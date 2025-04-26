"""
Final Project: Mapping Patient Transfers Between Hospitals
Author: Haichao Min
Date: 2024-04-19

This script performs two main analyses:
1. Static analysis using disease-level and hospital-level CSV datasets.
2. Network analysis based on real-world ZIP-to-hospital patient origin data.
"""

# Import necessary libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def find_similar_hospitals(location_csv, target_hospital, top_n=5):
    # Load location data
    location_df = pd.read_csv(location_csv)

    # Pivot on Facility_Name, MDC_DESC, and Count
    pivot = location_df.pivot_table(
        index='Facility_Name',
        columns='MDC_DESC',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )

    # Normalize rows
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(pivot)
    similarity = cosine_similarity(norm_data)

    # Create similarity DataFrame
    sim_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

    if target_hospital not in sim_df.index:
        print(f"[ERROR] '{target_hospital}' not found. Please check the spelling or use one of the following valid names:\n{sim_df.index[:5].tolist()}")
        return

    similar = sim_df[target_hospital].sort_values(ascending=False).drop(target_hospital).head(top_n)
    print(f"\n=== Top {top_n} Hospitals Similar to '{target_hospital}' ===")
    print(similar.to_string())


def load_and_clean_encounter_data(filepath):
    """Load and clean the encounter data CSV file."""
    df = pd.read_csv(filepath)
    # Remove percentage sign and convert to float
    df['TransferPercent'] = df['TransferPercent'].str.rstrip('%').astype(float)
    # Clean TransferCount and convert to int
    df['TransferCount'] = (
        df['TransferCount'].astype(str)
        .str.replace(',', '')
        .replace('Less than 11', '10')
        .astype(int)
    )
    return df


def analyze_top_diseases(encounter_df):
    """Print top 5 disease categories by transfer count."""
    top5 = encounter_df.sort_values('TransferCount', ascending=False).head(5)
    print("\n=== Top 5 Disease Categories by Transfer Volume ===")
    print(top5[['MDC_DESC', 'TransferCount', 'TransferPercent']].to_string(index=False))


def analyze_death_rates(outcome_df):
    """Print top 5 disease categories by inpatient death rate."""
    deaths = outcome_df[outcome_df['Outcome'] == 'Percent of Inpatient Deaths']
    top5 = deaths.sort_values('Value', ascending=False).head(5)
    print("\n=== Top 5 Disease Categories by Inpatient Death Rate ===")
    print(top5[['MDC_DESC', 'Value']].to_string(index=False))


def analyze_top_facilities(location_df):
    """Print top 5 hospitals by total patient count."""
    facility_counts = location_df.groupby('Facility_Name')['Count'].sum().reset_index()
    top5 = facility_counts.sort_values('Count', ascending=False).head(5)
    print("\n=== Top 5 Facilities by Total Transfer-Related Volume ===")
    print(top5.to_string(index=False))


def build_network_with_facility_names(origin_csv, facility_csv):
    """
    Build a directed patient transfer network from ZIP to facility names.
    - origin_csv: patient origin data (ZIP -> discharge count)
    - facility_csv: location info containing facility names and ZIPs
    Returns:
      G: networkx.DiGraph
      edges_df: DataFrame with columns ['from_zip', 'to_facility', 'weight']
    """
    # Load and clean the patient origin data (skip metadata)
    df = pd.read_csv(origin_csv, skiprows=9)
    df.rename(columns={'Row Labels': 'ZIP'}, inplace=True)
    # Keep only 5-digit ZIP codes
    df = df[df['ZIP'].str.match(r'^\d{5}$', na=False)]
    # Convert to long format for 2022 only
    df = df.melt(id_vars=['ZIP'], value_vars=['2022'],
                 var_name='year', value_name='discharges')
    # Ensure ZIP is a zero-padded string
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)

    # Load facility location data
    facilities = pd.read_csv(facility_csv)
    # Detect and rename the ZIP column
    zip_col = next((c for c in facilities.columns if 'zip' in c.lower()), None)
    if zip_col is None:
        raise ValueError("No ZIP-like column found in facility CSV.")
    facilities['ZIP'] = facilities[zip_col].astype(str).str.zfill(5)
    # Rename facility name column (FACNAME -> Facility_Name)
    if 'FACNAME' in facilities.columns:
        facilities.rename(columns={'FACNAME': 'Facility_Name'}, inplace=True)
    if 'Facility_Name' not in facilities.columns:
        raise ValueError("Expected 'Facility_Name' column missing in facility CSV.")

    # Merge on ZIP
    merged = pd.merge(df, facilities[['ZIP', 'Facility_Name']], on='ZIP', how='left')
    # Warn about unmatched ZIPs
    missing = merged['Facility_Name'].isna().sum()
    print(f"⚠️ Unmatched ZIPs: {missing} rows could not be matched to a facility.")

    # Build edge list
    edges = merged[['ZIP', 'Facility_Name', 'discharges']]
    edges = edges.dropna(subset=['Facility_Name', 'discharges']).copy()
    edges.columns = ['from_zip', 'to_facility', 'weight']

    # Create directed graph
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['from_zip'], row['to_facility'], weight=row['weight'])

    return G, edges


def print_top_central_facilities(G, top_n=5):
    """Print top facilities by in-degree centrality."""
    centrality = nx.in_degree_centrality(G)
    central_df = pd.DataFrame({
        'Node': list(centrality.keys()),
        'Centrality': list(centrality.values())
    })
    top = central_df.sort_values('Centrality', ascending=False).head(top_n)
    print(f"\n=== Top {top_n} Facilities by Network Centrality (In-Degree) ===")
    print(top.to_string(index=False))


def generate_hospital_summary(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    df = pd.DataFrame({
        'Node': list(G.nodes),
        'InDegree': [G.in_degree(n) for n in G.nodes],
        'OutDegree': [G.out_degree(n) for n in G.nodes],
        'DegreeCentrality': [degree_centrality.get(n, 0) for n in G.nodes],
        'BetweennessCentrality': [betweenness_centrality.get(n, 0) for n in G.nodes]
    })
    top10 = df.sort_values('BetweennessCentrality', ascending=False).head(10)
    print("\n=== Hospital Centrality Summary (Top 10 by Betweenness) ===")
    print(top10.to_string(index=False))


def draw_top_transfer_network(edges, top_n=50):
    """
    Draw a simplified network graph using only top-N transfer edges.
    
    Parameters:
        edges (DataFrame): DataFrame with ['from_zip', 'to_facility', 'weight']
        top_n (int): Number of most frequent transfer paths to include
    """
    top_edges = edges.sort_values('weight', ascending=False).head(top_n)

    G_top = nx.DiGraph()
    for _, row in top_edges.iterrows():
        G_top.add_edge(row['from_zip'], row['to_facility'], weight=row['weight'])

    pos = nx.spring_layout(G_top, k=0.8, seed=42)
    plt.figure(figsize=(14, 10))

    nx.draw_networkx_nodes(G_top, pos, node_size=300, node_color='skyblue')
    weights = [G_top[u][v]['weight'] for u, v in G_top.edges()]
    max_weight = max(weights)
    edge_widths = [2 + 5 * (w / max_weight) for w in weights]
    nx.draw_networkx_edges(G_top, pos, width=edge_widths, alpha=0.7, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G_top, pos, font_size=8)

    plt.title(f"Top {top_n} Patient Transfer Paths (by Volume)")
    plt.axis('off')
    plt.tight_layout()
    # Save figure if backend is non-GUI
    if plt.get_backend().lower() in ['agg', 'pdf', 'svg', 'ps']:
        plt.savefig(f'top_{top_n}_patient_transfer_paths.png')
    else:
        plt.show(block=False)
        plt.pause(0.1)


def shortest_transfer_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target, weight='weight')
        print(f"\n=== Shortest Transfer Path from '{source}' to '{target}' ===")
        print(" -> ".join(path))
    except nx.NetworkXNoPath:
        print(f"\nNo path found between '{source}' and '{target}'")
    except nx.NodeNotFound as e:
        print(f"\nNode not found: {e}")


def main():
    # Configurable file paths
    ENCOUNTER_CSV = 'data/Hospital_Transfers_by_Major_Diagnostic_Category__MDC__-Encounters.csv'
    LOCATION_CSV = 'data/Hospital_Transfers_by_Major_Diagnostic_Category__MDC__-Locations.csv'
    OUTCOME_CSV = 'data/Hospital_Transfers_by_Major_Diagnostic_Category__MDC__-Outcomes.csv'
    ORIGIN_CSV = 'data/2022-2023-patient-origin-market-share.csv'
    FACILITY_CSV = 'data/health_facility_locations.csv'

    enc_df = load_and_clean_encounter_data(ENCOUNTER_CSV)
    loc_df = pd.read_csv(LOCATION_CSV)
    out_df = pd.read_csv(OUTCOME_CSV)

    analyze_top_diseases(enc_df)
    analyze_death_rates(out_df)
    analyze_top_facilities(loc_df)

    G, edge_data = build_network_with_facility_names(
        ORIGIN_CSV,
        FACILITY_CSV
    )
    print_top_central_facilities(G, top_n=5)
    generate_hospital_summary(G)
    draw_top_transfer_network(edge_data, top_n=50)

    print("Welcome to the Hospital Patient Transfer Network Analyzer!")
    print("This tool helps you explore hospital-level patient transfer patterns in California.")
    print("You can view stats, calculate centrality, explore hospital similarities, or find shortest transfer paths.\n")

    # Command-line interaction loop for options
    try:
        while True:
            print("\nOptions:\n1. Shortest Path\n2. Hospital Stats\n3. Centrality\n4. Similar Hospitals\nq. Quit")
            choice = input("Choose a function (1-4) or 'q' to quit: ").strip()
            if choice == 'q':
                print("Exiting program.")
                break
            elif choice == '1':
                print(f"Example source: {list(G.nodes)[:3]}")
                print(f"Example target: {list(G.nodes)[-3:]}")
                source = input("\nEnter source ZIP or hospital name: ").strip()
                target = input("Enter target ZIP or hospital name: ").strip()
                if not source or not target:
                    print("[ERROR] Source and target cannot be empty.")
                    continue
                shortest_transfer_path(G, source=source, target=target)
            elif choice == '2':
                generate_hospital_summary(G)
            elif choice == '3':
                print_top_central_facilities(G, top_n=5)
            elif choice == '4':
                print(f"Example hospitals: {loc_df['Facility_Name'].dropna().unique()[:5].tolist()}")
                hospital_name = input("Enter hospital name to compare: ").strip()
                if not hospital_name:
                    print("[ERROR] Hospital name cannot be empty.")
                    continue
                find_similar_hospitals(
                    LOCATION_CSV,
                    target_hospital=hospital_name,
                    top_n=5
                )
            else:
                print("Invalid option.")
    except EOFError:
        print("\n[ERROR] No input received. Are you running this in an interactive terminal?")
    except Exception as e:
        print(f"\n[ERROR] Something went wrong during CLI interaction: {e}")


if __name__ == '__main__':
    main()
