"""
Task IV: Social Network Analysis
Creates customer network and calculates influence scores for viral marketing simulation.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class CustomerNetwork:
    """
    Manages customer social network for viral marketing simulation.

    Key Concepts (for midterm):
    - Degree Centrality: Number of connections a customer has
    - Betweenness Centrality: How important a customer is for connecting others
    - PageRank: Google's algorithm - importance based on connections from important nodes
    - Influence Score: Combined metric of customer's network power

    Network Construction:
    - Nodes = Customers
    - Edges = Co-purchase relationships (bought same products)
    - Weight = Number of common products purchased
    """

    def __init__(self, df, sample_size=500):
        """
        Initialize customer network from transaction data.

        Args:
            df: Cleaned transaction dataframe with columns:
                ['Customer ID', 'StockCode', 'Quantity', 'Price', 'Revenue']
            sample_size: Number of top customers to include (default 500 for performance)
        """
        self.df = df
        self.sample_size = sample_size
        self.G = nx.Graph()  # Undirected graph
        self.influence_scores = {}
        self.centrality_metrics = {}

        print("\n" + "=" * 80)
        print("CUSTOMER NETWORK INITIALIZATION")
        print("=" * 80)
        print(f"Dataset size: {len(df)} transactions")
        print(f"Target network size: {sample_size} customers")
        print("=" * 80 + "\n")

    def build_network_from_purchases(self):
        """
        Build co-purchase network where customers are connected if they bought similar products.

        Strategy: Two customers are "friends" if they purchased the same product.
        This simulates word-of-mouth: people who buy similar things likely influence each other.

        Returns:
            nx.Graph: The constructed network
        """
        print("[1/4] Building customer network from purchase patterns...")

        # Get top customers by number of transactions
        customer_counts = self.df.groupby('Customer ID').size()
        top_customers = customer_counts.nlargest(self.sample_size).index.tolist()

        print(f"  Selected top {len(top_customers)} customers")

        # Filter data to selected customers
        df_sample = self.df[self.df['Customer ID'].isin(top_customers)]
        print(f"  Filtered to {len(df_sample)} transactions")

        # Add all customers as nodes
        self.G.add_nodes_from(top_customers)
        print(f"  Added {self.G.number_of_nodes()} nodes")

        # Build edges based on co-purchases
        # Group by product to find customers who bought the same item
        product_groups = df_sample.groupby('StockCode')['Customer ID'].apply(list)

        edge_weights = defaultdict(int)

        for product, customers in product_groups.items():
            # Only consider products bought by 2-50 customers (not too rare, not too common)
            if 2 <= len(customers) <= 50:
                # Connect all pairs of customers who bought this product
                for i in range(len(customers)):
                    for j in range(i + 1, len(customers)):
                        customer_i = customers[i]
                        customer_j = customers[j]

                        # Increment edge weight (number of common products)
                        edge = tuple(sorted([customer_i, customer_j]))
                        edge_weights[edge] += 1

        # Add edges to graph with weights
        for (customer_i, customer_j), weight in edge_weights.items():
            if weight >= 2:  # Only add edges with at least 2 common products
                self.G.add_edge(customer_i, customer_j, weight=weight)

        print(f"  Created {self.G.number_of_edges()} connections")
        print(f"  Network density: {nx.density(self.G):.4f}")

        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)
        print(f"  Removed {len(isolated_nodes)} isolated customers")
        print(f"  Final network: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        return self.G

    def calculate_influence_scores(self):
        """
        Calculate influence score for each customer using multiple centrality measures.

        Influence Score combines:
        - Degree Centrality (30%): Number of direct connections
        - Betweenness Centrality (30%): How often node appears on shortest paths
        - PageRank (40%): Importance based on important connections

        Returns:
            dict: {customer_id: influence_score}
        """
        print("\n[2/4] Calculating influence scores...")

        # Calculate centrality measures
        print("  Computing degree centrality...")
        degree_cent = nx.degree_centrality(self.G)

        print("  Computing betweenness centrality...")
        betweenness_cent = nx.betweenness_centrality(self.G, weight='weight')

        print("  Computing PageRank...")
        pagerank = nx.pagerank(self.G, weight='weight', alpha=0.85)

        # Store centrality metrics
        self.centrality_metrics = {
            'degree': degree_cent,
            'betweenness': betweenness_cent,
            'pagerank': pagerank
        }

        # Normalize to [0, 1]
        max_degree = max(degree_cent.values()) if degree_cent else 1
        max_between = max(betweenness_cent.values()) if betweenness_cent else 1
        max_pagerank = max(pagerank.values()) if pagerank else 1

        # Calculate combined influence score
        for customer in self.G.nodes():
            degree_norm = degree_cent.get(customer, 0) / max_degree if max_degree > 0 else 0
            between_norm = betweenness_cent.get(customer, 0) / max_between if max_between > 0 else 0
            pagerank_norm = pagerank.get(customer, 0) / max_pagerank if max_pagerank > 0 else 0

            # Weighted combination (can adjust weights)
            influence = (0.30 * degree_norm +
                         0.30 * between_norm +
                         0.40 * pagerank_norm)

            self.influence_scores[customer] = influence

        influence_values = list(self.influence_scores.values())
        print(f"  Calculated influence for {len(self.influence_scores)} customers")
        print(f"  Average influence: {np.mean(influence_values):.4f}")
        print(f"  Max influence: {max(influence_values):.4f}")
        print(f"  Min influence: {min(influence_values):.4f}")

        return self.influence_scores

    def get_top_influencers(self, n=10):
        """
        Return top n most influential customers.

        Args:
            n: Number of top influencers to return

        Returns:
            list of tuples: [(customer_id, influence_score), ...]
        """
        sorted_influencers = sorted(self.influence_scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
        return sorted_influencers[:n]

    def get_network_statistics(self):
        """
        Calculate and return comprehensive network statistics.

        Returns:
            dict: Network metrics
        """
        if self.G.number_of_nodes() == 0:
            return {}

        # Calculate metrics
        degrees = [deg for node, deg in self.G.degree()]

        stats = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'density': nx.density(self.G),
            'num_components': nx.number_connected_components(self.G),
            'avg_clustering': nx.average_clustering(self.G),
            'avg_influence': np.mean(list(self.influence_scores.values())) if self.influence_scores else 0
        }

        return stats

    def print_network_statistics(self):
        """Display network statistics in formatted table."""
        stats = self.get_network_statistics()

        print("\n" + "=" * 80)
        print("NETWORK STATISTICS")
        print("=" * 80)
        print(f"Number of Customers (Nodes):        {stats['num_nodes']:,}")
        print(f"Number of Connections (Edges):      {stats['num_edges']:,}")
        print(f"Average Connections per Customer:   {stats['avg_degree']:.2f}")
        print(f"Maximum Connections (Hub):          {stats['max_degree']}")
        print(f"Network Density:                    {stats['density']:.4f}")
        print(f"Number of Communities:              {stats['num_components']}")
        print(f"Average Clustering Coefficient:     {stats['avg_clustering']:.4f}")
        print(f"Average Influence Score:            {stats['avg_influence']:.4f}")
        print("=" * 80 + "\n")

        return stats

    def visualize_network(self, save_path='social_network.png'):
        """
        Create comprehensive network visualization with 3 views.

        Args:
            save_path: Path to save the visualization
        """
        print("\n[3/4] Creating network visualization...")

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # === SUBPLOT 1: Full Network ===
        ax1 = axes[0]

        # Use spring layout for positioning
        print("  Computing layout...")
        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=42)

        # Node sizes proportional to influence
        node_sizes = [self.influence_scores.get(node, 0) * 800 + 50
                      for node in self.G.nodes()]

        # Node colors by influence
        node_colors = [self.influence_scores.get(node, 0)
                       for node in self.G.nodes()]

        # Draw network
        nx.draw_networkx_edges(self.G, pos, alpha=0.15, width=0.5, ax=ax1)
        nodes = nx.draw_networkx_nodes(self.G, pos,
                                       node_size=node_sizes,
                                       node_color=node_colors,
                                       cmap='YlOrRd',
                                       alpha=0.8,
                                       ax=ax1)

        ax1.set_title(f'Customer Social Network (n={self.G.number_of_nodes()})\n'
                      'Node Size & Color = Influence Score',
                      fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                   norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
        sm.set_array([])
        cbar1 = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Influence Score', rotation=270, labelpad=20, fontsize=10)

        # === SUBPLOT 2: Top Influencers Subgraph ===
        ax2 = axes[1]

        # Get top influencers and their neighbors
        top_influencers = self.get_top_influencers(n=30)
        top_nodes = [node for node, score in top_influencers]

        # Include neighbors
        subgraph_nodes = set(top_nodes)
        for node in top_nodes:
            neighbors = list(self.G.neighbors(node))
            subgraph_nodes.update(neighbors[:10])  # Add up to 10 neighbors

        subgraph = self.G.subgraph(list(subgraph_nodes))

        pos_sub = nx.spring_layout(subgraph, k=0.7, iterations=50, seed=42)

        # Different colors for top influencers
        node_colors_sub = ['#FF4444' if node in top_nodes else '#88CCFF'
                           for node in subgraph.nodes()]

        node_sizes_sub = [self.influence_scores.get(node, 0) * 1200 + 100
                          for node in subgraph.nodes()]

        nx.draw_networkx_edges(subgraph, pos_sub, alpha=0.3, ax=ax2)
        nx.draw_networkx_nodes(subgraph, pos_sub,
                               node_size=node_sizes_sub,
                               node_color=node_colors_sub,
                               alpha=0.8,
                               ax=ax2)

        # Label only top 10 influencers
        top_10_labels = {node: f"{node}\n{self.influence_scores[node]:.3f}"
                         for node, score in top_influencers[:10]
                         if node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos_sub, top_10_labels,
                                font_size=7, ax=ax2)

        ax2.set_title('Top 30 Influencers & Their Networks\n'
                      '(Red = Top Influencer, Blue = Connected Customer)',
                      fontsize=14, fontweight='bold')
        ax2.axis('off')

        # === SUBPLOT 3: Influence Distribution ===
        ax3 = axes[2]

        influence_values = list(self.influence_scores.values())

        # Histogram
        ax3.hist(influence_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(influence_values), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(influence_values):.3f}')
        ax3.axvline(np.median(influence_values), color='green', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(influence_values):.3f}')

        ax3.set_xlabel('Influence Score', fontsize=11)
        ax3.set_ylabel('Number of Customers', fontsize=11)
        ax3.set_title('Distribution of Influence Scores\n'
                      'Most customers have low influence, few are highly influential',
                      fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        # Add text box with statistics
        stats_text = f"Network Stats:\n" \
                     f"• Nodes: {self.G.number_of_nodes()}\n" \
                     f"• Edges: {self.G.number_of_edges()}\n" \
                     f"• Density: {nx.density(self.G):.4f}\n" \
                     f"• Avg Degree: {np.mean([d for n, d in self.G.degree()]):.1f}\n" \
                     f"• Avg Influence: {np.mean(influence_values):.3f}"

        ax3.text(0.65, 0.95, stats_text,
                 transform=ax3.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Visualization saved: {save_path}")

        return save_path


def main():
    """
    Main execution function for Task IV: Social Network Analysis
    """
    print("\n" + "=" * 80)
    print("TASK IV: SOCIAL NETWORK ANALYSIS")
    print("=" * 80)

    # Load cleaned data
    data_path = Path(__file__).parent / "Data/ProcessedData/cleaned_online_retail_data.csv"

    if not data_path.exists():
        # Try alternative path
        data_path = Path(__file__).parent.parent / "Data/ProcessedData/cleaned_online_retail_data.csv"

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions")

    # Create network
    network = CustomerNetwork(df, sample_size=500)

    # Build network
    G = network.build_network_from_purchases()

    # Calculate influence scores
    influence_scores = network.calculate_influence_scores()

    # Display top influencers
    print("\n" + "=" * 80)
    print("TOP 10 MOST INFLUENTIAL CUSTOMERS")
    print("=" * 80)
    top_influencers = network.get_top_influencers(n=10)

    print(f"{'Rank':<6} {'Customer ID':<15} {'Influence Score':<20} {'Degree':<10}")
    print("-" * 60)
    for i, (customer, score) in enumerate(top_influencers, 1):
        degree = G.degree(customer)
        print(f"{i:<6} {customer:<15} {score:<20.4f} {degree:<10}")
    print("=" * 80 + "\n")

    # Network statistics
    stats = network.print_network_statistics()

    # Create directory if it doesn't exist
    import os
    output_dir = Path(__file__).parent
    os.makedirs(output_dir, exist_ok=True)

    # Save in same directory as script
    network.visualize_network(output_dir / 'social_network.png')

    print("\n" + "=" * 80)
    print("✓ TASK IV: SOCIAL NETWORK ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nFiles created:")
    print("  • Task4/social_network.png - Network visualization")
    print("\nNext: Integrate with GameTheorySimulation.py to show network effects on Nash equilibrium")
    print("=" * 80 + "\n")

    return network, influence_scores, stats


if __name__ == "__main__":
    network, influence_scores, stats = main()
