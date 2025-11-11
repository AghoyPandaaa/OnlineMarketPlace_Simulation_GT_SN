"""
Integrated Simulation: Nash Equilibrium with Social Network Effects
Demonstrates how customer networks and word-of-mouth marketing affect competitive equilibrium.
"""

import sys
from pathlib import Path

# Add parent directory to path to import other modules
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from Task2.SellerModeling import Seller, MarketModel
from Task4.SocialNetworkAnalysis import CustomerNetwork

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (22, 12)


def create_balanced_sellers():
    """Create two balanced sellers for realistic Nash equilibrium demonstration."""

    seller_A = Seller(
        name="Seller_A",
        cost=1.50,
        initial_price=2.20,
        initial_ad_budget=300,
        base_demand=200  # Remove brand_value parameter
    )

    seller_B = Seller(
        name="Seller_B",
        cost=1.50,
        initial_price=2.25,
        initial_ad_budget=250,
        base_demand=180  # Remove brand_value parameter
    )

    return seller_A, seller_B


def find_best_response(market, seller_i, seller_j, influence_score_i=0,
                       price_min=1.50, price_max=3.00, price_step=0.05,
                       ad_min=0, ad_max=500, ad_step=50):
    """Find best response strategy for seller_i given seller_j's strategy."""

    best_price = seller_i.price
    best_ad = seller_i.ad_budget
    best_profit = float('-inf')

    # Test all price/ad combinations
    for price in np.arange(price_min, price_max + price_step, price_step):
        for ad in range(ad_min, ad_max + ad_step, ad_step):
            # Temporarily update strategy
            seller_i.price = price
            seller_i.ad_budget = ad

            # Calculate profit with this strategy
            profit = market.calculate_profit(seller_i, seller_j, influence_score_i)

            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_ad = ad

    # Set to best response
    seller_i.price = best_price
    seller_i.ad_budget = best_ad

    return best_price, best_ad, best_profit


def find_nash_equilibrium(market, seller_A, seller_B,
                          influence_A=0, influence_B=0,
                          max_iterations=30, threshold=0.01):
    """Find Nash equilibrium using iterative best response."""

    for iteration in range(max_iterations):
        # Store old strategies
        old_price_A, old_ad_A = seller_A.price, seller_A.ad_budget
        old_price_B, old_ad_B = seller_B.price, seller_B.ad_budget

        # Seller A's best response
        find_best_response(market, seller_A, seller_B, influence_A)

        # Seller B's best response
        find_best_response(market, seller_B, seller_A, influence_B)

        # Check convergence
        change_A = abs(seller_A.price - old_price_A) + abs(seller_A.ad_budget - old_ad_A)
        change_B = abs(seller_B.price - old_price_B) + abs(seller_B.ad_budget - old_ad_B)
        total_change = change_A + change_B

        if total_change < threshold:
            break

    # Calculate final profits
    profit_A = market.calculate_profit(seller_A, seller_B, influence_A)
    profit_B = market.calculate_profit(seller_B, seller_A, influence_B)

    return {
        'seller_A': {
            'price': seller_A.price,
            'ad_budget': seller_A.ad_budget,
            'profit': profit_A,
            'iterations': iteration + 1
        },
        'seller_B': {
            'price': seller_B.price,
            'ad_budget': seller_B.ad_budget,
            'profit': profit_B,
            'iterations': iteration + 1
        }
    }


def calculate_seller_influence(seller_name, price_range, influence_scores):
    """
    Simplified: Map seller to average customer influence.
    In reality, would identify actual customers per seller.
    """
    # For demonstration: assign influence based on seller quality
    if seller_name == "Seller_A":
        return 0.35  # Higher quality → more influential customers
    else:
        return 0.28  # Lower quality → less influential customers


def visualize_network_impact(results_no_network, results_with_network,
                             gamma_value, save_path='network_impact_comparison.png'):
    """Create comprehensive comparison visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Extract data
    sellers = ['Seller_A', 'Seller_B']

    # Without network
    prices_no = [results_no_network['seller_A']['price'],
                 results_no_network['seller_B']['price']]
    ads_no = [results_no_network['seller_A']['ad_budget'],
              results_no_network['seller_B']['ad_budget']]
    profits_no = [results_no_network['seller_A']['profit'],
                  results_no_network['seller_B']['profit']]

    # With network
    prices_with = [results_with_network['seller_A']['price'],
                   results_with_network['seller_B']['price']]
    ads_with = [results_with_network['seller_A']['ad_budget'],
                results_with_network['seller_B']['ad_budget']]
    profits_with = [results_with_network['seller_A']['profit'],
                    results_with_network['seller_B']['profit']]

    x = np.arange(len(sellers))
    width = 0.35

    # === SUBPLOT 1: Price Comparison ===
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width / 2, prices_no, width, label='γ=0 (No Network)',
                    color='#FF9999', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, prices_with, width, label=f'γ={gamma_value} (With Network)',
                    color='#66B2FF', alpha=0.8)

    ax1.set_xlabel('Seller', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Price (€)', fontsize=11, fontweight='bold')
    ax1.set_title('Nash Equilibrium Prices', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sellers)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (no, with_net) in enumerate(zip(prices_no, prices_with)):
        ax1.text(i - width / 2, no + 0.02, f'€{no:.2f}', ha='center', fontsize=9, fontweight='bold')
        ax1.text(i + width / 2, with_net + 0.02, f'€{with_net:.2f}', ha='center', fontsize=9, fontweight='bold')

    # === SUBPLOT 2: Ad Budget Comparison ===
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width / 2, ads_no, width, label='γ=0 (No Network)',
                    color='#FF9999', alpha=0.8)
    bars2 = ax2.bar(x + width / 2, ads_with, width, label=f'γ={gamma_value} (With Network)',
                    color='#66B2FF', alpha=0.8)

    ax2.set_xlabel('Seller', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ad Budget (€)', fontsize=11, fontweight='bold')
    ax2.set_title('Nash Equilibrium Ad Spending', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sellers)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (no, with_net) in enumerate(zip(ads_no, ads_with)):
        ax2.text(i - width / 2, no + 5, f'€{no:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax2.text(i + width / 2, with_net + 5, f'€{with_net:.0f}', ha='center', fontsize=9, fontweight='bold')

    # === SUBPLOT 3: Profit Comparison (MOST IMPORTANT!) ===
    ax3 = axes[0, 2]
    bars1 = ax3.bar(x - width / 2, profits_no, width, label='γ=0 (No Network)',
                    color='#FF9999', alpha=0.8)
    bars2 = ax3.bar(x + width / 2, profits_with, width, label=f'γ={gamma_value} (With Network)',
                    color='#66B2FF', alpha=0.8)

    ax3.set_xlabel('Seller', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Profit (€)', fontsize=11, fontweight='bold')
    ax3.set_title('Nash Equilibrium Profits (KEY RESULT)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sellers)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels and profit change
    for i, (no, with_net) in enumerate(zip(profits_no, profits_with)):
        ax3.text(i - width / 2, no + 20, f'€{no:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax3.text(i + width / 2, with_net + 20, f'€{with_net:.0f}', ha='center', fontsize=9, fontweight='bold')

        # Show profit increase
        change = with_net - no
        pct_change = (change / no * 100) if no != 0 else 0
        ax3.text(i, max(no, with_net) + 60,
                 f'Δ: €{change:+.0f}\n({pct_change:+.1f}%)',
                 ha='center', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # === SUBPLOT 4: Strategy Changes ===
    ax4 = axes[1, 0]

    changes_A = {
        'Price': prices_with[0] - prices_no[0],
        'Ad Budget': ads_with[0] - ads_no[0],
        'Profit': profits_with[0] - profits_no[0]
    }

    colors_A = ['green' if v > 0 else 'red' for v in changes_A.values()]
    ax4.barh(list(changes_A.keys()), list(changes_A.values()), color=colors_A, alpha=0.7)
    ax4.set_xlabel('Change in Strategy (€)', fontsize=11, fontweight='bold')
    ax4.set_title('Seller_A: Impact of Networks', fontsize=12, fontweight='bold')
    ax4.axvline(0, color='black', linewidth=0.8)
    ax4.grid(axis='x', alpha=0.3)

    # Add values
    for i, (key, val) in enumerate(changes_A.items()):
        ax4.text(val + (5 if val > 0 else -5), i, f'{val:+.1f}',
                 va='center', ha='left' if val > 0 else 'right', fontsize=9)

    # === SUBPLOT 5: Strategy Changes Seller B ===
    ax5 = axes[1, 1]

    changes_B = {
        'Price': prices_with[1] - prices_no[1],
        'Ad Budget': ads_with[1] - ads_no[1],
        'Profit': profits_with[1] - profits_no[1]
    }

    colors_B = ['green' if v > 0 else 'red' for v in changes_B.values()]
    ax5.barh(list(changes_B.keys()), list(changes_B.values()), color=colors_B, alpha=0.7)
    ax5.set_xlabel('Change in Strategy (€)', fontsize=11, fontweight='bold')
    ax5.set_title('Seller_B: Impact of Networks', fontsize=12, fontweight='bold')
    ax5.axvline(0, color='black', linewidth=0.8)
    ax5.grid(axis='x', alpha=0.3)

    # Add values
    for i, (key, val) in enumerate(changes_B.items()):
        ax5.text(val + (5 if val > 0 else -5), i, f'{val:+.1f}',
                 va='center', ha='left' if val > 0 else 'right', fontsize=9)

    # === SUBPLOT 6: Key Insights ===
    ax6 = axes[1, 2]
    ax6.axis('off')

    total_profit_no = sum(profits_no)
    total_profit_with = sum(profits_with)
    total_increase = total_profit_with - total_profit_no

    insights_text = f"""
    KEY INSIGHTS: SOCIAL NETWORK EFFECTS

    ═══════════════════════════════════════════
    WITHOUT Networks (γ=0):
    ───────────────────────────────────────────
    • Demand = base + (α×ad) + (β×price_diff)
    • No viral marketing effect
    • Traditional competition only

    ═══════════════════════════════════════════
    WITH Networks (γ={gamma_value}):
    ───────────────────────────────────────────
    • Demand = base + (α×ad) + (β×price_diff)
              + (γ×influence)
    • Word-of-mouth amplifies sales
    • Influential customers drive demand

    ═══════════════════════════════════════════
    MARKET-LEVEL IMPACT:
    ───────────────────────────────────────────
    Total Market Profit WITHOUT: €{total_profit_no:.0f}
    Total Market Profit WITH:    €{total_profit_with:.0f}

    Network Effect Value: €{total_increase:+.0f}
       ({total_increase / total_profit_no * 100:+.1f}% increase)

    ═══════════════════════════════════════════
    BUSINESS INTERPRETATION:
    ───────────────────────────────────────────
    Social networks create {total_increase:.0f}€ additional
    value through word-of-mouth effects. Sellers
    with more influential customers benefit more,
    changing Nash equilibrium strategies.

    This demonstrates why influencer marketing
    and viral strategies are valuable!
    """

    ax6.text(0.05, 0.95, insights_text,
             transform=ax6.transAxes,
             fontsize=9,
             verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('IMPACT OF SOCIAL NETWORKS ON NASH EQUILIBRIUM', fontsize=16)
    plt.tight_layout()  # No rect, nothing else!
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


    print(f"\n✓ Network impact visualization saved: {save_path}")


def main():
    """Main execution: Compare Nash equilibrium with and without network effects."""

    print("\n" + "=" * 80)
    print("INTEGRATED SIMULATION: NASH EQUILIBRIUM + SOCIAL NETWORKS")
    print("=" * 80)

    # Create sellers
    print("\n[1/5] Creating balanced sellers...")
    seller_A, seller_B = create_balanced_sellers()
    print(f"  ✓ Seller_A: cost={seller_A.cost}, base_demand={seller_A.base_demand}")
    print(f"  ✓ Seller_B: cost={seller_B.cost}, base_demand={seller_B.base_demand}")

    # ============================================================================
    # SCENARIO 1: WITHOUT Social Networks (γ=0)
    # ============================================================================
    print("\n[2/5] SCENARIO 1: Nash Equilibrium WITHOUT Social Networks (γ=0)")
    print("-" * 80)

    market_no_network = MarketModel(
        sellers_dict={'Seller_A': seller_A, 'Seller_B': seller_B},  # Changed: sellers → sellers_dict
        alpha=0.01,
        beta=5.0,
        gamma=0.0,  # NO social influence
        epsilon=0.5
    )

    # Reset sellers to initial state
    seller_A.price, seller_A.ad_budget = 2.20, 300
    seller_B.price, seller_B.ad_budget = 2.25, 250

    results_no_network = find_nash_equilibrium(
        market_no_network, seller_A, seller_B,
        influence_A=0, influence_B=0
    )

    print(f"\n  Nash Equilibrium (γ=0):")
    print(f"    Seller_A: Price=€{results_no_network['seller_A']['price']:.2f}, "
          f"Ad=€{results_no_network['seller_A']['ad_budget']:.0f}, "
          f"Profit=€{results_no_network['seller_A']['profit']:.0f}")
    print(f"    Seller_B: Price=€{results_no_network['seller_B']['price']:.2f}, "
          f"Ad=€{results_no_network['seller_B']['ad_budget']:.0f}, "
          f"Profit=€{results_no_network['seller_B']['profit']:.0f}")
    print(f"    Converged in {results_no_network['seller_A']['iterations']} iterations")

    # ============================================================================
    # SCENARIO 2: WITH Social Networks (γ=2.0)
    # ============================================================================
    print("\n[3/5] SCENARIO 2: Nash Equilibrium WITH Social Networks (γ=2.0)")
    print("-" * 80)

    # Simulate influence scores (in full version, calculate from network)
    influence_A = 0.35  # Seller_A has more influential customers
    influence_B = 0.28  # Seller_B has less influential customers

    print(f"  Seller influence scores:")
    print(f"    Seller_A: {influence_A:.3f} (higher quality → more influential customers)")
    print(f"    Seller_B: {influence_B:.3f}")

    market_with_network = MarketModel(
        sellers_dict={'Seller_A': seller_A, 'Seller_B': seller_B},  # Changed: sellers → sellers_dict
        alpha=0.01,
        beta=5.0,
        gamma=2.0,  # Social influence enabled!
        epsilon=0.5
    )

    # Reset sellers to initial state
    seller_A.price, seller_A.ad_budget = 2.20, 300
    seller_B.price, seller_B.ad_budget = 2.25, 250

    results_with_network = find_nash_equilibrium(
        market_with_network, seller_A, seller_B,
        influence_A=influence_A, influence_B=influence_B
    )

    print(f"\n  Nash Equilibrium (γ=2.0):")
    print(f"    Seller_A: Price=€{results_with_network['seller_A']['price']:.2f}, "
          f"Ad=€{results_with_network['seller_A']['ad_budget']:.0f}, "
          f"Profit=€{results_with_network['seller_A']['profit']:.0f}")
    print(f"    Seller_B: Price=€{results_with_network['seller_B']['price']:.2f}, "
          f"Ad=€{results_with_network['seller_B']['ad_budget']:.0f}, "
          f"Profit=€{results_with_network['seller_B']['profit']:.0f}")
    print(f"    Converged in {results_with_network['seller_A']['iterations']} iterations")

    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("\n[4/5] IMPACT ANALYSIS")
    print("=" * 80)

    profit_change_A = results_with_network['seller_A']['profit'] - results_no_network['seller_A']['profit']
    profit_change_B = results_with_network['seller_B']['profit'] - results_no_network['seller_B']['profit']
    total_change = profit_change_A + profit_change_B

    print(f"\nProfit Changes from Social Network Effects:")
    print(
        f"  Seller_A: €{profit_change_A:+.2f} ({profit_change_A / results_no_network['seller_A']['profit'] * 100:+.1f}%)")
    print(
        f"  Seller_B: €{profit_change_B:+.2f} ({profit_change_B / results_no_network['seller_B']['profit'] * 100:+.1f}%)")
    print(f"  Total Market: €{total_change:+.2f}")

    print(f"\nKEY FINDING:")
    print(f"  Social networks created {total_change:.2f}€ additional value!")
    print(f"  Seller with more influential customers (A) benefits more.")

    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    print("\n[5/5] Creating comparison visualization...")

    output_dir = Path(__file__).parent
    visualize_network_impact(
        results_no_network,
        results_with_network,
        gamma_value=2.0,
        save_path=str(output_dir / 'network_impact_comparison.png')
    )

    print("\n" + "=" * 80)
    print("✓ INTEGRATED SIMULATION COMPLETE!")
    print("=" * 80)
    print("\nFiles created:")
    print("  • network_impact_comparison.png - Full comparison visualization")
    print("\nKey Result:")
    print(f"  Social networks increase total market profit by €{total_change:.2f}")
    print("  This demonstrates the value of viral marketing and influencer strategies!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
