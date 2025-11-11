import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to access Task2 module
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from Task2.SellerModeling import sellers, market

# ============================================================================
# NASH EQUILIBRIUM FINDER - ITERATIVE BEST RESPONSE ALGORITHM
# ============================================================================
# This is the CORE algorithm for finding Nash Equilibrium in game theory!
# Teach this concept for midterm exam preparation.
# ============================================================================

def find_best_response(responding_seller, opponent_seller, market,
                       price_step=0.02, ad_step=100, verbose=False):
    """
    Find best response strategy for one seller given opponent's fixed strategy.

    This is a KEY CONCEPT in game theory:
    - Best Response = Strategy that maximizes your profit given opponent's strategy
    - Nash Equilibrium occurs when both players are playing best responses

    Parameters:
    - responding_seller: Seller finding their best response
    - opponent_seller: Seller with fixed strategy
    - market: MarketModel object
    - price_step: Price increment for search (default 0.02)
    - ad_step: Ad budget increment for search (default 100)
    - verbose: Print search progress

    Returns:
    - Dictionary with best price, ad budget, and resulting profit
    """

    # Define search ranges - VERY WIDE to find true optimum
    # Price range: From barely above cost to 5x cost
    min_price = responding_seller.production_cost * 1.01
    max_price = responding_seller.production_cost * 5.0  # EXPANDED to 5x to see full landscape

    # Ad budget range: £0 to £3,000
    min_ad = 0
    max_ad = 3000

    # Create search grid
    prices = np.arange(min_price, max_price, price_step)
    ad_budgets = np.arange(min_ad, max_ad + ad_step, ad_step)  # Include max_ad

    if verbose:
        print(f"    Search space: {len(prices)} prices × {len(ad_budgets)} ad budgets = {len(prices)*len(ad_budgets)} combinations")
        print(f"    Price range: €{min_price:.2f} to €{max_price:.2f}")
        print(f"    Ad range: €{min_ad:.0f} to €{max_ad:.0f}")

    # Track best strategy found
    best_profit = -np.inf
    best_price = None
    best_ad = None

    # Save original strategy
    original_price = responding_seller.price
    original_ad = responding_seller.advertising_budget

    # Grid search over all combinations
    tested = 0
    for price in prices:
        for ad_budget in ad_budgets:
            # Set temporary strategy
            responding_seller.price = price
            responding_seller.advertising_budget = ad_budget

            # Calculate profit with this strategy
            profit = market.calculate_profit(responding_seller, opponent_seller)

            # Update best if this is better
            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_ad = ad_budget

            tested += 1

    # Restore original strategy
    responding_seller.price = original_price
    responding_seller.advertising_budget = original_ad

    if verbose:
        print(f"    Best response: Price=€{best_price:.2f}, Ad=€{best_ad:.0f}, Profit=€{best_profit:.2f}")
        print(f"    Tested {tested} combinations")

    return {
        'price': best_price,
        'ad_budget': best_ad,
        'profit': best_profit
    }


def find_nash_equilibrium(market, seller_A, seller_B,
                          max_iterations=50,
                          convergence_threshold=0.01,
                          price_step=0.01, ad_step=50,
                          verbose=True):
    """
    Find Nash Equilibrium using iterative best response algorithm.

    ============================================================================
    ALGORITHM EXPLANATION (Study this for midterm exam!)
    ============================================================================

    1. WHAT IS NASH EQUILIBRIUM?
       - A state where NO player can improve by changing strategy alone
       - Both players are playing best responses to each other
       - Stable outcome: no incentive to deviate

    2. ITERATIVE BEST RESPONSE ALGORITHM:
       Step 1: Start with sellers' current strategies
       Step 2: Repeat until convergence:
           a) Seller A finds best response to B's current strategy
           b) Update A's strategy to best response
           c) Seller B finds best response to A's new strategy
           d) Update B's strategy to best response
           e) Check if strategies changed significantly
           f) If change < threshold → CONVERGED to Nash Equilibrium!
       Step 3: Return equilibrium strategies and history

    3. WHY DOES THIS WORK?
       - Each seller continuously adjusts to opponent's strategy
       - Eventually strategies stabilize (if equilibrium exists)
       - At equilibrium: no profitable deviations possible

    ============================================================================

    Parameters:
    - market: MarketModel object
    - seller_A, seller_B: Seller objects
    - max_iterations: Maximum iterations (default 50)
    - convergence_threshold: Stop when change < this (default 0.01)
    - price_step, ad_step: Search resolution
    - verbose: Print progress each iteration

    Returns:
    - Dictionary containing:
      * 'nash_equilibrium': Final equilibrium strategies and profits
      * 'converged': Boolean - True if converged, False if hit max_iterations
      * 'iterations': Number of iterations taken
      * 'history': List tracking strategies/profits at each iteration
      * 'convergence_metric': Change in last iteration

    CONVERGENCE METRIC:
    Euclidean distance of strategy changes:
    change = sqrt((p_A_new - p_A_old)^2 + (m_A_new - m_A_old)^2 +
                  (p_B_new - p_B_old)^2 + (m_B_new - m_B_old)^2)
    """

    # Make convergence threshold MUCH stricter to avoid premature convergence
    convergence_threshold = 0.001  # Changed from 0.01 to 0.001

    if verbose:
        print("\n" + "="*80)
        print("NASH EQUILIBRIUM FINDER - ITERATIVE BEST RESPONSE ALGORITHM")
        print("="*80)
        print(f"\nParameters:")
        print(f"  Max Iterations: {max_iterations}")
        print(f"  Convergence Threshold: {convergence_threshold} (STRICT)")
        print(f"  Price Step: €{price_step}")
        print(f"  Ad Budget Step: €{ad_step}")
        print("\n" + "-"*80)

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    # Save initial strategies
    initial_A = {
        'price': seller_A.price,
        'ad': seller_A.advertising_budget,
        'profit': market.calculate_profit(seller_A, seller_B)
    }

    initial_B = {
        'price': seller_B.price,
        'ad': seller_B.advertising_budget,
        'profit': market.calculate_profit(seller_B, seller_A)
    }

    if verbose:
        print(f"\nInitial State:")
        print(f"  {seller_A.name}: Price=£{initial_A['price']:.2f}, "
              f"Ad=£{initial_A['ad']:.0f}, Profit=£{initial_A['profit']:.2f}")
        print(f"  {seller_B.name}: Price=£{initial_B['price']:.2f}, "
              f"Ad=£{initial_B['ad']:.0f}, Profit=£{initial_B['profit']:.2f}")
        print("\n" + "-"*80)
        print("Beginning Iterative Best Response...\n")

    # History tracking
    history = []
    converged = False
    convergence_metric = np.inf

    # Store previous strategies for convergence check
    prev_A_price = seller_A.price
    prev_A_ad = seller_A.advertising_budget
    prev_B_price = seller_B.price
    prev_B_ad = seller_B.advertising_budget

    # ========================================================================
    # MAIN ITERATION LOOP
    # ========================================================================

    for iteration in range(max_iterations):

        if verbose:
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*80}")
            print(f"BEFORE UPDATE:")
            print(f"  {seller_A.name}: Price=€{seller_A.price:.2f}, Ad=€{seller_A.advertising_budget:.0f}")
            print(f"  {seller_B.name}: Price=€{seller_B.price:.2f}, Ad=€{seller_B.advertising_budget:.0f}")

        # Save state before updates for change calculation
        old_price_A = seller_A.price
        old_ad_A = seller_A.advertising_budget
        old_price_B = seller_B.price
        old_ad_B = seller_B.advertising_budget

        # --------------------------------------------------------------------
        # SELLER A'S TURN: Find best response to B's current strategy
        # --------------------------------------------------------------------

        if verbose:
            print(f"\n  [{seller_A.name} TURN] Finding best response to {seller_B.name}'s strategy...")

        best_A = find_best_response(
            seller_A, seller_B, market,
            price_step=price_step,
            ad_step=ad_step,
            verbose=verbose
        )

        # Update A's strategy to best response
        seller_A.price = best_A['price']
        seller_A.advertising_budget = best_A['ad_budget']
        profit_A = best_A['profit']

        if verbose:
            print(f"    → Updated {seller_A.name}: Price Δ={seller_A.price - old_price_A:+.2f}, Ad Δ={seller_A.advertising_budget - old_ad_A:+.0f}")

        # --------------------------------------------------------------------
        # SELLER B'S TURN: Find best response to A's NEW strategy
        # --------------------------------------------------------------------

        if verbose:
            print(f"\n  [{seller_B.name} TURN] Finding best response to {seller_A.name}'s NEW strategy...")

        best_B = find_best_response(
            seller_B, seller_A, market,
            price_step=price_step,
            ad_step=ad_step,
            verbose=verbose
        )

        # Update B's strategy to best response
        seller_B.price = best_B['price']
        seller_B.advertising_budget = best_B['ad_budget']
        profit_B = best_B['profit']

        if verbose:
            print(f"    → Updated {seller_B.name}: Price Δ={seller_B.price - old_price_B:+.2f}, Ad Δ={seller_B.advertising_budget - old_ad_B:+.0f}")

        # --------------------------------------------------------------------
        # RECORD HISTORY
        # --------------------------------------------------------------------

        history.append({
            'iteration': iteration + 1,
            'A_price': seller_A.price,
            'A_ad': seller_A.advertising_budget,
            'A_profit': profit_A,
            'B_price': seller_B.price,
            'B_ad': seller_B.advertising_budget,
            'B_profit': profit_B
        })

        # --------------------------------------------------------------------
        # CALCULATE CONVERGENCE METRIC
        # --------------------------------------------------------------------
        # Euclidean distance of strategy changes IN THIS ITERATION

        convergence_metric = np.sqrt(
            (seller_A.price - old_price_A)**2 +
            (seller_A.advertising_budget - old_ad_A)**2 +
            (seller_B.price - old_price_B)**2 +
            (seller_B.advertising_budget - old_ad_B)**2
        )

        # Add convergence metric to history
        history[-1]['change'] = convergence_metric

        # --------------------------------------------------------------------
        # PRINT ITERATION SUMMARY
        # --------------------------------------------------------------------

        if verbose:
            print(f"\nAFTER UPDATE:")
            print(f"  {seller_A.name}: Price=€{seller_A.price:.2f}, Ad=€{seller_A.advertising_budget:.0f}, Profit=€{profit_A:.2f}")
            print(f"  {seller_B.name}: Price=€{seller_B.price:.2f}, Ad=€{seller_B.advertising_budget:.0f}, Profit=€{profit_B:.2f}")
            print(f"\n  📊 Total Strategy Change: {convergence_metric:.6f} (threshold: {convergence_threshold})")

            # Show individual changes
            print(f"  Individual changes:")
            print(f"    {seller_A.name} price: €{old_price_A:.2f} → €{seller_A.price:.2f} (Δ={seller_A.price-old_price_A:+.2f})")
            print(f"    {seller_A.name} ad: €{old_ad_A:.0f} → €{seller_A.advertising_budget:.0f} (Δ={seller_A.advertising_budget-old_ad_A:+.0f})")
            print(f"    {seller_B.name} price: €{old_price_B:.2f} → €{seller_B.price:.2f} (Δ={seller_B.price-old_price_B:+.2f})")
            print(f"    {seller_B.name} ad: €{old_ad_B:.0f} → €{seller_B.advertising_budget:.0f} (Δ={seller_B.advertising_budget-old_ad_B:+.0f})")

        # --------------------------------------------------------------------
        # CHECK CONVERGENCE
        # --------------------------------------------------------------------

        if convergence_metric < convergence_threshold:
            converged = True
            if verbose:
                print(f"\n  ✓✓✓ CONVERGED! Change ({convergence_metric:.6f}) < Threshold ({convergence_threshold})")
            break

        # Update previous strategies for next iteration
        prev_A_price = seller_A.price
        prev_A_ad = seller_A.advertising_budget
        prev_B_price = seller_B.price
        prev_B_ad = seller_B.advertising_budget

        if verbose:
            print()  # Blank line between iterations

    # ========================================================================
    # POST-LOOP: FINAL STATUS
    # ========================================================================

    if verbose:
        print("-"*80)
        if converged:
            print(f"\n✓ CONVERGED to Nash Equilibrium in {len(history)} iterations!")
            print(f"  Final change: {convergence_metric:.6f} < {convergence_threshold}")
        else:
            print(f"\n⚠ Reached max iterations ({max_iterations}) without full convergence")
            print(f"  Final change: {convergence_metric:.6f} (threshold: {convergence_threshold})")
            print(f"  Note: Strategies may still be oscillating or converging slowly")

    # ========================================================================
    # PREPARE RESULTS
    # ========================================================================

    nash_equilibrium = {
        'seller_A': {
            'price': seller_A.price,
            'ad': seller_A.advertising_budget,
            'profit': profit_A
        },
        'seller_B': {
            'price': seller_B.price,
            'ad': seller_B.advertising_budget,
            'profit': profit_B
        }
    }

    # Print Nash Equilibrium
    if verbose:
        print("\n" + "="*80)
        print("NASH EQUILIBRIUM FOUND:")
        print("="*80)
        print(f"\n{seller_A.name}:")
        print(f"  Price: £{seller_A.price:.2f}")
        print(f"  Advertising Budget: £{seller_A.advertising_budget:.0f}")
        print(f"  Profit: £{profit_A:.2f}")

        print(f"\n{seller_B.name}:")
        print(f"  Price: £{seller_B.price:.2f}")
        print(f"  Advertising Budget: £{seller_B.advertising_budget:.0f}")
        print(f"  Profit: £{profit_B:.2f}")

        print("\n" + "="*80)
        print("\nKEY GAME THEORY CONCEPT:")
        print("At Nash Equilibrium, neither seller can improve profit by")
        print("changing their strategy alone. Both are playing best responses!")
        print("="*80 + "\n")

    return {
        'nash_equilibrium': nash_equilibrium,
        'converged': converged,
        'iterations': len(history),
        'history': history,
        'convergence_metric': convergence_metric,
        'initial_state': {
            'seller_A': initial_A,
            'seller_B': initial_B
        }

    }

def convert_history_to_dataframe(history):
    """
    Convert history list to pandas DataFrame for easy analysis and visualization.

    Parameters:
    - history: List of dictionaries from find_nash_equilibrium()

    Returns:
    - pandas DataFrame with columns:
      * iteration: Iteration number
      * A_price: Seller A's price
      * A_ad: Seller A's advertising budget
      * A_profit: Seller A's profit
      * B_price: Seller B's price
      * B_ad: Seller B's advertising budget
      * B_profit: Seller B's profit
      * change: Convergence metric (strategy change)
    """

    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)

    # Ensure proper column order
    columns = ['iteration', 'A_price', 'A_ad', 'A_profit',
               'B_price', 'B_ad', 'B_profit', 'change']

    return df[columns]

def visualize_nash_equilibrium(history_df, nash_result, save_path='nash_equilibrium.png'):
    """
    Create comprehensive 6-subplot visualization of Nash equilibrium process.

    Figure layout (3 rows × 2 columns):
    Row 1: Price Evolution | Ad Budget Evolution
    Row 2: Profit Evolution | Convergence Metric
    Row 3: Strategy Space Trajectory for A | Strategy Space Trajectory for B
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Nash Equilibrium Convergence Analysis', fontsize=16, fontweight='bold')

    # SUBPLOT 1: Price Evolution
    ax1 = axes[0, 0]
    ax1.plot(history_df['iteration'], history_df['A_price'],
             'b-', linewidth=2, label='Seller A', alpha=0.7)
    ax1.plot(history_df['iteration'], history_df['B_price'],
             'r-', linewidth=2, label='Seller B', alpha=0.7)

    # Mark Nash equilibrium
    final_iter = history_df['iteration'].max()
    nash_price_A = nash_result['nash_equilibrium']['seller_A']['price']
    nash_price_B = nash_result['nash_equilibrium']['seller_B']['price']

    ax1.plot(final_iter, nash_price_A, 'b*', markersize=20, label='Nash A')
    ax1.plot(final_iter, nash_price_B, 'r*', markersize=20, label='Nash B')
    ax1.axhline(nash_price_A, color='blue', linestyle='--', alpha=0.3)
    ax1.axhline(nash_price_B, color='red', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Price (€)')
    ax1.set_title('Price Evolution Toward Nash Equilibrium', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SUBPLOT 2: Ad Budget Evolution
    ax2 = axes[0, 1]
    ax2.plot(history_df['iteration'], history_df['A_ad'],
             'b-', linewidth=2, label='Seller A', alpha=0.7)
    ax2.plot(history_df['iteration'], history_df['B_ad'],
             'r-', linewidth=2, label='Seller B', alpha=0.7)

    nash_ad_A = nash_result['nash_equilibrium']['seller_A']['ad']
    nash_ad_B = nash_result['nash_equilibrium']['seller_B']['ad']

    ax2.plot(final_iter, nash_ad_A, 'b*', markersize=20, label='Nash A')
    ax2.plot(final_iter, nash_ad_B, 'r*', markersize=20, label='Nash B')
    ax2.axhline(nash_ad_A, color='blue', linestyle='--', alpha=0.3)
    ax2.axhline(nash_ad_B, color='red', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Ad Budget (€)')
    ax2.set_title('Ad Budget Evolution Toward Nash Equilibrium', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # SUBPLOT 3: Profit Evolution
    ax3 = axes[1, 0]
    ax3.plot(history_df['iteration'], history_df['A_profit'],
             'b-', linewidth=2, label='Seller A', alpha=0.7)
    ax3.plot(history_df['iteration'], history_df['B_profit'],
             'r-', linewidth=2, label='Seller B', alpha=0.7)

    nash_profit_A = nash_result['nash_equilibrium']['seller_A']['profit']
    nash_profit_B = nash_result['nash_equilibrium']['seller_B']['profit']

    ax3.plot(final_iter, nash_profit_A, 'b*', markersize=20, label='Nash A')
    ax3.plot(final_iter, nash_profit_B, 'r*', markersize=20, label='Nash B')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Break-even')

    # Annotate final profits
    ax3.text(final_iter * 0.95, nash_profit_A, f'€{nash_profit_A:.2f}',
             ha='right', va='bottom', fontsize=10, color='blue')
    ax3.text(final_iter * 0.95, nash_profit_B, f'€{nash_profit_B:.2f}',
             ha='right', va='top', fontsize=10, color='red')

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Profit (€)')
    ax3.set_title('Profit Evolution Toward Nash Equilibrium', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # SUBPLOT 4: Convergence Metric
    ax4 = axes[1, 1]
    if 'change' in history_df.columns:
        convergence_data = history_df['change'].replace(0, np.nan)
        ax4.semilogy(history_df['iteration'], convergence_data,
                     'g-', linewidth=2, label='Strategy Change')

        threshold = 0.01  # Default threshold
        ax4.axhline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold})')
        ax4.text(final_iter * 0.5, threshold * 1.5, f'Threshold = {threshold}',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Calculate convergence metric from price/ad changes
        price_change_A = history_df['A_price'].diff().abs()
        price_change_B = history_df['B_price'].diff().abs()
        ad_change_A = history_df['A_ad'].diff().abs()
        ad_change_B = history_df['B_ad'].diff().abs()

        total_change = price_change_A + price_change_B + ad_change_A + ad_change_B
        total_change = total_change.replace(0, np.nan)

        ax4.semilogy(history_df['iteration'], total_change,
                     'g-', linewidth=2, label='Total Strategy Change')

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Strategy Change (log scale)')
    ax4.set_title('Convergence Speed (Strategy Changes)', fontweight='bold')
    ax4.text(0.5, 0.02, 'Lower = more stable', transform=ax4.transAxes,
            ha='center', fontsize=9, style='italic')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # SUBPLOT 5: Seller A Strategy Trajectory
    ax5 = axes[2, 0]

    # Create color gradient
    n_points = len(history_df)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_points))

    # Plot trajectory with gradient
    for i in range(n_points - 1):
        ax5.plot(history_df['A_price'].iloc[i:i+2],
                history_df['A_ad'].iloc[i:i+2],
                color=colors[i], linewidth=2, alpha=0.7)

    # Mark start and end
    ax5.plot(history_df['A_price'].iloc[0], history_df['A_ad'].iloc[0],
            'go', markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax5.plot(nash_price_A, nash_ad_A,
            'r*', markersize=25, label='Nash Equilibrium', markeredgecolor='darkred', markeredgewidth=1)

    # Annotations
    ax5.annotate('Start', xy=(history_df['A_price'].iloc[0], history_df['A_ad'].iloc[0]),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color='darkgreen')
    ax5.annotate('Nash', xy=(nash_price_A, nash_ad_A),
                xytext=(10, -10), textcoords='offset points', fontsize=10, color='darkred')

    ax5.set_xlabel('Price (€)')
    ax5.set_ylabel('Ad Budget (€)')
    ax5.set_title('Seller A: Strategy Path to Nash Equilibrium', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # SUBPLOT 6: Seller B Strategy Trajectory
    ax6 = axes[2, 1]

    colors = plt.cm.Reds(np.linspace(0.3, 1.0, n_points))

    for i in range(n_points - 1):
        ax6.plot(history_df['B_price'].iloc[i:i+2],
                history_df['B_ad'].iloc[i:i+2],
                color=colors[i], linewidth=2, alpha=0.7)

    ax6.plot(history_df['B_price'].iloc[0], history_df['B_ad'].iloc[0],
            'go', markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax6.plot(nash_price_B, nash_ad_B,
            'r*', markersize=25, label='Nash Equilibrium', markeredgecolor='darkred', markeredgewidth=1)

    ax6.annotate('Start', xy=(history_df['B_price'].iloc[0], history_df['B_ad'].iloc[0]),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color='darkgreen')
    ax6.annotate('Nash', xy=(nash_price_B, nash_ad_B),
                xytext=(10, -10), textcoords='offset points', fontsize=10, color='darkred')

    ax6.set_xlabel('Price (€)')
    ax6.set_ylabel('Ad Budget (€)')
    ax6.set_title('Seller B: Strategy Path to Nash Equilibrium', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Nash equilibrium visualization saved to: {save_path}")
    plt.show()


def visualize_profit_comparison(nash_result, save_path='profit_comparison.png'):
    """
    Compare profits at initial strategies vs Nash equilibrium.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get initial profits from nash_result
    initial_profit_A = nash_result['initial_state']['seller_A']['profit']
    initial_profit_B = nash_result['initial_state']['seller_B']['profit']

    # Get Nash profits
    nash_profit_A = nash_result['nash_equilibrium']['seller_A']['profit']
    nash_profit_B = nash_result['nash_equilibrium']['seller_B']['profit']

    # Calculate changes
    change_A = nash_profit_A - initial_profit_A
    change_B = nash_profit_B - initial_profit_B
    pct_change_A = (change_A / abs(initial_profit_A) * 100) if initial_profit_A != 0 else 0
    pct_change_B = (change_B / abs(initial_profit_B) * 100) if initial_profit_B != 0 else 0
    total_change = change_A + change_B

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    sellers = ['Seller A', 'Seller B']
    x = np.arange(len(sellers))
    width = 0.35

    initial_profits = [initial_profit_A, initial_profit_B]
    nash_profits = [nash_profit_A, nash_profit_B]

    bars1 = ax.bar(x - width/2, initial_profits, width, label='Initial Strategy',
                   color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, nash_profits, width, label='Nash Equilibrium',
                   color=['darkblue', 'darkred'], alpha=0.9, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'€{height:.2f}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('Sellers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit (€)', fontsize=12, fontweight='bold')
    ax.set_title('Profit Comparison: Initial vs Nash Equilibrium',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sellers)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=1, linestyle='-')

    # Add insights text box
    insights_text = (
        f"Seller A profit change: {'+' if change_A >= 0 else ''}€{change_A:.2f} "
        f"({'+' if pct_change_A >= 0 else ''}{pct_change_A:.1f}%)\n"
        f"Seller B profit change: {'+' if change_B >= 0 else ''}€{change_B:.2f} "
        f"({'+' if pct_change_B >= 0 else ''}{pct_change_B:.1f}%)\n"
        f"Total market profit change: {'+' if total_change >= 0 else ''}€{total_change:.2f}"
    )

    ax.text(0.5, 0.98, insights_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Profit comparison saved to: {save_path}")
    plt.close()


def visualize_nash_on_profit_landscape(market, seller_A, seller_B, nash_result,
                                       save_path='nash_on_landscape.png'):
    """
    Show Nash equilibrium point on profit landscape heatmap.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print("Generating profit landscape visualization...")

    # Get Nash equilibrium values
    nash_price_A = nash_result['nash_equilibrium']['seller_A']['price']
    nash_ad_A = nash_result['nash_equilibrium']['seller_A']['ad']
    nash_price_B = nash_result['nash_equilibrium']['seller_B']['price']
    nash_ad_B = nash_result['nash_equilibrium']['seller_B']['ad']

    # Define search ranges based on seller costs
    price_min_A = seller_A.production_cost * 1.01
    price_max_A = seller_A.production_cost * 2.5
    price_range_A = np.linspace(price_min_A, price_max_A, 30)

    price_min_B = seller_B.production_cost * 1.01
    price_max_B = seller_B.production_cost * 2.5
    price_range_B = np.linspace(price_min_B, price_max_B, 30)

    ad_range = np.linspace(0, 2000, 30)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Nash Equilibrium on Profit Landscapes', fontsize=16, fontweight='bold')

    # Seller A landscape (with Seller B at Nash)
    ax1 = axes[0]
    profit_matrix_A = np.zeros((len(ad_range), len(price_range_A)))

    # Save original states
    orig_price_A = seller_A.price
    orig_ad_A = seller_A.advertising_budget
    orig_price_B = seller_B.price
    orig_ad_B = seller_B.advertising_budget

    # Fix seller B at Nash
    seller_B.price = nash_price_B
    seller_B.advertising_budget = nash_ad_B

    for i, ad in enumerate(ad_range):
        for j, price in enumerate(price_range_A):
            seller_A.price = price
            seller_A.advertising_budget = ad
            profit_matrix_A[i, j] = market.calculate_profit(seller_A, seller_B)

    im1 = ax1.contourf(price_range_A, ad_range, profit_matrix_A, levels=20, cmap='Blues')
    plt.colorbar(im1, ax=ax1, label='Profit (€)')

    ax1.plot(nash_price_A, nash_ad_A, 'rX', markersize=25, markeredgewidth=3,
            markeredgecolor='darkred', label='Nash Equilibrium')

    ax1.annotate('Nash Equilibrium\nOptimal response to B\'s strategy',
                xy=(nash_price_A, nash_ad_A), xytext=(20, 20),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))

    ax1.set_xlabel('Price (€)', fontsize=12)
    ax1.set_ylabel('Ad Budget (€)', fontsize=12)
    ax1.set_title('Seller A Profit Landscape\n(Seller B at Nash)', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Seller B landscape (with Seller A at Nash)
    ax2 = axes[1]
    profit_matrix_B = np.zeros((len(ad_range), len(price_range_B)))

    # Fix seller A at Nash
    seller_A.price = nash_price_A
    seller_A.advertising_budget = nash_ad_A

    for i, ad in enumerate(ad_range):
        for j, price in enumerate(price_range_B):
            seller_B.price = price
            seller_B.advertising_budget = ad
            profit_matrix_B[i, j] = market.calculate_profit(seller_B, seller_A)

    im2 = ax2.contourf(price_range_B, ad_range, profit_matrix_B, levels=20, cmap='Reds')
    plt.colorbar(im2, ax=ax2, label='Profit (€)')

    ax2.plot(nash_price_B, nash_ad_B, 'bX', markersize=25, markeredgewidth=3,
            markeredgecolor='darkblue', label='Nash Equilibrium')

    ax2.annotate('Nash Equilibrium\nOptimal response to A\'s strategy',
                xy=(nash_price_B, nash_ad_B), xytext=(20, 20),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='blue', lw=2))

    ax2.set_xlabel('Price (€)', fontsize=12)
    ax2.set_ylabel('Ad Budget (€)', fontsize=12)
    ax2.set_title('Seller B Profit Landscape\n(Seller A at Nash)', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Restore original states
    seller_A.price = orig_price_A
    seller_A.advertising_budget = orig_ad_A
    seller_B.price = orig_price_B
    seller_B.advertising_budget = orig_ad_B

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Nash equilibrium landscape visualization saved to: {save_path}")
    plt.close()


# Add these functions to GameTheorySimulation.py

def analyze_nash_equilibrium(market, seller_A, seller_B, nash_result, initial_state):
    """
    Perform comprehensive analysis of Nash equilibrium properties.

    Args:
        market: Market instance
        seller_A: Seller A instance
        seller_B: Seller B instance
        nash_result: Dictionary containing Nash equilibrium results
        initial_state: Dictionary with initial strategies and profits

    Returns:
        Dictionary with comprehensive analysis insights
    """
    analysis = {}

    # Extract data from nash_result
    converged = nash_result.get('converged', False)
    iterations = nash_result.get('iterations', 0)
    nash_equilibrium = nash_result.get('nash_equilibrium', {})

    # a) Convergence Analysis
    analysis['convergence'] = {
        'converged': converged,
        'iterations': iterations,
        'speed': 'fast' if iterations < 10 else ('medium' if iterations < 30 else 'slow'),
        'success': converged
    }

    # b) Strategy Changes
    initial_A = initial_state.get('seller_A', {})
    initial_B = initial_state.get('seller_B', {})
    nash_A = nash_equilibrium.get('seller_A', {})
    nash_B = nash_equilibrium.get('seller_B', {})

    # Calculate changes for Seller A
    price_change_A = nash_A.get('price', 0) - initial_A.get('price', 0)
    price_pct_A = (price_change_A / initial_A.get('price', 1)) * 100 if initial_A.get('price', 0) != 0 else 0
    ad_change_A = nash_A.get('ad', 0) - initial_A.get('ad_budget', 0)
    initial_ad_A = initial_A.get('ad_budget', 0)
    ad_pct_A = (ad_change_A / initial_ad_A) * 100 if initial_ad_A != 0 else (100 if ad_change_A > 0 else 0)

    # Calculate changes for Seller B
    price_change_B = nash_B.get('price', 0) - initial_B.get('price', 0)
    price_pct_B = (price_change_B / initial_B.get('price', 1)) * 100 if initial_B.get('price', 0) != 0 else 0
    ad_change_B = nash_B.get('ad', 0) - initial_B.get('ad_budget', 0)
    initial_ad_B = initial_B.get('ad_budget', 0)
    ad_pct_B = (ad_change_B / initial_ad_B) * 100 if initial_ad_B != 0 else (100 if ad_change_B > 0 else 0)

    total_change_A = abs(price_pct_A) + abs(ad_pct_A)
    total_change_B = abs(price_pct_B) + abs(ad_pct_B)

    analysis['strategy_changes'] = {
        'seller_A': {
            'price_change': price_change_A,
            'price_pct_change': price_pct_A,
            'ad_change': ad_change_A,
            'ad_pct_change': ad_pct_A,
            'total_change_magnitude': total_change_A
        },
        'seller_B': {
            'price_change': price_change_B,
            'price_pct_change': price_pct_B,
            'ad_change': ad_change_B,
            'ad_pct_change': ad_pct_B,
            'total_change_magnitude': total_change_B
        },
        'bigger_mover': 'seller_A' if total_change_A > total_change_B else 'seller_B'
    }

    # c) Profit Analysis
    initial_profit_A = initial_state.get('profits', {}).get('seller_A', 0)
    initial_profit_B = initial_state.get('profits', {}).get('seller_B', 0)
    nash_profit_A = nash_A.get('profit', 0)
    nash_profit_B = nash_B.get('profit', 0)

    profit_change_A = nash_profit_A - initial_profit_A
    profit_pct_A = (profit_change_A / abs(initial_profit_A)) * 100 if initial_profit_A != 0 else 0
    profit_change_B = nash_profit_B - initial_profit_B
    profit_pct_B = (profit_change_B / abs(initial_profit_B)) * 100 if initial_profit_B != 0 else 0

    total_profit_initial = initial_profit_A + initial_profit_B
    total_profit_nash = nash_profit_A + nash_profit_B
    total_profit_change = total_profit_nash - total_profit_initial

    analysis['profit_analysis'] = {
        'seller_A': {
            'initial_profit': initial_profit_A,
            'nash_profit': nash_profit_A,
            'change': profit_change_A,
            'pct_change': profit_pct_A,
            'is_positive': nash_profit_A > 0
        },
        'seller_B': {
            'initial_profit': initial_profit_B,
            'nash_profit': nash_profit_B,
            'change': profit_change_B,
            'pct_change': profit_pct_B,
            'is_positive': nash_profit_B > 0
        },
        'winner': 'seller_A' if profit_change_A > profit_change_B else 'seller_B',
        'total_market': {
            'initial': total_profit_initial,
            'nash': total_profit_nash,
            'change': total_profit_change
        },
        'both_sustainable': nash_profit_A > 0 and nash_profit_B > 0
    }

    # d) Market Dynamics
    price_gap = nash_A.get('price', 0) - nash_B.get('price', 0)
    ad_gap = nash_A.get('ad', 0) - nash_B.get('ad', 0)

    analysis['market_dynamics'] = {
        'price_gap': price_gap,
        'premium_seller': 'seller_A' if price_gap > 0 else 'seller_B',
        'discount_seller': 'seller_B' if price_gap > 0 else 'seller_A',
        'ad_gap': ad_gap,
        'higher_ad_spender': 'seller_A' if ad_gap > 0 else 'seller_B',
        'market_stable': nash_profit_A > 0 and nash_profit_B > 0,
        'price_competition_intensity': 'high' if abs(price_gap) < 5 else ('medium' if abs(price_gap) < 15 else 'low')
    }

    # e) Theoretical Insights
    analysis['theoretical_insights'] = {
        'is_best_response': converged,  # If algorithm converged, it's a mutual best response
        'equilibrium_type': 'Nash equilibrium' if converged else 'Non-equilibrium',
        'cooperative_potential': total_profit_nash < total_profit_initial * 1.1,  # Could cooperate for better outcome
        'competitive_pressure': abs(price_gap) / max(nash_A.get('price', 1), nash_B.get('price', 1)) < 0.1
    }

    return analysis


def generate_nash_report(analysis_dict, nash_result, output_path='nash_equilibrium_report.txt'):
    """
    Generate human-readable text report from analysis.

    Args:
        analysis_dict: Analysis dictionary from analyze_nash_equilibrium()
        nash_result: Original Nash equilibrium result
        output_path: Path to save the report
    """
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("NASH EQUILIBRIUM ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 1. Convergence Summary
    report_lines.append("1. CONVERGENCE SUMMARY")
    report_lines.append("-" * 80)
    conv = analysis_dict['convergence']
    report_lines.append(f"Algorithm converged: {'Yes' if conv['converged'] else 'No'}")
    report_lines.append(f"Iterations required: {conv['iterations']}")
    report_lines.append(f"Convergence speed: {conv['speed'].capitalize()}")
    report_lines.append("")

    # 2. Nash Equilibrium Strategies
    report_lines.append("2. NASH EQUILIBRIUM STRATEGIES")
    report_lines.append("-" * 80)

    nash_equilibrium = nash_result.get('nash_equilibrium', {})
    changes = analysis_dict['strategy_changes']
    profits = analysis_dict['profit_analysis']

    # Seller A
    report_lines.append("Seller A:")
    nash_A = nash_equilibrium.get('seller_A', {})
    change_A = changes['seller_A']
    profit_A = profits['seller_A']

    report_lines.append(f"  Price: £{nash_A.get('price', 0):.2f} "
                       f"(change: {change_A['price_pct_change']:+.1f}% from initial)")
    report_lines.append(f"  Ad Budget: £{nash_A.get('ad', 0):.2f} "
                       f"(change: {change_A['ad_pct_change']:+.1f}% from initial)")
    report_lines.append(f"  Profit: £{profit_A['nash_profit']:.2f} "
                       f"(change: {profit_A['pct_change']:+.1f}% from initial)")
    report_lines.append("")

    # Seller B
    report_lines.append("Seller B:")
    nash_B = nash_equilibrium.get('seller_B', {})
    change_B = changes['seller_B']
    profit_B = profits['seller_B']

    report_lines.append(f"  Price: £{nash_B.get('price', 0):.2f} "
                       f"(change: {change_B['price_pct_change']:+.1f}% from initial)")
    report_lines.append(f"  Ad Budget: £{nash_B.get('ad', 0):.2f} "
                       f"(change: {change_B['ad_pct_change']:+.1f}% from initial)")
    report_lines.append(f"  Profit: £{profit_B['nash_profit']:.2f} "
                       f"(change: {profit_B['pct_change']:+.1f}% from initial)")
    report_lines.append("")

    # 3. Market Dynamics
    report_lines.append("3. MARKET DYNAMICS AT EQUILIBRIUM")
    report_lines.append("-" * 80)
    dynamics = analysis_dict['market_dynamics']

    price_diff = dynamics['price_gap']
    premium = dynamics['premium_seller'].replace('seller_', 'Seller ')
    discount = dynamics['discount_seller'].replace('seller_', 'Seller ')

    report_lines.append(f"Price difference: £{abs(price_diff):.2f} ({premium} is Premium)")
    report_lines.append(f"Ad budget difference: £{abs(dynamics['ad_gap']):.2f} "
                       f"({dynamics['higher_ad_spender'].replace('seller_', 'Seller ')} spends more)")
    report_lines.append(f"Total market profit: £{profits['total_market']['nash']:.2f}")
    report_lines.append(f"Market stability: {'Stable' if dynamics['market_stable'] else 'Unstable'} "
                       f"({'both profitable' if profits['both_sustainable'] else 'at least one seller unprofitable'})")
    report_lines.append("")

    # 4. Key Insights
    report_lines.append("4. KEY INSIGHTS")
    report_lines.append("-" * 80)

    insights = []

    # Insight about profit winner
    winner = profits['winner'].replace('seller_', 'Seller ')
    winner_change = profits[profits['winner']]['pct_change']
    if winner_change > 0:
        insights.append(f"{winner} improved profit by {abs(winner_change):.1f}% at Nash equilibrium")
    else:
        insights.append(f"{winner} had better profit performance (smaller loss) at Nash equilibrium")

    # Insight about convergence
    if conv['speed'] == 'fast':
        insights.append(f"Equilibrium reached quickly ({conv['iterations']} iterations)")
    elif conv['speed'] == 'medium':
        insights.append(f"Equilibrium reached at moderate pace ({conv['iterations']} iterations)")
    else:
        insights.append(f"Equilibrium took longer to reach ({conv['iterations']} iterations)")

    # Insight about price competition
    if dynamics['price_competition_intensity'] == 'high':
        insights.append("Price gap narrowed significantly, indicating intense price competition")
    elif dynamics['price_competition_intensity'] == 'medium':
        insights.append("Moderate price differentiation maintained at equilibrium")
    else:
        insights.append("Sellers maintained significant price differentiation")

    # Insight about advertising
    if change_A['ad_pct_change'] < 0 and change_B['ad_pct_change'] < 0:
        insights.append("Both sellers reduced ad spending in equilibrium (cost optimization)")
    elif change_A['ad_pct_change'] > 0 and change_B['ad_pct_change'] > 0:
        insights.append("Both sellers increased ad spending (escalating ad competition)")
    else:
        insights.append("Asymmetric advertising strategies emerged at equilibrium")

    # Insight about stability
    if profits['both_sustainable']:
        insights.append("Nash equilibrium is stable with positive profits for both sellers")
    else:
        insights.append("Market sustainability concern: at least one seller has negative profits")

    # Insight about strategy changes
    bigger_mover = changes['bigger_mover'].replace('seller_', 'Seller ')
    insights.append(f"{bigger_mover} made larger strategic adjustments to reach equilibrium")

    for i, insight in enumerate(insights, 1):
        report_lines.append(f"  {i}. {insight}")
    report_lines.append("")

    # 5. Game Theory Validation (placeholder - will be filled by verify_nash_property)
    report_lines.append("5. GAME THEORY VALIDATION")
    report_lines.append("-" * 80)
    report_lines.append("Nash property verification: [To be completed by verify_nash_property()]")
    report_lines.append("")

    report_lines.append("=" * 80)

    # Write to file
    report_text = '\n'.join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    # Also print to console
    print(report_text)

    return report_text


def verify_nash_property(market, seller_A, seller_B, nash_result, deviation_size=0.05):
    """
    Verify that Nash equilibrium truly is an equilibrium:
    Neither seller can improve by deviating.

    Args:
        market: Market instance
        seller_A: Seller A instance
        seller_B: Seller B instance
        nash_result: Dictionary containing Nash equilibrium results
        deviation_size: Size of deviation to test (default 5%)

    Returns:
        Dictionary with verification results
    """
    nash_equilibrium = nash_result.get('nash_equilibrium', {})

    nash_A = nash_equilibrium.get('seller_A', {})
    nash_B = nash_equilibrium.get('seller_B', {})
    base_profit_A = nash_A.get('profit', 0)
    base_profit_B = nash_B.get('profit', 0)

    deviations_tested = []
    is_nash = True

    # Test deviations for Seller A
    test_scenarios_A = [
        ('price_increase', nash_A['price'] * (1 + deviation_size), nash_A['ad']),
        ('price_decrease', nash_A['price'] * (1 - deviation_size), nash_A['ad']),
        ('ad_increase', nash_A['price'], nash_A['ad'] * (1 + deviation_size)),
        ('ad_decrease', nash_A['price'], max(0, nash_A['ad'] * (1 - deviation_size)))
    ]

    for scenario_name, test_price, test_ad in test_scenarios_A:
        # Set test strategy for A
        seller_A.update_strategy(test_price, test_ad)
        seller_B.update_strategy(nash_B['price'], nash_B['ad'])

        # Calculate profit with this test strategy
        test_profit_A = market.calculate_profit(seller_A, seller_B)

        profit_improvement = test_profit_A - base_profit_A

        deviations_tested.append({
            'seller': 'A',
            'deviation': scenario_name,
            'test_price': test_price,
            'test_ad': test_ad,
            'base_profit': base_profit_A,
            'test_profit': test_profit_A,
            'improvement': profit_improvement,
            'profitable': profit_improvement > 0.01  # Small threshold for numerical stability
        })

        if profit_improvement > 0.01:
            is_nash = False

    # Test deviations for Seller B
    test_scenarios_B = [
        ('price_increase', nash_B['price'] * (1 + deviation_size), nash_B['ad']),
        ('price_decrease', nash_B['price'] * (1 - deviation_size), nash_B['ad']),
        ('ad_increase', nash_B['price'], nash_B['ad'] * (1 + deviation_size)),
        ('ad_decrease', nash_B['price'], max(0, nash_B['ad'] * (1 - deviation_size)))
    ]

    for scenario_name, test_price, test_ad in test_scenarios_B:
        # Set test strategy for B
        seller_A.update_strategy(nash_A['price'], nash_A['ad'])
        seller_B.update_strategy(test_price, test_ad)

        # Calculate profit with this test strategy
        test_profit_B = market.calculate_profit(seller_B, seller_A)

        profit_improvement = test_profit_B - base_profit_B

        deviations_tested.append({
            'seller': 'B',
            'deviation': scenario_name,
            'test_price': test_price,
            'test_ad': test_ad,
            'base_profit': base_profit_B,
            'test_profit': test_profit_B,
            'improvement': profit_improvement,
            'profitable': profit_improvement > 0.01
        })

        if profit_improvement > 0.01:
            is_nash = False

    # Reset to Nash strategies
    seller_A.update_strategy(nash_A['price'], nash_A['ad'])
    seller_B.update_strategy(nash_B['price'], nash_B['ad'])

    verification_result = {
        'is_nash_equilibrium': is_nash,
        'deviations_tested': deviations_tested,
        'num_tests': len(deviations_tested),
        'profitable_deviations_found': sum(1 for d in deviations_tested if d['profitable'])
    }

    return verification_result


def execute_complete_nash_analysis(market, seller_A, seller_B, nash_result, initial_state):
    """
    Execute complete Nash equilibrium analysis pipeline.

    Args:
        market: Market instance
        seller_A: Seller A instance
        seller_B: Seller B instance
        nash_result: Nash equilibrium computation result
        initial_state: Initial strategies and profits

    Returns:
        Dictionary containing all analysis results
    """
    print("\n" + "=" * 80)
    print("EXECUTING COMPREHENSIVE NASH EQUILIBRIUM ANALYSIS")
    print("=" * 80 + "\n")

    # Step 1: Analyze Nash equilibrium
    print("Step 1: Analyzing Nash equilibrium properties...")
    analysis = analyze_nash_equilibrium(market, seller_A, seller_B, nash_result, initial_state)
    print("✓ Analysis complete\n")

    # Step 2: Verify Nash property
    print("Step 2: Verifying Nash equilibrium property...")
    verification = verify_nash_property(market, seller_A, seller_B, nash_result)
    print(f"✓ Verification complete: {'VERIFIED' if verification['is_nash_equilibrium'] else 'NOT VERIFIED'}\n")

    # Step 3: Generate report
    print("Step 3: Generating detailed report...")
    report_text = generate_nash_report(analysis, nash_result)

    # Append verification results to report
    with open('nash_equilibrium_report.txt', 'a') as f:
        f.write("\n\n")
        f.write("NASH PROPERTY VERIFICATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Result: Nash equilibrium {'VERIFIED' if verification['is_nash_equilibrium'] else 'NOT VERIFIED'}\n")
        f.write(f"Tests performed: {verification['num_tests']}\n")
        f.write(f"Profitable deviations found: {verification['profitable_deviations_found']}\n\n")

        if verification['profitable_deviations_found'] > 0:
            f.write("Profitable deviations detected:\n")
            for dev in verification['deviations_tested']:
                if dev['profitable']:
                    f.write(f"  - Seller {dev['seller']}: {dev['deviation']} "
                           f"(profit improvement: £{dev['improvement']:.2f})\n")
        else:
            f.write("No profitable unilateral deviations found. Nash equilibrium confirmed.\n")

    print("✓ Report saved to 'nash_equilibrium_report.txt'\n")

    # Step 4: Print key findings
    print("=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    print(f"Convergence: {analysis['convergence']['speed'].upper()} "
          f"({analysis['convergence']['iterations']} iterations)")
    print(f"Nash Equilibrium Status: {'VERIFIED ✓' if verification['is_nash_equilibrium'] else 'NOT VERIFIED ✗'}")
    print(f"Market Stability: {'STABLE' if analysis['market_dynamics']['market_stable'] else 'UNSTABLE'}")
    print(f"Profit Winner: {analysis['profit_analysis']['winner'].replace('seller_', 'Seller ')}")
    print("=" * 80 + "\n")

    return {
        'analysis': analysis,
        'verification': verification,
        'report_path': 'nash_equilibrium_report.txt'
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("GAME THEORY SIMULATION - NASH EQUILIBRIUM FINDER")
    print("Laptop Market Competition: Apple MacBook vs Dell Inspiron")
    print("="*80)

    # ------------------------------------------------------------------------
    # INITIAL MARKET STATE
    # ------------------------------------------------------------------------

    print("\n" + "-"*80)
    print("INITIAL MARKET STATE")
    print("-"*80)

    print(f"\nMarket Parameters:")
    print(f"  Total Market Size: {market.total_market_size:,} customers")
    print(f"  Price Sensitivity: {market.beta:.4f}")
    print(f"  Advertising Effectiveness: {market.alpha:.4f}")

    seller_names = list(sellers.keys())

    # Check seller balance and select the two most balanced
    print("\n" + "="*80)
    print("SELLER SELECTION FOR NASH EQUILIBRIUM")
    print("="*80)
    print("\nAvailable sellers:")
    for name in seller_names:
        s = sellers[name]
        print(f"  {name}: base_demand={s.base_demand:.2f}, price=€{s.price:.2f}, ad=€{s.advertising_budget:.0f}")

    # Find the two sellers with most similar base_demand
    if len(sellers) >= 3:
        # Calculate demand ratios between all pairs
        best_ratio = float('inf')
        best_pair = None

        for i, name1 in enumerate(seller_names):
            for j, name2 in enumerate(seller_names[i+1:], start=i+1):
                s1, s2 = sellers[name1], sellers[name2]
                ratio = max(s1.base_demand, s2.base_demand) / min(s1.base_demand, s2.base_demand)
                print(f"\n{name1} vs {name2}: demand ratio = {ratio:.2f}x")
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (name1, name2)

        print(f"\n✓ Selected most balanced pair: {best_pair[0]} vs {best_pair[1]} (ratio: {best_ratio:.2f}x)")
        seller_A = sellers[best_pair[0]]
        seller_B = sellers[best_pair[1]]
    else:
        # Only 2 sellers available, use them
        seller_A = sellers[seller_names[0]]
        seller_B = sellers[seller_names[1]]
        print(f"\n⚠️  Only 2 sellers available, using: {seller_names[0]} vs {seller_names[1]}")

    print("="*80)

    print(f"\n{seller_A.name}:")
    print(f"  Brand Value: {seller_A.brand_value:.2f}")
    print(f"  Production Cost: £{seller_A.production_cost:.2f}")
    print(f"  Initial Price: £{seller_A.price:.2f}")
    print(f"  Initial Ad Budget: £{seller_A.advertising_budget:.0f}")

    print(f"\n{seller_B.name}:")
    print(f"  Brand Value: {seller_B.brand_value:.2f}")
    print(f"  Production Cost: £{seller_B.production_cost:.2f}")
    print(f"  Initial Price: £{seller_B.price:.2f}")
    print(f"  Initial Ad Budget: £{seller_B.advertising_budget:.0f}")

    # ------------------------------------------------------------------------
    # RESET TO REASONABLE INITIAL STRATEGIES
    # ------------------------------------------------------------------------

    print("\n" + "="*80)
    print("RESETTING TO REASONABLE INITIAL STRATEGIES")
    print("="*80)
    print(f"\nOriginal strategies (may be unrealistic):")
    print(f"  {seller_A.name}: Price=€{seller_A.price:.2f}, Ad=€{seller_A.advertising_budget:.0f}")
    print(f"  {seller_B.name}: Price=€{seller_B.price:.2f}, Ad=€{seller_B.advertising_budget:.0f}")

    # Reset to reasonable values
    # Use price slightly above cost, and modest ad budget
    seller_A.price = seller_A.production_cost * 1.50  # 50% markup
    seller_A.advertising_budget = 500  # Modest ad budget

    seller_B.price = seller_B.production_cost * 1.50  # 50% markup
    seller_B.advertising_budget = 500  # Modest ad budget

    print(f"\nReset to:")
    print(f"  {seller_A.name}: Price=€{seller_A.price:.2f}, Ad=€{seller_A.advertising_budget:.0f}")
    print(f"  {seller_B.name}: Price=€{seller_B.price:.2f}, Ad=€{seller_B.advertising_budget:.0f}")

    # ------------------------------------------------------------------------
    # FIND NASH EQUILIBRIUM
    # ------------------------------------------------------------------------

    results = find_nash_equilibrium(
        market, seller_A, seller_B,
        max_iterations=50,
        convergence_threshold=0.001,  # Stricter threshold
        price_step=0.02,  # Coarser for faster convergence
        ad_step=100,  # Coarser for faster convergence
        verbose=True
    )

    # ------------------------------------------------------------------------
    # CONVERT HISTORY TO DATAFRAME
    # ------------------------------------------------------------------------

    df_history = convert_history_to_dataframe(results['history'])

    print("\n" + "="*80)
    print("CONVERGENCE HISTORY (DataFrame)")
    print("="*80)

    # Show first 10 rows
    print("\nFirst 10 iterations:")
    print(df_history.head(10).to_string(index=False))

    # Show last 10 rows if more than 10 iterations
    if len(df_history) > 10:
        print(f"\n... ({len(df_history) - 10} more iterations) ...\n")
        print("Last 10 iterations:")
        print(df_history.tail(10).to_string(index=False))

    # ------------------------------------------------------------------------
    # SUMMARY STATISTICS
    # ------------------------------------------------------------------------

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    initial = results['initial_state']
    final = results['nash_equilibrium']

    print(f"\nTotal Iterations: {results['iterations']}")
    print(f"Converged: {'Yes ✓' if results['converged'] else 'No ⚠'}")
    print(f"Final Convergence Metric: {results['convergence_metric']:.6f}")

    print(f"\n{seller_A.name} - Initial vs Nash Equilibrium:")
    print(f"  Price:      £{initial['seller_A']['price']:.2f} → "
          f"£{final['seller_A']['price']:.2f} "
          f"(Δ: £{final['seller_A']['price'] - initial['seller_A']['price']:.2f})")
    print(f"  Ad Budget:  £{initial['seller_A']['ad']:.0f} → "
          f"£{final['seller_A']['ad']:.0f} "
          f"(Δ: £{final['seller_A']['ad'] - initial['seller_A']['ad']:.0f})")
    print(f"  Profit:     £{initial['seller_A']['profit']:.2f} → "
          f"£{final['seller_A']['profit']:.2f} "
          f"(Δ: £{final['seller_A']['profit'] - initial['seller_A']['profit']:.2f})")

    print(f"\n{seller_B.name} - Initial vs Nash Equilibrium:")
    print(f"  Price:      £{initial['seller_B']['price']:.2f} → "
          f"£{final['seller_B']['price']:.2f} "
          f"(Δ: £{final['seller_B']['price'] - initial['seller_B']['price']:.2f})")
    print(f"  Ad Budget:  £{initial['seller_B']['ad']:.0f} → "
          f"£{final['seller_B']['ad']:.0f} "
          f"(Δ: £{final['seller_B']['ad'] - initial['seller_B']['ad']:.0f})")
    print(f"  Profit:     £{initial['seller_B']['profit']:.2f} → "
          f"£{final['seller_B']['profit']:.2f} "
          f"(Δ: £{final['seller_B']['profit'] - initial['seller_B']['profit']:.2f})")

    print("\n" + "="*80)
    print("EXAM TIP: Remember Nash Equilibrium is where both players are")
    print("playing best responses - no one wants to deviate unilaterally!")
    print("="*80 + "\n")

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================

    print("\n" + "="*80)
    print("GENERATING NASH EQUILIBRIUM VISUALIZATIONS")
    print("="*80 + "\n")

    # Generate comprehensive 6-subplot convergence visualization
    print("[1/3] Creating Nash equilibrium convergence visualization...")
    visualize_nash_equilibrium(df_history, results, 'nash_equilibrium.png')

    # Generate profit comparison bar chart
    print("\n[2/3] Creating profit comparison visualization...")
    visualize_profit_comparison(results, 'profit_comparison.png')

    # Generate profit landscape with Nash equilibrium marked
    print("\n[3/3] Creating profit landscape with Nash equilibrium...")
    visualize_nash_on_profit_landscape(market, seller_A, seller_B, results,
                                      'nash_on_landscape.png')

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS SAVED!")
    print("="*80)
    print("Files created:")
    print("  ✓ nash_equilibrium.png - 6-subplot convergence analysis")
    print("  ✓ profit_comparison.png - Initial vs Nash profit comparison")
    print("  ✓ nash_on_landscape.png - Nash point on profit landscapes")
    print("="*80 + "\n")
    initial_state = {
        'seller_A': {
            'price': seller_A.price,
            'ad_budget': seller_A.advertising_budget
        },
        'seller_B': {
            'price': seller_B.price,
            'ad_budget': seller_B.advertising_budget
        },
        'profits': {
            'seller_A': seller_A.profit,
            'seller_B': seller_B.profit
        }
    }

    # Compute Nash equilibrium
    nash_result = find_nash_equilibrium(market, seller_A, seller_B)

    # Execute complete analysis
    complete_results = execute_complete_nash_analysis(
        market, seller_A, seller_B, nash_result, initial_state
    )

