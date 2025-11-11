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
                       price_step=0.01, ad_step=50, verbose=False):
    """
    Find best response strategy for one seller given opponent's fixed strategy.

    This is a KEY CONCEPT in game theory:
    - Best Response = Strategy that maximizes your profit given opponent's strategy
    - Nash Equilibrium occurs when both players are playing best responses

    Parameters:
    - responding_seller: Seller finding their best response
    - opponent_seller: Seller with fixed strategy
    - market: MarketModel object
    - price_step: Price increment for search (£0.01)
    - ad_step: Ad budget increment for search (£50)
    - verbose: Print search progress

    Returns:
    - Dictionary with best price, ad budget, and resulting profit
    """

    # Define search ranges
    # Price range: Cost + small margin to Cost + 100%
    min_price = responding_seller.production_cost * 1.01
    max_price = responding_seller.production_cost * 2.0

    # Ad budget range: £0 to £10,000
    min_ad = 0
    max_ad = 10000

    # Create search grid
    prices = np.arange(min_price, max_price, price_step)
    ad_budgets = np.arange(min_ad, max_ad, ad_step)

    # Track best strategy found
    best_profit = -np.inf
    best_price = None
    best_ad = None

    # Grid search over all combinations
    for price in prices:
        for ad_budget in ad_budgets:
            # Temporarily set responding seller's strategy
            original_price = responding_seller.price
            original_ad = responding_seller.advertising_budget

            responding_seller.price = price
            responding_seller.advertising_budget = ad_budget

            # Calculate profit with this strategy
            profit = market.calculate_profit(responding_seller, opponent_seller)

            # Update best if this is better
            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_ad = ad_budget

            # Restore original strategy
            responding_seller.price = original_price
            responding_seller.advertising_budget = original_ad

    if verbose:
        print(f"  Best response for {responding_seller.name}: "
              f"Price=£{best_price:.2f}, Ad=£{best_ad:.0f}, Profit=£{best_profit:.2f}")

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

    if verbose:
        print("\n" + "="*80)
        print("NASH EQUILIBRIUM FINDER - ITERATIVE BEST RESPONSE ALGORITHM")
        print("="*80)
        print(f"\nParameters:")
        print(f"  Max Iterations: {max_iterations}")
        print(f"  Convergence Threshold: {convergence_threshold}")
        print(f"  Price Step: £{price_step}")
        print(f"  Ad Budget Step: £{ad_step}")
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
            print(f"Iteration {iteration + 1}/{max_iterations}:")

        # --------------------------------------------------------------------
        # SELLER A'S TURN: Find best response to B's current strategy
        # --------------------------------------------------------------------

        if verbose:
            print(f"  {seller_A.name} finding best response to {seller_B.name}...")

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

        # --------------------------------------------------------------------
        # SELLER B'S TURN: Find best response to A's NEW strategy
        # --------------------------------------------------------------------

        if verbose:
            print(f"  {seller_B.name} finding best response to {seller_A.name}...")

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
        # Euclidean distance of strategy changes

        convergence_metric = np.sqrt(
            (seller_A.price - prev_A_price)**2 +
            (seller_A.advertising_budget - prev_A_ad)**2 +
            (seller_B.price - prev_B_price)**2 +
            (seller_B.advertising_budget - prev_B_ad)**2
        )

        # Add convergence metric to history
        history[-1]['change'] = convergence_metric

        # --------------------------------------------------------------------
        # PRINT ITERATION SUMMARY
        # --------------------------------------------------------------------

        if verbose:
            print(f"\n  Results:")
            print(f"    {seller_A.name}: Price=£{seller_A.price:.2f}, "
                  f"Ad=£{seller_A.advertising_budget:.0f}, Profit=£{profit_A:.2f}")
            print(f"    {seller_B.name}: Price=£{seller_B.price:.2f}, "
                  f"Ad=£{seller_B.advertising_budget:.0f}, Profit=£{profit_B:.2f}")
            print(f"    Strategy Change: {convergence_metric:.4f}")

        # --------------------------------------------------------------------
        # CHECK CONVERGENCE
        # --------------------------------------------------------------------

        if convergence_metric < convergence_threshold:
            converged = True
            if verbose:
                print(f"\n  ✓ CONVERGED! Change ({convergence_metric:.4f}) "
                      f"< Threshold ({convergence_threshold})")
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
    seller_A = sellers[seller_names[0]]  # First seller
    seller_B = sellers[seller_names[1]]

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
    # FIND NASH EQUILIBRIUM
    # ------------------------------------------------------------------------

    results = find_nash_equilibrium(
        market, seller_A, seller_B,
        max_iterations=50,
        convergence_threshold=0.01,
        price_step=0.01,
        ad_step=50,
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