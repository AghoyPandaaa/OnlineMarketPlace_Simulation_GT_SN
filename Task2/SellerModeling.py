import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("TASK II: SELLER MODELING AND ANALYSIS")
print("="*80)

# 1. Load the cleaned dataset
print("\n[1] Loading cleaned dataset...")
# Use absolute path relative to this file's location
data_path = Path(__file__).parent.parent / 'Data' / 'ProcessedData' / 'cleaned_online_retail_data.csv'
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# 2. Analyze products with multiple price points
print("\n[2] Analyzing products with price variations...")
product_analysis = df.groupby('StockCode').agg({
    'Price': ['mean', 'std', 'min', 'max', 'nunique'],
    'Quantity': ['sum', 'count'],
    'Description': 'first'
}).reset_index()

# Flatten multi-level columns
product_analysis.columns = ['StockCode', 'Avg_Price', 'Std_Price', 'Min_Price',
                             'Max_Price', 'Unique_Prices', 'Total_Quantity', 'Num_Transactions', 'Description']

# Filter products with at least 3 unique prices
products_with_variation = product_analysis[product_analysis['Unique_Prices'] >= 3].copy()
products_with_variation = products_with_variation.sort_values('Total_Quantity', ascending=False)

print(f"\nProducts with 3+ unique prices: {len(products_with_variation)}")
print("\nTop 20 products with price variations (by total quantity sold):")
print(products_with_variation.head(20)[['StockCode', 'Description', 'Min_Price', 'Max_Price', 'Unique_Prices', 'Total_Quantity']].to_string(index=False))

# 3. Select ONE popular product for simulation
print("\n[3] Selecting product for seller simulation...")
selected_product = products_with_variation.iloc[1]
selected_stock_code = selected_product['StockCode']

# Extract all transactions for this product
selected_product_data = df[df['StockCode'] == selected_stock_code].copy()

print(f"\n{'='*60}")
print("SELECTED PRODUCT DETAILS:")
print(f"{'='*60}")
print(f"Product Code: {selected_stock_code}")
print(f"Product Name: {selected_product['Description']}")
print(f"Price Range: €{selected_product['Min_Price']:.2f} - €{selected_product['Max_Price']:.2f}")
print(f"Unique Price Points: {int(selected_product['Unique_Prices'])}")
print(f"Total Transactions: {selected_product['Num_Transactions']}")
print(f"Total Quantity Sold: {int(selected_product['Total_Quantity'])}")
print(f"{'='*60}")

# 4. Create 2-3 sellers based on price tiers
print("\n[4] Creating sellers based on price quantiles...")

# Get unique prices and sort them
unique_prices = sorted(selected_product_data['Price'].unique())
print(f"Unique prices found: {len(unique_prices)} different price points")
print(f"Price range: €{min(unique_prices):.2f} to €{max(unique_prices):.2f}")

# If we have enough unique prices, create 3 sellers based on price ranges
# Otherwise, use transaction-based quantiles
if len(unique_prices) >= 3:
    # Use actual unique prices to define boundaries
    low_cutoff = unique_prices[len(unique_prices) // 3]
    high_cutoff = unique_prices[2 * len(unique_prices) // 3]
    print(f"\nUsing unique price boundaries:")
    print(f"  Low cutoff: €{low_cutoff:.2f}")
    print(f"  High cutoff: €{high_cutoff:.2f}")
else:
    # Fall back to transaction quantiles
    low_cutoff = selected_product_data['Price'].quantile(0.33)
    high_cutoff = selected_product_data['Price'].quantile(0.67)
    print(f"\nUsing transaction quantiles:")
    print(f"  33rd percentile: €{low_cutoff:.2f}")
    print(f"  67th percentile: €{high_cutoff:.2f}")

# Assign sellers based on price tiers
def assign_seller(price):
    if price < low_cutoff:
        return 'Seller_A'
    elif price < high_cutoff:
        return 'Seller_B'
    else:
        return 'Seller_C'

selected_product_data['Seller'] = selected_product_data['Price'].apply(assign_seller)

# Check how many sellers we actually have and their balance
actual_sellers = selected_product_data['Seller'].nunique()
seller_counts = selected_product_data['Seller'].value_counts()
print(f"\nNumber of sellers created: {actual_sellers}")
print("\nTransactions per seller:")
for seller, count in seller_counts.items():
    pct = (count / len(selected_product_data)) * 100
    print(f"  {seller}: {count} transactions ({pct:.1f}%)")

# If we only got 2 sellers, adjust boundaries to force 3
if actual_sellers < 3:
    print("\n⚠️  Only 2 sellers created. Adjusting boundaries to create 3 balanced sellers...")
    # Use different percentiles to ensure 3 groups
    low_cutoff = selected_product_data['Price'].quantile(0.25)
    high_cutoff = selected_product_data['Price'].quantile(0.75)

    def assign_seller_forced(price):
        if price <= low_cutoff:
            return 'Seller_A'
        elif price <= high_cutoff:
            return 'Seller_B'
        else:
            return 'Seller_C'

    selected_product_data['Seller'] = selected_product_data['Price'].apply(assign_seller_forced)
    actual_sellers = selected_product_data['Seller'].nunique()
    seller_counts = selected_product_data['Seller'].value_counts()
    print(f"After adjustment: {actual_sellers} sellers")
    print("\nAdjusted transactions per seller:")
    for seller, count in seller_counts.items():
        pct = (count / len(selected_product_data)) * 100
        print(f"  {seller}: {count} transactions ({pct:.1f}%)")

print("\nSeller Assignment:")
print(f"- Seller_A: Low-price/Discount seller (< €{low_cutoff:.2f})")
print(f"- Seller_B: Mid-price/Balanced seller (€{low_cutoff:.2f} - €{high_cutoff:.2f})")
print(f"- Seller_C: High-price/Premium seller (> €{high_cutoff:.2f})")

# 5. Calculate seller characteristics
print("\n[5] Calculating seller characteristics...")
seller_stats = selected_product_data.groupby('Seller').agg({
    'Price': ['mean', 'min', 'max'],
    'Quantity': ['sum', 'mean', 'count'],
    'Revenue': 'sum'
}).reset_index()

# Flatten columns
seller_stats.columns = ['Seller', 'Avg_Price', 'Min_Price', 'Max_Price',
                        'Total_Quantity', 'Avg_Quantity_Per_Transaction',
                        'Num_Transactions', 'Total_Revenue']

print("\n" + "="*80)
print("SELLER STATISTICS:")
print("="*80)
for idx, row in seller_stats.iterrows():
    print(f"\n{row['Seller']}:")
    print(f"  Average Price: €{row['Avg_Price']:.2f}")
    print(f"  Price Range: €{row['Min_Price']:.2f} - €{row['Max_Price']:.2f}")
    print(f"  Total Quantity Sold: {int(row['Total_Quantity'])}")
    print(f"  Avg Quantity per Transaction: {row['Avg_Quantity_Per_Transaction']:.2f}")
    print(f"  Total Revenue: €{row['Total_Revenue']:.2f}")
    print(f"  Number of Transactions: {int(row['Num_Transactions'])}")

# 6. Estimate production cost
print("\n[6] Estimating production cost...")
estimated_cost = 0.6 * seller_stats['Min_Price'].mean()
print(f"\nEstimated Production Cost: €{estimated_cost:.2f}")
print(f"(Calculated as 60% of average minimum price across sellers)")

# 7. Calculate base demand for each seller
print("\n[7] Calculating base demand...")
seller_stats['Base_Demand'] = seller_stats['Avg_Quantity_Per_Transaction']
print("\nBase Demand (avg quantity per transaction):")
for idx, row in seller_stats.iterrows():
    print(f"  {row['Seller']}: {row['Base_Demand']:.2f} units")

# 8. Visualize seller comparison
print("\n[8] Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Seller Analysis: {selected_product["Description"]}\n(StockCode: {selected_stock_code})',
             fontsize=16, fontweight='bold')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
seller_colors = [colors[i] for i in range(len(seller_stats))]

# Top-left: Average prices
axes[0, 0].bar(seller_stats['Seller'], seller_stats['Avg_Price'], color=seller_colors)
axes[0, 0].set_title('Average Price by Seller', fontweight='bold')
axes[0, 0].set_xlabel('Seller')
axes[0, 0].set_ylabel('Average Price (€)')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(seller_stats['Avg_Price']):
    axes[0, 0].text(i, v + (v * 0.02), f'€{v:.2f}', ha='center', va='bottom', fontweight='bold')

# Top-right: Total quantities
axes[0, 1].bar(seller_stats['Seller'], seller_stats['Total_Quantity'], color=seller_colors)
axes[0, 1].set_title('Total Quantity Sold by Seller', fontweight='bold')
axes[0, 1].set_xlabel('Seller')
axes[0, 1].set_ylabel('Total Quantity')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(seller_stats['Total_Quantity']):
    axes[0, 1].text(i, v + (v * 0.02), f'{int(v)}', ha='center', va='bottom', fontweight='bold')

# Bottom-left: Total revenue
axes[1, 0].bar(seller_stats['Seller'], seller_stats['Total_Revenue'], color=seller_colors)
axes[1, 0].set_title('Total Revenue by Seller', fontweight='bold')
axes[1, 0].set_xlabel('Seller')
axes[1, 0].set_ylabel('Total Revenue (€)')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(seller_stats['Total_Revenue']):
    axes[1, 0].text(i, v + (v * 0.02), f'€{v:.0f}', ha='center', va='bottom', fontweight='bold')

# Bottom-right: Box plot of price distributions
palette_dict = {seller: colors[i] for i, seller in enumerate(sorted(selected_product_data['Seller'].unique()))}
sns.boxplot(data=selected_product_data, x='Seller', y='Price',
            hue='Seller', palette=palette_dict, legend=False, ax=axes[1, 1])
axes[1, 1].set_title('Price Distribution by Seller', fontweight='bold')
axes[1, 1].set_xlabel('Seller')
axes[1, 1].set_ylabel('Price (€)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('seller_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'seller_analysis.png'")

# Summary output
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Selected Product: {selected_product['Description']}")
print(f"Estimated Production Cost: €{estimated_cost:.2f}")
print(f"Number of Sellers Created: {len(seller_stats)}")
print(f"Total Market Size: {selected_product_data['Quantity'].sum()} units")
print(f"Total Market Revenue: €{selected_product_data['Revenue'].sum():.2f}")
print("="*80)

print("\n✓ Seller modeling completed successfully!")

# ==================================================================================
# OBJECT-ORIENTED SELLER AND MARKET MODELING
# ==================================================================================

class Seller:
    """
    Represents a seller in the online marketplace competing in a duopoly/oligopoly.

    This class models a strategic seller in a game theory context, where each seller
    must decide on pricing and advertising strategies to maximize profit while
    considering competitor actions. Used in Nash Equilibrium simulations.

    Attributes:
        name (str): Seller identifier (e.g., 'Seller_A', 'Apple', 'Dell')
        production_cost (float): Cost to produce one unit (€). Must be positive.
        price (float): Current selling price per unit (€). Must be > production_cost.
        advertising_budget (float): Marketing/advertising spending (€). Must be non-negative.
        brand_value (float): Brand strength/market position [0-10]. Higher = stronger brand.
        demand (float): Current calculated demand in units (updated by MarketModel)
        revenue (float): Current revenue = price × demand (€)
        profit (float): Current profit = revenue - total_costs (€)

    Game Theory Context:
        - production_cost: Lower bound constraint on pricing strategy
        - price: Strategic decision variable for Nash Equilibrium
        - advertising_budget: Strategic decision variable for market share
        - brand_value: Asymmetry parameter affecting competitive advantage

    Nash Equilibrium Convergence:
        - Higher brand_value → More pricing power, faster convergence
        - Lower production_cost → Wider strategy space, may slow convergence
        - Initial price close to equilibrium → Faster convergence
        - Symmetric sellers → Unique pure strategy Nash Equilibrium
        - Asymmetric sellers → May have mixed or no pure equilibrium

    Expected Value Ranges:
        - production_cost: €0.10 - €100+ (product dependent)
        - price: production_cost * 1.1 to production_cost * 3.0 (typically)
        - advertising_budget: €0 - €50,000+ (market dependent)
        - brand_value: 0.0 (new/unknown) to 10.0 (market leader)
    """

    def __init__(self, name: str, cost: float, initial_price: float,
                 initial_ad_budget: float, base_demand: float):
        """
        Initialize a new seller with economic and strategic parameters.

        Parameters:
        -----------
        name : str
            Seller identifier (e.g., 'Seller_A', 'Apple MacBook')
        cost : float
            Production cost per unit in euros (€). Must be positive.
            Renamed to 'production_cost' internally for clarity.
        initial_price : float
            Starting selling price per unit (€). Must be > production_cost.
            This is a strategic variable in Nash Equilibrium finding.
        initial_ad_budget : float
            Starting advertising/marketing budget (€). Must be non-negative.
            Renamed to 'advertising_budget' internally for consistency.
        base_demand : float
            Brand strength or market position indicator [0-10].
            Renamed to 'brand_value' to reflect its role in market share.
            Higher values indicate stronger competitive position.

        Raises:
        -------
        ValueError
            If any economic constraints are violated:
            - production_cost <= 0
            - initial_price <= production_cost (selling at a loss)
            - initial_ad_budget < 0
            - base_demand < 0

        Notes:
        ------
        - Backward compatible: accepts old parameter names
        - Stores both old and new attribute names during transition
        - Essential for Nash Equilibrium convergence in Task III
        """

        # === Input Validation ===
        if cost <= 0:
            raise ValueError(f"Production cost must be positive. Got: €{cost:.2f}")

        if initial_price <= cost:
            raise ValueError(
                f"Price (€{initial_price:.2f}) must be greater than production cost (€{cost:.2f}). "
                f"Cannot sell at a loss!"
            )

        if initial_ad_budget < 0:
            raise ValueError(
                f"Advertising budget cannot be negative. Got: €{initial_ad_budget:.2f}"
            )

        if base_demand < 0:
            raise ValueError(
                f"Brand value must be non-negative. Got: {base_demand:.2f}"
            )

        # === Core Attributes ===
        self.name = name

        # Store with BOTH old and new names for compatibility
        self.cost = cost  # Backward compatibility with Task II
        self.production_cost = cost  # New name for Task III compatibility

        self.price = initial_price

        self.ad_budget = initial_ad_budget  # Backward compatibility
        self.advertising_budget = initial_ad_budget  # New name for Task III

        self.base_demand = base_demand  # Backward compatibility
        self.brand_value = base_demand  # New name for Task III

        # === Calculated Attributes (updated by MarketModel) ===
        self.demand = 0.0
        self.revenue = 0.0
        self.profit = 0.0

    def __repr__(self):
        """String representation for easy printing."""
        return (f"Seller(name={self.name}, price=€{self.price:.2f}, "
                f"ad_budget=€{self.ad_budget:.2f}, demand={self.demand:.2f}, "
                f"profit=€{self.profit:.2f})")

    def update_strategy(self, new_price: float, new_ad_budget: float):
        """
        Update seller's strategic decisions (price and advertising).

        This method allows sellers to change their competitive strategy.
        In game theory, this represents a "move" or "action" by the player
        during the iterative best response algorithm for finding Nash Equilibrium.

        Parameters:
        -----------
        new_price : float
            New selling price (€). Must be > production_cost.
        new_ad_budget : float
            New advertising budget (€). Must be non-negative.

        Raises:
        -------
        ValueError
            If new_price <= production_cost or new_ad_budget < 0

        Notes:
        ------
        Demand, revenue, and profit will be recalculated by MarketModel
        after strategy update.
        """
        if new_price <= self.production_cost:
            raise ValueError(
                f"New price (€{new_price:.2f}) must be greater than "
                f"production cost (€{self.production_cost:.2f})"
            )

        if new_ad_budget < 0:
            raise ValueError(f"Advertising budget cannot be negative: €{new_ad_budget:.2f}")

        self.price = new_price

        # Update both attribute names for compatibility
        self.ad_budget = new_ad_budget
        self.advertising_budget = new_ad_budget

    def get_profit_margin(self) -> float:
        """
        Calculate profit margin per unit.

        In game theory context, this represents the incentive to sell more units.
        Higher margins create stronger incentive for aggressive pricing strategies.

        Returns:
        --------
        float
            Profit per unit sold = price - production_cost (€)

        Notes:
        ------
        This is used in Nash Equilibrium calculations to determine
        optimal pricing strategies given competitor actions.
        """
        return self.price - self.production_cost

    def get_summary(self) -> dict:
        """
        Get current seller metrics as a dictionary.

        Returns:
        --------
        dict
            All seller metrics including strategic variables,
            calculated demand/profit, and brand positioning.
        """
        return {
            'Name': self.name,
            'Price': f'€{self.price:.2f}',
            'Production_Cost': f'€{self.production_cost:.2f}',
            'Advertising_Budget': f'€{self.advertising_budget:.2f}',
            'Brand_Value': f'{self.brand_value:.2f}',
            'Demand': f'{self.demand:.2f}',
            'Revenue': f'€{self.revenue:.2f}',
            'Profit': f'€{self.profit:.2f}',
            'Profit_Margin': f'€{self.get_profit_margin():.2f}'
        }


class MarketModel:
    """
    Models the competitive online marketplace with multiple sellers.

    This class implements the market demand function and calculates profits
    for sellers competing on price and advertising.

    DEMAND FUNCTION:
        D_i = base_demand + (α × m_i) + (β × (p_j - p_i)) + (γ × influence_score_i)

    Where:
        D_i: Demand for seller i (units)
        base_demand: Natural demand without any competitive effects

        α (alpha): Advertising effectiveness coefficient
            - How many additional units sold per €1 spent on advertising
            - Example: α = 0.01 means €100 advertising → +1 unit demand

        β (beta): Price sensitivity coefficient
            - How demand shifts based on price differences with competitors
            - Example: β = 5.0 means if competitor charges €1 more → +5 units demand

        γ (gamma): Social influence coefficient (for Task IV - network effects)
            - How social network influence affects demand
            - Default 0.0 for Tasks II & III (no social network)

    PROFIT FUNCTION:
        Profit_i = (price_i - cost_i) × demand_i - ad_budget_i

        Components:
        - (price - cost) × demand = Revenue - Cost of Goods Sold
        - ad_budget = Marketing cost

    Attributes:
        sellers (dict): Dictionary mapping seller names to Seller objects
        alpha (float): Advertising effectiveness coefficient
        beta (float): Price sensitivity coefficient
        gamma (float): Social influence coefficient
    """


    def __init__(self, sellers_dict, alpha=0.01, beta=5.0, gamma=0.0,total_market_size=0):
        """
        Initialize the market model.

        Args:
            sellers_dict (dict): Dictionary of {name: Seller object}
            alpha (float): Advertising effectiveness (default 0.01)
            beta (float): Price sensitivity (default 5.0)
            gamma (float): Social influence coefficient (default 0.0)
        """
        self.sellers = sellers_dict
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.total_market_size = total_market_size

        print(f"\n{'='*80}")
        print("MARKET MODEL INITIALIZED")
        print(f"{'='*80}")
        print(f"Number of sellers: {len(self.sellers)}")
        print(f"Alpha (advertising effectiveness): {self.alpha}")
        print(f"  → Each €1 advertising increases demand by {self.alpha} units")
        print(f"Beta (price sensitivity): {self.beta}")
        print(f"  → Each €1 price advantage vs competitor increases demand by {self.beta} units")
        print(f"Gamma (social influence): {self.gamma}")
        print(f"  → Currently {self.gamma} (no social network effects)")
        print(f"{'='*80}\n")

    def calculate_demand(self, seller_i, seller_j, influence_score_i=0):
        """
        Calculate demand for seller i given competitor j's actions.

        DEMAND EQUATION BREAKDOWN:
        1. base_demand: Natural baseline demand
        2. α × m_i: Additional demand from advertising
        3. β × (p_j - p_i): Demand gain/loss from price competition
        4. γ × influence_score: Demand from social influence (Task IV)

        Args:
            seller_i (Seller): The seller whose demand we're calculating
            seller_j (Seller): The competing seller
            influence_score_i (float): Social influence score (default 0)

        Returns:
            float: Calculated demand (units), always >= 0
        """
        # Component 1: Base demand (natural demand without any effects)
        base = seller_i.base_demand

        # Component 2: Advertising effect (α × m_i)
        # More advertising → more demand
        advertising_effect = self.alpha * seller_i.ad_budget

        # Component 3: Price competition effect (β × (p_j - p_i))
        # If competitor charges more → we get more demand
        # If we charge more → we lose demand
        price_competition_effect = self.beta * (seller_j.price - seller_i.price)

        # Component 4: Social influence (γ × influence_score)
        # Network effects from social connections (used in Task IV)
        social_influence_effect = self.gamma * influence_score_i

        # Total demand
        demand = base + advertising_effect + price_competition_effect + social_influence_effect

        # Ensure demand is never negative (economic constraint)
        demand = max(0, demand)

        return demand

    def calculate_profit(self, seller_i, seller_j, influence_score_i=0):
        """
        Calculate profit for seller i given competitor j's actions.

        PROFIT CALCULATION:
        1. Calculate demand using demand function
        2. Calculate revenue = price × demand
        3. Calculate cost of goods sold = unit_cost × demand
        4. Calculate profit = revenue - cost_of_goods - advertising_budget

        Args:
            seller_i (Seller): The seller whose profit we're calculating
            seller_j (Seller): The competing seller
            influence_score_i (float): Social influence score (default 0)

        Returns:
            float: Calculated profit (€)
        """
        # Step 1: Calculate demand
        demand = self.calculate_demand(seller_i, seller_j, influence_score_i)

        # Step 2: Calculate revenue
        revenue = seller_i.price * demand

        # Step 3: Calculate cost of goods sold
        cost_of_goods = seller_i.cost * demand

        # Step 4: Calculate profit = revenue - costs
        profit = revenue - cost_of_goods - seller_i.ad_budget

        # Update seller's metrics
        seller_i.demand = demand
        seller_i.revenue = revenue
        seller_i.profit = profit

        return profit

    def calculate_all_profits(self, influence_scores=None):
        """
        Calculate profits for all sellers considering all competitors.

        For 2 sellers (duopoly): Each seller faces one competitor
        For 3 sellers (triopoly): Each seller's demand considers average competitor effect

        Args:
            influence_scores (dict): Social influence scores {seller_name: score}
                                    Default None for no influence

        Returns:
            dict: {seller_name: profit}
        """
        if influence_scores is None:
            influence_scores = {name: 0 for name in self.sellers.keys()}

        seller_list = list(self.sellers.values())
        profits = {}

        if len(seller_list) == 2:
            # Duopoly: Simple head-to-head competition
            seller_a, seller_b = seller_list[0], seller_list[1]

            profits[seller_a.name] = self.calculate_profit(
                seller_a, seller_b, influence_scores.get(seller_a.name, 0)
            )
            profits[seller_b.name] = self.calculate_profit(
                seller_b, seller_a, influence_scores.get(seller_b.name, 0)
            )

        elif len(seller_list) == 3:
            # Triopoly: Each seller competes against average of others
            for i, seller_i in enumerate(seller_list):
                # Get the two competitors
                competitors = [s for j, s in enumerate(seller_list) if j != i]

                # Calculate average competitor price for price competition effect
                avg_competitor_price = sum(c.price for c in competitors) / len(competitors)

                # Create a "virtual average competitor" for calculation
                # Use minimum cost from all sellers to avoid validation error
                min_cost = min(s.production_cost for s in self.sellers.values())
                avg_competitor = Seller(
                    name="Avg_Competitor",
                    cost=min_cost,  # Use actual minimum cost instead of 0
                    initial_price=avg_competitor_price,
                    initial_ad_budget=0,
                    base_demand=0.1  # Minimal positive value to avoid validation error
                )

                profits[seller_i.name] = self.calculate_profit(
                    seller_i, avg_competitor, influence_scores.get(seller_i.name, 0)
                )

        return profits

    def print_market_state(self):
        """
        Display current market state with formatted table of all sellers' metrics.
        """
        print(f"\n{'='*100}")
        print("CURRENT MARKET STATE")
        print(f"{'='*100}")
        print(f"{'Seller':<12} {'Price':>10} {'Ad Budget':>12} {'Demand':>10} "
              f"{'Revenue':>12} {'Profit':>12} {'Margin':>10}")
        print(f"{'-'*100}")

        for seller in self.sellers.values():
            print(f"{seller.name:<12} €{seller.price:>9.2f} €{seller.ad_budget:>11.2f} "
                  f"{seller.demand:>10.2f} €{seller.revenue:>11.2f} "
                  f"€{seller.profit:>11.2f} €{seller.get_profit_margin():>9.2f}")

        print(f"{'='*100}\n")

    def get_market_summary(self):
        """
        Get market summary as a DataFrame for analysis.

        Returns:
            pd.DataFrame: Summary of all sellers' metrics
        """
        data = []
        for seller in self.sellers.values():
            data.append({
                'Seller': seller.name,
                'Price': seller.price,
                'Ad_Budget': seller.ad_budget,
                'Base_Demand': seller.base_demand,
                'Demand': seller.demand,
                'Revenue': seller.revenue,
                'Cost_per_Unit': seller.cost,
                'Profit': seller.profit,
                'Profit_Margin': seller.get_profit_margin()
            })

        return pd.DataFrame(data)


# ==================================================================================
# INITIALIZE SELLERS FROM PREVIOUS ANALYSIS
# ==================================================================================

print("\n" + "="*80)
print("INITIALIZING SELLERS FROM SELLER_STATS")
print("="*80)

# Use seller_stats and estimated_cost from the previous analysis
print(f"\nEstimated Production Cost: €{estimated_cost:.2f}")
print(f"Number of Sellers: {len(seller_stats)}\n")

# Create seller objects from seller_stats DataFrame
sellers = {}

for idx, row in seller_stats.iterrows():
    seller_name = row['Seller']

    # Calculate initial advertising budget as 10% of total revenue
    initial_ad_budget = 0.10 * row['Total_Revenue']

    # Create seller object
    seller = Seller(
        name=seller_name,
        cost=estimated_cost,
        initial_price=row['Avg_Price'],
        initial_ad_budget=initial_ad_budget,
        base_demand=row['Base_Demand']
    )

    sellers[seller_name] = seller

    print(f"Created {seller_name}:")
    print(f"  Cost: €{estimated_cost:.2f}")
    print(f"  Initial Price: €{row['Avg_Price']:.2f}")
    print(f"  Initial Ad Budget: €{initial_ad_budget:.2f} (10% of revenue)")
    print(f"  Base Demand: {row['Base_Demand']:.2f} units")
    print()

# ==================================================================================
# INITIALIZE MARKET MODEL
# ==================================================================================

# Create market with specified coefficients
market = MarketModel(
    sellers_dict=sellers,
    alpha=0.01,   # Each €1 advertising → +0.01 units demand
    beta=5.0,     # Each €1 price advantage → +5 units demand
    gamma=0.0     # No social influence yet (Task IV)
)

# ==================================================================================
# CALCULATE INITIAL PROFITS
# ==================================================================================

print("\n" + "="*80)
print("CALCULATING INITIAL MARKET EQUILIBRIUM")
print("="*80)

if len(sellers) == 2:
    # Duopoly scenario
    seller_names = list(sellers.keys())
    seller_a = sellers[seller_names[0]]
    seller_b = sellers[seller_names[1]]

    print(f"\nDUOPOLY: {seller_a.name} vs {seller_b.name}")
    print(f"{'-'*80}")

    profit_a = market.calculate_profit(seller_a, seller_b)
    profit_b = market.calculate_profit(seller_b, seller_a)

    print(f"\n{seller_a.name}:")
    print(f"  Demand: {seller_a.demand:.2f} units")
    print(f"  Revenue: €{seller_a.revenue:.2f}")
    print(f"  Profit: €{profit_a:.2f}")

    print(f"\n{seller_b.name}:")
    print(f"  Demand: {seller_b.demand:.2f} units")
    print(f"  Revenue: €{seller_b.revenue:.2f}")
    print(f"  Profit: €{profit_b:.2f}")

elif len(sellers) == 3:
    # Triopoly scenario
    print(f"\nTRIOPOLY: {len(sellers)} sellers competing")
    print(f"{'-'*80}")

    profits = market.calculate_all_profits()

    for seller_name, profit in profits.items():
        seller = sellers[seller_name]
        print(f"\n{seller_name}:")
        print(f"  Demand: {seller.demand:.2f} units")
        print(f"  Revenue: €{seller.revenue:.2f}")
        print(f"  Profit: €{profit:.2f}")

# Print formatted market state
market.print_market_state()
# Get summary DataFrame
market_summary_df = market.get_market_summary()
print("Market Summary DataFrame:")
print(market_summary_df.to_string(index=False))

print("\n✓ Object-oriented seller and market models initialized successfully!")

# ==================================================================================
# PROFIT LANDSCAPE VISUALIZATION
# ==================================================================================

print("\n" + "="*80)
print("CREATING PROFIT LANDSCAPE VISUALIZATIONS")
print("="*80)


def calculate_profit_landscape(market, seller_i, seller_j,
                                price_range, ad_range,
                                competitor_fixed_price, competitor_fixed_ad):
    """
    Calculate profit for seller_i across different price and ad budget combinations
    while competitor (seller_j) keeps fixed strategies.

    This function creates a 2D grid showing how seller_i's profit varies across
    all possible combinations of their price and advertising budget decisions,
    assuming the competitor maintains fixed strategies.

    Economic Interpretation:
    - Each cell in the grid represents a strategic choice (price, ad_budget)
    - The value is the resulting profit given competitor's fixed strategy
    - The optimal point is the Nash equilibrium response to competitor's strategy

    Parameters:
        market (MarketModel): The market model with demand/profit functions
        seller_i (Seller): Seller object to analyze
        seller_j (Seller): Competitor Seller object (kept fixed)
        price_range (np.array): Array of prices to test
        ad_range (np.array): Array of ad budgets to test
        competitor_fixed_price (float): Competitor's fixed price
        competitor_fixed_ad (float): Competitor's fixed ad budget

    Returns:
        np.array: 2D profit grid [len(ad_range), len(price_range)]
    """
    # Save original strategies
    original_price_i = seller_i.price
    original_ad_i = seller_i.ad_budget
    original_price_j = seller_j.price
    original_ad_j = seller_j.ad_budget

    # Fix competitor's strategy
    seller_j.price = competitor_fixed_price
    seller_j.ad_budget = competitor_fixed_ad

    # Initialize profit grid
    profit_grid = np.zeros((len(ad_range), len(price_range)))

    # Calculate profit for each combination
    for i, ad in enumerate(ad_range):
        for j, price in enumerate(price_range):
            # Update seller i's strategy
            seller_i.price = price
            seller_i.ad_budget = ad

            # Calculate profit
            profit = market.calculate_profit(seller_i, seller_j)
            profit_grid[i, j] = profit

    # Restore original strategies
    seller_i.price = original_price_i
    seller_i.ad_budget = original_ad_i
    seller_j.price = original_price_j
    seller_j.ad_budget = original_ad_j

    return profit_grid


# ==================================================================================
# GENERATE PROFIT GRIDS FOR BOTH SELLERS
# ==================================================================================

print("\n[1] Generating profit landscapes...")

# Check if we have enough sellers for landscape analysis
if len(sellers) < 2:
    print(f"\n⚠ WARNING: Only {len(sellers)} seller(s) found.")
    print("Profit landscape visualization requires at least 2 sellers (duopoly).")
    print("\nRECOMMENDATIONS:")
    print("1. Select a product with more price variation (different quantiles)")
    print("2. Adjust price quantiles to create multiple sellers")
    print("3. Use a different product from the top 20 list")
    print("\nSkipping profit landscape visualization...")
    print("="*80)
    print("\n✓ Seller modeling completed (landscape skipped due to single seller)")
else:
    # Get the two sellers (duopoly)
    seller_names = list(sellers.keys())
    seller_a = sellers[seller_names[0]]
    seller_b = sellers[seller_names[1]]

    # Define price range: from (cost + 0.01) to (cost + 0.20) with 50 points
    price_min = estimated_cost + 0.01
    price_max = estimated_cost + 0.20
    price_range = np.linspace(price_min, price_max, 50)

    # Define ad budget range: from 0 to 2000 with 50 points
    ad_range = np.linspace(0, 2000, 50)

    print(f"\nPrice range: €{price_min:.2f} to €{price_max:.2f} (50 points)")
    print(f"Ad budget range: €0 to €2000 (50 points)")
    print(f"Total combinations per seller: {len(price_range)} × {len(ad_range)} = {len(price_range) * len(ad_range)}")

    # Store current strategies
    current_price_a = seller_a.price
    current_ad_a = seller_a.ad_budget
    current_price_b = seller_b.price
    current_ad_b = seller_b.ad_budget

    # Calculate profit landscape for Seller A (with Seller B fixed)
    print(f"\nCalculating profit landscape for {seller_a.name}...")
    print(f"  (keeping {seller_b.name} fixed at price=€{current_price_b:.2f}, ad=€{current_ad_b:.2f})")
    profit_grid_A = calculate_profit_landscape(
        market, seller_a, seller_b,
        price_range, ad_range,
        current_price_b, current_ad_b
    )

    # Calculate profit landscape for Seller B (with Seller A fixed)
    print(f"\nCalculating profit landscape for {seller_b.name}...")
    print(f"  (keeping {seller_a.name} fixed at price=€{current_price_a:.2f}, ad=€{current_ad_a:.2f})")
    profit_grid_B = calculate_profit_landscape(
        market, seller_b, seller_a,
        price_range, ad_range,
        current_price_a, current_ad_a
    )

    print("\n✓ Profit landscapes calculated successfully!")

    # ==================================================================================
    # CREATE VISUALIZATION
    # ==================================================================================

    print("\n[2] Creating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle('Profit Landscape Analysis: Price and Advertising Optimization',
                 fontsize=18, fontweight='bold', y=0.995)

    cmap = 'RdYlGn'

    # Subplot 1: Seller A
    ax1 = axes[0]
    im1 = ax1.imshow(profit_grid_A, aspect='auto', origin='lower',
                     extent=[price_range[0], price_range[-1], ad_range[0], ad_range[-1]],
                     cmap=cmap)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Profit (€)', fontsize=12, fontweight='bold')
    ax1.plot(current_price_a, current_ad_a, 'wX', markersize=15,
             markeredgewidth=3, label='Current Strategy')
    ax1.set_xlabel('Price (€)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Advertising Budget (€)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{seller_a.name}: Profit Landscape ({seller_b.name} Fixed)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Subplot 2: Seller B
    ax2 = axes[1]
    im2 = ax2.imshow(profit_grid_B, aspect='auto', origin='lower',
                     extent=[price_range[0], price_range[-1], ad_range[0], ad_range[-1]],
                     cmap=cmap)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Profit (€)', fontsize=12, fontweight='bold')
    ax2.plot(current_price_b, current_ad_b, 'wX', markersize=15,
             markeredgewidth=3, label='Current Strategy')
    ax2.set_xlabel('Price (€)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Advertising Budget (€)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{seller_b.name}: Profit Landscape ({seller_a.name} Fixed)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Subplot 3: Profit vs Price
    ax3 = axes[2]
    profit_vs_price_A = []
    profit_vs_price_B = []

    for price in price_range:
        seller_a.price = price
        seller_a.ad_budget = current_ad_a
        seller_b.price = current_price_b
        seller_b.ad_budget = current_ad_b
        profit_a = market.calculate_profit(seller_a, seller_b)
        profit_vs_price_A.append(profit_a)

        seller_a.price = current_price_a
        seller_a.ad_budget = current_ad_a
        seller_b.price = price
        seller_b.ad_budget = current_ad_b
        profit_b = market.calculate_profit(seller_b, seller_a)
        profit_vs_price_B.append(profit_b)

    seller_a.price = current_price_a
    seller_a.ad_budget = current_ad_a
    seller_b.price = current_price_b
    seller_b.ad_budget = current_ad_b

    ax3.plot(price_range, profit_vs_price_A, 'b-', linewidth=2.5, label=f'{seller_a.name}')
    ax3.plot(price_range, profit_vs_price_B, 'r-', linewidth=2.5, label=f'{seller_b.name}')

    optimal_idx_a = np.argmax(profit_vs_price_A)
    optimal_price_a = price_range[optimal_idx_a]
    optimal_idx_b = np.argmax(profit_vs_price_B)
    optimal_price_b = price_range[optimal_idx_b]

    ax3.axvline(optimal_price_a, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                label=f'{seller_a.name} Optimal: €{optimal_price_a:.2f}')
    ax3.axvline(optimal_price_b, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'{seller_b.name} Optimal: €{optimal_price_b:.2f}')

    ax3.set_xlabel('Price (€)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Profit (€)', fontsize=12, fontweight='bold')
    ax3.set_title('Profit vs Price (Ad Budget Fixed)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('profit_landscape.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'profit_landscape.png'")

    # Find optimal strategies
    print("\n" + "="*80)
    print("OPTIMAL STRATEGIES (GIVEN COMPETITOR FIXED)")
    print("="*80)

    max_idx_a = np.unravel_index(np.argmax(profit_grid_A), profit_grid_A.shape)
    optimal_ad_idx_a, optimal_price_idx_a = max_idx_a
    optimal_price_a_full = price_range[optimal_price_idx_a]
    optimal_ad_a_full = ad_range[optimal_ad_idx_a]
    optimal_profit_a_full = profit_grid_A[optimal_ad_idx_a, optimal_price_idx_a]

    print(f"\n{seller_a.name}'s optimal strategy:")
    print(f"  Price: €{optimal_price_a_full:.2f}")
    print(f"  Ad Budget: €{optimal_ad_a_full:.2f}")
    print(f"  Expected Profit: €{optimal_profit_a_full:.2f}")

    max_idx_b = np.unravel_index(np.argmax(profit_grid_B), profit_grid_B.shape)
    optimal_ad_idx_b, optimal_price_idx_b = max_idx_b
    optimal_price_b_full = price_range[optimal_price_idx_b]
    optimal_ad_b_full = ad_range[optimal_ad_idx_b]
    optimal_profit_b_full = profit_grid_B[optimal_ad_idx_b, optimal_price_idx_b]

    print(f"\n{seller_b.name}'s optimal strategy:")
    print(f"  Price: €{optimal_price_b_full:.2f}")
    print(f"  Ad Budget: €{optimal_ad_b_full:.2f}")
    print(f"  Expected Profit: €{optimal_profit_b_full:.2f}")

    print("\n✓ Profit landscape analysis completed successfully!")


# ==================================================================================
# PARAMETER SENSITIVITY ANALYSIS
# ==================================================================================

print("\n" + "="*80)
print("PARAMETER SENSITIVITY ANALYSIS")
print("="*80)


def analyze_parameter_sensitivity(market, seller_a, seller_b):
    """
    Analyze how profits change with different α (advertising effectiveness)
    and β (price sensitivity) values.

    Tests multiple scenarios and returns DataFrame with results for all combinations.
    """
    # Define parameter ranges
    alpha_values = {
        'Low': 0.005,
        'Medium': 0.01,
        'High': 0.02
    }

    beta_values = {
        'Low': 2.0,
        'Medium': 5.0,
        'High': 10.0
    }

    results = []

    print("\nTesting 9 parameter combinations (3 α × 3 β)...")
    print("Alpha (α) - Advertising effectiveness")
    print("Beta (β) - Price sensitivity")

    # Store original parameters
    original_alpha = market.alpha
    original_beta = market.beta

    # Test all combinations
    for alpha_label, alpha_val in alpha_values.items():
        for beta_label, beta_val in beta_values.items():
            # Update market parameters
            market.alpha = alpha_val
            market.beta = beta_val

            # Calculate demands and profits
            demand_a = market.calculate_demand(seller_a, seller_b)
            profit_a = market.calculate_profit(seller_a, seller_b)

            demand_b = market.calculate_demand(seller_b, seller_a)
            profit_b = market.calculate_profit(seller_b, seller_a)

            results.append({
                'alpha': alpha_val,
                'beta': beta_val,
                'alpha_label': alpha_label,
                'beta_label': beta_label,
                'seller_A_demand': demand_a,
                'seller_A_profit': profit_a,
                'seller_B_demand': demand_b,
                'seller_B_profit': profit_b
            })

    # Restore original parameters
    market.alpha = original_alpha
    market.beta = original_beta

    return pd.DataFrame(results)


# Run sensitivity analysis
if len(sellers) >= 2:
    print("\n[1] Running parameter sensitivity analysis...")

    # Get first two sellers
    seller_names = list(sellers.keys())
    seller_a = sellers[seller_names[0]]
    seller_b = sellers[seller_names[1]]

    # Analyze sensitivity
    sensitivity_results = analyze_parameter_sensitivity(market, seller_a, seller_b)

    print("\n✓ Sensitivity analysis completed!")

    # ==================================================================================
    # PRINT FORMATTED RESULTS TABLE
    # ==================================================================================

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*80)

    # Create formatted table
    print("\nProfit Sensitivity to α (Advertising) and β (Price Sensitivity):")
    print("-" * 100)
    print(f"{'α Level':<10} {'β Level':<10} {'α Value':<10} {'β Value':<10} "
          f"{'A Demand':<12} {'A Profit':<12} {'B Demand':<12} {'B Profit':<12}")
    print("-" * 100)

    for idx, row in sensitivity_results.iterrows():
        # Highlight baseline scenario
        prefix = ">>> " if row['alpha_label'] == 'Medium' and row['beta_label'] == 'Medium' else "    "

        print(f"{prefix}{row['alpha_label']:<10} {row['beta_label']:<10} "
              f"{row['alpha']:<10.3f} {row['beta']:<10.1f} "
              f"{row['seller_A_demand']:<12.2f} {row['seller_A_profit']:<12.2f} "
              f"{row['seller_B_demand']:<12.2f} {row['seller_B_profit']:<12.2f}")

    print("-" * 100)
    print(">>> = Baseline scenario (Medium α, Medium β)")

    # ==================================================================================
    # CREATE VISUALIZATION
    # ==================================================================================

    print("\n[2] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter Sensitivity Analysis: Impact of α and β on Seller Profits',
                 fontsize=16, fontweight='bold', y=0.995)

    # Define colors for different levels
    colors = {'Low': '#FF6B6B', 'Medium': '#4ECDC4', 'High': '#45B7D1'}

    # Prepare data for grouped bar charts
    alpha_labels = ['Low', 'Medium', 'High']
    beta_labels = ['Low', 'Medium', 'High']

    x_positions = np.arange(len(beta_labels))
    bar_width = 0.25

    # ------------------------------------------------------------------------------
    # Subplot 1: Impact of α on Seller A Profit (grouped by β)
    # ------------------------------------------------------------------------------
    ax1 = axes[0, 0]

    for i, alpha_label in enumerate(alpha_labels):
        profits = []
        for beta_label in beta_labels:
            profit = sensitivity_results[
                (sensitivity_results['alpha_label'] == alpha_label) &
                (sensitivity_results['beta_label'] == beta_label)
            ]['seller_A_profit'].values[0]
            profits.append(profit)

        ax1.bar(x_positions + i * bar_width, profits, bar_width,
                label=f'α = {alpha_label}', color=colors[alpha_label], alpha=0.8)

    ax1.set_xlabel('Price Sensitivity (β)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Profit (€)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{seller_a.name}: Impact of Advertising Effectiveness (α)',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_positions + bar_width)
    ax1.set_xticklabels(beta_labels)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ------------------------------------------------------------------------------
    # Subplot 2: Impact of α on Seller B Profit (grouped by β)
    # ------------------------------------------------------------------------------
    ax2 = axes[0, 1]

    for i, alpha_label in enumerate(alpha_labels):
        profits = []
        for beta_label in beta_labels:
            profit = sensitivity_results[
                (sensitivity_results['alpha_label'] == alpha_label) &
                (sensitivity_results['beta_label'] == beta_label)
            ]['seller_B_profit'].values[0]
            profits.append(profit)

        ax2.bar(x_positions + i * bar_width, profits, bar_width,
                label=f'α = {alpha_label}', color=colors[alpha_label], alpha=0.8)

    ax2.set_xlabel('Price Sensitivity (β)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Profit (€)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{seller_b.name}: Impact of Advertising Effectiveness (α)',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x_positions + bar_width)
    ax2.set_xticklabels(beta_labels)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ------------------------------------------------------------------------------
    # Subplot 3: Impact of β on Seller A Profit (grouped by α)
    # ------------------------------------------------------------------------------
    ax3 = axes[1, 0]

    for i, beta_label in enumerate(beta_labels):
        profits = []
        for alpha_label in alpha_labels:
            profit = sensitivity_results[
                (sensitivity_results['alpha_label'] == alpha_label) &
                (sensitivity_results['beta_label'] == beta_label)
            ]['seller_A_profit'].values[0]
            profits.append(profit)

        ax3.bar(x_positions + i * bar_width, profits, bar_width,
                label=f'β = {beta_label}', color=colors[beta_label], alpha=0.8)

    ax3.set_xlabel('Advertising Effectiveness (α)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Profit (€)', fontsize=11, fontweight='bold')
    ax3.set_title(f'{seller_a.name}: Impact of Price Sensitivity (β)',
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x_positions + bar_width)
    ax3.set_xticklabels(alpha_labels)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ------------------------------------------------------------------------------
    # Subplot 4: Impact of β on Seller B Profit (grouped by α)
    # ------------------------------------------------------------------------------
    ax4 = axes[1, 1]

    for i, beta_label in enumerate(beta_labels):
        profits = []
        for alpha_label in alpha_labels:
            profit = sensitivity_results[
                (sensitivity_results['alpha_label'] == alpha_label) &
                (sensitivity_results['beta_label'] == beta_label)
            ]['seller_B_profit'].values[0]
            profits.append(profit)

        ax4.bar(x_positions + i * bar_width, profits, bar_width,
                label=f'β = {beta_label}', color=colors[beta_label], alpha=0.8)

    ax4.set_xlabel('Advertising Effectiveness (α)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Profit (€)', fontsize=11, fontweight='bold')
    ax4.set_title(f'{seller_b.name}: Impact of Price Sensitivity (β)',
                  fontsize=12, fontweight='bold')
    ax4.set_xticks(x_positions + bar_width)
    ax4.set_xticklabels(alpha_labels)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'parameter_sensitivity.png'")

    # ==================================================================================
    # GENERATE INSIGHTS
    # ==================================================================================

    print("\n" + "="*80)
    print("KEY INSIGHTS FROM SENSITIVITY ANALYSIS")
    print("="*80)

    # Calculate profit ranges
    baseline = sensitivity_results[
        (sensitivity_results['alpha_label'] == 'Medium') &
        (sensitivity_results['beta_label'] == 'Medium')
    ].iloc[0]

    profit_range_A_alpha = sensitivity_results.groupby('alpha_label')['seller_A_profit'].mean()
    profit_range_A_beta = sensitivity_results.groupby('beta_label')['seller_A_profit'].mean()

    profit_range_B_alpha = sensitivity_results.groupby('alpha_label')['seller_B_profit'].mean()
    profit_range_B_beta = sensitivity_results.groupby('beta_label')['seller_B_profit'].mean()

    alpha_impact_A = profit_range_A_alpha.max() - profit_range_A_alpha.min()
    beta_impact_A = profit_range_A_beta.max() - profit_range_A_beta.min()

    alpha_impact_B = profit_range_B_alpha.max() - profit_range_B_alpha.min()
    beta_impact_B = profit_range_B_beta.max() - profit_range_B_beta.min()

    insights = []

    # Insight 1: Overall parameter impact
    if alpha_impact_A > beta_impact_A and alpha_impact_B > beta_impact_B:
        insights.append(
            f"• Advertising effectiveness (α) has GREATER impact than price sensitivity (β) "
            f"for both sellers\n"
            f"  - {seller_a.name}: α impact = €{alpha_impact_A:.2f}, β impact = €{beta_impact_A:.2f}\n"
            f"  - {seller_b.name}: α impact = €{alpha_impact_B:.2f}, β impact = €{beta_impact_B:.2f}"
        )
    else:
        insights.append(
            f"• Price sensitivity (β) has GREATER impact than advertising effectiveness (α)\n"
            f"  - {seller_a.name}: β impact = €{beta_impact_A:.2f}, α impact = €{alpha_impact_A:.2f}\n"
            f"  - {seller_b.name}: β impact = €{beta_impact_B:.2f}, α impact = €{alpha_impact_B:.2f}"
        )

    # Insight 2: Best/worst scenarios
    best_scenario_A = sensitivity_results.loc[sensitivity_results['seller_A_profit'].idxmax()]
    worst_scenario_A = sensitivity_results.loc[sensitivity_results['seller_A_profit'].idxmin()]

    insights.append(
        f"\n• {seller_a.name}'s profit ranges from €{worst_scenario_A['seller_A_profit']:.2f} to €{best_scenario_A['seller_A_profit']:.2f}\n"
        f"  - Best: α = {best_scenario_A['alpha_label']}, β = {best_scenario_A['beta_label']}\n"
        f"  - Worst: α = {worst_scenario_A['alpha_label']}, β = {worst_scenario_A['beta_label']}"
    )

    best_scenario_B = sensitivity_results.loc[sensitivity_results['seller_B_profit'].idxmax()]
    worst_scenario_B = sensitivity_results.loc[sensitivity_results['seller_B_profit'].idxmin()]

    insights.append(
        f"\n• {seller_b.name}'s profit ranges from €{worst_scenario_B['seller_B_profit']:.2f} to €{best_scenario_B['seller_B_profit']:.2f}\n"
        f"  - Best: α = {best_scenario_B['alpha_label']}, β = {best_scenario_B['beta_label']}\n"
        f"  - Worst: α = {worst_scenario_B['alpha_label']}, β = {worst_scenario_B['beta_label']}"
    )

    # Insight 3: Baseline comparison
    insights.append(
        f"\n• At baseline (α=Medium, β=Medium):\n"
        f"  - {seller_a.name} profit: €{baseline['seller_A_profit']:.2f}\n"
        f"  - {seller_b.name} profit: €{baseline['seller_B_profit']:.2f}"
    )

    # Insight 4: Asymmetric effects
    if abs(alpha_impact_A - alpha_impact_B) > 50:
        stronger_seller = seller_a.name if alpha_impact_A > alpha_impact_B else seller_b.name
        insights.append(
            f"\n• Parameter changes affect sellers asymmetrically\n"
            f"  - {stronger_seller} is more sensitive to advertising effectiveness"
        )

    if abs(beta_impact_A - beta_impact_B) > 50:
        stronger_seller = seller_a.name if beta_impact_A > beta_impact_B else seller_b.name
        insights.append(
            f"\n• {stronger_seller} is more sensitive to price competition"
        )

    # Print insights
    for insight in insights[:5]:  # Limit to 5 insights
        print(insight)

    # ==================================================================================
    # RESET TO BASELINE
    # ==================================================================================

    print("\n" + "="*80)
    market.alpha = 0.01
    market.beta = 5.0
    print("✓ Market reset to baseline parameters (α=0.01, β=5.0)")
    print("="*80)

    print("\n✓ Parameter sensitivity analysis completed successfully!")

else:
    print("\n⚠ WARNING: Sensitivity analysis requires at least 2 sellers.")
    print("Skipping parameter sensitivity analysis...")