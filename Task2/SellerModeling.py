import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("TASK II: SELLER MODELING AND ANALYSIS")
print("="*80)

# 1. Load the cleaned dataset
print("\n[1] Loading cleaned dataset...")
df = pd.read_csv('../Data/ProcessedData/cleaned_online_retail_data.csv')
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
selected_product = products_with_variation.iloc[0]
selected_stock_code = selected_product['StockCode']

# Extract all transactions for this product
selected_product_data = df[df['StockCode'] == selected_stock_code].copy()

print(f"\n{'='*60}")
print("SELECTED PRODUCT DETAILS:")
print(f"{'='*60}")
print(f"Product Code: {selected_stock_code}")
print(f"Product Name: {selected_product['Description']}")
print(f"Price Range: ${selected_product['Min_Price']:.2f} - ${selected_product['Max_Price']:.2f}")
print(f"Unique Price Points: {int(selected_product['Unique_Prices'])}")
print(f"Total Transactions: {selected_product['Num_Transactions']}")
print(f"Total Quantity Sold: {int(selected_product['Total_Quantity'])}")
print(f"{'='*60}")

# 4. Create 2-3 sellers based on price tiers
print("\n[4] Creating sellers based on price quantiles...")
q33 = selected_product_data['Price'].quantile(0.33)
q67 = selected_product_data['Price'].quantile(0.67)

print(f"33rd Percentile Price: ${q33:.2f}")
print(f"67th Percentile Price: ${q67:.2f}")

# Assign sellers based on price tiers
def assign_seller(price):
    if price <= q33:
        return 'Seller_A'
    elif price <= q67:
        return 'Seller_B'
    else:
        return 'Seller_C'

selected_product_data['Seller'] = selected_product_data['Price'].apply(assign_seller)

# Check how many sellers we actually have
actual_sellers = selected_product_data['Seller'].nunique()
print(f"\nNumber of sellers created: {actual_sellers}")

print("\nSeller Assignment:")
print(f"- Seller_A: Low-price/Discount seller (<= ${q33:.2f})")
print(f"- Seller_B: Mid-price/Balanced seller (${q33:.2f} - ${q67:.2f})")
print(f"- Seller_C: High-price/Premium seller (> ${q67:.2f})")

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
    print(f"  Average Price: ${row['Avg_Price']:.2f}")
    print(f"  Price Range: ${row['Min_Price']:.2f} - ${row['Max_Price']:.2f}")
    print(f"  Total Quantity Sold: {int(row['Total_Quantity'])}")
    print(f"  Avg Quantity per Transaction: {row['Avg_Quantity_Per_Transaction']:.2f}")
    print(f"  Total Revenue: ${row['Total_Revenue']:.2f}")
    print(f"  Number of Transactions: {int(row['Num_Transactions'])}")

# 6. Estimate production cost
print("\n[6] Estimating production cost...")
estimated_cost = 0.6 * seller_stats['Min_Price'].mean()
print(f"\nEstimated Production Cost: ${estimated_cost:.2f}")
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
axes[0, 0].set_ylabel('Average Price ($)')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(seller_stats['Avg_Price']):
    axes[0, 0].text(i, v + (v * 0.02), f'${v:.2f}', ha='center', va='bottom', fontweight='bold')

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
axes[1, 0].set_ylabel('Total Revenue ($)')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(seller_stats['Total_Revenue']):
    axes[1, 0].text(i, v + (v * 0.02), f'${v:.0f}', ha='center', va='bottom', fontweight='bold')

# Bottom-right: Box plot of price distributions
palette_dict = {seller: colors[i] for i, seller in enumerate(sorted(selected_product_data['Seller'].unique()))}
sns.boxplot(data=selected_product_data, x='Seller', y='Price',
            hue='Seller', palette=palette_dict, legend=False, ax=axes[1, 1])
axes[1, 1].set_title('Price Distribution by Seller', fontweight='bold')
axes[1, 1].set_xlabel('Seller')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('seller_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'seller_analysis.png'")

# Summary output
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Selected Product: {selected_product['Description']}")
print(f"Estimated Production Cost: ${estimated_cost:.2f}")
print(f"Number of Sellers Created: {len(seller_stats)}")
print(f"Total Market Size: {selected_product_data['Quantity'].sum()} units")
print(f"Total Market Revenue: ${selected_product_data['Revenue'].sum():.2f}")
print("="*80)

print("\n✓ Seller modeling completed successfully!")

# ==================================================================================
# OBJECT-ORIENTED SELLER AND MARKET MODELING
# ==================================================================================

class Seller:
    """
    Represents a seller in the online marketplace.

    This class models a competitive seller with pricing, advertising, and demand.
    Each seller makes strategic decisions on price and advertising budget to
    maximize profit in a competitive environment.

    Attributes:
        name (str): Seller identifier (e.g., 'Seller_A')
        cost (float): Production cost per unit (£)
        price (float): Current selling price per unit (£)
        ad_budget (float): Current advertising budget (£)
        base_demand (float): Natural/baseline demand without advertising or competition effects
        demand (float): Current calculated demand in units (updated during simulation)
        revenue (float): Current revenue = price × demand (£)
        profit (float): Current profit = revenue - total_costs (£)

    Economics Explanation:
        - cost: What it costs to produce one unit
        - price: What customers pay per unit
        - profit_margin: price - cost (profit per unit sold)
        - ad_budget: Money spent on advertising to increase demand
        - base_demand: Natural demand if no advertising and equal competitor prices
        - demand: Actual demand considering advertising and competitor actions
    """

    def __init__(self, name, cost, initial_price, initial_ad_budget, base_demand):
        """
        Initialize a new seller.

        Args:
            name (str): Seller identifier
            cost (float): Production cost per unit (£)
            initial_price (float): Starting price per unit (£)
            initial_ad_budget (float): Starting advertising budget (£)
            base_demand (float): Baseline demand without competition/advertising effects
        """
        self.name = name
        self.cost = cost
        self.price = initial_price
        self.ad_budget = initial_ad_budget
        self.base_demand = base_demand

        # These will be updated by the market model during simulation
        self.demand = 0.0
        self.revenue = 0.0
        self.profit = 0.0

    def __repr__(self):
        """String representation for easy printing."""
        return (f"Seller(name={self.name}, price=£{self.price:.2f}, "
                f"ad_budget=£{self.ad_budget:.2f}, demand={self.demand:.2f}, "
                f"profit=£{self.profit:.2f})")

    def update_strategy(self, new_price, new_ad_budget):
        """
        Update seller's strategic decisions (price and advertising).

        This method allows sellers to change their competitive strategy.
        In game theory, this represents a "move" or "action" by the player.

        Args:
            new_price (float): New selling price (£)
            new_ad_budget (float): New advertising budget (£)
        """
        self.price = new_price
        self.ad_budget = new_ad_budget
        # Note: demand, revenue, and profit will be recalculated by market model

    def get_profit_margin(self):
        """
        Calculate profit margin per unit.

        Returns:
            float: Profit per unit sold = price - cost (£)
        """
        return self.price - self.cost

    def get_summary(self):
        """
        Get current seller metrics as a dictionary.

        Returns:
            dict: All seller metrics
        """
        return {
            'Name': self.name,
            'Price': f'£{self.price:.2f}',
            'Ad_Budget': f'£{self.ad_budget:.2f}',
            'Base_Demand': f'{self.base_demand:.2f}',
            'Demand': f'{self.demand:.2f}',
            'Revenue': f'£{self.revenue:.2f}',
            'Cost': f'£{self.cost:.2f}',
            'Profit': f'£{self.profit:.2f}',
            'Profit_Margin': f'£{self.get_profit_margin():.2f}'
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
            - How many additional units sold per £1 spent on advertising
            - Example: α = 0.01 means £100 advertising → +1 unit demand

        β (beta): Price sensitivity coefficient
            - How demand shifts based on price differences with competitors
            - Example: β = 5.0 means if competitor charges £1 more → +5 units demand

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

    def __init__(self, sellers_dict, alpha=0.01, beta=5.0, gamma=0.0):
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

        print(f"\n{'='*80}")
        print("MARKET MODEL INITIALIZED")
        print(f"{'='*80}")
        print(f"Number of sellers: {len(self.sellers)}")
        print(f"Alpha (advertising effectiveness): {self.alpha}")
        print(f"  → Each £1 advertising increases demand by {self.alpha} units")
        print(f"Beta (price sensitivity): {self.beta}")
        print(f"  → Each £1 price advantage vs competitor increases demand by {self.beta} units")
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
            float: Calculated profit (£)
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
                avg_competitor = Seller(
                    name="Avg_Competitor",
                    cost=0,
                    initial_price=avg_competitor_price,
                    initial_ad_budget=0,
                    base_demand=0
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
            print(f"{seller.name:<12} £{seller.price:>9.2f} £{seller.ad_budget:>11.2f} "
                  f"{seller.demand:>10.2f} £{seller.revenue:>11.2f} "
                  f"£{seller.profit:>11.2f} £{seller.get_profit_margin():>9.2f}")

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
print(f"\nEstimated Production Cost: £{estimated_cost:.2f}")
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
    print(f"  Cost: £{estimated_cost:.2f}")
    print(f"  Initial Price: £{row['Avg_Price']:.2f}")
    print(f"  Initial Ad Budget: £{initial_ad_budget:.2f} (10% of revenue)")
    print(f"  Base Demand: {row['Base_Demand']:.2f} units")
    print()

# ==================================================================================
# INITIALIZE MARKET MODEL
# ==================================================================================

# Create market with specified coefficients
market = MarketModel(
    sellers_dict=sellers,
    alpha=0.01,   # Each £1 advertising → +0.01 units demand
    beta=5.0,     # Each £1 price advantage → +5 units demand
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
    print(f"  Revenue: £{seller_a.revenue:.2f}")
    print(f"  Profit: £{profit_a:.2f}")

    print(f"\n{seller_b.name}:")
    print(f"  Demand: {seller_b.demand:.2f} units")
    print(f"  Revenue: £{seller_b.revenue:.2f}")
    print(f"  Profit: £{profit_b:.2f}")

elif len(sellers) == 3:
    # Triopoly scenario
    print(f"\nTRIOPOLY: {len(sellers)} sellers competing")
    print(f"{'-'*80}")

    profits = market.calculate_all_profits()

    for seller_name, profit in profits.items():
        seller = sellers[seller_name]
        print(f"\n{seller_name}:")
        print(f"  Demand: {seller.demand:.2f} units")
        print(f"  Revenue: £{seller.revenue:.2f}")
        print(f"  Profit: £{profit:.2f}")

# Print formatted market state
market.print_market_state()

# Get summary DataFrame
market_summary_df = market.get_market_summary()
print("Market Summary DataFrame:")
print(market_summary_df.to_string(index=False))

print("\n✓ Object-oriented seller and market models initialized successfully!")

