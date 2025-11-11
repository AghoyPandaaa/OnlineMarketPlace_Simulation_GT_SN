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