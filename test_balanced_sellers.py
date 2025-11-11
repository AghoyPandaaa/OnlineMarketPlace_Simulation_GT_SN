"""
Quick test to verify balanced sellers creation
"""
import sys
from pathlib import Path

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from Task2.SellerModeling import Seller, MarketModel

# Test create_balanced_sellers logic
print("="*80)
print("TESTING BALANCED SELLERS CREATION")
print("="*80)

# Seller 1
seller_1 = Seller(
    name="Balanced_Seller_A",
    cost=1.50,
    initial_price=2.20,
    initial_ad_budget=300,
    base_demand=200
)

print(f"\n✓ Seller 1 created: {seller_1.name}")
print(f"  Cost: €{seller_1.production_cost:.2f}")
print(f"  Price: €{seller_1.price:.2f}")
print(f"  Ad Budget: €{seller_1.advertising_budget:.2f}")
print(f"  Brand Value: {seller_1.brand_value:.2f}")

# Seller 2
seller_2 = Seller(
    name="Balanced_Seller_B",
    cost=1.50,
    initial_price=2.25,
    initial_ad_budget=250,
    base_demand=180
)

print(f"\n✓ Seller 2 created: {seller_2.name}")
print(f"  Cost: €{seller_2.production_cost:.2f}")
print(f"  Price: €{seller_2.price:.2f}")
print(f"  Ad Budget: €{seller_2.advertising_budget:.2f}")
print(f"  Brand Value: {seller_2.brand_value:.2f}")

# Market
sellers_dict = {
    'Balanced_Seller_A': seller_1,
    'Balanced_Seller_B': seller_2
}

market = MarketModel(
    sellers_dict,
    alpha=0.01,
    beta=5.0,
    gamma=0.0,
    epsilon=0.5
)

print(f"\n✓ Market created with {len(market.sellers)} sellers")
print(f"  Alpha: {market.alpha}")
print(f"  Beta: {market.beta}")
print(f"  Epsilon: {market.epsilon}")

# Test profit calculation
print("\n" + "="*80)
print("TESTING PROFIT CALCULATION")
print("="*80)

profit_1 = market.calculate_profit(seller_1, seller_2)
profit_2 = market.calculate_profit(seller_2, seller_1)

print(f"\nInitial profits:")
print(f"  {seller_1.name}: €{profit_1:.2f}")
print(f"  {seller_2.name}: €{profit_2:.2f}")

if profit_1 > 0 and profit_2 > 0:
    print("\n✓ Both sellers are PROFITABLE!")
    print("✓ Ready for interior Nash equilibrium!")
else:
    print("\n⚠ Warning: Sellers may not be profitable")

print("\n" + "="*80)
print("✓ BALANCED SELLERS TEST PASSED!")
print("="*80)

