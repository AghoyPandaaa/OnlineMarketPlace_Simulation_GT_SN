"""
Test script to verify Seller class updates work correctly
"""
import sys
sys.path.insert(0, 'Task2')

# Import only the Seller class (not the whole module which runs code)
import importlib.util
spec = importlib.util.spec_from_file_location("seller_module", "Task2/SellerModeling.py")

print("Testing Seller class updates...")
print("="*60)

# Read the Seller class definition directly
exec(open('Task2/SellerModeling.py').read().split('# 1. Load the cleaned dataset')[0])

# Test 1: Creating seller with old parameter names
print("\n[Test 1] Creating seller with old parameter names...")
try:
    seller = Seller(
        name='Test_Seller_A',
        cost=1.50,
        initial_price=2.50,
        initial_ad_budget=500,
        base_demand=25.0
    )
    print("✓ Seller created successfully")
    print(f"  Name: {seller.name}")
    print(f"  Old attributes: cost={seller.cost:.2f}, ad_budget={seller.ad_budget:.2f}, base_demand={seller.base_demand:.2f}")
    print(f"  New attributes: production_cost={seller.production_cost:.2f}, advertising_budget={seller.advertising_budget:.2f}, brand_value={seller.brand_value:.2f}")
    print(f"  Attributes match: {seller.cost == seller.production_cost and seller.ad_budget == seller.advertising_budget}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Validation - price must be > cost
print("\n[Test 2] Testing validation: price <= cost...")
try:
    bad_seller = Seller('Bad', 1.0, 0.5, 100, 10)
    print("✗ Validation failed - should have raised ValueError")
except ValueError as e:
    print(f"✓ Validation working: {e}")

# Test 3: Validation - negative ad budget
print("\n[Test 3] Testing validation: negative ad budget...")
try:
    bad_seller = Seller('Bad', 1.0, 2.0, -100, 10)
    print("✗ Validation failed - should have raised ValueError")
except ValueError as e:
    print(f"✓ Validation working: {e}")

# Test 4: Validation - zero/negative cost
print("\n[Test 4] Testing validation: zero cost...")
try:
    bad_seller = Seller('Bad', 0, 2.0, 100, 10)
    print("✗ Validation failed - should have raised ValueError")
except ValueError as e:
    print(f"✓ Validation working: {e}")

# Test 5: Update strategy method
print("\n[Test 5] Testing update_strategy method...")
try:
    seller.update_strategy(3.0, 600)
    print(f"✓ Strategy updated successfully")
    print(f"  New price: €{seller.price:.2f}")
    print(f"  New ad_budget: €{seller.ad_budget:.2f}")
    print(f"  New advertising_budget: €{seller.advertising_budget:.2f}")
    print(f"  Both updated: {seller.ad_budget == seller.advertising_budget}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 6: Profit margin calculation
print("\n[Test 6] Testing profit margin...")
try:
    margin = seller.get_profit_margin()
    expected = seller.price - seller.production_cost
    print(f"✓ Profit margin: €{margin:.2f}")
    print(f"  Calculated correctly: {abs(margin - expected) < 0.01}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*60)
print("All tests completed!")

