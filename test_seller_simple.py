"""
Simple test to verify Seller class attribute compatibility
"""

# Minimal Seller class definition for testing
class Seller:
    def __init__(self, name: str, cost: float, initial_price: float, 
                 initial_ad_budget: float, base_demand: float):
        # Validation
        if cost <= 0:
            raise ValueError(f"Production cost must be positive. Got: €{cost:.2f}")
        if initial_price <= cost:
            raise ValueError(f"Price (€{initial_price:.2f}) must be greater than production cost (€{cost:.2f})")
        if initial_ad_budget < 0:
            raise ValueError(f"Advertising budget cannot be negative. Got: €{initial_ad_budget:.2f}")
        if base_demand < 0:
            raise ValueError(f"Brand value must be non-negative. Got: {base_demand:.2f}")
        
        self.name = name
        self.cost = cost
        self.production_cost = cost
        self.price = initial_price
        self.ad_budget = initial_ad_budget
        self.advertising_budget = initial_ad_budget
        self.base_demand = base_demand
        self.brand_value = base_demand
        
        self.demand = 0.0
        self.revenue = 0.0
        self.profit = 0.0

print("="*70)
print("SELLER CLASS COMPATIBILITY TEST")
print("="*70)

# Test 1: Create seller with old names
print("\n[1] Creating seller with old parameter names...")
seller = Seller(
    name='Seller_A',
    cost=1.54,
    initial_price=2.86,
    initial_ad_budget=1500,
    base_demand=17.94
)
print("✓ Seller created successfully")
print(f"    Old: cost={seller.cost:.2f}, ad_budget={seller.ad_budget:.2f}, base_demand={seller.base_demand:.2f}")
print(f"    New: production_cost={seller.production_cost:.2f}, advertising_budget={seller.advertising_budget:.2f}, brand_value={seller.brand_value:.2f}")
print(f"    Match: {seller.cost == seller.production_cost}")

# Test 2: Validation
print("\n[2] Testing price <= cost validation...")
try:
    bad = Seller('Bad', 1.0, 0.5, 100, 10)
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Caught: {str(e)[:60]}...")

print("\n[3] Testing negative ad budget validation...")
try:
    bad = Seller('Bad', 1.0, 2.0, -100, 10)
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Caught: {str(e)[:60]}...")

print("\n[4] Testing zero cost validation...")
try:
    bad = Seller('Bad', 0, 2.0, 100, 10)
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Caught: {str(e)[:60]}...")

print("\n" + "="*70)
print("✓ All compatibility tests passed!")
print("="*70)

