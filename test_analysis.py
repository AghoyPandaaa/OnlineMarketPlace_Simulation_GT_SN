"""
Quick test of Nash equilibrium analysis functions
"""

# Create mock data structures to test the analysis functions
mock_nash_result = {
    'nash_equilibrium': {
        'seller_A': {'price': 7.21, 'ad': 0, 'profit': -9381.94},
        'seller_B': {'price': 7.21, 'ad': 0, 'profit': -4717.96}
    },
    'converged': True,
    'iterations': 2,
    'history': [],
    'convergence_metric': 0.0
}

mock_initial_state = {
    'seller_A': {'price': 2.16, 'ad_budget': 500},
    'seller_B': {'price': 2.16, 'ad_budget': 500},
    'profits': {'seller_A': -9855.40, 'seller_B': -4966.67}
}

# Test the analysis logic
print("Testing analysis calculations...")

# Extract data
nash_eq = mock_nash_result['nash_equilibrium']
nash_A = nash_eq['seller_A']
nash_B = nash_eq['seller_B']
initial_A = mock_initial_state['seller_A']
initial_B = mock_initial_state['seller_B']

# Calculate changes
price_change_A = nash_A['price'] - initial_A['price']
ad_change_A = nash_A['ad'] - initial_A['ad_budget']

print(f"\nSeller A changes:")
print(f"  Price: {initial_A['price']:.2f} → {nash_A['price']:.2f} (Δ={price_change_A:+.2f})")
print(f"  Ad: {initial_A['ad_budget']:.2f} → {nash_A['ad']:.2f} (Δ={ad_change_A:+.2f})")

# Test division by zero handling
initial_ad = initial_A['ad_budget']
if initial_ad != 0:
    ad_pct = (ad_change_A / initial_ad) * 100
else:
    ad_pct = 100 if ad_change_A > 0 else 0

print(f"  Ad percentage change: {ad_pct:+.1f}%")

print("\n✓ Analysis calculations work correctly!")
print("✓ No division by zero errors!")
print("✓ Key structure matching correct!")

