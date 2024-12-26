import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_simulations = 1000
num_days = 365 * 3
initial_points_value = .2
commission_rate = 0.01 * .05
daily_trades = 100000
trade_value = 1000
utilization_probability = 0.20
max_discount = 0.20

# Function to simulate TradePoints value


def simulate_tradepoints():
    points_value = initial_points_value
    points_values = []
    points_values_in_dollars = []
    total_points_available = 0
    for i in range(num_days):
        # Calculate points earned
        points_earned = daily_trades * trade_value * \
            commission_rate * np.random.rand() / points_value
        total_points_available += points_earned
        # print(" Points Earned: ", points_earned)
        # print(" Total Available Points: ", total_points_available)

        # Determine if points are used - assume that no burning happens in the first two days
        if (i > -1):
            discount_value = daily_trades * trade_value * max_discount
            points_used = min(discount_value * utilization_probability * np.random.rand() /
                              points_value, total_points_available * np.random.rand())
            total_points_available -= points_used
            points_value *= 1 + (points_used / total_points_available)

        if (points_earned == total_points_available or points_value < 0):
            points_value = initial_points_value
        else:
            points_value *= 1 - (points_earned / total_points_available)
        # print(" Points Value: ", points_value)
        # Put a cieling to the value:
        points_value = max(points_value, 1)

        points_values.append(points_value)
        points_values_in_dollars.append(points_value * total_points_available)

    return [points_values, points_values_in_dollars]


# Monte Carlo simulation
tradepoints_simulations = []
tradepoints_simulations_in_dollars = []
for _ in range(num_simulations):
    op = simulate_tradepoints()

    tradepoints_simulations.append(op[0])
    tradepoints_simulations_in_dollars.append(op[1])
# Calculate average points value over time
avg_tradepoints_value = np.mean(tradepoints_simulations, axis=0)
avg_tradepoints_value_in_dollars = np.mean(
    tradepoints_simulations_in_dollars, axis=0)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(avg_tradepoints_value, label='TradePoints Value')
plt.xlabel('Days')
plt.ylabel('Points Value')
plt.title('Monte Carlo Simulation of TradePoints Value Over a Year')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(avg_tradepoints_value_in_dollars, label='TradePoints Value')
plt.xlabel('Days')
plt.ylabel('Total Value')
plt.title('Monte Carlo Simulation of TradePoints Value Over a Year')
plt.legend()
plt.show()
