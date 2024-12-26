import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


# Parameters
MIN_TXN_VALUE = 1000
MAX_TXN_VALUE = 100000

COMMISSION_RATE = .01
EARN_PCT = 0.05
MAX_DISCOUNT_ON_COMMISSION = .20
INIT_POINTS_VALUE = .2
INIT_POINTS_IN_MARKET = 100000

# Every these many transactions, we will revise the market price
POINT_VALUE_REVISION_COUNT = 100

# Simulations Params
NUM_SIMULATIONS = 4
DAYS_TO_SIMULATE = 100
NUMBER_OF_TXNS_PER_DAY = 10000


def get_random_value(lower, upper):
    mean = (lower + upper) / 2
    # Approximation to cover 99.7% of values within the range
    std_dev = (upper - lower) / 6
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    return truncnorm.rvs(a, b, loc=mean, scale=std_dev)


# FLOW:
# Do N transactions a day
# Combine the total # of txns and the points earned and burned for the day
# After N transactions, calculate the new market price
# Keep a count of fixed points earning scope
# Do a total market value with fixed, v/s dynamic
# Do all days for a year
# Do multiple runs of this simulation - do an avg value for every day


# Return transaction value, points earned, points burned
def make_transaction(points_value):
    transaction_val = get_random_value(MIN_TXN_VALUE, MAX_TXN_VALUE)
    # User can spend anywhere between 0 and max discount allowed on the commission on each transaction
    points_burned = get_random_value(
        0,  transaction_val * COMMISSION_RATE * MAX_DISCOUNT_ON_COMMISSION * np.random.rand() * 0.2) / points_value

    # Points are earned based commission_rate and earn_pct
    points_earned = (transaction_val - (points_burned *
                     points_value)) * COMMISSION_RATE * EARN_PCT

    points_earned_fixed = ((transaction_val - (points_burned /
                                               INIT_POINTS_VALUE)) * COMMISSION_RATE * EARN_PCT)

    return [transaction_val, points_earned, points_burned, points_earned_fixed]


def run_simulation():
    # points_value, market_points, total_points_earned, total_points_burned, fixed_pts_value
    txns_array = [[], []]
    retrn_array = [[], [], [], []]
    master_incr = 0
    points_value = INIT_POINTS_VALUE
    market_points = INIT_POINTS_IN_MARKET
    for _ in range(DAYS_TO_SIMULATE):
        # Possible number of transactions this day
        txn_count = np.random.randint(
            NUMBER_OF_TXNS_PER_DAY * .50, NUMBER_OF_TXNS_PER_DAY * 1.50)
        total_points_earned = 0
        total_points_burned = 0
        total_points_fixed = 0
        sum_of_points_value = 0

        for _ in range(txn_count):
            # Revise the points value
            master_incr += 1
            if (master_incr % POINT_VALUE_REVISION_COUNT == 0):
                points_value = points_value * \
                    (1 + (total_points_earned - total_points_burned)/market_points)

            # Make a txn
            trx = make_transaction(points_value)
            sum_of_points_value += points_value
            # Holding the current market points value, total points earned , points burned
            total_points_earned += trx[1]
            total_points_burned += trx[2]
            total_points_fixed += trx[3]
            market_points += total_points_earned - total_points_burned

            txns_array[0].append(trx[1] * points_value)
            txns_array[1].append(trx[2] * points_value)

            # print(points_value, market_points,
            #       trx[1] * points_value, trx[2] * points_value, trx[3] * INIT_POINTS_VALUE)
        # point value avg for the day
        retrn_array[0].append(sum_of_points_value / txn_count)
        # market points available for the day
        retrn_array[1].append(market_points)
        retrn_array[2].append(sum(txns_array[0]))  # value of points earned
        retrn_array[3].append(sum(txns_array[1]))  # value of points burned

    return retrn_array


# iterate through daily
simulation_op = []
incr = 0
for _ in range(NUM_SIMULATIONS):
    print("NUM_SIMULATIONS count ", incr)
    simulation_op.append(run_simulation())
    incr += 1
    # Input list
print((simulation_op))

# Extract the first elements
first_elements = [sublist[0] for sublist in simulation_op]
print(len(first_elements[0]), len(first_elements[1]))
# Convert to NumPy array
first_elements_array = np.array(first_elements)

# Calculate the mean for each position
mean_values = np.mean(first_elements_array, axis=0)

# Convert to list
plt.figure(figsize=(12, 6))
plt.plot(mean_values, label='TradePoints Value')
plt.xlabel('Days')
plt.ylabel('Points Value')
plt.title('Monte Carlo Simulation of TradePoints Value Over a Year')
plt.legend()
plt.show()

# Output
# print(mean_values)
