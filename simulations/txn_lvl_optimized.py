import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from numba import jit

# Parameters
MIN_TXN_VALUE = 1000
MAX_TXN_VALUE = 100000
COMMISSION_RATE = 0.01
EARN_PCT = 0.05
MAX_DISCOUNT_ON_COMMISSION = 0.20
INIT_POINTS_VALUE = 0.2
INIT_POINTS_IN_MARKET = 100000
POINT_VALUE_REVISION_COUNT = 100
NUM_SIMULATIONS = 4
DAYS_TO_SIMULATE = 200
NUMBER_OF_TXNS_PER_DAY = 100000
BURN_PROB = 0.02
TRXS_PER_DAY = np.random.randint(
    NUMBER_OF_TXNS_PER_DAY * 0.50, NUMBER_OF_TXNS_PER_DAY * 1.50, size=DAYS_TO_SIMULATE
)


@jit
def get_random_value(lower, upper):
    mean = (lower + upper) / 2
    # Approximation to cover 99.7% of values within the range
    std_dev = (upper - lower) / 6

    return np.random.normal(mean, std_dev)


@jit
def make_transaction(points_value):
    transaction_val = get_random_value(MIN_TXN_VALUE, MAX_TXN_VALUE)
    if np.random.rand() < BURN_PROB:
        points_burned = (
            get_random_value(
                0, transaction_val * COMMISSION_RATE * MAX_DISCOUNT_ON_COMMISSION
            )
            / points_value
        )
    else:
        points_burned = 0

    points_earned = (
        (transaction_val - (points_burned * points_value)) * COMMISSION_RATE * EARN_PCT
    )
    points_earned_fixed = (
        (transaction_val - (points_burned * INIT_POINTS_VALUE))
        * COMMISSION_RATE
        * EARN_PCT
    )
    return transaction_val, points_earned, points_burned, points_earned_fixed


@jit(parallel=True)
def run_simulation():
    print("Simulation Started")
    points_value = INIT_POINTS_VALUE
    market_points = INIT_POINTS_IN_MARKET
    retrn_array = np.zeros((4, DAYS_TO_SIMULATE))
    master_incr = 0

    # For every transaction, is there a possibility of burn?

    for i in range(DAYS_TO_SIMULATE):
        txn_count = TRXS_PER_DAY[i]
        total_points_earned = 0.0
        total_points_burned = 0.0
        total_points_fixed = 0.0
        sum_of_points_value = 0.0

        for _ in range(txn_count):
            master_incr += 1
            if master_incr % POINT_VALUE_REVISION_COUNT == 0:
                points_value *= (
                    1 + (total_points_earned - total_points_burned) / market_points
                )

            trx = make_transaction(points_value)
            sum_of_points_value += points_value
            total_points_earned += trx[1]
            total_points_burned += trx[2]
            total_points_fixed += trx[3]
            market_points += total_points_earned - total_points_burned

        retrn_array[0, i] = sum_of_points_value / txn_count
        retrn_array[1, i] = market_points
        retrn_array[2, i] = total_points_earned * points_value
        retrn_array[3, i] = total_points_burned * points_value

    return retrn_array


simulation_op = [run_simulation() for _ in range(NUM_SIMULATIONS)]

first_elements = np.array([sublist[0] for sublist in simulation_op])
mean_values = np.mean(first_elements, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(mean_values, label="TradePoints Value")
plt.xlabel("Days")
plt.ylabel("Points Value")
plt.title("Monte Carlo Simulation of TradePoints Value Over a Year")
plt.legend()
plt.show()

print(simulation_op)
