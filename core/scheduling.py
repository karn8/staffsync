# core/scheduling.py

import pulp
import pandas as pd
from core.data import *

# Task 1: Minimize Cost (LP)
def solve_task1():
    model = pulp.LpProblem("Task1_Cost_Min", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0,
        cat=pulp.LpInteger
    )

    model += pulp.lpSum(WAGES[op] * x[(op, d)] for op in OPERATORS for d in DAYS)

    for op in OPERATORS:
        for d in DAYS:
            model += x[(op, d)] <= AVAILABILITY[op][d]

    for d in DAYS:
        model += pulp.lpSum(x[(op, d)] for op in OPERATORS) == DAILY_REQUIRED

    for op in OPERATORS:
        model += pulp.lpSum(x[(op, d)] for d in DAYS) >= MIN_WEEK[op]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    schedule = pd.DataFrame(index=OPERATORS, columns=DAYS)
    for op in OPERATORS:
        for d in DAYS:
            schedule.loc[op, d] = x[(op, d)].value()

    schedule["Weekly hours"] = schedule.sum(axis=1)
    cost = pulp.value(model.objective)

    return schedule, cost

'''# Task 2: Fairness in Scheduling (Minimize Max Deviation)
def solve_task2():
    total_required = DAILY_REQUIRED * len(DAYS)
    avg_hours = total_required / len(OPERATORS)

    model = pulp.LpProblem("Task2_Fairness", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    H = {op: pulp.lpSum(x[(op, d)] for d in DAYS) for op in OPERATORS}
    D = pulp.LpVariable("MaxDeviation", lowBound=0)

    # Objective: Minimize the maximum deviation from the average
    model += D

    # Constraints
    for op in OPERATORS:
        for d in DAYS:
            model += x[(op, d)] <= AVAILABILITY[op][d]

    for d in DAYS:
        model += pulp.lpSum(x[(op, d)] for op in OPERATORS) == DAILY_REQUIRED

    for op in OPERATORS:
        model += H[op] >= MIN_WEEK[op]
        # Fairness constraints: |H_op - average| <= D
        model += H[op] - avg_hours <= D
        model += avg_hours - H[op] <= D

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Calculate results
    schedule = pd.DataFrame(index=OPERATORS, columns=DAYS)
    for op in OPERATORS:
        for d in DAYS:
            schedule.loc[op, d] = x[(op, d)].value()

    schedule["Weekly hours"] = schedule.sum(axis=1)

    # Calculate Total Cost
    total_cost = sum(WAGES[op] * x[(op, d)].value() 
                     for op in OPERATORS for d in DAYS)

    return schedule, D.value(), total_cost'''


def solve_task2():

    total_required = DAILY_REQUIRED * len(DAYS)
    avg_hours = total_required / len(OPERATORS)

    # -------------------------------
    # STEP 1: Minimise Max Deviation
    # -------------------------------
    model_fair = pulp.LpProblem("Minimise_Max_Deviation", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    H = {op: pulp.lpSum(x[(op, d)] for d in DAYS) for op in OPERATORS}
    D = pulp.LpVariable("MaxDeviation", lowBound=0)

    # Objective
    model_fair += D

    # Constraints
    for op in OPERATORS:
        for d in DAYS:
            model_fair += x[(op, d)] <= AVAILABILITY[op][d]

    for d in DAYS:
        model_fair += pulp.lpSum(x[(op, d)] for op in OPERATORS) == DAILY_REQUIRED

    for op in OPERATORS:
        model_fair += H[op] >= MIN_WEEK[op]
        model_fair += H[op] - avg_hours <= D
        model_fair += avg_hours - H[op] <= D

    model_fair.solve(pulp.PULP_CBC_CMD(msg=False))
    D_star = D.value()

    # ----------------------------------------
    # STEP 2: Minimise Cost Given Fairness D*
    # ----------------------------------------
    model_cost = pulp.LpProblem("Minimise_Cost_Given_Fairness", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    H = {op: pulp.lpSum(x[(op, d)] for d in DAYS) for op in OPERATORS}

    # Objective: total wage cost
    model_cost += pulp.lpSum(
        WAGES[op] * x[(op, d)] for op in OPERATORS for d in DAYS
    )

    # Constraints
    for op in OPERATORS:
        for d in DAYS:
            model_cost += x[(op, d)] <= AVAILABILITY[op][d]

    for d in DAYS:
        model_cost += pulp.lpSum(x[(op, d)] for op in OPERATORS) == DAILY_REQUIRED

    for op in OPERATORS:
        model_cost += H[op] >= MIN_WEEK[op]
        model_cost += H[op] - avg_hours <= D_star
        model_cost += avg_hours - H[op] <= D_star

    model_cost.solve(pulp.PULP_CBC_CMD(msg=False))

    # -------------------------------
    # Results
    # -------------------------------
    schedule = pd.DataFrame(index=OPERATORS, columns=DAYS)
    for op in OPERATORS:
        for d in DAYS:
            schedule.loc[op, d] = x[(op, d)].value()

    schedule["Weekly hours"] = schedule.sum(axis=1)

    total_cost = pulp.value(model_cost.objective)

    return schedule, D_star, total_cost

# Task 3: Skill-Based Scheduling
def solve_task3():
    model = pulp.LpProblem("Task3_Skills", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    model += pulp.lpSum(WAGES[op] * x[(op, d)] for op in OPERATORS for d in DAYS)

    for op in OPERATORS:
        for d in DAYS:
            model += x[(op, d)] <= AVAILABILITY[op][d]

    for d in DAYS:
        model += pulp.lpSum(x[(op, d)] for op in OPERATORS) == DAILY_REQUIRED

    for op in OPERATORS:
        model += pulp.lpSum(x[(op, d)] for d in DAYS) >= MIN_WEEK[op]

    for d in DAYS:
        for skill in SKILLS:
            model += pulp.lpSum(
                x[(op, d)]
                for op in OPERATORS
                if skill in OPERATOR_SKILLS[op]
            ) >= 6

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    schedule = pd.DataFrame(index=OPERATORS, columns=DAYS)
    for op in OPERATORS:
        for d in DAYS:
            schedule.loc[op, d] = x[(op, d)].value()

    schedule["Weekly hours"] = schedule.sum(axis=1)
    cost = pulp.value(model.objective)

    return schedule, cost

