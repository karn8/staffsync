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

# Task 2: Fairness in Scheduling (Minimize Max Deviation)
def solve_task2():
    total_required = DAILY_REQUIRED * len(DAYS)
    avg_hours = total_required / len(OPERATORS)

    model = pulp.LpProblem("Task2_Fairness", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    H = {op: pulp.lpSum(x[(op, d)] for d in DAYS) for op in OPERATORS}
    D = pulp.LpVariable("MaxDeviation", lowBound=0)

    model += D

    for op in OPERATORS:
        for d in DAYS:
            model += x[(op, d)] <= AVAILABILITY[op][d]

    for d in DAYS:
        model += pulp.lpSum(x[(op, d)] for op in OPERATORS) == DAILY_REQUIRED

    for op in OPERATORS:
        model += H[op] >= MIN_WEEK[op]
        model += H[op] - avg_hours <= D
        model += avg_hours - H[op] <= D

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    schedule = pd.DataFrame(index=OPERATORS, columns=DAYS)
    for op in OPERATORS:
        for d in DAYS:
            schedule.loc[op, d] = x[(op, d)].value()

    schedule["Weekly hours"] = schedule.sum(axis=1)

    return schedule, D.value()

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

