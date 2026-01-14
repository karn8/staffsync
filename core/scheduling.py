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


def solve_task2():

    total_required = DAILY_REQUIRED * len(DAYS)
    avg_hours = total_required / len(OPERATORS)

    model_fair = pulp.LpProblem("Minimise_Max_Deviation", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    H = {op: pulp.lpSum(x[(op, d)] for d in DAYS) for op in OPERATORS}
    D = pulp.LpVariable("MaxDeviation", lowBound=0)

    model_fair += D

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

    model_cost = pulp.LpProblem("Minimise_Cost_Given_Fairness", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "hours",
        ((op, d) for op in OPERATORS for d in DAYS),
        lowBound=0
    )

    H = {op: pulp.lpSum(x[(op, d)] for d in DAYS) for op in OPERATORS}

    model_cost += pulp.lpSum(
        WAGES[op] * x[(op, d)] for op in OPERATORS for d in DAYS
    )

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

