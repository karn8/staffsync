# sampling.py
# =========================================
# SAMPLING MODULE
# Task 4: Random Sample & Descriptive Analysis
# Task 5: Statistical Inference
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =================================================
# TASK 4: Random Sample & Descriptive Analysis
# =================================================

def task4_descriptive_analysis(sample: pd.DataFrame):
    """
    Performs descriptive analysis and returns:
    - numeric variables
    - categorical variables
    - numerical summary
    - covariance matrix
    - correlation matrix
    """

    numeric_vars = sample.select_dtypes(include=np.number).columns
    categorical_vars = sample.select_dtypes(exclude=np.number).columns

    num_summary = sample[numeric_vars].describe().T
    cov_matrix = sample[numeric_vars].cov()
    cor_matrix = sample[numeric_vars].corr()

    return {
        "sample_shape": sample.shape,
        "numeric_vars": list(numeric_vars),
        "categorical_vars": list(categorical_vars),
        "numerical_summary": num_summary,
        "covariance_matrix": cov_matrix,
        "correlation_matrix": cor_matrix
    }


def task4_visualisations(sample: pd.DataFrame):
    """
    Generates all Task 4 visualisations
    """

    sns.set(style="whitegrid")

    # ---- FIGURE 1: Numeric Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    numeric_cols = [
        "Age",
        "LoS",
        "noofinvestigation",
        "nooftreatment",
        "noofpatients"
    ]

    for i, col in enumerate(numeric_cols):
        sns.histplot(sample[col], kde=True, ax=axes[i])
        axes[i].set_title(f"{col} distribution")

    plt.tight_layout()
    plt.show()

    # ---- FIGURE 2: Categorical Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    categorical_cols = ["DayofWeek", "Breachornot", "HRG"]

    for ax, col in zip(axes, categorical_cols):
        sample[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"{col} distribution")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    # ---- FIGURE 3: Relationships
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(
        x="Breachornot",
        y="LoS",
        data=sample,
        ax=axes[0]
    )
    axes[0].set_title("Length of Stay by Breach Status")

    sns.scatterplot(
        x="noofinvestigation",
        y="LoS",
        data=sample,
        ax=axes[1]
    )
    axes[1].set_title("Investigations vs Length of Stay")

    sns.scatterplot(
        x="noofpatients",
        y="LoS",
        data=sample,
        ax=axes[2]
    )
    axes[2].set_title("Congestion vs Length of Stay")

    plt.tight_layout()
    plt.show()


# =================================================
# TASK 5: Statistical Inference
# =================================================

def task5_confidence_intervals(sample: pd.DataFrame):
    """
    Computes confidence intervals for:
    - Mean Length of Stay
    - Breach rate
    """

    los = sample["LoS"]
    n = len(los)

    # ---- CI for mean LoS
    mean_los = los.mean()
    std_los = los.std(ddof=1)

    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin_los = t_crit * std_los / np.sqrt(n)
    ci_los = (mean_los - margin_los, mean_los + margin_los)

    # ---- CI for breach rate
    breach = sample["Breachornot"].map({"breach": 1, "non-breach": 0})
    p_hat = breach.mean()

    z = stats.norm.ppf(0.975)
    margin_breach = z * np.sqrt(p_hat * (1 - p_hat) / n)
    ci_breach = (p_hat - margin_breach, p_hat + margin_breach)

    return {
        "mean_los": mean_los,
        "ci_los": ci_los,
        "breach_rate": p_hat,
        "ci_breach_rate": ci_breach
    }


def task5_hypothesis_test(sample: pd.DataFrame):
    """
    Performs hypothesis test and effect size calculation
    """

    los_breach = sample[sample["Breachornot"] == "breach"]["LoS"]
    los_nonbreach = sample[sample["Breachornot"] == "non-breach"]["LoS"]

    t_stat, p_two = stats.ttest_ind(
        los_breach,
        los_nonbreach,
        equal_var=False
    )

    p_one = p_two / 2

    pooled_sd = np.sqrt(
        (los_breach.var(ddof=1) + los_nonbreach.var(ddof=1)) / 2
    )

    cohen_d = (los_breach.mean() - los_nonbreach.mean()) / pooled_sd

    return {
        "mean_los_breach": los_breach.mean(),
        "mean_los_nonbreach": los_nonbreach.mean(),
        "t_statistic": t_stat,
        "one_sided_p_value": p_one,
        "cohens_d": cohen_d
    }
