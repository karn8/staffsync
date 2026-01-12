# core/data.py

OPERATORS = ["E. Khan", "Y. Chen", "A. Taylor", "R. Zidane", "R. Perez", "C. Santos"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

AVAILABILITY = {
    "E. Khan": {"Mon": 6, "Tue": 0, "Wed": 6, "Thu": 0, "Fri": 6},
    "Y. Chen": {"Mon": 0, "Tue": 6, "Wed": 0, "Thu": 6, "Fri": 0},
    "A. Taylor": {"Mon": 4, "Tue": 8, "Wed": 4, "Thu": 0, "Fri": 4},
    "R. Zidane": {"Mon": 5, "Tue": 5, "Wed": 5, "Thu": 0, "Fri": 5},
    "R. Perez": {"Mon": 3, "Tue": 0, "Wed": 3, "Thu": 8, "Fri": 0},
    "C. Santos": {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 6, "Fri": 2},
}

WAGES = {
    "E. Khan": 25,
    "Y. Chen": 26,
    "A. Taylor": 24,
    "R. Zidane": 23,
    "R. Perez": 28,
    "C. Santos": 30
}

BACHELORS = {"E. Khan", "Y. Chen", "A. Taylor", "R. Zidane"}

MIN_WEEK = {op: (8 if op in BACHELORS else 7) for op in OPERATORS}

DAILY_REQUIRED = 14

SKILLS = ["Programming", "Troubleshooting"]

OPERATOR_SKILLS = {
    "E. Khan": ["Programming"],
    "Y. Chen": ["Programming"],
    "A. Taylor": ["Troubleshooting"],
    "R. Zidane": ["Troubleshooting"],
    "R. Perez": ["Programming"],
    "C. Santos": ["Programming", "Troubleshooting"],
}
