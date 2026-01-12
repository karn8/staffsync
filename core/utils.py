def get_operator_hours(schedule, operator):
    return schedule.loc[operator, "Weekly hours"]

def get_day_schedule(schedule, day):
    return schedule[day]
