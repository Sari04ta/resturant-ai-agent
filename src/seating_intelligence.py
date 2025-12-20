# Seating Intelligence Agent
# Adds seating satisfaction & waiting estimation (hooked later)

def seating_satisfaction_score(seating_capacity, seating_complaint_ratio, restaurant_type):
    penalty = seating_complaint_ratio * 100
    if restaurant_type == "cloud_kitchen":
        penalty *= 0.3
    if seating_capacity < 10:
        penalty *= 1.2
    return max(0, 100 - penalty)


def estimate_waiting(seating_capacity, demand_factor):
    waiting_people = int(seating_capacity * demand_factor)
    wait_time_min = waiting_people * 5
    return waiting_people, wait_time_min