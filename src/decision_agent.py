# Decision Agent
# Converts analytics into recommendations

def recommend(seating_score, waiting_people):
    actions = []
    if seating_score < 60:
        actions.append("Improve seating or layout")
    if waiting_people > 5:
        actions.append("Introduce waiting list / reservations")
    if not actions:
        actions.append("Current strategy is adequate")
    return actions