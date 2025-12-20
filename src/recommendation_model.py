class RecommendationModel:
    def predict(
        self,
        seating_score: float,
        waiting_people: int,
        restaurant_type: str
    ):
        recommendations = []

        if seating_score < 60:
            if restaurant_type == "dine_in":
                recommendations.append("Improve seating layout or increase seating capacity")
            else:
                recommendations.append("Reduce dine-in focus and prioritize delivery")

        if waiting_people > 5:
            recommendations.append("Introduce reservation or waiting list management")

        if not recommendations:
            recommendations.append("Current seating and demand strategy is adequate")

        return recommendations
