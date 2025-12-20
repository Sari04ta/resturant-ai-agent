class SeatingSatisfactionModel:
    def predict(
        self,
        seating_capacity: int,
        seating_complaint_ratio: float,
        avg_seating_sentiment: float,
        restaurant_type: str
    ) -> float:
        """
        Returns seating satisfaction score (0–100)
        """

        # Base penalty from complaints
        penalty = seating_complaint_ratio * abs(avg_seating_sentiment) * 100

        # Context awareness
        if restaurant_type == "cloud_kitchen":
            penalty *= 0.3
        else:  # dine-in
            penalty *= 1.0

        # Small seating risk
        if seating_capacity < 10:
            penalty *= 1.2

        score = max(0.0, 100.0 - penalty)
        return round(score, 2)
