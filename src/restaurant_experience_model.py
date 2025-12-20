from src.models.seating_model import SeatingSatisfactionModel
from src.models.waiting_model import WaitingListModel
from src.models.recommendation_model import RecommendationModel


class RestaurantExperienceAgent:
    def __init__(self):
        self.seating_model = SeatingSatisfactionModel()
        self.waiting_model = WaitingListModel()
        self.reco_model = RecommendationModel()

    def run(
        self,
        seating_capacity,
        seating_complaint_ratio,
        avg_seating_sentiment,
        restaurant_type,
        competition_intensity,
        avg_delivery_time
    ):
        seating_score = self.seating_model.predict(
            seating_capacity,
            seating_complaint_ratio,
            avg_seating_sentiment,
            restaurant_type
        )

        waiting_people, waiting_time = self.waiting_model.predict(
            seating_capacity,
            competition_intensity,
            avg_delivery_time
        )

        recommendations = self.reco_model.predict(
            seating_score,
            waiting_people,
            restaurant_type
        )

        return {
            "seating_satisfaction_score": seating_score,
            "estimated_waiting_people": waiting_people,
            "estimated_wait_time_min": waiting_time,
            "recommendations": recommendations
        }
