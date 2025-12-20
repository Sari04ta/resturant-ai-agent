class WaitingListModel:
    def predict(
        self,
        seating_capacity: int,
        competition_intensity: float,
        avg_delivery_time: float
    ):
        """
        Returns estimated waiting people & waiting time (minutes)
        """

        demand_factor = (competition_intensity / 100) * (avg_delivery_time / 30)

        waiting_people = max(0, int(seating_capacity * demand_factor))
        waiting_time = waiting_people * 5  # 5 min per person

        return waiting_people, waiting_time
