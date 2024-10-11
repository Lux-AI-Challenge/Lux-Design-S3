class Team:
    def __init__(self, team_id: int, agent: str, points: int) -> None:
        self.team_id = team_id
        self.agent = agent
        self.points = points
    def __str__(self) -> str:
        return f"[Player {self.team_id}]"