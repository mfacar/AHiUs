class UserStory:
    role: str
    feature: str
    goal: str

    def __init__(self, groups):
        self.role = groups[1] if len(groups) > 1 else None
        self.feature = groups[3] if len(groups) > 3 else None
        self.goal = groups[5] if len(groups) > 5 else None
