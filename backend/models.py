from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PlanRequest(BaseModel):
    age: int = Field(ge=13, le=100)
    sex: Optional[str] = Field(default=None, description="male/female/other")
    goal: str = Field(description="fat_loss | muscle_gain | strength | endurance | general_fitness")
    experience: str = Field(description="beginner | intermediate | advanced")
    injuries: Optional[List[str]] = Field(default_factory=list)
    days_per_week: int = Field(ge=1, le=7)
    time_per_workout_min: int = Field(ge=15, le=180)
    equipment_available: Optional[List[str]] = Field(default_factory=list)

class ExerciseItem(BaseModel):
    title: str
    body_part: Optional[str] = None
    equipment: Optional[str] = None
    level: Optional[str] = None
    type: Optional[str] = None
    mechanics: Optional[str] = None
    primary_muscles: Optional[List[str]] = None
    secondary_muscles: Optional[List[str]] = None
    notes: Optional[str] = None

class DailyPlan(BaseModel):
    day: str
    focus: str
    exercises: List[ExerciseItem]

class PlanResponse(BaseModel):
    summary: str
    weekly_schedule: List[DailyPlan]
    metadata: Dict[str, Any] = Field(default_factory=dict)
