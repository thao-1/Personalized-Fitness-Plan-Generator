from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import os
import pandas as pd
import numpy as np

from .models import PlanRequest, PlanResponse, ExerciseItem, DailyPlan


DATASET_PATH_CANDIDATES = [
    "megaGymDataset.csv",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "megaGymDataset.csv"),
    os.path.join(os.path.dirname(__file__), "megaGymDataset.csv"),
]


@dataclass
class Exercise:
    title: str
    body_part: str | None
    equipment: str | None
    level: str | None
    type: str | None
    mechanics: str | None
    primary_muscles: List[str] | None
    secondary_muscles: List[str] | None
    notes: str | None

    @staticmethod
    def from_row(row: pd.Series) -> "Exercise":
        def get(colnames: List[str]):
            for c in colnames:
                if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                    return str(row[c])
            return None

        def to_list(val: str | None) -> List[str] | None:
            if val is None:
                return None
            # Split on comma or semicolon
            parts = [p.strip() for p in str(val).replace(";", ",").split(",") if p.strip()]
            return parts if parts else None

        return Exercise(
            title=get(["Title", "title", "Exercise Name", "name"]) or "Unknown Exercise",
            body_part=get(["BodyPart", "body_part", "Body Part", "target", "muscle"]),
            equipment=get(["Equipment", "equipment"]),
            level=get(["Level", "level", "Difficulty"]),
            type=get(["Type", "type"]),
            mechanics=get(["Mechanics", "mechanics"]),
            primary_muscles=to_list(get(["PrimaryMuscles", "primary_muscles", "Primary Muscles"])) ,
            secondary_muscles=to_list(get(["SecondaryMuscles", "secondary_muscles", "Secondary Muscles"])) ,
            notes=get(["Instructions", "Description", "notes", "Instructions (short)"]),
        )


class Dataset:
    def __init__(self) -> None:
        self.df = self._load()
        self.exercises: List[Exercise] = [Exercise.from_row(r) for _, r in self.df.iterrows()]

    def _load(self) -> pd.DataFrame:
        last_err: Exception | None = None
        for path in DATASET_PATH_CANDIDATES:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    # Normalize columns for safer access
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise FileNotFoundError(f"Failed to load dataset from candidates: {DATASET_PATH_CANDIDATES}. Last error: {last_err}")
        raise FileNotFoundError(f"Dataset not found. Looked in: {DATASET_PATH_CANDIDATES}")


class Planner:
    def __init__(self) -> None:
        self.ds = Dataset()

    def _normalize_equipment(self, s: str) -> str:
        if not s:
            return "unknown"
        x = s.strip().lower()
        # common normalizations between UI and dataset
        mapping = {
            "bodyweight": "body only",
            "body weight": "body only",
            "none": "body only",  # treat none as body only moves
            "kettlebell": "kettlebells",
            "kb": "kettlebells",
            "band": "bands",
            "resistance band": "bands",
            "bar": "barbell",
            "dumbbells": "dumbbell",
        }
        return mapping.get(x, x)

    def generate_plan(self, req: PlanRequest) -> PlanResponse:
        # Determine training split by goal and days per week
        split = self._build_week_split(req.goal, req.days_per_week)
        # Candidate pool filtered by constraints
        candidates = self._filter_exercises(req)
        # Build daily workouts
        weekly_schedule: List[DailyPlan] = []
        for i, focus in enumerate(split):
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i]
            # Skip assigning exercises on rest days
            if "rest" in focus.lower():
                day_exs: List[Exercise] = []
            else:
                day_exs = self._pick_exercises_for_focus(candidates, focus, req, day_idx=i)
            weekly_schedule.append(
                DailyPlan(
                    day=day_name,
                    focus=focus,
                    exercises=[self._to_item(e) for e in day_exs]
                )
            )

        summary = self._summarize(req, weekly_schedule)
        metadata = {
            "num_candidates": len(candidates),
            "filters": {
                "equipment_available": req.equipment_available,
                "experience": req.experience,
                "injuries": req.injuries,
            },
        }
        return PlanResponse(summary=summary, weekly_schedule=weekly_schedule, metadata=metadata)

    # --- internals ---
    def _build_week_split(self, goal: str, days: int) -> List[str]:
        goal = goal.lower()
        splits = {
            3: ["Full Body", "Rest/Active Recovery", "Full Body"],
            4: ["Upper", "Lower", "Rest/Active Recovery", "Full Body"],
            5: ["Upper", "Lower", "Push", "Pull", "Legs"],
            6: ["Push", "Pull", "Legs", "Upper", "Lower", "Full Body"],
            7: ["Upper", "Lower", "Push", "Pull", "Legs", "Full Body", "Rest/Active Recovery"],
        }
        base = splits.get(days, ["Full Body"] * days)
        if goal in ("endurance",):
            base = ["Full Body" if b != "Rest/Active Recovery" else b for b in base]
        return base[:days]

    def _filter_exercises(self, req: PlanRequest) -> List[Exercise]:
        # Level filter
        level_rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
        req_level = level_rank.get(req.experience, 0)
        user_eq = set([self._normalize_equipment(e) for e in (req.equipment_available or [])])

        def pass_with_constraints(require_equipment: bool, require_level: bool) -> List[Exercise]:
            pool: List[Exercise] = []
            for e in self.ds.exercises:
                # injuries check
                if any(inj.lower() in (e.mechanics or "").lower() or inj.lower() in (e.title or "").lower() for inj in (req.injuries or [])):
                    continue
                # level check
                if require_level:
                    ex_level = level_rank.get((e.level or "").strip().lower(), 1)
                    if ex_level > req_level:
                        continue
                # equipment check
                ex_eq = self._normalize_equipment(e.equipment or "")
                # allow body-only synonyms: if user selected bodyweight, also accept 'none'
                if require_equipment and user_eq and ex_eq not in user_eq:
                    # special case: 'none' vs 'body only'
                    if ex_eq == "none" and "body only" in user_eq:
                        pass
                    else:
                        continue
                pool.append(e)
            return pool
            return tmp

        # Strict: respect equipment + level
        pool = pass_with_constraints(require_equipment=True, require_level=True)
        # Fallback 1: relax equipment
        if not pool:
            pool = pass_with_constraints(require_equipment=False, require_level=True)
        # Fallback 2: relax equipment and level
        if not pool:
            pool = pass_with_constraints(require_equipment=False, require_level=False)
        return pool

    def _pick_exercises_for_focus(self, pool: List[Exercise], focus: str, req: PlanRequest, day_idx: int = 0) -> List[Exercise]:
        # Use non-deterministic RNG (seeded by day index to vary days in the same run)
        rng = np.random.default_rng()
        focus = focus.lower()
        # target categories for focus
        categories = {
            "upper": ["chest", "back", "shoulders", "biceps", "triceps"],
            "lower": ["quadriceps", "hamstrings", "glutes", "calves"],
            "push": ["chest", "shoulders", "triceps"],
            "pull": ["back", "biceps"],
            "legs": ["quadriceps", "hamstrings", "glutes", "calves"],
            "full body": ["chest", "back", "shoulders", "quadriceps", "hamstrings", "glutes", "core"],
        }
        targets = categories.get(focus, categories.get("full body"))

        # Expand with common synonyms found in datasets
        synonyms = {
            "chest": ["pectorals", "pecs"],
            "back": ["lats", "latissimus dorsi", "middle back", "lower back", "upper back"],
            "shoulders": ["delts", "deltoids"],
            "quadriceps": ["quads"],
            "hamstrings": ["hams"],
            "glutes": ["gluteus", "gluteus maximus", "glute"],
            "calves": ["calf", "gastrocnemius", "soleus"],
            "core": ["abdominals", "abs", "obliques"],
            "biceps": ["bicep"],
            "triceps": ["tricep"],
        }
        expanded_targets = set()
        for t in targets:
            expanded_targets.add(t)
            for s in synonyms.get(t, []):
                expanded_targets.add(s)

        # score pool by how many target muscles match
        scored: List[tuple[float, Exercise]] = []
        for ex in pool:
            prim = [m.lower() for m in (ex.primary_muscles or [])]
            sec = [m.lower() for m in (ex.secondary_muscles or [])]
            score = 0
            for t in expanded_targets:
                if t in prim:
                    score += 2
                if t in sec:
                    score += 1
            # goal adjustment
            if req.goal.lower() == "strength" and ex.mechanics and ex.mechanics.lower() in {"compound"}:
                score += 0.5
            if req.goal.lower() == "fat_loss" and ex.type and ex.type.lower() in {"cardio", "plyometrics"}:
                score += 0.5
            if req.goal.lower() == "endurance" and ex.type and ex.type.lower() in {"cardio"}:
                score += 0.5
            if score > 0:
                scored.append((score, ex))

        # pick top-N with slight randomness; N based on available time
        n = max(5, min(10, int(req.time_per_workout_min / 12)))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[: 2 * n]]
        # If nothing scored, prefer compound or target-relevant movements
        if not top:
            target_set = set(expanded_targets)
            pref = [
                e for e in pool
                if (e.mechanics and e.mechanics.lower() == "compound")
                or any((m and m.lower() in target_set) for m in (e.primary_muscles or []))
            ]
            src = pref if pref else pool
            if len(src) <= n:
                # De-duplicate by title
                seen = set()
                out = []
                for e in src:
                    if e.title not in seen:
                        out.append(e)
                        seen.add(e.title)
                return out
            idx = rng.choice(len(src), size=n, replace=False)
            chosen = [src[i] for i in idx]
            # De-duplicate by title just in case
            seen = set()
            unique = []
            for e in chosen:
                if e.title not in seen:
                    unique.append(e)
                    seen.add(e.title)
            return unique
        if len(top) <= n:
            # De-duplicate by title
            seen = set()
            unique = []
            for e in top:
                if e.title not in seen:
                    unique.append(e)
                    seen.add(e.title)
            return unique
        idx = rng.choice(len(top), size=n, replace=False)
        picked = [top[i] for i in idx]
        # Try to diversify equipment if user has multiple options
        user_eq = set([e.lower() for e in (req.equipment_available or [])])
        if len(user_eq) > 1:
            eq_counts: Dict[str, int] = {}
            for e in picked:
                eq = (e.equipment or "").lower()
                eq_counts[eq] = eq_counts.get(eq, 0) + 1
            dominant_eq = max(eq_counts, key=lambda k: eq_counts[k]) if eq_counts else None
            if dominant_eq and eq_counts.get(dominant_eq, 0) > n // 2:
                # Attempt swaps with alternatives using other equipment in pool
                alternatives = [
                    e for e in top
                    if (e.equipment or "").lower() in user_eq and (e.equipment or "").lower() != dominant_eq
                ]
                for i in range(len(picked)):
                    if eq_counts.get((picked[i].equipment or "").lower(), "") == dominant_eq and alternatives:
                        repl = alternatives.pop(0)
                        picked[i] = repl
                        # update counts
                        eq_counts[dominant_eq] -= 1
        # Final de-duplication by title
        seen = set()
        unique = []
        for e in picked:
            if e.title not in seen:
                unique.append(e)
                seen.add(e.title)
        return unique

    def _to_item(self, e: Exercise) -> ExerciseItem:
        return ExerciseItem(
            title=e.title,
            body_part=e.body_part,
            equipment=e.equipment,
            level=e.level,
            type=e.type,
            mechanics=e.mechanics,
            primary_muscles=e.primary_muscles,
            secondary_muscles=e.secondary_muscles,
            notes=e.notes,
        )

    def _summarize(self, req: PlanRequest, days: List[DailyPlan]) -> str:
        return (
            f"Goal: {req.goal.replace('_', ' ').title()} | Experience: {req.experience.title()} | "
            f"Days/Week: {req.days_per_week} | Time/Workout: {req.time_per_workout_min} min"
        )
