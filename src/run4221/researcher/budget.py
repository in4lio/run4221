from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

from run4221.researcher.schemas import ResearchBudget


class BudgetCap(StrEnum):
    WALL_TIME = "wall_time_budget"
    TURNS = "agent_turn_budget"
    WEB_SEARCHES = "web_search_budget"
    OUTPUT_TOKENS = "output_token_budget"
    CANDIDATES = "candidate_budget"


class BudgetExhausted(RuntimeError):
    """Raised before a provider call that cannot fit inside the job budget."""

    def __init__(self, cap: BudgetCap) -> None:
        super().__init__(cap.value)
        self.cap = cap


@dataclass(frozen=True)
class ProviderCallLimits:
    max_turns: int
    max_output_tokens: int
    max_tool_calls: int | None
    max_retries: int
    wall_time_seconds: float


@dataclass(frozen=True)
class BudgetObservation:
    turns: int
    web_searches: int
    output_tokens: int | None


class JobBudgetTracker:
    """Mutable, process-local accounting for one otherwise stateless agent job."""

    def __init__(
        self,
        budget: ResearchBudget,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.policy = budget
        self._clock = clock
        self._deadline = clock() + budget.max_wall_time_seconds_per_job
        self.remaining_turns = budget.max_agent_turns_per_job
        self.remaining_web_searches = budget.max_web_searches_per_job
        self.remaining_output_tokens = budget.max_output_tokens_per_job
        self.remaining_retries = budget.max_retries_per_job

    def limits_for_call(
        self,
        *,
        needs_web_search: bool,
        reserve_assessment: bool = False,
    ) -> ProviderCallLimits:
        wall_time_seconds = self._deadline - self._clock()
        if wall_time_seconds <= 0:
            raise BudgetExhausted(BudgetCap.WALL_TIME)
        if self.remaining_turns <= 0:
            raise BudgetExhausted(BudgetCap.TURNS)
        if self.remaining_output_tokens <= 0:
            raise BudgetExhausted(BudgetCap.OUTPUT_TOKENS)
        if needs_web_search and self.remaining_web_searches <= 0:
            raise BudgetExhausted(BudgetCap.WEB_SEARCHES)

        max_turns = self.remaining_turns
        max_output_tokens = self.remaining_output_tokens
        max_retries = self.remaining_retries
        if reserve_assessment:
            max_turns -= 1
            max_output_tokens -= self._assessment_output_reserve()
            wall_time_seconds -= self._assessment_wall_time_reserve()
            if max_turns <= 0:
                raise BudgetExhausted(BudgetCap.TURNS)
            if max_output_tokens <= 0:
                raise BudgetExhausted(BudgetCap.OUTPUT_TOKENS)
            if wall_time_seconds <= 0:
                raise BudgetExhausted(BudgetCap.WALL_TIME)
            reserved_retries = min(1, self.remaining_retries)
            max_retries -= reserved_retries
            self.remaining_retries = reserved_retries
        else:
            self.remaining_retries = 0
        return ProviderCallLimits(
            max_turns=max_turns,
            max_output_tokens=max_output_tokens,
            max_tool_calls=(self.remaining_web_searches if needs_web_search else None),
            max_retries=max_retries,
            wall_time_seconds=wall_time_seconds,
        )

    def record(
        self,
        observation: BudgetObservation,
        *,
        preserve_assessment_reserve: bool = False,
    ) -> BudgetCap | None:
        turns = max(1, observation.turns)
        turn_overrun = turns > self.remaining_turns
        search_overrun = observation.web_searches > self.remaining_web_searches
        token_overrun = (
            observation.output_tokens is not None
            and observation.output_tokens > self.remaining_output_tokens
        )

        self.remaining_turns = max(0, self.remaining_turns - turns)
        self.remaining_web_searches = max(
            0,
            self.remaining_web_searches - max(0, observation.web_searches),
        )
        if observation.output_tokens is None:
            self.remaining_output_tokens = (
                min(
                    self.remaining_output_tokens,
                    self._assessment_output_reserve(),
                )
                if preserve_assessment_reserve
                else 0
            )
        else:
            self.remaining_output_tokens = max(
                0,
                self.remaining_output_tokens - max(0, observation.output_tokens),
            )

        if turn_overrun:
            return BudgetCap.TURNS
        if search_overrun:
            return BudgetCap.WEB_SEARCHES
        if token_overrun:
            return BudgetCap.OUTPUT_TOKENS
        return None

    def record_failed_call(
        self,
        *,
        exhaust_turns: bool = False,
        preserve_assessment_reserve: bool = False,
    ) -> None:
        if exhaust_turns:
            self.remaining_turns = 1 if preserve_assessment_reserve else 0
        else:
            floor = 1 if preserve_assessment_reserve else 0
            self.remaining_turns = max(floor, self.remaining_turns - 1)

    def candidate_cap_exceeded(self, count: int) -> bool:
        return count > self.policy.max_candidates_per_cycle

    def _assessment_output_reserve(self) -> int:
        return max(
            128,
            min(768, self.policy.max_output_tokens_per_job // 2),
        )

    def _assessment_wall_time_reserve(self) -> float:
        return min(15.0, self.policy.max_wall_time_seconds_per_job / 4)
