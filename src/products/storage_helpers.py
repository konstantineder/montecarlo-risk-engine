from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import math

import torch


DATE_TOL = 1e-12
VOLUME_TOL = 1e-12


@dataclass(order=True)
class _RatePoint:
    point: float
    rate: float


@dataclass(order=True)
class _DatedCost:
    date: float
    cost: float


@dataclass
class _RateSchedule:
    start_date: float
    end_date: float
    values: list[_RatePoint] = field(default_factory=list)

    def contains(self, date: float) -> bool:
        return StorageConfig._date_in_window(self.start_date, self.end_date, date)


@dataclass
class _VolumeWindow:
    start_date: float
    end_date: float
    vmin: float
    vmax: float
    penalty: float = 0.0

    def contains(self, date: float) -> bool:
        return StorageConfig._date_in_window(self.start_date, self.end_date, date)


class StorageConfig:
    @staticmethod
    def _date_in_window(start_date: float, end_date: float, date: float) -> bool:
        if math.isclose(start_date, end_date, abs_tol=DATE_TOL):
            return math.isclose(start_date, date, abs_tol=DATE_TOL)
        return (start_date - DATE_TOL) <= date < (end_date - DATE_TOL)

    @staticmethod
    def grid_step(vmin: float, vmax: float, num_states: int) -> float:
        if num_states <= 1 or math.isclose(vmin, vmax, abs_tol=VOLUME_TOL):
            return 0.0
        return (vmax - vmin) / (num_states - 1.0)

    @staticmethod
    def state_scale(vmin: float, vmax: float, num_states: int) -> float:
        if num_states <= 1 or math.isclose(vmin, vmax, abs_tol=VOLUME_TOL):
            return 0.0
        return (num_states - 1.0) / (vmax - vmin)

    @staticmethod
    def _interpolate_rate(point: float, rate_points: list[_RatePoint]) -> float:
        if not rate_points:
            raise ValueError("Flexibility slice is empty.")

        if len(rate_points) == 1:
            return rate_points[0].rate

        points = [item.point for item in rate_points]
        rates = [item.rate for item in rate_points]

        if point <= points[0]:
            return rates[0]
        if point >= points[-1]:
            return rates[-1]

        upper_idx = bisect_right(points, point)
        lower_idx = upper_idx - 1

        x0 = points[lower_idx]
        x1 = points[upper_idx]
        y0 = rates[lower_idx]
        y1 = rates[upper_idx]

        if math.isclose(x0, x1, abs_tol=VOLUME_TOL):
            return y1

        weight = (point - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)

    @staticmethod
    def interpolate_rate_tensor(
        point: torch.Tensor,
        rate_points: list[_RatePoint],
    ) -> torch.Tensor:
        if not rate_points:
            raise ValueError("Flexibility slice is empty.")

        if len(rate_points) == 1:
            return torch.full_like(point, rate_points[0].rate)

        xp = point.new_tensor([item.point for item in rate_points])
        fp = point.new_tensor([item.rate for item in rate_points])

        indices = torch.bucketize(point, xp)
        left_idx = torch.clamp(indices - 1, min=0, max=len(rate_points) - 2)
        right_idx = left_idx + 1

        x0 = xp[left_idx]
        x1 = xp[right_idx]
        y0 = fp[left_idx]
        y1 = fp[right_idx]

        weights = torch.where(
            torch.isclose(x0, x1),
            torch.zeros_like(point),
            (point - x0) / (x1 - x0),
        )
        interpolated = y0 + weights * (y1 - y0)
        interpolated = torch.where(point <= xp[0], fp[0], interpolated)
        interpolated = torch.where(point >= xp[-1], fp[-1], interpolated)
        return interpolated

    def __init__(self):
        self.initial_volume_constraints: list[_VolumeWindow] = []
        self.volume_constraints: list[_VolumeWindow] = []
        self.injection_flexibility: list[_RateSchedule] = []
        self.withdrawal_flexibility: list[_RateSchedule] = []
        self.injection_costs: list[_DatedCost] = []
        self.withdrawal_costs: list[_DatedCost] = []

    def add_volume_constraint(
        self,
        start_date: float,
        end_date: float,
        vmin: float,
        vmax: float,
        penalty: float = 0.0,
    ) -> None:
        self.initial_volume_constraints.append(
            _VolumeWindow(start_date, end_date, vmin, vmax, penalty)
        )
        self.initial_volume_constraints.sort(key=lambda item: item.start_date)

    def _get_volume_window(
        self,
        date: float,
        constraints: list[_VolumeWindow],
    ) -> _VolumeWindow:
        for constraint in constraints:
            if constraint.contains(date):
                return constraint
        if not constraints:
            raise ValueError("No volume constraints configured.")
        return constraints[-1]

    def get_initial_volume_constraint(self, date: float) -> _VolumeWindow:
        return self._get_volume_window(date, self.initial_volume_constraints)

    def get_volume_constraint(self, date: float) -> _VolumeWindow:
        constraints = self.volume_constraints or self.initial_volume_constraints
        return self._get_volume_window(date, constraints)

    def _add_rate_schedule(
        self,
        container: list[_RateSchedule],
        start_date: float,
        end_date: float,
        point: float,
        rate: float,
    ) -> None:
        for schedule in container:
            if math.isclose(schedule.start_date, start_date, abs_tol=DATE_TOL) and math.isclose(
                schedule.end_date, end_date, abs_tol=DATE_TOL
            ):
                schedule.values.append(_RatePoint(point, rate))
                schedule.values.sort(key=lambda item: item.point)
                return

        container.append(
            _RateSchedule(
                start_date=start_date,
                end_date=end_date,
                values=[_RatePoint(point, rate)],
            )
        )
        container.sort(key=lambda item: item.start_date)

    def _get_rate_schedule(
        self,
        date: float,
        container: list[_RateSchedule],
    ) -> list[_RatePoint]:
        for schedule in container:
            if schedule.contains(date):
                return schedule.values
        if not container:
            raise ValueError("No flexibility slice configured.")
        return container[-1].values

    def add_injection_flexibility(
        self,
        start_date: float,
        end_date: float,
        point: float,
        rate: float,
    ) -> None:
        self._add_rate_schedule(
            self.injection_flexibility,
            start_date,
            end_date,
            point,
            rate,
        )

    def get_injection_flexibility_slice(self, date: float) -> list[_RatePoint]:
        return self._get_rate_schedule(date, self.injection_flexibility)

    def get_injection_flexibility_rate(self, date: float, point: float) -> float:
        return self._interpolate_rate(point, self.get_injection_flexibility_slice(date))

    def add_withdrawal_flexibility(
        self,
        start_date: float,
        end_date: float,
        point: float,
        rate: float,
    ) -> None:
        self._add_rate_schedule(
            self.withdrawal_flexibility,
            start_date,
            end_date,
            point,
            rate,
        )

    def get_withdrawal_flexibility_slice(self, date: float) -> list[_RatePoint]:
        return self._get_rate_schedule(date, self.withdrawal_flexibility)

    def get_withdrawal_flexibility_rate(self, date: float, point: float) -> float:
        return self._interpolate_rate(point, self.get_withdrawal_flexibility_slice(date))

    def _add_dated_cost(
        self,
        container: list[_DatedCost],
        date: float,
        cost: float,
    ) -> None:
        container.append(_DatedCost(date, cost))
        container.sort(key=lambda item: item.date)

    def _get_dated_cost(
        self,
        date: float,
        container: list[_DatedCost],
    ) -> float:
        if not container:
            raise ValueError("No variable costs configured.")

        dates = [item.date for item in container]
        lower = bisect_left(dates, date)

        if lower == len(container):
            return container[-1].cost
        if lower == 0 or math.isclose(container[lower].date, date, abs_tol=DATE_TOL):
            return container[lower].cost
        return container[lower - 1].cost

    def add_variable_injection_cost(self, date: float, cost: float) -> None:
        self._add_dated_cost(self.injection_costs, date, cost)

    def get_variable_injection_cost(self, date: float) -> float:
        return self._get_dated_cost(date, self.injection_costs)

    def add_variable_withdrawal_cost(self, date: float, cost: float) -> None:
        self._add_dated_cost(self.withdrawal_costs, date, cost)

    def get_variable_withdrawal_cost(self, date: float) -> float:
        return self._get_dated_cost(date, self.withdrawal_costs)

    def _tighten_boundary_to_preserve_reachability(
        self,
        date_i: float,
        period: float,
        index: int,
        optimize_vmax: bool,
        constraints: list[_VolumeWindow],
    ) -> None:
        convergence = float("inf")

        if optimize_vmax:
            vmax_ip1 = constraints[index + 1].vmax
            vmax_i_lower = vmax_ip1
            vmax_i_upper = constraints[index].vmax
            convergence_threshold = (vmax_i_upper - vmax_i_lower) / 1000.0

            while convergence > convergence_threshold:
                vmax_mid = vmax_i_lower + 0.5 * (vmax_i_upper - vmax_i_lower)
                vmax_flex_wd = self.get_withdrawal_flexibility_rate(date_i, vmax_mid) * period

                if vmax_mid - vmax_flex_wd <= vmax_ip1:
                    vmax_i_lower = vmax_mid
                else:
                    vmax_i_upper = vmax_mid

                convergence = vmax_i_upper - vmax_i_lower

            constraints[index].vmax = vmax_i_lower
            return

        vmin_ip1 = constraints[index + 1].vmin
        vmin_i_upper = vmin_ip1
        vmin_i_lower = constraints[index].vmin
        convergence_threshold = (vmin_i_upper - vmin_i_lower) / 1000.0

        while convergence > convergence_threshold:
            vmin_mid = vmin_i_upper - 0.5 * (vmin_i_upper - vmin_i_lower)
            vmin_flex_inj = self.get_injection_flexibility_rate(date_i, vmin_mid) * period

            if vmin_mid + vmin_flex_inj <= vmin_ip1:
                vmin_i_lower = vmin_mid
            else:
                vmin_i_upper = vmin_mid

            convergence = vmin_i_upper - vmin_i_lower

        constraints[index].vmin = vmin_i_upper

    def optimize_volume_constraints(
        self,
        start_date: float,
        end_date: float,
        rollout_interval: float,
        initial_volume: float,
    ) -> None:
        dates: list[float] = []
        initial_constraints: list[_VolumeWindow] = []
        optimized_constraints: list[_VolumeWindow] = []

        date = start_date
        while date <= end_date + DATE_TOL:
            next_date = min(date + rollout_interval, end_date)
            constraint = self.get_initial_volume_constraint(date)
            vmin = constraint.vmin
            vmax = constraint.vmax

            if math.isclose(date, start_date, abs_tol=DATE_TOL):
                vmin = initial_volume
                vmax = initial_volume

            initial_constraints.append(constraint)
            optimized_constraints.append(
                _VolumeWindow(
                    start_date=date,
                    end_date=next_date,
                    vmin=vmin,
                    vmax=vmax,
                    penalty=constraint.penalty,
                )
            )
            dates.append(date)

            if date >= end_date - DATE_TOL:
                break
            date = next_date

        restart = True
        while restart:
            restart = False

            for i in range(len(optimized_constraints) - 1):
                date_i = optimized_constraints[i].start_date
                period = dates[i + 1] - dates[i]

                vmax_i = optimized_constraints[i].vmax
                vmax_ip1 = optimized_constraints[i + 1].vmax
                vmin_i = optimized_constraints[i].vmin
                vmin_ip1 = optimized_constraints[i + 1].vmin

                vmax_flex_wd = self.get_withdrawal_flexibility_rate(date_i, vmax_i) * period
                vmin_flex_wd = self.get_withdrawal_flexibility_rate(date_i, vmin_i) * period
                vmax_flex_inj = self.get_injection_flexibility_rate(date_i, vmax_i) * period
                vmin_flex_inj = self.get_injection_flexibility_rate(date_i, vmin_i) * period

                if vmax_i < vmax_ip1:
                    if vmax_i + vmax_flex_inj < vmax_ip1:
                        optimized_constraints[i + 1].vmax = vmax_i + vmax_flex_inj
                else:
                    if vmax_i - vmax_flex_wd > vmax_ip1:
                        self._tighten_boundary_to_preserve_reachability(
                            date_i=date_i,
                            period=period,
                            index=i,
                            optimize_vmax=True,
                            constraints=optimized_constraints,
                        )
                        restart = True

                if vmin_i < vmin_ip1:
                    if vmin_i + vmin_flex_inj < vmin_ip1:
                        self._tighten_boundary_to_preserve_reachability(
                            date_i=date_i,
                            period=period,
                            index=i,
                            optimize_vmax=False,
                            constraints=optimized_constraints,
                        )
                        restart = True
                else:
                    if vmin_i - vmin_flex_wd > vmin_ip1:
                        optimized_constraints[i + 1].vmin = vmin_i - vmin_flex_wd

                violated_i = (
                    optimized_constraints[i].vmin > initial_constraints[i].vmax
                    or optimized_constraints[i].vmax < initial_constraints[i].vmin
                )
                violated_ip1 = (
                    optimized_constraints[i + 1].vmin > initial_constraints[i + 1].vmax
                    or optimized_constraints[i + 1].vmax < initial_constraints[i + 1].vmin
                )

                if violated_i or violated_ip1:
                    violated_date = dates[i] if violated_i else dates[i + 1]
                    raise ValueError(
                        f"Initial volume constraints cannot be satisfied at date {violated_date}."
                    )

                if restart:
                    break

        self.volume_constraints = optimized_constraints
