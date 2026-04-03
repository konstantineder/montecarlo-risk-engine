import numpy as np
import torch


class SimulationResults:
    """
    Container for simulation results

    - Contains both metric outputs and corresponding derivatives if differentiation is enabled
    - If computation of second order derivatives are enabled these are stored as well
    - Results are converted into numpy arrays
    """

    def __init__(
        self,
        results,
        derivatives,
        second_derivatives,
        netting_set_names: list[str] | None = None,
        metric_names: list[str] | None = None,
        model_param_names: list[str] | None = None,
        product_names: list[str] | None = None,
    ):
        self.results = self._to_numpy_nested(results)
        self.derivatives = self._to_numpy_nested(derivatives)
        self.second_derivatives = self._to_numpy_nested(second_derivatives)
        num_netting_sets = len(self.results)
        num_metrics = len(self.results[0]) if num_netting_sets > 0 else 0

        if netting_set_names is not None and product_names is not None and netting_set_names != product_names:
            raise ValueError("Provide either 'netting_set_names' or legacy alias 'product_names', not conflicting values.")

        resolved_netting_set_names = netting_set_names if netting_set_names is not None else product_names
        self.netting_set_names = resolved_netting_set_names if resolved_netting_set_names is not None else [
            f"netting_set_{idx}" for idx in range(num_netting_sets)
        ]
        self.product_names = self.netting_set_names
        self.metric_names = metric_names if metric_names is not None else [
            f"metric_{idx}" for idx in range(num_metrics)
        ]
        self.model_param_names = model_param_names if model_param_names is not None else []

        self._netting_set_name_to_idx = {
            name.lower(): idx for idx, name in enumerate(self.netting_set_names)
        }
        self._metric_name_to_idx = {
            name.lower(): idx for idx, name in enumerate(self.metric_names)
        }
        self._model_param_name_to_idx = {
            name.lower(): idx for idx, name in enumerate(self.model_param_names)
        }

    @staticmethod
    def _extract_legacy_argument(
        legacy_kwargs: dict,
        new_name: str,
        legacy_names: tuple[str, ...],
    ):
        value = None
        for legacy_name in legacy_names:
            if legacy_name in legacy_kwargs:
                legacy_value = legacy_kwargs.pop(legacy_name)
                if value is None:
                    value = legacy_value
                elif legacy_value != value:
                    raise ValueError(
                        f"Conflicting values provided for '{new_name}' and legacy alias '{legacy_name}'."
                    )
        return value

    @staticmethod
    def _raise_on_unexpected_kwargs(legacy_kwargs: dict):
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    def _to_numpy_nested(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_numpy_nested(x) for x in obj)
        return obj

    def _resolve_netting_set_idx(self, netting_set: int | str) -> int:
        if isinstance(netting_set, str):
            key = netting_set.lower()
            if key not in self._netting_set_name_to_idx:
                raise KeyError(
                    f"Unknown netting set name '{netting_set}'. Available: {self.netting_set_names}"
                )
            return self._netting_set_name_to_idx[key]
        return netting_set

    def _resolve_metric_idx(self, metric: int | str) -> int:
        if isinstance(metric, str):
            key = metric.lower()
            if key not in self._metric_name_to_idx:
                raise KeyError(f"Unknown metric name '{metric}'. Available: {self.metric_names}")
            return self._metric_name_to_idx[key]
        return metric

    def _resolve_param_idx(self, param: int | str) -> int:
        if isinstance(param, str):
            key = param.lower()
            if key not in self._model_param_name_to_idx:
                raise KeyError(
                    f"Unknown model parameter name '{param}'. Available: {self.model_param_names}"
                )
            return self._model_param_name_to_idx[key]
        return param

    def get_product_names(self) -> list[str]:
        return list(self.netting_set_names)

    def get_netting_set_names(self) -> list[str]:
        return list(self.netting_set_names)

    def get_metric_names(self) -> list[str]:
        return list(self.metric_names)

    def get_model_param_names(self) -> list[str]:
        return list(self.model_param_names)

    def get_results(
        self,
        netting_set=None,
        metric=None,
        evaluation_idx: int | None = None,
        **legacy_kwargs,
    ):
        """Get metric outputs for each netting set and metric."""
        legacy_netting_set = self._extract_legacy_argument(
            legacy_kwargs,
            "netting_set",
            ("prod_idx", "product", "product_idx"),
        )
        legacy_metric = self._extract_legacy_argument(
            legacy_kwargs,
            "metric",
            ("metric_idx", "metric_set_idx"),
        )
        legacy_evaluation_idx = self._extract_legacy_argument(
            legacy_kwargs,
            "evaluation_idx",
            ("evaluation_index",),
        )
        self._raise_on_unexpected_kwargs(legacy_kwargs)

        if netting_set is None:
            netting_set = legacy_netting_set
        if metric is None:
            metric = legacy_metric
        if evaluation_idx is None:
            evaluation_idx = legacy_evaluation_idx

        netting_set_idx = self._resolve_netting_set_idx(netting_set)
        metric_idx = self._resolve_metric_idx(metric)
        results = self.results[netting_set_idx][metric_idx]
        simulation_results = [result[0] for result in results]
        simulation_results = np.array(simulation_results)
        if evaluation_idx is None:
            return simulation_results
        return simulation_results[evaluation_idx]

    def get_mc_error(
        self,
        netting_set=None,
        metric=None,
        evaluation_idx: int | None = None,
        **legacy_kwargs,
    ):
        """Get MC errors for metric outputs for each netting set and metric."""
        legacy_netting_set = self._extract_legacy_argument(
            legacy_kwargs,
            "netting_set",
            ("prod_idx", "product", "product_idx"),
        )
        legacy_metric = self._extract_legacy_argument(
            legacy_kwargs,
            "metric",
            ("metric_idx", "metric_set_idx"),
        )
        legacy_evaluation_idx = self._extract_legacy_argument(
            legacy_kwargs,
            "evaluation_idx",
            ("evaluation_index",),
        )
        self._raise_on_unexpected_kwargs(legacy_kwargs)

        if netting_set is None:
            netting_set = legacy_netting_set
        if metric is None:
            metric = legacy_metric
        if evaluation_idx is None:
            evaluation_idx = legacy_evaluation_idx

        netting_set_idx = self._resolve_netting_set_idx(netting_set)
        metric_idx = self._resolve_metric_idx(metric)
        results = self.results[netting_set_idx][metric_idx]
        mc_errors = [result[1] for result in results]
        mc_errors = np.array(mc_errors)
        if evaluation_idx is None:
            return mc_errors
        return mc_errors[evaluation_idx]

    def get_derivatives(
        self,
        netting_set=None,
        metric=None,
        param: int | str | None = None,
        evaluation_idx: int | None = None,
        **legacy_kwargs,
    ):
        """
        Get first order derivatives for each metric output
        For this the label 'differentiate' in the simulation
        controller needs to be enabled
        """
        legacy_netting_set = self._extract_legacy_argument(
            legacy_kwargs,
            "netting_set",
            ("prod_idx", "product", "product_idx"),
        )
        legacy_metric = self._extract_legacy_argument(
            legacy_kwargs,
            "metric",
            ("metric_idx", "metric_set_idx"),
        )
        legacy_evaluation_idx = self._extract_legacy_argument(
            legacy_kwargs,
            "evaluation_idx",
            ("evaluation_index",),
        )
        self._raise_on_unexpected_kwargs(legacy_kwargs)

        if netting_set is None:
            netting_set = legacy_netting_set
        if metric is None:
            metric = legacy_metric
        if evaluation_idx is None:
            evaluation_idx = legacy_evaluation_idx

        netting_set_idx = self._resolve_netting_set_idx(netting_set)
        metric_idx = self._resolve_metric_idx(metric)
        derivatives = self.derivatives[netting_set_idx][metric_idx]

        if param is None and evaluation_idx is None:
            return derivatives

        if evaluation_idx is not None:
            derivatives = derivatives[evaluation_idx]
            if param is None:
                return {
                    name: derivatives[idx] for idx, name in enumerate(self.model_param_names)
                }
            return derivatives[self._resolve_param_idx(param)]

        param_idx = self._resolve_param_idx(param)
        return np.array([evaluation[param_idx] for evaluation in derivatives])

    def get_second_derivatives(
        self,
        netting_set=None,
        metric=None,
        param1: int | str | None = None,
        param2: int | str | None = None,
        evaluation_idx: int | None = None,
        **legacy_kwargs,
    ):
        """
        Get second order derivatives for each metric output
        For this the label 'requires_higher_order_derivatives'
        in the simulation controller needs to be enables
        """
        legacy_netting_set = self._extract_legacy_argument(
            legacy_kwargs,
            "netting_set",
            ("prod_idx", "product", "product_idx"),
        )
        legacy_metric = self._extract_legacy_argument(
            legacy_kwargs,
            "metric",
            ("metric_idx", "metric_set_idx"),
        )
        legacy_evaluation_idx = self._extract_legacy_argument(
            legacy_kwargs,
            "evaluation_idx",
            ("evaluation_index",),
        )
        self._raise_on_unexpected_kwargs(legacy_kwargs)

        if netting_set is None:
            netting_set = legacy_netting_set
        if metric is None:
            metric = legacy_metric
        if evaluation_idx is None:
            evaluation_idx = legacy_evaluation_idx

        netting_set_idx = self._resolve_netting_set_idx(netting_set)
        metric_idx = self._resolve_metric_idx(metric)
        second_derivatives = self.second_derivatives[netting_set_idx][metric_idx]

        if param1 is None and param2 is None and evaluation_idx is None:
            return second_derivatives

        def row_to_named_dict(row):
            return {name: row[idx] for idx, name in enumerate(self.model_param_names)}

        def hessian_to_named_dict(hessian):
            return {
                row_name: row_to_named_dict(hessian[idx])
                for idx, row_name in enumerate(self.model_param_names)
            }

        if evaluation_idx is not None:
            second_derivatives = second_derivatives[evaluation_idx]
            if param1 is None and param2 is None:
                return hessian_to_named_dict(second_derivatives)
            if param1 is not None and param2 is None:
                return row_to_named_dict(second_derivatives[self._resolve_param_idx(param1)])
            if param1 is None and param2 is not None:
                col_idx = self._resolve_param_idx(param2)
                return {
                    row_name: second_derivatives[idx][col_idx]
                    for idx, row_name in enumerate(self.model_param_names)
                }
            return second_derivatives[self._resolve_param_idx(param1)][
                self._resolve_param_idx(param2)
            ]

        if param1 is not None and param2 is not None:
            row_idx = self._resolve_param_idx(param1)
            col_idx = self._resolve_param_idx(param2)
            return np.array([evaluation[row_idx][col_idx] for evaluation in second_derivatives])

        raise ValueError(
            "When evaluation_idx is omitted, provide both param1 and param2 or neither."
        )
