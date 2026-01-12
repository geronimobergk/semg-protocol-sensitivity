from __future__ import annotations

from typing import Any, Dict, Mapping, Type

from .st_cnn_gn import ST_Attn_CNN_GN, ST_CNN_GN

ModelCtor = Type


MODEL_REGISTRY: Dict[str, ModelCtor] = {
    "ST_CNN_GN": ST_CNN_GN,
    "ST_Attn_CNN_GN": ST_Attn_CNN_GN,
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def get_model(name: str) -> ModelCtor:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown model architecture: {name}. Available: {', '.join(available_models())}"
        ) from e


def create_model(name: str, params: Mapping[str, Any]) -> Any:
    """Create a model by name with the given parameters."""
    model_cls = get_model(name)
    kwargs = dict(params)
    return model_cls(**kwargs)
