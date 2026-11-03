from __future__ import annotations

from importlib import import_module

_METHOD_MODULES = {
    "moe_pba": "moe_pba.methods.moe_pba",
    "moe_pba_greedy": "moe_pba.methods.moe_pba_greedy",
}


def _load_method(name: str):
    module = import_module(_METHOD_MODULES[name])
    return module.run


METHOD_REGISTRY = {
    name: (lambda *args, _name=name, **kwargs: _load_method(_name)(*args, **kwargs))
    for name in _METHOD_MODULES
}


def __getattr__(name: str):
    if name in _METHOD_MODULES:
        return import_module(_METHOD_MODULES[name])
    raise AttributeError(name)


__all__ = [
    "METHOD_REGISTRY",
    "moe_pba",
    "moe_pba_greedy",
]
