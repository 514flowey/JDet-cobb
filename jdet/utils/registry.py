# reference mmcv.utils.registry
class Registry:
    def __init__(self):
        self._modules = {}

    def register_module(self, name=None):
        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self._modules,f"{key} is already registered."
            self._modules[key]=module
            return module
        return _register_module

    def get(self,name):
        assert name in self._modules,f"{name} is not registered."
        return self._modules[name]


def build_from_cfg(cfg,registry,**kwargs):
    if isinstance(cfg,str):
        return registry[cfg](**kwargs)
    elif isinstance(cfg,dict):
        args = cfg.copy()
        obj_type = args.pop('type')
        obj_cls = registry.get(obj_type)
        return obj_cls(**args,**kwargs)
    else:
        raise TypeError(f"type {type(cfg)} not support")


DATASETS = Registry()
TRANSFORMS = Registry()
META_ARCHS = Registry()
BACKBONES = Registry()
ROI_HEADS = Registry()
LOSSES = Registry()
OPTIMS = Registry()
SOLVERS = Registry()
HOOKs = Registry()


