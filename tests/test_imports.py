def test_project_imports():
    # Simple smoke test: import the package modules that should exist
    import importlib

    modules = [
        "gs_lora",
        "gs_lora.datasets",
        "gs_lora.data",
        "gs_lora.data.cifar100",
        "gs_lora.lora",
        "gs_lora.lora.adapters",
        "gs_lora.models",
        "gs_lora.models.resnet",
        "gs_lora.training.continual",
        "gs_lora.training.unlearning",
        "gs_lora.cli.continual",
        "gs_lora.cli.unlearning",
        "gs_lora.run_gslora_cl",
        "gs_lora.unlearning_demo",
    ]

    for mod in modules:
        importlib.import_module(mod)
