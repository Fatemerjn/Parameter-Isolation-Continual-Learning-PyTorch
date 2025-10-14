def test_project_imports():
    # Simple smoke test: import the package modules that should exist
    import importlib

    modules = [
        "gs_lora",
        "gs_lora.data_and_models",
        "gs_lora.lora_utils",
    ]

    for mod in modules:
        importlib.import_module(mod)
