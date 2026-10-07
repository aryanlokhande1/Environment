"""Explicit code semantics, independent of immutable empirical bundle versions."""
LEGACY = "legacy-v2r2"
CORRECTED = "corrected-v1"
RUNTIME_VERSIONS = (LEGACY, CORRECTED)


def validate_runtime_version(value: str) -> str:
    if value not in RUNTIME_VERSIONS:
        raise ValueError(f"unsupported runtime_version: {value}")
    return value
