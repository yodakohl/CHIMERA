from app.main import CUSTOM_ANALYSIS_MODEL, _resolve_analysis_model


def test_resolve_analysis_model_uses_legacy_by_default():
    config = _resolve_analysis_model(None)

    assert config.key == "legacy"
    assert config.backend == "legacy"
    assert config.model_id is None


def test_resolve_analysis_model_accepts_smolvlm_preset():
    config = _resolve_analysis_model("smolvlm2-500m")

    assert config.key == "smolvlm2-500m"
    assert config.backend == "smolvlm"
    assert config.model_id == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def test_resolve_analysis_model_accepts_custom_model():
    config = _resolve_analysis_model(
        CUSTOM_ANALYSIS_MODEL,
        "example-org/example-image-text-model",
    )

    assert config.key == CUSTOM_ANALYSIS_MODEL
    assert config.backend == "smolvlm"
    assert config.model_id == "example-org/example-image-text-model"


def test_resolve_analysis_model_falls_back_when_custom_is_blank():
    config = _resolve_analysis_model(CUSTOM_ANALYSIS_MODEL, " ")

    assert config.key == "legacy"
    assert config.backend == "legacy"
    assert config.model_id is None
