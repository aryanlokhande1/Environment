from pathlib import Path
from environment.models import (ArtifactLoader, CampaignDecisionPolicy, CampaignMotifModel,
                                PtpHazardModel, StageContinuationModel)

BUNDLE = Path(__file__).parents[1] / "artifacts" / "gold_events_v1"

def test_authoritative_models_load():
    assert len(ArtifactLoader(BUNDLE).validate()) == 26
    CampaignDecisionPolicy(BUNDLE)
    CampaignMotifModel(BUNDLE)
    PtpHazardModel(BUNDLE)
    StageContinuationModel(BUNDLE)
