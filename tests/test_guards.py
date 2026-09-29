from environment.core.guards import can_approve_aip, can_push_to_partner

def test_application_scoped_business_guards():
    state = {"form_filled_seen": True, "professional_details_seen": False,
             "aip_approved_seen": False, "done": False}
    assert not can_approve_aip(state)
    state["professional_details_seen"] = True
    assert can_approve_aip(state)
    assert not can_push_to_partner(state)
    state["aip_approved_seen"] = True
    assert can_push_to_partner(state)
