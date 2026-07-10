def test_drm_parity_template_matches_upstream_single_prompt():
    from src.templates import iwildcam_drm_template

    prompts = [template("deer") for template in iwildcam_drm_template]

    assert prompts == ["a photo of deer."]
