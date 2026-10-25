from src.intelligence.cue_filter import CueFilter
from src.intelligence.gemma_client import (
    CoachMetrics,
    GemmaClient,
    build_prompt,
    make_cue,
    offline_cue,
    parse_cue_json,
)


def test_offline_fallback_ranks_posture_then_volume_then_pace():
    posture = offline_cue(CoachMetrics(posture_alert=True, speak_up=True, rushing=True))
    assert posture["cue_type"] == "POSTURE"
    assert posture["display_text"] == "Straighten Posture"
    assert posture["icon"] == "arrow_up"
    assert posture["priority"] == "HIGH"
    assert posture["audio_chime"] is True
    assert set(posture) == {
        "priority",
        "cue_type",
        "display_text",
        "icon",
        "duration_ms",
        "audio_chime",
    }

    volume = offline_cue(CoachMetrics(speak_up=True, wpm=140))
    assert volume["cue_type"] == "VOLUME"
    pace = offline_cue(CoachMetrics(rushing=True, wpm=210))
    assert pace["display_text"] == "Slow down"
    breath = offline_cue(CoachMetrics(run_on=True, wpm=140))
    assert breath["display_text"] == "Take a breath"
    gaze = offline_cue(
        CoachMetrics(
            gaze_cue="look_right", gaze_text="Look right", gaze_icon="arrow_right"
        )
    )
    assert gaze["cue_type"] == "GAZE"
    confused = offline_cue(CoachMetrics(audience_confused=True))
    assert "reiterat" in confused["display_text"].lower()
    assert offline_cue(CoachMetrics()) is None


def test_prompt_includes_the_live_metrics_and_json_parser_rejects_junk():
    prompt = build_prompt(CoachMetrics(volume_db=37.5, wpm=190, posture_alert=True))
    assert "37.5" in prompt
    assert "190" in prompt
    assert "posture_alert: True" in prompt
    parsed = parse_cue_json(
        'sure {"priority":"LOW","cue_type":"PACE","display_text":"Slow down",'
        '"icon":"slow_down","duration_ms":3000,"audio_chime":false} thanks'
    )
    assert parsed["cue_type"] == "PACE"
    assert parse_cue_json("no json here") is None
    assert parse_cue_json('{"priority": "HIGH"}') is None


def test_client_uses_transport_json_and_falls_back_when_it_is_garbage():
    metrics = CoachMetrics(posture_alert=True)

    def good_transport(_prompt: str) -> str:
        return (
            '{"priority":"LOW","cue_type":"ENCOURAGE","display_text":"Great pace",'
            '"icon":"ok","duration_ms":2000,"audio_chime":false}'
        )

    client = GemmaClient(api_key="test-key", transport=good_transport)
    cue = client.arbitrate(metrics)
    assert cue["cue_type"] == "ENCOURAGE"
    assert client.used_fallback is False

    broken = GemmaClient(api_key="test-key", transport=lambda _prompt: "not json")
    fallback = broken.arbitrate(metrics)
    assert fallback["cue_type"] == "POSTURE"
    assert broken.used_fallback is True

    quiet = GemmaClient(api_key=None)
    assert quiet.arbitrate(CoachMetrics()) is None
    assert quiet.used_fallback is True


def test_none_priority_from_the_model_suppresses_the_cue():
    client = GemmaClient(
        api_key="test-key",
        transport=lambda _prompt: (
            '{"priority":"NONE","cue_type":"NONE","display_text":"",'
            '"icon":"none","duration_ms":0,"audio_chime":false}'
        ),
    )
    assert client.arbitrate(CoachMetrics(posture_alert=True)) is None


def test_cooldown_blocks_repeated_alerts_and_snooze_blocks_everything():
    cue_filter = CueFilter(
        cooldown_seconds=5.0, snooze_seconds=30.0, global_gap_seconds=1.0
    )
    posture = make_cue("HIGH", "POSTURE", "Straighten Posture", "arrow_up", 3000, True)
    volume = make_cue("HIGH", "VOLUME", "Speak up", "volume_up", 100, True)

    assert cue_filter.consider(posture, 0.0)["cue_type"] == "POSTURE"
    assert cue_filter.fired_at == 0.0
    assert cue_filter.consider(posture, 2.0)["cue_type"] == "POSTURE"
    assert cue_filter.fired_at is None
    assert cue_filter.consider(posture, 5.0)["cue_type"] == "POSTURE"

    cue_filter.dismiss(5.2)
    assert cue_filter.active is None
    assert cue_filter.consider(posture, 5.4) is None

    assert cue_filter.consider(volume, 6.2)["cue_type"] == "VOLUME"
    cue_filter.snooze(6.3)
    assert cue_filter.consider(posture, 10.0) is None
    assert cue_filter.is_snoozed(10.0) is True
    assert cue_filter.consider(volume, 36.3)["cue_type"] == "VOLUME"


def test_global_gap_spaces_different_cue_types():
    cue_filter = CueFilter(cooldown_seconds=5.0, global_gap_seconds=1.0)
    first = make_cue("HIGH", "POSTURE", "Straighten Posture", "arrow_up", 100, True)
    second = make_cue("MEDIUM", "GAZE", "Look right", "arrow_right", 1000, False)
    assert cue_filter.consider(first, 0.0) is not None
    assert cue_filter.consider(second, 0.2) is None
    assert cue_filter.consider(second, 1.0)["cue_type"] == "GAZE"
