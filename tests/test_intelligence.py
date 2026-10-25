import logging
import time

from src.config_loader import GemmaConfig, load_config
from src.intelligence.cue_filter import CueFilter
from src.intelligence.gemma_client import (
    DEFAULT_MODEL,
    CoachMetrics,
    GemmaClient,
    build_prompt,
    make_cue,
    normalize_bullet,
    normalize_bullets,
    offline_cue,
    parse_cue_json,
)

_ENCOURAGE = (
    '{"priority":"LOW","cue_type":"ENCOURAGE","display_text":"Great pace",'
    '"icon":"ok","duration_ms":2000,"audio_chime":false}'
)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the Gemma worker")


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
        "bullets",
    }
    assert posture["bullets"] == ["Straighten your posture"]

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
    assert "bullets" in prompt
    assert "3 to 6 words" in prompt
    parsed = parse_cue_json(
        'sure {"priority":"LOW","cue_type":"PACE","display_text":"Slow down",'
        '"icon":"slow_down","duration_ms":3000,"audio_chime":false} thanks'
    )
    assert parsed["cue_type"] == "PACE"
    assert parse_cue_json("no json here") is None
    assert parse_cue_json('{"priority": "HIGH"}') is None


def test_sidebar_bullets_are_trimmed_capped_and_expanded():
    assert normalize_bullet("Speak up") == "Speak up a bit"
    assert normalize_bullet("Straighten Posture") == "Straighten your posture"
    assert normalize_bullet("Slow down") == "Slow down your pace"
    assert normalize_bullet("Look right") == "Look toward the right"
    assert normalize_bullet("Take a breath") == "Take a breath now"
    assert normalize_bullet("   ") is None
    assert normalize_bullet("one two three four five six seven") == (
        "one two three four five six"
    )
    assert normalize_bullets(
        ["", "  ", "Look right", "one two three four five six seven eight"]
    ) == ["Look toward the right", "one two three four five six"]
    capped = normalize_bullets(["a b c", "d e f", "g h i", "j k l", "m n o"])
    assert capped == ["a b c", "d e f", "g h i", "j k l"]
    assert normalize_bullets("Speak up a bit\n\nSlow down your pace") == [
        "Speak up a bit",
        "Slow down your pace",
    ]


def test_parse_keeps_sidebar_bullets_and_derives_them_when_missing():
    parsed = parse_cue_json(
        '{"priority":"HIGH","cue_type":"VOLUME","display_text":"Speak up",'
        '"icon":"volume_up","duration_ms":3000,"audio_chime":true,'
        '"bullets":["Project your voice now", "", "Pause between your points"]}'
    )
    assert parsed["bullets"] == ["Project your voice now", "Pause between your points"]
    derived = parse_cue_json(
        '{"priority":"MEDIUM","cue_type":"PACE","display_text":"Slow down",'
        '"icon":"slow_down","duration_ms":3000,"audio_chime":false}'
    )
    assert derived["cue_type"] == "PACE"
    assert derived["bullets"] == ["Slow down your pace"]
    for cue in (
        offline_cue(CoachMetrics(posture_alert=True)),
        offline_cue(CoachMetrics(speak_up=True)),
        offline_cue(CoachMetrics(rushing=True)),
        offline_cue(CoachMetrics(run_on=True)),
        offline_cue(
            CoachMetrics(
                gaze_cue="look_right", gaze_text="Look right", gaze_icon="arrow_right"
            )
        ),
    ):
        assert cue["bullets"]
        for bullet in cue["bullets"]:
            assert 3 <= len(bullet.split()) <= 6


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


def test_gemma_model_env_overrides_the_config(monkeypatch):
    assert GemmaConfig().model == DEFAULT_MODEL == "gemma-4-26b-a4b-it"
    assert load_config().gemma.model == DEFAULT_MODEL
    monkeypatch.setenv("GEMMA_MODEL", "gemma-custom-model")
    overridden = GemmaClient.from_env(load_config().gemma.model, 3000, offline=True)
    assert overridden.model == "gemma-custom-model"
    monkeypatch.setenv("GEMMA_MODEL", "   ")
    blank = GemmaClient.from_env(DEFAULT_MODEL, 3000, offline=True)
    assert blank.model == DEFAULT_MODEL


def test_poll_does_not_block_on_a_slow_transport():
    def slow(_prompt: str) -> str:
        time.sleep(1.5)
        return _ENCOURAGE

    client = GemmaClient(api_key="test-key", transport=slow, min_interval=0, timeout=5)
    metrics = CoachMetrics(posture_alert=True)
    started = time.perf_counter()
    cues = [client.poll(metrics, now=float(i)) for i in range(12)]
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    assert client.pending is True
    assert all(cue["cue_type"] == "POSTURE" for cue in cues)
    client.close()


def test_finished_gemma_result_is_shown_and_errors_are_counted_once(caplog):
    caplog.set_level(logging.WARNING, logger="wingman.gemma")
    metrics = CoachMetrics(posture_alert=True)
    good = GemmaClient(
        api_key="test-key",
        transport=lambda _prompt: _ENCOURAGE,
        min_interval=2,
        timeout=2,
    )
    assert good.poll(metrics, now=0)["cue_type"] == "POSTURE"
    _wait_until(lambda: not good.pending)
    shown = good.poll(metrics, now=0.2)
    assert shown["cue_type"] == "ENCOURAGE"
    assert good.succeeded == 1
    assert good.failed == 0
    assert good.used_fallback is False
    assert good.pending is False

    def raise_404(_prompt: str) -> str:
        error = RuntimeError("NOT_FOUND")
        error.code = 404
        raise error

    bad = GemmaClient(
        api_key="test-key", transport=raise_404, min_interval=1, timeout=2
    )
    bad.poll(metrics, now=0)
    _wait_until(lambda: bad.failed >= 1 and not bad.pending)
    assert bad.poll(metrics, now=0.1)["cue_type"] == "POSTURE"
    assert bad.used_fallback is True
    bad.poll(metrics, now=5)
    _wait_until(lambda: bad.failed >= 2 and not bad.pending)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "wingman.gemma"
    ]
    assert warnings == ["Gemma call failed: 404 NOT_FOUND"]
    assert bad.succeeded == 0

    messy = GemmaClient(
        api_key="test-key",
        transport=lambda _prompt: "not json",
        min_interval=0,
        timeout=2,
    )
    messy.poll(CoachMetrics(), now=0)
    _wait_until(lambda: messy.failed >= 1 and not messy.pending)
    assert messy.failed == 1
    assert any("unparseable" in record.getMessage() for record in caplog.records)


def test_a_timed_out_call_is_a_failure_and_poll_does_not_wait_for_it():
    def slow(_prompt: str) -> str:
        time.sleep(0.6)
        return _ENCOURAGE

    client = GemmaClient(
        api_key="test-key", transport=slow, min_interval=5, timeout=0.15
    )
    started = time.perf_counter()
    cue = client.poll(CoachMetrics(posture_alert=True), now=0)
    assert cue["cue_type"] == "POSTURE"
    assert time.perf_counter() - started < 0.15
    _wait_until(lambda: client.failed >= 1, timeout=1)
    assert time.perf_counter() - started < 0.45
    _wait_until(lambda: not client.pending, timeout=2)
    assert client.failed == 1
    assert client.succeeded == 0
    client.close()


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
