import logging
import threading
import time

from src.config_loader import GemmaConfig, load_config
from src.intelligence.cue_filter import CueFilter
from src.intelligence.gemma_client import (
    DEFAULT_MODEL,
    CoachMetrics,
    CoachWindow,
    GemmaClient,
    active_cues,
    build_prompt,
    deterministic_report,
    http_timeout_ms,
    make_cue,
    normalize_bullet,
    normalize_bullets,
    offline_cue,
    parse_cue_json,
    parse_insight_json,
    render_report_markdown,
)
from google.genai import types

from src.intelligence.insight_client import (
    DEFAULT_INSIGHT_MODEL,
    InsightClient,
    flash_request_config,
)
from src.intelligence.session_stats import save_report

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


def test_prompt_is_compact_and_json_fences_still_parse():
    prompt = build_prompt(
        CoachWindow(
            avg_db=37, avg_wpm=190, posture_ratio=0.4, gaze_away_ratio=0.2, seconds=45
        )
    )
    assert "db=37" in prompt
    assert "wpm=190" in prompt
    assert "posture=0.40" in prompt
    assert "gaze=0.20" in prompt
    assert "bullets" in prompt
    assert "You are Wingman" not in prompt
    assert len(prompt) < 240
    fenced = parse_insight_json('```json\n{"bullets":["Posture slipping often"]}\n```')
    assert fenced == ["Posture slipping often"]
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
    window = CoachWindow(posture_ratio=0.8, avg_db=35, seconds=40)

    def good_transport(_prompt: str) -> str:
        return '{"bullets":["Posture slipping often"]}'

    client = InsightClient(api_key="test-key", transport=good_transport)
    bullets = client.insights(window)
    assert bullets == ["Posture slipping often"]
    assert client.used_fallback is False
    assert "db=" in build_prompt(window)

    broken = InsightClient(api_key="test-key", transport=lambda _prompt: "not json")
    assert broken.insights(window) is None
    assert broken.used_fallback is True

    quiet = InsightClient(api_key=None)
    assert quiet.insights(CoachWindow()) is None
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
    assert load_config().insight.model == DEFAULT_INSIGHT_MODEL
    monkeypatch.setenv("INSIGHT_MODEL", "gemini-custom-flash")
    flash = InsightClient.from_env(load_config().insight.model, offline=True)
    assert flash.model == "gemini-custom-flash"
    assert overridden.model == "gemma-custom-model"


def test_poll_does_not_block_on_a_slow_transport():
    def slow(_prompt: str) -> str:
        time.sleep(1.5)
        return '{"bullets":["Pace steadier now"]}'

    client = InsightClient(
        api_key="test-key", transport=slow, min_interval=0, timeout=5
    )
    window = CoachWindow(avg_wpm=150, seconds=30)
    started = time.perf_counter()
    results = [client.poll(window, now=float(i)) for i in range(12)]
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    assert client.pending is True
    assert client.status == "thinking"
    assert all(item is None for item in results)
    client.close()


def test_finished_gemma_result_is_shown_and_errors_are_counted_once(caplog):
    caplog.set_level(logging.WARNING, logger="wingman.insight")
    window = CoachWindow(posture_ratio=0.6, avg_db=36, seconds=40)
    good = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: '{"bullets":["Posture slipping often"]}',
        min_interval=2,
        timeout=2,
    )
    assert good.poll(window, now=0) is None
    assert good.status == "thinking"
    _wait_until(lambda: not good.pending)
    shown = good.poll(window, now=0.2)
    assert shown == ["Posture slipping often"]
    assert good.status == "flash"
    assert good.succeeded == 1
    assert good.failed == 0
    assert good.used_fallback is False
    assert good.pending is False
    assert good.average_latency is not None and good.average_latency >= 0

    def raise_404(_prompt: str) -> str:
        error = RuntimeError("NOT_FOUND")
        error.code = 404
        raise error

    bad = InsightClient(
        api_key="test-key", transport=raise_404, min_interval=1, timeout=2
    )
    bad.poll(window, now=0)
    _wait_until(lambda: bad.failed >= 1 and not bad.pending)
    assert bad.poll(window, now=0.1) is None
    assert bad.status == "offline"
    assert bad.used_fallback is True
    bad.poll(window, now=5)
    _wait_until(lambda: bad.failed >= 2 and not bad.pending)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "wingman.insight"
    ]
    assert warnings == [f"{DEFAULT_INSIGHT_MODEL} call failed: 404 NOT_FOUND"]
    assert bad.succeeded == 0

    messy = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: "not json",
        min_interval=0,
        timeout=2,
    )
    messy.poll(window, now=0)
    _wait_until(lambda: messy.failed >= 1 and not messy.pending)
    assert messy.failed == 1
    assert any("unparseable" in record.getMessage() for record in caplog.records)


def test_a_timed_out_call_is_a_failure_and_poll_does_not_wait_for_it():
    def slow(_prompt: str) -> str:
        time.sleep(0.6)
        return _ENCOURAGE

    client = InsightClient(
        api_key="test-key", transport=slow, min_interval=5, timeout=0.15
    )
    started = time.perf_counter()
    cue = client.poll(CoachWindow(posture_ratio=1, seconds=20), now=0)
    assert cue is None
    assert client.status == "thinking"
    assert time.perf_counter() - started < 0.15
    _wait_until(lambda: client.failed >= 1, timeout=1)
    assert time.perf_counter() - started < 0.45
    _wait_until(lambda: not client.pending, timeout=2)
    assert client.failed == 1
    assert client.succeeded == 0
    client.close()


def test_empty_insight_json_does_not_invent_a_trend():
    client = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: '{"bullets":[]}',
    )
    assert client.insights(CoachWindow(seconds=20)) is None


def test_posture_yields_so_volume_is_also_shown():
    cue_filter = CueFilter(cooldown_seconds=5.0, global_gap_seconds=1.0)
    metrics = CoachMetrics(posture_alert=True, speak_up=True)
    pending = active_cues(metrics)
    assert [cue["cue_type"] for cue in pending] == ["POSTURE", "VOLUME"]
    shown = []
    now = 0.0
    for _ in range(6):
        cue = cue_filter.consider_many(pending, now)
        if cue_filter.fired_at == now:
            shown.append(cue["cue_type"])
        now += 3.1
    assert "POSTURE" in shown
    assert "VOLUME" in shown
    assert shown[0] == "POSTURE"


def test_retryable_gemma_errors_back_off(caplog):
    caplog.set_level(logging.INFO, logger="wingman.insight")

    def raise_504(_prompt: str) -> str:
        error = RuntimeError("DEADLINE_EXCEEDED")
        error.code = 504
        raise error

    client = InsightClient(
        api_key="test-key", transport=raise_504, min_interval=10, timeout=2
    )
    window = CoachWindow(avg_db=33, seconds=30)
    client.poll(window, now=0)
    _wait_until(lambda: client.failed >= 1 and not client.pending)
    client.poll(window, now=10)
    time.sleep(0.05)
    assert client.failed == 1
    assert client.pending is False
    client.poll(window, now=30)
    _wait_until(lambda: client.failed >= 2 and not client.pending)
    assert client.average_latency is not None
    assert http_timeout_ms(8) == 10_000
    assert http_timeout_ms(45) == 45_000


def test_session_report_uses_gemma_or_the_local_summary(tmp_path):
    window = CoachWindow(
        cue_counts={"POSTURE": 4, "VOLUME": 2},
        avg_db=36,
        avg_wpm=150,
        gaze_away_ratio=0.1,
        posture_ratio=0.5,
        seconds=60,
    )
    offline = GemmaClient(api_key=None)
    source, strengths, improve = offline.report(window)
    assert source == "offline"
    assert len(strengths) == 3 and len(improve) == 3
    text = render_report_markdown(window, strengths, improve, source)
    path = save_report(text, tmp_path)
    saved = path.read_text(encoding="utf-8")
    assert saved.startswith("# Wingman session")
    assert "Strengths" in saved and "Work on" in saved
    assert "36 dB" in saved

    payload = (
        '{"strengths":["Clear voice throughout.","Steady eye contact today.",'
        '"Calm opening pace."],"improve":["Sit taller after the start.",'
        '"Pause between the sections.","Lift the quiet middle third."]}'
    )
    live = GemmaClient(
        api_key="test-key", transport=lambda _prompt: payload, report_timeout=2
    )
    source, strengths, improve = live.report(window)
    assert source == "gemma"
    assert strengths[0].startswith("Clear voice")
    assert len(improve) == 3
    local_strengths, _local_improve = deterministic_report(window)
    assert local_strengths


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


def test_flash_reuses_one_client_and_requests_minimal_json():
    class Response:
        text = '{"bullets":["Pace steadier now"]}'

    class Models:
        def __init__(self) -> None:
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return Response()

    class Sdk:
        def __init__(self) -> None:
            self.models = Models()

    insight = InsightClient(api_key="test-key", max_output_tokens=256)
    sdk = Sdk()
    insight._sdk = sdk
    assert insight._call_google("first") == Response.text
    assert insight._call_google("second") == Response.text
    assert insight._sdk is sdk
    assert len(sdk.models.calls) == 2
    config = sdk.models.calls[0]["config"]
    assert config.max_output_tokens == 256
    assert config.response_mime_type == "application/json"
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    built = flash_request_config(256, "minimal")
    assert built.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL

    gemma = GemmaClient(api_key="test-key", max_output_tokens=2048)
    gemma_sdk = Sdk()
    gemma._sdk = gemma_sdk

    class GemmaResponse:
        text = "report"

    gemma_sdk.models.generate_content = lambda **kwargs: (
        gemma_sdk.models.calls.append(kwargs) or GemmaResponse()
    )
    assert gemma.generate("slow question") == "report"
    assert gemma.generate("another") == "report"
    assert gemma._sdk is gemma_sdk
    gemma_config = gemma_sdk.models.calls[0]["config"]
    assert gemma_config.max_output_tokens == 2048
    assert gemma_config.response_mime_type is None
    assert gemma_config.thinking_config is None


def test_a_slow_gemma_call_does_not_delay_the_flash_loop():
    started = threading.Event()

    def slow_gemma(_prompt: str) -> str:
        started.set()
        time.sleep(0.8)
        return (
            '{"strengths":["Clear voice throughout.","Steady eye contact today.",'
            '"Calm opening pace."],"improve":["Sit taller after the start.",'
            '"Pause between the sections.","Lift the quiet middle third."]}'
        )

    gemma = GemmaClient(api_key="test-key", transport=slow_gemma, report_timeout=2)
    insight = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: '{"bullets":["Pace steadier now"]}',
        min_interval=0,
        timeout=2,
    )
    worker = threading.Thread(target=lambda: gemma.report(CoachWindow(seconds=30)))
    worker.start()
    assert started.wait(1)
    began = time.perf_counter()
    insight.poll(CoachWindow(avg_wpm=150, seconds=20), now=0)
    _wait_until(lambda: insight.bullets == ["Pace steadier now"])
    assert time.perf_counter() - began < 0.4
    assert insight.succeeded == 1
    worker.join(timeout=2)
    assert gemma.succeeded == 1
    insight.close()
    gemma.close()
