import pytest

from core.src.output_guard import StreamScrubber, scrub

# The answer FbrTaxAgent actually returned when its session had no Skill tool
LEAKED = (
    '\n<invoke name="Skill">\n<parameter name="skill">fbr-iris-filing</parameter>\n'
    '<parameter name="args">Read-only status check for tax year 2026.</parameter>\n</invoke>\n'
)
NS = "antml"  # namespaced variant, built here so the file itself holds no live markup


def test_answer_made_only_of_a_tool_call_becomes_empty():
    result = scrub(LEAKED)
    assert result.text == ""
    assert result.found == ("invoke",)  # removed whole, parameters included


def test_text_around_the_markup_is_kept():
    result = scrub(f"Checking your filing status…{LEAKED}Your 2025 return is submitted.")
    assert result.text == "Checking your filing status…\n\nYour 2025 return is submitted."


def test_namespaced_function_calls_block_is_removed_whole():
    text = (f"Sure.<{NS}:function_calls><{NS}:invoke name=\"x\"><{NS}:parameter name=\"a\">1"
            f"</{NS}:parameter></{NS}:invoke></{NS}:function_calls> Done.")
    result = scrub(text)
    assert result.text == "Sure. Done."
    assert result.found == ("function_calls",)


@pytest.mark.parametrize("tag", ["thinking", "system-reminder", "function_results", "tool_result"])
def test_internal_blocks_are_removed(tag):
    assert scrub(f"<{tag}>secret</{tag}>Hello").text == "Hello"


def test_truncated_tool_call_is_cut_to_the_end():
    result = scrub('Here you go.\n<invoke name="Skill">\n<parameter name="skill">fbr')
    assert result.text == "Here you go."
    assert "invoke" in result.found


def test_stray_closing_tags_are_removed():
    assert scrub("Done.</invoke></parameter>").text == "Done."


@pytest.mark.parametrize("text", [
    "Apartments under <1000 PLN: 3 found.",
    "Use <b>bold</b> or a <a href='x'>link</a>.",
    "Invoice <invoice-42> paid.",
])
def test_ordinary_text_is_untouched(text):
    result = scrub(text)
    assert result.text == text
    assert result.found == ()


def _stream(chunks):
    scrubber = StreamScrubber()
    out = [scrubber.feed(c) for c in chunks] + [scrubber.flush()]
    return out, scrubber


def test_stream_never_emits_markup_split_across_tokens():
    text = f"Checking…{LEAKED}Submitted."
    out, scrubber = _stream([text[i:i + 3] for i in range(0, len(text), 3)])
    assert all("<" not in piece and "parameter" not in piece for piece in out)
    assert "".join(out).replace("\n", "") == "Checking…Submitted."
    assert scrubber.found == {"invoke"}


def test_stream_releases_ordinary_text_as_it_goes():
    out, _ = _stream(["Hello ", "there, ", "under <1000", " PLN."])
    assert out[:2] == ["Hello ", "there, "]
    assert "".join(out) == "Hello there, under <1000 PLN."


def test_stream_drops_an_unfinished_tool_call_at_the_end():
    out, scrubber = _stream(["Done.", '<invoke name="Skill">', "<parameter>x"])
    assert "".join(out) == "Done."
    assert "invoke" in scrubber.found
