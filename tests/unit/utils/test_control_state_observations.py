"""Explicit XML state must survive compact observations without guessing from labels."""

from copy import deepcopy

import pytest

from artemis.utils.visualization import format_minimal_list_with_elements


@pytest.mark.parametrize("widget", ["Switch", "CheckBox"])
@pytest.mark.parametrize("checked", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_labeled_controls_preserve_independent_checked_and_enabled(widget, checked, enabled):
    node = {
        "class": f"android.widget.{widget}",
        "content-desc": "Dark theme",
        "resource-id": "com.android.settings:id/switchWidget",
        "checkable": "true",
        "checked": str(checked).lower(),
        "enabled": str(enabled).lower(),
        "bounds": "[901,1661][1038,1787]",
    }
    original = deepcopy(node)

    text, elements, labels = format_minimal_list_with_elements([node], 1080, 2400)

    assert text == (
        "[1] Text: 'Dark theme' | Bounds: [834,692][961,744]"
        f" | checked: {str(checked).lower()} | enabled: {str(enabled).lower()}"
    )
    assert elements == [
        {
            "index": 1,
            "center": [969, 1724],
            "text": "Dark theme",
            "bounds": [901, 1661, 1038, 1787],
            "class": node["class"],
            "resource_id": node["resource-id"],
            "is_ocr": False,
            "checked": checked,
            "enabled": enabled,
        }
    ]
    assert labels == ["1"]
    assert node == original


@pytest.mark.parametrize("attribute", ["checked", "enabled"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), (" TRUE ", True), (" False ", False)],
)
def test_control_state_accepts_explicit_booleans(attribute, raw, expected):
    node = {"text": "Control", "bounds": "[0,0][100,100]", attribute: raw}
    text, elements, _ = format_minimal_list_with_elements([node], 1000, 1000)
    assert f" | {attribute}: {str(expected).lower()}" in text
    assert elements[0][attribute] is expected


@pytest.mark.parametrize("attribute", ["checked", "enabled"])
@pytest.mark.parametrize("raw", [None, "", "unknown", "0", "1", 0, 1, [], {}])
def test_invalid_control_state_remains_unknown(attribute, raw):
    node = {"text": "Control", "bounds": "[0,0][100,100]", attribute: raw}
    text, elements, _ = format_minimal_list_with_elements([node], 1000, 1000)
    assert text == "[1] Text: 'Control' | Bounds: [0,0][100,100]"
    assert attribute not in elements[0]


def test_missing_control_state_is_not_inferred_from_label_or_class():
    node = {
        "text": "Wi-Fi on",
        "class": "android.widget.Switch",
        "checkable": "true",
        "bounds": "[0,0][100,100]",
    }
    text, elements, _ = format_minimal_list_with_elements([node], 1000, 1000)
    assert text == "[1] Text: 'Wi-Fi on' | Bounds: [0,0][100,100]"
    assert "checked" not in elements[0]
    assert "enabled" not in elements[0]


def test_hint_and_error_keep_explicit_disabled_state():
    node = {
        "hint": "Email",
        "error": "Required",
        "enabled": "false",
        "bounds": "[0,0][100,100]",
    }
    text, elements, _ = format_minimal_list_with_elements([node], 1000, 1000)
    assert text == (
        "[1] Hint: 'Email' | Bounds: [0,0][100,100] | Error: 'Required' | enabled: false"
    )
    assert elements[0]["enabled"] is False
    assert elements[0]["is_hint"] is True
    assert elements[0]["error"] == "Required"


def test_ocr_fragments_do_not_inherit_parent_control_state_or_change_indices():
    nodes = [
        {
            "text": "Parent",
            "bounds": "[0,0][200,200]",
            "checked": "true",
            "enabled": "false",
            "ocr_elements": [{"text": "Read from pixels", "bounds": "[10,10][90,90]"}],
        },
        {"text": "Next", "bounds": "[300,300][400,400]", "enabled": "true"},
    ]
    original = deepcopy(nodes)
    text, elements, labels = format_minimal_list_with_elements(nodes, 1000, 1000)
    assert text.splitlines() == [
        "[1] OCR Text: 'Read from pixels' | Bounds: [10,10][90,90]",
        "[2] Text: 'Next' | Bounds: [300,300][400,400] | enabled: true",
    ]
    assert labels == ["1", "2"]
    assert [element["index"] for element in elements] == [1, 2]
    assert [element["center"] for element in elements] == [[50, 50], [350, 350]]
    assert elements[0]["is_ocr"] is True
    assert "checked" not in elements[0]
    assert "enabled" not in elements[0]
    assert elements[1]["is_ocr"] is False
    assert nodes == original
