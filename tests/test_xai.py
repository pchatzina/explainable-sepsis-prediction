"""
Run: pytest tests/test_xai.py -v
"""

import pytest
from src.explainability.clinical_inference_xai import get_token_string_resilient


@pytest.fixture
def mock_vocab():
    return [
        {"type": "code", "code_string": "LOINC/LP432695-7"},
        {"type": "code", "code_string": "MIMIC_IV_LABITEM/52023"},
        {"type": "text", "property": "Note", "text_string": "Patient is stable"},
        {"val_start": 36.5, "val_end": 37.0, "property": "Temperature"},
        {"type": "unknown", "some_key": "some_value"},  # Fallback catch
    ]


def test_get_token_string_resilient(mock_vocab):
    athena_map = {"LOINC/LP432695-7": "Testing Concept"}
    mimic_map = {"52023": "Lactate"}

    # 1. Test Athena Resolution
    tok_0 = get_token_string_resilient(mock_vocab, 0, athena_map, mimic_map)
    assert "Testing Concept" in tok_0
    assert "LOINC/LP432695-7" in tok_0

    # 2. Test MIMIC Resolution
    tok_1 = get_token_string_resilient(mock_vocab, 1, athena_map, mimic_map)
    assert "Lactate" in tok_1
    assert "MIMIC_IV_LABITEM/52023" in tok_1

    # 3. Test Text formatting
    tok_2 = get_token_string_resilient(mock_vocab, 2)
    assert tok_2 == "Note: Patient is stable"

    # 4. Test Numeric Range formatting
    tok_3 = get_token_string_resilient(mock_vocab, 3)
    assert tok_3 == "Temperature: [36.50, 37.00)"

    # 5. Test Out of Bounds
    tok_out = get_token_string_resilient(mock_vocab, 999)
    assert "Out_of_Bounds_ID_999" == tok_out

    # 6. Test unknown type fallback
    tok_4 = get_token_string_resilient(mock_vocab, 4)
    assert "some_key" in tok_4  # falls through to the str({...}) branch
