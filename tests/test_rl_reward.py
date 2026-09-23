from scripts.rl import extract_sid_parts


def test_sid_reward_parser_supports_angle_tokens() -> None:
    assert extract_sid_parts("<a_14><b_31><c_28><d_1>") == [
        "<a_14>", "<b_31>", "<c_28>", "<d_1>"
    ]


def test_sid_reward_parser_keeps_legacy_square_tokens() -> None:
    assert extract_sid_parts("[A12][B99][C6]") == ["[A12]", "[B99]", "[C6]"]
