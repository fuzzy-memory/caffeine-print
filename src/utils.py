import json
import os
from typing import Dict, List

from singletons import SelectedChannel, TestingMode


def generate_prompt(article_text: str) -> List[Dict[str, str]]:
    selected_channel = SelectedChannel().get_channel().value
    base_prompt = (
        json.load(open("assets/control.json", "r"))
        .get(selected_channel)
        .get("base_prompt")
    )
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant to a news aggregator sorting through many news articles. Rate the "
            "given articles on the provided metrics as truthfully as possible.",
        },
        {
            "role": "user",
            "content": base_prompt + "\n" + article_text,
        },
    ]


def generate_asset_path(filename: str):
    test_mode = TestingMode().get_testing_mode()
    return os.path.join("assets", "test" if test_mode else "", filename)
