"""Detect emotions using the Watson NLP service."""

import json
import requests


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""
    url = (
        "https://sn-watson-emotion.labs.skills.network"
        "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        url, headers=headers, json=payload, timeout=30
    )
    data = json.loads(response.text)
    emotions = data["emotionPredictions"][0]["emotion"]
    result = {
        name: emotions[name]
        for name in ("anger", "disgust", "fear", "joy", "sadness")
    }
    result["dominant_emotion"] = max(result, key=result.get)
    return result
