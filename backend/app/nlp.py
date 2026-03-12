def basic_text_features(text):

    words = text.split()

    features = {
        "word_count": len(words),
        "char_count": len(text),
        "question_mark": "?" in text
    }

    return features