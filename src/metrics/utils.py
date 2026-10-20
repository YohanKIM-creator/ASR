def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != pred)))
        previous = current
    return previous[-1]


def calc_cer(target_text, predicted_text):
    return edit_distance(target_text, predicted_text) / max(1, len(target_text))


def calc_wer(target_text, predicted_text):
    target, predicted = target_text.split(), predicted_text.split()
    return edit_distance(target, predicted) / max(1, len(target))


class CorpusMetrics:
    """Aggregate edit counts, not the average of utterance percentages."""

    def __init__(self):
        self.word_errors = self.words = self.char_errors = self.chars = self.utterances = 0

    def update(self, reference, prediction):
        self.word_errors += edit_distance(reference.split(), prediction.split())
        self.words += len(reference.split())
        self.char_errors += edit_distance(reference, prediction)
        self.chars += len(reference)
        self.utterances += 1

    def result(self):
        return {
            "wer": self.word_errors / max(1, self.words),
            "cer": self.char_errors / max(1, self.chars),
            "word_errors": self.word_errors,
            "reference_words": self.words,
            "character_errors": self.char_errors,
            "reference_characters": self.chars,
            "utterances": self.utterances,
        }
