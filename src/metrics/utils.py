# Based on seminar materials

# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if not len(target_text):
        return 1.0

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text, sep=' ') -> float:
    if isinstance(target_text, str):
        target_text = target_text.split(sep)
    if isinstance(predicted_text, str):
        predicted_text = predicted_text.split(sep)

    if not len(target_text):
        return 1.0

    return editdistance.eval(target_text, predicted_text) / len(target_text)
