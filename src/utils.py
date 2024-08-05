import Levenshtein


def overlap_coefficient(set1, set2):
    intersection = set1.intersection(set2)
    return len(intersection) / min(len(set1), len(set2))


def levenshtein_distance(string1, string2):
    return Levenshtein.distance(string1, string2) / max(len(string1), len(string2))
