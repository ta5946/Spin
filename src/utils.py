def overlap_coefficient(set1, set2):
    intersection = set1.intersection(set2)
    return len(intersection) / min(len(set1), len(set2))
