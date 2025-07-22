def remove_key(d, k):
    return {i: d[i] for i in d if i not in k}
