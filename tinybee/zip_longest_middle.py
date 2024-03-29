"""Modified zip_longest: fillvalue in the middle.

zip_longest(iter1 [,iter2 [...]], [fillvalue=None]) --> zip_longest object
"""
# pylint: disable=broad-except, duplicate-code


def zip_longest_middle(list1, list2, fillvalue=None):
    """Zip longest but spread in the middle."""
    len1 = len(list1)
    len2 = len(list2)

    if len1 == len2:
        out1 = zip(list1, list2)
    elif len2 > len1:
        tmp = [fillvalue] * (len2 - len1)
        _ = (len1 + 1) // 2
        out1 = list1[: (len1 + 1) // 2] + tmp + list1[_:]
        out1 = zip(out1, list2)
    else:
        tmp = [fillvalue] * (len1 - len2)
        _ = (len2 + 1) // 2
        out1 = list2[: (len2 + 1) // 2] + tmp + list2[_:]
        out1 = zip(list1, out1)

    out = []
    for elm in out1:
        # out += list(elm)
        # out += elm  # list of numbers
        out.append(elm)  # list of tuples

    return out
