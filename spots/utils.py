from numba import jit

# Credits to https://stackoverflow.com/questions/47004506/check-if-a-numpy-array-is-sorted/47004533#47004533
@jit
def is_sorted(array, ascending=True):
    for i in range(array.size - 1):
        if ascending:
            if array[i+1] < array[i]:
                return False
        else:
            if array[i+1] > array[i]:
                return False
    return True
