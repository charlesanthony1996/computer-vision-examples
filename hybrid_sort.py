# combining linear and binary sort
# just an example

def hybrid_sort(arr):
    n = len(arr)

    # use linear sort for small arrays
    if n <= 10:
        arr.sort()
        return arr

    # use binary sort for larger arrays
    else:
        arr.sort()
        return binary_sort(arr)


def binary_sort(arr):
    n = len(arr)

    for i in range(1,n):
        key_item = arr[i]
        j = i - 1
        while j >= 0 and key_item < arr[j]:
            arr[j+ 1] = arr[j]
            j -= 1
        arr[j+ 1] = key_item
    return arr


# usage here
import random

# create a random array within 100 elements
arr = [random.randint(0, 100) for i in range(100)]

# sort the array using the hybrid sort function
sorted_arr = hybrid_sort(arr)

# print out the sorted array
print(sorted_arr)