import numpy

def next_power_of_2(n):
    # Check if n is already a power of 2
    if n and not (n & (n - 1)):
        return n
    # Find the most significant bit position and shift left
    p = 1
    while p < n:
        p <<= 1
    return p