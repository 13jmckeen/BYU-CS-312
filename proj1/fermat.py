import random
import math


def prime_test(N, k):
    return fermat(N, k), miller_rabin(N, k)


# This function uses y/2 iterations, it uses 2 mods each time & 1 multiplication.
# Multiplication is O(n^2),
# Regular MOD is O(n), but since the return is a multiplication MOD that is O(n^2)
# We are left with n * (n^2 + n + n + n^2) which is a BIG O on n^3
# RUNTIME: O(n^3)
def mod_exp(x, y, N):


    # Replica of algorithm from the book
    if y == 0:
        return 1
    z = mod_exp(x, math.floor(y / 2), N) # n times called
    zsq = z * z # O(n^2)
    if y % 2 == 0: # O(n)
        return zsq % N # O(n)
    else:
        return (x * zsq) % N # O(n^2)

# Probability for fermat is 1 - 1/2^k.
# This was calculated as each time you randomly pick a number, it has a 1/2 possibility of
# coming back true. If you run this multiple times, each time you get it back true,
# it increases the probability it is true overall by 1/2. Num of tests? 1 = 50% | 2 = 75% | 3 = 87.5%...etc

#THIS method runs in O(n^2) time as the power function is O(n^2) and num is O(1) constant time. which is
# O(n^2)
def fprobability(k):
    temp = 1 / pow(2, k) # O(n^2)
    num = 1 - temp # O(1)
    return num


# Probability is 1 / 100000^k per attempt.
# Finding the probability for each miller_rabin probability was found in the book,
# that if the exponent is even, and you divide it in 2, and then test THAT value under mod_exp,
# The probability of it also being prime is 1 / 100000. So each time you run it you multiply these odds

# This runs in O(n^2) as the power function is O(n^2) and the subtraction is O(1) constant time
# O(n^2)
def mprobability(k):
    temp = 1 / pow(100000, k)
    num = 1 - temp
    return num


    # Each loop is constant K iterations,
    # each mod_exp uses (log(N))^3 since the mod was n^3, and N is the biggest input we will get.
    # The number of bits in N is log N.
    # Total time is K * ((log(N^3)) + log(N)) or simplified is O(log(n^3))
def fermat(N, k):
    for i in range(k):
        num = random.randint(1, N - 1)
        if mod_exp(num, N - 1, N) != 1: # O(n^3)
            return 'composite'
    return 'prime'


    # This has log(N) iterations since exponent gets cut in half each time it's called
    # This also is called K number of times
    # It also has a mod outside the log so it is O(n)
    # This calls mod_exp which is n^3
    # This algorithm takes K n log(n^3), or since K is a constant: nlog(n^3).
def miller_rabin(N, k):
    for i in range(k):
        a = random.randint(1, N - 1) # O(1)
        n_1 = N - 1 # O(1)
        n = 1 # O(1)
        if n_1 % 2 == 1: # O(n)
            return 'composite'
        while n == 1 and n_1 % 2 == 0: # log(n)
            n = mod_exp(a, n_1, N) # O(n^3)
            n_1 = n_1 / 2
        if n == 1 or n == -1 or n == N - 1:
            return 'prime'
        return 'composite'
