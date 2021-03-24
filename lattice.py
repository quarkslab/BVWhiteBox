#!/usr/bin/env python3

"""
   Copyright 2021 Lucas Barthélémy and Quarkslab

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import random as rand

# main class to manipulate vectors in Z/(q, X^n + 1)
# and transformations NTT/CRT


class NTRU_vector():

    white = None

    def __init__(self, degree, modulus, ntt):
        self.vector = np.zeros(degree, dtype=np.int64)
        self.degree = degree
        self.modulus = modulus
        self.ntt = ntt

    def __add__(self, other):
        res = NTRU_vector(self.degree, self.modulus,
                          self.ntt)
        res.vector = np.array([self.vector[i] + other.vector[i]
                               % self.modulus for i in range(self.degree)])
        return res

    def __sub__(self, other):
        res = NTRU_vector(self.degree, self.modulus,
                          self.ntt)
        res.vector = np.array([self.vector[i] - other.vector[i]
                               % self.modulus for i in range(self.degree)])
        return res

    def __mul__(self, other):
        res = NTRU_vector(self.degree, self.modulus,
                          self.ntt)
        if self.ntt:
            for i in range(self.degree):
                x = int(self.vector[i])
                y = int(other.vector[i])
                z = x*y
                res.vector[i] = z
        else:
            for i in range(self.degree):
                for j in range(self.degree):
                    d = i+j
                    if d < self.degree:
                        res.vector[d] =\
                            (res.vector[d] +
                             self.vector[i] * other.vector[j]
                             ) % self.modulus
                    else:
                        d = d % self.degree
                        res.vector[d] =\
                            (res.vector[d] -
                             self.vector[i] * other.vector[j]
                             ) % self.modulus
        return res

    def __neg__(self):
        self.vector = np.array(
            [-self.vector[i] % self.modulus for i in range(self.degree)])
        return self

    # NTT transformation, need nth root of unity
    # Cooley-Tukey algorithm O(n log n)
    #  "negative wrapped convolution" to avoid shift to 2n dimensions

    def goto_ntt(self, root):
        if self.ntt:
            print("This vector is already ntt")
        else:
            n = self.degree
            self.ntt = True
            self.degree = n
            levels = n.bit_length() - 1
            powtable = []
            temp = 1
            for i in range(n):
                self.vector[i] = self.vector[i] * temp % self.modulus
                if not i % 2:
                    powtable.append(temp)
                temp = temp * root % self.modulus

            def reverse(x, bits):
                y = 0
                for i in range(bits):
                    y = (y << 1) | (x & 1)
                    x >>= 1
                return y
            for i in range(n):
                j = reverse(i, levels)
                if j > i:
                    self.vector[i], self.vector[j] =\
                        self.vector[j], self.vector[i]

            size = 2
            while size <= n:
                halfsize = size // 2
                tablestep = n // size
                for i in range(0, n, size):
                    k = 0
                    for j in range(i, i + halfsize):
                        l = j + halfsize
                        left = self.vector[j]
                        right = self.vector[l] * powtable[k]
                        self.vector[j] = (left + right)\
                            % self.modulus
                        self.vector[l] = (left - right)\
                            % self.modulus
                        k += tablestep
                size *= 2

    # inverse NTT transform, need root ring inverse,
    # that matches forward transform,
    # also needs dimension ring inverse,
    #  See forward NTT for spec

    def goback_ntt(self, unroot, ninv):
        if not self.ntt:
            print("This vector is not ntt")
        else:
            self.ntt = False
            n = self.degree
            res = NTRU_vector(n, self.modulus, False)
            res.vector = self.vector

            levels = n.bit_length() - 1
            powtable = []
            powtable2 = []
            temp = 1
            for i in range(n):
                if not i % 2:
                    powtable.append(temp)
                powtable2.append(temp)
                temp = temp * unroot % self.modulus

            def reverse(x, bits):
                y = 0
                for i in range(bits):
                    y = (y << 1) | (x & 1)
                    x >>= 1
                return y
            for i in range(n):
                j = reverse(i, levels)
                if j > i:
                    res.vector[i], res.vector[j] =\
                        res.vector[j], res.vector[i]

            size = 2
            while size <= n:
                halfsize = size // 2
                tablestep = n // size
                for i in range(0, n, size):
                    k = 0
                    for j in range(i, i + halfsize):
                        l = j + halfsize
                        left = res.vector[j]
                        right = res.vector[l] * powtable[k]
                        res.vector[j] = (left + right)\
                            % self.modulus
                        res.vector[l] = (left - right)\
                            % self.modulus
                        k += tablestep
                size *= 2

            self.vector = np.array([res.vector[i] * ninv * powtable2[i]
                                    % self.modulus
                                    for i in range(self.degree)])


# basic CRT transform
def goto_crt(x, base, l):
    return [x % base[i] for i in range(l)]


# basic inverse CRT transform
def goback_crt(x_b, base, l):
    x = 0
    B = np.prod(base)
    for i in range(l):
        B_i = B/base[i]
        x += (x_b[i] * B_i * xgcd(B_i, base[i])[1])
    return x % B


# Extended Euclidean alg
def xgcd(b, n):
    x0, x1, y0, y1 = 1, 0, 0, 1
    while n != 0:
        q, b, n = b // n, n, b % n
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


# generate public and secret keys with relaxed problem
# should still be safe (TBD)
def key_gen(degree, q):
    rand.seed()
    sk = NTRU_vector(degree, q, False)
    pka = NTRU_vector(degree, q, False)
    pkb = NTRU_vector(degree, q, False)

    # generate gaussian values for pk/sk
    for i in range(degree):
        sk.vector[i] = int(rand.gauss(0, 1))
        pka.vector[i] = rand.randint(0, q)
        pkb.vector[i] = 2*int(rand.gauss(0, 1))

    pkb = -(pkb + pka*sk)

    return pka, pkb, sk


# takes 512 bits and encrypt
def encrypt(m, pka, pkb, degree, modulus):
    u = NTRU_vector(degree, modulus, False)
    e1 = NTRU_vector(degree, modulus, False)
    e2 = NTRU_vector(degree, modulus, False)
    for i in range(degree):
        u.vector[i] = int(rand.gauss(0, 1))
        e1.vector[i] = 2*int(rand.gauss(0, 1))
        e2.vector[i] = 2*int(rand.gauss(0, 1))
    tmp = NTRU_vector(degree, modulus, False)
    for i in range(degree):
        tmp.vector[i] = m[i]
    a1 = pka*u+e1
    a2 = pkb*u+e2+tmp

    return a1, a2


# regular decryption, can be changed to use NTT
def decrypt(a1, a2, sk, degree, modulus):

    tmp = a2 + a1 * sk

    m = np.zeros(degree, dtype=int)
    for i in range(degree):
        m[i] = tmp.vector[i] % tmp.modulus
        if m[i] > modulus/2:
            m[i] = 1 - (m[i] % 2)
        else:
            m[i] = m[i] % 2
    return m
