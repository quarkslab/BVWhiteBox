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

import numbers
import numpy as np
import json
import random as rand
from lattice import key_gen, goto_crt, xgcd, encrypt

def print_progress(m,n,step):
    if (m % step == 0):
        print("%02.1f%%" % (m * 100.0 / n))


# write first row of lookup tables with rotate/zero encodings


def prepare_first_box_MM3(sk, a1_r, a2_r, a1_ma, a2_ma, root,
                          unroot, ninv, beta, k):

    # compute intermediate encoding values to limit
    # number of products in Montgomery
    rot = a2_r + a1_r * sk
    mask = -(a2_ma + a1_ma * sk)

    # create alternate key tmp_sk and fake key tmp_sz
    tmp_sk = sk * rot
    tmp_sz = sk * mask

    # go to ntt
    rot.goto_ntt(root)
    mask.goto_ntt(root)
    tmp_sk.goto_ntt(root)
    tmp_sz.goto_ntt(root)

    # prepare Montgomery parameters
    M = np.prod(beta)
    N = tmp_sk.modulus
    Ninv_M = goto_crt(xgcd(N, M)[1], beta, k)
    fb = {}
    for dim in range(tmp_sk.degree):
        print_progress(dim, tmp_sk.degree, 64)
        key = "fb_dim_"+str(dim)
        fb[key] = [[0]*32 for i in range(32)]
        # go to crt
        s = tmp_sk.vector[dim]
        s = goto_crt(s, beta, k)
        sz = tmp_sz.vector[dim]
        sz = goto_crt(sz, beta, k)
        r = rot.vector[dim]
        r = goto_crt(r, beta, k)
        m = mask.vector[dim]
        m = goto_crt(m, beta, k)
        # parse all possible input values
        for j in range(32):
            a = goto_crt(j, beta, k)
            for l in range(32):
                b = goto_crt(l, beta, k)
                # compute decoding function
                fb[key][j][l] = int(
                    (int(((a[0] * s[0] + b[0] * r[0] + r[0] * m[0])
                          * (-Ninv_M[0])) % beta[0])) +
                    (int(((a[1] * s[1] + b[1] * r[1] + r[1] * m[1])
                          * (-Ninv_M[1])) % beta[1]) << 5) +
                    (int(((a[2] * s[2] + b[2] * r[2] + r[2] * m[2])
                          * (-Ninv_M[2])) % beta[2]) << 10) +
                    (int(((a[3] * s[3] + b[3] * r[3] + r[3] * m[3])
                          * (-Ninv_M[3])) % beta[3]) << 15) +
                    (int(((a[4] * s[4] + b[4] * r[4] + r[4] * m[4])
                          * (-Ninv_M[4])) % beta[4]) << 20)
                )
    print_progress(tmp_sk.degree, tmp_sk.degree, 64)
    return fb


def prepare_second_box_MM3(sk, a1_r, a2_r, a1_ma, a2_ma, root,
                           unroot, ninv, beta, beta_p, k):

    # compute intermediate encoding values to limit
    # number of products in Montgomery
    rot = a2_r + a1_r * sk
    mask = -(a2_ma + a1_ma * sk)

    # create alternate key tmp_sk and fake key tmp_sz
    tmp_sk = sk * rot
    tmp_sz = sk * mask

    # go to ntt
    rot.goto_ntt(root)
    mask.goto_ntt(root)
    tmp_sk.goto_ntt(root)
    tmp_sz.goto_ntt(root)

    # prepare Montgomery parameters
    # N = tmp_sk.modulus
    M = np.prod(beta)
    M_p = np.prod(beta_p)
    Minv_M_p = goto_crt(xgcd(M, M_p)[1], beta_p, k)
    # N_M_p = goto_crt(N, beta_p, k)
    sb = {}
    for dim in range(tmp_sk.degree):
        key = "sb_dim_"+str(dim)
        sb[key] = [[0]*32 for i in range(32)]
        # go to crt
        s = tmp_sk.vector[dim]
        s = goto_crt(s, beta_p, k)
        sz = tmp_sz.vector[dim]
        sz = goto_crt(sz, beta_p, k)
        r = rot.vector[dim]
        r = goto_crt(r, beta_p, k)
        m = mask.vector[dim]
        m = goto_crt(m, beta_p, k)
        # parse all possible input values
        for j in range(32):
            a = goto_crt(j, beta_p, k)
            for l in range(32):
                b = goto_crt(l, beta_p, k)
                # compute decoding function
                sb[key][j][l] = int(
                    (int(((a[0] * s[0] + b[0] * r[0] + r[0] * m[0])
                          * Minv_M_p[0]) % beta_p[0])) +
                    (int(((a[1] * s[1] + b[1] * r[1] + r[1] * m[1])
                          * Minv_M_p[1]) % beta_p[1]) << 5) +
                    (int(((a[2] * s[2] + b[2] * r[2] + r[2] * m[2])
                          * Minv_M_p[2]) % beta_p[2]) << 10) +
                    (int(((a[3] * s[3] + b[3] * r[3] + r[3] * m[3])
                          * Minv_M_p[3]) % beta_p[3]) << 15) +
                    (int(((a[4] * s[4] + b[4] * r[4] + r[4] * m[4])
                          * Minv_M_p[4]) % beta_p[4]) << 20)
                )
        print_progress(dim, tmp_sk.degree, 64)
    return sb

# write first row of lookup tables with one/zero encodings


def prepare_first_box_MM2(sk, a1_o, a2_o, a1_z, a2_z, root,
                          unroot, ninv, beta, k):

    # compute intermediate encoding values to limit
    # number of products in Montgomery
    one = a2_o + a1_o * sk
    zero = -(a2_z + a1_z * sk)

    # create alternate key tmp_sk and fake key tmp_sz
    tmp_sk = sk * one
    tmp_sz = sk * zero

    # go to ntt
    one.goto_ntt(root)
    zero.goto_ntt(root)
    tmp_sk.goto_ntt(root)
    tmp_sz.goto_ntt(root)

    # prepare Montgomery parameters
    M = np.prod(beta)
    N = tmp_sk.modulus
    Ninv_M = goto_crt(xgcd(N, M)[1], beta, k)
    fb = {}
    for dim in range(tmp_sk.degree):
        print_progress(dim, tmp_sk.degree, 64)
        key = "fb_dim_"+str(dim)
        fb[key] = [[0]*32 for i in range(32)]
        # go to crt
        s = tmp_sk.vector[dim]
        s = goto_crt(s, beta, k)
        sz = tmp_sz.vector[dim]
        sz = goto_crt(sz, beta, k)
        o = one.vector[dim]
        o = goto_crt(o, beta, k)
        z = zero.vector[dim]
        z = goto_crt(z, beta, k)
        # parse all possible input values
        for j in range(32):
            a = goto_crt(j, beta, k)
            for l in range(32):
                b = goto_crt(l, beta, k)
                # compute decoding function
                fb[key][j][l] = int(
                    (int(((a[0] * s[0] + b[0] * o[0]
                           + a[0] * sz[0] + b[0] * z[0])
                          * (-Ninv_M[0])) % beta[0])) +
                    (int(((a[1] * s[1] + b[1] * o[1]
                           + a[1] * sz[1] + b[1] * z[1])
                          * (-Ninv_M[1])) % beta[1]) << 5) +
                    (int(((a[2] * s[2] + b[2] * o[2]
                           + a[2] * sz[2] + b[2] * z[2])
                          * (-Ninv_M[2])) % beta[2]) << 10) +
                    (int(((a[3] * s[3] + b[3] * o[3]
                           + a[3] * sz[3] + b[3] * z[3])
                          * (-Ninv_M[3])) % beta[3]) << 15) +
                    (int(((a[4] * s[4] + b[4] * o[4]
                           + a[4] * sz[4] + b[4] * z[4])
                          * (-Ninv_M[4])) % beta[4]) << 20)
                )
    print_progress(tmp_sk.degree, tmp_sk.degree, 64)
    return fb

# write second row of lookup tables with one/zero encodings


def prepare_second_box_MM2(sk, a1_o, a2_o, a1_z, a2_z, root,
                           unroot, ninv, beta, beta_p, k):

    # compute intermediate encoding values to limit
    # number of products in Montgomery
    one = a2_o + a1_o * sk
    zero = -(a2_z + a1_z * sk)

    # create alternate key tmp_sk and fake key tmp_sz
    tmp_sk = sk * one
    tmp_sz = sk * zero

    # go to ntt
    one.goto_ntt(root)
    zero.goto_ntt(root)
    tmp_sk.goto_ntt(root)
    tmp_sz.goto_ntt(root)

    # prepare Montgomery parameters
    # N = tmp_sk.modulus
    M = np.prod(beta)
    M_p = np.prod(beta_p)
    Minv_M_p = goto_crt(xgcd(M, M_p)[1], beta_p, k)
    # N_M_p = goto_crt(N, beta_p, k)
    sb = {}
    for dim in range(tmp_sk.degree):
        print_progress(dim, tmp_sk.degree, 64)
        key = "sb_dim_"+str(dim)
        sb[key] = [[0]*32 for i in range(32)]
        # go to crt
        s = tmp_sk.vector[dim]
        s = goto_crt(s, beta_p, k)
        sz = tmp_sz.vector[dim]
        sz = goto_crt(sz, beta_p, k)
        o = one.vector[dim]
        o = goto_crt(o, beta_p, k)
        z = zero.vector[dim]
        z = goto_crt(z, beta_p, k)
        # parse all possible input values
        for j in range(32):
            a = goto_crt(j, beta_p, k)
            for l in range(32):
                b = goto_crt(l, beta_p, k)
                # compute decoding function
                sb[key][j][l] = int(
                    (int(((a[0] * s[0] + b[0] * o[0]
                           + a[0] * sz[0] + b[0] * z[0])
                          * Minv_M_p[0]) % beta_p[0])) +
                    (int(((a[1] * s[1] + b[1] * o[1]
                           + a[1] * sz[1] + b[1] * z[1])
                          * Minv_M_p[1]) % beta_p[1]) << 5) +
                    (int(((a[2] * s[2] + b[2] * o[2]
                           + a[2] * sz[2] + b[2] * z[2])
                          * Minv_M_p[2]) % beta_p[2]) << 10) +
                    (int(((a[3] * s[3] + b[3] * o[3]
                           + a[3] * sz[3] + b[3] * z[3])
                          * Minv_M_p[3]) % beta_p[3]) << 15) +
                    (int(((a[4] * s[4] + b[4] * o[4]
                           + a[4] * sz[4] + b[4] * z[4])
                          * Minv_M_p[4]) % beta_p[4]) << 20)
                )
    print_progress(tmp_sk.degree, tmp_sk.degree, 64)
    return sb


# write first row of lookup tables with no encodings
def prepare_first_box_MM(sk, root, unroot, ninv, beta, k):

    sk.goto_ntt(root)

    M = np.prod(beta)
    N = sk.modulus
    Ninv_M = goto_crt(xgcd(N, M)[1], beta, k)
    fb = {}
    for dim in range(sk.degree):
        print_progress(dim, tmp_sk.degree, 64)
        key = "fb_dim_"+str(dim)
        fb[key] = [[0]*32 for i in range(32)]
        s = sk.vector[dim]
        s = goto_crt(s, beta, k)
        for j in range(32):
            a = goto_crt(j, beta, k)
            for l in range(32):
                b = goto_crt(l, beta, k)

                fb[key][j][l] = int(
                    (int(((a[0] * s[0] + b[0])
                          * (-Ninv_M[0])) % beta[0])) +
                    (int(((a[1] * s[1] + b[1])
                          * (-Ninv_M[1])) % beta[1]) << 5) +
                    (int(((a[2] * s[2] + b[2])
                          * (-Ninv_M[2])) % beta[2]) << 10) +
                    (int(((a[3] * s[3] + b[3])
                          * (-Ninv_M[3])) % beta[3]) << 15) +
                    (int(((a[4] * s[4] + b[4])
                          * (-Ninv_M[4])) % beta[4]) << 20)
                )
    print_progress(tmp_sk.degree, tmp_sk.degree, 64)
    sk.goback_ntt(unroot, ninv)
    return fb


# write second row of lookup tables with no encodings
def prepare_second_box_MM(sk, root, unroot, ninv, beta, beta_p, k):

    sk.goto_ntt(root)

    # N = sk.modulus
    M = np.prod(beta)
    M_p = np.prod(beta_p)
    Minv_M_p = goto_crt(xgcd(M, M_p)[1], beta_p, k)
    # N_M_p = goto_crt(N, beta_p, k)
    sb = {}
    for dim in range(sk.degree):
        print_progress(dim, tmp_sk.degree, 64)
        key = "sb_dim_"+str(dim)
        sb[key] = [[0]*32 for i in range(32)]
        s = sk.vector[dim]
        s = goto_crt(s, beta_p, k)
        for j in range(32):
            a = goto_crt(j, beta_p, k)
            for l in range(32):
                b = goto_crt(l, beta_p, k)

                sb[key][j][l] = int(
                    (int(((a[0] * s[0] + b[0])
                          * Minv_M_p[0]) % beta_p[0])) +
                    (int(((a[1] * s[1] + b[1])
                          * Minv_M_p[1]) % beta_p[1]) << 5) +
                    (int(((a[2] * s[2] + b[2])
                          * Minv_M_p[2]) % beta_p[2]) << 10) +
                    (int(((a[3] * s[3] + b[3])
                          * Minv_M_p[3]) % beta_p[3]) << 15) +
                    (int(((a[4] * s[4] + b[4])
                          * Minv_M_p[4]) % beta_p[4]) << 20)
                )
    print_progress(tmp_sk.degree, tmp_sk.degree, 64)
    sk.goback_ntt(unroot, ninv)
    return sb


# generating WB and writing data
def write_data(degree, modulus, beta, beta_p, k, chal=0):
    # set WB keys
    pka, pkb, sk = key_gen(degree, modulus)

    # set homomorphic masks if chal 1
    one = np.zeros(degree)
    one[0] = 1
    a1_o, a2_o = encrypt(one, pka, pkb, degree, modulus)
    zero = np.zeros(degree)
    a1_z, a2_z = encrypt(zero, pka, pkb, degree, modulus)

    # set homomorphic masks if chal 2
    rotate = np.zeros(degree)
    rot = rand.randint(0, degree)
    rotate[rot] = 1
    a1_rot, a2_rot = encrypt(rotate, pka, pkb, degree, modulus)

    mask = np.zeros(degree)
    for i in range(degree):
        mask[i] = rand.randint(0, 1)
    a1_ma, a2_ma = encrypt(mask, pka, pkb, degree, modulus)

    # set root for NTT, this need to be static (put in pub data)
    root = find_primitive_root(2*degree, modulus-1, modulus)
    unroot = xgcd(root, modulus)[1]
    ninv = xgcd(degree, modulus)[1]

    # writing private data
    # private data should only be accessed for testing purposes
    # remove otherwise
    d = {'sk': sk.vector.tolist(),
         'a1_o': a1_o.vector.tolist(),
         'a2_o': a2_o.vector.tolist(),
         'a1_z': a1_z.vector.tolist(),
         'a2_z': a2_z.vector.tolist()
         }

    with open('private_data.json', 'w') as f:
        json.dump(d, f)

    # writing public encryption data
    d = {'degree': degree,
         'modulus': modulus,
         'pka': pka.vector.tolist(),
         'pkb': pkb.vector.tolist()
         }
    with open('pub_enc_data.json', 'w') as f:
        json.dump(d, f)

    # writing whiteboxed decryption data
    d = {'root': root,
         'unroot': unroot,
         'ninv': ninv,
         'beta': beta,
         'beta_p': beta_p,
         'k': k,
         'mask': mask.tolist(),
         'rotate': rot,
         'chal': chal
         }
    # f = open('wb_dec_data.json', 'w')
    # json.dump(d, f)
    # writing lookup tables for white montgomery multiplication (need 2 set)
    # if chal, no encodings
    if chal == 0:
        print("prepare_first_box_MM")
        d.update(prepare_first_box_MM(sk, root, unroot, ninv, beta, k))
        print("prepare_second_box_MM")
        d.update(prepare_second_box_MM(
            sk, root, unroot, ninv, beta, beta_p, k))
    elif chal == 1:
        print("prepare_first_box_MM2")
        d.update(prepare_first_box_MM2(
            sk, a1_o, a2_o, a1_z, a2_z,
            root, unroot, ninv, beta, k))
        print("prepare_second_box_MM2")
        d.update(prepare_second_box_MM2(
            sk, a1_o, a2_o, a1_z, a2_z,
            root, unroot, ninv, beta, beta_p, k))
    elif chal == 2:
        print("prepare_first_box_MM3")
        d.update(prepare_first_box_MM3(
            sk, a1_rot, a2_rot, a1_ma, a2_ma,
            root, unroot, ninv, beta, k))
        print("prepare_second_box_MM3")
        d.update(prepare_second_box_MM3(
            sk, a1_rot, a2_rot, a1_ma, a2_ma,
            root, unroot, ninv, beta, beta_p, k))
    with open('wb_dec_data.json', 'w') as f:
        json.dump(d, f)

# >>>>>>>>>>>>>>>>>>> Project Nayuki starts here <<<<<<<<<<<<<<<<<<<<
# Functions below are used for the NTT transform and
# drawn with authorization from Project Nayuki
# Copyright (c) 2021 Project Nayuki
# All rights reserved. Contact Nayuki for licensing.
# https://www.nayuki.io/page/number-theoretic-transform-integer-dft

# Returns an arbitrary generator of the multiplicative group
# of integers modulo mod.
# totient must equal the Euler phi function of mod.
# If mod is prime, an answer must exist.


def find_generator(totient, mod):
    check_int(totient)
    check_int(mod)
    if not (1 <= totient < mod):
        raise ValueError()
    for i in range(1, mod):
        if is_generator(i, totient, mod):
            return i
    raise ValueError("No generator exists")


# Returns an arbitrary primitive degree-th root of unity modulo mod.
# totient must be a multiple of degree. If mod is prime, an answer must exist.
def find_primitive_root(degree, totient, mod):
    check_int(degree)
    check_int(totient)
    check_int(mod)
    if not (1 <= degree <= totient < mod):
        raise ValueError()
    if totient % degree != 0:
        raise ValueError()
    gen = find_generator(totient, mod)
    root = pow(gen, totient // degree, mod)
    assert 0 <= root < mod
    return root


# Tests whether val generates the multiplicative group of integers modulo mod.
# totient must equal the Euler phi function of mod.
# In other words, the set of numbers
# {val^0 % mod, val^1 % mod, ..., val^(totient-1) % mod} is equal to the set of
# all numbers in the range [0, mod) that are coprime to mod.
# If mod is prime, then totient = mod - 1, and powers of a generator produces
# all integers in the range [1, mod).
def is_generator(val, totient, mod):
    check_int(val)
    check_int(totient)
    check_int(mod)
    if not (0 <= val < mod):
        raise ValueError()
    if not (1 <= totient < mod):
        raise ValueError()
    pf = unique_prime_factors(totient)
    return pow(val, totient, mod) == 1 and \
        all((pow(val, totient // p, mod) != 1) for p in pf)


# Returns silently if the given value is an integer,
# otherwise raises a TypeError.
def check_int(n):
    if not isinstance(n, numbers.Integral):
        raise TypeError()


# Returns a list of unique prime factors of the given integer in
# ascending order. For example, unique_prime_factors(60) = [2, 3, 5].
def unique_prime_factors(n):
    check_int(n)
    if n < 1:
        raise ValueError()
    result = []
    i = 2
    end = sqrt(n)
    while i <= end:
        if n % i == 0:
            n //= i
            result.append(i)
            while n % i == 0:
                n //= i
            end = sqrt(n)
        i += 1
    if n > 1:
        result.append(n)
    return result


# Returns floor(sqrt(n)) for the given integer n >= 0.
def sqrt(n):
    check_int(n)
    if n < 0:
        raise ValueError()
    i = 1
    while i * i <= n:
        i *= 2
    result = 0
    while i > 0:
        if (result + i)**2 <= n:
            result += i
        i //= 2
    return result
# >>>>>>>>>>>>>>>>>>> Project Nayuki stops here <<<<<<<<<<<<<<<<<<<<


# create white box lookup tables,
# chal changes encodings
if __name__ == "__main__":
    # dimension of lattice
    degree = 512
    # prime modulus of the form (2*degree)*i +1
    # so that degree divides euler phi function
    modulus = 1231873
    # set bases for white montgomery multiplication (static)
    k = 5
    # beta = 3094416
    beta = [13, 16, 19, 27, 29]
    # beta_p = 3333275
    beta_p = [11, 17, 23, 25, 31]
    # chal level
    chal = 2
    write_data(degree, modulus, beta, beta_p, k, chal)
