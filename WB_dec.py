#!/usr/bin/env python3

import numpy as np
import json
import random as rand
from lattice import NTRU_vector, goto_crt, goback_crt, xgcd, encrypt


class WB_vector(NTRU_vector):

    def __mul__(self, other):
        res = NTRU_vector(self.degree, self.modulus,
                          self.ntt)
        if self.ntt:
            self.my_mult(other, res)
        else:
            print("WB vector must be turned in ntt form")
        return res

    def my_mult(self, other, res):
        for i in range(self.degree):
            x = int(self.vector[i])
            y = int(other.vector[i])
            z = WB_vector.mont_mult(i, x, y, self.modulus)
            res.vector[i] = z

    # white Montgomery mult,
    # for example, chal 1 compute (a*s + b)*one + (a*s + b)*zero,
    # (s, one, zero) are secrets
    # watchout, no sub allowed, need preprocess b <- -b
    @classmethod
    def mont_mult(cls, dim, a, b, N):
        # set up for Montgomery
        B = cls.white['beta']
        B_p = cls.white['beta_p']
        k = cls.white['k']
        # a and b in base M
        a_M = goto_crt(a, B, k)
        b_M = goto_crt(b, B, k)
        # a and b in base M'
        a_M_p = goto_crt(a, B_p, k)
        b_M_p = goto_crt(b, B_p, k)
        # base M
        M = np.prod(B)
        # base M'
        M_p = np.prod(B_p)
        # inv(N) mod M
        # Ninv_M = goto_crt(xgcd(N, M)[1], B, k)
        # inv(N) mod M'
        Minv_M_p = goto_crt(xgcd(M, M_p)[1], B_p, k)
        # N in base M'
        N_M_p = goto_crt(N, B_p, k)

        # start computing q
        # q = [((a_M[i] * s_M[i] + b_M[i]) *\
        #     (-Ninv_M[i]))%B[i] for i in range(k)]

        fb = cls.white['fb_dim_'+str(dim)]
        q = [0]*k
        q[0] = fb[a_M[0]][b_M[0]] % (1 << 5)
        q[1] = (fb[a_M[1]][b_M[1]] % (1 << 10)) >> 5
        q[2] = (fb[a_M[2]][b_M[2]] % (1 << 15)) >> 10
        q[3] = (fb[a_M[3]][b_M[3]] % (1 << 20)) >> 15
        q[4] = (fb[a_M[4]][b_M[4]]) >> 20

        # shift from base B to B'
        q = goback_crt(q, B, k)
        q = goto_crt(q, B_p, k)

        # compute r
        # r = [((a_M_p[i] * s_M_p[i] + b_M_p[i] + q[i]*N_M_p[i])\
        #     * Minv_M_p[i])%B_p[i] for i in range(k) ]

        sb = cls.white['sb_dim_'+str(dim)]
        r = [(q[i] * N_M_p[i] * Minv_M_p[i]) % B_p[i]
             for i in range(k)]
        r[0] += sb[a_M_p[0]][b_M_p[0]] % (1 << 5)
        r[1] += (sb[a_M_p[1]][b_M_p[1]] % (1 << 10)) >> 5
        r[2] += (sb[a_M_p[2]][b_M_p[2]] % (1 << 15)) >> 10
        r[3] += (sb[a_M_p[3]][b_M_p[3]] % (1 << 20)) >> 15
        r[4] += (sb[a_M_p[4]][b_M_p[4]]) >> 20

        # go back to B from B'
        r = goback_crt(r, B_p, k)

        return r*M

# takes a cipher and decrypts 512 bits


def decrypt_white(a1, a2, degree, modulus, debug=[]):
    # preprocess a2 -> (-a2)
    # because one is preprocess already, no need to postprocess
    tmp_a1 = WB_vector(degree, modulus, False)
    tmp_a1.vector = a1.vector
    tmp_a2 = WB_vector(degree, modulus, False)
    tmp_a2.vector = a2.vector

    # get public data and transform
    root = WB_vector.white['root']
    unroot = WB_vector.white['unroot']
    ninv = WB_vector.white['ninv']
    tmp_a1.goto_ntt(root)
    tmp_a2.goto_ntt(root)

    # go to MM -> box
    tmp = tmp_a1 * tmp_a2

    # reverse transform, optional for input
    # tmp_a1.goback_ntt(unroot, ninv)
    # tmp_a2.goback_ntt(unroot, ninv)
    tmp.goback_ntt(unroot, ninv)

    # fetch masks
    chal = WB_vector.white['chal']
    if chal == 2:
        mask = WB_vector.white['mask']
        rot = WB_vector.white['rotate']

    # get message
    m = np.zeros(degree, dtype=int)
    for i in range(degree):
        # if decoding needed
        if chal == 2:
            m[i] = tmp.vector[(i+rot) % degree] % tmp.modulus
            if m[i] > modulus/2:
                m[i] = 1 - ((m[i] + mask[i]) % 2)

            else:
                m[i] = (m[i] + mask[i]) % 2
        # else
        else:
            m[i] = tmp.vector[i] % tmp.modulus
            if m[i] > modulus/2:
                m[i] = 1 - (m[i] % 2)

            else:
                m[i] = m[i] % 2
    return m


# Main function performing decryption in white-box,
# decryption requires public data and lookup tables.


if __name__ == "__main__":

    print("Retrieving data")
    try:
        with open('pub_enc_data.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("There is no data for Encryption")
        exit()

    try:
        with open('wb_dec_data.json') as f:
            WB_vector.white = json.load(f)
    except FileNotFoundError:
        print("There is no data for WB Decryption")
        exit()

    degree = data['degree']
    modulus = data['modulus']
    pka = NTRU_vector(degree, modulus, False)
    pkb = NTRU_vector(degree, modulus, False)
    pka.vector = np.array(data['pka'])
    pkb.vector = np.array(data['pkb'])

    # check m == Dec(Enc(m)) z times
    cpt2 = 0
    iterate = 10
    print("Testing m == Dec(Enc(m)) over %i random messages" % iterate)
    for z in range(iterate):
        message = np.zeros(degree)
        for i in range(degree):
            message[i] = rand.randint(0, 1)
        a1, a2 = encrypt(message, pka, pkb, degree, modulus)
        m = decrypt_white(a1, a2, degree, modulus, debug=message)
        cpt = 0
        for i in range(degree):
            if not message[i] == m[i]:
                cpt += 1
                print("FATAL ERROR on bit " + str(i))
        if cpt:
            print("We raised "+str((cpt*100)/data['degree'])
                  + "% errors in decryption")
            cpt2 += 1
        print(str((z*100.)/iterate)+"% of ciphers done")
        print("##########################################")

    print("Done : "+str((cpt2*100)/iterate)+"% ciphers were wrong")
