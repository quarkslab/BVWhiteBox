# BV-WhiteBox

![GitHub](https://img.shields.io/github/license/quarkslab/BVWhiteBox)

This repo holds the proposal for an asymmetric lattice-based white-box scheme presented in https://eprint.iacr.org/2020/893

A copy of the thesis manuscript is included on this repo: [`thesis.pdf`](thesis.pdf).

# Copyright

Copyright Lucas Barthélémy & Quarkslab, 2021

This work is published under [Apache 2.0 license](LICENSE.md)

# Installing Requirements

```console
$ python3 -m pip install --user -r requirements.txt
```

# General Overview

This repo holds three different python scripts, each performing a task necessary for setting up/using our proposal for an asymmetric lattice-based white-box scheme:

[`lattice.py`](lattice.py) provides *basic functionalities* for the creation and manipulation of lattice vectors. There are a number of libraries dedicated to lattice cryptography, but we decided to implement our own script to keep things as minimal and simple as possible. In particular, this script holds methods for basic operations (addition, multiplication) as well as methods for the NTT transform and the RNS decomposition. *The user should note that the data type of lattice coefficients is forced to int64, this is required to run on Windows but can be switched to regular int or even set to float for the WB_dec.py script on Unix.*

[`create_WB.py`](create_WB.py) is the script *generating* an instance of our asymmetric lattice-based white-box. That is to say it generates the set of lookup tables that will be used in the white-box decryption algorithm. Given lattice parameters (dimension and modulus), it will generate a public and a secret key. Then, given white-box parameters (RNS basis, type of encodings), it will use the NTT transform, RNS decomposition and Montgomery's multiplication algorithm to generate a set of lookup tables later used by the white-box decryption algorithm. Finally, data is stored on three files: a public file (pub_enc_data.json), a white-box data file (wb_dec_data.json) and a private file (private_data.json). The public data include the parameters used during generation and the public key. The white-box data file include the set of lookup tables and (if applicable) the masking values needed for the final step of decoding. In a practical sense, knowledge of the private data (encrypted encodings and private key) is no longer necessary past this generation step. However, this data can be useful when testing that our white-box works properly.

[`WB_dec.py`](WB_dec.py) is the script *running* an instance of our asymmetric lattice-based white-box. First, the script uses public data to encrypt random 512 bits messages. Then, each message is decrypted using the white-box decryption method. This method only uses public data (parameters, lookup tables, final masks) to decrypt each message. Finally, the script checks that all messages are identical to their original counterpart.

A white-box user (or attacker) would only have access to the following resources:
- the [`lattice.py`](lattice.py) script,
- the [`WB_dec.py`](WB_dec.py) script,
- the public data file.

One way to break an instance of our proposal is to show you managed to retrieve a secret key of an instance of our asymmetric lattice-based white-box from those resources alone.

# Usage

Two scripts demonstrate usage of the library:

```console
$ ./create_WB.py
```
[`create_WB.py`](create_WB.py) generates data used by the white-box decryption algorithm. Parameters for our lattice and RNS bases are set to allow a couple of homomorphic products. While the dimension of our lattice can be increased easily (to increase security level), the reader should note that increasing the modulus (to accommodate better homomorphic capabilities) may also require different RNS bases to accommodate Montgomery's multiplication. In addition, bigger RNS bases may not be compatible with how we chose to store lookup tables. Public parameters and lookup tables are stored in json files named *pub_enc_data.json* and *wb_dec_data.json*. These data will be used by the second script. Private data are stored in a different json file *private_data.json*, **THIS IS DONE FOR DEBUGGING PURPOSES ONLY AND SHOULD NOT BE USED TO ATTACK THE DESIGN**.

```console
$ ./WB_dec.py
```

[`WB_dec.py`](WB_dec.py) encrypts random messages of 512 bits and decrypts them in the whitebox setup. It uses lookup tables generated with the first script to perform decryption. Finally, the script checks that all messages are identical to the original. 

