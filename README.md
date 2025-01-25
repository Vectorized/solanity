# Find a Solana vanity address using GPUs

I originally copied this from here: https://github.com/mcf-rocks/solanity

Then I made the following changes:
1. Optimize, especially the base58 encoding.
2. Make the PRNG cryptographically safe.
3. Allow per-character conditional case insensitivity.
4. Update `src/gpu-common.mk` for RTX4090.

When it finds a match, it will log a line starting with MATCH, you will see the vanity address found and the secret (seed) in hex.

A Solana keypair file is a text file with one line, that has 64 bytes as decimal numbers in json format. The first 32 bytes are the (secret) seed, the last 32 bytes are the public key. This public key, when represented in base58 format, is the (vanity) address. The line you are looking for is immediatley after the match line, something like this:

```
[59,140,24,207,208,39,85,22,191,118,230,168,183,34,21,196,25,202,215,167,74,68,74,29,50,247,170,102,19,66,27,104,136,17,198,97,155,247,112,195,114,159,140,43,11,156,171,32,112,188,1,46,231,106,16,148,200,105,30,83,19,235,139,5]
```

Paste this one line into a file keypair.json for example, and test it by sending funds to and from it. 

## Configure
Open `src/config.h` and add any prefixes you want to scan for to the list.

## Building
Make sure your cuda binary are in your path, and build:

```bash
$ export PATH=/usr/local/cuda/bin:$PATH
$ make -j$(nproc)
```

## Running

```bash
LD_LIBRARY_PATH=./src/release ./src/release/cuda_ed25519_vanity
```
