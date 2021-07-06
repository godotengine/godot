// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "anglebase/sha1.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "anglebase/sys_byteorder.h"

namespace angle
{

namespace base
{

// Implementation of SHA-1. Only handles data in byte-sized blocks,
// which simplifies the code a fair bit.

// Identifier names follow notation in FIPS PUB 180-3, where you'll
// also find a description of the algorithm:
// http://csrc.nist.gov/publications/fips/fips180-3/fips180-3_final.pdf

// Usage example:
//
// SecureHashAlgorithm sha;
// while(there is data to hash)
//   sha.Update(moredata, size of data);
// sha.Final();
// memcpy(somewhere, sha.Digest(), 20);
//
// to reuse the instance of sha, call sha.Init();

// TODO(jhawkins): Replace this implementation with a per-platform
// implementation using each platform's crypto library.  See
// http://crbug.com/47218

class SecureHashAlgorithm
{
  public:
    SecureHashAlgorithm() { Init(); }

    static const int kDigestSizeBytes;

    void Init();
    void Update(const void *data, size_t nbytes);
    void Final();

    // 20 bytes of message digest.
    const unsigned char *Digest() const { return reinterpret_cast<const unsigned char *>(H); }

  private:
    void Pad();
    void Process();

    uint32_t A, B, C, D, E;

    uint32_t H[5];

    union {
        uint32_t W[80];
        uint8_t M[64];
    };

    uint32_t cursor;
    uint64_t l;
};

static inline uint32_t f(uint32_t t, uint32_t B, uint32_t C, uint32_t D)
{
    if (t < 20)
    {
        return (B & C) | ((~B) & D);
    }
    else if (t < 40)
    {
        return B ^ C ^ D;
    }
    else if (t < 60)
    {
        return (B & C) | (B & D) | (C & D);
    }
    else
    {
        return B ^ C ^ D;
    }
}

static inline uint32_t S(uint32_t n, uint32_t X)
{
    return (X << n) | (X >> (32 - n));
}

static inline uint32_t K(uint32_t t)
{
    if (t < 20)
    {
        return 0x5a827999;
    }
    else if (t < 40)
    {
        return 0x6ed9eba1;
    }
    else if (t < 60)
    {
        return 0x8f1bbcdc;
    }
    else
    {
        return 0xca62c1d6;
    }
}

const int SecureHashAlgorithm::kDigestSizeBytes = 20;

void SecureHashAlgorithm::Init()
{
    A      = 0;
    B      = 0;
    C      = 0;
    D      = 0;
    E      = 0;
    cursor = 0;
    l      = 0;
    H[0]   = 0x67452301;
    H[1]   = 0xefcdab89;
    H[2]   = 0x98badcfe;
    H[3]   = 0x10325476;
    H[4]   = 0xc3d2e1f0;
}

void SecureHashAlgorithm::Final()
{
    Pad();
    Process();

    for (int t = 0; t < 5; ++t)
        H[t]   = ByteSwap(H[t]);
}

void SecureHashAlgorithm::Update(const void *data, size_t nbytes)
{
    const uint8_t *d = reinterpret_cast<const uint8_t *>(data);
    while (nbytes--)
    {
        M[cursor++] = *d++;
        if (cursor >= 64)
            Process();
        l += 8;
    }
}

void SecureHashAlgorithm::Pad()
{
    M[cursor++] = 0x80;

    if (cursor > 64 - 8)
    {
        // pad out to next block
        while (cursor < 64)
            M[cursor++] = 0;

        Process();
    }

    while (cursor < 64 - 8)
        M[cursor++] = 0;

    M[cursor++] = (l >> 56) & 0xff;
    M[cursor++] = (l >> 48) & 0xff;
    M[cursor++] = (l >> 40) & 0xff;
    M[cursor++] = (l >> 32) & 0xff;
    M[cursor++] = (l >> 24) & 0xff;
    M[cursor++] = (l >> 16) & 0xff;
    M[cursor++] = (l >> 8) & 0xff;
    M[cursor++] = l & 0xff;
}

void SecureHashAlgorithm::Process()
{
    uint32_t t;

    // Each a...e corresponds to a section in the FIPS 180-3 algorithm.

    // a.
    //
    // W and M are in a union, so no need to memcpy.
    // memcpy(W, M, sizeof(M));
    for (t   = 0; t < 16; ++t)
        W[t] = ByteSwap(W[t]);

    // b.
    for (t   = 16; t < 80; ++t)
        W[t] = S(1, W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16]);

    // c.
    A = H[0];
    B = H[1];
    C = H[2];
    D = H[3];
    E = H[4];

    // d.
    for (t = 0; t < 80; ++t)
    {
        uint32_t TEMP = S(5, A) + f(t, B, C, D) + E + W[t] + K(t);
        E             = D;
        D             = C;
        C             = S(30, B);
        B             = A;
        A             = TEMP;
    }

    // e.
    H[0] += A;
    H[1] += B;
    H[2] += C;
    H[3] += D;
    H[4] += E;

    cursor = 0;
}

std::string SHA1HashString(const std::string &str)
{
    char hash[SecureHashAlgorithm::kDigestSizeBytes];
    SHA1HashBytes(reinterpret_cast<const unsigned char *>(str.c_str()), str.length(),
                  reinterpret_cast<unsigned char *>(hash));
    return std::string(hash, SecureHashAlgorithm::kDigestSizeBytes);
}

void SHA1HashBytes(const unsigned char *data, size_t len, unsigned char *hash)
{
    SecureHashAlgorithm sha;
    sha.Update(data, len);
    sha.Final();

    memcpy(hash, sha.Digest(), SecureHashAlgorithm::kDigestSizeBytes);
}

}  // namespace base

}  // namespace angle
