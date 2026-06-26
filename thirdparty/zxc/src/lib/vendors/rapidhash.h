/*
 * rapidhash V3 - Very fast, high quality, platform-independent hashing algorithm.
 *
 * Based on 'wyhash', by Wang Yi <godspeed_china@yeah.net>
 * 
 * Copyright (C) 2025 Nicolas De Carli
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You can contact the author at:
 *   - rapidhash source repository: https://github.com/Nicoshev/rapidhash
 */

 #pragma once
 
/*
 *  Includes.
 */
 #include <stdint.h>
 #include <string.h>
 #if defined(_MSC_VER)
 # include <intrin.h>
 # if defined(_M_X64) && !defined(_M_ARM64EC)
 #   pragma intrinsic(_umul128)
 # endif
 #endif
 
 /*
  *  C/C++ macros.
  */
 
 #ifdef _MSC_VER
 # define RAPIDHASH_ALWAYS_INLINE __forceinline
 #elif defined(__GNUC__)
 # define RAPIDHASH_ALWAYS_INLINE inline __attribute__((__always_inline__))
 #else
 # define RAPIDHASH_ALWAYS_INLINE inline
 #endif
 
 #ifdef __cplusplus
 # define RAPIDHASH_NOEXCEPT noexcept
 # define RAPIDHASH_CONSTEXPR constexpr
 # ifndef RAPIDHASH_INLINE
 #   define RAPIDHASH_INLINE RAPIDHASH_ALWAYS_INLINE
 # endif
 # if __cplusplus >= 201402L && !defined(_MSC_VER)
 #   define RAPIDHASH_INLINE_CONSTEXPR RAPIDHASH_ALWAYS_INLINE constexpr
 # else
 #   define RAPIDHASH_INLINE_CONSTEXPR RAPIDHASH_ALWAYS_INLINE
 # endif
 #else
 # define RAPIDHASH_NOEXCEPT
 # define RAPIDHASH_CONSTEXPR static const
 # ifndef RAPIDHASH_INLINE
 #   define RAPIDHASH_INLINE static RAPIDHASH_ALWAYS_INLINE
 # endif
 # define RAPIDHASH_INLINE_CONSTEXPR RAPIDHASH_INLINE
 #endif

 /*
  *  Unrolled macro.
  *  Improves large input speed, but increases code size and worsens small input speed.
  *
  *  RAPIDHASH_COMPACT: Normal behavior.
  *  RAPIDHASH_UNROLLED: 
  *
  */
  #ifndef RAPIDHASH_UNROLLED
  # define RAPIDHASH_COMPACT
  #elif defined(RAPIDHASH_COMPACT)
  # error "cannot define RAPIDHASH_COMPACT and RAPIDHASH_UNROLLED simultaneously."
  #endif
 
 /*
  *  Protection macro, alters behaviour of rapid_mum multiplication function.
  *
  *  RAPIDHASH_FAST: Normal behavior, max speed.
  *  RAPIDHASH_PROTECTED: Extra protection against entropy loss.
  */
 #ifndef RAPIDHASH_PROTECTED
 # define RAPIDHASH_FAST
 #elif defined(RAPIDHASH_FAST)
 # error "cannot define RAPIDHASH_PROTECTED and RAPIDHASH_FAST simultaneously."
 #endif
 
 /*
  *  Likely and unlikely macros.
  */
 #if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
 # define _likely_(x)  __builtin_expect(x,1)
 # define _unlikely_(x)  __builtin_expect(x,0)
 #else
 # define _likely_(x) (x)
 # define _unlikely_(x) (x)
 #endif
 
 /*
  *  Endianness macros.
  */
 #ifndef RAPIDHASH_LITTLE_ENDIAN
 # if defined(_WIN32) || defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
 #   define RAPIDHASH_LITTLE_ENDIAN
 # elif defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
 #   define RAPIDHASH_BIG_ENDIAN
 # else
 #   warning "could not determine endianness! Falling back to little endian."
 #   define RAPIDHASH_LITTLE_ENDIAN
 # endif
 #endif
 
 /*
  *  Default secret parameters.
  */
   RAPIDHASH_CONSTEXPR uint64_t rapid_secret[8] = {
     0x2d358dccaa6c78a5ull,
     0x8bb84b93962eacc9ull,
     0x4b33a62ed433d4a3ull,
     0x4d5a2da51de1aa47ull,
     0xa0761d6478bd642full,
     0xe7037ed1a0b428dbull,
     0x90ed1765281c388cull,
     0xaaaaaaaaaaaaaaaaull};
 
 /*
  *  64*64 -> 128bit multiply function.
  *
  *  @param A  Address of 64-bit number.
  *  @param B  Address of 64-bit number.
  *
  *  Calculates 128-bit C = *A * *B.
  *
  *  When RAPIDHASH_FAST is defined:
  *  Overwrites A contents with C's low 64 bits.
  *  Overwrites B contents with C's high 64 bits.
  *
  *  When RAPIDHASH_PROTECTED is defined:
  *  Xors and overwrites A contents with C's low 64 bits.
  *  Xors and overwrites B contents with C's high 64 bits.
  */
 RAPIDHASH_INLINE_CONSTEXPR void rapid_mum(uint64_t *A, uint64_t *B) RAPIDHASH_NOEXCEPT {
 #if defined(__SIZEOF_INT128__)
   __uint128_t r=*A; r*=*B;
   #ifdef RAPIDHASH_PROTECTED
   *A^=(uint64_t)r; *B^=(uint64_t)(r>>64);
   #else
   *A=(uint64_t)r; *B=(uint64_t)(r>>64);
   #endif
 #elif defined(_MSC_VER) && (defined(_WIN64) || defined(_M_HYBRID_CHPE_ARM64))
   #if defined(_M_X64)
     #ifdef RAPIDHASH_PROTECTED
     uint64_t a, b;
     a=_umul128(*A,*B,&b);
     *A^=a;  *B^=b;
     #else
     *A=_umul128(*A,*B,B);
     #endif
   #else
     #ifdef RAPIDHASH_PROTECTED
     uint64_t a, b;
     b = __umulh(*A, *B);
     a = *A * *B;
     *A^=a;  *B^=b;
     #else
     uint64_t c = __umulh(*A, *B);
     *A = *A * *B;
     *B = c;
     #endif
   #endif
 #else
   uint64_t ha=*A>>32, hb=*B>>32, la=(uint32_t)*A, lb=(uint32_t)*B;
   uint64_t rh=ha*hb, rm0=ha*lb, rm1=hb*la, rl=la*lb, t=rl+(rm0<<32), c=t<rl;
   uint64_t lo=t+(rm1<<32); 
   c+=lo<t; 
   uint64_t hi=rh+(rm0>>32)+(rm1>>32)+c;
   #ifdef RAPIDHASH_PROTECTED
   *A^=lo;  *B^=hi;
   #else
   *A=lo;  *B=hi;
   #endif
 #endif
 }
 
 /*
  *  Multiply and xor mix function.
  *
  *  @param A  64-bit number.
  *  @param B  64-bit number.
  *
  *  Calculates 128-bit C = A * B.
  *  Returns 64-bit xor between high and low 64 bits of C.
  */
  RAPIDHASH_INLINE_CONSTEXPR uint64_t rapid_mix(uint64_t A, uint64_t B) RAPIDHASH_NOEXCEPT { rapid_mum(&A,&B); return A^B; }
 
 /*
  *  Read functions.
  */
 #ifdef RAPIDHASH_LITTLE_ENDIAN
 RAPIDHASH_INLINE uint64_t rapid_read64(const uint8_t *p) RAPIDHASH_NOEXCEPT { uint64_t v; memcpy(&v, p, sizeof(uint64_t)); return v;}
 RAPIDHASH_INLINE uint64_t rapid_read32(const uint8_t *p) RAPIDHASH_NOEXCEPT { uint32_t v; memcpy(&v, p, sizeof(uint32_t)); return v;}
 #elif defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
 RAPIDHASH_INLINE uint64_t rapid_read64(const uint8_t *p) RAPIDHASH_NOEXCEPT { uint64_t v; memcpy(&v, p, sizeof(uint64_t)); return __builtin_bswap64(v);}
 RAPIDHASH_INLINE uint64_t rapid_read32(const uint8_t *p) RAPIDHASH_NOEXCEPT { uint32_t v; memcpy(&v, p, sizeof(uint32_t)); return __builtin_bswap32(v);}
 #elif defined(_MSC_VER)
 RAPIDHASH_INLINE uint64_t rapid_read64(const uint8_t *p) RAPIDHASH_NOEXCEPT { uint64_t v; memcpy(&v, p, sizeof(uint64_t)); return _byteswap_uint64(v);}
 RAPIDHASH_INLINE uint64_t rapid_read32(const uint8_t *p) RAPIDHASH_NOEXCEPT { uint32_t v; memcpy(&v, p, sizeof(uint32_t)); return _byteswap_ulong(v);}
 #else
 RAPIDHASH_INLINE uint64_t rapid_read64(const uint8_t *p) RAPIDHASH_NOEXCEPT {
   uint64_t v; memcpy(&v, p, 8);
   return (((v >> 56) & 0xff)| ((v >> 40) & 0xff00)| ((v >> 24) & 0xff0000)| ((v >>  8) & 0xff000000)| ((v <<  8) & 0xff00000000)| ((v << 24) & 0xff0000000000)| ((v << 40) & 0xff000000000000)| ((v << 56) & 0xff00000000000000));
 }
 RAPIDHASH_INLINE uint64_t rapid_read32(const uint8_t *p) RAPIDHASH_NOEXCEPT {
   uint32_t v; memcpy(&v, p, 4);
   return (((v >> 24) & 0xff)| ((v >>  8) & 0xff00)| ((v <<  8) & 0xff0000)| ((v << 24) & 0xff000000));
 }
 #endif
 
 /*
  *  rapidhash main function.
  *
  *  @param key     Buffer to be hashed.
  *  @param len     @key length, in bytes.
  *  @param seed    64-bit seed used to alter the hash result predictably.
  *  @param secret  Triplet of 64-bit secrets used to alter hash result predictably.
  *
  *  Returns a 64-bit hash.
  */
RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhash_internal(const void *key, size_t len, uint64_t seed, const uint64_t* secret) RAPIDHASH_NOEXCEPT {
  const uint8_t *p=(const uint8_t *)key;
  seed ^= rapid_mix(seed ^ secret[2], secret[1]);
  uint64_t a=0, b=0;
  size_t i = len;
  if (_likely_(len <= 16)) {
    if (len >= 4) {
      seed ^= len;
      if (len >= 8) {
        const uint8_t* plast = p + len - 8;
        a = rapid_read64(p);
        b = rapid_read64(plast);
      } else {
        const uint8_t* plast = p + len - 4;
        a = rapid_read32(p);
        b = rapid_read32(plast);
      }
    } else if (len > 0) {
      a = (((uint64_t)p[0])<<45)|p[len-1];
      b = p[len>>1];
    } else
      a = b = 0;
  } else {
    if (len > 112) {
      uint64_t see1 = seed, see2 = seed;
      uint64_t see3 = seed, see4 = seed;
      uint64_t see5 = seed, see6 = seed;
#ifdef RAPIDHASH_COMPACT
      do {
        seed = rapid_mix(rapid_read64(p) ^ secret[0], rapid_read64(p + 8) ^ seed);
        see1 = rapid_mix(rapid_read64(p + 16) ^ secret[1], rapid_read64(p + 24) ^ see1);
        see2 = rapid_mix(rapid_read64(p + 32) ^ secret[2], rapid_read64(p + 40) ^ see2);
        see3 = rapid_mix(rapid_read64(p + 48) ^ secret[3], rapid_read64(p + 56) ^ see3);
        see4 = rapid_mix(rapid_read64(p + 64) ^ secret[4], rapid_read64(p + 72) ^ see4);
        see5 = rapid_mix(rapid_read64(p + 80) ^ secret[5], rapid_read64(p + 88) ^ see5);
        see6 = rapid_mix(rapid_read64(p + 96) ^ secret[6], rapid_read64(p + 104) ^ see6);
        p += 112;
        i -= 112;
      } while(i > 112);
#else
      while (i > 224) {
        seed = rapid_mix(rapid_read64(p) ^ secret[0], rapid_read64(p + 8) ^ seed);
        see1 = rapid_mix(rapid_read64(p + 16) ^ secret[1], rapid_read64(p + 24) ^ see1);
        see2 = rapid_mix(rapid_read64(p + 32) ^ secret[2], rapid_read64(p + 40) ^ see2);
        see3 = rapid_mix(rapid_read64(p + 48) ^ secret[3], rapid_read64(p + 56) ^ see3);
        see4 = rapid_mix(rapid_read64(p + 64) ^ secret[4], rapid_read64(p + 72) ^ see4);
        see5 = rapid_mix(rapid_read64(p + 80) ^ secret[5], rapid_read64(p + 88) ^ see5);
        see6 = rapid_mix(rapid_read64(p + 96) ^ secret[6], rapid_read64(p + 104) ^ see6);
        seed = rapid_mix(rapid_read64(p + 112) ^ secret[0], rapid_read64(p + 120) ^ seed);
        see1 = rapid_mix(rapid_read64(p + 128) ^ secret[1], rapid_read64(p + 136) ^ see1);
        see2 = rapid_mix(rapid_read64(p + 144) ^ secret[2], rapid_read64(p + 152) ^ see2);
        see3 = rapid_mix(rapid_read64(p + 160) ^ secret[3], rapid_read64(p + 168) ^ see3);
        see4 = rapid_mix(rapid_read64(p + 176) ^ secret[4], rapid_read64(p + 184) ^ see4);
        see5 = rapid_mix(rapid_read64(p + 192) ^ secret[5], rapid_read64(p + 200) ^ see5);
        see6 = rapid_mix(rapid_read64(p + 208) ^ secret[6], rapid_read64(p + 216) ^ see6);
        p += 224;
        i -= 224;
      }
      if (i > 112) {
        seed = rapid_mix(rapid_read64(p) ^ secret[0], rapid_read64(p + 8) ^ seed);
        see1 = rapid_mix(rapid_read64(p + 16) ^ secret[1], rapid_read64(p + 24) ^ see1);
        see2 = rapid_mix(rapid_read64(p + 32) ^ secret[2], rapid_read64(p + 40) ^ see2);
        see3 = rapid_mix(rapid_read64(p + 48) ^ secret[3], rapid_read64(p + 56) ^ see3);
        see4 = rapid_mix(rapid_read64(p + 64) ^ secret[4], rapid_read64(p + 72) ^ see4);
        see5 = rapid_mix(rapid_read64(p + 80) ^ secret[5], rapid_read64(p + 88) ^ see5);
        see6 = rapid_mix(rapid_read64(p + 96) ^ secret[6], rapid_read64(p + 104) ^ see6);
        p += 112;
        i -= 112;
      }
#endif
      seed ^= see1;
      see2 ^= see3;
      see4 ^= see5;
      seed ^= see6;
      see2 ^= see4;
      seed ^= see2;
    }
    if (i > 16) {
      seed = rapid_mix(rapid_read64(p) ^ secret[2], rapid_read64(p + 8) ^ seed);
      if (i > 32) {
          seed = rapid_mix(rapid_read64(p + 16) ^ secret[2], rapid_read64(p + 24) ^ seed);
          if (i > 48) {
              seed = rapid_mix(rapid_read64(p + 32) ^ secret[1], rapid_read64(p + 40) ^ seed);
              if (i > 64) {
                  seed = rapid_mix(rapid_read64(p + 48) ^ secret[1], rapid_read64(p + 56) ^ seed);
                  if (i > 80) {
                      seed = rapid_mix(rapid_read64(p + 64) ^ secret[2], rapid_read64(p + 72) ^ seed);
                      if (i > 96) {
                          seed = rapid_mix(rapid_read64(p + 80) ^ secret[1], rapid_read64(p + 88) ^ seed);
                      }
                  }
              }
          }
      }
    }
    a=rapid_read64(p+i-16) ^ i;  b=rapid_read64(p+i-8);
  }
  a ^= secret[1];
  b ^= seed;
  rapid_mum(&a, &b);
  return rapid_mix(a ^ secret[7], b ^ secret[1] ^ i);
}

 /*
  *  rapidhashMicro main function.
  *
  *  @param key     Buffer to be hashed.
  *  @param len     @key length, in bytes.
  *  @param seed    64-bit seed used to alter the hash result predictably.
  *  @param secret  Triplet of 64-bit secrets used to alter hash result predictably.
  *
  *  Returns a 64-bit hash.
  */
  RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhashMicro_internal(const void *key, size_t len, uint64_t seed, const uint64_t* secret) RAPIDHASH_NOEXCEPT {
    const uint8_t *p=(const uint8_t *)key;
    seed ^= rapid_mix(seed ^ secret[2], secret[1]);
    uint64_t a=0, b=0;
    size_t i = len;
    if (_likely_(len <= 16)) {
      if (len >= 4) {
        seed ^= len;
        if (len >= 8) {
          const uint8_t* plast = p + len - 8;
          a = rapid_read64(p);
          b = rapid_read64(plast);
        } else {
          const uint8_t* plast = p + len - 4;
          a = rapid_read32(p);
          b = rapid_read32(plast);
        }
      } else if (len > 0) {
        a = (((uint64_t)p[0])<<45)|p[len-1];
        b = p[len>>1];
      } else
        a = b = 0;
    } else {
      if (i > 80) {
        uint64_t see1 = seed, see2 = seed;
        uint64_t see3 = seed, see4 = seed;
        do {
          seed = rapid_mix(rapid_read64(p) ^ secret[0], rapid_read64(p + 8) ^ seed);
          see1 = rapid_mix(rapid_read64(p + 16) ^ secret[1], rapid_read64(p + 24) ^ see1);
          see2 = rapid_mix(rapid_read64(p + 32) ^ secret[2], rapid_read64(p + 40) ^ see2);
          see3 = rapid_mix(rapid_read64(p + 48) ^ secret[3], rapid_read64(p + 56) ^ see3);
          see4 = rapid_mix(rapid_read64(p + 64) ^ secret[4], rapid_read64(p + 72) ^ see4);
          p += 80;
          i -= 80;
        } while(i > 80);
        seed ^= see1;
        see2 ^= see3;
        seed ^= see4;
        seed ^= see2;
      }
      if (i > 16) {
        seed = rapid_mix(rapid_read64(p) ^ secret[2], rapid_read64(p + 8) ^ seed);
        if (i > 32) {
            seed = rapid_mix(rapid_read64(p + 16) ^ secret[2], rapid_read64(p + 24) ^ seed);
            if (i > 48) {
                seed = rapid_mix(rapid_read64(p + 32) ^ secret[1], rapid_read64(p + 40) ^ seed);
                if (i > 64) {
                    seed = rapid_mix(rapid_read64(p + 48) ^ secret[1], rapid_read64(p + 56) ^ seed);
                }
            }
        }
      }
      a=rapid_read64(p+i-16) ^ i;  b=rapid_read64(p+i-8);
    }
    a ^= secret[1];
    b ^= seed;
    rapid_mum(&a, &b);
    return rapid_mix(a ^ secret[7], b ^ secret[1] ^ i);
  }

  /*
  *  rapidhashNano main function.
  *
  *  @param key     Buffer to be hashed.
  *  @param len     @key length, in bytes.
  *  @param seed    64-bit seed used to alter the hash result predictably.
  *  @param secret  Triplet of 64-bit secrets used to alter hash result predictably.
  *
  *  Returns a 64-bit hash.
  */
  RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhashNano_internal(const void *key, size_t len, uint64_t seed, const uint64_t* secret) RAPIDHASH_NOEXCEPT {
    const uint8_t *p=(const uint8_t *)key;
    seed ^= rapid_mix(seed ^ secret[2], secret[1]);
    uint64_t a=0, b=0;
    size_t i = len;
    if (_likely_(len <= 16)) {
      if (len >= 4) {
        seed ^= len;
        if (len >= 8) {
          const uint8_t* plast = p + len - 8;
          a = rapid_read64(p);
          b = rapid_read64(plast);
        } else {
          const uint8_t* plast = p + len - 4;
          a = rapid_read32(p);
          b = rapid_read32(plast);
        }
      } else if (len > 0) {
        a = (((uint64_t)p[0])<<45)|p[len-1];
        b = p[len>>1];
      } else
        a = b = 0;
    } else {
      if (i > 48) {
        uint64_t see1 = seed, see2 = seed;
        do {
          seed = rapid_mix(rapid_read64(p) ^ secret[0], rapid_read64(p + 8) ^ seed);
          see1 = rapid_mix(rapid_read64(p + 16) ^ secret[1], rapid_read64(p + 24) ^ see1);
          see2 = rapid_mix(rapid_read64(p + 32) ^ secret[2], rapid_read64(p + 40) ^ see2);
          p += 48;
          i -= 48;
        } while(i > 48);
        seed ^= see1;
        seed ^= see2;
      }
      if (i > 16) {
        seed = rapid_mix(rapid_read64(p) ^ secret[2], rapid_read64(p + 8) ^ seed);
        if (i > 32) {
            seed = rapid_mix(rapid_read64(p + 16) ^ secret[2], rapid_read64(p + 24) ^ seed);
        }
      }
      a=rapid_read64(p+i-16) ^ i;  b=rapid_read64(p+i-8);
    }
    a ^= secret[1];
    b ^= seed;
    rapid_mum(&a, &b);
    return rapid_mix(a ^ secret[7], b ^ secret[1] ^ i);
  }
 
/*
 *  rapidhash seeded hash function.
 *
 *  @param key     Buffer to be hashed.
 *  @param len     @key length, in bytes.
 *  @param seed    64-bit seed used to alter the hash result predictably.
 *
 *  Calls rapidhash_internal using provided parameters and default secrets.
 *
 *  Returns a 64-bit hash.
 */
RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhash_withSeed(const void *key, size_t len, uint64_t seed) RAPIDHASH_NOEXCEPT {
  return rapidhash_internal(key, len, seed, rapid_secret);
}
 
/*
 *  rapidhash general purpose hash function.
 *
 *  @param key     Buffer to be hashed.
 *  @param len     @key length, in bytes.
 *
 *  Calls rapidhash_withSeed using provided parameters and the default seed.
 *
 *  Returns a 64-bit hash.
 */
RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhash(const void *key, size_t len) RAPIDHASH_NOEXCEPT {
  return rapidhash_withSeed(key, len, 0);
}

/*
 *  rapidhashMicro seeded hash function.
 *
 *  Designed for HPC and server applications, where cache misses make a noticeable performance detriment.
 *  Clang-18+ compiles it to ~140 instructions without stack usage, both on x86-64 and aarch64.
 *  Faster for sizes up to 512 bytes, just 15%-20% slower for inputs above 1kb.
 *
 *  @param key     Buffer to be hashed.
 *  @param len     @key length, in bytes.
 *  @param seed    64-bit seed used to alter the hash result predictably.
 *
 *  Calls rapidhash_internal using provided parameters and default secrets.
 *
 *  Returns a 64-bit hash.
 */
 RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhashMicro_withSeed(const void *key, size_t len, uint64_t seed) RAPIDHASH_NOEXCEPT {
  return rapidhashMicro_internal(key, len, seed, rapid_secret);
}
 
/*
 *  rapidhashMicro hash function.
 *
 *  @param key     Buffer to be hashed.
 *  @param len     @key length, in bytes.
 *
 *  Calls rapidhash_withSeed using provided parameters and the default seed.
 *
 *  Returns a 64-bit hash.
 */
RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhashMicro(const void *key, size_t len) RAPIDHASH_NOEXCEPT {
  return rapidhashMicro_withSeed(key, len, 0);
}

/*
 *  rapidhashNano seeded hash function.
 *
 *  @param key     Buffer to be hashed.
 *  @param len     @key length, in bytes.
 *  @param seed    64-bit seed used to alter the hash result predictably.
 *
 *  Calls rapidhash_internal using provided parameters and default secrets.
 *
 *  Returns a 64-bit hash.
 */
 RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhashNano_withSeed(const void *key, size_t len, uint64_t seed) RAPIDHASH_NOEXCEPT {
  return rapidhashNano_internal(key, len, seed, rapid_secret);
}
 
/*
 *  rapidhashNano hash function.
 *
 *  Designed for Mobile and embedded applications, where keeping a small code size is a top priority.
 *  Clang-18+ compiles it to less than 100 instructions without stack usage, both on x86-64 and aarch64.
 *  The fastest for sizes up to 48 bytes, but may be considerably slower for larger inputs.
 *
 *  @param key     Buffer to be hashed.
 *  @param len     @key length, in bytes.
 *
 *  Calls rapidhash_withSeed using provided parameters and the default seed.
 *
 *  Returns a 64-bit hash.
 */
RAPIDHASH_INLINE_CONSTEXPR uint64_t rapidhashNano(const void *key, size_t len) RAPIDHASH_NOEXCEPT {
  return rapidhashNano_withSeed(key, len, 0);
}
