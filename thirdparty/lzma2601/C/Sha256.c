/* Sha256.c -- SHA-256 Hash
: Igor Pavlov : Public domain
This code is based on public domain code from Wei Dai's Crypto++ library. */

#include "Precomp.h"

#include <string.h>

#include "Sha256.h"
#include "RotateDefs.h"
#include "CpuArch.h"

#ifdef MY_CPU_X86_OR_AMD64
  #if   defined(Z7_LLVM_CLANG_VERSION)  && (Z7_LLVM_CLANG_VERSION  >= 30800) \
     || defined(Z7_APPLE_CLANG_VERSION) && (Z7_APPLE_CLANG_VERSION >= 50100) \
     || defined(Z7_GCC_VERSION)         && (Z7_GCC_VERSION         >= 40900) \
     || defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1600) \
     || defined(_MSC_VER) && (_MSC_VER >= 1200)
      #define Z7_COMPILER_SHA256_SUPPORTED
  #endif
#elif defined(MY_CPU_ARM_OR_ARM64) && defined(MY_CPU_LE)

  #if   defined(__ARM_FEATURE_SHA2) \
     || defined(__ARM_FEATURE_CRYPTO)
    #define Z7_COMPILER_SHA256_SUPPORTED
  #else
    #if  defined(MY_CPU_ARM64) \
      || defined(__ARM_ARCH) && (__ARM_ARCH >= 4) \
      || defined(Z7_MSC_VER_ORIGINAL)
    #if  defined(__ARM_FP) && \
          (   defined(Z7_CLANG_VERSION) && (Z7_CLANG_VERSION >= 30800) \
           || defined(__GNUC__) && (__GNUC__ >= 6) \
          ) \
      || defined(Z7_MSC_VER_ORIGINAL) && (_MSC_VER >= 1910)
    #if  defined(MY_CPU_ARM64) \
      || !defined(Z7_CLANG_VERSION) \
      || defined(__ARM_NEON) && \
          (Z7_CLANG_VERSION < 170000 || \
           Z7_CLANG_VERSION > 170001)
      #define Z7_COMPILER_SHA256_SUPPORTED
    #endif
    #endif
    #endif
  #endif
#endif

void Z7_FASTCALL Sha256_UpdateBlocks(UInt32 state[8], const Byte *data, size_t numBlocks);

#ifdef Z7_COMPILER_SHA256_SUPPORTED
  void Z7_FASTCALL Sha256_UpdateBlocks_HW(UInt32 state[8], const Byte *data, size_t numBlocks);

  static SHA256_FUNC_UPDATE_BLOCKS g_SHA256_FUNC_UPDATE_BLOCKS = Sha256_UpdateBlocks;
  static SHA256_FUNC_UPDATE_BLOCKS g_SHA256_FUNC_UPDATE_BLOCKS_HW;

  #define SHA256_UPDATE_BLOCKS(p) p->v.vars.func_UpdateBlocks
#else
  #define SHA256_UPDATE_BLOCKS(p) Sha256_UpdateBlocks
#endif


BoolInt Sha256_SetFunction(CSha256 *p, unsigned algo)
{
  SHA256_FUNC_UPDATE_BLOCKS func = Sha256_UpdateBlocks;
  
  #ifdef Z7_COMPILER_SHA256_SUPPORTED
    if (algo != SHA256_ALGO_SW)
    {
      if (algo == SHA256_ALGO_DEFAULT)
        func = g_SHA256_FUNC_UPDATE_BLOCKS;
      else
      {
        if (algo != SHA256_ALGO_HW)
          return False;
        func = g_SHA256_FUNC_UPDATE_BLOCKS_HW;
        if (!func)
          return False;
      }
    }
  #else
    if (algo > 1)
      return False;
  #endif

  p->v.vars.func_UpdateBlocks = func;
  return True;
}


/* define it for speed optimization */

#ifdef Z7_SFX
  #define STEP_PRE 1
  #define STEP_MAIN 1
#else
  #define STEP_PRE 2
  #define STEP_MAIN 4
  // #define Z7_SHA256_UNROLL
#endif

#undef Z7_SHA256_BIG_W
#if STEP_MAIN != 16
  #define Z7_SHA256_BIG_W
#endif




void Sha256_InitState(CSha256 *p)
{
  p->v.vars.count = 0;
  p->state[0] = 0x6a09e667;
  p->state[1] = 0xbb67ae85;
  p->state[2] = 0x3c6ef372;
  p->state[3] = 0xa54ff53a;
  p->state[4] = 0x510e527f;
  p->state[5] = 0x9b05688c;
  p->state[6] = 0x1f83d9ab;
  p->state[7] = 0x5be0cd19;
}








void Sha256_Init(CSha256 *p)
{
  p->v.vars.func_UpdateBlocks =
  #ifdef Z7_COMPILER_SHA256_SUPPORTED
      g_SHA256_FUNC_UPDATE_BLOCKS;
  #else
      NULL;
  #endif
  Sha256_InitState(p);
}

#define S0(x) (rotrFixed(x, 2) ^ rotrFixed(x,13) ^ rotrFixed(x,22))
#define S1(x) (rotrFixed(x, 6) ^ rotrFixed(x,11) ^ rotrFixed(x,25))
#define s0(x) (rotrFixed(x, 7) ^ rotrFixed(x,18) ^ (x >> 3))
#define s1(x) (rotrFixed(x,17) ^ rotrFixed(x,19) ^ (x >>10))

#define Ch(x,y,z) (z^(x&(y^z)))
#define Maj(x,y,z) ((x&y)|(z&(x|y)))


#define W_PRE(i) (W[(i) + (size_t)(j)] = GetBe32(data + ((size_t)(j) + i) * 4))

#define blk2_main(j, i)  s1(w(j, (i)-2)) + w(j, (i)-7) + s0(w(j, (i)-15))

#ifdef Z7_SHA256_BIG_W
    // we use +i instead of +(i) to change the order to solve CLANG compiler warning for signed/unsigned.
    #define w(j, i)     W[(size_t)(j) + i]
    #define blk2(j, i)  (w(j, i) = w(j, (i)-16) + blk2_main(j, i))
#else
    #if STEP_MAIN == 16
        #define w(j, i)  W[(i) & 15]
    #else
        #define w(j, i)  W[((size_t)(j) + (i)) & 15]
    #endif
    #define blk2(j, i)  (w(j, i) += blk2_main(j, i))
#endif

#define W_MAIN(i)  blk2(j, i)


#define T1(wx, i) \
    tmp = h + S1(e) + Ch(e,f,g) + K[(i)+(size_t)(j)] + wx(i); \
    h = g; \
    g = f; \
    f = e; \
    e = d + tmp; \
    tmp += S0(a) + Maj(a, b, c); \
    d = c; \
    c = b; \
    b = a; \
    a = tmp; \

#define R1_PRE(i)  T1( W_PRE, i)
#define R1_MAIN(i) T1( W_MAIN, i)

#if (!defined(Z7_SHA256_UNROLL) || STEP_MAIN < 8) && (STEP_MAIN >= 4)
#define R2_MAIN(i) \
    R1_MAIN(i) \
    R1_MAIN(i + 1) \

#endif



#if defined(Z7_SHA256_UNROLL) && STEP_MAIN >= 8

#define T4( a,b,c,d,e,f,g,h, wx, i) \
    h += S1(e) + Ch(e,f,g) + K[(i)+(size_t)(j)] + wx(i); \
    tmp = h; \
    h += d; \
    d = tmp + S0(a) + Maj(a, b, c); \

#define R4( wx, i) \
    T4 ( a,b,c,d,e,f,g,h, wx, (i  )); \
    T4 ( d,a,b,c,h,e,f,g, wx, (i+1)); \
    T4 ( c,d,a,b,g,h,e,f, wx, (i+2)); \
    T4 ( b,c,d,a,f,g,h,e, wx, (i+3)); \

#define R4_PRE(i)  R4( W_PRE, i)
#define R4_MAIN(i) R4( W_MAIN, i)


#define T8( a,b,c,d,e,f,g,h, wx, i) \
    h += S1(e) + Ch(e,f,g) + K[(i)+(size_t)(j)] + wx(i); \
    d += h; \
    h += S0(a) + Maj(a, b, c); \

#define R8( wx, i) \
    T8 ( a,b,c,d,e,f,g,h, wx, i  ); \
    T8 ( h,a,b,c,d,e,f,g, wx, i+1); \
    T8 ( g,h,a,b,c,d,e,f, wx, i+2); \
    T8 ( f,g,h,a,b,c,d,e, wx, i+3); \
    T8 ( e,f,g,h,a,b,c,d, wx, i+4); \
    T8 ( d,e,f,g,h,a,b,c, wx, i+5); \
    T8 ( c,d,e,f,g,h,a,b, wx, i+6); \
    T8 ( b,c,d,e,f,g,h,a, wx, i+7); \

#define R8_PRE(i)  R8( W_PRE, i)
#define R8_MAIN(i) R8( W_MAIN, i)

#endif


extern
MY_ALIGN(64) const UInt32 SHA256_K_ARRAY[64];
MY_ALIGN(64) const UInt32 SHA256_K_ARRAY[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};





#define K SHA256_K_ARRAY

Z7_NO_INLINE
void Z7_FASTCALL Sha256_UpdateBlocks(UInt32 state[8], const Byte *data, size_t numBlocks)
{
  UInt32 W
#ifdef Z7_SHA256_BIG_W
      [64];
#else
      [16];
#endif
  unsigned j;
  UInt32 a,b,c,d,e,f,g,h;
#if !defined(Z7_SHA256_UNROLL) || (STEP_MAIN <= 4) || (STEP_PRE <= 4)
  UInt32 tmp;
#endif
  
  if (numBlocks == 0) return;

  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  f = state[5];
  g = state[6];
  h = state[7];

  do
  {

  for (j = 0; j < 16; j += STEP_PRE)
  {
    #if STEP_PRE > 4

      #if STEP_PRE < 8
      R4_PRE(0);
      #else
      R8_PRE(0);
      #if STEP_PRE == 16
      R8_PRE(8);
      #endif
      #endif

    #else

      R1_PRE(0)
      #if STEP_PRE >= 2
      R1_PRE(1)
      #if STEP_PRE >= 4
      R1_PRE(2)
      R1_PRE(3)
      #endif
      #endif
    
    #endif
  }

  for (j = 16; j < 64; j += STEP_MAIN)
  {
    #if defined(Z7_SHA256_UNROLL) && STEP_MAIN >= 8

      #if STEP_MAIN < 8
      R4_MAIN(0)
      #else
      R8_MAIN(0)
      #if STEP_MAIN == 16
      R8_MAIN(8)
      #endif
      #endif

    #else
      
      R1_MAIN(0)
      #if STEP_MAIN >= 2
      R1_MAIN(1)
      #if STEP_MAIN >= 4
      R2_MAIN(2)
      #if STEP_MAIN >= 8
      R2_MAIN(4)
      R2_MAIN(6)
      #if STEP_MAIN >= 16
      R2_MAIN(8)
      R2_MAIN(10)
      R2_MAIN(12)
      R2_MAIN(14)
      #endif
      #endif
      #endif
      #endif
    #endif
  }

  a += state[0]; state[0] = a;
  b += state[1]; state[1] = b;
  c += state[2]; state[2] = c;
  d += state[3]; state[3] = d;
  e += state[4]; state[4] = e;
  f += state[5]; state[5] = f;
  g += state[6]; state[6] = g;
  h += state[7]; state[7] = h;

  data += SHA256_BLOCK_SIZE;
  }
  while (--numBlocks);
}


#define Sha256_UpdateBlock(p) SHA256_UPDATE_BLOCKS(p)(p->state, p->buffer, 1)

void Sha256_Update(CSha256 *p, const Byte *data, size_t size)
{
  if (size == 0)
    return;
  {
    const unsigned pos = (unsigned)p->v.vars.count & (SHA256_BLOCK_SIZE - 1);
    const unsigned num = SHA256_BLOCK_SIZE - pos;
    p->v.vars.count += size;
    if (num > size)
    {
      memcpy(p->buffer + pos, data, size);
      return;
    }
    if (pos != 0)
    {
      size -= num;
      memcpy(p->buffer + pos, data, num);
      data += num;
      Sha256_UpdateBlock(p);
    }
  }
  {
    const size_t numBlocks = size >> 6;
    // if (numBlocks)
    SHA256_UPDATE_BLOCKS(p)(p->state, data, numBlocks);
    size &= SHA256_BLOCK_SIZE - 1;
    if (size == 0)
      return;
    data += (numBlocks << 6);
    memcpy(p->buffer, data, size);
  }
}


void Sha256_Final(CSha256 *p, Byte *digest)
{
  unsigned pos = (unsigned)p->v.vars.count & (SHA256_BLOCK_SIZE - 1);
  p->buffer[pos++] = 0x80;
  if (pos > (SHA256_BLOCK_SIZE - 4 * 2))
  {
    while (pos != SHA256_BLOCK_SIZE) { p->buffer[pos++] = 0; }
    // memset(&p->buf.buffer[pos], 0, SHA256_BLOCK_SIZE - pos);
    Sha256_UpdateBlock(p);
    pos = 0;
  }
  memset(&p->buffer[pos], 0, (SHA256_BLOCK_SIZE - 4 * 2) - pos);
  {
    const UInt64 numBits = p->v.vars.count << 3;
    SetBe32(p->buffer + SHA256_BLOCK_SIZE - 4 * 2, (UInt32)(numBits >> 32))
    SetBe32(p->buffer + SHA256_BLOCK_SIZE - 4 * 1, (UInt32)(numBits))
  }
  Sha256_UpdateBlock(p);
#if 1 && defined(MY_CPU_BE)
  memcpy(digest, p->state, SHA256_DIGEST_SIZE);
#else
  {
    unsigned i;
    for (i = 0; i < 8; i += 2)
    {
      const UInt32 v0 = p->state[i];
      const UInt32 v1 = p->state[(size_t)i + 1];
      SetBe32(digest    , v0)
      SetBe32(digest + 4, v1)
      digest += 4 * 2;
    }
  }




#endif
  Sha256_InitState(p);
}


void Sha256Prepare(void)
{
#ifdef Z7_COMPILER_SHA256_SUPPORTED
  SHA256_FUNC_UPDATE_BLOCKS f, f_hw;
  f = Sha256_UpdateBlocks;
  f_hw = NULL;
#ifdef MY_CPU_X86_OR_AMD64
  if (CPU_IsSupported_SHA()
      && CPU_IsSupported_SSSE3()
      )
#else
  if (CPU_IsSupported_SHA2())
#endif
  {
    // printf("\n========== HW SHA256 ======== \n");
    f = f_hw = Sha256_UpdateBlocks_HW;
  }
  g_SHA256_FUNC_UPDATE_BLOCKS    = f;
  g_SHA256_FUNC_UPDATE_BLOCKS_HW = f_hw;
#endif
}

#undef U64C
#undef K
#undef S0
#undef S1
#undef s0
#undef s1
#undef Ch
#undef Maj
#undef W_MAIN
#undef W_PRE
#undef w
#undef blk2_main
#undef blk2
#undef T1
#undef T4
#undef T8
#undef R1_PRE
#undef R1_MAIN
#undef R2_MAIN
#undef R4
#undef R4_PRE
#undef R4_MAIN
#undef R8
#undef R8_PRE
#undef R8_MAIN
#undef STEP_PRE
#undef STEP_MAIN
#undef Z7_SHA256_BIG_W
#undef Z7_SHA256_UNROLL
#undef Z7_COMPILER_SHA256_SUPPORTED
