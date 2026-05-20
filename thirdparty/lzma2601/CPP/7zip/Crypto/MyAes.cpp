// Crypto/MyAes.cpp

#include "StdAfx.h"

#include "../../../C/CpuArch.h"

#include "MyAes.h"

namespace NCrypto {

static struct CAesTabInit { CAesTabInit() { AesGenTables();} } g_AesTabInit;

CAesCoder::CAesCoder(
      // bool encodeMode,
      unsigned keySize
      // , bool ctrMode
      ):
  _keyIsSet(false),
  // _encodeMode(encodeMode),
  // _ctrMode(ctrMode),
  _keySize(keySize),
  // _ctrPos(0), // _ctrPos =0 will be set in Init()
  _aes(AES_NUM_IVMRK_WORDS * 4 + AES_BLOCK_SIZE * 2)
{
  // _offset = ((0 - (unsigned)(ptrdiff_t)_aes) & 0xF) / sizeof(UInt32);
  memset(_iv, 0, AES_BLOCK_SIZE);
  /*
  // we can use the following code to test 32-bit overflow case for AES-CTR
  for (unsigned i = 0; i < 16; i++) _iv[i] = (Byte)(i + 1);
  _iv[0] = 0xFE; _iv[1] = _iv[2] = _iv[3] = 0xFF;
  */
}

Z7_COM7F_IMF(CAesCoder::Init())
{
  _ctrPos = 0;
  AesCbc_Init(Aes(), _iv);
  return _keyIsSet ? S_OK : E_NOTIMPL; // E_FAIL
}

Z7_COM7F_IMF2(UInt32, CAesCoder::Filter(Byte *data, UInt32 size))
{
  if (!_keyIsSet)
    return 0;
  if (size < AES_BLOCK_SIZE)
  {
    if (size == 0)
      return 0;
    return AES_BLOCK_SIZE;
  }
  size >>= 4;
  // (data) must be aligned for 16-bytes here
  _codeFunc(Aes(), data, size);
  return size << 4;
}


Z7_COM7F_IMF(CAesCoder::SetKey(const Byte *data, UInt32 size))
{
  if ((size & 0x7) != 0 || size < 16 || size > 32)
    return E_INVALIDARG;
  if (_keySize != 0 && size != _keySize)
    return E_INVALIDARG;
  _setKeyFunc(Aes() + 4, data, size);
  _keyIsSet = true;
  return S_OK;
}

Z7_COM7F_IMF(CAesCoder::SetInitVector(const Byte *data, UInt32 size))
{
  if (size != AES_BLOCK_SIZE)
    return E_INVALIDARG;
  memcpy(_iv, data, size);
  /* we allow SetInitVector() call before SetKey() call.
     so we ignore possible error in Init() here */
  CAesCoder::Init(); // don't call virtual function here !!!
  return S_OK;
}


#ifndef Z7_SFX

/*
Z7_COM7F_IMF(CAesCtrCoder::Init())
{
  _ctrPos = 0;
  return CAesCoder::Init();
}
*/

Z7_COM7F_IMF2(UInt32, CAesCtrCoder::Filter(Byte *data, UInt32 size))
{
  if (!_keyIsSet)
    return 0;
  if (size == 0)
    return 0;
  
  if (_ctrPos != 0)
  {
    /* Optimized caller will not call here */
    const Byte *ctr = (Byte *)(Aes() + AES_NUM_IVMRK_WORDS);
    unsigned num = 0;
    for (unsigned i = _ctrPos; i != AES_BLOCK_SIZE; i++)
    {
      if (num == size)
      {
        _ctrPos = i;
        return num;
      }
      data[num++] ^= ctr[i];
    }
    _ctrPos = 0;
    /* if (num < size) {
       we can filter more data with _codeFunc().
       But it's supposed that the caller can work correctly,
       even if we do only partial filtering here.
       So we filter data only for current 16-byte block. }
    */
    /*
    size -= num;
    size >>= 4;
    // (data) must be aligned for 16-bytes here
    _codeFunc(Aes(), data + num, size);
    return num + (size << 4);
    */
    return num;
  }
  
  if (size < AES_BLOCK_SIZE)
  {
    /* The good optimized caller can call here only in last Filter() call.
       But we support also non-optimized callers,
       where another Filter() calls are allowed after this call.
    */
    Byte *ctr = (Byte *)(Aes() + AES_NUM_IVMRK_WORDS);
    memset(ctr, 0, AES_BLOCK_SIZE);
    memcpy(ctr, data, size);
    _codeFunc(Aes(), ctr, 1);
    memcpy(data, ctr, size);
    _ctrPos = size;
    return size;
  }
  
  size >>= 4;
  // (data) must be aligned for 16-bytes here
  _codeFunc(Aes(), data, size);
  return size << 4;
}

#endif // Z7_SFX


#ifndef Z7_EXTRACT_ONLY

#ifdef MY_CPU_X86_OR_AMD64

  #if defined(__INTEL_COMPILER)
    #if (__INTEL_COMPILER >= 1110)
      #define USE_HW_AES
      #if (__INTEL_COMPILER >= 1900)
        #define USE_HW_VAES
      #endif
    #endif
  #elif defined(Z7_CLANG_VERSION) && (Z7_CLANG_VERSION >= 30800) \
     || defined(Z7_GCC_VERSION)   && (Z7_GCC_VERSION   >= 40400)
    #define USE_HW_AES
      #if defined(__clang__) && (__clang_major__ >= 8) \
          || defined(__GNUC__) && (__GNUC__ >= 8)
        #define USE_HW_VAES
      #endif
  #elif defined(_MSC_VER)
    #define USE_HW_AES
    #define USE_HW_VAES
  #endif

#elif defined(MY_CPU_ARM_OR_ARM64) && defined(MY_CPU_LE)
  
  #if   defined(__ARM_FEATURE_AES) \
     || defined(__ARM_FEATURE_CRYPTO)
    #define USE_HW_AES
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
      #define USE_HW_AES
    #endif
    #endif
    #endif
  #endif
#endif

#ifdef USE_HW_AES
// #pragma message("=== MyAES.c USE_HW_AES === ")

    #define SET_AES_FUNC_2(f2) \
      if (algo == 2) if (g_Aes_SupportedFunctions_Flags & k_Aes_SupportedFunctions_HW) \
      { f = f2; }
  #ifdef USE_HW_VAES
    #define SET_AES_FUNC_23(f2, f3) \
      SET_AES_FUNC_2(f2) \
      if (algo == 3) if (g_Aes_SupportedFunctions_Flags & k_Aes_SupportedFunctions_HW_256) \
      { f = f3; }
  #else  // USE_HW_VAES
    #define SET_AES_FUNC_23(f2, f3) \
      SET_AES_FUNC_2(f2)
  #endif // USE_HW_VAES
#else  // USE_HW_AES
    #define SET_AES_FUNC_23(f2, f3)
#endif // USE_HW_AES

#define SET_AES_FUNCS(c, f0, f1, f2, f3) \
  bool c::SetFunctions(UInt32 algo) { \
  _codeFunc = f0; if (algo < 1) return true; \
  AES_CODE_FUNC f = NULL; \
  if (algo == 1) { f = f1; } \
  SET_AES_FUNC_23(f2, f3) \
  if (f) { _codeFunc = f; return true; } \
  return false; }



#ifndef Z7_SFX
SET_AES_FUNCS(
    CAesCtrCoder,
  g_AesCtr_Code,
    AesCtr_Code,
    AesCtr_Code_HW,
    AesCtr_Code_HW_256)
#endif

SET_AES_FUNCS(
    CAesCbcEncoder,
  g_AesCbc_Encode,
    AesCbc_Encode,
    AesCbc_Encode_HW,
    AesCbc_Encode_HW)

SET_AES_FUNCS(
    CAesCbcDecoder,
  g_AesCbc_Decode,
    AesCbc_Decode,
    AesCbc_Decode_HW,
    AesCbc_Decode_HW_256)

Z7_COM7F_IMF(CAesCoder::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *coderProps, UInt32 numProps))
{
  UInt32 algo = 0;
  for (UInt32 i = 0; i < numProps; i++)
  {
    if (propIDs[i] == NCoderPropID::kDefaultProp)
    {
      const PROPVARIANT &prop = coderProps[i];
      if (prop.vt != VT_UI4)
        return E_INVALIDARG;
      if (prop.ulVal > 3)
        return E_NOTIMPL;
      algo = prop.ulVal;
    }
  }
  if (!SetFunctions(algo))
    return E_NOTIMPL;
  return S_OK;
}

#endif // Z7_EXTRACT_ONLY

}
