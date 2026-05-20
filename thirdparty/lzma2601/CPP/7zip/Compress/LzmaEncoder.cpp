// LzmaEncoder.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "../Common/CWrappers.h"
#include "../Common/StreamUtils.h"

#include "LzmaEncoder.h"

// #define LOG_LZMA_THREADS

#ifdef LOG_LZMA_THREADS

#include <stdio.h>

#include "../../Common/IntToString.h"
#include "../../Windows/TimeUtils.h"

EXTERN_C_BEGIN
void LzmaEnc_GetLzThreads(CLzmaEncHandle pp, HANDLE lz_threads[2]);
EXTERN_C_END

#endif

namespace NCompress {
namespace NLzma {

CEncoder::CEncoder()
{
  _encoder = NULL;
  _encoder = LzmaEnc_Create(&g_AlignedAlloc);
  if (!_encoder)
    throw 1;
}

CEncoder::~CEncoder()
{
  if (_encoder)
    LzmaEnc_Destroy(_encoder, &g_AlignedAlloc, &g_BigAlloc);
}

static inline wchar_t GetLowCharFast(wchar_t c)
{
  return c |= 0x20;
}

static int ParseMatchFinder(const wchar_t *s, int *btMode, int *numHashBytes)
{
  const wchar_t c = GetLowCharFast(*s++);
  if (c == 'h')
  {
    if (GetLowCharFast(*s++) != 'c')
      return 0;
    const int num = (int)(*s++ - L'0');
    if (num < 4 || num > 5)
      return 0;
    if (*s != 0)
      return 0;
    *btMode = 0;
    *numHashBytes = num;
    return 1;
  }

  if (c != 'b')
    return 0;
  {
    if (GetLowCharFast(*s++) != 't')
      return 0;
    const int num = (int)(*s++ - L'0');
    if (num < 2 || num > 5)
      return 0;
    if (*s != 0)
      return 0;
    *btMode = 1;
    *numHashBytes = num;
    return 1;
  }
}

#define SET_PROP_32(_id_, _dest_) case NCoderPropID::_id_: ep._dest_ = (int)v; break;
#define SET_PROP_32U(_id_, _dest_) case NCoderPropID::_id_: ep._dest_ = v; break;

HRESULT SetLzmaProp(PROPID propID, const PROPVARIANT &prop, CLzmaEncProps &ep);
HRESULT SetLzmaProp(PROPID propID, const PROPVARIANT &prop, CLzmaEncProps &ep)
{
  if (propID == NCoderPropID::kMatchFinder)
  {
    if (prop.vt != VT_BSTR)
      return E_INVALIDARG;
    return ParseMatchFinder(prop.bstrVal, &ep.btMode, &ep.numHashBytes) ? S_OK : E_INVALIDARG;
  }

  if (propID == NCoderPropID::kAffinity)
  {
    if (prop.vt == VT_UI8)
      ep.affinity = prop.uhVal.QuadPart;
    else
      return E_INVALIDARG;
    return S_OK;
  }

  if (propID == NCoderPropID::kAffinityInGroup)
  {
    if (prop.vt == VT_UI8)
      ep.affinityInGroup = prop.uhVal.QuadPart;
    else
      return E_INVALIDARG;
    return S_OK;
  }

  if (propID == NCoderPropID::kThreadGroup)
  {
    if (prop.vt == VT_UI4)
      ep.affinityGroup = (Int32)(UInt32)prop.ulVal;
    else
      return E_INVALIDARG;
    return S_OK;
  }

  if (propID == NCoderPropID::kHashBits)
  {
    if (prop.vt == VT_UI4)
      ep.numHashOutBits = prop.ulVal;
    else
      return E_INVALIDARG;
    return S_OK;
  }

  if (propID > NCoderPropID::kReduceSize)
    return S_OK;
  
  if (propID == NCoderPropID::kReduceSize)
  {
    if (prop.vt == VT_UI8)
      ep.reduceSize = prop.uhVal.QuadPart;
    else
      return E_INVALIDARG;
    return S_OK;
  }

  if (propID == NCoderPropID::kDictionarySize)
  {
    if (prop.vt == VT_UI8)
    {
      // 21.03 : we support 64-bit VT_UI8 for dictionary and (dict == 4 GiB)
      const UInt64 v = prop.uhVal.QuadPart;
      if (v > ((UInt64)1 << 32))
        return E_INVALIDARG;
      UInt32 dict;
      if (v == ((UInt64)1 << 32))
        dict = (UInt32)(Int32)-1;
      else
        dict = (UInt32)v;
      ep.dictSize = dict;
      return S_OK;
    }
  }

  if (prop.vt != VT_UI4)
    return E_INVALIDARG;
  const UInt32 v = prop.ulVal;
  switch (propID)
  {
    case NCoderPropID::kDefaultProp:
      if (v > 32)
        return E_INVALIDARG;
      ep.dictSize = (v == 32) ? (UInt32)(Int32)-1 : (UInt32)1 << (unsigned)v;
      break;
    SET_PROP_32(kLevel, level)
    SET_PROP_32(kNumFastBytes, fb)
    SET_PROP_32U(kMatchFinderCycles, mc)
    SET_PROP_32(kAlgorithm, algo)
    SET_PROP_32U(kDictionarySize, dictSize)
    SET_PROP_32(kPosStateBits, pb)
    SET_PROP_32(kLitPosBits, lp)
    SET_PROP_32(kLitContextBits, lc)
    SET_PROP_32(kNumThreads, numThreads)
    default: return E_INVALIDARG;
  }
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs,
    const PROPVARIANT *coderProps, UInt32 numProps))
{
  CLzmaEncProps props;
  LzmaEncProps_Init(&props);

  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = coderProps[i];
    const PROPID propID = propIDs[i];
    switch (propID)
    {
      case NCoderPropID::kEndMarker:
        if (prop.vt != VT_BOOL)
          return E_INVALIDARG;
        props.writeEndMark = (prop.boolVal != VARIANT_FALSE);
        break;
      default:
        RINOK(SetLzmaProp(propID, prop, props))
    }
  }
  return SResToHRESULT(LzmaEnc_SetProps(_encoder, &props));
}


Z7_COM7F_IMF(CEncoder::SetCoderPropertiesOpt(const PROPID *propIDs,
    const PROPVARIANT *coderProps, UInt32 numProps))
{
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = coderProps[i];
    const PROPID propID = propIDs[i];
    if (propID == NCoderPropID::kExpectedDataSize)
      if (prop.vt == VT_UI8)
        LzmaEnc_SetDataSize(_encoder, prop.uhVal.QuadPart);
  }
  return S_OK;
}


Z7_COM7F_IMF(CEncoder::WriteCoderProperties(ISequentialOutStream *outStream))
{
  Byte props[LZMA_PROPS_SIZE];
  SizeT size = LZMA_PROPS_SIZE;
  RINOK(LzmaEnc_WriteProperties(_encoder, props, &size))
  return WriteStream(outStream, props, size);
}


#define RET_IF_WRAP_ERROR(wrapRes, sRes, sResErrorCode) \
  if (wrapRes != S_OK /* && (sRes == SZ_OK || sRes == sResErrorCode) */) return wrapRes;



#ifdef LOG_LZMA_THREADS

static inline UInt64 GetTime64(const FILETIME &t) { return ((UInt64)t.dwHighDateTime << 32) | t.dwLowDateTime; }

static void PrintNum(UInt64 val, unsigned numDigits, char c = ' ')
{
  char temp[64];
  char *p = temp + 32;
  ConvertUInt64ToString(val, p);
  unsigned len = (unsigned)strlen(p);
  for (; len < numDigits; len++)
    *--p = c;
  printf("%s", p);
}

static void PrintTime(const char *s, UInt64 val, UInt64 total)
{
  printf("  %s :", s);
  const UInt32 kFreq = 10000000;
  UInt64 sec = val / kFreq;
  PrintNum(sec, 6);
  printf(" .");
  UInt32 ms = (UInt32)(val - (sec * kFreq)) / (kFreq / 1000);
  PrintNum(ms, 3, '0');
  
  while (val > ((UInt64)1 << 56))
  {
    val >>= 1;
    total >>= 1;
  }

  UInt64 percent = 0;
  if (total != 0)
    percent = val * 100 / total;
  printf("  =");
  PrintNum(percent, 4);
  printf("%%");
}


struct CBaseStat
{
  UInt64 kernelTime, userTime;
  
  BOOL Get(HANDLE thread, const CBaseStat *prevStat)
  {
    FILETIME creationTimeFT, exitTimeFT, kernelTimeFT, userTimeFT;
    BOOL res = GetThreadTimes(thread
      , &creationTimeFT, &exitTimeFT, &kernelTimeFT, &userTimeFT);
    if (res)
    {
      kernelTime = GetTime64(kernelTimeFT);
      userTime = GetTime64(userTimeFT);
      if (prevStat)
      {
        kernelTime -= prevStat->kernelTime;
        userTime -= prevStat->userTime;
      }
    }
    return res;
  }
};


static void PrintStat(HANDLE thread, UInt64 totalTime, const CBaseStat *prevStat)
{
  CBaseStat newStat;
  if (!newStat.Get(thread, prevStat))
    return;

  PrintTime("K", newStat.kernelTime, totalTime);

  const UInt64 processTime = newStat.kernelTime + newStat.userTime;
  
  PrintTime("U", newStat.userTime, totalTime);
  PrintTime("S", processTime, totalTime);
  printf("\n");
  // PrintTime("G ", totalTime, totalTime);
}

#endif



Z7_COM7F_IMF(CEncoder::Code(ISequentialInStream *inStream, ISequentialOutStream *outStream,
    const UInt64 * /* inSize */, const UInt64 * /* outSize */, ICompressProgressInfo *progress))
{
  CSeqInStreamWrap inWrap;
  CSeqOutStreamWrap outWrap;
  CCompressProgressWrap progressWrap;

  inWrap.Init(inStream);
  outWrap.Init(outStream);
  progressWrap.Init(progress);

  #ifdef LOG_LZMA_THREADS

  FILETIME startTimeFT;
  NWindows::NTime::GetCurUtcFileTime(startTimeFT);
  UInt64 totalTime = GetTime64(startTimeFT);
  CBaseStat oldStat;
  if (!oldStat.Get(GetCurrentThread(), NULL))
    return E_FAIL;
  
  #endif
  
  
  SRes res = LzmaEnc_Encode(_encoder, &outWrap.vt, &inWrap.vt,
      progress ? &progressWrap.vt : NULL, &g_AlignedAlloc, &g_BigAlloc);

  _inputProcessed = inWrap.Processed;

  RET_IF_WRAP_ERROR(inWrap.Res, res, SZ_ERROR_READ)
  RET_IF_WRAP_ERROR(outWrap.Res, res, SZ_ERROR_WRITE)
  RET_IF_WRAP_ERROR(progressWrap.Res, res, SZ_ERROR_PROGRESS)

  
  #ifdef LOG_LZMA_THREADS
  
  NWindows::NTime::GetCurUtcFileTime(startTimeFT);
  totalTime = GetTime64(startTimeFT) - totalTime;
  HANDLE lz_threads[2];
  LzmaEnc_GetLzThreads(_encoder, lz_threads);
  printf("\n");
  printf("Main: ");  PrintStat(GetCurrentThread(), totalTime, &oldStat);
  printf("Hash: ");  PrintStat(lz_threads[0], totalTime, NULL);
  printf("BinT: ");  PrintStat(lz_threads[1], totalTime, NULL);
  // PrintTime("Total: ", totalTime, totalTime);
  printf("\n");

  #endif

  return SResToHRESULT(res);
}

}}
