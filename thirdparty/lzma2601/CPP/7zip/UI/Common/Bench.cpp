// Bench.cpp

#include "StdAfx.h"

// #include <stdio.h>

#ifndef _WIN32
#define USE_POSIX_TIME
#define USE_POSIX_TIME2
#endif // _WIN32

#ifdef USE_POSIX_TIME
#include <time.h>
#include <unistd.h>
#ifdef USE_POSIX_TIME2
#include <sys/time.h>
#include <sys/times.h>
#endif
#endif // USE_POSIX_TIME

#ifdef _WIN32
#define USE_ALLOCA
#endif

#ifdef USE_ALLOCA
#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#define BENCH_ALLOCA_VALUE(index) (((index) * 64 * 21) & 0x7FF)
#endif

#include "../../../../C/7zCrc.h"
#include "../../../../C/RotateDefs.h"
#include "../../../../C/CpuArch.h"

#ifndef Z7_ST
#include "../../../Windows/Synchronization.h"
#include "../../../Windows/Thread.h"
#endif

#include "../../../Windows/FileFind.h"
#include "../../../Windows/FileIO.h"
#include "../../../Windows/SystemInfo.h"

#include "../../../Common/MyBuffer2.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/StringToInt.h"
#include "../../../Common/Wildcard.h"

#include "../../Common/MethodProps.h"
#include "../../Common/StreamObjects.h"
#include "../../Common/StreamUtils.h"

#include "Bench.h"

using namespace NWindows;

#ifndef Z7_ST
static const UInt32 k_LZMA = 0x030101;
#endif

static const UInt64 kComplexInCommands = (UInt64)1 <<
  #ifdef UNDER_CE
    31;
  #else
    34;
  #endif

static const UInt32 kComplexInMs = 4000;

static void SetComplexCommandsMs(UInt32 complexInMs,
    bool isSpecifiedFreq, UInt64 cpuFreq, UInt64 &complexInCommands)
{
  complexInCommands = kComplexInCommands;
  const UInt64 kMinFreq = (UInt64)1000000 * 4;
  const UInt64 kMaxFreq = (UInt64)1000000 * 20000;
  if (cpuFreq < kMinFreq && !isSpecifiedFreq)
    cpuFreq = kMinFreq;
  if (cpuFreq < kMaxFreq || isSpecifiedFreq)
  {
    if (complexInMs != 0)
      complexInCommands = complexInMs * cpuFreq / 1000;
    else
      complexInCommands = cpuFreq >> 2;
  }
}

// const UInt64 kBenchmarkUsageMult = 1000000; // for debug
static const unsigned kBenchmarkUsageMultBits = 16;
static const UInt64 kBenchmarkUsageMult = 1 << kBenchmarkUsageMultBits;

UInt64 Benchmark_GetUsage_Percents(UInt64 usage)
{
  return (100 * usage + kBenchmarkUsageMult / 2) / kBenchmarkUsageMult;
}

static const unsigned kNumHashDictBits = 17;
static const UInt32 kFilterUnpackSize = (47 << 10); // + 5; // for test

static const unsigned kOldLzmaDictBits = 32;

// static const size_t kAdditionalSize = (size_t)1 << 32; // for debug
static const size_t kAdditionalSize = (size_t)1 << 16;
static const size_t kCompressedAdditionalSize = 1 << 10;

static const UInt32 kMaxMethodPropSize = 1 << 6;


#define ALLOC_WITH_HRESULT(_buffer_, _size_) \
  { (_buffer_)->Alloc(_size_); \
  if (_size_ && !(_buffer_)->IsAllocated()) return E_OUTOFMEMORY; }


class CBaseRandomGenerator
{
  UInt32 A1;
  UInt32 A2;
  UInt32 Salt;
public:
  CBaseRandomGenerator(UInt32 salt = 0): Salt(salt) { Init(); }
  void Init() { A1 = 362436069; A2 = 521288629;}
  Z7_FORCE_INLINE
  UInt32 GetRnd()
  {
#if 0
    // for debug:
    return 0x0c080400;
    // return 0;
#else
    return Salt ^
    (
      ((A1 = 36969 * (A1 & 0xffff) + (A1 >> 16)) << 16) +
      ((A2 = 18000 * (A2 & 0xffff) + (A2 >> 16)) )
    );
#endif
  }
};


static const size_t k_RandBuf_AlignMask = 4 - 1;

Z7_NO_INLINE
static void RandGen_BufAfterPad(Byte *buf, size_t size)
{
  CBaseRandomGenerator RG;
  for (size_t i = 0; i < size; i += 4)
  {
    const UInt32 v = RG.GetRnd();
    SetUi32a(buf + i, v)
  }
  /*
  UInt32 v = RG.GetRnd();
  for (; i < size; i++)
  {
    buf[i] = (Byte)v;
    v >>= 8;
  }
  */
}


class CBenchRandomGenerator: public CMidAlignedBuffer
{
  static UInt32 GetVal(UInt32 &res, unsigned numBits)
  {
    const UInt32 val = res & (((UInt32)1 << numBits) - 1);
    res >>= numBits;
    return val;
  }
  
  static UInt32 GetLen(UInt32 &r)
  {
    const unsigned len = (unsigned)GetVal(r, 2);
    return GetVal(r, 1 + len);
  }

public:
  
  void GenerateSimpleRandom(UInt32 salt)
  {
    CBaseRandomGenerator rg(salt);
    const size_t bufSize = Size();
    Byte *buf = (Byte *)*this;
    for (size_t i = 0; i < bufSize; i++)
      buf[i] = (Byte)rg.GetRnd();
  }

  void GenerateLz(unsigned dictBits, UInt32 salt)
  {
    CBaseRandomGenerator rg(salt);
    size_t pos = 0;
    size_t rep0 = 1;
    const size_t bufSize = Size();
    Byte *buf = (Byte *)*this;
    unsigned posBits = 1;

    // printf("\n dictBits = %d\n", (UInt32)dictBits);
    // printf("\n bufSize = 0x%p\n", (const void *)bufSize);
    
    while (pos < bufSize)
    {
      /*
      if (pos >= ((UInt32)1 << 31))
        printf(" %x\n", pos);
      */
      UInt32 r = rg.GetRnd();
      if (GetVal(r, 1) == 0 || pos < 1024)
        buf[pos++] = (Byte)(r & 0xFF);
      else
      {
        UInt32 len;
        len = 1 + GetLen(r);
        
        if (GetVal(r, 3) != 0)
        {
          len += GetLen(r);

          while (((size_t)1 << posBits) < pos)
            posBits++;

          unsigned numBitsMax = dictBits;
          if (numBitsMax > posBits)
            numBitsMax = posBits;

          const unsigned kAddBits = 6;
          unsigned numLogBits = 5;
          if (numBitsMax <= (1 << 4) - 1 + kAddBits)
            numLogBits = 4;

          for (;;)
          {
            const UInt32 ppp = GetVal(r, numLogBits) + kAddBits;
            r = rg.GetRnd();
            if (ppp > numBitsMax)
              continue;
            // rep0 = GetVal(r, ppp);
            rep0 = r & (((size_t)1 << ppp) - 1);
            if (rep0 < pos)
              break;
            r = rg.GetRnd();
          }
          rep0++;
        }

        // len *= 300; // for debug
        {
          const size_t rem = bufSize - pos;
          if (len > rem)
            len = (UInt32)rem;
        }
        Byte *dest = buf + pos;
        const Byte *src = dest - rep0;
        pos += len;
        for (UInt32 i = 0; i < len; i++)
          *dest++ = *src++;
      }
    }
    // printf("\n CRC = %x\n", CrcCalc(buf, bufSize));
  }
};


Z7_CLASS_IMP_NOQIB_1(
  CBenchmarkInStream
  , ISequentialInStream
)
  const Byte *Data;
  size_t Pos;
  size_t Size;
public:
  void Init(const Byte *data, size_t size)
  {
    Data = data;
    Size = size;
    Pos = 0;
  }
  bool WasFinished() const { return Pos == Size; }
};

Z7_COM7F_IMF(CBenchmarkInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  const UInt32 kMaxBlockSize = (1 << 20);
  if (size > kMaxBlockSize)
    size = kMaxBlockSize;
  const size_t remain = Size - Pos;
  if (size > remain)
    size = (UInt32)remain;

  if (size)
    memcpy(data, Data + Pos, size);

  Pos += size;
  if (processedSize)
    *processedSize = size;
  return S_OK;
}
  

class CBenchmarkOutStream Z7_final:
  public ISequentialOutStream,
  public CMyUnknownImp,
  public CMidAlignedBuffer
{
  Z7_COM_UNKNOWN_IMP_0
  Z7_IFACE_COM7_IMP(ISequentialOutStream)
  // bool _overflow;
public:
  size_t Pos;
  bool RealCopy;
  bool CalcCrc;
  UInt32 Crc;

  // CBenchmarkOutStream(): _overflow(false) {}
  void Init(bool realCopy, bool calcCrc)
  {
    Crc = CRC_INIT_VAL;
    RealCopy = realCopy;
    CalcCrc = calcCrc;
    // _overflow = false;
    Pos = 0;
  }

  void InitCrc()
  {
    Crc = CRC_INIT_VAL;
  }

  void Calc(const void *data, size_t size)
  {
    Crc = CrcUpdate(Crc, data, size);
  }

  size_t GetPos() const { return Pos; }

  // void Print() { printf("\n%8d %8d\n", (unsigned)BufferSize, (unsigned)Pos); }
};

Z7_COM7F_IMF(CBenchmarkOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  size_t curSize = Size() - Pos;
  if (curSize > size)
    curSize = size;
  if (curSize != 0)
  {
    if (RealCopy)
      memcpy(((Byte *)*this) + Pos, data, curSize);
    if (CalcCrc)
      Calc(data, curSize);
    Pos += curSize;
  }
  if (processedSize)
    *processedSize = (UInt32)curSize;
  if (curSize != size)
  {
    // _overflow = true;
    return E_FAIL;
  }
  return S_OK;
}
  

Z7_CLASS_IMP_NOQIB_1(
  CCrcOutStream
  , ISequentialOutStream
)
public:
  bool CalcCrc;
  UInt32 Crc;
  UInt64 Pos;
  
  CCrcOutStream(): CalcCrc(true) {}
  void Init() { Crc = CRC_INIT_VAL; Pos = 0; }
  void Calc(const void *data, size_t size)
  {
    Crc = CrcUpdate(Crc, data, size);
  }
};

Z7_COM7F_IMF(CCrcOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  if (CalcCrc)
    Calc(data, size);
  Pos += size;
  if (processedSize)
    *processedSize = size;
  return S_OK;
}
  
// #include "../../../../C/My_sys_time.h"

static UInt64 GetTimeCount()
{
  #ifdef USE_POSIX_TIME
  #ifdef USE_POSIX_TIME2
  timeval v;
  if (gettimeofday(&v, NULL) == 0)
    return (UInt64)(v.tv_sec) * 1000000 + (UInt64)v.tv_usec;
  return (UInt64)time(NULL) * 1000000;
  #else
  return time(NULL);
  #endif
  #else
  LARGE_INTEGER value;
  if (::QueryPerformanceCounter(&value))
    return (UInt64)value.QuadPart;
  return GetTickCount();
  #endif
}

static UInt64 GetFreq()
{
  #ifdef USE_POSIX_TIME
  #ifdef USE_POSIX_TIME2
  return 1000000;
  #else
  return 1;
  #endif
  #else
  LARGE_INTEGER value;
  if (::QueryPerformanceFrequency(&value))
    return (UInt64)value.QuadPart;
  return 1000;
  #endif
}


#ifdef USE_POSIX_TIME

struct CUserTime
{
  UInt64 Sum;
  clock_t Prev;
  
  void Init()
  {
    // Prev = clock();
    Sum = 0;
    Prev = 0;
    Update();
    Sum = 0;
  }

  void Update()
  {
    tms t;
    /* clock_t res = */ times(&t);
    clock_t newVal = t.tms_utime + t.tms_stime;
    Sum += (UInt64)(newVal - Prev);
    Prev = newVal;

    /*
    clock_t v = clock();
    if (v != -1)
    {
      Sum += v - Prev;
      Prev = v;
    }
    */
  }
  UInt64 GetUserTime()
  {
    Update();
    return Sum;
  }
};

#else


struct CUserTime
{
  bool UseTick;
  DWORD Prev_Tick;
  UInt64 Prev;
  UInt64 Sum;

  void Init()
  {
    UseTick = false;
    Prev_Tick = 0;
    Prev = 0;
    Sum = 0;
    Update();
    Sum = 0;
  }
  UInt64 GetUserTime()
  {
    Update();
    return Sum;
  }
  void Update();
};

static inline UInt64 GetTime64(const FILETIME &t) { return ((UInt64)t.dwHighDateTime << 32) | t.dwLowDateTime; }

void CUserTime::Update()
{
  DWORD new_Tick = GetTickCount();
  FILETIME creationTime, exitTime, kernelTime, userTime;
  if (!UseTick &&
      #ifdef UNDER_CE
        ::GetThreadTimes(::GetCurrentThread()
      #else
        ::GetProcessTimes(::GetCurrentProcess()
      #endif
      , &creationTime, &exitTime, &kernelTime, &userTime))
  {
    UInt64 newVal = GetTime64(userTime) + GetTime64(kernelTime);
    Sum += newVal - Prev;
    Prev = newVal;
  }
  else
  {
    UseTick = true;
    Sum += (UInt64)(new_Tick - (DWORD)Prev_Tick) * 10000;
  }
  Prev_Tick = new_Tick;
}


#endif

static UInt64 GetUserFreq()
{
  #ifdef USE_POSIX_TIME
  // return CLOCKS_PER_SEC;
  return (UInt64)sysconf(_SC_CLK_TCK);
  #else
  return 10000000;
  #endif
}

class CBenchProgressStatus Z7_final
{
  #ifndef Z7_ST
  NSynchronization::CCriticalSection CS;
  #endif
public:
  HRESULT Res;
  bool EncodeMode;
  void SetResult(HRESULT res)
  {
    #ifndef Z7_ST
    NSynchronization::CCriticalSectionLock lock(CS);
    #endif
    Res = res;
  }
  HRESULT GetResult()
  {
    #ifndef Z7_ST
    NSynchronization::CCriticalSectionLock lock(CS);
    #endif
    return Res;
  }
};

struct CBenchInfoCalc
{
  CBenchInfo BenchInfo;
  CUserTime UserTime;

  void SetStartTime();
  void SetFinishTime(CBenchInfo &dest);
};

void CBenchInfoCalc::SetStartTime()
{
  BenchInfo.GlobalFreq = GetFreq();
  BenchInfo.UserFreq = GetUserFreq();
  BenchInfo.GlobalTime = ::GetTimeCount();
  BenchInfo.UserTime = 0;
  UserTime.Init();
}

void CBenchInfoCalc::SetFinishTime(CBenchInfo &dest)
{
  dest = BenchInfo;
  dest.GlobalTime = ::GetTimeCount() - BenchInfo.GlobalTime;
  dest.UserTime = UserTime.GetUserTime();
}

class CBenchProgressInfo Z7_final:
  public ICompressProgressInfo,
  public CMyUnknownImp,
  public CBenchInfoCalc
{
  Z7_COM_UNKNOWN_IMP_0
  Z7_IFACE_COM7_IMP(ICompressProgressInfo)
public:
  CBenchProgressStatus *Status;
  IBenchCallback *Callback;

  CBenchProgressInfo(): Callback(NULL) {}
};


Z7_COM7F_IMF(CBenchProgressInfo::SetRatioInfo(const UInt64 *inSize, const UInt64 *outSize))
{
  HRESULT res = Status->GetResult();
  if (res != S_OK)
    return res;
  if (!Callback)
    return res;

  /*
  static UInt64 inSizePrev = 0;
  static UInt64 outSizePrev = 0;
  UInt64 delta1 = 0, delta2 = 0, val1 = 0, val2 = 0;
  if (inSize)   { val1 = *inSize;  delta1 = val1 - inSizePrev;  inSizePrev  = val1; }
  if (outSize)  { val2 = *outSize; delta2 = val2 - outSizePrev; outSizePrev = val2;  }
  UInt64 percents = delta2 * 1000;
  if (delta1 != 0)
    percents /= delta1;
  printf("=== %7d %7d     %7d %7d  ratio = %4d\n",
      (unsigned)(val1 >> 10), (unsigned)(delta1 >> 10),
      (unsigned)(val2 >> 10), (unsigned)(delta2 >> 10),
      (unsigned)percents);
  */

  CBenchInfo info;
  SetFinishTime(info);
  if (Status->EncodeMode)
  {
    info.UnpackSize = BenchInfo.UnpackSize + *inSize;
    info.PackSize = BenchInfo.PackSize + *outSize;
    res = Callback->SetEncodeResult(info, false);
  }
  else
  {
    info.PackSize = BenchInfo.PackSize + *inSize;
    info.UnpackSize = BenchInfo.UnpackSize + *outSize;
    res = Callback->SetDecodeResult(info, false);
  }
  if (res != S_OK)
    Status->SetResult(res);
  return res;
}

static const unsigned kSubBits = 8;

static unsigned GetLogSize(UInt64 size)
{
  unsigned i = 0;
  for (;;)
  {
    i++;  size >>= 1;  if (size == 0) break;
  }
  return i;
}


static UInt32 GetLogSize_Sub(UInt64 size)
{
  if (size <= 1)
    return 0;
  const unsigned i = GetLogSize(size) - 1;
  UInt32 v;
  if (i <= kSubBits)
    v = (UInt32)(size) << (kSubBits - i);
  else
    v = (UInt32)(size >> (i - kSubBits));
  return ((UInt32)i << kSubBits) + (v & (((UInt32)1 << kSubBits) - 1));
}


static UInt64 Get_UInt64_from_double(double v)
{
  const UInt64 kMaxVal = (UInt64)1 << 62;
  if (v > (double)(Int64)kMaxVal)
    return kMaxVal;
  return (UInt64)v;
}

static UInt64 MyMultDiv64(UInt64 m1, UInt64 m2, UInt64 d)
{
  if (d == 0)
    d = 1;
  const double v =
      (double)(Int64)m1 *
      (double)(Int64)m2 /
      (double)(Int64)d;
  return Get_UInt64_from_double(v);
  /*
  unsigned n1 = GetLogSize(m1);
  unsigned n2 = GetLogSize(m2);
  while (n1 + n2 > 64)
  {
    if (n1 >= n2)
    {
      m1 >>= 1;
      n1--;
    }
    else
    {
      m2 >>= 1;
      n2--;
    }
    d >>= 1;
  }

  if (d == 0)
    d = 1;
  return m1 * m2 / d;
  */
}


UInt64 CBenchInfo::GetUsage() const
{
  UInt64 userTime = UserTime;
  UInt64 userFreq = UserFreq;
  UInt64 globalTime = GlobalTime;
  UInt64 globalFreq = GlobalFreq;

  if (userFreq == 0)
    userFreq = 1;
  if (globalTime == 0)
    globalTime = 1;

  const double v =
        ((double)(Int64)userTime / (double)(Int64)userFreq)
      * ((double)(Int64)globalFreq / (double)(Int64)globalTime)
      * (double)(Int64)kBenchmarkUsageMult;
  return Get_UInt64_from_double(v);
  /*
  return MyMultDiv64(
        MyMultDiv64(kBenchmarkUsageMult, userTime, userFreq),
        globalFreq, globalTime);
  */
}


UInt64 CBenchInfo::GetRatingPerUsage(UInt64 rating) const
{
  if (UserTime == 0)
  {
    return 0;
    // userTime = 1;
  }
  UInt64 globalFreq = GlobalFreq;
  if (globalFreq == 0)
    globalFreq = 1;

  const double v =
        ((double)(Int64)GlobalTime / (double)(Int64)globalFreq)
      * ((double)(Int64)UserFreq  / (double)(Int64)UserTime)
      * (double)(Int64)rating;
  return Get_UInt64_from_double(v);
  /*
  return MyMultDiv64(
        MyMultDiv64(rating, UserFreq, UserTime),
        GlobalTime, globalFreq);
  */
}


UInt64 CBenchInfo::GetSpeed(UInt64 numUnits) const
{
  return MyMultDiv64(numUnits, GlobalFreq, GlobalTime);
}

static UInt64 GetNumCommands_from_Size_and_Complexity(UInt64 size, Int32 complexity)
{
  return complexity >= 0 ?
      size * (UInt32)complexity :
      size / (UInt32)(-complexity);
}

struct CBenchProps
{
  bool LzmaRatingMode;
  
  Int32 EncComplex;
  Int32 DecComplexCompr;
  Int32 DecComplexUnc;

  unsigned KeySize;

  CBenchProps():
      LzmaRatingMode(false),
      KeySize(0)
    {}

  void SetLzmaCompexity();

  UInt64 GetNumCommands_Enc(UInt64 unpackSize) const
  {
    const UInt32 kMinSize = 100;
    if (unpackSize < kMinSize)
      unpackSize = kMinSize;
    return GetNumCommands_from_Size_and_Complexity(unpackSize, EncComplex);
  }

  UInt64 GetNumCommands_Dec(UInt64 packSize, UInt64 unpackSize) const
  {
    return
        GetNumCommands_from_Size_and_Complexity(packSize, DecComplexCompr) +
        GetNumCommands_from_Size_and_Complexity(unpackSize, DecComplexUnc);
  }

  UInt64 GetRating_Enc(UInt64 dictSize, UInt64 elapsedTime, UInt64 freq, UInt64 size) const;
  UInt64 GetRating_Dec(UInt64 elapsedTime, UInt64 freq, UInt64 outSize, UInt64 inSize, UInt64 numIterations) const;
};

void CBenchProps::SetLzmaCompexity()
{
  EncComplex = 1200;
  DecComplexUnc = 4;
  DecComplexCompr = 190;
  LzmaRatingMode = true;
}

UInt64 CBenchProps::GetRating_Enc(UInt64 dictSize, UInt64 elapsedTime, UInt64 freq, UInt64 size) const
{
  if (dictSize < (1 << kBenchMinDicLogSize))
    dictSize = (1 << kBenchMinDicLogSize);
  Int32 encComplex = EncComplex;
  if (LzmaRatingMode)
  {
    /*
    for (UInt64 uu = 0; uu < (UInt64)0xf << 60;)
    {
      unsigned rr = GetLogSize_Sub(uu);
      printf("\n%16I64x , log = %4x", uu, rr);
      uu += 1;
      uu += uu / 50;
    }
    */
    // throw 1;
    const UInt32 t = GetLogSize_Sub(dictSize) - (kBenchMinDicLogSize << kSubBits);
    encComplex = 870 + ((t * t * 5) >> (2 * kSubBits));
  }
  const UInt64 numCommands = GetNumCommands_from_Size_and_Complexity(size, encComplex);
  return MyMultDiv64(numCommands, freq, elapsedTime);
}

UInt64 CBenchProps::GetRating_Dec(UInt64 elapsedTime, UInt64 freq, UInt64 outSize, UInt64 inSize, UInt64 numIterations) const
{
  const UInt64 numCommands = GetNumCommands_Dec(inSize, outSize) * numIterations;
  return MyMultDiv64(numCommands, freq, elapsedTime);
}



UInt64 CBenchInfo::GetRating_LzmaEnc(UInt64 dictSize) const
{
  CBenchProps props;
  props.SetLzmaCompexity();
  return props.GetRating_Enc(dictSize, GlobalTime, GlobalFreq, UnpackSize * NumIterations);
}

UInt64 CBenchInfo::GetRating_LzmaDec() const
{
  CBenchProps props;
  props.SetLzmaCompexity();
  return props.GetRating_Dec(GlobalTime, GlobalFreq, UnpackSize, PackSize, NumIterations);
}


#ifndef Z7_ST

#define NUM_CPU_LEVELS_MAX 3

struct CAffinityMode
{
  unsigned NumBundleThreads;
  unsigned NumLevels;
  unsigned NumCoreThreads;
  unsigned NumCores;
  // unsigned DivideNum;

#ifdef _WIN32
  unsigned NumGroups;
#endif

  UInt32 Sizes[NUM_CPU_LEVELS_MAX];

  void SetLevels(unsigned numCores, unsigned numCoreThreads);
  DWORD_PTR GetAffinityMask(UInt32 bundleIndex, CCpuSet *cpuSet) const;
  bool NeedAffinity() const { return NumBundleThreads != 0; }

#ifdef _WIN32
  bool NeedGroupsMode() const { return NumGroups > 1; }
#endif

  WRes CreateThread_WithAffinity(NWindows::CThread &thread, THREAD_FUNC_TYPE startAddress, LPVOID parameter, UInt32 bundleIndex) const
  {
#ifdef _WIN32
    if (NeedGroupsMode()) // we need fix for bundleIndex usage
      return thread.Create_With_Group(startAddress, parameter, bundleIndex % NumGroups);
#endif
    if (NeedAffinity())
    {
      CCpuSet cpuSet;
      GetAffinityMask(bundleIndex, &cpuSet);
      return thread.Create_With_CpuSet(startAddress, parameter, &cpuSet);
    }
    return thread.Create(startAddress, parameter);
  }

  CAffinityMode():
    NumBundleThreads(0),
    NumLevels(0),
    NumCoreThreads(1)
#ifdef _WIN32
    , NumGroups(0)
#endif
    // DivideNum(1)
    {}
};

void CAffinityMode::SetLevels(unsigned numCores, unsigned numCoreThreads)
{
  NumCores = numCores;
  NumCoreThreads = numCoreThreads;
  NumLevels = 0;
  if (numCoreThreads == 0 || numCores == 0 || numCores % numCoreThreads != 0)
    return;
  UInt32 c = numCores / numCoreThreads;
  UInt32 c2 = 1;
  while ((c & 1) == 0)
  {
    c >>= 1;
    c2 <<= 1;
  }
  if (c2 != 1)
    Sizes[NumLevels++] = c2;
  if (c != 1)
    Sizes[NumLevels++] = c;
  if (numCoreThreads != 1)
    Sizes[NumLevels++] = numCoreThreads;
  if (NumLevels == 0)
    Sizes[NumLevels++] = 1;

  /*
  printf("\n Cores:");
  for (unsigned i = 0; i < NumLevels; i++)
  {
    printf(" %d", Sizes[i]);
  }
  printf("\n");
  */
}


DWORD_PTR CAffinityMode::GetAffinityMask(UInt32 bundleIndex, CCpuSet *cpuSet) const
{
  CpuSet_Zero(cpuSet);

  if (NumLevels == 0)
    return 0;

  // printf("\n%2d", bundleIndex);

  /*
  UInt32 low = 0;
  if (DivideNum != 1)
  {
    low = bundleIndex % DivideNum;
    bundleIndex /= DivideNum;
  }
  */

  UInt32 numGroups = NumCores / NumBundleThreads;
  UInt32 m = bundleIndex % numGroups;
  UInt32 v = 0;
  for (unsigned i = 0; i < NumLevels; i++)
  {
    UInt32 size = Sizes[i];
    while ((size & 1) == 0)
    {
      v *= 2;
      v |= (m & 1);
      m >>= 1;
      size >>= 1;
    }
    v *= size;
    v += m % size;
    m /= size;
  }

  // UInt32 nb = NumBundleThreads / DivideNum;
  UInt32 nb = NumBundleThreads;

  DWORD_PTR mask = ((DWORD_PTR)1 << nb) - 1;
  // v += low;
  mask <<= v;

  // printf(" %2d %8x \n ", v, (unsigned)mask);
  #ifdef _WIN32
    *cpuSet = mask;
  #else
  {
    for (unsigned k = 0; k < nb; k++)
      CpuSet_Set(cpuSet, v + k);
  }
  #endif

  return mask;
}


struct CBenchSyncCommon
{
  bool ExitMode;
  NSynchronization::CManualResetEvent StartEvent;

  CBenchSyncCommon(): ExitMode(false) {}
};

#endif



enum E_CheckCrcMode
{
  k_CheckCrcMode_Never = 0,
  k_CheckCrcMode_Always = 1,
  k_CheckCrcMode_FirstPass = 2
};

class CEncoderInfo;

class CEncoderInfo Z7_final
{
  Z7_CLASS_NO_COPY(CEncoderInfo)

public:

  #ifndef Z7_ST
  NWindows::CThread thread[2];
  NSynchronization::CManualResetEvent ReadyEvent;
  UInt32 NumDecoderSubThreads;
  CBenchSyncCommon *Common;
  UInt32 EncoderIndex;
  UInt32 NumEncoderInternalThreads;
  CAffinityMode AffinityMode;
  bool IsGlobalMtMode; // if more than one benchmark encoder threads
  #endif

  CMyComPtr<ICompressCoder> _encoder;
  CMyComPtr<ICompressFilter> _encoderFilter;
  CBenchProgressInfo *progressInfoSpec[2];
  CMyComPtr<ICompressProgressInfo> progressInfo[2];
  UInt64 NumIterations;

  UInt32 Salt;

  #ifdef USE_ALLOCA
  size_t AllocaSize;
  #endif

  unsigned KeySize;
  Byte _key[32];
  Byte _iv[16];
  
  HRESULT Set_Key_and_IV(ICryptoProperties *cp)
  {
    RINOK(cp->SetKey(_key, KeySize))
    return cp->SetInitVector(_iv, sizeof(_iv));
  }

  Byte _psw[16];
  
  bool CheckCrc_Enc;    /* = 1, if we want to check packed data crcs after each pass
                                used for filter and usual coders */
  bool UseRealData_Enc; /* = 1, if we want to use only original data for each pass
                                used only for filter */
  E_CheckCrcMode CheckCrcMode_Dec;

  struct CDecoderInfo
  {
    CEncoderInfo *Encoder;
    UInt32 DecoderIndex;
    bool CallbackMode;
    
    #ifdef USE_ALLOCA
    size_t AllocaSize;
    #endif
  };
  CDecoderInfo decodersInfo[2];

  CMyComPtr<ICompressCoder> _decoders[2];
  CMyComPtr<ICompressFilter> _decoderFilter;

  HRESULT Results[2];
  CBenchmarkOutStream *outStreamSpec;
  CMyComPtr<ISequentialOutStream> outStream;
  IBenchCallback *callback;
  IBenchPrintCallback *printCallback;
  UInt32 crc;
  size_t kBufferSize;
  size_t compressedSize;
  const Byte *uncompressedDataPtr;

  const Byte *fileData;
  CBenchRandomGenerator rg;

  CMidAlignedBuffer rgCopy; // it must be 16-byte aligned !!!
  
  // CBenchmarkOutStream *propStreamSpec;
  Byte propsData[kMaxMethodPropSize];
  CBufPtrSeqOutStream *propStreamSpec;
  CMyComPtr<ISequentialOutStream> propStream;

  unsigned generateDictBits;
  COneMethodInfo _method;
  
  // for decode
  size_t _uncompressedDataSize;

  HRESULT Generate();
  HRESULT Encode();
  HRESULT Decode(UInt32 decoderIndex);

  CEncoderInfo():
    #ifndef Z7_ST
    Common(NULL),
    IsGlobalMtMode(true),
    #endif
    Salt(0),
    KeySize(0),
    CheckCrc_Enc(true),
    UseRealData_Enc(true),
    CheckCrcMode_Dec(k_CheckCrcMode_Always),
    outStreamSpec(NULL),
    callback(NULL),
    printCallback(NULL),
    fileData(NULL),
    propStreamSpec(NULL)
    {}

  #ifndef Z7_ST
  
  static THREAD_FUNC_DECL EncodeThreadFunction(void *param)
  {
    HRESULT res;
    CEncoderInfo *encoder = (CEncoderInfo *)param;
    try
    {
      #ifdef USE_ALLOCA
      alloca(encoder->AllocaSize);
      #endif

      res = encoder->Encode();
    }
    catch(...)
    {
      res = E_FAIL;
    }
    encoder->Results[0] = res;
    if (res != S_OK)
      encoder->progressInfoSpec[0]->Status->SetResult(res);
    encoder->ReadyEvent.Set();
    return THREAD_FUNC_RET_ZERO;
  }
  
  static THREAD_FUNC_DECL DecodeThreadFunction(void *param)
  {
    CDecoderInfo *decoder = (CDecoderInfo *)param;
    
    #ifdef USE_ALLOCA
    alloca(decoder->AllocaSize);
    // printf("\nalloca=%d\n", (unsigned)decoder->AllocaSize);
    #endif
    
    CEncoderInfo *encoder = decoder->Encoder;
    encoder->Results[decoder->DecoderIndex] = encoder->Decode(decoder->DecoderIndex);
    return THREAD_FUNC_RET_ZERO;
  }

  HRESULT CreateEncoderThread()
  {
    WRes res = 0;
    if (!ReadyEvent.IsCreated())
      res = ReadyEvent.Create();
    if (res == 0)
      res = AffinityMode.CreateThread_WithAffinity(thread[0], EncodeThreadFunction, this,
          EncoderIndex);
    return HRESULT_FROM_WIN32(res);
  }

  HRESULT CreateDecoderThread(unsigned index, bool callbackMode
      #ifdef USE_ALLOCA
      , size_t allocaSize
      #endif
      )
  {
    CDecoderInfo &decoder = decodersInfo[index];
    decoder.DecoderIndex = index;
    decoder.Encoder = this;
    
    #ifdef USE_ALLOCA
    decoder.AllocaSize = allocaSize;
    #endif
    
    decoder.CallbackMode = callbackMode;

    WRes res = AffinityMode.CreateThread_WithAffinity(thread[index], DecodeThreadFunction, &decoder,
        // EncoderIndex * NumEncoderInternalThreads + index
        EncoderIndex
        );

    return HRESULT_FROM_WIN32(res);
  }
  
  #endif
};




static size_t GetBenchCompressedSize(size_t bufferSize)
{
  return kCompressedAdditionalSize + bufferSize + bufferSize / 16;
  // kBufferSize / 2;
}


HRESULT CEncoderInfo::Generate()
{
  const COneMethodInfo &method = _method;

  // we need extra space, if input data is already compressed
  const size_t kCompressedBufferSize = _encoderFilter ?
      kBufferSize :
      GetBenchCompressedSize(kBufferSize);

  if (kCompressedBufferSize < kBufferSize)
    return E_FAIL;

  uncompressedDataPtr = fileData;
  if (fileData)
  {
    #if !defined(Z7_ST)
    if (IsGlobalMtMode)
    {
      /* we copy the data to local buffer of thread to eliminate
         using of shared buffer by different threads */
      ALLOC_WITH_HRESULT(&rg, kBufferSize)
      memcpy((Byte *)rg, fileData, kBufferSize);
      uncompressedDataPtr = (const Byte *)rg;
    }
    #endif
  }
  else
  {
    ALLOC_WITH_HRESULT(&rg, kBufferSize)
    // DWORD ttt = GetTickCount();
    if (generateDictBits == 0)
      rg.GenerateSimpleRandom(Salt);
    else
    {
      if (generateDictBits >= sizeof(size_t) * 8
          && kBufferSize > ((size_t)1 << (sizeof(size_t) * 8 - 1)))
        return E_INVALIDARG;
      rg.GenerateLz(generateDictBits, Salt);
      // return E_ABORT; // for debug
    }
    // printf("\n%d\n            ", GetTickCount() - ttt);

    crc = CrcCalc((const Byte *)rg, rg.Size());
    uncompressedDataPtr = (const Byte *)rg;
  }

  if (!outStream)
  {
    outStreamSpec = new CBenchmarkOutStream;
    outStream = outStreamSpec;
  }

  ALLOC_WITH_HRESULT(outStreamSpec, kCompressedBufferSize)

  if (_encoderFilter)
  {
    /* we try to reduce the number of memcpy() in main encoding loop.
       so we copy data to temp buffers here */
    ALLOC_WITH_HRESULT(&rgCopy, kBufferSize)
    memcpy((Byte *)*outStreamSpec, uncompressedDataPtr, kBufferSize);
    memcpy((Byte *)rgCopy, uncompressedDataPtr, kBufferSize);
  }

  if (!propStream)
  {
    propStreamSpec = new CBufPtrSeqOutStream; // CBenchmarkOutStream;
    propStream = propStreamSpec;
  }
  // ALLOC_WITH_HRESULT_2(propStreamSpec, kMaxMethodPropSize);
  // propStreamSpec->Init(true, false);
  propStreamSpec->Init(propsData, sizeof(propsData));
  
 
  CMyComPtr<IUnknown> coder;
  if (_encoderFilter)
    coder = _encoderFilter;
  else
    coder = _encoder;
  {
    CMyComPtr<ICompressSetCoderProperties> scp;
    coder.QueryInterface(IID_ICompressSetCoderProperties, &scp);
    if (scp)
    {
      const UInt64 reduceSize = kBufferSize;
      /* in posix : new thread uses same affinity as parent thread,
         so we don't need to send affinity to coder in posix */
      UInt64 affMask = 0;
      UInt32 affinityGroup = (UInt32)(Int32)-1;
      // UInt64 affinityInGroup = 0;
#if !defined(Z7_ST) && defined(_WIN32)
      {
        CCpuSet cpuSet;
        if (AffinityMode.NeedGroupsMode()) // we need fix for affinityInGroup also
          affinityGroup = EncoderIndex % AffinityMode.NumGroups;
        else
          affMask = AffinityMode.GetAffinityMask(EncoderIndex, &cpuSet);
      }
#endif
      // affMask <<= 3; // debug line: to test no affinity in coder
      // affMask = 0; // for debug
      // affinityGroup = 0; // for debug
      // affinityInGroup = 1; // for debug
      RINOK(method.SetCoderProps_DSReduce_Aff(scp, &reduceSize,
          affMask != 0 ? &affMask : NULL,
          affinityGroup != (UInt32)(Int32)-1 ? &affinityGroup : NULL,
          /* affinityInGroup != 0 ? &affinityInGroup : */ NULL))
    }
    else
    {
      if (method.AreThereNonOptionalProps())
        return E_INVALIDARG;
    }

    CMyComPtr<ICompressWriteCoderProperties> writeCoderProps;
    coder.QueryInterface(IID_ICompressWriteCoderProperties, &writeCoderProps);
    if (writeCoderProps)
    {
      RINOK(writeCoderProps->WriteCoderProperties(propStream))
    }

    {
      CMyComPtr<ICryptoSetPassword> sp;
      coder.QueryInterface(IID_ICryptoSetPassword, &sp);
      if (sp)
      {
        RINOK(sp->CryptoSetPassword(_psw, sizeof(_psw)))

        // we must call encoding one time to calculate password key for key cache.
        // it must be after WriteCoderProperties!
        Byte temp[16];
        memset(temp, 0, sizeof(temp));
        
        if (_encoderFilter)
        {
          _encoderFilter->Init();
          _encoderFilter->Filter(temp, sizeof(temp));
        }
        else
        {
          CBenchmarkInStream *inStreamSpec = new CBenchmarkInStream;
          CMyComPtr<ISequentialInStream> inStream = inStreamSpec;
          inStreamSpec->Init(temp, sizeof(temp));
          
          CCrcOutStream *crcStreamSpec = new CCrcOutStream;
          CMyComPtr<ISequentialOutStream> crcStream = crcStreamSpec;
          crcStreamSpec->Init();

          RINOK(_encoder->Code(inStream, crcStream, NULL, NULL, NULL))
        }
      }
    }
  }

  return S_OK;
}


static void My_FilterBench(ICompressFilter *filter, Byte *data, size_t size, UInt32 *crc)
{
  while (size != 0)
  {
    UInt32 cur = crc ? 1 << 17 : 1 << 24;
    if (cur > size)
      cur = (UInt32)size;
    UInt32 processed = filter->Filter(data, cur);
    /* if (processed > size) (in AES filter), we must fill last block with zeros.
       but it is not important for benchmark. So we just copy that data without filtering.
       if (processed == 0) then filter can't process more  */
    if (processed > size || processed == 0)
      processed = (UInt32)size;
    if (crc)
      *crc = CrcUpdate(*crc, data, processed);
    data += processed;
    size -= processed;
  }
}


HRESULT CEncoderInfo::Encode()
{
  // printf("\nCEncoderInfo::Generate\n");

  RINOK(Generate())

  // printf("\n2222\n");

  #ifndef Z7_ST
  if (Common)
  {
    Results[0] = S_OK;
    WRes wres = ReadyEvent.Set();
    if (wres == 0)
      wres = Common->StartEvent.Lock();
    if (wres != 0)
      return HRESULT_FROM_WIN32(wres);
    if (Common->ExitMode)
      return S_OK;
  }
  else
  #endif
  {
    CBenchProgressInfo *bpi = progressInfoSpec[0];
    bpi->SetStartTime();
  }


  CBenchInfo &bi = progressInfoSpec[0]->BenchInfo;
  bi.UnpackSize = 0;
  bi.PackSize = 0;
  CMyComPtr<ICryptoProperties> cp;
  CMyComPtr<IUnknown> coder;
  if (_encoderFilter)
    coder = _encoderFilter;
  else
    coder = _encoder;
  coder.QueryInterface(IID_ICryptoProperties, &cp);
  CBenchmarkInStream *inStreamSpec = new CBenchmarkInStream;
  CMyComPtr<ISequentialInStream> inStream = inStreamSpec;

  if (cp)
  {
    RINOK(Set_Key_and_IV(cp))
  }

  compressedSize = 0;
  if (_encoderFilter)
    compressedSize = kBufferSize;

  // CBenchmarkOutStream *outStreamSpec = this->outStreamSpec;
  UInt64 prev = 0;
  
  const UInt32 mask = (CheckCrc_Enc ? 0 : 0xFFFF);
  const bool useCrc = (mask < NumIterations);
  bool crcPrev_defined = false;
  UInt32 crcPrev = 0;
  
  bool useRealData_Enc = UseRealData_Enc;
  bool data_Was_Changed = false;
  if (useRealData_Enc)
  {
    /* we want memcpy() for each iteration including first iteration.
       So results will be equal for different number of iterations */
    data_Was_Changed = true;
  }

  const UInt64 numIterations = NumIterations;
  UInt64 i = numIterations;
    // printCallback->NewLine();

  while (i != 0)
  {
    i--;
    if (printCallback && bi.UnpackSize - prev >= (1 << 26))
    {
      prev = bi.UnpackSize;
      RINOK(printCallback->CheckBreak())
    }

    /*
    CBenchInfo info;
    progressInfoSpec[0]->SetStartTime();
    */
    
    bool calcCrc = false;
    if (useCrc)
      calcCrc = (((UInt32)i & mask) == 0);

    if (_encoderFilter)
    {
      Byte *filterData = rgCopy;
      if (i == numIterations - 1 || calcCrc || useRealData_Enc)
      {
        // printf("\nfilterData = (Byte *)*outStreamSpec;\n");
        filterData = (Byte *)*outStreamSpec;
        if (data_Was_Changed)
        {
          // printf("\nmemcpy(filterData, uncompressedDataPtr\n");
          memcpy(filterData, uncompressedDataPtr, kBufferSize);
        }
        data_Was_Changed = true;
      }
      _encoderFilter->Init();
      if (calcCrc)
      {
        // printf("\nInitCrc\n");
        outStreamSpec->InitCrc();
      }
      // printf("\nMy_FilterBench\n");
      My_FilterBench(_encoderFilter, filterData, kBufferSize,
          calcCrc ? &outStreamSpec->Crc : NULL);
    }
    else
    {
      outStreamSpec->Init(true, calcCrc); // write real data for speed consistency at any number of iterations
      inStreamSpec->Init(uncompressedDataPtr, kBufferSize);
      RINOK(_encoder->Code(inStream, outStream, NULL, NULL, progressInfo[0]))
      if (!inStreamSpec->WasFinished())
        return E_FAIL;
      if (compressedSize != outStreamSpec->Pos)
      {
        if (compressedSize != 0)
          return E_FAIL;
        compressedSize = outStreamSpec->Pos;
      }
    }

    // outStreamSpec->Print();

    if (calcCrc)
    {
      const UInt32 crc2 = CRC_GET_DIGEST(outStreamSpec->Crc);
      if (crcPrev_defined && crcPrev != crc2)
        return E_FAIL;
      crcPrev = crc2;
      crcPrev_defined = true;
    }
    
    bi.UnpackSize += kBufferSize;
    bi.PackSize += compressedSize;

    /*
    {
      progressInfoSpec[0]->SetFinishTime(info);
      info.UnpackSize = 0;
      info.PackSize = 0;
      info.NumIterations = 1;
      
      info.UnpackSize = kBufferSize;
      info.PackSize = compressedSize;
      // printf("\n%7d\n", encoder.compressedSize);
      
      RINOK(callback->SetEncodeResult(info, true))
      printCallback->NewLine();
    }
    */

  }
  
  _encoder.Release();
  _encoderFilter.Release();
  return S_OK;
}


HRESULT CEncoderInfo::Decode(UInt32 decoderIndex)
{
  CBenchmarkInStream *inStreamSpec = new CBenchmarkInStream;
  CMyComPtr<ISequentialInStream> inStream = inStreamSpec;
  CMyComPtr<ICompressCoder> &decoder = _decoders[decoderIndex];
  CMyComPtr<IUnknown> coder;
  if (_decoderFilter)
  {
    if (decoderIndex != 0)
      return E_FAIL;
    coder = _decoderFilter;
  }
  else
    coder = decoder;

  // printf("\ndecoderIndex = %d, stack = %p", decoderIndex, &coder);

  CMyComPtr<ICompressSetDecoderProperties2> setDecProps;
  coder.QueryInterface(IID_ICompressSetDecoderProperties2, &setDecProps);
  if (!setDecProps && propStreamSpec->GetPos() != 0)
    return E_FAIL;

  CCrcOutStream *crcOutStreamSpec = new CCrcOutStream;
  CMyComPtr<ISequentialOutStream> crcOutStream = crcOutStreamSpec;
    
  CBenchProgressInfo *pi = progressInfoSpec[decoderIndex];
  pi->BenchInfo.UnpackSize = 0;
  pi->BenchInfo.PackSize = 0;

  #ifndef Z7_ST
  {
    CMyComPtr<ICompressSetCoderMt> setCoderMt;
    coder.QueryInterface(IID_ICompressSetCoderMt, &setCoderMt);
    if (setCoderMt)
    {
      RINOK(setCoderMt->SetNumberOfThreads(NumDecoderSubThreads))
    }
  }
  #endif

  CMyComPtr<ICompressSetCoderProperties> scp;
  coder.QueryInterface(IID_ICompressSetCoderProperties, &scp);
  if (scp)
  {
    const UInt64 reduceSize = _uncompressedDataSize;
    RINOK(_method.SetCoderProps(scp, &reduceSize))
  }

  CMyComPtr<ICryptoProperties> cp;
  coder.QueryInterface(IID_ICryptoProperties, &cp);
  
  if (setDecProps)
  {
    RINOK(setDecProps->SetDecoderProperties2(
        /* (const Byte *)*propStreamSpec, */
        propsData,
        (UInt32)propStreamSpec->GetPos()))
  }

  {
    CMyComPtr<ICryptoSetPassword> sp;
    coder.QueryInterface(IID_ICryptoSetPassword, &sp);
    if (sp)
    {
      RINOK(sp->CryptoSetPassword(_psw, sizeof(_psw)))
    }
  }

  UInt64 prev = 0;
  
  if (cp)
  {
    RINOK(Set_Key_and_IV(cp))
  }

  CMyComPtr<ICompressSetFinishMode> setFinishMode;

  if (_decoderFilter)
  {
    if (compressedSize > rgCopy.Size())
      return E_FAIL;
  }
  else
  {
    decoder->QueryInterface(IID_ICompressSetFinishMode, (void **)&setFinishMode);
  }

  const UInt64 numIterations = NumIterations;
  const E_CheckCrcMode checkCrcMode = CheckCrcMode_Dec;

  for (UInt64 i = 0; i < numIterations; i++)
  {
    if (printCallback && pi->BenchInfo.UnpackSize - prev >= (1 << 26))
    {
      RINOK(printCallback->CheckBreak())
      prev = pi->BenchInfo.UnpackSize;
    }

    const UInt64 outSize = kBufferSize;
    bool calcCrc = (checkCrcMode != k_CheckCrcMode_Never);

    crcOutStreamSpec->Init();

    if (_decoderFilter)
    {
      Byte *filterData = (Byte *)*outStreamSpec;
      if (calcCrc)
      {
        calcCrc = (i == 0);
        if (checkCrcMode == k_CheckCrcMode_Always)
        {
          calcCrc = true;
          memcpy((Byte *)rgCopy, (const Byte *)*outStreamSpec, compressedSize);
          filterData = rgCopy;
        }
      }
      _decoderFilter->Init();
      My_FilterBench(_decoderFilter, filterData, compressedSize,
          calcCrc ? &crcOutStreamSpec->Crc : NULL);
    }
    else
    {
      crcOutStreamSpec->CalcCrc = calcCrc;
      inStreamSpec->Init((const Byte *)*outStreamSpec, compressedSize);

      if (setFinishMode)
      {
        RINOK(setFinishMode->SetFinishMode(BoolToUInt(true)))
      }

      RINOK(decoder->Code(inStream, crcOutStream, NULL, &outSize, progressInfo[decoderIndex]))

      if (setFinishMode)
      {
        if (!inStreamSpec->WasFinished())
          return S_FALSE;

        CMyComPtr<ICompressGetInStreamProcessedSize> getInStreamProcessedSize;
        decoder.QueryInterface(IID_ICompressGetInStreamProcessedSize, (void **)&getInStreamProcessedSize);
        
        if (getInStreamProcessedSize)
        {
          UInt64 processed;
          RINOK(getInStreamProcessedSize->GetInStreamProcessedSize(&processed))
          if (processed != compressedSize)
            return S_FALSE;
        }
      }

      if (crcOutStreamSpec->Pos != outSize)
        return S_FALSE;
    }
  
    if (calcCrc && CRC_GET_DIGEST(crcOutStreamSpec->Crc) != crc)
      return S_FALSE;

    pi->BenchInfo.UnpackSize += kBufferSize;
    pi->BenchInfo.PackSize += compressedSize;
  }
  
  decoder.Release();
  _decoderFilter.Release();
  return S_OK;
}


static const UInt32 kNumThreadsMax = (1 << 12);

struct CBenchEncoders
{
  CEncoderInfo *encoders;
  CBenchEncoders(UInt32 num): encoders(NULL) { encoders = new CEncoderInfo[num]; }
  ~CBenchEncoders() { delete []encoders; }
};


static UInt64 GetNumIterations(UInt64 numCommands, UInt64 complexInCommands)
{
  if (numCommands < (1 << 4))
    numCommands = (1 << 4);
  UInt64 res = complexInCommands / numCommands;
  return (res == 0 ? 1 : res);
}



#ifndef Z7_ST

// ---------- CBenchThreadsFlusher ----------

struct CBenchThreadsFlusher
{
  CBenchEncoders *EncodersSpec;
  CBenchSyncCommon Common;
  unsigned NumThreads;
  bool NeedClose;

  CBenchThreadsFlusher(): NumThreads(0), NeedClose(false) {}
  
  ~CBenchThreadsFlusher()
  {
    StartAndWait(true);
  }

  WRes StartAndWait(bool exitMode = false);
};


WRes CBenchThreadsFlusher::StartAndWait(bool exitMode)
{
  if (!NeedClose)
    return 0;

  Common.ExitMode = exitMode;
  WRes res = Common.StartEvent.Set();

  for (unsigned i = 0; i < NumThreads; i++)
  {
    NWindows::CThread &t = EncodersSpec->encoders[i].thread[0];
    if (t.IsCreated())
    {
      WRes res2 = t.Wait_Close();
      if (res == 0)
        res = res2;
    }
  }
  NeedClose = false;
  return res;
}

#endif // Z7_ST



static void SetPseudoRand(Byte *data, size_t size, UInt32 startValue)
{
  for (size_t i = 0; i < size; i++)
  {
    data[i] = (Byte)startValue;
    startValue++;
  }
}



static HRESULT MethodBench(
    DECL_EXTERNAL_CODECS_LOC_VARS
    UInt64 complexInCommands,
    #ifndef Z7_ST
      bool oldLzmaBenchMode,
      UInt32 numThreads,
      const CAffinityMode *affinityMode,
    #endif
    const COneMethodInfo &method2,
    size_t uncompressedDataSize,
    const Byte *fileData,
    unsigned generateDictBits,

    IBenchPrintCallback *printCallback,
    IBenchCallback *callback,
    CBenchProps *benchProps)
{
  COneMethodInfo method = method2;
  UInt64 methodId;
  UInt32 numStreams;
  bool isFilter;
  const int codecIndex = FindMethod_Index(
      EXTERNAL_CODECS_LOC_VARS
      method.MethodName, true,
      methodId, numStreams, isFilter);
  if (codecIndex < 0)
    return E_NOTIMPL;
  if (numStreams != 1)
    return E_INVALIDARG;

  UInt32 numEncoderThreads = 1;
  UInt32 numSubDecoderThreads = 1;
  
  #ifndef Z7_ST
    numEncoderThreads = numThreads;

    if (oldLzmaBenchMode)
    if (methodId == k_LZMA)
    {
      if (numThreads == 1 && method.Get_NumThreads() < 0)
        method.AddProp_NumThreads(1);
      const UInt32 numLzmaThreads = method.Get_Lzma_NumThreads();
      if (numThreads > 1 && numLzmaThreads > 1)
      {
        numEncoderThreads = (numThreads + 1) / 2; // 20.03
        numSubDecoderThreads = 2;
      }
    }

  const bool mtEncMode = (numEncoderThreads > 1) || affinityMode->NeedAffinity();

  #endif

  CBenchEncoders encodersSpec(numEncoderThreads);
  CEncoderInfo *encoders = encodersSpec.encoders;

  UInt32 i;
  
  for (i = 0; i < numEncoderThreads; i++)
  {
    CEncoderInfo &encoder = encoders[i];
    encoder.callback = (i == 0) ? callback : NULL;
    encoder.printCallback = printCallback;

    #ifndef Z7_ST
    encoder.EncoderIndex = i;
    encoder.NumEncoderInternalThreads = numSubDecoderThreads;
    encoder.AffinityMode = *affinityMode;

    /*
    if (numSubDecoderThreads > 1)
    if (encoder.AffinityMode.NeedAffinity()
        && encoder.AffinityMode.NumBundleThreads == 1)
    {
      // if old LZMA benchmark uses two threads in coder, we increase (NumBundleThreads) for old LZMA benchmark uses two threads instead of one
      if (encoder.AffinityMode.NumBundleThreads * 2 <= encoder.AffinityMode.NumCores)
        encoder.AffinityMode.NumBundleThreads *= 2;
    }
    */

    #endif

    {
      CCreatedCoder cod;
      RINOK(CreateCoder_Index(EXTERNAL_CODECS_LOC_VARS (unsigned)codecIndex, true, encoder._encoderFilter, cod))
      encoder._encoder = cod.Coder;
      if (!encoder._encoder && !encoder._encoderFilter)
        return E_NOTIMPL;
    }

    SetPseudoRand(encoder._iv,  sizeof(encoder._iv), 17);
    SetPseudoRand(encoder._key, sizeof(encoder._key), 51);
    SetPseudoRand(encoder._psw, sizeof(encoder._psw), 123);

    for (UInt32 j = 0; j < numSubDecoderThreads; j++)
    {
      CCreatedCoder cod;
      CMyComPtr<ICompressCoder> &decoder = encoder._decoders[j];
      RINOK(CreateCoder_Id(EXTERNAL_CODECS_LOC_VARS methodId, false, encoder._decoderFilter, cod))
      decoder = cod.Coder;
      if (!encoder._decoderFilter && !decoder)
        return E_NOTIMPL;
    }

    encoder.UseRealData_Enc =
    encoder.CheckCrc_Enc = (benchProps->EncComplex) > 30;

    encoder.CheckCrcMode_Dec = k_CheckCrcMode_Always;
    if (benchProps->DecComplexCompr +
        benchProps->DecComplexUnc <= 30)
      encoder.CheckCrcMode_Dec =
          k_CheckCrcMode_FirstPass; // for filters
          // k_CheckCrcMode_Never; // for debug
          // k_CheckCrcMode_Always; // for debug
    if (fileData)
    {
      encoder.UseRealData_Enc = true;
      encoder.CheckCrcMode_Dec = k_CheckCrcMode_Always;
    }
  }

  UInt32 crc = 0;
  if (fileData)
    crc = CrcCalc(fileData, uncompressedDataSize);

  for (i = 0; i < numEncoderThreads; i++)
  {
    CEncoderInfo &encoder = encoders[i];
    encoder._method = method;
    encoder.generateDictBits = generateDictBits;
    encoder._uncompressedDataSize = uncompressedDataSize;
    encoder.kBufferSize = uncompressedDataSize;
    encoder.fileData = fileData;
    encoder.crc = crc;
  }

  CBenchProgressStatus status;
  status.Res = S_OK;
  status.EncodeMode = true;

  #ifndef Z7_ST
  CBenchThreadsFlusher encoderFlusher;
  if (mtEncMode)
  {
    WRes wres = encoderFlusher.Common.StartEvent.Create();
    if (wres != 0)
      return HRESULT_FROM_WIN32(wres);
    encoderFlusher.NumThreads = numEncoderThreads;
    encoderFlusher.EncodersSpec = &encodersSpec;
    encoderFlusher.NeedClose = true;
  }
  #endif

  for (i = 0; i < numEncoderThreads; i++)
  {
    CEncoderInfo &encoder = encoders[i];
    encoder.NumIterations = GetNumIterations(benchProps->GetNumCommands_Enc(uncompressedDataSize), complexInCommands);
    // encoder.NumIterations = 3;
    {
#if 0
      #define kCrcPoly 0xEDB88320
      UInt32 r = i;
      unsigned num = numEncoderThreads < 256 ? 8 : 16;
      do
        r = (r >> 1) ^ (kCrcPoly & ((UInt32)0 - (r & 1)));
      while (--num);
      encoder.Salt = r;
#else
      UInt32 salt0 = g_CrcTable[(Byte)i];
      UInt32 salt1 = g_CrcTable[(Byte)(i >> 8)];
      encoder.Salt = salt0 ^ (salt1 << 3);
#endif
    }

    // (g_CrcTable[0] == 0), and (encoder.Salt == 0) for first thread
    // printf("\n encoder index = %d, Salt = %8x\n", i, encoder.Salt);

    encoder.KeySize = benchProps->KeySize;

    for (int j = 0; j < 2; j++)
    {
      CBenchProgressInfo *spec = new CBenchProgressInfo;
      encoder.progressInfoSpec[j] = spec;
      encoder.progressInfo[j] = spec;
      spec->Status = &status;
    }
    
    if (i == 0)
    {
      CBenchProgressInfo *bpi = encoder.progressInfoSpec[0];
      bpi->Callback = callback;
      bpi->BenchInfo.NumIterations = numEncoderThreads;
    }

    #ifndef Z7_ST
    if (mtEncMode)
    {
      #ifdef USE_ALLOCA
      encoder.AllocaSize = BENCH_ALLOCA_VALUE(i);
      #endif

      encoder.Common = &encoderFlusher.Common;
      encoder.IsGlobalMtMode = numEncoderThreads > 1;
      RINOK(encoder.CreateEncoderThread())
    }
    #endif
  }

  if (printCallback)
  {
    RINOK(printCallback->CheckBreak())
  }

  #ifndef Z7_ST
  if (mtEncMode)
  {
    for (i = 0; i < numEncoderThreads; i++)
    {
      CEncoderInfo &encoder = encoders[i];
      const WRes wres = encoder.ReadyEvent.Lock();
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
      RINOK(encoder.Results[0])
    }

    CBenchProgressInfo *bpi = encoders[0].progressInfoSpec[0];
    bpi->SetStartTime();

    const WRes wres = encoderFlusher.StartAndWait();
    if (status.Res == 0 && wres != 0)
      return HRESULT_FROM_WIN32(wres);
  }
  else
  #endif
  {
    RINOK(encoders[0].Encode())
  }

  RINOK(status.Res)

  CBenchInfo info;

  encoders[0].progressInfoSpec[0]->SetFinishTime(info);
  info.UnpackSize = 0;
  info.PackSize = 0;
  info.NumIterations = encoders[0].NumIterations;
  
  for (i = 0; i < numEncoderThreads; i++)
  {
    const CEncoderInfo &encoder = encoders[i];
    info.UnpackSize += encoder.kBufferSize;
    info.PackSize += encoder.compressedSize;
    // printf("\n%7d\n", encoder.compressedSize);
  }
  
  RINOK(callback->SetEncodeResult(info, true))




  // ---------- Decode ----------

  status.Res = S_OK;
  status.EncodeMode = false;

  const UInt32 numDecoderThreads = numEncoderThreads * numSubDecoderThreads;
  #ifndef Z7_ST
  const bool mtDecoderMode = (numDecoderThreads > 1) || affinityMode->NeedAffinity();
  #endif
  
  for (i = 0; i < numEncoderThreads; i++)
  {
    CEncoderInfo &encoder = encoders[i];

    /*
    #ifndef Z7_ST
    // encoder.affinityMode = *affinityMode;
    if (encoder.NumEncoderInternalThreads != 1)
      encoder.AffinityMode.DivideNum = encoder.NumEncoderInternalThreads;
    #endif
    */


    if (i == 0)
    {
      encoder.NumIterations = GetNumIterations(
          benchProps->GetNumCommands_Dec(
              encoder.compressedSize,
              encoder.kBufferSize),
          complexInCommands);
      CBenchProgressInfo *bpi = encoder.progressInfoSpec[0];
      bpi->Callback = callback;
      bpi->BenchInfo.NumIterations = numDecoderThreads;
      bpi->SetStartTime();
    }
    else
      encoder.NumIterations = encoders[0].NumIterations;

    #ifndef Z7_ST
    {
      const int numSubThreads = method.Get_NumThreads();
      encoder.NumDecoderSubThreads = (numSubThreads <= 0) ? 1 : (unsigned)numSubThreads;
    }
    if (mtDecoderMode)
    {
      for (UInt32 j = 0; j < numSubDecoderThreads; j++)
      {
        const HRESULT res = encoder.CreateDecoderThread(j, (i == 0 && j == 0)
            #ifdef USE_ALLOCA
            , BENCH_ALLOCA_VALUE(i * numSubDecoderThreads + j)
            #endif
            );
        RINOK(res)
      }
    }
    else
    #endif
    {
      RINOK(encoder.Decode(0))
    }
  }
  
  #ifndef Z7_ST
  if (mtDecoderMode)
  {
    WRes wres = 0;
    HRESULT res = S_OK;
    for (i = 0; i < numEncoderThreads; i++)
      for (UInt32 j = 0; j < numSubDecoderThreads; j++)
      {
        CEncoderInfo &encoder = encoders[i];
        const WRes wres2 = encoder.thread[j].
            // Wait(); // later we can get thread times from thread in UNDER_CE
            Wait_Close();
        if (wres == 0 && wres2 != 0)
          wres = wres2;
        const HRESULT res2 = encoder.Results[j];
        if (res == 0 && res2 != 0)
          res = res2;
      }
    if (wres != 0)
      return HRESULT_FROM_WIN32(wres);
    RINOK(res)
  }
  #endif // Z7_ST
 
  RINOK(status.Res)
  encoders[0].progressInfoSpec[0]->SetFinishTime(info);
 
  /*
  #ifndef Z7_ST
  #ifdef UNDER_CE
  if (mtDecoderMode)
    for (i = 0; i < numEncoderThreads; i++)
      for (UInt32 j = 0; j < numSubDecoderThreads; j++)
      {
        FILETIME creationTime, exitTime, kernelTime, userTime;
        if (::GetThreadTimes(encoders[i].thread[j], &creationTime, &exitTime, &kernelTime, &userTime) != 0)
          info.UserTime += GetTime64(userTime) + GetTime64(kernelTime);
      }
  #endif
  #endif
  */
  
  info.UnpackSize = 0;
  info.PackSize = 0;
  info.NumIterations = numSubDecoderThreads * encoders[0].NumIterations;
  
  for (i = 0; i < numEncoderThreads; i++)
  {
    const CEncoderInfo &encoder = encoders[i];
    info.UnpackSize += encoder.kBufferSize;
    info.PackSize += encoder.compressedSize;
  }
  
  // RINOK(callback->SetDecodeResult(info, false)) // why we called before 21.03 ??
  RINOK(callback->SetDecodeResult(info, true))
  
  return S_OK;
}



static inline UInt64 GetDictSizeFromLog(unsigned dictSizeLog)
{
  /*
  if (dictSizeLog < 32)
    return (UInt32)1 << dictSizeLog;
  else
    return (UInt32)(Int32)-1;
  */
  return (UInt64)1 << dictSizeLog;
}


// it's limit of current LZMA implementation that can be changed later
#define kLzmaMaxDictSize ((UInt32)15 << 28)

static inline UInt64 GetLZMAUsage(bool multiThread, int btMode, UInt64 dict)
{
  if (dict == 0)
    dict = 1;
  if (dict > kLzmaMaxDictSize)
    dict = kLzmaMaxDictSize;
  UInt32 hs = (UInt32)dict - 1;
  hs |= (hs >> 1);
  hs |= (hs >> 2);
  hs |= (hs >> 4);
  hs |= (hs >> 8);
  hs >>= 1;
  hs |= 0xFFFF;
  if (hs > (1 << 24))
    hs >>= 1;
  hs++;
  hs += (1 << 16);

  const UInt32 kBlockSizeMax = (UInt32)0 - (UInt32)(1 << 16);
  UInt64 blockSize = (UInt64)dict + (1 << 16)
      + (multiThread ? (1 << 20) : 0);
  blockSize += (blockSize >> (blockSize < ((UInt32)1 << 30) ? 1 : 2));
  if (blockSize >= kBlockSizeMax)
    blockSize = kBlockSizeMax;

  UInt64 son = (UInt64)dict;
  if (btMode)
    son *= 2;
  const UInt64 v = (hs + son) * 4 + blockSize +
      (1 << 20) + (multiThread ? (6 << 20) : 0);

  // printf("\nGetLZMAUsage = %d\n", (UInt32)(v >> 20));
  // printf("\nblockSize = %d\n", (UInt32)(blockSize >> 20));
  return v;
}


UInt64 GetBenchMemoryUsage(UInt32 numThreads, int level, UInt64 dictionary, bool totalBench)
{
  const size_t kBufferSize = (size_t)dictionary + kAdditionalSize;
  const UInt64 kCompressedBufferSize = GetBenchCompressedSize(kBufferSize); // / 2;
  if (level < 0)
    level = 5;
  const int algo = (level < 5 ? 0 : 1);
  const int btMode = (algo == 0 ? 0 : 1);

  UInt32 numBigThreads = numThreads;
  const bool lzmaMt = (totalBench || (numThreads > 1 && btMode));
  if (btMode)
  {
    if (!totalBench && lzmaMt)
      numBigThreads /= 2;
  }
  return ((UInt64)kBufferSize + kCompressedBufferSize +
    GetLZMAUsage(lzmaMt, btMode, dictionary) + (2 << 20)) * numBigThreads;
}

static UInt64 GetBenchMemoryUsage_Hash(UInt32 numThreads, UInt64 dictionary)
{
  // dictionary += (dictionary >> 9); // for page tables (virtual memory)
  return (UInt64)(dictionary + (1 << 15)) * numThreads + (2 << 20);
}


// ---------- CRC and HASH ----------

struct CCrcInfo_Base
{
  CMidAlignedBuffer Buffer;
  const Byte *Data;
  size_t Size;
  bool CreateLocalBuf;
  UInt32 CheckSum_Res;
  
  CCrcInfo_Base(): CreateLocalBuf(true), CheckSum_Res(0) {}
  
  HRESULT Generate(const Byte *data, size_t size);
  HRESULT CrcProcess(UInt64 numIterations,
      const UInt32 *checkSum, IHasher *hf,
      IBenchPrintCallback *callback);
};


// for debug: define it to test hash calling with unaligned data
// #define Z7_BENCH_HASH_ALIGN_BUF_OFFSET  3

HRESULT CCrcInfo_Base::Generate(const Byte *data, size_t size)
{
  Size = size;
  Data = data;
  if (!data || CreateLocalBuf)
  {
    Byte *buf;
    const size_t size2 = (size + k_RandBuf_AlignMask) & ~(size_t)k_RandBuf_AlignMask;
    if (size2 < size)
      return E_OUTOFMEMORY;
#ifdef Z7_BENCH_HASH_ALIGN_BUF_OFFSET
    ALLOC_WITH_HRESULT(&Buffer, size2 + Z7_BENCH_HASH_ALIGN_BUF_OFFSET)
    buf = Buffer + Z7_BENCH_HASH_ALIGN_BUF_OFFSET;
#else
    ALLOC_WITH_HRESULT(&Buffer, size2)
    buf = Buffer;
#endif
    Data = buf;
    if (!data)
      RandGen_BufAfterPad(buf, size);
    else if (size != 0) // (CreateLocalBuf == true)
      memcpy(buf, data, size);
  }
  return S_OK;
}


#if 1
#define HashUpdate(hf, data, size)  hf->Update(data, size)
#else
// for debug:
static void HashUpdate(IHasher *hf, const void *data, UInt32 size)
{
  for (;;)
  {
    if (size == 0)
      return;
    UInt32 size2 = (size * 0x85EBCA87) % size / 8;
    // UInt32 size2 = size / 2;
    if (size2 == 0)
      size2 = 1;
    hf->Update(data, size2);
    data = (const void *)((const Byte *)data + size2);
    size -= size2;
  }
}
#endif


HRESULT CCrcInfo_Base::CrcProcess(UInt64 numIterations,
    const UInt32 *checkSum, IHasher *hf,
    IBenchPrintCallback *callback)
{
  MY_ALIGN(16)
  UInt32 hash32[64 / 4];
  memset(hash32, 0, sizeof(hash32));

  CheckSum_Res = 0;

  const UInt32 hashSize = hf->GetDigestSize();
  if (hashSize > sizeof(hash32))
    return S_FALSE;

  const Byte *buf = Data;
  const size_t size = Size;
  UInt32 checkSum_Prev = 0;

  UInt64 prev = 0;
  UInt64 cur = 0;

  do
  {
    hf->Init();
    size_t pos = 0;
    do
    {
      const size_t rem = size - pos;
      const UInt32 kStep = ((UInt32)1 << 31);
      const UInt32 curSize = (rem < kStep) ? (UInt32)rem : kStep;
      HashUpdate(hf, buf + pos, curSize);
      pos += curSize;
    }
    while (pos != size);

    hf->Final((Byte *)(void *)hash32);
    UInt32 sum = 0;
    for (UInt32 j = 0; j < hashSize; j += 4)
    {
      sum = rotlFixed(sum, 11);
      sum += GetUi32((const Byte *)(const void *)hash32 + j);
    }
    if (checkSum)
    {
      if (sum != *checkSum)
        return S_FALSE;
    }
    else
    {
      checkSum_Prev = sum;
      checkSum = &checkSum_Prev;
    }
    if (callback)
    {
      cur += size;
      if (cur - prev >= ((UInt32)1 << 30))
      {
        prev = cur;
        RINOK(callback->CheckBreak())
      }
    }
  }
  while (--numIterations);

  CheckSum_Res = checkSum_Prev;
  return S_OK;
}

extern
UInt32 g_BenchCpuFreqTemp; // we need non-static variavble to disable compiler optimization
UInt32 g_BenchCpuFreqTemp = 1;

#define YY1 sum += val; sum ^= val;
#define YY3 YY1 YY1 YY1 YY1
#define YY5 YY3 YY3 YY3 YY3
#define YY7 YY5 YY5 YY5 YY5
static const UInt32 kNumFreqCommands = 128;

EXTERN_C_BEGIN

static UInt32 CountCpuFreq(UInt32 sum, UInt32 num, UInt32 val)
{
  for (UInt32 i = 0; i < num; i++)
  {
    YY7
  }
  return sum;
}

EXTERN_C_END


#ifndef Z7_ST

struct CBaseThreadInfo
{
  NWindows::CThread Thread;
  IBenchPrintCallback *Callback;
  HRESULT CallbackRes;

  WRes Wait_If_Created()
  {
    if (!Thread.IsCreated())
      return 0;
    return Thread.Wait_Close();
  }
};

struct CFreqInfo: public CBaseThreadInfo
{
  UInt32 ValRes;
  UInt32 Size;
  UInt64 NumIterations;
};

static THREAD_FUNC_DECL FreqThreadFunction(void *param)
{
  CFreqInfo *p = (CFreqInfo *)param;

  UInt32 sum = g_BenchCpuFreqTemp;
  for (UInt64 k = p->NumIterations; k > 0; k--)
  {
    if (p->Callback)
    {
      p->CallbackRes = p->Callback->CheckBreak();
      if (p->CallbackRes != S_OK)
        break;
    }
    sum = CountCpuFreq(sum, p->Size, g_BenchCpuFreqTemp);
  }
  p->ValRes = sum;
  return THREAD_FUNC_RET_ZERO;
}

struct CFreqThreads
{
  CFreqInfo *Items;
  UInt32 NumThreads;

  CFreqThreads(): Items(NULL), NumThreads(0) {}
  
  WRes WaitAll()
  {
    WRes wres = 0;
    for (UInt32 i = 0; i < NumThreads; i++)
    {
      WRes wres2 = Items[i].Wait_If_Created();
      if (wres == 0 && wres2 != 0)
        wres = wres2;
    }
    NumThreads = 0;
    return wres;
  }
  
  ~CFreqThreads()
  {
    WaitAll();
    delete []Items;
  }
};


static THREAD_FUNC_DECL CrcThreadFunction(void *param);

struct CCrcInfo: public CBaseThreadInfo
{
  const Byte *Data;
  size_t Size;
  UInt64 NumIterations;
  bool CheckSumDefined;
  UInt32 CheckSum;
  CMyComPtr<IHasher> Hasher;
  HRESULT Res;
  UInt32 CheckSum_Res;

  #ifndef Z7_ST
  NSynchronization::CManualResetEvent ReadyEvent;
  UInt32 ThreadIndex;
  CBenchSyncCommon *Common;
  CAffinityMode AffinityMode;
  #endif

  // we want to call CCrcInfo_Base::Buffer.Free() in main thread.
  // so we uses non-local CCrcInfo_Base.
  CCrcInfo_Base crcib;

  HRESULT CreateThread()
  {
    WRes res = 0;
    if (!ReadyEvent.IsCreated())
      res = ReadyEvent.Create();
    if (res == 0)
      res = AffinityMode.CreateThread_WithAffinity(Thread, CrcThreadFunction, this,
          ThreadIndex);
    return HRESULT_FROM_WIN32(res);
  }

  #ifdef USE_ALLOCA
  size_t AllocaSize;
  #endif

  void Process();

  CCrcInfo(): Res(E_FAIL) {}
};

static const bool k_Crc_CreateLocalBuf_For_File = true; // for total BW test
// static const bool k_Crc_CreateLocalBuf_For_File = false; // for shared memory read test

void CCrcInfo::Process()
{
  crcib.CreateLocalBuf = k_Crc_CreateLocalBuf_For_File;
  // we can use additional Generate() passes to reduce some time effects for new page allocation
  // for (unsigned y = 0; y < 10; y++)
  Res = crcib.Generate(Data, Size);

  // if (Common)
  {
    WRes wres = ReadyEvent.Set();
    if (wres != 0)
    {
      if (Res == 0)
        Res = HRESULT_FROM_WIN32(wres);
      return;
    }
    if (Res != 0)
      return;

    wres = Common->StartEvent.Lock();

    if (wres != 0)
    {
      Res = HRESULT_FROM_WIN32(wres);
      return;
    }
    if (Common->ExitMode)
      return;
  }

  Res = crcib.CrcProcess(NumIterations,
      CheckSumDefined ? &CheckSum : NULL, Hasher,
      Callback);
  CheckSum_Res = crcib.CheckSum_Res;
  /*
  We don't want to include the time of slow CCrcInfo_Base::Buffer.Free()
  to time of benchmark. So we don't free Buffer here
  */
  // crcib.Buffer.Free();
}


static THREAD_FUNC_DECL CrcThreadFunction(void *param)
{
  CCrcInfo *p = (CCrcInfo *)param;
  
  #ifdef USE_ALLOCA
  alloca(p->AllocaSize);
  #endif
  p->Process();
  return THREAD_FUNC_RET_ZERO;
}


struct CCrcThreads
{
  CCrcInfo *Items;
  unsigned NumThreads;
  CBenchSyncCommon Common;
  bool NeedClose;

  CCrcThreads(): Items(NULL), NumThreads(0), NeedClose(false) {}

  WRes StartAndWait(bool exitMode = false);

  ~CCrcThreads()
  {
    StartAndWait(true);
    delete []Items;
  }
};


WRes CCrcThreads::StartAndWait(bool exitMode)
{
  if (!NeedClose)
    return 0;

  Common.ExitMode = exitMode;
  WRes wres = Common.StartEvent.Set();

  for (unsigned i = 0; i < NumThreads; i++)
  {
    WRes wres2 = Items[i].Wait_If_Created();
    if (wres == 0 && wres2 != 0)
      wres = wres2;
  }
  NumThreads = 0;
  NeedClose = false;
  return wres;
}

#endif


/*
static UInt32 CrcCalc1(const Byte *buf, size_t size)
{
  UInt32 crc = CRC_INIT_VAL;
  for (size_t i = 0; i < size; i++)
    crc = CRC_UPDATE_BYTE(crc, buf[i]);
  return CRC_GET_DIGEST(crc);
}
*/

/*
static UInt32 RandGenCrc(Byte *buf, size_t size, CBaseRandomGenerator &RG)
{
  RandGen(buf, size, RG);
  return CrcCalc1(buf, size);
}
*/

static bool CrcInternalTest()
{
  CAlignedBuffer buffer;
  const size_t kBufSize = 1 << 11;
  const size_t kCheckSize = 1 << 6;
  buffer.Alloc(kBufSize);
  if (!buffer.IsAllocated())
    return false;
  Byte *buf = (Byte *)buffer;
  RandGen_BufAfterPad(buf, kBufSize);
  UInt32 sum = 0;
  for (size_t i = 0; i < kBufSize - kCheckSize * 2; i += kCheckSize - 1)
    for (size_t j = 0; j < kCheckSize; j++)
    {
      sum = rotlFixed(sum, 11);
      sum += CrcCalc(buf + i + j, j);
    }
  return sum == 0x28462c7c;
}

struct CBenchMethod
{
  unsigned Weight;
  unsigned DictBits;
  Int32 EncComplex;
  Int32 DecComplexCompr;
  Int32 DecComplexUnc;
  const char *Name;
  // unsigned KeySize;
};

// #define USE_SW_CMPLX

#ifdef USE_SW_CMPLX
#define CMPLX(x) ((x) * 1000)
#else
#define CMPLX(x) (x)
#endif

static const CBenchMethod g_Bench[] =
{
  // { 40, 17,  357,  145,   20, "LZMA:x1" },
  // { 20, 18,  360,  145,   20, "LZMA2:x1:mt2" },

  { 20, 18,  360,  145,   20, "LZMA:x1" },
  { 20, 22,  600,  145,   20, "LZMA:x3" },

  { 80, 24, 1220,  145,   20, "LZMA:x5:mt1" },
  { 80, 24, 1220,  145,   20, "LZMA:x5:mt2" },

  { 10, 16,  124,   40,   14, "Deflate:x1" },
  { 20, 16,  376,   40,   14, "Deflate:x5" },
  { 10, 16, 1082,   40,   14, "Deflate:x7" },
  { 10, 17,  422,   40,   14, "Deflate64:x5" },

  { 10, 15,  590,   69,   69, "BZip2:x1" },
  { 20, 19,  815,  122,  122, "BZip2:x5" },
  { 10, 19,  815,  122,  122, "BZip2:x5:mt2" },
  { 10, 19, 2530,  122,  122, "BZip2:x7" },

  // { 10, 18, 1010,    0, 1150, "PPMDZip:x1" },
  { 10, 18, 1010,    0, 1150, "PPMD:x1" },
  // { 10, 22, 1655,    0, 1830, "PPMDZip:x5" },
  { 10, 22, 1655,    0, 1830, "PPMD:x5" },

  // {  2,  0,  -16,    0,  -16, "Swap2" },
  {  2,  0,  -16,    0,  -16, "Swap4" },

  // {  2,  0,    3,    0,    4, "Delta:1" },
  // {  2,  0,    3,    0,    4, "Delta:2" },
  // {  2,  0,    3,    0,    4, "Delta:3" },
  {  2,  0,    3,    0,    4, "Delta:4" },
  // {  2,  0,    3,    0,    4, "Delta:8" },
  // {  2,  0,    3,    0,    4, "Delta:32" },

  {  2,  0,    2,    0,    2, "BCJ" },
  {  2,  0,    1,    0,    1, "ARM64" },
  {  2,  0,    1,    0,    1, "RISCV" },
  
  // { 10,  0,   18,    0,   18, "AES128CBC:1" },
  // { 10,  0,   21,    0,   21, "AES192CBC:1" },
  { 10,  0,   24,    0,   24, "AES256CBC:1" },

  // { 10,  0,   18,    0,   18, "AES128CTR:1" },
  // { 10,  0,   21,    0,   21, "AES192CTR:1" },
  // { 10,  0,   24,    0,   24, "AES256CTR:1" },
  // {  2,  0, CMPLX(6), 0, CMPLX(1), "AES128CBC:2" },
  // {  2,  0, CMPLX(7), 0, CMPLX(1), "AES192CBC:2" },
  {  2,  0, CMPLX(8), 0, CMPLX(1), "AES256CBC:2" },
  
  // {  2,  0, CMPLX(1), 0, CMPLX(1), "AES128CTR:2" },
  // {  2,  0, CMPLX(1), 0, CMPLX(1), "AES192CTR:2" },
  // {  2,  0, CMPLX(1), 0, CMPLX(1), "AES256CTR:2" },

  // {  1,  0, CMPLX(6), 0, -2, "AES128CBC:3" },
  // {  1,  0, CMPLX(7), 0, -2, "AES192CBC:3" },
  {  1,  0, CMPLX(8), 0, -2, "AES256CBC:3" }
  
  // {  1,  0, CMPLX(1), 0, -2, "AES128CTR:3" },
  // {  1,  0, CMPLX(1), 0, -2, "AES192CTR:3" },
  // {  1,  0, CMPLX(1), 0, -2, "AES256CTR:3" },
};

struct CBenchHash
{
  unsigned Weight;
  UInt32 Complex;
  UInt32 CheckSum;
  const char *Name;
};

// #define ARM_CRC_MUL 100
#define ARM_CRC_MUL 1

#define k_Hash_Complex_Mult 256

static const CBenchHash g_Hash[] =
{
  { 20,   256, 0x21e207bb, "CRC32:12" } ,
  {  2,   128 *ARM_CRC_MUL, 0x21e207bb, "CRC32:32" },
  {  2,    64 *ARM_CRC_MUL, 0x21e207bb, "CRC32:64" },
  { 10,   256, 0x41b901d1, "CRC64" },
  {  5,    64, 0x43eac94f, "XXH64" },
  {  2,  2340, 0x3398a904, "MD5" },
  { 10,  2340,                       0xff769021, "SHA1:1" },
  {  2, CMPLX((20 * 6 + 1) * 4 + 4), 0xff769021, "SHA1:2" },
  { 10,  5100,                       0x7913ba03, "SHA256:1" },
  {  2, CMPLX((32 * 4 + 1) * 4 + 4), 0x7913ba03, "SHA256:2" },
  {  5,  3200,                       0xe7aeb394, "SHA512:1" },
  {  2, CMPLX((40 * 4 + 1) * 4 + 4), 0xe7aeb394, "SHA512:2" },
  // { 10, 3428,       0x1cc99b18, "SHAKE128" },
  // { 10, 4235,       0x74eaddc3, "SHAKE256" },
  // { 10, 4000,       0xdf3e6863, "SHA3-224" },
  {  5, 4200,       0xcecac10d, "SHA3-256" },
  // { 10, 5538,       0x4e5d9163, "SHA3-384" },
  // { 10, 8000,       0x96a58289, "SHA3-512" },
  {  2,  4096, 0x85189d02, "BLAKE2sp:1" },
  {  2,  1024, 0x85189d02, "BLAKE2sp:2" }, // sse2-way4-fast
  {  2,   512, 0x85189d02, "BLAKE2sp:3" }  // avx2-way8-fast
#if 0
  , {  2,  2048, 0x85189d02, "BLAKE2sp:4" } // sse2-way1
  , {  2,  1024, 0x85189d02, "BLAKE2sp:5" } // sse2-way2
  , {  2,  1024, 0x85189d02, "BLAKE2sp:6" } // avx2-way2
  , {  2,  1024, 0x85189d02, "BLAKE2sp:7" } // avx2-way4
#endif
};

static void PrintNumber(IBenchPrintCallback &f, UInt64 value, unsigned size)
{
  char s[128];
  unsigned startPos = (unsigned)sizeof(s) - 32;
  memset(s, ' ', startPos);
  ConvertUInt64ToString(value, s + startPos);
  // if (withSpace)
  {
    startPos--;
    size++;
  }
  unsigned len = (unsigned)strlen(s + startPos);
  if (size > len)
  {
    size -= len;
    if (startPos < size)
      startPos = 0;
    else
      startPos -= size;
  }
  f.Print(s + startPos);
}

static const unsigned kFieldSize_Name = 12;
static const unsigned kFieldSize_SmallName = 4;
static const unsigned kFieldSize_Speed = 9;
static const unsigned kFieldSize_Usage = 5;
static const unsigned kFieldSize_RU = 6;
static const unsigned kFieldSize_Rating = 6;
static const unsigned kFieldSize_EU = 5;
static const unsigned kFieldSize_Effec = 5;
static const unsigned kFieldSize_CrcSpeed = 8;


static const unsigned kFieldSize_TotalSize = 4 + kFieldSize_Speed + kFieldSize_Usage + kFieldSize_RU + kFieldSize_Rating;
static const unsigned kFieldSize_EUAndEffec = 2 + kFieldSize_EU + kFieldSize_Effec;


static void PrintRating(IBenchPrintCallback &f, UInt64 rating, unsigned size)
{
  PrintNumber(f, (rating + 500000) / 1000000, size);
}


static void PrintPercents(IBenchPrintCallback &f, UInt64 val, UInt64 divider, unsigned size)
{
  UInt64 v = 0;
  if (divider != 0)
    v = (val * 100 + divider / 2) / divider;
  PrintNumber(f, v, size);
}

static void PrintChars(IBenchPrintCallback &f, char c, unsigned size)
{
  char s[256];
  memset(s, (Byte)c, size);
  s[size] = 0;
  f.Print(s);
}

static void PrintSpaces(IBenchPrintCallback &f, unsigned size)
{
  PrintChars(f, ' ', size);
}

static void PrintUsage(IBenchPrintCallback &f, UInt64 usage, unsigned size)
{
  PrintNumber(f, Benchmark_GetUsage_Percents(usage), size);
}

static void PrintResults(IBenchPrintCallback &f, UInt64 usage, UInt64 rpu, UInt64 rating, bool showFreq, UInt64 cpuFreq)
{
  PrintUsage(f, usage, kFieldSize_Usage);
  PrintRating(f, rpu, kFieldSize_RU);
  PrintRating(f, rating, kFieldSize_Rating);
  if (showFreq)
  {
    if (cpuFreq == 0)
      PrintSpaces(f, kFieldSize_EUAndEffec);
    else
    {
      PrintPercents(f, rating, cpuFreq * usage / kBenchmarkUsageMult, kFieldSize_EU);
      PrintPercents(f, rating, cpuFreq, kFieldSize_Effec);
    }
  }
}


void CTotalBenchRes::Generate_From_BenchInfo(const CBenchInfo &info)
{
  Speed = info.GetUnpackSizeSpeed();
  Usage = info.GetUsage();
  RPU = info.GetRatingPerUsage(Rating);
}

void CTotalBenchRes::Mult_For_Weight(unsigned weight)
{
  NumIterations2 *= weight;
  RPU *= weight;
  Rating *= weight;
  Usage *= weight;
  Speed *= weight;
}

void CTotalBenchRes::Update_With_Res(const CTotalBenchRes &r)
{
  Rating += r.Rating;
  Usage += r.Usage;
  RPU += r.RPU;
  Speed += r.Speed;
    // NumIterations1 = (r1.NumIterations1 + r2.NumIterations1);
  NumIterations2 += r.NumIterations2;
}

static void PrintResults(IBenchPrintCallback *f,
    const CBenchInfo &info,
    unsigned weight,
    UInt64 rating,
    bool showFreq, UInt64 cpuFreq,
    CTotalBenchRes *res)
{
  CTotalBenchRes t;
  t.Rating = rating;
  t.NumIterations2 = 1;
  t.Generate_From_BenchInfo(info);
  
  if (f)
  {
    if (t.Speed != 0)
      PrintNumber(*f, t.Speed / 1024, kFieldSize_Speed);
    else
      PrintSpaces(*f, 1 + kFieldSize_Speed);
  }
  if (f)
  {
    PrintResults(*f, t.Usage, t.RPU, rating, showFreq, cpuFreq);
  }

  if (res)
  {
    // res->NumIterations1++;
    t.Mult_For_Weight(weight);
    res->Update_With_Res(t);
  }
}

static void PrintTotals(IBenchPrintCallback &f,
    bool showFreq, UInt64 cpuFreq, bool showSpeed, const CTotalBenchRes &res)
{
  const UInt64 numIterations2 = res.NumIterations2 ? res.NumIterations2 : 1;
  const UInt64 speed = res.Speed / numIterations2;
  if (showSpeed && speed != 0)
    PrintNumber(f, speed / 1024, kFieldSize_Speed);
  else
    PrintSpaces(f, 1 + kFieldSize_Speed);

  // PrintSpaces(f, 1 + kFieldSize_Speed);
  // UInt64 numIterations1 = res.NumIterations1; if (numIterations1 == 0) numIterations1 = 1;
  PrintResults(f, res.Usage / numIterations2, res.RPU / numIterations2, res.Rating / numIterations2, showFreq, cpuFreq);
}


static void PrintHex(AString &s, UInt64 v)
{
  char temp[32];
  ConvertUInt64ToHex(v, temp);
  s += temp;
}

AString GetProcessThreadsInfo(const NSystem::CProcessAffinity &ti)
{
  AString s;
  // s.Add_UInt32(ti.numProcessThreads);
  const unsigned numSysThreads = ti.GetNumSystemThreads();
  if (ti.GetNumProcessThreads() != numSysThreads)
  {
    // if (ti.numProcessThreads != ti.numSysThreads)
    {
      s += " / ";
      s.Add_UInt32(numSysThreads);
    }
    s += " : ";
    #ifdef _WIN32
    PrintHex(s, ti.processAffinityMask);
    s += " / ";
    PrintHex(s, ti.systemAffinityMask);
    #else
    unsigned i = (numSysThreads + 3) & ~(unsigned)3;
    if (i == 0)
      i = 4;
    for (; i >= 4; )
    {
      i -= 4;
      unsigned val = 0;
      for (unsigned k = 0; k < 4; k++)
      {
        const unsigned bit = (ti.IsCpuSet(i + k) ? 1 : 0);
        val += (bit << k);
      }
      PrintHex(s, val);
    }
    #endif
  }
#ifdef _WIN32
  if (ti.Groups.GroupSizes.Size() > 1 ||
      (ti.Groups.GroupSizes.Size() == 1
       && ti.Groups.NumThreadsTotal != numSysThreads))
  {
    s += " : ";
    s.Add_UInt32(ti.Groups.GroupSizes.Size());
    s += " groups : ";
    if (ti.Groups.NumThreadsTotal == numSysThreads)
    {
      s.Add_UInt32(ti.Groups.NumThreadsTotal);
      s += " c : ";
    }
    UInt32 minSize, maxSize;
    ti.Groups.Get_GroupSize_Min_Max(minSize, maxSize);
    if (minSize == maxSize)
    {
      s.Add_UInt32(ti.Groups.GroupSizes[0]);
      s += " c/g";
    }
    else
    FOR_VECTOR (i, ti.Groups.GroupSizes)
    {
      if (i != 0)
        s.Add_Space();
      s.Add_UInt32(ti.Groups.GroupSizes[i]);
    }
  }
#endif
  return s;
}


#ifdef Z7_LARGE_PAGES

#ifdef _WIN32
extern bool g_LargePagesMode;
#endif
extern "C"
{
  extern size_t g_LargePageSize;
}

void Add_LargePages_String(AString &s)
{
  #ifdef _WIN32
  if (g_LargePagesMode || g_LargePageSize != 0)
  {
    s.Add_OptSpaced("(LP-");
    PrintSize_KMGT_Or_Hex(s, g_LargePageSize);
    #ifdef MY_CPU_X86_OR_AMD64
    if (CPU_IsSupported_PageGB())
      s += "-1G";
    #endif
    if (!g_LargePagesMode)
      s += "-NA";
    s.Add_Char(')');
  }
  #else
  if (g_LargePageSize != 0)
  {
    s.Add_OptSpaced(" (LP)");
    // PrintSize_KMGT_Or_Hex(s, g_LargePageSize);
  }
  #endif
}

#endif



static void PrintRequirements(IBenchPrintCallback &f, const char *sizeString,
    bool size_Defined, UInt64 size, const char *threadsString, UInt32 numThreads)
{
  f.Print("RAM ");
  f.Print(sizeString);
  if (size_Defined)
    PrintNumber(f, (size >> 20), 6);
  else
    f.Print("      ?");
  f.Print(" MB");
  
  #ifdef Z7_LARGE_PAGES
  {
    AString s;
    Add_LargePages_String(s);
    f.Print(s);
  }
  #endif
  
  f.Print(",  # ");
  f.Print(threadsString);
  PrintNumber(f, numThreads, 3);
}



struct CBenchCallbackToPrint Z7_final: public IBenchCallback
{
  bool NeedPrint;
  bool Use2Columns;
  bool ShowFreq;
  unsigned NameFieldSize;

  unsigned EncodeWeight;
  unsigned DecodeWeight;

  UInt64 CpuFreq;
  UInt64 DictSize;

  IBenchPrintCallback *_file;
  CBenchProps BenchProps;
  CTotalBenchRes EncodeRes;
  CTotalBenchRes DecodeRes;

  CBenchInfo BenchInfo_Results[2];
  
  CBenchCallbackToPrint():
      NeedPrint(true),
      Use2Columns(false),
      ShowFreq(false),
      NameFieldSize(0),
      EncodeWeight(1),
      DecodeWeight(1),
      CpuFreq(0)
      {}

  void Init() { EncodeRes.Init(); DecodeRes.Init(); }
  void Print(const char *s);
  void NewLine();
  
  HRESULT SetFreq(bool showFreq, UInt64 cpuFreq);
  HRESULT SetEncodeResult(const CBenchInfo &info, bool final) Z7_override;
  HRESULT SetDecodeResult(const CBenchInfo &info, bool final) Z7_override;
};

HRESULT CBenchCallbackToPrint::SetFreq(bool showFreq, UInt64 cpuFreq)
{
  ShowFreq = showFreq;
  CpuFreq = cpuFreq;
  return S_OK;
}

HRESULT CBenchCallbackToPrint::SetEncodeResult(const CBenchInfo &info, bool final)
{
  RINOK(_file->CheckBreak())
  if (final)
    BenchInfo_Results[0] = info;
  if (final)
  if (NeedPrint)
  {
    const UInt64 rating = BenchProps.GetRating_Enc(DictSize, info.GlobalTime, info.GlobalFreq, info.UnpackSize * info.NumIterations);
    PrintResults(_file, info,
        EncodeWeight, rating,
        ShowFreq, CpuFreq, &EncodeRes);
    if (!Use2Columns)
      _file->NewLine();
  }
  return S_OK;
}

static const char * const kSep = "  | ";

HRESULT CBenchCallbackToPrint::SetDecodeResult(const CBenchInfo &info, bool final)
{
  RINOK(_file->CheckBreak())
  if (final)
    BenchInfo_Results[1] = info;
  if (final)
  if (NeedPrint)
  {
    const UInt64 rating = BenchProps.GetRating_Dec(info.GlobalTime, info.GlobalFreq, info.UnpackSize, info.PackSize, info.NumIterations);
    if (Use2Columns)
      _file->Print(kSep);
    else
      PrintSpaces(*_file, NameFieldSize);
    CBenchInfo info2 = info;
    info2.UnpackSize *= info2.NumIterations;
    info2.PackSize *= info2.NumIterations;
    info2.NumIterations = 1;
    PrintResults(_file, info2,
        DecodeWeight, rating,
        ShowFreq, CpuFreq, &DecodeRes);
  }
  return S_OK;
}

void CBenchCallbackToPrint::Print(const char *s)
{
  _file->Print(s);
}

void CBenchCallbackToPrint::NewLine()
{
  _file->NewLine();
}

static void PrintLeft(IBenchPrintCallback &f, const char *s, unsigned size)
{
  f.Print(s);
  int numSpaces = (int)size - (int)MyStringLen(s);
  if (numSpaces > 0)
    PrintSpaces(f, (unsigned)numSpaces);
}

static void PrintRight(IBenchPrintCallback &f, const char *s, unsigned size)
{
  int numSpaces = (int)size - (int)MyStringLen(s);
  if (numSpaces > 0)
    PrintSpaces(f, (unsigned)numSpaces);
  f.Print(s);
}


static bool DoesWildcardMatchName_NoCase(const AString &mask, const char *name)
{
  UString wildc = GetUnicodeString(mask);
  UString bname = GetUnicodeString(name);
  wildc.MakeLower_Ascii();
  bname.MakeLower_Ascii();
  return DoesWildcardMatchName(wildc, bname);
}


static HRESULT TotalBench(
    DECL_EXTERNAL_CODECS_LOC_VARS
    const COneMethodInfo &methodMask,
    UInt64 complexInCommands,
  #ifndef Z7_ST
    UInt32 numThreads,
    const CAffinityMode *affinityMode,
  #endif
    bool forceUnpackSize,
    size_t unpackSize,
    const Byte *fileData,
    IBenchPrintCallback *printCallback, CBenchCallbackToPrint *callback)
{
  for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_Bench); i++)
  {
    const CBenchMethod &bench = g_Bench[i];
    if (!DoesWildcardMatchName_NoCase(methodMask.MethodName, bench.Name))
      continue;
    PrintLeft(*callback->_file, bench.Name, kFieldSize_Name);
    {
      unsigned keySize = 32;
           if (IsString1PrefixedByString2(bench.Name, "AES128")) keySize = 16;
      else if (IsString1PrefixedByString2(bench.Name, "AES192")) keySize = 24;
      callback->BenchProps.KeySize = keySize;
    }
    callback->BenchProps.DecComplexUnc = bench.DecComplexUnc;
    callback->BenchProps.DecComplexCompr = bench.DecComplexCompr;
    callback->BenchProps.EncComplex = bench.EncComplex;
    
    COneMethodInfo method;
    NCOM::CPropVariant propVariant;
    propVariant = bench.Name;
    RINOK(method.ParseMethodFromPROPVARIANT(UString(), propVariant))

    size_t unpackSize2 = unpackSize;
    if (!forceUnpackSize && bench.DictBits == 0)
      unpackSize2 = kFilterUnpackSize;

    callback->EncodeWeight = bench.Weight;
    callback->DecodeWeight = bench.Weight;

    const HRESULT res = MethodBench(
        EXTERNAL_CODECS_LOC_VARS
        complexInCommands,
        #ifndef Z7_ST
        false, numThreads, affinityMode,
        #endif
        method,
        unpackSize2, fileData,
        bench.DictBits,
        printCallback, callback, &callback->BenchProps);
    
    if (res == E_NOTIMPL)
    {
      // callback->Print(" ---");
      // we need additional empty line as line for decompression results
      if (!callback->Use2Columns)
        callback->NewLine();
    }
    else
    {
      RINOK(res)
    }
    
    callback->NewLine();
  }
  return S_OK;
}


struct CFreqBench
{
  // in:
  UInt64 complexInCommands;
  UInt32 numThreads;
  bool showFreq;
  UInt64 specifiedFreq;
  
  // out:
  UInt64 CpuFreqRes;
  UInt64 UsageRes;
  UInt32 res;

  CFreqBench()
    {}
    
  HRESULT FreqBench(IBenchPrintCallback *_file
      #ifndef Z7_ST
      , const CAffinityMode *affinityMode
      #endif
      );
};


HRESULT CFreqBench::FreqBench(IBenchPrintCallback *_file
    #ifndef Z7_ST
    , const CAffinityMode *affinityMode
    #endif
    )
{
  res = 0;
  CpuFreqRes = 0;
  UsageRes = 0;

  if (numThreads == 0)
    numThreads = 1;

  #ifdef Z7_ST
  numThreads = 1;
  #endif

  const UInt32 complexity = kNumFreqCommands;
  UInt64 numIterations = complexInCommands / complexity;
  UInt32 numIterations2 = 1 << 30;
  if (numIterations > numIterations2)
    numIterations /= numIterations2;
  else
  {
    numIterations2 = (UInt32)numIterations;
    numIterations = 1;
  }

  CBenchInfoCalc progressInfoSpec;

  #ifndef Z7_ST

  bool mtMode = (numThreads > 1) || affinityMode->NeedAffinity();

  if (mtMode)
  {
    CFreqThreads threads;
    threads.Items = new CFreqInfo[numThreads];
    UInt32 i;
    for (i = 0; i < numThreads; i++)
    {
      CFreqInfo &info = threads.Items[i];
      info.Callback = _file;
      info.CallbackRes = S_OK;
      info.NumIterations = numIterations;
      info.Size = numIterations2;
    }
    progressInfoSpec.SetStartTime();
    for (i = 0; i < numThreads; i++)
    {
      // Sleep(10);
      CFreqInfo &info = threads.Items[i];
      WRes wres = affinityMode->CreateThread_WithAffinity(info.Thread, FreqThreadFunction, &info, i);
      if (info.Thread.IsCreated())
        threads.NumThreads++;
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
    }
    WRes wres = threads.WaitAll();
    if (wres != 0)
      return HRESULT_FROM_WIN32(wres);
    for (i = 0; i < numThreads; i++)
    {
      RINOK(threads.Items[i].CallbackRes)
    }
  }
  else
  #endif
  {
    progressInfoSpec.SetStartTime();
    UInt32 sum = g_BenchCpuFreqTemp;
    UInt64 k = numIterations;
    do
    {
      sum = CountCpuFreq(sum, numIterations2, g_BenchCpuFreqTemp);
      if (_file)
      {
        RINOK(_file->CheckBreak())
      }
    }
    while (--k);
    res += sum;
  }

  if (res == 0x12345678)
  if (_file)
  {
    RINOK(_file->CheckBreak())
  }
  
  CBenchInfo info;
  progressInfoSpec.SetFinishTime(info);

  info.UnpackSize = 0;
  info.PackSize = 0;
  info.NumIterations = 1;

  const UInt64 numCommands = (UInt64)numIterations * numIterations2 * numThreads * complexity;
  const UInt64 rating = info.GetSpeed(numCommands);
  CpuFreqRes = rating / numThreads;
  UsageRes = info.GetUsage();

  if (_file)
  {
    PrintResults(_file, info,
          0, // weight
          rating,
          showFreq, showFreq ? (specifiedFreq != 0 ? specifiedFreq : CpuFreqRes) : 0, NULL);
    RINOK(_file->CheckBreak())
  }

  return S_OK;
}



static HRESULT CrcBench(
    DECL_EXTERNAL_CODECS_LOC_VARS
    UInt64 complexInCommands,
    UInt32 numThreads,
    const size_t bufferSize,
    const Byte *fileData,
    
    UInt64 &speed,
    UInt64 &usage,

    UInt32 complexity, unsigned benchWeight,
    const UInt32 *checkSum,
    const COneMethodInfo &method,
    IBenchPrintCallback *_file,
    #ifndef Z7_ST
    const CAffinityMode *affinityMode,
    #endif
    bool showRating,
    CTotalBenchRes *encodeRes,
    bool showFreq, UInt64 cpuFreq)
{
  if (numThreads == 0)
    numThreads = 1;

  #ifdef Z7_ST
  numThreads = 1;
  #endif

  const AString &methodName = method.MethodName;
  // methodName.RemoveChar(L'-');
  CMethodId hashID;
  if (!FindHashMethod(
      EXTERNAL_CODECS_LOC_VARS
      methodName, hashID))
    return E_NOTIMPL;

  /*
  // if will generate random data in each thread, instead of global data
  CMidAlignedBuffer buffer;
  if (!fileData)
  {
    ALLOC_WITH_HRESULT(&buffer, bufferSize)
    RandGen(buffer, bufferSize);
    fileData = buffer;
  }
  */

  const size_t bsize = (bufferSize == 0 ? 1 : bufferSize);
  UInt64 numIterations = complexInCommands * k_Hash_Complex_Mult / complexity / bsize;
  if (numIterations == 0)
    numIterations = 1;

  CBenchInfoCalc progressInfoSpec;
  CBenchInfo info;

  #ifndef Z7_ST
  bool mtEncMode = (numThreads > 1) || affinityMode->NeedAffinity();

  if (mtEncMode)
  {
    CCrcThreads threads;
    threads.Items = new CCrcInfo[numThreads];
    {
      WRes wres = threads.Common.StartEvent.Create();
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
      threads.NeedClose = true;
    }

    UInt32 i;
    for (i = 0; i < numThreads; i++)
    {
      CCrcInfo &ci = threads.Items[i];
      AString name;
      RINOK(CreateHasher(EXTERNAL_CODECS_LOC_VARS hashID, name, ci.Hasher))
      if (!ci.Hasher)
        return E_NOTIMPL;
      CMyComPtr<ICompressSetCoderProperties> scp;
      ci.Hasher.QueryInterface(IID_ICompressSetCoderProperties, &scp);
      if (scp)
      {
        RINOK(method.SetCoderProps(scp))
      }

      ci.Callback = _file;
      ci.Data = fileData;
      ci.NumIterations = numIterations;
      ci.Size = bufferSize;
      ci.CheckSumDefined = false;
      if (checkSum)
      {
        ci.CheckSum = *checkSum;
        ci.CheckSumDefined = true;
      }

      #ifdef USE_ALLOCA
      ci.AllocaSize = BENCH_ALLOCA_VALUE(i);
      #endif
    }

    for (i = 0; i < numThreads; i++)
    {
      CCrcInfo &ci = threads.Items[i];
      ci.ThreadIndex = i;
      ci.Common = &threads.Common;
      ci.AffinityMode = *affinityMode;
      HRESULT hres = ci.CreateThread();
      if (ci.Thread.IsCreated())
        threads.NumThreads++;
      if (hres != 0)
        return hres;
    }
    
    for (i = 0; i < numThreads; i++)
    {
      CCrcInfo &ci = threads.Items[i];
      WRes wres = ci.ReadyEvent.Lock();
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
      RINOK(ci.Res)
    }

    progressInfoSpec.SetStartTime();

    WRes wres = threads.StartAndWait();
    if (wres != 0)
      return HRESULT_FROM_WIN32(wres);

    progressInfoSpec.SetFinishTime(info);
    
    for (i = 0; i < numThreads; i++)
    {
      RINOK(threads.Items[i].Res)
      if (i != 0)
        if (threads.Items[i].CheckSum_Res !=
            threads.Items[i - 1].CheckSum_Res)
          return S_FALSE;
    }
  }
  else
  #endif
  {
    CMyComPtr<IHasher> hasher;
    AString name;
    RINOK(CreateHasher(EXTERNAL_CODECS_LOC_VARS hashID, name, hasher))
    if (!hasher)
      return E_NOTIMPL;
    CMyComPtr<ICompressSetCoderProperties> scp;
    hasher.QueryInterface(IID_ICompressSetCoderProperties, &scp);
    if (scp)
    {
      RINOK(method.SetCoderProps(scp))
    }
    CCrcInfo_Base crcib;
    crcib.CreateLocalBuf = false;
    RINOK(crcib.Generate(fileData, bufferSize))
    progressInfoSpec.SetStartTime();
    RINOK(crcib.CrcProcess(numIterations, checkSum, hasher, _file))
    progressInfoSpec.SetFinishTime(info);
  }


  UInt64 unpSize = numIterations * bufferSize;
  UInt64 unpSizeThreads = unpSize * numThreads;
  info.UnpackSize = unpSizeThreads;
  info.PackSize = unpSizeThreads;
  info.NumIterations = 1;

  if (_file)
  {
    if (showRating)
    {
      UInt64 unpSizeThreads2 = unpSizeThreads;
      if (unpSizeThreads2 == 0)
        unpSizeThreads2 = numIterations * 1 * numThreads;
      const UInt64 numCommands = unpSizeThreads2 * complexity / 256;
      const UInt64 rating = info.GetSpeed(numCommands);
      PrintResults(_file, info,
          benchWeight, rating,
          showFreq, cpuFreq, encodeRes);
    }
    RINOK(_file->CheckBreak())
  }

  speed = info.GetSpeed(unpSizeThreads);
  usage = info.GetUsage();

  return S_OK;
}



static HRESULT TotalBench_Hash(
    DECL_EXTERNAL_CODECS_LOC_VARS
    const COneMethodInfo &methodMask,
    UInt64 complexInCommands,
    UInt32 numThreads,
    size_t bufSize,
    const Byte *fileData,
    IBenchPrintCallback *printCallback, CBenchCallbackToPrint *callback,
    #ifndef Z7_ST
    const CAffinityMode *affinityMode,
    #endif
    CTotalBenchRes *encodeRes,
    bool showFreq, UInt64 cpuFreq)
{
  for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_Hash); i++)
  {
    const CBenchHash &bench = g_Hash[i];
    if (!DoesWildcardMatchName_NoCase(methodMask.MethodName, bench.Name))
      continue;
    PrintLeft(*callback->_file, bench.Name, kFieldSize_Name);
    // callback->BenchProps.DecComplexUnc = bench.DecComplexUnc;
    // callback->BenchProps.DecComplexCompr = bench.DecComplexCompr;
    // callback->BenchProps.EncComplex = bench.EncComplex;

    COneMethodInfo method;
    NCOM::CPropVariant propVariant;
    propVariant = bench.Name;
    RINOK(method.ParseMethodFromPROPVARIANT(UString(), propVariant))

    UInt64 speed, usage;

    const HRESULT res = CrcBench(
        EXTERNAL_CODECS_LOC_VARS
        complexInCommands,
        numThreads, bufSize, fileData,
        speed, usage,
        bench.Complex, bench.Weight,
        (!fileData && bufSize == (1 << kNumHashDictBits)) ? &bench.CheckSum : NULL,
        method,
        printCallback,
     #ifndef Z7_ST
        affinityMode,
     #endif
        true, // showRating
        encodeRes, showFreq, cpuFreq);
    if (res == E_NOTIMPL)
    {
      // callback->Print(" ---");
    }
    else
    {
      RINOK(res)
    }
    callback->NewLine();
  }
  return S_OK;
}

struct CTempValues
{
  UInt64 *Values;
  CTempValues(): Values(NULL) {}
  void Alloc(UInt32 num) { Values = new UInt64[num]; }
  ~CTempValues() { delete []Values; }
};

static void ParseNumberString(const UString &s, NCOM::CPropVariant &prop)
{
  const wchar_t *end;
  UInt64 result = ConvertStringToUInt64(s, &end);
  if (*end != 0 || s.IsEmpty())
    prop = s;
  else if (result <= (UInt32)0xFFFFFFFF)
    prop = (UInt32)result;
  else
    prop = result;
}


static bool AreSameMethodNames(const char *fullName, const char *shortName)
{
  return StringsAreEqualNoCase_Ascii(fullName, shortName);
}




static void Print_Usage_and_Threads(IBenchPrintCallback &f, UInt64 usage, UInt32 threads)
{
  PrintRequirements(f, "usage:", true, usage, "Benchmark threads:   ", threads);
}


static void Print_Delimiter(IBenchPrintCallback &f)
{
  f.Print(" |");
}

static void Print_Pow(IBenchPrintCallback &f, unsigned pow)
{
  char s[16];
  ConvertUInt32ToString(pow, s);
  unsigned pos = MyStringLen(s);
  s[pos++] = ':';
  s[pos] = 0;
  PrintLeft(f, s, kFieldSize_SmallName); // 4
}

static void Bench_BW_Print_Usage_Speed(IBenchPrintCallback &f,
    UInt64 usage, UInt64 speed)
{
  PrintUsage(f, usage, kFieldSize_Usage);
  PrintNumber(f, speed / 1000000, kFieldSize_CrcSpeed);
}


HRESULT Bench(
    DECL_EXTERNAL_CODECS_LOC_VARS
    IBenchPrintCallback *printCallback,
    IBenchCallback *benchCallback,
    const CObjectVector<CProperty> &props,
    UInt32 numIterations,
    bool multiDict,
    IBenchFreqCallback *freqCallback)
{
  // for (int y = 0; y < 10000; y++)
  if (!CrcInternalTest())
    return E_FAIL;

  UInt32 numCPUs = 1;
  size_t ramSize = (size_t)sizeof(size_t) << 29;

  NSystem::CProcessAffinity threadsInfo;
  threadsInfo.InitST();

  #ifndef Z7_ST

  if (!threadsInfo.Get()
      || (numCPUs = threadsInfo.GetNumProcessThreads()) == 0)
    numCPUs = NSystem::GetNumberOfProcessors();
  // numCPUs : is number of threads assigned to process with affinity,
  // or it's total number of threads in all groups, if IsGroupMode == true, and there is default affinity.

  #endif

  // numCPUs = 24;
  /*
  {
    DWORD_PTR mask = (1 << 0);
    DWORD_PTR old = SetThreadAffinityMask(GetCurrentThread(), mask);
    old = old;
    DWORD_PTR old2 = SetThreadAffinityMask(GetCurrentThread(), mask);
    old2 = old2;
    return 0;
  }
  */
  
  const bool ramSize_Defined = NSystem::GetRamSize(ramSize);

  UInt32 numThreadsSpecified = numCPUs;
  bool needSetComplexity = false;
  UInt32 testTimeMs = kComplexInMs;
  UInt32 startDicLog = 22;
  bool startDicLog_Defined = false;
  UInt64 specifiedFreq = 0;
  bool multiThreadTests = false;
  UInt64 complexInCommands = kComplexInCommands;
  UInt32 numThreads_Start = 1;
  
#ifndef Z7_ST
  CAffinityMode affinityMode;
#ifdef _WIN32
  if (threadsInfo.IsGroupMode && threadsInfo.Groups.GroupSizes.Size() > 1)
    affinityMode.NumGroups = threadsInfo.Groups.GroupSizes.Size();
#endif
#endif


  COneMethodInfo method;

  CMidAlignedBuffer fileDataBuffer;
  bool use_fileData = false;
  bool isFixedDict = false;

  {
  unsigned i;

  if (printCallback)
  {
    for (i = 0; i < props.Size(); i++)
    {
      const CProperty &property = props[i];
      printCallback->Print(" ");
      printCallback->Print(GetAnsiString(property.Name));
      if (!property.Value.IsEmpty())
      {
        printCallback->Print("=");
        printCallback->Print(GetAnsiString(property.Value));
      }
    }
    if (!props.IsEmpty())
      printCallback->NewLine();
  }


  for (i = 0; i < props.Size(); i++)
  {
    const CProperty &property = props[i];
    UString name (property.Name);
    name.MakeLower_Ascii();

    if (name.IsEqualTo("file"))
    {
      if (property.Value.IsEmpty())
        return E_INVALIDARG;

      NFile::NIO::CInFile file;
      if (!file.Open(us2fs(property.Value)))
        return GetLastError_noZero_HRESULT();
      size_t len;
      {
        UInt64 len64;
        if (!file.GetLength(len64))
          return GetLastError_noZero_HRESULT();
        if (printCallback)
        {
          printCallback->Print("file size =");
          PrintNumber(*printCallback, len64, 0);
          printCallback->NewLine();
        }
        len = (size_t)len64;
        if (len != len64)
          return E_INVALIDARG;
      }

      // (len == 0) is allowed. Also it's allowed if Alloc(0) returns NULL here

      ALLOC_WITH_HRESULT(&fileDataBuffer, len)
      use_fileData = true;
      
      {
        size_t processed;
        if (!file.ReadFull((Byte *)fileDataBuffer, len, processed))
          return GetLastError_noZero_HRESULT();
        if (processed != len)
          return E_FAIL;
      }
      continue;
    }

    NCOM::CPropVariant propVariant;
    if (!property.Value.IsEmpty())
      ParseNumberString(property.Value, propVariant);
    
    if (name.IsEqualTo("time"))
    {
      RINOK(ParsePropToUInt32(UString(), propVariant, testTimeMs))
      needSetComplexity = true;
      testTimeMs *= 1000;
      continue;
    }

    if (name.IsEqualTo("timems"))
    {
      RINOK(ParsePropToUInt32(UString(), propVariant, testTimeMs))
      needSetComplexity = true;
      continue;
    }

    if (name.IsEqualTo("tic"))
    {
      UInt32 v;
      RINOK(ParsePropToUInt32(UString(), propVariant, v))
      if (v >= 64)
        return E_INVALIDARG;
      complexInCommands = (UInt64)1 << v;
      continue;
    }

    const bool isCurrent_fixedDict = name.IsEqualTo("df");
    if (isCurrent_fixedDict)
      isFixedDict = true;
    if (isCurrent_fixedDict || name.IsEqualTo("ds"))
    {
      RINOK(ParsePropToUInt32(UString(), propVariant, startDicLog))
      if (startDicLog > 32)
        return E_INVALIDARG;
      startDicLog_Defined = true;
      continue;
    }

    if (name.IsEqualTo("mts"))
    {
      RINOK(ParsePropToUInt32(UString(), propVariant, numThreads_Start))
      continue;
    }

    if (name.IsEqualTo("af"))
    {
      UInt32 bundle;
      RINOK(ParsePropToUInt32(UString(), propVariant, bundle))
      if (bundle > 0 && bundle < numCPUs)
      {
        #ifndef Z7_ST
        affinityMode.SetLevels(numCPUs, 2);
        affinityMode.NumBundleThreads = bundle;
        #endif
      }
      continue;
    }
   
    if (name.IsEqualTo("freq"))
    {
      UInt32 freq32 = 0;
      RINOK(ParsePropToUInt32(UString(), propVariant, freq32))
      if (freq32 == 0)
        return E_INVALIDARG;
      specifiedFreq = (UInt64)freq32 * 1000000;

      if (printCallback)
      {
        printCallback->Print("freq=");
        PrintNumber(*printCallback, freq32, 0);
        printCallback->NewLine();
      }

      continue;
    }

    if (name.IsPrefixedBy_Ascii_NoCase("mt"))
    {
      const UString s = name.Ptr(2);
      if (s.IsEqualTo("*")
          || (s.IsEmpty()
            && propVariant.vt == VT_BSTR
            && StringsAreEqual_Ascii(propVariant.bstrVal, "*")))
      {
        multiThreadTests = true;
        continue;
      }
      #ifndef Z7_ST
      RINOK(ParseMtProp(s, propVariant, numCPUs, numThreadsSpecified))
      #endif
      continue;
    }
    
    RINOK(method.ParseMethodFromPROPVARIANT(name, propVariant))
  }
  }

  if (printCallback)
  {
    AString s;
    
#if 1 || !defined(Z7_MSC_VER_ORIGINAL) || (Z7_MSC_VER_ORIGINAL >= 1900)
    s += "Compiler: ";
    GetCompiler(s);
    printCallback->Print(s);
    printCallback->NewLine();
    s.Empty();
#endif

    GetSystemInfoText(s);
    printCallback->Print(s);
    printCallback->NewLine();
  }

  if (printCallback)
  {
    printCallback->Print("1T CPU Freq (MHz):");
  }

  if (printCallback || freqCallback)
  {
    UInt64 numMilCommands = 1 << 6;
    if (specifiedFreq != 0)
    {
      while (numMilCommands > 1 && specifiedFreq < (numMilCommands * 1000000))
        numMilCommands >>= 1;
    }

    for (int jj = 0;; jj++)
    {
      if (printCallback)
        RINOK(printCallback->CheckBreak())

      UInt64 start = ::GetTimeCount();
      UInt32 sum = (UInt32)start;
      sum = CountCpuFreq(sum, (UInt32)(numMilCommands * 1000000 / kNumFreqCommands), g_BenchCpuFreqTemp);
      if (sum == 0xF1541213)
        if (printCallback)
          printCallback->Print("");
      const UInt64 realDelta = ::GetTimeCount() - start;
      start = realDelta;
      if (start == 0)
        start = 1;
      if (start > (UInt64)1 << 61)
        start = 1;
      const UInt64 freq = GetFreq();
      // mips is constant in some compilers
      const UInt64 hzVal = MyMultDiv64(numMilCommands * 1000000, freq, start);
      const UInt64 mipsVal = numMilCommands * freq / start;
      if (printCallback)
      {
        if (realDelta == 0)
        {
          printCallback->Print(" -");
        }
        else
        {
          // PrintNumber(*printCallback, start, 0);
          PrintNumber(*printCallback, mipsVal, 5);
        }
      }
      if (freqCallback)
      {
        RINOK(freqCallback->AddCpuFreq(1, hzVal, kBenchmarkUsageMult))
      }

      if (jj >= 1)
      {
        bool needStop = (numMilCommands >= (1 <<
          #ifdef _DEBUG
            7
          #else
            11
          #endif
          ));
        if (start >= freq * 16)
        {
          printCallback->Print(" (Cmplx)");
          if (!freqCallback) // we don't want complexity change for old gui lzma benchmark
          {
            needSetComplexity = true;
          }
          needStop = true;
        }
        if (needSetComplexity)
          SetComplexCommandsMs(testTimeMs, false, mipsVal * 1000000, complexInCommands);
        if (needStop)
          break;
        numMilCommands <<= 1;
      }
    }
    if (freqCallback)
    {
      RINOK(freqCallback->FreqsFinished(1))
    }
  }

  if (printCallback || freqCallback)
  for (unsigned test = 0; test < 3; test++)
  {
    if (numThreadsSpecified < 2)
    {
      // if (test == 1)
      break;
    }
    if (test == 2 && numThreadsSpecified <= numCPUs)
      break;
    if (printCallback)
      printCallback->NewLine();

    /* it can show incorrect frequency for HT threads. */

    UInt32 numThreads = numThreadsSpecified;
    if (test < 2)
    {
      if (numThreads >= numCPUs)
        numThreads = numCPUs;
      if (test == 0)
        numThreads /= 2;
    }
    if (numThreads < 1)
      numThreads = 1;
   
    if (printCallback)
    {
      char s[128];
      ConvertUInt64ToString(numThreads, s);
      printCallback->Print(s);
      printCallback->Print("T CPU Freq (MHz):");
    }
    UInt64 numMilCommands = 1 <<
          #ifdef _DEBUG
            7;
          #else
            10;
          #endif

    if (specifiedFreq != 0)
    {
      while (numMilCommands > 1 && specifiedFreq < (numMilCommands * 1000000))
        numMilCommands >>= 1;
    }

    // for (int jj = 0;; jj++)
    for (;;)
    {
      if (printCallback)
        RINOK(printCallback->CheckBreak())

      {
        // PrintLeft(f, "CPU", kFieldSize_Name);
        
        // UInt32 resVal;
        
        CFreqBench fb;
        fb.complexInCommands = numMilCommands * 1000000;
        fb.numThreads = numThreads;
        // showFreq;
        // fb.showFreq = (freqTest == kNumCpuTests - 1 || specifiedFreq != 0);
        fb.showFreq = true;
        fb.specifiedFreq = 1;

        const HRESULT res = fb.FreqBench(NULL /* printCallback */
            #ifndef Z7_ST
              , &affinityMode
            #endif
            );
        RINOK(res)

        if (freqCallback)
        {
          RINOK(freqCallback->AddCpuFreq(numThreads, fb.CpuFreqRes, fb.UsageRes))
        }

        if (printCallback)
        {
          /*
          if (realDelta == 0)
          {
            printCallback->Print(" -");
          }
          else
          */
          {
            // PrintNumber(*printCallback, start, 0);
            PrintUsage(*printCallback, fb.UsageRes, 3);
            printCallback->Print("%");
            PrintNumber(*printCallback, fb.CpuFreqRes / 1000000, 0);
            printCallback->Print("  ");

            // PrintNumber(*printCallback, fb.UsageRes, 5);
          }
        }
      }
      // if (jj >= 1)
      {
        const bool needStop = (numMilCommands >= (1 <<
          #ifdef _DEBUG
            7
          #else
            11
          #endif
          ));
        if (needStop)
          break;
        numMilCommands <<= 1;
      }
    }
    if (freqCallback)
    {
      RINOK(freqCallback->FreqsFinished(numThreads))
    }
  }


  if (printCallback)
  {
    printCallback->NewLine();
    printCallback->NewLine();
    PrintRequirements(*printCallback, "size: ", ramSize_Defined, ramSize, "CPU hardware threads:", numCPUs);
    printCallback->Print(GetProcessThreadsInfo(threadsInfo));
    printCallback->NewLine();
  }

  if (numThreadsSpecified < 1 || numThreadsSpecified > kNumThreadsMax)
    return E_INVALIDARG;

  UInt64 dict = (UInt64)1 << startDicLog;
  const bool dictIsDefined = (isFixedDict || method.Get_DicSize(dict));

  const unsigned level = method.GetLevel();

  AString &methodName = method.MethodName;
  const AString original_MethodName = methodName;
  if (methodName.IsEmpty())
    methodName = "LZMA";

  if (benchCallback)
  {
    CBenchProps benchProps;
    benchProps.SetLzmaCompexity();
    const UInt64 dictSize = method.Get_Lzma_DicSize();

    size_t uncompressedDataSize;
    if (use_fileData)
    {
      uncompressedDataSize = fileDataBuffer.Size();
    }
    else
    {
      uncompressedDataSize = kAdditionalSize + (size_t)dictSize;
      if (uncompressedDataSize < dictSize)
        return E_INVALIDARG;
    }

    return MethodBench(
        EXTERNAL_CODECS_LOC_VARS
        complexInCommands,
      #ifndef Z7_ST
        true, numThreadsSpecified,
        &affinityMode,
      #endif
        method,
        uncompressedDataSize, (const Byte *)fileDataBuffer,
        kOldLzmaDictBits, printCallback, benchCallback, &benchProps);
  }

  if (methodName.IsEqualTo_Ascii_NoCase("CRC"))
    methodName = "crc32";

  CMethodId hashID;
  const bool isHashMethod = FindHashMethod(EXTERNAL_CODECS_LOC_VARS methodName, hashID);
  int codecIndex = -1;
  bool isFilter = false;
  if (!isHashMethod)
  {
    UInt32 numStreams;
    codecIndex = FindMethod_Index(EXTERNAL_CODECS_LOC_VARS original_MethodName,
        true,  // encode
        hashID, numStreams, isFilter);
    // we can allow non filter for BW tests
    if (!isFilter) codecIndex = -1;
  }

  CBenchCallbackToPrint callback;
  callback.Init();
  callback._file = printCallback;

  if (isHashMethod || codecIndex != -1)
  {
    if (!printCallback)
      return S_FALSE;
    IBenchPrintCallback &f = *printCallback;

    UInt64 dict64 = dict;
    if (!dictIsDefined)
      dict64 = (1 << 27);
    if (use_fileData)
    {
      if (!dictIsDefined)
        dict64 = fileDataBuffer.Size();
      else if (dict64 > fileDataBuffer.Size())
        dict64 = fileDataBuffer.Size();
    }

    for (;;)
    {
      const int index = method.FindProp(NCoderPropID::kDictionarySize);
      if (index < 0)
        break;
      method.Props.Delete((unsigned)index);
    }

    // methodName.RemoveChar(L'-');
    Int32 complexity = 16 * k_Hash_Complex_Mult; // for unknown hash method
    const UInt32 *checkSum = NULL;
    int benchIndex = -1;
    
    if (isHashMethod)
    {
      for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_Hash); i++)
      {
        const CBenchHash &h = g_Hash[i];
        AString benchMethod (h.Name);
        AString benchProps;
        const int propPos = benchMethod.Find(':');
        if (propPos >= 0)
        {
          benchProps = benchMethod.Ptr((unsigned)(propPos + 1));
          benchMethod.DeleteFrom((unsigned)propPos);
        }

        if (AreSameMethodNames(benchMethod, methodName))
        {
          const bool sameProps = method.PropsString.IsEqualTo_Ascii_NoCase(benchProps);
          /*
          bool isMainMethod = method.PropsString.IsEmpty();
          if (isMainMethod)
            isMainMethod = !checkSum
                || (benchMethod.IsEqualTo_Ascii_NoCase("crc32") && benchProps.IsEqualTo_Ascii_NoCase("8"));
          if (sameProps || isMainMethod)
          */
          {
            complexity = (Int32)h.Complex;
            checkSum = &h.CheckSum;
            if (sameProps)
              break;
            /*
            if property. is not specified, we use the complexity
            for latest fastest method (crc32:64)
            */
          }
        }
      }
      // if (!checkSum) return E_NOTIMPL;
    }
    else
    {
      for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_Bench); i++)
      {
        const CBenchMethod &bench = g_Bench[i];
        AString benchMethod (bench.Name);
        AString benchProps;
        const int propPos = benchMethod.Find(':');
        if (propPos >= 0)
        {
          benchProps = benchMethod.Ptr((unsigned)(propPos + 1));
          benchMethod.DeleteFrom((unsigned)propPos);
        }

        if (AreSameMethodNames(benchMethod, methodName))
        {
          const bool sameProps = method.PropsString.IsEqualTo_Ascii_NoCase(benchProps);
          // bool isMainMethod = method.PropsString.IsEmpty();
          // if (sameProps || isMainMethod)
          {
            benchIndex = (int)i;
            if (sameProps)
              break;
          }
        }
      }
      // if (benchIndex < 0) return E_NOTIMPL;
    }

    {
      /* we count usage only for crc and filter. non-filters are not supported */
      UInt64 usage = (1 << 20);
      UInt64 bufSize = dict64;
      UInt32 numBlocks = isHashMethod ? 1 : 3;
      if (use_fileData)
      {
        usage += fileDataBuffer.Size();
        if (bufSize > fileDataBuffer.Size())
          bufSize = fileDataBuffer.Size();
        if (isHashMethod)
        {
          numBlocks = 0;
          #ifndef Z7_ST
          if (numThreadsSpecified != 1)
            numBlocks = (k_Crc_CreateLocalBuf_For_File ? 1 : 0);
          #endif
        }
      }
      usage += numThreadsSpecified * bufSize * numBlocks;
      Print_Usage_and_Threads(f, usage, numThreadsSpecified);
    }

    CUIntVector numThreadsVector;
    {
      unsigned nt = numThreads_Start;
      for (;;)
      {
        if (nt > numThreadsSpecified)
          break;
        numThreadsVector.Add(nt);
        const unsigned next = nt * 2;
        const UInt32 ntHalf= numThreadsSpecified / 2;
        if (ntHalf > nt && ntHalf < next)
          numThreadsVector.Add(ntHalf);
        if (numThreadsSpecified > nt && numThreadsSpecified < next)
          numThreadsVector.Add(numThreadsSpecified);
        nt = next;
      }
    }

    unsigned numColumns = isHashMethod ? 1 : 2;
    CTempValues speedTotals;
    CTempValues usageTotals;
    {
      const unsigned numItems = numThreadsVector.Size() * numColumns;
      speedTotals.Alloc(numItems);
      usageTotals.Alloc(numItems);
      for (unsigned i = 0; i < numItems; i++)
      {
        speedTotals.Values[i] = 0;
        usageTotals.Values[i] = 0;
      }
    }

    f.NewLine();
    for (unsigned line = 0; line < 3; line++)
    {
      f.NewLine();
      f.Print(line == 0 ? "THRD" : line == 1 ? "    " : "Size");
      FOR_VECTOR (ti, numThreadsVector)
      {
        if (ti != 0)
          Print_Delimiter(f);
        if (line == 0)
        {
          PrintSpaces(f, (kFieldSize_CrcSpeed + kFieldSize_Usage + 2) * (numColumns - 1));
          PrintNumber(f, numThreadsVector[ti], 1 + kFieldSize_Usage + kFieldSize_CrcSpeed);
        }
        else
        {
          for (unsigned c = 0; c < numColumns; c++)
          {
            PrintRight(f, line == 1 ? "Usage" : "%",    kFieldSize_Usage + 1);
            PrintRight(f, line == 1 ? "BW"    : "MB/s", kFieldSize_CrcSpeed + 1);
          }
        }
      }
    }
    f.NewLine();

    UInt64 numSteps = 0;

    // for (UInt32 iter = 0; iter < numIterations; iter++)
    // {
    unsigned pow = 10; // kNumHashDictBits
    if (startDicLog_Defined)
      pow = startDicLog;

    // #define NUM_SUB_BITS 2
    // pow <<= NUM_SUB_BITS;
    for (;; pow++)
    {
      const UInt64 bufSize = (UInt64)1 << pow;
      // UInt64 bufSize = (UInt64)1 << (pow >> NUM_SUB_BITS);
      // bufSize += ((UInt64)pow & ((1 << NUM_SUB_BITS) - 1)) << ((pow >> NUM_SUB_BITS) - NUM_SUB_BITS);

      size_t dataSize = fileDataBuffer.Size();
      if (dataSize > bufSize || !use_fileData)
        dataSize = (size_t)bufSize;
      
      for (UInt32 iter = 0; iter < numIterations; iter++)
      {
        Print_Pow(f, pow);
        // PrintNumber(f, bufSize >> 10, 4);
 
        FOR_VECTOR (ti, numThreadsVector)
        {
          RINOK(f.CheckBreak())
          const UInt32 numThreads = numThreadsVector[ti];
          if (isHashMethod)
          {
            UInt64 speed = 0;
            UInt64 usage = 0;
            const HRESULT res = CrcBench(EXTERNAL_CODECS_LOC_VARS complexInCommands,
              numThreads,
              dataSize, (const Byte *)fileDataBuffer,
              speed, usage,
              (UInt32)complexity,
              1, // benchWeight,
              (pow == kNumHashDictBits && !use_fileData) ? checkSum : NULL,
              method,
              &f,
            #ifndef Z7_ST
              &affinityMode,
            #endif
              false, // showRating
              NULL, false, 0);
            RINOK(res)
            
            if (ti != 0)
              Print_Delimiter(f);
            
            Bench_BW_Print_Usage_Speed(f, usage, speed);
            speedTotals.Values[ti] += speed;
            usageTotals.Values[ti] += usage;
          }
          else
          {
            {
              unsigned keySize = 32;
                   if (IsString1PrefixedByString2(methodName, "AES128")) keySize = 16;
              else if (IsString1PrefixedByString2(methodName, "AES192")) keySize = 24;
              callback.BenchProps.KeySize = keySize;
            }

            COneMethodInfo method2 = method;
            unsigned bench_DictBits;

            if (benchIndex >= 0)
            {
              const CBenchMethod &bench = g_Bench[benchIndex];
              callback.BenchProps.EncComplex = bench.EncComplex;
              callback.BenchProps.DecComplexUnc = bench.DecComplexUnc;
              callback.BenchProps.DecComplexCompr = bench.DecComplexCompr;
              bench_DictBits = bench.DictBits;
              // bench_DictBits = kOldLzmaDictBits; = 32 default : for debug
            }
            else
            {
              bench_DictBits = kOldLzmaDictBits; // = 32 default
              if (isFilter)
              {
                const unsigned k_UnknownCoderComplexity = 4;
                callback.BenchProps.EncComplex = k_UnknownCoderComplexity;
                callback.BenchProps.DecComplexUnc = k_UnknownCoderComplexity;
              }
              else
              {
                callback.BenchProps.EncComplex = 1 << 10;
                callback.BenchProps.DecComplexUnc = 1 << 6;
              }
              callback.BenchProps.DecComplexCompr = 0;
            }
            callback.NeedPrint = false;

            if (StringsAreEqualNoCase_Ascii(method2.MethodName, "LZMA"))
            {
              const NCOM::CPropVariant propVariant = (UInt32)pow;
              RINOK(method2.ParseMethodFromPROPVARIANT((UString)"d", propVariant))
            }

            const HRESULT res = MethodBench(
                EXTERNAL_CODECS_LOC_VARS
                complexInCommands,
              #ifndef Z7_ST
                false, // oldLzmaBenchMode
                numThreadsVector[ti],
                &affinityMode,
              #endif
                method2,
                dataSize, (const Byte *)fileDataBuffer,
                bench_DictBits,
                printCallback,
                &callback,
                &callback.BenchProps);
            RINOK(res)

            if (ti != 0)
              Print_Delimiter(f);

            for (unsigned i = 0; i < 2; i++)
            {
              const CBenchInfo &bi = callback.BenchInfo_Results[i];
              const UInt64 usage = bi.GetUsage();
              const UInt64 speed = bi.GetUnpackSizeSpeed();
              usageTotals.Values[ti * 2 + i] += usage;
              speedTotals.Values[ti * 2 + i] += speed;
              Bench_BW_Print_Usage_Speed(f, usage, speed);
            }
          }
        }

        f.NewLine();
        numSteps++;
      }
      if (dataSize >= dict64)
        break;
    }

    if (numSteps != 0)
    {
      f.Print("Avg:");
      for (unsigned ti = 0; ti < numThreadsVector.Size(); ti++)
      {
        if (ti != 0)
          Print_Delimiter(f);
        for (unsigned i = 0; i < numColumns; i++)
          Bench_BW_Print_Usage_Speed(f,
              usageTotals.Values[ti * numColumns + i] / numSteps,
              speedTotals.Values[ti * numColumns + i] / numSteps);
      }
      f.NewLine();
    }

    return S_OK;
  }

  bool use2Columns = false;

  bool totalBenchMode = false;
  bool onlyHashBench = false;
  if (methodName.IsEqualTo_Ascii_NoCase("hash"))
  {
    onlyHashBench = true;
    methodName = "*";
    totalBenchMode = true;
  }
  else if (methodName.Find('*') >= 0)
    totalBenchMode = true;

  // ---------- Threads loop ----------
  for (unsigned threadsPassIndex = 0; threadsPassIndex < 3; threadsPassIndex++)
  {

  UInt32 numThreads = numThreadsSpecified;
    
  if (!multiThreadTests)
  {
    if (threadsPassIndex != 0)
      break;
  }
  else
  {
    numThreads = 1;
    if (threadsPassIndex != 0)
    {
      if (numCPUs < 2)
        break;
      numThreads = numCPUs;
      if (threadsPassIndex == 1)
      {
        if (numCPUs >= 4)
          numThreads = numCPUs / 2;
      }
      else if (numCPUs < 4)
        break;
    }
  }
 
  IBenchPrintCallback &f = *printCallback;

  if (threadsPassIndex > 0)
  {
    f.NewLine();
    f.NewLine();
  }

  if (!dictIsDefined && !onlyHashBench)
  {
    // we use dicSizeLog and dicSizeLog_Main for data size.
    // also we use it to reduce dictionary size of LZMA encoder via NCoderPropID::kReduceSize.
    const unsigned dicSizeLog_Main = (totalBenchMode ? 24 : 25);
    unsigned dicSizeLog = dicSizeLog_Main;
    
    #ifdef UNDER_CE
    dicSizeLog = (UInt64)1 << 20;
    #endif

    if (ramSize_Defined)
    for (; dicSizeLog > kBenchMinDicLogSize; dicSizeLog--)
      if (GetBenchMemoryUsage(numThreads, (int)level, ((UInt64)1 << dicSizeLog), totalBenchMode) + (8 << 20) <= ramSize)
        break;

    dict = (UInt64)1 << dicSizeLog;

    if (totalBenchMode && dicSizeLog != dicSizeLog_Main)
    {
      f.Print("Dictionary reduced to: ");
      PrintNumber(f, dicSizeLog, 1);
      f.NewLine();
    }
  }

  Print_Usage_and_Threads(f,
      onlyHashBench ?
        GetBenchMemoryUsage_Hash(numThreads, dict) :
        GetBenchMemoryUsage(numThreads, (int)level, dict, totalBenchMode),
      numThreads);

  f.NewLine();

  f.NewLine();

  if (totalBenchMode)
  {
    callback.NameFieldSize = kFieldSize_Name;
    use2Columns = false;
  }
  else
  {
    callback.NameFieldSize = kFieldSize_SmallName;
    use2Columns = true;
  }
  callback.Use2Columns = use2Columns;

  bool showFreq = false;
  UInt64 cpuFreq = 0;

  if (totalBenchMode)
  {
    showFreq = true;
  }

  unsigned fileldSize = kFieldSize_TotalSize;
  if (showFreq)
    fileldSize += kFieldSize_EUAndEffec;

  if (use2Columns)
  {
    PrintSpaces(f, callback.NameFieldSize);
    PrintRight(f, "Compressing", fileldSize);
    f.Print(kSep);
    PrintRight(f, "Decompressing", fileldSize);
  }
  f.NewLine();
  PrintLeft(f, totalBenchMode ? "Method" : "Dict", callback.NameFieldSize);

  int j;

  for (j = 0; j < 2; j++)
  {
    PrintRight(f, "Speed", kFieldSize_Speed + 1);
    PrintRight(f, "Usage", kFieldSize_Usage + 1);
    PrintRight(f, "R/U", kFieldSize_RU + 1);
    PrintRight(f, "Rating", kFieldSize_Rating + 1);
    if (showFreq)
    {
      PrintRight(f, "E/U", kFieldSize_EU + 1);
      PrintRight(f, "Effec", kFieldSize_Effec + 1);
    }
    if (!use2Columns)
      break;
    if (j == 0)
      f.Print(kSep);
  }
  
  f.NewLine();
  PrintSpaces(f, callback.NameFieldSize);
  
  for (j = 0; j < 2; j++)
  {
    PrintRight(f, "KiB/s", kFieldSize_Speed + 1);
    PrintRight(f, "%", kFieldSize_Usage + 1);
    PrintRight(f, "MIPS", kFieldSize_RU + 1);
    PrintRight(f, "MIPS", kFieldSize_Rating + 1);
    if (showFreq)
    {
      PrintRight(f, "%", kFieldSize_EU + 1);
      PrintRight(f, "%", kFieldSize_Effec + 1);
    }
    if (!use2Columns)
      break;
    if (j == 0)
      f.Print(kSep);
  }
  
  f.NewLine();
  f.NewLine();

  if (specifiedFreq != 0)
    cpuFreq = specifiedFreq;

  // bool showTotalSpeed = false;

  if (totalBenchMode)
  {
    for (UInt32 i = 0; i < numIterations; i++)
    {
      if (i != 0)
        printCallback->NewLine();

      const unsigned kNumCpuTests = 3;
      for (unsigned freqTest = 0; freqTest < kNumCpuTests; freqTest++)
      {
        PrintLeft(f, "CPU", kFieldSize_Name);
        
        // UInt32 resVal;
        
        CFreqBench fb;
        fb.complexInCommands = complexInCommands;
        fb.numThreads = numThreads;
        // showFreq;
        fb.showFreq = (freqTest == kNumCpuTests - 1 || specifiedFreq != 0);
        fb.specifiedFreq = specifiedFreq;

        const HRESULT res = fb.FreqBench(printCallback
            #ifndef Z7_ST
              , &affinityMode
            #endif
            );
        RINOK(res)

        cpuFreq = fb.CpuFreqRes;
        callback.NewLine();

        if (specifiedFreq != 0)
          cpuFreq = specifiedFreq;

        if (testTimeMs >= 1000)
        if (freqTest == kNumCpuTests - 1)
        {
          // SetComplexCommandsMs(testTimeMs, specifiedFreq != 0, cpuFreq, complexInCommands);
        }
      }
      callback.NewLine();

      // return S_OK; // change it

      callback.SetFreq(true, cpuFreq);

      if (!onlyHashBench)
      {
        size_t dataSize = (size_t)dict;
        if (use_fileData)
        {
          dataSize = fileDataBuffer.Size();
          if (dictIsDefined && dataSize > dict)
            dataSize = (size_t)dict;
        }

        const HRESULT res = TotalBench(EXTERNAL_CODECS_LOC_VARS
            method, complexInCommands,
          #ifndef Z7_ST
            numThreads,
            &affinityMode,
          #endif
            dictIsDefined || use_fileData, // forceUnpackSize
            dataSize,
            (const Byte *)fileDataBuffer,
            printCallback, &callback);
        RINOK(res)
      }

      {
        size_t dataSize = (size_t)1 << kNumHashDictBits;
        if (dictIsDefined)
        {
          dataSize = (size_t)dict;
          if (dataSize != dict)
            return E_OUTOFMEMORY;
        }
        if (use_fileData)
        {
          dataSize = fileDataBuffer.Size();
          if (dictIsDefined && dataSize > dict)
            dataSize = (size_t)dict;
        }

        const HRESULT res = TotalBench_Hash(EXTERNAL_CODECS_LOC_VARS
            method, complexInCommands,
            numThreads,
            dataSize, (const Byte *)fileDataBuffer,
            printCallback, &callback,
        #ifndef Z7_ST
          &affinityMode,
        #endif
          &callback.EncodeRes, true, cpuFreq);
        RINOK(res)
      }

      callback.NewLine();
      {
        PrintLeft(f, "CPU", kFieldSize_Name);

        CFreqBench fb;
        fb.complexInCommands = complexInCommands;
        fb.numThreads = numThreads;
        // showFreq;
        fb.showFreq = (specifiedFreq != 0);
        fb.specifiedFreq = specifiedFreq;

        const HRESULT res = fb.FreqBench(printCallback
          #ifndef Z7_ST
            , &affinityMode
          #endif
          );
        RINOK(res)
        callback.NewLine();
      }
    }
  }
  else
  {
    needSetComplexity = true;
    if (!methodName.IsEqualTo_Ascii_NoCase("LZMA"))
    {
      unsigned i;
      for (i = 0; i < Z7_ARRAY_SIZE(g_Bench); i++)
      {
        const CBenchMethod &h = g_Bench[i];
        AString benchMethod (h.Name);
        AString benchProps;
        const int propPos = benchMethod.Find(':');
        if (propPos >= 0)
        {
          benchProps = benchMethod.Ptr((unsigned)(propPos + 1));
          benchMethod.DeleteFrom((unsigned)propPos);
        }

        if (AreSameMethodNames(benchMethod, methodName))
        {
          if (benchProps.IsEmpty()
              || (benchProps.IsEqualTo("x5") && method.PropsString.IsEmpty())
              || method.PropsString.IsPrefixedBy_Ascii_NoCase(benchProps))
          {
            callback.BenchProps.EncComplex = h.EncComplex;
            callback.BenchProps.DecComplexCompr = h.DecComplexCompr;
            callback.BenchProps.DecComplexUnc = h.DecComplexUnc;
            needSetComplexity = false;
            break;
          }
        }
      }
      /*
      if (i == Z7_ARRAY_SIZE(g_Bench))
        return E_NOTIMPL;
      */
    }
    if (needSetComplexity)
      callback.BenchProps.SetLzmaCompexity();

  if (startDicLog < kBenchMinDicLogSize)
    startDicLog = kBenchMinDicLogSize;

  for (unsigned i = 0; i < numIterations; i++)
  {
    unsigned pow = (dict < GetDictSizeFromLog(startDicLog)) ? kBenchMinDicLogSize : (unsigned)startDicLog;
    if (!multiDict)
      pow = 32;
    while (GetDictSizeFromLog(pow) > dict && pow > 0)
      pow--;
    for (; GetDictSizeFromLog(pow) <= dict; pow++)
    {
      Print_Pow(f, pow);
      callback.DictSize = (UInt64)1 << pow;

      COneMethodInfo method2 = method;

      if (StringsAreEqualNoCase_Ascii(method2.MethodName, "LZMA"))
      {
        // We add dictionary size property.
        // method2 can have two different dictionary size properties.
        // And last property is main.
        NCOM::CPropVariant propVariant = (UInt32)pow;
        RINOK(method2.ParseMethodFromPROPVARIANT((UString)"d", propVariant))
      }

      size_t uncompressedDataSize;
      if (use_fileData)
      {
        uncompressedDataSize = fileDataBuffer.Size();
      }
      else
      {
        uncompressedDataSize = (size_t)callback.DictSize;
        if (uncompressedDataSize != callback.DictSize)
          return E_OUTOFMEMORY;
        if (uncompressedDataSize >= (1 << 18))
          uncompressedDataSize += kAdditionalSize;
      }

      const HRESULT res = MethodBench(
          EXTERNAL_CODECS_LOC_VARS
          complexInCommands,
        #ifndef Z7_ST
          true, numThreads,
          &affinityMode,
        #endif
          method2,
          uncompressedDataSize, (const Byte *)fileDataBuffer,
          kOldLzmaDictBits, printCallback, &callback, &callback.BenchProps);
      f.NewLine();
      RINOK(res)
      if (!multiDict)
        break;
    }
  }
  }

  PrintChars(f, '-', callback.NameFieldSize + fileldSize);
  
  if (use2Columns)
  {
    f.Print(kSep);
    PrintChars(f, '-', fileldSize);
  }

  f.NewLine();
  
  if (use2Columns)
  {
    PrintLeft(f, "Avr:", callback.NameFieldSize);
    PrintTotals(f, showFreq, cpuFreq, !totalBenchMode, callback.EncodeRes);
    f.Print(kSep);
    PrintTotals(f, showFreq, cpuFreq, !totalBenchMode, callback.DecodeRes);
    f.NewLine();
  }
  
  PrintLeft(f, "Tot:", callback.NameFieldSize);
  CTotalBenchRes midRes;
  midRes = callback.EncodeRes;
  midRes.Update_With_Res(callback.DecodeRes);

  // midRes.SetSum(callback.EncodeRes, callback.DecodeRes);
  PrintTotals(f, showFreq, cpuFreq, false, midRes);
  f.NewLine();

  }
  return S_OK;
}
