// Bench.h

#ifndef ZIP7_INC_7ZIP_BENCH_H
#define ZIP7_INC_7ZIP_BENCH_H

#include "../../../Windows/System.h"

#include "../../Common/CreateCoder.h"
#include "../../UI/Common/Property.h"

UInt64 Benchmark_GetUsage_Percents(UInt64 usage);

struct CBenchInfo
{
  UInt64 GlobalTime;
  UInt64 GlobalFreq;
  UInt64 UserTime;
  UInt64 UserFreq;
  UInt64 UnpackSize;
  UInt64 PackSize;
  UInt64 NumIterations;
    
  /*
     during Code(): we track benchInfo only from one thread (theads with index[0])
       NumIterations means number of threads
       UnpackSize and PackSize are total sizes of all iterations of current thread
     after Code():
       NumIterations means the number of Iterations
       UnpackSize and PackSize are total sizes of all threads
  */
  
  CBenchInfo(): NumIterations(0) {}

  UInt64 GetUsage() const;
  UInt64 GetRatingPerUsage(UInt64 rating) const;
  UInt64 GetSpeed(UInt64 numUnits) const;
  UInt64 GetUnpackSizeSpeed() const { return GetSpeed(UnpackSize * NumIterations); }

  UInt64 Get_UnpackSize_Full() const { return UnpackSize * NumIterations; }

  UInt64 GetRating_LzmaEnc(UInt64 dictSize) const;
  UInt64 GetRating_LzmaDec() const;
};


struct CTotalBenchRes
{
  // UInt64 NumIterations1; // for Usage
  UInt64 NumIterations2; // for Rating / RPU

  UInt64 Rating;
  UInt64 Usage;
  UInt64 RPU;
  UInt64 Speed;
  
  void Init() { /* NumIterations1 = 0; */ NumIterations2 = 0; Rating = 0; Usage = 0; RPU = 0; Speed = 0; }

  void SetSum(const CTotalBenchRes &r1, const CTotalBenchRes &r2)
  {
    Rating = (r1.Rating + r2.Rating);
    Usage = (r1.Usage + r2.Usage);
    RPU = (r1.RPU + r2.RPU);
    Speed = (r1.Speed + r2.Speed);
    // NumIterations1 = (r1.NumIterations1 + r2.NumIterations1);
    NumIterations2 = (r1.NumIterations2 + r2.NumIterations2);
  }

  void Generate_From_BenchInfo(const CBenchInfo &info);
  void Mult_For_Weight(unsigned weight);
  void Update_With_Res(const CTotalBenchRes &r);
};


const unsigned kBenchMinDicLogSize = 18;

UInt64 GetBenchMemoryUsage(UInt32 numThreads, int level, UInt64 dictionary, bool totalBench);

Z7_PURE_INTERFACES_BEGIN
DECLARE_INTERFACE(IBenchCallback)
{
  // virtual HRESULT SetFreq(bool showFreq, UInt64 cpuFreq) = 0;
  virtual HRESULT SetEncodeResult(const CBenchInfo &info, bool final) = 0;
  virtual HRESULT SetDecodeResult(const CBenchInfo &info, bool final) = 0;
};

DECLARE_INTERFACE(IBenchPrintCallback)
{
  virtual void Print(const char *s) = 0;
  virtual void NewLine() = 0;
  virtual HRESULT CheckBreak() = 0;
};

DECLARE_INTERFACE(IBenchFreqCallback)
{
  virtual HRESULT AddCpuFreq(unsigned numThreads, UInt64 freq, UInt64 usage) = 0;
  virtual HRESULT FreqsFinished(unsigned numThreads) = 0;
};
Z7_PURE_INTERFACES_END

HRESULT Bench(
    DECL_EXTERNAL_CODECS_LOC_VARS
    IBenchPrintCallback *printCallback,
    IBenchCallback *benchCallback,
    const CObjectVector<CProperty> &props,
    UInt32 numIterations,
    bool multiDict,
    IBenchFreqCallback *freqCallback = NULL);

AString GetProcessThreadsInfo(const NWindows::NSystem::CProcessAffinity &ti);

void GetSysInfo(AString &s1, AString &s2);
void GetCpuName(AString &s);
void AddCpuFeatures(AString &s);

#ifdef Z7_LARGE_PAGES
void Add_LargePages_String(AString &s);
#else
// #define Add_LargePages_String
#endif

#endif
