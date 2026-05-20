// RandGen.cpp

#include "StdAfx.h"

#include "RandGen.h"

#ifndef USE_STATIC_SYSTEM_RAND

#ifndef Z7_ST
#include "../../Windows/Synchronization.h"
#endif


#ifdef _WIN32

#ifdef _WIN64
#define USE_STATIC_RtlGenRandom
#endif

#ifdef USE_STATIC_RtlGenRandom

// #include <NTSecAPI.h>

EXTERN_C_BEGIN
#ifndef RtlGenRandom
  #define RtlGenRandom SystemFunction036
  BOOLEAN WINAPI RtlGenRandom(PVOID RandomBuffer, ULONG RandomBufferLength);
#endif
EXTERN_C_END

#else
EXTERN_C_BEGIN
typedef BOOLEAN (WINAPI * Func_RtlGenRandom)(PVOID RandomBuffer, ULONG RandomBufferLength);
EXTERN_C_END
#endif


#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#define USE_POSIX_TIME
#define USE_POSIX_TIME2
#endif

#ifdef USE_POSIX_TIME
#include <time.h>
#ifdef USE_POSIX_TIME2
#include <sys/time.h>
#endif
#endif

// The seed and first generated data block depend from processID,
// theadID, timer and system random generator, if available.
// Other generated data blocks depend from previous state

#define HASH_UPD(x) Sha256_Update(&hash, (const Byte *)&x, sizeof(x));

void CRandomGenerator::Init()
{
  MY_ALIGN (16)
  CSha256 hash;
  Sha256_Init(&hash);

  unsigned numIterations = 1000;

  {
  #ifndef UNDER_CE
  const unsigned kNumIterations_Small = 100;
  const unsigned kBufSize = 32;
  MY_ALIGN (16)
  Byte buf[kBufSize];
  #endif

  #ifdef _WIN32

  DWORD w = ::GetCurrentProcessId();
  HASH_UPD(w)
  w = ::GetCurrentThreadId();
  HASH_UPD(w)

  #ifdef UNDER_CE
  /*
  if (CeGenRandom(kBufSize, buf))
  {
    numIterations = kNumIterations_Small;
    Sha256_Update(&hash, buf, kBufSize);
  }
  */
  #elif defined(USE_STATIC_RtlGenRandom)
  if (RtlGenRandom(buf, kBufSize))
  {
    numIterations = kNumIterations_Small;
    Sha256_Update(&hash, buf, kBufSize);
  }
  #else
  {
    const HMODULE hModule = ::LoadLibrary(TEXT("advapi32.dll"));
    if (hModule)
    {
      // SystemFunction036() is real name of RtlGenRandom() function
      const
      Func_RtlGenRandom
        my_RtlGenRandom = Z7_GET_PROC_ADDRESS(
      Func_RtlGenRandom, hModule, "SystemFunction036");
      if (my_RtlGenRandom)
      {
        if (my_RtlGenRandom(buf, kBufSize))
        {
          numIterations = kNumIterations_Small;
          Sha256_Update(&hash, buf, kBufSize);
        }
      }
      ::FreeLibrary(hModule);
    }
  }
  #endif

  #else
  
  pid_t pid = getpid();
  HASH_UPD(pid)
  pid = getppid();
  HASH_UPD(pid)

  {
    int f = open("/dev/urandom", O_RDONLY);
    unsigned numBytes = kBufSize;
    if (f >= 0)
    {
      do
      {
        ssize_t n = read(f, buf, numBytes);
        if (n <= 0)
          break;
        Sha256_Update(&hash, buf, (size_t)n);
        numBytes -= (unsigned)n;
      }
      while (numBytes);
      close(f);
      if (numBytes == 0)
        numIterations = kNumIterations_Small;
    }
  }
  /*
  {
    int n = getrandom(buf, kBufSize, 0);
    if (n > 0)
    {
      Sha256_Update(&hash, buf, n);
      if (n == kBufSize)
        numIterations = kNumIterations_Small;
    }
  }
  */

  #endif
  }

  #ifdef _DEBUG
  numIterations = 2;
  #endif

  do
  {
    #ifdef _WIN32
    LARGE_INTEGER v;
    if (::QueryPerformanceCounter(&v))
      HASH_UPD(v.QuadPart)
    #endif

    #ifdef USE_POSIX_TIME
    #ifdef USE_POSIX_TIME2
    timeval v;
    if (gettimeofday(&v, NULL) == 0)
    {
      HASH_UPD(v.tv_sec)
      HASH_UPD(v.tv_usec)
    }
    #endif
    const time_t v2 = time(NULL);
    HASH_UPD(v2)
    #endif

    #ifdef _WIN32
    const DWORD tickCount = ::GetTickCount();
    HASH_UPD(tickCount)
    #endif
    
    for (unsigned j = 0; j < 100; j++)
    {
      Sha256_Final(&hash, _buff);
      Sha256_Init(&hash);
      Sha256_Update(&hash, _buff, SHA256_DIGEST_SIZE);
    }
  }
  while (--numIterations);

  Sha256_Final(&hash, _buff);
  _needInit = false;
}

#ifndef Z7_ST
  static NWindows::NSynchronization::CCriticalSection g_CriticalSection;
  #define MT_LOCK NWindows::NSynchronization::CCriticalSectionLock lock(g_CriticalSection);
#else
  #define MT_LOCK
#endif

void CRandomGenerator::Generate(Byte *data, unsigned size)
{
  MT_LOCK

  if (_needInit)
    Init();
  while (size != 0)
  {
    MY_ALIGN (16)
    CSha256 hash;
    
    Sha256_Init(&hash);
    Sha256_Update(&hash, _buff, SHA256_DIGEST_SIZE);
    Sha256_Final(&hash, _buff);
    
    Sha256_Init(&hash);
    UInt32 salt = 0xF672ABD1;
    HASH_UPD(salt)
    Sha256_Update(&hash, _buff, SHA256_DIGEST_SIZE);
    MY_ALIGN (16)
    Byte buff[SHA256_DIGEST_SIZE];
    Sha256_Final(&hash, buff);
    for (unsigned i = 0; i < SHA256_DIGEST_SIZE && size != 0; i++, size--)
      *data++ = buff[i];
  }
}

MY_ALIGN (16)
CRandomGenerator g_RandomGenerator;

#endif
