// Windows/System.cpp

#include "StdAfx.h"

#ifndef _WIN32
#include <unistd.h>
#include <limits.h>
#if defined(__APPLE__) || defined(__DragonFly__) \
    || defined(BSD) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) \
    || defined(__QNXNTO__)
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif
#endif

#include "../Common/Defs.h"
// #include "../Common/MyWindows.h"

// #include "../../C/CpuArch.h"

#include "System.h"

namespace NWindows {
namespace NSystem {

#ifdef _WIN32

/*
note: returned value in 32-bit version can be limited by value 32.
      while 64-bit version returns full value.
GetMaximumProcessorCount(groupNumber) can return higher value than
GetActiveProcessorCount(groupNumber) in some cases, because CPUs can be added.
*/
// typedef DWORD (WINAPI *Func_GetMaximumProcessorCount)(WORD GroupNumber);
typedef DWORD (WINAPI *Func_GetActiveProcessorCount)(WORD GroupNumber);
typedef WORD (WINAPI *Func_GetActiveProcessorGroupCount)(VOID);
/*
#if 0 && defined(ALL_PROCESSOR_GROUPS)
#define MY_ALL_PROCESSOR_GROUPS   ALL_PROCESSOR_GROUPS
#else
#define MY_ALL_PROCESSOR_GROUPS   0xffff
#endif
*/

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

bool CCpuGroups::Load()
{
  NumThreadsTotal = 0;
  GroupSizes.Clear();
  const HMODULE hmodule = ::GetModuleHandleA("kernel32.dll");
  // Is_Win11_Groups = GetProcAddress(hmodule, "SetThreadSelectedCpuSetMasks") != NULL;
  const
      Func_GetActiveProcessorGroupCount
        fn_GetActiveProcessorGroupCount = Z7_GET_PROC_ADDRESS(
      Func_GetActiveProcessorGroupCount, hmodule,
          "GetActiveProcessorGroupCount");
  const
      Func_GetActiveProcessorCount
        fn_GetActiveProcessorCount = Z7_GET_PROC_ADDRESS(
      Func_GetActiveProcessorCount, hmodule,
          "GetActiveProcessorCount");
  if (!fn_GetActiveProcessorGroupCount ||
      !fn_GetActiveProcessorCount)
    return false;

  const unsigned numGroups = fn_GetActiveProcessorGroupCount();
  if (numGroups == 0)
    return false;
  UInt32 sum = 0;
  for (unsigned i = 0; i < numGroups; i++)
  {
    const UInt32 num = fn_GetActiveProcessorCount((WORD)i);
    /*
    if (num == 0)
    {
      // it means error
      // but is it possible that some group is empty by some reason?
      // GroupSizes.Clear();
      // return false;
    }
    */
    sum += num;
    GroupSizes.Add(num);
  }
  NumThreadsTotal = sum;
  // NumThreadsTotal = fn_GetActiveProcessorCount(MY_ALL_PROCESSOR_GROUPS);
  return true;
}

UInt32 CountAffinity(DWORD_PTR mask)
{
  UInt32 num = 0;
  for (unsigned i = 0; i < sizeof(mask) * 8; i++)
  {
    num += (UInt32)(mask & 1);
    mask >>= 1;
  }
  return num;
}

BOOL CProcessAffinity::Get()
{
  IsGroupMode = false;
  Groups.Load();
  // SetThreadAffinityMask(GetCurrentThread(), 1);
  // SetProcessAffinityMask(GetCurrentProcess(), 1);
  BOOL res = GetProcessAffinityMask(GetCurrentProcess(),
      &processAffinityMask, &systemAffinityMask);
  /* DOCs: On a system with more than 64 processors, if the threads
     of the calling process  are in a single processor group, the
     function sets the variables pointed to by lpProcessAffinityMask
     and lpSystemAffinityMask to the process affinity mask and the
     processor mask of active logical processors for that group.
     If the calling process contains threads in multiple groups,
     the function returns zero for both affinity masks

     note: tested in Win10: GetProcessAffinityMask() doesn't return 0
           in (processAffinityMask) and (systemAffinityMask) masks.
     We need to test it in Win11: how to get mask==0 from GetProcessAffinityMask()?
  */
  if (!res)
  {
    processAffinityMask = 0;
    systemAffinityMask = 0;
  }
  if (Groups.GroupSizes.Size() > 1 && Groups.NumThreadsTotal)
    if (// !res ||
        processAffinityMask == 0 || // to support case described in DOCs and for (!res) case
        processAffinityMask == systemAffinityMask) // for default nonchanged affinity
    {
      // we set IsGroupMode only if processAffinity is default (not changed).
      res = TRUE;
      IsGroupMode = true;
    }
  return res;
}


UInt32 CProcessAffinity::Load_and_GetNumberOfThreads()
{
  if (Get())
  {
    const UInt32 numProcessors = GetNumProcessThreads();
    if (numProcessors)
      return numProcessors;
  }
  SYSTEM_INFO systemInfo;
  GetSystemInfo(&systemInfo);
  // the number of logical processors in the current group
  return systemInfo.dwNumberOfProcessors;
}

UInt32 GetNumberOfProcessors()
{
  // We need to know how many threads we can use.
  // By default the process is assigned to one group.
  CProcessAffinity pa;
  return pa.Load_and_GetNumberOfThreads();
}

#else


BOOL CProcessAffinity::Get()
{
  numSysThreads = GetNumberOfProcessors();

  /*
  numSysThreads = 8;
  for (unsigned i = 0; i < numSysThreads; i++)
    CpuSet_Set(&cpu_set, i);
  return TRUE;
  */
  
  #ifdef Z7_AFFINITY_SUPPORTED
  
  // numSysThreads = sysconf(_SC_NPROCESSORS_ONLN); // The number of processors currently online
  if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0)
    return FALSE;
  return TRUE;
  
  #else
  
  // cpu_set = ((CCpuSet)1 << (numSysThreads)) - 1;
  return TRUE;
  // errno = ENOSYS;
  // return FALSE;
  
  #endif
}

UInt32 GetNumberOfProcessors()
{
  #ifndef Z7_ST
  long n = sysconf(_SC_NPROCESSORS_CONF);  // The number of processors configured
  if (n < 1)
    n = 1;
  return (UInt32)n;
  #else
  return 1;
  #endif
}

#endif


#ifdef _WIN32

#ifndef UNDER_CE

#if !defined(_WIN64) && \
  (defined(__MINGW32_VERSION) || defined(Z7_OLD_WIN_SDK))

typedef struct {
  DWORD dwLength;
  DWORD dwMemoryLoad;
  DWORDLONG ullTotalPhys;
  DWORDLONG ullAvailPhys;
  DWORDLONG ullTotalPageFile;
  DWORDLONG ullAvailPageFile;
  DWORDLONG ullTotalVirtual;
  DWORDLONG ullAvailVirtual;
  DWORDLONG ullAvailExtendedVirtual;
} MY_MEMORYSTATUSEX, *MY_LPMEMORYSTATUSEX;

#else

#define MY_MEMORYSTATUSEX MEMORYSTATUSEX
#define MY_LPMEMORYSTATUSEX LPMEMORYSTATUSEX

#endif

typedef BOOL (WINAPI *Func_GlobalMemoryStatusEx)(MY_LPMEMORYSTATUSEX lpBuffer);

#endif // !UNDER_CE

  
bool GetRamSize(size_t &size)
{
  size = (size_t)sizeof(size_t) << 29;

  #ifndef UNDER_CE
    MY_MEMORYSTATUSEX stat;
    stat.dwLength = sizeof(stat);
  #endif
  
  #ifdef _WIN64
    
    if (!::GlobalMemoryStatusEx(&stat))
      return false;
    size = MyMin(stat.ullTotalVirtual, stat.ullTotalPhys);
    return true;

  #else
    
    #ifndef UNDER_CE
      const
      Func_GlobalMemoryStatusEx fn = Z7_GET_PROC_ADDRESS(
      Func_GlobalMemoryStatusEx, ::GetModuleHandleA("kernel32.dll"),
          "GlobalMemoryStatusEx");
      if (fn && fn(&stat))
      {
        // (MY_MEMORYSTATUSEX::ullTotalVirtual) < 4 GiB in 32-bit mode
        size_t size2 = (size_t)0 - 1;
        if (size2 > stat.ullTotalPhys)
            size2 = (size_t)stat.ullTotalPhys;
        if (size2 > stat.ullTotalVirtual)
            size2 = (size_t)stat.ullTotalVirtual;
        size = size2;
        return true;
      }
    #endif
  
    // On computers with more than 4 GB of memory:
    //   new docs  : GlobalMemoryStatus can report (-1) value to indicate an overflow.
    //   some old docs : GlobalMemoryStatus can report (modulo 4 GiB) value.
    //                   (for example, if 5 GB total memory, it could report 1 GB).
    // We don't want to get (modulo 4 GiB) value.
    // So we use GlobalMemoryStatusEx() instead.
    {
      MEMORYSTATUS stat2;
      stat2.dwLength = sizeof(stat2);
      ::GlobalMemoryStatus(&stat2);
      size = MyMin(stat2.dwTotalVirtual, stat2.dwTotalPhys);
      return true;
    }
  #endif
}
  
#else

// POSIX
// #include <stdio.h>

bool GetRamSize(size_t &size)
{
  UInt64 size64;
  size = (size_t)sizeof(size_t) << 29;
  size64 = size;

#if defined(__APPLE__) || defined(__DragonFly__) \
    || defined(BSD) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) \
    || defined(__QNXNTO__)

    uint64_t val = 0;
    int mib[2];
    mib[0] = CTL_HW;

    #ifdef HW_MEMSIZE
      mib[1] = HW_MEMSIZE;
      // printf("\n sysctl HW_MEMSIZE");
    #elif defined(HW_PHYSMEM64)
      mib[1] = HW_PHYSMEM64;
      // printf("\n sysctl HW_PHYSMEM64");
    #else
      mib[1] = HW_PHYSMEM;
      // printf("\n sysctl HW_PHYSMEM");
    #endif

    size_t size_sys = sizeof(val);
    int res = sysctl(mib, 2, &val, &size_sys, NULL, 0);
    // printf("\n sysctl res=%d val=%llx size_sys = %d, %d\n", res, (long long int)val, (int)size_sys, errno);
    // we use strict check (size_sys == sizeof(val)) for returned value
    // because big-endian encoding is possible:
    if (res == 0 && size_sys == sizeof(val) && val)
      size64 = val;
    else
    {
      uint32_t val32 = 0;
      size_sys = sizeof(val32);
      res = sysctl(mib, 2, &val32, &size_sys, NULL, 0);
      // printf("\n sysctl res=%d val=%llx size_sys = %d, %d\n", res, (long long int)val32, (int)size_sys, errno);
      if (res == 0 && size_sys == sizeof(val32) && val32)
        size64 = val32;
    }

  #elif defined(_AIX)
    #if defined(_SC_AIX_REALMEM) // AIX
      size64 = (UInt64)sysconf(_SC_AIX_REALMEM) * 1024;
    #endif
  #elif 0 || defined(__sun)
    #if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    // FreeBSD, Linux, OpenBSD, and Solaris.
    {
      const long phys_pages = sysconf(_SC_PHYS_PAGES);
      const long page_size = sysconf(_SC_PAGESIZE);
      // #pragma message("GetRamSize : sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE)")
      // printf("\n_SC_PHYS_PAGES (hex) = %lx", (unsigned long)phys_pages);
      // printf("\n_SC_PAGESIZE = %lu\n", (unsigned long)page_size);
      if (phys_pages != -1 && page_size != -1)
        size64 = (UInt64)(Int64)phys_pages * (UInt64)(Int64)page_size;
    }
    #endif
  #elif defined(__gnu_hurd__)
  // fixme
  #elif defined(__FreeBSD_kernel__) && defined(__GLIBC__)
  // GNU/kFreeBSD Debian
  // fixme
  #else

  struct sysinfo info;
  if (::sysinfo(&info) != 0)
    return false;
  size64 = (UInt64)info.mem_unit * info.totalram;
  /*
  printf("\n mem_unit  = %lld", (UInt64)info.mem_unit);
  printf("\n totalram  = %lld", (UInt64)info.totalram);
  printf("\n freeram   = %lld", (UInt64)info.freeram);
  */

  #endif

  size = (size_t)1 << (sizeof(size_t) * 8 - 1);
  if (size > size64)
      size = (size_t)size64;
  return true;
}

#endif


unsigned long Get_File_OPEN_MAX()
{
 #ifdef _WIN32
  return (1 << 24) - (1 << 16); // ~16M handles
 #else
  // some linux versions have default open file limit for user process of 1024 files.
  long n = sysconf(_SC_OPEN_MAX);
  // n = -1; // for debug
  // n = 9; // for debug
  if (n < 1)
  {
    // n = OPEN_MAX;  // ???
    // n = FOPEN_MAX; // = 16 : <stdio.h>
   #ifdef _POSIX_OPEN_MAX
    n = _POSIX_OPEN_MAX; // = 20 : <limits.h>
   #else
    n = 30; // our limit
   #endif
  }
  return (unsigned long)n;
 #endif
}

unsigned Get_File_OPEN_MAX_Reduced_for_3_tasks()
{
  unsigned long numFiles_OPEN_MAX = NSystem::Get_File_OPEN_MAX();
  const unsigned delta = 10; // the reserve for another internal needs of process
  if (numFiles_OPEN_MAX > delta)
    numFiles_OPEN_MAX -= delta;
  else
    numFiles_OPEN_MAX = 1;
  numFiles_OPEN_MAX /= 3; // we suppose that we have up to 3 tasks in total for multiple file processing
  numFiles_OPEN_MAX = MyMax(numFiles_OPEN_MAX, (unsigned long)1);
  unsigned n = (unsigned)(int)-1;
  if (n > numFiles_OPEN_MAX)
    n = (unsigned)numFiles_OPEN_MAX;
  return n;
}

}}
