/* Threads.c -- multithreading library
: Igor Pavlov : Public domain */

#include "Precomp.h"

#ifdef _WIN32

#ifndef USE_THREADS_CreateThread
#include <process.h>
#endif

#include "Threads.h"

static WRes GetError(void)
{
  const DWORD res = GetLastError();
  return res ? (WRes)res : 1;
}

static WRes HandleToWRes(HANDLE h) { return (h != NULL) ? 0 : GetError(); }
static WRes BOOLToWRes(BOOL v) { return v ? 0 : GetError(); }

WRes HandlePtr_Close(HANDLE *p)
{
  if (*p != NULL)
  {
    if (!CloseHandle(*p))
      return GetError();
    *p = NULL;
  }
  return 0;
}

WRes Handle_WaitObject(HANDLE h)
{
  DWORD dw = WaitForSingleObject(h, INFINITE);
  /*
    (dw) result:
    WAIT_OBJECT_0  // 0
    WAIT_ABANDONED // 0x00000080 : is not compatible with Win32 Error space
    WAIT_TIMEOUT   // 0x00000102 : is     compatible with Win32 Error space
    WAIT_FAILED    // 0xFFFFFFFF
  */
  if (dw == WAIT_FAILED)
  {
    dw = GetLastError();
    if (dw == 0)
      return WAIT_FAILED;
  }
  return (WRes)dw;
}

#define Thread_Wait(p) Handle_WaitObject(*(p))

WRes Thread_Wait_Close(CThread *p)
{
  WRes res = Thread_Wait(p);
  WRes res2 = Thread_Close(p);
  return (res != 0 ? res : res2);
}

typedef struct MY_PROCESSOR_NUMBER {
    WORD  Group;
    BYTE  Number;
    BYTE  Reserved;
} MY_PROCESSOR_NUMBER, *MY_PPROCESSOR_NUMBER;

typedef struct MY_GROUP_AFFINITY {
#if defined(Z7_GCC_VERSION) && (Z7_GCC_VERSION < 100000)
    // KAFFINITY is not defined in old mingw
    ULONG_PTR
#else
    KAFFINITY
#endif
      Mask;
    WORD   Group;
    WORD   Reserved[3];
} MY_GROUP_AFFINITY, *MY_PGROUP_AFFINITY;

typedef BOOL (WINAPI *Func_SetThreadGroupAffinity)(
    HANDLE hThread,
    CONST MY_GROUP_AFFINITY *GroupAffinity,
    MY_PGROUP_AFFINITY PreviousGroupAffinity);

typedef BOOL (WINAPI *Func_GetThreadGroupAffinity)(
    HANDLE hThread,
    MY_PGROUP_AFFINITY GroupAffinity);

typedef BOOL (WINAPI *Func_GetProcessGroupAffinity)(
    HANDLE hProcess,
    PUSHORT GroupCount,
    PUSHORT GroupArray);

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

#if 0
#include <stdio.h>
#define PRF(x) x
/*
--
  before call of SetThreadGroupAffinity()
    GetProcessGroupAffinity return one group.
  after call of SetThreadGroupAffinity():
    GetProcessGroupAffinity return more than group,
    if SetThreadGroupAffinity() was to another group.
--
  GetProcessAffinityMask MS DOCs:
  {
    If the calling process contains threads in multiple groups,
    the function returns zero for both affinity masks.
  }
  but tests in win10 with 2 groups (less than 64 cores total):
    GetProcessAffinityMask() still returns non-zero affinity masks
    even after SetThreadGroupAffinity() calls.
*/
static void PrintProcess_Info()
{
  {
    const
      Func_GetProcessGroupAffinity fn_GetProcessGroupAffinity =
     (Func_GetProcessGroupAffinity) Z7_CAST_FUNC_C GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
          "GetProcessGroupAffinity");
    if (fn_GetProcessGroupAffinity)
    {
      unsigned i;
      USHORT GroupCounts[64];
      USHORT GroupCount = Z7_ARRAY_SIZE(GroupCounts);
      BOOL boolRes = fn_GetProcessGroupAffinity(GetCurrentProcess(),
          &GroupCount, GroupCounts);
      printf("\n====== GetProcessGroupAffinity : "
          "boolRes=%u GroupCounts = %u :",
          boolRes, (unsigned)GroupCount);
      for (i = 0; i < GroupCount; i++)
        printf(" %u", GroupCounts[i]);
      printf("\n");
    }
  }
  {
    DWORD_PTR processAffinityMask, systemAffinityMask;
    if (GetProcessAffinityMask(GetCurrentProcess(), &processAffinityMask, &systemAffinityMask))
    {
      PRF(printf("\n====== GetProcessAffinityMask : "
        ": processAffinityMask=%x, systemAffinityMask=%x\n",
        (UInt32)processAffinityMask, (UInt32)systemAffinityMask);)
    }
    else
      printf("\n==GetProcessAffinityMask FAIL");
  }
}
#else
#ifndef USE_THREADS_CreateThread
// #define PRF(x)
#endif
#endif

/* if we send (stackSize=0) to CreateThread(), it will
   use default value PE::SizeOfStackReserve from exe file.
   PE::SizeOfStackReserve == 1 MiB in exe file with default linker options.
   Windows aligns specified value to the next 64 KB range. */
static const unsigned k_StackSize_ReserveSize =
  #ifdef UNDER_CE
    1 << 17;
  #else
    1 << 20;
  #endif

WRes Thread_Create(CThread *p, THREAD_FUNC_TYPE func, LPVOID param)
{
  /* Windows Me/98/95: threadId parameter may not be NULL in _beginthreadex/CreateThread functions */

  #ifdef USE_THREADS_CreateThread

  DWORD threadId;
  *p = CreateThread(NULL, k_StackSize_ReserveSize, func, param, STACK_SIZE_PARAM_IS_A_RESERVATION, &threadId);
  
  #else

#define CALL_beginthreadex(func2, param2, flags, threadIdPtr) \
    ((HANDLE)(_beginthreadex(NULL, k_StackSize_ReserveSize, func2, param2, (flags) | STACK_SIZE_PARAM_IS_A_RESERVATION, threadIdPtr)))
  
  unsigned threadId;
  *p = CALL_beginthreadex(func, param, 0, &threadId);

#if 0 // 1 : for debug
  {
      DWORD_PTR prevMask;
      DWORD_PTR affinity = 1 << 0;
      prevMask = SetThreadAffinityMask(*p, (DWORD_PTR)affinity);
      prevMask = prevMask;
  }
#endif
#if 0 // 1 : for debug
  {
      /* win10: new thread will be created in same group that is assigned to parent thread
                but affinity mask will contain all allowed threads of that group,
                even if affinity mask of parent group is not full
         win11: what group it will be created, if we have set
                affinity of parent thread with ThreadGroupAffinity?
      */
      const
         Func_GetThreadGroupAffinity fn =
        (Func_GetThreadGroupAffinity) Z7_CAST_FUNC_C GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
             "GetThreadGroupAffinity");
      if (fn)
      {
        // BOOL wres2;
        MY_GROUP_AFFINITY groupAffinity;
        memset(&groupAffinity, 0, sizeof(groupAffinity));
        /* wres2 = */ fn(*p, &groupAffinity);
        PRF(printf("\n==Thread_Create cur = %6u GetThreadGroupAffinity(): "
            "wres2_BOOL = %u, group=%u mask=%x\n",
            GetCurrentThreadId(),
            wres2,
            groupAffinity.Group,
            (UInt32)groupAffinity.Mask);)
      }
  }
#endif

  #endif

  /* maybe we must use errno here, but probably GetLastError() is also OK. */
  return HandleToWRes(*p);
}


WRes Thread_Create_With_Affinity(CThread *p, THREAD_FUNC_TYPE func, LPVOID param, CAffinityMask affinity)
{
  #ifdef USE_THREADS_CreateThread

  UNUSED_VAR(affinity)
  return Thread_Create(p, func, param);
  
  #else
  
  /* Windows Me/98/95: threadId parameter may not be NULL in _beginthreadex/CreateThread functions */
  HANDLE h;
  WRes wres;
  unsigned threadId;
  h = CALL_beginthreadex(func, param, CREATE_SUSPENDED, &threadId);
  *p = h;
  wres = HandleToWRes(h);
  if (h)
  {
    {
      // DWORD_PTR prevMask =
      SetThreadAffinityMask(h, (DWORD_PTR)affinity);
      /*
      if (prevMask == 0)
      {
        // affinity change is non-critical error, so we can ignore it
        // wres = GetError();
      }
      */
    }
    {
      const DWORD prevSuspendCount = ResumeThread(h);
      /* ResumeThread() returns:
         0 : was_not_suspended
         1 : was_resumed
        -1 : error
      */
      if (prevSuspendCount == (DWORD)-1)
        wres = GetError();
    }
  }

  /* maybe we must use errno here, but probably GetLastError() is also OK. */
  return wres;

  #endif
}


WRes Thread_Create_With_Group(CThread *p, THREAD_FUNC_TYPE func, LPVOID param, unsigned group, CAffinityMask affinityMask)
{
#ifdef USE_THREADS_CreateThread

  UNUSED_VAR(group)
  UNUSED_VAR(affinityMask)
  return Thread_Create(p, func, param);
  
#else
  
  /* Windows Me/98/95: threadId parameter may not be NULL in _beginthreadex/CreateThread functions */
  HANDLE h;
  WRes wres;
  unsigned threadId;
  h = CALL_beginthreadex(func, param, CREATE_SUSPENDED, &threadId);
  *p = h;
  wres = HandleToWRes(h);
  if (h)
  {
    // PrintProcess_Info();
    {
      const
         Func_SetThreadGroupAffinity fn =
        (Func_SetThreadGroupAffinity) Z7_CAST_FUNC_C GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
             "SetThreadGroupAffinity");
      if (fn)
      {
        // WRes wres2;
        MY_GROUP_AFFINITY groupAffinity, prev_groupAffinity;
        memset(&groupAffinity, 0, sizeof(groupAffinity));
        // groupAffinity.Mask must use only bits that supported by current group
        // (groupAffinity.Mask = 0) means all allowed bits
        groupAffinity.Mask = affinityMask;
        groupAffinity.Group = (WORD)group;
        // wres2 =
        fn(h, &groupAffinity, &prev_groupAffinity);
        /*
        if (groupAffinity.Group == prev_groupAffinity.Group)
          wres2 = wres2;
        else
          wres2 = wres2;
        if (wres2 == 0)
        {
          wres2 = GetError();
          PRF(printf("\n==SetThreadGroupAffinity error: %u\n", wres2);)
        }
        else
        {
          PRF(printf("\n==Thread_Create_With_Group::SetThreadGroupAffinity()"
            " threadId = %6u"
            " group=%u mask=%x\n",
            threadId,
            prev_groupAffinity.Group,
            (UInt32)prev_groupAffinity.Mask);)
        }
        */
      }
    }
    {
      const DWORD prevSuspendCount = ResumeThread(h);
      /* ResumeThread() returns:
         0 : was_not_suspended
         1 : was_resumed
        -1 : error
      */
      if (prevSuspendCount == (DWORD)-1)
        wres = GetError();
    }
  }

  /* maybe we must use errno here, but probably GetLastError() is also OK. */
  return wres;

  #endif
}


static WRes Event_Create(CEvent *p, BOOL manualReset, int signaled)
{
  *p = CreateEvent(NULL, manualReset, (signaled ? TRUE : FALSE), NULL);
  return HandleToWRes(*p);
}

WRes Event_Set(CEvent *p) { return BOOLToWRes(SetEvent(*p)); }
WRes Event_Reset(CEvent *p) { return BOOLToWRes(ResetEvent(*p)); }

WRes ManualResetEvent_Create(CManualResetEvent *p, int signaled) { return Event_Create(p, TRUE, signaled); }
WRes AutoResetEvent_Create(CAutoResetEvent *p, int signaled) { return Event_Create(p, FALSE, signaled); }
WRes ManualResetEvent_CreateNotSignaled(CManualResetEvent *p) { return ManualResetEvent_Create(p, 0); }
WRes AutoResetEvent_CreateNotSignaled(CAutoResetEvent *p) { return AutoResetEvent_Create(p, 0); }


WRes Semaphore_Create(CSemaphore *p, UInt32 initCount, UInt32 maxCount)
{
  // negative ((LONG)maxCount) is not supported in WIN32::CreateSemaphore()
  *p = CreateSemaphore(NULL, (LONG)initCount, (LONG)maxCount, NULL);
  return HandleToWRes(*p);
}

WRes Semaphore_OptCreateInit(CSemaphore *p, UInt32 initCount, UInt32 maxCount)
{
  // if (Semaphore_IsCreated(p))
  {
    WRes wres = Semaphore_Close(p);
    if (wres != 0)
      return wres;
  }
  return Semaphore_Create(p, initCount, maxCount);
}

static WRes Semaphore_Release(CSemaphore *p, LONG releaseCount, LONG *previousCount)
  { return BOOLToWRes(ReleaseSemaphore(*p, releaseCount, previousCount)); }
WRes Semaphore_ReleaseN(CSemaphore *p, UInt32 num)
  { return Semaphore_Release(p, (LONG)num, NULL); }
WRes Semaphore_Release1(CSemaphore *p) { return Semaphore_ReleaseN(p, 1); }

WRes CriticalSection_Init(CCriticalSection *p)
{
  /* InitializeCriticalSection() can raise exception:
     Windows XP, 2003 : can raise a STATUS_NO_MEMORY exception
     Windows Vista+   : no exceptions */
  #ifdef _MSC_VER
  #ifdef __clang__
    #pragma GCC diagnostic ignored "-Wlanguage-extension-token"
  #endif
  __try
  #endif
  {
    InitializeCriticalSection(p);
    /* InitializeCriticalSectionAndSpinCount(p, 0); */
  }
  #ifdef _MSC_VER
  __except (EXCEPTION_EXECUTE_HANDLER) { return ERROR_NOT_ENOUGH_MEMORY; }
  #endif
  return 0;
}




#else // _WIN32

// ---------- POSIX ----------

#if defined(__linux__) && !defined(__APPLE__) && !defined(_AIX) && !defined(__ANDROID__)
#ifndef Z7_AFFINITY_DISABLE
// _GNU_SOURCE can be required for pthread_setaffinity_np() / CPU_ZERO / CPU_SET
// clang < 3.6       : unknown warning group '-Wreserved-id-macro'
// clang 3.6 - 12.01 : gives warning "macro name is a reserved identifier"
// clang >= 13       : do not give warning
#if !defined(_GNU_SOURCE)
Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
// #define _GNU_SOURCE
Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
#endif // !defined(_GNU_SOURCE)
#endif // Z7_AFFINITY_DISABLE
#endif // __linux__

#include "Threads.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#ifdef Z7_AFFINITY_SUPPORTED
// #include <sched.h>
#endif


// #include <stdio.h>
// #define PRF(p) p
#define PRF(p)
#define Print(s) PRF(printf("\n%s\n", s);)

WRes Thread_Create_With_CpuSet(CThread *p, THREAD_FUNC_TYPE func, LPVOID param, const CCpuSet *cpuSet)
{
  // new thread in Posix probably inherits affinity from parrent thread
  Print("Thread_Create_With_CpuSet")

  pthread_attr_t attr;
  int ret;
  // int ret2;

  p->_created = 0;

  RINOK(pthread_attr_init(&attr))

  ret = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  if (!ret)
  {
    if (cpuSet)
    {
      // pthread_attr_setaffinity_np() is not supported for MUSL compile.
      // so we check for __GLIBC__ here
#if defined(Z7_AFFINITY_SUPPORTED) && defined( __GLIBC__)
      /*
      printf("\n affinity :");
      unsigned i;
      for (i = 0; i < sizeof(*cpuSet) && i < 8; i++)
      {
        Byte b = *((const Byte *)cpuSet + i);
        char temp[32];
        #define GET_HEX_CHAR(t) ((char)(((t < 10) ? ('0' + t) : ('A' + (t - 10)))))
        temp[0] = GET_HEX_CHAR((b & 0xF));
        temp[1] = GET_HEX_CHAR((b >> 4));
        // temp[0] = GET_HEX_CHAR((b >> 4));  // big-endian
        // temp[1] = GET_HEX_CHAR((b & 0xF));  // big-endian
        temp[2] = 0;
        printf("%s", temp);
      }
      printf("\n");
      */

      // ret2 =
      pthread_attr_setaffinity_np(&attr, sizeof(*cpuSet), cpuSet);
      // if (ret2) ret = ret2;
#endif
    }
    
    ret = pthread_create(&p->_tid, &attr, func, param);
    
    if (!ret)
    {
      p->_created = 1;
      /*
      if (cpuSet)
      {
        // ret2 =
        pthread_setaffinity_np(p->_tid, sizeof(*cpuSet), cpuSet);
        // if (ret2) ret = ret2;
      }
      */
    }
  }
  // ret2 =
  pthread_attr_destroy(&attr);
  // if (ret2 != 0) ret = ret2;
  return ret;
}


WRes Thread_Create(CThread *p, THREAD_FUNC_TYPE func, LPVOID param)
{
  return Thread_Create_With_CpuSet(p, func, param, NULL);
}

/*
WRes Thread_Create_With_Group(CThread *p, THREAD_FUNC_TYPE func, LPVOID param, unsigned group, CAffinityMask affinity)
{
  UNUSED_VAR(group)
  return Thread_Create_With_Affinity(p, func, param, affinity);
}
*/

WRes Thread_Create_With_Affinity(CThread *p, THREAD_FUNC_TYPE func, LPVOID param, CAffinityMask affinity)
{
  Print("Thread_Create_WithAffinity")
  CCpuSet cs;
  unsigned i;
  CpuSet_Zero(&cs);
  for (i = 0; i < sizeof(affinity) * 8; i++)
  {
    if (affinity == 0)
      break;
    if (affinity & 1)
    {
      CpuSet_Set(&cs, i);
    }
    affinity >>= 1;
  }
  return Thread_Create_With_CpuSet(p, func, param, &cs);
}


WRes Thread_Close(CThread *p)
{
  // Print("Thread_Close")
  int ret;
  if (!p->_created)
    return 0;
    
  ret = pthread_detach(p->_tid);
  p->_tid = 0;
  p->_created = 0;
  return ret;
}


WRes Thread_Wait_Close(CThread *p)
{
  // Print("Thread_Wait_Close")
  void *thread_return;
  int ret;
  if (!p->_created)
    return EINVAL;

  ret = pthread_join(p->_tid, &thread_return);
  // probably we can't use that (_tid) after pthread_join(), so we close thread here
  p->_created = 0;
  p->_tid = 0;
  return ret;
}



static WRes Event_Create(CEvent *p, int manualReset, int signaled)
{
  RINOK(pthread_mutex_init(&p->_mutex, NULL))
  RINOK(pthread_cond_init(&p->_cond, NULL))
  p->_manual_reset = manualReset;
  p->_state = (signaled ? True : False);
  p->_created = 1;
  return 0;
}

WRes ManualResetEvent_Create(CManualResetEvent *p, int signaled)
  { return Event_Create(p, True, signaled); }
WRes ManualResetEvent_CreateNotSignaled(CManualResetEvent *p)
  { return ManualResetEvent_Create(p, 0); }
WRes AutoResetEvent_Create(CAutoResetEvent *p, int signaled)
  { return Event_Create(p, False, signaled); }
WRes AutoResetEvent_CreateNotSignaled(CAutoResetEvent *p)
  { return AutoResetEvent_Create(p, 0); }


#if defined(Z7_LLVM_CLANG_VERSION) && (__clang_major__ == 13)
// freebsd:
#pragma GCC diagnostic ignored "-Wthread-safety-analysis"
#endif

WRes Event_Set(CEvent *p)
{
  RINOK(pthread_mutex_lock(&p->_mutex))
  p->_state = True;
  {
    const int res1 = pthread_cond_broadcast(&p->_cond);
    const int res2 = pthread_mutex_unlock(&p->_mutex);
    return (res2 ? res2 : res1);
  }
}

WRes Event_Reset(CEvent *p)
{
  RINOK(pthread_mutex_lock(&p->_mutex))
  p->_state = False;
  return pthread_mutex_unlock(&p->_mutex);
}
 
WRes Event_Wait(CEvent *p)
{
  RINOK(pthread_mutex_lock(&p->_mutex))
  while (p->_state == False)
  {
    // ETIMEDOUT
    // ret =
    pthread_cond_wait(&p->_cond, &p->_mutex);
    // if (ret != 0) break;
  }
  if (p->_manual_reset == False)
  {
    p->_state = False;
  }
  return pthread_mutex_unlock(&p->_mutex);
}

WRes Event_Close(CEvent *p)
{
  if (!p->_created)
    return 0;
  p->_created = 0;
  {
    const int res1 = pthread_mutex_destroy(&p->_mutex);
    const int res2 = pthread_cond_destroy(&p->_cond);
    return (res1 ? res1 : res2);
  }
}


WRes Semaphore_Create(CSemaphore *p, UInt32 initCount, UInt32 maxCount)
{
  if (initCount > maxCount || maxCount < 1)
    return EINVAL;
  RINOK(pthread_mutex_init(&p->_mutex, NULL))
  RINOK(pthread_cond_init(&p->_cond, NULL))
  p->_count = initCount;
  p->_maxCount = maxCount;
  p->_created = 1;
  return 0;
}


WRes Semaphore_OptCreateInit(CSemaphore *p, UInt32 initCount, UInt32 maxCount)
{
  if (Semaphore_IsCreated(p))
  {
    /*
    WRes wres = Semaphore_Close(p);
    if (wres != 0)
      return wres;
    */
    if (initCount > maxCount || maxCount < 1)
      return EINVAL;
    // return EINVAL; // for debug
    p->_count = initCount;
    p->_maxCount = maxCount;
    return 0;
  }
  return Semaphore_Create(p, initCount, maxCount);
}


WRes Semaphore_ReleaseN(CSemaphore *p, UInt32 releaseCount)
{
  UInt32 newCount;
  int ret;

  if (releaseCount < 1)
    return EINVAL;

  RINOK(pthread_mutex_lock(&p->_mutex))

  newCount = p->_count + releaseCount;
  if (newCount > p->_maxCount)
    ret = ERROR_TOO_MANY_POSTS; // EINVAL;
  else
  {
    p->_count = newCount;
    ret = pthread_cond_broadcast(&p->_cond);
  }
  RINOK(pthread_mutex_unlock(&p->_mutex))
  return ret;
}

WRes Semaphore_Wait(CSemaphore *p)
{
  RINOK(pthread_mutex_lock(&p->_mutex))
  while (p->_count < 1)
  {
    pthread_cond_wait(&p->_cond, &p->_mutex);
  }
  p->_count--;
  return pthread_mutex_unlock(&p->_mutex);
}

WRes Semaphore_Close(CSemaphore *p)
{
  if (!p->_created)
    return 0;
  p->_created = 0;
  {
    const int res1 = pthread_mutex_destroy(&p->_mutex);
    const int res2 = pthread_cond_destroy(&p->_cond);
    return (res1 ? res1 : res2);
  }
}



WRes CriticalSection_Init(CCriticalSection *p)
{
  // Print("CriticalSection_Init")
  if (!p)
    return EINTR;
  return pthread_mutex_init(&p->_mutex, NULL);
}

void CriticalSection_Enter(CCriticalSection *p)
{
  // Print("CriticalSection_Enter")
  if (p)
  {
    // int ret =
    pthread_mutex_lock(&p->_mutex);
  }
}

void CriticalSection_Leave(CCriticalSection *p)
{
  // Print("CriticalSection_Leave")
  if (p)
  {
    // int ret =
    pthread_mutex_unlock(&p->_mutex);
  }
}

void CriticalSection_Delete(CCriticalSection *p)
{
  // Print("CriticalSection_Delete")
  if (p)
  {
    // int ret =
    pthread_mutex_destroy(&p->_mutex);
  }
}

LONG InterlockedIncrement(LONG volatile *addend)
{
  // Print("InterlockedIncrement")
  #ifdef USE_HACK_UNSAFE_ATOMIC
    LONG val = *addend + 1;
    *addend = val;
    return val;
  #else

  #if defined(__clang__) && (__clang_major__ >= 8)
    #pragma GCC diagnostic ignored "-Watomic-implicit-seq-cst"
  #endif
    return __sync_add_and_fetch(addend, 1);
  #endif
}

LONG InterlockedDecrement(LONG volatile *addend)
{
  // Print("InterlockedDecrement")
  #ifdef USE_HACK_UNSAFE_ATOMIC
    LONG val = *addend - 1;
    *addend = val;
    return val;
  #else
    return __sync_sub_and_fetch(addend, 1);
  #endif
}

#endif // _WIN32

WRes AutoResetEvent_OptCreate_And_Reset(CAutoResetEvent *p)
{
  if (Event_IsCreated(p))
    return Event_Reset(p);
  return AutoResetEvent_CreateNotSignaled(p);
}

void ThreadNextGroup_Init(CThreadNextGroup *p, UInt32 numGroups, UInt32 startGroup)
{
  // printf("\n====== ThreadNextGroup_Init numGroups = %x: startGroup=%x\n", numGroups, startGroup);
  if (numGroups == 0)
      numGroups = 1;
  p->NumGroups = numGroups;
  p->NextGroup = startGroup % numGroups;
}


UInt32 ThreadNextGroup_GetNext(CThreadNextGroup *p)
{
  const UInt32 next = p->NextGroup;
  p->NextGroup = (next + 1) % p->NumGroups;
  return next;
}

#undef PRF
#undef Print
