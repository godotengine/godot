/**
* @file mingw.mutex.h
* @brief std::mutex et al implementation for MinGW
** (c) 2013-2016 by Mega Limited, Auckland, New Zealand
* @author Alexander Vassilev
*
* @copyright Simplified (2-clause) BSD License.
* You should have received a copy of the license along with this
* program.
*
* This code is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* @note
* This file may become part of the mingw-w64 runtime package. If/when this happens,
* the appropriate license will be added, i.e. this code will become dual-licensed,
* and the current BSD 2-clause license will stay.
*/

#ifndef WIN32STDMUTEX_H
#define WIN32STDMUTEX_H

#if !defined(__cplusplus) || (__cplusplus < 201103L)
#error A C++11 compiler is required!
#endif
// Recursion checks on non-recursive locks have some performance penalty, and
// the C++ standard does not mandate them. The user might want to explicitly
// enable or disable such checks. If the user has no preference, enable such
// checks in debug builds, but not in release builds.
#ifdef STDMUTEX_RECURSION_CHECKS
#elif defined(NDEBUG)
#define STDMUTEX_RECURSION_CHECKS 0
#else
#define STDMUTEX_RECURSION_CHECKS 1
#endif

#include <chrono>
#include <system_error>
#include <atomic>
#include <mutex> //need for call_once()

#if STDMUTEX_RECURSION_CHECKS || !defined(NDEBUG)
#include <cstdio>
#endif

#include <sdkddkver.h>  //  Detect Windows version.

#if (defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR))
#pragma message "The Windows API that MinGW-w32 provides is not fully compatible\
 with Microsoft's API. We'll try to work around this, but we can make no\
 guarantees. This problem does not exist in MinGW-w64."
#include <windows.h>    //  No further granularity can be expected.
#else
#if STDMUTEX_RECURSION_CHECKS
#include <processthreadsapi.h>  //  For GetCurrentThreadId
#endif
#include <synchapi.h> //  For InitializeCriticalSection, etc.
#include <errhandlingapi.h> //  For GetLastError
#include <handleapi.h>
#endif

//  Need for the implementation of invoke
#include "mingw.invoke.h"

#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0501)
#error To use the MinGW-std-threads library, you will need to define the macro _WIN32_WINNT to be 0x0501 (Windows XP) or higher.
#endif

namespace mingw_stdthread
{
//    The _NonRecursive class has mechanisms that do not play nice with direct
//  manipulation of the native handle. This forward declaration is part of
//  a friend class declaration.
#if STDMUTEX_RECURSION_CHECKS
namespace vista
{
class condition_variable;
}
#endif
//    To make this namespace equivalent to the thread-related subset of std,
//  pull in the classes and class templates supplied by std but not by this
//  implementation.
using std::lock_guard;
using std::unique_lock;
using std::adopt_lock_t;
using std::defer_lock_t;
using std::try_to_lock_t;
using std::adopt_lock;
using std::defer_lock;
using std::try_to_lock;

class recursive_mutex
{
    CRITICAL_SECTION mHandle;
public:
    typedef LPCRITICAL_SECTION native_handle_type;
    native_handle_type native_handle() {return &mHandle;}
    recursive_mutex() noexcept : mHandle()
    {
        InitializeCriticalSection(&mHandle);
    }
    recursive_mutex (const recursive_mutex&) = delete;
    recursive_mutex& operator=(const recursive_mutex&) = delete;
    ~recursive_mutex() noexcept
    {
        DeleteCriticalSection(&mHandle);
    }
    void lock()
    {
        EnterCriticalSection(&mHandle);
    }
    void unlock()
    {
        LeaveCriticalSection(&mHandle);
    }
    bool try_lock()
    {
        return (TryEnterCriticalSection(&mHandle)!=0);
    }
};

#if STDMUTEX_RECURSION_CHECKS
struct _OwnerThread
{
//    If this is to be read before locking, then the owner-thread variable must
//  be atomic to prevent a torn read from spuriously causing errors.
    std::atomic<DWORD> mOwnerThread;
    constexpr _OwnerThread () noexcept : mOwnerThread(0) {}
    static void on_deadlock (void)
    {
        using namespace std;
        fprintf(stderr, "FATAL: Recursive locking of non-recursive mutex\
 detected. Throwing system exception\n");
        fflush(stderr);
        __builtin_trap();
    }
    DWORD checkOwnerBeforeLock() const
    {
        DWORD self = GetCurrentThreadId();
        if (mOwnerThread.load(std::memory_order_relaxed) == self)
            on_deadlock();
        return self;
    }
    void setOwnerAfterLock(DWORD id)
    {
        mOwnerThread.store(id, std::memory_order_relaxed);
    }
    void checkSetOwnerBeforeUnlock()
    {
        DWORD self = GetCurrentThreadId();
        if (mOwnerThread.load(std::memory_order_relaxed) != self)
            on_deadlock();
        mOwnerThread.store(0, std::memory_order_relaxed);
    }
};
#endif

//    Though the Slim Reader-Writer (SRW) locks used here are not complete until
//  Windows 7, implementing partial functionality in Vista will simplify the
//  interaction with condition variables.

//Define SRWLOCK_INIT.
 
#if !defined(SRWLOCK_INIT)
#pragma message "SRWLOCK_INIT macro is not defined. Defining automatically."
#define SRWLOCK_INIT {0}
#endif
 
#if defined(_WIN32) && (WINVER >= _WIN32_WINNT_VISTA)
namespace windows7
{
class mutex
{
    SRWLOCK mHandle;
//  Track locking thread for error checking.
#if STDMUTEX_RECURSION_CHECKS
    friend class vista::condition_variable;
    _OwnerThread mOwnerThread {};
#endif
public:
    typedef PSRWLOCK native_handle_type;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
    constexpr mutex () noexcept : mHandle(SRWLOCK_INIT) { }
#pragma GCC diagnostic pop
    mutex (const mutex&) = delete;
    mutex & operator= (const mutex&) = delete;
    void lock (void)
    {
//  Note: Undefined behavior if called recursively.
#if STDMUTEX_RECURSION_CHECKS
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
#endif
        AcquireSRWLockExclusive(&mHandle);
#if STDMUTEX_RECURSION_CHECKS
        mOwnerThread.setOwnerAfterLock(self);
#endif
    }
    void unlock (void)
    {
#if STDMUTEX_RECURSION_CHECKS
        mOwnerThread.checkSetOwnerBeforeUnlock();
#endif
        ReleaseSRWLockExclusive(&mHandle);
    }
//  TryAcquireSRW functions are a Windows 7 feature.
#if (WINVER >= _WIN32_WINNT_WIN7)
    bool try_lock (void)
    {
#if STDMUTEX_RECURSION_CHECKS
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
#endif
        BOOL ret = TryAcquireSRWLockExclusive(&mHandle);
#if STDMUTEX_RECURSION_CHECKS
        if (ret)
            mOwnerThread.setOwnerAfterLock(self);
#endif
        return ret;
    }
#endif
    native_handle_type native_handle (void)
    {
        return &mHandle;
    }
};
} //  Namespace windows7
#endif  //  Compiling for Vista
namespace xp
{
class mutex
{
    CRITICAL_SECTION mHandle;
    std::atomic_uchar mState;
//  Track locking thread for error checking.
#if STDMUTEX_RECURSION_CHECKS
    friend class vista::condition_variable;
    _OwnerThread mOwnerThread {};
#endif
public:
    typedef PCRITICAL_SECTION native_handle_type;
    constexpr mutex () noexcept : mHandle(), mState(2) { }
    mutex (const mutex&) = delete;
    mutex & operator= (const mutex&) = delete;
    ~mutex() noexcept
    {
//    Undefined behavior if the mutex is held (locked) by any thread.
//    Undefined behavior if a thread terminates while holding ownership of the
//  mutex.
        DeleteCriticalSection(&mHandle);
    }
    void lock (void)
    {
        unsigned char state = mState.load(std::memory_order_acquire);
        while (state) {
            if ((state == 2) && mState.compare_exchange_weak(state, 1, std::memory_order_acquire))
            {
                InitializeCriticalSection(&mHandle);
                mState.store(0, std::memory_order_release);
                break;
            }
            if (state == 1)
            {
                Sleep(0);
                state = mState.load(std::memory_order_acquire);
            }
        }
#if STDMUTEX_RECURSION_CHECKS
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
#endif
        EnterCriticalSection(&mHandle);
#if STDMUTEX_RECURSION_CHECKS
        mOwnerThread.setOwnerAfterLock(self);
#endif
    }
    void unlock (void)
    {
#if STDMUTEX_RECURSION_CHECKS
        mOwnerThread.checkSetOwnerBeforeUnlock();
#endif
        LeaveCriticalSection(&mHandle);
    }
    bool try_lock (void)
    {
        unsigned char state = mState.load(std::memory_order_acquire);
        if ((state == 2) && mState.compare_exchange_strong(state, 1, std::memory_order_acquire))
        {
            InitializeCriticalSection(&mHandle);
            mState.store(0, std::memory_order_release);
        }
        if (state == 1)
            return false;
#if STDMUTEX_RECURSION_CHECKS
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
#endif
        BOOL ret = TryEnterCriticalSection(&mHandle);
#if STDMUTEX_RECURSION_CHECKS
        if (ret)
            mOwnerThread.setOwnerAfterLock(self);
#endif
        return ret;
    }
    native_handle_type native_handle (void)
    {
        return &mHandle;
    }
};
} //  Namespace "xp"
#if (WINVER >= _WIN32_WINNT_WIN7)
using windows7::mutex;
#else
using xp::mutex;
#endif

class recursive_timed_mutex
{
    static constexpr DWORD kWaitAbandoned = 0x00000080l;
    static constexpr DWORD kWaitObject0 = 0x00000000l;
    static constexpr DWORD kInfinite = 0xffffffffl;
    inline bool try_lock_internal (DWORD ms) noexcept
    {
        DWORD ret = WaitForSingleObject(mHandle, ms);
#ifndef NDEBUG
        if (ret == kWaitAbandoned)
        {
            using namespace std;
            fprintf(stderr, "FATAL: Thread terminated while holding a mutex.");
            terminate();
        }
#endif
        return (ret == kWaitObject0) || (ret == kWaitAbandoned);
    }
protected:
    HANDLE mHandle;
//    Track locking thread for error checking of non-recursive timed_mutex. For
//  standard compliance, this must be defined in same class and at the same
//  access-control level as every other variable in the timed_mutex.
#if STDMUTEX_RECURSION_CHECKS
    friend class vista::condition_variable;
    _OwnerThread mOwnerThread {};
#endif
public:
    typedef HANDLE native_handle_type;
    native_handle_type native_handle() const {return mHandle;}
    recursive_timed_mutex(const recursive_timed_mutex&) = delete;
    recursive_timed_mutex& operator=(const recursive_timed_mutex&) = delete;
    recursive_timed_mutex(): mHandle(CreateMutex(NULL, FALSE, NULL)) {}
    ~recursive_timed_mutex()
    {
        CloseHandle(mHandle);
    }
    void lock()
    {
        DWORD ret = WaitForSingleObject(mHandle, kInfinite);
//    If (ret == WAIT_ABANDONED), then the thread that held ownership was
//  terminated. Behavior is undefined, but Windows will pass ownership to this
//  thread.
#ifndef NDEBUG
        if (ret == kWaitAbandoned)
        {
            using namespace std;
            fprintf(stderr, "FATAL: Thread terminated while holding a mutex.");
            terminate();
        }
#endif
        if ((ret != kWaitObject0) && (ret != kWaitAbandoned))
        {
            __builtin_trap();
        }
    }
    void unlock()
    {
        if (!ReleaseMutex(mHandle))
            __builtin_trap();
    }
    bool try_lock()
    {
        return try_lock_internal(0);
    }
    template <class Rep, class Period>
    bool try_lock_for(const std::chrono::duration<Rep,Period>& dur)
    {
        using namespace std::chrono;
        auto timeout = duration_cast<milliseconds>(dur).count();
        while (timeout > 0)
        {
          constexpr auto kMaxStep = static_cast<decltype(timeout)>(kInfinite-1);
          auto step = (timeout < kMaxStep) ? timeout : kMaxStep;
          if (try_lock_internal(static_cast<DWORD>(step)))
            return true;
          timeout -= step;
        }
        return false;
    }
    template <class Clock, class Duration>
    bool try_lock_until(const std::chrono::time_point<Clock,Duration>& timeout_time)
    {
        return try_lock_for(timeout_time - Clock::now());
    }
};

//  Override if, and only if, it is necessary for error-checking.
#if STDMUTEX_RECURSION_CHECKS
class timed_mutex: recursive_timed_mutex
{
public:
    timed_mutex() = default;
    timed_mutex(const timed_mutex&) = delete;
    timed_mutex& operator=(const timed_mutex&) = delete;
    void lock()
    {
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
        recursive_timed_mutex::lock();
        mOwnerThread.setOwnerAfterLock(self);
    }
    void unlock()
    {
        mOwnerThread.checkSetOwnerBeforeUnlock();
        recursive_timed_mutex::unlock();
    }
    template <class Rep, class Period>
    bool try_lock_for(const std::chrono::duration<Rep,Period>& dur)
    {
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
        bool ret = recursive_timed_mutex::try_lock_for(dur);
        if (ret)
            mOwnerThread.setOwnerAfterLock(self);
        return ret;
    }
    template <class Clock, class Duration>
    bool try_lock_until(const std::chrono::time_point<Clock,Duration>& timeout_time)
    {
        return try_lock_for(timeout_time - Clock::now());
    }
    bool try_lock ()
    {
        return try_lock_for(std::chrono::milliseconds(0));
    }
};
#else
typedef recursive_timed_mutex timed_mutex;
#endif

class once_flag
{
//    When available, the SRW-based mutexes should be faster than the
//  CriticalSection-based mutexes. Only try_lock will be unavailable in Vista,
//  and try_lock is not used by once_flag.
#if (_WIN32_WINNT == _WIN32_WINNT_VISTA)
    windows7::mutex mMutex;
#else
    mutex mMutex;
#endif
    std::atomic_bool mHasRun;
    once_flag(const once_flag&) = delete;
    once_flag& operator=(const once_flag&) = delete;
    template<class Callable, class... Args>
    friend void call_once(once_flag& once, Callable&& f, Args&&... args);
public:
    constexpr once_flag() noexcept: mMutex(), mHasRun(false) {}
};

template<class Callable, class... Args>
void call_once(once_flag& flag, Callable&& func, Args&&... args)
{
    if (flag.mHasRun.load(std::memory_order_acquire))
        return;
    lock_guard<decltype(flag.mMutex)> lock(flag.mMutex);
    if (flag.mHasRun.load(std::memory_order_relaxed))
        return;
    detail::invoke(std::forward<Callable>(func),std::forward<Args>(args)...);
    flag.mHasRun.store(true, std::memory_order_release);
}
} //  Namespace mingw_stdthread

//  Push objects into std, but only if they are not already there.
namespace std
{
//    Because of quirks of the compiler, the common "using namespace std;"
//  directive would flatten the namespaces and introduce ambiguity where there
//  was none. Direct specification (std::), however, would be unaffected.
//    Take the safe option, and include only in the presence of MinGW's win32
//  implementation.
#if defined(__MINGW32__ ) && !defined(_GLIBCXX_HAS_GTHREADS) && !defined(__clang__)
using mingw_stdthread::recursive_mutex;
using mingw_stdthread::mutex;
using mingw_stdthread::recursive_timed_mutex;
using mingw_stdthread::timed_mutex;
using mingw_stdthread::once_flag;
using mingw_stdthread::call_once;
#elif !defined(MINGW_STDTHREAD_REDUNDANCY_WARNING)  //  Skip repetition
#define MINGW_STDTHREAD_REDUNDANCY_WARNING
#pragma message "This version of MinGW seems to include a win32 port of\
 pthreads, and probably already has C++11 std threading classes implemented,\
 based on pthreads. These classes, found in namespace std, are not overridden\
 by the mingw-std-thread library. If you would still like to use this\
 implementation (as it is more lightweight), use the classes provided in\
 namespace mingw_stdthread."
#endif
}
#endif // WIN32STDMUTEX_H
