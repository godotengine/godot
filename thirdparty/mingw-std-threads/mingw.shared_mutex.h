/// \file mingw.shared_mutex.h
/// \brief Standard-compliant shared_mutex for MinGW
///
/// (c) 2017 by Nathaniel J. McClatchey, Athens OH, United States
/// \author Nathaniel J. McClatchey
///
/// \copyright Simplified (2-clause) BSD License.
///
/// \note This file may become part of the mingw-w64 runtime package. If/when
/// this happens, the appropriate license will be added, i.e. this code will
/// become dual-licensed, and the current BSD 2-clause license will stay.
/// \note Target Windows version is determined by WINVER, which is determined in
/// <windows.h> from _WIN32_WINNT, which can itself be set by the user.

//  Notes on the namespaces:
//  - The implementation can be accessed directly in the namespace
//    mingw_stdthread.
//  - Objects will be brought into namespace std by a using directive. This
//    will cause objects declared in std (such as MinGW's implementation) to
//    hide this implementation's definitions.
//  - To avoid poluting the namespace with implementation details, all objects
//    to be pushed into std will be placed in mingw_stdthread::visible.
//  The end result is that if MinGW supplies an object, it is automatically
//  used. If MinGW does not supply an object, this implementation's version will
//  instead be used.

#ifndef MINGW_SHARED_MUTEX_H_
#define MINGW_SHARED_MUTEX_H_

#if !defined(__cplusplus) || (__cplusplus < 201103L)
#error A C++11 compiler is required!
#endif

#include <cassert>
//  For descriptive errors.
#include <system_error>
//    Implementing a shared_mutex without OS support will require atomic read-
//  modify-write capacity.
#include <atomic>
//  For timing in shared_lock and shared_timed_mutex.
#include <chrono>
#include <limits>

//    Use MinGW's shared_lock class template, if it's available. Requires C++14.
//  If unavailable (eg. because this library is being used in C++11), then an
//  implementation of shared_lock is provided by this header.
#if (__cplusplus >= 201402L)
#include <shared_mutex>
#endif

//  For defer_lock_t, adopt_lock_t, and try_to_lock_t
#include "mingw.mutex.h"
//  For this_thread::yield.
//#include "mingw.thread.h"

//  Might be able to use native Slim Reader-Writer (SRW) locks.
#ifdef _WIN32
#include <sdkddkver.h>  //  Detect Windows version.
#if (defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR))
#pragma message "The Windows API that MinGW-w32 provides is not fully compatible\
 with Microsoft's API. We'll try to work around this, but we can make no\
 guarantees. This problem does not exist in MinGW-w64."
#include <windows.h>    //  No further granularity can be expected.
#else
#include <synchapi.h>
#endif
#endif

namespace mingw_stdthread
{
//  Define a portable atomics-based shared_mutex
namespace portable
{
class shared_mutex
{
    typedef uint_fast16_t counter_type;
    std::atomic<counter_type> mCounter {0};
    static constexpr counter_type kWriteBit = 1 << (std::numeric_limits<counter_type>::digits - 1);

#if STDMUTEX_RECURSION_CHECKS
//  Runtime checker for verifying owner threads. Note: Exclusive mode only.
    _OwnerThread mOwnerThread {};
#endif
public:
    typedef shared_mutex * native_handle_type;

    shared_mutex () = default;

//  No form of copying or moving should be allowed.
    shared_mutex (const shared_mutex&) = delete;
    shared_mutex & operator= (const shared_mutex&) = delete;

    ~shared_mutex ()
    {
//  Terminate if someone tries to destroy an owned mutex.
        assert(mCounter.load(std::memory_order_relaxed) == 0);
    }

    void lock_shared (void)
    {
        counter_type expected = mCounter.load(std::memory_order_relaxed);
        do
        {
//  Delay if writing or if too many readers are attempting to read.
            if (expected >= kWriteBit - 1)
            {
                using namespace std;
                expected = mCounter.load(std::memory_order_relaxed);
                continue;
            }
            if (mCounter.compare_exchange_weak(expected,
                                               static_cast<counter_type>(expected + 1),
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed))
                break;
        }
        while (true);
    }

    bool try_lock_shared (void)
    {
        counter_type expected = mCounter.load(std::memory_order_relaxed) & static_cast<counter_type>(~kWriteBit);
        if (expected + 1 == kWriteBit)
            return false;
        else
            return mCounter.compare_exchange_strong( expected,
                                                    static_cast<counter_type>(expected + 1),
                                                    std::memory_order_acquire,
                                                    std::memory_order_relaxed);
    }

    void unlock_shared (void)
    {
        using namespace std;
#ifndef NDEBUG
        if (!(mCounter.fetch_sub(1, memory_order_release) & static_cast<counter_type>(~kWriteBit)))
            __builtin_trap();
#else
        mCounter.fetch_sub(1, memory_order_release);
#endif
    }

//  Behavior is undefined if a lock was previously acquired.
    void lock (void)
    {
#if STDMUTEX_RECURSION_CHECKS
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
#endif
        using namespace std;
//  Might be able to use relaxed memory order...
//  Wait for the write-lock to be unlocked, then claim the write slot.
        counter_type current;
        while ((current = mCounter.fetch_or(kWriteBit, std::memory_order_acquire)) & kWriteBit);
            //this_thread::yield();
//  Wait for readers to finish up.
        while (current != kWriteBit)
        {
            //this_thread::yield();
            current = mCounter.load(std::memory_order_acquire);
        }
#if STDMUTEX_RECURSION_CHECKS
        mOwnerThread.setOwnerAfterLock(self);
#endif
    }

    bool try_lock (void)
    {
#if STDMUTEX_RECURSION_CHECKS
        DWORD self = mOwnerThread.checkOwnerBeforeLock();
#endif
        counter_type expected = 0;
        bool ret = mCounter.compare_exchange_strong(expected, kWriteBit,
                                                    std::memory_order_acquire,
                                                    std::memory_order_relaxed);
#if STDMUTEX_RECURSION_CHECKS
        if (ret)
            mOwnerThread.setOwnerAfterLock(self);
#endif
        return ret;
    }

    void unlock (void)
    {
#if STDMUTEX_RECURSION_CHECKS
        mOwnerThread.checkSetOwnerBeforeUnlock();
#endif
        using namespace std;
#ifndef NDEBUG
        if (mCounter.load(memory_order_relaxed) != kWriteBit)
            __builtin_trap();
#endif
        mCounter.store(0, memory_order_release);
    }

    native_handle_type native_handle (void)
    {
        return this;
    }
};

} //  Namespace portable

//    The native shared_mutex implementation primarily uses features of Windows
//  Vista, but the features used for try_lock and try_lock_shared were not
//  introduced until Windows 7. To allow limited use while compiling for Vista,
//  I define the class without try_* functions in that case.
//    Only fully-featured implementations will be placed into namespace std.
#if defined(_WIN32) && (WINVER >= _WIN32_WINNT_VISTA)
namespace vista
{
class condition_variable_any;
}

namespace windows7
{
//  We already #include "mingw.mutex.h". May as well reduce redundancy.
class shared_mutex : windows7::mutex
{
//    Allow condition_variable_any (and only condition_variable_any) to treat a
//  shared_mutex as its base class.
    friend class vista::condition_variable_any;
public:
    using windows7::mutex::native_handle_type;
    using windows7::mutex::lock;
    using windows7::mutex::unlock;
    using windows7::mutex::native_handle;

    void lock_shared (void)
    {
        AcquireSRWLockShared(native_handle());
    }

    void unlock_shared (void)
    {
        ReleaseSRWLockShared(native_handle());
    }

//  TryAcquireSRW functions are a Windows 7 feature.
#if (WINVER >= _WIN32_WINNT_WIN7)
    bool try_lock_shared (void)
    {
        return TryAcquireSRWLockShared(native_handle()) != 0;
    }

    using windows7::mutex::try_lock;
#endif
};

} //  Namespace windows7
#endif  //  Compiling for Vista
#if (defined(_WIN32) && (WINVER >= _WIN32_WINNT_WIN7))
using windows7::shared_mutex;
#else
using portable::shared_mutex;
#endif

class shared_timed_mutex : shared_mutex
{
    typedef shared_mutex Base;
public:
    using Base::lock;
    using Base::try_lock;
    using Base::unlock;
    using Base::lock_shared;
    using Base::try_lock_shared;
    using Base::unlock_shared;

    template< class Clock, class Duration >
    bool try_lock_until ( const std::chrono::time_point<Clock,Duration>& cutoff )
    {
        do
        {
            if (try_lock())
                return true;
        }
        while (std::chrono::steady_clock::now() < cutoff);
        return false;
    }

    template< class Rep, class Period >
    bool try_lock_for (const std::chrono::duration<Rep,Period>& rel_time)
    {
        return try_lock_until(std::chrono::steady_clock::now() + rel_time);
    }

    template< class Clock, class Duration >
    bool try_lock_shared_until ( const std::chrono::time_point<Clock,Duration>& cutoff )
    {
        do
        {
            if (try_lock_shared())
                return true;
        }
        while (std::chrono::steady_clock::now() < cutoff);
        return false;
    }

    template< class Rep, class Period >
    bool try_lock_shared_for (const std::chrono::duration<Rep,Period>& rel_time)
    {
        return try_lock_shared_until(std::chrono::steady_clock::now() + rel_time);
    }
};

#if __cplusplus >= 201402L
using std::shared_lock;
#else
//    If not supplied by shared_mutex (eg. because C++14 is not supported), I
//  supply the various helper classes that the header should have defined.
template<class Mutex>
class shared_lock
{
    Mutex * mMutex;
    bool mOwns;
//  Reduce code redundancy
    void verify_lockable (void)
    {
        using namespace std;
        if (mMutex == nullptr)
            __builtin_trap();
        if (mOwns)
            __builtin_trap();
    }
public:
    typedef Mutex mutex_type;

    shared_lock (void) noexcept
        : mMutex(nullptr), mOwns(false)
    {
    }

    shared_lock (shared_lock<Mutex> && other) noexcept
        : mMutex(other.mutex_), mOwns(other.owns_)
    {
        other.mMutex = nullptr;
        other.mOwns = false;
    }

    explicit shared_lock (mutex_type & m)
        : mMutex(&m), mOwns(true)
    {
        mMutex->lock_shared();
    }

    shared_lock (mutex_type & m, defer_lock_t) noexcept
        : mMutex(&m), mOwns(false)
    {
    }

    shared_lock (mutex_type & m, adopt_lock_t)
        : mMutex(&m), mOwns(true)
    {
    }

    shared_lock (mutex_type & m, try_to_lock_t)
        : mMutex(&m), mOwns(m.try_lock_shared())
    {
    }

    template< class Rep, class Period >
    shared_lock( mutex_type& m, const std::chrono::duration<Rep,Period>& timeout_duration )
        : mMutex(&m), mOwns(m.try_lock_shared_for(timeout_duration))
    {
    }

    template< class Clock, class Duration >
    shared_lock( mutex_type& m, const std::chrono::time_point<Clock,Duration>& timeout_time )
        : mMutex(&m), mOwns(m.try_lock_shared_until(timeout_time))
    {
    }

    shared_lock& operator= (shared_lock<Mutex> && other) noexcept
    {
        if (&other != this)
        {
            if (mOwns)
                mMutex->unlock_shared();
            mMutex = other.mMutex;
            mOwns = other.mOwns;
            other.mMutex = nullptr;
            other.mOwns = false;
        }
        return *this;
    }


    ~shared_lock (void)
    {
        if (mOwns)
            mMutex->unlock_shared();
    }

    shared_lock (const shared_lock<Mutex> &) = delete;
    shared_lock& operator= (const shared_lock<Mutex> &) = delete;

//  Shared locking
    void lock (void)
    {
        verify_lockable();
        mMutex->lock_shared();
        mOwns = true;
    }

    bool try_lock (void)
    {
        verify_lockable();
        mOwns = mMutex->try_lock_shared();
        return mOwns;
    }

    template< class Clock, class Duration >
    bool try_lock_until( const std::chrono::time_point<Clock,Duration>& cutoff )
    {
        verify_lockable();
        do
        {
            mOwns = mMutex->try_lock_shared();
            if (mOwns)
                return mOwns;
        }
        while (std::chrono::steady_clock::now() < cutoff);
        return false;
    }

    template< class Rep, class Period >
    bool try_lock_for (const std::chrono::duration<Rep,Period>& rel_time)
    {
        return try_lock_until(std::chrono::steady_clock::now() + rel_time);
    }

    void unlock (void)
    {
        using namespace std;
        if (!mOwns)
            __builtin_trap();
        mMutex->unlock_shared();
        mOwns = false;
    }

//  Modifiers
    void swap (shared_lock<Mutex> & other) noexcept
    {
        using namespace std;
        swap(mMutex, other.mMutex);
        swap(mOwns, other.mOwns);
    }

    mutex_type * release (void) noexcept
    {
        mutex_type * ptr = mMutex;
        mMutex = nullptr;
        mOwns = false;
        return ptr;
    }
//  Observers
    mutex_type * mutex (void) const noexcept
    {
        return mMutex;
    }

    bool owns_lock (void) const noexcept
    {
        return mOwns;
    }

    explicit operator bool () const noexcept
    {
        return owns_lock();
    }
};

template< class Mutex >
void swap( shared_lock<Mutex>& lhs, shared_lock<Mutex>& rhs ) noexcept
{
    lhs.swap(rhs);
}
#endif  //  C++11
} //  Namespace mingw_stdthread

namespace std
{
//    Because of quirks of the compiler, the common "using namespace std;"
//  directive would flatten the namespaces and introduce ambiguity where there
//  was none. Direct specification (std::), however, would be unaffected.
//    Take the safe option, and include only in the presence of MinGW's win32
//  implementation.
#if (__cplusplus < 201703L) || (defined(__MINGW32__ ) && !defined(_GLIBCXX_HAS_GTHREADS) && !defined(__clang__))
using mingw_stdthread::shared_mutex;
#endif
#if (__cplusplus < 201402L) || (defined(__MINGW32__ ) && !defined(_GLIBCXX_HAS_GTHREADS) && !defined(__clang__))
using mingw_stdthread::shared_timed_mutex;
using mingw_stdthread::shared_lock;
#elif !defined(MINGW_STDTHREAD_REDUNDANCY_WARNING)  //  Skip repetition
#define MINGW_STDTHREAD_REDUNDANCY_WARNING
#pragma message "This version of MinGW seems to include a win32 port of\
 pthreads, and probably already has C++ std threading classes implemented,\
 based on pthreads. These classes, found in namespace std, are not overridden\
 by the mingw-std-thread library. If you would still like to use this\
 implementation (as it is more lightweight), use the classes provided in\
 namespace mingw_stdthread."
#endif
} //  Namespace std
#endif // MINGW_SHARED_MUTEX_H_
