/*
 * mptMutex.h
 * ----------
 * Purpose: Partially implement c++ mutexes as far as openmpt needs them. Can eventually go away when we only support c++11 compilers some time.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include <vector> // some C++ header in order to have the C++ standard library version information available

#if !MPT_PLATFORM_MULTITHREADED
#define MPT_MUTEX_STD     0
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_OS_EMSCRIPTEN
#define MPT_MUTEX_STD     0
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_COMPILER_GENERIC && !defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_MUTEX_STD     1
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_COMPILER_MSVC && !defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_MUTEX_STD     1
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_COMPILER_GCC && !MPT_OS_WINDOWS && !defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_MUTEX_STD     1
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_COMPILER_CLANG && defined(__GLIBCXX__) && !defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_MUTEX_STD     1
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif (MPT_OS_MACOSX_OR_IOS || MPT_OS_FREEBSD) && MPT_COMPILER_CLANG && !defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_MUTEX_STD     1
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_CLANG_AT_LEAST(3,6,0) && defined(_LIBCPP_VERSION) && !defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_MUTEX_STD     1
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#elif MPT_OS_WINDOWS
#define MPT_MUTEX_STD     0
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   1
#else
#define MPT_MUTEX_STD     0
#define MPT_MUTEX_PTHREAD 0
#define MPT_MUTEX_WIN32   0
#endif

#if !MPT_MUTEX_STD && !MPT_MUTEX_PTHREAD && !MPT_MUTEX_WIN32
#define MPT_MUTEX_NONE 1
#else
#define MPT_MUTEX_NONE 0
#endif

#if defined(MODPLUG_TRACKER) && MPT_MUTEX_NONE
#error "OpenMPT requires mutexes."
#endif

#if MPT_MUTEX_STD
#include <mutex>
#elif MPT_MUTEX_WIN32
#include <windows.h>
#elif MPT_MUTEX_PTHREAD
#include <pthread.h>
#endif // MPT_MUTEX

OPENMPT_NAMESPACE_BEGIN

namespace mpt {

#if MPT_MUTEX_STD

typedef std::mutex mutex;
typedef std::recursive_mutex recursive_mutex;

#elif MPT_MUTEX_WIN32

// compatible with c++11 std::mutex, can eventually be replaced without touching any usage site
class mutex {
private:
	CRITICAL_SECTION impl;
public:
	mutex() { InitializeCriticalSection(&impl); }
	~mutex() { DeleteCriticalSection(&impl); }
	void lock() { EnterCriticalSection(&impl); }
	bool try_lock() { return TryEnterCriticalSection(&impl) ? true : false; }
	void unlock() { LeaveCriticalSection(&impl); }
};

// compatible with c++11 std::recursive_mutex, can eventually be replaced without touching any usage site
class recursive_mutex {
private:
	CRITICAL_SECTION impl;
public:
	recursive_mutex() { InitializeCriticalSection(&impl); }
	~recursive_mutex() { DeleteCriticalSection(&impl); }
	void lock() { EnterCriticalSection(&impl); }
	bool try_lock() { return TryEnterCriticalSection(&impl) ? true : false; }
	void unlock() { LeaveCriticalSection(&impl); }
};

#elif MPT_MUTEX_PTHREAD

class mutex {
private:
	pthread_mutex_t hLock;
public:
	mutex()
	{
		pthread_mutexattr_t attr;
		pthread_mutexattr_init(&attr);
		pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
		pthread_mutex_init(&hLock, &attr);
		pthread_mutexattr_destroy(&attr);
	}
	~mutex() { pthread_mutex_destroy(&hLock); }
	void lock() { pthread_mutex_lock(&hLock); }
	bool try_lock() { return (pthread_mutex_trylock(&hLock) == 0); }
	void unlock() { pthread_mutex_unlock(&hLock); }
};

class recursive_mutex {
private:
	pthread_mutex_t hLock;
public:
	recursive_mutex()
	{
		pthread_mutexattr_t attr;
		pthread_mutexattr_init(&attr);
		pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
		pthread_mutex_init(&hLock, &attr);
		pthread_mutexattr_destroy(&attr);
	}
	~recursive_mutex() { pthread_mutex_destroy(&hLock); }
	void lock() { pthread_mutex_lock(&hLock); }
	bool try_lock() { return (pthread_mutex_trylock(&hLock) == 0); }
	void unlock() { pthread_mutex_unlock(&hLock); }
};

#else // MPT_MUTEX_NONE

class mutex {
public:
	mutex() { }
	~mutex() { }
	void lock() { }
	bool try_lock() { return true; }
	void unlock() { }
};

class recursive_mutex {
public:
	recursive_mutex() { }
	~recursive_mutex() { }
	void lock() { }
	bool try_lock() { return true; }
	void unlock() { }
};

#endif // MPT_MUTEX

#if MPT_MUTEX_STD

#define MPT_LOCK_GUARD std::lock_guard

#else // !MPT_MUTEX_STD

// compatible with c++11 std::lock_guard, can eventually be replaced without touching any usage site
template< typename mutex_type >
class lock_guard {
private:
	mutex_type & mutex;
public:
	lock_guard( mutex_type & m ) : mutex(m) { mutex.lock(); }
	~lock_guard() { mutex.unlock(); }
};

#define MPT_LOCK_GUARD mpt::lock_guard

#endif // MPT_MUTEX_STD

#ifdef MODPLUG_TRACKER

class recursive_mutex_with_lock_count {
private:
	mpt::recursive_mutex mutex;
	long lockCount;
public:
	recursive_mutex_with_lock_count()
		: lockCount(0)
	{
		return;
	}
	~recursive_mutex_with_lock_count()
	{
		return;
	}
	void lock()
	{
		mutex.lock();
		lockCount++;
	}
	void unlock()
	{
		lockCount--;
		mutex.unlock();
	}
public:
	bool IsLockedByCurrentThread() // DEBUGGING only
	{
		bool islocked = false;
		if(mutex.try_lock())
		{
			islocked = (lockCount > 0);
			mutex.unlock();
		}
		return islocked;
	}
};

#endif // MODPLUG_TRACKER

} // namespace mpt

OPENMPT_NAMESPACE_END

