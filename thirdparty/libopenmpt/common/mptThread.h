/*
 * mptThread.h
 * -----------
 * Purpose: Helper class for running threads, with a more or less platform-independent interface.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#if defined(MPT_ENABLE_THREAD)

#include <vector> // some C++ header in order to have the C++ standard library version information available

#if defined(MPT_QUIRK_NO_CPP_THREAD)
#define MPT_STD_THREAD 0
#elif MPT_COMPILER_GENERIC
#define MPT_STD_THREAD 1
#elif MPT_COMPILER_MSVC
#define MPT_STD_THREAD 1
#elif MPT_COMPILER_GCC && !MPT_OS_WINDOWS
#define MPT_STD_THREAD 1
#elif MPT_COMPILER_CLANG && defined(__GLIBCXX__)
#define MPT_STD_THREAD 1
#elif (MPT_OS_MACOSX_OR_IOS || MPT_OS_FREEBSD) && MPT_COMPILER_CLANG
#define MPT_STD_THREAD 1
#elif MPT_CLANG_AT_LEAST(3,6,0) && defined(_LIBCPP_VERSION)
#define MPT_STD_THREAD 1
#else
#define MPT_STD_THREAD 0
#endif

#if MPT_STD_THREAD
#include <thread>
#else // !MPT_STD_THREAD
#if MPT_OS_WINDOWS
#include <windows.h>
#else // !MPT_OS_WINDOWS
#include <pthread.h>
#endif // MPT_OS_WINDOWS
#endif // MPT_STD_THREAD

#if defined(MODPLUG_TRACKER)
#if MPT_OS_WINDOWS
#include <windows.h>
#endif // MPT_OS_WINDOWS
#endif // MODPLUG_TRACKER

#endif // MPT_ENABLE_THREAD


OPENMPT_NAMESPACE_BEGIN


#if defined(MPT_ENABLE_THREAD)

namespace mpt
{



#if MPT_STD_THREAD



typedef std::thread::native_handle_type native_handle_type;
typedef std::thread thread;



#else // !MPT_STD_THREAD



#if MPT_OS_WINDOWS



typedef HANDLE native_handle_type;

// std::thread
// NOTE: This implementation is not movable and prevents copying.
// Therefore, it is not as versatile as a full C++11 std::thread implementation.
// It is only a strict subset.
class thread
{

private:

	thread(const thread &) = delete;
	thread & operator = (const thread &) = delete;

private:

	class functor_helper_base {
	protected:
		functor_helper_base() {}
	public:
		virtual ~functor_helper_base() {}
	public:
		virtual void operator () () = 0;
	};

	template<typename Tfunc>
	class functor_helper : public functor_helper_base {
	private:
		Tfunc func;
	public:
		functor_helper(Tfunc func_) : func(func_) { return; }
		virtual ~functor_helper() { return; }
		virtual void operator () () { func(); }
	};

	enum FunctionMode
	{
		FunctionModeNone            = 0,
		FunctionModeParamNone       = 1,
		FunctionModeParamPointer    = 2,
		FunctionModeFunctor         = 3,
	};

	native_handle_type threadHandle;

	// Thread startup accesses members of mpt::thread.
	// If the mpt::thread instanced gets detached and destroyed directly after initialization,
	//  there is a race between thread startup and detach/destroy.
	// startupDoneEvent protects against this race.
	HANDLE startupDoneEvent;

	FunctionMode functionMode;
	union {
		struct {
			void (*function)(void);
		} ModeParamNone; 
		struct {
			void (*function)(void*);
			void * userdata;
		} ModeParamPointer;
		struct {
			functor_helper_base * pfunctor;
		} ModeFunctor;
	} modeState;

private:

	uintptr_t ThreadFuntion()
	{
		switch(functionMode)
		{
			case FunctionModeNone:
				SetEvent(startupDoneEvent);
				return 0; 
				break;
			case FunctionModeParamNone:
				{
					void (*f)(void) = modeState.ModeParamNone.function;
					SetEvent(startupDoneEvent);
					f();
				}
				return 0; 
				break;
			case FunctionModeParamPointer:
				{
					void (*f)(void*) = modeState.ModeParamPointer.function;
					void * d = modeState.ModeParamPointer.userdata;
					SetEvent(startupDoneEvent);
					f(d);
				}
				return 0; 
				break;
			case FunctionModeFunctor:
				{
					functor_helper_base * pf = modeState.ModeFunctor.pfunctor;
					SetEvent(startupDoneEvent);
					(*pf)();
					delete pf;
				}
				return 0; 
				break;
			default:
				SetEvent(startupDoneEvent);
				return 0;
				break;
		}
		SetEvent(startupDoneEvent);
		return 0;
	}

	static DWORD WINAPI ThreadFunctionWrapper(LPVOID param)
	{
		reinterpret_cast<mpt::thread*>(param)->ThreadFuntion();
		return 0;
	}

public:

	mpt::native_handle_type native_handle()
	{
		return threadHandle;
	}

	bool joinable() const
	{
		return (threadHandle != nullptr);
	}

	void join()
 	{
		if(!joinable())
 		{
			throw std::invalid_argument("thread::joinable() == false");
		}
		WaitForSingleObject(threadHandle, INFINITE);
		CloseHandle(threadHandle);
		threadHandle = nullptr;
	}

	void detach()
	{
		if(!joinable())
		{
			throw std::invalid_argument("thread::joinable() == false");
		}
		CloseHandle(threadHandle);
		threadHandle = nullptr;
	}

	void swap(thread & other) noexcept
	{
		using std::swap;
		swap(threadHandle, other.threadHandle);
		swap(startupDoneEvent, other.startupDoneEvent);
		swap(functionMode, other.functionMode);
	}

	friend void swap(thread & a, thread & b) noexcept
	{
		a.swap(b);
	}

	thread(thread && other) noexcept
		: threadHandle(nullptr)
		, startupDoneEvent(nullptr)
		, functionMode(FunctionModeNone)
	{
		swap(other);
	}

	thread & operator=(thread && other) noexcept
	{
		if(joinable())
		{
			std::terminate();
		}
		swap(other);
		return *this;
	}

	thread()
		: threadHandle(nullptr)
		, startupDoneEvent(nullptr)
		, functionMode(FunctionModeNone)
	{
		std::memset(&modeState, 0, sizeof(modeState));
	}

	thread(void (*function)(void))
		: threadHandle(nullptr)
		, startupDoneEvent(nullptr)
		, functionMode(FunctionModeParamNone)
	{
		std::memset(&modeState, 0, sizeof(modeState));
		modeState.ModeParamNone.function = function;
		startupDoneEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		if(!startupDoneEvent) { throw std::runtime_error("unable to start thread"); }
		DWORD dummy = 0;  // For Win9x
		threadHandle = CreateThread(NULL, 0, ThreadFunctionWrapper, this, 0, &dummy);
		if(threadHandle)
		{
			WaitForSingleObject(startupDoneEvent, INFINITE);
		}
		CloseHandle(startupDoneEvent);
		startupDoneEvent = nullptr;
		if(!threadHandle) { throw std::runtime_error("unable to start thread"); }
	}

	thread(void (*function)(void*), void * userdata)
 		: threadHandle(nullptr)
		, startupDoneEvent(nullptr)
		, functionMode(FunctionModeParamPointer)
	{
		std::memset(&modeState, 0, sizeof(modeState));
		modeState.ModeParamPointer.function = function;
		modeState.ModeParamPointer.userdata = userdata;
		startupDoneEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		if(!startupDoneEvent) { throw std::runtime_error("unable to start thread"); }
		DWORD dummy = 0;  // For Win9x
		threadHandle = CreateThread(NULL, 0, ThreadFunctionWrapper, this, 0, &dummy);
		if(threadHandle)
		{
			WaitForSingleObject(startupDoneEvent, INFINITE);
		}
		CloseHandle(startupDoneEvent);
		startupDoneEvent = nullptr;
		if(!threadHandle) { throw std::runtime_error("unable to start thread"); }
	}

	template<typename Tfunctor>
	thread(Tfunctor functor)
		: threadHandle(nullptr)
		, startupDoneEvent(nullptr)
		, functionMode(FunctionModeFunctor)
	{
		std::memset(&modeState, 0, sizeof(modeState));
		modeState.ModeFunctor.pfunctor = new functor_helper<Tfunctor>(functor);
		startupDoneEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		if(!startupDoneEvent) { throw std::runtime_error("unable to start thread"); }
		DWORD dummy = 0;  // For Win9x
		threadHandle = CreateThread(NULL, 0, ThreadFunctionWrapper, this, 0, &dummy);
		if(threadHandle)
		{
			WaitForSingleObject(startupDoneEvent, INFINITE);
		}
		CloseHandle(startupDoneEvent);
		startupDoneEvent = nullptr;
		if(!threadHandle) { throw std::runtime_error("unable to start thread"); }
	}

	~thread()
	{
		MPT_ASSERT(!joinable());
	}

public:

	static unsigned int hardware_concurrency()
	{
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return std::max<unsigned int>(sysInfo.dwNumberOfProcessors, 1);
	}

};



#endif // MPT_OS_WINDOWS



#endif // MPT_STD_THREAD



#if defined(MODPLUG_TRACKER)

#if MPT_OS_WINDOWS

enum ThreadPriority
{
	ThreadPriorityLowest  = THREAD_PRIORITY_LOWEST,
	ThreadPriorityLower   = THREAD_PRIORITY_BELOW_NORMAL,
	ThreadPriorityNormal  = THREAD_PRIORITY_NORMAL,
	ThreadPriorityHigh    = THREAD_PRIORITY_ABOVE_NORMAL,
	ThreadPriorityHighest = THREAD_PRIORITY_HIGHEST
};

inline void SetThreadPriority(mpt::thread &t, mpt::ThreadPriority priority)
{
	::SetThreadPriority(t.native_handle(), priority);
}

inline void SetCurrentThreadPriority(mpt::ThreadPriority priority)
{
	::SetThreadPriority(GetCurrentThread(), priority);
}

#else // !MPT_OS_WINDOWS

enum ThreadPriority
{
	ThreadPriorityLowest  = -2,
	ThreadPriorityLower   = -1,
	ThreadPriorityNormal  =  0,
	ThreadPriorityHigh    =  1,
	ThreadPriorityHighest =  2
};

inline void SetThreadPriority(mpt::thread & /*t*/ , mpt::ThreadPriority /*priority*/ )
{
	// nothing
}

inline void SetCurrentThreadPriority(mpt::ThreadPriority /*priority*/ )
{
	// nothing
}

#endif // MPT_OS_WINDOWS

#endif // MODPLUG_TRACKER



#if defined(MODPLUG_TRACKER)

#if MPT_OS_WINDOWS

// Default WinAPI thread
class UnmanagedThread
{
protected:
	HANDLE threadHandle;

public:

	operator HANDLE& () { return threadHandle; }
	operator bool () const { return threadHandle != nullptr; }

	UnmanagedThread() : threadHandle(nullptr) { }
	UnmanagedThread(LPTHREAD_START_ROUTINE function, void *userData = nullptr)
	{
		DWORD dummy = 0;	// For Win9x
		threadHandle = CreateThread(NULL, 0, function, userData, 0, &dummy);
	}
};

// Thread that operates on a member function
template<typename T, void (T::*Fun)()>
class UnmanagedThreadMember : public mpt::UnmanagedThread
{
protected:
	static DWORD WINAPI wrapperFunc(LPVOID param)
	{
		(static_cast<T *>(param)->*Fun)();
		return 0;
	}

public:

	UnmanagedThreadMember(T *instance) : mpt::UnmanagedThread(wrapperFunc, instance) { }
};

inline void SetThreadPriority(mpt::UnmanagedThread &t, mpt::ThreadPriority priority)
{
	::SetThreadPriority(t, priority);
}

#endif // MPT_OS_WINDOWS

#endif // MODPLUG_TRACKER



}	// namespace mpt

#endif // MPT_ENABLE_THREAD

OPENMPT_NAMESPACE_END
