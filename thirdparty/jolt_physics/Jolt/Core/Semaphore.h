// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/Atomics.h>

// Determine which platform specific construct we'll use
JPH_SUPPRESS_WARNINGS_STD_BEGIN
#ifdef JPH_PLATFORM_WINDOWS
	// We include windows.h in the cpp file, the semaphore itself is a void pointer
#elif defined(JPH_PLATFORM_LINUX) || defined(JPH_PLATFORM_ANDROID) || defined(JPH_PLATFORM_BSD) || defined(JPH_PLATFORM_WASM)
	#include <semaphore.h>
	#define JPH_USE_PTHREADS
#elif defined(JPH_PLATFORM_MACOS) || defined(JPH_PLATFORM_IOS)
	#include <dispatch/dispatch.h>
	#define JPH_USE_GRAND_CENTRAL_DISPATCH
#elif defined(JPH_PLATFORM_BLUE)
	// Jolt/Core/PlatformBlue.h should have defined everything that is needed below
#else
	#include <mutex>
	#include <condition_variable>
#endif
JPH_SUPPRESS_WARNINGS_STD_END

JPH_NAMESPACE_BEGIN

/// Implements a semaphore
/// When we switch to C++20 we can use counting_semaphore to unify this
class JPH_EXPORT Semaphore
{
public:
	/// Constructor
							Semaphore();
							~Semaphore();

	/// Release the semaphore, signaling the thread waiting on the barrier that there may be work
	void					Release(uint inNumber = 1);

	/// Acquire the semaphore inNumber times
	void					Acquire(uint inNumber = 1);

	/// Get the current value of the semaphore
	inline int				GetValue() const								{ return mCount.load(std::memory_order_relaxed); }

private:
#if defined(JPH_PLATFORM_WINDOWS) || defined(JPH_USE_PTHREADS) || defined(JPH_USE_GRAND_CENTRAL_DISPATCH) || defined(JPH_PLATFORM_BLUE)
#ifdef JPH_PLATFORM_WINDOWS
	using SemaphoreType = void *;
#elif defined(JPH_USE_PTHREADS)
	using SemaphoreType = sem_t;
#elif defined(JPH_USE_GRAND_CENTRAL_DISPATCH)
	using SemaphoreType = dispatch_semaphore_t;
#elif defined(JPH_PLATFORM_BLUE)
	using SemaphoreType = JPH_PLATFORM_BLUE_SEMAPHORE;
#endif
	alignas(JPH_CACHE_LINE_SIZE) atomic<int> mCount { 0 };					///< We increment mCount for every release, to acquire we decrement the count. If the count is negative we know that we are waiting on the actual semaphore.
	SemaphoreType			mSemaphore { };									///< The semaphore is an expensive construct so we only acquire/release it if we know that we need to wait/have waiting threads
#else
	// Other platforms: Emulate a semaphore using a mutex, condition variable and count
	std::mutex				mLock;
	std::condition_variable	mWaitVariable;
	atomic<int>				mCount { 0 };
#endif
};

JPH_NAMESPACE_END
