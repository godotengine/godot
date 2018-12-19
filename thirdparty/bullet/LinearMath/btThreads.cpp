/*
Copyright (c) 2003-2014 Erwin Coumans  http://bullet.googlecode.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "btThreads.h"
#include "btQuickprof.h"
#include <algorithm>  // for min and max


#if BT_USE_OPENMP && BT_THREADSAFE

#include <omp.h>

#endif // #if BT_USE_OPENMP && BT_THREADSAFE


#if BT_USE_PPL && BT_THREADSAFE

// use Microsoft Parallel Patterns Library (installed with Visual Studio 2010 and later)
#include <ppl.h>  // if you get a compile error here, check whether your version of Visual Studio includes PPL
// Visual Studio 2010 and later should come with it
#include <concrtrm.h>  // for GetProcessorCount()

#endif // #if BT_USE_PPL && BT_THREADSAFE


#if BT_USE_TBB && BT_THREADSAFE

// use Intel Threading Building Blocks for thread management
#define __TBB_NO_IMPLICIT_LINKAGE 1
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#endif // #if BT_USE_TBB && BT_THREADSAFE


#if BT_THREADSAFE
//
// Lightweight spin-mutex based on atomics
// Using ordinary system-provided mutexes like Windows critical sections was noticeably slower
// presumably because when it fails to lock at first it would sleep the thread and trigger costly
// context switching.
// 

#if __cplusplus >= 201103L

// for anything claiming full C++11 compliance, use C++11 atomics
// on GCC or Clang you need to compile with -std=c++11
#define USE_CPP11_ATOMICS 1

#elif defined( _MSC_VER )

// on MSVC, use intrinsics instead
#define USE_MSVC_INTRINSICS 1

#elif defined( __GNUC__ ) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))

// available since GCC 4.7 and some versions of clang
// todo: check for clang
#define USE_GCC_BUILTIN_ATOMICS 1

#elif defined( __GNUC__ ) && (__GNUC__ == 4 && __GNUC_MINOR__ >= 1)

// available since GCC 4.1
#define USE_GCC_BUILTIN_ATOMICS_OLD 1

#endif


#if USE_CPP11_ATOMICS

#include <atomic>
#include <thread>

#define THREAD_LOCAL_STATIC thread_local static

bool btSpinMutex::tryLock()
{
    std::atomic<int>* aDest = reinterpret_cast<std::atomic<int>*>(&mLock);
    int expected = 0;
    return std::atomic_compare_exchange_weak_explicit( aDest, &expected, int(1), std::memory_order_acq_rel, std::memory_order_acquire );
}

void btSpinMutex::lock()
{
    // note: this lock does not sleep the thread.
    while (! tryLock())
    {
        // spin
    }
}

void btSpinMutex::unlock()
{
    std::atomic<int>* aDest = reinterpret_cast<std::atomic<int>*>(&mLock);
    std::atomic_store_explicit( aDest, int(0), std::memory_order_release );
}


#elif USE_MSVC_INTRINSICS

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <intrin.h>

#define THREAD_LOCAL_STATIC __declspec( thread ) static


bool btSpinMutex::tryLock()
{
    volatile long* aDest = reinterpret_cast<long*>(&mLock);
    return ( 0 == _InterlockedCompareExchange( aDest, 1, 0) );
}

void btSpinMutex::lock()
{
    // note: this lock does not sleep the thread
    while (! tryLock())
    {
        // spin
    }
}

void btSpinMutex::unlock()
{
    volatile long* aDest = reinterpret_cast<long*>( &mLock );
    _InterlockedExchange( aDest, 0 );
}

#elif USE_GCC_BUILTIN_ATOMICS

#define THREAD_LOCAL_STATIC static __thread


bool btSpinMutex::tryLock()
{
    int expected = 0;
    bool weak = false;
    const int memOrderSuccess = __ATOMIC_ACQ_REL;
    const int memOrderFail = __ATOMIC_ACQUIRE;
    return __atomic_compare_exchange_n(&mLock, &expected, int(1), weak, memOrderSuccess, memOrderFail);
}

void btSpinMutex::lock()
{
    // note: this lock does not sleep the thread
    while (! tryLock())
    {
        // spin
    }
}

void btSpinMutex::unlock()
{
    __atomic_store_n(&mLock, int(0), __ATOMIC_RELEASE);
}

#elif USE_GCC_BUILTIN_ATOMICS_OLD


#define THREAD_LOCAL_STATIC static __thread

bool btSpinMutex::tryLock()
{
    return __sync_bool_compare_and_swap(&mLock, int(0), int(1));
}

void btSpinMutex::lock()
{
    // note: this lock does not sleep the thread
    while (! tryLock())
    {
        // spin
    }
}

void btSpinMutex::unlock()
{
    // write 0
    __sync_fetch_and_and(&mLock, int(0));
}

#else //#elif USE_MSVC_INTRINSICS

#error "no threading primitives defined -- unknown platform"

#endif  //#else //#elif USE_MSVC_INTRINSICS

#else //#if BT_THREADSAFE

// These should not be called ever
void btSpinMutex::lock()
{
    btAssert( !"unimplemented btSpinMutex::lock() called" );
}

void btSpinMutex::unlock()
{
    btAssert( !"unimplemented btSpinMutex::unlock() called" );
}

bool btSpinMutex::tryLock()
{
    btAssert( !"unimplemented btSpinMutex::tryLock() called" );
    return true;
}

#define THREAD_LOCAL_STATIC static

#endif // #else //#if BT_THREADSAFE


struct ThreadsafeCounter
{
    unsigned int mCounter;
    btSpinMutex mMutex;

    ThreadsafeCounter()
    {
        mCounter = 0;
        --mCounter; // first count should come back 0
    }

    unsigned int getNext()
    {
        // no need to optimize this with atomics, it is only called ONCE per thread!
        mMutex.lock();
        mCounter++;
        if ( mCounter >= BT_MAX_THREAD_COUNT )
        {
            btAssert( !"thread counter exceeded" );
            // wrap back to the first worker index
            mCounter = 1;
        }
        unsigned int val = mCounter;
        mMutex.unlock();
        return val;
    }
};


static btITaskScheduler* gBtTaskScheduler;
static int gThreadsRunningCounter = 0;  // useful for detecting if we are trying to do nested parallel-for calls
static btSpinMutex gThreadsRunningCounterMutex;
static ThreadsafeCounter gThreadCounter;


//
// BT_DETECT_BAD_THREAD_INDEX tries to detect when there are multiple threads assigned the same thread index.
//
// BT_DETECT_BAD_THREAD_INDEX is a developer option to test if
// certain assumptions about how the task scheduler manages its threads
// holds true.
// The main assumption is:
//   - when the threadpool is resized, the task scheduler either
//      1. destroys all worker threads and creates all new ones in the correct number, OR
//      2. never destroys a worker thread
//
// We make that assumption because we can't easily enumerate the worker threads of a task scheduler
// to assign nice sequential thread-indexes. We also do not get notified if a worker thread is destroyed,
// so we can't tell when a thread-index is no longer being used.
// We allocate thread-indexes as needed with a sequential global thread counter.
//
// Our simple thread-counting scheme falls apart if the task scheduler destroys some threads but
// continues to re-use other threads and the application repeatedly resizes the thread pool of the 
// task scheduler.
// In order to prevent the thread-counter from exceeding the global max (BT_MAX_THREAD_COUNT), we
// wrap the thread counter back to 1. This should only happen if the worker threads have all been
// destroyed and re-created.
//
// BT_DETECT_BAD_THREAD_INDEX only works for Win32 right now,
// but could be adapted to work with pthreads
#define BT_DETECT_BAD_THREAD_INDEX 0

#if BT_DETECT_BAD_THREAD_INDEX

typedef DWORD ThreadId_t;
const static ThreadId_t kInvalidThreadId = 0;
ThreadId_t gDebugThreadIds[ BT_MAX_THREAD_COUNT ];

static ThreadId_t getDebugThreadId()
{
    return GetCurrentThreadId();
}

#endif // #if BT_DETECT_BAD_THREAD_INDEX


// return a unique index per thread, main thread is 0, worker threads are in [1, BT_MAX_THREAD_COUNT)
unsigned int btGetCurrentThreadIndex()
{
    const unsigned int kNullIndex = ~0U;
    THREAD_LOCAL_STATIC unsigned int sThreadIndex = kNullIndex;
    if ( sThreadIndex == kNullIndex )
    {
        sThreadIndex = gThreadCounter.getNext();
        btAssert( sThreadIndex < BT_MAX_THREAD_COUNT );
    }
#if BT_DETECT_BAD_THREAD_INDEX
    if ( gBtTaskScheduler && sThreadIndex > 0 )
    {
        ThreadId_t tid = getDebugThreadId();
        // if not set
        if ( gDebugThreadIds[ sThreadIndex ] == kInvalidThreadId )
        {
            // set it
            gDebugThreadIds[ sThreadIndex ] = tid;
        }
        else
        {
            if ( gDebugThreadIds[ sThreadIndex ] != tid )
            {
                // this could indicate the task scheduler is breaking our assumptions about
                // how threads are managed when threadpool is resized
                btAssert( !"there are 2 or more threads with the same thread-index!" );
                __debugbreak();
            }
        }
    }
#endif // #if BT_DETECT_BAD_THREAD_INDEX
    return sThreadIndex;
}

bool btIsMainThread()
{
    return btGetCurrentThreadIndex() == 0;
}

void btResetThreadIndexCounter()
{
    // for when all current worker threads are destroyed
    btAssert( btIsMainThread() );
    gThreadCounter.mCounter = 0;
}

btITaskScheduler::btITaskScheduler( const char* name )
{
    m_name = name;
    m_savedThreadCounter = 0;
    m_isActive = false;
}

void btITaskScheduler::activate()
{
    // gThreadCounter is used to assign a thread-index to each worker thread in a task scheduler.
    // The main thread is always thread-index 0, and worker threads are numbered from 1 to 63 (BT_MAX_THREAD_COUNT-1)
    // The thread-indexes need to be unique amongst the threads that can be running simultaneously.
    // Since only one task scheduler can be used at a time, it is OK for a pair of threads that belong to different
    // task schedulers to share the same thread index because they can't be running at the same time.
    // So each task scheduler needs to keep its own thread counter value
    if ( !m_isActive )
    {
        gThreadCounter.mCounter = m_savedThreadCounter;  // restore saved thread counter
        m_isActive = true;
    }
}

void btITaskScheduler::deactivate()
{
    if ( m_isActive )
    {
        m_savedThreadCounter = gThreadCounter.mCounter;  // save thread counter
        m_isActive = false;
    }
}

void btPushThreadsAreRunning()
{
    gThreadsRunningCounterMutex.lock();
    gThreadsRunningCounter++;
    gThreadsRunningCounterMutex.unlock();
}

void btPopThreadsAreRunning()
{
    gThreadsRunningCounterMutex.lock();
    gThreadsRunningCounter--;
    gThreadsRunningCounterMutex.unlock();
}

bool btThreadsAreRunning()
{
    return gThreadsRunningCounter != 0;
}


void btSetTaskScheduler( btITaskScheduler* ts )
{
    int threadId = btGetCurrentThreadIndex();  // make sure we call this on main thread at least once before any workers run
    if ( threadId != 0 )
    {
        btAssert( !"btSetTaskScheduler must be called from the main thread!" );
        return;
    }
    if ( gBtTaskScheduler )
    {
        // deactivate old task scheduler
        gBtTaskScheduler->deactivate();
    }
    gBtTaskScheduler = ts;
    if ( ts )
    {
        // activate new task scheduler
        ts->activate();
    }
}


btITaskScheduler* btGetTaskScheduler()
{
    return gBtTaskScheduler;
}


void btParallelFor( int iBegin, int iEnd, int grainSize, const btIParallelForBody& body )
{
#if BT_THREADSAFE

#if BT_DETECT_BAD_THREAD_INDEX
    if ( !btThreadsAreRunning() )
    {
        // clear out thread ids
        for ( int i = 0; i < BT_MAX_THREAD_COUNT; ++i )
        {
            gDebugThreadIds[ i ] = kInvalidThreadId;
        }
    }
#endif // #if BT_DETECT_BAD_THREAD_INDEX

    btAssert( gBtTaskScheduler != NULL );  // call btSetTaskScheduler() with a valid task scheduler first!
    gBtTaskScheduler->parallelFor( iBegin, iEnd, grainSize, body );

#else // #if BT_THREADSAFE

    // non-parallel version of btParallelFor
    btAssert( !"called btParallelFor in non-threadsafe build. enable BT_THREADSAFE" );
    body.forLoop( iBegin, iEnd );

#endif// #if BT_THREADSAFE
}

btScalar btParallelSum( int iBegin, int iEnd, int grainSize, const btIParallelSumBody& body )
{
#if BT_THREADSAFE

#if BT_DETECT_BAD_THREAD_INDEX
    if ( !btThreadsAreRunning() )
    {
        // clear out thread ids
        for ( int i = 0; i < BT_MAX_THREAD_COUNT; ++i )
        {
            gDebugThreadIds[ i ] = kInvalidThreadId;
        }
    }
#endif // #if BT_DETECT_BAD_THREAD_INDEX

    btAssert( gBtTaskScheduler != NULL );  // call btSetTaskScheduler() with a valid task scheduler first!
    return gBtTaskScheduler->parallelSum( iBegin, iEnd, grainSize, body );

#else // #if BT_THREADSAFE

    // non-parallel version of btParallelSum
    btAssert( !"called btParallelFor in non-threadsafe build. enable BT_THREADSAFE" );
    return body.sumLoop( iBegin, iEnd );

#endif //#else // #if BT_THREADSAFE
}


///
/// btTaskSchedulerSequential -- non-threaded implementation of task scheduler
///                              (really just useful for testing performance of single threaded vs multi)
///
class btTaskSchedulerSequential : public btITaskScheduler
{
public:
    btTaskSchedulerSequential() : btITaskScheduler( "Sequential" ) {}
    virtual int getMaxNumThreads() const BT_OVERRIDE { return 1; }
    virtual int getNumThreads() const BT_OVERRIDE { return 1; }
    virtual void setNumThreads( int numThreads ) BT_OVERRIDE {}
    virtual void parallelFor( int iBegin, int iEnd, int grainSize, const btIParallelForBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelFor_sequential" );
        body.forLoop( iBegin, iEnd );
    }
    virtual btScalar parallelSum( int iBegin, int iEnd, int grainSize, const btIParallelSumBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelSum_sequential" );
        return body.sumLoop( iBegin, iEnd );
    }
};


#if BT_USE_OPENMP && BT_THREADSAFE
///
/// btTaskSchedulerOpenMP -- wrapper around OpenMP task scheduler
///
class btTaskSchedulerOpenMP : public btITaskScheduler
{
    int m_numThreads;
public:
    btTaskSchedulerOpenMP() : btITaskScheduler( "OpenMP" )
    {
        m_numThreads = 0;
    }
    virtual int getMaxNumThreads() const BT_OVERRIDE
    {
        return omp_get_max_threads();
    }
    virtual int getNumThreads() const BT_OVERRIDE
    {
        return m_numThreads;
    }
    virtual void setNumThreads( int numThreads ) BT_OVERRIDE
    {
        // With OpenMP, because it is a standard with various implementations, we can't
        // know for sure if every implementation has the same behavior of destroying all
        // previous threads when resizing the threadpool
        m_numThreads = ( std::max )( 1, ( std::min )( int( BT_MAX_THREAD_COUNT ), numThreads ) );
        omp_set_num_threads( 1 );  // hopefully, all previous threads get destroyed here
        omp_set_num_threads( m_numThreads );
        m_savedThreadCounter = 0;
        if ( m_isActive )
        {
            btResetThreadIndexCounter();
        }
    }
    virtual void parallelFor( int iBegin, int iEnd, int grainSize, const btIParallelForBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelFor_OpenMP" );
        btPushThreadsAreRunning();
#pragma omp parallel for schedule( static, 1 )
        for ( int i = iBegin; i < iEnd; i += grainSize )
        {
            BT_PROFILE( "OpenMP_forJob" );
            body.forLoop( i, ( std::min )( i + grainSize, iEnd ) );
        }
        btPopThreadsAreRunning();
    }
    virtual btScalar parallelSum( int iBegin, int iEnd, int grainSize, const btIParallelSumBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelFor_OpenMP" );
        btPushThreadsAreRunning();
        btScalar sum = btScalar( 0 );
#pragma omp parallel for schedule( static, 1 ) reduction(+:sum)
        for ( int i = iBegin; i < iEnd; i += grainSize )
        {
            BT_PROFILE( "OpenMP_sumJob" );
            sum += body.sumLoop( i, ( std::min )( i + grainSize, iEnd ) );
        }
        btPopThreadsAreRunning();
        return sum;
    }
};
#endif // #if BT_USE_OPENMP && BT_THREADSAFE


#if BT_USE_TBB && BT_THREADSAFE
///
/// btTaskSchedulerTBB -- wrapper around Intel Threaded Building Blocks task scheduler
///
class btTaskSchedulerTBB : public btITaskScheduler
{
    int m_numThreads;
    tbb::task_scheduler_init* m_tbbSchedulerInit;

public:
    btTaskSchedulerTBB() : btITaskScheduler( "IntelTBB" )
    {
        m_numThreads = 0;
        m_tbbSchedulerInit = NULL;
    }
    ~btTaskSchedulerTBB()
    {
        if ( m_tbbSchedulerInit )
        {
            delete m_tbbSchedulerInit;
            m_tbbSchedulerInit = NULL;
        }
    }

    virtual int getMaxNumThreads() const BT_OVERRIDE
    {
        return tbb::task_scheduler_init::default_num_threads();
    }
    virtual int getNumThreads() const BT_OVERRIDE
    {
        return m_numThreads;
    }
    virtual void setNumThreads( int numThreads ) BT_OVERRIDE
    {
        m_numThreads = ( std::max )( 1, ( std::min )( int(BT_MAX_THREAD_COUNT), numThreads ) );
        if ( m_tbbSchedulerInit )
        {
            // destroys all previous threads
            delete m_tbbSchedulerInit;
            m_tbbSchedulerInit = NULL;
        }
        m_tbbSchedulerInit = new tbb::task_scheduler_init( m_numThreads );
        m_savedThreadCounter = 0;
        if ( m_isActive )
        {
            btResetThreadIndexCounter();
        }
    }
    struct ForBodyAdapter
    {
        const btIParallelForBody* mBody;

        ForBodyAdapter( const btIParallelForBody* body ) : mBody( body ) {}
        void operator()( const tbb::blocked_range<int>& range ) const
        {
            BT_PROFILE( "TBB_forJob" );
            mBody->forLoop( range.begin(), range.end() );
        }
    };
    virtual void parallelFor( int iBegin, int iEnd, int grainSize, const btIParallelForBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelFor_TBB" );
        ForBodyAdapter tbbBody( &body );
        btPushThreadsAreRunning();
        tbb::parallel_for( tbb::blocked_range<int>( iBegin, iEnd, grainSize ),
            tbbBody,
            tbb::simple_partitioner()
        );
        btPopThreadsAreRunning();
    }
    struct SumBodyAdapter
    {
        const btIParallelSumBody* mBody;
        btScalar mSum;

        SumBodyAdapter( const btIParallelSumBody* body ) : mBody( body ), mSum( btScalar( 0 ) ) {}
        SumBodyAdapter( const SumBodyAdapter& src, tbb::split ) : mBody( src.mBody ), mSum( btScalar( 0 ) ) {}
        void join( const SumBodyAdapter& src ) { mSum += src.mSum; }
        void operator()( const tbb::blocked_range<int>& range )
        {
            BT_PROFILE( "TBB_sumJob" );
            mSum += mBody->sumLoop( range.begin(), range.end() );
        }
    };
    virtual btScalar parallelSum( int iBegin, int iEnd, int grainSize, const btIParallelSumBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelSum_TBB" );
        SumBodyAdapter tbbBody( &body );
        btPushThreadsAreRunning();
        tbb::parallel_deterministic_reduce( tbb::blocked_range<int>( iBegin, iEnd, grainSize ), tbbBody );
        btPopThreadsAreRunning();
        return tbbBody.mSum;
    }
};
#endif // #if BT_USE_TBB && BT_THREADSAFE


#if BT_USE_PPL && BT_THREADSAFE
///
/// btTaskSchedulerPPL -- wrapper around Microsoft Parallel Patterns Lib task scheduler
///
class btTaskSchedulerPPL : public btITaskScheduler
{
    int m_numThreads;
    concurrency::combinable<btScalar> m_sum;  // for parallelSum
public:
    btTaskSchedulerPPL() : btITaskScheduler( "PPL" )
    {
        m_numThreads = 0;
    }
    virtual int getMaxNumThreads() const BT_OVERRIDE
    {
        return concurrency::GetProcessorCount();
    }
    virtual int getNumThreads() const BT_OVERRIDE
    {
        return m_numThreads;
    }
    virtual void setNumThreads( int numThreads ) BT_OVERRIDE
    {
        // capping the thread count for PPL due to a thread-index issue
        const int maxThreadCount = (std::min)(int(BT_MAX_THREAD_COUNT), 31);
        m_numThreads = ( std::max )( 1, ( std::min )( maxThreadCount, numThreads ) );
        using namespace concurrency;
        if ( CurrentScheduler::Id() != -1 )
        {
            CurrentScheduler::Detach();
        }
        SchedulerPolicy policy;
        {
            // PPL seems to destroy threads when threadpool is shrunk, but keeps reusing old threads
            // force it to destroy old threads
            policy.SetConcurrencyLimits( 1, 1 );
            CurrentScheduler::Create( policy );
            CurrentScheduler::Detach();
        }
        policy.SetConcurrencyLimits( m_numThreads, m_numThreads );
        CurrentScheduler::Create( policy );
        m_savedThreadCounter = 0;
        if ( m_isActive )
        {
            btResetThreadIndexCounter();
        }
    }
    struct ForBodyAdapter
    {
        const btIParallelForBody* mBody;
        int mGrainSize;
        int mIndexEnd;

        ForBodyAdapter( const btIParallelForBody* body, int grainSize, int end ) : mBody( body ), mGrainSize( grainSize ), mIndexEnd( end ) {}
        void operator()( int i ) const
        {
            BT_PROFILE( "PPL_forJob" );
            mBody->forLoop( i, ( std::min )( i + mGrainSize, mIndexEnd ) );
        }
    };
    virtual void parallelFor( int iBegin, int iEnd, int grainSize, const btIParallelForBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelFor_PPL" );
        // PPL dispatch
        ForBodyAdapter pplBody( &body, grainSize, iEnd );
        btPushThreadsAreRunning();
        // note: MSVC 2010 doesn't support partitioner args, so avoid them
        concurrency::parallel_for( iBegin,
            iEnd,
            grainSize,
            pplBody
        );
        btPopThreadsAreRunning();
    }
    struct SumBodyAdapter
    {
        const btIParallelSumBody* mBody;
        concurrency::combinable<btScalar>* mSum;
        int mGrainSize;
        int mIndexEnd;

        SumBodyAdapter( const btIParallelSumBody* body, concurrency::combinable<btScalar>* sum, int grainSize, int end ) : mBody( body ), mSum(sum), mGrainSize( grainSize ), mIndexEnd( end ) {}
        void operator()( int i ) const
        {
            BT_PROFILE( "PPL_sumJob" );
            mSum->local() += mBody->sumLoop( i, ( std::min )( i + mGrainSize, mIndexEnd ) );
        }
    };
    static btScalar sumFunc( btScalar a, btScalar b ) { return a + b; }
    virtual btScalar parallelSum( int iBegin, int iEnd, int grainSize, const btIParallelSumBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelSum_PPL" );
        m_sum.clear();
        SumBodyAdapter pplBody( &body, &m_sum, grainSize, iEnd );
        btPushThreadsAreRunning();
        // note: MSVC 2010 doesn't support partitioner args, so avoid them
        concurrency::parallel_for( iBegin,
            iEnd,
            grainSize,
            pplBody
        );
        btPopThreadsAreRunning();
        return m_sum.combine( sumFunc );
    }
};
#endif // #if BT_USE_PPL && BT_THREADSAFE


// create a non-threaded task scheduler (always available)
btITaskScheduler* btGetSequentialTaskScheduler()
{
    static btTaskSchedulerSequential sTaskScheduler;
    return &sTaskScheduler;
}


// create an OpenMP task scheduler (if available, otherwise returns null)
btITaskScheduler* btGetOpenMPTaskScheduler()
{
#if BT_USE_OPENMP && BT_THREADSAFE
    static btTaskSchedulerOpenMP sTaskScheduler;
    return &sTaskScheduler;
#else
    return NULL;
#endif
}


// create an Intel TBB task scheduler (if available, otherwise returns null)
btITaskScheduler* btGetTBBTaskScheduler()
{
#if BT_USE_TBB && BT_THREADSAFE
    static btTaskSchedulerTBB sTaskScheduler;
    return &sTaskScheduler;
#else
    return NULL;
#endif
}


// create a PPL task scheduler (if available, otherwise returns null)
btITaskScheduler* btGetPPLTaskScheduler()
{
#if BT_USE_PPL && BT_THREADSAFE
    static btTaskSchedulerPPL sTaskScheduler;
    return &sTaskScheduler;
#else
    return NULL;
#endif
}

