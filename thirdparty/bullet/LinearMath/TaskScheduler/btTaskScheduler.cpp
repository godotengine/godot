
#include "LinearMath/btMinMax.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btThreads.h"
#include "LinearMath/btQuickprof.h"
#include <stdio.h>
#include <algorithm>



#if BT_THREADSAFE

#include "btThreadSupportInterface.h"

#if defined( _WIN32 )

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#endif


typedef unsigned long long btU64;
static const int kCacheLineSize = 64;

void btSpinPause()
{
#if defined( _WIN32 )
    YieldProcessor();
#endif
}


struct WorkerThreadStatus
{
    enum Type
    {
        kInvalid,
        kWaitingForWork,
        kWorking,
        kSleeping,
    };
};


ATTRIBUTE_ALIGNED64(class) WorkerThreadDirectives
{
    static const int kMaxThreadCount = BT_MAX_THREAD_COUNT;
    // directives for all worker threads packed into a single cacheline
    char m_threadDirs[kMaxThreadCount];

public:
    enum Type
    {
        kInvalid,
        kGoToSleep,         // go to sleep
        kStayAwakeButIdle,  // wait for not checking job queue
        kScanForJobs,       // actively scan job queue for jobs
    };
    WorkerThreadDirectives()
    {
        for ( int i = 0; i < kMaxThreadCount; ++i )
        {
            m_threadDirs[ i ] = 0;
        }
    }

    Type getDirective(int threadId)
    {
        btAssert(threadId < kMaxThreadCount);
        return static_cast<Type>(m_threadDirs[threadId]);
    }

    void setDirectiveByRange(int threadBegin, int threadEnd, Type dir)
    {
        btAssert( threadBegin < threadEnd );
        btAssert( threadEnd <= kMaxThreadCount );
        char dirChar = static_cast<char>(dir);
        for ( int i = threadBegin; i < threadEnd; ++i )
        {
            m_threadDirs[ i ] = dirChar;
        }
    }
};

class JobQueue;

ATTRIBUTE_ALIGNED64(struct) ThreadLocalStorage
{
    int m_threadId;
    WorkerThreadStatus::Type m_status;
    int m_numJobsFinished;
    btSpinMutex m_mutex;
    btScalar m_sumResult;
    WorkerThreadDirectives * m_directive;
    JobQueue* m_queue;
    btClock* m_clock;
    unsigned int m_cooldownTime;
};


struct IJob
{
    virtual void executeJob(int threadId) = 0;
};

class ParallelForJob : public IJob
{
    const btIParallelForBody* m_body;
    int m_begin;
    int m_end;

public:
    ParallelForJob( int iBegin, int iEnd, const btIParallelForBody& body )
    {
        m_body = &body;
        m_begin = iBegin;
        m_end = iEnd;
    }
    virtual void executeJob(int threadId) BT_OVERRIDE
    {
        BT_PROFILE( "executeJob" );

        // call the functor body to do the work
        m_body->forLoop( m_begin, m_end );
    }
};


class ParallelSumJob : public IJob
{
    const btIParallelSumBody* m_body;
    ThreadLocalStorage* m_threadLocalStoreArray;
    int m_begin;
    int m_end;

public:
    ParallelSumJob( int iBegin, int iEnd, const btIParallelSumBody& body, ThreadLocalStorage* tls )
    {
        m_body = &body;
        m_threadLocalStoreArray = tls;
        m_begin = iBegin;
        m_end = iEnd;
    }
    virtual void executeJob( int threadId ) BT_OVERRIDE
    {
        BT_PROFILE( "executeJob" );

        // call the functor body to do the work
        btScalar val = m_body->sumLoop( m_begin, m_end );
#if BT_PARALLEL_SUM_DETERMINISTISM
        // by truncating bits of the result, we can make the parallelSum deterministic (at the expense of precision)
        const float TRUNC_SCALE = float(1<<19);
        val = floor(val*TRUNC_SCALE+0.5f)/TRUNC_SCALE;  // truncate some bits
#endif
        m_threadLocalStoreArray[threadId].m_sumResult += val;
    }
};


ATTRIBUTE_ALIGNED64(class) JobQueue
{
    btThreadSupportInterface* m_threadSupport;
    btCriticalSection* m_queueLock;
    btSpinMutex m_mutex;

    btAlignedObjectArray<IJob*> m_jobQueue;
    char* m_jobMem;
    int m_jobMemSize;
    bool m_queueIsEmpty;
    int m_tailIndex;
    int m_headIndex;
    int m_allocSize;
    bool m_useSpinMutex;
    btAlignedObjectArray<JobQueue*> m_neighborContexts;
    char m_cachePadding[kCacheLineSize];  // prevent false sharing

    void freeJobMem()
    {
        if ( m_jobMem )
        {
            // free old
            btAlignedFree(m_jobMem);
            m_jobMem = NULL;
        }
    }
    void resizeJobMem(int newSize)
    {
        if (newSize > m_jobMemSize)
        {
            freeJobMem();
            m_jobMem = static_cast<char*>(btAlignedAlloc(newSize, kCacheLineSize));
            m_jobMemSize = newSize;
        }
    }

public:

    JobQueue()
    {
        m_jobMem = NULL;
        m_jobMemSize = 0;
        m_threadSupport = NULL;
        m_queueLock = NULL;
        m_headIndex = 0;
        m_tailIndex = 0;
        m_useSpinMutex = false;
    }
    ~JobQueue()
    {
		exit();
    }
	void exit()
    {
		freeJobMem();
        if (m_queueLock && m_threadSupport)
        {
            m_threadSupport->deleteCriticalSection(m_queueLock);
            m_queueLock = NULL;
			m_threadSupport = 0;
        }
	}

    void init(btThreadSupportInterface* threadSup, btAlignedObjectArray<JobQueue>* contextArray)
    {
        m_threadSupport = threadSup;
        if (threadSup)
        {
            m_queueLock = m_threadSupport->createCriticalSection();
        }
        setupJobStealing(contextArray, contextArray->size());
    }
    void setupJobStealing(btAlignedObjectArray<JobQueue>* contextArray, int numActiveContexts)
    {
        btAlignedObjectArray<JobQueue>& contexts = *contextArray;
        int selfIndex = 0;
        for (int i = 0; i < contexts.size(); ++i)
        {
            if ( this == &contexts[ i ] )
            {
                selfIndex = i;
                break;
            }
        }
        int numNeighbors = btMin(2, contexts.size() - 1);
        int neighborOffsets[ ] = {-1, 1, -2, 2, -3, 3};
        int numOffsets = sizeof(neighborOffsets)/sizeof(neighborOffsets[0]);
        m_neighborContexts.reserve( numNeighbors );
        m_neighborContexts.resizeNoInitialize(0);
        for (int i = 0; i < numOffsets && m_neighborContexts.size() < numNeighbors; i++)
        {
            int neighborIndex = selfIndex + neighborOffsets[i];
            if ( neighborIndex >= 0 && neighborIndex < numActiveContexts)
            {
                m_neighborContexts.push_back( &contexts[ neighborIndex ] );
            }
        }
    }

    bool isQueueEmpty() const {return m_queueIsEmpty;}
    void lockQueue()
    {
        if ( m_useSpinMutex )
        {
            m_mutex.lock();
        }
        else
        {
            m_queueLock->lock();
        }
    }
    void unlockQueue()
    {
        if ( m_useSpinMutex )
        {
            m_mutex.unlock();
        }
        else
        {
            m_queueLock->unlock();
        }
    }
    void clearQueue(int jobCount, int jobSize)
    {
        lockQueue();
        m_headIndex = 0;
        m_tailIndex = 0;
        m_allocSize = 0;
        m_queueIsEmpty = true;
        int jobBufSize = jobSize * jobCount;
        // make sure we have enough memory allocated to store jobs
        if ( jobBufSize > m_jobMemSize )
        {
            resizeJobMem( jobBufSize );
        }
        // make sure job queue is big enough
        if ( jobCount > m_jobQueue.capacity() )
        {
            m_jobQueue.reserve( jobCount );
        }
        unlockQueue();
        m_jobQueue.resizeNoInitialize( 0 );
    }
    void* allocJobMem(int jobSize)
    {
        btAssert(m_jobMemSize >= (m_allocSize + jobSize));
        void* jobMem = &m_jobMem[m_allocSize];
        m_allocSize += jobSize;
        return jobMem;
    }
    void submitJob( IJob* job )
    {
        btAssert( reinterpret_cast<char*>( job ) >= &m_jobMem[ 0 ] && reinterpret_cast<char*>( job ) < &m_jobMem[ 0 ] + m_allocSize );
        m_jobQueue.push_back( job );
        lockQueue();
        m_tailIndex++;
        m_queueIsEmpty = false;
        unlockQueue();
    }
    IJob* consumeJobFromOwnQueue()
    {
        if ( m_queueIsEmpty )
        {
            // lock free path. even if this is taken erroneously it isn't harmful
            return NULL;
        }
        IJob* job = NULL;
        lockQueue();
        if ( !m_queueIsEmpty )
        {
            job = m_jobQueue[ m_headIndex++ ];
            btAssert( reinterpret_cast<char*>( job ) >= &m_jobMem[ 0 ] && reinterpret_cast<char*>( job ) < &m_jobMem[ 0 ] + m_allocSize );
            if ( m_headIndex == m_tailIndex )
            {
                m_queueIsEmpty = true;
            }
        }
        unlockQueue();
        return job;
    }
    IJob* consumeJob()
    {
        if (IJob* job = consumeJobFromOwnQueue())
        {
            return job;
        }
        // own queue is empty, try to steal from neighbor
        for (int i = 0; i < m_neighborContexts.size(); ++i)
        {
            JobQueue* otherContext = m_neighborContexts[ i ];
            if ( IJob* job = otherContext->consumeJobFromOwnQueue() )
            {
                return job;
            }
        }
        return NULL;
    }
};


static void WorkerThreadFunc( void* userPtr )
{
    BT_PROFILE( "WorkerThreadFunc" );
    ThreadLocalStorage* localStorage = (ThreadLocalStorage*) userPtr;
    JobQueue* jobQueue = localStorage->m_queue;

    bool shouldSleep = false;
    int threadId = localStorage->m_threadId;
    while (! shouldSleep)
    {
        // do work
        localStorage->m_mutex.lock();
        while ( IJob* job = jobQueue->consumeJob() )
        {
            localStorage->m_status = WorkerThreadStatus::kWorking;
            job->executeJob( threadId );
            localStorage->m_numJobsFinished++;
        }
        localStorage->m_status = WorkerThreadStatus::kWaitingForWork;
        localStorage->m_mutex.unlock();
        btU64 clockStart = localStorage->m_clock->getTimeMicroseconds();
        // while queue is empty,
        while (jobQueue->isQueueEmpty())
        {
            // todo: spin wait a bit to avoid hammering the empty queue
            btSpinPause();
            if ( localStorage->m_directive->getDirective(threadId) == WorkerThreadDirectives::kGoToSleep )
            {
                shouldSleep = true;
                break;
            }
            // if jobs are incoming,
            if ( localStorage->m_directive->getDirective( threadId ) == WorkerThreadDirectives::kScanForJobs )
            {
                clockStart = localStorage->m_clock->getTimeMicroseconds(); // reset clock
            }
            else
            {
                for ( int i = 0; i < 50; ++i )
                {
                    btSpinPause();
                    btSpinPause();
                    btSpinPause();
                    btSpinPause();
                    if (localStorage->m_directive->getDirective( threadId ) == WorkerThreadDirectives::kScanForJobs || !jobQueue->isQueueEmpty())
                    {
                        break;
                    }
                }
                // if no jobs incoming and queue has been empty for the cooldown time, sleep
                btU64 timeElapsed = localStorage->m_clock->getTimeMicroseconds() - clockStart;
                if (timeElapsed > localStorage->m_cooldownTime)
                {
                    shouldSleep = true;
                    break;
                }
            }
        }
    }
	{
		BT_PROFILE("sleep");
		// go sleep
		localStorage->m_mutex.lock();
		localStorage->m_status = WorkerThreadStatus::kSleeping;
		localStorage->m_mutex.unlock();
	}
}


class btTaskSchedulerDefault : public btITaskScheduler
{
    btThreadSupportInterface* m_threadSupport;
    WorkerThreadDirectives* m_workerDirective;
    btAlignedObjectArray<JobQueue> m_jobQueues;
    btAlignedObjectArray<JobQueue*> m_perThreadJobQueues;
    btAlignedObjectArray<ThreadLocalStorage> m_threadLocalStorage;
    btSpinMutex m_antiNestingLock;  // prevent nested parallel-for
    btClock m_clock;
    int m_numThreads;
    int m_numWorkerThreads;
    int m_numActiveJobQueues;
    int m_maxNumThreads;
    int m_numJobs;
    static const int kFirstWorkerThreadId = 1;
public:

    btTaskSchedulerDefault() : btITaskScheduler("ThreadSupport")
    {
        m_threadSupport = NULL;
        m_workerDirective = NULL;
    }

    virtual ~btTaskSchedulerDefault()
    {
        waitForWorkersToSleep();

		for ( int i = 0; i < m_jobQueues.size(); ++i )
        {
            m_jobQueues[i].exit();
        }

        if (m_threadSupport)
        {
            delete m_threadSupport;
            m_threadSupport = NULL;
        }
        if (m_workerDirective)
        {
            btAlignedFree(m_workerDirective);
            m_workerDirective = NULL;
        }
    }

    void init()
    {
        btThreadSupportInterface::ConstructionInfo constructionInfo( "TaskScheduler", WorkerThreadFunc );
        m_threadSupport = btThreadSupportInterface::create( constructionInfo );
        m_workerDirective = static_cast<WorkerThreadDirectives*>(btAlignedAlloc(sizeof(*m_workerDirective), 64));

        m_numWorkerThreads = m_threadSupport->getNumWorkerThreads();
        m_maxNumThreads = m_threadSupport->getNumWorkerThreads() + 1;
        m_numThreads = m_maxNumThreads;
        // ideal to have one job queue for each physical processor (except for the main thread which needs no queue)
        int numThreadsPerQueue = m_threadSupport->getLogicalToPhysicalCoreRatio();
        int numJobQueues = (numThreadsPerQueue == 1) ? (m_maxNumThreads-1) : (m_maxNumThreads / numThreadsPerQueue);
        m_jobQueues.resize(numJobQueues);
        m_numActiveJobQueues = numJobQueues;
        for ( int i = 0; i < m_jobQueues.size(); ++i )
        {
            m_jobQueues[i].init( m_threadSupport, &m_jobQueues );
        }
        m_perThreadJobQueues.resize(m_numThreads);
        for ( int i = 0; i < m_numThreads; i++ )
        {
            JobQueue* jq = NULL;
            // only worker threads get a job queue
            if (i > 0)
            {
                if (numThreadsPerQueue == 1)
                {
                    // one queue per worker thread
                    jq = &m_jobQueues[ i - kFirstWorkerThreadId ];
                }
                else
                {
                    // 2 threads share each queue
                    jq = &m_jobQueues[ i / numThreadsPerQueue ];
                }
            }
            m_perThreadJobQueues[i] = jq;
        }
        m_threadLocalStorage.resize(m_numThreads);
        for ( int i = 0; i < m_numThreads; i++ )
        {
            ThreadLocalStorage& storage = m_threadLocalStorage[i];
            storage.m_threadId = i;
            storage.m_directive = m_workerDirective;
            storage.m_status = WorkerThreadStatus::kSleeping;
            storage.m_cooldownTime = 100; // 100 microseconds, threads go to sleep after this long if they have nothing to do
            storage.m_clock = &m_clock;
            storage.m_queue = m_perThreadJobQueues[i];
        }
        setWorkerDirectives( WorkerThreadDirectives::kGoToSleep ); // no work for them yet
        setNumThreads( m_threadSupport->getCacheFriendlyNumThreads() );
    }

    void setWorkerDirectives(WorkerThreadDirectives::Type dir)
    {
        m_workerDirective->setDirectiveByRange(kFirstWorkerThreadId, m_numThreads, dir);
    }

    virtual int getMaxNumThreads() const BT_OVERRIDE
    {
        return m_maxNumThreads;
    }

    virtual int getNumThreads() const BT_OVERRIDE
    {
        return m_numThreads;
    }

    virtual void setNumThreads( int numThreads ) BT_OVERRIDE
    {
        m_numThreads = btMax( btMin(numThreads, int(m_maxNumThreads)), 1 );
        m_numWorkerThreads = m_numThreads - 1;
        m_numActiveJobQueues = 0;
        // if there is at least 1 worker,
        if ( m_numWorkerThreads > 0 )
        {
            // re-setup job stealing between queues to avoid attempting to steal from an inactive job queue
            JobQueue* lastActiveContext = m_perThreadJobQueues[ m_numThreads - 1 ];
            int iLastActiveContext = lastActiveContext - &m_jobQueues[0];
            m_numActiveJobQueues = iLastActiveContext + 1;
            for ( int i = 0; i < m_jobQueues.size(); ++i )
            {
                m_jobQueues[ i ].setupJobStealing( &m_jobQueues, m_numActiveJobQueues );
            }
        }
        m_workerDirective->setDirectiveByRange(m_numThreads, BT_MAX_THREAD_COUNT, WorkerThreadDirectives::kGoToSleep);
    }

    void waitJobs()
    {
        BT_PROFILE( "waitJobs" );
        // have the main thread work until the job queues are empty
        int numMainThreadJobsFinished = 0;
        for ( int i = 0; i < m_numActiveJobQueues; ++i )
        {
            while ( IJob* job = m_jobQueues[i].consumeJob() )
            {
                job->executeJob( 0 );
                numMainThreadJobsFinished++;
            }
        }

        // done with jobs for now, tell workers to rest (but not sleep)
        setWorkerDirectives( WorkerThreadDirectives::kStayAwakeButIdle );

        btU64 clockStart = m_clock.getTimeMicroseconds();
        // wait for workers to finish any jobs in progress
        while ( true )
        {
            int numWorkerJobsFinished = 0;
            for ( int iThread = kFirstWorkerThreadId; iThread < m_numThreads; ++iThread )
            {
                ThreadLocalStorage* storage = &m_threadLocalStorage[iThread];
                storage->m_mutex.lock();
                numWorkerJobsFinished += storage->m_numJobsFinished;
                storage->m_mutex.unlock();
            }
            if (numWorkerJobsFinished + numMainThreadJobsFinished == m_numJobs)
            {
                break;
            }
            btU64 timeElapsed = m_clock.getTimeMicroseconds() - clockStart;
            btAssert(timeElapsed < 1000);
            if (timeElapsed > 100000)
            {
                break;
            }
            btSpinPause();
        }
    }

    void wakeWorkers(int numWorkersToWake)
    {
        BT_PROFILE( "wakeWorkers" );
        btAssert( m_workerDirective->getDirective(1) == WorkerThreadDirectives::kScanForJobs );
        int numDesiredWorkers = btMin(numWorkersToWake, m_numWorkerThreads);
        int numActiveWorkers = 0;
        for ( int iWorker = 0; iWorker < m_numWorkerThreads; ++iWorker )
        {
            // note this count of active workers is not necessarily totally reliable, because a worker thread could be
            // just about to put itself to sleep. So we may on occasion fail to wake up all the workers. It should be rare.
            ThreadLocalStorage& storage = m_threadLocalStorage[ kFirstWorkerThreadId + iWorker ];
            if (storage.m_status != WorkerThreadStatus::kSleeping)
            {
                numActiveWorkers++;
            }
        }
        for ( int iWorker = 0; iWorker < m_numWorkerThreads && numActiveWorkers < numDesiredWorkers; ++iWorker )
        {
            ThreadLocalStorage& storage = m_threadLocalStorage[ kFirstWorkerThreadId + iWorker ];
            if (storage.m_status == WorkerThreadStatus::kSleeping)
            {
                m_threadSupport->runTask( iWorker, &storage );
                numActiveWorkers++;
            }
        }
    }

    void waitForWorkersToSleep()
    {
        BT_PROFILE( "waitForWorkersToSleep" );
        setWorkerDirectives( WorkerThreadDirectives::kGoToSleep );
        m_threadSupport->waitForAllTasks();
        for ( int i = kFirstWorkerThreadId; i < m_numThreads; i++ )
        {
            ThreadLocalStorage& storage = m_threadLocalStorage[i];
            btAssert( storage.m_status == WorkerThreadStatus::kSleeping );
        }
    }

    virtual void sleepWorkerThreadsHint() BT_OVERRIDE
    {
        BT_PROFILE( "sleepWorkerThreadsHint" );
        // hint the task scheduler that we may not be using these threads for a little while
        setWorkerDirectives( WorkerThreadDirectives::kGoToSleep );
    }

    void prepareWorkerThreads()
    {
        for ( int i = kFirstWorkerThreadId; i < m_numThreads; ++i )
        {
            ThreadLocalStorage& storage = m_threadLocalStorage[i];
            storage.m_mutex.lock();
            storage.m_numJobsFinished = 0;
            storage.m_mutex.unlock();
        }
        setWorkerDirectives( WorkerThreadDirectives::kScanForJobs );
    }

    virtual void parallelFor( int iBegin, int iEnd, int grainSize, const btIParallelForBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelFor_ThreadSupport" );
        btAssert( iEnd >= iBegin );
        btAssert( grainSize >= 1 );
        int iterationCount = iEnd - iBegin;
        if ( iterationCount > grainSize && m_numWorkerThreads > 0 && m_antiNestingLock.tryLock() )
        {
            typedef ParallelForJob JobType;
            int jobCount = ( iterationCount + grainSize - 1 ) / grainSize;
            m_numJobs = jobCount;
            btAssert( jobCount >= 2 );  // need more than one job for multithreading
            int jobSize = sizeof( JobType );

            for (int i = 0; i < m_numActiveJobQueues; ++i)
            {
                m_jobQueues[i].clearQueue( jobCount, jobSize );
            }
            // prepare worker threads for incoming work
            prepareWorkerThreads();
            // submit all of the jobs
            int iJob = 0;
            int iThread = kFirstWorkerThreadId;  // first worker thread
            for ( int i = iBegin; i < iEnd; i += grainSize )
            {
                btAssert( iJob < jobCount );
                int iE = btMin( i + grainSize, iEnd );
                JobQueue* jq = m_perThreadJobQueues[ iThread ];
                btAssert(jq);
                btAssert((jq - &m_jobQueues[0]) < m_numActiveJobQueues);
                void* jobMem = jq->allocJobMem(jobSize);
                JobType* job = new ( jobMem ) ParallelForJob( i, iE, body );  // placement new
                jq->submitJob( job );
                iJob++;
                iThread++;
                if ( iThread >= m_numThreads )
                {
                    iThread = kFirstWorkerThreadId;  // first worker thread
                }
            }
            wakeWorkers( jobCount - 1 );

            // put the main thread to work on emptying the job queue and then wait for all workers to finish
            waitJobs();
            m_antiNestingLock.unlock();
        }
        else
        {
            BT_PROFILE( "parallelFor_mainThread" );
            // just run on main thread
            body.forLoop( iBegin, iEnd );
        }
    }
    virtual btScalar parallelSum( int iBegin, int iEnd, int grainSize, const btIParallelSumBody& body ) BT_OVERRIDE
    {
        BT_PROFILE( "parallelSum_ThreadSupport" );
        btAssert( iEnd >= iBegin );
        btAssert( grainSize >= 1 );
        int iterationCount = iEnd - iBegin;
        if ( iterationCount > grainSize && m_numWorkerThreads > 0 && m_antiNestingLock.tryLock() )
        {
            typedef ParallelSumJob JobType;
            int jobCount = ( iterationCount + grainSize - 1 ) / grainSize;
            m_numJobs = jobCount;
            btAssert( jobCount >= 2 );  // need more than one job for multithreading
            int jobSize = sizeof( JobType );
            for (int i = 0; i < m_numActiveJobQueues; ++i)
            {
                m_jobQueues[i].clearQueue( jobCount, jobSize );
            }

            // initialize summation
            for ( int iThread = 0; iThread < m_numThreads; ++iThread )
            {
                m_threadLocalStorage[iThread].m_sumResult = btScalar(0);
            }

            // prepare worker threads for incoming work
            prepareWorkerThreads();
            // submit all of the jobs
            int iJob = 0;
            int iThread = kFirstWorkerThreadId;  // first worker thread
            for ( int i = iBegin; i < iEnd; i += grainSize )
            {
                btAssert( iJob < jobCount );
                int iE = btMin( i + grainSize, iEnd );
                JobQueue* jq = m_perThreadJobQueues[ iThread ];
                btAssert(jq);
                btAssert((jq - &m_jobQueues[0]) < m_numActiveJobQueues);
                void* jobMem = jq->allocJobMem(jobSize);
                JobType* job = new ( jobMem ) ParallelSumJob( i, iE, body, &m_threadLocalStorage[0] );  // placement new
                jq->submitJob( job );
                iJob++;
                iThread++;
                if ( iThread >= m_numThreads )
                {
                    iThread = kFirstWorkerThreadId;  // first worker thread
                }
            }
            wakeWorkers( jobCount - 1 );

            // put the main thread to work on emptying the job queue and then wait for all workers to finish
            waitJobs();

            // add up all the thread sums
            btScalar sum = btScalar(0);
            for ( int iThread = 0; iThread < m_numThreads; ++iThread )
            {
                sum += m_threadLocalStorage[ iThread ].m_sumResult;
            }
            m_antiNestingLock.unlock();
            return sum;
        }
        else
        {
            BT_PROFILE( "parallelSum_mainThread" );
            // just run on main thread
            return body.sumLoop( iBegin, iEnd );
        }
    }
};



btITaskScheduler* btCreateDefaultTaskScheduler()
{
    btTaskSchedulerDefault* ts = new btTaskSchedulerDefault();
    ts->init();
    return ts;
}

#else // #if BT_THREADSAFE

btITaskScheduler* btCreateDefaultTaskScheduler()
{
    return NULL;
}

#endif // #else // #if BT_THREADSAFE
