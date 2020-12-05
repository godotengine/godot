
/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2018 Erwin Coumans  http://bulletphysics.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#if BT_THREADSAFE && !defined(_WIN32)

#include "LinearMath/btScalar.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btThreads.h"
#include "LinearMath/btMinMax.h"
#include "btThreadSupportInterface.h"

#include <stdio.h>
#include <errno.h>
#include <unistd.h>

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600  //for definition of pthread_barrier_t, see http://pages.cs.wisc.edu/~travitch/pthreads_primer.html
#endif                     //_XOPEN_SOURCE
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>  //for sysconf

///
/// getNumHardwareThreads()
///
///
/// https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
///
#if __cplusplus >= 201103L

#include <thread>

int btGetNumHardwareThreads()
{
	return btMax(1u, btMin(BT_MAX_THREAD_COUNT, std::thread::hardware_concurrency()));
}

#else

int btGetNumHardwareThreads()
{
	return btMax(1, btMin<int>(BT_MAX_THREAD_COUNT, sysconf(_SC_NPROCESSORS_ONLN)));
}

#endif

// btThreadSupportPosix helps to initialize/shutdown libspe2, start/stop SPU tasks and communication
class btThreadSupportPosix : public btThreadSupportInterface
{
public:
	struct btThreadStatus
	{
		int m_taskId;
		int m_commandId;
		int m_status;

		ThreadFunc m_userThreadFunc;
		void* m_userPtr;  //for taskDesc etc

		pthread_t thread;
		//each tread will wait until this signal to start its work
		sem_t* startSemaphore;
		btCriticalSection* m_cs;
		// this is a copy of m_mainSemaphore,
		//each tread will signal once it is finished with its work
		sem_t* m_mainSemaphore;
		unsigned long threadUsed;
	};

private:
	typedef unsigned long long UINT64;

	btAlignedObjectArray<btThreadStatus> m_activeThreadStatus;
	// m_mainSemaphoresemaphore will signal, if and how many threads are finished with their work
	sem_t* m_mainSemaphore;
	int m_numThreads;
	UINT64 m_startedThreadsMask;
	void startThreads(const ConstructionInfo& threadInfo);
	void stopThreads();
	int waitForResponse();
	btCriticalSection* m_cs;
public:
	btThreadSupportPosix(const ConstructionInfo& threadConstructionInfo);
	virtual ~btThreadSupportPosix();

	virtual int getNumWorkerThreads() const BT_OVERRIDE { return m_numThreads; }
	// TODO: return the number of logical processors sharing the first L3 cache
	virtual int getCacheFriendlyNumThreads() const BT_OVERRIDE { return m_numThreads + 1; }
	// TODO: detect if CPU has hyperthreading enabled
	virtual int getLogicalToPhysicalCoreRatio() const BT_OVERRIDE { return 1; }

	virtual void runTask(int threadIndex, void* userData) BT_OVERRIDE;
	virtual void waitForAllTasks() BT_OVERRIDE;

	virtual btCriticalSection* createCriticalSection() BT_OVERRIDE;
	virtual void deleteCriticalSection(btCriticalSection* criticalSection) BT_OVERRIDE;
};

#define checkPThreadFunction(returnValue)                                                                 \
	if (0 != returnValue)                                                                                 \
	{                                                                                                     \
		printf("PThread problem at line %i in file %s: %i %d\n", __LINE__, __FILE__, returnValue, errno); \
	}

// The number of threads should be equal to the number of available cores
// Todo: each worker should be linked to a single core, using SetThreadIdealProcessor.

btThreadSupportPosix::btThreadSupportPosix(const ConstructionInfo& threadConstructionInfo)
{
	m_cs = createCriticalSection();
	startThreads(threadConstructionInfo);
}

// cleanup/shutdown Libspe2
btThreadSupportPosix::~btThreadSupportPosix()
{
	stopThreads();
	deleteCriticalSection(m_cs);
	m_cs=0;
}

#if (defined(__APPLE__))
#define NAMED_SEMAPHORES
#endif

static sem_t* createSem(const char* baseName)
{
	static int semCount = 0;
#ifdef NAMED_SEMAPHORES
	/// Named semaphore begin
	char name[32];
	snprintf(name, 32, "/%8.s-%4.d-%4.4d", baseName, getpid(), semCount++);
	sem_t* tempSem = sem_open(name, O_CREAT, 0600, 0);

	if (tempSem != reinterpret_cast<sem_t*>(SEM_FAILED))
	{
		//        printf("Created \"%s\" Semaphore %p\n", name, tempSem);
	}
	else
	{
		//printf("Error creating Semaphore %d\n", errno);
		exit(-1);
	}
	/// Named semaphore end
#else
	sem_t* tempSem = new sem_t;
	checkPThreadFunction(sem_init(tempSem, 0, 0));
#endif
	return tempSem;
}

static void destroySem(sem_t* semaphore)
{
#ifdef NAMED_SEMAPHORES
	checkPThreadFunction(sem_close(semaphore));
#else
	checkPThreadFunction(sem_destroy(semaphore));
	delete semaphore;
#endif
}

static void* threadFunction(void* argument)
{
	btThreadSupportPosix::btThreadStatus* status = (btThreadSupportPosix::btThreadStatus*)argument;

	while (1)
	{
		checkPThreadFunction(sem_wait(status->startSemaphore));
		void* userPtr = status->m_userPtr;

		if (userPtr)
		{
			btAssert(status->m_status);
			status->m_userThreadFunc(userPtr);
			status->m_cs->lock();
			status->m_status = 2;
			status->m_cs->unlock();
			checkPThreadFunction(sem_post(status->m_mainSemaphore));
			status->threadUsed++;
		}
		else
		{
			//exit Thread
			status->m_cs->lock();
			status->m_status = 3;
			status->m_cs->unlock();
			checkPThreadFunction(sem_post(status->m_mainSemaphore));
			break;
		}
	}

	return 0;
}

///send messages to SPUs
void btThreadSupportPosix::runTask(int threadIndex, void* userData)
{
	///we should spawn an SPU task here, and in 'waitForResponse' it should wait for response of the (one of) the first tasks that finished
	btThreadStatus& threadStatus = m_activeThreadStatus[threadIndex];
	btAssert(threadIndex >= 0);
	btAssert(threadIndex < m_activeThreadStatus.size());
	threadStatus.m_cs = m_cs;
	threadStatus.m_commandId = 1;
	threadStatus.m_status = 1;
	threadStatus.m_userPtr = userData;
	m_startedThreadsMask |= UINT64(1) << threadIndex;

	// fire event to start new task
	checkPThreadFunction(sem_post(threadStatus.startSemaphore));
}

///check for messages from SPUs
int btThreadSupportPosix::waitForResponse()
{
	///We should wait for (one of) the first tasks to finish (or other SPU messages), and report its response
	///A possible response can be 'yes, SPU handled it', or 'no, please do a PPU fallback'

	btAssert(m_activeThreadStatus.size());

	// wait for any of the threads to finish
	checkPThreadFunction(sem_wait(m_mainSemaphore));
	// get at least one thread which has finished
	size_t last = -1;

	for (size_t t = 0; t < size_t(m_activeThreadStatus.size()); ++t)
	{
		m_cs->lock();
		bool hasFinished = (2 == m_activeThreadStatus[t].m_status);
		m_cs->unlock(); 
		if (hasFinished)
		{
			last = t;
			break;
		}
	}

	btThreadStatus& threadStatus = m_activeThreadStatus[last];

	btAssert(threadStatus.m_status > 1);
	threadStatus.m_status = 0;

	// need to find an active spu
	btAssert(last >= 0);
	m_startedThreadsMask &= ~(UINT64(1) << last);

	return last;
}

void btThreadSupportPosix::waitForAllTasks()
{
	while (m_startedThreadsMask)
	{
		waitForResponse();
	}
}

void btThreadSupportPosix::startThreads(const ConstructionInfo& threadConstructionInfo)
{
	m_numThreads = btGetNumHardwareThreads() - 1;  // main thread exists already
	m_activeThreadStatus.resize(m_numThreads);
	m_startedThreadsMask = 0;

	m_mainSemaphore = createSem("main");
	//checkPThreadFunction(sem_wait(mainSemaphore));

	for (int i = 0; i < m_numThreads; i++)
	{
		btThreadStatus& threadStatus = m_activeThreadStatus[i];
		threadStatus.startSemaphore = createSem("threadLocal");
		threadStatus.m_userPtr = 0;
		threadStatus.m_cs = m_cs;
		threadStatus.m_taskId = i;
		threadStatus.m_commandId = 0;
		threadStatus.m_status = 0;
		threadStatus.m_mainSemaphore = m_mainSemaphore;
		threadStatus.m_userThreadFunc = threadConstructionInfo.m_userThreadFunc;
		threadStatus.threadUsed = 0;
		checkPThreadFunction(pthread_create(&threadStatus.thread, NULL, &threadFunction, (void*)&threadStatus));

	}
}

///tell the task scheduler we are done with the SPU tasks
void btThreadSupportPosix::stopThreads()
{
	for (size_t t = 0; t < size_t(m_activeThreadStatus.size()); ++t)
	{
		btThreadStatus& threadStatus = m_activeThreadStatus[t];

		threadStatus.m_userPtr = 0;
		checkPThreadFunction(sem_post(threadStatus.startSemaphore));
		checkPThreadFunction(sem_wait(m_mainSemaphore));

		checkPThreadFunction(pthread_join(threadStatus.thread, 0));
		destroySem(threadStatus.startSemaphore);
	}
	destroySem(m_mainSemaphore);
	m_activeThreadStatus.clear();
}

class btCriticalSectionPosix : public btCriticalSection
{
	pthread_mutex_t m_mutex;

public:
	btCriticalSectionPosix()
	{
		pthread_mutex_init(&m_mutex, NULL);
	}
	virtual ~btCriticalSectionPosix()
	{
		pthread_mutex_destroy(&m_mutex);
	}

	virtual void lock()
	{
		pthread_mutex_lock(&m_mutex);
	}
	virtual void unlock()
	{
		pthread_mutex_unlock(&m_mutex);
	}
};

btCriticalSection* btThreadSupportPosix::createCriticalSection()
{
	return new btCriticalSectionPosix();
}

void btThreadSupportPosix::deleteCriticalSection(btCriticalSection* cs)
{
	delete cs;
}

btThreadSupportInterface* btThreadSupportInterface::create(const ConstructionInfo& info)
{
	return new btThreadSupportPosix(info);
}

#endif  // BT_THREADSAFE && !defined( _WIN32 )
