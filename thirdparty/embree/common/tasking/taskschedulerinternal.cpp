// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "taskschedulerinternal.h"
#include "../math/math.h"
#include "../sys/sysinfo.h"
#include <algorithm>

namespace embree
{
  RTC_NAMESPACE_BEGIN
  
  static MutexSys g_mutex;
  size_t TaskScheduler::g_numThreads = 0;
  __thread TaskScheduler* TaskScheduler::g_instance = nullptr;
  std::vector<Ref<TaskScheduler>> g_instance_vector;
  __thread TaskScheduler::Thread* TaskScheduler::thread_local_thread = nullptr;
  TaskScheduler::ThreadPool* TaskScheduler::threadPool = nullptr;

  template<typename Predicate, typename Body>
  __forceinline void TaskScheduler::steal_loop(Thread& thread, const Predicate& pred, const Body& body)
  {
    while (true)
    {
      /*! some rounds that yield */
      for (size_t i=0; i<32; i++)
      {
        /*! some spinning rounds */
        const size_t threadCount = thread.threadCount();
        for (size_t j=0; j<1024; j+=threadCount)
        {
          if (!pred()) return;
          if (thread.scheduler->steal_from_other_threads(thread)) {
            i=j=0;
            body();
          }
        }
        yield();
      }
    }
  }

  /*! run this task */
  void TaskScheduler::Task::run_internal (Thread& thread) // FIXME: avoid as many dll_exports as possible
  {
    /* try to run if not already stolen */
    if (try_switch_state(INITIALIZED,DONE))
    {
      Task* prevTask = thread.task;
      thread.task = this;
      try {
        if (thread.scheduler->cancellingException == nullptr)
          closure->execute();
      } catch (...) {
        if (thread.scheduler->cancellingException == nullptr)
          thread.scheduler->cancellingException = std::current_exception();
      }
      thread.task = prevTask;
      add_dependencies(-1);
    }

    /* steal until all dependencies have completed */
    steal_loop(thread,
               [&] () { return dependencies>0; },
               [&] () { while (thread.tasks.execute_local_internal(thread,this)); });

    /* now signal our parent task that we are finished */
    if (parent)
      parent->add_dependencies(-1);
  }

    /*! run this task */
  dll_export void TaskScheduler::Task::run (Thread& thread) {
    run_internal(thread);
  }

  bool TaskScheduler::TaskQueue::execute_local_internal(Thread& thread, Task* parent)
  {
    /* stop if we run out of local tasks or reach the waiting task */
    if (right == 0 || &tasks[right-1] == parent)
      return false;

    /* execute task */
    size_t oldRight = right;
    tasks[right-1].run_internal(thread);
    if (right != oldRight) {
      THROW_RUNTIME_ERROR("you have to wait for spawned subtasks");
    }

    /* pop task and closure from stack */
    right--;
    if (tasks[right].stackPtr != size_t(-1))
      stackPtr = tasks[right].stackPtr;

    /* also move left pointer */
    if (left >= right) left.store(right.load());

    return right != 0;
  }

  dll_export bool TaskScheduler::TaskQueue::execute_local(Thread& thread, Task* parent) {
    return execute_local_internal(thread,parent);
  }

  bool TaskScheduler::TaskQueue::steal(Thread& thread)
  {
    size_t l = left;
    size_t r = right;
    if (l < r)
    {
      l = left++;
       if (l >= r)
         return false;
    }
    else
      return false;

    if (!tasks[l].try_steal(thread.tasks.tasks[thread.tasks.right]))
      return false;

    thread.tasks.right++;
    return true;
  }

  /* we steal from the left */
  size_t TaskScheduler::TaskQueue::getTaskSizeAtLeft()
  {
    if (left >= right) return 0;
    return tasks[left].N;
  }

  void threadPoolFunction(std::pair<TaskScheduler::ThreadPool*,size_t>* pair)
  {
    TaskScheduler::ThreadPool* pool = pair->first;
    size_t threadIndex = pair->second;
    delete pair;
    pool->thread_loop(threadIndex);
  }

  TaskScheduler::ThreadPool::ThreadPool(bool set_affinity)
    : numThreads(0), numThreadsRunning(0), set_affinity(set_affinity), running(false) {}

  dll_export void TaskScheduler::ThreadPool::startThreads()
  {
    if (running) return;
    setNumThreads(numThreads,true);
  }

  void TaskScheduler::ThreadPool::setNumThreads(size_t newNumThreads, bool startThreads)
  {
    Lock<MutexSys> lock(g_mutex);
    assert(newNumThreads);
    newNumThreads = min(newNumThreads, (size_t) getNumberOfLogicalThreads());

    numThreads = newNumThreads;
    if (!startThreads && !running) return;
    running = true;
    size_t numThreadsActive = numThreadsRunning;

    mutex.lock();
    numThreadsRunning = newNumThreads;
    mutex.unlock();
    condition.notify_all();

    /* start new threads */
    for (size_t t=numThreadsActive; t<numThreads; t++)
    {
      if (t == 0) continue;
      auto pair = new std::pair<TaskScheduler::ThreadPool*,size_t>(this,t);
      threads.push_back(createThread((thread_func)threadPoolFunction,pair,4*1024*1024,set_affinity ? t : -1));
    }

    /* stop some threads if we reduce the number of threads */
    for (ssize_t t=numThreadsActive-1; t>=ssize_t(numThreadsRunning); t--) {
      if (t == 0) continue;
      embree::join(threads.back());
      threads.pop_back();
    }
  }

  TaskScheduler::ThreadPool::~ThreadPool()
  {
    /* leave all taskschedulers */
    mutex.lock();
    numThreadsRunning = 0;
    mutex.unlock();
    condition.notify_all();

    /* wait for threads to terminate */
    for (size_t i=0; i<threads.size(); i++)
      embree::join(threads[i]);
  }

  dll_export void TaskScheduler::ThreadPool::add(const Ref<TaskScheduler>& scheduler)
  {
    mutex.lock();
    schedulers.push_back(scheduler);
    mutex.unlock();
    condition.notify_all();
  }

  dll_export void TaskScheduler::ThreadPool::remove(const Ref<TaskScheduler>& scheduler)
  {
    Lock<MutexSys> lock(mutex);
    for (std::list<Ref<TaskScheduler> >::iterator it = schedulers.begin(); it != schedulers.end(); it++) {
      if (scheduler == *it) {
        schedulers.erase(it);
        return;
      }
    }
  }

  void TaskScheduler::ThreadPool::thread_loop(size_t globalThreadIndex)
  {
    while (globalThreadIndex < numThreadsRunning)
    {
      Ref<TaskScheduler> scheduler = NULL;
      ssize_t threadIndex = -1;
      {
        Lock<MutexSys> lock(mutex);
        condition.wait(mutex, [&] () { return globalThreadIndex >= numThreadsRunning || !schedulers.empty(); });
        if (globalThreadIndex >= numThreadsRunning) break;
        scheduler = schedulers.front();
        threadIndex = scheduler->allocThreadIndex();
      }
      scheduler->thread_loop(threadIndex);
    }
  }

  TaskScheduler::TaskScheduler()
    : threadCounter(0), anyTasksRunning(0), hasRootTask(false)
  {
    threadLocal.resize(2*getNumberOfLogicalThreads()); // FIXME: this has to be 2x as in the compatibility join mode with rtcCommitScene the worker threads also join. When disallowing rtcCommitScene to join a build we can remove the 2x.
    for (size_t i=0; i<threadLocal.size(); i++)
      threadLocal[i].store(nullptr);
  }

  TaskScheduler::~TaskScheduler()
  {
    assert(threadCounter == 0);
  }

  dll_export size_t TaskScheduler::threadID()
  {
    Thread* thread = TaskScheduler::thread();
    if (thread) return thread->threadIndex;
    else        return 0;
  }

  dll_export size_t TaskScheduler::threadIndex()
  {
    Thread* thread = TaskScheduler::thread();
    if (thread) return thread->threadIndex;
    else        return 0;
  }

  dll_export size_t TaskScheduler::threadCount() {
    return threadPool->size();
  }

  dll_export TaskScheduler* TaskScheduler::instance()
  {
    if (g_instance == NULL) {
      Lock<MutexSys> lock(g_mutex);
      g_instance = new TaskScheduler;
      g_instance_vector.push_back(g_instance);
    }
    return g_instance;
  }

  void TaskScheduler::create(size_t numThreads, bool set_affinity, bool start_threads)
  {
    if (!threadPool) threadPool = new TaskScheduler::ThreadPool(set_affinity);
    threadPool->setNumThreads(numThreads,start_threads);
  }

  void TaskScheduler::destroy() {
    delete threadPool; threadPool = nullptr;
  }

  dll_export ssize_t TaskScheduler::allocThreadIndex()
  {
    size_t threadIndex = threadCounter++;
    assert(threadIndex < threadLocal.size());
    return threadIndex;
  }

  void TaskScheduler::join()
  {
    mutex.lock();
    size_t threadIndex = allocThreadIndex();
    condition.wait(mutex, [&] () { return hasRootTask.load(); });
    mutex.unlock();
    std::exception_ptr except = thread_loop(threadIndex);
    if (except != nullptr) std::rethrow_exception(except);
  }

  void TaskScheduler::reset() {
    hasRootTask = false;
  }

  void TaskScheduler::wait_for_threads(size_t threadCount)
  {
    while (threadCounter < threadCount-1)
      pause_cpu();
  }

  dll_export TaskScheduler::Thread* TaskScheduler::thread() {
    return thread_local_thread;
  }

  dll_export TaskScheduler::Thread* TaskScheduler::swapThread(Thread* thread)
  {
    Thread* old = thread_local_thread;
    thread_local_thread = thread;
    return old;
  }

  dll_export bool TaskScheduler::wait()
  {
    Thread* thread = TaskScheduler::thread();
    if (thread == nullptr) return true;
    while (thread->tasks.execute_local_internal(*thread,thread->task)) {};
    return thread->scheduler->cancellingException == nullptr;
  }

  std::exception_ptr TaskScheduler::thread_loop(size_t threadIndex)
  {
    /* allocate thread structure */
    std::unique_ptr<Thread> mthread(new Thread(threadIndex,this)); // too large for stack allocation
    Thread& thread = *mthread;
    threadLocal[threadIndex].store(&thread);
    Thread* oldThread = swapThread(&thread);

    /* main thread loop */
    while (anyTasksRunning)
    {
      steal_loop(thread,
                 [&] () { return anyTasksRunning > 0; },
                 [&] () {
                   anyTasksRunning++;
                   while (thread.tasks.execute_local_internal(thread,nullptr));
                   anyTasksRunning--;
                 });
    }
    threadLocal[threadIndex].store(nullptr);
    swapThread(oldThread);

    /* remember exception to throw */
    std::exception_ptr except = nullptr;
    if (cancellingException != nullptr) except = cancellingException;

    /* wait for all threads to terminate */
    threadCounter--;
#if defined(__WIN32__)
	size_t loopIndex = 1;
#endif
#define LOOP_YIELD_THRESHOLD (4096)
	while (threadCounter > 0) {
#if defined(__WIN32__)
          if ((loopIndex % LOOP_YIELD_THRESHOLD) == 0)
            yield();
          else
            _mm_pause();
	  loopIndex++;
#else
          yield();
#endif
	}
    return except;
  }

  bool TaskScheduler::steal_from_other_threads(Thread& thread)
  {
    const size_t threadIndex = thread.threadIndex;
    const size_t threadCount = this->threadCounter;

    for (size_t i=1; i<threadCount; i++)
    {
      pause_cpu(32);
      size_t otherThreadIndex = threadIndex+i;
      if (otherThreadIndex >= threadCount) otherThreadIndex -= threadCount;

      Thread* othread = threadLocal[otherThreadIndex].load();
      if (!othread)
        continue;

      if (othread->tasks.steal(thread))
        return true;
    }

    return false;
  }

  dll_export void TaskScheduler::startThreads() {
    threadPool->startThreads();
  }

  dll_export void TaskScheduler::addScheduler(const Ref<TaskScheduler>& scheduler) {
    threadPool->add(scheduler);
  }

  dll_export void TaskScheduler::removeScheduler(const Ref<TaskScheduler>& scheduler) {
    threadPool->remove(scheduler);
  }

  RTC_NAMESPACE_END
}
