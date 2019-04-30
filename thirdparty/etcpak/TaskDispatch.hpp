#ifndef __DARKRL__TASKDISPATCH_HPP__
#define __DARKRL__TASKDISPATCH_HPP__

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class TaskDispatch
{
public:
    TaskDispatch( size_t workers );
    ~TaskDispatch();

    static void Queue( const std::function<void(void)>& f );
    static void Queue( std::function<void(void)>&& f );

    static void Sync();

private:
    void Worker();

    std::vector<std::function<void(void)>> m_queue;
    std::mutex m_queueLock;
    std::condition_variable m_cvWork, m_cvJobs;
    std::atomic<bool> m_exit;
    size_t m_jobs;

    std::vector<std::thread> m_workers;
};

#endif
