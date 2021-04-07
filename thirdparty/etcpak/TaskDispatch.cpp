#include <assert.h>
#include <stdio.h>

#include "Debug.hpp"
#include "System.hpp"
#include "TaskDispatch.hpp"

static TaskDispatch* s_instance = nullptr;

TaskDispatch::TaskDispatch( size_t workers )
    : m_exit( false )
    , m_jobs( 0 )
{
    assert( !s_instance );
    s_instance = this;

    assert( workers >= 1 );
    workers--;

    m_workers.reserve( workers );
    for( size_t i=0; i<workers; i++ )
    {
        char tmp[16];
        sprintf( tmp, "Worker %zu", i );
#ifdef __APPLE__
        auto worker = std::thread( [this, tmp]{
            pthread_setname_np( tmp );
            Worker();
        } );
#else
        auto worker = std::thread( [this]{ Worker(); } );
#endif
        System::SetThreadName( worker, tmp );
        m_workers.emplace_back( std::move( worker ) );
    }

    DBGPRINT( "Task dispatcher with " << m_workers.size() + 1 << " workers" );
}

TaskDispatch::~TaskDispatch()
{
    m_exit = true;
    m_queueLock.lock();
    m_cvWork.notify_all();
    m_queueLock.unlock();

    for( auto& worker : m_workers )
    {
        worker.join();
    }

    assert( s_instance );
    s_instance = nullptr;
}

void TaskDispatch::Queue( const std::function<void(void)>& f )
{
    std::unique_lock<std::mutex> lock( s_instance->m_queueLock );
    s_instance->m_queue.emplace_back( f );
    const auto size = s_instance->m_queue.size();
    lock.unlock();
    if( size > 1 )
    {
        s_instance->m_cvWork.notify_one();
    }
}

void TaskDispatch::Queue( std::function<void(void)>&& f )
{
    std::unique_lock<std::mutex> lock( s_instance->m_queueLock );
    s_instance->m_queue.emplace_back( std::move( f ) );
    const auto size = s_instance->m_queue.size();
    lock.unlock();
    if( size > 1 )
    {
        s_instance->m_cvWork.notify_one();
    }
}

void TaskDispatch::Sync()
{
    std::unique_lock<std::mutex> lock( s_instance->m_queueLock );
    while( !s_instance->m_queue.empty() )
    {
        auto f = s_instance->m_queue.back();
        s_instance->m_queue.pop_back();
        lock.unlock();
        f();
        lock.lock();
    }
    s_instance->m_cvJobs.wait( lock, []{ return s_instance->m_jobs == 0; } );
}

void TaskDispatch::Worker()
{
    for(;;)
    {
        std::unique_lock<std::mutex> lock( m_queueLock );
        m_cvWork.wait( lock, [this]{ return !m_queue.empty() || m_exit; } );
        if( m_exit ) return;
        auto f = m_queue.back();
        m_queue.pop_back();
        m_jobs++;
        lock.unlock();
        f();
        lock.lock();
        m_jobs--;
        bool notify = m_jobs == 0 && m_queue.empty();
        lock.unlock();
        if( notify )
        {
            m_cvJobs.notify_all();
        }
    }
}
