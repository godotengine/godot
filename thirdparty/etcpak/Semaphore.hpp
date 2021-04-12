#ifndef __DARKRL__SEMAPHORE_HPP__
#define __DARKRL__SEMAPHORE_HPP__

#include <condition_variable>
#include <mutex>

class Semaphore
{
public:
    Semaphore( int count ) : m_count( count ) {}

    void lock()
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_cv.wait( lock, [this](){ return m_count != 0; } );
        m_count--;
    }

    void unlock()
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_count++;
        m_cv.notify_one();
    }

    bool try_lock()
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        if( m_count == 0 )
        {
            return false;
        }
        else
        {
            m_count--;
            return true;
        }
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    unsigned int m_count;
};

#endif
