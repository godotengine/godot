/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_TIMER_H
#define VHACD_TIMER_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#endif
#include <windows.h>
#elif __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

namespace VHACD {
#ifdef _WIN32
class Timer {
public:
    Timer(void)
    {
        m_start.QuadPart = 0;
        m_stop.QuadPart = 0;
        QueryPerformanceFrequency(&m_freq);
    };
    ~Timer(void){};
    void Tic()
    {
        QueryPerformanceCounter(&m_start);
    }
    void Toc()
    {
        QueryPerformanceCounter(&m_stop);
    }
    double GetElapsedTime() // in ms
    {
        LARGE_INTEGER delta;
        delta.QuadPart = m_stop.QuadPart - m_start.QuadPart;
        return (1000.0 * delta.QuadPart) / (double)m_freq.QuadPart;
    }

private:
    LARGE_INTEGER m_start;
    LARGE_INTEGER m_stop;
    LARGE_INTEGER m_freq;
};

#elif __MACH__
class Timer {
public:
    Timer(void)
    {
        memset(this, 0, sizeof(Timer));
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &m_cclock);
    };
    ~Timer(void)
    {
        mach_port_deallocate(mach_task_self(), m_cclock);
    };
    void Tic()
    {
        clock_get_time(m_cclock, &m_start);
    }
    void Toc()
    {
        clock_get_time(m_cclock, &m_stop);
    }
    double GetElapsedTime() // in ms
    {
        return 1000.0 * (m_stop.tv_sec - m_start.tv_sec + (1.0E-9) * (m_stop.tv_nsec - m_start.tv_nsec));
    }

private:
    clock_serv_t m_cclock;
    mach_timespec_t m_start;
    mach_timespec_t m_stop;
};
#else
class Timer {
public:
    Timer(void)
    {
        memset(this, 0, sizeof(Timer));
    };
    ~Timer(void){};
    void Tic()
    {
        clock_gettime(CLOCK_REALTIME, &m_start);
    }
    void Toc()
    {
        clock_gettime(CLOCK_REALTIME, &m_stop);
    }
    double GetElapsedTime() // in ms
    {
        return 1000.0 * (m_stop.tv_sec - m_start.tv_sec + (1.0E-9) * (m_stop.tv_nsec - m_start.tv_nsec));
    }

private:
    struct timespec m_start;
    struct timespec m_stop;
};
#endif
}
#endif // VHACD_TIMER_H
