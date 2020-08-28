/*!
**
** Copyright (c) 2009 by John W. Ratcliff mailto:jratcliffscarab@gmail.com
**
** Portions of this source has been released with the PhysXViewer application, as well as
** Rocket, CreateDynamics, ODF, and as a number of sample code snippets.
**
** If you find this code useful or you are feeling particularily generous I would
** ask that you please go to http://www.amillionpixels.us and make a donation
** to Troy DeMolay.
**
** DeMolay is a youth group for young men between the ages of 12 and 21.
** It teaches strong moral principles, as well as leadership skills and
** public speaking.  The donations page uses the 'pay for pixels' paradigm
** where, in this case, a pixel is only a single penny.  Donations can be
** made for as small as $4 or as high as a $100 block.  Each person who donates
** will get a link to their own site as well as acknowledgement on the
** donations blog located here http://www.amillionpixels.blogspot.com/
**
** If you wish to contact me you can use the following methods:
**
** Skype ID: jratcliff63367
** Yahoo: jratcliff63367
** AOL: jratcliff1961
** email: jratcliffscarab@gmail.com
**
**
** The MIT license:
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is furnished
** to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in all
** copies or substantial portions of the Software.

** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
** WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#pragma once
#ifndef VHACD_MUTEX_H
#define VHACD_MUTEX_H

#if defined(WIN32)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x400
#endif
#include <windows.h>
#pragma comment(lib, "winmm.lib")
#endif

#if defined(__linux__)
//#include <sys/time.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#define __stdcall
#endif

#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif

// -- GODOT start --
#if defined(__APPLE__) || !defined(__GLIBC__)
// -- GODOT end --
#define PTHREAD_MUTEX_RECURSIVE_NP PTHREAD_MUTEX_RECURSIVE
#endif

#define VHACD_DEBUG

//#define VHACD_NDEBUG
#ifdef VHACD_NDEBUG
#define VHACD_VERIFY(x) (x)
#else
#define VHACD_VERIFY(x) assert((x))
#endif

namespace VHACD {
class Mutex {
public:
    Mutex(void)
    {
#if defined(WIN32) || defined(_XBOX)
        InitializeCriticalSection(&m_mutex);
#elif defined(__APPLE__) || defined(__linux__)
        pthread_mutexattr_t mutexAttr; // Mutex Attribute
        VHACD_VERIFY(pthread_mutexattr_init(&mutexAttr) == 0);
        VHACD_VERIFY(pthread_mutexattr_settype(&mutexAttr, PTHREAD_MUTEX_RECURSIVE_NP) == 0);
        VHACD_VERIFY(pthread_mutex_init(&m_mutex, &mutexAttr) == 0);
        VHACD_VERIFY(pthread_mutexattr_destroy(&mutexAttr) == 0);
#endif
    }
    ~Mutex(void)
    {
#if defined(WIN32) || defined(_XBOX)
        DeleteCriticalSection(&m_mutex);
#elif defined(__APPLE__) || defined(__linux__)
        VHACD_VERIFY(pthread_mutex_destroy(&m_mutex) == 0);
#endif
    }
    void Lock(void)
    {
#if defined(WIN32) || defined(_XBOX)
        EnterCriticalSection(&m_mutex);
#elif defined(__APPLE__) || defined(__linux__)
        VHACD_VERIFY(pthread_mutex_lock(&m_mutex) == 0);
#endif
    }
    bool TryLock(void)
    {
#if defined(WIN32) || defined(_XBOX)
        bool bRet = false;
        //assert(("TryEnterCriticalSection seems to not work on XP???", 0));
        bRet = TryEnterCriticalSection(&m_mutex) ? true : false;
        return bRet;
#elif defined(__APPLE__) || defined(__linux__)
        int32_t result = pthread_mutex_trylock(&m_mutex);
        return (result == 0);
#endif
    }

    void Unlock(void)
    {
#if defined(WIN32) || defined(_XBOX)
        LeaveCriticalSection(&m_mutex);
#elif defined(__APPLE__) || defined(__linux__)
        VHACD_VERIFY(pthread_mutex_unlock(&m_mutex) == 0);
#endif
    }

private:
#if defined(WIN32) || defined(_XBOX)
    CRITICAL_SECTION m_mutex;
#elif defined(__APPLE__) || defined(__linux__)
    pthread_mutex_t m_mutex;
#endif
};
}
#endif // VHACD_MUTEX_H
