/*
 * Copyright (c) 2024 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TVG_LOCK_H_
#define _TVG_LOCK_H_

#include <mutex>
#include "tvgTaskScheduler.h"

namespace tvg
{
#ifdef THORVG_THREAD_SUPPORT
    struct Key
    {
        std::mutex mtx;

        void lock()
        {
            mtx.lock();
        }

        void unlock()
        {
            mtx.unlock();
        }
    };

    struct StrictKey : Key
    {
    };

    struct ScopedLock
    {
        Key* key = nullptr;

        ScopedLock(Key& k)
        {
            if (TaskScheduler::threads() > 0) {
                k.mtx.lock();
                key = &k;
            }
        }

        ScopedLock(StrictKey& k)
        {
            k.mtx.lock();
            key = &k;
        }

        ~ScopedLock()
        {
            if (key) key->mtx.unlock();
        }
    };
#else //THORVG_THREAD_SUPPORT
    struct Key {};

    struct StrictKey : Key
    {
#ifdef __STDCPP_THREADS__
        std::mutex mtx;
#endif

        void lock()
        {
#ifdef __STDCPP_THREADS__
            mtx.lock();
#endif
        }

        void unlock()
        {
#ifdef __STDCPP_THREADS__
            mtx.unlock();
#endif
        }
    };

    struct ScopedLock
    {
        StrictKey* key = nullptr;

        ScopedLock(Key& k) {}

        ScopedLock(StrictKey& k)
        {
            k.lock();
            key = &k;
        }

        ~ScopedLock()
        {
            if (key) key->unlock();
        }
    };
#endif //THORVG_THREAD_SUPPORT
}

#endif //_TVG_LOCK_H_
