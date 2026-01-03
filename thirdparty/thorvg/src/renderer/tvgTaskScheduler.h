/*
 * Copyright (c) 2020 - 2024 the ThorVG project. All rights reserved.

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

#ifndef _TVG_TASK_SCHEDULER_H_
#define _TVG_TASK_SCHEDULER_H_

#include <mutex>
#include <condition_variable>

#include "tvgCommon.h"
#include "tvgInlist.h"

namespace tvg {

#ifdef THORVG_THREAD_SUPPORT

struct Task
{
private:
    mutex                   mtx;
    condition_variable      cv;
    bool                    ready = true;
    bool                    pending = false;

public:
    INLIST_ITEM(Task);

    virtual ~Task() = default;

    void done()
    {
        if (!pending) return;

        unique_lock<mutex> lock(mtx);
        while (!ready) cv.wait(lock);
        pending = false;
    }

protected:
    virtual void run(unsigned tid) = 0;

private:
    void operator()(unsigned tid)
    {
        run(tid);

        lock_guard<mutex> lock(mtx);
        ready = true;
        cv.notify_one();
    }

    void prepare()
    {
        ready = false;
        pending = true;
    }

    friend struct TaskSchedulerImpl;
};

#else  //THORVG_THREAD_SUPPORT

struct Task
{
public:
    INLIST_ITEM(Task);

    virtual ~Task() = default;
    void done() {}

protected:
    virtual void run(unsigned tid) = 0;

private:
    friend struct TaskSchedulerImpl;
};

#endif  //THORVG_THREAD_SUPPORT


struct TaskScheduler
{
    static uint32_t threads();
    static void init(uint32_t threads);
    static void term();
    static void request(Task* task);
};

}  //namespace

#endif //_TVG_TASK_SCHEDULER_H_
 
