/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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

#include <deque>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>
#include "tvgTaskScheduler.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

namespace tvg {

struct TaskQueue {
    deque<Task*>             taskDeque;
    mutex                    mtx;
    condition_variable       ready;
    bool                     done = false;

    bool tryPop(Task** task)
    {
        unique_lock<mutex> lock{mtx, try_to_lock};
        if (!lock || taskDeque.empty()) return false;
        *task = taskDeque.front();
        taskDeque.pop_front();

        return true;
    }

    bool tryPush(Task* task)
    {
        {
            unique_lock<mutex> lock{mtx, try_to_lock};
            if (!lock) return false;
            taskDeque.push_back(task);
        }

        ready.notify_one();

        return true;
    }

    void complete()
    {
        {
            unique_lock<mutex> lock{mtx};
            done = true;
        }
        ready.notify_all();
    }

    bool pop(Task** task)
    {
        unique_lock<mutex> lock{mtx};

        while (taskDeque.empty() && !done) {
            ready.wait(lock);
        }

        if (taskDeque.empty()) return false;

        *task = taskDeque.front();
        taskDeque.pop_front();

        return true;
    }

    void push(Task* task)
    {
        {
            unique_lock<mutex> lock{mtx};
            taskDeque.push_back(task);
        }

        ready.notify_one();
    }

};


struct TaskSchedulerImpl
{
    uint32_t                       threadCnt;
    vector<thread>                 threads;
    vector<TaskQueue>              taskQueues;
    atomic<uint32_t>               idx{0};

    TaskSchedulerImpl(unsigned threadCnt) : threadCnt(threadCnt), taskQueues(threadCnt)
    {
        for (unsigned i = 0; i < threadCnt; ++i) {
            threads.emplace_back([&, i] { run(i); });
        }
    }

    ~TaskSchedulerImpl()
    {
        for (auto& queue : taskQueues) queue.complete();
        for (auto& thread : threads) thread.join();
    }

    void run(unsigned i)
    {
        Task* task;

        //Thread Loop
        while (true) {
            auto success = false;
            for (unsigned x = 0; x < threadCnt * 2; ++x) {
                if (taskQueues[(i + x) % threadCnt].tryPop(&task)) {
                    success = true;
                    break;
                }
            }

            if (!success && !taskQueues[i].pop(&task)) break;
            (*task)(i + 1);
        }
    }

    void request(Task* task)
    {
        //Async
        if (threadCnt > 0) {
            task->prepare();
            auto i = idx++;
            for (unsigned n = 0; n < threadCnt; ++n) {
                if (taskQueues[(i + n) % threadCnt].tryPush(task)) return;
            }
            taskQueues[i % threadCnt].push(task);
        //Sync
        } else {
            task->run(0);
        }
    }
};

}

static TaskSchedulerImpl* inst = nullptr;

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void TaskScheduler::init(unsigned threads)
{
    if (inst) return;
    inst = new TaskSchedulerImpl(threads);
}


void TaskScheduler::term()
{
    if (!inst) return;
    delete(inst);
    inst = nullptr;
}


void TaskScheduler::request(Task* task)
{
    if (inst) inst->request(task);
}


unsigned TaskScheduler::threads()
{
    if (inst) return inst->threadCnt;
    return 0;
}
