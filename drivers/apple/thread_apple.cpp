/**************************************************************************/
/*  thread_apple.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "thread_apple.h"

#include "core/error/error_macros.h"
#include "core/object/script_language.h"
#include "core/string/ustring.h"
#if defined(__APPLE__)
#define QUARK_APPLE 1
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <deque>

#if QUARK_APPLE
  #include <pthread.h>
  #include <dispatch/dispatch.h>
  #include <TargetConditionals.h>
  #include <os/lock.h>     // os_unfair_lock
  #include <os/activity.h> // os_activity_create
  #include <sys/resource.h>
#endif

namespace godot_apple {

// Cooperative cancellation token used across all driver components.
class CancelToken {
public:
    void cancel() noexcept { flag_.store(true, std::memory_order_release); }
    bool cancelled() const noexcept { return flag_.load(std::memory_order_acquire); }
private:
    std::atomic<bool> flag_{false};
};

// QoS-based priority mapping for Apple threads.
enum class ThreadPriority { Background, Utility, Default, UserInitiated, UserInteractive };

inline void set_current_thread_name(const char* name) noexcept {
#if QUARK_APPLE
    if (name) pthread_setname_np(name); // length limit ~63 chars
#else
    (void)name;
#endif
}

inline void set_current_thread_priority(ThreadPriority p) noexcept {
#if QUARK_APPLE
    qos_class_t qos = QOS_CLASS_DEFAULT;
    switch (p) {
        case ThreadPriority::Background:      qos = QOS_CLASS_BACKGROUND; break;
        case ThreadPriority::Utility:         qos = QOS_CLASS_UTILITY; break;
        case ThreadPriority::Default:         qos = QOS_CLASS_DEFAULT; break;
        case ThreadPriority::UserInitiated:   qos = QOS_CLASS_USER_INITIATED; break;
        case ThreadPriority::UserInteractive: qos = QOS_CLASS_USER_INTERACTIVE; break;
    }
    pthread_set_qos_class_self_np(qos, 0);
#else
    (void)p;
#endif
}

// Fast mutex: os_unfair_lock on Apple, std::mutex fallback otherwise.
class MutexApple {
public:
    MutexApple() noexcept {
#if QUARK_APPLE
        lock_ = OS_UNFAIR_LOCK_INIT;
#endif
    }
    void lock() noexcept {
#if QUARK_APPLE
        os_unfair_lock_lock(&lock_);
#else
        m_.lock();
#endif
    }
    bool try_lock() noexcept {
#if QUARK_APPLE
        return os_unfair_lock_trylock(&lock_);
#else
        return m_.try_lock();
#endif
    }
    void unlock() noexcept {
#if QUARK_APPLE
        os_unfair_lock_unlock(&lock_);
#else
        m_.unlock();
#endif
    }
private:
#if QUARK_APPLE
    os_unfair_lock lock_;
#else
    std::mutex m_;
#endif
};

// Read-write lock implemented via condition variable (fair-ish behavior).
class RWLockApple {
public:
    void lock_read() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return writers_ == 0 && !writer_active_; });
        ++readers_;
    }
    void unlock_read() {
        std::lock_guard<std::mutex> lk(m_);
        if (--readers_ == 0) cv_.notify_all();
    }
    void lock_write() {
        std::unique_lock<std::mutex> lk(m_);
        ++writers_;
        cv_.wait(lk, [&]{ return readers_ == 0 && !writer_active_; });
        writer_active_ = true;
        --writers_;
    }
    void unlock_write() {
        std::lock_guard<std::mutex> lk(m_);
        writer_active_ = false;
        cv_.notify_all();
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    int readers_ = 0;
    int writers_ = 0;
    bool writer_active_ = false;
};

// Condition variable wrapper for signaling/waiting with predicate.
class CondVarApple {
public:
    void wait(std::unique_lock<std::mutex>& lk) { cv_.wait(lk); }
    template <class Pred>
    void wait(std::unique_lock<std::mutex>& lk, Pred pred) { cv_.wait(lk, pred); }
    void notify_one() noexcept { cv_.notify_one(); }
    void notify_all() noexcept { cv_.notify_all(); }
private:
    std::condition_variable cv_;
};

// Counting semaphore implemented with CV; supports timeouts.
class SemaphoreApple {
public:
    explicit SemaphoreApple(int initial = 0) : count_(initial) {}
    void post(int n = 1) {
        std::lock_guard<std::mutex> lk(m_);
        count_ += n;
        for (int i = 0; i < n; ++i) cv_.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return count_ > 0; });
        --count_;
    }
    bool try_wait_for(std::chrono::milliseconds ms) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(lk, ms, [&]{ return count_ > 0; })) return false;
        --count_;
        return true;
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    int count_;
};

// Thread abstraction compatible with Godot-style usage.
class ThreadApple {
public:
    using Task = std::function<void(const CancelToken&)>;

    ThreadApple() = default;

    // Start a new thread with name and priority.
    void start(Task task,
               const char* name = "godot-apple-thread",
               ThreadPriority prio = ThreadPriority::Default) {
        if (!task) throw std::invalid_argument("Thread task must not be empty");
        if (t_.joinable()) throw std::runtime_error("Thread already started");

        cancel_ = std::make_shared<CancelToken>();
        running_.store(true, std::memory_order_release);

        t_ = std::thread([this, task, name, prio](){
            set_current_thread_name(name);
            set_current_thread_priority(prio);

#if QUARK_APPLE
            os_activity_t act = os_activity_create("Godot Apple Thread", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_DEFAULT);
#endif

            try { task(*cancel_); }
            catch (const std::exception& e) { std::fprintf(stderr, "Thread exception: %s\n", e.what()); }
            catch (...) { std::fprintf(stderr, "Thread unknown exception\n"); }

#if QUARK_APPLE
            if (act) os_release(act);
#endif
            running_.store(false, std::memory_order_release);
        });
    }

    // Request cooperative cancellation; task should check token regularly.
    void request_cancel() noexcept { if (cancel_) cancel_->cancel(); }

    // Wait for thread to finish. Returns true when done.
    bool wait_to_finish() {
        if (t_.joinable()) { t_.join(); return true; }
        return true;
    }

    // Try join with timeout by polling; returns false if timed out.
    bool wait_to_finish_for(std::chrono::milliseconds timeout) {
        if (!t_.joinable()) return true;
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        using namespace std::chrono_literals;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!running_.load(std::memory_order_acquire)) break;
            std::this_thread::sleep_for(5ms);
        }
        if (t_.joinable()) t_.join();
        return true;
    }

    bool is_active() const noexcept { return running_.load(std::memory_order_acquire); }

    // Return a lightweight string ID (not OS TID, portable).
    std::string get_id() const {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        return oss.str();
    }

    ~ThreadApple() {
        request_cancel();
        if (t_.joinable()) t_.join();
    }

private:
    std::thread t_;
    std::shared_ptr<CancelToken> cancel_;
    std::atomic<bool> running_{false};
};

// Priority-aware thread pool for parallel tasks.
class ThreadPoolApple {
public:
    enum class Priority { Low, Normal, High };
    using Task = std::function<void(const CancelToken&)>;

    explicit ThreadPoolApple(size_t threads = std::max(2u, std::thread::hardware_concurrency()),
                             size_t queue_capacity = 1024)
        : stop_(false), capacity_(queue_capacity), cancel_(std::make_shared<CancelToken>()) {
        workers_.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this, i]{
                set_current_thread_name(("godot-pool-" + std::to_string(i)).c_str());
                set_current_thread_priority(ThreadPriority::Default);
                worker_loop();
            });
        }
    }

    ~ThreadPoolApple() { shutdown(); }

    void enqueue(Task task, Priority prio = Priority::Normal) {
        if (!task) throw std::invalid_argument("ThreadPool task must not be empty");
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&]{ return stop_ || q_.size() < capacity_; });
        if (stop_) throw std::runtime_error("ThreadPool is stopping");
        push_unlocked(std::move(task), prio);
        not_empty_.notify_one();
    }

    bool try_enqueue(Task task, Priority prio, std::chrono::milliseconds timeout) {
        if (!task) return false;
        std::unique_lock<std::mutex> lk(m_);
        if (!not_full_.wait_for(lk, timeout, [&]{ return stop_ || q_.size() < capacity_; })) return false;
        if (stop_) return false;
        push_unlocked(std::move(task), prio);
        not_empty_.notify_one();
        return true;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) return;
            stop_ = true;
            cancel_->cancel();
        }
        not_empty_.notify_all();
        not_full_.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
        workers_.clear();
        q_.clear();
    }

    std::shared_ptr<CancelToken> cancel_token() const { return cancel_; }

private:
    struct Item {
        Task task;
        Priority prio;
        uint64_t seq;
    };

    std::vector<std::thread> workers_;
    std::deque<Item> q_;
    size_t capacity_;
    std::mutex m_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::atomic<bool> stop_;
    std::shared_ptr<CancelToken> cancel_;
    uint64_t next_seq_ = 0;

    void push_unlocked(Task t, Priority p) {
        q_.push_back(Item{std::move(t), p, next_seq_++});
        std::stable_sort(q_.begin(), q_.end(), [](const Item& a, const Item& b){
            int pa = static_cast<int>(a.prio), pb = static_cast<int>(b.prio);
            if (pa != pb) return pa > pb; // High > Normal > Low
            return a.seq < b.seq;
        });
    }

    bool pop_unlocked(Item& out) {
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void worker_loop() {
        while (true) {
            Item item;
            {
                std::unique_lock<std::mutex> lk(m_);
                not_empty_.wait(lk, [&]{ return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) break;
                pop_unlocked(item);
                not_full_.notify_one();
            }
            try {
                if (!cancel_->cancelled()) item.task(*cancel_);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "ThreadPool task error: %s\n", e.what());
            } catch (...) {
                std::fprintf(stderr, "ThreadPool task unknown error\n");
            }
        }
    }
};

// GCD-based timers with cancellation.
class GcdTimerApple {
public:
    using Task = std::function<void(const CancelToken&)>;

    GcdTimerApple() = default;

    void start_repeating(Task task, std::chrono::milliseconds interval) {
        stop();
#if QUARK_APPLE
        cancel_ = std::make_shared<CancelToken>();
        queue_ = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
        timer_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
        dispatch_source_set_timer(timer_, dispatch_time(DISPATCH_TIME_NOW, interval.count() * NSEC_PER_MSEC),
                                  interval.count() * NSEC_PER_MSEC, interval.count() * NSEC_PER_MSEC / 10);
        auto cancel = cancel_;
        dispatch_source_set_event_handler(timer_, ^{
            if (cancel->cancelled()) return;
            try { task(*cancel); } catch (const std::exception& e) { std::fprintf(stderr, "Timer task error: %s\n", e.what()); }
        });
        dispatch_resume(timer_);
#else
        (void)task; (void)interval;
#endif
    }

    static void after(Task task, std::chrono::milliseconds delay) {
#if QUARK_APPLE
        auto cancel = std::make_shared<CancelToken>();
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay.count() * NSEC_PER_MSEC),
                       dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            if (cancel->cancelled()) return;
            try { task(*cancel); } catch (const std::exception& e) { std::fprintf(stderr, "After task error: %s\n", e.what()); }
        });
#else
        (void)task; (void)delay;
#endif
    }

    void stop() {
#if QUARK_APPLE
        if (cancel_) cancel_->cancel();
        if (timer_) {
            dispatch_source_cancel(timer_);
            timer_ = nullptr;
        }
        queue_ = nullptr;
#endif
    }

    ~GcdTimerApple() { stop(); }

private:
#if QUARK_APPLE
    dispatch_queue_t queue_ = nullptr;
    dispatch_source_t timer_ = nullptr;
#endif
    std::shared_ptr<CancelToken> cancel_;
};

// GCD helpers for async/after/on_queue execution.
class AppleDispatch {
public:
    using Task = std::function<void(const CancelToken&)>;

    static void async(Task t) {
#if QUARK_APPLE
        auto cancel = std::make_shared<CancelToken>();
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            if (!cancel->cancelled()) safe_run(t, *cancel);
        });
#else
        (void)t;
#endif
    }

    static void after(Task t, std::chrono::milliseconds delay) {
#if QUARK_APPLE
        auto cancel = std::make_shared<CancelToken>();
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay.count() * NSEC_PER_MSEC),
                       dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            if (!cancel->cancelled()) safe_run(t, *cancel);
        });
#else
        (void)t; (void)delay;
#endif
    }

    static void on_queue(dispatch_queue_t q, Task t) {
#if QUARK_APPLE
        auto cancel = std::make_shared<CancelToken>();
        dispatch_async(q, ^{
            if (!cancel->cancelled()) safe_run(t, *cancel);
        });
#else
        (void)q; (void)t;
#endif
    }

private:
    static void safe_run(Task& t, const CancelToken& c) {
        try { t(c); }
        catch (const std::exception& e) { std::fprintf(stderr, "GCD task error: %s\n", e.what()); }
        catch (...) { std::fprintf(stderr, "GCD task unknown error\n"); }
    }
};

} // namespace godot_apple
