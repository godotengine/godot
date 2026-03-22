#ifndef GS_RENDER_THREAD_DISPATCHER_H
#define GS_RENDER_THREAD_DISPATCHER_H

#include "core/error/error_list.h"
#include "core/variant/callable.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
#include <atomic>

class IRenderThreadDispatcher {
public:
    virtual ~IRenderThreadDispatcher() = default;

    virtual bool dispatch_call_on_render_thread_blocking(const Callable &p_callable, bool *r_dispatched,
            bool p_allow_timeout, uint64_t *r_request_id, const char *p_log_prefix) = 0;
    virtual void notify_completed(uint64_t p_request_id) = 0;
    virtual void set_timeout_usec(uint64_t p_timeout_usec) = 0;
    virtual uint64_t get_timeout_usec() const = 0;
    virtual uint64_t get_next_request_id() const = 0;
    virtual uint64_t get_completed_request_id() const = 0;
    virtual void promote_latest_data_request_id(uint64_t p_request_id) = 0;
    virtual uint64_t get_latest_data_request_id() const = 0;
    virtual void set_latest_data_result(Error p_error) = 0;
    virtual Error get_latest_data_result() const = 0;
};

class RenderThreadDispatcher : public IRenderThreadDispatcher {
public:
    RenderThreadDispatcher() = default;
    ~RenderThreadDispatcher() override = default;

    bool dispatch_call_on_render_thread_blocking(const Callable &p_callable, bool *r_dispatched,
            bool p_allow_timeout, uint64_t *r_request_id, const char *p_log_prefix) override;
    void notify_completed(uint64_t p_request_id) override;
    void set_timeout_usec(uint64_t p_timeout_usec) override;
    uint64_t get_timeout_usec() const override;
    uint64_t get_next_request_id() const override;
    uint64_t get_completed_request_id() const override;
    void promote_latest_data_request_id(uint64_t p_request_id) override;
    uint64_t get_latest_data_request_id() const override;
    void set_latest_data_result(Error p_error) override;
    Error get_latest_data_result() const override;

private:
    mutable Mutex dispatch_mutex;
    mutable Semaphore dispatch_semaphore;
    std::atomic<uint64_t> next_request_id{1};
    std::atomic<uint64_t> completed_request_id{0};
    std::atomic<uint64_t> timeout_usec{15000000};
    std::atomic<uint64_t> latest_data_request_id{0};
    std::atomic<int> latest_data_result{int(OK)};
};

#endif // GS_RENDER_THREAD_DISPATCHER_H
