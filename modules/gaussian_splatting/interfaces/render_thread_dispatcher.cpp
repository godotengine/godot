#include "render_thread_dispatcher.h"

#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "servers/rendering_server.h"

bool RenderThreadDispatcher::dispatch_call_on_render_thread_blocking(const Callable &p_callable, bool *r_dispatched,
        bool p_allow_timeout, uint64_t *r_request_id, const char *p_log_prefix) {
    if (r_dispatched) {
        *r_dispatched = false;
    }
    if (r_request_id) {
        *r_request_id = 0;
    }

    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs || rs->is_on_render_thread()) {
        return false;
    }
    if (!rs->is_render_loop_enabled()) {
        return false;
    }
    OS *os = OS::get_singleton();
    ERR_FAIL_NULL_V(os, false);

    const String log_prefix = p_log_prefix ? String(p_log_prefix) : String("[RenderThreadDispatcher]");
    MutexLock dispatch_lock(dispatch_mutex);
    const uint64_t request_id = next_request_id.fetch_add(1, std::memory_order_acq_rel);
    if (r_request_id) {
        *r_request_id = request_id;
    }
    const uint64_t dispatch_wait_start_usec = os->get_ticks_usec();
    static constexpr uint64_t STALL_LOG_INTERVAL_USEC = 5000000;
    static constexpr uint64_t DEFAULT_TIMEOUT_USEC = 15000000;
    static constexpr uint32_t POLL_SLEEP_USEC = 1000;
    uint64_t effective_timeout_usec = timeout_usec.load(std::memory_order_acquire);
    if (effective_timeout_usec == 0) {
        effective_timeout_usec = DEFAULT_TIMEOUT_USEC;
    }
    uint64_t next_stall_log_usec = STALL_LOG_INTERVAL_USEC;
    bool logged_stall = false;
    uint64_t next_render_loop_disabled_log_usec = 0;

    rs->call_on_render_thread(p_callable.bind(request_id));
    if (r_dispatched) {
        *r_dispatched = true;
    }
    while (completed_request_id.load(std::memory_order_acquire) < request_id) {
        if (dispatch_semaphore.try_wait()) {
            continue;
        }

        const uint64_t elapsed_usec = os->get_ticks_usec() - dispatch_wait_start_usec;
        if (!rs->is_render_loop_enabled()) {
            const uint64_t completed = completed_request_id.load(std::memory_order_acquire);
            if (p_allow_timeout || next_render_loop_disabled_log_usec == 0 ||
                    elapsed_usec >= next_render_loop_disabled_log_usec) {
                ERR_PRINT(vformat("%s stalled request_id=%d (completed=%d elapsed_ms=%.2f): render loop disabled while waiting",
                        log_prefix,
                        uint64_t(request_id),
                        uint64_t(completed),
                        double(elapsed_usec) / 1000.0));
                logged_stall = true;
                if (!p_allow_timeout) {
                    next_render_loop_disabled_log_usec = elapsed_usec + STALL_LOG_INTERVAL_USEC;
                }
            }
            if (p_allow_timeout) {
                return false;
            }
            os->delay_usec(POLL_SLEEP_USEC);
            continue;
        }
        if (p_allow_timeout && elapsed_usec >= effective_timeout_usec) {
            const uint64_t completed = completed_request_id.load(std::memory_order_acquire);
            ERR_PRINT(vformat("%s timed out request_id=%d (completed=%d timeout_ms=%.2f elapsed_ms=%.2f); escaping blocking wait",
                    log_prefix,
                    uint64_t(request_id),
                    uint64_t(completed),
                    double(effective_timeout_usec) / 1000.0,
                    double(elapsed_usec) / 1000.0));
            return false;
        }
        if (elapsed_usec >= next_stall_log_usec) {
            const uint64_t completed = completed_request_id.load(std::memory_order_acquire);
            ERR_PRINT(vformat("%s stalled request_id=%d (completed=%d render_loop_enabled=%s elapsed_ms=%.2f); continuing to wait for callback completion",
                    log_prefix,
                    uint64_t(request_id),
                    uint64_t(completed),
                    rs->is_render_loop_enabled() ? "true" : "false",
                    double(elapsed_usec) / 1000.0));
            logged_stall = true;
            next_stall_log_usec = elapsed_usec + STALL_LOG_INTERVAL_USEC;
        }

        os->delay_usec(POLL_SLEEP_USEC);
    }

    if (logged_stall) {
        const uint64_t elapsed_usec = os->get_ticks_usec() - dispatch_wait_start_usec;
        WARN_PRINT(vformat("%s recovered request_id=%d after %.2f ms",
                log_prefix, uint64_t(request_id), double(elapsed_usec) / 1000.0));
    }

    return true;
}

void RenderThreadDispatcher::notify_completed(uint64_t p_request_id) {
    uint64_t completed = completed_request_id.load(std::memory_order_acquire);
    while (completed < p_request_id &&
            !completed_request_id.compare_exchange_weak(
                    completed, p_request_id, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    dispatch_semaphore.post();
}

void RenderThreadDispatcher::set_timeout_usec(uint64_t p_timeout_usec) {
    timeout_usec.store(p_timeout_usec, std::memory_order_release);
}

uint64_t RenderThreadDispatcher::get_timeout_usec() const {
    return timeout_usec.load(std::memory_order_acquire);
}

uint64_t RenderThreadDispatcher::get_next_request_id() const {
    return next_request_id.load(std::memory_order_acquire);
}

uint64_t RenderThreadDispatcher::get_completed_request_id() const {
    return completed_request_id.load(std::memory_order_acquire);
}

void RenderThreadDispatcher::promote_latest_data_request_id(uint64_t p_request_id) {
    uint64_t previous_request_id = latest_data_request_id.load(std::memory_order_acquire);
    while (previous_request_id < p_request_id &&
            !latest_data_request_id.compare_exchange_weak(
                    previous_request_id, p_request_id, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
}

uint64_t RenderThreadDispatcher::get_latest_data_request_id() const {
    return latest_data_request_id.load(std::memory_order_acquire);
}

void RenderThreadDispatcher::set_latest_data_result(Error p_error) {
    latest_data_result.store(int(p_error), std::memory_order_release);
}

Error RenderThreadDispatcher::get_latest_data_result() const {
    return Error(latest_data_result.load(std::memory_order_acquire));
}
