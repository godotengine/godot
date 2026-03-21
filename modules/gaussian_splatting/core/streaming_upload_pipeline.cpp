#include "streaming_upload_pipeline.h"
#include "gaussian_streaming.h"
#include "gs_project_settings.h"
#include "streaming_queue_pressure_controller.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "quality_tier_config.h"
#include "../interfaces/sync_policy.h"
#include "../renderer/gpu_debug_utils.h"
#include <algorithm>
#include <utility>

namespace {
static constexpr uint32_t STREAMING_DEFAULT_MAX_UPLOAD_MB_PER_FRAME = 128;
static constexpr uint32_t STREAMING_DEFAULT_MAX_UPLOAD_MB_PER_SLICE = 16;
static constexpr uint32_t STREAMING_DEFAULT_MAX_UPLOAD_MB_PER_SECOND = 0;

struct StreamingTierCapPolicy {
    String tier_preset = "custom";
    bool active = false;
    uint32_t upload_mb_per_frame = 0;
    uint32_t upload_mb_per_slice = 0;
    uint32_t upload_mb_per_second = 0;
    uint32_t vram_budget_mb = 0;
    uint32_t min_chunks_in_vram = 0;
    uint32_t max_chunks_in_vram = 0;
};

bool _project_setting_has_override(ProjectSettings *ps, const StringName &name) {
    if (!ps || !ps->has_setting(name) || !ps->property_can_revert(name)) {
        return false;
    }
    return ps->get_setting_with_override(name) != ps->property_get_revert(name);
}

uint32_t _resolve_tiered_cap_uint(ProjectSettings *ps, const StringName &name, uint32_t fallback,
        bool tier_active, uint32_t tier_value, String &r_source) {
    const uint32_t configured_value = gs::settings::get_uint(ps, name, fallback);
    const bool has_project_override = _project_setting_has_override(ps, name);
    if (tier_active && !has_project_override) {
        r_source = "tier_preset";
        return tier_value;
    }
    r_source = has_project_override ? "project_override" : "project_default";
    return configured_value;
}

StreamingTierCapPolicy _resolve_streaming_tier_cap_policy(ProjectSettings *ps) {
    StreamingTierCapPolicy policy;
    if (!ps) {
        return policy;
    }

    const StringName tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const Variant tier_preset_value = ps->has_setting(tier_preset_setting)
            ? ps->get_setting_with_override(tier_preset_setting)
            : Variant("custom");
    policy.tier_preset = String(tier_preset_value)
                                 .strip_edges()
                                 .to_lower();
    const bool apply_tier_budgets =
            gs::settings::get_bool(ps, "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", true);
    if (!apply_tier_budgets) {
        return policy;
    }

    QualityTierConfig tier_config;
    if (!get_quality_tier_config(policy.tier_preset, tier_config)) {
        return policy;
    }

    policy.active = true;
    policy.upload_mb_per_frame = tier_config.streaming_upload_mb_per_frame;
    policy.upload_mb_per_slice = tier_config.streaming_upload_mb_per_slice;
    policy.upload_mb_per_second = tier_config.streaming_upload_mb_per_second;
    policy.vram_budget_mb = tier_config.streaming_vram_budget_mb;
    policy.min_chunks_in_vram = tier_config.streaming_min_chunks_in_vram;
    policy.max_chunks_in_vram = tier_config.streaming_max_chunks_in_vram;
    return policy;
}

void _atomic_saturating_sub(std::atomic<uint32_t> &p_value, uint32_t p_amount) {
    if (p_amount == 0) {
        return;
    }
    uint32_t current = p_value.load(std::memory_order_relaxed);
    while (true) {
        const uint32_t next = current > p_amount ? (current - p_amount) : 0;
        if (p_value.compare_exchange_weak(current, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return;
        }
    }
}

uint64_t _ticks_usec_now() {
    OS *os = OS::get_singleton();
    return os ? os->get_ticks_usec() : 0;
}

StreamingQueuePressureController::PressureSummary _summarize_queue_pressure_checked(
        const StreamingQueuePressureController::PressureSample &p_sample,
        const char *p_context) {
    StreamingQueuePressureController::PressureSummary summary =
            StreamingQueuePressureController::summarize(p_sample);
    String summary_error;
    if (StreamingQueuePressureController::validate_summary_invariants(summary, p_sample, &summary_error)) {
        return summary;
    }

    WARN_PRINT(vformat("[Streaming] Queue pressure summary invariant violated (%s): %s",
            p_context ? String(p_context) : String("unknown"),
            summary_error));
    summary.active = false;
    summary.cap_active = false;
    summary.pack_source_active = false;
    summary.upload_source_active = false;
    summary.sync_source_active = false;
    summary.backlog_depth = MAX(p_sample.sync_fallback_queue_depth,
            MAX(p_sample.pack_queue_depth, p_sample.upload_queue_depth));
    summary.source = StreamingQueuePressureController::SOURCE_NONE;
    summary.reason = StreamingQueuePressureController::REASON_NONE;
    return summary;
}

void _validate_queue_pressure_latched_state(bool &r_active, String &r_source, String &r_reason, const char *p_context) {
    String latch_error;
    if (StreamingQueuePressureController::validate_latched_state_invariants(r_active, r_source, r_reason, &latch_error)) {
        return;
    }
    WARN_PRINT(vformat("[Streaming] Queue pressure latch invariant violated (%s): %s",
            p_context ? String(p_context) : String("unknown"),
            latch_error));
    StreamingQueuePressureController::reset_latched_state(r_active, r_source, r_reason);
}

void _release_chunk_slot_if_matches(GaussianAtlasAllocator &p_allocator, uint64_t p_chunk_key, uint32_t p_expected_slot) {
    uint32_t mapped_slot = UINT32_MAX;
    if (!p_allocator.get_slot(p_chunk_key, mapped_slot)) {
        return;
    }
    if (p_expected_slot != UINT32_MAX && mapped_slot != p_expected_slot) {
        return;
    }
    p_allocator.release_slot(p_chunk_key);
}

bool _chunk_slot_matches_allocator(const GaussianAtlasAllocator &p_allocator, uint64_t p_chunk_key, uint32_t p_expected_slot, uint32_t *r_mapped_slot = nullptr) {
    uint32_t mapped_slot = UINT32_MAX;
    const bool has_slot = p_allocator.get_slot(p_chunk_key, mapped_slot);
    if (r_mapped_slot) {
        *r_mapped_slot = mapped_slot;
    }
    return has_slot && mapped_slot == p_expected_slot;
}
} // namespace

#ifdef DEV_ENABLED

StreamingUploadPipeline::PackTelemetry::Snapshot
StreamingUploadPipeline::PackTelemetry::exchange_and_reset() {
    Snapshot s;
    // SafeNumeric has no exchange(), so we get-then-set to zero.
    // This runs on the main thread once per frame; the tiny race window
    // between get() and set() is acceptable for telemetry counters.
    s.pack_time_total = pack_time_usec_total.get();
    pack_time_usec_total.set(0);
    s.pack_time_max = pack_time_usec_max.get();
    pack_time_usec_max.set(0);
    s.pack_jobs = pack_jobs_completed.get();
    pack_jobs_completed.set(0);
    s.upload_bytes = upload_bytes_total.get();
    upload_bytes_total.set(0);
    s.upload_chunks = upload_chunks_completed.get();
    upload_chunks_completed.set(0);
    s.pack_queue_lat_total = pack_queue_latency_usec_total.get();
    pack_queue_latency_usec_total.set(0);
    s.pack_queue_lat_max = pack_queue_latency_usec_max.get();
    pack_queue_latency_usec_max.set(0);
    s.pack_queue_lat_samples = pack_queue_latency_samples.get();
    pack_queue_latency_samples.set(0);
    s.upload_queue_lat_total = upload_queue_latency_usec_total.get();
    upload_queue_latency_usec_total.set(0);
    s.upload_queue_lat_max = upload_queue_latency_usec_max.get();
    upload_queue_latency_usec_max.set(0);
    s.upload_queue_lat_samples = upload_queue_latency_samples.get();
    upload_queue_latency_samples.set(0);
    s.mutex_wait_total = pack_mutex_wait_usec_total.get();
    pack_mutex_wait_usec_total.set(0);
    s.mutex_wait_max = pack_mutex_wait_usec_max.get();
    pack_mutex_wait_usec_max.set(0);
    s.mutex_wait_samples = pack_mutex_wait_samples.get();
    pack_mutex_wait_samples.set(0);
    return s;
}

StreamingUploadPipeline::PackTelemetry::Snapshot
StreamingUploadPipeline::PackTelemetry::read_current() const {
    Snapshot s;
    s.pack_time_total = pack_time_usec_total.get();
    s.pack_time_max = pack_time_usec_max.get();
    s.pack_jobs = pack_jobs_completed.get();
    s.upload_bytes = upload_bytes_total.get();
    s.upload_chunks = upload_chunks_completed.get();
    s.pack_queue_lat_total = pack_queue_latency_usec_total.get();
    s.pack_queue_lat_max = pack_queue_latency_usec_max.get();
    s.pack_queue_lat_samples = pack_queue_latency_samples.get();
    s.upload_queue_lat_total = upload_queue_latency_usec_total.get();
    s.upload_queue_lat_max = upload_queue_latency_usec_max.get();
    s.upload_queue_lat_samples = upload_queue_latency_samples.get();
    s.mutex_wait_total = pack_mutex_wait_usec_total.get();
    s.mutex_wait_max = pack_mutex_wait_usec_max.get();
    s.mutex_wait_samples = pack_mutex_wait_samples.get();
    return s;
}

#endif // DEV_ENABLED


void StreamingUploadPipeline::load_streaming_tuning_config_from_project_settings(GaussianStreamingSystem &system) {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }
    const StreamingTierCapPolicy tier_policy = _resolve_streaming_tier_cap_policy(ps);
    cap_tier_preset = tier_policy.tier_preset;
    cap_tier_active = tier_policy.active;

    async_pack_enabled = gs::settings::get_bool(ps, "rendering/gaussian_splatting/streaming/async_pack_enabled", async_pack_enabled);
    pack_worker_threads = gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/pack_worker_threads", pack_worker_threads);
    max_pack_jobs_in_flight = gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/max_pack_jobs_in_flight", max_pack_jobs_in_flight);
    max_chunk_loads_per_frame = gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame", max_chunk_loads_per_frame);
    system.scheduler.max_prefetch_loads_per_frame = MIN<uint32_t>(
            gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/max_prefetch_loads_per_frame",
                    system.scheduler.max_prefetch_loads_per_frame),
            GaussianStreamingSystem::SchedulerState::MAX_PREFETCH_LOADS_PER_FRAME);
    system.scheduler.max_visible_chunk_scan_per_frame = gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/max_visible_chunk_scan_per_frame",
            system.scheduler.max_visible_chunk_scan_per_frame);
    system.scheduler.max_prefetch_chunk_scan_per_frame = gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/max_prefetch_chunk_scan_per_frame",
            system.scheduler.max_prefetch_chunk_scan_per_frame);
    system.scheduler.queue_pressure_candidate_scan_throttle_enabled = gs::settings::get_bool(ps,
            "rendering/gaussian_splatting/streaming/queue_pressure_candidate_scan_throttle_enabled",
            system.scheduler.queue_pressure_candidate_scan_throttle_enabled);
    system.scheduler.queue_pressure_candidate_scan_throttle_min_queue_depth = MAX<uint32_t>(1u,
            gs::settings::get_uint(ps,
                    "rendering/gaussian_splatting/streaming/queue_pressure_candidate_scan_throttle_min_queue_depth",
                    system.scheduler.queue_pressure_candidate_scan_throttle_min_queue_depth));
    const uint32_t queue_pressure_visible_scan_cap = MAX<uint32_t>(1u, gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/queue_pressure_visible_scan_cap",
            system.scheduler.queue_pressure_candidate_scan_throttle_visible_scan_cap));
    const uint32_t queue_pressure_prefetch_scan_cap = MAX<uint32_t>(1u, gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/queue_pressure_prefetch_scan_cap",
            system.scheduler.queue_pressure_candidate_scan_throttle_prefetch_scan_cap));
    system.scheduler.queue_pressure_candidate_scan_throttle_visible_scan_cap =
            system.scheduler.max_visible_chunk_scan_per_frame > 0
            ? MIN(queue_pressure_visible_scan_cap, system.scheduler.max_visible_chunk_scan_per_frame)
            : queue_pressure_visible_scan_cap;
    system.scheduler.queue_pressure_candidate_scan_throttle_prefetch_scan_cap =
            system.scheduler.max_prefetch_chunk_scan_per_frame > 0
            ? MIN(queue_pressure_prefetch_scan_cap, system.scheduler.max_prefetch_chunk_scan_per_frame)
            : queue_pressure_prefetch_scan_cap;
    system.scheduler.max_sync_fallback_loads_per_frame = MIN<uint32_t>(
            MAX<uint32_t>(1u, gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/max_sync_fallback_loads_per_frame",
                                       system.scheduler.max_sync_fallback_loads_per_frame)),
            GaussianStreamingSystem::SchedulerState::MAX_SYNC_FALLBACK_LOADS_PER_FRAME);
    system.scheduler.max_sync_fallback_queue_size = MAX<uint32_t>(64u, gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/max_sync_fallback_queue_size",
            system.scheduler.max_sync_fallback_queue_size));

    const uint32_t upload_mb = _resolve_tiered_cap_uint(ps,
            "rendering/gaussian_splatting/streaming/max_upload_mb_per_frame",
            STREAMING_DEFAULT_MAX_UPLOAD_MB_PER_FRAME,
            tier_policy.active,
            tier_policy.upload_mb_per_frame,
            cap_source_upload_mb_per_frame);
    const uint32_t slice_mb = _resolve_tiered_cap_uint(ps,
            "rendering/gaussian_splatting/streaming/max_upload_mb_per_slice",
            STREAMING_DEFAULT_MAX_UPLOAD_MB_PER_SLICE,
            tier_policy.active,
            tier_policy.upload_mb_per_slice,
            cap_source_upload_mb_per_slice);
    const uint32_t upload_mbps = _resolve_tiered_cap_uint(ps,
            "rendering/gaussian_splatting/streaming/max_upload_mb_per_second",
            STREAMING_DEFAULT_MAX_UPLOAD_MB_PER_SECOND,
            tier_policy.active,
            tier_policy.upload_mb_per_second,
            cap_source_upload_mb_per_second);

    effective_upload_cap_mb_per_frame = upload_mb;
    effective_upload_cap_mb_per_slice = slice_mb;
    effective_upload_cap_mb_per_second = upload_mbps;
    max_upload_bytes_per_frame = upload_mb == 0 ? 0 : uint64_t(upload_mb) * 1024 * 1024;
    max_upload_bytes_per_slice = slice_mb == 0 ? 0 : uint64_t(slice_mb) * 1024 * 1024;
    max_upload_bytes_per_second = upload_mbps == 0 ? 0 : uint64_t(upload_mbps) * 1024 * 1024;
    if (max_upload_bytes_per_second == 0) {
        upload_budget_tokens = 0;
        upload_budget_last_update_usec = 0;
    } else {
        upload_budget_tokens = MIN<uint64_t>(upload_budget_tokens, max_upload_bytes_per_second * 2);
        if (upload_budget_tokens == 0) {
            upload_budget_tokens = max_upload_bytes_per_second;
        }
    }

    system.eviction.eviction_hysteresis_frames = gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/eviction_hysteresis_frames",
            system.eviction.eviction_hysteresis_frames);
    system.eviction.max_evictions_per_frame = gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/max_evictions_per_frame",
            system.eviction.max_evictions_per_frame);

    if (async_pack_enabled && pack_worker_threads == 0) {
        async_pack_enabled = false;
    }

    const bool want_async = async_pack_enabled && pack_worker_threads > 0;
    const bool threads_mismatch = pack_thread_running.load() &&
            uint32_t(pack_threads.size()) != pack_worker_threads;
    if (want_async && (!pack_thread_running.load() || threads_mismatch)) {
        if (pack_thread_running.load()) {
            stop_pack_threads(system);
        }
        start_pack_threads(system);
    } else if (!want_async && pack_thread_running.load()) {
        stop_pack_threads(system);
    }
}

void StreamingUploadPipeline::start_pack_threads(GaussianStreamingSystem &system) {
    if (pack_thread_running.load()) {
        return;
    }
    pack_thread_exit = false;

    uint32_t thread_count = MAX(1u, pack_worker_threads);
    pack_threads.clear();
    pack_thread_contexts.clear();
    pack_threads.resize(thread_count);
    pack_thread_contexts.resize(thread_count);

    PackThreadContext *contexts = pack_thread_contexts.ptr();
    uint32_t started_count = 0;
    for (uint32_t i = 0; i < thread_count; i++) {
        contexts[i].system = &system;
        contexts[i].thread_index = i;
        pack_threads[i] = memnew(Thread);
        Thread::ID tid = pack_threads[i]->start(StreamingUploadPipeline::pack_thread_entry, &contexts[i]);
        if (tid == Thread::UNASSIGNED_ID) {
            ERR_PRINT(vformat("GaussianStreaming: pack thread %d/%d failed to start.", i, thread_count));
            memdelete(pack_threads[i]);
            pack_threads[i] = nullptr;
        } else {
            started_count++;
        }
    }

    // Compact the arrays so only successfully-started threads remain.
    // This keeps stop_pack_threads() simple: every non-null entry is joinable.
    if (started_count < thread_count) {
        uint32_t write_idx = 0;
        for (uint32_t i = 0; i < thread_count; i++) {
            if (pack_threads[i] != nullptr) {
                if (write_idx != i) {
                    pack_threads[write_idx] = pack_threads[i];
                    pack_thread_contexts[write_idx] = pack_thread_contexts[i];
                    // Update the context's thread_index to match its new slot.
                    pack_thread_contexts[write_idx].thread_index = write_idx;
                }
                write_idx++;
            }
        }
        pack_threads.resize(started_count);
        pack_thread_contexts.resize(started_count);
    }

    if (started_count == 0) {
        WARN_PRINT("GaussianStreaming: all pack threads failed to start; falling back to synchronous packing.");
        pack_threads.clear();
        pack_thread_contexts.clear();
        pack_thread_running = false;
        async_pack_enabled = false;
        return;
    }

    if (started_count < thread_count) {
        WARN_PRINT(vformat("GaussianStreaming: only %d of %d pack threads started; degraded throughput.", started_count, thread_count));
    }

    pack_thread_running = true;
}

void StreamingUploadPipeline::stop_pack_threads(GaussianStreamingSystem &system) {
    if (pack_thread_running.load()) {
        pack_thread_exit = true;

        for (uint32_t i = 0; i < pack_threads.size(); i++) {
            pack_semaphore.post();
        }

        for (uint32_t i = 0; i < pack_threads.size(); i++) {
            if (pack_threads[i]) {
                pack_threads[i]->wait_to_finish();
                memdelete(pack_threads[i]);
                pack_threads[i] = nullptr;
            }
        }
    }

    pack_threads.clear();
    pack_thread_contexts.clear();
    pack_thread_running = false;
    pack_thread_exit = false;
    clear_pending_uploads(system);
}

void StreamingUploadPipeline::pack_thread_entry(void *p_userdata) {
    PackThreadContext *context = static_cast<PackThreadContext *>(p_userdata);
    if (!context || !context->system) {
        return;
    }
    context->system->upload_pipeline.pack_thread_func(*context->system, context->thread_index);
}

void StreamingUploadPipeline::pack_thread_func(GaussianStreamingSystem &system, uint32_t p_thread_index) {
    (void)p_thread_index;
    static constexpr uint32_t PACK_DEQUEUE_BATCH = 4;
    while (!pack_thread_exit.load()) {
        pack_semaphore.wait();
        if (pack_thread_exit.load()) {
            break;
        }

        PackJob jobs[PACK_DEQUEUE_BATCH];
        uint32_t job_count = 0;
        uint32_t semaphore_tokens_to_consume = 0;
        {
            const uint64_t lock_wait_start_usec = _ticks_usec_now();
            MutexLock lock(pack_mutex);
            record_pack_mutex_wait(lock_wait_start_usec);
            const uint64_t dequeue_usec = _ticks_usec_now();
            while (job_count < PACK_DEQUEUE_BATCH && pack_queue_read_idx < pack_queue.size()) {
                jobs[job_count++] = std::move(pack_queue[pack_queue_read_idx++]);
                const PackJob &dequeued_job = jobs[job_count - 1];
                if (dequeue_usec > 0 && dequeued_job.enqueue_usec > 0 && dequeue_usec >= dequeued_job.enqueue_usec) {
                    const uint64_t latency_usec = dequeue_usec - dequeued_job.enqueue_usec;
                    telemetry.add_pack_queue_latency(latency_usec);
                }
            }
            // queue_chunk_load() posts one semaphore token per enqueued job.
            // This worker may dequeue several jobs after a single wait(), so
            // consume extra ready tokens while the queue lock is held. That
            // keeps token accounting aligned with the queue snapshot and avoids
            // stealing wake credits for jobs enqueued after this dequeue.
            semaphore_tokens_to_consume = job_count > 0 ? (job_count - 1) : 0;
            while (semaphore_tokens_to_consume > 0) {
                if (pack_thread_exit.load()) {
                    break;
                }
                if (!pack_semaphore.try_wait()) {
                    break;
                }
                // Shutdown posts one wake token per worker. If exit flips after
                // try_wait() succeeds, return that token so blocked workers can
                // still observe pack_thread_exit and terminate cleanly.
                if (pack_thread_exit.load()) {
                    pack_semaphore.post();
                    break;
                }
                semaphore_tokens_to_consume--;
            }
            compact_queues_locked();
        }

        if (job_count == 0) {
            continue;
        }

        for (uint32_t job_idx = 0; job_idx < job_count; job_idx++) {
            const PackJob &job = jobs[job_idx];
            PendingChunkUpload *upload = memnew(PendingChunkUpload);
            upload->asset_id = job.asset_id;
            upload->chunk_idx = job.chunk_idx;
            upload->buffer_slot = job.buffer_slot;
            upload->asset_generation = job.asset_generation;

            if (job.chunk_count == 0 || !job.data_ref.is_valid()) {
                const uint64_t lock_wait_start_usec = _ticks_usec_now();
                MutexLock lock(pack_mutex);
                record_pack_mutex_wait(lock_wait_start_usec);
                upload->enqueue_usec = _ticks_usec_now();
                upload_queue.push_back(upload);
                sync_cached_queue_depths_locked();
                _atomic_saturating_sub(pack_jobs_in_flight, 1);
                continue;
            }

            LocalVector<Gaussian> gaussian_snapshot;
            LocalVector<Vector3> sh_high_order_snapshot;
            uint32_t sh_first_order = 0;
            uint32_t sh_high_order = 0;
            bool snapshot_ok = false;
            if (job.uses_explicit_source_indices) {
                snapshot_ok = job.chunk_count == static_cast<uint32_t>(job.source_indices.size()) &&
                        job.data_ref->capture_indexed_chunk_snapshot(job.source_indices.ptr(), job.chunk_count,
                                gaussian_snapshot,
                                sh_high_order_snapshot,
                                sh_first_order,
                                sh_high_order);
            } else {
                snapshot_ok = job.data_ref->capture_chunk_snapshot(job.chunk_start, job.chunk_count,
                        gaussian_snapshot,
                        sh_high_order_snapshot,
                        sh_first_order,
                        sh_high_order);
            }
            if (!snapshot_ok || job.chunk_count > static_cast<uint32_t>(gaussian_snapshot.size())) {
                const uint64_t lock_wait_start_usec = _ticks_usec_now();
                MutexLock lock(pack_mutex);
                record_pack_mutex_wait(lock_wait_start_usec);
                upload->enqueue_usec = _ticks_usec_now();
                upload_queue.push_back(upload);
                sync_cached_queue_depths_locked();
                _atomic_saturating_sub(pack_jobs_in_flight, 1);
                continue;
            }

            const Vector3 *sh_coeffs = sh_high_order_snapshot.is_empty()
                    ? nullptr
                    : sh_high_order_snapshot.ptr();

            const bool telemetry_on = telemetry.is_enabled();
            uint64_t pack_start_usec = 0;
            if (telemetry_on) {
                pack_start_usec = OS::get_singleton()->get_ticks_usec();
            }
            pack_gaussians_range(gaussian_snapshot,
                    0,
                    job.chunk_count,
                    upload->packed_data,
                    upload->metrics,
                    sh_coeffs,
                    sh_first_order,
                    sh_high_order);
            if (telemetry_on) {
                const uint64_t pack_end_usec = OS::get_singleton()->get_ticks_usec();
                const uint64_t duration = pack_end_usec - pack_start_usec;
                telemetry.add_pack_time(duration);
            }

            {
                const uint64_t lock_wait_start_usec = _ticks_usec_now();
                MutexLock lock(pack_mutex);
                record_pack_mutex_wait(lock_wait_start_usec);
                upload->enqueue_usec = _ticks_usec_now();
                upload_queue.push_back(upload);
                sync_cached_queue_depths_locked();
            }

            _atomic_saturating_sub(pack_jobs_in_flight, 1);
        }
    }
}

bool StreamingUploadPipeline::queue_chunk_load(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx) {
    if (!async_pack_enabled) {
        return false;
    }

    GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
    if (!asset || !asset->data.is_valid()) {
        return false;
    }

    LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
    if (chunk_idx >= asset_chunks.size()) {
        return false;
    }

    GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[chunk_idx];
    if (!chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX) {
        system._rollback_pending_chunk(asset_id, chunk_idx, chunk, true);
    }
    system._assert_chunk_state_invariant(asset_id, chunk_idx, chunk, "queue_chunk_load.pre");
    if (chunk.is_loaded || chunk.upload_pending) {
        return false;
    }
    if (chunk.count == 0 || chunk.count > GaussianStreamingSystem::CHUNK_SIZE) {
        return false;
    }

    if (max_chunk_loads_per_frame > 0 && queued_chunk_loads_this_frame >= max_chunk_loads_per_frame) {
        chunk_load_cap_hit_this_frame = true;
        StreamingQueuePressureController::mark_latched_state(
                queue_pressure_active,
                queue_pressure_source,
                queue_pressure_reason,
                StreamingQueuePressureController::SOURCE_UPLOAD,
                StreamingQueuePressureController::REASON_CHUNK_LOAD_CAP);
        _validate_queue_pressure_latched_state(
                queue_pressure_active, queue_pressure_source, queue_pressure_reason,
                "queue_chunk_load.chunk_load_cap");
        return false;
    }

    if (max_pack_jobs_in_flight > 0 && pack_jobs_in_flight.load() >= max_pack_jobs_in_flight) {
        StreamingQueuePressureController::mark_latched_state(
                queue_pressure_active,
                queue_pressure_source,
                queue_pressure_reason,
                StreamingQueuePressureController::SOURCE_PACK,
                StreamingQueuePressureController::REASON_PACK_INFLIGHT_CAP);
        _validate_queue_pressure_latched_state(
                queue_pressure_active, queue_pressure_source, queue_pressure_reason,
                "queue_chunk_load.pack_inflight_cap");
        return false;
    }

    PackJob job;
    job.asset_id = asset_id;
    job.chunk_idx = chunk_idx;
    job.asset_generation = asset->generation;
    job.chunk_start = chunk.start_idx;
    job.chunk_count = chunk.count;
    if (chunk.source_index_remapped && asset_id == GaussianStreamingSystem::PRIMARY_ASSET_ID) {
        job.uses_explicit_source_indices = true;
        job.source_indices.resize(chunk.count);
        for (uint32_t i = 0; i < chunk.count; i++) {
            uint32_t source_index = 0;
            if (!system._resolve_primary_chunk_source_index(chunk, i, source_index)) {
                return false;
            }
            job.source_indices[i] = source_index;
        }
    }
    job.data_ref = asset->data;
    if (!job.data_ref.is_valid()) {
        return false;
    }

    if (!system._ensure_atlas_slot_available(asset_id)) {
        return false;
    }

    const uint64_t chunk_key = system._make_chunk_key(asset_id, chunk_idx);
    uint32_t buffer_slot = UINT32_MAX;
    if (!system.atlas_sync.allocator.allocate_slot(chunk_key, buffer_slot)) {
        return false;
    }

    if (chunk.is_loaded || chunk.upload_pending || chunk.buffer_slot != UINT32_MAX) {
        if (!chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX) {
            system._rollback_pending_chunk(asset_id, chunk_idx, chunk, true);
        } else {
            system._assert_chunk_state_invariant(asset_id, chunk_idx, chunk, "queue_chunk_load.recheck");
        }
        if (!chunk.is_loaded && !chunk.upload_pending) {
            _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, buffer_slot);
        }
        return false;
    }

    job.buffer_slot = buffer_slot;
    if (!system._begin_chunk_upload(asset_id, chunk_idx, chunk, buffer_slot)) {
        _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, buffer_slot);
        return false;
    }
    queued_chunk_loads_this_frame++;
    pack_jobs_in_flight.fetch_add(1);

    {
        const uint64_t lock_wait_start_usec = _ticks_usec_now();
        MutexLock lock(pack_mutex);
        record_pack_mutex_wait(lock_wait_start_usec);
        job.enqueue_usec = _ticks_usec_now();
        pack_queue.push_back(job);
        sync_cached_queue_depths_locked();
    }
    pack_semaphore.post();
    return true;
}

void StreamingUploadPipeline::process_upload_queue(GaussianStreamingSystem &system) {
    if (!has_pending_uploads()) {
        return;
    }

    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *submission_rd = system._resolve_submission_device(manager, submission_lock);
    if (!submission_rd || !system.persistent_buffer.is_valid()) {
        clear_pending_uploads(system);
        return;
    }

    auto resolve_upload_chunk = [&](PendingChunkUpload *job, GaussianStreamingSystem::StreamingChunk *&chunk) -> bool {
        const uint64_t chunk_key = system._make_chunk_key(job->asset_id, job->chunk_idx);
        GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(job->asset_id);
        if (!asset) {
            _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, job->buffer_slot);
            memdelete(job);
            return false;
        }

        LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
        if (job->chunk_idx >= asset_chunks.size()) {
            _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, job->buffer_slot);
            memdelete(job);
            return false;
        }

        if (asset->generation != job->asset_generation) {
            system.diagnostics.invariant_generation_violations++;
            system.diagnostics.last_invariant_context = "resolve_upload_chunk.asset_generation";
            system.diagnostics.last_invariant_message = vformat(
                    "[Streaming] Stale upload job dropped: asset=%d chunk=%d queued_generation=%d current_generation=%d.",
                    job->asset_id, job->chunk_idx, job->asset_generation, asset->generation);
            GaussianStreamingSystem::StreamingChunk &stale_chunk = asset_chunks[job->chunk_idx];
            const bool current_chunk_owns_job_slot = (stale_chunk.buffer_slot == job->buffer_slot) &&
                    (stale_chunk.upload_pending || stale_chunk.is_loaded);
            if (!current_chunk_owns_job_slot) {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, job->buffer_slot);
            }
            memdelete(job);
            return false;
        }

        GaussianStreamingSystem::StreamingChunk &resolved_chunk = asset_chunks[job->chunk_idx];
        if (!resolved_chunk.upload_pending || resolved_chunk.buffer_slot != job->buffer_slot) {
            if (!resolved_chunk.is_loaded && resolved_chunk.buffer_slot == job->buffer_slot) {
                system._rollback_pending_chunk(job->asset_id, job->chunk_idx, resolved_chunk, true);
            } else if (!resolved_chunk.is_loaded) {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, job->buffer_slot);
            }
            memdelete(job);
            return false;
        }

        uint32_t mapped_slot = UINT32_MAX;
        if (!_chunk_slot_matches_allocator(system.atlas_sync.allocator, chunk_key, job->buffer_slot, &mapped_slot)) {
            if (!resolved_chunk.is_loaded && resolved_chunk.upload_pending) {
                system._rollback_pending_chunk(job->asset_id, job->chunk_idx, resolved_chunk, false);
            }
            if (mapped_slot != UINT32_MAX) {
                system.atlas_sync.allocator.release_slot(chunk_key);
            } else {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, job->buffer_slot);
            }
            memdelete(job);
            return false;
        }

        system._assert_chunk_state_invariant(job->asset_id, job->chunk_idx, resolved_chunk, "resolve_upload_chunk");
        chunk = &resolved_chunk;
        return true;
    };

    auto upload_job_slices = [&](GaussianStreamingSystem::StreamingChunk &chunk,
                                 PendingChunkUpload *job,
                                 uint64_t total_bytes,
                                 uint64_t &upload_budget,
                                 uint64_t slice_limit,
                                 bool &submitted) -> bool {
        if (!submission_rd || !system.persistent_buffer.is_valid() || chunk.buffer_slot == UINT32_MAX) {
            return false;
        }
        if (!chunk.upload_pending || chunk.is_loaded || chunk.buffer_slot != job->buffer_slot) {
            return false;
        }
        if (job->bytes_uploaded > total_bytes) {
            return false;
        }
        const uint64_t slot_capacity_bytes = uint64_t(GaussianStreamingSystem::CHUNK_SIZE) * sizeof(PackedGaussian);
        if (total_bytes > slot_capacity_bytes) {
            return false;
        }
        if (slice_limit != UINT64_MAX && total_bytes > slice_limit) {
            upload_slice_cap_hit_this_frame = true;
        }

        const uint64_t slot_offset = uint64_t(chunk.buffer_slot) * GaussianStreamingSystem::CHUNK_SIZE * sizeof(PackedGaussian);
        if (slot_offset >= system.persistent_buffer_size) {
            return false;
        }

        while (job->bytes_uploaded < total_bytes && upload_budget > 0) {
            uint64_t remaining = total_bytes - job->bytes_uploaded;
            uint64_t slice_bytes = MIN(remaining, slice_limit);
            if (upload_budget != UINT64_MAX) {
                slice_bytes = MIN(slice_bytes, upload_budget);
            }
            if (slice_bytes == 0) {
                break;
            }

            if (job->bytes_uploaded > UINT32_MAX || slice_bytes > UINT32_MAX) {
                return false;
            }
            const uint64_t write_offset = slot_offset + uint64_t(job->bytes_uploaded);
            if (slice_bytes > system.persistent_buffer_size ||
                    write_offset > system.persistent_buffer_size - slice_bytes) {
                return false;
            }
            if (write_offset > uint64_t(UINT32_MAX)) {
                return false;
            }
            const uint32_t write_offset_u32 = static_cast<uint32_t>(write_offset);
            const uint32_t slice_bytes_u32 = uint32_t(slice_bytes);
            const uint8_t *data_ptr = reinterpret_cast<const uint8_t *>(job->packed_data.ptr());
            submission_rd->buffer_update(system.persistent_buffer, write_offset_u32,
                    slice_bytes_u32, data_ptr + job->bytes_uploaded);
            job->bytes_uploaded += slice_bytes_u32;

            if (upload_budget != UINT64_MAX) {
                upload_budget -= slice_bytes;
            }
            submitted = true;
            if (telemetry.is_enabled()) {
                telemetry.add_upload_bytes(slice_bytes);
            }
        }
        return true;
    };

    auto finalize_upload_job = [&](PendingChunkUpload *job,
                                   GaussianStreamingSystem::StreamingChunk &chunk,
                                   UploadBudgetState &budget_state) {
        if (chunk.is_loaded || !chunk.upload_pending || chunk.buffer_slot != job->buffer_slot) {
            if (!chunk.is_loaded) {
                system._rollback_pending_chunk(job->asset_id, job->chunk_idx, chunk, true);
            }
            memdelete(job);
            return;
        }

        const uint64_t chunk_key = system._make_chunk_key(job->asset_id, job->chunk_idx);
        uint32_t mapped_slot = UINT32_MAX;
        if (!_chunk_slot_matches_allocator(system.atlas_sync.allocator, chunk_key, chunk.buffer_slot, &mapped_slot)) {
            if (!chunk.is_loaded && chunk.upload_pending) {
                system._rollback_pending_chunk(job->asset_id, job->chunk_idx, chunk, false);
            }
            if (mapped_slot != UINT32_MAX) {
                system.atlas_sync.allocator.release_slot(chunk_key);
            }
            memdelete(job);
            return;
        }

        system._complete_chunk_load_common(job->asset_id, job->chunk_idx, chunk);
        system.budget.chunks_loaded_this_frame++;
        system.total_sh_metrics.raw_bytes += job->metrics.raw_bytes;
        system.total_sh_metrics.compressed_bytes += job->metrics.compressed_bytes;
        system.total_sh_metrics.coefficient_count += job->metrics.coefficient_count;

        memdelete(job);
        budget_state.completed_chunks++;
        if (telemetry.is_enabled()) {
            telemetry.add_upload_chunk();
        }
    };

    UploadBudgetState upload_budget_state = prepare_upload_budget_state();
    const uint64_t upload_budget_start = upload_budget_state.upload_budget;
    const uint64_t frame_budget_limit = max_upload_bytes_per_frame == 0 ? UINT64_MAX : max_upload_bytes_per_frame;
    const uint64_t bandwidth_budget_limit = max_upload_bytes_per_second == 0 ? UINT64_MAX : upload_budget_tokens;
    bool submitted = false;

    while (upload_budget_state.upload_budget > 0 && upload_budget_state.completed_chunks < upload_budget_state.chunk_limit) {
        PendingChunkUpload *job = nullptr;
        if (!pop_upload_job(job) || !job) {
            break;
        }

        GaussianStreamingSystem::StreamingChunk *chunk = nullptr;
        if (!resolve_upload_chunk(job, chunk)) {
            continue;
        }
        if (chunk->count == 0 || chunk->count > GaussianStreamingSystem::CHUNK_SIZE) {
            system._rollback_pending_chunk(job->asset_id, job->chunk_idx, *chunk, true);
            memdelete(job);
            continue;
        }
        if (job->packed_data.size() != static_cast<int>(chunk->count)) {
            system._rollback_pending_chunk(job->asset_id, job->chunk_idx, *chunk, true);
            memdelete(job);
            continue;
        }

        const uint64_t total_bytes = uint64_t(job->packed_data.size()) * sizeof(PackedGaussian);
        if (total_bytes == 0) {
            system._rollback_pending_chunk(job->asset_id, job->chunk_idx, *chunk, true);
            memdelete(job);
            continue;
        }

        const bool upload_ok = upload_job_slices(*chunk, job, total_bytes,
                upload_budget_state.upload_budget, upload_budget_state.slice_limit, submitted);
        if (!upload_ok) {
            system._rollback_pending_chunk(job->asset_id, job->chunk_idx, *chunk, true);
            memdelete(job);
            continue;
        }

        if (job->bytes_uploaded >= total_bytes) {
            finalize_upload_job(job, *chunk, upload_budget_state);
        } else {
            requeue_upload_job(job);
        }
    }

    if (submitted) {
        gs_device_utils::safe_submit(submission_rd);
    }

    uint32_t remaining_pack_queue_depth = 0;
    uint32_t remaining_upload_queue_depth = 0;
    get_pending_queue_depths_cached(remaining_pack_queue_depth, remaining_upload_queue_depth);
    const bool pending_after_budget = remaining_pack_queue_depth > 0 || remaining_upload_queue_depth > 0;
    if (pending_after_budget) {
        queue_pressure_active = true;
        if (remaining_pack_queue_depth > 0 && remaining_upload_queue_depth > 0) {
            queue_pressure_source = "combined";
        } else if (remaining_pack_queue_depth > 0) {
            queue_pressure_source = "pack";
        } else {
            queue_pressure_source = "upload";
        }
        queue_pressure_reason = "queue_backlog";
    }
    if (pending_after_budget && upload_budget_state.upload_budget == 0) {
        const bool frame_cap_active = frame_budget_limit != UINT64_MAX;
        const bool bandwidth_cap_active = bandwidth_budget_limit != UINT64_MAX;
        if (frame_cap_active && upload_budget_start == frame_budget_limit) {
            upload_frame_cap_hit_this_frame = true;
        }
        if (bandwidth_cap_active && upload_budget_start == bandwidth_budget_limit) {
            upload_bandwidth_cap_hit_this_frame = true;
        }
    }
    if (pending_after_budget &&
            upload_budget_state.chunk_limit != UINT32_MAX &&
            upload_budget_state.completed_chunks >= upload_budget_state.chunk_limit) {
        chunk_load_cap_hit_this_frame = true;
    }

    StreamingQueuePressureController::PressureSample pressure_sample;
    pressure_sample.pack_queue_depth = remaining_pack_queue_depth;
    pressure_sample.upload_queue_depth = remaining_upload_queue_depth;
    pressure_sample.sync_fallback_queue_depth = 0;
    pressure_sample.pack_inflight_saturated =
            max_pack_jobs_in_flight > 0 &&
            pack_jobs_in_flight.load(std::memory_order_relaxed) >= max_pack_jobs_in_flight;
    pressure_sample.upload_frame_cap_hit = upload_frame_cap_hit_this_frame;
    pressure_sample.upload_bandwidth_cap_hit = upload_bandwidth_cap_hit_this_frame;
    pressure_sample.chunk_load_cap_hit = chunk_load_cap_hit_this_frame;
    pressure_sample.vram_chunk_cap_hit = false;
    pressure_sample.sync_backpressure = false;
    const StreamingQueuePressureController::PressureSummary pressure_summary =
            _summarize_queue_pressure_checked(pressure_sample, "StreamingUploadPipeline::process_upload_queue");
    StreamingQueuePressureController::latch_summary(
            pressure_summary, queue_pressure_active, queue_pressure_source, queue_pressure_reason);
    _validate_queue_pressure_latched_state(
            queue_pressure_active, queue_pressure_source, queue_pressure_reason,
            "StreamingUploadPipeline::process_upload_queue.latch");

    if (max_upload_bytes_per_second > 0 && upload_budget_start != UINT64_MAX) {
        const uint64_t consumed = upload_budget_start > upload_budget_state.upload_budget
                ? (upload_budget_start - upload_budget_state.upload_budget)
                : 0;
        if (consumed >= upload_budget_tokens) {
            upload_budget_tokens = 0;
        } else {
            upload_budget_tokens -= consumed;
        }
    }
}
uint32_t StreamingUploadPipeline::get_pack_queue_depth_unsafe() const {
    return pack_queue_read_idx < pack_queue.size() ? (pack_queue.size() - pack_queue_read_idx) : 0;
}

uint32_t StreamingUploadPipeline::get_upload_queue_depth_unsafe() const {
    return upload_queue_read_idx < upload_queue.size() ? (upload_queue.size() - upload_queue_read_idx) : 0;
}

uint32_t StreamingUploadPipeline::get_pack_queue_depth_cached() const {
    return pack_queue_depth_cached.load(std::memory_order_acquire);
}

uint32_t StreamingUploadPipeline::get_upload_queue_depth_cached() const {
    return upload_queue_depth_cached.load(std::memory_order_acquire);
}

void StreamingUploadPipeline::get_pending_queue_depths_cached(
        uint32_t &r_pack_queue_depth, uint32_t &r_upload_queue_depth) const {
    r_pack_queue_depth = get_pack_queue_depth_cached();
    r_upload_queue_depth = get_upload_queue_depth_cached();
}

void StreamingUploadPipeline::compact_queues_locked() {
    if (pack_queue_read_idx >= pack_queue.size()) {
        pack_queue.clear();
        pack_queue_read_idx = 0;
    } else if (pack_queue_read_idx >= QUEUE_COMPACT_MIN_PREFIX &&
            pack_queue_read_idx * 2 >= pack_queue.size()) {
        const uint32_t remaining = pack_queue.size() - pack_queue_read_idx;
        for (uint32_t i = 0; i < remaining; i++) {
            pack_queue[i] = pack_queue[pack_queue_read_idx + i];
        }
        pack_queue.resize(remaining);
        pack_queue_read_idx = 0;
    }

    if (upload_queue_read_idx >= upload_queue.size()) {
        upload_queue.clear();
        upload_queue_read_idx = 0;
    } else if (upload_queue_read_idx >= QUEUE_COMPACT_MIN_PREFIX &&
            upload_queue_read_idx * 2 >= upload_queue.size()) {
        const uint32_t remaining = upload_queue.size() - upload_queue_read_idx;
        for (uint32_t i = 0; i < remaining; i++) {
            upload_queue[i] = upload_queue[upload_queue_read_idx + i];
        }
        upload_queue.resize(remaining);
        upload_queue_read_idx = 0;
    }
    sync_cached_queue_depths_locked();
}

void StreamingUploadPipeline::sync_cached_queue_depths_locked() {
    pack_queue_depth_cached.store(get_pack_queue_depth_unsafe(), std::memory_order_release);
    upload_queue_depth_cached.store(get_upload_queue_depth_unsafe(), std::memory_order_release);
}

void StreamingUploadPipeline::record_pack_mutex_wait(uint64_t wait_start_usec) {
    const uint64_t now_usec = _ticks_usec_now();
    uint64_t wait_usec = 0;
    if (wait_start_usec > 0 && now_usec > 0 && now_usec >= wait_start_usec) {
        wait_usec = now_usec - wait_start_usec;
    }
    telemetry.add_mutex_wait(wait_usec);
}

void StreamingUploadPipeline::record_upload_queue_latency(uint64_t enqueue_usec) {
    const uint64_t now_usec = _ticks_usec_now();
    if (enqueue_usec == 0 || now_usec == 0 || now_usec < enqueue_usec) {
        return;
    }
    const uint64_t latency_usec = now_usec - enqueue_usec;
    telemetry.add_upload_queue_latency(latency_usec);
}

bool StreamingUploadPipeline::has_pending_uploads() {
    return get_upload_queue_depth_cached() > 0 || get_pack_queue_depth_cached() > 0;
}

bool StreamingUploadPipeline::pop_upload_job(PendingChunkUpload *&job) {
    const uint64_t lock_wait_start_usec = _ticks_usec_now();
    MutexLock lock(pack_mutex);
    record_pack_mutex_wait(lock_wait_start_usec);
    if (upload_queue_read_idx >= upload_queue.size()) {
        return false;
    }
    job = upload_queue[upload_queue_read_idx++];
    if (job) {
        record_upload_queue_latency(job->enqueue_usec);
    }
    compact_queues_locked();
    return true;
}

StreamingUploadPipeline::UploadBudgetState
StreamingUploadPipeline::prepare_upload_budget_state() {
    UploadBudgetState result;
    uint64_t upload_budget_limit = max_upload_bytes_per_frame == 0 ? UINT64_MAX : max_upload_bytes_per_frame;
    if (max_upload_bytes_per_second > 0) {
        OS *os = OS::get_singleton();
        const uint64_t now_usec = os ? os->get_ticks_usec() : 0;
        if (upload_budget_last_update_usec == 0 || now_usec == 0 || now_usec < upload_budget_last_update_usec) {
            upload_budget_last_update_usec = now_usec;
            if (upload_budget_tokens == 0) {
                upload_budget_tokens = max_upload_bytes_per_second;
            }
        } else if (now_usec > upload_budget_last_update_usec) {
            const uint64_t elapsed_usec = now_usec - upload_budget_last_update_usec;
            const uint64_t refill_bytes = uint64_t((double(max_upload_bytes_per_second) * double(elapsed_usec)) / 1000000.0);
            const uint64_t max_tokens = max_upload_bytes_per_second * 2;
            upload_budget_tokens = MIN<uint64_t>(max_tokens, upload_budget_tokens + refill_bytes);
            upload_budget_last_update_usec = now_usec;
        }

        if (upload_budget_limit == UINT64_MAX) {
            upload_budget_limit = upload_budget_tokens;
        } else {
            upload_budget_limit = MIN(upload_budget_limit, upload_budget_tokens);
        }
    }
    result.slice_limit = max_upload_bytes_per_slice == 0 ? UINT64_MAX : max_upload_bytes_per_slice;
    result.upload_budget = upload_budget_limit;
    result.completed_chunks = 0;
    result.chunk_limit = max_chunk_loads_per_frame == 0 ? UINT32_MAX : max_chunk_loads_per_frame;
    return result;
}

void StreamingUploadPipeline::requeue_upload_job(PendingChunkUpload *job) {
    const uint64_t lock_wait_start_usec = _ticks_usec_now();
    MutexLock lock(pack_mutex);
    record_pack_mutex_wait(lock_wait_start_usec);
    if (job) {
        job->enqueue_usec = _ticks_usec_now();
    }
    upload_queue.push_back(job);
    sync_cached_queue_depths_locked();
}

void StreamingUploadPipeline::cancel_chunk_jobs(
        GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx, uint32_t buffer_slot) {
    const uint64_t chunk_key = system._make_chunk_key(asset_id, chunk_idx);
    uint32_t removed_pack_jobs = 0;
    bool release_slot = false;

    {
        const uint64_t lock_wait_start_usec = _ticks_usec_now();
        MutexLock lock(pack_mutex);
        record_pack_mutex_wait(lock_wait_start_usec);
        uint32_t pack_write_idx = pack_queue_read_idx;
        for (uint32_t read_idx = pack_queue_read_idx; read_idx < pack_queue.size(); read_idx++) {
            const PackJob &job = pack_queue[read_idx];
            if (job.asset_id != asset_id || job.chunk_idx != chunk_idx) {
                if (pack_write_idx != read_idx) {
                    pack_queue[pack_write_idx] = pack_queue[read_idx];
                }
                pack_write_idx++;
                continue;
            }
            if (buffer_slot != UINT32_MAX && job.buffer_slot != buffer_slot) {
                if (pack_write_idx != read_idx) {
                    pack_queue[pack_write_idx] = pack_queue[read_idx];
                }
                pack_write_idx++;
                continue;
            }
            removed_pack_jobs++;
            release_slot = true;
        }
        pack_queue.resize(pack_write_idx);

        uint32_t upload_write_idx = upload_queue_read_idx;
        for (uint32_t read_idx = upload_queue_read_idx; read_idx < upload_queue.size(); read_idx++) {
            PendingChunkUpload *job = upload_queue[read_idx];
            if (!job || job->asset_id != asset_id || job->chunk_idx != chunk_idx) {
                if (upload_write_idx != read_idx) {
                    upload_queue[upload_write_idx] = upload_queue[read_idx];
                }
                upload_write_idx++;
                continue;
            }
            if (buffer_slot != UINT32_MAX && job->buffer_slot != buffer_slot) {
                if (upload_write_idx != read_idx) {
                    upload_queue[upload_write_idx] = upload_queue[read_idx];
                }
                upload_write_idx++;
                continue;
            }
            memdelete(job);
            release_slot = true;
        }
        upload_queue.resize(upload_write_idx);
        compact_queues_locked();
    }

    if (removed_pack_jobs > 0) {
        _atomic_saturating_sub(pack_jobs_in_flight, removed_pack_jobs);
    }

    GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
    if (asset) {
        LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
        if (chunk_idx < asset_chunks.size()) {
            GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[chunk_idx];
            const bool slot_match = (buffer_slot == UINT32_MAX || chunk.buffer_slot == buffer_slot);
            if (slot_match && chunk.upload_pending && !chunk.is_loaded) {
                release_slot = true;
                system._rollback_pending_chunk(asset_id, chunk_idx, chunk, false);
            }
            if (!chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX) {
                system._rollback_pending_chunk(asset_id, chunk_idx, chunk, true);
                release_slot = true;
            }
            if (!chunk.is_loaded && chunk.upload_pending) {
                uint32_t mapped_slot = UINT32_MAX;
                if (!_chunk_slot_matches_allocator(system.atlas_sync.allocator, chunk_key, chunk.buffer_slot, &mapped_slot)) {
                    if (mapped_slot != UINT32_MAX) {
                        system.atlas_sync.allocator.release_slot(chunk_key);
                    }
                    system._rollback_pending_chunk(asset_id, chunk_idx, chunk, false);
                }
            }
            if (!chunk.is_loaded && !chunk.upload_pending) {
                uint32_t mapped_slot = UINT32_MAX;
                if (system.atlas_sync.allocator.get_slot(chunk_key, mapped_slot)) {
                    system.atlas_sync.allocator.release_slot(chunk_key);
                }
            }
            system._assert_chunk_state_invariant(asset_id, chunk_idx, chunk, "cancel_chunk_jobs.post");
        }
    }

    if (release_slot) {
        _release_chunk_slot_if_matches(system.atlas_sync.allocator, chunk_key, buffer_slot);
    }
}

void StreamingUploadPipeline::cancel_asset_jobs(GaussianStreamingSystem &system, uint32_t asset_id) {
    struct SlotRelease {
        uint64_t chunk_key = 0;
        uint32_t slot = UINT32_MAX;
    };

    uint32_t removed_pack_jobs = 0;
    LocalVector<SlotRelease> slots_to_release;

    {
        const uint64_t lock_wait_start_usec = _ticks_usec_now();
        MutexLock lock(pack_mutex);
        record_pack_mutex_wait(lock_wait_start_usec);
        uint32_t pack_write_idx = pack_queue_read_idx;
        for (uint32_t read_idx = pack_queue_read_idx; read_idx < pack_queue.size(); read_idx++) {
            const PackJob &job = pack_queue[read_idx];
            if (job.asset_id != asset_id) {
                if (pack_write_idx != read_idx) {
                    pack_queue[pack_write_idx] = pack_queue[read_idx];
                }
                pack_write_idx++;
                continue;
            }
            SlotRelease release;
            release.chunk_key = system._make_chunk_key(job.asset_id, job.chunk_idx);
            release.slot = job.buffer_slot;
            slots_to_release.push_back(release);
            removed_pack_jobs++;
        }
        pack_queue.resize(pack_write_idx);

        uint32_t upload_write_idx = upload_queue_read_idx;
        for (uint32_t read_idx = upload_queue_read_idx; read_idx < upload_queue.size(); read_idx++) {
            PendingChunkUpload *job = upload_queue[read_idx];
            if (!job || job->asset_id != asset_id) {
                if (upload_write_idx != read_idx) {
                    upload_queue[upload_write_idx] = upload_queue[read_idx];
                }
                upload_write_idx++;
                continue;
            }
            SlotRelease release;
            release.chunk_key = system._make_chunk_key(job->asset_id, job->chunk_idx);
            release.slot = job->buffer_slot;
            slots_to_release.push_back(release);
            memdelete(job);
        }
        upload_queue.resize(upload_write_idx);
        compact_queues_locked();
    }

    if (removed_pack_jobs > 0) {
        _atomic_saturating_sub(pack_jobs_in_flight, removed_pack_jobs);
    }

    for (uint32_t i = 0; i < slots_to_release.size(); i++) {
        _release_chunk_slot_if_matches(system.atlas_sync.allocator, slots_to_release[i].chunk_key, slots_to_release[i].slot);
    }

    GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
    if (!asset) {
        return;
    }

    LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
    for (uint32_t i = 0; i < asset_chunks.size(); i++) {
        GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[i];
        const uint64_t chunk_key = system._make_chunk_key(asset_id, i);
        if (!chunk.upload_pending || chunk.is_loaded) {
            if (!chunk.is_loaded && chunk.buffer_slot != UINT32_MAX) {
                system._rollback_pending_chunk(asset_id, i, chunk, true);
            }
            if (!chunk.is_loaded && !chunk.upload_pending) {
                uint32_t mapped_slot = UINT32_MAX;
                if (system.atlas_sync.allocator.get_slot(chunk_key, mapped_slot)) {
                    system.atlas_sync.allocator.release_slot(chunk_key);
                }
            }
            system._assert_chunk_state_invariant(asset_id, i, chunk, "cancel_asset_jobs.post");
            continue;
        }
        if (!chunk.is_loaded) {
            uint32_t mapped_slot = UINT32_MAX;
            if (!_chunk_slot_matches_allocator(system.atlas_sync.allocator, chunk_key, chunk.buffer_slot, &mapped_slot)) {
                if (mapped_slot != UINT32_MAX) {
                    system.atlas_sync.allocator.release_slot(chunk_key);
                }
                system._rollback_pending_chunk(asset_id, i, chunk, false);
            }
        }
        if (chunk.upload_pending && !chunk.is_loaded) {
            system._rollback_pending_chunk(asset_id, i, chunk, true);
        }
        system._assert_chunk_state_invariant(asset_id, i, chunk, "cancel_asset_jobs.post");
    }
}

void StreamingUploadPipeline::clear_pending_uploads(GaussianStreamingSystem &system) {
    uint32_t removed_pack_jobs = 0;
    {
        const uint64_t lock_wait_start_usec = _ticks_usec_now();
        MutexLock lock(pack_mutex);
        record_pack_mutex_wait(lock_wait_start_usec);

        for (uint32_t i = pack_queue_read_idx; i < pack_queue.size(); i++) {
            const PackJob &job = pack_queue[i];
            removed_pack_jobs++;
            GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(job.asset_id);
            if (!asset) {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator,
                        system._make_chunk_key(job.asset_id, job.chunk_idx), job.buffer_slot);
                continue;
            }
            LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
            if (job.chunk_idx >= asset_chunks.size()) {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator,
                        system._make_chunk_key(job.asset_id, job.chunk_idx), job.buffer_slot);
                continue;
            }
            GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[job.chunk_idx];
            if (chunk.upload_pending && chunk.buffer_slot == job.buffer_slot && !chunk.is_loaded) {
                system._rollback_pending_chunk(job.asset_id, job.chunk_idx, chunk, true);
            } else {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator,
                        system._make_chunk_key(job.asset_id, job.chunk_idx), job.buffer_slot);
            }
        }
        pack_queue.clear();
        pack_queue_read_idx = 0;

        for (uint32_t i = upload_queue_read_idx; i < upload_queue.size(); i++) {
            PendingChunkUpload *job = upload_queue[i];
            if (!job) {
                continue;
            }
            GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(job->asset_id);
            if (!asset) {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator,
                        system._make_chunk_key(job->asset_id, job->chunk_idx), job->buffer_slot);
                memdelete(job);
                continue;
            }
            LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
            if (job->chunk_idx < asset_chunks.size()) {
                GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[job->chunk_idx];
                if (chunk.upload_pending && chunk.buffer_slot == job->buffer_slot && !chunk.is_loaded) {
                    system._rollback_pending_chunk(job->asset_id, job->chunk_idx, chunk, true);
                } else {
                    _release_chunk_slot_if_matches(system.atlas_sync.allocator,
                            system._make_chunk_key(job->asset_id, job->chunk_idx), job->buffer_slot);
                }
            } else {
                _release_chunk_slot_if_matches(system.atlas_sync.allocator,
                        system._make_chunk_key(job->asset_id, job->chunk_idx), job->buffer_slot);
            }
            memdelete(job);
        }
        upload_queue.clear();
        upload_queue_read_idx = 0;
        sync_cached_queue_depths_locked();
    }

    for (uint32_t asset_order_idx = 0; asset_order_idx < system.asset_registry.atlas_asset_order.size(); asset_order_idx++) {
        const uint32_t asset_id = system.asset_registry.atlas_asset_order[asset_order_idx];
        GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
        if (!asset) {
            continue;
        }
        LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
        for (uint32_t chunk_idx = 0; chunk_idx < asset_chunks.size(); chunk_idx++) {
            GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[chunk_idx];
            const uint64_t chunk_key = system._make_chunk_key(asset_id, chunk_idx);
            if (!chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX) {
                system._rollback_pending_chunk(asset_id, chunk_idx, chunk, true);
            }
            if (!chunk.is_loaded && chunk.upload_pending) {
                uint32_t mapped_slot = UINT32_MAX;
                if (!_chunk_slot_matches_allocator(system.atlas_sync.allocator, chunk_key, chunk.buffer_slot, &mapped_slot)) {
                    if (mapped_slot != UINT32_MAX) {
                        system.atlas_sync.allocator.release_slot(chunk_key);
                    }
                    system._rollback_pending_chunk(asset_id, chunk_idx, chunk, false);
                }
            }
            if (!chunk.is_loaded && !chunk.upload_pending) {
                uint32_t mapped_slot = UINT32_MAX;
                if (system.atlas_sync.allocator.get_slot(chunk_key, mapped_slot)) {
                    system.atlas_sync.allocator.release_slot(chunk_key);
                }
            }
            system._assert_chunk_state_invariant(asset_id, chunk_idx, chunk, "clear_pending_upload_pipeline.post");
        }
    }

    if (removed_pack_jobs > 0) {
        _atomic_saturating_sub(pack_jobs_in_flight, removed_pack_jobs);
    }

}
