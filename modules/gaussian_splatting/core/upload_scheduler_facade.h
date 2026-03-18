#ifndef UPLOAD_SCHEDULER_FACADE_H
#define UPLOAD_SCHEDULER_FACADE_H

#include <cstdint>

class GaussianStreamingSystem;

class UploadSchedulerFacade {
public:
    static bool enqueue_chunk_load(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx);
    static void process_uploads(GaussianStreamingSystem &system);
    static void clear_pending_uploads(GaussianStreamingSystem &system);
    static uint32_t get_pending_pack_jobs(GaussianStreamingSystem &system);
    static uint32_t get_pending_upload_jobs(GaussianStreamingSystem &system);
    static void get_pending_queue_depths(
            GaussianStreamingSystem &system, uint32_t &r_pack_queue_depth, uint32_t &r_upload_queue_depth);
};

#endif // UPLOAD_SCHEDULER_FACADE_H
