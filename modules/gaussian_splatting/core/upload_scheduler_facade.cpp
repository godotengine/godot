#include "upload_scheduler_facade.h"

#include "gaussian_streaming.h"

bool UploadSchedulerFacade::enqueue_chunk_load(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx) {
    return system._internal_get_upload_state().queue_chunk_load(system, asset_id, chunk_idx);
}

void UploadSchedulerFacade::process_uploads(GaussianStreamingSystem &system) {
    system._internal_get_upload_state().process_upload_queue(system);
}

void UploadSchedulerFacade::clear_pending_uploads(GaussianStreamingSystem &system) {
    system._internal_get_upload_state().clear_pending_uploads(system);
}

uint32_t UploadSchedulerFacade::get_pending_pack_jobs(GaussianStreamingSystem &system) {
    return system._internal_get_upload_state().get_pack_queue_depth_cached();
}

uint32_t UploadSchedulerFacade::get_pending_upload_jobs(GaussianStreamingSystem &system) {
    return system._internal_get_upload_state().get_upload_queue_depth_cached();
}

void UploadSchedulerFacade::get_pending_queue_depths(
        GaussianStreamingSystem &system, uint32_t &r_pack_queue_depth, uint32_t &r_upload_queue_depth) {
    system._internal_get_upload_state().get_pending_queue_depths_cached(r_pack_queue_depth, r_upload_queue_depth);
}
