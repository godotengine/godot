#include "batched_async_readback.h"
#include "../logger/gs_logger.h"
#include "../interfaces/sync_policy.h"

// PERF (#633, #634): Implementation of batched async buffer readback
// Consolidates multiple GPU->CPU transfers into single staging buffer operation

void BatchedAsyncReadback::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_initialized"), &BatchedAsyncReadback::is_initialized);
	ClassDB::bind_method(D_METHOD("get_pending_request_count"), &BatchedAsyncReadback::get_pending_request_count);
	ClassDB::bind_method(D_METHOD("get_staging_buffer_usage"), &BatchedAsyncReadback::get_staging_buffer_usage);
	ClassDB::bind_method(D_METHOD("get_total_failed_requests"), &BatchedAsyncReadback::get_total_failed_requests);
}

BatchedAsyncReadback::BatchedAsyncReadback() {
	pending_requests.reserve(MAX_READBACK_SLOTS);
	submitted_requests.reserve(MAX_READBACK_SLOTS);
}

BatchedAsyncReadback::~BatchedAsyncReadback() {
	shutdown();
}

Error BatchedAsyncReadback::initialize(RenderingDevice *p_rd, uint32_t p_staging_buffer_size) {
	if (initialized) {
		return OK;
	}

	if (!p_rd) {
		GS_LOG_WARN_DEFAULT("[BatchedAsyncReadback] No RenderingDevice provided");
		return ERR_INVALID_PARAMETER;
	}

	rd = p_rd;
	staging_buffer_size = p_staging_buffer_size;

	// Create staging buffer for consolidated readback
	Vector<uint8_t> zero_data;
	zero_data.resize(staging_buffer_size);
	staging_buffer = rd->storage_buffer_create(staging_buffer_size, zero_data);
	if (!staging_buffer.is_valid()) {
		GS_LOG_WARN_DEFAULT("[BatchedAsyncReadback] Failed to create staging buffer");
		return ERR_CANT_CREATE;
	}
	rd->set_resource_name(staging_buffer, "GS_BatchedReadbackStaging");

	initialized = true;
	GS_LOG_WARN_DEFAULT(vformat("[BatchedAsyncReadback] Initialized with %d KB staging buffer", staging_buffer_size / 1024));
	return OK;
}

void BatchedAsyncReadback::shutdown() {
	cancel_batch();

	if (staging_buffer.is_valid() && rd) {
		rd->free(staging_buffer);
	}
	staging_buffer = RID();

	pending_requests.clear();
	submitted_requests.clear();
	staging_offset = 0;
	batch_state = BATCH_IDLE;
	initialized = false;
	rd = nullptr;
	total_batches_completed = 0;
	total_failed_requests = 0;
	last_polled_batches_completed = 0;
}

bool BatchedAsyncReadback::add_request(RID p_source_buffer, uint32_t p_offset, uint32_t p_size,
		const Callable &p_callback, int64_t p_user_data) {
	if (!initialized) {
		GS_LOG_WARN_DEFAULT("[BatchedAsyncReadback] Not initialized");
		return false;
	}

	if (batch_state == BATCH_PENDING) {
		GS_LOG_WARN_DEFAULT("[BatchedAsyncReadback] Cannot add request while batch is pending");
		return false;
	}

	if (pending_requests.size() >= MAX_READBACK_SLOTS) {
		GS_LOG_WARN_DEFAULT("[BatchedAsyncReadback] Batch is full");
		return false;
	}

	// Align to 16 bytes for GPU compatibility
	uint32_t aligned_offset = (staging_offset + 15) & ~15;
	if (aligned_offset + p_size > staging_buffer_size) {
		GS_LOG_WARN_DEFAULT(vformat("[BatchedAsyncReadback] Staging buffer capacity exceeded: need %d, have %d",
				aligned_offset + p_size, staging_buffer_size));
		return false;
	}

	ReadbackRequest request;
	request.source_buffer = p_source_buffer;
	request.offset = p_offset;
	request.size = p_size;
	request.staging_offset = aligned_offset;
	request.callback = p_callback;
	request.user_data = p_user_data;

	pending_requests.push_back(request);
	staging_offset = aligned_offset + p_size;

	return true;
}

bool BatchedAsyncReadback::submit_batch() {
	if (!initialized || pending_requests.is_empty()) {
		return false;
	}

	if (batch_state == BATCH_PENDING) {
		GS_LOG_WARN_DEFAULT("[BatchedAsyncReadback] Batch already pending");
		return false;
	}

	// Copy source buffers to staging buffer. On per-request failure, log the specific
	// index and mark it failed, but continue with remaining requests so partial
	// batches still deliver data for the successful copies.
	uint32_t batch_size = pending_requests.size();
	uint32_t failed_in_batch = 0;
	for (uint32_t i = 0; i < batch_size; i++) {
		ReadbackRequest &request = pending_requests[i];
		if (!request.source_buffer.is_valid()) {
			WARN_PRINT(vformat("Async readback: request %d/%d failed — invalid source buffer", i, batch_size));
			request.callback = Callable(); // Invalidate so callback is skipped during dispatch
			failed_in_batch++;
			continue;
		}
		Error err = rd->buffer_copy(request.source_buffer, staging_buffer,
				request.offset, request.staging_offset, request.size);
		if (err != OK) {
			WARN_PRINT(vformat("Async readback: request %d/%d failed — buffer_copy error %d", i, batch_size, err));
			request.callback = Callable(); // Invalidate so callback is skipped during dispatch
			failed_in_batch++;
			continue;
		}
	}
	total_failed_requests += failed_in_batch;
	// If every request in the batch failed, cancel rather than submit an empty GPU op.
	if (failed_in_batch == batch_size) {
		cancel_batch();
		return false;
	}

	// Submit async readback for the staging buffer
	Callable callback = callable_mp(this, &BatchedAsyncReadback::_on_batch_readback);
	Error err = rd->buffer_get_data_async(staging_buffer, callback, 0, staging_offset);
	if (err != OK) {
		GS_LOG_WARN_DEFAULT(vformat("[BatchedAsyncReadback] Async readback submission failed: %d", err));
		cancel_batch();
		return false;
	}

	// Move to submitted state
	submitted_requests = pending_requests;
	pending_requests.clear();
	batch_state = BATCH_PENDING;
	staging_offset = 0;
	total_batches_submitted++;

	return true;
}

void BatchedAsyncReadback::_on_batch_readback(const Vector<uint8_t> &p_data) {
	if (batch_state != BATCH_PENDING) {
		return;
	}

	batch_state = BATCH_COMPLETE;

	// Dispatch callbacks with their slice of data, verifying bounds for each request.
	for (uint32_t i = 0; i < submitted_requests.size(); i++) {
		const ReadbackRequest &request = submitted_requests[i];
		if (!request.callback.is_valid()) {
			continue;
		}

		// Verify data bounds before invoking callback to guard against
		// truncated readbacks or requests that failed during copy.
		if (request.staging_offset + request.size > (uint32_t)p_data.size()) {
			WARN_PRINT(vformat("Async readback: callback %d/%d skipped — data bounds exceeded (offset=%d size=%d data=%d)",
					i, (uint32_t)submitted_requests.size(),
					request.staging_offset, request.size, (uint32_t)p_data.size()));
			total_failed_requests++;
			continue;
		}

		// Extract the slice of data for this request
		Vector<uint8_t> slice;
		slice.resize(request.size);
		memcpy(slice.ptrw(), p_data.ptr() + request.staging_offset, request.size);

		// Call the user callback
		Variant args[2] = { slice, request.user_data };
		const Variant *argp[2] = { &args[0], &args[1] };
		Callable::CallError ce;
		Variant ret;
		request.callback.callp(argp, 2, ret, ce);
		total_requests_processed++;
	}

	submitted_requests.clear();
	total_batches_completed++;
	batch_state = BATCH_IDLE;
}

bool BatchedAsyncReadback::poll_and_dispatch() {
	// In Godot's async model, the callback is called automatically
	// This method is for compatibility with polling-based usage
	if (batch_state != BATCH_IDLE) {
		return false;
	}

	if (total_batches_completed == last_polled_batches_completed) {
		return false;
	}

	last_polled_batches_completed = total_batches_completed;
	return true;
}

void BatchedAsyncReadback::wait_for_completion() {
	if (!initialized || batch_state != BATCH_PENDING) {
		return;
	}

	// Force synchronous completion
	if (rd) {
		gs_device_utils::safe_submit(rd);
		gs_device_utils::safe_sync(rd);
	}

	// The callback should have been called by now
	if (batch_state == BATCH_PENDING) {
		// Fallback: manually retrieve data
		Vector<uint8_t> data = rd->buffer_get_data(staging_buffer, 0, staging_offset > 0 ? staging_offset : staging_buffer_size);
		_on_batch_readback(data);
	}
}

void BatchedAsyncReadback::cancel_batch() {
	if (batch_state == BATCH_PENDING) {
		// Can't truly cancel GPU work, but we can ignore the results
		submitted_requests.clear();
	}
	pending_requests.clear();
	staging_offset = 0;
	batch_state = BATCH_IDLE;
}

float BatchedAsyncReadback::get_average_batch_size() const {
	if (total_batches_submitted == 0) {
		return 0.0f;
	}
	return static_cast<float>(total_requests_processed) / static_cast<float>(total_batches_submitted);
}
