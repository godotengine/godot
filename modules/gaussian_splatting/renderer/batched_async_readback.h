#ifndef BATCHED_ASYNC_READBACK_H
#define BATCHED_ASYNC_READBACK_H

// PERF (#633, #634): Batched async buffer readback to reduce CPU/GPU sync points
// This class consolidates multiple buffer readback requests into a single operation,
// using a single staging buffer to minimize synchronization overhead.

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"

class BatchedAsyncReadback : public RefCounted {
	GDCLASS(BatchedAsyncReadback, RefCounted);

public:
	// Maximum number of readback slots per batch
	static constexpr uint32_t MAX_READBACK_SLOTS = 8;

	// Readback request descriptor
	struct ReadbackRequest {
		RID source_buffer;           // GPU buffer to read from
		uint32_t offset = 0;         // Byte offset in source buffer
		uint32_t size = 0;           // Byte size to read
		uint32_t staging_offset = 0; // Offset in staging buffer (computed internally)
		Callable callback;           // Called when data is ready
		int64_t user_data = 0;       // User-defined data passed to callback
	};

	// Batch state
	enum BatchState {
		BATCH_IDLE = 0,      // No pending batch
		BATCH_PENDING = 1,   // Batch submitted, awaiting GPU completion
		BATCH_COMPLETE = 2,  // GPU work done, data ready for dispatch
	};

	BatchedAsyncReadback();
	~BatchedAsyncReadback() override;

	// Lifecycle
	Error initialize(RenderingDevice *p_rd, uint32_t p_staging_buffer_size = 1024 * 1024);
	void shutdown();
	bool is_initialized() const { return initialized && rd != nullptr; }

	// Add a readback request to the current batch
	// Returns false if batch is full or size exceeds remaining capacity
	bool add_request(RID p_source_buffer, uint32_t p_offset, uint32_t p_size,
			const Callable &p_callback, int64_t p_user_data = 0);

	// Submit the current batch for async readback
	// Returns false if no requests pending or already in-flight
	bool submit_batch();

	// Poll for completion and dispatch callbacks
	// Returns true if batch completed and callbacks were dispatched
	bool poll_and_dispatch();

	// Force synchronous wait for current batch
	void wait_for_completion();

	// Cancel pending batch (no callbacks will be called)
	void cancel_batch();

	// Query state
	BatchState get_state() const { return batch_state; }
	uint32_t get_pending_request_count() const { return static_cast<uint32_t>(pending_requests.size()); }
	uint32_t get_staging_buffer_usage() const { return staging_offset; }
	uint32_t get_staging_buffer_capacity() const { return staging_buffer_size; }

	// Statistics
	uint32_t get_total_batches_submitted() const { return total_batches_submitted; }
	uint32_t get_total_requests_processed() const { return total_requests_processed; }
	uint32_t get_total_failed_requests() const { return total_failed_requests; }
	float get_average_batch_size() const;

protected:
	static void _bind_methods();

private:
	RenderingDevice *rd = nullptr;
	bool initialized = false;

	// Staging buffer for consolidated readback
	RID staging_buffer;
	uint32_t staging_buffer_size = 0;
	uint32_t staging_offset = 0;  // Current write offset

	// Pending requests in current batch
	LocalVector<ReadbackRequest> pending_requests;

	// Submitted batch state
	BatchState batch_state = BATCH_IDLE;
	LocalVector<ReadbackRequest> submitted_requests;
	uint64_t batch_fence_value = 0;

	// Statistics
	uint32_t total_batches_submitted = 0;
	uint32_t total_batches_completed = 0;
	uint32_t total_requests_processed = 0;
	uint32_t total_failed_requests = 0;
	uint32_t last_polled_batches_completed = 0;

	// Callback for async readback
	void _on_batch_readback(const Vector<uint8_t> &p_data);
};

#endif // BATCHED_ASYNC_READBACK_H
