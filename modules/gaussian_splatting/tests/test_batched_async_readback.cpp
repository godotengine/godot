#include "test_macros.h"

#include "../renderer/batched_async_readback.h"

TEST_CASE("[GaussianSplatting][AsyncReadback] Batched readback tolerates per-request failures") {
	REQUIRE_GPU_DEVICE();

	Ref<BatchedAsyncReadback> batched_readback;
	batched_readback.instantiate();
	REQUIRE(batched_readback.is_valid());

	CHECK(batched_readback->initialize(rd, 4096) == OK);

	Vector<uint8_t> payload;
	payload.resize(16);
	uint8_t *payload_ptr = payload.ptrw();
	for (int i = 0; i < payload.size(); i++) {
		payload_ptr[i] = static_cast<uint8_t>(i);
	}

	const RID valid_source = rd->storage_buffer_create(payload.size(), payload);
	REQUIRE(valid_source.is_valid());

	CHECK(batched_readback->add_request(RID(), 0, 16, Callable(), 1));
	CHECK(batched_readback->add_request(valid_source, 0, 16, Callable(), 2));
	CHECK(batched_readback->submit_batch());

	batched_readback->wait_for_completion();

	CHECK_EQ(batched_readback->get_total_batches_submitted(), 1u);
	CHECK_EQ(batched_readback->get_total_batches_completed(), 1u);
	CHECK_EQ(batched_readback->get_total_failed_requests(), 1u);
	CHECK_EQ(batched_readback->get_pending_request_count(), 0u);
	CHECK_EQ(batched_readback->get_state(), BatchedAsyncReadback::BATCH_IDLE);

	rd->free(valid_source);
	batched_readback->shutdown();
}

TEST_CASE("[GaussianSplatting][AsyncReadback] Batched readback rejects fully failed batches cleanly") {
	REQUIRE_GPU_DEVICE();

	Ref<BatchedAsyncReadback> batched_readback;
	batched_readback.instantiate();
	REQUIRE(batched_readback.is_valid());

	CHECK(batched_readback->initialize(rd, 1024) == OK);
	CHECK(batched_readback->add_request(RID(), 0, 16, Callable(), 7));
	CHECK_FALSE(batched_readback->submit_batch());

	CHECK_EQ(batched_readback->get_total_batches_submitted(), 0u);
	CHECK_EQ(batched_readback->get_total_batches_completed(), 0u);
	CHECK_EQ(batched_readback->get_total_failed_requests(), 1u);
	CHECK_EQ(batched_readback->get_pending_request_count(), 0u);
	CHECK_EQ(batched_readback->get_state(), BatchedAsyncReadback::BATCH_IDLE);

	batched_readback->shutdown();
}
