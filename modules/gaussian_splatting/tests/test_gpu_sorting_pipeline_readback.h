#pragma once

#include "test_macros.h"

#include "../interfaces/gpu_sorting_pipeline.h"
#include "../renderer/pipeline_io_contracts.h"

#include <cstring>

namespace TestGaussianSplatting {
namespace {

struct TestSortResultSink : public ISortResultSink {
	int publish_count = 0;
	SortPublicationPayload last_payload;

	void publish_sorted_indices(const SortPublicationPayload &p_payload) override {
		publish_count++;
		last_payload = p_payload;
	}
};

static Vector<uint8_t> _make_indirect_dispatch_payload(uint32_t p_element_count, uint32_t p_overflow_flag,
		uint32_t p_unclamped_total) {
	GaussianSplatting::IndirectDispatchLayout layout = {};
	layout.element_count = p_element_count;
	layout.overflow_flag = p_overflow_flag;
	layout.unclamped_total = p_unclamped_total;

	Vector<uint8_t> payload;
	payload.resize(int(sizeof(layout)));
	std::memcpy(payload.ptrw(), &layout, sizeof(layout));
	return payload;
}

static Vector<uint8_t> _make_sort_indices_payload(const Vector<uint32_t> &p_indices) {
	Vector<uint8_t> payload;
	payload.resize(int(p_indices.size() * sizeof(uint32_t)));
	if (!payload.is_empty()) {
		std::memcpy(payload.ptrw(), p_indices.ptr(), size_t(payload.size()));
	}
	return payload;
}

} // namespace

TEST_CASE("[GaussianSplatting][GPU Sort Pipeline] Stale sort readbacks are ignored after generation advances") {
	Ref<GPUSortingPipeline> pipeline;
	pipeline.instantiate();
	REQUIRE(pipeline.is_valid());

	auto &sort_state = pipeline->_test_sort_readback_state();
	sort_state.pending = true;
	sort_state.generation = 17;
	sort_state.expected_count = 4;
	sort_state.snapshot_indices.resize(4);
	sort_state.snapshot_indices.write[0] = 3;
	sort_state.snapshot_indices.write[1] = 1;
	sort_state.snapshot_indices.write[2] = 0;
	sort_state.snapshot_indices.write[3] = 2;
	TestSortResultSink sink;
	pipeline->set_sort_result_sink(&sink);

	const Vector<uint8_t> payload = _make_sort_indices_payload(sort_state.snapshot_indices);

	pipeline->shutdown();
	CHECK(sort_state.generation == 18);
	CHECK_FALSE(sort_state.pending);
	CHECK(pipeline->_test_get_sort_result_sink() == nullptr);

	sort_state.pending = true;
	sort_state.expected_count = 4;
	sort_state.snapshot_indices.resize(4);
	sort_state.snapshot_indices.write[0] = 3;
	sort_state.snapshot_indices.write[1] = 1;
	sort_state.snapshot_indices.write[2] = 0;
	sort_state.snapshot_indices.write[3] = 2;
	pipeline->set_sort_result_sink(&sink);

	pipeline->_on_sort_readback(payload, 17);
	CHECK(sort_state.pending);
	CHECK(sort_state.generation == 18);
	CHECK(sink.publish_count == 0);
	REQUIRE(sort_state.snapshot_indices.size() == 4);
	CHECK(sort_state.snapshot_indices[0] == 3);
	CHECK(sort_state.snapshot_indices[1] == 1);
	CHECK(sort_state.snapshot_indices[2] == 0);
	CHECK(sort_state.snapshot_indices[3] == 2);
}

TEST_CASE("[GaussianSplatting][GPU Sort Pipeline] Instance-count readbacks reject stale generations and accept the current one") {
	Ref<GPUSortingPipeline> pipeline;
	pipeline.instantiate();
	REQUIRE(pipeline.is_valid());

	auto &count_state = pipeline->_test_instance_count_readback_state();
	count_state.pending = true;
	count_state.generation = 33;
	count_state.pending_frame_counter = 71;
	pipeline->_test_set_last_instance_visible_splat_count_state(12, false, 0);

	const Vector<uint8_t> stale_payload = _make_indirect_dispatch_payload(27, 1, 99);
	pipeline->_on_instance_count_readback(stale_payload, 32);

	CHECK(count_state.pending);
	CHECK(count_state.generation == 33);
	CHECK(count_state.pending_frame_counter == 71);
	CHECK(pipeline->get_last_instance_visible_splat_count() == 12);
	CHECK_FALSE(pipeline->_test_get_last_instance_visible_splat_count_valid());
	CHECK(pipeline->_test_get_last_instance_visible_splat_count_frame() == 0);

	const Vector<uint8_t> current_payload = _make_indirect_dispatch_payload(27, 1, 99);
	pipeline->_on_instance_count_readback(current_payload, 33);

	CHECK_FALSE(count_state.pending);
	CHECK(count_state.generation == 33);
	CHECK(count_state.pending_frame_counter == 0);
	CHECK(pipeline->get_last_instance_visible_splat_count() == 27);
	CHECK(pipeline->_test_get_last_instance_visible_splat_count_valid());
	CHECK(pipeline->_test_get_last_instance_visible_splat_count_frame() == 71);
}

TEST_CASE("[GaussianSplatting][GPU Sort Pipeline] Clearing instance pipeline inputs resets readback ownership state") {
	Ref<GPUSortingPipeline> pipeline;
	pipeline.instantiate();
	REQUIRE(pipeline.is_valid());

	auto &count_state = pipeline->_test_instance_count_readback_state();
	count_state.pending = true;
	count_state.generation = 11;
	count_state.pending_frame_counter = 29;
	count_state.bootstrap_sync_attempted = true;
	pipeline->_test_set_last_instance_visible_splat_count_state(42, true, 17);

	pipeline->clear_instance_pipeline_inputs();

	CHECK_FALSE(count_state.pending);
	CHECK(count_state.generation == 12);
	CHECK(count_state.pending_frame_counter == 0);
	CHECK_FALSE(count_state.bootstrap_sync_attempted);
	CHECK(pipeline->get_last_instance_visible_splat_count() == 0);
	CHECK_FALSE(pipeline->_test_get_last_instance_visible_splat_count_valid());
	CHECK(pipeline->_test_get_last_instance_visible_splat_count_frame() == 0);
}

} // namespace TestGaussianSplatting
