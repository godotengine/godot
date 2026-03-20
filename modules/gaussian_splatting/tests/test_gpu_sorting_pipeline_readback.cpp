#include "test_macros.h"

#define private public
#include "../interfaces/gpu_sorting_pipeline.h"
#undef private

#include "../renderer/pipeline_io_contracts.h"

#include <cstring>

namespace {

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

	pipeline->sort_readback_state.pending = true;
	pipeline->sort_readback_state.generation = 17;
	pipeline->sort_readback_state.expected_count = 4;
	pipeline->sort_readback_state.snapshot_indices.resize(4);
	pipeline->sort_readback_state.snapshot_indices.write[0] = 3;
	pipeline->sort_readback_state.snapshot_indices.write[1] = 1;
	pipeline->sort_readback_state.snapshot_indices.write[2] = 0;
	pipeline->sort_readback_state.snapshot_indices.write[3] = 2;
	pipeline->pending_renderer = reinterpret_cast<GaussianSplatRenderer *>(uintptr_t(0x1));

	const Vector<uint8_t> payload = _make_sort_indices_payload(pipeline->sort_readback_state.snapshot_indices);

	pipeline->shutdown();
	CHECK(pipeline->sort_readback_state.generation == 18);
	CHECK_FALSE(pipeline->sort_readback_state.pending);
	CHECK(pipeline->pending_renderer == nullptr);

	// Re-arm only the minimal test state needed to verify stale callbacks are ignored.
	pipeline->sort_readback_state.pending = true;
	pipeline->sort_readback_state.expected_count = 4;
	pipeline->sort_readback_state.snapshot_indices.resize(4);
	pipeline->sort_readback_state.snapshot_indices.write[0] = 3;
	pipeline->sort_readback_state.snapshot_indices.write[1] = 1;
	pipeline->sort_readback_state.snapshot_indices.write[2] = 0;
	pipeline->sort_readback_state.snapshot_indices.write[3] = 2;
	pipeline->pending_renderer = reinterpret_cast<GaussianSplatRenderer *>(uintptr_t(0x1));

	pipeline->_on_sort_readback(payload, 17);
	CHECK(pipeline->sort_readback_state.pending);
	CHECK(pipeline->sort_readback_state.generation == 18);
	REQUIRE(pipeline->sort_readback_state.snapshot_indices.size() == 4);
	CHECK(pipeline->sort_readback_state.snapshot_indices[0] == 3);
	CHECK(pipeline->sort_readback_state.snapshot_indices[1] == 1);
	CHECK(pipeline->sort_readback_state.snapshot_indices[2] == 0);
	CHECK(pipeline->sort_readback_state.snapshot_indices[3] == 2);
}

TEST_CASE("[GaussianSplatting][GPU Sort Pipeline] Instance-count readbacks reject stale generations and accept the current one") {
	Ref<GPUSortingPipeline> pipeline;
	pipeline.instantiate();
	REQUIRE(pipeline.is_valid());

	pipeline->instance_count_readback_state.pending = true;
	pipeline->instance_count_readback_state.generation = 33;
	pipeline->instance_count_readback_state.pending_frame_counter = 71;
	pipeline->last_instance_visible_splat_count = 12;
	pipeline->last_instance_visible_splat_count_valid = false;
	pipeline->last_instance_visible_splat_count_frame = 0;

	const Vector<uint8_t> stale_payload = _make_indirect_dispatch_payload(27, 1, 99);
	pipeline->_on_instance_count_readback(stale_payload, 32);

	CHECK(pipeline->instance_count_readback_state.pending);
	CHECK(pipeline->instance_count_readback_state.generation == 33);
	CHECK(pipeline->instance_count_readback_state.pending_frame_counter == 71);
	CHECK(pipeline->last_instance_visible_splat_count == 12);
	CHECK_FALSE(pipeline->last_instance_visible_splat_count_valid);
	CHECK(pipeline->last_instance_visible_splat_count_frame == 0);

	const Vector<uint8_t> current_payload = _make_indirect_dispatch_payload(27, 1, 99);
	pipeline->_on_instance_count_readback(current_payload, 33);

	CHECK_FALSE(pipeline->instance_count_readback_state.pending);
	CHECK(pipeline->instance_count_readback_state.generation == 33);
	CHECK(pipeline->instance_count_readback_state.pending_frame_counter == 0);
	CHECK(pipeline->last_instance_visible_splat_count == 27);
	CHECK(pipeline->last_instance_visible_splat_count_valid);
	CHECK(pipeline->last_instance_visible_splat_count_frame == 71);
}
