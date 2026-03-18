#include "test_macros.h"

#define private public
#include "../renderer/tile_renderer.h"
#undef private

#include <cstring>

namespace {

static Vector<uint8_t> _make_overflow_readback_payload(uint32_t p_element_count, uint32_t p_overflow_flag,
		uint32_t p_unclamped_total) {
	uint32_t words[3] = { p_element_count, p_overflow_flag, p_unclamped_total };
	Vector<uint8_t> payload;
	payload.resize(int(sizeof(words)));
	std::memcpy(payload.ptrw(), words, sizeof(words));
	return payload;
}

static Vector<uint8_t> _make_tile_counts_payload(uint32_t p_a, uint32_t p_b) {
	uint32_t words[2] = { p_a, p_b };
	Vector<uint8_t> payload;
	payload.resize(int(sizeof(words)));
	std::memcpy(payload.ptrw(), words, sizeof(words));
	return payload;
}

} // namespace

TEST_CASE("[TileRenderer] Async overflow readback rejects stale callbacks") {
	TileRenderer renderer;
	auto &state = renderer.async_readback.overflow_state;

	state.pending_readback = true;
	state.requested_frame_serial = 42;
	state.overflow_detected = false;
	state.last_unclamped_total = 64;
	state.first_frame_complete = true;

	const Vector<uint8_t> payload = _make_overflow_readback_payload(9, 1, 2048);

	renderer._on_overflow_flag_readback(payload, 41);
	CHECK(state.pending_readback);
	CHECK(state.requested_frame_serial == 42);
	CHECK_FALSE(state.overflow_detected);
	CHECK(state.last_unclamped_total == 64);

	renderer._on_overflow_flag_readback(payload, 42);
	CHECK_FALSE(state.pending_readback);
	CHECK(state.requested_frame_serial == 0);
	CHECK(state.overflow_detected);
	CHECK(state.last_unclamped_total == 2048);
}

TEST_CASE("[TileRenderer] Async tile-count readback rejects stale callbacks") {
	TileRenderer renderer;
	auto &state = renderer.async_readback.tile_counts_state;

	state.pending_readback = true;
	state.requested_frame_serial = 7;
	state.cached_total_tiles = 2;
	state.cached_counts.clear();
	state.first_frame_complete = false;

	const Vector<uint8_t> payload = _make_tile_counts_payload(5, 9);

	renderer._on_tile_counts_readback(payload, 6);
	CHECK(state.pending_readback);
	CHECK(state.requested_frame_serial == 7);
	CHECK_FALSE(state.first_frame_complete);
	CHECK(state.cached_counts.is_empty());

	renderer._on_tile_counts_readback(payload, 7);
	REQUIRE_FALSE(state.pending_readback);
	CHECK(state.requested_frame_serial == 0);
	CHECK(state.first_frame_complete);
	REQUIRE(state.cached_counts.size() == 2);
	CHECK(state.cached_counts[0] == 5);
	CHECK(state.cached_counts[1] == 9);
}
