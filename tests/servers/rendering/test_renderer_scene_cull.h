/**************************************************************************/
/*  test_renderer_scene_cull.h                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "servers/rendering/renderer_scene_cull.h"

#include "tests/test_macros.h"

namespace TestRendererSceneCull {

TEST_CASE("[RendererSceneCull] Hidden indexing policy gates Gaussian exemption") {
	CHECK(RendererSceneCull::allows_hidden_visibility_indexing(
			RendererSceneCull::HIDDEN_VISIBILITY_INDEXING_ALLOW_GAUSSIAN_SPLATS,
			RS::INSTANCE_GAUSSIAN_SPLAT));
	CHECK_FALSE(RendererSceneCull::allows_hidden_visibility_indexing(
			RendererSceneCull::HIDDEN_VISIBILITY_INDEXING_ALLOW_GAUSSIAN_SPLATS,
			RS::INSTANCE_MESH));
	CHECK_FALSE(RendererSceneCull::allows_hidden_visibility_indexing(
			RendererSceneCull::HIDDEN_VISIBILITY_INDEXING_DISABLED,
			RS::INSTANCE_GAUSSIAN_SPLAT));
	CHECK(RendererSceneCull::should_skip_visibility_indexing(false, true, 1.0f, true));
	CHECK(RendererSceneCull::should_skip_visibility_indexing(true, true, 0.0f, true));
}

TEST_CASE("[RendererSceneCull] Hidden-visible transitions stay deterministic across threaded merge/reset") {
	PagedArrayPool<RID> rid_pool;
	PagedArrayPool<RenderGeometryInstance *> geometry_pool;
	PagedArrayPool<RendererSceneCull::Instance *> instance_pool;

	RendererSceneCull::InstanceCullResult merged_result;
	RendererSceneCull::InstanceCullResult thread_results[2];

	merged_result.init(&rid_pool, &geometry_pool, &instance_pool);
	thread_results[0].init(&rid_pool, &geometry_pool, &instance_pool);
	thread_results[1].init(&rid_pool, &geometry_pool, &instance_pool);

	auto simulate_frame = [&](bool p_gaussian_visible, bool p_non_gaussian_visible, bool p_allow_hidden_gaussian, uint64_t p_frame_id) {
		merged_result.clear();
		thread_results[0].clear();
		thread_results[1].clear();

		if (!RendererSceneCull::should_skip_visibility_indexing(true, p_gaussian_visible, 1.0f, p_allow_hidden_gaussian)) {
			thread_results[0].gaussian_splats.push_back(RID::from_uint64(1000 + p_frame_id));
		}
		if (!RendererSceneCull::should_skip_visibility_indexing(true, p_non_gaussian_visible, 1.0f, false)) {
			thread_results[1].fog_volumes.push_back(RID::from_uint64(2000 + p_frame_id));
		}

		merged_result.append_from(thread_results[0]);
		merged_result.append_from(thread_results[1]);
	};

	SUBCASE("Policy enabled keeps Gaussian indexed while hidden") {
		const bool allow_hidden_gaussian = RendererSceneCull::allows_hidden_visibility_indexing(
				RendererSceneCull::HIDDEN_VISIBILITY_INDEXING_ALLOW_GAUSSIAN_SPLATS,
				RS::INSTANCE_GAUSSIAN_SPLAT);

		simulate_frame(false, false, allow_hidden_gaussian, 1);
		CHECK_EQ(merged_result.gaussian_splats.size(), 1);
		CHECK(merged_result.gaussian_splats[0] == RID::from_uint64(1001));
		CHECK_EQ(merged_result.fog_volumes.size(), 0);

		simulate_frame(true, true, allow_hidden_gaussian, 2);
		CHECK_EQ(merged_result.gaussian_splats.size(), 1);
		CHECK(merged_result.gaussian_splats[0] == RID::from_uint64(1002));
		CHECK_EQ(merged_result.fog_volumes.size(), 1);
		CHECK(merged_result.fog_volumes[0] == RID::from_uint64(2002));

		simulate_frame(false, false, allow_hidden_gaussian, 3);
		CHECK_EQ(merged_result.gaussian_splats.size(), 1);
		CHECK(merged_result.gaussian_splats[0] == RID::from_uint64(1003));
		CHECK_EQ(merged_result.fog_volumes.size(), 0);
	}

	SUBCASE("Policy disabled matches non-Gaussian hidden behavior") {
		const bool allow_hidden_gaussian = RendererSceneCull::allows_hidden_visibility_indexing(
				RendererSceneCull::HIDDEN_VISIBILITY_INDEXING_DISABLED,
				RS::INSTANCE_GAUSSIAN_SPLAT);

		simulate_frame(false, false, allow_hidden_gaussian, 4);
		CHECK_EQ(merged_result.gaussian_splats.size(), 0);
		CHECK_EQ(merged_result.fog_volumes.size(), 0);

		simulate_frame(true, true, allow_hidden_gaussian, 5);
		CHECK_EQ(merged_result.gaussian_splats.size(), 1);
		CHECK(merged_result.gaussian_splats[0] == RID::from_uint64(1005));
		CHECK_EQ(merged_result.fog_volumes.size(), 1);
		CHECK(merged_result.fog_volumes[0] == RID::from_uint64(2005));

		simulate_frame(false, false, allow_hidden_gaussian, 6);
		CHECK_EQ(merged_result.gaussian_splats.size(), 0);
		CHECK_EQ(merged_result.fog_volumes.size(), 0);
	}

	merged_result.reset();
	thread_results[0].reset();
	thread_results[1].reset();
	rid_pool.reset();
	geometry_pool.reset();
	instance_pool.reset();
}

} // namespace TestRendererSceneCull
