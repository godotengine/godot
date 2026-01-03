/**************************************************************************/
/*  texture_streaming.cpp                                                 */
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

#include "texture_streaming.h"

#include "core/config/project_settings.h"
#include "core/templates/sort_array.h"
#include "servers/rendering/rendering_server.h"

#include <sys/types.h>

// Uncomment to enable function-level timing instrumentation for debugging.
// This prints slow function execution times and phase breakdowns.
// #define TEXTURE_STREAMING_FUNCTION_TIMING

// Helper macros for code outside of the rendering server, but that is
// called by the rendering server.
#ifdef DEBUG_ENABLED
#define ERR_NOT_ON_RENDER_THREAD                                          \
	RenderingServer *rendering_server = RenderingServer::get_singleton(); \
	ERR_FAIL_NULL(rendering_server);                                      \
	ERR_FAIL_COND(!rendering_server->is_on_render_thread());
#define ERR_NOT_ON_RENDER_THREAD_V(m_ret)                                 \
	RenderingServer *rendering_server = RenderingServer::get_singleton(); \
	ERR_FAIL_NULL_V(rendering_server, m_ret);                             \
	ERR_FAIL_COND_V(!rendering_server->is_on_render_thread(), m_ret);
#else
#define ERR_NOT_ON_RENDER_THREAD
#define ERR_NOT_ON_RENDER_THREAD_V(m_ret)
#endif

TextureStreaming *TextureStreaming::singleton = nullptr;

void TextureStreaming::MaterialFeedbackBuffer::clear() {
	ERR_NOT_ON_RENDER_THREAD;
	rid_map.clear();
	RD::get_singleton()->buffer_clear(buffer, 0, buffer_size);
}

void TextureStreaming::MaterialFeedbackBuffer::resize() {
	uint32_t material_info_count = TextureStreaming::get_singleton()->material_info_owner.get_count();
	uint32_t material_info_count_bytes = material_info_count * 4;
	if (buffer_size >= material_info_count_bytes && material_info_count_bytes > 0) {
		return;
	}

	// Free the old buffer before creating a new one to avoid RID leaks.
	if (buffer.is_valid()) {
		RD::get_singleton()->free_rid(buffer);
	}

	buffer_size = nearest_power_of_2_templated(MAX(4096u, material_info_count_bytes));
	buffer = RD::get_singleton()->storage_buffer_create(buffer_size * sizeof(uint32_t), Vector<uint8_t>(), 0, RD::BufferCreationBits::BUFFER_CREATION_AS_STORAGE_BIT);

	// Make the vector match the size of the buffer
	rid_map.resize(buffer_size / 4);
}

TextureStreaming::MaterialFeedbackBuffer::MaterialFeedbackBuffer() {}

TextureStreaming::MaterialFeedbackBuffer::~MaterialFeedbackBuffer() {
	if (buffer.is_valid()) {
		RS::get_singleton()->call_on_render_thread(callable_mp(RD::get_singleton(), &RD::free_rid).bind(buffer));
	}

	buffer = RID();
	buffer_size = 0;
	rid_map.clear();
}

TextureStreaming *TextureStreaming::get_singleton() {
	return singleton;
}

uint64_t TextureStreaming::get_memory_budget_bytes_used() {
	return texture_streaming_total_memory.load();
}

void TextureStreaming::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_budget_enabled"), &TextureStreaming::get_budget_enabled);
	ClassDB::bind_method(D_METHOD("set_budget_enabled", "enabled"), &TextureStreaming::set_budget_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "budget_enabled", PROPERTY_HINT_NONE), "set_budget_enabled", "get_budget_enabled");

	ClassDB::bind_method(D_METHOD("get_min_resolution"), &TextureStreaming::get_streaming_min_resolution);
	ClassDB::bind_method(D_METHOD("set_min_resolution", "resolution"), &TextureStreaming::set_streaming_min_resolution);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "min_resolution", PROPERTY_HINT_ENUM, "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), "set_min_resolution", "get_min_resolution");

	ClassDB::bind_method(D_METHOD("get_max_resolution"), &TextureStreaming::get_streaming_max_resolution);
	ClassDB::bind_method(D_METHOD("set_max_resolution", "resolution"), &TextureStreaming::set_streaming_max_resolution);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_resolution", PROPERTY_HINT_ENUM, "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), "set_max_resolution", "get_max_resolution");

	ClassDB::bind_method(D_METHOD("get_memory_budget_mb"), &TextureStreaming::get_memory_budget_mb);
	ClassDB::bind_method(D_METHOD("set_memory_budget_mb", "mb"), &TextureStreaming::set_memory_budget_mb);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "memory_budget_mb", PROPERTY_HINT_RANGE, "0,65535,1"), "set_memory_budget_mb", "get_memory_budget_mb");

	ClassDB::bind_method(D_METHOD("get_memory_budget_bytes_used"), &TextureStreaming::get_memory_budget_bytes_used);
}

RID TextureStreaming::texture_configure_streaming(RID p_texture, Image::Format p_format, int p_width, int p_height, int p_min_resolution, int p_max_resolution, Callable p_reload_callable) {
	ERR_FAIL_COND_V(p_texture.is_null(), RID());
	ERR_FAIL_COND_V(p_width == 0, RID());
	ERR_FAIL_COND_V(p_height == 0, RID());
	ERR_FAIL_COND_V(!p_reload_callable.is_valid(), RID());

	return push_and_wait<RID>("_texture_configure_streaming_impl", this, &TextureStreaming::_texture_configure_streaming_impl,
			p_texture, p_format, p_width, p_height, p_min_resolution, p_max_resolution, p_reload_callable);
}

void TextureStreaming::_texture_configure_streaming_impl(Completion<RID> *p_completion, RID p_texture, Image::Format p_format, int p_width, int p_height, int p_min_resolution, int p_max_resolution, const Callable &p_reload_callable) {
	print_verbose(vformat("TextureStreaming: Configuring streaming for texture RID(%d) size(%dx%d) format(%d) min_res(%d) max_res(%d)",
			p_texture.get_id(), p_width, p_height, int(p_format), p_min_resolution, p_max_resolution));

	RID rid = streaming_info_owner.allocate_rid();
	StreamingState state;
	state.texture = p_texture;
	state.format = p_format;
	state.width = p_width;
	state.height = p_height;
	state.min_resolution = CLAMP(p_min_resolution, 0, 8192);
	state.max_resolution = CLAMP(p_max_resolution, 0, 8192);
	state.reload_callable = p_reload_callable;
	streaming_info_owner.initialize_rid(rid, state);

	RS::get_singleton()->texture_2d_attach_streaming_state(p_texture, rid);

	p_completion->complete(rid);
}

void TextureStreaming::texture_remove(RID p_rid) {
	push_and_sync("_texture_remove_impl", this, &TextureStreaming::_texture_remove_impl, p_rid);
}

void TextureStreaming::_texture_remove_impl(CompletionVoid *p_completion, RID p_rid) {
	StreamingState *state = streaming_info_owner.get_or_null(p_rid);
	if (state) {
		RS::get_singleton()->texture_2d_attach_streaming_state(state->texture, RID());
		streaming_info_owner.free(p_rid);
	}
	p_completion->complete();
}

void TextureStreaming::texture_update(RID p_rid, int p_width, int p_height, int p_min_resolution, int p_max_resolution) {
	push_and_sync("_texture_update_impl", this, &TextureStreaming::_texture_update_impl, p_rid, p_width, p_height, p_min_resolution, p_max_resolution);
}

void TextureStreaming::_texture_update_impl(CompletionVoid *p_completion, RID p_rid, int p_width, int p_height, int p_min_resolution, int p_max_resolution) {
	StreamingState *state = streaming_info_owner.get_or_null(p_rid);
	if (state) {
		state->width = p_width;
		state->height = p_height;
		state->min_resolution = CLAMP(p_min_resolution, 0, 8192);
		state->max_resolution = CLAMP(p_max_resolution, 0, 8192);
	}
	p_completion->complete();
}

RID TextureStreaming::material_set_textures(RID p_feedback_rid, const Vector<RID> &p_textures) {
	ERR_NOT_ON_RENDER_THREAD_V(RID());

	MaterialInfo *info = nullptr;
	if (p_feedback_rid.is_null()) {
		info = material_info_owner.allocate(p_feedback_rid);
	} else {
		info = material_info_owner.get_or_null(p_feedback_rid);
	}

	ERR_FAIL_NULL_V(info, RID());

	if (p_textures.is_empty()) {
		material_info_owner.free(p_feedback_rid);
		return RID();
	}

	info->textures = p_textures;

	return p_feedback_rid;
}

TextureStreaming::TextureStreaming() {
	singleton = this;

	// Define global settings with defaults.
	GLOBAL_DEF_RST("rendering/textures/streaming/enabled", false);
	GLOBAL_DEF("rendering/textures/streaming/import_high_quality", false);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/initial_size", PROPERTY_HINT_ENUM, "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), 7); // default to 128
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/default_max_dimension", PROPERTY_HINT_ENUM, "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), 13); // default to 8192
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/default_min_dimension", PROPERTY_HINT_ENUM, "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), 7); // default to 128
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/wait_time"), 100);
	GLOBAL_DEF("rendering/textures/streaming/memory_budget_enabled", true);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/memory_budget_mb"), 512);

	// Load initial settings.
	setting_texture_change_wait_msec = GLOBAL_GET("rendering/textures/streaming/wait_time");
	setting_texture_min_resolution = 1 << int(GLOBAL_GET("rendering/textures/streaming/default_min_dimension"));
	setting_texture_max_resolution = 1 << int(GLOBAL_GET("rendering/textures/streaming/default_max_dimension"));
	setting_streaming_is_enabled = GLOBAL_GET("rendering/textures/streaming/enabled");
	setting_budget_enabled = GLOBAL_GET("rendering/textures/streaming/memory_budget_enabled");
	setting_budget_mb = GLOBAL_GET("rendering/textures/streaming/memory_budget_mb");

	print_verbose(vformat("TextureStreaming settings changed: enabled=%d, budget_enabled=%d, budget_mb=%d, min_res=%d, max_res=%d wait_time=%d",
			(int)setting_streaming_is_enabled,
			(int)setting_budget_enabled,
			setting_budget_mb,
			setting_texture_min_resolution,
			setting_texture_max_resolution,
			setting_texture_change_wait_msec));

	if (!setting_streaming_is_enabled) {
		return;
	}

	const String rendering_method = OS::get_singleton()->get_current_rendering_method();
	if (rendering_method == "gl_compatibility") {
		WARN_PRINT("Texture streaming is not supported with the Compatibility renderer.");
		return;
	}

	feedback_buffer_thread.start(_feedback_buffer_thread_func, this);
	texture_reload_thread.start(_texture_reload_thread_func, this);

	// Connect to the frame post draw signal to submit feedback buffers. The signal is used instead of request_frame_drawn_callback
	// since request_frame_drawn_callback is only called once per request, and we need continuous feedback each frame.
	Error err = RenderingServer::get_singleton()->connect("frame_post_draw", callable_mp(this, &TextureStreaming::feedback_frame_done_callback));
	if (err != OK) {
		ERR_PRINT("Failed to connect frame post draw signal.");
	}

	// There is some initialization that must be done on the render thread since it interacts with RD
	// which requires being on the render thread.
	RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &TextureStreaming::render_thread_specific_initialization));
}

TextureStreaming::~TextureStreaming() {
	// Signal both command queues to exit (this will wake up the processing threads)
	command_queue.request_exit();
	io_command_queue.request_exit();

	if (feedback_buffer_thread.is_started()) {
		feedback_buffer_thread.wait_to_finish();
	}

	if (texture_reload_thread.is_started()) {
		texture_reload_thread.wait_to_finish();
	}

	// Clean up buffer pool
	{
		MutexLock lock(buffer_pool_mutex);

		if (current_feedback_buffer.is_valid()) {
			buffer_pool.push_back(current_feedback_buffer);
			current_feedback_buffer = RID();
		}

		LocalVector<RID> buffers = feedback_buffer_owner.get_owned_list();
		for (uint32_t i = 0; i < buffers.size(); i++) {
			feedback_buffer_owner.free(buffers[i]);
		}
		buffer_pool.clear();
	}

	// Stop processing thread
	singleton = nullptr;
}

void TextureStreaming::render_thread_specific_initialization() {
	ERR_NOT_ON_RENDER_THREAD;

	// Initialize buffer pool
	{
		MutexLock lock(buffer_pool_mutex);
		while (buffer_pool.size() < 3) {
			RID buffer = feedback_buffer_owner.allocate_rid();
			MaterialFeedbackBuffer materialFeedbackBuffer;
			materialFeedbackBuffer.buffer_size = 0;
			materialFeedbackBuffer.self = buffer;
			feedback_buffer_owner.initialize_rid(buffer, materialFeedbackBuffer);
			buffer_pool.push_back(buffer);

			MaterialFeedbackBuffer *mb = feedback_buffer_owner.get_or_null(buffer);
			mb->resize();
			mb->clear();

			RD::get_singleton()->set_resource_name(mb->buffer, "MaterialFeedbackBuffer_" + itos(buffer.get_id()));
		}
	}

	// Get an initial buffer.
	current_feedback_buffer = feedback_buffer_get_next();

	initialized = true;
}

void TextureStreaming::feedback_frame_done_callback() {
	// feedback_frame_done_callback_render_thread must be called on the render thread since it interacts with RD.
	RS::get_singleton()->call_on_render_thread(callable_mp(this, &TextureStreaming::feedback_frame_done_callback_render_thread));
}

void TextureStreaming::feedback_frame_done_callback_render_thread() {
	if (!initialized) {
		return;
	}

	// Throttle feedback buffer submission to once every 30ms
	const uint64_t current_ticks = OS::get_singleton()->get_ticks_msec();
	if (current_ticks - feedback_buffer_last_submit_ticks < 30) {
		return;
	}

	if (current_feedback_buffer.is_null()) {
		WARN_PRINT("TextureStreaming feedback_frame_done_callback called with invalid feedback buffer RID.");
		return;
	}

	// Try to get a new buffer for the next frame
	RID next_buffer = feedback_buffer_get_next();
	if (next_buffer.is_null()) {
		print_verbose("TextureStreaming feedback_frame_done_callback called but no feedback buffers are available.");
		feedback_buffer_last_submit_ticks = current_ticks; //
		return;
	}

	// Submit the current feedback buffer
	MaterialFeedbackBuffer *mb = feedback_buffer_owner.get_or_null(current_feedback_buffer);
	RD::get_singleton()->buffer_get_data_async(mb->buffer, callable_mp(this, &TextureStreaming::feedback_handle_data).bind(current_feedback_buffer));
	feedback_buffer_last_submit_ticks = current_ticks;

	// Swap to the next buffer
	current_feedback_buffer = next_buffer;
}

RID TextureStreaming::feedback_buffer_get_uniform_rid() {
	ERR_NOT_ON_RENDER_THREAD_V(RID());
	MaterialFeedbackBuffer *_buffer = feedback_buffer_owner.get_or_null(current_feedback_buffer);

	if (!initialized) {
		// WARN_PRINT("TextureStreaming feedback_buffer_get_uniform_rid: TextureStreaming not initialized yet.");
		return RID();
	}

	if (_buffer && _buffer->buffer.is_valid() && _buffer->buffer_size > 0) {
		return _buffer->buffer;
	}

	if (setting_streaming_is_enabled) {
		WARN_PRINT("TextureStreaming feedback_buffer_get_uniform_rid: No valid feedback buffer available, but streaming is enabled.");
	} else {
		// Streaming is disabled, so this is expected.
	}

	return RID();
}

RID TextureStreaming::feedback_buffer_get_next() {
	ERR_NOT_ON_RENDER_THREAD_V(RID());
	MutexLock lock(buffer_pool_mutex);

	RID buffer;
	if (buffer_pool.size() > 0) {
		// Use a buffer from the pool
		buffer = buffer_pool[buffer_pool.size() - 1];
		buffer_pool.remove_at(buffer_pool.size() - 1);
	}

	if (buffer.is_valid()) {
		MaterialFeedbackBuffer *mb = feedback_buffer_owner.get_or_null(buffer);
		ERR_FAIL_NULL_V(mb, RID());
		mb->resize();
		mb->clear();
	}

	return buffer;
}

uint32_t TextureStreaming::feedback_buffer_material_index(RID p_material) {
	ERR_NOT_ON_RENDER_THREAD_V(UINT32_MAX);
	ERR_FAIL_COND_V(p_material.is_null(), UINT32_MAX);

	// Put the RID for the material into the feedback buffer's rid_map at the index of the material info.
	MaterialFeedbackBuffer *_buffer = feedback_buffer_owner.get_or_null(current_feedback_buffer);
	if (_buffer) {
		// material_info_owner.get_index() is thread-safe (only reads from RID_Owner which is thread-safe)
		uint32_t index = material_info_owner.get_index(p_material);
		if (_buffer->rid_map.size() <= (index + 1)) {
			_buffer->rid_map.resize(index + 1);
		}

		_buffer->rid_map[index] = p_material;

		return index;
	}

	return UINT32_MAX;
}

void TextureStreaming::feedback_handle_data(const PackedByteArray &p_array, RID p_buffer) {
	if (TextureStreaming::get_singleton() == nullptr) {
		return;
	}
	ERR_FAIL_COND(!p_buffer.is_valid());
	MaterialFeedbackBuffer *mb = feedback_buffer_owner.get_or_null(p_buffer);
	if (mb == nullptr) {
		return;
	}

	mb->data = p_array;

	uint64_t current_ticks = OS::get_singleton()->get_ticks_msec();
	command_queue.push_internal("_process_material_feedback_buffer", this, &TextureStreaming::_process_material_feedback_buffer, mb, current_ticks);
}

void TextureStreaming::_process_material_feedback_buffer(MaterialFeedbackBuffer *p_mb, uint64_t p_ticks_msec) {
	uint32_t *data_ptr = (uint32_t *)p_mb->data.ptrw();
	ERR_FAIL_NULL(data_ptr);

	// Distribute feedback to materials and their associated textures
	// OPTIMIZATION: Hold the lock for the entire loop instead of per-iteration.
	// This avoids lock acquire/release overhead for every material (was O(N) locks, now O(1)).
	const uint32_t *p_data = (uint32_t *)data_ptr;
	{
		MutexLock lock(material_mutex);
		for (uint32_t i = 0; i < p_mb->rid_map.size(); i++) {
			uint32_t requested_resolution = p_data[i];
			RID material_rid = p_mb->rid_map[i];

			MaterialInfo *info = material_info_owner.get_or_null(material_rid);

			if (!info) {
				continue;
			}

			const Vector<RID> &textures = info->textures;
			requested_resolution = info->update(requested_resolution, p_ticks_msec);

			for (int j = 0; j < textures.size(); j++) {
				RID texture_rid = textures[j];
				StreamingState *state = streaming_info_owner.get_or_null(texture_rid);

				// If the texture has been deleted but the material hasn't updated the list of textures yet, skip it.
				if (!state) {
					continue;
				}

				// Just a safety clamp. This is really to just ensure we never get crazy values.
				uint32_t clamped_resolution = CLAMP(requested_resolution, 1u, 8192u);

				if (clamped_resolution >= state->feedback_resolution) {
					state->requested_tick_msec = p_ticks_msec;
					state->feedback_resolution = clamped_resolution;
				}
			}
		}
	}

	{
		// Return buffer to the pool since we're done with it.
		MutexLock lock(buffer_pool_mutex);
		buffer_pool.push_back(p_mb->self);
	}

	{
		// Run fit/process algorithm
		_feedback_buffer_process(p_ticks_msec);
	}
}

void TextureStreaming::_feedback_buffer_thread_func(void *p_udata) {
	Thread::set_name("TexStreaming");

	TextureStreaming *texture_streaming = static_cast<TextureStreaming *>(p_udata);

	print_verbose("Texture Streaming process thread starting...");

	texture_streaming->_feedback_buffer_thread_main();
}

void TextureStreaming::_feedback_buffer_thread_main() {
	// Main loop: wait for commands and process them
	// wait_and_flush() returns false when exit is requested
	while (command_queue.wait_and_flush()) {
		// Commands are already flushed by wait_and_flush()
		// Add any per-iteration work here if needed
	}
}

// Budget-aware texture resolution fitting algorithm.
// This function processes all streaming textures and determines their target resolutions
// based on shader feedback and memory budget constraints.
//
// Algorithm overview:
// 1. Build candidates: Start each texture at MAX(requested_resolution, current_resolution)
//    to preserve quality when budget allows.
// 2. Budget fitting: If over budget, iteratively reduce the "least important" texture
//    by one mip level until within budget or all textures hit minimum.
// 3. Gradual application: Smoothly transition current_resolution toward fit_resolution
//    to avoid visual popping.
//
// Priority for reductions (when over budget):
// - First: Textures above their requested resolution (cached quality)
// - Second: Textures with longer inactivity times (not recently used)
// - Third: Textures with larger memory footprint (more savings per reduction)
// - Fourth: Higher resolution textures (tiebreaker)
void TextureStreaming::_feedback_buffer_process(uint64_t p_ticks_msec) {
	const LocalVector<RID> buffers = streaming_info_owner.get_owned_list();

	// Reuse fit_candidates vector to reduce allocations. Clear it for this frame.
	fit_candidates.clear();
	fit_candidates.reserve(buffers.size());

	// Phase 1: Build candidate list with initial target resolutions.
	// Strategy: Preserve current resolution if higher than requested (budget-aware caching).
	uint64_t total_requested_bytes = 0;
	for (uint32_t i = 0; i < buffers.size(); i++) {
		StreamingState *state = streaming_info_owner.get_or_null(buffers[i]);
		if (!state) {
			continue;
		}

		// Compute effective min/max resolution for this texture.
		// Respects per-texture limits, global settings, and actual texture dimensions.
		uint16_t max_res = MIN(state->max_resolution > 0 ? state->max_resolution : setting_texture_max_resolution, MAX(state->width, state->height));
		uint16_t min_res = MIN(MIN(state->min_resolution > 0 ? state->min_resolution : setting_texture_min_resolution, MAX(state->width, state->height)), max_res);

		// request_resolution comes from shader feedback (what shaders actually need).
		state->request_resolution = state->feedback_resolution;
		uint16_t desired_res = CLAMP(state->request_resolution, min_res, max_res);

		// Key optimization: Preserve current resolution if higher than requested.
		// This avoids unnecessary quality drops when budget allows - textures that were
		// once needed at high resolution stay there until budget pressure demands otherwise.
		uint16_t initial_target = state->current_resolution > 0 ? MAX(desired_res, state->current_resolution) : desired_res;
		initial_target = CLAMP(initial_target, min_res, max_res);

		FitCandidate candidate;
		candidate.state = state;
		candidate.min_res = min_res;
		candidate.max_res = max_res;
		candidate.target_res = initial_target;

		// Track inactivity for priority during budget fitting.
		// Longer inactivity = lower priority = reduced first when over budget.
		const uint64_t msecs_since_request = p_ticks_msec - state->requested_tick_msec;
		candidate.inactivity_msec = state->requested_tick_msec == 0 ? 0 : msecs_since_request;

		candidate.bytes = Image::get_image_data_size(initial_target, initial_target, state->format, true);
		total_requested_bytes += candidate.bytes;
		fit_candidates.push_back(candidate);

		state->feedback_resolution = 0; // Reset for feedback results
	}

	// Phase 2: Budget fitting - iteratively reduce textures until within memory budget.
	// Uses a heap-based approach for O(M log N) complexity instead of O(N×M).
	uint64_t assigned_bytes = total_requested_bytes;
	if (setting_budget_enabled) {
		const uint64_t budget_bytes = uint64_t(setting_budget_mb) * 1024ull * 1024ull;

		// Early exit if already within budget (no Phase 2 work needed)
		if (budget_bytes > 0 && assigned_bytes > budget_bytes) {
			// Build heap of reducible candidate pointers
			reduction_heap.clear();
			reduction_heap.reserve(fit_candidates.size());
			for (uint32_t i = 0; i < fit_candidates.size(); i++) {
				if (fit_candidates[i].state && fit_candidates[i].target_res > fit_candidates[i].min_res) {
					reduction_heap.push_back(&fit_candidates[i]);
				}
			}

			// Create max-heap where "top" element is the least important texture to reduce
			// SortArray with custom comparator for FitCandidate pointers
			SortArray<FitCandidate *, FitCandidateComparator> heap_sorter{};
			heap_sorter.make_heap(0, reduction_heap.size(), reduction_heap.ptr());

			// Iteratively reduce textures one mip level at a time until budget is satisfied
			int64_t heap_size = reduction_heap.size();
			while (assigned_bytes > budget_bytes && heap_size > 0) {
				// Pop the least important candidate from heap
				heap_sorter.pop_heap(0, heap_size, reduction_heap.ptr());
				heap_size--;
				FitCandidate *best = reduction_heap[heap_size];

				// Skip if candidate became exhausted
				if (!best->state || best->target_res <= best->min_res) {
					continue;
				}

				// Reduce selected texture by one mip level (halve resolution)
				uint16_t previous_res = best->target_res;
				uint16_t decreased_res = MAX(best->min_res, previous_res >> 1u);
				if (decreased_res == previous_res) {
					// Already at minimum - mark exhausted and continue
					best->state = nullptr;
					continue;
				}

				// Update candidate's memory footprint after reduction
				uint64_t previous_bytes = best->bytes;
				uint64_t new_bytes = Image::get_image_data_size(decreased_res, decreased_res, best->state->format, true);
				best->target_res = decreased_res;
				best->bytes = new_bytes;

				// Sanity check: reducing resolution should always reduce memory
				ERR_FAIL_COND(new_bytes >= previous_bytes);
				assigned_bytes -= (previous_bytes - new_bytes);

				// Re-insert candidate if still reducible (will be re-prioritized in heap)
				if (best->target_res > best->min_res) {
					reduction_heap[heap_size] = best;
					heap_size++;
					heap_sorter.push_heap(0, heap_size - 1, 0, best, reduction_heap.ptr());
				}
			}

			// Check if budget was satisfied after all reductions
			if (assigned_bytes > budget_bytes) {
				WARN_PRINT_ONCE("Texture streaming budget cannot be satisfied even at minimum resolutions.");
			}
		}
	}

	// Phase 3: Apply fitted resolutions and gradually transition current_resolution.
	// Smooth transitions (one mip level per wait period) prevent visual popping.
	uint64_t memory = 0;
	for (uint32_t i = 0; i < fit_candidates.size(); i++) {
		FitCandidate &candidate = fit_candidates[i];
		StreamingState *state = candidate.state;
		if (!state) {
			continue;
		}

		// Set the fit_resolution (the goal resolution after budget constraints).
		state->fit_resolution = candidate.target_res;

		// Gradually adjust current_resolution toward fit_resolution.
		// This prevents visual "popping" by limiting changes to one mip level per frame
		// and enforcing a minimum time delay between changes (setting_texture_change_wait_msec).
		{
			uint64_t msecs_since_change = p_ticks_msec - state->changed_tick_msec;
			if (msecs_since_change >= setting_texture_change_wait_msec) {
				// fit_resolution is already clamped via candidate.target_res during budget fitting.
				state->current_resolution = CLAMP(state->current_resolution, 1, 8192);

				// Upgrade: Move up one mip level if fit_resolution is higher.
				if (state->fit_resolution > state->current_resolution && state->current_resolution < candidate.max_res) {
					state->current_resolution <<= 1u;
					state->changed_tick_msec = p_ticks_msec;
				}
				// Downgrade: Move down one mip level if fit_resolution is lower.
				else if (state->fit_resolution < state->current_resolution && state->current_resolution > candidate.min_res) {
					state->current_resolution >>= 1u;
					state->changed_tick_msec = p_ticks_msec;
				}
			}
		}

		// Accumulate actual memory usage based on current_resolution.
		memory += Image::get_image_data_size(state->current_resolution, state->current_resolution, state->format, true);

		// Queue texture reload if resolution changed and not already pending.
		// Use atomic compare-exchange to avoid duplicate queue entries.
		if (state->current_resolution != state->last_resolution) {
			uint16_t expected = 0;
			// Only queue if no reload is pending (pending_reload_resolution == 0)
			if (state->pending_reload_resolution.compare_exchange_strong(expected, state->current_resolution)) {
				state->last_resolution = state->current_resolution;
				// Push reload command to I/O thread - passes state pointer for coalescing
				io_command_queue.push_internal("_do_texture_reload", this, &TextureStreaming::_do_texture_reload, state);
			}
			// If a reload is already pending, we skip - the pending reload will pick up the latest resolution
		}
	}

	texture_streaming_total_memory = memory;
}

void TextureStreaming::_texture_reload_thread_func(void *p_udata) {
	Thread::set_name("TexStreaming IO");

	TextureStreaming *tss = static_cast<TextureStreaming *>(p_udata);

	print_verbose("Texture Streaming i/o thread starting...");

	tss->_texture_reload_thread_main();
}

void TextureStreaming::_texture_reload_thread_main() {
	// Main loop: wait for commands and process them
	// wait_and_flush() returns false when exit is requested
	while (io_command_queue.wait_and_flush()) {
		// Commands are already flushed by wait_and_flush()
	}
}

void TextureStreaming::_do_texture_reload(StreamingState *p_state) {
	// Read the pending resolution and clear it atomically
	uint16_t resolution = p_state->pending_reload_resolution.exchange(0);

	if (resolution == 0) {
		// Already processed or cancelled
		return;
	}

	// This runs on the I/O thread - perform the actual texture reload
	p_state->reload_callable.call(resolution);
}
