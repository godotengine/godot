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
#include "core/math/math_funcs.h"
#include "core/templates/local_vector.h"
#include "core/templates/sort_array.h"
#include "core/typedefs.h"
#include "core/variant/variant.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h"

// Helper macros for code outside of the rendering server, but that is
// called by the rendering server.
#ifdef DEBUG_ENABLED
#define ERR_NOT_ON_RENDER_THREAD \
	RenderingServer *rendering_server = RenderingServer::get_singleton(); \
	ERR_FAIL_NULL(rendering_server); \
	ERR_FAIL_COND(!rendering_server->is_on_render_thread());
#define ERR_NOT_ON_RENDER_THREAD_V(m_ret) \
	RenderingServer *rendering_server = RenderingServer::get_singleton(); \
	ERR_FAIL_NULL_V(rendering_server, m_ret); \
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

	buffer_size = Math::nearest_power_of_2_templated(MAX(4096u, material_info_count_bytes));
	buffer = RD::get_singleton()->storage_buffer_create(buffer_size, Vector<uint8_t>(), 0, RD::BufferCreationBits::BUFFER_CREATION_AS_STORAGE_BIT);

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
	ClassDB::bind_method(D_METHOD("get_min_lod_override"), &TextureStreaming::get_min_lod_override);
	ClassDB::bind_method(D_METHOD("set_min_lod_override", "min_lod"), &TextureStreaming::set_min_lod_override);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "min_lod_override", PROPERTY_HINT_RANGE, "0,13,1"), "set_min_lod_override", "get_min_lod_override");

	ClassDB::bind_method(D_METHOD("get_max_lod_override"), &TextureStreaming::get_max_lod_override);
	ClassDB::bind_method(D_METHOD("set_max_lod_override", "max_lod"), &TextureStreaming::set_max_lod_override);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lod_override", PROPERTY_HINT_RANGE, "0,13,1"), "set_max_lod_override", "get_max_lod_override");

	ClassDB::bind_method(D_METHOD("get_memory_budget_mb_override"), &TextureStreaming::get_memory_budget_mb_override);
	ClassDB::bind_method(D_METHOD("set_memory_budget_mb_override", "mb"), &TextureStreaming::set_memory_budget_mb_override);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "memory_budget_mb_override", PROPERTY_HINT_RANGE, "0,65535,1"), "set_memory_budget_mb_override", "get_memory_budget_mb_override");

	ClassDB::bind_method(D_METHOD("get_memory_budget_bytes_used"), &TextureStreaming::get_memory_budget_bytes_used);
}

RID TextureStreaming::texture_configure_streaming(RID p_texture, Image::Format p_format, int p_width, int p_height, int p_min_lod, int p_max_lod, Callable p_reload_callable) {
	ERR_FAIL_COND_V(p_texture.is_null(), RID());
	ERR_FAIL_COND_V(p_width == 0, RID());
	ERR_FAIL_COND_V(p_height == 0, RID());
	ERR_FAIL_COND_V(!p_reload_callable.is_valid(), RID());

	return push_and_wait<RID>("_texture_configure_streaming_impl", this, &TextureStreaming::_texture_configure_streaming_impl,
			p_texture, p_format, p_width, p_height, p_min_lod, p_max_lod, p_reload_callable);
}

void TextureStreaming::_texture_configure_streaming_impl(Completion<RID> *p_completion, RID p_texture, Image::Format p_format, int p_width, int p_height, int p_min_lod, int p_max_lod, const Callable &p_reload_callable) {
	print_verbose(vformat("TextureStreaming: Configuring streaming for texture RID(%d) size(%dx%d) format(%d) min_lod(%d) max_lod(%d)",
			p_texture.get_id(), p_width, p_height, int(p_format), p_min_lod, p_max_lod));

	RID rid = streaming_info_owner.allocate_rid();
	StreamingState state;
	state.texture = p_texture;
	state.format = p_format;
	state.width = p_width;
	state.height = p_height;
	state.min_lod_override = uint8_t(CLAMP(p_min_lod, 0, 14));
	state.max_lod_override = uint8_t(CLAMP(p_max_lod, 0, 14));
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
		state->pending_reload_mip.exchange(INVALID_MIP);
		RS::get_singleton()->texture_2d_attach_streaming_state(state->texture, RID());
		streaming_info_owner.free(p_rid);
	}
	p_completion->complete();
}

void TextureStreaming::texture_update(RID p_rid, int p_width, int p_height, int p_min_lod, int p_max_lod) {
	push_and_sync("_texture_update_impl", this, &TextureStreaming::_texture_update_impl, p_rid, p_width, p_height, p_min_lod, p_max_lod);
}

void TextureStreaming::_texture_update_impl(CompletionVoid *p_completion, RID p_rid, int p_width, int p_height, int p_min_lod, int p_max_lod) {
	StreamingState *state = streaming_info_owner.get_or_null(p_rid);
	if (state) {
		state->width = p_width;
		state->height = p_height;
		state->min_lod_override = uint8_t(CLAMP(p_min_lod, 0, 14));
		state->max_lod_override = uint8_t(CLAMP(p_max_lod, 0, 14));
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

	// Global settings
	GLOBAL_DEF_RST("rendering/textures/streaming/enabled", false);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/wait_time"), 100);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/memory_budget_mb"), 512);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/max_ops_per_second"), 5000);

	// Mip level settings: 0 = full resolution (best quality), higher = lower resolution
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/min_lod", PROPERTY_HINT_RANGE, "0,13,1"), 0);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/max_lod", PROPERTY_HINT_RANGE, "0,13,1"), 3);

	// Inactivity decay: time per mip level of decay in milliseconds.
	// The first decay happens after one rate period. Set to 0 to disable decay.
	GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/textures/streaming/inactivity_decay_rate_ms", PROPERTY_HINT_RANGE, "0,60000,100,or_greater"), 5000);

	// Load initial settings.
	setting_streaming_is_enabled = GLOBAL_GET("rendering/textures/streaming/enabled");
	setting_texture_change_wait_msec = GLOBAL_GET("rendering/textures/streaming/wait_time");
	setting_max_ops_per_second = GLOBAL_GET("rendering/textures/streaming/max_ops_per_second");
	setting_min_lod = uint8_t(CLAMP(int(GLOBAL_GET("rendering/textures/streaming/min_lod")), 0, 13));
	setting_max_lod = uint8_t(CLAMP(int(GLOBAL_GET("rendering/textures/streaming/max_lod")), 0, 13));
	setting_budget_mb = GLOBAL_GET("rendering/textures/streaming/memory_budget_mb");
	setting_inactivity_decay_rate_ms = GLOBAL_GET("rendering/textures/streaming/inactivity_decay_rate_ms");

	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &TextureStreaming::_on_settings_changed));

	print_verbose(vformat("TextureStreaming settings changed: enabled=%d, budget_mb=%d, min_lod=%d, max_lod=%d wait_time=%d",
			(int)setting_streaming_is_enabled,
			(int)setting_budget_mb,
			setting_min_lod,
			setting_max_lod,
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
	const String display_server_name = DisplayServer::get_singleton()->get_name();
	if (display_server_name != "headless") {
		RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &TextureStreaming::_render_thread_specific_initialization));
	}
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

void TextureStreaming::_on_settings_changed() {
	setting_texture_change_wait_msec = GLOBAL_GET("rendering/textures/streaming/wait_time");
	setting_max_ops_per_second = GLOBAL_GET("rendering/textures/streaming/max_ops_per_second");

	setting_min_lod = uint8_t(CLAMP(int(GLOBAL_GET("rendering/textures/streaming/min_lod")), 0, 13));
	setting_max_lod = uint8_t(CLAMP(int(GLOBAL_GET("rendering/textures/streaming/max_lod")), 0, 13));
	setting_budget_mb = GLOBAL_GET("rendering/textures/streaming/memory_budget_mb");
	setting_inactivity_decay_rate_ms = GLOBAL_GET("rendering/textures/streaming/inactivity_decay_rate_ms");
}

void TextureStreaming::_render_thread_specific_initialization() {
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
	const uint32_t *p_data = (uint32_t *)data_ptr;
	{
		MutexLock lock(material_mutex);
		for (uint32_t i = 0; i < p_mb->rid_map.size(); i++) {
			RID material_rid = p_mb->rid_map[i];
			uint32_t material_feedback = p_data[i];

			if (material_feedback == 0) {
				continue;
			}

			// Undo the bitwise NOT applied in the shader (reverse of ~floatBitsToUint in shader)
			uint32_t original_bits = ~material_feedback;
			float min_uv_sq;
			memcpy(&min_uv_sq, &original_bits, sizeof(float));
			min_uv_sq = Math::is_nan(min_uv_sq) ? 1.0f : min_uv_sq;
			float min_uv_len = Math::sqrt(min_uv_sq);
			MaterialInfo *info = material_info_owner.get_or_null(material_rid);
			if (!info) {
				continue;
			}

			const Vector<RID> &textures = info->textures;
			for (int j = 0; j < textures.size(); j++) {
				RID texture_rid = textures[j];
				StreamingState *state = streaming_info_owner.get_or_null(texture_rid);

				// If the texture has been deleted but the material hasn't updated the list of textures yet, skip it.
				if (!state) {
					continue;
				}

				// Determine required mip level based on smoothed UV length and texture size.
				// For rectangular textures, use the smaller dimension to avoid undersampling.
				float min_dim = MIN(float(state->width), float(state->height));
				float texel_coverage = min_uv_len * min_dim;

				// Guard against invalid values that would cause undefined behavior in log2.
				// If texel_coverage is <= 0, request highest quality (mip 0).
				int required_mip = (texel_coverage > 0.0f) ? int(Math::floor(Math::log2(texel_coverage))) : 0;

				// Clamp to valid mip range for this texture.
				uint8_t clamped_mip = uint8_t(CLAMP(required_mip, 0, int(state->get_mip_count()) - 1));

				// Update feedback if this is a better (lower) mip level than previously recorded
				// Lower mip = higher quality needed
				if (clamped_mip < state->feedback_mip) {
					state->requested_tick_msec = p_ticks_msec;
					state->feedback_mip = clamped_mip;
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

	// Reuse vector to reduce allocations.
	fit_candidates.clear();
	fit_candidates.reserve(buffers.size());

	// Phase 1: Build candidate list with initial target mip levels.
	uint64_t total_requested_bytes = 0;
	for (uint32_t i = 0; i < buffers.size(); i++) {
		StreamingState *state = streaming_info_owner.get_or_null(buffers[i]);
		if (!state) {
			continue;
		}

		// Compute effective best/worst mip (0 = highest quality).
		uint8_t min_lod = state->min_lod_override > 0 ? state->min_lod_override - 1 : _get_system_min_lod();
		uint8_t max_lod = state->max_lod_override > 0 ? state->max_lod_override - 1 : _get_system_max_lod();
		max_lod = MAX(max_lod, min_lod);

		uint8_t raw_request_mip = state->feedback_mip;
		state->request_mip = state->update(raw_request_mip, p_ticks_msec, 0.00005f);
		uint8_t desired_mip = CLAMP(state->request_mip, min_lod, max_lod);

		// Only allow improvements freely; reductions happen via budget fitting or decay.
		uint8_t initial_target;
		if (state->fit_mip < INVALID_MIP && state->fit_mip >= min_lod && state->fit_mip <= max_lod) {
			initial_target = MIN(desired_mip, state->fit_mip);
		} else if (state->current_mip < INVALID_MIP) {
			initial_target = MIN(desired_mip, state->current_mip);
		} else {
			initial_target = desired_mip;
		}

		// Apply inactivity decay.
		const uint64_t msecs_since_request = (state->requested_tick_msec > 0) ? (p_ticks_msec - state->requested_tick_msec) : 0;
		if (setting_inactivity_decay_rate_ms > 0) {
			const uint32_t decay_mips = uint32_t(msecs_since_request / setting_inactivity_decay_rate_ms);
			if (decay_mips > 0) {
				// Increase target mip (reduce quality) by decay amount, clamped to max_lod
				initial_target = uint8_t(MIN(uint32_t(initial_target) + decay_mips, uint32_t(max_lod)));
			}
		}

		initial_target = CLAMP(initial_target, min_lod, max_lod);

		FitCandidate candidate;
		candidate.state = state;
		candidate.min_lod = min_lod;
		candidate.max_lod = max_lod;
		candidate.target_mip = initial_target;
		candidate.inactivity_msec = msecs_since_request;

		candidate.bytes = state->get_bytes_at_mip(initial_target);
		total_requested_bytes += candidate.bytes;
		fit_candidates.push_back(candidate);

		state->feedback_mip = INVALID_MIP;
	}

	// Phase 2: Budget fitting - iteratively reduce textures until within memory budget.
	uint64_t assigned_bytes = total_requested_bytes;
	const uint64_t budget_bytes = uint64_t(_get_memory_budget_mb()) * 1024ull * 1024ull;
	if (budget_bytes > 0) {
		// Symmetric hysteresis (10% deadband).
		const uint64_t budget_high = budget_bytes + (budget_bytes / 20); // +5% = trigger reduction
		const uint64_t budget_low = budget_bytes - (budget_bytes / 20); // -5% = stop reduction

		if (budget_bytes > 0 && assigned_bytes > budget_high) {
			// Build heap of reducible candidates.
			reduction_heap.clear();
			reduction_heap.reserve(fit_candidates.size());
			for (uint32_t i = 0; i < fit_candidates.size(); i++) {
				if (fit_candidates[i].state && fit_candidates[i].target_mip < fit_candidates[i].max_lod) {
					reduction_heap.push_back(&fit_candidates[i]);
				}
			}

			// Create max-heap (least important texture at top).
			SortArray<FitCandidate *, FitCandidateComparator> heap_sorter{};
			heap_sorter.make_heap(0, reduction_heap.size(), reduction_heap.ptr());

			int64_t heap_size = reduction_heap.size();
			while (assigned_bytes > budget_low && heap_size > 0) {
				heap_sorter.pop_heap(0, heap_size, reduction_heap.ptr());
				heap_size--;
				FitCandidate *best = reduction_heap[heap_size];

				if (!best->state || best->target_mip >= best->max_lod) {
					continue;
				}

				// Reduce texture by one mip level.
				uint8_t previous_mip = best->target_mip;
				uint8_t increased_mip = MIN(best->max_lod, uint8_t(previous_mip + 1));
				if (increased_mip == previous_mip) {
					// Already at minimum quality - mark exhausted and continue
					best->state = nullptr;
					continue;
				}

				uint64_t previous_bytes = best->bytes;
				uint64_t new_bytes = best->state->get_bytes_at_mip(increased_mip);
				best->target_mip = increased_mip;
				best->bytes = new_bytes;

				ERR_FAIL_COND(new_bytes > previous_bytes);
				assigned_bytes -= (previous_bytes - new_bytes);

				// Re-insert candidate if still reducible.
				if (best->target_mip < best->max_lod) {
					reduction_heap[heap_size] = best;
					heap_size++;
					heap_sorter.push_heap(0, heap_size - 1, 0, best, reduction_heap.ptr());
				}
			}

			if (assigned_bytes > budget_low) {
				WARN_PRINT_ONCE(vformat(
						"Texture streaming budget cannot be satisfied even at minimum quality. Used %s, budget %s bytes.", String::humanize_size(assigned_bytes), String::humanize_size(budget_bytes)));
			}
		}
	}

	// Phase 3: Apply fitted mip levels and gradually transition current_mip.
	// Smooth transitions (one mip level per wait period) prevent visual popping.
	uint64_t memory = 0;
	for (uint32_t i = 0; i < fit_candidates.size(); i++) {
		FitCandidate &candidate = fit_candidates[i];
		StreamingState *state = candidate.state;
		if (!state) {
			continue;
		}

		// Set the fit_mip (the goal mip level after budget constraints).
		state->fit_mip = candidate.target_mip;

		// Adjust current_mip toward fit_mip.
		{
			if (state->current_mip == INVALID_MIP) {
				state->current_mip = candidate.max_lod;
			}

			const uint64_t msecs_since_change = p_ticks_msec - state->changed_tick_msec;
			if (msecs_since_change >= setting_texture_change_wait_msec) {
				if (state->fit_mip < state->current_mip && state->current_mip > candidate.min_lod) {
					state->current_mip--;
					state->changed_tick_msec = p_ticks_msec;
				} else if (state->fit_mip > state->current_mip && state->current_mip < candidate.max_lod) {
					state->current_mip++;
					state->changed_tick_msec = p_ticks_msec;
				}
			}
		}

		memory += state->get_bytes_at_mip(state->current_mip);

		// Queue texture reload if mip changed and not already pending.
		if (state->current_mip != state->last_mip) {
			uint8_t expected = INVALID_MIP;
			if (state->pending_reload_mip.compare_exchange_strong(expected, state->current_mip)) {
				state->last_mip = state->current_mip;
				// Push reload command to I/O thread - passes state pointer for coalescing
				io_command_queue.push_internal("_do_texture_reload", this, &TextureStreaming::_do_texture_reload, state);
			}
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
		// Add any per-iteration work here if needed
	}
}

void TextureStreaming::_do_texture_reload(StreamingState *p_state) {
	// Read the pending mip level and clear it atomically
	uint8_t mip_level = p_state->pending_reload_mip.exchange(INVALID_MIP);

	if (mip_level == INVALID_MIP) {
		// Already processed or cancelled
		return;
	}

	// Throttle I/O operations to limit disk/memory bandwidth usage.
	// This prevents overwhelming the system when many textures need reloading.
	if (setting_max_ops_per_second > 0) {
		const uint64_t min_interval_usec = 1000000 / setting_max_ops_per_second;
		if (min_interval_usec > 0) {
			uint64_t current_ticks = OS::get_singleton()->get_ticks_usec();
			uint64_t elapsed = current_ticks - last_io_op_ticks;
			if (elapsed < min_interval_usec) {
				OS::get_singleton()->delay_usec(min_interval_usec - elapsed);
			}
			last_io_op_ticks = OS::get_singleton()->get_ticks_usec();
		}
	}

	// This runs on the I/O thread - perform the actual texture reload
	// Pass the mip level to the callable
	p_state->reload_callable.call(mip_level);
}
