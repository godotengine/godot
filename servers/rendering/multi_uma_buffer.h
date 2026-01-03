/**************************************************************************/
/*  multi_uma_buffer.h                                                    */
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

#include "servers/rendering/rendering_server.h"

class MultiUmaBufferBase {
protected:
	LocalVector<RID> buffers;
	uint32_t curr_idx = UINT32_MAX;
	uint64_t last_frame_mapped = UINT64_MAX;
	const uint32_t max_extra_buffers;
#ifdef DEBUG_ENABLED
	const char *debug_name;
#endif

	MultiUmaBufferBase(uint32_t p_max_extra_buffers, const char *p_debug_name) :
			max_extra_buffers(p_max_extra_buffers)
#ifdef DEBUG_ENABLED
			,
			debug_name(p_debug_name)
#endif
	{
	}

#ifdef DEV_ENABLED
	~MultiUmaBufferBase() {
		DEV_ASSERT(buffers.is_empty() && "Forgot to call uninit()!");
	}
#endif

public:
	void uninit() {
		if (is_print_verbose_enabled()) {
			print_line("MultiUmaBuffer '"
#ifdef DEBUG_ENABLED
					+ String(debug_name) +
#else
					   "{DEBUG_ENABLED unavailable}"
#endif
					"' used a total of " + itos(buffers.size()) +
					" buffers. A large number may indicate a waste of VRAM and can be brought down by tweaking MAX_EXTRA_BUFFERS for this buffer.");
		}

		RenderingDevice *rd = RD::RenderingDevice::get_singleton();

		for (RID buffer : buffers) {
			if (buffer.is_valid()) {
				rd->free_rid(buffer);
			}
		}

		buffers.clear();
	}

	void shrink_to_max_extra_buffers() {
		DEV_ASSERT(curr_idx == 0u && "This function can only be called after reset and before being upload_and_advance again!");

		RenderingDevice *rd = RD::RenderingDevice::get_singleton();

		uint32_t elem_count = buffers.size();

		if (elem_count > max_extra_buffers) {
			if (is_print_verbose_enabled()) {
				print_line("MultiUmaBuffer '"
#ifdef DEBUG_ENABLED
						+ String(debug_name) +
#else
						   "{DEBUG_ENABLED unavailable}"
#endif
						"' peaked to " + itos(elem_count) + " elements and shrinking it to " + itos(max_extra_buffers) +
						". If you see this message often, then something is wrong with rendering or MAX_EXTRA_BUFFERS needs to be increased.");
			}
		}

		while (elem_count > max_extra_buffers) {
			--elem_count;
			if (buffers[elem_count].is_valid()) {
				rd->free_rid(buffers[elem_count]);
			}
			buffers.remove_at(elem_count);
		}
	}
};

enum class MultiUmaBufferType : uint8_t {
	UNIFORM,
	STORAGE,
	VERTEX,
};

/// Interface for making it easier to work with UMA.
///
/// # What is UMA?
///
/// It stands for Unified Memory Architecture. There are two kinds of UMA:
///	 1. HW UMA. This is the case of iGPUs (specially Android, iOS, Apple ARM-based macOS, PS4 & PS5)
///		The CPU and GPU share the same die and same memory. So regular RAM and VRAM are internally the
///		same thing. There may be some differences between them in practice due to cache synchronization
///		behaviors or the regular BW RAM may be purposely throttled (as is the case of PS4 & PS5).
///  2. "Pretended UMA". On PC Desktop GPUs with ReBAR enabled can pretend VRAM behaves like normal
///		RAM, while internally the data is moved across the PCIe Bus. This can cause differences
///		in execution time of the routines that write to GPU buffers as the region is often uncached
///		(i.e. write-combined) and PCIe latency and BW is vastly different from regular RAM.
///		Without ReBAR, the amount of UMA memory is limited to 256MB (shared by the entire system).
///
/// Since often this type of memory is uncached, it is not well-suited for downloading GPU -> CPU,
/// but rather for uploading CPU -> GPU.
///
/// # When to use UMA buffers?
///
/// UMA buffers have various caveats and improper usage might lead to visual glitches. Therefore they
/// should be used sparingly, where it makes a difference. Does all of the following check?:
///	  1. Data is uploaded from CPU to GPU every (or almost every) frame.
///   2. Data is always uploaded from scratch. Partial uploads are unsupported.
///	  3. If uploading multiple times per frame (e.g. for multiple passes). The amount of times
///      per frame is relatively stable (occasional spikes are fine if using MAX_EXTRA_BUFFERS).
///
/// # Why the caveats?
///
///	This is due to our inability to detect race conditions. If you write to an UMA buffer, submit
///	GPU commands and then write more data to it, we can't guarantee that you won't be writing to a
/// region the GPU is currently reading from. Tools like the validation layers cannot detect this
/// race condition at all, making it very hard to troubleshoot.
///
/// Therefore the safest approach is to use an interface that forces users to upload everything at once.
/// There is one exception for performance: map_raw_for_upload() will return a pointer, and it is your
/// responsibility to make sure you don't use that pointer again after submitting.
/// USE THIS API CALL SPARINGLY AND WITH CARE.
///
/// Since we forbid uploading more data after we've uploaded to it, this Interface will create
/// more buffers. This means users will need more UniformSets (i.e. uniform_set_create).
///
/// # How to use
///
/// Example code 01:
///		MultiUmaBuffer<1> uma_buffer = MultiUmaBuffer<1>("Debug name displayed if run with --verbose");
///		uma_buffer.set_uniform_size(0, max_size_bytes);
///
///		for(uint32_t i = 0u; i < num_passes; ++i) {
///			uma_buffer.prepare_for_upload(); // Creates a new buffer (if none exists already)
///											 // of max_size_bytes. Must be called.
///			uma_buffer.upload(0, src_data, size_bytes);
///
///			if(!uniform_set[i]) {
///				RD::Uniform u;
///				u.binding = 1;
///				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC;
///				u.append_id(uma_buffer._get(0u));
///				uniform_set[i] = rd->uniform_set_create( ... );
///			}
///		}
///
///	  // On shutdown (or if you need to call set_size again).
///	  uma_buffer.uninit();
///
/// Example code 02:
///
///		uma_buffer.prepare_for_upload();
///		RID rid = uma_buffer.get_for_upload(0u);
///		rd->buffer_update(rid, 0, sizeof(BakeParameters), &bake_parameters);
///		RD::Uniform u; // Skipping full initialization of u. See Example 01.
///		u.append_id(rid);
///
/// Example code 03:
///
///		void *dst_data = uma_buffer.map_raw_for_upload(0u);
///		memcpy(dst_data, src_data, size_bytes);
///		rd->buffer_flush(uma_buffer._get(0u));
///		RD::Uniform u; // Skipping full initialization of u. See Example 01.
///		u.append_id(rid);
///
/// # Tricks
///
///	Godot's shadow mapping code calls uma_buffer.uniform_buffers._get(-p_pass_offset) (i.e. a negative value)
/// because for various reasons its shadow mapping code was written like this:
///
///		for( uint32_t i = 0u; i < num_passes; ++i ) {
///			uma_buffer.prepare_for_upload();
///			uma_buffer.upload(0, src_data, size_bytes);
///		}
///		for( uint32_t i = 0u; i < num_passes; ++i ) {
///			RD::Uniform u;
///			u.binding = 1;
///			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC;
///			u.append_id(uma_buffer._get(-(num_passes - 1u - i)));
///			uniform_set[i] = rd->uniform_set_create( ... );
///		}
///
/// Every time prepare_for_upload() is called, uma_buffer._get(-idx) will return a different RID(*).
/// Thus with a negative value we can address previous ones. This is fine as long as the value idx
/// doesn't exceed the number of times the user called prepare_for_upload() for this frame.
///
/// (*)This RID will be returned again on the next frame after the same amount of prepare_for_upload()
/// calls; unless the number of times it was called exceeded MAX_EXTRA_BUFFERS.
///
/// # Template parameters
///
///	## NUM_BUFFERS
///
/// How many buffers we should track. e.g. instead of doing this:
///		MultiUmaBuffer<1> omni_lights = /*...*/;
///		MultiUmaBuffer<1> spot_lights = /*...*/;
///		MultiUmaBuffer<1> directional_lights = /*...*/;
///
///		omni_lights.set_uniform_size(0u, omni_size);
///		spot_lights.set_uniform_size(0u, spot_size);
///		directional_lights.set_uniform_size(0u, dir_size);
///
///		omni_lights.prepare_for_upload();
///		spot_lights.prepare_for_upload();
///		directional_lights.prepare_for_upload();
///
/// You can do this:
///
///		MultiUmaBuffer<3> lights = /*...*/;
///
///		lights.set_uniform_size(0u, omni_size);
///		lights.set_uniform_size(1u, spot_size);
///		lights.set_uniform_size(2u, dir_size);
///
///		lights.prepare_for_upload();
///
/// This approach works as long as all buffers would call prepare_for_upload() at the same time.
/// It saves some overhead.
///
/// ## MAX_EXTRA_BUFFERS
///
/// Upper limit on the number of buffers per frame.
///
/// There are times where rendering might spike for exceptional reasons, calling prepare_for_upload()
/// too many times, never to do that again. This will cause an increase in memory usage that will
/// never be reclaimed until shutdown.
///
/// MAX_EXTRA_BUFFERS can be used to handle such spikes, by deallocating the extra buffers.
/// Example:
///		MultiUmaBuffer<1, 6> buffer;
///
///		// Normal frame (assuming up to 6 passes is considered normal):
///		for(uint32_t i = 0u; i < 6u; ++i) {
///			buffer.prepare_for_upload();
///			...
///			buffer.upload(...);
///		}
///
///		// Exceptional frame:
///		for(uint32_t i = 0u; i < 24u; ++i) {
///			buffer.prepare_for_upload();
///			...
///			buffer.upload(...);
///		}
///
///	After the frame is done, those extra 18 buffers will be deleted.
/// Launching godot with --verbose will print diagnostic information.
template <uint32_t NUM_BUFFERS, uint32_t MAX_EXTRA_BUFFERS = UINT32_MAX>
class MultiUmaBuffer : public MultiUmaBufferBase {
	struct BufferInfo {
		uint32_t size_bytes = 0;
		MultiUmaBufferType type = MultiUmaBufferType::UNIFORM;
	};
	BufferInfo buffer_info[NUM_BUFFERS];
#ifdef DEV_ENABLED
	bool can_upload[NUM_BUFFERS] = {};
#endif

	void push() {
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();
		for (uint32_t i = 0u; i < NUM_BUFFERS; ++i) {
			const BufferInfo &info = buffer_info[i];
			RID buffer;
			switch (info.type) {
				case MultiUmaBufferType::STORAGE:
					buffer = rd->storage_buffer_create(info.size_bytes, Vector<uint8_t>(), BitField<RenderingDevice::StorageBufferUsage>(), RD::BUFFER_CREATION_DYNAMIC_PERSISTENT_BIT);
					break;
				case MultiUmaBufferType::VERTEX:
					buffer = rd->vertex_buffer_create(info.size_bytes, Vector<uint8_t>(), RD::BUFFER_CREATION_DYNAMIC_PERSISTENT_BIT);
					break;
				case MultiUmaBufferType::UNIFORM:
				default:
					buffer = rd->uniform_buffer_create(info.size_bytes, Vector<uint8_t>(), RD::BUFFER_CREATION_DYNAMIC_PERSISTENT_BIT);
					break;
			}
			buffers.push_back(buffer);
		}
	}

public:
	MultiUmaBuffer(const char *p_debug_name) :
			MultiUmaBufferBase(MAX_EXTRA_BUFFERS, p_debug_name) {}

	uint32_t get_curr_idx() const { return curr_idx; }

	void set_size(uint32_t p_idx, uint32_t p_size_bytes, MultiUmaBufferType p_type) {
		DEV_ASSERT(buffers.is_empty());
		buffer_info[p_idx].size_bytes = p_size_bytes;
		buffer_info[p_idx].type = p_type;
		curr_idx = UINT32_MAX;
		last_frame_mapped = UINT64_MAX;
	}

	void set_size(uint32_t p_idx, uint32_t p_size_bytes, bool p_is_storage) {
		set_size(p_idx, p_size_bytes, p_is_storage ? MultiUmaBufferType::STORAGE : MultiUmaBufferType::UNIFORM);
	}

	void set_uniform_size(uint32_t p_idx, uint32_t p_size_bytes) {
		set_size(p_idx, p_size_bytes, MultiUmaBufferType::UNIFORM);
	}

	void set_storage_size(uint32_t p_idx, uint32_t p_size_bytes) {
		set_size(p_idx, p_size_bytes, MultiUmaBufferType::STORAGE);
	}

	void set_vertex_size(uint32_t p_idx, uint32_t p_size_bytes) {
		set_size(p_idx, p_size_bytes, MultiUmaBufferType::VERTEX);
	}

	uint32_t get_size(uint32_t p_idx) const { return buffer_info[p_idx].size_bytes; }

	// Gets the raw buffer. Use with care.
	// If you call this function, make sure to have called prepare_for_upload() first.
	// Do not call _get() then prepare_for_upload().
	RID _get(uint32_t p_idx) {
		return buffers[curr_idx * NUM_BUFFERS + p_idx];
	}

	/**
	 * @param p_append	True if you wish to append more data to existing buffer.
	 * @return			False if it's possible to append. True if the internal buffer changed.
	 */
	bool prepare_for_map(bool p_append) {
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();
		const uint64_t frames_drawn = rd->get_frames_drawn();

		if (last_frame_mapped == frames_drawn) {
			if (!p_append) {
				++curr_idx;
			}
		} else {
			p_append = false;
			curr_idx = 0u;
			if (max_extra_buffers != UINT32_MAX) {
				shrink_to_max_extra_buffers();
			}
		}
		last_frame_mapped = frames_drawn;
		if (curr_idx * NUM_BUFFERS >= buffers.size()) {
			push();
		}

#ifdef DEV_ENABLED
		if (!p_append) {
			for (size_t i = 0u; i < NUM_BUFFERS; ++i) {
				can_upload[i] = true;
			}
		}
#endif
		return !p_append;
	}

	void prepare_for_upload() {
		prepare_for_map(false);
	}

	void *map_raw_for_upload(uint32_t p_idx) {
#ifdef DEV_ENABLED
		DEV_ASSERT(can_upload[p_idx] && "Forgot to prepare_for_upload first! Or called get_for_upload/upload() twice.");
		can_upload[p_idx] = false;
#endif
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();
		return rd->buffer_persistent_map_advance(buffers[curr_idx * NUM_BUFFERS + p_idx]);
	}

	RID get_for_upload(uint32_t p_idx) {
#ifdef DEV_ENABLED
		DEV_ASSERT(can_upload[p_idx] && "Forgot to prepare_for_upload first! Or called get_for_upload/upload() twice.");
		can_upload[p_idx] = false;
#endif
		return buffers[curr_idx * NUM_BUFFERS + p_idx];
	}

	void upload(uint32_t p_idx, const void *p_src_data, uint32_t p_size_bytes) {
#ifdef DEV_ENABLED
		DEV_ASSERT(can_upload[p_idx] && "Forgot to prepare_for_upload first! Or called get_for_upload/upload() twice.");
		can_upload[p_idx] = false;
#endif
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();
		rd->buffer_update(buffers[curr_idx * NUM_BUFFERS + p_idx], 0, p_size_bytes, p_src_data, true);
	}
};
