/**************************************************************************/
/*  update_once_per_frame_uniform_buffer_pool.h                           */
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

class UpdateOncePerFrameUniformBufferPool {
	LocalVector<RID> buffers;
	uint32_t size_bytes = 0;
	uint32_t used_buffer_count = 0;
	uint64_t last_frame_acquired = 0;

public:
	_FORCE_INLINE_ void set_size_bytes(uint32_t p_size_bytes) {
		size_bytes = p_size_bytes;
	}

	_FORCE_INLINE_ uint32_t get_next() {
		uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
		if (frames_drawn != last_frame_acquired) {
			used_buffer_count = 0;
			last_frame_acquired = frames_drawn;
		}

		uint32_t index = used_buffer_count;
		++used_buffer_count;

		if (buffers.size() < used_buffer_count) {
			uint32_t from = buffers.size();
			buffers.resize(used_buffer_count);

			for (uint32_t i = from; i < used_buffer_count; i++) {
				buffers[i] = RD::get_singleton()->uniform_buffer_create(size_bytes, Span<uint8_t>(), RD::BUFFER_CREATION_UPDATE_ONCE_PER_FRAME_BIT);
			}
		}

		return index;
	}

	_FORCE_INLINE_ RID get_next_rid() {
		return buffers[get_next()];
	}

	_FORCE_INLINE_ RID get_at_index(uint32_t p_index) const {
		return buffers[p_index];
	}

	_FORCE_INLINE_ void clear() {
		for (RID buffer : buffers) {
			RD::get_singleton()->free_rid(buffer);
		}
		buffers.clear();
		used_buffer_count = 0;
	}

	~UpdateOncePerFrameUniformBufferPool() {
		for (RID buffer : buffers) {
			RD::get_singleton()->free_rid(buffer);
		}
	}
};
