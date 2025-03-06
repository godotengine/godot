/**************************************************************************/
/*  push_constants_emu.h                                                  */
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

#include "servers/rendering_server.h"

class PushConstantsEmuBase {
public:
	struct ParamsUniform {
		RID buffer;
		RID set;
	};

protected:
	LocalVector<ParamsUniform> params_uniform;
	uint32_t curr_idx = 0u;
	const uint32_t max_extra_buffers;
#ifdef DEBUG_ENABLED
	const char *debug_name;
#endif

	PushConstantsEmuBase(uint32_t p_max_extra_buffers, const char *p_debug_name) :
			max_extra_buffers(p_max_extra_buffers)
#ifdef DEBUG_ENABLED
			,
			debug_name(p_debug_name)
#endif
	{
	}

#ifdef DEV_ENABLED
	~PushConstantsEmuBase() {
		DEV_ASSERT(params_uniform.is_empty() && "Forgot to call uninit()!");
	}
#endif

	void init_base() {
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();
		rd->_register_push_constant_emu(this);
	}

	void uninit_base() {
		if (is_print_verbose_enabled()) {
			print_line("PushConstantsEmu '"
#ifdef DEBUG_ENABLED
					+ String(debug_name) +
#else
					   "{DEBUG_ENABLED unavailable}"
#endif
					"' used a total of " + itos(params_uniform.size()) +
					" buffers. A large number may indicate a waste of VRAM and can be brought down by tweaking MAX_EXTRA_BUFFERS for this buffer.");
		}

		RenderingDevice *rd = RD::RenderingDevice::get_singleton();

		rd->_unregister_push_constant_emu(this);

		for (const ParamsUniform &pu : params_uniform) {
			if (pu.set.is_valid()) {
				rd->free(pu.set);
			}
			if (pu.buffer.is_valid()) {
				rd->free(pu.buffer);
			}
		}

		params_uniform.clear();
	}

	void shrink_to_max_extra_buffers() {
		DEV_ASSERT(curr_idx == 0u && "This function can only be called after reset and before being upload_and_advance again!");

		RenderingDevice *rd = RD::RenderingDevice::get_singleton();

		uint32_t elem_count = params_uniform.size();

		if (elem_count > max_extra_buffers) {
			if (is_print_verbose_enabled()) {
				print_line("PushConstantsEmu '"
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
			if (params_uniform[elem_count].set.is_valid()) {
				rd->free(params_uniform[elem_count].set);
			}
			if (params_uniform[elem_count].buffer.is_valid()) {
				rd->free(params_uniform[elem_count].buffer);
			}
			params_uniform.remove_at(elem_count);
		}
	}

public:
	void _reset() {
		curr_idx = 0u;
		if (max_extra_buffers != UINT32_MAX) {
			shrink_to_max_extra_buffers();
		}
	}
};

/// See MultiUmaBuffer documentation. This is extremely similar, but specifically tailored for PushConstants.
template <typename T, uint32_t SET_IDX = 1u, uint32_t MAX_EXTRA_BUFFERS = UINT32_MAX>
class PushConstantsEmu : public PushConstantsEmuBase {
private:
	RID shader;

	void push() {
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();

		ParamsUniform pu;
		pu.buffer = rd->uniform_buffer_create(sizeof(T), Vector<uint8_t>(), RD::BUFFER_CREATION_DYNAMIC_PERSISTENT_BIT);

		Vector<RD::Uniform> params_uniforms;
		RD::Uniform u;
		u.binding = 0;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC;
		u.append_id(pu.buffer);
		params_uniforms.push_back(u);

		pu.set = rd->uniform_set_create(params_uniforms, shader, SET_IDX);

		params_uniform.push_back(pu);
	}

public:
	PushConstantsEmu(const char *p_debug_name) :
			PushConstantsEmuBase(MAX_EXTRA_BUFFERS, p_debug_name) {}

	void init(RID p_shader) {
		init_base();
		shader = p_shader;
	}

	void uninit() {
		shader = RID();
		uninit_base();
	}

	static uint32_t set_idx() { return SET_IDX; }

	ParamsUniform upload_and_advance(const T &p_src_data) {
		if (curr_idx >= params_uniform.size()) {
			push();
		}

		RD::RenderingDevice::get_singleton()->buffer_update(params_uniform[curr_idx].buffer, 0, sizeof(T), &p_src_data, true);

		return params_uniform[curr_idx++];
	}
};

/// See PushConstantsEmu. The difference is that PushConstantsEmu creates the set (which is exclusively
/// used for the push constant) while this version can be used to share the emulated push constants with
/// other sets.
///
/// This code expects that class S implements _create_push_constant_uniform_set(RID) like this:
///
///		class MyEffect
///		{
///		public:
///			RID _create_push_constant_uniform_set(RID buffer) {
///				RD::Uniform u;
///				u.binding = ...;
///				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC;
///				u.append_id(buffer);
///				return rd->uniform_set_create(...);
///			}
///		};
///
///		PushConstantsEmuEmbedded<MyParams, MyEffect> buffer =
///										PushConstantsEmuEmbedded<MyParams, MyEffect>("debug name");
template <typename T, typename S, uint32_t MAX_EXTRA_BUFFERS = UINT32_MAX>
class PushConstantsEmuEmbedded : public PushConstantsEmuBase {
private:
#ifdef DEV_ENABLED
	bool initialized = false;
#endif

	void push(S *p_embed_owner) {
		RenderingDevice *rd = RD::RenderingDevice::get_singleton();

		ParamsUniform pu;
		pu.buffer = rd->uniform_buffer_create(sizeof(T), Vector<uint8_t>(), RD::BUFFER_CREATION_DYNAMIC_PERSISTENT_BIT);
		pu.set = p_embed_owner->_create_push_constant_uniform_set(pu.buffer);
		params_uniform.push_back(pu);
	}

public:
	PushConstantsEmuEmbedded(const char *p_debug_name) :
			PushConstantsEmuBase(MAX_EXTRA_BUFFERS, p_debug_name) {}

	void init() {
#ifdef DEV_ENABLED
		initialized = true;
#endif
		init_base();
	}

	void uninit() {
#ifdef DEV_ENABLED
		initialized = false;
#endif
		uninit_base();
	}

	ParamsUniform upload_and_advance(const T &p_src_data, S *p_embed_owner) {
		DEV_ASSERT(initialized);

		if (curr_idx >= params_uniform.size()) {
			push(p_embed_owner);
		}

		RD::RenderingDevice::get_singleton()->buffer_update(params_uniform[curr_idx].buffer, 0, sizeof(T), &p_src_data, true);

		return params_uniform[curr_idx++];
	}
};
