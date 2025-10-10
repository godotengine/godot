/**************************************************************************/
/*  ffx_common.h                                                          */
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
#include "servers/rendering/renderer_rd/shader_rd.h"
#include "thirdparty/amd-ffx/ffx_interface.h"

#define FFX_UBO_RING_BUFFER_SIZE 144

namespace RendererRD {
class FFXCommonContext {
public:
	enum ResourceID : uint32_t {
		RESOURCE_ID_DYNAMIC = 0xFFFFFFFF
	};

	struct Resources {
		LocalVector<RID> rids;
		LocalVector<LocalVector<RID>> mip_slice_rids;
		LocalVector<uint32_t> ids;
		LocalVector<FfxResourceDescription> descriptions;
		LocalVector<uint32_t> dynamic_list;
		LocalVector<uint32_t> free_list;

		uint32_t add(RID p_rid, bool p_dynamic, uint32_t p_id, FfxResourceDescription p_description) {
			uint32_t ret_index;
			if (free_list.is_empty()) {
				ret_index = rids.size();
				uint32_t new_size = ret_index + 1;
				rids.resize(new_size);
				mip_slice_rids.resize(new_size);
				ids.resize(new_size);
				descriptions.resize(new_size);
			} else {
				uint32_t end_index = free_list.size() - 1;
				ret_index = free_list[end_index];
				free_list.resize(end_index);
			}

			rids[ret_index] = p_rid;
			mip_slice_rids[ret_index].clear();
			ids[ret_index] = p_id;
			descriptions[ret_index] = p_description;

			if (p_dynamic) {
				dynamic_list.push_back(ret_index);
			}

			return ret_index;
		}

		void remove(uint32_t p_index) {
			DEV_ASSERT(p_index < rids.size());
			free_list.push_back(p_index);
			rids[p_index] = RID();
			mip_slice_rids[p_index].clear();
			ids[p_index] = 0;
			descriptions[p_index] = {};
			dynamic_list.erase(p_index);
		}

		uint32_t size() const {
			return rids.size();
		}
	};

	struct RootSignature {
		// Proxy structure to store the shader required by RD that uses the terminology used by the FSR2 API.
		RID shader_rid;
	};

	struct Pipeline {
		RID pipeline_rid;
	};

	struct Pass {
		ShaderRD *shader;
		RID shader_version;
		RootSignature root_signature;
		uint32_t shader_variant = 0;
		Pipeline pipeline;
		Vector<FfxResourceBinding> sampled_texture_bindings;
		Vector<FfxResourceBinding> storage_texture_bindings;
		Vector<FfxResourceBinding> uniform_bindings;
	};

	struct EffectContext {
		Pass passes[FFX_MAX_PASS_COUNT];
	};

	struct Device {
		RID point_clamp_sampler;
		RID linear_clamp_sampler;
		FfxDeviceCapabilities capabilities;
		EffectContext effect_contexts;
	} device;

	struct Scratch {
		Resources resources;
		LocalVector<FfxGpuJobDescription> gpu_jobs;
		// Uniform ring buffer
		RID ubo_ring_buffer[FFX_UBO_RING_BUFFER_SIZE];
		uint32_t ubo_ring_buffer_index = 0;
		// Staging buffer for constant buffer data.
		uint8_t* staging_constant_buffer;
		size_t staging_constant_buffer_base = 0;
		// Pointer to the device common to all contexts.
		// Static functions cannot access class members, so we store it here.
		FfxDevice device;
	};

	Scratch scratch;

	void init_device();
	void create_ffx_interface(FfxInterface* p_interface);
	static FfxResource get_resource_rd(RID *p_rid, const wchar_t *p_name);

	static FFXCommonContext *get_singleton() {
		if (singleton == nullptr) {
			singleton = memnew(RendererRD::FFXCommonContext);
			singleton->init_device();
		}
		return singleton;
	}

	static void free_singleton() {
		if (singleton) {
			memdelete(singleton);
		}
	}

	~FFXCommonContext();
private:
	static FFXCommonContext *singleton;
};
}
