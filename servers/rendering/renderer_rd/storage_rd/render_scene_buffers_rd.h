/*************************************************************************/
/*  render_scene_buffers_rd.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef RENDER_SCENE_BUFFERS_RD_H
#define RENDER_SCENE_BUFFERS_RD_H

#include "core/templates/hash_map.h"
#include "servers/rendering/renderer_rd/effects/vrs.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_buffer_custom_data_rd.h"
#include "servers/rendering/renderer_scene.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/storage/render_scene_buffers.h"

// These can be retired in due time
#include "servers/rendering/renderer_rd/cluster_builder_rd.h"
#include "servers/rendering/renderer_rd/effects/ss_effects.h"
#include "servers/rendering/renderer_rd/environment/fog.h"

#define RB_SCOPE_BUFFERS SNAME("render_buffers")
#define RB_SCOPE_VRS SNAME("VRS")

#define RB_TEXTURE SNAME("texture")
#define RB_TEX_COLOR SNAME("color")
#define RB_TEX_COLOR_MSAA SNAME("color_msaa")
#define RB_TEX_DEPTH SNAME("depth")
#define RB_TEX_DEPTH_MSAA SNAME("depth_msaa")
#define RB_TEX_VELOCITY SNAME("velocity")
#define RB_TEX_VELOCITY_MSAA SNAME("velocity_msaa")

#define RB_TEX_BLUR_0 SNAME("blur_0")
#define RB_TEX_BLUR_1 SNAME("blur_1")
#define RB_TEX_HALF_BLUR SNAME("half_blur") // only for raster!

#define RB_TEX_BACK_DEPTH SNAME("back_depth")

class RenderSceneBuffersRD : public RenderSceneBuffers {
	GDCLASS(RenderSceneBuffersRD, RenderSceneBuffers);

private:
	bool can_be_storage = true;
	uint32_t max_cluster_elements = 512;
	RD::DataFormat base_data_format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	RendererRD::SSEffects *sse = nullptr;
	RendererRD::VRS *vrs = nullptr;
	uint64_t auto_exposure_version = 1;

	// Our render target represents our final destination that we display on screen.
	RID render_target;
	Size2i target_size = Size2i(0, 0);
	uint32_t view_count = 1;

	// The internal size of the textures we render 3D to in case we render at a lower resolution and upscale
	Size2i internal_size = Size2i(0, 0);
	float fsr_sharpness = 0.2f;

	// Aliassing settings
	RS::ViewportMSAA msaa_3d = RS::VIEWPORT_MSAA_DISABLED;
	RS::ViewportScreenSpaceAA screen_space_aa = RS::VIEWPORT_SCREEN_SPACE_AA_DISABLED;
	bool use_taa = false;
	bool use_debanding = false;

	// Named Textures

	struct NTKey {
		StringName context;
		StringName buffer_name;

		bool operator==(const NTKey &p_val) const {
			return (context == p_val.context) && (buffer_name == p_val.buffer_name);
		}

		static uint32_t hash(const NTKey &p_val) {
			// FIXME, properly hash two stringnames together
			uint32_t h = p_val.context.hash();
			h = hash_murmur3_one_32(p_val.buffer_name.hash(), h);
			return hash_fmix32(h);
		}

		NTKey() {}
		NTKey(const StringName p_context, const StringName p_texture_name) {
			context = p_context;
			buffer_name = p_texture_name;
		}
	};

	struct NamedTexture {
		// Cache the data used to create our texture
		RD::TextureFormat format;
		bool is_unique; // If marked as unique, we return it into our pool

		// Our texture objects, slices are lazy (i.e. only created when requested).
		RID texture;
		Vector<RID> slices;
		Vector<Size2i> sizes;
	};

	mutable HashMap<NTKey, NamedTexture, NTKey> named_textures;
	void update_sizes(NamedTexture &p_named_texture);
	void free_named_texture(NamedTexture &p_named_texture);

	// Data buffers
	mutable HashMap<StringName, Ref<RenderBufferCustomDataRD>> data_buffers;

protected:
	static void _bind_methods();

public:
	RenderSceneBuffersRD();
	virtual ~RenderSceneBuffersRD();

	// info from our renderer
	void set_can_be_storage(const bool p_can_be_storage) { can_be_storage = p_can_be_storage; }
	void set_max_cluster_elements(const uint32_t p_max_elements) { max_cluster_elements = p_max_elements; }
	void set_base_data_format(const RD::DataFormat p_base_data_format) { base_data_format = p_base_data_format; }
	RD::DataFormat get_base_data_format() const { return base_data_format; }
	void set_sseffects(RendererRD::SSEffects *p_ss_effects) { sse = p_ss_effects; }
	void set_vrs(RendererRD::VRS *p_vrs) { vrs = p_vrs; }

	void cleanup();
	virtual void configure(RID p_render_target, const Size2i p_internal_size, const Size2i p_target_size, float p_fsr_sharpness, float p_texture_mipmap_bias, RS::ViewportMSAA p_msaa_3d, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count) override;
	virtual void set_fsr_sharpness(float p_fsr_sharpness) override;
	virtual void set_texture_mipmap_bias(float p_texture_mipmap_bias) override;
	virtual void set_use_debanding(bool p_use_debanding) override;

	// Named Textures

	bool has_texture(const StringName &p_context, const StringName &p_texture_name) const;
	RID create_texture(const StringName &p_context, const StringName &p_texture_name, const RD::DataFormat p_data_format, const uint32_t p_usage_bits, const RD::TextureSamples p_texture_samples = RD::TEXTURE_SAMPLES_1, const Size2i p_size = Size2i(0, 0), const uint32_t p_layers = 0, const uint32_t p_mipmaps = 1, bool p_unique = true);
	RID create_texture_from_format(const StringName &p_context, const StringName &p_texture_name, const RD::TextureFormat &p_texture_format, RD::TextureView p_view = RD::TextureView(), bool p_unique = true);
	RID create_texture_view(const StringName &p_context, const StringName &p_texture_name, const StringName p_view_name, RD::TextureView p_view = RD::TextureView());
	RID get_texture(const StringName &p_context, const StringName &p_texture_name) const;
	const RD::TextureFormat get_texture_format(const StringName &p_context, const StringName &p_texture_name) const;
	RID get_texture_slice(const StringName &p_context, const StringName &p_texture_name, const uint32_t p_layer, const uint32_t p_mipmap);
	Size2i get_texture_slice_size(const StringName &p_context, const StringName &p_texture_name, const uint32_t p_layer, const uint32_t p_mipmap);

	void clear_context(const StringName &p_context);

	// Allocate shared buffers
	void allocate_blur_textures();

	// Custom data
	bool has_custom_data(const StringName &p_name);
	void set_custom_data(const StringName &p_name, Ref<RenderBufferCustomDataRD> p_data);
	Ref<RenderBufferCustomDataRD> get_custom_data(const StringName &p_name) const;

	// Getters

	_FORCE_INLINE_ RID get_render_target() const { return render_target; }
	_FORCE_INLINE_ uint32_t get_view_count() const { return view_count; }
	_FORCE_INLINE_ Size2i get_internal_size() const { return internal_size; }
	_FORCE_INLINE_ Size2i get_target_size() const { return target_size; }
	_FORCE_INLINE_ float get_fsr_sharpness() const { return fsr_sharpness; }
	_FORCE_INLINE_ RS::ViewportMSAA get_msaa_3d() const { return msaa_3d; }
	_FORCE_INLINE_ RS::ViewportScreenSpaceAA get_screen_space_aa() const { return screen_space_aa; }
	_FORCE_INLINE_ bool get_use_taa() const { return use_taa; }
	_FORCE_INLINE_ bool get_use_debanding() const { return use_debanding; }

	uint64_t get_auto_exposure_version() const { return auto_exposure_version; }
	void set_auto_exposure_version(const uint64_t p_auto_exposure_version) { auto_exposure_version = p_auto_exposure_version; }

	// For our internal textures we provide some easy access methods.

	_FORCE_INLINE_ RID get_internal_texture() const {
		return get_texture(RB_SCOPE_BUFFERS, RB_TEX_COLOR);
	}
	_FORCE_INLINE_ RID get_internal_texture(const uint32_t p_layer) {
		return get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_COLOR, p_layer, 0);
	}

	_FORCE_INLINE_ RID get_depth_texture() const {
		return get_texture(RB_SCOPE_BUFFERS, RB_TEX_DEPTH);
	}
	_FORCE_INLINE_ RID get_depth_texture(const uint32_t p_layer) {
		return get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_DEPTH, p_layer, 0);
	}

	// back buffer (color)
	RID get_back_buffer_texture() const { return has_texture(RB_SCOPE_BUFFERS, RB_TEX_BLUR_0) ? get_texture(RB_SCOPE_BUFFERS, RB_TEX_BLUR_0) : RID(); } // We (re)use our blur texture here.

	// Velocity, currently only used by TAA (Clustered) but we'll be using this in other places soon too.

	void ensure_velocity();
	bool has_velocity_buffer(bool p_has_msaa) { return has_texture(RB_SCOPE_BUFFERS, p_has_msaa ? RB_TEX_VELOCITY_MSAA : RB_TEX_VELOCITY); }
	RID get_velocity_buffer(bool p_get_msaa);
	RID get_velocity_buffer(bool p_get_msaa, uint32_t p_layer) { return get_texture_slice(RB_SCOPE_BUFFERS, p_get_msaa ? RB_TEX_VELOCITY_MSAA : RB_TEX_VELOCITY, p_layer, 0); }

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Everything after this needs to be re-evaluated, this is all old implementation

	ClusterBuilderRD *cluster_builder = nullptr;

	struct WeightBuffers {
		RID weight;
		RID fb; // FB with both texture and weight writing into one level lower
	};

	// 2 full size, 2 half size
	WeightBuffers weight_buffers[4]; // Only used in raster

	struct Luminance {
		Vector<RID> reduce;
		RID current;

		// used only on mobile renderer
		Vector<RID> fb;
		RID current_fb;
	} luminance;

	struct SSEffects {
		RID linear_depth;
		Vector<RID> linear_depth_slices;

		RID downsample_uniform_set;

		Projection last_frame_projection;
		Transform3D last_frame_transform;

		RendererRD::SSEffects::SSAORenderBuffers ssao;
		RendererRD::SSEffects::SSILRenderBuffers ssil;
	} ss_effects;

	RendererRD::SSEffects::SSRRenderBuffers ssr;

	RID get_ao_texture() const { return ss_effects.ssao.ao_final; }
	RID get_ssil_texture() const { return ss_effects.ssil.ssil_final; }
};

#endif // RENDER_SCENE_BUFFERS_RD_H
