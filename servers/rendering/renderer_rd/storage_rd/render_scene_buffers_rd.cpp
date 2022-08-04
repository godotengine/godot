/*************************************************************************/
/*  render_scene_buffers_rd.cpp                                          */
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

#include "render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"

RenderSceneBuffersRD::RenderSceneBuffersRD() {
}

RenderSceneBuffersRD::~RenderSceneBuffersRD() {
	cleanup();

	data_buffers.clear();

	// need to investigate if we can remove these things.
	if (cluster_builder) {
		memdelete(cluster_builder);
	}
}

void RenderSceneBuffersRD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_texture", "context", "name"), &RenderSceneBuffersRD::has_texture);
	// FIXME we can't pass RD::DataFormat, RD::TextureSamples and RD::TextureView in ClassDB, need to solve views differently...
	// ClassDB::bind_method(D_METHOD("create_texture", "context", "name", "data_format", "usage_bits", "texture_samples", "size", "layers", "mipmaps", "unique"), &RenderSceneBuffersRD::create_texture);
	// ClassDB::bind_method(D_METHOD("create_texture_from_format", "context", "name", "format", "view", "unique"), &RenderSceneBuffersRD::create_texture_from_format);
	// ClassDB::bind_method(D_METHOD("create_texture_view", "context", "name", "view_name", "view"), &RenderSceneBuffersRD::has_texture);
	ClassDB::bind_method(D_METHOD("get_texture", "context", "name"), &RenderSceneBuffersRD::get_texture);
	// ClassDB::bind_method(D_METHOD("get_texture_format", "context", "name"), &RenderSceneBuffersRD::get_texture_format);
	ClassDB::bind_method(D_METHOD("get_texture_slice", "context", "name", "layer", "mipmap"), &RenderSceneBuffersRD::get_texture_slice);
	ClassDB::bind_method(D_METHOD("get_texture_slice_size", "context", "name", "layer", "mipmap"), &RenderSceneBuffersRD::get_texture_slice_size);
	ClassDB::bind_method(D_METHOD("clear_context", "context"), &RenderSceneBuffersRD::clear_context);
}

void RenderSceneBuffersRD::update_sizes(NamedTexture &p_named_texture) {
	ERR_FAIL_COND(p_named_texture.texture.is_null());

	uint32_t size = p_named_texture.format.array_layers * p_named_texture.format.mipmaps;
	p_named_texture.sizes.resize(size);

	Size2i mipmap_size = Size2i(p_named_texture.format.width, p_named_texture.format.height);

	for (uint32_t mipmap = 0; mipmap < p_named_texture.format.mipmaps; mipmap++) {
		for (uint32_t layer = 0; layer < p_named_texture.format.array_layers; layer++) {
			uint32_t index = layer * p_named_texture.format.mipmaps + mipmap;

			p_named_texture.sizes.ptrw()[index] = mipmap_size;
		}

		mipmap_size.width = MAX(1, mipmap_size.width >> 1);
		mipmap_size.height = MAX(1, mipmap_size.height >> 1);
	}
}

void RenderSceneBuffersRD::free_named_texture(NamedTexture &p_named_texture) {
	if (p_named_texture.texture.is_valid()) {
		RD::get_singleton()->free(p_named_texture.texture);
	}
	p_named_texture.texture = RID();
	p_named_texture.slices.clear(); // slices should be freed automatically as dependents...
}

void RenderSceneBuffersRD::cleanup() {
	// Free our data buffers (but don't destroy them)
	for (KeyValue<StringName, Ref<RenderBufferCustomDataRD>> &E : data_buffers) {
		E.value->free_data();
	}

	// Clear our named textures
	for (KeyValue<NTKey, NamedTexture> &E : named_textures) {
		free_named_texture(E.value);
	}
	named_textures.clear();

	// old stuff, to be re-evaluated...

	for (int i = 0; i < luminance.fb.size(); i++) {
		RD::get_singleton()->free(luminance.fb[i]);
	}
	luminance.fb.clear();

	for (int i = 0; i < luminance.reduce.size(); i++) {
		RD::get_singleton()->free(luminance.reduce[i]);
	}
	luminance.reduce.clear();

	if (luminance.current_fb.is_valid()) {
		RD::get_singleton()->free(luminance.current_fb);
		luminance.current_fb = RID();
	}

	if (luminance.current.is_valid()) {
		RD::get_singleton()->free(luminance.current);
		luminance.current = RID();
	}

	if (ss_effects.linear_depth.is_valid()) {
		RD::get_singleton()->free(ss_effects.linear_depth);
		ss_effects.linear_depth = RID();
		ss_effects.linear_depth_slices.clear();
	}

	sse->ssao_free(ss_effects.ssao);
	sse->ssil_free(ss_effects.ssil);
	sse->ssr_free(ssr);
}

void RenderSceneBuffersRD::configure(RID p_render_target, const Size2i p_internal_size, const Size2i p_target_size, float p_fsr_sharpness, float p_texture_mipmap_bias, RS::ViewportMSAA p_msaa_3d, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count) {
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	ERR_FAIL_COND_MSG(p_view_count == 0, "Must have at least 1 view");

	target_size = p_target_size;
	internal_size = p_internal_size;

	// FIXME, right now we do this because only our clustered renderer supports FSR upscale
	// this does mean that with linear upscale if we use subpasses, we could get into trouble.
	if (!can_be_storage) {
		internal_size = target_size;
	}

	if (p_use_taa) {
		// Use negative mipmap LOD bias when TAA is enabled to compensate for loss of sharpness.
		// This restores sharpness in still images to be roughly at the same level as without TAA,
		// but moving scenes will still be blurrier.
		p_texture_mipmap_bias -= 0.5;
	}

	if (p_screen_space_aa == RS::VIEWPORT_SCREEN_SPACE_AA_FXAA) {
		// Use negative mipmap LOD bias when FXAA is enabled to compensate for loss of sharpness.
		// If both TAA and FXAA are enabled, combine their negative LOD biases together.
		p_texture_mipmap_bias -= 0.25;
	}

	material_storage->sampler_rd_configure_custom(p_texture_mipmap_bias);

	// need to check if we really need to do this here..
	RendererSceneRenderRD::get_singleton()->update_uniform_sets();

	render_target = p_render_target;
	fsr_sharpness = p_fsr_sharpness;
	msaa_3d = p_msaa_3d;
	screen_space_aa = p_screen_space_aa;
	use_taa = p_use_taa;
	use_debanding = p_use_debanding;
	view_count = p_view_count;

	/* may move this into our clustered renderer data object */
	if (can_be_storage) {
		if (cluster_builder == nullptr) {
			cluster_builder = memnew(ClusterBuilderRD);
		}
		cluster_builder->set_shared(RendererSceneRenderRD::get_singleton()->get_cluster_builder_shared());
	}

	// cleanout any old buffers we had.
	cleanup();

	// create our 3D render buffers
	{
		// Create our color buffer(s)
		uint32_t usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | (can_be_storage ? RD::TEXTURE_USAGE_STORAGE_BIT : 0) | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		usage_bits |= RD::TEXTURE_USAGE_INPUT_ATTACHMENT_BIT; // only needed when using subpasses in the mobile renderer

		// our internal texture should have MSAA support if applicable
		if (msaa_3d != RS::VIEWPORT_MSAA_DISABLED) {
			usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
		}

		create_texture(RB_SCOPE_BUFFERS, RB_TEX_COLOR, base_data_format, usage_bits);
	}

	// Create our depth buffer
	{
		// TODO If we have depth buffer supplied externally, pick this up

		RD::DataFormat format;
		uint32_t usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;

		if (msaa_3d == RS::VIEWPORT_MSAA_DISABLED) {
			format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, (RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT)) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
			usage_bits |= RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		} else {
			format = RD::DATA_FORMAT_R32_SFLOAT;
			usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		}

		create_texture(RB_SCOPE_BUFFERS, RB_TEX_DEPTH, format, usage_bits);
	}

	// VRS (note, our vrs object will only be set if VRS is supported)
	RID vrs_texture;
	RS::ViewportVRSMode vrs_mode = texture_storage->render_target_get_vrs_mode(p_render_target);
	if (vrs && vrs_mode != RS::VIEWPORT_VRS_DISABLED) {
		uint32_t usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_VRS_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		vrs_texture = create_texture(RB_SCOPE_VRS, RB_TEXTURE, RD::DATA_FORMAT_R8_UINT, usage_bits, RD::TEXTURE_SAMPLES_1, vrs->get_vrs_texture_size(internal_size));
	}

	for (KeyValue<StringName, Ref<RenderBufferCustomDataRD>> &E : data_buffers) {
		E.value->configure(this);
	}

	if (cluster_builder) {
		RID sampler = RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		cluster_builder->setup(internal_size, max_cluster_elements, get_depth_texture(), sampler, get_internal_texture());
	}
}

void RenderSceneBuffersRD::set_fsr_sharpness(float p_fsr_sharpness) {
	fsr_sharpness = p_fsr_sharpness;
}

void RenderSceneBuffersRD::set_texture_mipmap_bias(float p_texture_mipmap_bias) {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	material_storage->sampler_rd_configure_custom(p_texture_mipmap_bias);
}

void RenderSceneBuffersRD::set_use_debanding(bool p_use_debanding) {
	use_debanding = p_use_debanding;
}

// Named textures

bool RenderSceneBuffersRD::has_texture(const StringName &p_context, const StringName &p_texture_name) const {
	NTKey key(p_context, p_texture_name);

	return named_textures.has(key);
}

RID RenderSceneBuffersRD::create_texture(const StringName &p_context, const StringName &p_texture_name, const RD::DataFormat p_data_format, const uint32_t p_usage_bits, const RD::TextureSamples p_texture_samples, const Size2i p_size, const uint32_t p_layers, const uint32_t p_mipmaps, bool p_unique) {
	// Keep some useful data, we use default values when these are 0.
	Size2i size = p_size == Size2i(0, 0) ? internal_size : p_size;
	uint32_t layers = p_layers == 0 ? view_count : p_layers;
	uint32_t mipmaps = p_mipmaps == 0 ? 1 : p_mipmaps;

	// Create our texture
	RD::TextureFormat tf;
	tf.format = p_data_format;
	if (layers > 1) {
		tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
	}

	tf.width = size.x;
	tf.height = size.y;
	tf.depth = 1;
	tf.array_layers = layers;
	tf.mipmaps = mipmaps;
	tf.usage_bits = p_usage_bits;
	tf.samples = p_texture_samples;

	return create_texture_from_format(p_context, p_texture_name, tf, RD::TextureView(), p_unique);
}

RID RenderSceneBuffersRD::create_texture_from_format(const StringName &p_context, const StringName &p_texture_name, const RD::TextureFormat &p_texture_format, RD::TextureView p_view, bool p_unique) {
	// TODO p_unique, if p_unique is true, this is a texture that can be shared. This will be implemented later as an optimisation.

	NTKey key(p_context, p_texture_name);

	// check if this is a known texture
	if (named_textures.has(key)) {
		return named_textures[key].texture;
	}

	// Add a new entry..
	NamedTexture &named_texture = named_textures[key];
	named_texture.format = p_texture_format;
	named_texture.is_unique = p_unique;
	named_texture.texture = RD::get_singleton()->texture_create(p_texture_format, p_view);

	Array arr;
	arr.push_back(p_context);
	arr.push_back(p_texture_name);
	RD::get_singleton()->set_resource_name(named_texture.texture, String("RenderBuffer {0}/{1}").format(arr));

	update_sizes(named_texture);

	// The rest is lazy created..

	return named_texture.texture;
}

RID RenderSceneBuffersRD::create_texture_view(const StringName &p_context, const StringName &p_texture_name, const StringName p_view_name, RD::TextureView p_view) {
	NTKey view_key(p_context, p_view_name);

	// check if this is a known texture
	if (named_textures.has(view_key)) {
		return named_textures[view_key].texture;
	}

	NTKey key(p_context, p_texture_name);

	ERR_FAIL_COND_V(!named_textures.has(key), RID());

	NamedTexture &named_texture = named_textures[key];
	NamedTexture &view_texture = named_textures[view_key];

	view_texture.format = named_texture.format;
	view_texture.is_unique = named_texture.is_unique;

	view_texture.texture = RD::get_singleton()->texture_create_shared(p_view, named_texture.texture);

	Array arr;
	arr.push_back(p_context);
	arr.push_back(p_view_name);
	RD::get_singleton()->set_resource_name(view_texture.texture, String("RenderBuffer View {0}/{1}").format(arr));

	update_sizes(named_texture);

	return view_texture.texture;
}

RID RenderSceneBuffersRD::get_texture(const StringName &p_context, const StringName &p_texture_name) const {
	NTKey key(p_context, p_texture_name);

	ERR_FAIL_COND_V(!named_textures.has(key), RID());

	return named_textures[key].texture;
}

const RD::TextureFormat RenderSceneBuffersRD::get_texture_format(const StringName &p_context, const StringName &p_texture_name) const {
	NTKey key(p_context, p_texture_name);

	ERR_FAIL_COND_V(!named_textures.has(key), RD::TextureFormat());

	return named_textures[key].format;
}

RID RenderSceneBuffersRD::get_texture_slice(const StringName &p_context, const StringName &p_texture_name, const uint32_t p_layer, const uint32_t p_mipmap) {
	NTKey key(p_context, p_texture_name);

	// check if this is a known texture
	ERR_FAIL_COND_V(!named_textures.has(key), RID());
	NamedTexture &named_texture = named_textures[key];
	ERR_FAIL_COND_V(named_texture.texture.is_null(), RID());

	// check if we're in bounds
	ERR_FAIL_UNSIGNED_INDEX_V(p_layer, named_texture.format.array_layers, RID());
	ERR_FAIL_UNSIGNED_INDEX_V(p_mipmap, named_texture.format.mipmaps, RID());

	// if we don't have multiple layers or mipmaps, we can just return our texture as is
	if (named_texture.format.array_layers == 1 && named_texture.format.mipmaps == 1) {
		return named_texture.texture;
	}

	// get our index and make sure we have enough entries in our slices vector
	uint32_t index = p_layer * named_texture.format.mipmaps + p_mipmap;
	while (named_texture.slices.size() <= int(index)) {
		named_texture.slices.push_back(RID());
	}

	// create our slice if we don't have it already
	if (named_texture.slices[index].is_null()) {
		named_texture.slices.ptrw()[index] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), named_texture.texture, p_layer, p_mipmap);

		Array arr;
		arr.push_back(p_context);
		arr.push_back(p_texture_name);
		arr.push_back(itos(p_layer));
		arr.push_back(itos(p_mipmap));
		RD::get_singleton()->set_resource_name(named_texture.slices[index], String("RenderBuffer {0}/{1} slice {2}/{3}").format(arr));
	}

	// and return our slice
	return named_texture.slices[index];
}

Size2i RenderSceneBuffersRD::get_texture_slice_size(const StringName &p_context, const StringName &p_texture_name, const uint32_t p_layer, const uint32_t p_mipmap) {
	NTKey key(p_context, p_texture_name);

	// check if this is a known texture
	ERR_FAIL_COND_V(!named_textures.has(key), Size2i());
	NamedTexture &named_texture = named_textures[key];
	ERR_FAIL_COND_V(named_texture.texture.is_null(), Size2i());

	// check if we're in bounds
	ERR_FAIL_UNSIGNED_INDEX_V(p_layer, named_texture.format.array_layers, Size2i());
	ERR_FAIL_UNSIGNED_INDEX_V(p_mipmap, named_texture.format.mipmaps, Size2i());

	// get our index
	uint32_t index = p_layer * named_texture.format.mipmaps + p_mipmap;

	// and return our size
	return named_texture.sizes[index];
}

void RenderSceneBuffersRD::clear_context(const StringName &p_context) {
	Vector<NTKey> to_free; // free these

	// Find all entries for our context, we don't want to free them yet or our loop fails.
	for (KeyValue<NTKey, NamedTexture> &E : named_textures) {
		if (E.key.context == p_context) {
			to_free.push_back(E.key);
		}
	}

	// Now free these and remove them from our textures
	for (NTKey &key : to_free) {
		free_named_texture(named_textures[key]);
		named_textures.erase(key);
	}
}

// Allocate shared buffers
void RenderSceneBuffersRD::allocate_blur_textures() {
	if (has_texture(RB_SCOPE_BUFFERS, RB_TEX_BLUR_0)) {
		// already allocated...
		return;
	}

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(internal_size.x, internal_size.y, Image::FORMAT_RGBAH);

	uint32_t usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	if (can_be_storage) {
		usage_bits += RD::TEXTURE_USAGE_STORAGE_BIT;
	} else {
		usage_bits += RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	create_texture(RB_SCOPE_BUFFERS, RB_TEX_BLUR_0, base_data_format, usage_bits, RD::TEXTURE_SAMPLES_1, internal_size, view_count, mipmaps_required);
	create_texture(RB_SCOPE_BUFFERS, RB_TEX_BLUR_1, base_data_format, usage_bits, RD::TEXTURE_SAMPLES_1, Size2i(internal_size.x >> 1, internal_size.y >> 1), view_count, mipmaps_required - 1);

	// if !can_be_storage we need a half width version
	if (!can_be_storage) {
		create_texture(RB_SCOPE_BUFFERS, RB_TEX_HALF_BLUR, base_data_format, usage_bits, RD::TEXTURE_SAMPLES_1, Size2i(internal_size.x >> 1, internal_size.y), 1, mipmaps_required);
	}

	// TODO redo this:
	if (!can_be_storage) {
		// create 4 weight textures, 2 full size, 2 half size

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16_SFLOAT; // We could probably use DATA_FORMAT_R8_SNORM if we don't pre-multiply by blur_size but that depends on whether we can remove DEPTH_GAP
		tf.width = internal_size.x;
		tf.height = internal_size.y;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.array_layers = 1; // Our DOF effect handles one eye per turn
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		tf.mipmaps = 1;
		for (uint32_t i = 0; i < 4; i++) {
			// associated blur texture
			RID texture;
			if (i == 1) {
				texture = get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_BLUR_0, 0, 0);
			} else if (i == 2) {
				texture = get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_BLUR_1, 0, 0);
			} else if (i == 3) {
				texture = get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_BLUR_0, 0, 1);
			}

			// create weight texture
			weight_buffers[i].weight = RD::get_singleton()->texture_create(tf, RD::TextureView());

			// create frame buffer
			Vector<RID> fb;
			if (i != 0) {
				fb.push_back(texture);
			}
			fb.push_back(weight_buffers[i].weight);
			weight_buffers[i].fb = RD::get_singleton()->framebuffer_create(fb);

			if (i == 1) {
				// next 2 are half size
				tf.width = MAX(1u, tf.width >> 1);
				tf.height = MAX(1u, tf.height >> 1);
			}
		}
	}
}

// Data buffers

bool RenderSceneBuffersRD::has_custom_data(const StringName &p_name) {
	return data_buffers.has(p_name);
}

void RenderSceneBuffersRD::set_custom_data(const StringName &p_name, Ref<RenderBufferCustomDataRD> p_data) {
	if (p_data.is_valid()) {
		data_buffers[p_name] = p_data;
	} else if (has_custom_data(p_name)) {
		data_buffers.erase(p_name);
	}
}

Ref<RenderBufferCustomDataRD> RenderSceneBuffersRD::get_custom_data(const StringName &p_name) const {
	ERR_FAIL_COND_V(!data_buffers.has(p_name), Ref<RenderBufferCustomDataRD>());

	Ref<RenderBufferCustomDataRD> ret = data_buffers[p_name];

	return ret;
}

// Velocity texture.

void RenderSceneBuffersRD::ensure_velocity() {
	if (!has_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY)) {
		uint32_t usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

		if (msaa_3d != RS::VIEWPORT_MSAA_DISABLED) {
			uint32_t msaa_usage_bits = usage_bits | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

			const RD::TextureSamples ts[RS::VIEWPORT_MSAA_MAX] = {
				RD::TEXTURE_SAMPLES_1,
				RD::TEXTURE_SAMPLES_2,
				RD::TEXTURE_SAMPLES_4,
				RD::TEXTURE_SAMPLES_8,
			};

			create_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY_MSAA, RD::DATA_FORMAT_R16G16_SFLOAT, msaa_usage_bits, ts[msaa_3d]);
		}

		create_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY, RD::DATA_FORMAT_R16G16_SFLOAT, usage_bits);
	}
}

RID RenderSceneBuffersRD::get_velocity_buffer(bool p_get_msaa) {
	if (p_get_msaa) {
		if (!has_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY_MSAA)) {
			return RID();
		} else {
			return get_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY_MSAA);
		}
	} else {
		if (!has_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY)) {
			return RID();
		} else {
			return get_texture(RB_SCOPE_BUFFERS, RB_TEX_VELOCITY);
		}
	}
}
