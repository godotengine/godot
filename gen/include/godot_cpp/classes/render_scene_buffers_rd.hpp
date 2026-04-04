/**************************************************************************/
/*  render_scene_buffers_rd.hpp                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/render_scene_buffers.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RDTextureFormat;
class RDTextureView;
class StringName;

class RenderSceneBuffersRD : public RenderSceneBuffers {
	GDEXTENSION_CLASS(RenderSceneBuffersRD, RenderSceneBuffers)

public:
	bool has_texture(const StringName &p_context, const StringName &p_name) const;
	RID create_texture(const StringName &p_context, const StringName &p_name, RenderingDevice::DataFormat p_data_format, uint32_t p_usage_bits, RenderingDevice::TextureSamples p_texture_samples, const Vector2i &p_size, uint32_t p_layers, uint32_t p_mipmaps, bool p_unique, bool p_discardable);
	RID create_texture_from_format(const StringName &p_context, const StringName &p_name, const Ref<RDTextureFormat> &p_format, const Ref<RDTextureView> &p_view, bool p_unique);
	RID create_texture_view(const StringName &p_context, const StringName &p_name, const StringName &p_view_name, const Ref<RDTextureView> &p_view);
	RID get_texture(const StringName &p_context, const StringName &p_name) const;
	Ref<RDTextureFormat> get_texture_format(const StringName &p_context, const StringName &p_name) const;
	RID get_texture_slice(const StringName &p_context, const StringName &p_name, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_layers, uint32_t p_mipmaps);
	RID get_texture_slice_view(const StringName &p_context, const StringName &p_name, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_layers, uint32_t p_mipmaps, const Ref<RDTextureView> &p_view);
	Vector2i get_texture_slice_size(const StringName &p_context, const StringName &p_name, uint32_t p_mipmap);
	void clear_context(const StringName &p_context);
	RID get_color_texture(bool p_msaa = false);
	RID get_color_layer(uint32_t p_layer, bool p_msaa = false);
	RID get_depth_texture(bool p_msaa = false);
	RID get_depth_layer(uint32_t p_layer, bool p_msaa = false);
	RID get_velocity_texture(bool p_msaa = false);
	RID get_velocity_layer(uint32_t p_layer, bool p_msaa = false);
	RID get_render_target() const;
	uint32_t get_view_count() const;
	Vector2i get_internal_size() const;
	Vector2i get_target_size() const;
	RenderingServer::ViewportScaling3DMode get_scaling_3d_mode() const;
	float get_fsr_sharpness() const;
	RenderingServer::ViewportMSAA get_msaa_3d() const;
	RenderingDevice::TextureSamples get_texture_samples() const;
	RenderingServer::ViewportScreenSpaceAA get_screen_space_aa() const;
	bool get_use_taa() const;
	bool get_use_debanding() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RenderSceneBuffers::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

