/**************************************************************************/
/*  render_scene_buffers_configuration.hpp                                */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RenderSceneBuffersConfiguration : public RefCounted {
	GDEXTENSION_CLASS(RenderSceneBuffersConfiguration, RefCounted)

public:
	RID get_render_target() const;
	void set_render_target(const RID &p_render_target);
	Vector2i get_internal_size() const;
	void set_internal_size(const Vector2i &p_internal_size);
	Vector2i get_target_size() const;
	void set_target_size(const Vector2i &p_target_size);
	uint32_t get_view_count() const;
	void set_view_count(uint32_t p_view_count);
	RenderingServer::ViewportScaling3DMode get_scaling_3d_mode() const;
	void set_scaling_3d_mode(RenderingServer::ViewportScaling3DMode p_scaling_3d_mode);
	RenderingServer::ViewportMSAA get_msaa_3d() const;
	void set_msaa_3d(RenderingServer::ViewportMSAA p_msaa_3d);
	RenderingServer::ViewportScreenSpaceAA get_screen_space_aa() const;
	void set_screen_space_aa(RenderingServer::ViewportScreenSpaceAA p_screen_space_aa);
	float get_fsr_sharpness() const;
	void set_fsr_sharpness(float p_fsr_sharpness);
	float get_texture_mipmap_bias() const;
	void set_texture_mipmap_bias(float p_texture_mipmap_bias);
	RenderingServer::ViewportAnisotropicFiltering get_anisotropic_filtering_level() const;
	void set_anisotropic_filtering_level(RenderingServer::ViewportAnisotropicFiltering p_anisotropic_filtering_level);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

