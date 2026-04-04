/**************************************************************************/
/*  render_scene_buffers_extension.hpp                                    */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RenderSceneBuffersConfiguration;

class RenderSceneBuffersExtension : public RenderSceneBuffers {
	GDEXTENSION_CLASS(RenderSceneBuffersExtension, RenderSceneBuffers)

public:
	virtual void _configure(const Ref<RenderSceneBuffersConfiguration> &p_config);
	virtual void _set_fsr_sharpness(float p_fsr_sharpness);
	virtual void _set_texture_mipmap_bias(float p_texture_mipmap_bias);
	virtual void _set_anisotropic_filtering_level(int32_t p_anisotropic_filtering_level);
	virtual void _set_use_debanding(bool p_use_debanding);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RenderSceneBuffers::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_configure), decltype(&T::_configure)>) {
			BIND_VIRTUAL_METHOD(T, _configure, 3072623270);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_fsr_sharpness), decltype(&T::_set_fsr_sharpness)>) {
			BIND_VIRTUAL_METHOD(T, _set_fsr_sharpness, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_texture_mipmap_bias), decltype(&T::_set_texture_mipmap_bias)>) {
			BIND_VIRTUAL_METHOD(T, _set_texture_mipmap_bias, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_anisotropic_filtering_level), decltype(&T::_set_anisotropic_filtering_level)>) {
			BIND_VIRTUAL_METHOD(T, _set_anisotropic_filtering_level, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_use_debanding), decltype(&T::_set_use_debanding)>) {
			BIND_VIRTUAL_METHOD(T, _set_use_debanding, 2586408642);
		}
	}

public:
};

} // namespace godot

