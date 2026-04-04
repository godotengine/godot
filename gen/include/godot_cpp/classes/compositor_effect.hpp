/**************************************************************************/
/*  compositor_effect.hpp                                                 */
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
#include <godot_cpp/classes/resource.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RenderData;

class CompositorEffect : public Resource {
	GDEXTENSION_CLASS(CompositorEffect, Resource)

public:
	enum EffectCallbackType {
		EFFECT_CALLBACK_TYPE_PRE_OPAQUE = 0,
		EFFECT_CALLBACK_TYPE_POST_OPAQUE = 1,
		EFFECT_CALLBACK_TYPE_POST_SKY = 2,
		EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT = 3,
		EFFECT_CALLBACK_TYPE_POST_TRANSPARENT = 4,
		EFFECT_CALLBACK_TYPE_MAX = 5,
	};

	void set_enabled(bool p_enabled);
	bool get_enabled() const;
	void set_effect_callback_type(CompositorEffect::EffectCallbackType p_effect_callback_type);
	CompositorEffect::EffectCallbackType get_effect_callback_type() const;
	void set_access_resolved_color(bool p_enable);
	bool get_access_resolved_color() const;
	void set_access_resolved_depth(bool p_enable);
	bool get_access_resolved_depth() const;
	void set_needs_motion_vectors(bool p_enable);
	bool get_needs_motion_vectors() const;
	void set_needs_normal_roughness(bool p_enable);
	bool get_needs_normal_roughness() const;
	void set_needs_separate_specular(bool p_enable);
	bool get_needs_separate_specular() const;
	virtual void _render_callback(int32_t p_effect_callback_type, RenderData *p_render_data);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_render_callback), decltype(&T::_render_callback)>) {
			BIND_VIRTUAL_METHOD(T, _render_callback, 2153422729);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CompositorEffect::EffectCallbackType);

