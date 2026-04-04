/**************************************************************************/
/*  open_xr_composition_layer.hpp                                         */
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

#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class JavaObject;
class SubViewport;
struct Vector3;

class OpenXRCompositionLayer : public Node3D {
	GDEXTENSION_CLASS(OpenXRCompositionLayer, Node3D)

public:
	enum Filter {
		FILTER_NEAREST = 0,
		FILTER_LINEAR = 1,
		FILTER_CUBIC = 2,
	};

	enum MipmapMode {
		MIPMAP_MODE_DISABLED = 0,
		MIPMAP_MODE_NEAREST = 1,
		MIPMAP_MODE_LINEAR = 2,
	};

	enum Wrap {
		WRAP_CLAMP_TO_BORDER = 0,
		WRAP_CLAMP_TO_EDGE = 1,
		WRAP_REPEAT = 2,
		WRAP_MIRRORED_REPEAT = 3,
		WRAP_MIRROR_CLAMP_TO_EDGE = 4,
	};

	enum Swizzle {
		SWIZZLE_RED = 0,
		SWIZZLE_GREEN = 1,
		SWIZZLE_BLUE = 2,
		SWIZZLE_ALPHA = 3,
		SWIZZLE_ZERO = 4,
		SWIZZLE_ONE = 5,
	};

	void set_layer_viewport(SubViewport *p_viewport);
	SubViewport *get_layer_viewport() const;
	void set_use_android_surface(bool p_enable);
	bool get_use_android_surface() const;
	void set_android_surface_size(const Vector2i &p_size);
	Vector2i get_android_surface_size() const;
	void set_enable_hole_punch(bool p_enable);
	bool get_enable_hole_punch() const;
	void set_sort_order(int32_t p_order);
	int32_t get_sort_order() const;
	void set_alpha_blend(bool p_enabled);
	bool get_alpha_blend() const;
	Ref<JavaObject> get_android_surface();
	bool is_natively_supported() const;
	bool is_protected_content() const;
	void set_protected_content(bool p_protected_content);
	void set_min_filter(OpenXRCompositionLayer::Filter p_mode);
	OpenXRCompositionLayer::Filter get_min_filter() const;
	void set_mag_filter(OpenXRCompositionLayer::Filter p_mode);
	OpenXRCompositionLayer::Filter get_mag_filter() const;
	void set_mipmap_mode(OpenXRCompositionLayer::MipmapMode p_mode);
	OpenXRCompositionLayer::MipmapMode get_mipmap_mode() const;
	void set_horizontal_wrap(OpenXRCompositionLayer::Wrap p_mode);
	OpenXRCompositionLayer::Wrap get_horizontal_wrap() const;
	void set_vertical_wrap(OpenXRCompositionLayer::Wrap p_mode);
	OpenXRCompositionLayer::Wrap get_vertical_wrap() const;
	void set_red_swizzle(OpenXRCompositionLayer::Swizzle p_mode);
	OpenXRCompositionLayer::Swizzle get_red_swizzle() const;
	void set_green_swizzle(OpenXRCompositionLayer::Swizzle p_mode);
	OpenXRCompositionLayer::Swizzle get_green_swizzle() const;
	void set_blue_swizzle(OpenXRCompositionLayer::Swizzle p_mode);
	OpenXRCompositionLayer::Swizzle get_blue_swizzle() const;
	void set_alpha_swizzle(OpenXRCompositionLayer::Swizzle p_mode);
	OpenXRCompositionLayer::Swizzle get_alpha_swizzle() const;
	void set_max_anisotropy(float p_value);
	float get_max_anisotropy() const;
	void set_border_color(const Color &p_color);
	Color get_border_color() const;
	Vector2 intersects_ray(const Vector3 &p_origin, const Vector3 &p_direction) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRCompositionLayer::Filter);
VARIANT_ENUM_CAST(OpenXRCompositionLayer::MipmapMode);
VARIANT_ENUM_CAST(OpenXRCompositionLayer::Wrap);
VARIANT_ENUM_CAST(OpenXRCompositionLayer::Swizzle);

