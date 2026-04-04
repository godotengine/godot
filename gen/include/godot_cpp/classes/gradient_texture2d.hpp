/**************************************************************************/
/*  gradient_texture2d.hpp                                                */
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
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Gradient;

class GradientTexture2D : public Texture2D {
	GDEXTENSION_CLASS(GradientTexture2D, Texture2D)

public:
	enum Fill {
		FILL_LINEAR = 0,
		FILL_RADIAL = 1,
		FILL_SQUARE = 2,
	};

	enum Repeat {
		REPEAT_NONE = 0,
		REPEAT = 1,
		REPEAT_MIRROR = 2,
	};

	void set_gradient(const Ref<Gradient> &p_gradient);
	Ref<Gradient> get_gradient() const;
	void set_width(int32_t p_width);
	void set_height(int32_t p_height);
	void set_use_hdr(bool p_enabled);
	bool is_using_hdr() const;
	void set_fill(GradientTexture2D::Fill p_fill);
	GradientTexture2D::Fill get_fill() const;
	void set_fill_from(const Vector2 &p_fill_from);
	Vector2 get_fill_from() const;
	void set_fill_to(const Vector2 &p_fill_to);
	Vector2 get_fill_to() const;
	void set_repeat(GradientTexture2D::Repeat p_repeat);
	GradientTexture2D::Repeat get_repeat() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Texture2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GradientTexture2D::Fill);
VARIANT_ENUM_CAST(GradientTexture2D::Repeat);

