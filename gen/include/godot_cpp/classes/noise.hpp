/**************************************************************************/
/*  noise.hpp                                                             */
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
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Image;
struct Vector2;
struct Vector3;

class Noise : public Resource {
	GDEXTENSION_CLASS(Noise, Resource)

public:
	float get_noise_1d(float p_x) const;
	float get_noise_2d(float p_x, float p_y) const;
	float get_noise_2dv(const Vector2 &p_v) const;
	float get_noise_3d(float p_x, float p_y, float p_z) const;
	float get_noise_3dv(const Vector3 &p_v) const;
	Ref<Image> get_image(int32_t p_width, int32_t p_height, bool p_invert = false, bool p_in_3d_space = false, bool p_normalize = true) const;
	Ref<Image> get_seamless_image(int32_t p_width, int32_t p_height, bool p_invert = false, bool p_in_3d_space = false, float p_skirt = 0.1, bool p_normalize = true) const;
	TypedArray<Ref<Image>> get_image_3d(int32_t p_width, int32_t p_height, int32_t p_depth, bool p_invert = false, bool p_normalize = true) const;
	TypedArray<Ref<Image>> get_seamless_image_3d(int32_t p_width, int32_t p_height, int32_t p_depth, bool p_invert = false, float p_skirt = 0.1, bool p_normalize = true) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

