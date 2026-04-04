/**************************************************************************/
/*  noise_texture2d.hpp                                                   */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Gradient;
class Noise;

class NoiseTexture2D : public Texture2D {
	GDEXTENSION_CLASS(NoiseTexture2D, Texture2D)

public:
	void set_width(int32_t p_width);
	void set_height(int32_t p_height);
	void set_generate_mipmaps(bool p_invert);
	bool is_generating_mipmaps() const;
	void set_noise(const Ref<Noise> &p_noise);
	Ref<Noise> get_noise();
	void set_color_ramp(const Ref<Gradient> &p_gradient);
	Ref<Gradient> get_color_ramp() const;
	void set_seamless(bool p_seamless);
	bool get_seamless();
	void set_invert(bool p_invert);
	bool get_invert() const;
	void set_in_3d_space(bool p_enable);
	bool is_in_3d_space() const;
	void set_as_normal_map(bool p_as_normal_map);
	bool is_normal_map();
	void set_normalize(bool p_normalize);
	bool is_normalized() const;
	void set_seamless_blend_skirt(float p_seamless_blend_skirt);
	float get_seamless_blend_skirt();
	void set_bump_strength(float p_bump_strength);
	float get_bump_strength();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Texture2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

