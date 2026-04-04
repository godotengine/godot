/**************************************************************************/
/*  mobile_vr_interface.hpp                                               */
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
#include <godot_cpp/classes/xr_interface.hpp>
#include <godot_cpp/variant/rect2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MobileVRInterface : public XRInterface {
	GDEXTENSION_CLASS(MobileVRInterface, XRInterface)

public:
	void set_eye_height(double p_eye_height);
	double get_eye_height() const;
	void set_iod(double p_iod);
	double get_iod() const;
	void set_display_width(double p_display_width);
	double get_display_width() const;
	void set_display_to_lens(double p_display_to_lens);
	double get_display_to_lens() const;
	void set_offset_rect(const Rect2 &p_offset_rect);
	Rect2 get_offset_rect() const;
	void set_oversample(double p_oversample);
	double get_oversample() const;
	void set_k1(double p_k);
	double get_k1() const;
	void set_k2(double p_k);
	double get_k2() const;
	float get_vrs_min_radius() const;
	void set_vrs_min_radius(float p_radius);
	float get_vrs_strength() const;
	void set_vrs_strength(float p_strength);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRInterface::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

