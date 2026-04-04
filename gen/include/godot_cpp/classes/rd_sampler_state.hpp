/**************************************************************************/
/*  rd_sampler_state.hpp                                                  */
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
#include <godot_cpp/classes/rendering_device.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RDSamplerState : public RefCounted {
	GDEXTENSION_CLASS(RDSamplerState, RefCounted)

public:
	void set_mag_filter(RenderingDevice::SamplerFilter p_member);
	RenderingDevice::SamplerFilter get_mag_filter() const;
	void set_min_filter(RenderingDevice::SamplerFilter p_member);
	RenderingDevice::SamplerFilter get_min_filter() const;
	void set_mip_filter(RenderingDevice::SamplerFilter p_member);
	RenderingDevice::SamplerFilter get_mip_filter() const;
	void set_repeat_u(RenderingDevice::SamplerRepeatMode p_member);
	RenderingDevice::SamplerRepeatMode get_repeat_u() const;
	void set_repeat_v(RenderingDevice::SamplerRepeatMode p_member);
	RenderingDevice::SamplerRepeatMode get_repeat_v() const;
	void set_repeat_w(RenderingDevice::SamplerRepeatMode p_member);
	RenderingDevice::SamplerRepeatMode get_repeat_w() const;
	void set_lod_bias(float p_member);
	float get_lod_bias() const;
	void set_use_anisotropy(bool p_member);
	bool get_use_anisotropy() const;
	void set_anisotropy_max(float p_member);
	float get_anisotropy_max() const;
	void set_enable_compare(bool p_member);
	bool get_enable_compare() const;
	void set_compare_op(RenderingDevice::CompareOperator p_member);
	RenderingDevice::CompareOperator get_compare_op() const;
	void set_min_lod(float p_member);
	float get_min_lod() const;
	void set_max_lod(float p_member);
	float get_max_lod() const;
	void set_border_color(RenderingDevice::SamplerBorderColor p_member);
	RenderingDevice::SamplerBorderColor get_border_color() const;
	void set_unnormalized_uvw(bool p_member);
	bool get_unnormalized_uvw() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

