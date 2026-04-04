/**************************************************************************/
/*  rd_pipeline_color_blend_state_attachment.hpp                          */
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

class RDPipelineColorBlendStateAttachment : public RefCounted {
	GDEXTENSION_CLASS(RDPipelineColorBlendStateAttachment, RefCounted)

public:
	void set_as_mix();
	void set_enable_blend(bool p_member);
	bool get_enable_blend() const;
	void set_src_color_blend_factor(RenderingDevice::BlendFactor p_member);
	RenderingDevice::BlendFactor get_src_color_blend_factor() const;
	void set_dst_color_blend_factor(RenderingDevice::BlendFactor p_member);
	RenderingDevice::BlendFactor get_dst_color_blend_factor() const;
	void set_color_blend_op(RenderingDevice::BlendOperation p_member);
	RenderingDevice::BlendOperation get_color_blend_op() const;
	void set_src_alpha_blend_factor(RenderingDevice::BlendFactor p_member);
	RenderingDevice::BlendFactor get_src_alpha_blend_factor() const;
	void set_dst_alpha_blend_factor(RenderingDevice::BlendFactor p_member);
	RenderingDevice::BlendFactor get_dst_alpha_blend_factor() const;
	void set_alpha_blend_op(RenderingDevice::BlendOperation p_member);
	RenderingDevice::BlendOperation get_alpha_blend_op() const;
	void set_write_r(bool p_member);
	bool get_write_r() const;
	void set_write_g(bool p_member);
	bool get_write_g() const;
	void set_write_b(bool p_member);
	bool get_write_b() const;
	void set_write_a(bool p_member);
	bool get_write_a() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

