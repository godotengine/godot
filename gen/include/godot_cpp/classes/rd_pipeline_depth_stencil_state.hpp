/**************************************************************************/
/*  rd_pipeline_depth_stencil_state.hpp                                   */
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

class RDPipelineDepthStencilState : public RefCounted {
	GDEXTENSION_CLASS(RDPipelineDepthStencilState, RefCounted)

public:
	void set_enable_depth_test(bool p_member);
	bool get_enable_depth_test() const;
	void set_enable_depth_write(bool p_member);
	bool get_enable_depth_write() const;
	void set_depth_compare_operator(RenderingDevice::CompareOperator p_member);
	RenderingDevice::CompareOperator get_depth_compare_operator() const;
	void set_enable_depth_range(bool p_member);
	bool get_enable_depth_range() const;
	void set_depth_range_min(float p_member);
	float get_depth_range_min() const;
	void set_depth_range_max(float p_member);
	float get_depth_range_max() const;
	void set_enable_stencil(bool p_member);
	bool get_enable_stencil() const;
	void set_front_op_fail(RenderingDevice::StencilOperation p_member);
	RenderingDevice::StencilOperation get_front_op_fail() const;
	void set_front_op_pass(RenderingDevice::StencilOperation p_member);
	RenderingDevice::StencilOperation get_front_op_pass() const;
	void set_front_op_depth_fail(RenderingDevice::StencilOperation p_member);
	RenderingDevice::StencilOperation get_front_op_depth_fail() const;
	void set_front_op_compare(RenderingDevice::CompareOperator p_member);
	RenderingDevice::CompareOperator get_front_op_compare() const;
	void set_front_op_compare_mask(uint32_t p_member);
	uint32_t get_front_op_compare_mask() const;
	void set_front_op_write_mask(uint32_t p_member);
	uint32_t get_front_op_write_mask() const;
	void set_front_op_reference(uint32_t p_member);
	uint32_t get_front_op_reference() const;
	void set_back_op_fail(RenderingDevice::StencilOperation p_member);
	RenderingDevice::StencilOperation get_back_op_fail() const;
	void set_back_op_pass(RenderingDevice::StencilOperation p_member);
	RenderingDevice::StencilOperation get_back_op_pass() const;
	void set_back_op_depth_fail(RenderingDevice::StencilOperation p_member);
	RenderingDevice::StencilOperation get_back_op_depth_fail() const;
	void set_back_op_compare(RenderingDevice::CompareOperator p_member);
	RenderingDevice::CompareOperator get_back_op_compare() const;
	void set_back_op_compare_mask(uint32_t p_member);
	uint32_t get_back_op_compare_mask() const;
	void set_back_op_write_mask(uint32_t p_member);
	uint32_t get_back_op_write_mask() const;
	void set_back_op_reference(uint32_t p_member);
	uint32_t get_back_op_reference() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

