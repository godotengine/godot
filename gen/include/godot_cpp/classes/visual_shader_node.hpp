/**************************************************************************/
/*  visual_shader_node.hpp                                                */
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
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VisualShaderNode : public Resource {
	GDEXTENSION_CLASS(VisualShaderNode, Resource)

public:
	enum PortType {
		PORT_TYPE_SCALAR = 0,
		PORT_TYPE_SCALAR_INT = 1,
		PORT_TYPE_SCALAR_UINT = 2,
		PORT_TYPE_VECTOR_2D = 3,
		PORT_TYPE_VECTOR_3D = 4,
		PORT_TYPE_VECTOR_4D = 5,
		PORT_TYPE_BOOLEAN = 6,
		PORT_TYPE_TRANSFORM = 7,
		PORT_TYPE_SAMPLER = 8,
		PORT_TYPE_MAX = 9,
	};

	int32_t get_default_input_port(VisualShaderNode::PortType p_type) const;
	void set_output_port_for_preview(int32_t p_port);
	int32_t get_output_port_for_preview() const;
	void set_input_port_default_value(int32_t p_port, const Variant &p_value, const Variant &p_prev_value = nullptr);
	Variant get_input_port_default_value(int32_t p_port) const;
	void remove_input_port_default_value(int32_t p_port);
	void clear_default_input_values();
	void set_default_input_values(const Array &p_values);
	Array get_default_input_values() const;
	void set_frame(int32_t p_frame);
	int32_t get_frame() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNode::PortType);

