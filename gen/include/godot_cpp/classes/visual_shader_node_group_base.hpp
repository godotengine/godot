/**************************************************************************/
/*  visual_shader_node_group_base.hpp                                     */
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
#include <godot_cpp/classes/visual_shader_node_resizable_base.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VisualShaderNodeGroupBase : public VisualShaderNodeResizableBase {
	GDEXTENSION_CLASS(VisualShaderNodeGroupBase, VisualShaderNodeResizableBase)

public:
	void set_inputs(const String &p_inputs);
	String get_inputs() const;
	void set_outputs(const String &p_outputs);
	String get_outputs() const;
	bool is_valid_port_name(const String &p_name) const;
	void add_input_port(int32_t p_id, int32_t p_type, const String &p_name);
	void remove_input_port(int32_t p_id);
	int32_t get_input_port_count() const;
	bool has_input_port(int32_t p_id) const;
	void clear_input_ports();
	void add_output_port(int32_t p_id, int32_t p_type, const String &p_name);
	void remove_output_port(int32_t p_id);
	int32_t get_output_port_count() const;
	bool has_output_port(int32_t p_id) const;
	void clear_output_ports();
	void set_input_port_name(int32_t p_id, const String &p_name);
	void set_input_port_type(int32_t p_id, int32_t p_type);
	void set_output_port_name(int32_t p_id, const String &p_name);
	void set_output_port_type(int32_t p_id, int32_t p_type);
	int32_t get_free_input_port_id() const;
	int32_t get_free_output_port_id() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNodeResizableBase::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

