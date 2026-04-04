/**************************************************************************/
/*  visual_shader_node_float_parameter.hpp                                */
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
#include <godot_cpp/classes/visual_shader_node_parameter.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VisualShaderNodeFloatParameter : public VisualShaderNodeParameter {
	GDEXTENSION_CLASS(VisualShaderNodeFloatParameter, VisualShaderNodeParameter)

public:
	enum Hint {
		HINT_NONE = 0,
		HINT_RANGE = 1,
		HINT_RANGE_STEP = 2,
		HINT_MAX = 3,
	};

	void set_hint(VisualShaderNodeFloatParameter::Hint p_hint);
	VisualShaderNodeFloatParameter::Hint get_hint() const;
	void set_min(float p_value);
	float get_min() const;
	void set_max(float p_value);
	float get_max() const;
	void set_step(float p_value);
	float get_step() const;
	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;
	void set_default_value(float p_value);
	float get_default_value() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNodeParameter::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNodeFloatParameter::Hint);

