/**************************************************************************/
/*  vmproperty.h                                                          */
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

#pragma once
#include <godot_cpp/variant/variant.hpp>
using namespace godot;

class Sandbox;

// This class is used to represent a property in the guest.
class SandboxProperty {
	StringName m_name;
	Variant::Type m_type = Variant::Type::NIL;
	uint64_t m_setter_address = 0;
	uint64_t m_getter_address = 0;
	Variant m_def_val;

public:
	SandboxProperty(const StringName &name, Variant::Type type, uint64_t setter, uint64_t getter, const Variant &def = "") :
			m_name(name), m_type(type), m_setter_address(setter), m_getter_address(getter), m_def_val(def) {}

	// Get the name of the property.
	const StringName &name() const { return m_name; }

	// Get the type of the property.
	Variant::Type type() const { return m_type; }

	// Get the address of the setter function.
	uint64_t setter_address() const { return m_setter_address; }
	// Get the address of the getter function.
	uint64_t getter_address() const { return m_getter_address; }

	// Get the default value of the property.
	const Variant &default_value() const { return m_def_val; }

	// Call the setter function.
	void set(Sandbox &sandbox, const Variant &value);
	// Call the getter function.
	Variant get(const Sandbox &sandbox) const;
};
