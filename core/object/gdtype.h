/**************************************************************************/
/*  gdtype.h                                                              */
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

#include "core/string/string_name.h"
#include "core/templates/a_hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"

class GDType {
public:
	enum class InitState {
		UNINITIALIZED,
		INITIALIZED,
		FINALIZED,
	};

	struct EnumInfo {
		StringName name;
		AHashMap<StringName, int64_t> values;
		bool is_bitfield = false;
	};

protected:
	const GDType *super_type;
	mutable InitState init_state = InitState::UNINITIALIZED;

	StringName name;
	/// Contains all the class names in order:
	/// `name` is the first element and `Object` is the last.
	Vector<StringName> name_hierarchy;

	AHashMap<StringName, int64_t> constant_map;
	AHashMap<StringName, int64_t> self_constant_map;

	AHashMap<StringName, const EnumInfo *> enum_map;
	AHashMap<StringName, const EnumInfo *> self_enum_map;

public:
	GDType(const GDType *p_super_type, StringName p_name);
	~GDType();

	InitState get_init_state() const { return init_state; }
	void initialize();

	const GDType *get_super_type() const { return super_type; }
	const StringName &get_name() const { return name; }
	const Vector<StringName> &get_name_hierarchy() const { return name_hierarchy; }

	void bind_integer_constant(const StringName &p_enum, const StringName &p_name, int64_t p_constant, bool p_is_bitfield = false);
	const AHashMap<StringName, int64_t> &get_integer_constant_map(bool p_no_inheritance = false) const { return p_no_inheritance ? self_constant_map : constant_map; }
	const AHashMap<StringName, const EnumInfo *> &get_enum_map(bool p_no_inheritance = false) const { return p_no_inheritance ? self_enum_map : enum_map; }
	const EnumInfo *get_integer_constant_enum(const StringName &p_name, bool p_no_inheritance = false) const;
};
