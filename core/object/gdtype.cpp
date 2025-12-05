/**************************************************************************/
/*  gdtype.cpp                                                            */
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

#include "gdtype.h"

#include "core/os/thread.h"
#include "core/variant/variant.h"

GDType::GDType(const GDType *p_super_type, StringName p_name) :
		super_type(p_super_type), name(std::move(p_name)) {
	name_hierarchy.push_back(name);

	if (super_type) {
		for (const StringName &ancestor_name : super_type->name_hierarchy) {
			name_hierarchy.push_back(ancestor_name);
		}
	}
}

GDType::~GDType() {
	for (const KeyValue<StringName, const EnumInfo *> &kv : self_enum_map) {
		memdelete((EnumInfo *)kv.value);
	}
}

void GDType::initialize() {
	ERR_FAIL_COND(init_state != InitState::UNINITIALIZED);

	if (super_type) {
		// Now that a subtype is registered, the supertype cannot change anymore.
		// Otherwise, our caches would become invalid.
		// This shouldn't be a problem, since classes should register all their
		// parts in _bind_methods, which is called on registration.
		super_type->init_state = InitState::FINALIZED;

		constant_map = super_type->constant_map;
		enum_map = super_type->enum_map;
	}

	init_state = InitState::INITIALIZED;
}

void GDType::bind_integer_constant(const StringName &p_enum, const StringName &p_name, int64_t p_constant, bool p_is_bitfield) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(init_state != InitState::INITIALIZED);
	ERR_FAIL_COND(self_constant_map.has(p_name));

	constant_map[p_name] = p_constant;
	self_constant_map[p_name] = p_constant;

	String enum_name = p_enum;
	if (!enum_name.is_empty()) {
		if (enum_name.contains_char('.')) {
			enum_name = enum_name.get_slicec('.', 1);
		}

		const EnumInfo **_enum_info = self_enum_map.getptr(enum_name);

		if (_enum_info) {
			EnumInfo *enum_info = (EnumInfo *)*_enum_info;
			enum_info->values.insert(p_name, p_constant);
			enum_info->is_bitfield = p_is_bitfield;
		} else {
			EnumInfo *enum_info = memnew(EnumInfo);
			enum_info->name = enum_name;
			enum_info->is_bitfield = p_is_bitfield;
			enum_info->values.insert(p_name, p_constant);
			self_enum_map[enum_name] = enum_info;
			enum_map[enum_name] = enum_info;
		}
	}
}

const GDType::EnumInfo *GDType::get_integer_constant_enum(const StringName &p_name, bool p_no_inheritance) const {
	for (const KeyValue<StringName, const EnumInfo *> &kv : get_enum_map(p_no_inheritance)) {
		if (kv.value->values.has(p_name)) {
			return kv.value;
		}
	}

	return nullptr;
}
