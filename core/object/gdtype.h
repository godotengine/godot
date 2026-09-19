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

#include "core/object/method_info.h"
#include "core/string/string_name.h"
#include "core/templates/a_hash_map.h"
#include "core/templates/vector.h"

class MethodBind;

class GDType {
public:
	enum class InitState {
		UNINITIALIZED,
		MUTABLE,
		FINALIZED,
	};

	struct EnumInfo {
		StringName name;
		AHashMap<StringName, int64_t> values;
		bool is_bitfield = false;
	};

	struct Member {
		enum class Type {
			PROPERTY,
			INTEGER_CONSTANT,
			ENUM,
			METHOD,
			SIGNAL
		};

		struct Property {
			const PropertyInfo *property_info;
			const MethodBind *setter;
			const MethodBind *getter;
			int index;
		};

		struct IntegerConstant {
			int64_t value;
			const EnumInfo *enum_info;
		};

		union Payload {
			Property property;
			IntegerConstant integer_constant;
			const EnumInfo *enum_info;
			const MethodBind *method;
			const MethodInfo *signal;
		};

		Type type;
		Payload payload;

		Member &operator=(const Member &) = default;

		static Member create_property(const Property &p_property) {
			Payload payload;
			payload.property = p_property;
			return Member(Type::PROPERTY, payload);
		}

		static Member create_integer_constant(IntegerConstant p_constant) {
			Payload payload;
			payload.integer_constant = p_constant;
			return Member(Type::INTEGER_CONSTANT, payload);
		}

		static Member create_enum(const EnumInfo *p_enum_info) {
			Payload payload;
			payload.enum_info = p_enum_info;
			return Member(Type::ENUM, payload);
		}

		static Member create_method(const MethodBind *p_method) {
			Payload payload;
			payload.method = p_method;
			return Member(Type::METHOD, payload);
		}

		static Member create_signal(const MethodInfo *p_method) {
			Payload payload;
			payload.signal = p_method;
			return Member(Type::SIGNAL, payload);
		}

		Member(const Member &) = default;
		Member(Type p_type, const Payload &p_payload) : type(p_type), payload(p_payload) {}
	};

protected:
	const GDType *super_type;
	mutable InitState init_state = InitState::UNINITIALIZED;
	uint64_t owning_thread_id = 0;

	StringName name;
	/// Contains all the class names in order:
	/// `name` is the first element and `Object` is the last (for `Object` types).
	Vector<StringName> name_hierarchy;

	/// This needs to be tracked separately because
	/// bind_method supports binding non-owned methods.
	LocalVector<MethodBind *> owned_method_map;

	AHashMap<StringName, LocalVector<MethodBind *>> self_compatibility_method_map;

	/// Contains all members that can be obtained or set with dot notation (`object.member`).
	AHashMap<StringName, Member> _members;
	AHashMap<StringName, Member> _self_members;
	LocalVector<const PropertyInfo *> ordered_self_properties;

public:
	GDType(const GDType *p_super_type, StringName p_name);
	~GDType();

	InitState get_init_state() const { return init_state; }
	void initialize();

	const GDType *get_super_type() const { return super_type; }
	const StringName &get_name() const { return name; }
	const StringName &get_super_type_name() const {
		static const StringName EMPTY;
		return super_type ? super_type->name : EMPTY;
	}
	const Vector<StringName> &get_name_hierarchy() const { return name_hierarchy; }

	// Binding
	void bind_integer_constant(const StringName &p_enum, const StringName &p_name, int64_t p_constant, bool p_is_bitfield = false);

	void add_signal(MethodInfo p_signal);

	bool bind_method(MethodBind *p_method, bool p_take_ownership = true);

	void set_method_flags(const StringName &p_method, int p_flags);

	bool bind_compatibility_method(MethodBind *p_method);
	const AHashMap<StringName, LocalVector<MethodBind *>> &get_self_compatibility_method_map() const { return self_compatibility_method_map; }

	void add_property(const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index);
	void add_to_ordered_properties(const PropertyInfo &p_pinfo);
	const LocalVector<const PropertyInfo *> &get_ordered_self_properties() const { return ordered_self_properties; }

	// Access
	const AHashMap<StringName, Member> &members(bool p_no_inheritance = false) const { return p_no_inheritance ? _self_members : _members; }
	const EnumInfo *get_integer_constant_enum(const StringName &p_name, bool p_no_inheritance = false) const;
};
