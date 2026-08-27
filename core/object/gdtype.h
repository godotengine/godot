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

	struct Property {
		enum class Type {
			SETGET,
			INTEGER_CONSTANT,
			ENUM,
			METHOD,
			SIGNAL
		};

		struct SetGet {
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
			SetGet setget;
			IntegerConstant integer_constant;
			const EnumInfo *enum_info;
			const MethodBind *method;
			const MethodInfo *signal;
		};

		Type type;
		Payload payload;

		Property &operator=(const Property &) = default;

		static Property create_setget(const SetGet &p_setget) {
			Payload payload;
			payload.setget = p_setget;
			return Property(Type::SETGET, payload);
		}

		static Property create_integer_constant(IntegerConstant p_constant) {
			Payload payload;
			payload.integer_constant = p_constant;
			return Property(Type::INTEGER_CONSTANT, payload);
		}

		static Property create_enum(const EnumInfo *p_enum_info) {
			Payload payload;
			payload.enum_info = p_enum_info;
			return Property(Type::ENUM, payload);
		}

		static Property create_method(const MethodBind *p_method) {
			Payload payload;
			payload.method = p_method;
			return Property(Type::METHOD, payload);
		}

		static Property create_signal(const MethodInfo *p_method) {
			Payload payload;
			payload.signal = p_method;
			return Property(Type::SIGNAL, payload);
		}

		Property(const Property &) = default;
		Property(Type p_type, const Payload &p_payload) : type(p_type), payload(p_payload) {}
	};

protected:
	const GDType *super_type;
	mutable InitState init_state = InitState::UNINITIALIZED;

	StringName name;
	/// Contains all the class names in order:
	/// `name` is the first element and `Object` is the last (for `Object` types).
	Vector<StringName> name_hierarchy;

	/// This needs to be tracked separately because
	/// bind_method supports binding non-owned methods.
	LocalVector<MethodBind *> owned_method_map;

	AHashMap<StringName, LocalVector<MethodBind *>> self_compatibility_method_map;

	/// Contains all properties that can be obtained or set with `object.property`.
	AHashMap<StringName, Property> property_map;
	AHashMap<StringName, Property> self_property_map;
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
	const AHashMap<StringName, Property> &get_property_map(bool p_no_inheritance = false) const { return p_no_inheritance ? self_property_map : property_map; }
	const EnumInfo *get_integer_constant_enum(const StringName &p_name, bool p_no_inheritance = false) const;
};
