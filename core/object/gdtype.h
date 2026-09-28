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

		StringName name;
		/// The type that declared this member.
		const GDType *owner = nullptr;
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

	/// Iterates members in declaration order, starting with the root type's.
	class MemberRange {
		const GDType *const *types_begin = nullptr;
		const GDType *const *types_end = nullptr;

	public:
		class Iterator {
			const GDType *const *type = nullptr;
			const GDType *const *type_end = nullptr;
			uint32_t index = 0;

			_FORCE_INLINE_ void _skip_empty() {
				while (type != type_end && index >= (*type)->_self_members.size()) {
					type++;
					index = 0;
				}
			}

		public:
			_FORCE_INLINE_ const Member *operator*() const { return (*type)->_self_members[index]; }
			_FORCE_INLINE_ Iterator &operator++() {
				index++;
				_skip_empty();
				return *this;
			}
			_FORCE_INLINE_ bool operator==(const Iterator &p_other) const { return type == p_other.type && index == p_other.index; }
			_FORCE_INLINE_ bool operator!=(const Iterator &p_other) const { return !(*this == p_other); }

			Iterator(const GDType *const *p_type, const GDType *const *p_type_end) :
					type(p_type), type_end(p_type_end) {
				_skip_empty();
			}
		};

		Iterator begin() const { return Iterator(types_begin, types_end); }
		Iterator end() const { return Iterator(types_end, types_end); }

		MemberRange(const GDType *const *p_types_begin, const GDType *const *p_types_end) :
				types_begin(p_types_begin), types_end(p_types_end) {}
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

	/// The root type first, down to this type. Used to iterate inherited members.
	LocalVector<const GDType *> _type_hierarchy;

	/// Members declared by this type, in declaration order. Owned.
	LocalVector<const Member *> _self_members;
	/// All members that can be obtained or set with dot notation (`object.member`),
	/// including inherited ones. Members are shared with every subtype by pointer,
	/// so that each member is stored only once.
	AHashMap<StringName, const Member *> _members;
	LocalVector<const PropertyInfo *> ordered_self_properties;

	void _add_member(const StringName &p_name, const Member &p_member);

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
	// Minimal footprint helper for codebase callers. Helps avoid binary size.
	void bind_integer_constant_raw(const char *p_enum_qualified_name, const char *p_value_name, int64_t p_constant, bool p_is_bitfield = false);

	void add_signal(MethodInfo p_signal);

	bool bind_method(MethodBind *p_method, bool p_take_ownership = true);

	void set_method_flags(const StringName &p_method, int p_flags);

	bool bind_compatibility_method(MethodBind *p_method);
	const AHashMap<StringName, LocalVector<MethodBind *>> &get_self_compatibility_method_map() const { return self_compatibility_method_map; }

	void add_property(const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index);
	void add_to_ordered_properties(const PropertyInfo &p_pinfo);
	const LocalVector<const PropertyInfo *> &get_ordered_self_properties() const { return ordered_self_properties; }

	// Access
	MemberRange members(bool p_no_inheritance = false) const {
		const GDType *const *types_end = _type_hierarchy.ptr() + _type_hierarchy.size();
		return MemberRange(p_no_inheritance ? types_end - 1 : _type_hierarchy.ptr(), types_end);
	}
	_FORCE_INLINE_ const Member *get_member(const StringName &p_name, bool p_no_inheritance = false) const {
		const Member *const *member = _members.getptr(p_name);
		return member && (!p_no_inheritance || (*member)->owner == this) ? *member : nullptr;
	}
	const EnumInfo *get_integer_constant_enum(const StringName &p_name, bool p_no_inheritance = false) const;
};
