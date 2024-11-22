/**************************************************************************/
/*  method_info.h                                                         */
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

#ifndef METHOD_INFO_H
#define METHOD_INFO_H

#include "core/object/property_info.h"
#include "core/variant/variant.h"

enum MethodFlags {
	METHOD_FLAG_NORMAL = 1,
	METHOD_FLAG_EDITOR = 2,
	METHOD_FLAG_CONST = 4,
	METHOD_FLAG_VIRTUAL = 8,
	METHOD_FLAG_VARARG = 16,
	METHOD_FLAG_STATIC = 32,
	METHOD_FLAG_OBJECT_CORE = 64,
	METHOD_FLAG_VIRTUAL_REQUIRED = 128,
	METHOD_FLAGS_DEFAULT = METHOD_FLAG_NORMAL,
};

struct MethodInfo {
	STRUCT_DECLARE(MethodInfo);
	STRUCT_MEMBER(String, name, String());
	STRUCT_MEMBER_FROM_TO_ALIAS(List<PropertyInfo>, arguments, "args", List<PropertyInfo>());
	STRUCT_MEMBER_ALIAS(Vector<Variant>, default_arguments, "default_args", Vector<Variant>());
	STRUCT_MEMBER(uint32_t, flags, METHOD_FLAGS_DEFAULT);
	STRUCT_MEMBER(int, id, 0);
	STRUCT_MEMBER_STRUCT_FROM_TO_ALIAS(PropertyInfo, return_val, "return", PropertyInfo());
	STRUCT_MEMBER(int, return_val_metadata, 0);
	STRUCT_MEMBER(Vector<int>, arguments_metadata, Vector<int>());
	STRUCT_LAYOUT(Object, MethodInfo, struct name, struct arguments, struct default_arguments, struct flags, struct id, struct return_val, struct return_val_metadata, struct arguments_metadata);

	int get_argument_meta(int p_arg) const {
		ERR_FAIL_COND_V(p_arg < -1 || p_arg > arguments.size(), 0);
		if (p_arg == -1) {
			return return_val_metadata;
		}
		return arguments_metadata.size() > p_arg ? arguments_metadata[p_arg] : 0;
	}

	inline bool operator==(const MethodInfo &p_method) const { return id == p_method.id; }
	inline bool operator<(const MethodInfo &p_method) const { return id == p_method.id ? (name < p_method.name) : (id < p_method.id); }

	operator Dictionary() const;

	static MethodInfo from_dict(const Dictionary &p_dict);

	MethodInfo() {}

	explicit MethodInfo(const GDExtensionMethodInfo &pinfo) :
			name(*reinterpret_cast<StringName *>(pinfo.name)) {
		for (uint32_t j = 0; j < pinfo.argument_count; j++) {
			arguments.push_back(PropertyInfo(pinfo.arguments[j]));
		}
		const Variant *def_values = (const Variant *)pinfo.default_arguments;
		for (uint32_t j = 0; j < pinfo.default_argument_count; j++) {
			default_arguments.push_back(def_values[j]);
		}
		flags = pinfo.flags;
		id = pinfo.id;
		return_val = PropertyInfo(pinfo.return_value);
	}

	void _push_params(const PropertyInfo &p_param) {
		arguments.push_back(p_param);
	}

	template <typename... VarArgs>
	void _push_params(const PropertyInfo &p_param, VarArgs... p_params) {
		arguments.push_back(p_param);
		_push_params(p_params...);
	}

	MethodInfo(const String &p_name) { name = p_name; }

	template <typename... VarArgs>
	MethodInfo(const String &p_name, VarArgs... p_params) {
		name = p_name;
		_push_params(p_params...);
	}

	MethodInfo(Variant::Type ret) { return_val.type = ret; }
	MethodInfo(Variant::Type ret, const String &p_name) {
		return_val.type = ret;
		name = p_name;
	}

	template <typename... VarArgs>
	MethodInfo(Variant::Type ret, const String &p_name, VarArgs... p_params) {
		name = p_name;
		return_val.type = ret;
		_push_params(p_params...);
	}

	MethodInfo(const PropertyInfo &p_ret, const String &p_name) {
		return_val = p_ret;
		name = p_name;
	}

	template <typename... VarArgs>
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, VarArgs... p_params) {
		return_val = p_ret;
		name = p_name;
		_push_params(p_params...);
	}
};

#endif // METHOD_INFO_H
