/**************************************************************************/
/*  method_info.cpp                                                       */
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

#include "method_info.h"
#include "property_info.h"

MethodInfo::operator Dictionary() const {
	Dictionary d;
	d["name"] = name;
	d["args"] = convert_property_list(arguments);
	Array da;
	for (int i = 0; i < default_arguments.size(); i++) {
		da.push_back(default_arguments[i]);
	}
	d["default_args"] = da;
	d["flags"] = flags;
	d["id"] = id;
	Dictionary r = return_val;
	d["return"] = r;
	return d;
}

MethodInfo MethodInfo::from_dict(const Dictionary &p_dict) {
	MethodInfo mi;

	if (p_dict.has("name")) {
		mi.name = p_dict["name"];
	}
	Array args;
	if (p_dict.has("args")) {
		args = p_dict["args"];
	}

	for (const Variant &arg : args) {
		Dictionary d = arg;
		mi.arguments.push_back(PropertyInfo::from_dict(d));
	}
	Array defargs;
	if (p_dict.has("default_args")) {
		defargs = p_dict["default_args"];
	}
	for (const Variant &defarg : defargs) {
		mi.default_arguments.push_back(defarg);
	}

	if (p_dict.has("return")) {
		mi.return_val = PropertyInfo::from_dict(p_dict["return"]);
	}

	if (p_dict.has("flags")) {
		mi.flags = p_dict["flags"];
	}

	return mi;
}

uint32_t MethodInfo::get_compatibility_hash() const {
	bool has_return = (return_val.type != Variant::NIL) || (return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT);

	uint32_t hash = hash_murmur3_one_32(has_return);
	hash = hash_murmur3_one_32(arguments.size(), hash);

	if (has_return) {
		hash = hash_murmur3_one_32(return_val.type, hash);
		if (return_val.class_name != StringName()) {
			hash = hash_murmur3_one_32(return_val.class_name.hash(), hash);
		}
	}

	for (const PropertyInfo &arg : arguments) {
		hash = hash_murmur3_one_32(arg.type, hash);
		if (arg.class_name != StringName()) {
			hash = hash_murmur3_one_32(arg.class_name.hash(), hash);
		}
	}

	hash = hash_murmur3_one_32(default_arguments.size(), hash);
	for (const Variant &v : default_arguments) {
		hash = hash_murmur3_one_32(v.hash(), hash);
	}

	hash = hash_murmur3_one_32(flags & METHOD_FLAG_CONST ? 1 : 0, hash);
	hash = hash_murmur3_one_32(flags & METHOD_FLAG_VARARG ? 1 : 0, hash);

	return hash_fmix32(hash);
}
