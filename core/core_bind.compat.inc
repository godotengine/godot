/**************************************************************************/
/*  core_bind.compat.inc                                                  */
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

#ifndef DISABLE_DEPRECATED
#include "core_bind.h"

namespace core_bind {
namespace special {

Dictionary ClassDB::class_get_signal_compat_82198(const StringName &p_class, const StringName &p_signal) const {
	MethodInfo signal;
	if (::ClassDB::get_signal(p_class, p_signal, &signal)) {
		return signal.operator Dictionary();
	} else {
		return Dictionary();
	}
}

TypedArray<Dictionary> ClassDB::class_get_signal_list_compat_82198(const StringName &p_class, bool p_no_inheritance) const {
	List<MethodInfo> signals;
	::ClassDB::get_signal_list(p_class, &signals, p_no_inheritance);
	TypedArray<Dictionary> ret;

	for (const MethodInfo &E : signals) {
		ret.push_back(E.operator Dictionary());
	}

	return ret;
}

TypedArray<Dictionary> ClassDB::class_get_property_list_compat_82198(const StringName &p_class, bool p_no_inheritance) const {
	List<PropertyInfo> plist;
	::ClassDB::get_property_list(p_class, &plist, p_no_inheritance);
	TypedArray<Dictionary> ret;
	for (const PropertyInfo &E : plist) {
		ret.push_back(E.operator Dictionary());
	}

	return ret;
}

TypedArray<Dictionary> ClassDB::class_get_method_list_compat_82198(const StringName &p_class, bool p_no_inheritance) const {
	List<MethodInfo> methods;
	::ClassDB::get_method_list(p_class, &methods, p_no_inheritance);
	TypedArray<Dictionary> ret;

	for (const MethodInfo &E : methods) {
#ifdef DEBUG_METHODS_ENABLED
		ret.push_back(E.operator Dictionary());
#else
		Dictionary dict;
		dict["name"] = E.name;
		ret.push_back(dict);
#endif
	}

	return ret;
}

void ClassDB::_bind_compatibility_methods() {
	::ClassDB::bind_compatibility_method(D_METHOD("class_get_signal", "class", "signal"), &ClassDB::class_get_signal_compat_82198);
	::ClassDB::bind_compatibility_method(D_METHOD("class_get_signal_list", "class", "no_inheritance"), &ClassDB::class_get_signal_list_compat_82198, DEFVAL(false));
	::ClassDB::bind_compatibility_method(D_METHOD("class_get_property_list", "class", "no_inheritance"), &ClassDB::class_get_property_list_compat_82198, DEFVAL(false));
	::ClassDB::bind_compatibility_method(D_METHOD("class_get_method_list", "class", "no_inheritance"), &ClassDB::class_get_method_list_compat_82198, DEFVAL(false));
}
} //namespace special
} //namespace core_bind

#endif