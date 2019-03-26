/*************************************************************************/
/*  method_watcher.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef METHOD_OVERRIDE_H
#define METHOD_OVERRIDE_H

#include "core/func_ref.h"
#include "core/map.h"
#include "core/pair.h"
#include "core/reference.h"

class MethodWatcher {
public:
	typedef Pair<StringName, StringName> SetGetPair;
	typedef Map<StringName, SetGetPair> PropertyMap;

	typedef Vector<Variant> Args;
	struct MethodInfo {
		Variant m_return;
		Vector<Args> m_calls;
	};
	typedef Map<StringName, MethodInfo> MethodMap;

private:
	PropertyMap m_properties;
	mutable MethodMap m_methods;

public:
	void bind_method(const String &p_name, const Variant &p_return);
	void add_property(const String &p_name, const StringName p_setter, const StringName p_getter);
	const Vector<Args> get_calls(const String &p_name) const;

	Variant get(const Variant &p_key, bool *r_valid = NULL);
	void set(const Variant &p_key, const Variant &p_value, bool *r_valid = NULL);
	Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	bool has_method(const StringName &p_method) const;
};

#endif // METHOD_OVERRIDE_H
