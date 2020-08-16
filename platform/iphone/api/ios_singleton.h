/*************************************************************************/
/*  ios_singleton.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef IOS_SINGLETON_H
#define IOS_SINGLETON_H

#include "core/object.h"
#ifdef IPHONE_ENABLED
#include "objc_utils.h"
#endif
#include <core/engine.h>
#include <core/variant.h>

class iOSSingleton : public Object {

	GDCLASS(iOSSingleton, Object);
#ifdef IPHONE_ENABLED
	id instance;
#endif
	struct MethodData {

		String method_name;
		String method_sel_name;
		Variant::Type ret_type;
		Vector<Variant::Type> argtypes;
	};

	Map<StringName, MethodData> method_map;

public:
	void add_method(const StringName &p_name, const StringName &p_sel_name, const Vector<Variant::Type> &p_args, Variant::Type p_ret_type);
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	void add_signal(const StringName &p_name, const Vector<Variant::Type> &p_args);
#ifdef IPHONE_ENABLED
	void set_instance(id p_instance);
	id get_instance() const;
#endif
	iOSSingleton();
};

#endif
