/*************************************************************************/
/*  ios_singleton.mm                                                               */
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

#include "ios_singleton.h"

#import <Foundation/Foundation.h>

void iOSSingleton::add_method(const StringName &p_name, const StringName &p_sel_name, const Vector<Variant::Type> &p_args, Variant::Type p_ret_type) {
	MethodData md;
	md.method_name = p_name;
	md.method_sel_name = p_sel_name;
	md.argtypes = p_args;
	md.ret_type = p_ret_type;
	method_map[p_name] = md;
}

Variant iOSSingleton::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
#ifdef IPHONE_ENABLED
	Map<StringName, MethodData>::Element *E = method_map.find(p_method);

	// Check the method we're looking for is in the iOSSingleton map and that
	// the arguments match.
	bool call_error = !E || E->get().argtypes.size() != p_argcount;
	if (!call_error) {
		for (int i = 0; i < p_argcount; i++) {

			if (!Variant::can_convert(p_args[i]->get_type(), E->get().argtypes[i])) {
				call_error = true;
				break;
			}
		}
	}

	if (call_error) {
		// The method is not in this map, defaulting to the regular instance calls.
		return Object::call(p_method, p_args, p_argcount, r_error);
	}

	ERR_FAIL_COND_V(!instance, Variant());

	r_error.error = Variant::CallError::CALL_OK;

	NSString *method_sel_name = [[[NSString alloc] initWithUTF8String:E->get().method_sel_name.utf8().get_data()] autorelease];
	SEL method_selector = NSSelectorFromString(method_sel_name);

	NSMethodSignature *method_signature = [instance methodSignatureForSelector:method_selector];
	NSInvocation *method_invocation = [NSInvocation invocationWithMethodSignature:method_signature];

	[method_invocation setTarget:instance];
	[method_invocation setSelector:method_selector];

	id *v = NULL;

	if (p_argcount) {

		v = (id *)alloca(sizeof(id) * p_argcount);
	}

	for (int i = 0; i < p_argcount; i++) {

		id vr = _variant_to_id(E->get().argtypes[i], p_args[i]);
		v[i] = vr;

		[method_invocation setArgument:&vr atIndex:(i + 2)];
	}
	[method_invocation retainArguments];
	[method_invocation invoke];

	Variant::Type ret_type = E->get().ret_type;
	if (ret_type == Variant::NIL) {
		return Variant::NIL;
	} else {
		id ret = _create_objc_object(ret_type);
		if (ret == nil)
			return Variant::NIL;
		[method_invocation getReturnValue:&ret];
		return _id_to_variant(ret, ret_type);
	}
#else
	return Object::call(p_method, p_args, p_argcount, r_error);
#endif
}

void iOSSingleton::add_signal(const StringName &p_name, const Vector<Variant::Type> &p_args) {
	if (p_args.size() == 0)
		ADD_SIGNAL(MethodInfo(p_name));
	else if (p_args.size() == 1)
		ADD_SIGNAL(MethodInfo(p_name, PropertyInfo(p_args[0], "arg1")));
	else if (p_args.size() == 2)
		ADD_SIGNAL(MethodInfo(p_name, PropertyInfo(p_args[0], "arg1"), PropertyInfo(p_args[1], "arg2")));
	else if (p_args.size() == 3)
		ADD_SIGNAL(MethodInfo(p_name, PropertyInfo(p_args[0], "arg1"), PropertyInfo(p_args[1], "arg2"), PropertyInfo(p_args[2], "arg3")));
	else if (p_args.size() == 4)
		ADD_SIGNAL(MethodInfo(p_name, PropertyInfo(p_args[0], "arg1"), PropertyInfo(p_args[1], "arg2"), PropertyInfo(p_args[2], "arg3"), PropertyInfo(p_args[3], "arg4")));
	else if (p_args.size() == 5)
		ADD_SIGNAL(MethodInfo(p_name, PropertyInfo(p_args[0], "arg1"), PropertyInfo(p_args[1], "arg2"), PropertyInfo(p_args[2], "arg3"), PropertyInfo(p_args[3], "arg4"), PropertyInfo(p_args[4], "arg5")));
}

#ifdef IPHONE_ENABLED
void iOSSingleton::set_instance(id p_instance) {
	this->instance = p_instance;
}

id iOSSingleton::get_instance() const {
	return this->instance;
}
#endif

iOSSingleton::iOSSingleton(){};
