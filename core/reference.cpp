/*************************************************************************/
/*  reference.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "reference.h"

#include "script_language.h"

bool Reference::init_ref() {

	if (reference()) {

		// this may fail in the scenario of two threads assigning the pointer for the FIRST TIME
		// at the same time, which is never likely to happen (would be crazy to do)
		// so don't do it.

		if (refcount_init.get() > 0) {
			refcount_init.unref();
			unreference(); // first referencing is already 1, so compensate for the ref above
		}

		return true;
	} else {

		return false;
	}
}

void Reference::_bind_methods() {

	ClassDB::bind_method(D_METHOD("init_ref"), &Reference::init_ref);
	ClassDB::bind_method(D_METHOD("reference"), &Reference::reference);
	ClassDB::bind_method(D_METHOD("unreference"), &Reference::unreference);
}

int Reference::reference_get_count() const {
	return refcount.get();
}

bool Reference::reference() {
	bool success = refcount.ref();

	if (success && get_script_instance()) {
		get_script_instance()->refcount_incremented();
	}

	return success;
}

bool Reference::unreference() {

	bool die = refcount.unref();

	if (get_script_instance()) {
		bool script_ret = get_script_instance()->refcount_decremented();
		die = die && script_ret;
	}

	return die;
}

Reference::Reference() {

	refcount.init();
	refcount_init.init();
}

Reference::~Reference() {
}

Variant WeakRef::get_ref() const {

	if (ref == 0)
		return Variant();

	Object *obj = ObjectDB::get_instance(ref);
	if (!obj)
		return Variant();
	Reference *r = cast_to<Reference>(obj);
	if (r) {

		return REF(r);
	}

	return obj;
}

void WeakRef::set_obj(Object *p_object) {
	ref = p_object ? p_object->get_instance_id() : 0;
}

void WeakRef::set_ref(const REF &p_ref) {

	ref = p_ref.is_valid() ? p_ref->get_instance_id() : 0;
}

WeakRef::WeakRef() {
	ref = 0;
}

void WeakRef::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_ref"), &WeakRef::get_ref);
}
