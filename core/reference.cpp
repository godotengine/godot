/*************************************************************************/
/*  reference.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

	if (refcount.ref()) {

		// this may fail in the scenario of two threads assigning the pointer for the FIRST TIME
		// at the same time, which is never likely to happen (would be crazy to do)
		// so don't do it.

		if (refcount_init.get() > 0) {
			refcount_init.unref();
			refcount.unref(); // first referencing is already 1, so compensate for the ref above
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

void Reference::reference() {

	refcount.ref();
	if (get_script_instance()) {
		get_script_instance()->refcount_incremented();
	}
}
bool Reference::unreference() {

	bool die = refcount.unref();

	if (get_script_instance()) {
		die = die && get_script_instance()->refcount_decremented();
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
	Reference *r = obj->cast_to<Reference>();
	if (r) {

		return REF(r);
	}

	return obj;
}

void WeakRef::set_obj(Object *p_object) {
	ref = p_object ? p_object->get_instance_ID() : 0;
}

void WeakRef::set_ref(const REF &p_ref) {

	ref = p_ref.is_valid() ? p_ref->get_instance_ID() : 0;
}

WeakRef::WeakRef() {
	ref = 0;
}

void WeakRef::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_ref:Object"), &WeakRef::get_ref);
}
#if 0

Reference * RefBase::get_reference_from_ref(const RefBase &p_base) {

	return p_base.get_reference();
}
void RefBase::ref_inc(Reference *p_reference) {

	p_reference->refcount.ref();
}
bool RefBase::ref_dec(Reference *p_reference) {

	bool ref = p_reference->refcount.unref();
	return ref;
}

Reference *RefBase::first_ref(Reference *p_reference) {

	if (p_reference->refcount.ref()) {

		// this may fail in the scenario of two threads assigning the pointer for the FIRST TIME
		// at the same time, which is never likely to happen (would be crazy to do)
		// so don't do it.

		if (p_reference->refcount_init.get()>0) {
			p_reference->refcount_init.unref();
			p_reference->refcount.unref(); // first referencing is already 1, so compensate for the ref above
		}

		return p_reference;
	} else {

		return 0;
	}

}
char * RefBase::get_refptr_data(const RefPtr &p_refptr) const {

	return p_refptr.data;
}
#endif
