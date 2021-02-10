/*************************************************************************/
/*  reference.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/object/script_language.h"

bool Reference::init_ref() {
	if (reference()) {
		if (!is_referenced() && refcount_init.unref()) {
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
	uint32_t rc_val = refcount.refval();
	bool success = rc_val != 0;

	if (success && rc_val <= 2 /* higher is not relevant */) {
		if (get_script_instance()) {
			get_script_instance()->refcount_incremented();
		}
		if (instance_binding_count.get() > 0 && !ScriptServer::are_languages_finished()) {
			for (int i = 0; i < MAX_SCRIPT_INSTANCE_BINDINGS; i++) {
				if (_script_instance_bindings[i]) {
					ScriptServer::get_language(i)->refcount_incremented_instance_binding(this);
				}
			}
		}
	}

	return success;
}

bool Reference::unreference() {
	uint32_t rc_val = refcount.unrefval();
	bool die = rc_val == 0;

	if (rc_val <= 1 /* higher is not relevant */) {
		if (get_script_instance()) {
			bool script_ret = get_script_instance()->refcount_decremented();
			die = die && script_ret;
		}
		if (instance_binding_count.get() > 0 && !ScriptServer::are_languages_finished()) {
			for (int i = 0; i < MAX_SCRIPT_INSTANCE_BINDINGS; i++) {
				if (_script_instance_bindings[i]) {
					bool script_ret = ScriptServer::get_language(i)->refcount_decremented_instance_binding(this);
					die = die && script_ret;
				}
			}
		}
	}

	return die;
}

Reference::Reference() :
		Object(true) {
	refcount.init();
	refcount_init.init();
}

Variant WeakRef::get_ref() const {
	if (ref.is_null()) {
		return Variant();
	}

	Object *obj = ObjectDB::get_instance(ref);
	if (!obj) {
		return Variant();
	}
	Reference *r = cast_to<Reference>(obj);
	if (r) {
		return REF(r);
	}

	return obj;
}

void WeakRef::set_obj(Object *p_object) {
	ref = p_object ? p_object->get_instance_id() : ObjectID();
}

void WeakRef::set_ref(const REF &p_ref) {
	ref = p_ref.is_valid() ? p_ref->get_instance_id() : ObjectID();
}

void WeakRef::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_ref"), &WeakRef::get_ref);
}
