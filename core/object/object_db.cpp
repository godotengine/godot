/**************************************************************************/
/*  object_db.cpp                                                         */
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

#include "core/object/object_db.h"

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/os/os.h"

void ObjectDB::debug_objects(DebugFunc p_func, void *p_user_data) {
	spin_lock.lock();

	for (uint32_t i = 0, count = slot_count; i < slot_max && count != 0; i++) {
		if (object_slots[i].validator) {
			p_func(object_slots[i].object, p_user_data);
			count--;
		}
	}
	spin_lock.unlock();
}

SpinLock ObjectDB::spin_lock;
uint32_t ObjectDB::slot_count = 0;
uint32_t ObjectDB::slot_max = 0;
ObjectDB::ObjectSlot *ObjectDB::object_slots = nullptr;
uint64_t ObjectDB::validator_counter = 0;

int ObjectDB::get_object_count() {
	return slot_count;
}

ObjectID ObjectDB::add_instance(Object *p_object) {
	spin_lock.lock();
	if (unlikely(slot_count == slot_max)) {
		CRASH_COND(slot_count == (1 << OBJECTDB_SLOT_MAX_COUNT_BITS));

		uint32_t new_slot_max = slot_max > 0 ? slot_max * 2 : 1;
		object_slots = (ObjectSlot *)memrealloc(object_slots, sizeof(ObjectSlot) * new_slot_max);
		for (uint32_t i = slot_max; i < new_slot_max; i++) {
			object_slots[i].object = nullptr;
			object_slots[i].is_ref_counted = false;
			object_slots[i].next_free = i;
			object_slots[i].validator = 0;
		}
		slot_max = new_slot_max;
	}

	uint32_t slot = object_slots[slot_count].next_free;
	if (object_slots[slot].object != nullptr) {
		spin_lock.unlock();
		ERR_FAIL_COND_V(object_slots[slot].object != nullptr, ObjectID());
	}
	object_slots[slot].object = p_object;
	object_slots[slot].is_ref_counted = p_object->is_ref_counted();
	validator_counter = (validator_counter + 1) & OBJECTDB_VALIDATOR_MASK;
	if (unlikely(validator_counter == 0)) {
		validator_counter = 1;
	}
	object_slots[slot].validator = validator_counter;

	uint64_t id = validator_counter;
	id <<= OBJECTDB_SLOT_MAX_COUNT_BITS;
	id |= uint64_t(slot);

	if (p_object->is_ref_counted()) {
		id |= OBJECTDB_REFERENCE_BIT;
	}

	slot_count++;

	spin_lock.unlock();

	return ObjectID(id);
}

void ObjectDB::remove_instance(Object *p_object) {
	uint64_t t = p_object->get_instance_id();
	uint32_t slot = t & OBJECTDB_SLOT_MAX_COUNT_MASK; //slot is always valid on valid object

	spin_lock.lock();

#ifdef DEBUG_ENABLED

	if (object_slots[slot].object != p_object) {
		spin_lock.unlock();
		ERR_FAIL_COND(object_slots[slot].object != p_object);
	}
	{
		uint64_t validator = (t >> OBJECTDB_SLOT_MAX_COUNT_BITS) & OBJECTDB_VALIDATOR_MASK;
		if (object_slots[slot].validator != validator) {
			spin_lock.unlock();
			ERR_FAIL_COND(object_slots[slot].validator != validator);
		}
	}

#endif
	//decrease slot count
	slot_count--;
	//set the free slot properly
	object_slots[slot_count].next_free = slot;
	//invalidate, so checks against it fail
	object_slots[slot].validator = 0;
	object_slots[slot].is_ref_counted = false;
	object_slots[slot].object = nullptr;

	spin_lock.unlock();
}

void ObjectDB::setup() {
	//nothing to do now
}

void ObjectDB::cleanup() {
	spin_lock.lock();

	if (slot_count > 0) {
		WARN_PRINT(vformat("%d ObjectDB %s leaked at exit (run with `--verbose` for details).", slot_count, slot_count == 1 ? "instance was" : "instances were"));
		if (OS::get_singleton()->is_stdout_verbose()) {
			// Ensure calling the native classes because if a leaked instance has a script
			// that overrides any of those methods, it'd not be OK to call them at this point,
			// now the scripting languages have already been terminated.
			MethodBind *node_get_path = ClassDB::get_method("Node", "get_path");
			MethodBind *resource_get_path = ClassDB::get_method("Resource", "get_path");
			Callable::CallError call_error;

			for (uint32_t i = 0, count = slot_count; i < slot_max && count != 0; i++) {
				if (object_slots[i].validator) {
					Object *obj = object_slots[i].object;

					String extra_info;
					if (obj->is_class("Node")) {
						extra_info = " - Node path: " + String(node_get_path->call(obj, nullptr, 0, call_error));
					}
					if (obj->is_class("Resource")) {
						extra_info = " - Resource path: " + String(resource_get_path->call(obj, nullptr, 0, call_error));
					}
					if (obj->is_class("RefCounted")) {
						extra_info = " - Reference count: " + itos((static_cast<RefCounted *>(obj))->get_reference_count());
					}

					uint64_t id = uint64_t(i) | (uint64_t(object_slots[i].validator) << OBJECTDB_SLOT_MAX_COUNT_BITS) | (object_slots[i].is_ref_counted ? OBJECTDB_REFERENCE_BIT : 0);
					DEV_ASSERT(id == (uint64_t)obj->get_instance_id()); // We could just use the id from the object, but this check may help catching memory corruption catastrophes.
					print_line("Leaked instance: " + String(obj->get_class()) + ":" + uitos(id) + extra_info);

					count--;
				}
			}
			print_line("Hint: Leaked instances typically happen when nodes are removed from the scene tree (with `remove_child()`) but not freed (with `free()` or `queue_free()`).");
		}
	}

	if (object_slots) {
		memfree(object_slots);
	}

	spin_lock.unlock();
}
