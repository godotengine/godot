/**************************************************************************/
/*  festival_world.cpp                                                    */
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

#include "festival_world.h"

FestivalWorld *FestivalWorld::singleton = nullptr;

FestivalWorld *FestivalWorld::get_singleton() { return singleton; }

void FestivalWorld::set_flag(const StringName &p_flag, const Variant &p_value) {
	flags[p_flag] = p_value;
	emit_signal(SNAME("flag_changed"), p_flag, p_value);
}

Variant FestivalWorld::get_flag(const StringName &p_flag, const Variant &p_default) const {
	const Variant *v = flags.getptr(p_flag);
	return v ? *v : p_default;
}

bool FestivalWorld::has_flag(const StringName &p_flag) const { return flags.has(p_flag); }

void FestivalWorld::erase_flag(const StringName &p_flag) {
	if (flags.erase(p_flag)) {
		emit_signal(SNAME("flag_changed"), p_flag, Variant());
	}
}

Dictionary FestivalWorld::get_flags() const {
	Dictionary d;
	for (const KeyValue<StringName, Variant> &E : flags) {
		d[E.key] = E.value;
	}
	return d;
}

void FestivalWorld::add_item(const StringName &p_id, int p_count) {
	if (p_count <= 0) {
		return;
	}
	inventory[p_id] = get_item_count(p_id) + p_count;
	emit_signal(SNAME("inventory_changed"));
}

void FestivalWorld::remove_item(const StringName &p_id, int p_count) {
	if (p_count <= 0) {
		return;
	}
	int c = get_item_count(p_id);
	if (c <= 0) {
		return;
	}
	c -= p_count;
	if (c <= 0) {
		inventory.erase(p_id);
		presented.erase(p_id);
	} else {
		inventory[p_id] = c;
	}
	emit_signal(SNAME("inventory_changed"));
}

bool FestivalWorld::has_item(const StringName &p_id, int p_count) const {
	return get_item_count(p_id) >= p_count;
}

int FestivalWorld::get_item_count(const StringName &p_id) const {
	const int *c = inventory.getptr(p_id);
	return c ? *c : 0;
}

Dictionary FestivalWorld::get_items() const {
	Dictionary d;
	for (const KeyValue<StringName, int> &E : inventory) {
		d[E.key] = E.value;
	}
	return d;
}

void FestivalWorld::set_outfit(const StringName &p_outfit) {
	if (current_outfit == p_outfit) {
		return;
	}
	current_outfit = p_outfit;
	emit_signal(SNAME("outfit_changed"), current_outfit);
}

StringName FestivalWorld::get_outfit() const { return current_outfit; }

void FestivalWorld::present_item(const StringName &p_id) {
	if (presented.has(p_id)) {
		return;
	}
	presented.insert(p_id);
	emit_signal(SNAME("presented_changed"));
}

void FestivalWorld::unpresent_item(const StringName &p_id) {
	if (presented.erase(p_id)) {
		emit_signal(SNAME("presented_changed"));
	}
}

bool FestivalWorld::is_presented(const StringName &p_id) const { return presented.has(p_id); }

PackedStringArray FestivalWorld::get_presented() const {
	PackedStringArray out;
	for (const StringName &E : presented) {
		out.push_back(E);
	}
	return out;
}

void FestivalWorld::clear_presented() {
	if (presented.is_empty()) {
		return;
	}
	presented.clear();
	emit_signal(SNAME("presented_changed"));
}

void FestivalWorld::reset() {
	flags.clear();
	inventory.clear();
	presented.clear();
	current_outfit = StringName();
	emit_signal(SNAME("world_reset"));
}

FestivalWorld::FestivalWorld() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalWorld::~FestivalWorld() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void FestivalWorld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_flag", "flag", "value"), &FestivalWorld::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag", "default"), &FestivalWorld::get_flag, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("has_flag", "flag"), &FestivalWorld::has_flag);
	ClassDB::bind_method(D_METHOD("erase_flag", "flag"), &FestivalWorld::erase_flag);
	ClassDB::bind_method(D_METHOD("get_flags"), &FestivalWorld::get_flags);

	ClassDB::bind_method(D_METHOD("add_item", "id", "count"), &FestivalWorld::add_item, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("remove_item", "id", "count"), &FestivalWorld::remove_item, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("has_item", "id", "count"), &FestivalWorld::has_item, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("get_item_count", "id"), &FestivalWorld::get_item_count);
	ClassDB::bind_method(D_METHOD("get_items"), &FestivalWorld::get_items);

	ClassDB::bind_method(D_METHOD("set_outfit", "outfit"), &FestivalWorld::set_outfit);
	ClassDB::bind_method(D_METHOD("get_outfit"), &FestivalWorld::get_outfit);

	ClassDB::bind_method(D_METHOD("present_item", "id"), &FestivalWorld::present_item);
	ClassDB::bind_method(D_METHOD("unpresent_item", "id"), &FestivalWorld::unpresent_item);
	ClassDB::bind_method(D_METHOD("is_presented", "id"), &FestivalWorld::is_presented);
	ClassDB::bind_method(D_METHOD("get_presented"), &FestivalWorld::get_presented);
	ClassDB::bind_method(D_METHOD("clear_presented"), &FestivalWorld::clear_presented);

	ClassDB::bind_method(D_METHOD("reset"), &FestivalWorld::reset);

	ADD_SIGNAL(MethodInfo("flag_changed", PropertyInfo(Variant::STRING_NAME, "flag"), PropertyInfo(Variant::NIL, "value")));
	ADD_SIGNAL(MethodInfo("inventory_changed"));
	ADD_SIGNAL(MethodInfo("outfit_changed", PropertyInfo(Variant::STRING_NAME, "outfit")));
	ADD_SIGNAL(MethodInfo("presented_changed"));
	ADD_SIGNAL(MethodInfo("world_reset"));
}
