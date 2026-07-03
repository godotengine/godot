/**************************************************************************/
/*  festival_notebook.cpp                                                 */
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

#include "festival_notebook.h"

#include "core/object/class_db.h"

#include "core/io/config_file.h"

FestivalNotebook *FestivalNotebook::singleton = nullptr;

FestivalNotebook *FestivalNotebook::get_singleton() { return singleton; }

void FestivalNotebook::learn(const StringName &p_id) {
	if (p_id == StringName() || known.has(p_id)) {
		return;
	}
	known.insert(p_id);
	emit_signal(SNAME("knowledge_learned"), p_id);
}

bool FestivalNotebook::knows(const StringName &p_id) const { return known.has(p_id); }

void FestivalNotebook::forget(const StringName &p_id) { known.erase(p_id); }

PackedStringArray FestivalNotebook::get_known() const {
	PackedStringArray out;
	for (const StringName &E : known) {
		out.push_back(E);
	}
	return out;
}

int FestivalNotebook::get_known_count() const { return known.size(); }

void FestivalNotebook::clear() { known.clear(); }

bool FestivalNotebook::save(const String &p_path) const {
	Ref<ConfigFile> cf;
	cf.instantiate();
	cf->set_value("notebook", "known", get_known());
	return cf->save(p_path) == OK;
}

bool FestivalNotebook::load(const String &p_path) {
	Ref<ConfigFile> cf;
	cf.instantiate();
	if (cf->load(p_path) != OK) {
		return false;
	}
	const PackedStringArray arr = cf->get_value("notebook", "known", PackedStringArray());
	known.clear();
	for (int i = 0; i < arr.size(); i++) {
		known.insert(arr[i]);
	}
	emit_signal(SNAME("notebook_loaded"));
	return true;
}

FestivalNotebook::FestivalNotebook() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalNotebook::~FestivalNotebook() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void FestivalNotebook::_bind_methods() {
	ClassDB::bind_method(D_METHOD("learn", "id"), &FestivalNotebook::learn);
	ClassDB::bind_method(D_METHOD("knows", "id"), &FestivalNotebook::knows);
	ClassDB::bind_method(D_METHOD("forget", "id"), &FestivalNotebook::forget);
	ClassDB::bind_method(D_METHOD("get_known"), &FestivalNotebook::get_known);
	ClassDB::bind_method(D_METHOD("get_known_count"), &FestivalNotebook::get_known_count);
	ClassDB::bind_method(D_METHOD("clear"), &FestivalNotebook::clear);
	ClassDB::bind_method(D_METHOD("save", "path"), &FestivalNotebook::save, DEFVAL("user://festival_notebook.cfg"));
	ClassDB::bind_method(D_METHOD("load", "path"), &FestivalNotebook::load, DEFVAL("user://festival_notebook.cfg"));

	ADD_SIGNAL(MethodInfo("knowledge_learned", PropertyInfo(Variant::STRING_NAME, "id")));
	ADD_SIGNAL(MethodInfo("notebook_loaded"));
}
