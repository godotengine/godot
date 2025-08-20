/**************************************************************************/
/*  spx_base_mgr.cpp                                                      */
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

#include "spx_base_mgr.h"
#include "spx_engine.h"
#include "scene/2d/node_2d.h"
#include "scene/main/window.h"
#include <cstdio>

GdInt SpxBaseMgr::get_unique_id() {
	return SpxEngine::get_singleton()->get_unique_id();
}

void SpxBaseMgr::free_return_cstr(GdString str_ptr) {
	free((void*)str_ptr);
}

GdString SpxBaseMgr::to_return_cstr(const String& ret_val) {
	auto cstr = ret_val.utf8();
	char* result = (char*)malloc(cstr.size() + 1);
	strcpy(result, cstr.get_data());
	return result;
}

Window *SpxBaseMgr::get_root() {
	return SpxEngine::get_singleton()->get_root();
}

Node *SpxBaseMgr::get_spx_root() {
	return SpxEngine::get_singleton()->get_spx_root();
}

SceneTree *SpxBaseMgr::get_tree() {
	return SpxEngine::get_singleton()->get_tree();
}

void SpxBaseMgr::on_awake() {
	owner = memnew(Node2D);
	owner->set_name(get_class_name());
	get_spx_root()->add_child(owner);
}

void SpxBaseMgr::on_start() {
}

void SpxBaseMgr::on_update(float delta) {
}

void SpxBaseMgr::on_exit(int exit_code) {
}

void SpxBaseMgr::on_fixed_update(float delta) {
}

void SpxBaseMgr::on_destroy() {
	if (owner != nullptr) {
		owner->queue_free();
		owner = nullptr;
	}
}

void SpxBaseMgr::on_pause() {
	// Default implementation - override in derived classes if needed
}

void SpxBaseMgr::on_resume() {
	// Default implementation - override in derived classes if needed
}
