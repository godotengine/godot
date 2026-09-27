/**************************************************************************/
/*  script_editor_navigation_marker.cpp                                   */
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

#include "script_editor_navigation_marker.h"

#include "core/config/engine.h"
#include "core/os/memory.h"

ScriptEditorNavigationMarker *ScriptEditorNavigationMarker::get_singleton() {
	if (!singleton) {
		singleton = memnew(ScriptEditorNavigationMarker);
	}
	return singleton;
}

void ScriptEditorNavigationMarker::release_singleton() {
	if (!singleton) {
		return;
	}
	memdelete(singleton);
	singleton = nullptr;
}

void ScriptEditorNavigationMarker::init_begin() {
	init_in_progress = true;
}

void ScriptEditorNavigationMarker::init_end() {
	init_in_progress = false;
}

void ScriptEditorNavigationMarker::locate_begin() {
	locate_in_progress = true;
}

void ScriptEditorNavigationMarker::locate_end() {
	locate_in_progress = false;
	locate_end_physics_frame = Engine::get_singleton()->get_physics_frames();
	locate_end_process_frame = Engine::get_singleton()->get_process_frames();
}

void ScriptEditorNavigationMarker::traverse_begin() {
	traverse_in_progress = true;
}

void ScriptEditorNavigationMarker::traverse_end() {
	traverse_in_progress = false;
	traverse_end_physics_frame = Engine::get_singleton()->get_physics_frames();
	traverse_end_process_frame = Engine::get_singleton()->get_process_frames();
}

bool ScriptEditorNavigationMarker::is_initializing() const {
	return init_in_progress;
}

bool ScriptEditorNavigationMarker::is_locating() const {
	return locate_in_progress;
}

bool ScriptEditorNavigationMarker::is_traversing() const {
	return traverse_in_progress;
}

bool ScriptEditorNavigationMarker::is_locate_just_occured() const {
	if (Engine::get_singleton()->is_in_physics_frame()) {
		return locate_end_physics_frame == Engine::get_singleton()->get_physics_frames() || locate_end_physics_frame == Engine::get_singleton()->get_physics_frames() - 1;
	} else {
		return locate_end_process_frame == Engine::get_singleton()->get_process_frames();
	}
}

bool ScriptEditorNavigationMarker::is_traverse_just_occured() const {
	if (Engine::get_singleton()->is_in_physics_frame()) {
		return traverse_end_physics_frame == Engine::get_singleton()->get_physics_frames() || traverse_end_physics_frame == Engine::get_singleton()->get_physics_frames() - 1;
	} else {
		return traverse_end_process_frame == Engine::get_singleton()->get_process_frames();
	}
}
