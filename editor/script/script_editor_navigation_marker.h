/**************************************************************************/
/*  script_editor_navigation_marker.h                                     */
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

#pragma once

#include <cstdint>

class ScriptEditorNavigationMarker {
	static inline ScriptEditorNavigationMarker *singleton = nullptr;

private:
	bool init_in_progress = false;

	bool locate_in_progress = false;
	uint64_t locate_end_physics_frame = 0;
	uint64_t locate_end_process_frame = 0;

	bool traverse_in_progress = false;
	uint64_t traverse_end_physics_frame = 0;
	uint64_t traverse_end_process_frame = 0;

public:
	static ScriptEditorNavigationMarker *get_singleton();
	static void release_singleton();

	void init_begin();
	void init_end();
	void locate_begin();
	void locate_end();
	void traverse_begin();
	void traverse_end();

	bool is_initializing() const;
	bool is_locating() const;
	bool is_traversing() const;

	bool is_locate_just_occured() const;
	bool is_traverse_just_occured() const;
};
