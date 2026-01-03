/**************************************************************************/
/*  scene_cache.h                                                         */
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

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"

class Node;
class EditorFileSystemDirectory;
class PackedScene;

/**
 * Used to load and cache scene instances for LSP autocompletion.
 *
 * This implementation is not thread safe.
 */
class SceneCache {
	// Always contains the path to the scene which is currently loaded via the `ResourceLoader`.
	// If this is not empty, `script_path_queue` must have at least one element.
	String current_loaded_owner = "";
	LocalVector<String> script_path_queue;

	HashMap<String, Node *> cache;

	void _get_owner_paths(EditorFileSystemDirectory *p_dir, const String &p_script_path, LocalVector<String> &r_owner_paths);
	void _finalize_scene_load();

public:
	void poll();

	void clear();
	void request_load(const String &p_script_path);
	void unload(const String &p_script_path);

	Node *get(const String &p_script_path);
};
