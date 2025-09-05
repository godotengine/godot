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


GdArray SpxBaseMgr::create_array(int32_t type, int32_t size) {
	if (size < 0) {
		return nullptr;
	}
	
	GdArray array = (GdArray)malloc(sizeof(GdArrayInfo));
	if (!array) {
		return nullptr;
	}
	
	array->size = size;
	array->type = type;
	
	if (size == 0) {
		array->data = nullptr;
		return array;
	}
	
	size_t element_size = 0;
	switch (type) {
		case GD_ARRAY_TYPE_INT64:
			element_size = sizeof(int64_t);
			break;
		case GD_ARRAY_TYPE_FLOAT:
			element_size = sizeof(float);
			break;
		case GD_ARRAY_TYPE_BOOL:
			element_size = sizeof(uint8_t); // Store as int64_t for alignment
			break;
		case GD_ARRAY_TYPE_STRING:
			element_size = sizeof(char*);
			break;
		case GD_ARRAY_TYPE_BYTE:
			element_size = sizeof(uint8_t);
			break;
		case GD_ARRAY_TYPE_GDOBJ:
			element_size = sizeof(GdObj);
			break;
		default:
			free(array);
			return nullptr;
	}
	
	array->data = malloc(size * element_size);
	if (!array->data && size > 0) {
		free(array);
		return nullptr;
	}
	
	return array;
}

void SpxBaseMgr::free_array(GdArray array) {
	if (!array) {
		return;
	}
	// Special handling for string arrays - need to free each string
	if (array->type == GD_ARRAY_TYPE_STRING && array->data) {
		char** strings = (char**)array->data;
		for (int64_t i = 0; i < array->size; i++) {
			if (strings[i]) {
				free(strings[i]);
			}
		}
	}
	
	if (array->data) {
		free(array->data);
	}
	free(array);
}

void* SpxBaseMgr::_get_array(GdArray array, int64_t index, int type_size) {
	if (!array || index < 0 || index >= array->size) {
		return nullptr;
	}
	return static_cast<char*>(array->data) + (index * type_size);
}