/**************************************************************************/
/*  scene_saveload.cpp                                                    */
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

#include "scene_saveload.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"

#include <stdint.h>

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#endif

TypedArray<SaveloadSpawner> SceneSaveload::get_spawners() const {
	return saveloader->get_spawners();
}

TypedArray<SaveloadSynchronizer> SceneSaveload::get_synchers() const {
	return saveloader->get_synchers();
}

Error SceneSaveload::track(Object *p_object) {
	return saveloader->track(p_object);
}

Error SceneSaveload::untrack(Object *p_object) {
	return saveloader->untrack(p_object);
}

Variant SceneSaveload::serialize(const Variant &p_configuration_data) {
	return saveloader->get_saveload_state().to_dict();
}

Error SceneSaveload::deserialize(const Variant &p_serialized_state, const Variant &p_configuration_data) {
	return saveloader->load_saveload_state(SceneSaveloadInterface::SaveloadState(p_serialized_state));
}

Error SceneSaveload::save(const String &p_path, const Variant &p_configuration_data) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	if (err != OK) {
		return err;
	}
	file->store_var(serialize(), false);
	file->close();
	return err;
}

Error SceneSaveload::load(const String &p_path, const Variant &p_configuration_data) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err != OK) {
		return err;
	}
	return deserialize(file->get_var(false));
}

void SceneSaveload::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_synchers"), &SceneSaveload::get_synchers);
	ClassDB::bind_method(D_METHOD("get_spawners"), &SceneSaveload::get_spawners);
}

SceneSaveload::SceneSaveload() {
	saveloader = Ref<SceneSaveloadInterface>(memnew(SceneSaveloadInterface(this)));
}

SceneSaveload::~SceneSaveload() {

}
