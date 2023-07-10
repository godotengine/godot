/**************************************************************************/
/*  scene_saveload.h                                                      */
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

#ifndef SCENE_SAVELOAD_H
#define SCENE_SAVELOAD_H

#include "scene/main/saveload_api.h"

#include "scene_saveload_interface.h"

class SceneSaveload : public SaveloadAPI {
	GDCLASS(SceneSaveload, SaveloadAPI);

private:
	NodePath root_path;
	bool allow_object_decoding = false;

	Ref<SceneSaveloadInterface> saveloader;

#ifdef DEBUG_ENABLED
	_FORCE_INLINE_ void _profile_bandwidth(const String &p_what, int p_value);
#endif

protected:
	static void _bind_methods();

public:

	TypedArray<SaveloadSpawner> get_spawn_nodes();
	TypedArray<SaveloadSynchronizer> get_sync_nodes();
	Dictionary get_dict();

	virtual Variant get_state(Object *p_object, const StringName section) override;
	virtual Error set_state(Variant p_value, Object *p_object, const StringName section) override;

	virtual PackedByteArray encode(Object *p_object, const StringName section) override;
	virtual Error decode(PackedByteArray p_bytes, Object *p_object, const StringName section) override;

	virtual Error save(const String p_path, Object *p_object, const StringName section) override;
	virtual Error load(const String p_path, Object *p_object, const StringName section) override;

	//set root path, configure spawn, configure sync
	virtual Error object_configuration_add(Object *p_obj, Variant p_config) override;
	virtual Error object_configuration_remove(Object *p_obj, Variant p_config) override;

	void clear();

	// Usually from object_configuration_add/remove
	void set_root_path(const NodePath &p_path);
	NodePath get_root_path() const;

	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;

	Ref<SceneSaveloadInterface> get_saveloader() { return saveloader; }

	SceneSaveload();
	~SceneSaveload();
};

#endif // SCENE_SAVELOAD_H
