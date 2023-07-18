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

#include "saveload_api.h"

#include "scene_saveload_interface.h"

class SceneSaveload : public SaveloadAPI {
	GDCLASS(SceneSaveload, SaveloadAPI);

private:
	bool allow_object_decoding = false;

	Ref<SceneSaveloadInterface> saveloader;

protected:
	static void _bind_methods();

public:
	TypedArray<SaveloadSpawner> get_spawners() const;
	TypedArray<SaveloadSynchronizer> get_synchers() const;

	virtual Error track (Object *p_object) override;
	virtual Error untrack (Object *p_object) override;

	virtual Variant serialize(const Variant &p_configuration_data = Variant()) override;
	virtual Error deserialize(const Variant &p_serialized_state, const Variant &p_configuration_data = Variant()) override;

	virtual Error save(const String &p_path, const Variant &p_configuration_data = Variant()) override;
	virtual Error load(const String &p_path, const Variant &p_configuration_data = Variant()) override;

	Ref<SceneSaveloadInterface> get_saveloader() const { return saveloader; }

	SceneSaveload();
	~SceneSaveload();
};

#endif // SCENE_SAVELOAD_H
