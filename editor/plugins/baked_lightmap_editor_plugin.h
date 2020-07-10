/*************************************************************************/
/*  baked_lightmap_editor_plugin.h                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef BAKED_LIGHTMAP_EDITOR_PLUGIN_H
#define BAKED_LIGHTMAP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/baked_lightmap.h"
#include "scene/resources/material.h"

class BakedLightmapEditorPlugin : public EditorPlugin {
	GDCLASS(BakedLightmapEditorPlugin, EditorPlugin);

	BakedLightmap *lightmap;

	Button *bake;
	EditorNode *editor;

	EditorFileDialog *file_dialog;
	static EditorProgress *tmp_progress;
	static bool bake_func_step(float p_progress, const String &p_description, void *, bool p_refresh);
	static void bake_func_end();

	void _bake_select_file(const String &p_file);
	void _bake();

protected:
	static void _bind_methods();

public:
	virtual String get_name() const override { return "BakedLightmap"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	BakedLightmapEditorPlugin(EditorNode *p_node);
	~BakedLightmapEditorPlugin();
};

#endif
