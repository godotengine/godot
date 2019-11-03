/*************************************************************************/
/*  baked_lightmap_editor_plugin.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "baked_lightmap_editor_plugin.h"

void BakedLightmapEditorPlugin::_bake() {

	if (lightmap) {
		BakedLightmap::BakeError err;
		if (get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root() == lightmap) {
			err = lightmap->bake(lightmap);
		} else {
			err = lightmap->bake(lightmap->get_parent());
		}

		switch (err) {
			case BakedLightmap::BAKE_ERROR_NO_SAVE_PATH:
				EditorNode::get_singleton()->show_warning(TTR("Can't determine a save path for lightmap images.\nSave your scene (for images to be saved in the same dir), or pick a save path from the BakedLightmap properties."));
				break;
			case BakedLightmap::BAKE_ERROR_NO_MESHES:
				EditorNode::get_singleton()->show_warning(TTR("No meshes to bake. Make sure they contain an UV2 channel and that the 'Bake Light' flag is on."));
				break;
			case BakedLightmap::BAKE_ERROR_CANT_CREATE_IMAGE:
				EditorNode::get_singleton()->show_warning(TTR("Failed creating lightmap images, make sure path is writable."));
				break;
			default: {
			}
		}
	}
}

void BakedLightmapEditorPlugin::edit(Object *p_object) {

	BakedLightmap *s = Object::cast_to<BakedLightmap>(p_object);
	if (!s)
		return;

	lightmap = s;
}

bool BakedLightmapEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("BakedLightmap");
}

void BakedLightmapEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		bake->show();
	} else {

		bake->hide();
	}
}

EditorProgress *BakedLightmapEditorPlugin::tmp_progress = NULL;

void BakedLightmapEditorPlugin::bake_func_begin(int p_steps) {

	ERR_FAIL_COND(tmp_progress != NULL);

	tmp_progress = memnew(EditorProgress("bake_lightmaps", TTR("Bake Lightmaps"), p_steps, true));
}

bool BakedLightmapEditorPlugin::bake_func_step(int p_step, const String &p_description) {

	ERR_FAIL_COND_V(tmp_progress == NULL, false);
	return tmp_progress->step(p_description, p_step, false);
}

void BakedLightmapEditorPlugin::bake_func_end() {
	ERR_FAIL_COND(tmp_progress == NULL);
	memdelete(tmp_progress);
	tmp_progress = NULL;
}

void BakedLightmapEditorPlugin::_bind_methods() {

	ClassDB::bind_method("_bake", &BakedLightmapEditorPlugin::_bake);
}

BakedLightmapEditorPlugin::BakedLightmapEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	bake = memnew(ToolButton);
	bake->set_icon(editor->get_gui_base()->get_icon("Bake", "EditorIcons"));
	bake->set_text(TTR("Bake Lightmaps"));
	bake->hide();
	bake->connect("pressed", this, "_bake");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake);
	lightmap = NULL;

	BakedLightmap::bake_begin_function = bake_func_begin;
	BakedLightmap::bake_step_function = bake_func_step;
	BakedLightmap::bake_end_function = bake_func_end;
}

BakedLightmapEditorPlugin::~BakedLightmapEditorPlugin() {
}
