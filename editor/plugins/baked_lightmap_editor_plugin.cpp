/*************************************************************************/
/*  baked_lightmap_editor_plugin.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

void BakedLightmapEditorPlugin::_bake_select_file(const String &p_file) {
	if (lightmap) {
		BakedLightmap::BakeError err;
		if (get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root() == lightmap) {
			err = lightmap->bake(lightmap, p_file);
		} else {
			err = lightmap->bake(lightmap->get_parent(), p_file);
		}

		bake_func_end();

		switch (err) {
			case BakedLightmap::BAKE_ERROR_NO_SAVE_PATH: {
				String scene_path = lightmap->get_filename();
				if (scene_path == String()) {
					scene_path = lightmap->get_owner()->get_filename();
				}
				if (scene_path == String()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't determine a save path for lightmap images.\nSave your scene and try again."));
					break;
				}
				scene_path = scene_path.get_basename() + ".lmbake";

				file_dialog->set_current_path(scene_path);
				file_dialog->popup_centered_ratio();

			} break;
			case BakedLightmap::BAKE_ERROR_NO_MESHES:
				EditorNode::get_singleton()->show_warning(TTR("No meshes to bake. Make sure they contain an UV2 channel and that the 'Use In Baked Light' and 'Generate Lightmap' flags are on."));
				break;
			case BakedLightmap::BAKE_ERROR_CANT_CREATE_IMAGE:
				EditorNode::get_singleton()->show_warning(TTR("Failed creating lightmap images, make sure path is writable."));
				break;
			case BakedLightmap::BAKE_ERROR_LIGHTMAP_SIZE:
				EditorNode::get_singleton()->show_warning(TTR("Failed determining lightmap size. Maximum lightmap size too small?"));
				break;
			case BakedLightmap::BAKE_ERROR_INVALID_MESH:
				EditorNode::get_singleton()->show_warning(TTR("Some mesh is invalid. Make sure the UV2 channel values are contained within the [0.0,1.0] square region."));
				break;
			case BakedLightmap::BAKE_ERROR_NO_LIGHTMAPPER:
				EditorNode::get_singleton()->show_warning(TTR("Godot editor was built without ray tracing support, lightmaps can't be baked."));
				break;
			default: {
			}
		}
	}
}

void BakedLightmapEditorPlugin::_bake() {
	_bake_select_file("");
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
EditorProgress *BakedLightmapEditorPlugin::tmp_subprogress = NULL;

bool BakedLightmapEditorPlugin::bake_func_step(float p_progress, const String &p_description, void *, bool p_force_refresh) {
	if (!tmp_progress) {
		tmp_progress = memnew(EditorProgress("bake_lightmaps", TTR("Bake Lightmaps"), 1000, true));
		ERR_FAIL_COND_V(tmp_progress == nullptr, false);
	}
	return tmp_progress->step(p_description, p_progress * 1000, p_force_refresh);
}

bool BakedLightmapEditorPlugin::bake_func_substep(float p_progress, const String &p_description, void *, bool p_force_refresh) {
	if (!tmp_subprogress) {
		tmp_subprogress = memnew(EditorProgress("bake_lightmaps_substep", "", 1000, true));
		ERR_FAIL_COND_V(tmp_subprogress == nullptr, false);
	}
	return tmp_subprogress->step(p_description, p_progress * 1000, p_force_refresh);
}

void BakedLightmapEditorPlugin::bake_func_end() {
	if (tmp_progress != nullptr) {
		memdelete(tmp_progress);
		tmp_progress = nullptr;
	}

	if (tmp_subprogress != nullptr) {
		memdelete(tmp_subprogress);
		tmp_subprogress = nullptr;
	}
}

void BakedLightmapEditorPlugin::_bind_methods() {

	ClassDB::bind_method("_bake", &BakedLightmapEditorPlugin::_bake);
	ClassDB::bind_method("_bake_select_file", &BakedLightmapEditorPlugin::_bake_select_file);
}

BakedLightmapEditorPlugin::BakedLightmapEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	bake = memnew(ToolButton);
	bake->set_icon(editor->get_gui_base()->get_icon("Bake", "EditorIcons"));
	bake->set_text(TTR("Bake Lightmaps"));
	bake->hide();
	bake->connect("pressed", this, "_bake");

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	file_dialog->add_filter("*.lmbake ; LightMap Bake");
	file_dialog->set_title(TTR("Select lightmap bake file:"));
	file_dialog->connect("file_selected", this, "_bake_select_file");
	bake->add_child(file_dialog);

	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake);
	lightmap = NULL;

	BakedLightmap::bake_step_function = bake_func_step;
	BakedLightmap::bake_substep_function = bake_func_substep;
}

BakedLightmapEditorPlugin::~BakedLightmapEditorPlugin() {
}
