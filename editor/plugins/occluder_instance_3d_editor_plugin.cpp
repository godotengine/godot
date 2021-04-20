/*************************************************************************/
/*  occluder_instance_3d_editor_plugin.cpp                               */
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

#include "occluder_instance_3d_editor_plugin.h"

void OccluderInstance3DEditorPlugin::_bake_select_file(const String &p_file) {
	if (occluder_instance) {
		OccluderInstance3D::BakeError err;
		if (get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root() == occluder_instance) {
			err = occluder_instance->bake(occluder_instance, p_file);
		} else {
			err = occluder_instance->bake(occluder_instance->get_parent(), p_file);
		}

		switch (err) {
			case OccluderInstance3D::BAKE_ERROR_NO_SAVE_PATH: {
				String scene_path = occluder_instance->get_filename();
				if (scene_path == String()) {
					scene_path = occluder_instance->get_owner()->get_filename();
				}
				if (scene_path == String()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't determine a save path for the occluder.\nSave your scene and try again."));
					break;
				}
				scene_path = scene_path.get_basename() + ".occ";

				file_dialog->set_current_path(scene_path);
				file_dialog->popup_file_dialog();

			} break;
			case OccluderInstance3D::BAKE_ERROR_NO_MESHES: {
				EditorNode::get_singleton()->show_warning(TTR("No meshes to bake."));
				break;
			}
			default: {
			}
		}
	}
}

void OccluderInstance3DEditorPlugin::_bake() {
	_bake_select_file("");
}

void OccluderInstance3DEditorPlugin::edit(Object *p_object) {
	OccluderInstance3D *s = Object::cast_to<OccluderInstance3D>(p_object);
	if (!s) {
		return;
	}

	occluder_instance = s;
}

bool OccluderInstance3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("OccluderInstance3D");
}

void OccluderInstance3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bake->show();
	} else {
		bake->hide();
	}
}

void OccluderInstance3DEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_bake", &OccluderInstance3DEditorPlugin::_bake);
}

OccluderInstance3DEditorPlugin::OccluderInstance3DEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	bake = memnew(Button);
	bake->set_flat(true);
	bake->set_icon(editor->get_gui_base()->get_theme_icon("Bake", "EditorIcons"));
	bake->set_text(TTR("Bake Occluders"));
	bake->hide();
	bake->connect("pressed", Callable(this, "_bake"));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake);
	occluder_instance = nullptr;

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->add_filter("*.occ ; Occluder3D");
	file_dialog->set_title(TTR("Select occluder bake file:"));
	file_dialog->connect("file_selected", callable_mp(this, &OccluderInstance3DEditorPlugin::_bake_select_file));
	bake->add_child(file_dialog);
}

OccluderInstance3DEditorPlugin::~OccluderInstance3DEditorPlugin() {
}
