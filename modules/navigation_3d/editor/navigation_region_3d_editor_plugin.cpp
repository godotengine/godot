/**************************************************************************/
/*  navigation_region_3d_editor_plugin.cpp                                */
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

#include "navigation_region_3d_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/inspector/multi_node_edit.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "scene/3d/navigation/navigation_region_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "servers/navigation_3d/navigation_server_3d.h"

void NavigationRegion3DEditor::_node_removed(Node *p_node) {
	if (selected_regions.is_empty()) {
		return;
	}

	NavigationRegion3D *region = Object::cast_to<NavigationRegion3D>(p_node);

	if (region && selected_regions.has(region)) {
		selected_regions.erase(region);
		if (selected_regions.is_empty()) {
			hide();
		}
	}
}

void NavigationRegion3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			button_bake->set_button_icon(get_theme_icon(SNAME("Bake"), EditorStringName(EditorIcons)));
			button_reset->set_button_icon(get_theme_icon(SNAME("Reload"), EditorStringName(EditorIcons)));
		} break;
		case NOTIFICATION_PROCESS: {
			if (currently_baking_region) {
				const String bake_state_msg = NavigationServer3D::get_singleton()->get_baking_navigation_mesh_state_msg(currently_baking_region->get_navigation_mesh());
				multibake_dialog->set_text(itos(processed_regions_to_bake_count) + " / " + itos(processed_regions_to_bake_count_max) + " - Baking navmesh from region '" + currently_baking_region->get_name() + "'.\n\nBake state: " + bake_state_msg + "\n\nDo NOT change nodes by any means while the baking is parsing the SceneTree.");
			} else {
				multibake_dialog->set_text("");
			}
		}
	}
}

void NavigationRegion3DEditor::_bake_pressed() {
	button_bake->set_pressed(false);

	if (selected_regions.is_empty()) {
		return;
	}

	if (bake_in_process) {
		return;
	}

	HashSet<Ref<NavigationMesh>> unique_navmeshes;
	regions_to_bake.clear();
	regions_with_navmesh_to_bake.clear();

	for (NavigationRegion3D *region : selected_regions) {
		ERR_CONTINUE(region == nullptr);
		Ref<NavigationMesh> navmesh = region->get_navigation_mesh();
		if (navmesh.is_null()) {
			err_dialog->set_text(TTR("A NavigationMesh resource must be set or created for this node to work."));
			err_dialog->popup_centered();
			return;
		}

		String path = navmesh->get_path();
		if (!path.is_resource_file()) {
			int srpos = path.find("::");
			if (srpos != -1) {
				String base = path.substr(0, srpos);
				if (ResourceLoader::get_resource_type(base) == "PackedScene") {
					if (!get_tree()->get_edited_scene_root() || get_tree()->get_edited_scene_root()->get_scene_file_path() != base) {
						err_dialog->set_text(TTR("Cannot generate navigation mesh because it does not belong to the edited scene. Make it unique first."));
						err_dialog->popup_centered();
						return;
					}
				} else {
					if (FileAccess::exists(base + ".import")) {
						err_dialog->set_text(TTR("Cannot generate navigation mesh because it belongs to a resource which was imported."));
						err_dialog->popup_centered();
						return;
					}
				}
			}
		} else {
			if (FileAccess::exists(path + ".import")) {
				err_dialog->set_text(TTR("Cannot generate navigation mesh because the resource was imported from another type."));
				err_dialog->popup_centered();
				return;
			}
		}

		regions_to_bake.push_back(region);

		if (unique_navmeshes.has(navmesh)) {
			// No point (re)baking the same resource in case of multi select.
			// Trying to bake the same navmesh twice would trigger an error.
			continue;
		}
		unique_navmeshes.insert(navmesh);
		regions_with_navmesh_to_bake.push_back(region);
	}

	if (!regions_with_navmesh_to_bake.is_empty()) {
		multibake_dialog->set_ok_button_text(TTR("Bake"));
		multibake_dialog->get_ok_button()->set_disabled(false);
		multibake_dialog->set_text("Attempting to bake " + itos(regions_with_navmesh_to_bake.size()) + " unique navmesh(es) from " + itos(regions_to_bake.size()) + " selected NavigationRegion3D node(s).\n\nThis can take some time and freeze the Editor temporarily.\n\nDo NOT change nodes by any means while the baking is parsing the SceneTree.");
		multibake_dialog->popup_centered();
		if (regions_with_navmesh_to_bake.size() == 1) {
			// If we only have a single region start bake immediately.
			_on_navmesh_multibake_confirmed();
		}
	}
}

void NavigationRegion3DEditor::_on_navmesh_multibake_confirmed() {
	multibake_dialog->get_ok_button()->set_disabled(true);

	bake_in_process = true;
	region_baking_canceled = false;
	processed_regions_to_bake_count = 0;
	processed_regions_to_bake_count_max = regions_with_navmesh_to_bake.size();

	set_process(true);
	_process_regions_to_bake();
}

void NavigationRegion3DEditor::_process_regions_to_bake() {
	if (region_baking_canceled) {
		region_baking_canceled = false;
		regions_with_navmesh_to_bake.clear();
	}

	if (regions_with_navmesh_to_bake.is_empty()) {
		regions_to_bake.clear();
		multibake_dialog->set_visible(false);
		set_process(false);
		currently_baking_region = nullptr;
		bake_in_process = false;
		return;
	}

	NavigationRegion3D *region_to_bake = regions_with_navmesh_to_bake[0];
	regions_with_navmesh_to_bake.remove_at_unordered(0);
	processed_regions_to_bake_count += 1;
	if (region_to_bake && region_to_bake->get_navigation_mesh().is_valid()) {
		currently_baking_region = region_to_bake;
		region_to_bake->connect(SNAME("bake_finished"), callable_mp(this, &NavigationRegion3DEditor::_process_regions_to_bake), CONNECT_ONE_SHOT);
		region_to_bake->bake_navigation_mesh(true);
		return;
	} else {
		_process_regions_to_bake();
	}
}

void NavigationRegion3DEditor::_on_navmesh_multibake_canceled() {
	if (bake_in_process) {
		region_baking_canceled = true;
		return;
	}

	multibake_dialog->set_visible(false);
	regions_to_bake.clear();
	regions_with_navmesh_to_bake.clear();
	processed_regions_to_bake_count = 0;
	processed_regions_to_bake_count_max = 0;
	region_baking_canceled = false;
	currently_baking_region = nullptr;
	bake_in_process = false;
}

void NavigationRegion3DEditor::_clear_pressed() {
	button_bake->set_pressed(false);
	bake_info->set_text("");

	if (!selected_regions.is_empty()) {
		for (NavigationRegion3D *region : selected_regions) {
			if (region->get_navigation_mesh().is_valid()) {
				region->get_navigation_mesh()->clear();
				region->update_gizmos();
			}
		}
	}
}

void NavigationRegion3DEditor::edit(LocalVector<NavigationRegion3D *> p_regions) {
	if (p_regions.is_empty()) {
		return;
	}

	selected_regions = p_regions;
}

NavigationRegion3DEditor::NavigationRegion3DEditor() {
	bake_hbox = memnew(HBoxContainer);

	button_bake = memnew(Button);
	button_bake->set_theme_type_variation(SceneStringName(FlatButton));
	bake_hbox->add_child(button_bake);
	button_bake->set_toggle_mode(true);
	button_bake->set_text(TTR("Bake NavigationMesh"));
	button_bake->set_tooltip_text(TTR("Bakes the NavigationMesh by first parsing the scene for source geometry and then creating the navigation mesh vertices and polygons."));
	button_bake->connect(SceneStringName(pressed), callable_mp(this, &NavigationRegion3DEditor::_bake_pressed));

	button_reset = memnew(Button);
	button_reset->set_theme_type_variation(SceneStringName(FlatButton));
	bake_hbox->add_child(button_reset);
	button_reset->set_text(TTR("Clear NavigationMesh"));
	button_reset->set_tooltip_text(TTR("Clears the internal NavigationMesh vertices and polygons."));
	button_reset->connect(SceneStringName(pressed), callable_mp(this, &NavigationRegion3DEditor::_clear_pressed));

	bake_info = memnew(Label);
	bake_info->set_focus_mode(FOCUS_ACCESSIBILITY);
	bake_hbox->add_child(bake_info);

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);

	multibake_dialog = memnew(ConfirmationDialog);
	add_child(multibake_dialog);
	multibake_dialog->connect(SceneStringName(confirmed), callable_mp(this, &NavigationRegion3DEditor::_on_navmesh_multibake_confirmed));
	multibake_dialog->connect(SNAME("canceled"), callable_mp(this, &NavigationRegion3DEditor::_on_navmesh_multibake_canceled));
	multibake_dialog->set_hide_on_ok(false);
	multibake_dialog->set_title(TTR("Baking NavigationMesh ..."));
}

void NavigationRegion3DEditorPlugin::edit(Object *p_object) {
	LocalVector<NavigationRegion3D *> regions;

	{
		NavigationRegion3D *region = Object::cast_to<NavigationRegion3D>(p_object);
		if (region) {
			regions.push_back(region);
			navigation_region_editor->edit(LocalVector<NavigationRegion3D *>(regions));
			return;
		}
	}

	Ref<MultiNodeEdit> mne = Ref<MultiNodeEdit>(p_object);
	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (mne.is_valid() && edited_scene) {
		for (int i = 0; i < mne->get_node_count(); i++) {
			NavigationRegion3D *region = Object::cast_to<NavigationRegion3D>(edited_scene->get_node(mne->get_node(i)));
			if (region) {
				regions.push_back(region);
			}
		}
	}

	navigation_region_editor->edit(LocalVector<NavigationRegion3D *>(regions));
}

bool NavigationRegion3DEditorPlugin::handles(Object *p_object) const {
	if (Object::cast_to<NavigationRegion3D>(p_object)) {
		return true;
	}

	Ref<MultiNodeEdit> mne = Ref<MultiNodeEdit>(p_object);
	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (mne.is_valid() && edited_scene) {
		for (int i = 0; i < mne->get_node_count(); i++) {
			if (Object::cast_to<NavigationRegion3D>(edited_scene->get_node(mne->get_node(i)))) {
				return true;
			}
		}
	}
	return false;
}

void NavigationRegion3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		navigation_region_editor->show();
		navigation_region_editor->bake_hbox->show();
	} else {
		navigation_region_editor->hide();
		navigation_region_editor->bake_hbox->hide();
		navigation_region_editor->edit(LocalVector<NavigationRegion3D *>());
	}
}

NavigationRegion3DEditorPlugin::NavigationRegion3DEditorPlugin() {
	navigation_region_editor = memnew(NavigationRegion3DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(navigation_region_editor);
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, navigation_region_editor->bake_hbox);
	navigation_region_editor->hide();
	navigation_region_editor->bake_hbox->hide();

	gizmo_plugin.instantiate();
	Node3DEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
}
