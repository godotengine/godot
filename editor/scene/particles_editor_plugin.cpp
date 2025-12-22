/**************************************************************************/
/*  particles_editor_plugin.cpp                                           */
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

#include "particles_editor_plugin.h"

#include "editor/docks/scene_tree_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/spin_box.h"

void ParticlesEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (handled_type.ends_with("2D")) {
				add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);
			} else if (handled_type.ends_with("3D")) {
				add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, toolbar);
			} else {
				DEV_ASSERT(false);
			}

			menu->set_button_icon(menu->get_editor_theme_icon(handled_type));
			menu->set_text(handled_type);

			PopupMenu *popup = menu->get_popup();
			popup->add_shortcut(ED_SHORTCUT("particles/restart_emission", TTRC("Restart Emission"), KeyModifierMask::CTRL | Key::R), MENU_RESTART);
			_add_menu_options(popup);
			popup->add_item(conversion_option_name, MENU_OPTION_CONVERT);
			popup->set_item_metadata(-1, "disable_on_multiselect");
		} break;
	}
}

bool ParticlesEditorPlugin::need_show_lifetime_dialog(SpinBox *p_seconds) {
	// Add one second to the default generation lifetime, since the progress is updated every second.
	p_seconds->set_value(MAX(1.0, std::trunc(edited_node->get("lifetime").operator double()) + 1.0));

	if (p_seconds->get_value() >= 11.0 + CMP_EPSILON) {
		// Only pop up the time dialog if the particle's lifetime is long enough to warrant shortening it.
		return true;
	} else {
		// Generate the visibility rect/AABB immediately.
		return false;
	}
}

void ParticlesEditorPlugin::disable_single_select_operations(bool p_disabled) const {
	PopupMenu *popup = menu->get_popup();
	for (int i = 0; i < popup->get_item_count(); i++) {
		if (popup->get_item_metadata(i) == "disable_on_multiselect") {
			popup->set_item_disabled(i, p_disabled);
			popup->set_item_tooltip(i, p_disabled ? TTR("Functionality not available in multi-selection.") : TTR(""));
		}
	}
}

void ParticlesEditorPlugin::_menu_callback(int p_idx) {
	switch (p_idx) {
		case MENU_OPTION_CONVERT: {
			Node *converted_node = _convert_particles();

			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			ur->create_action(conversion_option_name, UndoRedo::MERGE_DISABLE, edited_node);
			SceneTreeDock::get_singleton()->replace_node(edited_node, converted_node);
			ur->commit_action(false);
		} break;

		case MENU_RESTART: {
			if (EditorNode::get_singleton()->get_editor_selection()->get_selected_nodes().size() == 1) {
				edited_node->call("restart");
			} else {
				for (int i = 0; i < mne->get_node_count(); i++) {
					Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
					if (edited_scene->get_node(mne->get_node(i))->is_class(handled_type)) {
						edited_scene->get_node(mne->get_node(i))->call("restart");
					}
				}
			}
		}
	}
}

void ParticlesEditorPlugin::edit(Object *p_object) {
	{
		edited_node = Object::cast_to<Node>(p_object);
		if (edited_node) {
			return;
		}
	}

	mne = Ref<MultiNodeEdit>(p_object);
	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (mne.is_valid() && edited_scene) {
		for (int i = 0; i < mne->get_node_count(); i++) {
			edited_node = Object::cast_to<Node>(edited_scene->get_node(mne->get_node(i)));
			if (edited_node && edited_node->is_class(handled_type)) {
				return;
			}
		}
	}
	edited_node = nullptr;
}

bool ParticlesEditorPlugin::handles(Object *p_object) const {
	if (p_object->is_class(handled_type)) {
		disable_single_select_operations(false);
		return true;
	}

	Ref<MultiNodeEdit> tmp_mne = Ref<MultiNodeEdit>(p_object);
	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (tmp_mne.is_valid() && edited_scene) {
		bool has_particles = false;
		for (int i = 0; i < tmp_mne->get_node_count(); i++) {
			if (edited_scene->get_node(tmp_mne->get_node(i))->is_class(handled_type)) {
				if (has_particles) {
					disable_single_select_operations(true);
					return true;
				} else {
					has_particles = true;
				}
			}
		}
	}

	return false;
}

void ParticlesEditorPlugin::make_visible(bool p_visible) {
	toolbar->set_visible(p_visible);
}

ParticlesEditorPlugin::ParticlesEditorPlugin() {
	toolbar = memnew(HBoxContainer);
	toolbar->hide();

	menu = memnew(MenuButton);
	menu->set_switch_on_hover(true);
	menu->set_flat(false);
	menu->set_theme_type_variation("FlatMenuButton");
	toolbar->add_child(menu);
	menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &ParticlesEditorPlugin::_menu_callback));
}
