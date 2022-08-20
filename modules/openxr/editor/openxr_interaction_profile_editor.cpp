/*************************************************************************/
/*  openxr_interaction_profile_editor.cpp                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "openxr_interaction_profile_editor.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/text_edit.h"

///////////////////////////////////////////////////////////////////////////
// Interaction profile editor base

void OpenXRInteractionProfileEditorBase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_add_binding", "action", "path"), &OpenXRInteractionProfileEditorBase::_add_binding);
	ClassDB::bind_method(D_METHOD("_remove_binding", "action", "path"), &OpenXRInteractionProfileEditorBase::_remove_binding);
	ClassDB::bind_method(D_METHOD("_update_interaction_profile"), &OpenXRInteractionProfileEditorBase::_update_interaction_profile);
}

void OpenXRInteractionProfileEditorBase::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_interaction_profile();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_theme_changed();
		} break;
	}
}

void OpenXRInteractionProfileEditorBase::_add_binding(const String p_action, const String p_path) {
	ERR_FAIL_COND(action_map.is_null());
	ERR_FAIL_COND(interaction_profile.is_null());

	Ref<OpenXRAction> action = action_map->get_action(p_action);
	ERR_FAIL_COND(action.is_null());

	Ref<OpenXRIPBinding> binding = interaction_profile->get_binding_for_action(action);
	if (binding.is_null()) {
		// create a new binding
		binding.instantiate();
		binding->set_action(action);
		interaction_profile->add_binding(binding);
	}

	binding->add_path(p_path);

	// Update our toplevel paths
	action->set_toplevel_paths(action_map->get_top_level_paths(action));

	call_deferred("_update_interaction_profile");
}

void OpenXRInteractionProfileEditorBase::_remove_binding(const String p_action, const String p_path) {
	ERR_FAIL_COND(action_map.is_null());
	ERR_FAIL_COND(interaction_profile.is_null());

	Ref<OpenXRAction> action = action_map->get_action(p_action);
	ERR_FAIL_COND(action.is_null());

	Ref<OpenXRIPBinding> binding = interaction_profile->get_binding_for_action(action);
	if (binding.is_valid()) {
		binding->remove_path(p_path);

		if (binding->get_path_count() == 0) {
			interaction_profile->remove_binding(binding);
		}

		// Update our toplevel paths
		action->set_toplevel_paths(action_map->get_top_level_paths(action));

		call_deferred("_update_interaction_profile");
	}
}

OpenXRInteractionProfileEditorBase::OpenXRInteractionProfileEditorBase(Ref<OpenXRActionMap> p_action_map, Ref<OpenXRInteractionProfile> p_interaction_profile) {
	action_map = p_action_map;
	interaction_profile = p_interaction_profile;
	String profile_path = interaction_profile->get_interaction_profile_path();
	String profile_name = profile_path;

	profile_def = OpenXRDefs::get_profile(profile_path);
	if (profile_def != nullptr) {
		profile_name = profile_def->display_name;
	}

	set_name(profile_name);
	set_h_size_flags(SIZE_EXPAND_FILL);
	set_v_size_flags(SIZE_EXPAND_FILL);
}

///////////////////////////////////////////////////////////////////////////
// Default interaction profile editor

void OpenXRInteractionProfileEditor::select_action_for(const String p_io_path) {
	selecting_for_io_path = p_io_path;
	select_action_dialog->open();
}

void OpenXRInteractionProfileEditor::action_selected(const String p_action) {
	_add_binding(p_action, selecting_for_io_path);
	selecting_for_io_path = "";
}

void OpenXRInteractionProfileEditor::_add_io_path(VBoxContainer *p_container, const OpenXRDefs::IOPath *p_io_path) {
	HBoxContainer *path_hb = memnew(HBoxContainer);
	path_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_container->add_child(path_hb);

	Label *path_label = memnew(Label);
	path_label->set_text(p_io_path->display_name);
	path_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	path_hb->add_child(path_label);

	Label *type_label = memnew(Label);
	switch (p_io_path->action_type) {
		case OpenXRAction::OPENXR_ACTION_BOOL: {
			type_label->set_text(TTR("Boolean"));
		} break;
		case OpenXRAction::OPENXR_ACTION_FLOAT: {
			type_label->set_text(TTR("Float"));
		} break;
		case OpenXRAction::OPENXR_ACTION_VECTOR2: {
			type_label->set_text(TTR("Vector2"));
		} break;
		case OpenXRAction::OPENXR_ACTION_POSE: {
			type_label->set_text(TTR("Pose"));
		} break;
		case OpenXRAction::OPENXR_ACTION_HAPTIC: {
			type_label->set_text(TTR("Haptic"));
		} break;
		default: {
			type_label->set_text(TTR("Unknown"));
		} break;
	}
	type_label->set_custom_minimum_size(Size2(50.0, 0.0));
	path_hb->add_child(type_label);

	Button *path_add = memnew(Button);
	path_add->set_icon(get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	path_add->set_flat(true);
	path_add->connect("pressed", callable_mp(this, &OpenXRInteractionProfileEditor::select_action_for).bind(String(p_io_path->openxr_path)));
	path_hb->add_child(path_add);

	if (interaction_profile.is_valid()) {
		String io_path = String(p_io_path->openxr_path);
		Array bindings = interaction_profile->get_bindings();
		for (int i = 0; i < bindings.size(); i++) {
			Ref<OpenXRIPBinding> binding = bindings[i];
			if (binding->has_path(io_path)) {
				Ref<OpenXRAction> action = binding->get_action();

				HBoxContainer *action_hb = memnew(HBoxContainer);
				action_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				p_container->add_child(action_hb);

				Control *indent_node = memnew(Control);
				indent_node->set_custom_minimum_size(Size2(10.0, 0.0));
				action_hb->add_child(indent_node);

				Label *action_label = memnew(Label);
				action_label->set_text(action->get_name_with_set() + ": " + action->get_localized_name());
				action_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				action_hb->add_child(action_label);

				Button *action_rem = memnew(Button);
				action_rem->set_flat(true);
				action_rem->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
				action_rem->connect("pressed", callable_mp((OpenXRInteractionProfileEditorBase *)this, &OpenXRInteractionProfileEditorBase::_remove_binding).bind(action->get_name_with_set(), String(p_io_path->openxr_path)));
				action_hb->add_child(action_rem);
			}
		}
	}
}

void OpenXRInteractionProfileEditor::_update_interaction_profile() {
	ERR_FAIL_NULL(profile_def);

	// out with the old...
	while (main_hb->get_child_count() > 0) {
		memdelete(main_hb->get_child(0));
	}

	// in with the new...

	// Determine toplevel paths
	Vector<const OpenXRDefs::TopLevelPath *> top_level_paths;
	for (int i = 0; i < profile_def->io_path_count; i++) {
		const OpenXRDefs::IOPath *io_path = &profile_def->io_paths[i];

		if (!top_level_paths.has(io_path->top_level_path)) {
			top_level_paths.push_back(io_path->top_level_path);
		}
	}

	for (int i = 0; i < top_level_paths.size(); i++) {
		PanelContainer *panel = memnew(PanelContainer);
		panel->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		main_hb->add_child(panel);
		panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("TabContainer")));

		VBoxContainer *container = memnew(VBoxContainer);
		panel->add_child(container);

		Label *label = memnew(Label);
		label->set_text(top_level_paths[i]->display_name);
		container->add_child(label);

		for (int j = 0; j < profile_def->io_path_count; j++) {
			const OpenXRDefs::IOPath *io_path = &profile_def->io_paths[j];
			if (io_path->top_level_path == top_level_paths[i]) {
				_add_io_path(container, io_path);
			}
		}
	}
}

void OpenXRInteractionProfileEditor::_theme_changed() {
	for (int i = 0; i < main_hb->get_child_count(); i++) {
		Control *panel = static_cast<Control *>(main_hb->get_child(i));
		if (panel) {
			panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("TabContainer")));
		}
	}
}

OpenXRInteractionProfileEditor::OpenXRInteractionProfileEditor(Ref<OpenXRActionMap> p_action_map, Ref<OpenXRInteractionProfile> p_interaction_profile) :
		OpenXRInteractionProfileEditorBase(p_action_map, p_interaction_profile) {
	// TODO background of scrollbox should be darker with our VBoxContainers we're adding in _update_interaction_profile the normal color

	main_hb = memnew(HBoxContainer);
	add_child(main_hb);

	select_action_dialog = memnew(OpenXRSelectActionDialog(p_action_map));
	select_action_dialog->connect("action_selected", callable_mp(this, &OpenXRInteractionProfileEditor::action_selected));
	add_child(select_action_dialog);

	_update_interaction_profile();
}
