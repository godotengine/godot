/**************************************************************************/
/*  scene_create_dialog.cpp                                               */
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

#include "scene_create_dialog.h"

#include "core/io/dir_access.h"
#include "editor/create_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/packed_scene.h"

void SceneCreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			select_node_button->set_icon(get_theme_icon(SNAME("ClassList"), SNAME("EditorIcons")));
			node_type_2d->set_icon(get_theme_icon(SNAME("Node2D"), SNAME("EditorIcons")));
			node_type_3d->set_icon(get_theme_icon(SNAME("Node3D"), SNAME("EditorIcons")));
			node_type_gui->set_icon(get_theme_icon(SNAME("Control"), SNAME("EditorIcons")));
			node_type_other->add_theme_icon_override(SNAME("icon"), get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
			status_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("Tree")));
		} break;
	}
}

void SceneCreateDialog::config(const String &p_dir) {
	directory = p_dir;
	root_name_edit->set_text("");
	scene_name_edit->set_text("");
	scene_name_edit->call_deferred(SNAME("grab_focus"));
	update_dialog();
}

void SceneCreateDialog::accept_create() {
	if (!get_ok_button()->is_disabled()) {
		hide();
		emit_signal(SNAME("confirmed"));
	}
}

void SceneCreateDialog::browse_types() {
	select_node_dialog->popup_create(true);
	select_node_dialog->set_title(TTR("Pick Root Node Type"));
	select_node_dialog->set_ok_button_text(TTR("Pick"));
}

void SceneCreateDialog::on_type_picked() {
	other_type_display->set_text(select_node_dialog->get_selected_type().get_slice(" ", 0));
	if (node_type_other->is_pressed()) {
		update_dialog();
	} else {
		node_type_other->set_pressed(true); // Calls update_dialog() via group.
	}
}

void SceneCreateDialog::update_dialog() {
	scene_name = scene_name_edit->get_text().strip_edges();
	update_error(file_error_label, MSG_OK, TTR("Scene name is valid."));

	bool is_valid = true;
	if (scene_name.is_empty()) {
		update_error(file_error_label, MSG_ERROR, TTR("Scene name is empty."));
		is_valid = false;
	}

	if (is_valid) {
		if (!scene_name.ends_with(".")) {
			scene_name += ".";
		}
		scene_name += scene_extension_picker->get_selected_metadata().operator String();
	}

	if (is_valid && !scene_name.is_valid_filename()) {
		update_error(file_error_label, MSG_ERROR, TTR("File name invalid."));
		is_valid = false;
	}

	if (is_valid) {
		scene_name = directory.path_join(scene_name);
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (da->file_exists(scene_name)) {
			update_error(file_error_label, MSG_ERROR, TTR("File already exists."));
			is_valid = false;
		}
	}

	const StringName root_type_name = StringName(other_type_display->get_text());
	if (has_theme_icon(root_type_name, SNAME("EditorIcons"))) {
		node_type_other->set_icon(get_theme_icon(root_type_name, SNAME("EditorIcons")));
	} else {
		node_type_other->set_icon(nullptr);
	}

	update_error(node_error_label, MSG_OK, "Root node valid.");

	root_name = root_name_edit->get_text().strip_edges();
	if (root_name.is_empty()) {
		root_name = scene_name.get_file().get_basename();
	}

	if (root_name.is_empty() || root_name.validate_node_name().size() != root_name.size()) {
		update_error(node_error_label, MSG_ERROR, TTR("Invalid root node name."));
		is_valid = false;
	}

	get_ok_button()->set_disabled(!is_valid);
}

void SceneCreateDialog::update_error(Label *p_label, MsgType p_type, const String &p_msg) {
	p_label->set_text(String::utf8("•  ") + p_msg);
	switch (p_type) {
		case MSG_OK:
			p_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
			break;
		case MSG_ERROR:
			p_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
			break;
	}
}

String SceneCreateDialog::get_scene_path() const {
	return scene_name;
}

Node *SceneCreateDialog::create_scene_root() {
	ERR_FAIL_NULL_V(node_type_group->get_pressed_button(), nullptr);
	RootType type = (RootType)node_type_group->get_pressed_button()->get_meta(type_meta).operator int();

	Node *root = nullptr;
	switch (type) {
		case ROOT_2D_SCENE:
			root = memnew(Node2D);
			break;
		case ROOT_3D_SCENE:
			root = memnew(Node3D);
			break;
		case ROOT_USER_INTERFACE: {
			Control *gui_ctl = memnew(Control);
			// Making the root control full rect by default is more useful for resizable UIs.
			gui_ctl->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
			gui_ctl->set_grow_direction_preset(Control::PRESET_FULL_RECT);
			root = gui_ctl;
		} break;
		case ROOT_OTHER:
			root = Object::cast_to<Node>(select_node_dialog->instantiate_selected());
			break;
	}

	ERR_FAIL_NULL_V(root, nullptr);
	root->set_name(root_name);
	return root;
}

SceneCreateDialog::SceneCreateDialog() {
	select_node_dialog = memnew(CreateDialog);
	add_child(select_node_dialog);
	select_node_dialog->set_base_type("Node");
	select_node_dialog->select_base();
	select_node_dialog->connect("create", callable_mp(this, &SceneCreateDialog::on_type_picked));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	GridContainer *gc = memnew(GridContainer);
	main_vb->add_child(gc);
	gc->set_columns(2);

	{
		Label *label = memnew(Label(TTR("Root Type:")));
		gc->add_child(label);
		label->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);

		VBoxContainer *vb = memnew(VBoxContainer);
		gc->add_child(vb);

		node_type_group.instantiate();

		node_type_2d = memnew(CheckBox);
		vb->add_child(node_type_2d);
		node_type_2d->set_text(TTR("2D Scene"));
		node_type_2d->set_button_group(node_type_group);
		node_type_2d->set_meta(type_meta, ROOT_2D_SCENE);
		node_type_2d->set_pressed(true);

		node_type_3d = memnew(CheckBox);
		vb->add_child(node_type_3d);
		node_type_3d->set_text(TTR("3D Scene"));
		node_type_3d->set_button_group(node_type_group);
		node_type_3d->set_meta(type_meta, ROOT_3D_SCENE);

		node_type_gui = memnew(CheckBox);
		vb->add_child(node_type_gui);
		node_type_gui->set_text(TTR("User Interface"));
		node_type_gui->set_button_group(node_type_group);
		node_type_gui->set_meta(type_meta, ROOT_USER_INTERFACE);

		HBoxContainer *hb = memnew(HBoxContainer);
		vb->add_child(hb);

		node_type_other = memnew(CheckBox);
		hb->add_child(node_type_other);
		node_type_other->set_button_group(node_type_group);
		node_type_other->set_meta(type_meta, ROOT_OTHER);

		Control *spacing = memnew(Control);
		hb->add_child(spacing);
		spacing->set_custom_minimum_size(Size2(4 * EDSCALE, 0));

		other_type_display = memnew(LineEdit);
		hb->add_child(other_type_display);
		other_type_display->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		other_type_display->set_editable(false);
		other_type_display->set_text("Node");

		select_node_button = memnew(Button);
		hb->add_child(select_node_button);
		select_node_button->connect("pressed", callable_mp(this, &SceneCreateDialog::browse_types));

		node_type_group->connect("pressed", callable_mp(this, &SceneCreateDialog::update_dialog).unbind(1));
	}

	{
		Label *label = memnew(Label(TTR("Scene Name:")));
		gc->add_child(label);

		HBoxContainer *hb = memnew(HBoxContainer);
		gc->add_child(hb);

		scene_name_edit = memnew(LineEdit);
		hb->add_child(scene_name_edit);
		scene_name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		scene_name_edit->connect("text_changed", callable_mp(this, &SceneCreateDialog::update_dialog).unbind(1));
		scene_name_edit->connect("text_submitted", callable_mp(this, &SceneCreateDialog::accept_create).unbind(1));

		List<String> extensions;
		Ref<PackedScene> sd = memnew(PackedScene);
		ResourceSaver::get_recognized_extensions(sd, &extensions);

		scene_extension_picker = memnew(OptionButton);
		hb->add_child(scene_extension_picker);
		for (const String &E : extensions) {
			scene_extension_picker->add_item("." + E);
			scene_extension_picker->set_item_metadata(-1, E);
		}
	}

	{
		Label *label = memnew(Label(TTR("Root Name:")));
		gc->add_child(label);

		root_name_edit = memnew(LineEdit);
		gc->add_child(root_name_edit);
		root_name_edit->set_placeholder(TTR("Leave empty to use scene name"));
		root_name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		root_name_edit->connect("text_changed", callable_mp(this, &SceneCreateDialog::update_dialog).unbind(1));
		root_name_edit->connect("text_submitted", callable_mp(this, &SceneCreateDialog::accept_create).unbind(1));
	}

	Control *spacing = memnew(Control);
	main_vb->add_child(spacing);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	status_panel = memnew(PanelContainer);
	main_vb->add_child(status_panel);
	status_panel->set_h_size_flags(Control::SIZE_FILL);
	status_panel->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	VBoxContainer *status_vb = memnew(VBoxContainer);
	status_panel->add_child(status_vb);

	file_error_label = memnew(Label);
	status_vb->add_child(file_error_label);

	node_error_label = memnew(Label);
	status_vb->add_child(node_error_label);

	set_title(TTR("Create New Scene"));
	set_min_size(Size2i(400 * EDSCALE, 0));
}
