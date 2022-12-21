/*************************************************************************/
/*  version_control_editor_plugin.cpp                                    */
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

#include "version_control_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"

VersionControlEditorPlugin *VersionControlEditorPlugin::singleton = nullptr;

void VersionControlEditorPlugin::_bind_methods() {
	// No binds required so far.
}

VersionControlEditorPlugin *VersionControlEditorPlugin::get_singleton() {
	return singleton ? singleton : memnew(VersionControlEditorPlugin);
}

bool VersionControlEditorPlugin::_assign_plugin_singleton(String p_plugin_name) {
	if (EditorVCSInterface::get_singleton()) {
		_destroy_plugin_and_ui();
	}

	Object *extension_instance = ClassDB::instantiate(p_plugin_name);
	ERR_FAIL_NULL_V_MSG(extension_instance, false, "Received a nullptr VCS extension instance during construction.");

	EditorVCSInterface *vcs_plugin = Object::cast_to<EditorVCSInterface>(extension_instance);
	ERR_FAIL_NULL_V_MSG(vcs_plugin, false, vformat("Could not cast VCS extension instance to %s.", EditorVCSInterface::get_class_static()));

	String res_dir = OS::get_singleton()->get_resource_dir();

	ERR_FAIL_COND_V_MSG(!vcs_plugin->initialize(res_dir), false, vformat("Could not initialize %s.", p_plugin_name));

	EditorVCSInterface::set_singleton(vcs_plugin);
	return true;
}

void VersionControlEditorPlugin::_instantiate_plugin_and_ui(String p_plugin_name) {
	if (!_assign_plugin_singleton(p_plugin_name)) {
		toggle_vcs_choice->set_disabled(true);
		ERR_FAIL_MSG(false, "Could not assign VCS plugin instance as singleton.");
	}

	EditorVCSInterface::get_singleton()->attach_ui(this);
	toggle_vcs_choice->set_disabled(false);
}

void VersionControlEditorPlugin::_destroy_plugin_and_ui() {
	if (!EditorVCSInterface::get_singleton()) {
		return;
	}

	EditorVCSInterface::get_singleton()->remove_ui(this);

	EditorVCSInterface::get_singleton()->shut_down();
	memdelete(EditorVCSInterface::get_singleton());
	EditorVCSInterface::set_singleton(nullptr);

	toggle_vcs_choice->set_disabled(true);
}

void VersionControlEditorPlugin::_toggle_vcs_integration(bool p_toggled) {
	if (p_toggled) {
		_instantiate_plugin_and_ui(set_up_choice->get_item_text(set_up_choice->get_selected_id()));
	} else {
		_destroy_plugin_and_ui();
	}
}

void VersionControlEditorPlugin::_create_vcs_metadata_files() {
	String dir = "res://";
	EditorVCSInterface::create_vcs_metadata_files(EditorVCSInterface::VCSMetadata(metadata_selection->get_selected()), dir);
}

void VersionControlEditorPlugin::popup_vcs_metadata_dialog() {
	metadata_dialog->popup_centered();
}

void VersionControlEditorPlugin::popup_vcs_set_up_dialog(const Control *p_gui_base) {
	List<StringName> available_plugins = fetch_available_vcs_plugin_names();
	if (!available_plugins.is_empty()) {
		Size2 popup_size = Size2(400, 100);
		Size2 window_size = p_gui_base->get_viewport_rect().size;
		popup_size.x = MIN(window_size.x * 0.5, popup_size.x);
		popup_size.y = MIN(window_size.y * 0.5, popup_size.y);

		set_up_choice->clear();
		for (int i = 0; i < available_plugins.size(); i++) {
			set_up_choice->add_item(available_plugins[i]);
		}

		set_up_dialog->popup_centered_clamped(popup_size * EDSCALE);
	} else {
		// TODO: Give info to user on how to fix this error.
		EditorNode::get_singleton()->show_warning(TTR("No VCS plugins are available in the project. Install a VCS plugin to use VCS integration features."), TTR("Error"));
	}
}

List<StringName> VersionControlEditorPlugin::fetch_available_vcs_plugin_names() {
	List<StringName> available_plugins;
	ClassDB::get_direct_inheriters_from_class(EditorVCSInterface::get_class_static(), &available_plugins);
	return available_plugins;
}

VersionControlEditorPlugin::VersionControlEditorPlugin() {
	singleton = this;

	// VCS Metadata creation
	metadata_dialog = memnew(ConfirmationDialog);
	metadata_dialog->set_title(TTR("Create Version Control Metadata"));
	metadata_dialog->set_min_size(Size2(200, 40));
	metadata_dialog->get_ok_button()->connect(SNAME("pressed"), callable_mp(this, &VersionControlEditorPlugin::_create_vcs_metadata_files));
	add_child(metadata_dialog);

	VBoxContainer *metadata_vb = memnew(VBoxContainer);
	metadata_dialog->add_child(metadata_vb);

	HBoxContainer *metadata_hb = memnew(HBoxContainer);
	metadata_hb->set_custom_minimum_size(Size2(200, 20));
	metadata_vb->add_child(metadata_hb);

	Label *l = memnew(Label);
	l->set_text(TTR("Create VCS metadata files for:"));
	metadata_hb->add_child(l);

	metadata_selection = memnew(OptionButton);
	metadata_selection->set_custom_minimum_size(Size2(100, 20));
	metadata_selection->add_item("None", (int)EditorVCSInterface::VCSMetadata::NONE);
	metadata_selection->add_item("Git", (int)EditorVCSInterface::VCSMetadata::GIT);
	metadata_selection->select((int)EditorVCSInterface::VCSMetadata::GIT);
	metadata_hb->add_child(metadata_selection);

	l = memnew(Label);
	l->set_text(TTR("Existing VCS metadata files will be overwritten."));
	metadata_vb->add_child(l);

	// Settings UI
	set_up_dialog = memnew(AcceptDialog);
	set_up_dialog->set_title(TTR("Local Settings"));
	set_up_dialog->set_min_size(Size2(600, 100));
	set_up_dialog->set_hide_on_ok(true);
	add_child(set_up_dialog);

	VBoxContainer *set_up_vbc = memnew(VBoxContainer);
	set_up_vbc->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	set_up_dialog->add_child(set_up_vbc);

	HBoxContainer *set_up_hbc = memnew(HBoxContainer);
	set_up_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_up_vbc->add_child(set_up_hbc);

	// VCS Provider
	Label *set_up_vcs_label = memnew(Label);
	set_up_vcs_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_up_vcs_label->set_text(TTR("VCS Provider"));
	set_up_hbc->add_child(set_up_vcs_label);

	set_up_choice = memnew(OptionButton);
	set_up_choice->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_up_hbc->add_child(set_up_choice);

	// Connect to VCS
	HBoxContainer *toggle_vcs_hbc = memnew(HBoxContainer);
	toggle_vcs_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_up_vbc->add_child(toggle_vcs_hbc);

	Label *toggle_vcs_label = memnew(Label);
	toggle_vcs_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle_vcs_label->set_text(TTR("Connect to VCS"));
	toggle_vcs_hbc->add_child(toggle_vcs_label);

	toggle_vcs_choice = memnew(CheckButton);
	toggle_vcs_choice->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle_vcs_choice->set_pressed_no_signal(false);
	toggle_vcs_choice->connect(SNAME("toggled"), callable_mp(this, &VersionControlEditorPlugin::_toggle_vcs_integration));
	toggle_vcs_hbc->add_child(toggle_vcs_choice);

	// Separator before VCS settings
	set_up_hs = memnew(HSeparator);
	set_up_hs->hide();
	set_up_vbc->add_child(set_up_hs);

	// VCS plugin settings box
	vcs_plugin_settings = memnew(VBoxContainer);
	vcs_plugin_settings->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_up_vbc->add_child(vcs_plugin_settings);
}

VersionControlEditorPlugin::~VersionControlEditorPlugin() {
	_toggle_vcs_integration(false);
}
