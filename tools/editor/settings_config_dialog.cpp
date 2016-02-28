/*************************************************************************/
/*  settings_config_dialog.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "settings_config_dialog.h"
#include "editor_settings.h"
#include "scene/gui/margin_container.h"
#include "globals.h"
#include "editor_file_system.h"
void EditorSettingsDialog::ok_pressed() {

	if (!EditorSettings::get_singleton())
		return;

	_settings_save();
	timer->stop();

}

void EditorSettingsDialog::_settings_changed() {


	timer->start();
}

void EditorSettingsDialog::_settings_save() {


	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();

}

void EditorSettingsDialog::cancel_pressed() {

	if (!EditorSettings::get_singleton())
		return;

	EditorSettings::get_singleton()->notify_changes();

}


void EditorSettingsDialog::popup_edit_settings() {

	if (!EditorSettings::get_singleton())
		return;

	property_editor->edit(EditorSettings::get_singleton());
	property_editor->get_property_editor()->update_tree();

	search_box->select_all();
	search_box->grab_focus();

	popup_centered_ratio(0.7);
}



void EditorSettingsDialog::_clear_search_box() {

	if (search_box->get_text()=="")
		return;

	search_box->clear();
	property_editor->get_property_editor()->update_tree();
}

void EditorSettingsDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		clear_button->set_icon(get_icon("Close","EditorIcons"));
	}
}

void EditorSettingsDialog::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_settings_save"),&EditorSettingsDialog::_settings_save);
	ObjectTypeDB::bind_method(_MD("_settings_changed"),&EditorSettingsDialog::_settings_changed);
	ObjectTypeDB::bind_method(_MD("_clear_search_box"),&EditorSettingsDialog::_clear_search_box);
}

EditorSettingsDialog::EditorSettingsDialog() {

	set_title("Editor Settings");

	tabs = memnew( TabContainer );
	add_child(tabs);
	set_child_rect(tabs);

	VBoxContainer *vbc = memnew( VBoxContainer );
	tabs->add_child(vbc);
	vbc->set_name("General");

	HBoxContainer *hbc = memnew( HBoxContainer );
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hbc);

	Label *l = memnew( Label );
	l->set_text("Search: ");
	hbc->add_child(l);

	search_box = memnew( LineEdit );
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(search_box);

	clear_button = memnew( ToolButton );
	hbc->add_child(clear_button);
	clear_button->connect("pressed",this,"_clear_search_box");

	property_editor = memnew( SectionedPropertyEditor );
	//property_editor->hide_top_label();
	property_editor->get_property_editor()->set_use_filter(true);
	property_editor->get_property_editor()->register_text_enter(search_box);
	property_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(property_editor);

	vbc = memnew( VBoxContainer );
	tabs->add_child(vbc);
	vbc->set_name("Plugins");

	hbc = memnew( HBoxContainer );
	vbc->add_child(hbc);
	hbc->add_child( memnew( Label("Plugin List: ")));
	hbc->add_spacer();
	//Button *load = memnew( Button );
	//load->set_text("Load..");
	//hbc->add_child(load);


	//get_ok()->set_text("Apply");
	set_hide_on_ok(true);
	//get_cancel()->set_text("Close");

	timer = memnew( Timer );
	timer->set_wait_time(1.5);
	timer->connect("timeout",this,"_settings_save");
	timer->set_one_shot(true);
	add_child(timer);
	EditorSettings::get_singleton()->connect("settings_changed",this,"_settings_changed");
	get_ok()->set_text("Close");

	updating=false;

}
