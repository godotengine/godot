/*************************************************************************/
/*  settings_config_dialog.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
	property_editor->update_tree();
	popup_centered_ratio(0.7);
}


void EditorSettingsDialog::_plugin_install() {

	EditorSettings::Plugin plugin =	EditorSettings::get_singleton()->get_plugins()[plugin_setting_edit];

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	da->change_dir("res://");
	if (da->change_dir("plugins")!=OK) {

		Error err = da->make_dir("plugins");
		if (err)
			memdelete(da);
		ERR_FAIL_COND(err!=OK);
		err = da->change_dir("plugins");
		if (err)
			memdelete(da);
		ERR_FAIL_COND(err!=OK);
	}

	if (da->change_dir(plugin_setting_edit)!=OK) {

		Error err = da->make_dir(plugin_setting_edit);
		if (err)
			memdelete(da);
		ERR_FAIL_COND(err!=OK);
		err = da->change_dir(plugin_setting_edit);
		if (err)
			memdelete(da);
		ERR_FAIL_COND(err!=OK);
	}

	Vector<String> ifiles=plugin.install_files;

	if (ifiles.find("plugin.cfg")==-1) {
		ifiles.push_back("plugin.cfg");
	}

	if (ifiles.find(plugin.script)==-1) {
		ifiles.push_back(plugin.script);
	}

	for(int i=0;i<ifiles.size();i++) {

		String target = "res://plugins/"+plugin_setting_edit+"/"+ifiles[i];
		Error err = da->copy(EditorSettings::get_singleton()->get_settings_path().plus_file("plugins/"+plugin_setting_edit+"/"+ifiles[i]),target);
		if (err)
			memdelete(da);
		ERR_EXPLAIN("Error copying to file "+target);
		ERR_FAIL_COND(err!=OK);
		EditorFileSystem::get_singleton()->update_file(target);
	}

	memdelete(da);

	Globals::get_singleton()->set("plugins/"+plugin_setting_edit,"res://plugins/"+plugin_setting_edit);
	Globals::get_singleton()->set_persisting("plugins/"+plugin_setting_edit,true);
	EditorSettings::get_singleton()->load_installed_plugin(plugin_setting_edit);
	Globals::get_singleton()->save();


	_update_plugins();
}

void EditorSettingsDialog::_rescan_plugins() {

	EditorSettings::get_singleton()->scan_plugins();
	_update_plugins();
}

void EditorSettingsDialog::_plugin_edited() {

	if (updating)
		return;

	TreeItem *ti=plugins->get_edited();
	if (!ti)
		return;

	String plugin = ti->get_metadata(0);
	bool enabled = ti->is_checked(0);

	EditorSettings::get_singleton()->set_plugin_enabled(plugin,enabled);
}

void EditorSettingsDialog::_plugin_settings(Object *p_obj,int p_cell,int p_index) {

	TreeItem *ti=p_obj->cast_to<TreeItem>();

	EditorSettings::Plugin plugin =	EditorSettings::get_singleton()->get_plugins()[ti->get_metadata(0)];

	plugin_description->clear();
	plugin_description->parse_bbcode(plugin.description);
	plugin_setting_edit = ti->get_metadata(0);
	if (plugin.installs) {
		if (Globals::get_singleton()->has("plugins/"+plugin_setting_edit))
			plugin_setting->get_ok()->set_text("Re-Install to Project");
		else
			plugin_setting->get_ok()->set_text("Install to Project");
		plugin_setting->get_ok()->show();
		plugin_setting->get_cancel()->set_text("Close");
	} else {
		plugin_setting->get_ok()->hide();
		plugin_setting->get_cancel()->set_text("Close");
	}

	plugin_setting->set_title(plugin.name);
	plugin_setting->popup_centered(Size2(300,200));
}

void EditorSettingsDialog::_update_plugins() {


	updating=true;

	plugins->clear();
	TreeItem *root = plugins->create_item(NULL);
	plugins->set_hide_root(true);

	Color sc = get_color("prop_subsection","Editor");
	TreeItem *editor = plugins->create_item(root);
	editor->set_text(0,"Editor Plugins");
	editor->set_custom_bg_color(0,sc);
	editor->set_custom_bg_color(1,sc);
	editor->set_custom_bg_color(2,sc);

	TreeItem *install = plugins->create_item(root);
	install->set_text(0,"Installable Plugins");
	install->set_custom_bg_color(0,sc);
	install->set_custom_bg_color(1,sc);
	install->set_custom_bg_color(2,sc);

	for (const Map<String,EditorSettings::Plugin>::Element *E=EditorSettings::get_singleton()->get_plugins().front();E;E=E->next()) {


		TreeItem *ti = plugins->create_item(E->get().installs?install:editor);
		if (!E->get().installs) {
			ti->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
			ti->set_editable(0,true);
			if (EditorSettings::get_singleton()->is_plugin_enabled(E->key()))
				ti->set_checked(0,true);

			ti->set_text(0,E->get().name);
		} else {

			if (Globals::get_singleton()->has("plugins/"+E->key())) {

				ti->set_text(0,E->get().name+" (Installed)");
			} else {
				ti->set_text(0,E->get().name);
			}
		}


		ti->add_button(0,get_icon("Tools","EditorIcons"),0);
		ti->set_text(1,E->get().author);
		ti->set_text(2,E->get().version);
		ti->set_metadata(0,E->key());

	}

	if (!editor->get_children())
		memdelete(editor);
	if (!install->get_children())
		memdelete(install);

	updating=false;

}

void EditorSettingsDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		rescan_plugins->set_icon(get_icon("Reload","EditorIcons"));
		_update_plugins();
	}
}

void EditorSettingsDialog::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_settings_save"),&EditorSettingsDialog::_settings_save);
	ObjectTypeDB::bind_method(_MD("_settings_changed"),&EditorSettingsDialog::_settings_changed);
	ObjectTypeDB::bind_method(_MD("_rescan_plugins"),&EditorSettingsDialog::_rescan_plugins);
	ObjectTypeDB::bind_method(_MD("_plugin_settings"),&EditorSettingsDialog::_plugin_settings);
	ObjectTypeDB::bind_method(_MD("_plugin_edited"),&EditorSettingsDialog::_plugin_edited);
	ObjectTypeDB::bind_method(_MD("_plugin_install"),&EditorSettingsDialog::_plugin_install);
}

EditorSettingsDialog::EditorSettingsDialog() {

	set_title("Editor Settings");

	tabs = memnew( TabContainer );
	add_child(tabs);
	set_child_rect(tabs);

	property_editor = memnew( PropertyEditor );
	property_editor->hide_top_label();
	tabs->add_child(property_editor);
	property_editor->set_name("General");

	VBoxContainer *vbc = memnew( VBoxContainer );
	tabs->add_child(vbc);
	vbc->set_name("Plugins");

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_child(hbc);
	hbc->add_child( memnew( Label("Plugin List: ")));
	hbc->add_spacer();
	Button *load = memnew( Button );
	load->set_text("Load..");
	Button *rescan = memnew( Button );
	rescan_plugins=rescan;
	rescan_plugins->connect("pressed",this,"_rescan_plugins");
	hbc->add_child(load);
	hbc->add_child(rescan);
	plugins = memnew( Tree );
	MarginContainer *mc = memnew( MarginContainer);
	vbc->add_child(mc);
	mc->add_child(plugins);
	mc->set_v_size_flags(SIZE_EXPAND_FILL);
	plugins->set_columns(3);
	plugins->set_column_title(0,"Name");
	plugins->set_column_title(1,"Author");
	plugins->set_column_title(2,"Version");
	plugins->set_column_expand(2,false);
	plugins->set_column_min_width(2,100);
	plugins->set_column_titles_visible(true);
	plugins->connect("button_pressed",this,"_plugin_settings");
	plugins->connect("item_edited",this,"_plugin_edited");

	plugin_setting = memnew( ConfirmationDialog );
	add_child(plugin_setting);
	plugin_description = memnew( RichTextLabel );
	plugin_setting->add_child(plugin_description);
	plugin_setting->set_child_rect(plugin_description);
	plugin_setting->connect("confirmed",this,"_plugin_install");



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
	install_confirm = memnew( ConfirmationDialog );
	add_child(install_confirm);

	updating=false;

}
