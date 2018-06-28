#include "plugin_create_dialog.h"

#include "../core/io/config_file.h"
#include "../core/os/dir_access.h"
#include "../editor/editor_node.h"
#include "../editor/editor_plugin.h"
#include "../modules/gdscript/gdscript.h"
#include "../scene/gui/grid_container.h"

void PluginCreateDialog::_clear_fields() {
	name_edit->set_text("");
	subfolder_edit->set_text("");
	desc_edit->set_text("");
	author_edit->set_text("");
	version_edit->set_text("");
	script_edit->set_text("");
}

void PluginCreateDialog::_on_confirmed() {

	String path = "res://addons/" + subfolder_edit->get_text();

	if (!_edit_mode) {
		DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (!d || d->make_dir_recursive(path) != OK)
			return;
	}

	Ref<ConfigFile> cf = memnew(ConfigFile);
	cf->set_value("plugin", "name", name_edit->get_text());
	cf->set_value("plugin", "description", desc_edit->get_text());
	cf->set_value("plugin", "author", author_edit->get_text());
	cf->set_value("plugin", "version", version_edit->get_text());
	cf->set_value("plugin", "script", script_edit->get_text());

	cf->save(path.plus_file("plugin.cfg"));

	if (!_edit_mode) {
		String type = script_option_edit->get_item_text(script_option_edit->get_selected());

		Ref<Script> script;

		if (type == GDScriptLanguage::get_singleton()->get_name()) {
			Ref<GDScript> gdscript = memnew(GDScript);
			gdscript->set_source_code(
					"tool\n"
					"extends EditorPlugin\n"
					"\n"
					"func _enter_tree():\n"
					"\tpass");
			ResourceSaver::save(path.plus_file(script_edit->get_text()), gdscript);
			script = gdscript;
		}
		//TODO: other languages

		emit_signal("plugin_ready", script.operator->(), active_edit->is_pressed() ? name_edit->get_text() : "");
	} else {
		EditorNode::get_singleton()->get_project_settings()->update_plugins();
	}
	_clear_fields();
}

void PluginCreateDialog::_on_cancelled() {
	_clear_fields();
}

void PluginCreateDialog::_on_required_text_changed(const String &p_text) {
	String ext = script_option_edit->get_item_metadata(script_option_edit->get_selected());
	get_ok()->set_disabled(script_edit->get_text().get_basename().empty() || script_edit->get_text().get_extension() != ext || name_edit->get_text().empty());
}

void PluginCreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect("confirmed", this, "_on_confirmed");
			get_cancel()->connect("pressed", this, "_on_cancelled");
		} break;
	}
}

void PluginCreateDialog::config(const String &p_plugin_dir_name) {
	if (p_plugin_dir_name.length()) {
		EditorPlugin *plugin = EditorNode::get_singleton()->get_addon_plugin(p_plugin_dir_name);
		Ref<ConfigFile> cf = plugin->get_config();

		name_edit->set_text(cf->get_value("plugin", "name"));
		subfolder_edit->set_text(p_plugin_dir_name); //make read-only
		desc_edit->set_text(cf->get_value("plugin", "description"));
		author_edit->set_text(cf->get_value("plugin", "author"));
		version_edit->set_text(cf->get_value("plugin", "version"));
		script_edit->set_text(cf->get_value("plugin", "script"));

		_edit_mode = true;
		active_edit->hide();
		Object::cast_to<Label>(active_edit->get_parent()->get_child(active_edit->get_index() - 1))->hide();
		subfolder_edit->hide();
		Object::cast_to<Label>(subfolder_edit->get_parent()->get_child(subfolder_edit->get_index() - 1))->hide();
		set_title(TTR("Edit a Plugin"));
	} else {
		_clear_fields();
		_edit_mode = false;
		active_edit->show();
		Object::cast_to<Label>(active_edit->get_parent()->get_child(active_edit->get_index() - 1))->show();
		subfolder_edit->show();
		Object::cast_to<Label>(subfolder_edit->get_parent()->get_child(subfolder_edit->get_index() - 1))->show();
		set_title(TTR("Create a Plugin"));
	}
	get_ok()->set_disabled(!_edit_mode);
	get_ok()->set_text(_edit_mode ? TTR("Update") : TTR("Create"));
}

void PluginCreateDialog::_bind_methods() {
	ClassDB::bind_method("_on_required_text_changed", &PluginCreateDialog::_on_required_text_changed);
	ClassDB::bind_method("_on_confirmed", &PluginCreateDialog::_on_confirmed);
	ClassDB::bind_method("_on_cancelled", &PluginCreateDialog::_on_cancelled);
	ADD_SIGNAL(MethodInfo("plugin_ready", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script"), PropertyInfo(Variant::STRING, "activate_name")));
}

PluginCreateDialog::PluginCreateDialog() {
	get_ok()->set_disabled(true);
	set_hide_on_ok(true);

	GridContainer *grid = memnew(GridContainer);
	grid->set_columns(2);
	add_child(grid);

	Label *name_lb = memnew(Label);
	name_lb->set_text(TTR("Plugin Name:"));
	grid->add_child(name_lb);

	name_edit = memnew(LineEdit);
	name_edit->connect("text_changed", this, "_on_required_text_changed");
	name_edit->set_placeholder("MyPlugin");
	grid->add_child(name_edit);

	Label *subfolder_lb = memnew(Label);
	subfolder_lb->set_text(TTR("Subfolder:"));
	grid->add_child(subfolder_lb);

	subfolder_edit = memnew(LineEdit);
	subfolder_edit->set_placeholder("\"my_plugin\" -> res://addons/my_plugin");
	grid->add_child(subfolder_edit);

	Label *desc_lb = memnew(Label);
	desc_lb->set_text(TTR("Description:"));
	grid->add_child(desc_lb);

	desc_edit = memnew(TextEdit);
	desc_edit->set_custom_minimum_size(Size2(400.0f, 50.0f));
	grid->add_child(desc_edit);

	Label *author_lb = memnew(Label);
	author_lb->set_text(TTR("Author:"));
	grid->add_child(author_lb);

	author_edit = memnew(LineEdit);
	author_edit->set_placeholder("Godette");
	grid->add_child(author_edit);

	Label *version_lb = memnew(Label);
	version_lb->set_text(TTR("Version:"));
	grid->add_child(version_lb);

	version_edit = memnew(LineEdit);
	version_edit->set_placeholder("1.0");
	grid->add_child(version_edit);

	Label *script_option_lb = memnew(Label);
	script_option_lb->set_text(TTR("Language:"));
	grid->add_child(script_option_lb);

	script_option_edit = memnew(OptionButton);
	script_option_edit->add_item(GDScriptLanguage::get_singleton()->get_name());
	script_option_edit->set_item_metadata(0, GDScriptLanguage::get_singleton()->get_extension());
	script_option_edit->select(0);
	//TODO: add other languages
	grid->add_child(script_option_edit);

	Label *script_lb = memnew(Label);
	script_lb->set_text(TTR("Script Name:"));
	grid->add_child(script_lb);

	script_edit = memnew(LineEdit);
	script_edit->connect("text_changed", this, "_on_required_text_changed");
	script_edit->set_placeholder("\"plugin.gd\" -> res://addons/my_plugin/plugin.gd");
	grid->add_child(script_edit);

	Label *active_lb = memnew(Label);
	active_lb->set_text(TTR("Activate now?"));
	grid->add_child(active_lb);

	active_edit = memnew(CheckBox);
	active_edit->set_pressed(true);
	grid->add_child(active_edit);
}

PluginCreateDialog::~PluginCreateDialog() {
}
