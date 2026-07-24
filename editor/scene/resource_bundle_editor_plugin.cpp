/**************************************************************************/
/*  resource_bundle_editor_plugin.cpp                                     */
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

#include "resource_bundle_editor_plugin.h"

#include "core/io/dir_access.h"
#include "core/io/resource_saver.h"
#include "core/object/callable_mp.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_icon_manager.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/tab_bar.h"
#include "scene/resources/resource_bundle.h"

// ResourceBundleEditor

void ResourceBundleEditor::_bundle_tabs_resized() {
	const Size2 add_button_size = Size2(bundle_tab_add->get_size().x, bundle_tabs->get_size().y);
	if (bundle_tabs->get_offset_buttons_visible()) {
		// Move the add button to a fixed position.
		if (bundle_tab_add->get_parent() == bundle_tabs) {
			bundle_tabs->remove_child(bundle_tab_add);
			bundle_tab_add_ph->add_child(bundle_tab_add);
			bundle_tab_add->set_rect(Rect2(Point2(), add_button_size));
		}
	} else {
		// Move the add button to be after the last tab.
		if (bundle_tab_add->get_parent() == bundle_tab_add_ph) {
			bundle_tab_add_ph->remove_child(bundle_tab_add);
			bundle_tabs->add_child(bundle_tab_add);
		}

		if (bundle_tabs->get_tab_count() == 0) {
			bundle_tab_add->set_rect(Rect2(Point2(), add_button_size));
			return;
		}

		Rect2 last_tab = bundle_tabs->get_tab_rect(bundle_tabs->get_tab_count() - 1);
		int hsep = bundle_tabs->get_theme_constant(SNAME("h_separation"));
		if (bundle_tabs->is_layout_rtl()) {
			bundle_tab_add->set_rect(Rect2(Point2(last_tab.position.x - add_button_size.x - hsep, last_tab.position.y), add_button_size));
		} else {
			bundle_tab_add->set_rect(Rect2(Point2(last_tab.position.x + last_tab.size.width + hsep, last_tab.position.y), add_button_size));
		}
	}
}

void ResourceBundleEditor::_add_tab(const String &p_bundle, const String &p_schema) {
	String tab_name = p_bundle.is_empty() ? "empty" : p_bundle.get_basename().get_file();

	bundle_tabs->add_tab(tab_name);

	ResourceBundleTab *new_tab = memnew(ResourceBundleTab(p_bundle, p_schema));
	new_tab->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_container->add_child(new_tab);
}

void ResourceBundleEditor::_notification(int p_what) {
}

bool ResourceBundleEditor::make_bundle(const String &p_path) {
	Ref<ResourceBundle> bundle;
	bundle.instantiate();
	bundle->set_owned_path(p_path);
	Error err = ResourceSaver::save(bundle, p_path.path_join(".bundle"));
	return err == OK;
}

bool ResourceBundleEditor::remove_bundle(const String &p_path) {
	String path = p_path.path_join(".bundle");
	if (FileAccess::exists(path)) {
		Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		dir->remove(path);
		return true;
	}
	return false;
}

ResourceBundleEditor::ResourceBundleEditor() {
	singleton = this;

	set_name(TTR("Bundle"));
	set_icon_name("ResourceBundle");
	set_default_slot(EditorDock::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);

	set_global(false);
	set_transient(true);
	set_closable(true);

	set_focus_mode(FOCUS_ALL);
	set_process_shortcut_input(true);

	file_dialog = memnew(EditorFileDialog);
	add_child(file_dialog);

	tab_container = memnew(VBoxContainer);
	add_child(tab_container);

	tabbar_panel = memnew(PanelContainer);
	tabbar_panel->set_theme_type_variation("PanelContainerTabbarInner");
	tab_container->add_child(tabbar_panel);

	tabbar_container = memnew(HBoxContainer);
	tabbar_panel->add_child(tabbar_container);

	bundle_tabs = memnew(TabBar);
	bundle_tabs->set_theme_type_variation("TabBarInner");
	bundle_tabs->set_tab_close_display_policy(TabBar::CloseButtonDisplayPolicy::CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	bundle_tabs->set_max_tab_width(150 * EDSCALE);
	bundle_tabs->set_drag_to_rearrange_enabled(true);
	bundle_tabs->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	bundle_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tabbar_container->add_child(bundle_tabs);

	//bundle_tabs->connect("tab_changed", callable_mp(this, &EditorSceneTabs::_scene_tab_changed));
	//bundle_tabs->connect("tab_button_pressed", callable_mp(this, &EditorSceneTabs::_scene_tab_script_edited));
	//bundle_tabs->connect("tab_close_pressed", callable_mp(this, &EditorSceneTabs::_scene_tab_closed));
	//bundle_tabs->connect("tab_hovered", callable_mp(this, &EditorSceneTabs::_scene_tab_hovered));
	//bundle_tabs->connect(SceneStringName(mouse_exited), callable_mp(this, &EditorSceneTabs::_scene_tab_exit));
	//bundle_tabs->connect(SceneStringName(gui_input), callable_mp(this, &EditorSceneTabs::_scene_tab_input));
	//bundle_tabs->connect("active_tab_rearranged", callable_mp(this, &EditorSceneTabs::_reposition_active_tab));
	bundle_tabs->connect(SceneStringName(resized), callable_mp(this, &ResourceBundleEditor::_bundle_tabs_resized), CONNECT_DEFERRED);

	bundle_tab_add = memnew(Button);
	bundle_tab_add->set_flat(true);
	bundle_tab_add->set_tooltip_text(TTR("Add a new bundle."));
	bundle_tab_add->set_button_icon(EditorIconManager::get_icon(SNAME("Add")));
	bundle_tabs->add_child(bundle_tab_add);
	//bundle_tab_add->connect(SceneStringName(pressed), callable_mp(EditorNode::get_singleton(), &EditorNode::trigger_menu_option).bind(EditorNode::SCENE_NEW_SCENE, false));

	bundle_tab_add_ph = memnew(Control);
	bundle_tab_add_ph->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	bundle_tab_add_ph->set_custom_minimum_size(bundle_tab_add->get_minimum_size());
	tabbar_container->add_child(bundle_tab_add_ph);

	_add_tab();
}

// ResourceBundleTab

ResourceBundleTab::ResourceBundleTab(const String &p_bundle, const String &p_schema) {
	HBoxContainer *path_hb = memnew(HBoxContainer);
	add_child(path_hb);

	HBoxContainer *bundle_path_hb = memnew(HBoxContainer);
	bundle_path_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	path_hb->add_child(bundle_path_hb);

	Label *bundle_path_label = memnew(Label);
	bundle_path_label->set_text(TTRC("Bundle Path:"));
	bundle_path_hb->add_child(bundle_path_label);

	LineEdit *bundle_path_edit = memnew(LineEdit);
	bundle_path_edit->set_text(p_bundle);
	bundle_path_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bundle_path_hb->add_child(bundle_path_edit);

	Button *bundle_path_browse = memnew(Button);
	bundle_path_browse->set_flat(true);
	bundle_path_browse->set_tooltip_text(TTRC("Browse for a bundle."));
	bundle_path_browse->set_button_icon(EditorIconManager::get_icon(SNAME("FileBrowse")));
	bundle_path_hb->add_child(bundle_path_browse);

	HBoxContainer *schema_path_hb = memnew(HBoxContainer);
	schema_path_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	path_hb->add_child(schema_path_hb);

	Label *schema_path_label = memnew(Label);
	schema_path_label->set_text(TTRC("Schema Path:"));
	schema_path_hb->add_child(schema_path_label);

	LineEdit *schema_path_edit = memnew(LineEdit);
	schema_path_edit->set_text(p_schema);
	schema_path_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	schema_path_hb->add_child(schema_path_edit);

	Button *schema_path_browse = memnew(Button);
	schema_path_browse->set_flat(true);
	schema_path_browse->set_tooltip_text(TTRC("Browse for a schema."));
	schema_path_browse->set_button_icon(EditorIconManager::get_icon(SNAME("FileBrowse")));
	schema_path_hb->add_child(schema_path_browse);

	table = memnew(ResourceBundleTable);
	table->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	table->set_theme_type_variation("ScrollContainerSecondary");
	add_child(table);
}

// ResourceBundleTable

ResourceBundleTable::ResourceBundleTable() {
}

// ResourceBundleEditorPlugin

void ResourceBundleEditorPlugin::edit(Object *p_object) {
}

bool ResourceBundleEditorPlugin::handles(Object *p_object) const {
	return true;
}

void ResourceBundleEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bundle_editor->make_visible();
	}
}

ResourceBundleEditorPlugin::ResourceBundleEditorPlugin() {
	bundle_editor = memnew(ResourceBundleEditor);
	EditorDockManager::get_singleton()->add_dock(bundle_editor);
}
