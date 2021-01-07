/*************************************************************************/
/*  ot_features_plugin.cpp                                               */
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

#include "ot_features_plugin.h"

#include "editor/editor_scale.h"

void OpenTypeFeaturesEditor::_value_changed(double val) {
	if (setting) {
		return;
	}

	emit_changed(get_edited_property(), spin->get_value());
}

void OpenTypeFeaturesEditor::update_property() {
	double val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin->set_value(val);
	setting = false;
}

void OpenTypeFeaturesEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_theme_color("accent_color", "Editor");

		button->set_icon(get_theme_icon("Remove", "EditorIcons"));
		button->set_size(get_theme_icon("Remove", "EditorIcons")->get_size());
		spin->set_custom_label_color(true, base);
	}
}

void OpenTypeFeaturesEditor::_remove_feature() {
	get_edited_object()->set(get_edited_property(), -1);
}

void OpenTypeFeaturesEditor::_bind_methods() {
}

OpenTypeFeaturesEditor::OpenTypeFeaturesEditor() {
	HBoxContainer *bc = memnew(HBoxContainer);
	add_child(bc);

	spin = memnew(EditorSpinSlider);
	spin->set_flat(true);
	bc->add_child(spin);
	add_focusable(spin);
	spin->connect("value_changed", callable_mp(this, &OpenTypeFeaturesEditor::_value_changed));
	spin->set_h_size_flags(SIZE_EXPAND_FILL);

	spin->set_min(0);
	spin->set_max(65536);
	spin->set_step(1);
	spin->set_hide_slider(false);
	spin->set_allow_greater(false);
	spin->set_allow_lesser(false);

	button = memnew(Button);
	button->set_tooltip(RTR("Remove feature"));
	button->set_flat(true);
	bc->add_child(button);

	button->connect("pressed", callable_mp(this, &OpenTypeFeaturesEditor::_remove_feature));

	setting = false;
}

/*************************************************************************/

void OpenTypeFeaturesAdd::_add_feature(int p_option) {
	get_edited_object()->set("opentype_features/" + TS->tag_to_name(p_option), 1);
}

void OpenTypeFeaturesAdd::update_property() {
	menu->clear();
	menu_ss->clear();
	menu_cv->clear();
	menu_cu->clear();
	bool have_ss = false;
	bool have_cv = false;
	bool have_cu = false;
	Dictionary features = Object::cast_to<Control>(get_edited_object())->get_theme_font("font")->get_feature_list();
	for (const Variant *ftr = features.next(nullptr); ftr != nullptr; ftr = features.next(ftr)) {
		String ftr_name = TS->tag_to_name(*ftr);
		if (ftr_name.begins_with("stylistic_set_")) {
			menu_ss->add_item(ftr_name.capitalize(), (int32_t)*ftr);
			have_ss = true;
		} else if (ftr_name.begins_with("character_variant_")) {
			menu_cv->add_item(ftr_name.capitalize(), (int32_t)*ftr);
			have_cv = true;
		} else if (ftr_name.begins_with("custom_")) {
			menu_cu->add_item(ftr_name.replace("custom_", ""), (int32_t)*ftr);
			have_cu = true;
		} else {
			menu->add_item(ftr_name.capitalize(), (int32_t)*ftr);
		}
	}
	if (have_ss) {
		menu->add_submenu_item(RTR("Stylistic Sets"), "SSMenu");
	}
	if (have_cv) {
		menu->add_submenu_item(RTR("Character Variants"), "CVMenu");
	}
	if (have_cu) {
		menu->add_submenu_item(RTR("Custom"), "CUMenu");
	}
}

void OpenTypeFeaturesAdd::_features_menu() {
	Size2 size = get_size();
	menu->set_position(get_screen_position() + Size2(0, size.height * get_global_transform().get_scale().y));
	menu->popup();
}

void OpenTypeFeaturesAdd::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED || p_what == NOTIFICATION_ENTER_TREE) {
		set_label("");
		button->set_icon(get_theme_icon("Add", "EditorIcons"));
		button->set_size(get_theme_icon("Add", "EditorIcons")->get_size());
	}
}

void OpenTypeFeaturesAdd::_bind_methods() {
}

OpenTypeFeaturesAdd::OpenTypeFeaturesAdd() {
	menu = memnew(PopupMenu);
	add_child(menu);

	menu_cv = memnew(PopupMenu);
	menu_cv->set_name("CVMenu");
	menu->add_child(menu_cv);

	menu_ss = memnew(PopupMenu);
	menu_ss->set_name("SSMenu");
	menu->add_child(menu_ss);

	menu_cu = memnew(PopupMenu);
	menu_cu->set_name("CUMenu");
	menu->add_child(menu_cu);

	button = memnew(Button);
	button->set_flat(true);
	button->set_text(RTR("Add feature..."));
	button->set_tooltip(RTR("Add feature..."));
	add_child(button);

	button->connect("pressed", callable_mp(this, &OpenTypeFeaturesAdd::_features_menu));
	menu->connect("id_pressed", callable_mp(this, &OpenTypeFeaturesAdd::_add_feature));
	menu_cv->connect("id_pressed", callable_mp(this, &OpenTypeFeaturesAdd::_add_feature));
	menu_ss->connect("id_pressed", callable_mp(this, &OpenTypeFeaturesAdd::_add_feature));
	menu_cu->connect("id_pressed", callable_mp(this, &OpenTypeFeaturesAdd::_add_feature));
}

/*************************************************************************/

bool EditorInspectorPluginOpenTypeFeatures::can_handle(Object *p_object) {
	return (Object::cast_to<Control>(p_object) != nullptr);
}

void EditorInspectorPluginOpenTypeFeatures::parse_begin(Object *p_object) {
}

void EditorInspectorPluginOpenTypeFeatures::parse_category(Object *p_object, const String &p_parse_category) {
}

bool EditorInspectorPluginOpenTypeFeatures::parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage, bool p_wide) {
	if (p_path == "opentype_features/_new") {
		OpenTypeFeaturesAdd *editor = memnew(OpenTypeFeaturesAdd);
		add_property_editor(p_path, editor);
		return true;
	} else if (p_path.begins_with("opentype_features")) {
		OpenTypeFeaturesEditor *editor = memnew(OpenTypeFeaturesEditor);
		add_property_editor(p_path, editor);
		return true;
	}
	return false;
}

/*************************************************************************/

OpenTypeFeaturesEditorPlugin::OpenTypeFeaturesEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginOpenTypeFeatures> ftr_plugin;
	ftr_plugin.instance();
	EditorInspector::add_inspector_plugin(ftr_plugin);
}
