/*************************************************************************/
/*  item_list_editor_plugin.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "item_list_editor_plugin.h"

#include "io/resource_loader.h"

bool ItemListPlugin::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name;
	int idx = name.get_slice("/", 0).to_int();
	String what = name.get_slice("/", 1);

	if (what == "text")
		set_item_text(idx, p_value);
	else if (what == "icon")
		set_item_icon(idx, p_value);
	else if (what == "checkable")
		set_item_checkable(idx, p_value);
	else if (what == "checked")
		set_item_checked(idx, p_value);
	else if (what == "id")
		set_item_id(idx, p_value);
	else if (what == "enabled")
		set_item_enabled(idx, p_value);
	else if (what == "separator")
		set_item_separator(idx, p_value);
	else
		return false;

	return true;
}

bool ItemListPlugin::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;
	int idx = name.get_slice("/", 0).to_int();
	String what = name.get_slice("/", 1);

	if (what == "text")
		r_ret = get_item_text(idx);
	else if (what == "icon")
		r_ret = get_item_icon(idx);
	else if (what == "checkable")
		r_ret = is_item_checkable(idx);
	else if (what == "checked")
		r_ret = is_item_checked(idx);
	else if (what == "id")
		r_ret = get_item_id(idx);
	else if (what == "enabled")
		r_ret = is_item_enabled(idx);
	else if (what == "separator")
		r_ret = is_item_separator(idx);
	else
		return false;

	return true;
}
void ItemListPlugin::_get_property_list(List<PropertyInfo> *p_list) const {

	for (int i = 0; i < get_item_count(); i++) {

		String base = itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, base + "text"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, base + "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture"));

		int flags = get_flags();

		if (flags & FLAG_CHECKABLE) {
			p_list->push_back(PropertyInfo(Variant::BOOL, base + "checkable"));
			p_list->push_back(PropertyInfo(Variant::BOOL, base + "checked"));
		}

		if (flags & FLAG_ID)
			p_list->push_back(PropertyInfo(Variant::INT, base + "id", PROPERTY_HINT_RANGE, "-1,4096"));

		if (flags & FLAG_ENABLE)
			p_list->push_back(PropertyInfo(Variant::BOOL, base + "enabled"));

		if (flags & FLAG_SEPARATOR)
			p_list->push_back(PropertyInfo(Variant::BOOL, base + "separator"));
	}
}

///////////////////////////////////////////////////////////////
///////////////////////// PLUGINS /////////////////////////////
///////////////////////////////////////////////////////////////

void ItemListOptionButtonPlugin::set_object(Object *p_object) {

	ob = p_object->cast_to<OptionButton>();
}

bool ItemListOptionButtonPlugin::handles(Object *p_object) const {

	return p_object->is_class("OptionButton");
}

int ItemListOptionButtonPlugin::get_flags() const {

	return FLAG_ICON | FLAG_ID | FLAG_ENABLE;
}

void ItemListOptionButtonPlugin::add_item() {

	ob->add_item(vformat(TTR("Item %d"), ob->get_item_count()));
	_change_notify();
}

int ItemListOptionButtonPlugin::get_item_count() const {

	return ob->get_item_count();
}

void ItemListOptionButtonPlugin::erase(int p_idx) {

	ob->remove_item(p_idx);
	_change_notify();
}

ItemListOptionButtonPlugin::ItemListOptionButtonPlugin() {

	ob = NULL;
}

///////////////////////////////////////////////////////////////

void ItemListPopupMenuPlugin::set_object(Object *p_object) {

	if (p_object->is_class("MenuButton"))
		pp = p_object->cast_to<MenuButton>()->get_popup();
	else
		pp = p_object->cast_to<PopupMenu>();
}

bool ItemListPopupMenuPlugin::handles(Object *p_object) const {

	return p_object->is_class("PopupMenu") || p_object->is_class("MenuButton");
}

int ItemListPopupMenuPlugin::get_flags() const {

	return FLAG_ICON | FLAG_CHECKABLE | FLAG_ID | FLAG_ENABLE | FLAG_SEPARATOR;
}

void ItemListPopupMenuPlugin::add_item() {

	pp->add_item(vformat(TTR("Item %d"), pp->get_item_count()));
	_change_notify();
}

int ItemListPopupMenuPlugin::get_item_count() const {

	return pp->get_item_count();
}

void ItemListPopupMenuPlugin::erase(int p_idx) {

	pp->remove_item(p_idx);
	_change_notify();
}

ItemListPopupMenuPlugin::ItemListPopupMenuPlugin() {

	pp = NULL;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void ItemListEditor::_node_removed(Node *p_node) {

	if (p_node == item_list) {
		item_list = NULL;
		hide();
		dialog->hide();
	}
}

void ItemListEditor::_notification(int p_notification) {

	if (p_notification == NOTIFICATION_ENTER_TREE) {

		add_button->set_icon(get_icon("Add", "EditorIcons"));
		del_button->set_icon(get_icon("Remove", "EditorIcons"));
	}
}

void ItemListEditor::_add_pressed() {

	if (selected_idx == -1)
		return;

	item_plugins[selected_idx]->add_item();
}

void ItemListEditor::_delete_pressed() {

	TreeItem *ti = tree->get_selected();

	if (!ti)
		return;

	if (ti->get_parent() != tree->get_root())
		return;

	int idx = ti->get_text(0).to_int();

	if (selected_idx == -1)
		return;

	item_plugins[selected_idx]->erase(idx);
}

void ItemListEditor::_edit_items() {

	dialog->popup_centered(Vector2(300, 400));
}

void ItemListEditor::edit(Node *p_item_list) {

	item_list = p_item_list;

	if (!item_list) {
		selected_idx = -1;
		property_editor->edit(NULL);
		return;
	}

	for (int i = 0; i < item_plugins.size(); i++) {
		if (item_plugins[i]->handles(p_item_list)) {

			item_plugins[i]->set_object(p_item_list);
			property_editor->edit(item_plugins[i]);

			if (has_icon(item_list->get_class(), "EditorIcons"))
				toolbar_button->set_icon(get_icon(item_list->get_class(), "EditorIcons"));
			else
				toolbar_button->set_icon(Ref<Texture>());

			selected_idx = i;
			return;
		}
	}

	selected_idx = -1;
	property_editor->edit(NULL);
}

bool ItemListEditor::handles(Object *p_object) const {

	for (int i = 0; i < item_plugins.size(); i++) {
		if (item_plugins[i]->handles(p_object)) {
			return true;
		}
	}

	return false;
}

void ItemListEditor::_bind_methods() {

	ClassDB::bind_method("_edit_items", &ItemListEditor::_edit_items);
	ClassDB::bind_method("_add_button", &ItemListEditor::_add_pressed);
	ClassDB::bind_method("_delete_button", &ItemListEditor::_delete_pressed);
}

ItemListEditor::ItemListEditor() {

	selected_idx = -1;

	add_child(memnew(VSeparator));

	toolbar_button = memnew(ToolButton);
	toolbar_button->set_text(TTR("Items"));
	add_child(toolbar_button);
	toolbar_button->connect("pressed", this, "_edit_items");

	dialog = memnew(AcceptDialog);
	dialog->set_title(TTR("Item List Editor"));
	add_child(dialog);

	VBoxContainer *vbc = memnew(VBoxContainer);
	dialog->add_child(vbc);
	//dialog->set_child_rect(vbc);

	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(hbc);

	add_button = memnew(Button);
	add_button->set_text(TTR("Add"));
	hbc->add_child(add_button);
	add_button->connect("pressed", this, "_add_button");

	hbc->add_spacer();

	del_button = memnew(Button);
	del_button->set_text(TTR("Delete"));
	hbc->add_child(del_button);
	del_button->connect("pressed", this, "_delete_button");

	property_editor = memnew(PropertyEditor);
	property_editor->hide_top_label();
	property_editor->set_subsection_selectable(true);
	vbc->add_child(property_editor);
	property_editor->set_v_size_flags(SIZE_EXPAND_FILL);

	tree = property_editor->get_scene_tree();
}

ItemListEditor::~ItemListEditor() {

	for (int i = 0; i < item_plugins.size(); i++)
		memdelete(item_plugins[i]);
}

void ItemListEditorPlugin::edit(Object *p_object) {

	item_list_editor->edit(p_object->cast_to<Node>());
}

bool ItemListEditorPlugin::handles(Object *p_object) const {

	return item_list_editor->handles(p_object);
}

void ItemListEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		item_list_editor->show();
	} else {

		item_list_editor->hide();
		item_list_editor->edit(NULL);
	}
}

ItemListEditorPlugin::ItemListEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	item_list_editor = memnew(ItemListEditor);
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(item_list_editor);

	item_list_editor->hide();
	item_list_editor->add_plugin(memnew(ItemListOptionButtonPlugin));
	item_list_editor->add_plugin(memnew(ItemListPopupMenuPlugin));
}

ItemListEditorPlugin::~ItemListEditorPlugin() {
}
