/*************************************************************************/
/*  style_box_editor_plugin.cpp                                          */
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
#include "style_box_editor_plugin.h"

void StyleBoxEditor::edit(const Ref<StyleBox> &p_stylebox) {

	if (stylebox.is_valid())
		stylebox->disconnect("changed", this, "_sb_changed");
	stylebox = p_stylebox;
	if (p_stylebox.is_valid()) {
		preview->add_style_override("panel", stylebox);
		stylebox->connect("changed", this, "_sb_changed");
	}
}

void StyleBoxEditor::_sb_changed() {

	preview->update();
}

void StyleBoxEditor::_bind_methods() {

	ClassDB::bind_method("_sb_changed", &StyleBoxEditor::_sb_changed);
	//ClassDB::bind_method("_import",&StyleBoxEditor::_import);
	//ClassDB::bind_method("_import_accept",&StyleBoxEditor::_import_accept);
	//ClassDB::bind_method("_preview_text_changed",&StyleBoxEditor::_preview_text_changed);
}

StyleBoxEditor::StyleBoxEditor() {

	panel = memnew(Panel);
	add_child(panel);
	panel->set_area_as_parent_rect();

	Label *l = memnew(Label);
	l->set_text(TTR("StyleBox Preview:"));
	l->set_pos(Point2(5, 5));
	panel->add_child(l);

	preview = memnew(Panel);
	panel->add_child(preview);
	preview->set_pos(Point2(50, 50));
	preview->set_size(Size2(200, 100));
}

void StyleBoxEditorPlugin::edit(Object *p_node) {

	if (p_node && p_node->cast_to<StyleBox>()) {
		stylebox_editor->edit(p_node->cast_to<StyleBox>());
		stylebox_editor->show();
	} else
		stylebox_editor->hide();
}

bool StyleBoxEditorPlugin::handles(Object *p_node) const {

	return p_node->is_class("StyleBox");
}

void StyleBoxEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		button->show();
		EditorNode::get_singleton()->make_bottom_panel_item_visible(stylebox_editor);

	} else {
		if (stylebox_editor->is_visible_in_tree())
			EditorNode::get_singleton()->hide_bottom_panel();
		button->hide();
	}
}

StyleBoxEditorPlugin::StyleBoxEditorPlugin(EditorNode *p_node) {

	stylebox_editor = memnew(StyleBoxEditor);
	stylebox_editor->set_custom_minimum_size(Size2(0, 250));

	//p_node->get_viewport()->add_child(stylebox_editor);
	button = p_node->add_bottom_panel_item("StyleBox", stylebox_editor);
	button->hide();
}
