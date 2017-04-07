/*************************************************************************/
/*  rich_text_editor_plugin.cpp                                          */
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
#include "rich_text_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "os/file_access.h"

void RichTextEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}
void RichTextEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		node = NULL;
		hide();
	}
}

void RichTextEditor::_file_selected(const String &p_path) {

	CharString cs;
	FileAccess *fa = FileAccess::open(p_path, FileAccess::READ);
	if (!fa) {
		ERR_FAIL();
	}

	while (!fa->eof_reached())
		cs.push_back(fa->get_8());
	cs.push_back(0);
	memdelete(fa);

	String bbcode;
	bbcode.parse_utf8(&cs[0]);
	node->parse_bbcode(bbcode);
}

void RichTextEditor::_menu_option(int p_option) {

	switch (p_option) {

		case PARSE_BBCODE: {

			file_dialog->popup_centered_ratio();
		} break;
		case CLEAR: {

			node->clear();

		} break;
	}
}

void RichTextEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_menu_option"), &RichTextEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_file_selected"), &RichTextEditor::_file_selected);
}

void RichTextEditor::edit(Node *p_rich_text) {

	node = p_rich_text->cast_to<RichTextLabel>();
}
RichTextEditor::RichTextEditor() {

	options = memnew(MenuButton);
	//add_child(options);
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(options);
	options->set_area_as_parent_rect();

	options->set_text("RichText");
	options->get_popup()->add_item(TTR("Parse BBCode"), PARSE_BBCODE);
	options->get_popup()->add_item(TTR("Clear"), CLEAR);

	options->get_popup()->connect("id_pressed", this, "_menu_option");
	file_dialog = memnew(EditorFileDialog);
	add_child(file_dialog);
	file_dialog->add_filter("*.txt");
	file_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	file_dialog->connect("file_selected", this, "_file_selected");
}

void RichTextEditorPlugin::edit(Object *p_object) {

	rich_text_editor->edit(p_object->cast_to<Node>());
}

bool RichTextEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("RichTextLabel");
}

void RichTextEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		rich_text_editor->options->show();
	} else {

		rich_text_editor->options->hide();
		rich_text_editor->edit(NULL);
	}
}

RichTextEditorPlugin::RichTextEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	rich_text_editor = memnew(RichTextEditor);
	editor->get_viewport()->add_child(rich_text_editor);

	rich_text_editor->set_margin(MARGIN_LEFT, 184);
	rich_text_editor->set_margin(MARGIN_RIGHT, 230);
	rich_text_editor->set_margin(MARGIN_TOP, 0);
	rich_text_editor->set_margin(MARGIN_BOTTOM, 10);

	rich_text_editor->options->hide();
}

RichTextEditorPlugin::~RichTextEditorPlugin() {
}
