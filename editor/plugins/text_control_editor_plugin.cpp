/*************************************************************************/
/*  text_control_editor_plugin.cpp                                       */
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

#include "text_control_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/multi_node_edit.h"

void TextControlEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &TextControlEditor::_reload_fonts))) {
				EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &TextControlEditor::_reload_fonts), make_binds(""));
			}
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			clear_formatting->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &TextControlEditor::_reload_fonts))) {
				EditorFileSystem::get_singleton()->disconnect("filesystem_changed", callable_mp(this, &TextControlEditor::_reload_fonts));
			}
		} break;
	}
}

void TextControlEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_control"), &TextControlEditor::_update_control);
}

void TextControlEditor::_find_resources(EditorFileSystemDirectory *p_dir) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_find_resources(p_dir->get_subdir(i));
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "FontData") {
			Ref<FontData> fd = ResourceLoader::load(p_dir->get_file_path(i));
			if (fd.is_valid()) {
				String name = fd->get_font_name();
				String sty = fd->get_font_style_name();
				if (sty.is_empty()) {
					sty = "Default";
				}
				fonts[name][sty] = p_dir->get_file_path(i);
			}
		}
	}
}

void TextControlEditor::_reload_fonts(const String &p_path) {
	fonts.clear();
	_find_resources(EditorFileSystem::get_singleton()->get_filesystem());
	_update_control();
}

void TextControlEditor::_update_fonts_menu() {
	font_list->clear();
	font_list->add_item(TTR("[Theme Default]"), FONT_INFO_THEME_DEFAULT);
	if (custom_font.is_valid()) {
		font_list->add_item(TTR("[Custom Font]"), FONT_INFO_USER_CUSTOM);
	}

	int id = FONT_INFO_ID;
	for (const KeyValue<String, HashMap<String, String>> &E : fonts) {
		font_list->add_item(E.key, id++);
	}

	if (font_list->get_item_count() > 1) {
		font_list->show();
	} else {
		font_list->hide();
	}
}

void TextControlEditor::_update_styles_menu() {
	font_style_list->clear();
	if ((font_list->get_selected_id() >= FONT_INFO_ID)) {
		const String &name = font_list->get_item_text(font_list->get_selected());
		for (KeyValue<String, String> &E : fonts[name]) {
			font_style_list->add_item(E.key);
		}
	} else if (font_list->get_selected() >= 0) {
		font_style_list->add_item("Default");
	}

	if (font_style_list->get_item_count() > 1) {
		font_style_list->show();
	} else {
		font_style_list->hide();
	}
}

void TextControlEditor::_update_control() {
	if (!edited_controls.is_empty()) {
		String font_selected;
		bool same_font = true;
		String style_selected;
		bool same_style = true;
		int font_size = 0;
		bool same_font_size = true;
		int outline_size = 0;
		bool same_outline_size = true;
		Color font_color = Color{ 1.0f, 1.0f, 1.0f };
		bool same_font_color = true;
		Color outline_color = Color{ 1.0f, 1.0f, 1.0f };
		bool same_outline_color = true;

		int count = edited_controls.size();
		for (int i = 0; i < count; ++i) {
			Control *edited_control = edited_controls[i];

			StringName edited_color;
			StringName edited_font;
			StringName edited_font_size;

			// Get override names.
			if (Object::cast_to<RichTextLabel>(edited_control)) {
				edited_color = SNAME("default_color");
				edited_font = SNAME("normal_font");
				edited_font_size = SNAME("normal_font_size");
			} else {
				edited_color = SNAME("font_color");
				edited_font = SNAME("font");
				edited_font_size = SNAME("font_size");
			}

			// Get font override.
			Ref<Font> font;
			if (edited_control->has_theme_font_override(edited_font)) {
				font = edited_control->get_theme_font(edited_font);
			}

			if (font.is_valid()) {
				if (font->get_data_count() != 1) {
					if (i > 0) {
						same_font = same_font && (custom_font == font);
					}
					custom_font = font;

					font_selected = TTR("[Custom Font]");
					same_style = false;
				} else {
					String name = font->get_data(0)->get_font_name();
					String style = font->get_data(0)->get_font_style_name();
					if (fonts.has(name) && fonts[name].has(style)) {
						if (i > 0) {
							same_font = same_font && (name == font_selected);
							same_style = same_style && (style == style_selected);
						}
						font_selected = name;
						style_selected = style;
					} else {
						if (i > 0) {
							same_font = same_font && (custom_font == font);
						}
						custom_font = font;

						font_selected = TTR("[Custom Font]");
						same_style = false;
					}
				}
			} else {
				if (i > 0) {
					same_font = same_font && (font_selected == TTR("[Theme Default]"));
				}

				font_selected = TTR("[Theme Default]");
				same_style = false;
			}

			int current_font_size = edited_control->get_theme_font_size(edited_font_size);
			int current_outline_size = edited_control->get_theme_constant(SNAME("outline_size"));
			Color current_font_color = edited_control->get_theme_color(edited_color);
			Color current_outline_color = edited_control->get_theme_color(SNAME("font_outline_color"));
			if (i > 0) {
				same_font_size = same_font_size && (font_size == current_font_size);
				same_outline_size = same_outline_size && (outline_size == current_outline_size);
				same_font_color = same_font_color && (font_color == current_font_color);
				same_outline_color = same_outline_color && (outline_color == current_outline_color);
			}

			font_size = current_font_size;
			outline_size = current_outline_size;
			font_color = current_font_color;
			outline_color = current_outline_color;
		}
		_update_fonts_menu();
		if (same_font) {
			for (int j = 0; j < font_list->get_item_count(); j++) {
				if (font_list->get_item_text(j) == font_selected) {
					font_list->select(j);
					break;
				}
			}
		} else {
			custom_font = Ref<Font>();
			font_list->select(-1);
		}

		_update_styles_menu();
		if (same_style) {
			for (int j = 0; j < font_style_list->get_item_count(); j++) {
				if (font_style_list->get_item_text(j) == style_selected) {
					font_style_list->select(j);
					break;
				}
			}
		} else {
			font_style_list->select(-1);
		}

		// Get other theme overrides.
		font_size_list->set_block_signals(true);
		if (same_font_size) {
			font_size_list->get_line_edit()->set_text(String::num_uint64(font_size));
			font_size_list->set_value(font_size);
		} else {
			font_size_list->get_line_edit()->set_text("");
		}
		font_size_list->set_block_signals(false);

		outline_size_list->set_block_signals(true);
		if (same_outline_size) {
			outline_size_list->get_line_edit()->set_text(String::num_uint64(outline_size));
			outline_size_list->set_value(outline_size);
		} else {
			outline_size_list->get_line_edit()->set_text("");
		}
		outline_size_list->set_block_signals(false);

		if (!same_font_color) {
			font_color = Color{ 1.0f, 1.0f, 1.0f };
		}
		font_color_picker->set_pick_color(font_color);

		if (!same_outline_color) {
			outline_color = Color{ 1.0f, 1.0f, 1.0f };
		}
		outline_color_picker->set_pick_color(outline_color);
	}
}

void TextControlEditor::_font_selected(int p_id) {
	_update_styles_menu();
	_set_font();
}

void TextControlEditor::_font_style_selected(int p_id) {
	_set_font();
}

void TextControlEditor::_set_font() {
	if (edited_controls.is_empty()) {
		return;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Font"));

	int count = edited_controls.size();
	for (int i = 0; i < count; ++i) {
		Control *edited_control = edited_controls[i];

		StringName edited_font;
		if (Object::cast_to<RichTextLabel>(edited_control)) {
			edited_font = SNAME("normal_font");
		} else {
			edited_font = SNAME("font");
		}

		if (font_list->get_selected_id() == FONT_INFO_THEME_DEFAULT) {
			// Remove font override.
			ur->add_do_method(edited_control, "remove_theme_font_override", edited_font);
		} else if (font_list->get_selected_id() == FONT_INFO_USER_CUSTOM) {
			// Restore "custom_font".
			ur->add_do_method(edited_control, "add_theme_font_override", edited_font, custom_font);
		} else if (font_list->get_selected() >= 0) {
			// Load new font resource using selected name and style.
			String name = font_list->get_item_text(font_list->get_selected());
			String style = font_style_list->get_item_text(font_style_list->get_selected());
			if (style.is_empty()) {
				style = "Default";
			}
			if (fonts.has(name)) {
				Ref<FontData> fd = ResourceLoader::load(fonts[name][style]);
				if (fd.is_valid()) {
					Ref<Font> font;
					font.instantiate();
					font->add_data(fd);
					ur->add_do_method(edited_control, "add_theme_font_override", edited_font, font);
				}
			}
		}

		if (edited_control->has_theme_font_override(edited_font)) {
			ur->add_undo_method(edited_control, "add_theme_font_override", edited_font, edited_control->get_theme_font(edited_font));
		} else {
			ur->add_undo_method(edited_control, "remove_theme_font_override", edited_font);
		}
	}

	ur->add_do_method(this, "_update_control");
	ur->add_undo_method(this, "_update_control");

	ur->commit_action();
}

void TextControlEditor::_font_size_selected(double p_size) {
	if (edited_controls.is_empty()) {
		return;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Font Size"));

	int count = edited_controls.size();
	for (int i = 0; i < count; ++i) {
		Control *edited_control = edited_controls[i];

		StringName edited_font_size;
		if (Object::cast_to<RichTextLabel>(edited_control)) {
			edited_font_size = SNAME("normal_font_size");
		} else {
			edited_font_size = SNAME("font_size");
		}

		ur->add_do_method(edited_control, "add_theme_font_size_override", edited_font_size, p_size);
		if (edited_control->has_theme_font_size_override(edited_font_size)) {
			ur->add_undo_method(edited_control, "add_theme_font_size_override", edited_font_size, edited_control->get_theme_font_size(edited_font_size));
		} else {
			ur->add_undo_method(edited_control, "remove_theme_font_size_override", edited_font_size);
		}
	}

	ur->add_do_method(this, "_update_control");
	ur->add_undo_method(this, "_update_control");

	ur->commit_action();
}

void TextControlEditor::_outline_size_selected(double p_size) {
	if (edited_controls.is_empty()) {
		return;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Font Outline Size"));

	int count = edited_controls.size();
	for (int i = 0; i < count; ++i) {
		Control *edited_control = edited_controls[i];

		ur->add_do_method(edited_control, "add_theme_constant_override", "outline_size", p_size);
		if (edited_control->has_theme_constant_override("outline_size")) {
			ur->add_undo_method(edited_control, "add_theme_constant_override", "outline_size", edited_control->get_theme_constant(SNAME("outline_size")));
		} else {
			ur->add_undo_method(edited_control, "remove_theme_constant_override", "outline_size");
		}
	}

	ur->add_do_method(this, "_update_control");
	ur->add_undo_method(this, "_update_control");

	ur->commit_action();
}

void TextControlEditor::_font_color_changed(const Color &p_color) {
	if (edited_controls.is_empty()) {
		return;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Font Color"), UndoRedo::MERGE_ENDS);

	int count = edited_controls.size();
	for (int i = 0; i < count; ++i) {
		Control *edited_control = edited_controls[i];

		StringName edited_color;
		if (Object::cast_to<RichTextLabel>(edited_control)) {
			edited_color = SNAME("default_color");
		} else {
			edited_color = SNAME("font_color");
		}

		ur->add_do_method(edited_control, "add_theme_color_override", edited_color, p_color);
		if (edited_control->has_theme_color_override(edited_color)) {
			ur->add_undo_method(edited_control, "add_theme_color_override", edited_color, edited_control->get_theme_color(edited_color));
		} else {
			ur->add_undo_method(edited_control, "remove_theme_color_override", edited_color);
		}
	}

	ur->add_do_method(this, "_update_control");
	ur->add_undo_method(this, "_update_control");

	ur->commit_action();
}

void TextControlEditor::_outline_color_changed(const Color &p_color) {
	if (edited_controls.is_empty()) {
		return;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Font Outline Color"), UndoRedo::MERGE_ENDS);

	int count = edited_controls.size();
	for (int i = 0; i < count; ++i) {
		Control *edited_control = edited_controls[i];

		ur->add_do_method(edited_control, "add_theme_color_override", "font_outline_color", p_color);
		if (edited_control->has_theme_color_override("font_outline_color")) {
			ur->add_undo_method(edited_control, "add_theme_color_override", "font_outline_color", edited_control->get_theme_color(SNAME("font_outline_color")));
		} else {
			ur->add_undo_method(edited_control, "remove_theme_color_override", "font_outline_color");
		}
	}

	ur->add_do_method(this, "_update_control");
	ur->add_undo_method(this, "_update_control");

	ur->commit_action();
}

void TextControlEditor::_clear_formatting() {
	if (edited_controls.is_empty()) {
		return;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Clear Control Formatting"));

	int count = edited_controls.size();
	for (int i = 0; i < count; ++i) {
		Control *edited_control = edited_controls[i];

		StringName edited_color;
		StringName edited_font;
		StringName edited_font_size;

		// Get override names.
		if (Object::cast_to<RichTextLabel>(edited_control)) {
			edited_color = SNAME("default_color");
			edited_font = SNAME("normal_font");
			edited_font_size = SNAME("normal_font_size");
		} else {
			edited_color = SNAME("font_color");
			edited_font = SNAME("font");
			edited_font_size = SNAME("font_size");
		}

		ur->add_do_method(edited_control, "begin_bulk_theme_override");
		ur->add_undo_method(edited_control, "begin_bulk_theme_override");

		ur->add_do_method(edited_control, "remove_theme_font_override", edited_font);
		if (edited_control->has_theme_font_override(edited_font)) {
			ur->add_undo_method(edited_control, "add_theme_font_override", edited_font, edited_control->get_theme_font(edited_font));
		}

		ur->add_do_method(edited_control, "remove_theme_font_size_override", edited_font_size);
		if (edited_control->has_theme_font_size_override(edited_font_size)) {
			ur->add_undo_method(edited_control, "add_theme_font_size_override", edited_font_size, edited_control->get_theme_font_size(edited_font_size));
		}

		ur->add_do_method(edited_control, "remove_theme_color_override", edited_color);
		if (edited_control->has_theme_color_override(edited_color)) {
			ur->add_undo_method(edited_control, "add_theme_color_override", edited_color, edited_control->get_theme_color(edited_color));
		}

		ur->add_do_method(edited_control, "remove_theme_color_override", "font_outline_color");
		if (edited_control->has_theme_color_override("font_outline_color")) {
			ur->add_undo_method(edited_control, "add_theme_color_override", "font_outline_color", edited_control->get_theme_color(SNAME("font_outline_color")));
		}

		ur->add_do_method(edited_control, "remove_theme_constant_override", "outline_size");
		if (edited_control->has_theme_constant_override("outline_size")) {
			ur->add_undo_method(edited_control, "add_theme_constant_override", "outline_size", edited_control->get_theme_constant(SNAME("outline_size")));
		}

		ur->add_do_method(edited_control, "end_bulk_theme_override");
		ur->add_undo_method(edited_control, "end_bulk_theme_override");
	}

	ur->add_do_method(this, "_update_control");
	ur->add_undo_method(this, "_update_control");

	ur->commit_action();
}

void TextControlEditor::edit(Object *p_object) {
	Control *ctrl = Object::cast_to<Control>(p_object);
	MultiNodeEdit *multi_node = Object::cast_to<MultiNodeEdit>(p_object);

	edited_controls.clear();
	custom_font = Ref<Font>();
	if (ctrl) {
		edited_controls.append(ctrl);
		_update_control();
	} else if (multi_node && handles(multi_node)) {
		int count = multi_node->get_node_count();
		Node *scene = EditorNode::get_singleton()->get_edited_scene();

		for (int i = 0; i < count; ++i) {
			Control *child = Object::cast_to<Control>(scene->get_node(multi_node->get_node(i)));
			edited_controls.append(child);
		}
		_update_control();
	}
}

bool TextControlEditor::handles(Object *p_object) const {
	Control *ctrl = Object::cast_to<Control>(p_object);
	MultiNodeEdit *multi_node = Object::cast_to<MultiNodeEdit>(p_object);

	if (!ctrl && !multi_node) {
		return false;
	} else if (ctrl) {
		bool valid = false;
		ctrl->get("text", &valid);
		return valid;
	} else {
		bool valid = true;
		int count = multi_node->get_node_count();
		Node *scene = EditorNode::get_singleton()->get_edited_scene();

		for (int i = 0; i < count; ++i) {
			bool temp_valid = false;
			Control *child = Object::cast_to<Control>(scene->get_node(multi_node->get_node(i)));
			if (child) {
				child->get("text", &temp_valid);
			}
			valid = valid && temp_valid;

			if (!valid) {
				break;
			}
		}

		return valid;
	}
}

TextControlEditor::TextControlEditor() {
	add_child(memnew(VSeparator));

	font_list = memnew(OptionButton);
	font_list->set_flat(true);
	font_list->set_tooltip(TTR("Font"));
	add_child(font_list);
	font_list->connect("item_selected", callable_mp(this, &TextControlEditor::_font_selected));

	font_style_list = memnew(OptionButton);
	font_style_list->set_flat(true);
	font_style_list->set_tooltip(TTR("Font style"));
	font_style_list->set_toggle_mode(true);
	add_child(font_style_list);
	font_style_list->connect("item_selected", callable_mp(this, &TextControlEditor::_font_style_selected));

	font_size_list = memnew(SpinBox);
	font_size_list->set_tooltip(TTR("Font Size"));
	font_size_list->get_line_edit()->add_theme_constant_override("minimum_character_width", 2);
	font_size_list->set_min(6);
	font_size_list->set_step(1);
	font_size_list->set_max(96);
	font_size_list->get_line_edit()->set_flat(true);
	add_child(font_size_list);
	font_size_list->connect("value_changed", callable_mp(this, &TextControlEditor::_font_size_selected));

	font_color_picker = memnew(ColorPickerButton);
	font_color_picker->set_custom_minimum_size(Size2(20, 0) * EDSCALE);
	font_color_picker->set_flat(true);
	font_color_picker->set_tooltip(TTR("Text Color"));
	add_child(font_color_picker);
	font_color_picker->connect("color_changed", callable_mp(this, &TextControlEditor::_font_color_changed));

	add_child(memnew(VSeparator));

	outline_size_list = memnew(SpinBox);
	outline_size_list->set_tooltip(TTR("Outline Size"));
	outline_size_list->get_line_edit()->add_theme_constant_override("minimum_character_width", 2);
	outline_size_list->set_min(0);
	outline_size_list->set_step(1);
	outline_size_list->set_max(96);
	outline_size_list->get_line_edit()->set_flat(true);
	add_child(outline_size_list);
	outline_size_list->connect("value_changed", callable_mp(this, &TextControlEditor::_outline_size_selected));

	outline_color_picker = memnew(ColorPickerButton);
	outline_color_picker->set_custom_minimum_size(Size2(20, 0) * EDSCALE);
	outline_color_picker->set_flat(true);
	outline_color_picker->set_tooltip(TTR("Outline Color"));
	add_child(outline_color_picker);
	outline_color_picker->connect("color_changed", callable_mp(this, &TextControlEditor::_outline_color_changed));

	add_child(memnew(VSeparator));

	clear_formatting = memnew(Button);
	clear_formatting->set_flat(true);
	clear_formatting->set_tooltip(TTR("Clear Formatting"));
	add_child(clear_formatting);
	clear_formatting->connect("pressed", callable_mp(this, &TextControlEditor::_clear_formatting));
}

/*************************************************************************/

void TextControlEditorPlugin::edit(Object *p_object) {
	text_ctl_editor->edit(p_object);
}

bool TextControlEditorPlugin::handles(Object *p_object) const {
	return text_ctl_editor->handles(p_object);
}

void TextControlEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		text_ctl_editor->show();
	} else {
		text_ctl_editor->hide();
		text_ctl_editor->edit(nullptr);
	}
}

TextControlEditorPlugin::TextControlEditorPlugin() {
	text_ctl_editor = memnew(TextControlEditor);
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(text_ctl_editor);

	text_ctl_editor->hide();
}
