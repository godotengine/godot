/*************************************************************************/
/*  font_editor_plugin.cpp                                               */
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

#include "font_editor_plugin.h"

#include "editor/editor_scale.h"

void FontDataPreview::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		Color text_color = get_theme_color("font_color", "Label");
		Color line_color = text_color;
		line_color.a *= 0.6;
		Vector2 pos = (get_size() - line->get_size()) / 2;
		line->draw(get_canvas_item(), pos, text_color);
		draw_line(Vector2(0, pos.y + line->get_line_ascent()), Vector2(pos.x - 5, pos.y + line->get_line_ascent()), line_color);
		draw_line(Vector2(pos.x + line->get_size().x + 5, pos.y + line->get_line_ascent()), Vector2(get_size().x, pos.y + line->get_line_ascent()), line_color);
	}
}

void FontDataPreview::_bind_methods() {}

Size2 FontDataPreview::get_minimum_size() const {
	return Vector2(64, 64) * EDSCALE;
}

struct FSample {
	String script;
	String sample;
};

static FSample _samples[] = {
	{ "hani", U"漢語" },
	{ "armn", U"Աբ" },
	{ "copt", U"Αα" },
	{ "cyrl", U"Аб" },
	{ "grek", U"Αα" },
	{ "hebr", U"אב" },
	{ "arab", U"اب" },
	{ "syrc", U"ܐܒ" },
	{ "thaa", U"ހށ" },
	{ "deva", U"आ" },
	{ "beng", U"আ" },
	{ "guru", U"ਆ" },
	{ "gujr", U"આ" },
	{ "orya", U"ଆ" },
	{ "taml", U"ஆ" },
	{ "telu", U"ఆ" },
	{ "knda", U"ಆ" },
	{ "mylm", U"ആ" },
	{ "sinh", U"ආ" },
	{ "thai", U"กิ" },
	{ "laoo", U"ກິ" },
	{ "tibt", U"ༀ" },
	{ "mymr", U"က" },
	{ "geor", U"Ⴀა" },
	{ "hang", U"한글" },
	{ "ethi", U"ሀ" },
	{ "cher", U"Ꭳ" },
	{ "cans", U"ᐁ" },
	{ "ogam", U"ᚁ" },
	{ "runr", U"ᚠ" },
	{ "tglg", U"ᜀ" },
	{ "hano", U"ᜠ" },
	{ "buhd", U"ᝀ" },
	{ "tagb", U"ᝠ" },
	{ "khmr", U"ក" },
	{ "mong", U"ᠠ" },
	{ "limb", U"ᤁ" },
	{ "tale", U"ᥐ" },
	{ "latn", U"Ab" },
	{ "zyyy", U"😀" },
	{ "", U"" }
};

void FontDataPreview::set_data(const Ref<FontData> &p_data) {
	Ref<Font> f = memnew(Font);
	f->add_data(p_data);

	line->clear();

	String sample;
	for (int i = 0; _samples[i].script != String(); i++) {
		if (p_data->is_script_supported(_samples[i].script)) {
			if (p_data->has_char(_samples[i].sample[0])) {
				sample += _samples[i].sample;
			}
		}
	}
	line->add_string(sample, f, 72);

	update();
}

FontDataPreview::FontDataPreview() {
	line.instance();
}

/*************************************************************************/

void FontDataEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_SORT_CHILDREN) {
		int split_width = get_name_split_ratio() * get_size().width;
		button->set_size(Size2(get_theme_icon("Add", "EditorIcons")->get_width(), get_size().height));
		if (is_layout_rtl()) {
			if (le != nullptr) {
				fit_child_in_rect(le, Rect2(Vector2(split_width, 0), Size2(split_width, get_size().height)));
			}
			fit_child_in_rect(chk, Rect2(Vector2(split_width - chk->get_size().x, 0), Size2(chk->get_size().x, get_size().height)));
			fit_child_in_rect(button, Rect2(Vector2(0, 0), Size2(button->get_size().width, get_size().height)));
		} else {
			if (le != nullptr) {
				fit_child_in_rect(le, Rect2(Vector2(0, 0), Size2(split_width, get_size().height)));
			}
			fit_child_in_rect(chk, Rect2(Vector2(split_width, 0), Size2(chk->get_size().x, get_size().height)));
			fit_child_in_rect(button, Rect2(Vector2(get_size().width - button->get_size().width, 0), Size2(button->get_size().width, get_size().height)));
		}
		update();
	}
	if (p_what == NOTIFICATION_DRAW) {
		int split_width = get_name_split_ratio() * get_size().width;
		Color dark_color = get_theme_color("dark_color_2", "Editor");
		if (is_layout_rtl()) {
			draw_rect(Rect2(Vector2(0, 0), Size2(split_width, get_size().height)), dark_color);
		} else {
			draw_rect(Rect2(Vector2(split_width, 0), Size2(split_width, get_size().height)), dark_color);
		}
	}
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		if (le != nullptr) {
			button->set_icon(get_theme_icon("Add", "EditorIcons"));
		} else {
			button->set_icon(get_theme_icon("Remove", "EditorIcons"));
		}
		queue_sort();
	}
	if (p_what == NOTIFICATION_RESIZED) {
		queue_sort();
	}
}

void FontDataEditor::update_property() {
	if (le == nullptr) {
		bool c = get_edited_object()->get(get_edited_property());
		chk->set_pressed(c);
		chk->set_disabled(is_read_only());
	}
}

Size2 FontDataEditor::get_minimum_size() const {
	return Size2(0, 60);
}

void FontDataEditor::_bind_methods() {
}

void FontDataEditor::init_lang_add() {
	le = memnew(LineEdit);
	le->set_placeholder("Language code");
	le->set_custom_minimum_size(Size2(get_size().width / 2, 0));
	le->set_editable(true);
	add_child(le);

	button->set_icon(get_theme_icon("Add", "EditorIcons"));
	button->connect("pressed", callable_mp(this, &FontDataEditor::add_lang));
}

void FontDataEditor::init_lang_edit() {
	button->set_icon(get_theme_icon("Remove", "EditorIcons"));
	button->connect("pressed", callable_mp(this, &FontDataEditor::remove_lang));
	chk->connect("toggled", callable_mp(this, &FontDataEditor::toggle_lang));
}

void FontDataEditor::init_script_add() {
	le = memnew(LineEdit);
	le->set_placeholder("Script code");
	le->set_custom_minimum_size(Size2(get_size().width / 2, 0));
	le->set_editable(true);
	add_child(le);

	button->set_icon(get_theme_icon("Add", "EditorIcons"));
	button->connect("pressed", callable_mp(this, &FontDataEditor::add_script));
}

void FontDataEditor::init_script_edit() {
	button->set_icon(get_theme_icon("Remove", "EditorIcons"));
	button->connect("pressed", callable_mp(this, &FontDataEditor::remove_script));
	chk->connect("toggled", callable_mp(this, &FontDataEditor::toggle_script));
}

void FontDataEditor::add_lang() {
	FontData *fd = Object::cast_to<FontData>(get_edited_object());
	if (fd != nullptr && !le->get_text().is_empty()) {
		fd->set_language_support_override(le->get_text(), chk->is_pressed());
		le->set_text("");
		chk->set_pressed(false);
	}
}

void FontDataEditor::add_script() {
	FontData *fd = Object::cast_to<FontData>(get_edited_object());
	if (fd != nullptr && le->get_text().length() == 4) {
		fd->set_script_support_override(le->get_text(), chk->is_pressed());
		le->set_text("");
		chk->set_pressed(false);
	}
}

void FontDataEditor::toggle_lang(bool p_pressed) {
	FontData *fd = Object::cast_to<FontData>(get_edited_object());
	if (fd != nullptr) {
		String lang = String(get_edited_property()).replace("language_support_override/", "");
		fd->set_language_support_override(lang, p_pressed);
	}
}

void FontDataEditor::toggle_script(bool p_pressed) {
	FontData *fd = Object::cast_to<FontData>(get_edited_object());
	if (fd != nullptr) {
		String script = String(get_edited_property()).replace("script_support_override/", "");
		fd->set_script_support_override(script, p_pressed);
	}
}

void FontDataEditor::remove_lang() {
	FontData *fd = Object::cast_to<FontData>(get_edited_object());
	if (fd != nullptr) {
		String lang = String(get_edited_property()).replace("language_support_override/", "");
		fd->remove_language_support_override(lang);
	}
}

void FontDataEditor::remove_script() {
	FontData *fd = Object::cast_to<FontData>(get_edited_object());
	if (fd != nullptr) {
		String script = String(get_edited_property()).replace("script_support_override/", "");
		fd->remove_script_support_override(script);
	}
}

FontDataEditor::FontDataEditor() {
	chk = memnew(CheckBox);
	chk->set_text(TTR("On"));
	chk->set_flat(true);
	add_child(chk);

	button = memnew(Button);
	button->set_flat(true);
	add_child(button);
}

/*************************************************************************/

bool EditorInspectorPluginFont::can_handle(Object *p_object) {
	return Object::cast_to<FontData>(p_object) != nullptr;
}

void EditorInspectorPluginFont::parse_begin(Object *p_object) {
	FontData *fd = Object::cast_to<FontData>(p_object);
	ERR_FAIL_COND(!fd);

	FontDataPreview *editor = memnew(FontDataPreview);
	editor->set_data(fd);
	add_custom_control(editor);
}

bool EditorInspectorPluginFont::parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage, bool p_wide) {
	if (p_path.begins_with("language_support_override/") && p_object->is_class("FontData")) {
		String lang = p_path.replace("language_support_override/", "");

		FontDataEditor *editor = memnew(FontDataEditor);
		if (lang != "_new") {
			editor->init_lang_edit();
		} else {
			editor->init_lang_add();
		}
		add_property_editor(p_path, editor);

		return true;
	}

	if (p_path.begins_with("script_support_override/") && p_object->is_class("FontData")) {
		String script = p_path.replace("script_support_override/", "");

		FontDataEditor *editor = memnew(FontDataEditor);
		if (script != "_new") {
			editor->init_script_edit();
		} else {
			editor->init_script_add();
		}
		add_property_editor(p_path, editor);

		return true;
	}

	return false;
}

/*************************************************************************/

FontEditorPlugin::FontEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginFont> fd_plugin;
	fd_plugin.instance();
	EditorInspector::add_inspector_plugin(fd_plugin);
}
