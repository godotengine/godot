/*************************************************************************/
/*  font_editor_plugin.cpp                                               */
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

#include "font_editor_plugin.h"

#include "editor/editor_scale.h"

void FontDataPreview::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		Color text_color = get_theme_color(SNAME("font_color"), SNAME("Label"));
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

void FontDataPreview::set_data(const Ref<FontData> &p_data) {
	Ref<Font> f = memnew(Font);
	f->add_data(p_data);

	line->clear();
	if (p_data.is_valid()) {
		String sample;
		static const String sample_base = U"12漢字ԱբΑαАбΑαאבابܐܒހށआআਆઆଆஆఆಆആආกิກິༀကႠა한글ሀᎣᐁᚁᚠᜀᜠᝀᝠកᠠᤁᥐAb😀";
		for (int i = 0; i < sample_base.length(); i++) {
			if (p_data->has_char(sample_base[i])) {
				sample += sample_base[i];
			}
		}
		if (sample.is_empty()) {
			sample = p_data->get_supported_chars().substr(0, 6);
		}
		line->add_string(sample, f, 72);
	}

	update();
}

FontDataPreview::FontDataPreview() {
	line.instantiate();
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

bool EditorInspectorPluginFont::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide) {
	return false;
}

/*************************************************************************/

FontEditorPlugin::FontEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginFont> fd_plugin;
	fd_plugin.instantiate();
	EditorInspector::add_inspector_plugin(fd_plugin);
}
