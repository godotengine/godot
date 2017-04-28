/*************************************************************************/
/*  texture_editor_plugin.cpp                                            */
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
#include "texture_editor_plugin.h"

#include "editor/editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"

void TextureEditor::_gui_input(InputEvent p_event) {
}

void TextureEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_FIXED_PROCESS) {
	}

	if (p_what == NOTIFICATION_READY) {

		//get_scene()->connect("node_removed",this,"_node_removed");
	}

	if (p_what == NOTIFICATION_DRAW) {

		Ref<Texture> checkerboard = get_icon("Checkerboard", "EditorIcons");
		Size2 size = get_size();

		draw_texture_rect(checkerboard, Rect2(Point2(), size), true);

		int tex_width = texture->get_width() * size.height / texture->get_height();
		int tex_height = size.height;

		if (tex_width > size.width) {
			tex_width = size.width;
			tex_height = texture->get_height() * tex_width / texture->get_width();
		}

		int ofs_x = (size.width - tex_width) / 2;
		int ofs_y = (size.height - tex_height) / 2;

		draw_texture_rect(texture, Rect2(ofs_x, ofs_y, tex_width, tex_height));

		Ref<Font> font = get_font("font", "Label");

		String format;
		if (texture->cast_to<ImageTexture>()) {
			format = Image::get_format_name(texture->cast_to<ImageTexture>()->get_format());
		} else {
			format = texture->get_class();
		}
		String text = itos(texture->get_width()) + "x" + itos(texture->get_height()) + " " + format;

		Size2 rect = font->get_string_size(text);

		Vector2 draw_from = size - rect + Size2(-2, font->get_ascent() - 2);
		if (draw_from.x < 0)
			draw_from.x = 0;

		draw_string(font, draw_from + Vector2(2, 2), text, Color(0, 0, 0, 0.5), size.width);
		draw_string(font, draw_from - Vector2(2, 2), text, Color(0, 0, 0, 0.5), size.width);
		draw_string(font, draw_from, text, Color(1, 1, 1, 1), size.width);
	}
}

void TextureEditor::edit(Ref<Texture> p_texture) {

	texture = p_texture;

	if (!texture.is_null())
		update();
	else {

		hide();
	}
}

void TextureEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &TextureEditor::_gui_input);
}

TextureEditor::TextureEditor() {

	set_custom_minimum_size(Size2(1, 150));
}

void TextureEditorPlugin::edit(Object *p_object) {

	Texture *s = p_object->cast_to<Texture>();
	if (!s)
		return;

	texture_editor->edit(Ref<Texture>(s));
}

bool TextureEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Texture");
}

void TextureEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		texture_editor->show();
		//texture_editor->set_process(true);
	} else {

		texture_editor->hide();
		//texture_editor->set_process(false);
	}
}

TextureEditorPlugin::TextureEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	texture_editor = memnew(TextureEditor);
	add_control_to_container(CONTAINER_PROPERTY_EDITOR_BOTTOM, texture_editor);
	texture_editor->hide();
}

TextureEditorPlugin::~TextureEditorPlugin() {
}
