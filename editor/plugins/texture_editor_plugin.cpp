/*************************************************************************/
/*  texture_editor_plugin.cpp                                            */
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

#include "texture_editor_plugin.h"

#include "editor/editor_scale.h"
#include "scene/resources/dynamic_font.h"

TextureRect *TexturePreview::get_texture_display() {
	return texture_display;
}

void TexturePreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (!is_inside_tree()) {
				// TODO: This is a workaround because `NOTIFICATION_THEME_CHANGED`
				// is getting called for some reason when the `TexturePreview` is
				// getting destroyed, which causes `get_theme_font()` to return `nullptr`.
				// See https://github.com/godotengine/godot/issues/50743.
				break;
			}

			if (metadata_label) {
				Ref<DynamicFont> metadata_label_font = get_font("expression", "EditorFonts")->duplicate();
				metadata_label_font->set_size(16 * EDSCALE);
				metadata_label_font->set_outline_size(2 * EDSCALE);
				metadata_label_font->set_outline_color(Color::named("black"));
				metadata_label->add_font_override("font", metadata_label_font);
			}

			checkerboard->set_texture(get_icon("Checkerboard", "EditorIcons"));
		} break;
	}
}

void TexturePreview::_update_metadata_label_text() {
	Ref<Texture> texture = texture_display->get_texture();

	String format;
	if (Object::cast_to<ImageTexture>(*texture)) {
		format = Image::get_format_name(Object::cast_to<ImageTexture>(*texture)->get_format());
	} else if (Object::cast_to<StreamTexture>(*texture)) {
		format = Image::get_format_name(Object::cast_to<StreamTexture>(*texture)->get_format());
	} else {
		format = texture->get_class();
	}

	metadata_label->set_text(itos(texture->get_width()) + "x" + itos(texture->get_height()) + " " + format);
}

void TexturePreview::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_metadata_label_text"), &TexturePreview::_update_metadata_label_text);
}

TexturePreview::TexturePreview(Ref<Texture> p_texture, bool p_show_metadata) {
	checkerboard = memnew(TextureRect);
	checkerboard->set_stretch_mode(TextureRect::STRETCH_TILE);
	checkerboard->set_custom_minimum_size(Size2(0.0, 256.0) * EDSCALE);
	add_child(checkerboard);

	texture_display = memnew(TextureRect);
	texture_display->set_texture(p_texture);
	texture_display->set_anchors_preset(TextureRect::PRESET_WIDE);
	texture_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	texture_display->set_expand(true);
	add_child(texture_display);

	if (p_show_metadata) {
		metadata_label = memnew(Label);

		_update_metadata_label_text();
		p_texture->connect("changed", this, "_update_metadata_label_text");

		// It's okay that these colors are static since the grid color is static too.
		metadata_label->add_color_override("font_color", Color::named("white"));
		metadata_label->add_color_override("font_color_shadow", Color::named("black"));

		metadata_label->add_constant_override("shadow_as_outline", 1);
		metadata_label->set_h_size_flags(Control::SIZE_SHRINK_END);
		metadata_label->set_v_size_flags(Control::SIZE_SHRINK_END);

		add_child(metadata_label);
	}
}

bool EditorInspectorPluginTexture::can_handle(Object *p_object) {
	return Object::cast_to<ImageTexture>(p_object) != nullptr || Object::cast_to<AtlasTexture>(p_object) != nullptr || Object::cast_to<StreamTexture>(p_object) != nullptr || Object::cast_to<LargeTexture>(p_object) != nullptr || Object::cast_to<AnimatedTexture>(p_object) != nullptr;
}

void EditorInspectorPluginTexture::parse_begin(Object *p_object) {
	Ref<Texture> texture(Object::cast_to<Texture>(p_object));

	add_custom_control(memnew(TexturePreview(texture, true)));
}

TextureEditorPlugin::TextureEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginTexture> plugin;
	plugin.instance();
	add_inspector_plugin(plugin);
}
