/*************************************************************************/
/*  texture_button.cpp                                                   */
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
#include "texture_button.h"

Size2 TextureButton::get_minimum_size() const {

	Size2 rscale = Control::get_minimum_size();

	if (!expand) {
		if (normal.is_null()) {
			if (pressed.is_null()) {
				if (hover.is_null())
					if (click_mask.is_null())
						rscale = Size2();
					else
						rscale = click_mask->get_size();
				else
					rscale = hover->get_size();
			} else
				rscale = pressed->get_size();

		} else
			rscale = normal->get_size();
	}

	return rscale.abs();
}

bool TextureButton::has_point(const Point2 &p_point) const {

	if (click_mask.is_valid()) {

		Point2i p = p_point;
		if (p.x < 0 || p.x >= click_mask->get_size().width || p.y < 0 || p.y >= click_mask->get_size().height)
			return false;

		return click_mask->get_bit(p);
	}

	return Control::has_point(p_point);
}

void TextureButton::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_DRAW: {
			DrawMode draw_mode = get_draw_mode();

			Ref<Texture> texdraw;

			switch (draw_mode) {
				case DRAW_NORMAL: {

					if (normal.is_valid())
						texdraw = normal;
				} break;
				case DRAW_PRESSED: {

					if (pressed.is_null()) {
						if (hover.is_null()) {
							if (normal.is_valid())
								texdraw = normal;
						} else
							texdraw = hover;

					} else
						texdraw = pressed;
				} break;
				case DRAW_HOVER: {

					if (hover.is_null()) {
						if (pressed.is_valid() && is_pressed())
							texdraw = pressed;
						else if (normal.is_valid())
							texdraw = normal;
					} else
						texdraw = hover;
				} break;
				case DRAW_DISABLED: {

					if (disabled.is_null()) {
						if (normal.is_valid())
							texdraw = normal;
					} else
						texdraw = disabled;
				} break;
			}

			if (texdraw.is_valid()) {
				Point2 ofs;
				Size2 size = texdraw->get_size();
				Rect2 tex_regin = Rect2(Point2(), texdraw->get_size());
				bool tile = false;
				if (expand) {
					switch (stretch_mode) {
						case STRETCH_KEEP:
							size = texdraw->get_size();
							break;
						case STRETCH_SCALE:
							size = get_size();
							break;
						case STRETCH_TILE:
							size = get_size();
							tile = true;
							break;
						case STRETCH_KEEP_CENTERED:
							ofs = (get_size() - texdraw->get_size()) / 2;
							size = texdraw->get_size();
							break;
						case STRETCH_KEEP_ASPECT_CENTERED:
						case STRETCH_KEEP_ASPECT: {
							Size2 _size = get_size();
							float tex_width = texdraw->get_width() * _size.height / texdraw->get_height();
							float tex_height = _size.height;

							if (tex_width > _size.width) {
								tex_width = _size.width;
								tex_height = texdraw->get_height() * tex_width / texdraw->get_width();
							}

							if (stretch_mode == STRETCH_KEEP_ASPECT_CENTERED) {
								ofs.x = (_size.width - tex_width) / 2;
								ofs.y = (_size.height - tex_height) / 2;
							}
							size.width = tex_width;
							size.height = tex_height;
						} break;
						case STRETCH_KEEP_ASPECT_COVERED: {
							size = get_size();
							Size2 tex_size = texdraw->get_size();
							Size2 scaleSize(size.width / tex_size.width, size.height / tex_size.height);
							float scale = scaleSize.width > scaleSize.height ? scaleSize.width : scaleSize.height;
							Size2 scaledTexSize = tex_size * scale;
							Point2 ofs = ((scaledTexSize - size) / scale).abs() / 2.0f;
							tex_regin = Rect2(ofs, size / scale);
						} break;
					}
				}
				if (tile)
					draw_texture_rect(texdraw, Rect2(ofs, size), tile);
				else
					draw_texture_rect_region(texdraw, Rect2(ofs, size), tex_regin);
			}
			if (has_focus() && focused.is_valid()) {

				Rect2 drect(Point2(), get_size());
				draw_texture_rect(focused, drect, false);
			};
		} break;
	}
}

void TextureButton::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_normal_texture", "texture:Texture"), &TextureButton::set_normal_texture);
	ClassDB::bind_method(D_METHOD("set_pressed_texture", "texture:Texture"), &TextureButton::set_pressed_texture);
	ClassDB::bind_method(D_METHOD("set_hover_texture", "texture:Texture"), &TextureButton::set_hover_texture);
	ClassDB::bind_method(D_METHOD("set_disabled_texture", "texture:Texture"), &TextureButton::set_disabled_texture);
	ClassDB::bind_method(D_METHOD("set_focused_texture", "texture:Texture"), &TextureButton::set_focused_texture);
	ClassDB::bind_method(D_METHOD("set_click_mask", "mask:BitMap"), &TextureButton::set_click_mask);
	ClassDB::bind_method(D_METHOD("set_expand", "p_expand"), &TextureButton::set_expand);
	ClassDB::bind_method(D_METHOD("set_stretch_mode", "p_mode"), &TextureButton::set_stretch_mode);

	ClassDB::bind_method(D_METHOD("get_normal_texture:Texture"), &TextureButton::get_normal_texture);
	ClassDB::bind_method(D_METHOD("get_pressed_texture:Texture"), &TextureButton::get_pressed_texture);
	ClassDB::bind_method(D_METHOD("get_hover_texture:Texture"), &TextureButton::get_hover_texture);
	ClassDB::bind_method(D_METHOD("get_disabled_texture:Texture"), &TextureButton::get_disabled_texture);
	ClassDB::bind_method(D_METHOD("get_focused_texture:Texture"), &TextureButton::get_focused_texture);
	ClassDB::bind_method(D_METHOD("get_click_mask:BitMap"), &TextureButton::get_click_mask);
	ClassDB::bind_method(D_METHOD("get_expand"), &TextureButton::get_expand);
	ClassDB::bind_method(D_METHOD("get_stretch_mode"), &TextureButton::get_stretch_mode);

	ADD_GROUP("Textures", "texture_");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture_normal", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_normal_texture", "get_normal_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture_pressed", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_pressed_texture", "get_pressed_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture_hover", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_hover_texture", "get_hover_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture_disabled", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_disabled_texture", "get_disabled_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture_focused", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_focused_texture", "get_focused_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture_click_mask", PROPERTY_HINT_RESOURCE_TYPE, "BitMap"), "set_click_mask", "get_click_mask");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "expand", PROPERTY_HINT_RESOURCE_TYPE, "bool"), "set_expand", "get_expand");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "stretch_mode", PROPERTY_HINT_ENUM, "Scale,Tile,Keep,Keep Centered,Keep Aspect,Keep Aspect Centered,Keep Aspect Covered"), "set_stretch_mode", "get_stretch_mode");

	BIND_CONSTANT(STRETCH_SCALE);
	BIND_CONSTANT(STRETCH_TILE);
	BIND_CONSTANT(STRETCH_KEEP);
	BIND_CONSTANT(STRETCH_KEEP_CENTERED);
	BIND_CONSTANT(STRETCH_KEEP_ASPECT);
	BIND_CONSTANT(STRETCH_KEEP_ASPECT_CENTERED);
	BIND_CONSTANT(STRETCH_KEEP_ASPECT_COVERED);
}

void TextureButton::set_normal_texture(const Ref<Texture> &p_normal) {

	normal = p_normal;
	update();
	minimum_size_changed();
}

void TextureButton::set_pressed_texture(const Ref<Texture> &p_pressed) {

	pressed = p_pressed;
	update();
}
void TextureButton::set_hover_texture(const Ref<Texture> &p_hover) {

	hover = p_hover;
	update();
}
void TextureButton::set_disabled_texture(const Ref<Texture> &p_disabled) {

	disabled = p_disabled;
	update();
}
void TextureButton::set_click_mask(const Ref<BitMap> &p_click_mask) {

	click_mask = p_click_mask;
	update();
}

Ref<Texture> TextureButton::get_normal_texture() const {

	return normal;
}
Ref<Texture> TextureButton::get_pressed_texture() const {

	return pressed;
}
Ref<Texture> TextureButton::get_hover_texture() const {

	return hover;
}
Ref<Texture> TextureButton::get_disabled_texture() const {

	return disabled;
}
Ref<BitMap> TextureButton::get_click_mask() const {

	return click_mask;
}

Ref<Texture> TextureButton::get_focused_texture() const {

	return focused;
};

void TextureButton::set_focused_texture(const Ref<Texture> &p_focused) {

	focused = p_focused;
};

bool TextureButton::get_expand() const {
	return expand;
}

void TextureButton::set_expand(bool p_expand) {
	expand = p_expand;
	minimum_size_changed();
	update();
}

void TextureButton::set_stretch_mode(StretchMode p_mode) {
	stretch_mode = p_mode;
	update();
}

TextureButton::StretchMode TextureButton::get_stretch_mode() const {
	return stretch_mode;
}

TextureButton::TextureButton() {
	expand = false;
	stretch_mode = STRETCH_SCALE;
}
