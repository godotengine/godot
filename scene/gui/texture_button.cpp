/*************************************************************************/
/*  texture_button.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

	Size2 size = Control::get_minimum_size();
	if (resize_mode == RESIZE_SCALE) {
		if (normal.is_null()) {
			if (pressed.is_null()) {
				if (hover.is_null())
					if (click_mask.is_null())
						size = Size2();
					else
						size = click_mask->get_size();
				else
					size = hover->get_size();
			} else
				size = pressed->get_size();
		} else
			size = normal->get_size();
		size = size * scale.abs();
	}
	return size;
}

bool TextureButton::has_point(const Point2 &p_point) const {
	if (resize_mode == RESIZE_SCALE && (scale[0] == 0 || scale[1] == 0)) {
		return false;
	}
	Point2 ppos = (resize_mode == RESIZE_SCALE) ? (p_point / scale.abs()) : p_point;
	if (click_mask.is_valid()) {

		Point2i p = ppos;
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
				if (resize_mode == RESIZE_STRETCH) {
					switch (stretch_mode) {
						case STRETCH_KEEP:
							size = texdraw->get_size();
							break;
						case STRETCH_SCALE_ON_EXPAND:
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
				} else {
					size = texdraw->get_size() * scale;
				}
				if (tile)
					draw_texture_rect(texdraw, Rect2(ofs, size), tile, modulate);
				else
					draw_texture_rect_region(texdraw, Rect2(ofs, size), tex_regin, modulate);
			}
			if (has_focus() && focused.is_valid()) {

				Rect2 drect(Point2(), get_size());
				draw_texture_rect(focused, drect, false, modulate);
			};
		} break;
	}
}

void TextureButton::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_normal_texture", "texture:Texture"), &TextureButton::set_normal_texture);
	ObjectTypeDB::bind_method(_MD("set_pressed_texture", "texture:Texture"), &TextureButton::set_pressed_texture);
	ObjectTypeDB::bind_method(_MD("set_hover_texture", "texture:Texture"), &TextureButton::set_hover_texture);
	ObjectTypeDB::bind_method(_MD("set_disabled_texture", "texture:Texture"), &TextureButton::set_disabled_texture);
	ObjectTypeDB::bind_method(_MD("set_focused_texture", "texture:Texture"), &TextureButton::set_focused_texture);
	ObjectTypeDB::bind_method(_MD("set_click_mask", "mask:BitMap"), &TextureButton::set_click_mask);
	ObjectTypeDB::bind_method(_MD("set_texture_scale", "scale"), &TextureButton::set_texture_scale);
	ObjectTypeDB::bind_method(_MD("set_modulate", "color"), &TextureButton::set_modulate);
	ObjectTypeDB::bind_method(_MD("set_resize_mode", "p_mode"), &TextureButton::set_resize_mode);
	ObjectTypeDB::bind_method(_MD("set_stretch_mode", "p_mode"), &TextureButton::set_stretch_mode);

	ObjectTypeDB::bind_method(_MD("get_normal_texture:Texture"), &TextureButton::get_normal_texture);
	ObjectTypeDB::bind_method(_MD("get_pressed_texture:Texture"), &TextureButton::get_pressed_texture);
	ObjectTypeDB::bind_method(_MD("get_hover_texture:Texture"), &TextureButton::get_hover_texture);
	ObjectTypeDB::bind_method(_MD("get_disabled_texture:Texture"), &TextureButton::get_disabled_texture);
	ObjectTypeDB::bind_method(_MD("get_focused_texture:Texture"), &TextureButton::get_focused_texture);
	ObjectTypeDB::bind_method(_MD("get_click_mask:BitMap"), &TextureButton::get_click_mask);
	ObjectTypeDB::bind_method(_MD("get_texture_scale"), &TextureButton::get_texture_scale);
	ObjectTypeDB::bind_method(_MD("get_modulate"), &TextureButton::get_modulate);
	ObjectTypeDB::bind_method(_MD("get_resize_mode"), &TextureButton::get_resize_mode);
	ObjectTypeDB::bind_method(_MD("get_stretch_mode"), &TextureButton::get_stretch_mode);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "textures/normal", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_normal_texture"), _SCS("get_normal_texture"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "textures/pressed", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_pressed_texture"), _SCS("get_pressed_texture"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "textures/hover", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_hover_texture"), _SCS("get_hover_texture"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "textures/disabled", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_disabled_texture"), _SCS("get_disabled_texture"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "textures/focused", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_focused_texture"), _SCS("get_focused_texture"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "textures/click_mask", PROPERTY_HINT_RESOURCE_TYPE, "BitMap"), _SCS("set_click_mask"), _SCS("get_click_mask"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params/resize_mode", PROPERTY_HINT_ENUM, "Scale (Compat),Stretch"), _SCS("set_resize_mode"), _SCS("get_resize_mode"));
	ADD_PROPERTYNO(PropertyInfo(Variant::VECTOR2, "params/scale", PROPERTY_HINT_RANGE, "0.01,1024,0.01"), _SCS("set_texture_scale"), _SCS("get_texture_scale"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params/stretch_mode", PROPERTY_HINT_ENUM, "Scale On Expand (Compat),Scale,Tile,Keep,Keep Centered,Keep Aspect,Keep Aspect Centered,Keep Aspect Covered"), _SCS("set_stretch_mode"), _SCS("get_stretch_mode"));
	ADD_PROPERTYNO(PropertyInfo(Variant::COLOR, "params/modulate"), _SCS("set_modulate"), _SCS("get_modulate"));

	BIND_CONSTANT(RESIZE_SCALE);
	BIND_CONSTANT(RESIZE_STRETCH);

	BIND_CONSTANT(STRETCH_SCALE_ON_EXPAND);
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

void TextureButton::set_modulate(const Color &p_modulate) {
	modulate = p_modulate;
	update();
}

Color TextureButton::get_modulate() const {
	return modulate;
}

TextureButton::ResizeMode TextureButton::get_resize_mode() const {
	return resize_mode;
}

void TextureButton::set_resize_mode(TextureButton::ResizeMode p_mode) {
	resize_mode = p_mode;
	minimum_size_changed();
	update();
}

void TextureButton::set_texture_scale(Size2 p_scale) {
	scale = p_scale;
	minimum_size_changed();
	update();
}

Size2 TextureButton::get_texture_scale() const {
	return scale;
}

void TextureButton::set_stretch_mode(TextureButton::StretchMode p_mode) {
	stretch_mode = p_mode;
	update();
}

TextureButton::StretchMode TextureButton::get_stretch_mode() const {
	return stretch_mode;
}

TextureButton::TextureButton() {
	modulate = Color(1, 1, 1);
	resize_mode = RESIZE_SCALE;
	scale = Size2(1.0, 1.0);
	stretch_mode = STRETCH_SCALE_ON_EXPAND;
}
