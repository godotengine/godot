/*************************************************************************/
/*  text_layout_rect.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without startation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT startED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "text_layout_rect.h"
#include "core_string_names.h"
#include "scene/resources/text_layout.h"

void TextLayoutRect::_notification(int p_what) {

	if (layout.is_null())
		return;

	if (p_what == NOTIFICATION_RESIZED) {

		if (autosize) {
			layout->set_min_area(get_size());
			layout->set_max_area(get_size());
		}
		if (clip) {
			layout->set_clip_rect(Rect2(Point2(0, 0), get_size()));
		}
	} else if (p_what == NOTIFICATION_DRAW) {

		Point2 cell_offset = Point2(0, 0);
		switch (layout->get_parent_valign()) {
			case V_ALIGN_TOP: {
				cell_offset.y = 0;
			} break;
			case V_ALIGN_CENTER: {
				cell_offset.y = (get_size().y - layout->get_bounds().size.y) / 2;
			} break;
			case V_ALIGN_BOTTOM: {
				cell_offset.y = (get_size().y - layout->get_bounds().size.y);
			} break;
		}
		switch (layout->get_parent_halign()) {
			case H_ALIGN_LEFT: {
				cell_offset.x = 0;
			} break;
			case H_ALIGN_CENTER: {
				cell_offset.x = (get_size().x - layout->get_bounds().size.x) / 2;
			} break;
			case H_ALIGN_RIGHT: {
				cell_offset.x = (get_size().x - layout->get_bounds().size.x);
			} break;
		}
		layout->draw(get_canvas_item(), cell_offset);
	}
};

void TextLayoutRect::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_layout", "layout"), &TextLayoutRect::set_layout);
	ClassDB::bind_method(D_METHOD("get_layout"), &TextLayoutRect::get_layout);

	ClassDB::bind_method(D_METHOD("set_autosize", "autosize"), &TextLayoutRect::set_autosize);
	ClassDB::bind_method(D_METHOD("get_autosize"), &TextLayoutRect::get_autosize);

	ClassDB::bind_method(D_METHOD("set_clip", "clip"), &TextLayoutRect::set_clip);
	ClassDB::bind_method(D_METHOD("get_clip"), &TextLayoutRect::get_clip);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "layout", PROPERTY_HINT_RESOURCE_TYPE, "TextLayout"), "set_layout", "get_layout");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "autosize"), "set_autosize", "get_autosize");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "clip"), "set_clip", "get_clip");
};

void TextLayoutRect::set_autosize(bool p_autosize) {

	autosize = p_autosize;
	update();
};

bool TextLayoutRect::get_autosize() {

	return autosize;
};

void TextLayoutRect::set_clip(bool p_clip) {

	clip = p_clip;
	if ((!clip) && (!layout.is_null())) {
		layout->set_clip_rect(Rect2(0, 0, -1, -1));
	}
	update();
};

bool TextLayoutRect::get_clip() {

	return clip;
};

void TextLayoutRect::set_layout(const Ref<TextLayout> &p_layout) {

	if (!layout.is_null()) layout->disconnect(CoreStringNames::get_singleton()->changed, this, "update");
	layout = p_layout;
	if (!layout.is_null()) layout->connect(CoreStringNames::get_singleton()->changed, this, "update");
	update();
};

Ref<TextLayout> TextLayoutRect::get_layout() const {

	return layout;
};

TextLayoutRect::TextLayoutRect() {

	autosize = false;
	clip = true;
};

TextLayoutRect::~TextLayoutRect(){};
