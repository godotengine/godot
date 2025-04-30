/**************************************************************************/
/*  label_settings.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "label_settings.h"

void LabelSettings::_font_changed() {
	emit_changed();
}

void LabelSettings::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_line_spacing", "spacing"), &LabelSettings::set_line_spacing);
	ClassDB::bind_method(D_METHOD("get_line_spacing"), &LabelSettings::get_line_spacing);

	ClassDB::bind_method(D_METHOD("set_paragraph_spacing", "spacing"), &LabelSettings::set_paragraph_spacing);
	ClassDB::bind_method(D_METHOD("get_paragraph_spacing"), &LabelSettings::get_paragraph_spacing);

	ClassDB::bind_method(D_METHOD("set_font", "font"), &LabelSettings::set_font);
	ClassDB::bind_method(D_METHOD("get_font"), &LabelSettings::get_font);

	ClassDB::bind_method(D_METHOD("set_font_size", "size"), &LabelSettings::set_font_size);
	ClassDB::bind_method(D_METHOD("get_font_size"), &LabelSettings::get_font_size);

	ClassDB::bind_method(D_METHOD("set_font_color", "color"), &LabelSettings::set_font_color);
	ClassDB::bind_method(D_METHOD("get_font_color"), &LabelSettings::get_font_color);

	ClassDB::bind_method(D_METHOD("set_outline_size", "size"), &LabelSettings::set_outline_size);
	ClassDB::bind_method(D_METHOD("get_outline_size"), &LabelSettings::get_outline_size);

	ClassDB::bind_method(D_METHOD("set_outline_color", "color"), &LabelSettings::set_outline_color);
	ClassDB::bind_method(D_METHOD("get_outline_color"), &LabelSettings::get_outline_color);

	ClassDB::bind_method(D_METHOD("set_shadow_size", "size"), &LabelSettings::set_shadow_size);
	ClassDB::bind_method(D_METHOD("get_shadow_size"), &LabelSettings::get_shadow_size);

	ClassDB::bind_method(D_METHOD("set_shadow_color", "color"), &LabelSettings::set_shadow_color);
	ClassDB::bind_method(D_METHOD("get_shadow_color"), &LabelSettings::get_shadow_color);

	ClassDB::bind_method(D_METHOD("set_shadow_offset", "offset"), &LabelSettings::set_shadow_offset);
	ClassDB::bind_method(D_METHOD("get_shadow_offset"), &LabelSettings::get_shadow_offset);

	// Stacked outlines
	ClassDB::bind_method(D_METHOD("get_stacked_outline_count"), &LabelSettings::get_stacked_outline_count);
	ClassDB::bind_method(D_METHOD("set_stacked_outline_count", "count"), &LabelSettings::set_stacked_outline_count);
	ClassDB::bind_method(D_METHOD("add_stacked_outline", "index"), &LabelSettings::add_stacked_outline, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("move_stacked_outline", "from_index", "to_position"), &LabelSettings::move_stacked_outline);
	ClassDB::bind_method(D_METHOD("remove_stacked_outline", "index"), &LabelSettings::remove_stacked_outline);
	ClassDB::bind_method(D_METHOD("set_stacked_outline_size", "index", "size"), &LabelSettings::set_stacked_outline_size);
	ClassDB::bind_method(D_METHOD("get_stacked_outline_size", "index"), &LabelSettings::get_stacked_outline_size);
	ClassDB::bind_method(D_METHOD("set_stacked_outline_color", "index", "color"), &LabelSettings::set_stacked_outline_color);
	ClassDB::bind_method(D_METHOD("get_stacked_outline_color", "index"), &LabelSettings::get_stacked_outline_color);

	// Stacked shadows
	ClassDB::bind_method(D_METHOD("get_stacked_shadow_count"), &LabelSettings::get_stacked_shadow_count);
	ClassDB::bind_method(D_METHOD("set_stacked_shadow_count", "count"), &LabelSettings::set_stacked_shadow_count);
	ClassDB::bind_method(D_METHOD("add_stacked_shadow", "index"), &LabelSettings::add_stacked_shadow, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("move_stacked_shadow", "from_index", "to_position"), &LabelSettings::move_stacked_shadow);
	ClassDB::bind_method(D_METHOD("remove_stacked_shadow", "index"), &LabelSettings::remove_stacked_shadow);
	ClassDB::bind_method(D_METHOD("set_stacked_shadow_offset", "index", "offset"), &LabelSettings::set_stacked_shadow_offset);
	ClassDB::bind_method(D_METHOD("get_stacked_shadow_offset", "index"), &LabelSettings::get_stacked_shadow_offset);
	ClassDB::bind_method(D_METHOD("set_stacked_shadow_color", "index", "color"), &LabelSettings::set_stacked_shadow_color);
	ClassDB::bind_method(D_METHOD("get_stacked_shadow_color", "index"), &LabelSettings::get_stacked_shadow_color);
	ClassDB::bind_method(D_METHOD("set_stacked_shadow_outline_size", "index", "size"), &LabelSettings::set_stacked_shadow_outline_size);
	ClassDB::bind_method(D_METHOD("get_stacked_shadow_outline_size", "index"), &LabelSettings::get_stacked_shadow_outline_size);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "line_spacing", PROPERTY_HINT_NONE, "suffix:px"), "set_line_spacing", "get_line_spacing");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "paragraph_spacing", PROPERTY_HINT_NONE, "suffix:px"), "set_paragraph_spacing", "get_paragraph_spacing");

	ADD_GROUP("Font", "font_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "font_size", PROPERTY_HINT_RANGE, "1,1024,1,or_greater,suffix:px"), "set_font_size", "get_font_size");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "font_color"), "set_font_color", "get_font_color");

	ADD_GROUP("Outline", "outline_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "outline_size", PROPERTY_HINT_RANGE, "0,127,1,or_greater,suffix:px"), "set_outline_size", "get_outline_size");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "outline_color"), "set_outline_color", "get_outline_color");

	ADD_GROUP("Shadow", "shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_size", PROPERTY_HINT_RANGE, "0,127,1,or_greater,suffix:px"), "set_shadow_size", "get_shadow_size");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color"), "set_shadow_color", "get_shadow_color");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "shadow_offset", PROPERTY_HINT_NONE, "suffix:px"), "set_shadow_offset", "get_shadow_offset");

	ADD_GROUP("Stacked Effects", "");
	ADD_ARRAY_COUNT("Stacked Outlines", "stacked_outline_count", "set_stacked_outline_count", "get_stacked_outline_count", "stacked_outline_");
	ADD_ARRAY_COUNT("Stacked Shadows", "stacked_shadow_count", "set_stacked_shadow_count", "get_stacked_shadow_count", "stacked_shadow_");

	constexpr StackedOutlineData stacked_outline_defaults;

	stacked_outline_base_property_helper.set_prefix("stacked_outline_");
	stacked_outline_base_property_helper.set_array_length_getter(&LabelSettings::get_stacked_outline_count);
	stacked_outline_base_property_helper.register_property(PropertyInfo(Variant::INT, "size", PROPERTY_HINT_NONE, "0,127,1,or_greater,suffix:px"), stacked_outline_defaults.size, &LabelSettings::set_stacked_outline_size, &LabelSettings::get_stacked_outline_size);
	stacked_outline_base_property_helper.register_property(PropertyInfo(Variant::COLOR, "color"), stacked_outline_defaults.color, &LabelSettings::set_stacked_outline_color, &LabelSettings::get_stacked_outline_color);
	PropertyListHelper::register_base_helper(&stacked_outline_base_property_helper);

	constexpr StackedShadowData stacked_shadow_defaults;

	stacked_shadow_base_property_helper.set_prefix("stacked_shadow_");
	stacked_shadow_base_property_helper.set_array_length_getter(&LabelSettings::get_stacked_shadow_count);
	stacked_shadow_base_property_helper.register_property(PropertyInfo(Variant::VECTOR2, "offset", PROPERTY_HINT_NONE, "suffix:px"), stacked_shadow_defaults.offset, &LabelSettings::set_stacked_shadow_offset, &LabelSettings::get_stacked_shadow_offset);
	stacked_shadow_base_property_helper.register_property(PropertyInfo(Variant::COLOR, "color"), stacked_shadow_defaults.color, &LabelSettings::set_stacked_shadow_color, &LabelSettings::get_stacked_shadow_color);
	stacked_shadow_base_property_helper.register_property(PropertyInfo(Variant::INT, "outline_size", PROPERTY_HINT_NONE, "0,127,1,or_greater,suffix:px"), stacked_shadow_defaults.outline_size, &LabelSettings::set_stacked_shadow_outline_size, &LabelSettings::get_stacked_shadow_outline_size);
	PropertyListHelper::register_base_helper(&stacked_shadow_base_property_helper);
}

void LabelSettings::set_line_spacing(real_t p_spacing) {
	if (line_spacing != p_spacing) {
		line_spacing = p_spacing;
		emit_changed();
	}
}

real_t LabelSettings::get_line_spacing() const {
	return line_spacing;
}

void LabelSettings::set_paragraph_spacing(real_t p_spacing) {
	if (paragraph_spacing != p_spacing) {
		paragraph_spacing = p_spacing;
		emit_changed();
	}
}

real_t LabelSettings::get_paragraph_spacing() const {
	return paragraph_spacing;
}

void LabelSettings::set_font(const Ref<Font> &p_font) {
	if (font != p_font) {
		if (font.is_valid()) {
			font->disconnect_changed(callable_mp(this, &LabelSettings::_font_changed));
		}
		font = p_font;
		if (font.is_valid()) {
			font->connect_changed(callable_mp(this, &LabelSettings::_font_changed), CONNECT_REFERENCE_COUNTED);
		}
		emit_changed();
	}
}

Ref<Font> LabelSettings::get_font() const {
	return font;
}

void LabelSettings::set_font_size(int p_size) {
	if (font_size != p_size) {
		font_size = p_size;
		emit_changed();
	}
}

int LabelSettings::get_font_size() const {
	return font_size;
}

void LabelSettings::set_font_color(const Color &p_color) {
	if (font_color != p_color) {
		font_color = p_color;
		emit_changed();
	}
}

Color LabelSettings::get_font_color() const {
	return font_color;
}

void LabelSettings::set_outline_size(int p_size) {
	if (outline_size != p_size) {
		outline_size = p_size;
		emit_changed();
	}
}

int LabelSettings::get_outline_size() const {
	return outline_size;
}

void LabelSettings::set_outline_color(const Color &p_color) {
	if (outline_color != p_color) {
		outline_color = p_color;
		emit_changed();
	}
}

Color LabelSettings::get_outline_color() const {
	return outline_color;
}

void LabelSettings::set_shadow_size(int p_size) {
	if (shadow_size != p_size) {
		shadow_size = p_size;
		emit_changed();
	}
}

int LabelSettings::get_shadow_size() const {
	return shadow_size;
}

void LabelSettings::set_shadow_color(const Color &p_color) {
	if (shadow_color != p_color) {
		shadow_color = p_color;
		emit_changed();
	}
}

Color LabelSettings::get_shadow_color() const {
	return shadow_color;
}

void LabelSettings::set_shadow_offset(const Vector2 &p_offset) {
	if (shadow_offset != p_offset) {
		shadow_offset = p_offset;
		emit_changed();
	}
}

Vector2 LabelSettings::get_shadow_offset() const {
	return shadow_offset;
}

Vector<LabelSettings::StackedOutlineData> LabelSettings::get_stacked_outline_data() const {
	return stacked_outline_data;
}

int LabelSettings::get_stacked_outline_count() const {
	return stacked_outline_data.size();
}

void LabelSettings::set_stacked_outline_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	if (stacked_outline_data.size() != p_count) {
		stacked_outline_data.resize(p_count);
		notify_property_list_changed();
		emit_changed();
	}
}

void LabelSettings::add_stacked_outline(int p_index) {
	if (p_index < 0) {
		p_index = stacked_outline_data.size();
	}
	ERR_FAIL_INDEX(p_index, stacked_outline_data.size() + 1);
	stacked_outline_data.insert(p_index, StackedOutlineData());
	notify_property_list_changed();
	emit_changed();
}

void LabelSettings::move_stacked_outline(int p_from_index, int p_to_position) {
	ERR_FAIL_INDEX(p_from_index, stacked_outline_data.size());
	ERR_FAIL_INDEX(p_to_position, stacked_outline_data.size() + 1);
	stacked_outline_data.insert(p_to_position, stacked_outline_data[p_from_index]);
	stacked_outline_data.remove_at(p_to_position < p_from_index ? p_from_index + 1 : p_from_index);
	notify_property_list_changed();
	emit_changed();
}

void LabelSettings::remove_stacked_outline(int p_index) {
	ERR_FAIL_INDEX(p_index, stacked_outline_data.size());
	stacked_outline_data.remove_at(p_index);
	notify_property_list_changed();
	emit_changed();
}

void LabelSettings::set_stacked_outline_size(int p_index, int p_size) {
	ERR_FAIL_INDEX(p_index, stacked_outline_data.size());
	if (stacked_outline_data[p_index].size != p_size) {
		stacked_outline_data.write[p_index].size = p_size;
		emit_changed();
	}
}

int LabelSettings::get_stacked_outline_size(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, stacked_outline_data.size(), 0);
	return stacked_outline_data[p_index].size;
}

void LabelSettings::set_stacked_outline_color(int p_index, const Color &p_color) {
	ERR_FAIL_INDEX(p_index, stacked_outline_data.size());
	if (stacked_outline_data[p_index].color != p_color) {
		stacked_outline_data.write[p_index].color = p_color;
		emit_changed();
	}
}

Color LabelSettings::get_stacked_outline_color(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, stacked_outline_data.size(), Color());
	return stacked_outline_data[p_index].color;
}

Vector<LabelSettings::StackedShadowData> LabelSettings::get_stacked_shadow_data() const {
	return stacked_shadow_data;
}

int LabelSettings::get_stacked_shadow_count() const {
	return stacked_shadow_data.size();
}

void LabelSettings::set_stacked_shadow_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	if (stacked_shadow_data.size() != p_count) {
		stacked_shadow_data.resize(p_count);
		notify_property_list_changed();
		emit_changed();
	}
}

void LabelSettings::add_stacked_shadow(int p_index) {
	if (p_index < 0) {
		p_index = stacked_shadow_data.size();
	}
	ERR_FAIL_INDEX(p_index, stacked_shadow_data.size() + 1);
	stacked_shadow_data.insert(p_index, StackedShadowData());
	notify_property_list_changed();
	emit_changed();
}

void LabelSettings::move_stacked_shadow(int p_from_index, int p_to_position) {
	ERR_FAIL_INDEX(p_from_index, stacked_shadow_data.size());
	ERR_FAIL_INDEX(p_to_position, stacked_shadow_data.size() + 1);
	stacked_shadow_data.insert(p_to_position, stacked_shadow_data[p_from_index]);
	stacked_shadow_data.remove_at(p_to_position < p_from_index ? p_from_index + 1 : p_from_index);
	notify_property_list_changed();
	emit_changed();
}

void LabelSettings::remove_stacked_shadow(int p_index) {
	ERR_FAIL_INDEX(p_index, stacked_shadow_data.size());
	stacked_shadow_data.remove_at(p_index);
	notify_property_list_changed();
	emit_changed();
}

void LabelSettings::set_stacked_shadow_offset(int p_index, const Vector2 &p_offset) {
	ERR_FAIL_INDEX(p_index, stacked_shadow_data.size());
	if (stacked_shadow_data[p_index].offset != p_offset) {
		stacked_shadow_data.write[p_index].offset = p_offset;
		emit_changed();
	}
}

Vector2 LabelSettings::get_stacked_shadow_offset(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, stacked_shadow_data.size(), Vector2());
	return stacked_shadow_data[p_index].offset;
}

void LabelSettings::set_stacked_shadow_color(int p_index, const Color &p_color) {
	ERR_FAIL_INDEX(p_index, stacked_shadow_data.size());
	if (stacked_shadow_data[p_index].color != p_color) {
		stacked_shadow_data.write[p_index].color = p_color;
		emit_changed();
	}
}

Color LabelSettings::get_stacked_shadow_color(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, stacked_shadow_data.size(), Color());
	return stacked_shadow_data[p_index].color;
}

void LabelSettings::set_stacked_shadow_outline_size(int p_index, int p_size) {
	ERR_FAIL_INDEX(p_index, stacked_shadow_data.size());
	if (stacked_shadow_data[p_index].outline_size != p_size) {
		stacked_shadow_data.write[p_index].outline_size = p_size;
		emit_changed();
	}
}

int LabelSettings::get_stacked_shadow_outline_size(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, stacked_shadow_data.size(), 0);
	return stacked_shadow_data[p_index].outline_size;
}
