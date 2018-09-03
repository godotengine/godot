/*************************************************************************/
/*  text_layout.cpp                                                      */
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

#include "text_layout.h"
#include "core/os/os.h"
#include "core_string_names.h"

/*************************************************************************/
/*  TextHitInfo                                                          */
/*************************************************************************/

void TextHitInfo::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_index_start"), &TextHitInfo::get_index_start);
	ClassDB::bind_method(D_METHOD("get_index_end"), &TextHitInfo::get_index_end);
	ClassDB::bind_method(D_METHOD("get_index_insertion"), &TextHitInfo::get_index_insertion);

	ClassDB::bind_method(D_METHOD("get_bounds"), &TextHitInfo::get_bounds);
	ClassDB::bind_method(D_METHOD("get_leading_edge"), &TextHitInfo::get_leading_edge);
	ClassDB::bind_method(D_METHOD("get_trailing_edge"), &TextHitInfo::get_trailing_edge);

	ClassDB::bind_method(D_METHOD("get_type"), &TextHitInfo::get_type);

	BIND_ENUM_CONSTANT(OBJECT_TEXT);
	BIND_ENUM_CONSTANT(OBJECT_IMAGE);
	BIND_ENUM_CONSTANT(OBJECT_SPAN);
	BIND_ENUM_CONSTANT(OBJECT_TABLE);
};

TextLayoutItemType TextHitInfo::get_type() {

	return type;
};

int TextHitInfo::get_index_start() {

	return index_start;
};

int TextHitInfo::get_index_end() {

	return index_end;
};

int TextHitInfo::get_index_insertion() {

	return index_insertion;
};

Rect2 TextHitInfo::get_bounds() {

	return bounds;
};

Point2 TextHitInfo::get_leading_edge() {

	return leading_edge;
};

Point2 TextHitInfo::get_trailing_edge() {

	return trailing_edge;
};

/*************************************************************************/
/*  TextLayout::Script                                                   */
/*************************************************************************/

bool TextLayout::Script::same_script(int32_t p_script_one, int32_t p_script_two) {

	return p_script_one <= USCRIPT_INHERITED || p_script_two <= USCRIPT_INHERITED || p_script_one == p_script_two;
}

bool TextLayout::Script::next() {

	int32_t start_sp = paren_sp;
	UErrorCode error = U_ZERO_ERROR;

	if (script_end >= char_limit) return false;

	script_code = USCRIPT_COMMON;
	for (script_start = script_end; script_end < char_limit; script_end += 1) {
		UChar high = char_array[script_end];
		UChar32 ch = high;

		if (high >= 0xD800 && high <= 0xDBFF && script_end < char_limit - 1) {
			UChar low = char_array[script_end + 1];

			if (low >= 0xDC00 && low <= 0xDFFF) {
				ch = (high - 0xD800) * 0x0400 + low - 0xDC00 + 0x10000;
				script_end += 1;
			}
		}

		UScriptCode sc = uscript_getScript(ch, &error);
		if (U_FAILURE(error)) {
			ERR_EXPLAIN(String(u_errorName(error)));
			ERR_FAIL_COND_V(true, false);
		}
		if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) != U_BPT_NONE) {
			if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_OPEN) {
				paren_stack[++paren_sp].pair_index = ch;
				paren_stack[paren_sp].script_code = script_code;
			} else if (paren_sp >= 0) {
				UChar32 paired_ch = u_getBidiPairedBracket(ch);
				while (paren_sp >= 0 && paren_stack[paren_sp].pair_index != paired_ch)
					paren_sp -= 1;
				if (paren_sp < start_sp) start_sp = paren_sp;
				if (paren_sp >= 0) sc = paren_stack[paren_sp].script_code;
			}
		}

		if (same_script(script_code, sc)) {
			if (script_code <= USCRIPT_INHERITED && sc > USCRIPT_INHERITED) {
				script_code = sc;
				while (start_sp < paren_sp)
					paren_stack[++start_sp].script_code = script_code;
			}
			if ((u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_CLOSE) && paren_sp >= 0) {
				paren_sp -= 1;
				start_sp -= 1;
			}
		} else {
			if (ch >= 0x10000) script_end -= 1;
			break;
		}
	}
	return true;
}

int32_t TextLayout::Script::get_start() {

	return script_start;
}

int32_t TextLayout::Script::get_end() {

	return script_end;
}

hb_script_t TextLayout::Script::get_script() {

	return hb_icu_script_to_script(script_code);
}

void TextLayout::Script::reset() {

	script_start = char_start;
	script_end = char_start;
	script_code = USCRIPT_INVALID_CODE;
	paren_sp = -1;
}

TextLayout::Script::Script(const Vector<UChar> &p_chars, int32_t p_start, int32_t p_length) {

	char_array = (const uint16_t *)p_chars.ptr();
	char_start = p_start;
	char_limit = p_start + p_length;
	reset();
}

/*************************************************************************/
/*  TextLayoutItem                                                       */
/*************************************************************************/

void TextLayoutItem::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_invalidate"), &TextLayoutItem::_invalidate);

	ClassDB::bind_method(D_METHOD("set_align", "align"), &TextLayoutItem::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &TextLayoutItem::get_align);
	ClassDB::bind_method(D_METHOD("set_align_percentage", "align_percentage"), &TextLayoutItem::set_align_percentage);
	ClassDB::bind_method(D_METHOD("get_align_percentage"), &TextLayoutItem::get_align_percentage);
	ClassDB::bind_method(D_METHOD("set_margins", "margins"), &TextLayoutItem::set_margins);
	ClassDB::bind_method(D_METHOD("get_margins"), &TextLayoutItem::get_margins);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Top,Center,Bottom,Percentage"), "set_align", "get_align");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "align_percentage"), "set_align_percentage", "get_align_percentage");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "margins"), "set_margins", "get_margins");
};

void TextLayoutItem::set_align(TextVAlign p_align) {

	if (align != p_align) {
		align = p_align;
		_invalidate();
	}
};

TextVAlign TextLayoutItem::get_align() {

	return align;
};

void TextLayoutItem::set_align_percentage(float p_align_pcnt) {

	if (pcnt != p_align_pcnt) {
		pcnt = p_align_pcnt;
		_invalidate();
	}
};

float TextLayoutItem::get_align_percentage() {

	return pcnt;
};

void TextLayoutItem::set_margins(const Rect2 &p_margins) {

	if (margins != p_margins) {
		margins = p_margins;
		_invalidate();
	}
};

Rect2 TextLayoutItem::get_margins() {

	return margins;
};

bool TextLayoutItem::has(const Ref<TextLayout> &p_layout) {

	return false;
};

void TextLayoutItem::_invalidate() {

	dirty = true;
	emit_changed();
};

void TextLayoutItem::update(){};

/*************************************************************************/
/*  TextLayoutItemText                                                   */
/*************************************************************************/

void TextLayoutItemText::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_string", "string"), &TextLayoutItemText::set_string);
	ClassDB::bind_method(D_METHOD("get_string"), &TextLayoutItemText::get_string);

	ClassDB::bind_method(D_METHOD("set_rise", "rise"), &TextLayoutItemText::set_rise);
	ClassDB::bind_method(D_METHOD("get_rise"), &TextLayoutItemText::get_rise);

	ClassDB::bind_method(D_METHOD("set_fore_color", "color"), &TextLayoutItemText::set_fore_color);
	ClassDB::bind_method(D_METHOD("get_fore_color"), &TextLayoutItemText::get_fore_color);

	ClassDB::bind_method(D_METHOD("set_invis_color", "color"), &TextLayoutItemText::set_invis_color);
	ClassDB::bind_method(D_METHOD("get_invis_color"), &TextLayoutItemText::get_invis_color);

	ClassDB::bind_method(D_METHOD("set_back_color", "color"), &TextLayoutItemText::set_back_color);
	ClassDB::bind_method(D_METHOD("get_back_color"), &TextLayoutItemText::get_back_color);

	ClassDB::bind_method(D_METHOD("set_underline_color", "color"), &TextLayoutItemText::set_underline_color);
	ClassDB::bind_method(D_METHOD("get_underline_color"), &TextLayoutItemText::get_underline_color);

	ClassDB::bind_method(D_METHOD("set_overline_color", "color"), &TextLayoutItemText::set_overline_color);
	ClassDB::bind_method(D_METHOD("get_overline_color"), &TextLayoutItemText::get_overline_color);

	ClassDB::bind_method(D_METHOD("set_strikeout_color", "color"), &TextLayoutItemText::set_strikeout_color);
	ClassDB::bind_method(D_METHOD("get_strikeout_color"), &TextLayoutItemText::get_strikeout_color);

	ClassDB::bind_method(D_METHOD("set_underline_width", "width"), &TextLayoutItemText::set_underline_width);
	ClassDB::bind_method(D_METHOD("get_underline_width"), &TextLayoutItemText::get_underline_width);

	ClassDB::bind_method(D_METHOD("set_overline_width", "width"), &TextLayoutItemText::set_overline_width);
	ClassDB::bind_method(D_METHOD("get_overline_width"), &TextLayoutItemText::get_overline_width);

	ClassDB::bind_method(D_METHOD("set_strikeout_width", "width"), &TextLayoutItemText::set_strikeout_width);
	ClassDB::bind_method(D_METHOD("get_strikeout_width"), &TextLayoutItemText::get_strikeout_width);

	ClassDB::bind_method(D_METHOD("set_font", "font"), &TextLayoutItemText::set_font);
	ClassDB::bind_method(D_METHOD("get_font"), &TextLayoutItemText::get_font);

	ClassDB::bind_method(D_METHOD("set_language", "language"), &TextLayoutItemText::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &TextLayoutItemText::get_language);

	ClassDB::bind_method(D_METHOD("set_features", "features"), &TextLayoutItemText::set_features);
	ClassDB::bind_method(D_METHOD("get_features"), &TextLayoutItemText::get_features);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "string", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_string", "get_string");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "rise"), "set_rise", "get_rise");

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "fore_color"), "set_fore_color", "get_fore_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "invis_color"), "set_invis_color", "get_invis_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "back_color"), "set_back_color", "get_back_color");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "underline_width"), "set_underline_width", "get_underline_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "overline_width"), "set_overline_width", "get_overline_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "strikeout_width"), "set_strikeout_width", "get_strikeout_width");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "underline_color"), "set_underline_color", "get_underline_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "overline_color"), "set_overline_color", "get_overline_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "strikeout_color"), "set_strikeout_color", "get_strikeout_color");

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "features"), "set_features", "get_features");
};

void TextLayoutItemText::set_string(const String &p_string) {

	if (string != p_string) {
		string = p_string;
		_invalidate();
	}
};

String TextLayoutItemText::get_string() {

	return string;
};

void TextLayoutItemText::set_rise(float p_rise) {

	if (rise != p_rise) {
		rise = p_rise;
		_invalidate();
	}
};

float TextLayoutItemText::get_rise() const {

	return rise;
};

void TextLayoutItemText::set_fore_color(const Color &p_color) {

	if (fore_color != p_color) {
		fore_color = p_color;
		_invalidate();
	}
};

Color TextLayoutItemText::get_fore_color() const {

	return fore_color;
};

void TextLayoutItemText::set_invis_color(const Color &p_color) {

	if (fore_color_invis != p_color) {
		fore_color_invis = p_color;
		_invalidate();
	}
};

Color TextLayoutItemText::get_invis_color() const {

	return fore_color_invis;
};

void TextLayoutItemText::set_back_color(const Color &p_color) {

	if (back_color != p_color) {
		back_color = p_color;
		_invalidate();
	}
};

Color TextLayoutItemText::get_back_color() const {

	return back_color;
};

void TextLayoutItemText::set_underline_color(const Color &p_color) {

	if (underline_color != p_color) {
		underline_color = p_color;
		_invalidate();
	}
};

Color TextLayoutItemText::get_underline_color() const {

	return underline_color;
};

void TextLayoutItemText::set_overline_color(const Color &p_color) {

	if (overline_color != p_color) {
		overline_color = p_color;
		_invalidate();
	}
};

Color TextLayoutItemText::get_overline_color() const {

	return overline_color;
};

void TextLayoutItemText::set_strikeout_color(const Color &p_color) {

	if (strikeout_color != p_color) {
		strikeout_color = p_color;
		_invalidate();
	}
};

Color TextLayoutItemText::get_strikeout_color() const {

	return strikeout_color;
};

void TextLayoutItemText::set_underline_width(int p_width) {

	if (underline_width != p_width) {
		underline_width = p_width;
		_invalidate();
	}
};

int TextLayoutItemText::get_underline_width() const {

	return underline_width;
};

void TextLayoutItemText::set_overline_width(int p_width) {

	if (overline_width != p_width) {
		overline_width = p_width;
		_invalidate();
	}
};

int TextLayoutItemText::get_overline_width() const {

	return overline_width;
};

void TextLayoutItemText::set_strikeout_width(int p_width) {

	if (strikeout_width != p_width) {
		strikeout_width = p_width;
		_invalidate();
	}
};

int TextLayoutItemText::get_strikeout_width() const {

	return strikeout_width;
};

void TextLayoutItemText::set_font(Ref<Font> p_font) {

	if (font != p_font) {
		font = p_font;
		_invalidate();
	}
};

Ref<Font> TextLayoutItemText::get_font() const {

	return font;
}

void TextLayoutItemText::set_language(const String &p_language) {

	if (language != p_language) {
		language = p_language;
		h_language = hb_language_from_string(language.ascii().get_data(), -1);
		_invalidate();
	}
};

String TextLayoutItemText::get_language() const {

	return language;
};

void TextLayoutItemText::set_features(const String &p_features) {

	if (features != p_features) {
		features = p_features;

		h_features.clear();
		Vector<String> v_features = features.split(",");
		for (int i = 0; i < v_features.size(); i++) {
			hb_feature_t feature;
			if (hb_feature_from_string(v_features[i].ascii().get_data(), -1, &feature)) h_features.push_back(feature);
		}
		_invalidate();
	}
};

String TextLayoutItemText::get_features() const {

	return features;
}

void TextLayoutItemImage::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_image", "image"), &TextLayoutItemImage::set_image);
	ClassDB::bind_method(D_METHOD("get_image"), &TextLayoutItemImage::get_image);
	ClassDB::bind_method(D_METHOD("set_modulate", "modulate"), &TextLayoutItemImage::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &TextLayoutItemImage::get_modulate);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "image", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_image", "get_image");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), "set_modulate", "get_modulate");
};

void TextLayoutItemImage::set_image(const Ref<Texture> &p_image) {

	if (image != p_image) {
		image = p_image;
		_invalidate();
	}
};

Ref<Texture> TextLayoutItemImage::get_image() {

	return image;
};

void TextLayoutItemImage::set_modulate(Color p_modulate) {

	if (modulate != p_modulate) {
		modulate = p_modulate;
		_invalidate();
	}
};

Color TextLayoutItemImage::get_modulate() {

	return modulate;
};

Size2 TextLayoutItemImage::get_size() {

	update();
	if (image.is_null())
		return Size2(0, 0);

	return Size2(image->get_width(), image->get_height()) + margins.size + margins.position;
};

void TextLayoutItemImage::draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip) {

	update();
	if (!image.is_null())
		image->draw(p_canvas_item, p_pos + margins.position, modulate);
};

/*************************************************************************/
/*  TextLayoutSpan                                                       */
/*************************************************************************/

void TextLayoutItemSpan::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_layout", "layout"), &TextLayoutItemSpan::set_layout);
	ClassDB::bind_method(D_METHOD("get_layout"), &TextLayoutItemSpan::get_layout);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "layout", PROPERTY_HINT_RESOURCE_TYPE, "TextLayout"), "set_layout", "get_layout");
};

void TextLayoutItemSpan::set_layout(const Ref<TextLayout> &p_layout) {

	if (layout != p_layout) {
		if (!layout.is_null()) layout->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
		layout = p_layout;
		if (!layout.is_null()) layout->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
		_invalidate();
	}
};

Ref<TextLayout> TextLayoutItemSpan::get_layout() {

	return layout;
};

bool TextLayoutItemSpan::has(const Ref<TextLayout> &p_layout) {

	return (p_layout == layout);
};

Size2 TextLayoutItemSpan::get_size() {

	update();
	return layout->get_bounds().size + margins.size + margins.position;
};

void TextLayoutItemSpan::draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip) {

	update();
	if (!layout.is_null()) {
		layout->set_clip_rect(p_clip);
		layout->draw(p_canvas_item, p_pos + margins.position);
	}
};

/*************************************************************************/
/*  TextLayoutTable                                                      */
/*************************************************************************/

bool TextLayoutItemTable::_set(const StringName &p_name, const Variant &p_value) {

	String str = p_name;
	if (str.begins_with("cells/")) {
		int idx = str.get_slicec('/', 1).to_int();
		Ref<TextLayout> fd = p_value;

		if (fd.is_valid()) {
			if (idx == cells.size()) {
				add_cell(fd);
				return true;
			} else if (idx >= 0 && idx < cells.size()) {
				set_cell(idx, fd);
				return true;
			} else {
				return false;
			}
		} else if (idx >= 0 && idx < cells.size()) {
			remove_cell(idx);
			return true;
		}
	}

	return false;
}

bool TextLayoutItemTable::_get(const StringName &p_name, Variant &r_ret) const {

	String str = p_name;
	if (str.begins_with("cells/")) {
		int idx = str.get_slicec('/', 1).to_int();

		if (idx == cells.size()) {
			r_ret = Ref<TextLayout>();
			return true;
		} else if (idx >= 0 && idx < cells.size()) {
			r_ret = get_cell(idx);
			return true;
		}
	}

	return false;
};

void TextLayoutItemTable::_get_property_list(List<PropertyInfo> *p_list) const {

	for (int i = 0; i < cells.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "cells/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "TextLayout"));
	}

	p_list->push_back(PropertyInfo(Variant::OBJECT, "cells/" + itos(cells.size()), PROPERTY_HINT_RESOURCE_TYPE, "TextLayout"));
};

void TextLayoutItemTable::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_column_count", "columns"), &TextLayoutItemTable::set_column_count);
	ClassDB::bind_method(D_METHOD("get_column_count"), &TextLayoutItemTable::get_column_count);

	ClassDB::bind_method(D_METHOD("set_vseparation", "vseparation"), &TextLayoutItemTable::set_vseparation);
	ClassDB::bind_method(D_METHOD("get_vseparation"), &TextLayoutItemTable::get_vseparation);

	ClassDB::bind_method(D_METHOD("set_hseparation", "hseparation"), &TextLayoutItemTable::set_hseparation);
	ClassDB::bind_method(D_METHOD("get_hseparation"), &TextLayoutItemTable::get_hseparation);

	ClassDB::bind_method(D_METHOD("get_cell", "idx"), &TextLayoutItemTable::get_cell);
	ClassDB::bind_method(D_METHOD("get_cell_count"), &TextLayoutItemTable::get_cell_count);
	ClassDB::bind_method(D_METHOD("add_cell", "layout"), &TextLayoutItemTable::add_cell);
	ClassDB::bind_method(D_METHOD("set_cell", "idx", "layout"), &TextLayoutItemTable::set_cell);
	ClassDB::bind_method(D_METHOD("insert_cell", "idx", "layout"), &TextLayoutItemTable::insert_cell);
	ClassDB::bind_method(D_METHOD("remove_cell", "idx"), &TextLayoutItemTable::remove_cell);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "column_count"), "set_column_count", "get_column_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vseparation"), "set_vseparation", "get_vseparation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hseparation"), "set_hseparation", "get_hseparation");
};

void TextLayoutItemTable::set_vseparation(int p_vseparation) {

	if (vseparation != p_vseparation) {
		vseparation = p_vseparation;
		_invalidate();
	}
};

int TextLayoutItemTable::get_vseparation() {

	return vseparation;
};

void TextLayoutItemTable::set_hseparation(int p_hseparation) {

	if (hseparation != p_hseparation) {
		hseparation = p_hseparation;
		_invalidate();
	}
};

int TextLayoutItemTable::get_hseparation() {

	return hseparation;
};

void TextLayoutItemTable::set_column_count(int p_columns) {

	if (columns.size() != p_columns) {
		columns.resize(p_columns);
		_invalidate();
	}
};

int TextLayoutItemTable::get_column_count() {

	return columns.size();
};

Ref<TextLayout> TextLayoutItemTable::get_cell(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, cells.size(), Ref<TextLayout>());

	return cells[p_idx];
};

int TextLayoutItemTable::get_cell_count() const {

	return cells.size();
};

void TextLayoutItemTable::add_cell(const Ref<TextLayout> &p_layout) {

	ERR_FAIL_COND(p_layout.is_null());
	cells.push_back(p_layout);
	cells[cells.size() - 1]->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	_invalidate();
};

void TextLayoutItemTable::insert_cell(int p_idx, const Ref<TextLayout> &p_layout) {

	ERR_FAIL_COND(p_layout.is_null());
	ERR_FAIL_INDEX(p_idx, cells.size());
	cells.insert(p_idx, p_layout);
	cells[p_idx]->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	_invalidate();
};

void TextLayoutItemTable::set_cell(int p_idx, const Ref<TextLayout> &p_layout) {

	ERR_FAIL_COND(p_layout.is_null());
	ERR_FAIL_INDEX(p_idx, cells.size());
	cells[p_idx]->disconnect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	cells[p_idx] = p_layout;
	cells[p_idx]->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	_invalidate();
};

void TextLayoutItemTable::remove_cell(int p_idx) {

	ERR_FAIL_INDEX(p_idx, cells.size());
	cells[p_idx]->disconnect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	cells.remove(p_idx);
	_invalidate();
};

bool TextLayoutItemTable::has(const Ref<TextLayout> &p_layout) {

	for (int i = 0; i < cells.size(); i++) {
		if (p_layout == cells[i])
			return true;
	};
	return false;
};

void TextLayoutItemTable::update() {

	if (dirty) {
		size = Size2(0, 0);
		for (int i = 0; i < columns.size(); i++) {
			columns[i] = 0;
		}

		rows.clear();
		rows.push_back(0);
		int r = 0;
		for (int i = 0; i < cells.size(); i++) {
			int c = i % columns.size();
			if (!cells[i].is_null()) {
				columns[c] = MAX(columns[c], cells[i]->get_bounds().size.x);
				rows[r] = MAX(rows[r], cells[i]->get_bounds().size.y);
			}
			if (c == columns.size() - 1) {
				size.y += rows[r] + vseparation;
				rows.push_back(0);
				r++;
			}
		}

		for (int i = 0; i < columns.size(); i++) {
			size.x += columns[i] + hseparation;
		}

		size += margins.size + margins.position;

		dirty = false;
	}
};

Size2 TextLayoutItemTable::get_size() {

	update();
	return size;
};

Ref<TextLayout> TextLayoutItemTable::hit_test(const Point2 &p_point) {

	update();

	Point2 offset = Point2(0, 0);
	int r = 0;
	for (int i = 0; i < cells.size(); i++) {
		int c = i % columns.size();
		if (!cells[i].is_null()) {
			Point2 cell_offset = Point2(0, 0);
			switch (cells[i]->get_parent_valign()) {
				case V_ALIGN_TOP: {
					cell_offset.y = 0;
				} break;
				case V_ALIGN_CENTER: {
					cell_offset.y = (rows[r] - cells[i]->get_bounds().size.y) / 2;
				} break;
				case V_ALIGN_BOTTOM: {
					cell_offset.y = (rows[r] - cells[i]->get_bounds().size.y);
				} break;
			}
			switch (cells[i]->get_parent_halign()) {
				case H_ALIGN_LEFT: {
					cell_offset.x = 0;
				} break;
				case H_ALIGN_CENTER: {
					cell_offset.x = (columns[c] - cells[i]->get_bounds().size.x) / 2;
				} break;
				case H_ALIGN_RIGHT: {
					cell_offset.x = (columns[c] - cells[i]->get_bounds().size.x);
				} break;
			}
			if ((p_point.x >= (margins.position.x + offset.x + cell_offset.x)) && (p_point.x <= (margins.position.x + offset.x + cell_offset.x + cells[i]->get_bounds().size.x)) && (p_point.y >= (margins.position.y + offset.y + cell_offset.y)) && (p_point.y <= (margins.position.y + offset.y + cell_offset.y + cells[i]->get_bounds().size.y))) {
				return cells[i];
			}

			offset.x += columns[c] + hseparation;
		}
		if (c == columns.size() - 1) {
			offset.y += rows[r] + vseparation;
			offset.x = 0;
			r++;
		}
	}

	return NULL;
};

void TextLayoutItemTable::draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip) {

	update();

	Point2 offset = Point2(0, 0);
	int r = 0;
	for (int i = 0; i < cells.size(); i++) {
		int c = i % columns.size();
		if (!cells[i].is_null()) {
			Point2 cell_offset = Point2(0, 0);
			switch (cells[i]->get_parent_valign()) {
				case V_ALIGN_TOP: {
					cell_offset.y = 0;
				} break;
				case V_ALIGN_CENTER: {
					cell_offset.y = (rows[r] - cells[i]->get_bounds().size.y) / 2;
				} break;
				case V_ALIGN_BOTTOM: {
					cell_offset.y = (rows[r] - cells[i]->get_bounds().size.y);
				} break;
			}
			switch (cells[i]->get_parent_halign()) {
				case H_ALIGN_LEFT: {
					cell_offset.x = 0;
				} break;
				case H_ALIGN_CENTER: {
					cell_offset.x = (columns[c] - cells[i]->get_bounds().size.x) / 2;
				} break;
				case H_ALIGN_RIGHT: {
					cell_offset.x = (columns[c] - cells[i]->get_bounds().size.x);
				} break;
			}
			cells[i]->set_clip_rect(p_clip);
			cells[i]->draw(p_canvas_item, p_pos + margins.position + offset + cell_offset);
		}
		offset.x += columns[c] + hseparation;
		if (c == columns.size() - 1) {
			offset.y += rows[r] + vseparation;
			offset.x = 0;
			r++;
		}
	}
};

/*************************************************************************/
/*  TextLayout                                                           */
/*************************************************************************/

bool TextLayout::_set(const StringName &p_name, const Variant &p_value) {

	String str = p_name;
	if (str.begins_with("items/")) {
		int idx = str.get_slicec('/', 1).to_int();
		Ref<TextLayoutItem> fd = p_value;

		if (fd.is_valid()) {
			if (idx == items.size()) {
				add_item(fd);
				return true;
			} else if (idx >= 0 && idx < items.size()) {
				set_item(idx, fd);
				return true;
			} else {
				return false;
			}
		} else if (idx >= 0 && idx < items.size()) {
			remove_item(idx);
			return true;
		}
	}
	return false;
}

bool TextLayout::_get(const StringName &p_name, Variant &r_ret) const {

	String str = p_name;
	if (str.begins_with("items/")) {
		int idx = str.get_slicec('/', 1).to_int();

		if (idx == items.size()) {
			r_ret = Ref<TextLayoutItem>();
			return true;
		} else if (idx >= 0 && idx < items.size()) {
			r_ret = get_item(idx);
			return true;
		}
	}
	return false;
}

void TextLayout::_get_property_list(List<PropertyInfo> *p_list) const {

	for (int i = 0; i < items.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "items/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "TextLayoutItem"));
	}

	p_list->push_back(PropertyInfo(Variant::OBJECT, "items/" + itos(items.size()), PROPERTY_HINT_RESOURCE_TYPE, "TextLayoutItem"));
}

void TextLayout::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_invalidate"), &TextLayout::_invalidate);

	ClassDB::bind_method(D_METHOD("add_item", "data"), &TextLayout::add_item);
	ClassDB::bind_method(D_METHOD("set_item", "idx", "data"), &TextLayout::set_item);
	ClassDB::bind_method(D_METHOD("insert_item", "idx", "data"), &TextLayout::insert_item);
	ClassDB::bind_method(D_METHOD("get_item", "idx"), &TextLayout::get_item);
	ClassDB::bind_method(D_METHOD("remove_item", "idx"), &TextLayout::remove_item);
	ClassDB::bind_method(D_METHOD("get_item_count"), &TextLayout::get_item_count);

	ClassDB::bind_method(D_METHOD("set_base_direction", "direction"), &TextLayout::set_base_direction);
	ClassDB::bind_method(D_METHOD("get_base_direction"), &TextLayout::get_base_direction);

	ClassDB::bind_method(D_METHOD("set_clip_rect", "clip_rect"), &TextLayout::set_clip_rect);
	ClassDB::bind_method(D_METHOD("get_clip_rect"), &TextLayout::get_clip_rect);

	ClassDB::bind_method(D_METHOD("set_border_left_color", "border_left_color"), &TextLayout::set_border_left_color);
	ClassDB::bind_method(D_METHOD("get_border_left_color"), &TextLayout::get_border_left_color);

	ClassDB::bind_method(D_METHOD("set_border_top_color", "border_top_color"), &TextLayout::set_border_top_color);
	ClassDB::bind_method(D_METHOD("get_border_top_color"), &TextLayout::get_border_top_color);

	ClassDB::bind_method(D_METHOD("set_border_right_color", "border_right_color"), &TextLayout::set_border_right_color);
	ClassDB::bind_method(D_METHOD("get_border_right_color"), &TextLayout::get_border_right_color);

	ClassDB::bind_method(D_METHOD("set_border_bottom_color", "border_bottom_color"), &TextLayout::set_border_bottom_color);
	ClassDB::bind_method(D_METHOD("get_border_bottom_color"), &TextLayout::get_border_bottom_color);

	ClassDB::bind_method(D_METHOD("set_border_left_width", "border_left_width"), &TextLayout::set_border_left_width);
	ClassDB::bind_method(D_METHOD("get_border_left_width"), &TextLayout::get_border_left_width);

	ClassDB::bind_method(D_METHOD("set_border_top_width", "border_top_width"), &TextLayout::set_border_top_width);
	ClassDB::bind_method(D_METHOD("get_border_top_width"), &TextLayout::get_border_top_width);

	ClassDB::bind_method(D_METHOD("set_border_right_width", "border_right_width"), &TextLayout::set_border_right_width);
	ClassDB::bind_method(D_METHOD("get_border_right_width"), &TextLayout::get_border_right_width);

	ClassDB::bind_method(D_METHOD("set_border_bottom_width", "border_bottom_width"), &TextLayout::set_border_bottom_width);
	ClassDB::bind_method(D_METHOD("get_border_bottom_width"), &TextLayout::get_border_bottom_width);

	ClassDB::bind_method(D_METHOD("set_back_color", "back_color"), &TextLayout::set_back_color);
	ClassDB::bind_method(D_METHOD("get_back_color"), &TextLayout::get_back_color);

	ClassDB::bind_method(D_METHOD("set_padding", "padding"), &TextLayout::set_padding);
	ClassDB::bind_method(D_METHOD("get_padding"), &TextLayout::get_padding);

	ClassDB::bind_method(D_METHOD("set_max_area", "area"), &TextLayout::set_max_area);
	ClassDB::bind_method(D_METHOD("get_max_area"), &TextLayout::get_max_area);

	ClassDB::bind_method(D_METHOD("set_min_area", "area"), &TextLayout::set_min_area);
	ClassDB::bind_method(D_METHOD("get_min_area"), &TextLayout::get_min_area);

	ClassDB::bind_method(D_METHOD("set_line_spacing", "spacing"), &TextLayout::set_line_spacing);
	ClassDB::bind_method(D_METHOD("get_line_spacing"), &TextLayout::get_line_spacing);

	ClassDB::bind_method(D_METHOD("set_autowrap", "autowrap"), &TextLayout::set_autowrap);
	ClassDB::bind_method(D_METHOD("get_autowrap"), &TextLayout::get_autowrap);

	ClassDB::bind_method(D_METHOD("set_hard_breaks", "hard_breaks"), &TextLayout::set_hard_breaks);
	ClassDB::bind_method(D_METHOD("get_hard_breaks"), &TextLayout::get_hard_breaks);

	ClassDB::bind_method(D_METHOD("set_valign", "valign"), &TextLayout::set_valign);
	ClassDB::bind_method(D_METHOD("get_valign"), &TextLayout::get_valign);

	ClassDB::bind_method(D_METHOD("set_align", "align"), &TextLayout::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &TextLayout::get_align);

	ClassDB::bind_method(D_METHOD("set_parent_valign", "parent_valign"), &TextLayout::set_parent_valign);
	ClassDB::bind_method(D_METHOD("get_parent_valign"), &TextLayout::get_parent_valign);

	ClassDB::bind_method(D_METHOD("set_parent_halign", "parent_halign"), &TextLayout::set_parent_halign);
	ClassDB::bind_method(D_METHOD("get_parent_halign"), &TextLayout::get_parent_halign);

	ClassDB::bind_method(D_METHOD("set_align_last", "align"), &TextLayout::set_align_last);
	ClassDB::bind_method(D_METHOD("get_align_last"), &TextLayout::get_align_last);

	ClassDB::bind_method(D_METHOD("set_tab_stops", "tab_stops"), &TextLayout::set_tab_stops);
	ClassDB::bind_method(D_METHOD("get_tab_stops"), &TextLayout::get_tab_stops);

	ClassDB::bind_method(D_METHOD("set_show_invisible_characters", "enable"), &TextLayout::set_show_invisible_characters);
	ClassDB::bind_method(D_METHOD("get_show_invisible_characters"), &TextLayout::get_show_invisible_characters);

	ClassDB::bind_method(D_METHOD("set_show_control_characters", "enable"), &TextLayout::set_show_control_characters);
	ClassDB::bind_method(D_METHOD("get_show_control_characters"), &TextLayout::get_show_control_characters);

	ClassDB::bind_method(D_METHOD("set_invisible_characters", "placeholders"), &TextLayout::set_invisible_characters);
	ClassDB::bind_method(D_METHOD("get_invisible_characters"), &TextLayout::get_invisible_characters);

	ClassDB::bind_method(D_METHOD("set_kasida_to_space_ratio", "ratio"), &TextLayout::set_kasida_to_space_ratio);
	ClassDB::bind_method(D_METHOD("get_kasida_to_space_ratio"), &TextLayout::get_kasida_to_space_ratio);

	ClassDB::bind_method(D_METHOD("set_enable_shaping", "enable"), &TextLayout::set_enable_shaping);
	ClassDB::bind_method(D_METHOD("get_enable_shaping"), &TextLayout::get_enable_shaping);

	ClassDB::bind_method(D_METHOD("set_enable_bidi", "enable"), &TextLayout::set_enable_bidi);
	ClassDB::bind_method(D_METHOD("get_enable_bidi"), &TextLayout::get_enable_bidi);

	ClassDB::bind_method(D_METHOD("set_enable_fallback", "enable"), &TextLayout::set_enable_fallback);
	ClassDB::bind_method(D_METHOD("get_enable_fallback"), &TextLayout::get_enable_fallback);

	ClassDB::bind_method(D_METHOD("set_enable_kasida_justification", "enable"), &TextLayout::set_enable_kasida_justification);
	ClassDB::bind_method(D_METHOD("get_enable_kasida_justification"), &TextLayout::get_enable_kasida_justification);

	ClassDB::bind_method(D_METHOD("set_enable_interword_justification", "enable"), &TextLayout::set_enable_interword_justification);
	ClassDB::bind_method(D_METHOD("get_enable_interword_justification"), &TextLayout::get_enable_interword_justification);

	ClassDB::bind_method(D_METHOD("set_enable_intercluster_justification", "enable"), &TextLayout::set_enable_intercluster_justification);
	ClassDB::bind_method(D_METHOD("get_enable_intercluster_justification"), &TextLayout::get_enable_intercluster_justification);

	ClassDB::bind_method(D_METHOD("set_enable_fallback_line_break", "enable"), &TextLayout::set_enable_fallback_line_break);
	ClassDB::bind_method(D_METHOD("get_enable_fallback_line_break"), &TextLayout::get_enable_fallback_line_break);

	ClassDB::bind_method(D_METHOD("get_bounds"), &TextLayout::get_bounds);

	ClassDB::bind_method(D_METHOD("get_line_count"), &TextLayout::get_line_count);
	ClassDB::bind_method(D_METHOD("get_line_bounds", "line"), &TextLayout::get_line_bounds);
	ClassDB::bind_method(D_METHOD("get_line_start", "line"), &TextLayout::get_line_start);
	ClassDB::bind_method(D_METHOD("get_line_end", "line"), &TextLayout::get_line_end);

	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "position"), &TextLayout::draw);

	ClassDB::bind_method(D_METHOD("hit_test", "point"), &TextLayout::hit_test);
	ClassDB::bind_method(D_METHOD("hit_test_layout", "point"), &TextLayout::hit_test_layout);

	ClassDB::bind_method(D_METHOD("has", "layout"), &TextLayout::has);

	ClassDB::bind_method(D_METHOD("highlight_shapes_hit", "first_hit", "second_hit", "selection"), &TextLayout::highlight_shapes_hit, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("highlight_shapes_char", "start", "end"), &TextLayout::highlight_shapes_char);

	ClassDB::bind_method(D_METHOD("caret_shapes_hit", "hit"), &TextLayout::caret_shapes_hit);
	ClassDB::bind_method(D_METHOD("caret_shapes_char", "index"), &TextLayout::caret_shapes_char);

	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "clip_rect"), "set_clip_rect", "get_clip_rect");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "base_direction", PROPERTY_HINT_ENUM, "LTR,RTL,Auto"), "set_base_direction", "get_base_direction");

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "max_area"), "set_max_area", "get_max_area");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "min_area"), "set_min_area", "get_min_area");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "padding"), "set_padding", "get_padding");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_left_width"), "set_border_left_width", "get_border_left_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_top_width"), "set_border_top_width", "get_border_top_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_right_width"), "set_border_right_width", "get_border_right_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "border_bottom_width"), "set_border_bottom_width", "get_border_bottom_width");

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_left_color"), "set_border_left_color", "get_border_left_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_top_color"), "set_border_top_color", "get_border_top_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_right_color"), "set_border_right_color", "get_border_right_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_bottom_color"), "set_border_bottom_color", "get_border_bottom_color");

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "back_color"), "set_back_color", "get_back_color");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "line_spacing"), "set_line_spacing", "get_line_spacing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autowrap"), "set_autowrap", "get_autowrap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hard_breaks"), "set_hard_breaks", "get_hard_breaks");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "valign", PROPERTY_HINT_ENUM, "Top,Center,Bottom"), "set_valign", "get_valign");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Start,End,Justify"), "set_align", "get_align");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align_last", PROPERTY_HINT_ENUM, "Left,Center,Right,Start,End,Justify"), "set_align_last", "get_align_last");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "parent_valign", PROPERTY_HINT_ENUM, "Top,Center,Bottom"), "set_parent_valign", "get_parent_valign");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "parent_halign", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_parent_halign", "get_parent_halign");

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "tab_stops", PROPERTY_HINT_NONE, "", 0), "set_tab_stops", "get_tab_stops");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_invisible_characters"), "set_show_invisible_characters", "get_show_invisible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_control_characters"), "set_show_control_characters", "get_show_control_characters");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "invisible_characters"), "set_invisible_characters", "get_invisible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "kasida_to_space_ratio", PROPERTY_HINT_RANGE, "0.0,1.0,1.0"), "set_kasida_to_space_ratio", "get_kasida_to_space_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_shaping"), "set_enable_shaping", "get_enable_shaping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_bidi"), "set_enable_bidi", "get_enable_bidi");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_fallback"), "set_enable_fallback", "get_enable_fallback");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_kasida_justification"), "set_enable_kasida_justification", "get_enable_kasida_justification");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_interword_justification"), "set_enable_interword_justification", "get_enable_interword_justification");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_intercluster_justification"), "set_enable_intercluster_justification", "get_enable_intercluster_justification");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_fallback_line_break"), "set_enable_fallback_line_break", "get_enable_fallback_line_break");

	BIND_ENUM_CONSTANT(H_ALIGN_LEFT);
	BIND_ENUM_CONSTANT(H_ALIGN_CENTER);
	BIND_ENUM_CONSTANT(H_ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(H_ALIGN_START);
	BIND_ENUM_CONSTANT(H_ALIGN_END);
	BIND_ENUM_CONSTANT(H_ALIGN_FILL);

	BIND_ENUM_CONSTANT(V_ALIGN_TOP);
	BIND_ENUM_CONSTANT(V_ALIGN_CENTER);
	BIND_ENUM_CONSTANT(V_ALIGN_BOTTOM);
	BIND_ENUM_CONSTANT(V_ALIGN_PERCENTAGE);

	BIND_ENUM_CONSTANT(DIR_LTR);
	BIND_ENUM_CONSTANT(DIR_RTL);
	BIND_ENUM_CONSTANT(DIR_AUTO);
};

void TextLayout::_invalidate() {

	invalidate(INVALIDATE_ALL);
};

void TextLayout::invalidate(TextInvalidateType p_level) {

	if (p_level >= INVALIDATE_LAYOUT) {
		_dirty_layout = true;
		first_logical = NULL;
		first_visual = NULL;
		last_logical = NULL;
		last_visual = NULL;
	}
	if (p_level >= INVALIDATE_LINES) _dirty_lines = true;
	if (p_level >= INVALIDATE_RUNS) _dirty_text_runs = true;
	if (p_level >= INVALIDATE_ALL) _dirty_text_boundaries = true;

	emit_changed();
};

bool TextLayout::has(const Ref<TextLayout> &p_layout) const {

	for (int i = 0; i < items.size(); i++) {
		if (items[i]->type == OBJECT_TABLE) {
			const TextLayoutItemTable *table = static_cast<const TextLayoutItemTable *>(items[i].ptr());
			for (int i = 0; i < table->cells.size(); i++) {
				if (table->cells[i] == p_layout) return true;
			}
		} else if (items[i]->type == OBJECT_SPAN) {
			const TextLayoutItemSpan *span = static_cast<const TextLayoutItemSpan *>(items[i].ptr());
			if (span->layout == p_layout) return true;
		}
	}
	return false;
};

void TextLayout::clear_cache() {

	_offset = 0;

	utf16_text.clear();
	_style_runs.clear();

	for (int i = 0; i < _lines.size(); i++) {
		memdelete(_lines[i]);
	}
	_lines.clear();

	for (int i = 0; i < _runs.size(); i++) {
		memdelete(_runs[i]);
	};
	_runs.clear();
	_runs_logical.clear();

	_justification_opportunies.clear();
	_break_opportunies.clear();

	first_logical = NULL;
	first_visual = NULL;
	last_logical = NULL;
	last_visual = NULL;
};

void TextLayout::add_item(const Ref<TextLayoutItem> &p_data) {

	ERR_FAIL_COND(p_data.is_null());
	items.push_back(p_data);
	items[items.size() - 1]->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	invalidate(INVALIDATE_ALL);
};

void TextLayout::insert_item(int p_idx, const Ref<TextLayoutItem> &p_data) {

	ERR_FAIL_COND(p_data.is_null());
	ERR_FAIL_INDEX(p_idx, items.size());
	items.insert(p_idx, p_data);
	items[p_idx]->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	invalidate(INVALIDATE_ALL);
};

void TextLayout::set_item(int p_idx, const Ref<TextLayoutItem> &p_data) {

	ERR_FAIL_COND(p_data.is_null());
	ERR_FAIL_INDEX(p_idx, items.size());
	items[p_idx]->disconnect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	items[p_idx] = p_data;
	items[p_idx]->connect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	invalidate(INVALIDATE_ALL);
};

int TextLayout::get_item_count() const {

	return items.size();
};

Ref<TextLayoutItem> TextLayout::get_item(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<TextLayoutItem>());

	return items[p_idx];
};

void TextLayout::remove_item(int p_idx) {

	ERR_FAIL_INDEX(p_idx, items.size());
	items[p_idx]->disconnect(CoreStringNames::get_singleton()->changed, this, "_invalidate");
	items.remove(p_idx);
	invalidate(INVALIDATE_ALL);
};

void TextLayout::generate_kashida_justification_opportunies(size_t p_start, size_t p_end) {

	int32_t kashida_pos = -1;
	int8_t priority = 100;
	int32_t i = p_start;

	uint32_t c, pc = 0;

	while ((p_end > p_start) && is_transparent(utf16_text[p_end - 1]))
		p_end--;

	while (i < p_end) {
		c = utf16_text[i];

		if (c == 0x0640) {
			kashida_pos = i;
			priority = 0;
		}
		if (priority >= 1 && i < p_end - 1) {
			if (is_seen_sad(c) && (utf16_text[i + 1] != 0x200C)) {
				kashida_pos = i;
				priority = 1;
			}
		}
		if (priority >= 2 && i > p_start) {
			if (is_teh_marbuta(c) || is_dal(c) || (is_heh(c) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 2;
				}
			}
		}
		if (priority >= 3 && i > p_start) {
			if (is_alef(c) || ((is_lam(c) || is_tah(c) || is_kaf(c) || is_gaf(c)) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 3;
				}
			}
		}
		if (priority >= 4 && i > p_start && i < p_end - 1) {
			if (is_beh(c)) {
				if (is_reh(utf16_text[i + 1]) || is_yeh(utf16_text[i + 1])) {
					if (is_connected_to_prev(c, pc)) {
						kashida_pos = i - 1;
						priority = 4;
					}
				}
			}
		}
		if (priority >= 5 && i > p_start) {
			if (is_waw(c) || ((is_ain(c) || is_qaf(c) || is_feh(c)) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 5;
				}
			}
		}
		if (priority >= 6 && i > p_start) {
			if (is_reh(c)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 6;
				}
			}
		}
		if (!is_transparent(c)) pc = c;
		i++;
	}
	if (kashida_pos > -1) _justification_opportunies.push_back(JustificationOpportunity(kashida_pos, JUSTIFICATION_KASHIDA));
};

void TextLayout::generate_break_opportunies(int32_t p_start, int32_t p_end, const String &p_lang) {

	UErrorCode err = U_ZERO_ERROR;
	UBreakIterator *bi = ubrk_open(UBRK_LINE, p_lang.ascii().get_data(), (const uint16_t *)&utf16_text[p_start], p_end - p_start, &err);
	if (U_SUCCESS(err)) {
		while (ubrk_next(bi) != UBRK_DONE) {
			_break_opportunies.push_back(BreakOpportunity(p_start + ubrk_current(bi), (ubrk_getRuleStatus(bi) == UBRK_LINE_HARD) ? BREAK_HARD : BREAK_LINE));
		}
	} else {
		if (err == U_MISSING_RESOURCE_ERROR) {
			//use fallback
			if (show_no_icu_data_warning) {
				WARN_PRINTS(TTR("ICU break iterator data is missing, using fallback whitespace line breaking and justification"));
				show_no_icu_data_warning = false;
			}
			for (int i = p_start; i < p_end; i++) {
				if ((utf16_text[i] == 0x000A) || (utf16_text[i] == 0x000D)) {
					_break_opportunies.push_back(BreakOpportunity(i + 1, BREAK_HARD));
				} else if (u_isWhitespace(utf16_text[i])) {
					_break_opportunies.push_back(BreakOpportunity(i + 1, BREAK_LINE));
				}
			}
		} else {
			ERR_EXPLAIN(String(u_errorName(err)));
			ERR_FAIL_COND(true);
		}
	}
	ubrk_close(bi);
};

void TextLayout::generate_justification_opportunies(int32_t p_start, int32_t p_end, const String &p_lang) {

	UErrorCode err = U_ZERO_ERROR;
	UBreakIterator *bi = ubrk_open(UBRK_WORD, p_lang.ascii().get_data(), (const uint16_t *)&utf16_text[p_start], p_end - p_start, &err);
	if (U_SUCCESS(err)) {
		int limit = 0;
		while (ubrk_next(bi) != UBRK_DONE) {
			if (ubrk_getRuleStatus(bi) != UBRK_WORD_NONE) {
				generate_kashida_justification_opportunies(p_start + limit, p_start + ubrk_current(bi));
				_justification_opportunies.push_back(JustificationOpportunity(p_start + ubrk_current(bi), JUSTIFICATION_WORD));
				limit = ubrk_current(bi);
			}
		}
	} else {
		if (err == U_MISSING_RESOURCE_ERROR) {
			//use fallback
			if (show_no_icu_data_warning) {
				WARN_PRINTS(TTR("ICU break iterator data is missing, using fallback whitespace line breaking and justification"));
				show_no_icu_data_warning = false;
			}
			int limit = p_start;
			for (int i = p_start; i < p_end; i++) {
				if (u_isWhitespace(utf16_text[i])) {
					generate_kashida_justification_opportunies(limit, i + 1);
					_justification_opportunies.push_back(JustificationOpportunity(i + 1, JUSTIFICATION_WORD));
					limit = i + 1;
				}
			}
			generate_kashida_justification_opportunies(limit, p_end);
		} else {
			ERR_EXPLAIN(String(u_errorName(err)));
			ERR_FAIL_COND(true);
		}
	}
	ubrk_close(bi);
};

void TextLayout::update_text_boundaries() {

	if (!_dirty_text_boundaries)
		return;

	clear_cache();

	for (int i = 0; i < items.size(); i++) {
		if (items[i]->type == OBJECT_TEXT) {
			UErrorCode err = U_ZERO_ERROR;
			TextLayoutItemText *text = static_cast<TextLayoutItemText *>(items[i].ptr());
			if ((!text->font.is_null()) && (text->string.length() > 0)) {
				size_t old_length = utf16_text.size();

				if (sizeof(CharType) == 4) {
					//:'‑( wchar_t why???
					int32_t utf16_length = 0;
					u_strFromUTF32(NULL, 0, &utf16_length, (const UChar32 *)text->string.ptr(), text->string.length(), &err);
					if (err != U_BUFFER_OVERFLOW_ERROR) {
						ERR_EXPLAIN(String(u_errorName(err)));
						ERR_FAIL_COND(true);
					} else {
						err = U_ZERO_ERROR;
						utf16_text.resize(utf16_text.size() + utf16_length);
						u_strFromUTF32((UChar *)&utf16_text[old_length], utf16_length, &utf16_length, (const UChar32 *)text->string.ptr(), text->string.length(), &err);
						if (U_FAILURE(err)) {
							utf16_text.resize(old_length);
							ERR_EXPLAIN(String(u_errorName(err)));
							ERR_FAIL_COND(true);
						}
					}
				} else {
					utf16_text.resize(utf16_text.size() + text->string.length());
					memcpy((UChar *)&utf16_text[old_length], text->string.ptr(), text->string.length() * sizeof(UChar));
				}

				if ((_style_runs.size() > 0) && (_style_runs[_style_runs.size() - 1].item == items[i])) {
					_style_runs[_style_runs.size() - 1].end = old_length + text->string.length();
				} else {
					_style_runs.push_back(TextRange(items[i], old_length, old_length + text->string.length()));
				}
			}
		} else {
			if (!items[i].ptr()->has(this)) {
				_style_runs.push_back(TextRange(items[i], utf16_text.size(), utf16_text.size() + 1));
				utf16_text.push_back(0xFFFC);
			}
		}
	}

	if (utf16_text.size() == 0)
		return;

	_break_opportunies.clear();
	_justification_opportunies.clear();

	for (int i = 0; i < _style_runs.size(); i++) {
		if (_style_runs[i].item->type == OBJECT_TEXT) {
			TextLayoutItemText *text = static_cast<TextLayoutItemText *>(_style_runs[i].item.ptr());
			generate_break_opportunies(_style_runs[i].start, _style_runs[i].end, text->language);
			generate_justification_opportunies(_style_runs[i].start, _style_runs[i].end, text->language);
		}
	}

	_dirty_text_boundaries = false;

	return;
};

void TextLayout::update_text_runs() {

	if (!_dirty_text_runs)
		return;

	if (_dirty_text_boundaries) update_text_boundaries();

	for (int i = 0; i < _runs.size(); i++) {
		memdelete(_runs[i]);
	};
	_runs.clear();
	_runs_logical.clear();

	Metrics metrics = Metrics();
	shape_text(0, _runs, _runs_logical, 0, utf16_text.size(), metrics, false);

	_runs_logical.sort_custom<RunCompare>();

	_dirty_text_runs = false;

	return;
};

void TextLayout::update_lines() {

	if (!_dirty_lines)
		return;

	if (_dirty_text_runs) update_text_runs();

	for (int i = 0; i < _lines.size(); i++) {
		memdelete(_lines[i]);
	}
	_lines.clear();

	float _max_width_act = _max_area.x - border_left_width - border_right_width - _padding.position.x - _padding.size.x;

	int next_hard = -1;
	if (_hard_breaks || _autowrap) {
		for (int b = 0; b < _break_opportunies.size(); b++) {
			if (_break_opportunies[b].break_type == BREAK_HARD) {
				next_hard = b;
				break;
			}
		}
	}

	Vector<BreakOpportunity> breaks;
	float line_width = 0.0;
	if ((next_hard > -1) || (_autowrap && (_max_width_act > 0) && (_break_opportunies.size() > 0))) {
		int brk = 0;
		for (int i = 0; i < _runs_logical.size(); i++) {
			int _start = (_runs_logical[i]->direction == HB_DIRECTION_LTR) ? 0 : _runs_logical[i]->clusters.size() - 1;
			int _end = (_runs_logical[i]->direction == HB_DIRECTION_LTR) ? _runs_logical[i]->clusters.size() : -1;
			int _delta = (_runs_logical[i]->direction == HB_DIRECTION_LTR) ? +1 : -1;

			bool hard_brk = ((next_hard > -1) && (_break_opportunies[next_hard].position >= _runs_logical[i]->start) && (_break_opportunies[next_hard].position <= _runs_logical[i]->end));
			if (hard_brk || ((_max_width_act > 0) && (line_width + _runs_logical[i]->size.x >= _max_width_act))) {
				int prev_break_position = 0;

				for (int j = _start; j != _end; j += _delta) {
					//HARD breaks
					if ((next_hard > -1) && (_break_opportunies[next_hard].position >= _runs_logical[i]->clusters[j]->start) && (_break_opportunies[next_hard].position <= _runs_logical[i]->clusters[j]->end)) {
						breaks.push_back(_break_opportunies[next_hard]);
						line_width = 0.0;
						prev_break_position = _break_opportunies[next_hard].position;
						if (next_hard < _break_opportunies.size()) {
							for (int b = next_hard + 1; b < _break_opportunies.size(); b++) {
								if (_break_opportunies[b].break_type == BREAK_HARD) {
									next_hard = b;
									break;
								}
							}
						}
					}
					//SOFT breaks
					if (_autowrap && (_max_width_act > 0) && (line_width + _runs_logical[i]->clusters[j]->size.x >= _max_width_act)) {
						int break_position = 0;
						int max_break_position = _runs_logical[i]->clusters[j]->start;
						while ((brk < _break_opportunies.size()) && (_break_opportunies[brk].position > prev_break_position) && (_break_opportunies[brk].position < _runs_logical[i]->clusters[j]->end) && (_break_opportunies[brk].position < max_break_position)) {
							if (_break_opportunies[brk].position >= _runs_logical[i]->start) break_position = _break_opportunies[brk].position;
							brk++;
						}
						if (break_position == 0) {
							break_position = (_break_anywhere) ? max_break_position : _runs_logical[i]->start;
						}
						if (break_position > prev_break_position) breaks.push_back(BreakOpportunity(break_position, BREAK_LINE));

						line_width = 0.0;
						for (int k = _start; k != j + _delta; k += _delta) {
							if (_runs_logical[i]->clusters[k]->start >= break_position) {
								line_width += _runs_logical[i]->clusters[k]->size.x;
							}
						}
						prev_break_position = break_position;
					} else {
						line_width += _runs_logical[i]->clusters[j]->size.x;
					}
				}
			} else {
				line_width += _runs_logical[i]->size.x;
			}
		}
	}
	breaks.push_back(BreakOpportunity(utf16_text.size() + 1, BREAK_HARD));

	int prev = 0;
	for (int i = 0; i < breaks.size(); i++) {

		Line *line = memnew(Line());
		line->start = prev;
		line->end = breaks[i].position;

		line->hard_break = (breaks[i].break_type == BREAK_HARD);

		Metrics metrics = Metrics();
		if (i > 0)
			for (int k = prev; k < breaks[i].position - 1; k++) {
				if ((utf16_text[k] != 0x009) && u_isWhitespace(utf16_text[k])) {
					prev++;
				} else {
					break;
				}
			}
		if (i < breaks.size() - 1)
			for (int k = breaks[i].position - 1; k > prev; k--) {
				if ((utf16_text[k] != 0x009) && u_isWhitespace(utf16_text[k])) {
					breaks[i].position--;
				} else {
					break;
				}
			}

		if (prev < breaks[i].position) shape_text(0, line->runs, line->runs_logical, prev, breaks[i].position, metrics, true);
		if (line->ascent < metrics.ascent) line->ascent = metrics.ascent;
		if (line->descent < metrics.descent) line->descent = metrics.descent;
		if (line->leading < metrics.leading) line->leading = metrics.leading;
		if (line->max_neg_glyph_displacement < metrics.max_neg_glyph_displacement) line->max_neg_glyph_displacement = metrics.max_neg_glyph_displacement;
		if (line->max_pos_glyph_displacement < metrics.max_pos_glyph_displacement) line->max_pos_glyph_displacement = metrics.max_pos_glyph_displacement;
		if (line->size.y < (line->ascent + line->descent)) line->size.y = (line->ascent + line->descent);

		line->size.x = metrics.width;
		line->runs_logical.sort_custom<RunCompare>();
		_lines.push_back(line);
		prev = breaks[i].position;
	}

	_dirty_lines = false;
	return;
};

void TextLayout::update_layout() {

	if (!_dirty_layout) return;

	if (_dirty_lines) update_lines();

	if (_lines.size() == 0) return;

	float y_offset = 0;
	float max_line_width = 0;
	float _max_width_act = _max_area.x - border_left_width - border_right_width - _padding.position.x - _padding.size.x;

	for (int i = 0; i < _lines.size(); i++) {
		if (_lines[i]->runs.size() == 0) {
			continue;
		}
		if ((_max_width_act < 0) ||
				((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_START) && _lines[i]->hard_break && (_para_direction == HB_DIRECTION_LTR)) ||
				((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_END) && _lines[i]->hard_break && (_para_direction == HB_DIRECTION_RTL)) ||
				((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_LEFT) && _lines[i]->hard_break) ||
				((_align == H_ALIGN_START) && (_para_direction == HB_DIRECTION_LTR)) ||
				((_align == H_ALIGN_END) && (_para_direction == HB_DIRECTION_RTL)) ||
				(_align == H_ALIGN_LEFT)) {
			//Left
			Point2 offset = Point2(0, y_offset);
			_lines[i]->offset = offset;

			for (int k = 0; k < _lines[i]->runs.size(); k++) {
				_lines[i]->runs[k]->offset = offset;
				for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
					_lines[i]->runs[k]->clusters[l]->offset = offset;
					offset.x += _lines[i]->runs[k]->clusters[l]->size.x;
				}
			}

			y_offset += MAX(_lines[i]->size.y, _lines[i]->leading) + _line_spacing + _lines[i]->max_neg_glyph_displacement + _lines[i]->max_pos_glyph_displacement;
			if (max_line_width < _lines[i]->size.x) max_line_width = _lines[i]->size.x;
		} else if (
				((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_START) && _lines[i]->hard_break && (_para_direction == HB_DIRECTION_RTL)) ||
				((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_END) && _lines[i]->hard_break && (_para_direction == HB_DIRECTION_LTR)) ||
				((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_RIGHT) && _lines[i]->hard_break) ||
				((_align == H_ALIGN_START) && (_para_direction == HB_DIRECTION_RTL)) ||
				((_align == H_ALIGN_END) && (_para_direction == HB_DIRECTION_LTR)) ||
				(_align == H_ALIGN_RIGHT)) {
			//Right
			Point2 offset = Point2(_max_width_act - _lines[i]->size.x, y_offset);
			_lines[i]->offset = offset;

			for (int k = 0; k < _lines[i]->runs.size(); k++) {
				_lines[i]->runs[k]->offset = offset;
				for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
					_lines[i]->runs[k]->clusters[l]->offset = offset;
					offset.x += _lines[i]->runs[k]->clusters[l]->size.x;
				}
			}

			y_offset += MAX(_lines[i]->size.y, _lines[i]->leading) + _line_spacing + _lines[i]->max_neg_glyph_displacement + _lines[i]->max_pos_glyph_displacement;
			if (max_line_width < offset.x) max_line_width = offset.x;
		} else if ((_align == H_ALIGN_CENTER) ||
				   ((_align == H_ALIGN_FILL) && (_align_last == H_ALIGN_CENTER) && _lines[i]->hard_break)) {
			//Center
			Point2 offset = Point2((_max_width_act - _lines[i]->size.x) / 2, y_offset);
			_lines[i]->offset = offset;

			for (int k = 0; k < _lines[i]->runs.size(); k++) {
				_lines[i]->runs[k]->offset = offset;
				for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
					_lines[i]->runs[k]->clusters[l]->offset = offset;
					offset.x += _lines[i]->runs[k]->clusters[l]->size.x;
				}
			}

			y_offset += MAX(_lines[i]->size.y, _lines[i]->leading) + _line_spacing + _lines[i]->max_neg_glyph_displacement + _lines[i]->max_pos_glyph_displacement;
			if (max_line_width < offset.x) max_line_width = offset.x;
		} else if (_align == H_ALIGN_FILL) {
			//Jusifiy
			float w_dif = _max_width_act - _lines[i]->size.x;

			int start_run = 0;
			int end_run = _lines[i]->runs.size() - 1;
			int start_cluster = 0;
			int end_cluster = _lines[i]->runs[end_run]->clusters.size() - 1;

			if (_para_direction == HB_DIRECTION_LTR) {
				//justify only to the right of last tab
				for (int k = _lines[i]->runs.size() - 1; k >= 0; k--) {
					for (int l = _lines[i]->runs[k]->clusters.size() - 1; l >= 0; l--) {
						if (_lines[i]->runs[k]->clusters[l]->glyphs[0].is_tab) {
							start_run = k;
							start_cluster = l;
							goto break_external;
						}
					}
				}
			} else {
				//justify only to the left of last tab
				for (int k = 0; k < _lines[i]->runs.size(); k++) {
					for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
						if (_lines[i]->runs[k]->clusters[l]->glyphs[0].is_tab) {
							end_run = k;
							end_cluster = l;
							goto break_external;
						}
					}
				}
			}
		break_external:

			//count valid inter-word & kashida opportunies
			int space_count = 0;
			int cluster_count = 0;
			Vector<JustificationOpportunity> ks_sel;
			for (int k = start_run; k <= end_run; k++) {
				for (int l = (k == start_run) ? start_cluster : 0; l <= ((k == end_run) ? end_cluster : _lines[i]->runs[k]->clusters.size() - 1); l++) {
					if ((k > start_run) || ((k == start_run) && (l > start_cluster))) {
						if ((k < end_run) || ((k == end_run) && (l < end_cluster))) {
							if ((k > 0) || ((k == 0) && (l > 0))) {
								if ((k < _lines[i]->runs.size() - 1) || ((k == _lines[i]->runs.size() - 1) && (l < _lines[i]->runs[k]->clusters.size() - 1))) {
									cluster_count++;
									_lines[i]->runs[k]->clusters[l]->elongation = 0;
									_lines[i]->runs[k]->clusters[l]->elongation_offset = 0;
									for (int m = 0; m < _justification_opportunies.size(); m++) {
										if ((_justification_opportunies[m].just_type == JUSTIFICATION_WORD) && (_justification_opportunies[m].position >= _lines[i]->runs[k]->clusters[l]->start) && (_justification_opportunies[m].position <= _lines[i]->runs[k]->clusters[l]->end)) {
											space_count++;
										}
										if ((_justification_opportunies[m].just_type == JUSTIFICATION_KASHIDA) && (_justification_opportunies[m].position >= _lines[i]->runs[k]->clusters[l]->start) && (_justification_opportunies[m].position <= _lines[i]->runs[k]->clusters[l]->end)) {
											ks_sel.push_back(_justification_opportunies[m]);
										}
									}
								}
							}
						}
					}
				}
			}

			//kashida justification
			if (_kashida && (_kashida_to_space_ratio > 0.01) && (ks_sel.size() > 0)) {
				float ks_w_dif = (w_dif * _kashida_to_space_ratio) / ks_sel.size();
				for (int j = 0; j < ks_sel.size(); j++) {
					for (int k = start_run; k <= end_run; k++) {
						for (int l = (k == start_run) ? start_cluster : 0; l <= ((k == end_run) ? end_cluster : _lines[i]->runs[k]->clusters.size() - 1); l++) {
							if (ks_sel[j].position == _lines[i]->runs[k]->clusters[l]->start) {
								if (_lines[i]->runs[k]->clusters[l]->item->type == OBJECT_TEXT) {
									TextLayoutItemText *item = static_cast<TextLayoutItemText *>(_lines[i]->runs[k]->clusters[l]->item.ptr());
									if ((item->font->get_char_size(0x0640).x > 0)) {
										float ks_w_dif_c = ks_w_dif;
										float ks_shift = 0;
										while ((ks_w_dif_c > item->font->get_char_size(0x0640).x)) {
											ks_w_dif_c -= item->font->get_char_size(0x0640).x;
											w_dif -= item->font->get_char_size(0x0640).x;
											ks_shift += item->font->get_char_size(0x0640).x;
											_lines[i]->runs[k]->clusters[l]->elongation++;
											_lines[i]->runs[k]->clusters[l]->elongation_offset += item->font->get_char_size(0x0640).x;
										}
									}
								}
							}
						}
					}
				}
			}

			//inter-word justification
			float space_dif = (space_count == 0) ? 0 : w_dif / space_count;
			Point2 offset = Point2(0, y_offset);
			_lines[i]->offset = offset;

			if ((space_dif > 0) && _inter_word) {
				for (int k = 0; k < _lines[i]->runs.size(); k++) {
					_lines[i]->runs[k]->offset = offset;
					for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
						if ((k > start_run) || ((k == start_run) && (l > start_cluster))) {
							if ((k < end_run) || ((k == end_run) && (l < end_cluster))) {
								if ((k > 0) || ((k == 0) && (l > 0))) {
									if ((k < _lines[i]->runs.size() - 1) || ((k == _lines[i]->runs.size() - 1) && (l < _lines[i]->runs[k]->clusters.size() - 1))) {
										for (int m = 0; m < _justification_opportunies.size(); m++) {
											if ((_justification_opportunies[m].just_type == JUSTIFICATION_WORD) && (_justification_opportunies[m].position >= _lines[i]->runs[k]->clusters[l]->start) && (_justification_opportunies[m].position <= _lines[i]->runs[k]->clusters[l]->end)) {
												offset.x += space_dif;
												break;
											}
										}
									}
								}
							}
						}
						_lines[i]->runs[k]->clusters[l]->offset = offset;
						offset.x += _lines[i]->runs[k]->clusters[l]->size.x + _lines[i]->runs[k]->clusters[l]->elongation_offset;
					}
				}
			} else {
				//inter-cluster justification
				space_dif = ((cluster_count == 0) || (!_inter_char)) ? 0 : w_dif / cluster_count;

				for (int k = 0; k < _lines[i]->runs.size(); k++) {
					_lines[i]->runs[k]->offset = offset;
					for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
						if ((k > start_run) || ((k == start_run) && (l > start_cluster))) {
							if ((k < end_run) || ((k == end_run) && (l < end_cluster))) {
								if ((k > 0) || ((k == 0) && (l > 0))) {
									if ((k < _lines[i]->runs.size() - 1) || ((k == _lines[i]->runs.size() - 1) && (l < _lines[i]->runs[k]->clusters.size() - 1))) {
										offset.x += space_dif;
									}
								}
							}
						}
						_lines[i]->runs[k]->clusters[l]->offset = offset;
						offset.x += _lines[i]->runs[k]->clusters[l]->size.x + _lines[i]->runs[k]->clusters[l]->elongation_offset;
					}
				}
			}

			_lines[i]->size.x = offset.x;

			y_offset += MAX(_lines[i]->size.y, _lines[i]->leading) + _line_spacing + _lines[i]->max_neg_glyph_displacement + _lines[i]->max_pos_glyph_displacement;
			if (max_line_width < offset.x) max_line_width = offset.x;
		}
	}

	_size.x = MAX(max_line_width, _min_area.x - border_left_width - border_right_width - _padding.position.x - _padding.size.x);
	_size.y = MAX(y_offset, _min_area.y - border_top_width - border_bottom_width - _padding.position.y - _padding.size.y);

	if (_max_area.x > 0) _size.x = MIN(_size.x, _max_area.x - border_left_width - border_right_width - _padding.position.x - _padding.size.x);
	if (_max_area.y > 0) _size.y = MIN(_size.y, _max_area.y - border_top_width - border_bottom_width - _padding.position.y - _padding.size.y);

	float y_shift = 0;

	if (_valign == V_ALIGN_CENTER) {
		y_shift = (_size.y - y_offset) / 2;
	} else if (_valign == V_ALIGN_BOTTOM) {
		y_shift = _size.y - y_offset;
	}

	Cluster *prev = NULL;

	for (int i = 0; i < _lines.size(); i++) {
		_lines[i]->offset.y += y_shift;
		for (int k = 0; k < _lines[i]->runs.size(); k++) {
			_lines[i]->runs[k]->offset.y += y_shift;
			for (int l = 0; l < _lines[i]->runs[k]->clusters.size(); l++) {
				_lines[i]->runs[k]->clusters[l]->offset.y += y_shift + _lines[i]->ascent - _lines[i]->runs[k]->clusters[l]->ascent;
				_lines[i]->runs[k]->clusters[l]->direction = _lines[i]->runs[k]->direction;
				_lines[i]->runs[k]->clusters[l]->line_index = i;
				_lines[i]->runs[k]->clusters[l]->prev_v = prev;
				_lines[i]->runs[k]->clusters[l]->next_v = NULL;
				if (prev) {
					prev->next_v = _lines[i]->runs[k]->clusters[l];
				} else {
					first_visual = _lines[i]->runs[k]->clusters[l];
				}
				prev = _lines[i]->runs[k]->clusters[l];
			}
		}
	}
	last_visual = prev;
	prev = NULL;
	for (int i = 0; i < _lines.size(); i++) {
		for (int k = 0; k < _lines[i]->runs_logical.size(); k++) {
			int _start = (_lines[i]->runs_logical[k]->direction == HB_DIRECTION_LTR) ? 0 : _lines[i]->runs_logical[k]->clusters.size() - 1;
			int _end = (_lines[i]->runs_logical[k]->direction == HB_DIRECTION_LTR) ? _lines[i]->runs_logical[k]->clusters.size() : -1;
			int _delta = (_lines[i]->runs_logical[k]->direction == HB_DIRECTION_LTR) ? +1 : -1;
			for (int l = _start; l != _end; l += _delta) {

				_lines[i]->runs_logical[k]->clusters[l]->prev_l = prev;
				_lines[i]->runs_logical[k]->clusters[l]->next_l = NULL;
				if (prev) {
					prev->next_l = _lines[i]->runs_logical[k]->clusters[l];
				} else {
					first_logical = _lines[i]->runs_logical[k]->clusters[l];
				}
				prev = _lines[i]->runs_logical[k]->clusters[l];
			}
		}
	}

	last_logical = prev;

	_dirty_layout = false;

	return;
};

bool TextLayout::shape_text(int p_paragraph, Vector<Run *> &p_runs, Vector<Run *> &p_runs_log, size_t p_line_start, size_t p_line_end, Metrics &p_metrics, bool p_temp) {

	if (p_line_start < 0) p_line_start = 0;
	if (p_line_end > utf16_text.size()) p_line_end = utf16_text.size();

	if (utf16_text.size() == 0)
		return false;

	UErrorCode err = U_ZERO_ERROR;

	UBiDi *para = NULL;
	UBiDi *target = NULL;
	UBiDi *line = NULL;

	if (_bidi) {
		para = ubidi_openSized(utf16_text.size(), 0, &err);
		if (U_FAILURE(err)) {
			ERR_EXPLAIN(String(u_errorName(err)));
			ERR_FAIL_COND_V(true, false);
		}
		switch (_base_direction) {
			case DIR_LTR: {
				ubidi_setPara(para, (const uint16_t *)&utf16_text[0], utf16_text.size(), UBIDI_LTR, NULL, &err);
				_para_direction = HB_DIRECTION_LTR;
			} break;
			case DIR_RTL: {
				ubidi_setPara(para, (const uint16_t *)&utf16_text[0], utf16_text.size(), UBIDI_RTL, NULL, &err);
				_para_direction = HB_DIRECTION_RTL;
			} break;
			case DIR_AUTO: {
				UBiDiDirection direction = ubidi_getBaseDirection((const uint16_t *)&utf16_text[0], utf16_text.size());
				ubidi_setPara(para, (const uint16_t *)&utf16_text[0], utf16_text.size(), direction, NULL, &err);
				_para_direction = (direction == UBIDI_RTL) ? HB_DIRECTION_RTL : HB_DIRECTION_LTR; //LTR - for LTR and neutral
			} break;
		}
	} else {
		switch (_base_direction) {
			case DIR_LTR: {
				_para_direction = HB_DIRECTION_LTR;
			} break;
			case DIR_RTL: {
				_para_direction = HB_DIRECTION_RTL;
			} break;
			case DIR_AUTO: {
				_para_direction = HB_DIRECTION_LTR; //LTR - for neutral
			} break;
		}
	}
	if (U_SUCCESS(err)) {
		int32_t start = 0;
		int32_t length;
		UBiDiLevel level;

		Vector<TextRange> script_runs;
		TextLayout::Script scr_runs = TextLayout::Script(utf16_text, 0, utf16_text.size());
		while (scr_runs.next())
			script_runs.push_back(TextRange(scr_runs.get_script(), scr_runs.get_start(), scr_runs.get_end()));

		if (_bidi) {
			if ((p_line_end != utf16_text.size()) || (p_line_start != 0)) {
				line = ubidi_openSized(p_line_end - p_line_start, 0, &err);
				if (U_FAILURE(err)) {
					ERR_EXPLAIN(String(u_errorName(err)));
					ERR_FAIL_COND_V(true, false);
				}
				ubidi_setLine(para, p_line_start, p_line_end, line, &err);
				if (U_FAILURE(err)) {
					ERR_EXPLAIN(String(u_errorName(err)));
					ERR_FAIL_COND_V(true, false);
				}
				target = line;
			} else {
				target = para;
			}
		}
		int count = (_bidi) ? ubidi_countRuns(target, &err) : 1;
		if (U_FAILURE(err)) {
			ERR_EXPLAIN(String(u_errorName(err)));
			ERR_FAIL_COND_V(true, false);
		}
		for (int i = 0; i < count; i++) {
			UBiDiDirection ub_direction = (_bidi) ? ubidi_getVisualRun(target, i, &start, &length) : ((_para_direction == HB_DIRECTION_LTR) ? UBIDI_LTR : UBIDI_RTL);

			int script = (ub_direction == UBIDI_LTR) ? 0 : script_runs.size() - 1;
			int style = (ub_direction == UBIDI_LTR) ? 0 : _style_runs.size() - 1;

			int limit = (ub_direction == UBIDI_LTR) ? p_line_start : p_line_end;
			int next_limit;
			int end = (ub_direction == UBIDI_LTR) ? p_line_start + start + length : p_line_start + start;

			hb_script_t script_;
			Ref<TextLayoutItem> _style;

			while (true) {
				if (ub_direction == UBIDI_LTR) {
					next_limit = end;
					if (script_runs[script].start <= limit) {
						script_ = script_runs[script].script;
						if (script < script_runs.size() - 1) {
							script++;
							if (next_limit > script_runs[script].start) next_limit = script_runs[script].start;
						}
					} else {
						if (next_limit > script_runs[script].start) next_limit = script_runs[script].start;
					}
					if (_style_runs[style].start <= limit) {
						_style = _style_runs[style].item;
						if (style < _style_runs.size() - 1) {
							style++;
							if (next_limit > _style_runs[style].start) next_limit = _style_runs[style].start;
						}
					} else {
						if (next_limit > _style_runs[style].start) next_limit = _style_runs[style].start;
					}
					if ((limit < p_line_start + start + length) && (next_limit > p_line_start + start)) {
						int shape_start = limit > p_line_start + start ? limit : p_line_start + start;
						int shape_end = next_limit < p_line_start + start + length ? next_limit : p_line_start + start + length;
						TextLayout::Run *run;

						bool found_in_cache = false;
						for (int j = 0; j < _runs.size(); j++) {
							if ((!_runs[j]->has_tabs) && (_runs[j]->start == shape_start) && (_runs[j]->end == shape_end) && (_runs[j]->direction == (ub_direction == UBIDI_LTR) ? HB_DIRECTION_LTR : HB_DIRECTION_RTL)) {
								run = _runs[j];
								found_in_cache = true;
								break;
							}
						}
						if (!found_in_cache) {
							run = memnew(TextLayout::Run());
							run->start = shape_start;
							run->end = shape_end;
							run->temp = p_temp;
							run->direction = (ub_direction == UBIDI_LTR) ? HB_DIRECTION_LTR : HB_DIRECTION_RTL;
							run->offset = Point2(p_metrics.width, 0);
							shape_run(shape_start, shape_end, _style, script_, run, p_metrics);
						} else {
							run->offset = Point2(p_metrics.width, 0);
						}

						if (run->clusters.size() > 0) {
							p_runs.push_back(run);
							p_runs_log.push_back(run);
							p_metrics.width += run->size.x;
							if (p_metrics.ascent < run->ascent) p_metrics.ascent = run->ascent;
							if (p_metrics.descent < run->descent) p_metrics.descent = run->descent;
							if (p_metrics.leading < run->leading) p_metrics.leading = run->leading;
							if (p_metrics.max_neg_glyph_displacement < run->max_neg_glyph_displacement) p_metrics.max_neg_glyph_displacement = run->max_neg_glyph_displacement;
							if (p_metrics.max_pos_glyph_displacement < run->max_pos_glyph_displacement) p_metrics.max_pos_glyph_displacement = run->max_pos_glyph_displacement;
						}
					}
					limit = next_limit;
					if (limit >= end)
						break;
				} else {
					next_limit = end;
					if (script_runs[script].end >= limit) {
						script_ = script_runs[script].script;
						if (script > 0) {
							script--;
							if (next_limit < script_runs[script].end) next_limit = script_runs[script].end;
						}
					} else {
						if (next_limit < script_runs[script].end) next_limit = script_runs[script].end;
					}
					if (_style_runs[style].end >= limit) {
						_style = _style_runs[style].item;
						if (style > 0) {
							style--;
							if (next_limit < _style_runs[style].end) next_limit = _style_runs[style].end;
						}
					} else {
						if (next_limit < _style_runs[style].end) next_limit = _style_runs[style].end;
					}
					if ((next_limit < p_line_start + start + length) && (limit > p_line_start + start)) {
						int shape_start = next_limit > start ? next_limit : start;
						int shape_end = limit < p_line_start + start + length ? limit : p_line_start + start + length;
						TextLayout::Run *run;

						bool found_in_cache = false;
						for (int j = 0; j < _runs.size(); j++) {
							if ((!_runs[j]->has_tabs) && (_runs[j]->start == shape_start) && (_runs[j]->end == shape_end) && (_runs[j]->direction == (ub_direction == UBIDI_LTR) ? HB_DIRECTION_LTR : HB_DIRECTION_RTL)) {
								run = _runs[j];
								found_in_cache = true;
								break;
							}
						}
						if (!found_in_cache) {
							run = memnew(TextLayout::Run());
							run->start = shape_start;
							run->end = shape_end;
							run->temp = p_temp;
							run->direction = (ub_direction == UBIDI_LTR) ? HB_DIRECTION_LTR : HB_DIRECTION_RTL;
							run->offset = Point2(p_metrics.width, 0);
							shape_run(shape_start, shape_end, _style, script_, run, p_metrics);
						} else {
							run->offset = Point2(p_metrics.width, 0);
						}
						if (run->clusters.size() > 0) {
							p_runs.push_back(run);
							p_runs_log.push_back(run);
							p_metrics.width += run->size.x;
							if (p_metrics.ascent < run->ascent) p_metrics.ascent = run->ascent;
							if (p_metrics.descent < run->descent) p_metrics.descent = run->descent;
							if (p_metrics.leading < run->leading) p_metrics.leading = run->leading;
							if (p_metrics.max_neg_glyph_displacement < run->max_neg_glyph_displacement) p_metrics.max_neg_glyph_displacement = run->max_neg_glyph_displacement;
							if (p_metrics.max_pos_glyph_displacement < run->max_pos_glyph_displacement) p_metrics.max_pos_glyph_displacement = run->max_pos_glyph_displacement;
						}
					}
					limit = next_limit;
					if (limit <= end)
						break;
				}
			}
		}
		if (line != NULL) {
			if (_bidi) ubidi_close(line);
		}
		if (_bidi) ubidi_close(para);
	} else {
		ERR_EXPLAIN(String(u_errorName(err)));
		ERR_FAIL_COND_V(true, false);
	}

	return true;
};

void TextLayout::shape_run(size_t p_start, size_t p_end, Ref<TextLayoutItem> p_item, hb_script_t p_script, TextLayout::Run *p_run, Metrics &p_metrics, int p_fallback_index) {

	if (utf16_text.size() == 0)
		return;

	if (p_item->type != OBJECT_TEXT) {
		if (utf16_text[p_start] == 0xFFFC) {
			//object run
			Cluster *new_cluster = memnew(Cluster());
			new_cluster->glyphs.clear();
			new_cluster->fallback_index = 0;
			new_cluster->valid = true;
			new_cluster->start = p_start;
			new_cluster->end = p_end;
			new_cluster->item = p_item;
			new_cluster->size = p_item->get_size();
			new_cluster->glyphs.push_back(Glyph(0, 0, Point2(0, 0), Size2(new_cluster->size)));

			if (p_item->align == V_ALIGN_TOP) {
				new_cluster->ascent = 0;
				new_cluster->descent = new_cluster->size.y;
				if (p_run->descent < new_cluster->descent) p_run->descent = new_cluster->descent;
			} else if (p_item->align == V_ALIGN_CENTER) {
				new_cluster->ascent = new_cluster->size.y / 2;
				new_cluster->descent = new_cluster->size.y / 2;
				if (p_run->ascent < new_cluster->ascent) p_run->ascent = new_cluster->ascent;
				if (p_run->descent < new_cluster->descent) p_run->descent = new_cluster->descent;
			} else if (p_item->align == V_ALIGN_BOTTOM) {
				new_cluster->ascent = new_cluster->size.y;
				new_cluster->descent = 0;
				if (p_run->ascent < new_cluster->ascent) p_run->ascent = new_cluster->ascent;
			} else if (p_item->align == V_ALIGN_PERCENTAGE) {
				new_cluster->ascent = new_cluster->size.y * p_item->pcnt;
				new_cluster->descent = new_cluster->size.y - new_cluster->ascent;
				if (p_run->ascent < new_cluster->ascent) p_run->ascent = new_cluster->ascent;
				if (p_run->descent < new_cluster->descent) p_run->descent = new_cluster->descent;
			}

			p_run->clusters.push_back(new_cluster);
			p_run->size.x += new_cluster->size.x;
		}
	} else {
		TextLayoutItemText *item = static_cast<TextLayoutItemText *>(p_item.ptr());
		//text run
		hb_font_t *h_font = item->font->get_hb_font(p_fallback_index);
		hb_buffer_clear_contents(_hb_buffer);

		Vector<Cluster *> clusters;
		float offset = p_run->offset.x;

		if (p_run->ascent < item->font->get_ascent()) p_run->ascent = item->font->get_ascent();
		if (p_run->descent < item->font->get_descent()) p_run->descent = item->font->get_descent();
		if (p_run->leading < item->font->get_leading()) p_run->leading = item->font->get_leading();

		if ((h_font != NULL) || !_shaping) {
			//shape using harfbuzz
			hb_buffer_set_direction(_hb_buffer, p_run->direction);

			if (_controls) {
				hb_buffer_set_flags(_hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_PRESERVE_DEFAULT_IGNORABLES | (p_start == 0 ? HB_BUFFER_FLAG_BOT : 0) | (p_end == utf16_text.size() ? HB_BUFFER_FLAG_EOT : 0)));
			} else {
				hb_buffer_set_flags(_hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT | (p_start == 0 ? HB_BUFFER_FLAG_BOT : 0) | (p_end == utf16_text.size() ? HB_BUFFER_FLAG_EOT : 0)));
			}

			hb_buffer_set_script(_hb_buffer, p_script);
			if (item->h_language) hb_buffer_set_language(_hb_buffer, item->h_language);

			hb_buffer_add_utf16(_hb_buffer, (const uint16_t *)utf16_text.ptr(), utf16_text.size(), p_start, p_end - p_start);

			hb_shape(h_font, _hb_buffer, item->h_features.empty() ? NULL : item->h_features.ptr(), item->h_features.size());

			unsigned int glyph_count;
			hb_glyph_info_t *glyph_info = hb_buffer_get_glyph_infos(_hb_buffer, &glyph_count);
			hb_glyph_position_t *glyph_pos = hb_buffer_get_glyph_positions(_hb_buffer, &glyph_count);

			if (glyph_count > 0) {
				uint32_t last_cluster_id = -1;
				for (int i = 0; i < glyph_count; i++) {
					if (glyph_pos[i].y_offset < 0) {
						if (p_run->max_neg_glyph_displacement < -(glyph_pos[i].y_offset / 64)) p_run->max_neg_glyph_displacement = -(glyph_pos[i].y_offset / 64);
					} else {
						if (p_run->max_pos_glyph_displacement < (glyph_pos[i].y_offset / 64)) p_run->max_pos_glyph_displacement = (glyph_pos[i].y_offset / 64);
					}
					if (last_cluster_id != glyph_info[i].cluster) {

						Cluster *new_cluster = memnew(Cluster());

						new_cluster->fallback_index = p_fallback_index;
						new_cluster->glyphs.clear();
						new_cluster->glyphs.push_back(Glyph(utf16_text[glyph_info[i].cluster], glyph_info[i].codepoint, Point2((glyph_pos[i].x_offset) / 64, -(glyph_pos[i].y_offset / 64)), Point2((glyph_pos[i].x_advance) / 64, (glyph_pos[i].y_advance / 64))));
						new_cluster->valid = (item->font->has_glyph(glyph_info[i].codepoint, p_fallback_index) || !u_isgraph(utf16_text[glyph_info[i].cluster])); //have valid glyph index or non graphic char
						new_cluster->start = glyph_info[i].cluster;
						new_cluster->end = glyph_info[i].cluster;
						new_cluster->ascent = item->font->get_ascent();
						new_cluster->descent = item->font->get_descent();
						new_cluster->item = p_item;
						new_cluster->size = Point2(glyph_pos[i].x_advance / 64, item->font->get_height());
						if (i != 0) {
							if (p_run->direction == HB_DIRECTION_LTR) {
								clusters[clusters.size() - 1]->end = glyph_info[i].cluster - 1;
							} else {
								new_cluster->end = clusters[clusters.size() - 1]->start - 1;
							}
						}

						if (utf16_text[glyph_info[i].cluster] == 0x0009) {
							float tab = 0;
							int j = 0;
							while (tab <= offset + new_cluster->size.x) {
								tab += _tab_stop[j] * item->font->get_char_size('m').x;
								j++;
								if (j == _tab_stop.size()) j = 0;
							}
							new_cluster->size.x = tab - offset;
							p_run->has_tabs = true;
						}

						offset += new_cluster->size.x;

						clusters.push_back(new_cluster);

						last_cluster_id = glyph_info[i].cluster;
					} else {
						clusters[clusters.size() - 1]->glyphs.push_back(Glyph(utf16_text[glyph_info[i].cluster], glyph_info[i].codepoint, Point2(clusters[clusters.size() - 1]->size.x + (glyph_pos[i].x_offset) / 64, -(glyph_pos[i].y_offset / 64)), Point2((glyph_pos[i].x_advance) / 64, (glyph_pos[i].y_advance / 64))));
						clusters[clusters.size() - 1]->valid &= (item->font->has_glyph(glyph_info[i].codepoint, p_fallback_index) || !u_isgraph(utf16_text[glyph_info[i].cluster])); //have valid glyph index or non graphic char
						clusters[clusters.size() - 1]->size.x += glyph_pos[i].x_advance / 64;

						offset += clusters[clusters.size() - 1]->size.x;
					}
				}
			}
		} else {
			//no harfbuzz shaping for bitmap fonts
			int start_char = (p_run->direction == HB_DIRECTION_LTR) ? p_start : p_end - 1;
			int end_char = (p_run->direction == HB_DIRECTION_LTR) ? p_end : p_start - 1;
			int delta = (p_run->direction == HB_DIRECTION_LTR) ? +1 : -1;

			for (int i = start_char; i != end_char; i += delta) {
				Cluster *new_cluster = memnew(Cluster());

				uint32_t glyph = item->font->char_to_glyph(utf16_text[i]).first;

				new_cluster->glyphs.clear();
				new_cluster->fallback_index = item->font->char_to_glyph(utf16_text[i]).second;
				new_cluster->glyphs.push_back(Glyph(utf16_text[i], glyph, Point2(0, 0), Point2(item->font->get_char_size(utf16_text[i]).x, 0)));
				new_cluster->valid = item->font->has_glyph(glyph, p_fallback_index);
				new_cluster->start = i;
				new_cluster->ascent = item->font->get_ascent();
				new_cluster->descent = item->font->get_descent();
				new_cluster->end = i;
				new_cluster->item = p_item;
				new_cluster->size = Point2(item->font->get_char_size(utf16_text[i]).x, item->font->get_height());

				if (utf16_text[i] == 0x0009) {
					float tab = 0;
					int j = 0;
					while (tab <= offset + new_cluster->size.x) {
						tab += _tab_stop[j] * item->font->get_char_size('m').x;
						j++;
						if (j == _tab_stop.size()) j = 0;
					}
					new_cluster->size.x = tab - offset;
					p_run->has_tabs = true;
				}

				clusters.push_back(new_cluster);
				offset += item->font->get_char_size(utf16_text[i]).x;
			}
		}
		//find and process fallback subruns
		if (clusters.size() > 0) {
			uint32_t failed_subrun_start = p_end + 1;
			uint32_t failed_subrun_end = p_start;
			for (int i = 0; i < clusters.size(); i++) {
				if ((clusters[i]->valid) || (p_fallback_index == item->font->get_fallback_count() - 1) || (!_fallback)) {
					if ((failed_subrun_start != p_end + 1) && _fallback) {
						shape_run(failed_subrun_start, failed_subrun_end + 1, p_item, p_script, p_run, p_metrics, p_fallback_index + 1);
						failed_subrun_start = p_end + 1;
						failed_subrun_end = p_start;
					}
					if (!clusters[i]->valid) {
						Size2 box_size = Size2(0, 0);
						for (int j = clusters[i]->start; j <= clusters[i]->end; j++) {
							box_size.x += item->font->get_hex_box_size(utf16_text[j]).x;
							box_size.y = MAX(box_size.y, item->font->get_hex_box_size(utf16_text[j]).y);
						}
						clusters[i]->size = box_size; //draw hex box instead of cluster
					}
					p_run->clusters.push_back(clusters[i]);
					p_run->size.x += clusters[i]->size.x;
				} else {
					if (failed_subrun_start >= clusters[i]->start) failed_subrun_start = clusters[i]->start;
					if (failed_subrun_end <= clusters[i]->end) failed_subrun_end = clusters[i]->end;
					memdelete(clusters[i]);
				}
			}
			if ((failed_subrun_start != p_end + 1) && _fallback) {
				shape_run(failed_subrun_start, failed_subrun_end + 1, p_item, p_script, p_run, p_metrics, p_fallback_index + 1);
			}
		}
	}
};

void TextLayout::set_base_direction(TextDirection p_base_direction) {

	if (_base_direction != p_base_direction) {
		_base_direction = p_base_direction;
		invalidate(INVALIDATE_RUNS);
	}
};

TextDirection TextLayout::get_base_direction() const {

	return _base_direction;
};

void TextLayout::set_border_left_color(const Color &p_color) {

	if (border_left_color != p_color) {
		border_left_color = p_color;
		invalidate(INVALIDATE_NONE);
	}
};

Color TextLayout::get_border_left_color() const {

	return border_left_color;
};

void TextLayout::set_border_top_color(const Color &p_color) {

	if (border_top_color != p_color) {
		border_top_color = p_color;
		invalidate(INVALIDATE_NONE);
	}
};

Color TextLayout::get_border_top_color() const {

	return border_top_color;
};

void TextLayout::set_border_right_color(const Color &p_color) {

	if (border_right_color != p_color) {
		border_right_color = p_color;
		invalidate(INVALIDATE_NONE);
	}
};

Color TextLayout::get_border_right_color() const {

	return border_right_color;
};

void TextLayout::set_border_bottom_color(const Color &p_color) {

	if (border_bottom_color != p_color) {
		border_bottom_color = p_color;
		invalidate(INVALIDATE_NONE);
	}
};

Color TextLayout::get_border_bottom_color() const {

	return border_bottom_color;
};

void TextLayout::set_border_left_width(int p_width) {

	if (border_left_width != p_width) {
		border_left_width = p_width;
		invalidate(INVALIDATE_LINES);
	}
};

int TextLayout::get_border_left_width() const {

	return border_left_width;
};

void TextLayout::set_border_top_width(int p_width) {

	if (border_top_width != p_width) {
		border_top_width = p_width;
		invalidate(INVALIDATE_LAYOUT);
	}
};

int TextLayout::get_border_top_width() const {

	return border_top_width;
};

void TextLayout::set_border_right_width(int p_width) {

	if (border_right_width != p_width) {
		border_right_width = p_width;
		invalidate(INVALIDATE_LAYOUT);
	}
};

int TextLayout::get_border_right_width() const {

	return border_right_width;
};

void TextLayout::set_border_bottom_width(int p_width) {

	if (border_bottom_width != p_width) {
		border_bottom_width = p_width;
		invalidate(INVALIDATE_LAYOUT);
	}
};

int TextLayout::get_border_bottom_width() const {

	return border_bottom_width;
};

void TextLayout::set_back_color(const Color &p_color) {

	if (back_color != p_color) {
		back_color = p_color;
		invalidate(INVALIDATE_NONE);
	}
};

Color TextLayout::get_back_color() const {

	return back_color;
};

void TextLayout::set_clip_rect(const Rect2 &p_clip) {

	if (_clip_rect != p_clip) {
		_clip_rect = p_clip;
		invalidate(INVALIDATE_NONE);
	}
};

Rect2 TextLayout::get_clip_rect() const {

	return _clip_rect;
};

void TextLayout::set_padding(const Rect2 &p_padding) {

	if (_padding != p_padding) {
		_padding = p_padding;
		invalidate(INVALIDATE_LINES);
	}
};

Rect2 TextLayout::get_padding() const {

	return _padding;
};

void TextLayout::set_max_area(const Size2 &p_area) {

	if (_max_area != p_area) {
		_max_area = p_area;
		invalidate(INVALIDATE_LINES);
	}
};

Size2 TextLayout::get_max_area() const {

	return _max_area;
};

void TextLayout::set_min_area(const Size2 &p_area) {

	if (_min_area != p_area) {
		_min_area = p_area;
		invalidate(INVALIDATE_LAYOUT);
	}
};

Size2 TextLayout::get_min_area() const {

	return _min_area;
};

void TextLayout::set_line_spacing(int p_spacing) {

	if (_line_spacing != p_spacing) {
		_line_spacing = p_spacing;
		invalidate(INVALIDATE_LINES);
	}
};

int TextLayout::get_line_spacing() const {

	return _line_spacing;
};

void TextLayout::set_hard_breaks(bool p_aw) {

	if (_hard_breaks != p_aw) {
		_hard_breaks = p_aw;
		invalidate(INVALIDATE_LINES);
	}
};

bool TextLayout::get_hard_breaks() const {

	return _hard_breaks;
};

void TextLayout::set_autowrap(bool p_aw) {

	if (_autowrap != p_aw) {
		_autowrap = p_aw;
		invalidate(INVALIDATE_LINES);
	}
};

bool TextLayout::get_autowrap() const {

	return _autowrap;
};

void TextLayout::set_valign(TextVAlign p_align) {

	if (_valign != p_align) {
		_valign = p_align;
		invalidate(INVALIDATE_LAYOUT);
	}
};

TextVAlign TextLayout::get_valign() const {

	return _valign;
};

void TextLayout::set_align(TextHAlign p_align) {

	if (_align != p_align) {
		_align = p_align;
		invalidate(INVALIDATE_LINES);
	}
};

TextHAlign TextLayout::get_align() const {

	return _align;
};

void TextLayout::set_parent_valign(TextVAlign p_align) {

	if (_parent_valign != p_align) {
		_parent_valign = p_align;
		invalidate(INVALIDATE_NONE);
	}
};

TextVAlign TextLayout::get_parent_valign() const {

	return _parent_valign;
};

void TextLayout::set_parent_halign(TextHAlign p_align) {

	if (_parent_halign != p_align) {
		_parent_halign = p_align;
		invalidate(INVALIDATE_NONE);
	}
};

TextHAlign TextLayout::get_parent_halign() const {

	return _parent_halign;
};

void TextLayout::set_align_last(TextHAlign p_align) {

	if (_align_last != p_align) {
		_align_last = p_align;
		invalidate(INVALIDATE_LINES);
	}
};

TextHAlign TextLayout::get_align_last() const {

	return _align_last;
};

void TextLayout::set_tab_stops(const Vector<int> &p_tab_stops) {

	_tab_stop = p_tab_stops;
	invalidate(INVALIDATE_RUNS);
};

void TextLayout::set_kasida_to_space_ratio(float p_ratio) {

	if (p_ratio > 1.0) {
		_kashida_to_space_ratio = 1.0;
	} else if (p_ratio < 0.0) {
		_kashida_to_space_ratio = 0.0;
	} else {
		_kashida_to_space_ratio = p_ratio;
	}
	invalidate(INVALIDATE_LAYOUT);
};

float TextLayout::get_kasida_to_space_ratio() const {

	return _kashida_to_space_ratio;
};

void TextLayout::set_show_invisible_characters(bool p_enable) {

	if (_invisibles != p_enable) {
		_invisibles = p_enable;
		invalidate(INVALIDATE_NONE);
	}
};

bool TextLayout::get_show_invisible_characters() const {

	return _invisibles;
};

void TextLayout::set_show_control_characters(bool p_enable) {

	if (_controls != p_enable) {
		_controls = p_enable;
		invalidate(INVALIDATE_RUNS);
	}
}

bool TextLayout::get_show_control_characters() const {

	return _controls;
}

void TextLayout::set_invisible_characters(const String &p_string) {

	if (p_string.length() == 4) {
		_invis = p_string;
		invalidate(INVALIDATE_NONE);
	}
};

String TextLayout::get_invisible_characters() {

	return _invis;
};

void TextLayout::set_enable_kasida_justification(bool p_enable) {

	if (_kashida != p_enable) {
		_kashida = p_enable;
		invalidate(INVALIDATE_LINES);
	}
};

bool TextLayout::get_enable_kasida_justification() const {

	return _kashida;
};

void TextLayout::set_enable_interword_justification(bool p_enable) {

	if (_inter_word != p_enable) {
		_inter_word = p_enable;
		invalidate(INVALIDATE_LINES);
	}
};

bool TextLayout::get_enable_interword_justification() const {

	return _inter_word;
};

void TextLayout::set_enable_intercluster_justification(bool p_enable) {

	if (_inter_char != p_enable) {
		_inter_char = p_enable;
		invalidate(INVALIDATE_LINES);
	}
};

bool TextLayout::get_enable_intercluster_justification() const {

	return _inter_char;
};

void TextLayout::set_enable_fallback_line_break(bool p_enable) {

	if (_break_anywhere != p_enable) {
		_break_anywhere = p_enable;
		invalidate(INVALIDATE_LINES);
	}
};

bool TextLayout::get_enable_fallback_line_break() const {

	return _break_anywhere;
};

void TextLayout::set_enable_bidi(bool p_enable) {

	if (_bidi != p_enable) {
		_bidi = p_enable;
		invalidate(INVALIDATE_ALL);
	}
};

bool TextLayout::get_enable_bidi() const {

	return _bidi;
};

void TextLayout::set_enable_shaping(bool p_enable) {

	if (_shaping != p_enable) {
		_shaping = p_enable;
		invalidate(INVALIDATE_ALL);
	}
};

bool TextLayout::get_enable_shaping() const {

	return _shaping;
};

void TextLayout::set_enable_fallback(bool p_enable) {

	if (_fallback != p_enable) {
		_fallback = p_enable;
		invalidate(INVALIDATE_ALL);
	}
};

bool TextLayout::get_enable_fallback() const {

	return _fallback;
};

Vector<int> TextLayout::get_tab_stops() const {

	return _tab_stop;
};

Rect2 TextLayout::get_bounds() {

	update_layout();

	return Rect2(Point2(0, 0), Size2(_size.x + _padding.position.x + _padding.size.x + border_left_width + border_right_width, _size.y + _padding.position.y + _padding.size.y + border_top_width + border_bottom_width));
};

int TextLayout::get_line_count() {

	update_layout();

	return _lines.size();
};

int TextLayout::get_line_start(int p_line) {

	update_layout();

	if ((p_line < 0) || (p_line >= _lines.size()))
		return -1;

	return _lines[p_line]->start;
};

int TextLayout::get_line_end(int p_line) {

	update_layout();

	if ((p_line < 0) || (p_line >= _lines.size()))
		return -1;

	return _lines[p_line]->end;
};

Rect2 TextLayout::get_line_bounds(int p_line) {

	update_layout();

	if ((p_line < 0) || (p_line >= _lines.size()))
		return Rect2();

	return Rect2(_lines[p_line]->offset, _lines[p_line]->size);
};

Ref<TextLayout> TextLayout::hit_test_layout(const Point2 &p_point) {

	update_layout();

	Cluster *cluster = first_logical;
	Point2 pos = p_point + _padding.position + Point2(border_left_width, border_top_width);
	while (cluster) {
		if ((pos.x >= cluster->offset.x) && (pos.y >= cluster->offset.y) && (pos.x <= (cluster->offset.x + cluster->size.x)) && (pos.y <= (cluster->offset.y + cluster->size.y))) {
			if (cluster->item->type == OBJECT_TABLE) {
				TextLayoutItemTable *table = static_cast<TextLayoutItemTable *>(cluster->item.ptr());
				return table->hit_test(pos - cluster->offset);
			} else if (cluster->item->type == OBJECT_SPAN) {
				TextLayoutItemSpan *span = static_cast<TextLayoutItemSpan *>(cluster->item.ptr());
				return span->layout;
			}
			return Ref<TextLayout>(this);
		}
		cluster = cluster->next_l;
	}
	return NULL;
};

Ref<TextHitInfo> TextLayout::hit_test(const Point2 &p_point) {

	update_layout();

	Cluster *cluster = first_logical;
	Point2 pos = p_point + _padding.position + Point2(border_left_width, border_top_width);
	while (cluster) {
		if ((pos.x >= cluster->offset.x) && (pos.y >= cluster->offset.y) && (pos.x <= (cluster->offset.x + cluster->size.x)) && (pos.y <= (cluster->offset.y + cluster->size.y))) {
			Ref<TextHitInfo> hit = Ref<TextHitInfo>(memnew(TextHitInfo()));
			hit->index_start = cluster->start;
			hit->index_end = cluster->end;
			bool left_hit = (pos.x <= (cluster->offset.x + cluster->size.x / 2));
			if (cluster->direction == HB_DIRECTION_RTL) {
				hit->index_insertion = (left_hit) ? cluster->end + 1 : cluster->start;
			} else {
				hit->index_insertion = (left_hit) ? cluster->start : cluster->end + 1;
			}
			hit->bounds = Rect2(cluster->offset + _padding.position + Point2(border_left_width, border_top_width), cluster->size);
			hit->leading_edge = cluster->leading_edge() + _padding.position + Point2(border_left_width, border_top_width);
			hit->trailing_edge = cluster->trailing_edge() + _padding.position + Point2(border_left_width, border_top_width);
			hit->type = cluster->item->type;
			return hit;
		}
		cluster = cluster->next_l;
	}
	return NULL;
};

Vector<Variant> TextLayout::highlight_shapes_hit(Ref<TextHitInfo> p_first_hit, Ref<TextHitInfo> p_second_hit, bool p_selection) {

	if (p_selection) {
		return highlight_shapes_char(MIN(p_first_hit->index_insertion, p_second_hit->index_insertion), MAX(p_first_hit->index_insertion, p_second_hit->index_insertion));
	} else {
		return highlight_shapes_char(MIN(p_first_hit->index_start, p_second_hit->index_start), MAX(p_first_hit->index_end, p_second_hit->index_end));
	}
};

Vector<Variant> TextLayout::highlight_shapes_char(int p_start, int p_end) {

	update_layout();

	Vector<Variant> output;
	Cluster *cluster = first_logical;
	while (cluster) {
		if ((cluster->start >= p_start) && (cluster->start < p_end)) {
			TextLayoutItemText *item = static_cast<TextLayoutItemText *>(cluster->item.ptr());
			if (!item->font.is_null()) output.push_back(clip_rect(Rect2(cluster->offset, Size2(cluster->size.x, item->font->get_height()))));
		}
		cluster = cluster->next_l;
	}
	return output;
};

Vector<Variant> TextLayout::caret_shapes_hit(Ref<TextHitInfo> p_hit) {

	return caret_shapes_char(p_hit->index_insertion);
};

Vector<Variant> TextLayout::caret_shapes_char(int p_index) {

	update_layout();

	Vector<Variant> output;

	Cluster *leading = NULL;
	Cluster *trailing = NULL;
	bool leading_strong = true;

	Cluster *cluster = first_logical;
	while (cluster) {
		if (cluster->start < p_index) {
			leading = cluster;
			leading_strong = (_para_direction == cluster->direction);
		} else {
			trailing = cluster;
			break;
		}
		cluster = cluster->next_l;
	}

	if (leading_strong) {
		if (leading) {
			output.push_back(clip_rect(Rect2(leading->leading_edge(), Size2(leading->size.x, _lines[leading->line_index]->size.y))));
		}
		if ((trailing) && (trailing != leading)) {
			output.push_back(clip_rect(Rect2(trailing->trailing_edge(), Size2(trailing->size.x, _lines[trailing->line_index]->size.y))));
		}
	} else {
		if (trailing) {
			output.push_back(clip_rect(Rect2(trailing->trailing_edge(), Size2(trailing->size.x, _lines[trailing->line_index]->size.y))));
		}
		if ((leading) && (trailing != leading)) {
			output.push_back(clip_rect(Rect2(leading->leading_edge(), Size2(leading->size.x, _lines[leading->line_index]->size.y))));
		}
	}

	return output;
};

void TextLayout::draw_line(RID p_canvas_item, Point2 p_pos, float p_width, const Color &p_color, int p_size) const {

	if ((p_size > 0) && (p_color.a > 0)) {
		VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(p_pos, Size2(p_width, p_size))), p_color);
	}
};

Rect2 TextLayout::draw(RID p_canvas_item, const Point2 &p_pos) {

	update_layout();

	Point2 pos = p_pos;

	if (back_color.a > 0) {
		VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(pos, _size + _padding.position + _padding.size + Size2(border_left_width + border_right_width, border_top_width + border_bottom_width))), back_color);
	};

	if ((border_left_width > 0) && (border_left_color.a > 0)) {
		VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(pos, Size2(border_left_width, _size.y + _padding.size.y + _padding.position.y + border_top_width + border_bottom_width))), border_left_color);
	};
	if ((border_right_width > 0) && (border_right_color.a > 0)) {
		VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(pos + Point2(_size.x + _padding.size.x + _padding.position.x + border_left_width, 0), Size2(border_right_width, _size.y + _padding.size.y + _padding.position.y + border_top_width + border_bottom_width))), border_right_color);
	};
	if ((border_top_width > 0) && (border_top_color.a > 0)) {
		VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(pos, Size2(_size.x + _padding.size.x + _padding.position.x + border_left_width + border_right_width, border_top_width))), border_top_color);
	};
	if ((border_bottom_width > 0) && (border_bottom_color.a > 0)) {
		VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(pos + Point2(0, _size.y + _padding.size.y + _padding.position.y + border_top_width), Size2(_size.x + _padding.size.x + _padding.position.x + border_left_width + border_right_width, border_bottom_width))), border_bottom_color);
	};

	pos += _padding.position + Point2(border_left_width, border_top_width);
	Cluster *cluster = first_visual;
	while (cluster) {
		if (cluster->item->type == OBJECT_TEXT) {
			TextLayoutItemText *item = static_cast<TextLayoutItemText *>(cluster->item.ptr());
			if (item->back_color.a > 0) VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, clip_rect(Rect2(pos + cluster->offset, cluster->size)), item->back_color);
			draw_line(p_canvas_item, pos + cluster->offset + Point2(0, cluster->ascent), cluster->size.x, item->underline_color, item->underline_width);
			draw_line(p_canvas_item, pos + cluster->offset + Point2(0, -item->overline_width), cluster->size.x, item->overline_color, item->overline_width);
			Point2 offset = Point2(0, 0);
			if (cluster->valid) {
				if (_kashida && (_kashida_to_space_ratio > 0.01) && cluster->elongation > 0) {
					for (int i = 0; i < cluster->elongation; i++) {
						offset.y = cluster->glyphs[cluster->glyphs.size() - 1].offset.y;
						item->font->draw_raw_glyph(p_canvas_item, pos + cluster->offset + offset, item->font->char_to_glyph(0x0640).first, item->font->char_to_glyph(0x0640).second, item->fore_color, _clip_rect);
						offset.x += item->font->get_char_size(0x0640).x;
					}
					offset.y = 0;
				}
				for (int j = 0; j < cluster->glyphs.size(); j++) {
					if (_invisibles && (cluster->glyphs[j].is_tab)) {
						item->font->draw_raw_glyph(p_canvas_item, pos + cluster->offset + cluster->glyphs[j].offset + offset, item->font->char_to_glyph(_invis[1]).first, item->font->char_to_glyph(_invis[1]).second, item->fore_color_invis, _clip_rect);
					} else if (_invisibles && (cluster->glyphs[j].is_cr)) {
						item->font->draw_raw_glyph(p_canvas_item, pos + cluster->offset + cluster->glyphs[j].offset + offset, item->font->char_to_glyph(_invis[2]).first, item->font->char_to_glyph(_invis[2]).second, item->fore_color_invis, _clip_rect);
					} else if (_invisibles && (cluster->glyphs[j].is_ws)) {
						item->font->draw_raw_glyph(p_canvas_item, pos + cluster->offset + cluster->glyphs[j].offset + offset, item->font->char_to_glyph(_invis[0]).first, item->font->char_to_glyph(_invis[0]).second, item->fore_color_invis, _clip_rect);
					} else if (_controls && (cluster->glyphs[j].is_ctrl)) {
						item->font->draw_raw_glyph(p_canvas_item, pos + cluster->offset + cluster->glyphs[j].offset + offset, item->font->char_to_glyph(_invis[3]).first, item->font->char_to_glyph(_invis[3]).second, item->fore_color_invis, _clip_rect);
					} else if (cluster->glyphs[j].is_gr) {
						item->font->draw_raw_glyph(p_canvas_item, pos + cluster->offset + cluster->glyphs[j].offset + offset, cluster->glyphs[j].codepoint, cluster->fallback_index, item->fore_color, _clip_rect);
					}
				}
			} else {
				//draw fallback hex boxes
				for (int i = cluster->start; i <= cluster->end; i++) {
					item->font->draw_hex_box(p_canvas_item, pos + cluster->offset + offset, utf16_text[i], item->fore_color, _clip_rect);
					offset.x += item->font->get_hex_box_size(utf16_text[i]).x;
				}
			}
			draw_line(p_canvas_item, pos + cluster->offset + Point2(0, (cluster->ascent - item->strikeout_width) / 2), cluster->size.x, item->strikeout_color, item->strikeout_width);
		} else {
			cluster->item->draw(p_canvas_item, pos + cluster->offset, _clip_rect);
		}
		cluster = cluster->next_v;
	}
	return Rect2(p_pos, Size2(_size.x + _padding.position.x + _padding.size.x + border_left_width + border_right_width, _size.y + _padding.position.y + _padding.size.y + border_top_width + border_bottom_width));
};

bool TextLayout::show_no_icu_data_warning = true;

void TextLayout::initialize_icu() {

	//init ICU data, u_setDataDirectory should be called at most once in a process, before the first ICU operation (e.g., u_init())
	String cwd = ProjectSettings::get_singleton()->get_resource_path();
	u_setDataDirectory((const char *)cwd.utf8().get_data());

	UErrorCode err = U_ZERO_ERROR;
	u_init(&err);
	if (U_FAILURE(err)) ERR_PRINT(u_errorName(err));
};

void TextLayout::finish_icu() {

	u_cleanup();
};

TextLayout::TextLayout() {

	_base_direction = DIR_LTR;
	_max_area = Size2(-1, -1);
	_min_area = Size2(0, 0);
	_autowrap = false;
	_line_spacing = 0;
	_tab_stop.push_back(4);

	_valign = V_ALIGN_TOP;
	_align = H_ALIGN_START;
	_align_last = H_ALIGN_START;

	_parent_valign = V_ALIGN_CENTER;
	_parent_halign = H_ALIGN_CENTER;

	_offset = 0;
	_kashida_to_space_ratio = 1.0;
	_para_direction = HB_DIRECTION_LTR;
	_size = Size2(0, 0);

	_padding = Rect2(0, 0, 0, 0);
	_clip_rect = Rect2(0, 0, -1, -1);

	border_left_color = Color(0, 0, 0, 0);
	border_top_color = Color(0, 0, 0, 0);
	border_right_color = Color(0, 0, 0, 0);
	border_bottom_color = Color(0, 0, 0, 0);

	border_left_width = 0;
	border_top_width = 0;
	border_right_width = 0;
	border_bottom_width = 0;

	first_logical = NULL;
	first_visual = NULL;
	last_logical = NULL;
	last_visual = NULL;

	back_color = Color(0, 0, 0, 0);

	_invisibles = false;
	_controls = false;

	_invis = String::chr(0x00B7) + String::chr(0x00BB) + String::chr(0x00AC) + String::chr(0x00A4);
	//0x00B7 MIDDLE DOT
	//0x00BB RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
	//0x00AC NOT SIGN
	//0x00A4 CURRENCY SIGN

	_shaping = true;
	_bidi = true;
	_fallback = true;

	_kashida = true;
	_inter_word = true;
	_inter_char = false;

	_break_anywhere = false;
	_hard_breaks = true;

	_dirty_text_boundaries = true;
	_dirty_text_runs = true;
	_dirty_lines = true;
	_dirty_layout = true;

	_hb_buffer = hb_buffer_create();
};

TextLayout::~TextLayout() {

	clear_cache();
	items.clear();
	hb_buffer_destroy(_hb_buffer);
};