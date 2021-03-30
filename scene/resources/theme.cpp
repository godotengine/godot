/*************************************************************************/
/*  theme.cpp                                                            */
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

#include "theme.h"
#include "core/os/file_access.h"
#include "core/string/print_string.h"

void Theme::_emit_theme_changed() {
	emit_changed();
}

Vector<String> Theme::_get_icon_list(const String &p_theme_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_icon_list(p_theme_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_icon_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_icon_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_stylebox_list(const String &p_theme_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_stylebox_list(p_theme_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_stylebox_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_stylebox_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_font_list(const String &p_theme_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_font_list(p_theme_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_font_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_font_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_font_size_list(const String &p_theme_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_font_size_list(p_theme_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_font_size_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_font_size_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_color_list(const String &p_theme_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_color_list(p_theme_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_color_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_color_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_constant_list(const String &p_theme_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_constant_list(p_theme_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_constant_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_constant_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_theme_item_list(DataType p_data_type, const String &p_theme_type) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			return _get_color_list(p_theme_type);
		case DATA_TYPE_CONSTANT:
			return _get_constant_list(p_theme_type);
		case DATA_TYPE_FONT:
			return _get_font_list(p_theme_type);
		case DATA_TYPE_FONT_SIZE:
			return _get_font_size_list(p_theme_type);
		case DATA_TYPE_ICON:
			return _get_icon_list(p_theme_type);
		case DATA_TYPE_STYLEBOX:
			return _get_stylebox_list(p_theme_type);
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	return Vector<String>();
}

Vector<String> Theme::_get_theme_item_type_list(DataType p_data_type) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			return _get_color_type_list();
		case DATA_TYPE_CONSTANT:
			return _get_constant_type_list();
		case DATA_TYPE_FONT:
			return _get_font_type_list();
		case DATA_TYPE_FONT_SIZE:
			return _get_font_size_type_list();
		case DATA_TYPE_ICON:
			return _get_icon_type_list();
		case DATA_TYPE_STYLEBOX:
			return _get_stylebox_type_list();
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	return Vector<String>();
}

Vector<String> Theme::_get_type_list() const {
	Vector<String> ilret;
	List<StringName> il;

	get_type_list(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

bool Theme::_set(const StringName &p_name, const Variant &p_value) {
	String sname = p_name;

	if (sname.find("/") != -1) {
		String type = sname.get_slicec('/', 1);
		String theme_type = sname.get_slicec('/', 0);
		String name = sname.get_slicec('/', 2);

		if (type == "icons") {
			set_icon(name, theme_type, p_value);
		} else if (type == "styles") {
			set_stylebox(name, theme_type, p_value);
		} else if (type == "fonts") {
			set_font(name, theme_type, p_value);
		} else if (type == "colors") {
			set_color(name, theme_type, p_value);
		} else if (type == "constants") {
			set_constant(name, theme_type, p_value);
		} else {
			return false;
		}

		return true;
	}

	return false;
}

bool Theme::_get(const StringName &p_name, Variant &r_ret) const {
	String sname = p_name;

	if (sname.find("/") != -1) {
		String type = sname.get_slicec('/', 1);
		String theme_type = sname.get_slicec('/', 0);
		String name = sname.get_slicec('/', 2);

		if (type == "icons") {
			if (!has_icon(name, theme_type)) {
				r_ret = Ref<Texture2D>();
			} else {
				r_ret = get_icon(name, theme_type);
			}
		} else if (type == "styles") {
			if (!has_stylebox(name, theme_type)) {
				r_ret = Ref<StyleBox>();
			} else {
				r_ret = get_stylebox(name, theme_type);
			}
		} else if (type == "fonts") {
			if (!has_font(name, theme_type)) {
				r_ret = Ref<Font>();
			} else {
				r_ret = get_font(name, theme_type);
			}
		} else if (type == "colors") {
			r_ret = get_color(name, theme_type);
		} else if (type == "constants") {
			r_ret = get_constant(name, theme_type);
		} else {
			return false;
		}

		return true;
	}

	return false;
}

void Theme::_get_property_list(List<PropertyInfo> *p_list) const {
	List<PropertyInfo> list;

	const StringName *key = nullptr;

	while ((key = icon_map.next(key))) {
		const StringName *key2 = nullptr;

		while ((key2 = icon_map[*key].next(key2))) {
			list.push_back(PropertyInfo(Variant::OBJECT, String() + *key + "/icons/" + *key2, PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		}
	}

	key = nullptr;

	while ((key = style_map.next(key))) {
		const StringName *key2 = nullptr;

		while ((key2 = style_map[*key].next(key2))) {
			list.push_back(PropertyInfo(Variant::OBJECT, String() + *key + "/styles/" + *key2, PROPERTY_HINT_RESOURCE_TYPE, "StyleBox", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		}
	}

	key = nullptr;

	while ((key = font_map.next(key))) {
		const StringName *key2 = nullptr;

		while ((key2 = font_map[*key].next(key2))) {
			list.push_back(PropertyInfo(Variant::OBJECT, String() + *key + "/fonts/" + *key2, PROPERTY_HINT_RESOURCE_TYPE, "Font", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		}
	}

	key = nullptr;

	while ((key = color_map.next(key))) {
		const StringName *key2 = nullptr;

		while ((key2 = color_map[*key].next(key2))) {
			list.push_back(PropertyInfo(Variant::COLOR, String() + *key + "/colors/" + *key2));
		}
	}

	key = nullptr;

	while ((key = constant_map.next(key))) {
		const StringName *key2 = nullptr;

		while ((key2 = constant_map[*key].next(key2))) {
			list.push_back(PropertyInfo(Variant::INT, String() + *key + "/constants/" + *key2));
		}
	}

	list.sort();
	for (List<PropertyInfo>::Element *E = list.front(); E; E = E->next()) {
		p_list->push_back(E->get());
	}
}

void Theme::set_default_theme_font(const Ref<Font> &p_default_font) {
	if (default_theme_font == p_default_font) {
		return;
	}

	if (default_theme_font.is_valid()) {
		default_theme_font->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	default_theme_font = p_default_font;

	if (default_theme_font.is_valid()) {
		default_theme_font->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	notify_property_list_changed();
	emit_changed();
}

Ref<Font> Theme::get_default_theme_font() const {
	return default_theme_font;
}

void Theme::set_default_theme_font_size(int p_font_size) {
	if (default_theme_font_size == p_font_size) {
		return;
	}

	default_theme_font_size = p_font_size;

	notify_property_list_changed();
	emit_changed();
}

int Theme::get_default_theme_font_size() const {
	return default_theme_font_size;
}

Ref<Theme> Theme::project_default_theme;
Ref<Theme> Theme::default_theme;
Ref<Texture2D> Theme::default_icon;
Ref<StyleBox> Theme::default_style;
Ref<Font> Theme::default_font;
int Theme::default_font_size = 16;

Ref<Theme> Theme::get_default() {
	return default_theme;
}

void Theme::set_default(const Ref<Theme> &p_default) {
	default_theme = p_default;
}

Ref<Theme> Theme::get_project_default() {
	return project_default_theme;
}

void Theme::set_project_default(const Ref<Theme> &p_project_default) {
	project_default_theme = p_project_default;
}

void Theme::set_default_icon(const Ref<Texture2D> &p_icon) {
	default_icon = p_icon;
}

void Theme::set_default_style(const Ref<StyleBox> &p_style) {
	default_style = p_style;
}

void Theme::set_default_font(const Ref<Font> &p_font) {
	default_font = p_font;
}

void Theme::set_default_font_size(int p_font_size) {
	default_font_size = p_font_size;
}

void Theme::set_icon(const StringName &p_name, const StringName &p_theme_type, const Ref<Texture2D> &p_icon) {
	bool new_value = !icon_map.has(p_theme_type) || !icon_map[p_theme_type].has(p_name);

	if (icon_map[p_theme_type].has(p_name) && icon_map[p_theme_type][p_name].is_valid()) {
		icon_map[p_theme_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	icon_map[p_theme_type][p_name] = p_icon;

	if (p_icon.is_valid()) {
		icon_map[p_theme_type][p_name]->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	if (new_value) {
		notify_property_list_changed();
		emit_changed();
	}
}

Ref<Texture2D> Theme::get_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (icon_map.has(p_theme_type) && icon_map[p_theme_type].has(p_name) && icon_map[p_theme_type][p_name].is_valid()) {
		return icon_map[p_theme_type][p_name];
	} else {
		return default_icon;
	}
}

bool Theme::has_icon(const StringName &p_name, const StringName &p_theme_type) const {
	return (icon_map.has(p_theme_type) && icon_map[p_theme_type].has(p_name) && icon_map[p_theme_type][p_name].is_valid());
}

bool Theme::has_icon_nocheck(const StringName &p_name, const StringName &p_theme_type) const {
	return (icon_map.has(p_theme_type) && icon_map[p_theme_type].has(p_name));
}

void Theme::rename_icon(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!icon_map.has(p_theme_type), "Cannot rename the icon '" + String(p_old_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(icon_map[p_theme_type].has(p_name), "Cannot rename the icon '" + String(p_old_name) + "' because the new name '" + String(p_name) + "' already exists.");
	ERR_FAIL_COND_MSG(!icon_map[p_theme_type].has(p_old_name), "Cannot rename the icon '" + String(p_old_name) + "' because it does not exist.");

	icon_map[p_theme_type][p_name] = icon_map[p_theme_type][p_old_name];
	icon_map[p_theme_type].erase(p_old_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::clear_icon(const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!icon_map.has(p_theme_type), "Cannot clear the icon '" + String(p_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(!icon_map[p_theme_type].has(p_name), "Cannot clear the icon '" + String(p_name) + "' because it does not exist.");

	if (icon_map[p_theme_type][p_name].is_valid()) {
		icon_map[p_theme_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	icon_map[p_theme_type].erase(p_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::get_icon_list(StringName p_theme_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!icon_map.has(p_theme_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = icon_map[p_theme_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::add_icon_type(const StringName &p_theme_type) {
	icon_map[p_theme_type] = HashMap<StringName, Ref<Texture2D>>();
}

void Theme::get_icon_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = icon_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_stylebox(const StringName &p_name, const StringName &p_theme_type, const Ref<StyleBox> &p_style) {
	bool new_value = !style_map.has(p_theme_type) || !style_map[p_theme_type].has(p_name);

	if (style_map[p_theme_type].has(p_name) && style_map[p_theme_type][p_name].is_valid()) {
		style_map[p_theme_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	style_map[p_theme_type][p_name] = p_style;

	if (p_style.is_valid()) {
		style_map[p_theme_type][p_name]->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	if (new_value) {
		notify_property_list_changed();
	}
	emit_changed();
}

Ref<StyleBox> Theme::get_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (style_map.has(p_theme_type) && style_map[p_theme_type].has(p_name) && style_map[p_theme_type][p_name].is_valid()) {
		return style_map[p_theme_type][p_name];
	} else {
		return default_style;
	}
}

bool Theme::has_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	return (style_map.has(p_theme_type) && style_map[p_theme_type].has(p_name) && style_map[p_theme_type][p_name].is_valid());
}

bool Theme::has_stylebox_nocheck(const StringName &p_name, const StringName &p_theme_type) const {
	return (style_map.has(p_theme_type) && style_map[p_theme_type].has(p_name));
}

void Theme::rename_stylebox(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!style_map.has(p_theme_type), "Cannot rename the stylebox '" + String(p_old_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(style_map[p_theme_type].has(p_name), "Cannot rename the stylebox '" + String(p_old_name) + "' because the new name '" + String(p_name) + "' already exists.");
	ERR_FAIL_COND_MSG(!style_map[p_theme_type].has(p_old_name), "Cannot rename the stylebox '" + String(p_old_name) + "' because it does not exist.");

	style_map[p_theme_type][p_name] = style_map[p_theme_type][p_old_name];
	style_map[p_theme_type].erase(p_old_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::clear_stylebox(const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!style_map.has(p_theme_type), "Cannot clear the stylebox '" + String(p_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(!style_map[p_theme_type].has(p_name), "Cannot clear the stylebox '" + String(p_name) + "' because it does not exist.");

	if (style_map[p_theme_type][p_name].is_valid()) {
		style_map[p_theme_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	style_map[p_theme_type].erase(p_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::get_stylebox_list(StringName p_theme_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!style_map.has(p_theme_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = style_map[p_theme_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::add_stylebox_type(const StringName &p_theme_type) {
	style_map[p_theme_type] = HashMap<StringName, Ref<StyleBox>>();
}

void Theme::get_stylebox_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = style_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_font(const StringName &p_name, const StringName &p_theme_type, const Ref<Font> &p_font) {
	bool new_value = !font_map.has(p_theme_type) || !font_map[p_theme_type].has(p_name);

	if (font_map[p_theme_type][p_name].is_valid()) {
		font_map[p_theme_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	font_map[p_theme_type][p_name] = p_font;

	if (p_font.is_valid()) {
		font_map[p_theme_type][p_name]->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	if (new_value) {
		notify_property_list_changed();
		emit_changed();
	}
}

Ref<Font> Theme::get_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (font_map.has(p_theme_type) && font_map[p_theme_type].has(p_name) && font_map[p_theme_type][p_name].is_valid()) {
		return font_map[p_theme_type][p_name];
	} else if (default_theme_font.is_valid()) {
		return default_theme_font;
	} else {
		return default_font;
	}
}

bool Theme::has_font(const StringName &p_name, const StringName &p_theme_type) const {
	return ((font_map.has(p_theme_type) && font_map[p_theme_type].has(p_name) && font_map[p_theme_type][p_name].is_valid()) || default_theme_font.is_valid());
}

bool Theme::has_font_nocheck(const StringName &p_name, const StringName &p_theme_type) const {
	return (font_map.has(p_theme_type) && font_map[p_theme_type].has(p_name));
}

void Theme::rename_font(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!font_map.has(p_theme_type), "Cannot rename the font '" + String(p_old_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(font_map[p_theme_type].has(p_name), "Cannot rename the font '" + String(p_old_name) + "' because the new name '" + String(p_name) + "' already exists.");
	ERR_FAIL_COND_MSG(!font_map[p_theme_type].has(p_old_name), "Cannot rename the font '" + String(p_old_name) + "' because it does not exist.");

	font_map[p_theme_type][p_name] = font_map[p_theme_type][p_old_name];
	font_map[p_theme_type].erase(p_old_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::clear_font(const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!font_map.has(p_theme_type), "Cannot clear the font '" + String(p_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(!font_map[p_theme_type].has(p_name), "Cannot clear the font '" + String(p_name) + "' because it does not exist.");

	if (font_map[p_theme_type][p_name].is_valid()) {
		font_map[p_theme_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	font_map[p_theme_type].erase(p_name);
	notify_property_list_changed();
	emit_changed();
}

void Theme::get_font_list(StringName p_theme_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!font_map.has(p_theme_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = font_map[p_theme_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::add_font_type(const StringName &p_theme_type) {
	font_map[p_theme_type] = HashMap<StringName, Ref<Font>>();
}

void Theme::get_font_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = font_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_font_size(const StringName &p_name, const StringName &p_theme_type, int p_font_size) {
	bool new_value = !font_size_map.has(p_theme_type) || !font_size_map[p_theme_type].has(p_name);

	font_size_map[p_theme_type][p_name] = p_font_size;

	if (new_value) {
		notify_property_list_changed();
		emit_changed();
	}
}

int Theme::get_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (font_size_map.has(p_theme_type) && font_size_map[p_theme_type].has(p_name) && (font_size_map[p_theme_type][p_name] > 0)) {
		return font_size_map[p_theme_type][p_name];
	} else if (default_theme_font_size > 0) {
		return default_theme_font_size;
	} else {
		return default_font_size;
	}
}

bool Theme::has_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	return ((font_size_map.has(p_theme_type) && font_size_map[p_theme_type].has(p_name) && (font_size_map[p_theme_type][p_name] > 0)) || (default_theme_font_size > 0));
}

bool Theme::has_font_size_nocheck(const StringName &p_name, const StringName &p_theme_type) const {
	return (font_size_map.has(p_theme_type) && font_size_map[p_theme_type].has(p_name));
}

void Theme::rename_font_size(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!font_size_map.has(p_theme_type), "Cannot rename the font size '" + String(p_old_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(font_size_map[p_theme_type].has(p_name), "Cannot rename the font size '" + String(p_old_name) + "' because the new name '" + String(p_name) + "' already exists.");
	ERR_FAIL_COND_MSG(!font_size_map[p_theme_type].has(p_old_name), "Cannot rename the font size '" + String(p_old_name) + "' because it does not exist.");

	font_size_map[p_theme_type][p_name] = font_size_map[p_theme_type][p_old_name];
	font_size_map[p_theme_type].erase(p_old_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::clear_font_size(const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!font_size_map.has(p_theme_type), "Cannot clear the font size '" + String(p_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(!font_size_map[p_theme_type].has(p_name), "Cannot clear the font size '" + String(p_name) + "' because it does not exist.");

	font_size_map[p_theme_type].erase(p_name);
	notify_property_list_changed();
	emit_changed();
}

void Theme::get_font_size_list(StringName p_theme_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!font_size_map.has(p_theme_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = font_size_map[p_theme_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::add_font_size_type(const StringName &p_theme_type) {
	font_size_map[p_theme_type] = HashMap<StringName, int>();
}

void Theme::get_font_size_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = font_size_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_color(const StringName &p_name, const StringName &p_theme_type, const Color &p_color) {
	bool new_value = !color_map.has(p_theme_type) || !color_map[p_theme_type].has(p_name);

	color_map[p_theme_type][p_name] = p_color;

	if (new_value) {
		notify_property_list_changed();
		emit_changed();
	}
}

Color Theme::get_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (color_map.has(p_theme_type) && color_map[p_theme_type].has(p_name)) {
		return color_map[p_theme_type][p_name];
	} else {
		return Color();
	}
}

bool Theme::has_color(const StringName &p_name, const StringName &p_theme_type) const {
	return (color_map.has(p_theme_type) && color_map[p_theme_type].has(p_name));
}

bool Theme::has_color_nocheck(const StringName &p_name, const StringName &p_theme_type) const {
	return (color_map.has(p_theme_type) && color_map[p_theme_type].has(p_name));
}

void Theme::rename_color(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!color_map.has(p_theme_type), "Cannot rename the color '" + String(p_old_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(color_map[p_theme_type].has(p_name), "Cannot rename the color '" + String(p_old_name) + "' because the new name '" + String(p_name) + "' already exists.");
	ERR_FAIL_COND_MSG(!color_map[p_theme_type].has(p_old_name), "Cannot rename the color '" + String(p_old_name) + "' because it does not exist.");

	color_map[p_theme_type][p_name] = color_map[p_theme_type][p_old_name];
	color_map[p_theme_type].erase(p_old_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::clear_color(const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!color_map.has(p_theme_type), "Cannot clear the color '" + String(p_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(!color_map[p_theme_type].has(p_name), "Cannot clear the color '" + String(p_name) + "' because it does not exist.");

	color_map[p_theme_type].erase(p_name);
	notify_property_list_changed();
	emit_changed();
}

void Theme::get_color_list(StringName p_theme_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!color_map.has(p_theme_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = color_map[p_theme_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::add_color_type(const StringName &p_theme_type) {
	color_map[p_theme_type] = HashMap<StringName, Color>();
}

void Theme::get_color_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = color_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_constant(const StringName &p_name, const StringName &p_theme_type, int p_constant) {
	bool new_value = !constant_map.has(p_theme_type) || !constant_map[p_theme_type].has(p_name);
	constant_map[p_theme_type][p_name] = p_constant;

	if (new_value) {
		notify_property_list_changed();
		emit_changed();
	}
}

int Theme::get_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (constant_map.has(p_theme_type) && constant_map[p_theme_type].has(p_name)) {
		return constant_map[p_theme_type][p_name];
	} else {
		return 0;
	}
}

bool Theme::has_constant(const StringName &p_name, const StringName &p_theme_type) const {
	return (constant_map.has(p_theme_type) && constant_map[p_theme_type].has(p_name));
}

bool Theme::has_constant_nocheck(const StringName &p_name, const StringName &p_theme_type) const {
	return (constant_map.has(p_theme_type) && constant_map[p_theme_type].has(p_name));
}

void Theme::rename_constant(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!constant_map.has(p_theme_type), "Cannot rename the constant '" + String(p_old_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(constant_map[p_theme_type].has(p_name), "Cannot rename the constant '" + String(p_old_name) + "' because the new name '" + String(p_name) + "' already exists.");
	ERR_FAIL_COND_MSG(!constant_map[p_theme_type].has(p_old_name), "Cannot rename the constant '" + String(p_old_name) + "' because it does not exist.");

	constant_map[p_theme_type][p_name] = constant_map[p_theme_type][p_old_name];
	constant_map[p_theme_type].erase(p_old_name);

	notify_property_list_changed();
	emit_changed();
}

void Theme::clear_constant(const StringName &p_name, const StringName &p_theme_type) {
	ERR_FAIL_COND_MSG(!constant_map.has(p_theme_type), "Cannot clear the constant '" + String(p_name) + "' because the node type '" + String(p_theme_type) + "' does not exist.");
	ERR_FAIL_COND_MSG(!constant_map[p_theme_type].has(p_name), "Cannot clear the constant '" + String(p_name) + "' because it does not exist.");

	constant_map[p_theme_type].erase(p_name);
	notify_property_list_changed();
	emit_changed();
}

void Theme::get_constant_list(StringName p_theme_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!constant_map.has(p_theme_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = constant_map[p_theme_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::add_constant_type(const StringName &p_theme_type) {
	constant_map[p_theme_type] = HashMap<StringName, int>();
}

void Theme::get_constant_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = constant_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_theme_item(DataType p_data_type, const StringName &p_name, const StringName &p_theme_type, const Variant &p_value) {
	switch (p_data_type) {
		case DATA_TYPE_COLOR: {
			ERR_FAIL_COND_MSG(p_value.get_type() != Variant::COLOR, "Theme item's data type (Color) does not match Variant's type (" + Variant::get_type_name(p_value.get_type()) + ").");

			Color color_value = p_value;
			set_color(p_name, p_theme_type, color_value);
		} break;
		case DATA_TYPE_CONSTANT: {
			ERR_FAIL_COND_MSG(p_value.get_type() != Variant::INT, "Theme item's data type (int) does not match Variant's type (" + Variant::get_type_name(p_value.get_type()) + ").");

			int constant_value = p_value;
			set_constant(p_name, p_theme_type, constant_value);
		} break;
		case DATA_TYPE_FONT: {
			ERR_FAIL_COND_MSG(p_value.get_type() != Variant::OBJECT, "Theme item's data type (Object) does not match Variant's type (" + Variant::get_type_name(p_value.get_type()) + ").");

			Ref<Font> font_value = Object::cast_to<Font>(p_value.get_validated_object());
			set_font(p_name, p_theme_type, font_value);
		} break;
		case DATA_TYPE_FONT_SIZE: {
			ERR_FAIL_COND_MSG(p_value.get_type() != Variant::INT, "Theme item's data type (int) does not match Variant's type (" + Variant::get_type_name(p_value.get_type()) + ").");

			int font_size_value = p_value;
			set_font_size(p_name, p_theme_type, font_size_value);
		} break;
		case DATA_TYPE_ICON: {
			ERR_FAIL_COND_MSG(p_value.get_type() != Variant::OBJECT, "Theme item's data type (Object) does not match Variant's type (" + Variant::get_type_name(p_value.get_type()) + ").");

			Ref<Texture2D> icon_value = Object::cast_to<Texture2D>(p_value.get_validated_object());
			set_icon(p_name, p_theme_type, icon_value);
		} break;
		case DATA_TYPE_STYLEBOX: {
			ERR_FAIL_COND_MSG(p_value.get_type() != Variant::OBJECT, "Theme item's data type (Object) does not match Variant's type (" + Variant::get_type_name(p_value.get_type()) + ").");

			Ref<StyleBox> stylebox_value = Object::cast_to<StyleBox>(p_value.get_validated_object());
			set_stylebox(p_name, p_theme_type, stylebox_value);
		} break;
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

Variant Theme::get_theme_item(DataType p_data_type, const StringName &p_name, const StringName &p_theme_type) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			return get_color(p_name, p_theme_type);
		case DATA_TYPE_CONSTANT:
			return get_constant(p_name, p_theme_type);
		case DATA_TYPE_FONT:
			return get_font(p_name, p_theme_type);
		case DATA_TYPE_FONT_SIZE:
			return get_font_size(p_name, p_theme_type);
		case DATA_TYPE_ICON:
			return get_icon(p_name, p_theme_type);
		case DATA_TYPE_STYLEBOX:
			return get_stylebox(p_name, p_theme_type);
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	return Variant();
}

bool Theme::has_theme_item(DataType p_data_type, const StringName &p_name, const StringName &p_theme_type) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			return has_color(p_name, p_theme_type);
		case DATA_TYPE_CONSTANT:
			return has_constant(p_name, p_theme_type);
		case DATA_TYPE_FONT:
			return has_font(p_name, p_theme_type);
		case DATA_TYPE_FONT_SIZE:
			return has_font_size(p_name, p_theme_type);
		case DATA_TYPE_ICON:
			return has_icon(p_name, p_theme_type);
		case DATA_TYPE_STYLEBOX:
			return has_stylebox(p_name, p_theme_type);
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	return false;
}

bool Theme::has_theme_item_nocheck(DataType p_data_type, const StringName &p_name, const StringName &p_theme_type) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			return has_color_nocheck(p_name, p_theme_type);
		case DATA_TYPE_CONSTANT:
			return has_constant_nocheck(p_name, p_theme_type);
		case DATA_TYPE_FONT:
			return has_font_nocheck(p_name, p_theme_type);
		case DATA_TYPE_FONT_SIZE:
			return has_font_size_nocheck(p_name, p_theme_type);
		case DATA_TYPE_ICON:
			return has_icon_nocheck(p_name, p_theme_type);
		case DATA_TYPE_STYLEBOX:
			return has_stylebox_nocheck(p_name, p_theme_type);
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	return false;
}

void Theme::rename_theme_item(DataType p_data_type, const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type) {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			rename_color(p_old_name, p_name, p_theme_type);
			break;
		case DATA_TYPE_CONSTANT:
			rename_constant(p_old_name, p_name, p_theme_type);
			break;
		case DATA_TYPE_FONT:
			rename_font(p_old_name, p_name, p_theme_type);
			break;
		case DATA_TYPE_FONT_SIZE:
			rename_font_size(p_old_name, p_name, p_theme_type);
			break;
		case DATA_TYPE_ICON:
			rename_icon(p_old_name, p_name, p_theme_type);
			break;
		case DATA_TYPE_STYLEBOX:
			rename_stylebox(p_old_name, p_name, p_theme_type);
			break;
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

void Theme::clear_theme_item(DataType p_data_type, const StringName &p_name, const StringName &p_theme_type) {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			clear_color(p_name, p_theme_type);
			break;
		case DATA_TYPE_CONSTANT:
			clear_constant(p_name, p_theme_type);
			break;
		case DATA_TYPE_FONT:
			clear_font(p_name, p_theme_type);
			break;
		case DATA_TYPE_FONT_SIZE:
			clear_font_size(p_name, p_theme_type);
			break;
		case DATA_TYPE_ICON:
			clear_icon(p_name, p_theme_type);
			break;
		case DATA_TYPE_STYLEBOX:
			clear_stylebox(p_name, p_theme_type);
			break;
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

void Theme::get_theme_item_list(DataType p_data_type, StringName p_theme_type, List<StringName> *p_list) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			get_color_list(p_theme_type, p_list);
			break;
		case DATA_TYPE_CONSTANT:
			get_constant_list(p_theme_type, p_list);
			break;
		case DATA_TYPE_FONT:
			get_font_list(p_theme_type, p_list);
			break;
		case DATA_TYPE_FONT_SIZE:
			get_font_size_list(p_theme_type, p_list);
			break;
		case DATA_TYPE_ICON:
			get_icon_list(p_theme_type, p_list);
			break;
		case DATA_TYPE_STYLEBOX:
			get_stylebox_list(p_theme_type, p_list);
			break;
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

void Theme::add_theme_item_type(DataType p_data_type, const StringName &p_theme_type) {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			add_color_type(p_theme_type);
			break;
		case DATA_TYPE_CONSTANT:
			add_constant_type(p_theme_type);
			break;
		case DATA_TYPE_FONT:
			add_font_type(p_theme_type);
			break;
		case DATA_TYPE_FONT_SIZE:
			add_font_size_type(p_theme_type);
			break;
		case DATA_TYPE_ICON:
			add_icon_type(p_theme_type);
			break;
		case DATA_TYPE_STYLEBOX:
			add_stylebox_type(p_theme_type);
			break;
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

void Theme::get_theme_item_type_list(DataType p_data_type, List<StringName> *p_list) const {
	switch (p_data_type) {
		case DATA_TYPE_COLOR:
			get_color_type_list(p_list);
			break;
		case DATA_TYPE_CONSTANT:
			get_constant_type_list(p_list);
			break;
		case DATA_TYPE_FONT:
			get_font_type_list(p_list);
			break;
		case DATA_TYPE_FONT_SIZE:
			get_font_size_type_list(p_list);
			break;
		case DATA_TYPE_ICON:
			get_icon_type_list(p_list);
			break;
		case DATA_TYPE_STYLEBOX:
			get_stylebox_type_list(p_list);
			break;
		case DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

void Theme::clear() {
	//these need disconnecting
	{
		const StringName *K = nullptr;
		while ((K = icon_map.next(K))) {
			const StringName *L = nullptr;
			while ((L = icon_map[*K].next(L))) {
				Ref<Texture2D> icon = icon_map[*K][*L];
				if (icon.is_valid()) {
					icon->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
				}
			}
		}
	}

	{
		const StringName *K = nullptr;
		while ((K = style_map.next(K))) {
			const StringName *L = nullptr;
			while ((L = style_map[*K].next(L))) {
				Ref<StyleBox> style = style_map[*K][*L];
				if (style.is_valid()) {
					style->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
				}
			}
		}
	}

	{
		const StringName *K = nullptr;
		while ((K = font_map.next(K))) {
			const StringName *L = nullptr;
			while ((L = font_map[*K].next(L))) {
				Ref<Font> font = font_map[*K][*L];
				if (font.is_valid()) {
					font->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
				}
			}
		}
	}

	icon_map.clear();
	style_map.clear();
	font_map.clear();
	color_map.clear();
	constant_map.clear();

	notify_property_list_changed();
	emit_changed();
}

void Theme::copy_default_theme() {
	Ref<Theme> default_theme2 = get_default();
	copy_theme(default_theme2);
}

void Theme::copy_theme(const Ref<Theme> &p_other) {
	if (p_other.is_null()) {
		clear();
		return;
	}

	// These items need reconnecting, so add them normally.
	{
		const StringName *K = nullptr;
		while ((K = p_other->icon_map.next(K))) {
			const StringName *L = nullptr;
			while ((L = p_other->icon_map[*K].next(L))) {
				set_icon(*L, *K, p_other->icon_map[*K][*L]);
			}
		}
	}

	{
		const StringName *K = nullptr;
		while ((K = p_other->style_map.next(K))) {
			const StringName *L = nullptr;
			while ((L = p_other->style_map[*K].next(L))) {
				set_stylebox(*L, *K, p_other->style_map[*K][*L]);
			}
		}
	}

	{
		const StringName *K = nullptr;
		while ((K = p_other->font_map.next(K))) {
			const StringName *L = nullptr;
			while ((L = p_other->font_map[*K].next(L))) {
				set_font(*L, *K, p_other->font_map[*K][*L]);
			}
		}
	}

	// These items can be simply copied.
	font_size_map = p_other->font_size_map;
	color_map = p_other->color_map;
	constant_map = p_other->constant_map;

	notify_property_list_changed();
	emit_changed();
}

void Theme::get_type_list(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	Set<StringName> types;
	const StringName *key = nullptr;

	while ((key = icon_map.next(key))) {
		types.insert(*key);
	}

	key = nullptr;

	while ((key = style_map.next(key))) {
		types.insert(*key);
	}

	key = nullptr;

	while ((key = font_map.next(key))) {
		types.insert(*key);
	}

	key = nullptr;

	while ((key = color_map.next(key))) {
		types.insert(*key);
	}

	key = nullptr;

	while ((key = constant_map.next(key))) {
		types.insert(*key);
	}

	for (Set<StringName>::Element *E = types.front(); E; E = E->next()) {
		p_list->push_back(E->get());
	}
}

void Theme::get_type_dependencies(const StringName &p_theme_type, List<StringName> *p_list) {
	ERR_FAIL_NULL(p_list);

	StringName class_name = p_theme_type;
	while (class_name != StringName()) {
		p_list->push_back(class_name);
		class_name = ClassDB::get_parent_class_nocheck(class_name);
	}
}

void Theme::reset_state() {
	clear();
}
void Theme::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_icon", "name", "theme_type", "texture"), &Theme::set_icon);
	ClassDB::bind_method(D_METHOD("get_icon", "name", "theme_type"), &Theme::get_icon);
	ClassDB::bind_method(D_METHOD("has_icon", "name", "theme_type"), &Theme::has_icon);
	ClassDB::bind_method(D_METHOD("rename_icon", "old_name", "name", "theme_type"), &Theme::rename_icon);
	ClassDB::bind_method(D_METHOD("clear_icon", "name", "theme_type"), &Theme::clear_icon);
	ClassDB::bind_method(D_METHOD("get_icon_list", "theme_type"), &Theme::_get_icon_list);
	ClassDB::bind_method(D_METHOD("get_icon_type_list"), &Theme::_get_icon_type_list);

	ClassDB::bind_method(D_METHOD("set_stylebox", "name", "theme_type", "texture"), &Theme::set_stylebox);
	ClassDB::bind_method(D_METHOD("get_stylebox", "name", "theme_type"), &Theme::get_stylebox);
	ClassDB::bind_method(D_METHOD("has_stylebox", "name", "theme_type"), &Theme::has_stylebox);
	ClassDB::bind_method(D_METHOD("rename_stylebox", "old_name", "name", "theme_type"), &Theme::rename_stylebox);
	ClassDB::bind_method(D_METHOD("clear_stylebox", "name", "theme_type"), &Theme::clear_stylebox);
	ClassDB::bind_method(D_METHOD("get_stylebox_list", "theme_type"), &Theme::_get_stylebox_list);
	ClassDB::bind_method(D_METHOD("get_stylebox_type_list"), &Theme::_get_stylebox_type_list);

	ClassDB::bind_method(D_METHOD("set_font", "name", "theme_type", "font"), &Theme::set_font);
	ClassDB::bind_method(D_METHOD("get_font", "name", "theme_type"), &Theme::get_font);
	ClassDB::bind_method(D_METHOD("has_font", "name", "theme_type"), &Theme::has_font);
	ClassDB::bind_method(D_METHOD("rename_font", "old_name", "name", "theme_type"), &Theme::rename_font);
	ClassDB::bind_method(D_METHOD("clear_font", "name", "theme_type"), &Theme::clear_font);
	ClassDB::bind_method(D_METHOD("get_font_list", "theme_type"), &Theme::_get_font_list);
	ClassDB::bind_method(D_METHOD("get_font_type_list"), &Theme::_get_font_type_list);

	ClassDB::bind_method(D_METHOD("set_font_size", "name", "theme_type", "font_size"), &Theme::set_font_size);
	ClassDB::bind_method(D_METHOD("get_font_size", "name", "theme_type"), &Theme::get_font_size);
	ClassDB::bind_method(D_METHOD("has_font_size", "name", "theme_type"), &Theme::has_font_size);
	ClassDB::bind_method(D_METHOD("rename_font_size", "old_name", "name", "theme_type"), &Theme::rename_font_size);
	ClassDB::bind_method(D_METHOD("clear_font_size", "name", "theme_type"), &Theme::clear_font_size);
	ClassDB::bind_method(D_METHOD("get_font_size_list", "theme_type"), &Theme::_get_font_size_list);
	ClassDB::bind_method(D_METHOD("get_font_size_type_list"), &Theme::_get_font_size_type_list);

	ClassDB::bind_method(D_METHOD("set_color", "name", "theme_type", "color"), &Theme::set_color);
	ClassDB::bind_method(D_METHOD("get_color", "name", "theme_type"), &Theme::get_color);
	ClassDB::bind_method(D_METHOD("has_color", "name", "theme_type"), &Theme::has_color);
	ClassDB::bind_method(D_METHOD("rename_color", "old_name", "name", "theme_type"), &Theme::rename_color);
	ClassDB::bind_method(D_METHOD("clear_color", "name", "theme_type"), &Theme::clear_color);
	ClassDB::bind_method(D_METHOD("get_color_list", "theme_type"), &Theme::_get_color_list);
	ClassDB::bind_method(D_METHOD("get_color_type_list"), &Theme::_get_color_type_list);

	ClassDB::bind_method(D_METHOD("set_constant", "name", "theme_type", "constant"), &Theme::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant", "name", "theme_type"), &Theme::get_constant);
	ClassDB::bind_method(D_METHOD("has_constant", "name", "theme_type"), &Theme::has_constant);
	ClassDB::bind_method(D_METHOD("rename_constant", "old_name", "name", "theme_type"), &Theme::rename_constant);
	ClassDB::bind_method(D_METHOD("clear_constant", "name", "theme_type"), &Theme::clear_constant);
	ClassDB::bind_method(D_METHOD("get_constant_list", "theme_type"), &Theme::_get_constant_list);
	ClassDB::bind_method(D_METHOD("get_constant_type_list"), &Theme::_get_constant_type_list);

	ClassDB::bind_method(D_METHOD("clear"), &Theme::clear);

	ClassDB::bind_method(D_METHOD("set_default_font", "font"), &Theme::set_default_theme_font);
	ClassDB::bind_method(D_METHOD("get_default_font"), &Theme::get_default_theme_font);

	ClassDB::bind_method(D_METHOD("set_default_font_size", "font_size"), &Theme::set_default_theme_font_size);
	ClassDB::bind_method(D_METHOD("get_default_font_size"), &Theme::get_default_theme_font_size);

	ClassDB::bind_method(D_METHOD("set_theme_item", "data_type", "name", "theme_type", "value"), &Theme::set_theme_item);
	ClassDB::bind_method(D_METHOD("get_theme_item", "data_type", "name", "theme_type"), &Theme::get_theme_item);
	ClassDB::bind_method(D_METHOD("has_theme_item", "data_type", "name", "theme_type"), &Theme::has_theme_item);
	ClassDB::bind_method(D_METHOD("rename_theme_item", "data_type", "old_name", "name", "theme_type"), &Theme::rename_theme_item);
	ClassDB::bind_method(D_METHOD("clear_theme_item", "data_type", "name", "theme_type"), &Theme::clear_theme_item);
	ClassDB::bind_method(D_METHOD("get_theme_item_list", "data_type", "theme_type"), &Theme::_get_theme_item_list);
	ClassDB::bind_method(D_METHOD("get_theme_item_type_list", "data_type"), &Theme::_get_theme_item_type_list);

	ClassDB::bind_method(D_METHOD("get_type_list"), &Theme::_get_type_list);

	ClassDB::bind_method("copy_default_theme", &Theme::copy_default_theme);
	ClassDB::bind_method(D_METHOD("copy_theme", "other"), &Theme::copy_theme);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "default_font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_default_font", "get_default_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_font_size"), "set_default_font_size", "get_default_font_size");

	BIND_ENUM_CONSTANT(DATA_TYPE_COLOR);
	BIND_ENUM_CONSTANT(DATA_TYPE_CONSTANT);
	BIND_ENUM_CONSTANT(DATA_TYPE_FONT);
	BIND_ENUM_CONSTANT(DATA_TYPE_FONT_SIZE);
	BIND_ENUM_CONSTANT(DATA_TYPE_ICON);
	BIND_ENUM_CONSTANT(DATA_TYPE_STYLEBOX);
	BIND_ENUM_CONSTANT(DATA_TYPE_MAX);
}

Theme::Theme() {
}

Theme::~Theme() {
}
