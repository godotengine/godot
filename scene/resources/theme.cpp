/*************************************************************************/
/*  theme.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/print_string.h"

void Theme::_emit_theme_changed() {
	emit_changed();
}

Vector<String> Theme::_get_icon_list(const String &p_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_icon_list(p_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_stylebox_list(const String &p_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_stylebox_list(p_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_stylebox_types() const {
	Vector<String> ilret;
	List<StringName> il;

	get_stylebox_types(&il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_font_list(const String &p_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_font_list(p_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_color_list(const String &p_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_color_list(p_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_constant_list(const String &p_type) const {
	Vector<String> ilret;
	List<StringName> il;

	get_constant_list(p_type, &il);
	ilret.resize(il.size());

	int i = 0;
	String *w = ilret.ptrw();
	for (List<StringName>::Element *E = il.front(); E; E = E->next(), i++) {
		w[i] = E->get();
	}
	return ilret;
}

Vector<String> Theme::_get_type_list(const String &p_type) const {
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
		String node_type = sname.get_slicec('/', 0);
		String name = sname.get_slicec('/', 2);

		if (type == "icons") {
			set_icon(name, node_type, p_value);
		} else if (type == "styles") {
			set_stylebox(name, node_type, p_value);
		} else if (type == "fonts") {
			set_font(name, node_type, p_value);
		} else if (type == "colors") {
			set_color(name, node_type, p_value);
		} else if (type == "constants") {
			set_constant(name, node_type, p_value);
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
		String node_type = sname.get_slicec('/', 0);
		String name = sname.get_slicec('/', 2);

		if (type == "icons") {
			if (!has_icon(name, node_type)) {
				r_ret = Ref<Texture2D>();
			} else {
				r_ret = get_icon(name, node_type);
			}
		} else if (type == "styles") {
			if (!has_stylebox(name, node_type)) {
				r_ret = Ref<StyleBox>();
			} else {
				r_ret = get_stylebox(name, node_type);
			}
		} else if (type == "fonts") {
			if (!has_font(name, node_type)) {
				r_ret = Ref<Font>();
			} else {
				r_ret = get_font(name, node_type);
			}
		} else if (type == "colors") {
			r_ret = get_color(name, node_type);
		} else if (type == "constants") {
			r_ret = get_constant(name, node_type);
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

	_change_notify();
	emit_changed();
}

Ref<Font> Theme::get_default_theme_font() const {
	return default_theme_font;
}

Ref<Theme> Theme::project_default_theme;
Ref<Theme> Theme::default_theme;
Ref<Texture2D> Theme::default_icon;
Ref<StyleBox> Theme::default_style;
Ref<Font> Theme::default_font;

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

void Theme::set_icon(const StringName &p_name, const StringName &p_type, const Ref<Texture2D> &p_icon) {
	//ERR_FAIL_COND(p_icon.is_null());

	bool new_value = !icon_map.has(p_type) || !icon_map[p_type].has(p_name);

	if (icon_map[p_type].has(p_name) && icon_map[p_type][p_name].is_valid()) {
		icon_map[p_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	icon_map[p_type][p_name] = p_icon;

	if (p_icon.is_valid()) {
		icon_map[p_type][p_name]->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}

Ref<Texture2D> Theme::get_icon(const StringName &p_name, const StringName &p_type) const {
	if (icon_map.has(p_type) && icon_map[p_type].has(p_name) && icon_map[p_type][p_name].is_valid()) {
		return icon_map[p_type][p_name];
	} else {
		return default_icon;
	}
}

bool Theme::has_icon(const StringName &p_name, const StringName &p_type) const {
	return (icon_map.has(p_type) && icon_map[p_type].has(p_name) && icon_map[p_type][p_name].is_valid());
}

void Theme::clear_icon(const StringName &p_name, const StringName &p_type) {
	ERR_FAIL_COND(!icon_map.has(p_type));
	ERR_FAIL_COND(!icon_map[p_type].has(p_name));

	if (icon_map[p_type][p_name].is_valid()) {
		icon_map[p_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	icon_map[p_type].erase(p_name);

	_change_notify();
	emit_changed();
}

void Theme::get_icon_list(StringName p_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!icon_map.has(p_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = icon_map[p_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_shader(const StringName &p_name, const StringName &p_type, const Ref<Shader> &p_shader) {
	bool new_value = !shader_map.has(p_type) || !shader_map[p_type].has(p_name);

	shader_map[p_type][p_name] = p_shader;

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}

Ref<Shader> Theme::get_shader(const StringName &p_name, const StringName &p_type) const {
	if (shader_map.has(p_type) && shader_map[p_type].has(p_name) && shader_map[p_type][p_name].is_valid()) {
		return shader_map[p_type][p_name];
	} else {
		return nullptr;
	}
}

bool Theme::has_shader(const StringName &p_name, const StringName &p_type) const {
	return (shader_map.has(p_type) && shader_map[p_type].has(p_name) && shader_map[p_type][p_name].is_valid());
}

void Theme::clear_shader(const StringName &p_name, const StringName &p_type) {
	ERR_FAIL_COND(!shader_map.has(p_type));
	ERR_FAIL_COND(!shader_map[p_type].has(p_name));

	shader_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_shader_list(const StringName &p_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!shader_map.has(p_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = shader_map[p_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_stylebox(const StringName &p_name, const StringName &p_type, const Ref<StyleBox> &p_style) {
	//ERR_FAIL_COND(p_style.is_null());

	bool new_value = !style_map.has(p_type) || !style_map[p_type].has(p_name);

	if (style_map[p_type].has(p_name) && style_map[p_type][p_name].is_valid()) {
		style_map[p_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	style_map[p_type][p_name] = p_style;

	if (p_style.is_valid()) {
		style_map[p_type][p_name]->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	if (new_value) {
		_change_notify();
	}
	emit_changed();
}

Ref<StyleBox> Theme::get_stylebox(const StringName &p_name, const StringName &p_type) const {
	if (style_map.has(p_type) && style_map[p_type].has(p_name) && style_map[p_type][p_name].is_valid()) {
		return style_map[p_type][p_name];
	} else {
		return default_style;
	}
}

bool Theme::has_stylebox(const StringName &p_name, const StringName &p_type) const {
	return (style_map.has(p_type) && style_map[p_type].has(p_name) && style_map[p_type][p_name].is_valid());
}

void Theme::clear_stylebox(const StringName &p_name, const StringName &p_type) {
	ERR_FAIL_COND(!style_map.has(p_type));
	ERR_FAIL_COND(!style_map[p_type].has(p_name));

	if (style_map[p_type][p_name].is_valid()) {
		style_map[p_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	style_map[p_type].erase(p_name);

	_change_notify();
	emit_changed();
}

void Theme::get_stylebox_list(StringName p_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!style_map.has(p_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = style_map[p_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::get_stylebox_types(List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	const StringName *key = nullptr;
	while ((key = style_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_font(const StringName &p_name, const StringName &p_type, const Ref<Font> &p_font) {
	//ERR_FAIL_COND(p_font.is_null());

	bool new_value = !font_map.has(p_type) || !font_map[p_type].has(p_name);

	if (font_map[p_type][p_name].is_valid()) {
		font_map[p_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	font_map[p_type][p_name] = p_font;

	if (p_font.is_valid()) {
		font_map[p_type][p_name]->connect("changed", callable_mp(this, &Theme::_emit_theme_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}

Ref<Font> Theme::get_font(const StringName &p_name, const StringName &p_type) const {
	if (font_map.has(p_type) && font_map[p_type].has(p_name) && font_map[p_type][p_name].is_valid()) {
		return font_map[p_type][p_name];
	} else if (default_theme_font.is_valid()) {
		return default_theme_font;
	} else {
		return default_font;
	}
}

bool Theme::has_font(const StringName &p_name, const StringName &p_type) const {
	return (font_map.has(p_type) && font_map[p_type].has(p_name) && font_map[p_type][p_name].is_valid());
}

void Theme::clear_font(const StringName &p_name, const StringName &p_type) {
	ERR_FAIL_COND(!font_map.has(p_type));
	ERR_FAIL_COND(!font_map[p_type].has(p_name));

	if (font_map[p_type][p_name].is_valid()) {
		font_map[p_type][p_name]->disconnect("changed", callable_mp(this, &Theme::_emit_theme_changed));
	}

	font_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_font_list(StringName p_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!font_map.has(p_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = font_map[p_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_color(const StringName &p_name, const StringName &p_type, const Color &p_color) {
	bool new_value = !color_map.has(p_type) || !color_map[p_type].has(p_name);

	color_map[p_type][p_name] = p_color;

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}

Color Theme::get_color(const StringName &p_name, const StringName &p_type) const {
	if (color_map.has(p_type) && color_map[p_type].has(p_name)) {
		return color_map[p_type][p_name];
	} else {
		return Color();
	}
}

bool Theme::has_color(const StringName &p_name, const StringName &p_type) const {
	return (color_map.has(p_type) && color_map[p_type].has(p_name));
}

void Theme::clear_color(const StringName &p_name, const StringName &p_type) {
	ERR_FAIL_COND(!color_map.has(p_type));
	ERR_FAIL_COND(!color_map[p_type].has(p_name));

	color_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_color_list(StringName p_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!color_map.has(p_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = color_map[p_type].next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_constant(const StringName &p_name, const StringName &p_type, int p_constant) {
	bool new_value = !constant_map.has(p_type) || !constant_map[p_type].has(p_name);
	constant_map[p_type][p_name] = p_constant;

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}

int Theme::get_constant(const StringName &p_name, const StringName &p_type) const {
	if (constant_map.has(p_type) && constant_map[p_type].has(p_name)) {
		return constant_map[p_type][p_name];
	} else {
		return 0;
	}
}

bool Theme::has_constant(const StringName &p_name, const StringName &p_type) const {
	return (constant_map.has(p_type) && constant_map[p_type].has(p_name));
}

void Theme::clear_constant(const StringName &p_name, const StringName &p_type) {
	ERR_FAIL_COND(!constant_map.has(p_type));
	ERR_FAIL_COND(!constant_map[p_type].has(p_name));

	constant_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_constant_list(StringName p_type, List<StringName> *p_list) const {
	ERR_FAIL_NULL(p_list);

	if (!constant_map.has(p_type)) {
		return;
	}

	const StringName *key = nullptr;

	while ((key = constant_map[p_type].next(key))) {
		p_list->push_back(*key);
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
	shader_map.clear();
	color_map.clear();
	constant_map.clear();

	_change_notify();
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

	//these need reconnecting, so add normally
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

	//these are ok to just copy

	color_map = p_other->color_map;
	constant_map = p_other->constant_map;
	shader_map = p_other->shader_map;

	_change_notify();
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

void Theme::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_icon", "name", "type", "texture"), &Theme::set_icon);
	ClassDB::bind_method(D_METHOD("get_icon", "name", "type"), &Theme::get_icon);
	ClassDB::bind_method(D_METHOD("has_icon", "name", "type"), &Theme::has_icon);
	ClassDB::bind_method(D_METHOD("clear_icon", "name", "type"), &Theme::clear_icon);
	ClassDB::bind_method(D_METHOD("get_icon_list", "type"), &Theme::_get_icon_list);

	ClassDB::bind_method(D_METHOD("set_stylebox", "name", "type", "texture"), &Theme::set_stylebox);
	ClassDB::bind_method(D_METHOD("get_stylebox", "name", "type"), &Theme::get_stylebox);
	ClassDB::bind_method(D_METHOD("has_stylebox", "name", "type"), &Theme::has_stylebox);
	ClassDB::bind_method(D_METHOD("clear_stylebox", "name", "type"), &Theme::clear_stylebox);
	ClassDB::bind_method(D_METHOD("get_stylebox_list", "type"), &Theme::_get_stylebox_list);
	ClassDB::bind_method(D_METHOD("get_stylebox_types"), &Theme::_get_stylebox_types);

	ClassDB::bind_method(D_METHOD("set_font", "name", "type", "font"), &Theme::set_font);
	ClassDB::bind_method(D_METHOD("get_font", "name", "type"), &Theme::get_font);
	ClassDB::bind_method(D_METHOD("has_font", "name", "type"), &Theme::has_font);
	ClassDB::bind_method(D_METHOD("clear_font", "name", "type"), &Theme::clear_font);
	ClassDB::bind_method(D_METHOD("get_font_list", "type"), &Theme::_get_font_list);

	ClassDB::bind_method(D_METHOD("set_color", "name", "type", "color"), &Theme::set_color);
	ClassDB::bind_method(D_METHOD("get_color", "name", "type"), &Theme::get_color);
	ClassDB::bind_method(D_METHOD("has_color", "name", "type"), &Theme::has_color);
	ClassDB::bind_method(D_METHOD("clear_color", "name", "type"), &Theme::clear_color);
	ClassDB::bind_method(D_METHOD("get_color_list", "type"), &Theme::_get_color_list);

	ClassDB::bind_method(D_METHOD("set_constant", "name", "type", "constant"), &Theme::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant", "name", "type"), &Theme::get_constant);
	ClassDB::bind_method(D_METHOD("has_constant", "name", "type"), &Theme::has_constant);
	ClassDB::bind_method(D_METHOD("clear_constant", "name", "type"), &Theme::clear_constant);
	ClassDB::bind_method(D_METHOD("get_constant_list", "type"), &Theme::_get_constant_list);

	ClassDB::bind_method(D_METHOD("clear"), &Theme::clear);

	ClassDB::bind_method(D_METHOD("set_default_font", "font"), &Theme::set_default_theme_font);
	ClassDB::bind_method(D_METHOD("get_default_font"), &Theme::get_default_theme_font);

	ClassDB::bind_method(D_METHOD("get_type_list", "type"), &Theme::_get_type_list);

	ClassDB::bind_method("copy_default_theme", &Theme::copy_default_theme);
	ClassDB::bind_method(D_METHOD("copy_theme", "other"), &Theme::copy_theme);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "default_font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_default_font", "get_default_font");
}

Theme::Theme() {
}

Theme::~Theme() {
}
