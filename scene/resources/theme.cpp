/*************************************************************************/
/*  theme.cpp                                                            */
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
#include "theme.h"
#include "os/file_access.h"
#include "print_string.h"

Ref<Theme> Theme::default_theme;

void Theme::_emit_theme_changed() {

	emit_changed();
}

void Theme::_ref_font(Ref<Font> p_sc) {

	if (!font_refcount.has(p_sc)) {
		font_refcount[p_sc] = 1;
		p_sc->connect("changed", this, "_emit_theme_changed");
	} else {
		font_refcount[p_sc] += 1;
	}
}

void Theme::_unref_font(Ref<Font> p_sc) {

	ERR_FAIL_COND(!font_refcount.has(p_sc));
	font_refcount[p_sc]--;
	if (font_refcount[p_sc] == 0) {
		p_sc->disconnect("changed", this, "_emit_theme_changed");
		font_refcount.erase(p_sc);
	}
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
		} else
			return false;

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

			if (!has_icon(name, node_type))
				r_ret = Ref<Texture>();
			else
				r_ret = get_icon(name, node_type);
		} else if (type == "styles") {

			if (!has_stylebox(name, node_type))
				r_ret = Ref<StyleBox>();
			else
				r_ret = get_stylebox(name, node_type);
		} else if (type == "fonts") {

			if (!has_font(name, node_type))
				r_ret = Ref<Font>();
			else
				r_ret = get_font(name, node_type);
		} else if (type == "colors") {

			r_ret = get_color(name, node_type);
		} else if (type == "constants") {

			r_ret = get_constant(name, node_type);
		} else
			return false;

		return true;
	}

	return false;
}

void Theme::_get_property_list(List<PropertyInfo> *p_list) const {

	List<PropertyInfo> list;

	const StringName *key = NULL;

	while ((key = icon_map.next(key))) {

		const StringName *key2 = NULL;

		while ((key2 = icon_map[*key].next(key2))) {

			list.push_back(PropertyInfo(Variant::OBJECT, String() + *key + "/icons/" + *key2, PROPERTY_HINT_RESOURCE_TYPE, "Texture", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		}
	}

	key = NULL;

	while ((key = style_map.next(key))) {

		const StringName *key2 = NULL;

		while ((key2 = style_map[*key].next(key2))) {

			list.push_back(PropertyInfo(Variant::OBJECT, String() + *key + "/styles/" + *key2, PROPERTY_HINT_RESOURCE_TYPE, "StyleBox", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		}
	}

	key = NULL;

	while ((key = font_map.next(key))) {

		const StringName *key2 = NULL;

		while ((key2 = font_map[*key].next(key2))) {

			list.push_back(PropertyInfo(Variant::OBJECT, String() + *key + "/fonts/" + *key2, PROPERTY_HINT_RESOURCE_TYPE, "Font", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		}
	}

	key = NULL;

	while ((key = color_map.next(key))) {

		const StringName *key2 = NULL;

		while ((key2 = color_map[*key].next(key2))) {

			list.push_back(PropertyInfo(Variant::COLOR, String() + *key + "/colors/" + *key2));
		}
	}

	key = NULL;

	while ((key = constant_map.next(key))) {

		const StringName *key2 = NULL;

		while ((key2 = constant_map[*key].next(key2))) {

			list.push_back(PropertyInfo(Variant::INT, String() + *key + "/constants/" + *key2));
		}
	}

	list.sort();
	for (List<PropertyInfo>::Element *E = list.front(); E; E = E->next()) {
		p_list->push_back(E->get());
	}
}

Ref<Theme> Theme::get_default() {

	return default_theme;
}

void Theme::set_default_theme_font(const Ref<Font> &p_default_font) {

	if (default_theme_font == p_default_font)
		return;

	if (default_theme_font.is_valid()) {
		_unref_font(default_theme_font);
	}

	default_theme_font = p_default_font;

	if (default_theme_font.is_valid()) {
		_ref_font(default_theme_font);
	}

	_change_notify();
	emit_changed();
}

Ref<Font> Theme::get_default_theme_font() const {

	return default_theme_font;
}

void Theme::set_default(const Ref<Theme> &p_default) {

	default_theme = p_default;
}

Ref<Texture> Theme::default_icon;
Ref<StyleBox> Theme::default_style;
Ref<Font> Theme::default_font;

void Theme::set_default_icon(const Ref<Texture> &p_icon) {

	default_icon = p_icon;
}
void Theme::set_default_style(const Ref<StyleBox> &p_style) {

	default_style = p_style;
}
void Theme::set_default_font(const Ref<Font> &p_font) {

	default_font = p_font;
}

void Theme::set_icon(const StringName &p_name, const StringName &p_type, const Ref<Texture> &p_icon) {

	//ERR_FAIL_COND(p_icon.is_null());

	bool new_value = !icon_map.has(p_type) || !icon_map[p_type].has(p_name);

	icon_map[p_type][p_name] = p_icon;

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}
Ref<Texture> Theme::get_icon(const StringName &p_name, const StringName &p_type) const {

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

	icon_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_icon_list(StringName p_type, List<StringName> *p_list) const {

	if (!icon_map.has(p_type))
		return;

	const StringName *key = NULL;

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
		return NULL;
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
	if (!shader_map.has(p_type))
		return;

	const StringName *key = NULL;

	while ((key = shader_map[p_type].next(key))) {

		p_list->push_back(*key);
	}
}

void Theme::set_stylebox(const StringName &p_name, const StringName &p_type, const Ref<StyleBox> &p_style) {

	//ERR_FAIL_COND(p_style.is_null());

	bool new_value = !style_map.has(p_type) || !style_map[p_type].has(p_name);

	style_map[p_type][p_name] = p_style;

	if (new_value)
		_change_notify();
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

	style_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_stylebox_list(StringName p_type, List<StringName> *p_list) const {

	if (!style_map.has(p_type))
		return;

	const StringName *key = NULL;

	while ((key = style_map[p_type].next(key))) {

		p_list->push_back(*key);
	}
}

void Theme::get_stylebox_types(List<StringName> *p_list) const {
	const StringName *key = NULL;
	while ((key = style_map.next(key))) {
		p_list->push_back(*key);
	}
}

void Theme::set_font(const StringName &p_name, const StringName &p_type, const Ref<Font> &p_font) {

	//ERR_FAIL_COND(p_font.is_null());

	bool new_value = !font_map.has(p_type) || !font_map[p_type].has(p_name);

	if (!new_value) {
		if (font_map[p_type][p_name].is_valid()) {
			_unref_font(font_map[p_type][p_name]);
		}
	}
	font_map[p_type][p_name] = p_font;

	if (p_font.is_valid()) {
		_ref_font(p_font);
	}

	if (new_value) {
		_change_notify();
		emit_changed();
	}
}
Ref<Font> Theme::get_font(const StringName &p_name, const StringName &p_type) const {

	if (font_map.has(p_type) && font_map[p_type].has(p_name) && font_map[p_type][p_name].is_valid())
		return font_map[p_type][p_name];
	else if (default_theme_font.is_valid())
		return default_theme_font;
	else
		return default_font;
}

bool Theme::has_font(const StringName &p_name, const StringName &p_type) const {

	return (font_map.has(p_type) && font_map[p_type].has(p_name) && font_map[p_type][p_name].is_valid());
}

void Theme::clear_font(const StringName &p_name, const StringName &p_type) {

	ERR_FAIL_COND(!font_map.has(p_type));
	ERR_FAIL_COND(!font_map[p_type].has(p_name));

	if (font_map.has(p_type) && font_map[p_type].has(p_name) && font_map[p_type][p_name].is_valid()) {
		_unref_font(font_map[p_type][p_name]);
	}

	font_map[p_type].erase(p_name);
	_change_notify();
	emit_changed();
}

void Theme::get_font_list(StringName p_type, List<StringName> *p_list) const {

	if (!font_map.has(p_type))
		return;

	const StringName *key = NULL;

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

	if (color_map.has(p_type) && color_map[p_type].has(p_name))
		return color_map[p_type][p_name];
	else
		return Color();
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

	if (!color_map.has(p_type))
		return;

	const StringName *key = NULL;

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

	if (constant_map.has(p_type) && constant_map[p_type].has(p_name))
		return constant_map[p_type][p_name];
	else {
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

	if (!constant_map.has(p_type))
		return;

	const StringName *key = NULL;

	while ((key = constant_map[p_type].next(key))) {

		p_list->push_back(*key);
	}
}

void Theme::copy_default_theme() {

	Ref<Theme> default_theme = get_default();

	icon_map = default_theme->icon_map;
	style_map = default_theme->style_map;
	font_map = default_theme->font_map;
	color_map = default_theme->color_map;
	constant_map = default_theme->constant_map;
	_change_notify();
	emit_changed();
}

void Theme::get_type_list(List<StringName> *p_list) const {

	Set<StringName> types;

	const StringName *key = NULL;

	while ((key = icon_map.next(key))) {

		types.insert(*key);
	}

	key = NULL;

	while ((key = style_map.next(key))) {

		types.insert(*key);
	}

	key = NULL;

	while ((key = font_map.next(key))) {

		types.insert(*key);
	}

	key = NULL;

	while ((key = color_map.next(key))) {

		types.insert(*key);
	}

	key = NULL;

	while ((key = constant_map.next(key))) {

		types.insert(*key);
	}

	for (Set<StringName>::Element *E = types.front(); E; E = E->next()) {

		p_list->push_back(E->get());
	}
}

void Theme::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_icon", "name", "type", "texture:Texture"), &Theme::set_icon);
	ClassDB::bind_method(D_METHOD("get_icon:Texture", "name", "type"), &Theme::get_icon);
	ClassDB::bind_method(D_METHOD("has_icon", "name", "type"), &Theme::has_icon);
	ClassDB::bind_method(D_METHOD("clear_icon", "name", "type"), &Theme::clear_icon);
	ClassDB::bind_method(D_METHOD("get_icon_list", "type"), &Theme::_get_icon_list);

	ClassDB::bind_method(D_METHOD("set_stylebox", "name", "type", "texture:StyleBox"), &Theme::set_stylebox);
	ClassDB::bind_method(D_METHOD("get_stylebox:StyleBox", "name", "type"), &Theme::get_stylebox);
	ClassDB::bind_method(D_METHOD("has_stylebox", "name", "type"), &Theme::has_stylebox);
	ClassDB::bind_method(D_METHOD("clear_stylebox", "name", "type"), &Theme::clear_stylebox);
	ClassDB::bind_method(D_METHOD("get_stylebox_list", "type"), &Theme::_get_stylebox_list);
	ClassDB::bind_method(D_METHOD("get_stylebox_types"), &Theme::_get_stylebox_types);

	ClassDB::bind_method(D_METHOD("set_font", "name", "type", "font:Font"), &Theme::set_font);
	ClassDB::bind_method(D_METHOD("get_font:Font", "name", "type"), &Theme::get_font);
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

	ClassDB::bind_method(D_METHOD("set_default_font", "font"), &Theme::set_default_theme_font);
	ClassDB::bind_method(D_METHOD("get_default_font"), &Theme::get_default_theme_font);

	ClassDB::bind_method(D_METHOD("get_type_list", "type"), &Theme::_get_type_list);

	ClassDB::bind_method(D_METHOD("_emit_theme_changed"), &Theme::_emit_theme_changed);

	ClassDB::bind_method("copy_default_theme", &Theme::copy_default_theme);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "default_font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_default_font", "get_default_font");
}

Theme::Theme() {
}

Theme::~Theme() {
}

RES ResourceFormatLoaderTheme::load(const String &p_path, const String &p_original_path, Error *r_error) {
	if (r_error)
		*r_error = ERR_CANT_OPEN;

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);

	ERR_EXPLAIN("Unable to open theme file: " + p_path);
	ERR_FAIL_COND_V(err, RES());
	String base_path = p_path.get_base_dir();
	Ref<Theme> theme(memnew(Theme));
	Map<StringName, Variant> library;
	if (r_error)
		*r_error = ERR_FILE_CORRUPT;

	bool reading_library = false;
	int line = 0;

	while (!f->eof_reached()) {

		String l = f->get_line().strip_edges();
		line++;

		int comment = l.find(";");
		if (comment != -1)
			l = l.substr(0, comment);
		if (l == "")
			continue;

		if (l.begins_with("[")) {
			if (l == "[library]") {
				reading_library = true;
			} else if (l == "[theme]") {
				reading_library = false;
			} else {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Unknown section type: '" + l + "'.");
				ERR_FAIL_V(RES());
			}
			continue;
		}

		int eqpos = l.find("=");
		if (eqpos == -1) {
			memdelete(f);
			ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected '='.");
			ERR_FAIL_V(RES());
		}

		String right = l.substr(eqpos + 1, l.length()).strip_edges();
		if (right == "") {
			memdelete(f);
			ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected value after '='.");
			ERR_FAIL_V(RES());
		}

		Variant value;

		if (right.is_valid_integer()) {
			//is number
			value = right.to_int();
		} else if (right.is_valid_html_color()) {
			//is html color
			value = Color::html(right);
		} else if (right.begins_with("@")) { //reference

			String reference = right.substr(1, right.length());
			if (!library.has(reference)) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid reference to '" + reference + "'.");
				ERR_FAIL_V(RES());
			}

			value = library[reference];

		} else if (right.begins_with("default")) { //use default
			//do none
		} else {
			//attempt to parse a constructor
			int popenpos = right.find("(");

			if (popenpos == -1) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid constructor syntax: " + right);
				ERR_FAIL_V(RES());
			}

			int pclosepos = right.find_last(")");

			if (pclosepos == -1) {
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid constructor parameter syntax: " + right);
				ERR_FAIL_V(RES());
			}

			String type = right.substr(0, popenpos);
			String param = right.substr(popenpos + 1, pclosepos - popenpos - 1);

			if (type == "icon") {

				String path;

				if (param.is_abs_path())
					path = param;
				else
					path = base_path + "/" + param;

				Ref<Texture> texture = ResourceLoader::load(path);
				if (!texture.is_valid()) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Couldn't find icon at path: " + path);
					ERR_FAIL_V(RES());
				}

				value = texture;

			} else if (type == "sbox") {

				String path;

				if (param.is_abs_path())
					path = param;
				else
					path = base_path + "/" + param;

				Ref<StyleBox> stylebox = ResourceLoader::load(path);
				if (!stylebox.is_valid()) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Couldn't find stylebox at path: " + path);
					ERR_FAIL_V(RES());
				}

				value = stylebox;

			} else if (type == "sboxt") {

				Vector<String> params = param.split(",");
				if (params.size() != 5 && params.size() != 9) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid param count for sboxt(): '" + right + "'.");
					ERR_FAIL_V(RES());
				}

				String path = params[0];

				if (!param.is_abs_path())
					path = base_path + "/" + path;

				Ref<Texture> tex = ResourceLoader::load(path);
				if (tex.is_null()) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Could not open texture for sboxt at path: '" + params[0] + "'.");
					ERR_FAIL_V(RES());
				}

				Ref<StyleBoxTexture> sbtex(memnew(StyleBoxTexture));

				sbtex->set_texture(tex);

				for (int i = 0; i < 4; i++) {
					if (!params[i + 1].is_valid_integer()) {

						memdelete(f);
						ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid expand margin parameter for sboxt #" + itos(i + 1) + ", expected integer constant, got: '" + params[i + 1] + "'.");
						ERR_FAIL_V(RES());
					}

					int margin = params[i + 1].to_int();
					sbtex->set_expand_margin_size(Margin(i), margin);
				}

				if (params.size() == 9) {

					for (int i = 0; i < 4; i++) {

						if (!params[i + 5].is_valid_integer()) {
							memdelete(f);
							ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid expand margin parameter for sboxt #" + itos(i + 5) + ", expected integer constant, got: '" + params[i + 5] + "'.");
							ERR_FAIL_V(RES());
						}

						int margin = params[i + 5].to_int();
						sbtex->set_margin_size(Margin(i), margin);
					}
				}

				value = sbtex;
			} else if (type == "sboxf") {

				Vector<String> params = param.split(",");
				if (params.size() < 2) {

					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid param count for sboxf(): '" + right + "'.");
					ERR_FAIL_V(RES());
				}

				Ref<StyleBoxFlat> sbflat(memnew(StyleBoxFlat));

				if (!params[0].is_valid_integer()) {

					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected integer numeric constant for parameter 0 (border size).");
					ERR_FAIL_V(RES());
				}

				sbflat->set_border_size(params[0].to_int());

				if (!params[0].is_valid_integer()) {

					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected integer numeric constant for parameter 0 (border size).");
					ERR_FAIL_V(RES());
				}

				int left = MIN(params.size() - 1, 3);

				int ccodes = 0;

				for (int i = 0; i < left; i++) {

					if (params[i + 1].is_valid_html_color())
						ccodes++;
					else
						break;
				}

				Color normal;
				Color bright;
				Color dark;

				if (ccodes < 1) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected at least 1, 2 or 3 html color codes.");
					ERR_FAIL_V(RES());
				} else if (ccodes == 1) {

					normal = Color::html(params[1]);
					bright = Color::html(params[1]);
					dark = Color::html(params[1]);
				} else if (ccodes == 2) {

					normal = Color::html(params[1]);
					bright = Color::html(params[2]);
					dark = Color::html(params[2]);
				} else {

					normal = Color::html(params[1]);
					bright = Color::html(params[2]);
					dark = Color::html(params[3]);
				}

				sbflat->set_dark_color(dark);
				sbflat->set_light_color(bright);
				sbflat->set_bg_color(normal);

				if (params.size() == ccodes + 5) {
					//margins
					for (int i = 0; i < 4; i++) {

						if (!params[i + ccodes + 1].is_valid_integer()) {
							memdelete(f);
							ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid expand margin parameter for sboxf #" + itos(i + ccodes + 1) + ", expected integer constant, got: '" + params[i + ccodes + 1] + "'.");
							ERR_FAIL_V(RES());
						}

						//int margin = params[i+ccodes+1].to_int();
						//sbflat->set_margin_size(Margin(i),margin);
					}
				} else if (params.size() != ccodes + 1) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid amount of margin parameters for sboxt.");
					ERR_FAIL_V(RES());
				}

				value = sbflat;

			} else {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid constructor type: '" + type + "'.");
				ERR_FAIL_V(RES());
			}
		}

		//parse left and do something with it
		String left = l.substr(0, eqpos);

		if (reading_library) {

			left = left.strip_edges();
			if (!left.is_valid_identifier()) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": <LibraryItem> is not a valid identifier.");
				ERR_FAIL_V(RES());
			}
			if (library.has(left)) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Already in library: '" + left + "'.");
				ERR_FAIL_V(RES());
			}

			library[left] = value;
		} else {

			int pointpos = left.find(".");
			if (pointpos == -1) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected 'control.item=..' assign syntax.");
				ERR_FAIL_V(RES());
			}

			String control = left.substr(0, pointpos).strip_edges();
			if (!control.is_valid_identifier()) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": <Control> is not a valid identifier.");
				ERR_FAIL_V(RES());
			}
			String item = left.substr(pointpos + 1, left.size()).strip_edges();
			if (!item.is_valid_identifier()) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": <Item> is not a valid identifier.");
				ERR_FAIL_V(RES());
			}

			if (value.get_type() == Variant::NIL) {
				//try to use exiting
				if (Theme::get_default()->has_stylebox(item, control))
					value = Theme::get_default()->get_stylebox(item, control);
				else if (Theme::get_default()->has_font(item, control))
					value = Theme::get_default()->get_font(item, control);
				else if (Theme::get_default()->has_icon(item, control))
					value = Theme::get_default()->get_icon(item, control);
				else if (Theme::get_default()->has_color(item, control))
					value = Theme::get_default()->get_color(item, control);
				else if (Theme::get_default()->has_constant(item, control))
					value = Theme::get_default()->get_constant(item, control);
				else {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Default not present for: '" + control + "." + item + "'.");
					ERR_FAIL_V(RES());
				}
			}

			if (value.get_type() == Variant::OBJECT) {

				Ref<Resource> res = value;
				if (!res.is_valid()) {

					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid resource (NULL).");
					ERR_FAIL_V(RES());
				}

				if (res->cast_to<StyleBox>()) {

					theme->set_stylebox(item, control, res);
				} else if (res->cast_to<Font>()) {
					theme->set_font(item, control, res);
				} else if (res->cast_to<Font>()) {
					theme->set_font(item, control, res);
				} else if (res->cast_to<Texture>()) {
					theme->set_icon(item, control, res);
				} else {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid resource type.");
					ERR_FAIL_V(RES());
				}
			} else if (value.get_type() == Variant::COLOR) {

				theme->set_color(item, control, value);

			} else if (value.get_type() == Variant::INT) {

				theme->set_constant(item, control, value);

			} else {

				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Couldn't even determine what this setting is! what did you do!?");
				ERR_FAIL_V(RES());
			}
		}
	}

	f->close();
	memdelete(f);

	if (r_error)
		*r_error = OK;

	return theme;
}

void ResourceFormatLoaderTheme::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("theme");
}

bool ResourceFormatLoaderTheme::handles_type(const String &p_type) const {

	return p_type == "Theme";
}

String ResourceFormatLoaderTheme::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "theme")
		return "Theme";
	return "";
}
