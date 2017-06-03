/*************************************************************************/
/*  theme.h                                                              */
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
#ifndef THEME_H
#define THEME_H

#include "io/resource_loader.h"
#include "resource.h"
#include "scene/resources/font.h"
#include "scene/resources/shader.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Theme : public Resource {

	GDCLASS(Theme, Resource);
	RES_BASE_EXTENSION("thm");

	static Ref<Theme> default_theme;

	//keep a reference count to font, so each time the font changes, we emit theme changed too
	Map<Ref<Font>, int> font_refcount;

	void _ref_font(Ref<Font> p_sc);
	void _unref_font(Ref<Font> p_sc);
	void _emit_theme_changed();

	HashMap<StringName, HashMap<StringName, Ref<Texture>, StringNameHasher>, StringNameHasher> icon_map;
	HashMap<StringName, HashMap<StringName, Ref<StyleBox>, StringNameHasher>, StringNameHasher> style_map;
	HashMap<StringName, HashMap<StringName, Ref<Font>, StringNameHasher>, StringNameHasher> font_map;
	HashMap<StringName, HashMap<StringName, Ref<Shader>, StringNameHasher>, StringNameHasher> shader_map;
	HashMap<StringName, HashMap<StringName, Color, StringNameHasher>, StringNameHasher> color_map;
	HashMap<StringName, HashMap<StringName, int, StringNameHasher>, StringNameHasher> constant_map;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static Ref<Texture> default_icon;
	static Ref<StyleBox> default_style;
	static Ref<Font> default_font;

	Ref<Font> default_theme_font;

	PoolVector<String> _get_icon_list(const String &p_type) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_icon_list(p_type, &il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}
	PoolVector<String> _get_stylebox_list(const String &p_type) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_stylebox_list(p_type, &il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}
	PoolVector<String> _get_stylebox_types(void) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_stylebox_types(&il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}
	PoolVector<String> _get_font_list(const String &p_type) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_font_list(p_type, &il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}
	PoolVector<String> _get_color_list(const String &p_type) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_color_list(p_type, &il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}
	PoolVector<String> _get_constant_list(const String &p_type) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_constant_list(p_type, &il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}
	PoolVector<String> _get_type_list(const String &p_type) const {
		PoolVector<String> ilret;
		List<StringName> il;
		get_type_list(&il);
		for (List<StringName>::Element *E = il.front(); E; E = E->next()) {
			ilret.push_back(E->get());
		}
		return ilret;
	}

	static void _bind_methods();

public:
	static Ref<Theme> get_default();
	static void set_default(const Ref<Theme> &p_default);

	static void set_default_icon(const Ref<Texture> &p_icon);
	static void set_default_style(const Ref<StyleBox> &p_default_style);
	static void set_default_font(const Ref<Font> &p_default_font);

	void set_default_theme_font(const Ref<Font> &p_default_font);
	Ref<Font> get_default_theme_font() const;

	void set_icon(const StringName &p_name, const StringName &p_type, const Ref<Texture> &p_icon);
	Ref<Texture> get_icon(const StringName &p_name, const StringName &p_type) const;
	bool has_icon(const StringName &p_name, const StringName &p_type) const;
	void clear_icon(const StringName &p_name, const StringName &p_type);
	void get_icon_list(StringName p_type, List<StringName> *p_list) const;

	void set_shader(const StringName &p_name, const StringName &p_type, const Ref<Shader> &p_shader);
	Ref<Shader> get_shader(const StringName &p_name, const StringName &p_type) const;
	bool has_shader(const StringName &p_name, const StringName &p_type) const;
	void clear_shader(const StringName &p_name, const StringName &p_type);
	void get_shader_list(const StringName &p_name, List<StringName> *p_list) const;

	void set_stylebox(const StringName &p_name, const StringName &p_type, const Ref<StyleBox> &p_style);
	Ref<StyleBox> get_stylebox(const StringName &p_name, const StringName &p_type) const;
	bool has_stylebox(const StringName &p_name, const StringName &p_type) const;
	void clear_stylebox(const StringName &p_name, const StringName &p_type);
	void get_stylebox_list(StringName p_type, List<StringName> *p_list) const;
	void get_stylebox_types(List<StringName> *p_list) const;

	void set_font(const StringName &p_name, const StringName &p_type, const Ref<Font> &p_font);
	Ref<Font> get_font(const StringName &p_name, const StringName &p_type) const;
	bool has_font(const StringName &p_name, const StringName &p_type) const;
	void clear_font(const StringName &p_name, const StringName &p_type);
	void get_font_list(StringName p_type, List<StringName> *p_list) const;

	void set_color(const StringName &p_name, const StringName &p_type, const Color &p_color);
	Color get_color(const StringName &p_name, const StringName &p_type) const;
	bool has_color(const StringName &p_name, const StringName &p_type) const;
	void clear_color(const StringName &p_name, const StringName &p_type);
	void get_color_list(StringName p_type, List<StringName> *p_list) const;

	void set_constant(const StringName &p_name, const StringName &p_type, int p_constant);
	int get_constant(const StringName &p_name, const StringName &p_type) const;
	bool has_constant(const StringName &p_name, const StringName &p_type) const;
	void clear_constant(const StringName &p_name, const StringName &p_type);
	void get_constant_list(StringName p_type, List<StringName> *p_list) const;

	void get_type_list(List<StringName> *p_list) const;

	void copy_default_theme();

	Theme();
	~Theme();
};

class ResourceFormatLoaderTheme : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif
