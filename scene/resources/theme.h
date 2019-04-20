/*************************************************************************/
/*  theme.h                                                              */
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

#ifndef THEME_H
#define THEME_H

#include "core/io/resource_loader.h"
#include "core/resource.h"

#include "scene/gui/control.h"
#include "scene/resources/font.h"
#include "scene/resources/shader.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Control;

class Theme : public Resource {

	GDCLASS(Theme, Resource);
	RES_BASE_EXTENSION("theme");

	static Ref<Theme> default_theme;
	void _emit_theme_changed();

	HashMap<StringName, HashMap<StringName, Ref<Texture> > > icon_map;
	HashMap<StringName, HashMap<StringName, Ref<StyleBox> > > style_map;
	HashMap<StringName, HashMap<StringName, Ref<Font> > > font_map;
	HashMap<StringName, HashMap<StringName, Ref<Shader> > > shader_map;
	HashMap<StringName, HashMap<StringName, Color> > color_map;
	HashMap<StringName, HashMap<StringName, int> > constant_map;

protected:
	float dpi_scale;

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
	static void set_default_style(const Ref<StyleBox> &p_style);
	static void set_default_font(const Ref<Font> &p_font);

	void set_default_theme_font(const Ref<Font> &p_default_font);
	Ref<Font> get_default_theme_font(const Control *p_context = NULL) const;

	void set_dpi(float p_dpi);
	float get_dpi() const;

	void set_icon(const StringName &p_name, const StringName &p_type, const Ref<Texture> &p_icon);
	Ref<Texture> get_icon(const StringName &p_name, const StringName &p_type, const Control *p_context = NULL) const;
	bool has_icon(const StringName &p_name, const StringName &p_type) const;
	void clear_icon(const StringName &p_name, const StringName &p_type);
	void get_icon_list(StringName p_type, List<StringName> *p_list) const;

	void set_shader(const StringName &p_name, const StringName &p_type, const Ref<Shader> &p_shader);
	Ref<Shader> get_shader(const StringName &p_name, const StringName &p_type) const;
	bool has_shader(const StringName &p_name, const StringName &p_type) const;
	void clear_shader(const StringName &p_name, const StringName &p_type);
	void get_shader_list(const StringName &p_type, List<StringName> *p_list) const;

	void set_stylebox(const StringName &p_name, const StringName &p_type, const Ref<StyleBox> &p_style);
	Ref<StyleBox> get_stylebox(const StringName &p_name, const StringName &p_type, const Control *p_context = NULL) const;
	bool has_stylebox(const StringName &p_name, const StringName &p_type) const;
	void clear_stylebox(const StringName &p_name, const StringName &p_type);
	void get_stylebox_list(StringName p_type, List<StringName> *p_list) const;
	void get_stylebox_types(List<StringName> *p_list) const;

	void set_font(const StringName &p_name, const StringName &p_type, const Ref<Font> &p_font);
	Ref<Font> get_font(const StringName &p_name, const StringName &p_type, const Control *p_context = NULL) const;
	bool has_font(const StringName &p_name, const StringName &p_type) const;
	void clear_font(const StringName &p_name, const StringName &p_type);
	void get_font_list(StringName p_type, List<StringName> *p_list) const;

	void set_color(const StringName &p_name, const StringName &p_type, const Color &p_color);
	Color get_color(const StringName &p_name, const StringName &p_type) const;
	bool has_color(const StringName &p_name, const StringName &p_type) const;
	void clear_color(const StringName &p_name, const StringName &p_type);
	void get_color_list(StringName p_type, List<StringName> *p_list) const;

	void set_constant(const StringName &p_name, const StringName &p_type, int p_constant);
	int get_constant(const StringName &p_name, const StringName &p_type, const Control *p_context = NULL) const;
	bool has_constant(const StringName &p_name, const StringName &p_type) const;
	void clear_constant(const StringName &p_name, const StringName &p_type);
	void get_constant_list(StringName p_type, List<StringName> *p_list) const;

	bool has_type(StringName p_type) const;
	void _get_type_set(Set<StringName> *p_set) const;
	void get_type_list(List<StringName> *p_list) const;

	void copy_default_theme();
	void copy_theme(const Ref<Theme> &p_other);
	void clear();

	Theme();
	~Theme();
};

class ScaledThemeTexture : public Texture {

private:
	Ref<Texture> unscaled_texture;
	Ref<Theme> theme;
	Size2i icon_size;
	Color modulate_color;

public:
	void set_unscaled_texture(const Ref<Texture> p_texture);
	Ref<Texture> get_unscaled_texture() const;

	void set_dpi_theme(const Ref<Theme> p_theme);
	Ref<Theme> get_dpi_theme() const;

	void set_modulate_color(const Color p_color);
	Color get_modulate_color() const;

	virtual int get_width() const { return icon_size.width * theme->get_dpi(); };
	virtual int get_height() const { return icon_size.height * theme->get_dpi(); };
	virtual int get_original_width() const { return unscaled_texture->get_original_width(); };
	virtual int get_original_height() const { return unscaled_texture->get_original_height(); };
	virtual void set_size_override(const Size2 &p_size);

	virtual RID get_rid() const { return unscaled_texture->get_rid(); };
	virtual bool is_pixel_opaque(int p_x, int p_y) const { return unscaled_texture->is_pixel_opaque(p_x, p_y); };
	virtual bool has_alpha() const { return unscaled_texture->has_alpha(); };
	virtual void set_flags(uint32_t p_flags){};
	virtual uint32_t get_flags() const { return unscaled_texture->get_flags(); };

	virtual Ref<Image> get_data() const { return unscaled_texture->get_data(); }

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture> &p_normal_map = Ref<Texture>()) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture> &p_normal_map = Ref<Texture>()) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture> &p_normal_map = Ref<Texture>(), bool p_clip_uv = true) const;

	ScaledThemeTexture();
};

#endif
