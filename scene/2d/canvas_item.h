/*************************************************************************/
/*  canvas_item.h                                                        */
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
#ifndef CANVAS_ITEM_H
#define CANVAS_ITEM_H

#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/material.h"
#include "scene/resources/shader.h"
#include "scene/resources/texture.h"

class CanvasLayer;
class Viewport;
class Font;

class StyleBox;

class CanvasItemMaterial : public Material {

	GDCLASS(CanvasItemMaterial, Material)

public:
	enum BlendMode {
		BLEND_MODE_MIX,
		BLEND_MODE_ADD,
		BLEND_MODE_SUB,
		BLEND_MODE_MUL,
		BLEND_MODE_PREMULT_ALPHA
	};

	enum LightMode {
		LIGHT_MODE_NORMAL,
		LIGHT_MODE_UNSHADED,
		LIGHT_MODE_LIGHT_ONLY
	};

private:
	union MaterialKey {

		struct {
			uint32_t blend_mode : 4;
			uint32_t light_mode : 4;
			uint32_t invalid_key : 1;
		};

		uint32_t key;

		bool operator<(const MaterialKey &p_key) const {
			return key < p_key.key;
		}
	};

	struct ShaderData {
		RID shader;
		int users;
	};

	static Map<MaterialKey, ShaderData> shader_map;

	MaterialKey current_key;

	_FORCE_INLINE_ MaterialKey _compute_key() const {

		MaterialKey mk;
		mk.key = 0;
		mk.blend_mode = blend_mode;
		mk.light_mode = light_mode;
		return mk;
	}

	static Mutex *material_mutex;
	static SelfList<CanvasItemMaterial>::List dirty_materials;
	SelfList<CanvasItemMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

	BlendMode blend_mode;
	LightMode light_mode;

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;

public:
	void set_blend_mode(BlendMode p_blend_mode);
	BlendMode get_blend_mode() const;

	void set_light_mode(LightMode p_light_mode);
	LightMode get_light_mode() const;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	CanvasItemMaterial();
	virtual ~CanvasItemMaterial();
};

VARIANT_ENUM_CAST(CanvasItemMaterial::BlendMode)
VARIANT_ENUM_CAST(CanvasItemMaterial::LightMode)

class CanvasItem : public Node {

	GDCLASS(CanvasItem, Node);

public:
	enum BlendMode {

		BLEND_MODE_MIX, //default
		BLEND_MODE_ADD,
		BLEND_MODE_SUB,
		BLEND_MODE_MUL,
		BLEND_MODE_PREMULT_ALPHA
	};

private:
	mutable SelfList<Node> xform_change;

	RID canvas_item;
	String group;

	CanvasLayer *canvas_layer;

	Color modulate;
	Color self_modulate;

	List<CanvasItem *> children_items;
	List<CanvasItem *>::Element *C;

	int light_mask;

	bool first_draw;
	bool visible;
	bool pending_update;
	bool toplevel;
	bool drawing;
	bool block_transform_notify;
	bool behind;
	bool use_parent_material;
	bool notify_local_transform;
	bool notify_transform;

	Ref<Material> material;

	mutable Transform2D global_transform;
	mutable bool global_invalid;

	void _toplevel_raise_self();

	void _propagate_visibility_changed(bool p_visible);

	void _update_callback();

	void _enter_canvas();
	void _exit_canvas();

	void _notify_transform(CanvasItem *p_node);

	void _set_on_top(bool p_on_top) { set_draw_behind_parent(!p_on_top); }
	bool _is_on_top() const { return !is_draw_behind_parent_enabled(); }

protected:
	_FORCE_INLINE_ void _notify_transform() {
		if (!is_inside_tree()) return;
		_notify_transform(this);
		if (!block_transform_notify && notify_local_transform) notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}

	void item_rect_changed(bool p_size_changed = true);

	void _notification(int p_what);
	static void _bind_methods();

public:
	enum {
		NOTIFICATION_TRANSFORM_CHANGED = SceneTree::NOTIFICATION_TRANSFORM_CHANGED, //unique
		NOTIFICATION_DRAW = 30,
		NOTIFICATION_VISIBILITY_CHANGED = 31,
		NOTIFICATION_ENTER_CANVAS = 32,
		NOTIFICATION_EXIT_CANVAS = 33,
		NOTIFICATION_LOCAL_TRANSFORM_CHANGED = 35,
		NOTIFICATION_WORLD_2D_CHANGED = 36,

	};

	/* EDITOR */

	virtual Variant edit_get_state() const;
	virtual void edit_set_state(const Variant &p_state);
	virtual void edit_set_rect(const Rect2 &p_edit_rect);
	virtual void edit_rotate(float p_rot);
	virtual Size2 edit_get_minimum_size() const;

	/* VISIBILITY */

	void set_visible(bool p_visible);
	bool is_visible() const;
	bool is_visible_in_tree() const;
	void show();
	void hide();

	void update();

	virtual void set_light_mask(int p_light_mask);
	int get_light_mask() const;

	void set_modulate(const Color &p_modulate);
	Color get_modulate() const;

	void set_self_modulate(const Color &p_self_modulate);
	Color get_self_modulate() const;

	/* DRAWING API */

	void draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width = 1.0, bool p_antialiased = false);
	void draw_polyline(const Vector<Point2> &p_points, const Color &p_color, float p_width = 1.0, bool p_antialiased = false);
	void draw_polyline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, float p_width = 1.0, bool p_antialiased = false);
	void draw_rect(const Rect2 &p_rect, const Color &p_color, bool p_filled = true);
	void draw_circle(const Point2 &p_pos, float p_radius, const Color &p_color);
	void draw_texture(const Ref<Texture> &p_texture, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1, 1), const Ref<Texture> &p_normal_map = Ref<Texture>());
	void draw_texture_rect(const Ref<Texture> &p_texture, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture> &p_normal_map = Ref<Texture>());
	void draw_texture_rect_region(const Ref<Texture> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture> &p_normal_map = Ref<Texture>(), bool p_clip_uv = true);
	void draw_style_box(const Ref<StyleBox> &p_style_box, const Rect2 &p_rect);
	void draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, Ref<Texture> p_texture = Ref<Texture>(), float p_width = 1, const Ref<Texture> &p_normal_map = Ref<Texture>());
	void draw_polygon(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), Ref<Texture> p_texture = Ref<Texture>(), const Ref<Texture> &p_normal_map = Ref<Texture>(), bool p_antialiased = false);
	void draw_colored_polygon(const Vector<Point2> &p_points, const Color &p_color, const Vector<Point2> &p_uvs = Vector<Point2>(), Ref<Texture> p_texture = Ref<Texture>(), const Ref<Texture> &p_normal_map = Ref<Texture>(), bool p_antialiased = false);

	void draw_string(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_text, const Color &p_modulate = Color(1, 1, 1), int p_clip_w = -1);
	float draw_char(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_char, const String &p_next = "", const Color &p_modulate = Color(1, 1, 1));

	void draw_set_transform(const Point2 &p_offset, float p_rot, const Size2 &p_scale);
	void draw_set_transform_matrix(const Transform2D &p_matrix);

	/* RECT / TRANSFORM */

	void set_as_toplevel(bool p_toplevel);
	bool is_set_as_toplevel() const;

	void set_draw_behind_parent(bool p_enable);
	bool is_draw_behind_parent_enabled() const;

	CanvasItem *get_parent_item() const;

	virtual Rect2 get_item_rect() const = 0;
	virtual Transform2D get_transform() const = 0;

	virtual Transform2D get_global_transform() const;
	virtual Transform2D get_global_transform_with_canvas() const;

	Rect2 get_item_and_children_rect() const;

	CanvasItem *get_toplevel() const;
	_FORCE_INLINE_ RID get_canvas_item() const { return canvas_item; }

	void set_block_transform_notify(bool p_enable);
	bool is_block_transform_notify_enabled() const;

	Transform2D get_canvas_transform() const;
	Transform2D get_viewport_transform() const;
	Rect2 get_viewport_rect() const;
	RID get_viewport_rid() const;
	RID get_canvas() const;
	Ref<World2D> get_world_2d() const;

	virtual void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	virtual void set_use_parent_material(bool p_use_parent_material);
	bool get_use_parent_material() const;

	Ref<InputEvent> make_input_local(const Ref<InputEvent> &p_event) const;
	Vector2 make_canvas_position_local(const Vector2 &screen_point) const;

	Vector2 get_global_mouse_position() const;
	Vector2 get_local_mouse_position() const;

	void set_notify_local_transform(bool p_enable);
	bool is_local_transform_notification_enabled() const;

	void set_notify_transform(bool p_enable);
	bool is_transform_notification_enabled() const;

	int get_canvas_layer() const;

	CanvasItem();
	~CanvasItem();
};

VARIANT_ENUM_CAST(CanvasItem::BlendMode);

#endif // CANVAS_ITEM_H
