/*************************************************************************/
/*  canvas_item.h                                                        */
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

#ifndef CANVAS_ITEM_H
#define CANVAS_ITEM_H

#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/canvas_item_material.h"
#include "scene/resources/font.h"
#include "servers/text_server.h"

class CanvasLayer;
class MultiMesh;
class StyleBox;
class Window;
class World2D;

class CanvasItem : public Node {
	GDCLASS(CanvasItem, Node);

public:
	enum TextureFilter {
		TEXTURE_FILTER_PARENT_NODE,
		TEXTURE_FILTER_NEAREST,
		TEXTURE_FILTER_LINEAR,
		TEXTURE_FILTER_NEAREST_WITH_MIPMAPS,
		TEXTURE_FILTER_LINEAR_WITH_MIPMAPS,
		TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC,
		TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC,
		TEXTURE_FILTER_MAX
	};

	enum TextureRepeat {
		TEXTURE_REPEAT_PARENT_NODE,
		TEXTURE_REPEAT_DISABLED,
		TEXTURE_REPEAT_ENABLED,
		TEXTURE_REPEAT_MIRROR,
		TEXTURE_REPEAT_MAX,
	};

private:
	mutable SelfList<Node> xform_change;

	RID canvas_item;
	StringName group;

	CanvasLayer *canvas_layer = nullptr;

	Color modulate = Color(1, 1, 1, 1);
	Color self_modulate = Color(1, 1, 1, 1);

	List<CanvasItem *> children_items;
	List<CanvasItem *>::Element *C = nullptr;

	int light_mask = 1;

	Window *window = nullptr;
	bool first_draw = false;
	bool visible = true;
	bool clip_children = false;
	bool pending_update = false;
	bool top_level = false;
	bool drawing = false;
	bool block_transform_notify = false;
	bool behind = false;
	bool use_parent_material = false;
	bool notify_local_transform = false;
	bool notify_transform = false;

	RS::CanvasItemTextureFilter texture_filter_cache = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
	RS::CanvasItemTextureRepeat texture_repeat_cache = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
	TextureFilter texture_filter = TEXTURE_FILTER_PARENT_NODE;
	TextureRepeat texture_repeat = TEXTURE_REPEAT_PARENT_NODE;

	Ref<Material> material;

	mutable Transform2D global_transform;
	mutable bool global_invalid = true;

	void _top_level_raise_self();

	void _propagate_visibility_changed(bool p_visible);

	void _update_callback();

	void _enter_canvas();
	void _exit_canvas();

	void _window_visibility_changed();

	void _notify_transform(CanvasItem *p_node);

	void _set_on_top(bool p_on_top) { set_draw_behind_parent(!p_on_top); }
	bool _is_on_top() const { return !is_draw_behind_parent_enabled(); }

	static CanvasItem *current_item_drawn;
	friend class Viewport;
	void _update_texture_repeat_changed(bool p_propagate);
	void _update_texture_filter_changed(bool p_propagate);

protected:
	_FORCE_INLINE_ void _notify_transform() {
		if (!is_inside_tree()) {
			return;
		}
		_notify_transform(this);
		if (!block_transform_notify && notify_local_transform) {
			notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
		}
	}

	void item_rect_changed(bool p_size_changed = true);

	void _notification(int p_what);
	static void _bind_methods();

	GDVIRTUAL0(_draw)
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
#ifdef TOOLS_ENABLED
	// Select the node
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const;

	// Save and restore a CanvasItem state
	virtual void _edit_set_state(const Dictionary &p_state) {}
	virtual Dictionary _edit_get_state() const { return Dictionary(); };

	// Used to move the node
	virtual void _edit_set_position(const Point2 &p_position) = 0;
	virtual Point2 _edit_get_position() const = 0;

	// Used to scale the node
	virtual void _edit_set_scale(const Size2 &p_scale) = 0;
	virtual Size2 _edit_get_scale() const = 0;

	// Used to rotate the node
	virtual bool _edit_use_rotation() const { return false; };
	virtual void _edit_set_rotation(real_t p_rotation) {}
	virtual real_t _edit_get_rotation() const { return 0.0; };

	// Used to resize/move the node
	virtual bool _edit_use_rect() const { return false; }; // MAYBE REPLACE BY A _edit_get_editmode()
	virtual void _edit_set_rect(const Rect2 &p_rect) {}
	virtual Rect2 _edit_get_rect() const { return Rect2(0, 0, 0, 0); };
	virtual Size2 _edit_get_minimum_size() const { return Size2(-1, -1); }; // LOOKS WEIRD

	// Used to set a pivot
	virtual bool _edit_use_pivot() const { return false; };
	virtual void _edit_set_pivot(const Point2 &p_pivot) {}
	virtual Point2 _edit_get_pivot() const { return Point2(); };

	virtual Transform2D _edit_get_transform() const;
#endif

	/* VISIBILITY */

	void set_visible(bool p_visible);
	bool is_visible() const;
	bool is_visible_in_tree() const;
	void show();
	void hide();

	void update();

	void set_clip_children(bool p_enabled);
	bool is_clipping_children() const;

	virtual void set_light_mask(int p_light_mask);
	int get_light_mask() const;

	void set_modulate(const Color &p_modulate);
	Color get_modulate() const;

	void set_self_modulate(const Color &p_self_modulate);
	Color get_self_modulate() const;

	/* DRAWING API */

	void draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, real_t p_width = 1.0);
	void draw_polyline(const Vector<Point2> &p_points, const Color &p_color, real_t p_width = 1.0, bool p_antialiased = false);
	void draw_polyline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, real_t p_width = 1.0, bool p_antialiased = false);
	void draw_arc(const Vector2 &p_center, real_t p_radius, real_t p_start_angle, real_t p_end_angle, int p_point_count, const Color &p_color, real_t p_width = 1.0, bool p_antialiased = false);
	void draw_multiline(const Vector<Point2> &p_points, const Color &p_color, real_t p_width = 1.0);
	void draw_multiline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, real_t p_width = 1.0);
	void draw_rect(const Rect2 &p_rect, const Color &p_color, bool p_filled = true, real_t p_width = 1.0);
	void draw_circle(const Point2 &p_pos, real_t p_radius, const Color &p_color);
	void draw_texture(const Ref<Texture2D> &p_texture, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1, 1));
	void draw_texture_rect(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false);
	void draw_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, bool p_clip_uv = false);
	void draw_msdf_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), double p_outline = 0.0, double p_pixel_range = 4.0);
	void draw_style_box(const Ref<StyleBox> &p_style_box, const Rect2 &p_rect);
	void draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, Ref<Texture2D> p_texture = Ref<Texture2D>(), real_t p_width = 1);
	void draw_polygon(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), Ref<Texture2D> p_texture = Ref<Texture2D>());
	void draw_colored_polygon(const Vector<Point2> &p_points, const Color &p_color, const Vector<Point2> &p_uvs = Vector<Point2>(), Ref<Texture2D> p_texture = Ref<Texture2D>());

	void draw_mesh(const Ref<Mesh> &p_mesh, const Ref<Texture2D> &p_texture, const Transform2D &p_transform = Transform2D(), const Color &p_modulate = Color(1, 1, 1));
	void draw_multimesh(const Ref<MultiMesh> &p_multimesh, const Ref<Texture2D> &p_texture);

	void draw_string(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_text, HAlign p_align = HALIGN_LEFT, real_t p_width = -1, int p_size = Font::DEFAULT_FONT_SIZE, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0), uint16_t p_flags = TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;
	void draw_multiline_string(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_text, HAlign p_align = HALIGN_LEFT, real_t p_width = -1, int p_max_lines = -1, int p_size = Font::DEFAULT_FONT_SIZE, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0), uint16_t p_flags = TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;
	real_t draw_char(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_char, const String &p_next = "", int p_size = Font::DEFAULT_FONT_SIZE, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0)) const;

	void draw_set_transform(const Point2 &p_offset, real_t p_rot = 0.0, const Size2 &p_scale = Size2(1.0, 1.0));
	void draw_set_transform_matrix(const Transform2D &p_matrix);
	void draw_animation_slice(double p_animation_length, double p_slice_begin, double p_slice_end, double p_offset = 0);
	void draw_end_animation();

	static CanvasItem *get_current_item_drawn();

	/* RECT / TRANSFORM */

	void set_as_top_level(bool p_top_level);
	bool is_set_as_top_level() const;

	void set_draw_behind_parent(bool p_enable);
	bool is_draw_behind_parent_enabled() const;

	CanvasItem *get_parent_item() const;

	virtual Transform2D get_transform() const = 0;

	virtual Transform2D get_global_transform() const;
	virtual Transform2D get_global_transform_with_canvas() const;
	virtual Transform2D get_screen_transform() const;

	CanvasItem *get_top_level() const;
	_FORCE_INLINE_ RID get_canvas_item() const {
		return canvas_item;
	}

	void set_block_transform_notify(bool p_enable);
	bool is_block_transform_notify_enabled() const;

	Transform2D get_canvas_transform() const;
	Transform2D get_viewport_transform() const;
	Rect2 get_viewport_rect() const;
	RID get_viewport_rid() const;
	RID get_canvas() const;
	ObjectID get_canvas_layer_instance_id() const;
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

	void force_update_transform();

	virtual void set_texture_filter(TextureFilter p_texture_filter);
	TextureFilter get_texture_filter() const;

	virtual void set_texture_repeat(TextureRepeat p_texture_repeat);
	TextureRepeat get_texture_repeat() const;

	// Used by control nodes to retrieve the parent's anchorable area
	virtual Rect2 get_anchorable_rect() const { return Rect2(0, 0, 0, 0); };

	int get_canvas_layer() const;

	CanvasItem();
	~CanvasItem();
};

VARIANT_ENUM_CAST(CanvasItem::TextureFilter)
VARIANT_ENUM_CAST(CanvasItem::TextureRepeat)

class CanvasTexture : public Texture2D {
	GDCLASS(CanvasTexture, Texture2D);
	OBJ_SAVE_TYPE(Texture2D); // Saves derived classes with common type so they can be interchanged.

	Ref<Texture2D> diffuse_texture;
	Ref<Texture2D> normal_texture;
	Ref<Texture2D> specular_texture;
	Color specular = Color(1, 1, 1, 1);
	real_t shininess = 1.0;

	RID canvas_texture;

	CanvasItem::TextureFilter texture_filter = CanvasItem::TEXTURE_FILTER_PARENT_NODE;
	CanvasItem::TextureRepeat texture_repeat = CanvasItem::TEXTURE_REPEAT_PARENT_NODE;

protected:
	static void _bind_methods();

public:
	void set_diffuse_texture(const Ref<Texture2D> &p_diffuse);
	Ref<Texture2D> get_diffuse_texture() const;

	void set_normal_texture(const Ref<Texture2D> &p_normal);
	Ref<Texture2D> get_normal_texture() const;

	void set_specular_texture(const Ref<Texture2D> &p_specular);
	Ref<Texture2D> get_specular_texture() const;

	void set_specular_color(const Color &p_color);
	Color get_specular_color() const;

	void set_specular_shininess(real_t p_shininess);
	real_t get_specular_shininess() const;

	void set_texture_filter(CanvasItem::TextureFilter p_filter);
	CanvasItem::TextureFilter get_texture_filter() const;

	void set_texture_repeat(CanvasItem::TextureRepeat p_repeat);
	CanvasItem::TextureRepeat get_texture_repeat() const;

	virtual int get_width() const override;
	virtual int get_height() const override;

	virtual bool is_pixel_opaque(int p_x, int p_y) const override;
	virtual bool has_alpha() const override;

	virtual Ref<Image> get_image() const override;

	virtual RID get_rid() const override;

	CanvasTexture();
	~CanvasTexture();
};

#endif // CANVAS_ITEM_H
