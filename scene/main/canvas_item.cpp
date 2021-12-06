/*************************************************************************/
/*  canvas_item.cpp                                                      */
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

#include "canvas_item.h"

#include "core/object/message_queue.h"
#include "scene/2d/canvas_group.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/window.h"
#include "scene/resources/canvas_item_material.h"
#include "scene/resources/font.h"
#include "scene/resources/multimesh.h"
#include "scene/resources/style_box.h"
#include "scene/resources/world_2d.h"
#include "scene/scene_string_names.h"

#ifdef TOOLS_ENABLED
bool CanvasItem::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	if (_edit_use_rect()) {
		return _edit_get_rect().has_point(p_point);
	} else {
		return p_point.length() < p_tolerance;
	}
}

Transform2D CanvasItem::_edit_get_transform() const {
	return Transform2D(_edit_get_rotation(), _edit_get_position() + _edit_get_pivot());
}
#endif

bool CanvasItem::is_visible_in_tree() const {
	if (!is_inside_tree()) {
		return false;
	}

	const CanvasItem *p = this;

	while (p) {
		if (!p->visible) {
			return false;
		}
		if (p->window && !p->window->is_visible()) {
			return false;
		}
		p = p->get_parent_item();
	}

	return true;
}

void CanvasItem::_propagate_visibility_changed(bool p_visible) {
	if (p_visible && first_draw) { //avoid propagating it twice
		first_draw = false;
	}
	notification(NOTIFICATION_VISIBILITY_CHANGED);

	if (p_visible) {
		update(); //todo optimize
	} else {
		emit_signal(SceneStringNames::get_singleton()->hidden);
	}
	_block();

	for (int i = 0; i < get_child_count(); i++) {
		CanvasItem *c = Object::cast_to<CanvasItem>(get_child(i));

		if (c && c->visible) { //should the top_levels stop propagation? i think so but..
			c->_propagate_visibility_changed(p_visible);
		}
	}

	_unblock();
}

void CanvasItem::show() {
	if (visible) {
		return;
	}

	visible = true;
	RenderingServer::get_singleton()->canvas_item_set_visible(canvas_item, true);

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed(true);
}

void CanvasItem::hide() {
	if (!visible) {
		return;
	}

	visible = false;
	RenderingServer::get_singleton()->canvas_item_set_visible(canvas_item, false);

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed(false);
}

CanvasItem *CanvasItem::current_item_drawn = nullptr;
CanvasItem *CanvasItem::get_current_item_drawn() {
	return current_item_drawn;
}

void CanvasItem::_update_callback() {
	if (!is_inside_tree()) {
		pending_update = false;
		return;
	}

	RenderingServer::get_singleton()->canvas_item_clear(get_canvas_item());
	//todo updating = true - only allow drawing here
	if (is_visible_in_tree()) { //todo optimize this!!
		if (first_draw) {
			notification(NOTIFICATION_VISIBILITY_CHANGED);
			first_draw = false;
		}
		drawing = true;
		current_item_drawn = this;
		notification(NOTIFICATION_DRAW);
		emit_signal(SceneStringNames::get_singleton()->draw);
		GDVIRTUAL_CALL(_draw);
		current_item_drawn = nullptr;
		drawing = false;
	}
	//todo updating = false
	pending_update = false; // don't change to false until finished drawing (avoid recursive update)
}

Transform2D CanvasItem::get_global_transform_with_canvas() const {
	if (canvas_layer) {
		return canvas_layer->get_transform() * get_global_transform();
	} else if (is_inside_tree()) {
		return get_viewport()->get_canvas_transform() * get_global_transform();
	} else {
		return get_global_transform();
	}
}

Transform2D CanvasItem::get_screen_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform2D());
	Transform2D xform = get_global_transform_with_canvas();

	Window *w = Object::cast_to<Window>(get_viewport());
	if (w && !w->is_embedding_subwindows()) {
		Transform2D s;
		s.set_origin(w->get_position());

		xform = s * xform;
	}

	return xform;
}

Transform2D CanvasItem::get_global_transform() const {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!is_inside_tree(), get_transform());
#endif
	if (global_invalid) {
		const CanvasItem *pi = get_parent_item();
		if (pi) {
			global_transform = pi->get_global_transform() * get_transform();
		} else {
			global_transform = get_transform();
		}

		global_invalid = false;
	}

	return global_transform;
}

void CanvasItem::_top_level_raise_self() {
	if (!is_inside_tree()) {
		return;
	}

	if (canvas_layer) {
		RenderingServer::get_singleton()->canvas_item_set_draw_index(canvas_item, canvas_layer->get_sort_index());
	} else {
		RenderingServer::get_singleton()->canvas_item_set_draw_index(canvas_item, get_viewport()->gui_get_canvas_sort_index());
	}
}

void CanvasItem::_enter_canvas() {
	if ((!Object::cast_to<CanvasItem>(get_parent())) || top_level) {
		Node *n = this;

		canvas_layer = nullptr;

		while (n) {
			canvas_layer = Object::cast_to<CanvasLayer>(n);
			if (canvas_layer) {
				break;
			}
			if (Object::cast_to<Viewport>(n)) {
				break;
			}
			n = n->get_parent();
		}

		RID canvas;
		if (canvas_layer) {
			canvas = canvas_layer->get_canvas();
		} else {
			canvas = get_viewport()->find_world_2d()->get_canvas();
		}

		RenderingServer::get_singleton()->canvas_item_set_parent(canvas_item, canvas);

		group = "root_canvas" + itos(canvas.get_id());

		add_to_group(group);
		if (canvas_layer) {
			canvas_layer->reset_sort_index();
		} else {
			get_viewport()->gui_reset_canvas_sort_index();
		}

		get_tree()->call_group_flags(SceneTree::GROUP_CALL_UNIQUE, group, SNAME("_top_level_raise_self"));

	} else {
		CanvasItem *parent = get_parent_item();
		canvas_layer = parent->canvas_layer;
		RenderingServer::get_singleton()->canvas_item_set_parent(canvas_item, parent->get_canvas_item());
		RenderingServer::get_singleton()->canvas_item_set_draw_index(canvas_item, get_index());
	}

	pending_update = false;
	update();

	notification(NOTIFICATION_ENTER_CANVAS);
}

void CanvasItem::_exit_canvas() {
	notification(NOTIFICATION_EXIT_CANVAS, true); //reverse the notification
	RenderingServer::get_singleton()->canvas_item_set_parent(canvas_item, RID());
	canvas_layer = nullptr;
	group = StringName();
}

void CanvasItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			ERR_FAIL_COND(!is_inside_tree());
			first_draw = true;
			Node *parent = get_parent();
			if (parent) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(parent);
				if (ci) {
					C = ci->children_items.push_back(this);
				}
				if (!ci) {
					//look for a window
					Viewport *viewport = nullptr;

					while (parent) {
						viewport = Object::cast_to<Viewport>(parent);
						if (viewport) {
							break;
						}
						parent = parent->get_parent();
					}

					ERR_FAIL_COND(!viewport);

					window = Object::cast_to<Window>(viewport);
					if (window) {
						window->connect(SceneStringNames::get_singleton()->visibility_changed, callable_mp(this, &CanvasItem::_window_visibility_changed));
					}
				}
			}
			_enter_canvas();

			_update_texture_filter_changed(false);
			_update_texture_repeat_changed(false);

			if (!block_transform_notify && !xform_change.in_list()) {
				get_tree()->xform_change_list.add(&xform_change);
			}
		} break;
		case NOTIFICATION_MOVED_IN_PARENT: {
			if (!is_inside_tree()) {
				break;
			}

			if (group != StringName()) {
				get_tree()->call_group_flags(SceneTree::GROUP_CALL_UNIQUE, group, "_top_level_raise_self");
			} else {
				CanvasItem *p = get_parent_item();
				ERR_FAIL_COND(!p);
				RenderingServer::get_singleton()->canvas_item_set_draw_index(canvas_item, get_index());
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (xform_change.in_list()) {
				get_tree()->xform_change_list.remove(&xform_change);
			}
			_exit_canvas();
			if (C) {
				Object::cast_to<CanvasItem>(get_parent())->children_items.erase(C);
				C = nullptr;
			}
			if (window) {
				window->disconnect(SceneStringNames::get_singleton()->visibility_changed, callable_mp(this, &CanvasItem::_window_visibility_changed));
			}
			global_invalid = true;
		} break;
		case NOTIFICATION_DRAW:
		case NOTIFICATION_TRANSFORM_CHANGED: {
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			emit_signal(SceneStringNames::get_singleton()->visibility_changed);
		} break;
	}
}

void CanvasItem::set_visible(bool p_visible) {
	if (p_visible) {
		show();
	} else {
		hide();
	}
}

void CanvasItem::_window_visibility_changed() {
	if (visible) {
		_propagate_visibility_changed(window->is_visible());
	}
}

bool CanvasItem::is_visible() const {
	return visible;
}

void CanvasItem::update() {
	if (!is_inside_tree()) {
		return;
	}
	if (pending_update) {
		return;
	}

	pending_update = true;

	MessageQueue::get_singleton()->push_call(this, SNAME("_update_callback"));
}

void CanvasItem::set_modulate(const Color &p_modulate) {
	if (modulate == p_modulate) {
		return;
	}

	modulate = p_modulate;
	RenderingServer::get_singleton()->canvas_item_set_modulate(canvas_item, modulate);
}

Color CanvasItem::get_modulate() const {
	return modulate;
}

void CanvasItem::set_as_top_level(bool p_top_level) {
	if (top_level == p_top_level) {
		return;
	}

	if (!is_inside_tree()) {
		top_level = p_top_level;
		return;
	}

	_exit_canvas();
	top_level = p_top_level;
	_enter_canvas();

	_notify_transform();
}

bool CanvasItem::is_set_as_top_level() const {
	return top_level;
}

CanvasItem *CanvasItem::get_parent_item() const {
	if (top_level) {
		return nullptr;
	}

	return Object::cast_to<CanvasItem>(get_parent());
}

void CanvasItem::set_self_modulate(const Color &p_self_modulate) {
	if (self_modulate == p_self_modulate) {
		return;
	}

	self_modulate = p_self_modulate;
	RenderingServer::get_singleton()->canvas_item_set_self_modulate(canvas_item, self_modulate);
}

Color CanvasItem::get_self_modulate() const {
	return self_modulate;
}

void CanvasItem::set_light_mask(int p_light_mask) {
	if (light_mask == p_light_mask) {
		return;
	}

	light_mask = p_light_mask;
	RS::get_singleton()->canvas_item_set_light_mask(canvas_item, p_light_mask);
}

int CanvasItem::get_light_mask() const {
	return light_mask;
}

void CanvasItem::item_rect_changed(bool p_size_changed) {
	if (p_size_changed) {
		update();
	}
	emit_signal(SceneStringNames::get_singleton()->item_rect_changed);
}

void CanvasItem::draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, real_t p_width) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_line(canvas_item, p_from, p_to, p_color, p_width);
}

void CanvasItem::draw_polyline(const Vector<Point2> &p_points, const Color &p_color, real_t p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Vector<Color> colors;
	colors.push_back(p_color);
	RenderingServer::get_singleton()->canvas_item_add_polyline(canvas_item, p_points, colors, p_width, p_antialiased);
}

void CanvasItem::draw_polyline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, real_t p_width, bool p_antialiased) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_polyline(canvas_item, p_points, p_colors, p_width, p_antialiased);
}

void CanvasItem::draw_arc(const Vector2 &p_center, real_t p_radius, real_t p_start_angle, real_t p_end_angle, int p_point_count, const Color &p_color, real_t p_width, bool p_antialiased) {
	Vector<Point2> points;
	points.resize(p_point_count);
	const real_t delta_angle = p_end_angle - p_start_angle;
	for (int i = 0; i < p_point_count; i++) {
		real_t theta = (i / (p_point_count - 1.0f)) * delta_angle + p_start_angle;
		points.set(i, p_center + Vector2(Math::cos(theta), Math::sin(theta)) * p_radius);
	}

	draw_polyline(points, p_color, p_width, p_antialiased);
}

void CanvasItem::draw_multiline(const Vector<Point2> &p_points, const Color &p_color, real_t p_width) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Vector<Color> colors;
	colors.push_back(p_color);
	RenderingServer::get_singleton()->canvas_item_add_multiline(canvas_item, p_points, colors, p_width);
}

void CanvasItem::draw_multiline_colors(const Vector<Point2> &p_points, const Vector<Color> &p_colors, real_t p_width) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_multiline(canvas_item, p_points, p_colors, p_width);
}

void CanvasItem::draw_rect(const Rect2 &p_rect, const Color &p_color, bool p_filled, real_t p_width) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	if (p_filled) {
		if (p_width != 1.0) {
			WARN_PRINT("The draw_rect() \"width\" argument has no effect when \"filled\" is \"true\".");
		}

		RenderingServer::get_singleton()->canvas_item_add_rect(canvas_item, p_rect, p_color);
	} else {
		// Thick lines are offset depending on their width to avoid partial overlapping.
		// Thin lines don't require an offset, so don't apply one in this case
		real_t offset;
		if (p_width >= 2) {
			offset = p_width / 2.0;
		} else {
			offset = 0.0;
		}

		RenderingServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(-offset, 0),
				p_rect.position + Size2(p_rect.size.width + offset, 0),
				p_color,
				p_width);
		RenderingServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(p_rect.size.width, offset),
				p_rect.position + Size2(p_rect.size.width, p_rect.size.height - offset),
				p_color,
				p_width);
		RenderingServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(p_rect.size.width + offset, p_rect.size.height),
				p_rect.position + Size2(-offset, p_rect.size.height),
				p_color,
				p_width);
		RenderingServer::get_singleton()->canvas_item_add_line(
				canvas_item,
				p_rect.position + Size2(0, p_rect.size.height - offset),
				p_rect.position + Size2(0, offset),
				p_color,
				p_width);
	}
}

void CanvasItem::draw_circle(const Point2 &p_pos, real_t p_radius, const Color &p_color) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_circle(canvas_item, p_pos, p_radius, p_color);
}

void CanvasItem::draw_texture(const Ref<Texture2D> &p_texture, const Point2 &p_pos, const Color &p_modulate) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_texture.is_null());

	p_texture->draw(canvas_item, p_pos, p_modulate, false);
}

void CanvasItem::draw_texture_rect(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_texture.is_null());
	p_texture->draw_rect(canvas_item, p_rect, p_tile, p_modulate, p_transpose);
}

void CanvasItem::draw_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
	ERR_FAIL_COND(p_texture.is_null());
	p_texture->draw_rect_region(canvas_item, p_rect, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

void CanvasItem::draw_msdf_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, double p_outline, double p_pixel_range) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
	ERR_FAIL_COND(p_texture.is_null());
	RenderingServer::get_singleton()->canvas_item_add_msdf_texture_rect_region(canvas_item, p_rect, p_texture->get_rid(), p_src_rect, p_modulate, p_outline, p_pixel_range);
}

void CanvasItem::draw_style_box(const Ref<StyleBox> &p_style_box, const Rect2 &p_rect) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	ERR_FAIL_COND(p_style_box.is_null());

	p_style_box->draw(canvas_item, p_rect);
}

void CanvasItem::draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, Ref<Texture2D> p_texture, real_t p_width) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RenderingServer::get_singleton()->canvas_item_add_primitive(canvas_item, p_points, p_colors, p_uvs, rid, p_width);
}

void CanvasItem::draw_set_transform(const Point2 &p_offset, real_t p_rot, const Size2 &p_scale) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Transform2D xform(p_rot, p_offset);
	xform.scale_basis(p_scale);
	RenderingServer::get_singleton()->canvas_item_add_set_transform(canvas_item, xform);
}

void CanvasItem::draw_set_transform_matrix(const Transform2D &p_matrix) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_set_transform(canvas_item, p_matrix);
}
void CanvasItem::draw_animation_slice(double p_animation_length, double p_slice_begin, double p_slice_end, double p_offset) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_animation_slice(canvas_item, p_animation_length, p_slice_begin, p_slice_end, p_offset);
}

void CanvasItem::draw_end_animation() {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RenderingServer::get_singleton()->canvas_item_add_animation_slice(canvas_item, 1, 0, 2, 0);
}

void CanvasItem::draw_polygon(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, Ref<Texture2D> p_texture) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();

	RenderingServer::get_singleton()->canvas_item_add_polygon(canvas_item, p_points, p_colors, p_uvs, rid);
}

void CanvasItem::draw_colored_polygon(const Vector<Point2> &p_points, const Color &p_color, const Vector<Point2> &p_uvs, Ref<Texture2D> p_texture) {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");

	Vector<Color> colors;
	colors.push_back(p_color);
	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RenderingServer::get_singleton()->canvas_item_add_polygon(canvas_item, p_points, colors, p_uvs, rid);
}

void CanvasItem::draw_mesh(const Ref<Mesh> &p_mesh, const Ref<Texture2D> &p_texture, const Transform2D &p_transform, const Color &p_modulate) {
	ERR_FAIL_COND(p_mesh.is_null());
	RID texture_rid = p_texture.is_valid() ? p_texture->get_rid() : RID();

	RenderingServer::get_singleton()->canvas_item_add_mesh(canvas_item, p_mesh->get_rid(), p_transform, p_modulate, texture_rid);
}

void CanvasItem::draw_multimesh(const Ref<MultiMesh> &p_multimesh, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND(p_multimesh.is_null());
	RID texture_rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RenderingServer::get_singleton()->canvas_item_add_multimesh(canvas_item, p_multimesh->get_rid(), texture_rid);
}

void CanvasItem::draw_string(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_text, HAlign p_align, real_t p_width, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate, uint16_t p_flags) const {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
	ERR_FAIL_COND(p_font.is_null());
	p_font->draw_string(canvas_item, p_pos, p_text, p_align, p_width, p_size, p_modulate, p_outline_size, p_outline_modulate, p_flags);
}

void CanvasItem::draw_multiline_string(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_text, HAlign p_align, real_t p_width, int p_max_lines, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate, uint16_t p_flags) const {
	ERR_FAIL_COND_MSG(!drawing, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
	ERR_FAIL_COND(p_font.is_null());
	p_font->draw_multiline_string(canvas_item, p_pos, p_text, p_align, p_width, p_max_lines, p_size, p_modulate, p_outline_size, p_outline_modulate, p_flags);
}

real_t CanvasItem::draw_char(const Ref<Font> &p_font, const Point2 &p_pos, const String &p_char, const String &p_next, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate) const {
	ERR_FAIL_COND_V_MSG(!drawing, 0.f, "Drawing is only allowed inside NOTIFICATION_DRAW, _draw() function or 'draw' signal.");
	ERR_FAIL_COND_V(p_font.is_null(), 0.f);
	ERR_FAIL_COND_V(p_char.length() != 1, 0.f);

	return p_font->draw_char(canvas_item, p_pos, p_char[0], p_next.get_data()[0], p_size, p_modulate, p_outline_size, p_outline_modulate);
}

void CanvasItem::_notify_transform(CanvasItem *p_node) {
	/* This check exists to avoid re-propagating the transform
	 * notification down the tree on dirty nodes. It provides
	 * optimization by avoiding redundancy (nodes are dirty, will get the
	 * notification anyway).
	 */

	if (/*p_node->xform_change.in_list() &&*/ p_node->global_invalid) {
		return; //nothing to do
	}

	p_node->global_invalid = true;

	if (p_node->notify_transform && !p_node->xform_change.in_list()) {
		if (!p_node->block_transform_notify) {
			if (p_node->is_inside_tree()) {
				get_tree()->xform_change_list.add(&p_node->xform_change);
			}
		}
	}

	for (CanvasItem *ci : p_node->children_items) {
		if (ci->top_level) {
			continue;
		}
		_notify_transform(ci);
	}
}

Rect2 CanvasItem::get_viewport_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	return get_viewport()->get_visible_rect();
}

RID CanvasItem::get_canvas() const {
	ERR_FAIL_COND_V(!is_inside_tree(), RID());

	if (canvas_layer) {
		return canvas_layer->get_canvas();
	} else {
		return get_viewport()->find_world_2d()->get_canvas();
	}
}

ObjectID CanvasItem::get_canvas_layer_instance_id() const {
	if (canvas_layer) {
		return canvas_layer->get_instance_id();
	} else {
		return ObjectID();
	}
}

CanvasItem *CanvasItem::get_top_level() const {
	CanvasItem *ci = const_cast<CanvasItem *>(this);
	while (!ci->top_level && Object::cast_to<CanvasItem>(ci->get_parent())) {
		ci = Object::cast_to<CanvasItem>(ci->get_parent());
	}

	return ci;
}

Ref<World2D> CanvasItem::get_world_2d() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Ref<World2D>());

	CanvasItem *tl = get_top_level();

	if (tl->get_viewport()) {
		return tl->get_viewport()->find_world_2d();
	} else {
		return Ref<World2D>();
	}
}

RID CanvasItem::get_viewport_rid() const {
	ERR_FAIL_COND_V(!is_inside_tree(), RID());
	return get_viewport()->get_viewport_rid();
}

void CanvasItem::set_block_transform_notify(bool p_enable) {
	block_transform_notify = p_enable;
}

bool CanvasItem::is_block_transform_notify_enabled() const {
	return block_transform_notify;
}

void CanvasItem::set_draw_behind_parent(bool p_enable) {
	if (behind == p_enable) {
		return;
	}
	behind = p_enable;
	RenderingServer::get_singleton()->canvas_item_set_draw_behind_parent(canvas_item, behind);
}

bool CanvasItem::is_draw_behind_parent_enabled() const {
	return behind;
}

void CanvasItem::set_material(const Ref<Material> &p_material) {
	material = p_material;
	RID rid;
	if (material.is_valid()) {
		rid = material->get_rid();
	}
	RS::get_singleton()->canvas_item_set_material(canvas_item, rid);
	notify_property_list_changed(); //properties for material exposed
}

void CanvasItem::set_use_parent_material(bool p_use_parent_material) {
	use_parent_material = p_use_parent_material;
	RS::get_singleton()->canvas_item_set_use_parent_material(canvas_item, p_use_parent_material);
}

bool CanvasItem::get_use_parent_material() const {
	return use_parent_material;
}

Ref<Material> CanvasItem::get_material() const {
	return material;
}

Vector2 CanvasItem::make_canvas_position_local(const Vector2 &screen_point) const {
	ERR_FAIL_COND_V(!is_inside_tree(), screen_point);

	Transform2D local_matrix = (get_canvas_transform() * get_global_transform()).affine_inverse();

	return local_matrix.xform(screen_point);
}

Ref<InputEvent> CanvasItem::make_input_local(const Ref<InputEvent> &p_event) const {
	ERR_FAIL_COND_V(p_event.is_null(), p_event);
	ERR_FAIL_COND_V(!is_inside_tree(), p_event);

	return p_event->xformed_by((get_canvas_transform() * get_global_transform()).affine_inverse());
}

Vector2 CanvasItem::get_global_mouse_position() const {
	ERR_FAIL_COND_V(!get_viewport(), Vector2());
	return get_canvas_transform().affine_inverse().xform(get_viewport()->get_mouse_position());
}

Vector2 CanvasItem::get_local_mouse_position() const {
	ERR_FAIL_COND_V(!get_viewport(), Vector2());

	return get_global_transform().affine_inverse().xform(get_global_mouse_position());
}

void CanvasItem::force_update_transform() {
	ERR_FAIL_COND(!is_inside_tree());
	if (!xform_change.in_list()) {
		return;
	}

	get_tree()->xform_change_list.remove(&xform_change);

	notification(NOTIFICATION_TRANSFORM_CHANGED);
}

void CanvasItem::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_top_level_raise_self"), &CanvasItem::_top_level_raise_self);
	ClassDB::bind_method(D_METHOD("_update_callback"), &CanvasItem::_update_callback);

#ifdef TOOLS_ENABLED
	ClassDB::bind_method(D_METHOD("_edit_set_state", "state"), &CanvasItem::_edit_set_state);
	ClassDB::bind_method(D_METHOD("_edit_get_state"), &CanvasItem::_edit_get_state);
	ClassDB::bind_method(D_METHOD("_edit_set_position", "position"), &CanvasItem::_edit_set_position);
	ClassDB::bind_method(D_METHOD("_edit_get_position"), &CanvasItem::_edit_get_position);
	ClassDB::bind_method(D_METHOD("_edit_set_scale", "scale"), &CanvasItem::_edit_set_scale);
	ClassDB::bind_method(D_METHOD("_edit_get_scale"), &CanvasItem::_edit_get_scale);
	ClassDB::bind_method(D_METHOD("_edit_set_rect", "rect"), &CanvasItem::_edit_set_rect);
	ClassDB::bind_method(D_METHOD("_edit_get_rect"), &CanvasItem::_edit_get_rect);
	ClassDB::bind_method(D_METHOD("_edit_use_rect"), &CanvasItem::_edit_use_rect);
	ClassDB::bind_method(D_METHOD("_edit_set_rotation", "degrees"), &CanvasItem::_edit_set_rotation);
	ClassDB::bind_method(D_METHOD("_edit_get_rotation"), &CanvasItem::_edit_get_rotation);
	ClassDB::bind_method(D_METHOD("_edit_use_rotation"), &CanvasItem::_edit_use_rotation);
	ClassDB::bind_method(D_METHOD("_edit_set_pivot", "pivot"), &CanvasItem::_edit_set_pivot);
	ClassDB::bind_method(D_METHOD("_edit_get_pivot"), &CanvasItem::_edit_get_pivot);
	ClassDB::bind_method(D_METHOD("_edit_use_pivot"), &CanvasItem::_edit_use_pivot);
	ClassDB::bind_method(D_METHOD("_edit_get_transform"), &CanvasItem::_edit_get_transform);
#endif

	ClassDB::bind_method(D_METHOD("get_canvas_item"), &CanvasItem::get_canvas_item);

	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &CanvasItem::set_visible);
	ClassDB::bind_method(D_METHOD("is_visible"), &CanvasItem::is_visible);
	ClassDB::bind_method(D_METHOD("is_visible_in_tree"), &CanvasItem::is_visible_in_tree);
	ClassDB::bind_method(D_METHOD("show"), &CanvasItem::show);
	ClassDB::bind_method(D_METHOD("hide"), &CanvasItem::hide);

	ClassDB::bind_method(D_METHOD("update"), &CanvasItem::update);

	ClassDB::bind_method(D_METHOD("set_as_top_level", "enable"), &CanvasItem::set_as_top_level);
	ClassDB::bind_method(D_METHOD("is_set_as_top_level"), &CanvasItem::is_set_as_top_level);

	ClassDB::bind_method(D_METHOD("set_light_mask", "light_mask"), &CanvasItem::set_light_mask);
	ClassDB::bind_method(D_METHOD("get_light_mask"), &CanvasItem::get_light_mask);

	ClassDB::bind_method(D_METHOD("set_modulate", "modulate"), &CanvasItem::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &CanvasItem::get_modulate);
	ClassDB::bind_method(D_METHOD("set_self_modulate", "self_modulate"), &CanvasItem::set_self_modulate);
	ClassDB::bind_method(D_METHOD("get_self_modulate"), &CanvasItem::get_self_modulate);

	ClassDB::bind_method(D_METHOD("set_draw_behind_parent", "enable"), &CanvasItem::set_draw_behind_parent);
	ClassDB::bind_method(D_METHOD("is_draw_behind_parent_enabled"), &CanvasItem::is_draw_behind_parent_enabled);

	ClassDB::bind_method(D_METHOD("_set_on_top", "on_top"), &CanvasItem::_set_on_top);
	ClassDB::bind_method(D_METHOD("_is_on_top"), &CanvasItem::_is_on_top);
	//ClassDB::bind_method(D_METHOD("get_transform"),&CanvasItem::get_transform);

	ClassDB::bind_method(D_METHOD("draw_line", "from", "to", "color", "width"), &CanvasItem::draw_line, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("draw_polyline", "points", "color", "width", "antialiased"), &CanvasItem::draw_polyline, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_polyline_colors", "points", "colors", "width", "antialiased"), &CanvasItem::draw_polyline_colors, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_arc", "center", "radius", "start_angle", "end_angle", "point_count", "color", "width", "antialiased"), &CanvasItem::draw_arc, DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_multiline", "points", "color", "width"), &CanvasItem::draw_multiline, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("draw_multiline_colors", "points", "colors", "width"), &CanvasItem::draw_multiline_colors, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("draw_rect", "rect", "color", "filled", "width"), &CanvasItem::draw_rect, DEFVAL(true), DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("draw_circle", "position", "radius", "color"), &CanvasItem::draw_circle);
	ClassDB::bind_method(D_METHOD("draw_texture", "texture", "position", "modulate"), &CanvasItem::draw_texture, DEFVAL(Color(1, 1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_texture_rect", "texture", "rect", "tile", "modulate", "transpose"), &CanvasItem::draw_texture_rect, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_texture_rect_region", "texture", "rect", "src_rect", "modulate", "transpose", "clip_uv"), &CanvasItem::draw_texture_rect_region, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("draw_msdf_texture_rect_region", "texture", "rect", "src_rect", "modulate", "outline", "pixel_range"), &CanvasItem::draw_msdf_texture_rect_region, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(0.0), DEFVAL(4.0));
	ClassDB::bind_method(D_METHOD("draw_style_box", "style_box", "rect"), &CanvasItem::draw_style_box);
	ClassDB::bind_method(D_METHOD("draw_primitive", "points", "colors", "uvs", "texture", "width"), &CanvasItem::draw_primitive, DEFVAL(Ref<Texture2D>()), DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("draw_polygon", "points", "colors", "uvs", "texture"), &CanvasItem::draw_polygon, DEFVAL(PackedVector2Array()), DEFVAL(Ref<Texture2D>()));
	ClassDB::bind_method(D_METHOD("draw_colored_polygon", "points", "color", "uvs", "texture"), &CanvasItem::draw_colored_polygon, DEFVAL(PackedVector2Array()), DEFVAL(Ref<Texture2D>()));
	ClassDB::bind_method(D_METHOD("draw_string", "font", "pos", "text", "align", "width", "size", "modulate", "outline_size", "outline_modulate", "flags"), &CanvasItem::draw_string, DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(Font::DEFAULT_FONT_SIZE), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("draw_multiline_string", "font", "pos", "text", "align", "width", "max_lines", "size", "modulate", "outline_size", "outline_modulate", "flags"), &CanvasItem::draw_multiline_string, DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(Font::DEFAULT_FONT_SIZE), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("draw_char", "font", "pos", "char", "next", "size", "modulate", "outline_size", "outline_modulate"), &CanvasItem::draw_char, DEFVAL(""), DEFVAL(Font::DEFAULT_FONT_SIZE), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)));
	ClassDB::bind_method(D_METHOD("draw_mesh", "mesh", "texture", "transform", "modulate"), &CanvasItem::draw_mesh, DEFVAL(Transform2D()), DEFVAL(Color(1, 1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_multimesh", "multimesh", "texture"), &CanvasItem::draw_multimesh);
	ClassDB::bind_method(D_METHOD("draw_set_transform", "position", "rotation", "scale"), &CanvasItem::draw_set_transform, DEFVAL(0.0), DEFVAL(Size2(1.0, 1.0)));
	ClassDB::bind_method(D_METHOD("draw_set_transform_matrix", "xform"), &CanvasItem::draw_set_transform_matrix);
	ClassDB::bind_method(D_METHOD("draw_animation_slice", "animation_length", "slice_begin", "slice_end", "offset"), &CanvasItem::draw_animation_slice, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("draw_end_animation"), &CanvasItem::draw_end_animation);
	ClassDB::bind_method(D_METHOD("get_transform"), &CanvasItem::get_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform"), &CanvasItem::get_global_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform_with_canvas"), &CanvasItem::get_global_transform_with_canvas);
	ClassDB::bind_method(D_METHOD("get_viewport_transform"), &CanvasItem::get_viewport_transform);
	ClassDB::bind_method(D_METHOD("get_viewport_rect"), &CanvasItem::get_viewport_rect);
	ClassDB::bind_method(D_METHOD("get_canvas_transform"), &CanvasItem::get_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_local_mouse_position"), &CanvasItem::get_local_mouse_position);
	ClassDB::bind_method(D_METHOD("get_global_mouse_position"), &CanvasItem::get_global_mouse_position);
	ClassDB::bind_method(D_METHOD("get_canvas"), &CanvasItem::get_canvas);
	ClassDB::bind_method(D_METHOD("get_world_2d"), &CanvasItem::get_world_2d);
	//ClassDB::bind_method(D_METHOD("get_viewport"),&CanvasItem::get_viewport);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CanvasItem::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CanvasItem::get_material);

	ClassDB::bind_method(D_METHOD("set_use_parent_material", "enable"), &CanvasItem::set_use_parent_material);
	ClassDB::bind_method(D_METHOD("get_use_parent_material"), &CanvasItem::get_use_parent_material);

	ClassDB::bind_method(D_METHOD("set_notify_local_transform", "enable"), &CanvasItem::set_notify_local_transform);
	ClassDB::bind_method(D_METHOD("is_local_transform_notification_enabled"), &CanvasItem::is_local_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("set_notify_transform", "enable"), &CanvasItem::set_notify_transform);
	ClassDB::bind_method(D_METHOD("is_transform_notification_enabled"), &CanvasItem::is_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("force_update_transform"), &CanvasItem::force_update_transform);

	ClassDB::bind_method(D_METHOD("make_canvas_position_local", "screen_point"), &CanvasItem::make_canvas_position_local);
	ClassDB::bind_method(D_METHOD("make_input_local", "event"), &CanvasItem::make_input_local);

	ClassDB::bind_method(D_METHOD("set_texture_filter", "mode"), &CanvasItem::set_texture_filter);
	ClassDB::bind_method(D_METHOD("get_texture_filter"), &CanvasItem::get_texture_filter);

	ClassDB::bind_method(D_METHOD("set_texture_repeat", "mode"), &CanvasItem::set_texture_repeat);
	ClassDB::bind_method(D_METHOD("get_texture_repeat"), &CanvasItem::get_texture_repeat);

	ClassDB::bind_method(D_METHOD("set_clip_children", "enable"), &CanvasItem::set_clip_children);
	ClassDB::bind_method(D_METHOD("is_clipping_children"), &CanvasItem::is_clipping_children);

	GDVIRTUAL_BIND(_draw);

	ADD_GROUP("Visibility", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "self_modulate"), "set_self_modulate", "get_self_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_behind_parent"), "set_draw_behind_parent", "is_draw_behind_parent_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "top_level"), "set_as_top_level", "is_set_as_top_level");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_on_top", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "_set_on_top", "_is_on_top"); //compatibility
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_children"), "set_clip_children", "is_clipping_children");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_mask", PROPERTY_HINT_LAYERS_2D_RENDER), "set_light_mask", "get_light_mask");

	ADD_GROUP("Texture", "texture_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_filter", PROPERTY_HINT_ENUM, "Inherit,Nearest,Linear,Nearest Mipmap,Linear Mipmap,Nearest Mipmap Aniso.,Linear Mipmap Aniso."), "set_texture_filter", "get_texture_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_repeat", PROPERTY_HINT_ENUM, "Inherit,Disabled,Enabled,Mirror"), "set_texture_repeat", "get_texture_repeat");

	ADD_GROUP("Material", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "CanvasItemMaterial,ShaderMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_parent_material"), "set_use_parent_material", "get_use_parent_material");
	// ADD_PROPERTY(PropertyInfo(Variant::BOOL,"transform/notify"),"set_transform_notify","is_transform_notify_enabled");

	ADD_SIGNAL(MethodInfo("draw"));
	ADD_SIGNAL(MethodInfo("visibility_changed"));
	ADD_SIGNAL(MethodInfo("hidden"));
	ADD_SIGNAL(MethodInfo("item_rect_changed"));

	BIND_CONSTANT(NOTIFICATION_TRANSFORM_CHANGED);
	BIND_CONSTANT(NOTIFICATION_DRAW);
	BIND_CONSTANT(NOTIFICATION_VISIBILITY_CHANGED);
	BIND_CONSTANT(NOTIFICATION_ENTER_CANVAS);
	BIND_CONSTANT(NOTIFICATION_EXIT_CANVAS);

	BIND_ENUM_CONSTANT(TEXTURE_FILTER_PARENT_NODE);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_NEAREST);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_LINEAR);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_LINEAR_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_MAX);

	BIND_ENUM_CONSTANT(TEXTURE_REPEAT_PARENT_NODE);
	BIND_ENUM_CONSTANT(TEXTURE_REPEAT_DISABLED);
	BIND_ENUM_CONSTANT(TEXTURE_REPEAT_ENABLED);
	BIND_ENUM_CONSTANT(TEXTURE_REPEAT_MIRROR);
	BIND_ENUM_CONSTANT(TEXTURE_REPEAT_MAX);
}

Transform2D CanvasItem::get_canvas_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform2D());

	if (canvas_layer) {
		return canvas_layer->get_transform();
	} else if (Object::cast_to<CanvasItem>(get_parent())) {
		return Object::cast_to<CanvasItem>(get_parent())->get_canvas_transform();
	} else {
		return get_viewport()->get_canvas_transform();
	}
}

Transform2D CanvasItem::get_viewport_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform2D());

	if (canvas_layer) {
		if (get_viewport()) {
			return get_viewport()->get_final_transform() * canvas_layer->get_transform();
		} else {
			return canvas_layer->get_transform();
		}

	} else {
		return get_viewport()->get_final_transform() * get_viewport()->get_canvas_transform();
	}
}

void CanvasItem::set_notify_local_transform(bool p_enable) {
	notify_local_transform = p_enable;
}

bool CanvasItem::is_local_transform_notification_enabled() const {
	return notify_local_transform;
}

void CanvasItem::set_notify_transform(bool p_enable) {
	if (notify_transform == p_enable) {
		return;
	}

	notify_transform = p_enable;

	if (notify_transform && is_inside_tree()) {
		//this ensures that invalid globals get resolved, so notifications can be received
		get_global_transform();
	}
}

bool CanvasItem::is_transform_notification_enabled() const {
	return notify_transform;
}

int CanvasItem::get_canvas_layer() const {
	if (canvas_layer) {
		return canvas_layer->get_layer();
	} else {
		return 0;
	}
}

void CanvasItem::_update_texture_filter_changed(bool p_propagate) {
	if (!is_inside_tree()) {
		return;
	}

	if (texture_filter == TEXTURE_FILTER_PARENT_NODE) {
		CanvasItem *parent_item = get_parent_item();
		if (parent_item) {
			texture_filter_cache = parent_item->texture_filter_cache;
		} else {
			texture_filter_cache = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT;
		}
	} else {
		texture_filter_cache = RS::CanvasItemTextureFilter(texture_filter);
	}
	RS::get_singleton()->canvas_item_set_default_texture_filter(get_canvas_item(), texture_filter_cache);
	update();

	if (p_propagate) {
		for (CanvasItem *E : children_items) {
			if (!E->top_level && E->texture_filter == TEXTURE_FILTER_PARENT_NODE) {
				E->_update_texture_filter_changed(true);
			}
		}
	}
}

void CanvasItem::set_texture_filter(TextureFilter p_texture_filter) {
	ERR_FAIL_INDEX(p_texture_filter, TEXTURE_FILTER_MAX);
	if (texture_filter == p_texture_filter) {
		return;
	}
	texture_filter = p_texture_filter;
	_update_texture_filter_changed(true);
	notify_property_list_changed();
}

CanvasItem::TextureFilter CanvasItem::get_texture_filter() const {
	return texture_filter;
}

void CanvasItem::_update_texture_repeat_changed(bool p_propagate) {
	if (!is_inside_tree()) {
		return;
	}

	if (texture_repeat == TEXTURE_REPEAT_PARENT_NODE) {
		CanvasItem *parent_item = get_parent_item();
		if (parent_item) {
			texture_repeat_cache = parent_item->texture_repeat_cache;
		} else {
			texture_repeat_cache = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT;
		}
	} else {
		texture_repeat_cache = RS::CanvasItemTextureRepeat(texture_repeat);
	}
	RS::get_singleton()->canvas_item_set_default_texture_repeat(get_canvas_item(), texture_repeat_cache);
	update();
	if (p_propagate) {
		for (CanvasItem *E : children_items) {
			if (!E->top_level && E->texture_repeat == TEXTURE_REPEAT_PARENT_NODE) {
				E->_update_texture_repeat_changed(true);
			}
		}
	}
}

void CanvasItem::set_texture_repeat(TextureRepeat p_texture_repeat) {
	ERR_FAIL_INDEX(p_texture_repeat, TEXTURE_REPEAT_MAX);
	if (texture_repeat == p_texture_repeat) {
		return;
	}
	texture_repeat = p_texture_repeat;
	_update_texture_repeat_changed(true);
	notify_property_list_changed();
}

void CanvasItem::set_clip_children(bool p_enabled) {
	if (clip_children == p_enabled) {
		return;
	}
	clip_children = p_enabled;

	if (Object::cast_to<CanvasGroup>(this) != nullptr) {
		//avoid accidental bugs, make this not work on CanvasGroup
		return;
	}
	RS::get_singleton()->canvas_item_set_canvas_group_mode(get_canvas_item(), clip_children ? RS::CANVAS_GROUP_MODE_OPAQUE : RS::CANVAS_GROUP_MODE_DISABLED);
}
bool CanvasItem::is_clipping_children() const {
	return clip_children;
}

CanvasItem::TextureRepeat CanvasItem::get_texture_repeat() const {
	return texture_repeat;
}

CanvasItem::CanvasItem() :
		xform_change(this) {
	canvas_item = RenderingServer::get_singleton()->canvas_item_create();
}

CanvasItem::~CanvasItem() {
	RenderingServer::get_singleton()->free(canvas_item);
}

///////////////////////////////////////////////////////////////////

void CanvasTexture::set_diffuse_texture(const Ref<Texture2D> &p_diffuse) {
	ERR_FAIL_COND_MSG(Object::cast_to<CanvasTexture>(p_diffuse.ptr()) != nullptr, "Can't self-assign a CanvasTexture");
	diffuse_texture = p_diffuse;

	RID tex_rid = diffuse_texture.is_valid() ? diffuse_texture->get_rid() : RID();
	RS::get_singleton()->canvas_texture_set_channel(canvas_texture, RS::CANVAS_TEXTURE_CHANNEL_DIFFUSE, tex_rid);
	emit_changed();
}
Ref<Texture2D> CanvasTexture::get_diffuse_texture() const {
	return diffuse_texture;
}

void CanvasTexture::set_normal_texture(const Ref<Texture2D> &p_normal) {
	ERR_FAIL_COND_MSG(Object::cast_to<CanvasTexture>(p_normal.ptr()) != nullptr, "Can't self-assign a CanvasTexture");
	normal_texture = p_normal;
	RID tex_rid = normal_texture.is_valid() ? normal_texture->get_rid() : RID();
	RS::get_singleton()->canvas_texture_set_channel(canvas_texture, RS::CANVAS_TEXTURE_CHANNEL_NORMAL, tex_rid);
}
Ref<Texture2D> CanvasTexture::get_normal_texture() const {
	return normal_texture;
}

void CanvasTexture::set_specular_texture(const Ref<Texture2D> &p_specular) {
	ERR_FAIL_COND_MSG(Object::cast_to<CanvasTexture>(p_specular.ptr()) != nullptr, "Can't self-assign a CanvasTexture");
	specular_texture = p_specular;
	RID tex_rid = specular_texture.is_valid() ? specular_texture->get_rid() : RID();
	RS::get_singleton()->canvas_texture_set_channel(canvas_texture, RS::CANVAS_TEXTURE_CHANNEL_SPECULAR, tex_rid);
}

Ref<Texture2D> CanvasTexture::get_specular_texture() const {
	return specular_texture;
}

void CanvasTexture::set_specular_color(const Color &p_color) {
	specular = p_color;
	RS::get_singleton()->canvas_texture_set_shading_parameters(canvas_texture, specular, shininess);
}

Color CanvasTexture::get_specular_color() const {
	return specular;
}

void CanvasTexture::set_specular_shininess(real_t p_shininess) {
	shininess = p_shininess;
	RS::get_singleton()->canvas_texture_set_shading_parameters(canvas_texture, specular, shininess);
}

real_t CanvasTexture::get_specular_shininess() const {
	return shininess;
}

void CanvasTexture::set_texture_filter(CanvasItem::TextureFilter p_filter) {
	texture_filter = p_filter;
	RS::get_singleton()->canvas_texture_set_texture_filter(canvas_texture, RS::CanvasItemTextureFilter(p_filter));
}
CanvasItem::TextureFilter CanvasTexture::get_texture_filter() const {
	return texture_filter;
}

void CanvasTexture::set_texture_repeat(CanvasItem::TextureRepeat p_repeat) {
	texture_repeat = p_repeat;
	RS::get_singleton()->canvas_texture_set_texture_repeat(canvas_texture, RS::CanvasItemTextureRepeat(p_repeat));
}
CanvasItem::TextureRepeat CanvasTexture::get_texture_repeat() const {
	return texture_repeat;
}

int CanvasTexture::get_width() const {
	if (diffuse_texture.is_valid()) {
		return diffuse_texture->get_width();
	} else {
		return 1;
	}
}
int CanvasTexture::get_height() const {
	if (diffuse_texture.is_valid()) {
		return diffuse_texture->get_height();
	} else {
		return 1;
	}
}

bool CanvasTexture::is_pixel_opaque(int p_x, int p_y) const {
	if (diffuse_texture.is_valid()) {
		return diffuse_texture->is_pixel_opaque(p_x, p_y);
	} else {
		return false;
	}
}

bool CanvasTexture::has_alpha() const {
	if (diffuse_texture.is_valid()) {
		return diffuse_texture->has_alpha();
	} else {
		return false;
	}
}

Ref<Image> CanvasTexture::get_image() const {
	if (diffuse_texture.is_valid()) {
		return diffuse_texture->get_image();
	} else {
		return Ref<Image>();
	}
}

RID CanvasTexture::get_rid() const {
	return canvas_texture;
}

void CanvasTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_diffuse_texture", "texture"), &CanvasTexture::set_diffuse_texture);
	ClassDB::bind_method(D_METHOD("get_diffuse_texture"), &CanvasTexture::get_diffuse_texture);

	ClassDB::bind_method(D_METHOD("set_normal_texture", "texture"), &CanvasTexture::set_normal_texture);
	ClassDB::bind_method(D_METHOD("get_normal_texture"), &CanvasTexture::get_normal_texture);

	ClassDB::bind_method(D_METHOD("set_specular_texture", "texture"), &CanvasTexture::set_specular_texture);
	ClassDB::bind_method(D_METHOD("get_specular_texture"), &CanvasTexture::get_specular_texture);

	ClassDB::bind_method(D_METHOD("set_specular_color", "color"), &CanvasTexture::set_specular_color);
	ClassDB::bind_method(D_METHOD("get_specular_color"), &CanvasTexture::get_specular_color);

	ClassDB::bind_method(D_METHOD("set_specular_shininess", "shininess"), &CanvasTexture::set_specular_shininess);
	ClassDB::bind_method(D_METHOD("get_specular_shininess"), &CanvasTexture::get_specular_shininess);

	ClassDB::bind_method(D_METHOD("set_texture_filter", "filter"), &CanvasTexture::set_texture_filter);
	ClassDB::bind_method(D_METHOD("get_texture_filter"), &CanvasTexture::get_texture_filter);

	ClassDB::bind_method(D_METHOD("set_texture_repeat", "repeat"), &CanvasTexture::set_texture_repeat);
	ClassDB::bind_method(D_METHOD("get_texture_repeat"), &CanvasTexture::get_texture_repeat);

	ADD_GROUP("Diffuse", "diffuse_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "diffuse_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_diffuse_texture", "get_diffuse_texture");
	ADD_GROUP("Normalmap", "normal_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "normal_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_normal_texture", "get_normal_texture");
	ADD_GROUP("Specular", "specular_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "specular_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_specular_texture", "get_specular_texture");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "specular_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_specular_color", "get_specular_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "specular_shininess", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_specular_shininess", "get_specular_shininess");
	ADD_GROUP("Texture", "texture_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_filter", PROPERTY_HINT_ENUM, "Inherit,Nearest,Linear,Nearest Mipmap,Linear Mipmap,Nearest Mipmap Aniso.,Linear Mipmap Aniso."), "set_texture_filter", "get_texture_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_repeat", PROPERTY_HINT_ENUM, "Inherit,Disabled,Enabled,Mirror"), "set_texture_repeat", "get_texture_repeat");
}

CanvasTexture::CanvasTexture() {
	canvas_texture = RS::get_singleton()->canvas_texture_create();
}
CanvasTexture::~CanvasTexture() {
	RS::get_singleton()->free(canvas_texture);
}
