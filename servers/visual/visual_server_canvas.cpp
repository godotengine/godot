/*************************************************************************/
/*  visual_server_canvas.cpp                                             */
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
#include "visual_server_canvas.h"
#include "visual_server_global.h"
#include "visual_server_viewport.h"

void VisualServerCanvas::_render_canvas_item_tree(Item *p_canvas_item, const Transform2D &p_transform, const Rect2 &p_clip_rect, const Color &p_modulate, RasterizerCanvas::Light *p_lights) {

	static const int z_range = VS::CANVAS_ITEM_Z_MAX - VS::CANVAS_ITEM_Z_MIN + 1;
	RasterizerCanvas::Item *z_list[z_range];
	RasterizerCanvas::Item *z_last_list[z_range];

	for (int i = 0; i < z_range; i++) {
		z_list[i] = NULL;
		z_last_list[i] = NULL;
	}

	_render_canvas_item(p_canvas_item, p_transform, p_clip_rect, Color(1, 1, 1, 1), 0, z_list, z_last_list, NULL, NULL);

	for (int i = 0; i < z_range; i++) {
		if (!z_list[i])
			continue;
		VSG::canvas_render->canvas_render_items(z_list[i], VS::CANVAS_ITEM_Z_MIN + i, p_modulate, p_lights);
	}
}

void VisualServerCanvas::_render_canvas_item(Item *p_canvas_item, const Transform2D &p_transform, const Rect2 &p_clip_rect, const Color &p_modulate, int p_z, RasterizerCanvas::Item **z_list, RasterizerCanvas::Item **z_last_list, Item *p_canvas_clip, Item *p_material_owner) {

	Item *ci = p_canvas_item;

	if (!ci->visible)
		return;

	Rect2 rect = ci->get_rect();
	Transform2D xform = p_transform * ci->xform;
	Rect2 global_rect = xform.xform(rect);
	global_rect.pos += p_clip_rect.pos;

	if (ci->use_parent_material && p_material_owner)
		ci->material_owner = p_material_owner;
	else {
		p_material_owner = ci;
		ci->material_owner = NULL;
	}

	Color modulate(ci->modulate.r * p_modulate.r, ci->modulate.g * p_modulate.g, ci->modulate.b * p_modulate.b, ci->modulate.a * p_modulate.a);

	if (modulate.a < 0.007)
		return;

	int child_item_count = ci->child_items.size();
	Item **child_items = (Item **)alloca(child_item_count * sizeof(Item *));
	copymem(child_items, ci->child_items.ptr(), child_item_count * sizeof(Item *));

	if (ci->clip) {
		if (p_canvas_clip != NULL) {
			ci->final_clip_rect = p_canvas_clip->final_clip_rect.clip(global_rect);
		} else {
			ci->final_clip_rect = global_rect;
		}
		ci->final_clip_owner = ci;

	} else {
		ci->final_clip_owner = p_canvas_clip;
	}

	if (ci->sort_y) {

		SortArray<Item *, ItemPtrSort> sorter;
		sorter.sort(child_items, child_item_count);
	}

	if (ci->z_relative)
		p_z = CLAMP(p_z + ci->z, VS::CANVAS_ITEM_Z_MIN, VS::CANVAS_ITEM_Z_MAX);
	else
		p_z = ci->z;

	for (int i = 0; i < child_item_count; i++) {

		if (!child_items[i]->behind)
			continue;
		_render_canvas_item(child_items[i], xform, p_clip_rect, modulate, p_z, z_list, z_last_list, (Item *)ci->final_clip_owner, p_material_owner);
	}

	if (ci->copy_back_buffer) {

		ci->copy_back_buffer->screen_rect = xform.xform(ci->copy_back_buffer->rect).clip(p_clip_rect);
	}

	if ((!ci->commands.empty() && p_clip_rect.intersects(global_rect)) || ci->vp_render || ci->copy_back_buffer) {
		//something to draw?
		ci->final_transform = xform;
		ci->final_modulate = Color(modulate.r * ci->self_modulate.r, modulate.g * ci->self_modulate.g, modulate.b * ci->self_modulate.b, modulate.a * ci->self_modulate.a);
		ci->global_rect_cache = global_rect;
		ci->global_rect_cache.pos -= p_clip_rect.pos;
		ci->light_masked = false;

		int zidx = p_z - VS::CANVAS_ITEM_Z_MIN;

		if (z_last_list[zidx]) {
			z_last_list[zidx]->next = ci;
			z_last_list[zidx] = ci;

		} else {
			z_list[zidx] = ci;
			z_last_list[zidx] = ci;
		}

		ci->next = NULL;
	}

	for (int i = 0; i < child_item_count; i++) {

		if (child_items[i]->behind)
			continue;
		_render_canvas_item(child_items[i], xform, p_clip_rect, modulate, p_z, z_list, z_last_list, (Item *)ci->final_clip_owner, p_material_owner);
	}
}

void VisualServerCanvas::_light_mask_canvas_items(int p_z, RasterizerCanvas::Item *p_canvas_item, RasterizerCanvas::Light *p_masked_lights) {

	if (!p_masked_lights)
		return;

	RasterizerCanvas::Item *ci = p_canvas_item;

	while (ci) {

		RasterizerCanvas::Light *light = p_masked_lights;
		while (light) {

			if (ci->light_mask & light->item_mask && p_z >= light->z_min && p_z <= light->z_max && ci->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {
				ci->light_masked = true;
			}

			light = light->mask_next_ptr;
		}

		ci = ci->next;
	}
}

void VisualServerCanvas::render_canvas(Canvas *p_canvas, const Transform2D &p_transform, RasterizerCanvas::Light *p_lights, RasterizerCanvas::Light *p_masked_lights, const Rect2 &p_clip_rect) {

	VSG::canvas_render->canvas_begin();

	int l = p_canvas->child_items.size();
	Canvas::ChildItem *ci = p_canvas->child_items.ptr();

	bool has_mirror = false;
	for (int i = 0; i < l; i++) {
		if (ci[i].mirror.x || ci[i].mirror.y) {
			has_mirror = true;
			break;
		}
	}

	if (!has_mirror) {

		static const int z_range = VS::CANVAS_ITEM_Z_MAX - VS::CANVAS_ITEM_Z_MIN + 1;
		RasterizerCanvas::Item *z_list[z_range];
		RasterizerCanvas::Item *z_last_list[z_range];

		for (int i = 0; i < z_range; i++) {
			z_list[i] = NULL;
			z_last_list[i] = NULL;
		}
		for (int i = 0; i < l; i++) {
			_render_canvas_item(ci[i].item, p_transform, p_clip_rect, Color(1, 1, 1, 1), 0, z_list, z_last_list, NULL, NULL);
		}

		for (int i = 0; i < z_range; i++) {
			if (!z_list[i])
				continue;

			if (p_masked_lights) {
				_light_mask_canvas_items(VS::CANVAS_ITEM_Z_MIN + i, z_list[i], p_masked_lights);
			}

			VSG::canvas_render->canvas_render_items(z_list[i], VS::CANVAS_ITEM_Z_MIN + i, p_canvas->modulate, p_lights);
		}
	} else {

		for (int i = 0; i < l; i++) {

			Canvas::ChildItem &ci = p_canvas->child_items[i];
			_render_canvas_item_tree(ci.item, p_transform, p_clip_rect, p_canvas->modulate, p_lights);

			//mirroring (useful for scrolling backgrounds)
			if (ci.mirror.x != 0) {

				Transform2D xform2 = p_transform * Transform2D(0, Vector2(ci.mirror.x, 0));
				_render_canvas_item_tree(ci.item, xform2, p_clip_rect, p_canvas->modulate, p_lights);
			}
			if (ci.mirror.y != 0) {

				Transform2D xform2 = p_transform * Transform2D(0, Vector2(0, ci.mirror.y));
				_render_canvas_item_tree(ci.item, xform2, p_clip_rect, p_canvas->modulate, p_lights);
			}
			if (ci.mirror.y != 0 && ci.mirror.x != 0) {

				Transform2D xform2 = p_transform * Transform2D(0, ci.mirror);
				_render_canvas_item_tree(ci.item, xform2, p_clip_rect, p_canvas->modulate, p_lights);
			}
		}
	}
}

RID VisualServerCanvas::canvas_create() {

	Canvas *canvas = memnew(Canvas);
	ERR_FAIL_COND_V(!canvas, RID());
	RID rid = canvas_owner.make_rid(canvas);

	return rid;
}

void VisualServerCanvas::canvas_set_item_mirroring(RID p_canvas, RID p_item, const Point2 &p_mirroring) {

	Canvas *canvas = canvas_owner.getornull(p_canvas);
	ERR_FAIL_COND(!canvas);
	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	int idx = canvas->find_item(canvas_item);
	ERR_FAIL_COND(idx == -1);
	canvas->child_items[idx].mirror = p_mirroring;
}
void VisualServerCanvas::canvas_set_modulate(RID p_canvas, const Color &p_color) {

	Canvas *canvas = canvas_owner.get(p_canvas);
	ERR_FAIL_COND(!canvas);
	canvas->modulate = p_color;
}

RID VisualServerCanvas::canvas_item_create() {

	Item *canvas_item = memnew(Item);
	ERR_FAIL_COND_V(!canvas_item, RID());

	return canvas_item_owner.make_rid(canvas_item);
}

void VisualServerCanvas::canvas_item_set_parent(RID p_item, RID p_parent) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	if (canvas_item->parent.is_valid()) {

		if (canvas_owner.owns(canvas_item->parent)) {

			Canvas *canvas = canvas_owner.get(canvas_item->parent);
			canvas->erase_item(canvas_item);
		} else if (canvas_item_owner.owns(canvas_item->parent)) {

			Item *item_owner = canvas_item_owner.get(canvas_item->parent);
			item_owner->child_items.erase(canvas_item);
		}

		canvas_item->parent = RID();
	}

	if (p_parent.is_valid()) {
		if (canvas_owner.owns(p_parent)) {

			Canvas *canvas = canvas_owner.get(p_parent);
			Canvas::ChildItem ci;
			ci.item = canvas_item;
			canvas->child_items.push_back(ci);
			canvas->children_order_dirty = true;
		} else if (canvas_item_owner.owns(p_parent)) {

			Item *item_owner = canvas_item_owner.get(p_parent);
			item_owner->child_items.push_back(canvas_item);
			item_owner->children_order_dirty = true;

		} else {

			ERR_EXPLAIN("Invalid parent");
			ERR_FAIL();
		}
	}

	canvas_item->parent = p_parent;
}
void VisualServerCanvas::canvas_item_set_visible(RID p_item, bool p_visible) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->visible = p_visible;
}
void VisualServerCanvas::canvas_item_set_light_mask(RID p_item, int p_mask) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->light_mask = p_mask;
}

void VisualServerCanvas::canvas_item_set_transform(RID p_item, const Transform2D &p_transform) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->xform = p_transform;
}
void VisualServerCanvas::canvas_item_set_clip(RID p_item, bool p_clip) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->clip = p_clip;
}
void VisualServerCanvas::canvas_item_set_distance_field_mode(RID p_item, bool p_enable) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->distance_field = p_enable;
}
void VisualServerCanvas::canvas_item_set_custom_rect(RID p_item, bool p_custom_rect, const Rect2 &p_rect) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->custom_rect = p_custom_rect;
	canvas_item->rect = p_rect;
}
void VisualServerCanvas::canvas_item_set_modulate(RID p_item, const Color &p_color) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->modulate = p_color;
}
void VisualServerCanvas::canvas_item_set_self_modulate(RID p_item, const Color &p_color) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->self_modulate = p_color;
}

void VisualServerCanvas::canvas_item_set_draw_behind_parent(RID p_item, bool p_enable) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->behind = p_enable;
}

void VisualServerCanvas::canvas_item_add_line(RID p_item, const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandLine *line = memnew(Item::CommandLine);
	ERR_FAIL_COND(!line);
	line->color = p_color;
	line->from = p_from;
	line->to = p_to;
	line->width = p_width;
	line->antialiased = p_antialiased;
	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(line);
}

void VisualServerCanvas::canvas_item_add_rect(RID p_item, const Rect2 &p_rect, const Color &p_color) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandRect *rect = memnew(Item::CommandRect);
	ERR_FAIL_COND(!rect);
	rect->modulate = p_color;
	rect->rect = p_rect;
	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(rect);
}

void VisualServerCanvas::canvas_item_add_circle(RID p_item, const Point2 &p_pos, float p_radius, const Color &p_color) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandCircle *circle = memnew(Item::CommandCircle);
	ERR_FAIL_COND(!circle);
	circle->color = p_color;
	circle->pos = p_pos;
	circle->radius = p_radius;

	canvas_item->commands.push_back(circle);
}

void VisualServerCanvas::canvas_item_add_texture_rect(RID p_item, const Rect2 &p_rect, RID p_texture, bool p_tile, const Color &p_modulate, bool p_transpose) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandRect *rect = memnew(Item::CommandRect);
	ERR_FAIL_COND(!rect);
	rect->modulate = p_modulate;
	rect->rect = p_rect;
	rect->flags = 0;
	if (p_tile) {
		rect->flags |= RasterizerCanvas::CANVAS_RECT_TILE;
		rect->flags |= RasterizerCanvas::CANVAS_RECT_REGION;
		rect->source = Rect2(0, 0, p_rect.size.width, p_rect.size.height);
	}

	if (p_rect.size.x < 0) {

		rect->flags |= RasterizerCanvas::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_rect.size.y < 0) {

		rect->flags |= RasterizerCanvas::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_transpose) {
		rect->flags |= RasterizerCanvas::CANVAS_RECT_TRANSPOSE;
		SWAP(rect->rect.size.x, rect->rect.size.y);
	}
	rect->texture = p_texture;
	canvas_item->rect_dirty = true;
	canvas_item->commands.push_back(rect);
}

void VisualServerCanvas::canvas_item_add_texture_rect_region(RID p_item, const Rect2 &p_rect, RID p_texture, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandRect *rect = memnew(Item::CommandRect);
	ERR_FAIL_COND(!rect);
	rect->modulate = p_modulate;
	rect->rect = p_rect;
	rect->texture = p_texture;
	rect->source = p_src_rect;
	rect->flags = RasterizerCanvas::CANVAS_RECT_REGION;

	if (p_rect.size.x < 0) {

		rect->flags |= RasterizerCanvas::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_rect.size.y < 0) {

		rect->flags |= RasterizerCanvas::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_transpose) {
		rect->flags |= RasterizerCanvas::CANVAS_RECT_TRANSPOSE;
		SWAP(rect->rect.size.x, rect->rect.size.y);
	}

	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(rect);
}

void VisualServerCanvas::canvas_item_add_nine_patch(RID p_item, const Rect2 &p_rect, const Rect2 &p_source, RID p_texture, const Vector2 &p_topleft, const Vector2 &p_bottomright, VS::NinePatchAxisMode p_x_axis_mode, VS::NinePatchAxisMode p_y_axis_mode, bool p_draw_center, const Color &p_modulate) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandNinePatch *style = memnew(Item::CommandNinePatch);
	ERR_FAIL_COND(!style);
	style->texture = p_texture;
	style->rect = p_rect;
	style->source = p_source;
	style->draw_center = p_draw_center;
	style->color = p_modulate;
	style->margin[MARGIN_LEFT] = p_topleft.x;
	style->margin[MARGIN_TOP] = p_topleft.y;
	style->margin[MARGIN_RIGHT] = p_bottomright.x;
	style->margin[MARGIN_BOTTOM] = p_bottomright.y;
	style->axis_x = p_x_axis_mode;
	style->axis_y = p_y_axis_mode;
	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(style);
}
void VisualServerCanvas::canvas_item_add_primitive(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture, float p_width) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandPrimitive *prim = memnew(Item::CommandPrimitive);
	ERR_FAIL_COND(!prim);
	prim->texture = p_texture;
	prim->points = p_points;
	prim->uvs = p_uvs;
	prim->colors = p_colors;
	prim->width = p_width;
	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(prim);
}

void VisualServerCanvas::canvas_item_add_polygon(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);
#ifdef DEBUG_ENABLED
	int pointcount = p_points.size();
	ERR_FAIL_COND(pointcount < 3);
	int color_size = p_colors.size();
	int uv_size = p_uvs.size();
	ERR_FAIL_COND(color_size != 0 && color_size != 1 && color_size != pointcount);
	ERR_FAIL_COND(uv_size != 0 && (uv_size != pointcount || !p_texture.is_valid()));
#endif
	Vector<int> indices = Geometry::triangulate_polygon(p_points);

	if (indices.empty()) {

		ERR_EXPLAIN("Bad Polygon!");
		ERR_FAIL_V();
	}

	Item::CommandPolygon *polygon = memnew(Item::CommandPolygon);
	ERR_FAIL_COND(!polygon);
	polygon->texture = p_texture;
	polygon->points = p_points;
	polygon->uvs = p_uvs;
	polygon->colors = p_colors;
	polygon->indices = indices;
	polygon->count = indices.size();
	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(polygon);
}

void VisualServerCanvas::canvas_item_add_triangle_array(RID p_item, const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture, int p_count) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	int ps = p_points.size();
	ERR_FAIL_COND(!p_colors.empty() && p_colors.size() != ps && p_colors.size() != 1);
	ERR_FAIL_COND(!p_uvs.empty() && p_uvs.size() != ps);

	Vector<int> indices = p_indices;

	int count = p_count * 3;

	if (indices.empty()) {

		ERR_FAIL_COND(ps % 3 != 0);
		if (p_count == -1)
			count = ps;
	} else {

		ERR_FAIL_COND(indices.size() % 3 != 0);
		if (p_count == -1)
			count = indices.size();
	}

	Item::CommandPolygon *polygon = memnew(Item::CommandPolygon);
	ERR_FAIL_COND(!polygon);
	polygon->texture = p_texture;
	polygon->points = p_points;
	polygon->uvs = p_uvs;
	polygon->colors = p_colors;
	polygon->indices = indices;
	polygon->count = count;
	canvas_item->rect_dirty = true;

	canvas_item->commands.push_back(polygon);
}

void VisualServerCanvas::canvas_item_add_set_transform(RID p_item, const Transform2D &p_transform) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandTransform *tr = memnew(Item::CommandTransform);
	ERR_FAIL_COND(!tr);
	tr->xform = p_transform;

	canvas_item->commands.push_back(tr);
}

void VisualServerCanvas::canvas_item_add_mesh(RID p_item, const RID &p_mesh, RID p_skeleton) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandMesh *m = memnew(Item::CommandMesh);
	ERR_FAIL_COND(!m);
	m->mesh = p_mesh;
	m->skeleton = p_skeleton;

	canvas_item->commands.push_back(m);
}
void VisualServerCanvas::canvas_item_add_multimesh(RID p_item, RID p_mesh, RID p_skeleton) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandMultiMesh *mm = memnew(Item::CommandMultiMesh);
	ERR_FAIL_COND(!mm);
	mm->multimesh = p_mesh;
	mm->skeleton = p_skeleton;

	canvas_item->commands.push_back(mm);
}

void VisualServerCanvas::canvas_item_add_clip_ignore(RID p_item, bool p_ignore) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	Item::CommandClipIgnore *ci = memnew(Item::CommandClipIgnore);
	ERR_FAIL_COND(!ci);
	ci->ignore = p_ignore;

	canvas_item->commands.push_back(ci);
}
void VisualServerCanvas::canvas_item_set_sort_children_by_y(RID p_item, bool p_enable) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->sort_y = p_enable;
}
void VisualServerCanvas::canvas_item_set_z(RID p_item, int p_z) {

	ERR_FAIL_COND(p_z < VS::CANVAS_ITEM_Z_MIN || p_z > VS::CANVAS_ITEM_Z_MAX);

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->z = p_z;
}
void VisualServerCanvas::canvas_item_set_z_as_relative_to_parent(RID p_item, bool p_enable) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->z_relative = p_enable;
}
void VisualServerCanvas::canvas_item_set_copy_to_backbuffer(RID p_item, bool p_enable, const Rect2 &p_rect) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);
	if (bool(canvas_item->copy_back_buffer != NULL) != p_enable) {
		if (p_enable) {
			canvas_item->copy_back_buffer = memnew(RasterizerCanvas::Item::CopyBackBuffer);
		} else {
			memdelete(canvas_item->copy_back_buffer);
			canvas_item->copy_back_buffer = NULL;
		}
	}

	if (p_enable) {
		canvas_item->copy_back_buffer->rect = p_rect;
		canvas_item->copy_back_buffer->full = p_rect == Rect2();
	}
}

void VisualServerCanvas::canvas_item_clear(RID p_item) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->clear();
}
void VisualServerCanvas::canvas_item_set_draw_index(RID p_item, int p_index) {

	Item *canvas_item = canvas_item_owner.getornull(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->index = p_index;

	if (canvas_item_owner.owns(canvas_item->parent)) {
		Item *canvas_item_parent = canvas_item_owner.getornull(canvas_item->parent);
		canvas_item_parent->children_order_dirty = true;
		return;
	}

	Canvas *canvas = canvas_owner.getornull(canvas_item->parent);
	if (canvas) {
		canvas->children_order_dirty = true;
		return;
	}
}

void VisualServerCanvas::canvas_item_set_material(RID p_item, RID p_material) {

	Item *canvas_item = canvas_item_owner.get(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->material = p_material;
}

void VisualServerCanvas::canvas_item_set_use_parent_material(RID p_item, bool p_enable) {

	Item *canvas_item = canvas_item_owner.get(p_item);
	ERR_FAIL_COND(!canvas_item);

	canvas_item->use_parent_material = p_enable;
}

RID VisualServerCanvas::canvas_light_create() {

	RasterizerCanvas::Light *clight = memnew(RasterizerCanvas::Light);
	clight->light_internal = VSG::canvas_render->light_internal_create();
	return canvas_light_owner.make_rid(clight);
}
void VisualServerCanvas::canvas_light_attach_to_canvas(RID p_light, RID p_canvas) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	if (clight->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.getornull(clight->canvas);
		canvas->lights.erase(clight);
	}

	if (!canvas_owner.owns(p_canvas))
		p_canvas = RID();

	clight->canvas = p_canvas;

	if (clight->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(clight->canvas);
		canvas->lights.insert(clight);
	}
}

void VisualServerCanvas::canvas_light_set_enabled(RID p_light, bool p_enabled) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->enabled = p_enabled;
}
void VisualServerCanvas::canvas_light_set_scale(RID p_light, float p_scale) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->scale = p_scale;
}
void VisualServerCanvas::canvas_light_set_transform(RID p_light, const Transform2D &p_transform) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->xform = p_transform;
}
void VisualServerCanvas::canvas_light_set_texture(RID p_light, RID p_texture) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->texture = p_texture;
}
void VisualServerCanvas::canvas_light_set_texture_offset(RID p_light, const Vector2 &p_offset) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->texture_offset = p_offset;
}
void VisualServerCanvas::canvas_light_set_color(RID p_light, const Color &p_color) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->color = p_color;
}
void VisualServerCanvas::canvas_light_set_height(RID p_light, float p_height) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->height = p_height;
}
void VisualServerCanvas::canvas_light_set_energy(RID p_light, float p_energy) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->energy = p_energy;
}
void VisualServerCanvas::canvas_light_set_z_range(RID p_light, int p_min_z, int p_max_z) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->z_min = p_min_z;
	clight->z_max = p_max_z;
}
void VisualServerCanvas::canvas_light_set_layer_range(RID p_light, int p_min_layer, int p_max_layer) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->layer_max = p_max_layer;
	clight->layer_min = p_min_layer;
}
void VisualServerCanvas::canvas_light_set_item_cull_mask(RID p_light, int p_mask) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->item_mask = p_mask;
}
void VisualServerCanvas::canvas_light_set_item_shadow_cull_mask(RID p_light, int p_mask) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->item_shadow_mask = p_mask;
}
void VisualServerCanvas::canvas_light_set_mode(RID p_light, VS::CanvasLightMode p_mode) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->mode = p_mode;
}

void VisualServerCanvas::canvas_light_set_shadow_enabled(RID p_light, bool p_enabled) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	if (clight->shadow_buffer.is_valid() == p_enabled)
		return;
	if (p_enabled) {
		clight->shadow_buffer = VSG::storage->canvas_light_shadow_buffer_create(clight->shadow_buffer_size);
	} else {
		VSG::storage->free(clight->shadow_buffer);
		clight->shadow_buffer = RID();
	}
}
void VisualServerCanvas::canvas_light_set_shadow_buffer_size(RID p_light, int p_size) {

	ERR_FAIL_COND(p_size < 32 || p_size > 16384);

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	int new_size = nearest_power_of_2(p_size);
	if (new_size == clight->shadow_buffer_size)
		return;

	clight->shadow_buffer_size = nearest_power_of_2(p_size);

	if (clight->shadow_buffer.is_valid()) {
		VSG::storage->free(clight->shadow_buffer);
		clight->shadow_buffer = VSG::storage->canvas_light_shadow_buffer_create(clight->shadow_buffer_size);
	}
}

void VisualServerCanvas::canvas_light_set_shadow_gradient_length(RID p_light, float p_length) {

	ERR_FAIL_COND(p_length < 0);

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->shadow_gradient_length = p_length;
}
void VisualServerCanvas::canvas_light_set_shadow_filter(RID p_light, VS::CanvasLightShadowFilter p_filter) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->shadow_filter = p_filter;
}
void VisualServerCanvas::canvas_light_set_shadow_color(RID p_light, const Color &p_color) {

	RasterizerCanvas::Light *clight = canvas_light_owner.get(p_light);
	ERR_FAIL_COND(!clight);

	clight->shadow_color = p_color;
}

RID VisualServerCanvas::canvas_light_occluder_create() {

	RasterizerCanvas::LightOccluderInstance *occluder = memnew(RasterizerCanvas::LightOccluderInstance);

	return canvas_light_occluder_owner.make_rid(occluder);
}
void VisualServerCanvas::canvas_light_occluder_attach_to_canvas(RID p_occluder, RID p_canvas) {

	RasterizerCanvas::LightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	if (occluder->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(occluder->canvas);
		canvas->occluders.erase(occluder);
	}

	if (!canvas_owner.owns(p_canvas))
		p_canvas = RID();

	occluder->canvas = p_canvas;

	if (occluder->canvas.is_valid()) {

		Canvas *canvas = canvas_owner.get(occluder->canvas);
		canvas->occluders.insert(occluder);
	}
}
void VisualServerCanvas::canvas_light_occluder_set_enabled(RID p_occluder, bool p_enabled) {

	RasterizerCanvas::LightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->enabled = p_enabled;
}
void VisualServerCanvas::canvas_light_occluder_set_polygon(RID p_occluder, RID p_polygon) {

	RasterizerCanvas::LightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	if (occluder->polygon.is_valid()) {
		LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_polygon);
		if (occluder_poly) {
			occluder_poly->owners.erase(occluder);
		}
	}

	occluder->polygon = p_polygon;
	occluder->polygon_buffer = RID();

	if (occluder->polygon.is_valid()) {
		LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_polygon);
		if (!occluder_poly)
			occluder->polygon = RID();
		ERR_FAIL_COND(!occluder_poly);
		occluder_poly->owners.insert(occluder);
		occluder->polygon_buffer = occluder_poly->occluder;
		occluder->aabb_cache = occluder_poly->aabb;
		occluder->cull_cache = occluder_poly->cull_mode;
	}
}
void VisualServerCanvas::canvas_light_occluder_set_transform(RID p_occluder, const Transform2D &p_xform) {

	RasterizerCanvas::LightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->xform = p_xform;
}
void VisualServerCanvas::canvas_light_occluder_set_light_mask(RID p_occluder, int p_mask) {

	RasterizerCanvas::LightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->light_mask = p_mask;
}

RID VisualServerCanvas::canvas_occluder_polygon_create() {

	LightOccluderPolygon *occluder_poly = memnew(LightOccluderPolygon);
	occluder_poly->occluder = VSG::storage->canvas_light_occluder_create();
	return canvas_light_occluder_polygon_owner.make_rid(occluder_poly);
}
void VisualServerCanvas::canvas_occluder_polygon_set_shape(RID p_occluder_polygon, const PoolVector<Vector2> &p_shape, bool p_closed) {

	if (p_shape.size() < 3) {
		canvas_occluder_polygon_set_shape_as_lines(p_occluder_polygon, p_shape);
		return;
	}

	PoolVector<Vector2> lines;
	int lc = p_shape.size() * 2;

	lines.resize(lc - (p_closed ? 0 : 2));
	{
		PoolVector<Vector2>::Write w = lines.write();
		PoolVector<Vector2>::Read r = p_shape.read();

		int max = lc / 2;
		if (!p_closed) {
			max--;
		}
		for (int i = 0; i < max; i++) {

			Vector2 a = r[i];
			Vector2 b = r[(i + 1) % (lc / 2)];
			w[i * 2 + 0] = a;
			w[i * 2 + 1] = b;
		}
	}

	canvas_occluder_polygon_set_shape_as_lines(p_occluder_polygon, lines);
}
void VisualServerCanvas::canvas_occluder_polygon_set_shape_as_lines(RID p_occluder_polygon, const PoolVector<Vector2> &p_shape) {

	LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_occluder_polygon);
	ERR_FAIL_COND(!occluder_poly);
	ERR_FAIL_COND(p_shape.size() & 1);

	int lc = p_shape.size();
	occluder_poly->aabb = Rect2();
	{
		PoolVector<Vector2>::Read r = p_shape.read();
		for (int i = 0; i < lc; i++) {
			if (i == 0)
				occluder_poly->aabb.pos = r[i];
			else
				occluder_poly->aabb.expand_to(r[i]);
		}
	}

	VSG::storage->canvas_light_occluder_set_polylines(occluder_poly->occluder, p_shape);
	for (Set<RasterizerCanvas::LightOccluderInstance *>::Element *E = occluder_poly->owners.front(); E; E = E->next()) {
		E->get()->aabb_cache = occluder_poly->aabb;
	}
}

void VisualServerCanvas::canvas_occluder_polygon_set_cull_mode(RID p_occluder_polygon, VS::CanvasOccluderPolygonCullMode p_mode) {

	LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_occluder_polygon);
	ERR_FAIL_COND(!occluder_poly);
	occluder_poly->cull_mode = p_mode;
	for (Set<RasterizerCanvas::LightOccluderInstance *>::Element *E = occluder_poly->owners.front(); E; E = E->next()) {
		E->get()->cull_cache = p_mode;
	}
}

bool VisualServerCanvas::free(RID p_rid) {

	if (canvas_owner.owns(p_rid)) {

		Canvas *canvas = canvas_owner.get(p_rid);
		ERR_FAIL_COND_V(!canvas, false);

		while (canvas->viewports.size()) {

			VisualServerViewport::Viewport *vp = VSG::viewport->viewport_owner.get(canvas->viewports.front()->get());
			ERR_FAIL_COND_V(!vp, true);

			Map<RID, VisualServerViewport::Viewport::CanvasData>::Element *E = vp->canvas_map.find(p_rid);
			ERR_FAIL_COND_V(!E, true);
			vp->canvas_map.erase(p_rid);

			canvas->viewports.erase(canvas->viewports.front());
		}

		for (int i = 0; i < canvas->child_items.size(); i++) {

			canvas->child_items[i].item->parent = RID();
		}

		for (Set<RasterizerCanvas::Light *>::Element *E = canvas->lights.front(); E; E = E->next()) {

			E->get()->canvas = RID();
		}

		for (Set<RasterizerCanvas::LightOccluderInstance *>::Element *E = canvas->occluders.front(); E; E = E->next()) {

			E->get()->canvas = RID();
		}

		canvas_owner.free(p_rid);

		memdelete(canvas);

	} else if (canvas_item_owner.owns(p_rid)) {

		Item *canvas_item = canvas_item_owner.get(p_rid);
		ERR_FAIL_COND_V(!canvas_item, true);

		if (canvas_item->parent.is_valid()) {

			if (canvas_owner.owns(canvas_item->parent)) {

				Canvas *canvas = canvas_owner.get(canvas_item->parent);
				canvas->erase_item(canvas_item);
			} else if (canvas_item_owner.owns(canvas_item->parent)) {

				Item *item_owner = canvas_item_owner.get(canvas_item->parent);
				item_owner->child_items.erase(canvas_item);
			}
		}

		for (int i = 0; i < canvas_item->child_items.size(); i++) {

			canvas_item->child_items[i]->parent = RID();
		}

		/*
		if (canvas_item->material) {
			canvas_item->material->owners.erase(canvas_item);
		}
		*/

		canvas_item_owner.free(p_rid);

		memdelete(canvas_item);

	} else if (canvas_light_owner.owns(p_rid)) {

		RasterizerCanvas::Light *canvas_light = canvas_light_owner.get(p_rid);
		ERR_FAIL_COND_V(!canvas_light, true);

		if (canvas_light->canvas.is_valid()) {
			Canvas *canvas = canvas_owner.get(canvas_light->canvas);
			if (canvas)
				canvas->lights.erase(canvas_light);
		}

		if (canvas_light->shadow_buffer.is_valid())
			VSG::storage->free(canvas_light->shadow_buffer);

		VSG::canvas_render->light_internal_free(canvas_light->light_internal);

		canvas_light_owner.free(p_rid);
		memdelete(canvas_light);

	} else if (canvas_light_occluder_owner.owns(p_rid)) {

		RasterizerCanvas::LightOccluderInstance *occluder = canvas_light_occluder_owner.get(p_rid);
		ERR_FAIL_COND_V(!occluder, true);

		if (occluder->polygon.is_valid()) {

			LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(occluder->polygon);
			if (occluder_poly) {
				occluder_poly->owners.erase(occluder);
			}
		}

		if (occluder->canvas.is_valid() && canvas_owner.owns(occluder->canvas)) {

			Canvas *canvas = canvas_owner.get(occluder->canvas);
			canvas->occluders.erase(occluder);
		}

		canvas_light_occluder_owner.free(p_rid);
		memdelete(occluder);

	} else if (canvas_light_occluder_polygon_owner.owns(p_rid)) {

		LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get(p_rid);
		ERR_FAIL_COND_V(!occluder_poly, true);
		VSG::storage->free(occluder_poly->occluder);

		while (occluder_poly->owners.size()) {

			occluder_poly->owners.front()->get()->polygon = RID();
			occluder_poly->owners.erase(occluder_poly->owners.front());
		}

		canvas_light_occluder_polygon_owner.free(p_rid);
		memdelete(occluder_poly);
	} else {
		return false;
	}

	return true;
}

VisualServerCanvas::VisualServerCanvas() {
}
