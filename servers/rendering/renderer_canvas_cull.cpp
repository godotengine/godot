/**************************************************************************/
/*  renderer_canvas_cull.cpp                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "renderer_canvas_cull.h"

#include "core/config/project_settings.h"
#include "core/math/geometry_2d.h"
#include "renderer_viewport.h"
#include "rendering_server_default.h"
#include "rendering_server_globals.h"
#include "servers/rendering/storage/texture_storage.h"

void RendererCanvasCull::_render_canvas_item_tree(RID p_to_render_target, Canvas::ChildItem *p_child_items, int p_child_item_count, Item *p_canvas_item, const Transform2D &p_transform, const Rect2 &p_clip_rect, const Color &p_modulate, RendererCanvasRender::Light *p_lights, RendererCanvasRender::Light *p_directional_lights, RenderingServer::CanvasItemTextureFilter p_default_filter, RenderingServer::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, uint32_t canvas_cull_mask) {
	RENDER_TIMESTAMP("Cull CanvasItem Tree");

	memset(z_list, 0, z_range * sizeof(RendererCanvasRender::Item *));
	memset(z_last_list, 0, z_range * sizeof(RendererCanvasRender::Item *));

	for (int i = 0; i < p_child_item_count; i++) {
		_cull_canvas_item(p_child_items[i].item, p_transform, p_clip_rect, Color(1, 1, 1, 1), 0, z_list, z_last_list, nullptr, nullptr, true, canvas_cull_mask);
	}
	if (p_canvas_item) {
		_cull_canvas_item(p_canvas_item, p_transform, p_clip_rect, Color(1, 1, 1, 1), 0, z_list, z_last_list, nullptr, nullptr, true, canvas_cull_mask);
	}

	RendererCanvasRender::Item *list = nullptr;
	RendererCanvasRender::Item *list_end = nullptr;

	for (int i = 0; i < z_range; i++) {
		if (!z_list[i]) {
			continue;
		}
		if (!list) {
			list = z_list[i];
			list_end = z_last_list[i];
		} else {
			list_end->next = z_list[i];
			list_end = z_last_list[i];
		}
	}

	RENDER_TIMESTAMP("Render CanvasItems");

	bool sdf_flag;
	RSG::canvas_render->canvas_render_items(p_to_render_target, list, p_modulate, p_lights, p_directional_lights, p_transform, p_default_filter, p_default_repeat, p_snap_2d_vertices_to_pixel, sdf_flag);
	if (sdf_flag) {
		sdf_used = true;
	}
}

void _collect_ysort_children(RendererCanvasCull::Item *p_canvas_item, Transform2D p_transform, RendererCanvasCull::Item *p_material_owner, const Color &p_modulate, RendererCanvasCull::Item **r_items, int &r_index, int p_z) {
	int child_item_count = p_canvas_item->child_items.size();
	RendererCanvasCull::Item **child_items = p_canvas_item->child_items.ptrw();
	for (int i = 0; i < child_item_count; i++) {
		int abs_z = 0;
		if (child_items[i]->visible) {
			if (r_items) {
				r_items[r_index] = child_items[i];
				child_items[i]->ysort_xform = p_transform;
				child_items[i]->ysort_pos = p_transform.xform(child_items[i]->xform.columns[2]);
				child_items[i]->material_owner = child_items[i]->use_parent_material ? p_material_owner : nullptr;
				child_items[i]->ysort_modulate = p_modulate;
				child_items[i]->ysort_index = r_index;
				child_items[i]->ysort_parent_abs_z_index = p_z;

				// Y sorted canvas items are flattened into r_items. Calculate their absolute z index to use when rendering r_items.
				if (child_items[i]->z_relative) {
					abs_z = CLAMP(p_z + child_items[i]->z_index, RS::CANVAS_ITEM_Z_MIN, RS::CANVAS_ITEM_Z_MAX);
				} else {
					abs_z = child_items[i]->z_index;
				}
			}

			r_index++;

			if (child_items[i]->sort_y) {
				_collect_ysort_children(child_items[i], p_transform * child_items[i]->xform, child_items[i]->use_parent_material ? p_material_owner : child_items[i], p_modulate * child_items[i]->modulate, r_items, r_index, abs_z);
			}
		}
	}
}

void _mark_ysort_dirty(RendererCanvasCull::Item *ysort_owner, RID_Owner<RendererCanvasCull::Item, true> &canvas_item_owner) {
	do {
		ysort_owner->ysort_children_count = -1;
		ysort_owner = canvas_item_owner.owns(ysort_owner->parent) ? canvas_item_owner.get_or_null(ysort_owner->parent) : nullptr;
	} while (ysort_owner && ysort_owner->sort_y);
}

void RendererCanvasCull::_attach_canvas_item_for_draw(RendererCanvasCull::Item *ci, RendererCanvasCull::Item *p_canvas_clip, RendererCanvasRender::Item **r_z_list, RendererCanvasRender::Item **r_z_last_list, const Transform2D &xform, const Rect2 &p_clip_rect, Rect2 global_rect, const Color &modulate, int p_z, RendererCanvasCull::Item *p_material_owner, bool p_use_canvas_group, RendererCanvasRender::Item *canvas_group_from, const Transform2D &p_xform) {
	if (ci->copy_back_buffer) {
		ci->copy_back_buffer->screen_rect = xform.xform(ci->copy_back_buffer->rect).intersection(p_clip_rect);
	}

	if (p_use_canvas_group) {
		int zidx = p_z - RS::CANVAS_ITEM_Z_MIN;
		if (canvas_group_from == nullptr) {
			// no list before processing this item, means must put stuff in group from the beginning of list.
			canvas_group_from = r_z_list[zidx];
		} else {
			// there was a list before processing, so begin group from this one.
			canvas_group_from = canvas_group_from->next;
		}

		if (canvas_group_from) {
			// Has a place to begin the group from!

			//compute a global rect (in global coords) for children in the same z layer
			Rect2 rect_accum;
			RendererCanvasRender::Item *c = canvas_group_from;
			while (c) {
				if (c == canvas_group_from) {
					rect_accum = c->global_rect_cache;
				} else {
					rect_accum = rect_accum.merge(c->global_rect_cache);
				}

				c = c->next;
			}

			// We have two choices now, if user has drawn something, we must assume users wants to draw the "mask", so compute the size based on this.
			// If nothing has been drawn, we just take it over and draw it ourselves.
			if (ci->canvas_group->fit_empty && (ci->commands == nullptr || (ci->commands->next == nullptr && ci->commands->type == RendererCanvasCull::Item::Command::TYPE_RECT && (static_cast<RendererCanvasCull::Item::CommandRect *>(ci->commands)->flags & RendererCanvasRender::CANVAS_RECT_IS_GROUP)))) {
				// No commands, or sole command is the one used to draw, so we (re)create the draw command.
				ci->clear();

				if (rect_accum == Rect2()) {
					rect_accum.size = Size2(1, 1);
				}

				rect_accum = rect_accum.grow(ci->canvas_group->fit_margin);

				//draw it?
				RendererCanvasRender::Item::CommandRect *crect = ci->alloc_command<RendererCanvasRender::Item::CommandRect>();

				crect->flags = RendererCanvasRender::CANVAS_RECT_IS_GROUP; // so we can recognize it later
				crect->rect = xform.affine_inverse().xform(rect_accum);
				crect->modulate = Color(1, 1, 1, 1);

				//the global rect is used to do the copying, so update it
				global_rect = rect_accum.grow(ci->canvas_group->clear_margin); //grow again by clear margin
				global_rect.position += p_clip_rect.position;
			} else {
				global_rect.position -= p_clip_rect.position;

				global_rect = global_rect.merge(rect_accum); //must use both rects for this
				global_rect = global_rect.grow(ci->canvas_group->clear_margin); //grow by clear margin

				global_rect.position += p_clip_rect.position;
			}

			// Very important that this is cleared after used in RendererCanvasRender to avoid
			// potential crashes.
			canvas_group_from->canvas_group_owner = ci;
		}
	}

	if (((ci->commands != nullptr || ci->visibility_notifier) && p_clip_rect.intersects(global_rect, true)) || ci->vp_render || ci->copy_back_buffer) {
		//something to draw?

		if (ci->update_when_visible) {
			RenderingServerDefault::redraw_request();
		}

		if (ci->commands != nullptr || ci->copy_back_buffer) {
			ci->final_transform = xform;
			ci->final_modulate = modulate * ci->self_modulate;
			ci->global_rect_cache = global_rect;
			ci->global_rect_cache.position -= p_clip_rect.position;
			ci->light_masked = false;

			int zidx = p_z - RS::CANVAS_ITEM_Z_MIN;

			if (r_z_last_list[zidx]) {
				r_z_last_list[zidx]->next = ci;
				r_z_last_list[zidx] = ci;

			} else {
				r_z_list[zidx] = ci;
				r_z_last_list[zidx] = ci;
			}

			ci->z_final = p_z;

			ci->next = nullptr;
		}

		if (ci->visibility_notifier) {
			if (!ci->visibility_notifier->visible_element.in_list()) {
				visibility_notifier_list.add(&ci->visibility_notifier->visible_element);
				ci->visibility_notifier->just_visible = true;
			}

			ci->visibility_notifier->visible_in_frame = RSG::rasterizer->get_frame_number();
		}
	}
}

void RendererCanvasCull::_cull_canvas_item(Item *p_canvas_item, const Transform2D &p_transform, const Rect2 &p_clip_rect, const Color &p_modulate, int p_z, RendererCanvasRender::Item **r_z_list, RendererCanvasRender::Item **r_z_last_list, Item *p_canvas_clip, Item *p_material_owner, bool allow_y_sort, uint32_t canvas_cull_mask) {
	Item *ci = p_canvas_item;

	if (!ci->visible) {
		return;
	}

	if (!(ci->visibility_layer & canvas_cull_mask)) {
		return;
	}

	if (ci->children_order_dirty) {
		ci->child_items.sort_custom<ItemIndexSort>();
		ci->children_order_dirty = false;
	}

	Rect2 rect = ci->get_rect();

	if (ci->visibility_notifier) {
		if (ci->visibility_notifier->area.size != Vector2()) {
			rect = rect.merge(ci->visibility_notifier->area);
		}
	}

	Transform2D xform = ci->xform;
	if (snapping_2d_transforms_to_pixel) {
		xform.columns[2] = xform.columns[2].floor();
	}
	xform = p_transform * xform;

	Rect2 global_rect = xform.xform(rect);
	global_rect.position += p_clip_rect.position;

	if (ci->use_parent_material && p_material_owner) {
		ci->material_owner = p_material_owner;
	} else {
		p_material_owner = ci;
		ci->material_owner = nullptr;
	}

	Color modulate(ci->modulate.r * p_modulate.r, ci->modulate.g * p_modulate.g, ci->modulate.b * p_modulate.b, ci->modulate.a * p_modulate.a);

	if (modulate.a < 0.007) {
		return;
	}

	int child_item_count = ci->child_items.size();
	Item **child_items = ci->child_items.ptrw();

	if (ci->clip) {
		if (p_canvas_clip != nullptr) {
			ci->final_clip_rect = p_canvas_clip->final_clip_rect.intersection(global_rect);
		} else {
			ci->final_clip_rect = p_clip_rect.intersection(global_rect);
		}
		if (ci->final_clip_rect.size.width < 0.5 || ci->final_clip_rect.size.height < 0.5) {
			// The clip rect area is 0, so don't draw the item.
			return;
		}
		ci->final_clip_rect.position = ci->final_clip_rect.position.round();
		ci->final_clip_rect.size = ci->final_clip_rect.size.round();
		ci->final_clip_owner = ci;

	} else {
		ci->final_clip_owner = p_canvas_clip;
	}

	int parent_z = p_z;
	if (ci->z_relative) {
		p_z = CLAMP(p_z + ci->z_index, RS::CANVAS_ITEM_Z_MIN, RS::CANVAS_ITEM_Z_MAX);
	} else {
		p_z = ci->z_index;
	}

	if (ci->sort_y) {
		if (allow_y_sort) {
			if (ci->ysort_children_count == -1) {
				ci->ysort_children_count = 0;
				_collect_ysort_children(ci, Transform2D(), p_material_owner, Color(1, 1, 1, 1), nullptr, ci->ysort_children_count, p_z);
			}

			child_item_count = ci->ysort_children_count + 1;
			child_items = (Item **)alloca(child_item_count * sizeof(Item *));

			ci->ysort_parent_abs_z_index = parent_z;
			child_items[0] = ci;
			int i = 1;
			_collect_ysort_children(ci, Transform2D(), p_material_owner, Color(1, 1, 1, 1), child_items, i, p_z);
			ci->ysort_xform = ci->xform.affine_inverse();
			ci->ysort_modulate = Color(1, 1, 1, 1);

			SortArray<Item *, ItemPtrSort> sorter;
			sorter.sort(child_items, child_item_count);

			for (i = 0; i < child_item_count; i++) {
				_cull_canvas_item(child_items[i], xform * child_items[i]->ysort_xform, p_clip_rect, modulate * child_items[i]->ysort_modulate, child_items[i]->ysort_parent_abs_z_index, r_z_list, r_z_last_list, (Item *)ci->final_clip_owner, (Item *)child_items[i]->material_owner, false, canvas_cull_mask);
			}
		} else {
			RendererCanvasRender::Item *canvas_group_from = nullptr;
			bool use_canvas_group = ci->canvas_group != nullptr && (ci->canvas_group->fit_empty || ci->commands != nullptr);
			if (use_canvas_group) {
				int zidx = p_z - RS::CANVAS_ITEM_Z_MIN;
				canvas_group_from = r_z_last_list[zidx];
			}

			_attach_canvas_item_for_draw(ci, p_canvas_clip, r_z_list, r_z_last_list, xform, p_clip_rect, global_rect, modulate, p_z, p_material_owner, use_canvas_group, canvas_group_from, xform);
		}
	} else {
		RendererCanvasRender::Item *canvas_group_from = nullptr;
		bool use_canvas_group = ci->canvas_group != nullptr && (ci->canvas_group->fit_empty || ci->commands != nullptr);
		if (use_canvas_group) {
			int zidx = p_z - RS::CANVAS_ITEM_Z_MIN;
			canvas_group_from = r_z_last_list[zidx];
		}

		for (int i = 0; i < child_item_count; i++) {
			if (!child_items[i]->behind && !use_canvas_group) {
				continue;
			}
			_cull_canvas_item(child_items[i], xform, p_clip_rect, modulate, p_z, r_z_list, r_z_last_list, (Item *)ci->final_clip_owner, p_material_owner, true, canvas_cull_mask);
		}
		_attach_canvas_item_for_draw(ci, p_canvas_clip, r_z_list, r_z_last_list, xform, p_clip_rect, global_rect, modulate, p_z, p_material_owner, use_canvas_group, canvas_group_from, xform);
		for (int i = 0; i < child_item_count; i++) {
			if (child_items[i]->behind || use_canvas_group) {
				continue;
			}
			_cull_canvas_item(child_items[i], xform, p_clip_rect, modulate, p_z, r_z_list, r_z_last_list, (Item *)ci->final_clip_owner, p_material_owner, true, canvas_cull_mask);
		}
	}
}

void RendererCanvasCull::render_canvas(RID p_render_target, Canvas *p_canvas, const Transform2D &p_transform, RendererCanvasRender::Light *p_lights, RendererCanvasRender::Light *p_directional_lights, const Rect2 &p_clip_rect, RenderingServer::CanvasItemTextureFilter p_default_filter, RenderingServer::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_transforms_to_pixel, bool p_snap_2d_vertices_to_pixel, uint32_t canvas_cull_mask) {
	RENDER_TIMESTAMP("> Render Canvas");

	sdf_used = false;
	snapping_2d_transforms_to_pixel = p_snap_2d_transforms_to_pixel;

	if (p_canvas->children_order_dirty) {
		p_canvas->child_items.sort();
		p_canvas->children_order_dirty = false;
	}

	int l = p_canvas->child_items.size();
	Canvas::ChildItem *ci = p_canvas->child_items.ptrw();

	bool has_mirror = false;
	for (int i = 0; i < l; i++) {
		if (ci[i].mirror.x || ci[i].mirror.y) {
			has_mirror = true;
			break;
		}
	}

	if (!has_mirror) {
		_render_canvas_item_tree(p_render_target, ci, l, nullptr, p_transform, p_clip_rect, p_canvas->modulate, p_lights, p_directional_lights, p_default_filter, p_default_repeat, p_snap_2d_vertices_to_pixel, canvas_cull_mask);

	} else {
		//used for parallaxlayer mirroring
		for (int i = 0; i < l; i++) {
			const Canvas::ChildItem &ci2 = p_canvas->child_items[i];
			_render_canvas_item_tree(p_render_target, nullptr, 0, ci2.item, p_transform, p_clip_rect, p_canvas->modulate, p_lights, p_directional_lights, p_default_filter, p_default_repeat, p_snap_2d_vertices_to_pixel, canvas_cull_mask);

			//mirroring (useful for scrolling backgrounds)
			if (ci2.mirror.x != 0) {
				Transform2D xform2 = p_transform * Transform2D(0, Vector2(ci2.mirror.x, 0));
				_render_canvas_item_tree(p_render_target, nullptr, 0, ci2.item, xform2, p_clip_rect, p_canvas->modulate, p_lights, p_directional_lights, p_default_filter, p_default_repeat, p_snap_2d_vertices_to_pixel, canvas_cull_mask);
			}
			if (ci2.mirror.y != 0) {
				Transform2D xform2 = p_transform * Transform2D(0, Vector2(0, ci2.mirror.y));
				_render_canvas_item_tree(p_render_target, nullptr, 0, ci2.item, xform2, p_clip_rect, p_canvas->modulate, p_lights, p_directional_lights, p_default_filter, p_default_repeat, p_snap_2d_vertices_to_pixel, canvas_cull_mask);
			}
			if (ci2.mirror.y != 0 && ci2.mirror.x != 0) {
				Transform2D xform2 = p_transform * Transform2D(0, ci2.mirror);
				_render_canvas_item_tree(p_render_target, nullptr, 0, ci2.item, xform2, p_clip_rect, p_canvas->modulate, p_lights, p_directional_lights, p_default_filter, p_default_repeat, p_snap_2d_vertices_to_pixel, canvas_cull_mask);
			}
		}
	}

	RENDER_TIMESTAMP("< Render Canvas");
}

bool RendererCanvasCull::was_sdf_used() {
	return sdf_used;
}

RID RendererCanvasCull::canvas_allocate() {
	return canvas_owner.allocate_rid();
}
void RendererCanvasCull::canvas_initialize(RID p_rid) {
	canvas_owner.initialize_rid(p_rid);
}

void RendererCanvasCull::canvas_set_item_mirroring(RID p_canvas, RID p_item, const Point2 &p_mirroring) {
	Canvas *canvas = canvas_owner.get_or_null(p_canvas);
	ERR_FAIL_NULL(canvas);
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	int idx = canvas->find_item(canvas_item);
	ERR_FAIL_COND(idx == -1);
	canvas->child_items.write[idx].mirror = p_mirroring;
}

void RendererCanvasCull::canvas_set_modulate(RID p_canvas, const Color &p_color) {
	Canvas *canvas = canvas_owner.get_or_null(p_canvas);
	ERR_FAIL_NULL(canvas);
	canvas->modulate = p_color;
}

void RendererCanvasCull::canvas_set_disable_scale(bool p_disable) {
	disable_scale = p_disable;
}

void RendererCanvasCull::canvas_set_parent(RID p_canvas, RID p_parent, float p_scale) {
	Canvas *canvas = canvas_owner.get_or_null(p_canvas);
	ERR_FAIL_NULL(canvas);

	canvas->parent = p_parent;
	canvas->parent_scale = p_scale;
}

RID RendererCanvasCull::canvas_item_allocate() {
	return canvas_item_owner.allocate_rid();
}
void RendererCanvasCull::canvas_item_initialize(RID p_rid) {
	canvas_item_owner.initialize_rid(p_rid);
}

void RendererCanvasCull::canvas_item_set_parent(RID p_item, RID p_parent) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	if (canvas_item->parent.is_valid()) {
		if (canvas_owner.owns(canvas_item->parent)) {
			Canvas *canvas = canvas_owner.get_or_null(canvas_item->parent);
			canvas->erase_item(canvas_item);
		} else if (canvas_item_owner.owns(canvas_item->parent)) {
			Item *item_owner = canvas_item_owner.get_or_null(canvas_item->parent);
			item_owner->child_items.erase(canvas_item);

			if (item_owner->sort_y) {
				_mark_ysort_dirty(item_owner, canvas_item_owner);
			}
		}

		canvas_item->parent = RID();
	}

	if (p_parent.is_valid()) {
		if (canvas_owner.owns(p_parent)) {
			Canvas *canvas = canvas_owner.get_or_null(p_parent);
			Canvas::ChildItem ci;
			ci.item = canvas_item;
			canvas->child_items.push_back(ci);
			canvas->children_order_dirty = true;
		} else if (canvas_item_owner.owns(p_parent)) {
			Item *item_owner = canvas_item_owner.get_or_null(p_parent);
			item_owner->child_items.push_back(canvas_item);
			item_owner->children_order_dirty = true;

			if (item_owner->sort_y) {
				_mark_ysort_dirty(item_owner, canvas_item_owner);
			}

		} else {
			ERR_FAIL_MSG("Invalid parent.");
		}
	}

	canvas_item->parent = p_parent;
}

void RendererCanvasCull::canvas_item_set_visible(RID p_item, bool p_visible) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->visible = p_visible;

	_mark_ysort_dirty(canvas_item, canvas_item_owner);
}

void RendererCanvasCull::canvas_item_set_light_mask(RID p_item, int p_mask) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->light_mask = p_mask;
}

void RendererCanvasCull::canvas_item_set_transform(RID p_item, const Transform2D &p_transform) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->xform = p_transform;
}

void RendererCanvasCull::canvas_item_set_visibility_layer(RID p_item, uint32_t p_visibility_layer) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->visibility_layer = p_visibility_layer;
}

uint32_t RendererCanvasCull::canvas_item_get_visibility_layer(RID p_item) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	if (!canvas_item)
		return 0;
	return canvas_item->visibility_layer;
}

void RendererCanvasCull::canvas_item_set_clip(RID p_item, bool p_clip) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->clip = p_clip;
}

void RendererCanvasCull::canvas_item_set_distance_field_mode(RID p_item, bool p_enable) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->distance_field = p_enable;
}

void RendererCanvasCull::canvas_item_set_custom_rect(RID p_item, bool p_custom_rect, const Rect2 &p_rect) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->custom_rect = p_custom_rect;
	canvas_item->rect = p_rect;
}

void RendererCanvasCull::canvas_item_set_modulate(RID p_item, const Color &p_color) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->modulate = p_color;
}

void RendererCanvasCull::canvas_item_set_self_modulate(RID p_item, const Color &p_color) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->self_modulate = p_color;
}

void RendererCanvasCull::canvas_item_set_draw_behind_parent(RID p_item, bool p_enable) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->behind = p_enable;
}

void RendererCanvasCull::canvas_item_set_update_when_visible(RID p_item, bool p_update) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->update_when_visible = p_update;
}

void RendererCanvasCull::canvas_item_add_line(RID p_item, const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandPrimitive *line = canvas_item->alloc_command<Item::CommandPrimitive>();
	ERR_FAIL_NULL(line);

	Vector2 diff = (p_from - p_to);
	Vector2 dir = diff.orthogonal().normalized();
	Vector2 t = dir * p_width * 0.5;

	Vector2 begin_left;
	Vector2 begin_right;
	Vector2 end_left;
	Vector2 end_right;

	if (p_width >= 0.0) {
		begin_left = p_from + t;
		begin_right = p_from - t;
		end_left = p_to + t;
		end_right = p_to - t;

		line->points[0] = begin_left;
		line->points[1] = begin_right;
		line->points[2] = end_right;
		line->points[3] = end_left;
		line->point_count = 4;
	} else {
		begin_left = p_from;
		begin_right = p_from;
		end_left = p_to;
		end_right = p_to;

		line->points[0] = p_from;
		line->points[1] = p_to;
		line->point_count = 2;
	}
	for (uint32_t i = 0; i < line->point_count; i++) {
		line->colors[i] = p_color;
	}

	if (p_antialiased) {
		// Use the same antialiasing feather size as StyleBoxFlat's default
		// (but doubled, as it's specified for both sides here).
		// This value is empirically determined to provide good antialiasing quality
		// while not making lines appear too soft.
		float border_size = 1.25f;
		if (0.0f <= p_width && p_width < 1.0f) {
			border_size *= p_width;
		}
		Vector2 dir2 = diff.normalized();

		Vector2 border = dir * border_size;
		Vector2 border2 = dir2 * border_size;

		Color transparent = Color(p_color.r, p_color.g, p_color.b, 0.0);

		{
			Item::CommandPrimitive *left_border = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(left_border);

			left_border->points[0] = begin_left;
			left_border->points[1] = begin_left + border;
			left_border->points[2] = end_left + border;
			left_border->points[3] = end_left;

			left_border->colors[0] = p_color;
			left_border->colors[1] = transparent;
			left_border->colors[2] = transparent;
			left_border->colors[3] = p_color;

			left_border->point_count = 4;
		}
		{
			Item::CommandPrimitive *right_border = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(right_border);

			right_border->points[0] = begin_right;
			right_border->points[1] = begin_right - border;
			right_border->points[2] = end_right - border;
			right_border->points[3] = end_right;

			right_border->colors[0] = p_color;
			right_border->colors[1] = transparent;
			right_border->colors[2] = transparent;
			right_border->colors[3] = p_color;

			right_border->point_count = 4;
		}
		{
			Item::CommandPrimitive *top_border = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(top_border);

			top_border->points[0] = begin_left;
			top_border->points[1] = begin_left + border2;
			top_border->points[2] = begin_right + border2;
			top_border->points[3] = begin_right;

			top_border->colors[0] = p_color;
			top_border->colors[1] = transparent;
			top_border->colors[2] = transparent;
			top_border->colors[3] = p_color;

			top_border->point_count = 4;
		}
		{
			Item::CommandPrimitive *bottom_border = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(bottom_border);

			bottom_border->points[0] = end_left;
			bottom_border->points[1] = end_left - border2;
			bottom_border->points[2] = end_right - border2;
			bottom_border->points[3] = end_right;

			bottom_border->colors[0] = p_color;
			bottom_border->colors[1] = transparent;
			bottom_border->colors[2] = transparent;
			bottom_border->colors[3] = p_color;

			bottom_border->point_count = 4;
		}
		{
			Item::CommandPrimitive *top_left_corner = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(top_left_corner);

			top_left_corner->points[0] = begin_left;
			top_left_corner->points[1] = begin_left + border2;
			top_left_corner->points[2] = begin_left + border + border2;
			top_left_corner->points[3] = begin_left + border;

			top_left_corner->colors[0] = p_color;
			top_left_corner->colors[1] = transparent;
			top_left_corner->colors[2] = transparent;
			top_left_corner->colors[3] = transparent;

			top_left_corner->point_count = 4;
		}
		{
			Item::CommandPrimitive *top_right_corner = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(top_right_corner);

			top_right_corner->points[0] = begin_right;
			top_right_corner->points[1] = begin_right + border2;
			top_right_corner->points[2] = begin_right - border + border2;
			top_right_corner->points[3] = begin_right - border;

			top_right_corner->colors[0] = p_color;
			top_right_corner->colors[1] = transparent;
			top_right_corner->colors[2] = transparent;
			top_right_corner->colors[3] = transparent;

			top_right_corner->point_count = 4;
		}
		{
			Item::CommandPrimitive *bottom_left_corner = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(bottom_left_corner);

			bottom_left_corner->points[0] = end_left;
			bottom_left_corner->points[1] = end_left - border2;
			bottom_left_corner->points[2] = end_left + border - border2;
			bottom_left_corner->points[3] = end_left + border;

			bottom_left_corner->colors[0] = p_color;
			bottom_left_corner->colors[1] = transparent;
			bottom_left_corner->colors[2] = transparent;
			bottom_left_corner->colors[3] = transparent;

			bottom_left_corner->point_count = 4;
		}
		{
			Item::CommandPrimitive *bottom_right_corner = canvas_item->alloc_command<Item::CommandPrimitive>();
			ERR_FAIL_NULL(bottom_right_corner);

			bottom_right_corner->points[0] = end_right;
			bottom_right_corner->points[1] = end_right - border2;
			bottom_right_corner->points[2] = end_right - border - border2;
			bottom_right_corner->points[3] = end_right - border;

			bottom_right_corner->colors[0] = p_color;
			bottom_right_corner->colors[1] = transparent;
			bottom_right_corner->colors[2] = transparent;
			bottom_right_corner->colors[3] = transparent;

			bottom_right_corner->point_count = 4;
		}
	}
}

static Vector2 compute_polyline_segment_dir(const Vector<Point2> &p_points, int p_index, const Vector2 &p_prev_segment_dir) {
	int point_count = p_points.size();

	bool is_last_point = (p_index == point_count - 1);

	Vector2 segment_dir;

	if (is_last_point) {
		segment_dir = p_prev_segment_dir;
	} else {
		segment_dir = (p_points[p_index + 1] - p_points[p_index]).normalized();

		if (segment_dir.is_zero_approx()) {
			segment_dir = p_prev_segment_dir;
		}
	}

	return segment_dir;
}

static Vector2 compute_polyline_edge_offset_clamped(const Vector2 &p_segment_dir, const Vector2 &p_prev_segment_dir) {
	Vector2 bisector;
	float length = 1.0f;

	bisector = (p_prev_segment_dir * p_segment_dir.length() - p_segment_dir * p_prev_segment_dir.length()).normalized();

	float angle = atan2f(bisector.cross(p_prev_segment_dir), bisector.dot(p_prev_segment_dir));
	float sin_angle = sinf(angle);

	if (!Math::is_zero_approx(sin_angle) && !p_segment_dir.is_equal_approx(p_prev_segment_dir)) {
		length = 1.0f / sin_angle;
		length = CLAMP(length, -3.0f, 3.0f);
	} else {
		bisector = p_segment_dir.orthogonal();
	}

	if (bisector.is_zero_approx()) {
		bisector = p_segment_dir.orthogonal();
	}

	return bisector * length;
}

void RendererCanvasCull::canvas_item_add_polyline(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, float p_width, bool p_antialiased) {
	ERR_FAIL_COND(p_points.size() < 2);
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Color color = Color(1, 1, 1, 1);

	Vector<int> indices;
	int point_count = p_points.size();

	Item::CommandPolygon *pline = canvas_item->alloc_command<Item::CommandPolygon>();
	ERR_FAIL_NULL(pline);

	if (p_width < 0) {
		if (p_antialiased) {
			WARN_PRINT("Antialiasing is not supported for thin polylines drawn using line strips (`p_width < 0`).");
		}

		pline->primitive = RS::PRIMITIVE_LINE_STRIP;

		if (p_colors.size() == 1 || p_colors.size() == point_count) {
			pline->polygon.create(indices, p_points, p_colors);
		} else {
			Vector<Color> colors;
			if (p_colors.is_empty()) {
				colors.push_back(color);
			} else {
				colors.resize(point_count);
				Color *colors_ptr = colors.ptrw();
				for (int i = 0; i < point_count; i++) {
					if (i < p_colors.size()) {
						color = p_colors[i];
					}
					colors_ptr[i] = color;
				}
			}
			pline->polygon.create(indices, p_points, colors);
		}
		return;
	}

	int polyline_point_count = point_count * 2;

	bool loop = p_points[0].is_equal_approx(p_points[point_count - 1]);
	Vector2 first_segment_dir;
	Vector2 last_segment_dir;

	// Search for first non-zero vector between two segments.
	for (int i = 1; i < point_count; i++) {
		first_segment_dir = (p_points[i] - p_points[i - 1]).normalized();

		if (!first_segment_dir.is_zero_approx()) {
			break;
		}
	}

	// Search for last non-zero vector between two segments.
	for (int i = point_count - 1; i >= 1; i--) {
		last_segment_dir = (p_points[i] - p_points[i - 1]).normalized();

		if (!last_segment_dir.is_zero_approx()) {
			break;
		}
	}

	PackedColorArray colors;
	PackedVector2Array points;

	// Additional 2+2 vertices to antialias begin+end of the middle triangle strip.
	colors.resize(polyline_point_count + ((p_antialiased && !loop) ? 4 : 0));
	points.resize(polyline_point_count + ((p_antialiased && !loop) ? 4 : 0));

	Vector2 *points_ptr = points.ptrw();
	Color *colors_ptr = colors.ptrw();

	if (p_antialiased) {
		// Use the same antialiasing feather size as StyleBoxFlat's default
		// (but doubled, as it's specified for both sides here).
		// This value is empirically determined to provide good antialiasing quality
		// while not making lines appear too soft.
		float border_size = 1.25f;
		if (p_width < 1.0f) {
			border_size *= p_width;
		}
		Color color2 = Color(1, 1, 1, 0);

		Item::CommandPolygon *pline_left = canvas_item->alloc_command<Item::CommandPolygon>();
		ERR_FAIL_NULL(pline_left);

		Item::CommandPolygon *pline_right = canvas_item->alloc_command<Item::CommandPolygon>();
		ERR_FAIL_NULL(pline_right);

		PackedColorArray colors_left;
		PackedVector2Array points_left;

		PackedColorArray colors_right;
		PackedVector2Array points_right;

		// 2+2 additional vertices for begin+end corners.
		// 1 additional vertex to swap the orientation of the triangles within the end corner's quad.
		colors_left.resize(polyline_point_count + (loop ? 0 : 5));
		points_left.resize(polyline_point_count + (loop ? 0 : 5));

		colors_right.resize(polyline_point_count + (loop ? 0 : 5));
		points_right.resize(polyline_point_count + (loop ? 0 : 5));

		Color *colors_left_ptr = colors_left.ptrw();
		Vector2 *points_left_ptr = points_left.ptrw();

		Vector2 *points_right_ptr = points_right.ptrw();
		Color *colors_right_ptr = colors_right.ptrw();

		Vector2 prev_segment_dir;
		for (int i = 0; i < point_count; i++) {
			bool is_first_point = (i == 0);
			bool is_last_point = (i == point_count - 1);

			Vector2 segment_dir = compute_polyline_segment_dir(p_points, i, prev_segment_dir);
			if (is_first_point && loop) {
				prev_segment_dir = last_segment_dir;
			} else if (is_last_point && loop) {
				prev_segment_dir = first_segment_dir;
			}

			Vector2 base_edge_offset;
			if (is_first_point && !loop) {
				base_edge_offset = first_segment_dir.orthogonal();
			} else if (is_last_point && !loop) {
				base_edge_offset = last_segment_dir.orthogonal();
			} else {
				base_edge_offset = compute_polyline_edge_offset_clamped(segment_dir, prev_segment_dir);
			}

			Vector2 edge_offset = base_edge_offset * (p_width * 0.5f);
			Vector2 border = base_edge_offset * border_size;
			Vector2 pos = p_points[i];

			int j = i * 2 + (loop ? 0 : 2);

			points_ptr[j + 0] = pos + edge_offset;
			points_ptr[j + 1] = pos - edge_offset;

			points_left_ptr[j + 0] = pos + edge_offset;
			points_left_ptr[j + 1] = pos + edge_offset + border;

			points_right_ptr[j + 0] = pos - edge_offset;
			points_right_ptr[j + 1] = pos - edge_offset - border;

			if (i < p_colors.size()) {
				color = p_colors[i];
				color2 = Color(color.r, color.g, color.b, 0);
			}

			colors_ptr[j + 0] = color;
			colors_ptr[j + 1] = color;

			colors_left_ptr[j + 0] = color;
			colors_left_ptr[j + 1] = color2;

			colors_right_ptr[j + 0] = color;
			colors_right_ptr[j + 1] = color2;

			if (is_first_point && !loop) {
				Vector2 begin_border = -segment_dir * border_size;

				points_ptr[0] = pos + edge_offset + begin_border;
				points_ptr[1] = pos - edge_offset + begin_border;

				colors_ptr[0] = color2;
				colors_ptr[1] = color2;

				points_left_ptr[0] = pos + edge_offset + begin_border;
				points_left_ptr[1] = pos + edge_offset + begin_border + border;

				colors_left_ptr[0] = color2;
				colors_left_ptr[1] = color2;

				points_right_ptr[0] = pos - edge_offset + begin_border;
				points_right_ptr[1] = pos - edge_offset + begin_border - border;

				colors_right_ptr[0] = color2;
				colors_right_ptr[1] = color2;
			}

			if (is_last_point && !loop) {
				Vector2 end_border = prev_segment_dir * border_size;
				int end_index = polyline_point_count + 2;

				points_ptr[end_index + 0] = pos + edge_offset + end_border;
				points_ptr[end_index + 1] = pos - edge_offset + end_border;

				colors_ptr[end_index + 0] = color2;
				colors_ptr[end_index + 1] = color2;

				// Swap orientation of the triangles within both end corner quads so the visual seams
				// between triangles goes from the edge corner. Done by going back to the edge corner
				// (1 additional vertex / zero-area triangle per left/right corner).
				points_left_ptr[end_index + 0] = pos + edge_offset;
				points_left_ptr[end_index + 1] = pos + edge_offset + end_border + border;
				points_left_ptr[end_index + 2] = pos + edge_offset + end_border;

				colors_left_ptr[end_index + 0] = color;
				colors_left_ptr[end_index + 1] = color2;
				colors_left_ptr[end_index + 2] = color2;

				points_right_ptr[end_index + 0] = pos - edge_offset;
				points_right_ptr[end_index + 1] = pos - edge_offset + end_border - border;
				points_right_ptr[end_index + 2] = pos - edge_offset + end_border;

				colors_right_ptr[end_index + 0] = color;
				colors_right_ptr[end_index + 1] = color2;
				colors_right_ptr[end_index + 2] = color2;
			}

			prev_segment_dir = segment_dir;
		}

		pline_left->primitive = RS::PRIMITIVE_TRIANGLE_STRIP;
		pline_left->polygon.create(indices, points_left, colors_left);

		pline_right->primitive = RS::PRIMITIVE_TRIANGLE_STRIP;
		pline_right->polygon.create(indices, points_right, colors_right);
	} else {
		// Makes a single triangle strip for drawing the line.

		Vector2 prev_segment_dir;
		for (int i = 0; i < point_count; i++) {
			bool is_first_point = (i == 0);
			bool is_last_point = (i == point_count - 1);

			Vector2 segment_dir = compute_polyline_segment_dir(p_points, i, prev_segment_dir);
			if (is_first_point && loop) {
				prev_segment_dir = last_segment_dir;
			} else if (is_last_point && loop) {
				prev_segment_dir = first_segment_dir;
			}

			Vector2 base_edge_offset;
			if (is_first_point && !loop) {
				base_edge_offset = first_segment_dir.orthogonal();
			} else if (is_last_point && !loop) {
				base_edge_offset = last_segment_dir.orthogonal();
			} else {
				base_edge_offset = compute_polyline_edge_offset_clamped(segment_dir, prev_segment_dir);
			}

			Vector2 edge_offset = base_edge_offset * (p_width * 0.5f);
			Vector2 pos = p_points[i];

			points_ptr[i * 2 + 0] = pos + edge_offset;
			points_ptr[i * 2 + 1] = pos - edge_offset;

			if (i < p_colors.size()) {
				color = p_colors[i];
			}

			colors_ptr[i * 2 + 0] = color;
			colors_ptr[i * 2 + 1] = color;

			prev_segment_dir = segment_dir;
		}
	}

	pline->primitive = RS::PRIMITIVE_TRIANGLE_STRIP;
	pline->polygon.create(indices, points, colors);
}

void RendererCanvasCull::canvas_item_add_multiline(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, float p_width) {
	ERR_FAIL_COND(p_points.is_empty() || p_points.size() % 2 != 0);
	ERR_FAIL_COND(p_colors.size() != 1 && p_colors.size() * 2 != p_points.size());

	// TODO: `canvas_item_add_line`(`multiline`, `polyline`) share logic, should factor out.
	if (p_width < 0) {
		Item *canvas_item = canvas_item_owner.get_or_null(p_item);
		ERR_FAIL_NULL(canvas_item);

		Vector<Color> colors;
		if (p_colors.size() == 1) {
			colors = p_colors;
		} else { //} else if (p_colors.size() << 1 == p_points.size()) {
			colors.resize(p_points.size());
			Color *colors_ptr = colors.ptrw();
			for (int i = 0; i < p_colors.size(); i++) {
				Color color = p_colors[i];
				colors_ptr[i * 2 + 0] = color;
				colors_ptr[i * 2 + 1] = color;
			}
		}

		Item::CommandPolygon *pline = canvas_item->alloc_command<Item::CommandPolygon>();
		ERR_FAIL_NULL(pline);
		pline->primitive = RS::PRIMITIVE_LINES;
		pline->polygon.create(Vector<int>(), p_points, colors);
	} else {
		if (p_colors.size() == 1) {
			Color color = p_colors[0];
			for (int i = 0; i < p_points.size() >> 1; i++) {
				Vector2 from = p_points[i * 2 + 0];
				Vector2 to = p_points[i * 2 + 1];

				canvas_item_add_line(p_item, from, to, color, p_width);
			}
		} else { //} else if (p_colors.size() << 1 == p_points.size()) {
			for (int i = 0; i < p_colors.size(); i++) {
				Color color = p_colors[i];
				Vector2 from = p_points[i * 2 + 0];
				Vector2 to = p_points[i * 2 + 1];

				canvas_item_add_line(p_item, from, to, color, p_width);
			}
		}
	}
}

void RendererCanvasCull::canvas_item_add_rect(RID p_item, const Rect2 &p_rect, const Color &p_color) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandRect *rect = canvas_item->alloc_command<Item::CommandRect>();
	ERR_FAIL_NULL(rect);
	rect->modulate = p_color;
	rect->rect = p_rect;
}

void RendererCanvasCull::canvas_item_add_circle(RID p_item, const Point2 &p_pos, float p_radius, const Color &p_color) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandPolygon *circle = canvas_item->alloc_command<Item::CommandPolygon>();
	ERR_FAIL_NULL(circle);

	circle->primitive = RS::PRIMITIVE_TRIANGLES;

	Vector<int> indices;
	Vector<Vector2> points;

	static const int circle_points = 64;

	points.resize(circle_points);
	Vector2 *points_ptr = points.ptrw();
	const real_t circle_point_step = Math_TAU / circle_points;

	for (int i = 0; i < circle_points; i++) {
		float angle = i * circle_point_step;
		points_ptr[i].x = Math::cos(angle) * p_radius;
		points_ptr[i].y = Math::sin(angle) * p_radius;
		points_ptr[i] += p_pos;
	}

	indices.resize((circle_points - 2) * 3);
	int *indices_ptr = indices.ptrw();

	for (int i = 0; i < circle_points - 2; i++) {
		indices_ptr[i * 3 + 0] = 0;
		indices_ptr[i * 3 + 1] = i + 1;
		indices_ptr[i * 3 + 2] = i + 2;
	}

	Vector<Color> color;
	color.push_back(p_color);
	circle->polygon.create(indices, points, color);
}

void RendererCanvasCull::canvas_item_add_texture_rect(RID p_item, const Rect2 &p_rect, RID p_texture, bool p_tile, const Color &p_modulate, bool p_transpose) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandRect *rect = canvas_item->alloc_command<Item::CommandRect>();
	ERR_FAIL_NULL(rect);
	rect->modulate = p_modulate;
	rect->rect = p_rect;
	rect->flags = 0;
	if (p_tile) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_TILE;
		rect->flags |= RendererCanvasRender::CANVAS_RECT_REGION;
		rect->source = Rect2(0, 0, ABS(p_rect.size.width), ABS(p_rect.size.height));
	}

	if (p_rect.size.x < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_rect.size.y < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_transpose) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_TRANSPOSE;
		SWAP(rect->rect.size.x, rect->rect.size.y);
	}

	rect->texture = p_texture;
}

void RendererCanvasCull::canvas_item_add_msdf_texture_rect_region(RID p_item, const Rect2 &p_rect, RID p_texture, const Rect2 &p_src_rect, const Color &p_modulate, int p_outline_size, float p_px_range, float p_scale) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandRect *rect = canvas_item->alloc_command<Item::CommandRect>();
	ERR_FAIL_NULL(rect);
	rect->modulate = p_modulate;
	rect->rect = p_rect;

	rect->texture = p_texture;

	rect->source = p_src_rect;
	rect->flags = RendererCanvasRender::CANVAS_RECT_REGION | RendererCanvasRender::CANVAS_RECT_MSDF;

	if (p_rect.size.x < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_src_rect.size.x < 0) {
		rect->flags ^= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->source.size.x = -rect->source.size.x;
	}
	if (p_rect.size.y < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_src_rect.size.y < 0) {
		rect->flags ^= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->source.size.y = -rect->source.size.y;
	}
	rect->outline = (float)p_outline_size / p_scale / 4.0;
	rect->px_range = p_px_range;
}

void RendererCanvasCull::canvas_item_add_lcd_texture_rect_region(RID p_item, const Rect2 &p_rect, RID p_texture, const Rect2 &p_src_rect, const Color &p_modulate) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandRect *rect = canvas_item->alloc_command<Item::CommandRect>();
	ERR_FAIL_NULL(rect);
	rect->modulate = p_modulate;
	rect->rect = p_rect;

	rect->texture = p_texture;

	rect->source = p_src_rect;
	rect->flags = RendererCanvasRender::CANVAS_RECT_REGION | RendererCanvasRender::CANVAS_RECT_LCD;

	if (p_rect.size.x < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_src_rect.size.x < 0) {
		rect->flags ^= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->source.size.x = -rect->source.size.x;
	}
	if (p_rect.size.y < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_src_rect.size.y < 0) {
		rect->flags ^= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->source.size.y = -rect->source.size.y;
	}
}

void RendererCanvasCull::canvas_item_add_texture_rect_region(RID p_item, const Rect2 &p_rect, RID p_texture, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandRect *rect = canvas_item->alloc_command<Item::CommandRect>();
	ERR_FAIL_NULL(rect);
	rect->modulate = p_modulate;
	rect->rect = p_rect;

	rect->texture = p_texture;

	rect->source = p_src_rect;
	rect->flags = RendererCanvasRender::CANVAS_RECT_REGION;

	if (p_rect.size.x < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->rect.size.x = -rect->rect.size.x;
	}
	if (p_src_rect.size.x < 0) {
		rect->flags ^= RendererCanvasRender::CANVAS_RECT_FLIP_H;
		rect->source.size.x = -rect->source.size.x;
	}
	if (p_rect.size.y < 0) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->rect.size.y = -rect->rect.size.y;
	}
	if (p_src_rect.size.y < 0) {
		rect->flags ^= RendererCanvasRender::CANVAS_RECT_FLIP_V;
		rect->source.size.y = -rect->source.size.y;
	}

	if (p_transpose) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_TRANSPOSE;
		SWAP(rect->rect.size.x, rect->rect.size.y);
	}

	if (p_clip_uv) {
		rect->flags |= RendererCanvasRender::CANVAS_RECT_CLIP_UV;
	}
}

void RendererCanvasCull::canvas_item_add_nine_patch(RID p_item, const Rect2 &p_rect, const Rect2 &p_source, RID p_texture, const Vector2 &p_topleft, const Vector2 &p_bottomright, RS::NinePatchAxisMode p_x_axis_mode, RS::NinePatchAxisMode p_y_axis_mode, bool p_draw_center, const Color &p_modulate) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandNinePatch *style = canvas_item->alloc_command<Item::CommandNinePatch>();
	ERR_FAIL_NULL(style);

	style->texture = p_texture;

	style->rect = p_rect;
	style->source = p_source;
	style->draw_center = p_draw_center;
	style->color = p_modulate;
	style->margin[SIDE_LEFT] = p_topleft.x;
	style->margin[SIDE_TOP] = p_topleft.y;
	style->margin[SIDE_RIGHT] = p_bottomright.x;
	style->margin[SIDE_BOTTOM] = p_bottomright.y;
	style->axis_x = p_x_axis_mode;
	style->axis_y = p_y_axis_mode;
}

void RendererCanvasCull::canvas_item_add_primitive(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture) {
	uint32_t pc = p_points.size();
	ERR_FAIL_COND(pc == 0 || pc > 4);

	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandPrimitive *prim = canvas_item->alloc_command<Item::CommandPrimitive>();
	ERR_FAIL_NULL(prim);

	for (int i = 0; i < p_points.size(); i++) {
		prim->points[i] = p_points[i];
		if (i < p_uvs.size()) {
			prim->uvs[i] = p_uvs[i];
		}
		if (i < p_colors.size()) {
			prim->colors[i] = p_colors[i];
		} else if (p_colors.size()) {
			prim->colors[i] = p_colors[0];
		} else {
			prim->colors[i] = Color(1, 1, 1, 1);
		}
	}

	prim->point_count = p_points.size();

	prim->texture = p_texture;
}

void RendererCanvasCull::canvas_item_add_polygon(RID p_item, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);
#ifdef DEBUG_ENABLED
	int pointcount = p_points.size();
	ERR_FAIL_COND(pointcount < 3);
	int color_size = p_colors.size();
	int uv_size = p_uvs.size();
	ERR_FAIL_COND(color_size != 0 && color_size != 1 && color_size != pointcount);
	ERR_FAIL_COND(uv_size != 0 && (uv_size != pointcount));
#endif
	Vector<int> indices = Geometry2D::triangulate_polygon(p_points);
	ERR_FAIL_COND_MSG(indices.is_empty(), "Invalid polygon data, triangulation failed.");

	Item::CommandPolygon *polygon = canvas_item->alloc_command<Item::CommandPolygon>();
	ERR_FAIL_NULL(polygon);
	polygon->primitive = RS::PRIMITIVE_TRIANGLES;
	polygon->texture = p_texture;
	polygon->polygon.create(indices, p_points, p_colors, p_uvs);
}

void RendererCanvasCull::canvas_item_add_triangle_array(RID p_item, const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, const Vector<int> &p_bones, const Vector<float> &p_weights, RID p_texture, int p_count) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	int vertex_count = p_points.size();
	ERR_FAIL_COND(vertex_count == 0);
	ERR_FAIL_COND(!p_colors.is_empty() && p_colors.size() != vertex_count && p_colors.size() != 1);
	ERR_FAIL_COND(!p_uvs.is_empty() && p_uvs.size() != vertex_count);
	ERR_FAIL_COND(!p_bones.is_empty() && p_bones.size() != vertex_count * 4);
	ERR_FAIL_COND(!p_weights.is_empty() && p_weights.size() != vertex_count * 4);

	Vector<int> indices = p_indices;

	Item::CommandPolygon *polygon = canvas_item->alloc_command<Item::CommandPolygon>();
	ERR_FAIL_NULL(polygon);

	polygon->texture = p_texture;

	polygon->polygon.create(indices, p_points, p_colors, p_uvs, p_bones, p_weights);

	polygon->primitive = RS::PRIMITIVE_TRIANGLES;
}

void RendererCanvasCull::canvas_item_add_set_transform(RID p_item, const Transform2D &p_transform) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandTransform *tr = canvas_item->alloc_command<Item::CommandTransform>();
	ERR_FAIL_NULL(tr);
	tr->xform = p_transform;
}

void RendererCanvasCull::canvas_item_add_mesh(RID p_item, const RID &p_mesh, const Transform2D &p_transform, const Color &p_modulate, RID p_texture) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!p_mesh.is_valid());

	Item::CommandMesh *m = canvas_item->alloc_command<Item::CommandMesh>();
	ERR_FAIL_NULL(m);
	m->mesh = p_mesh;
	if (canvas_item->skeleton.is_valid()) {
		m->mesh_instance = RSG::mesh_storage->mesh_instance_create(p_mesh);
		RSG::mesh_storage->mesh_instance_set_skeleton(m->mesh_instance, canvas_item->skeleton);
	}

	m->texture = p_texture;

	m->transform = p_transform;
	m->modulate = p_modulate;
}

void RendererCanvasCull::canvas_item_add_particles(RID p_item, RID p_particles, RID p_texture) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandParticles *part = canvas_item->alloc_command<Item::CommandParticles>();
	ERR_FAIL_NULL(part);
	part->particles = p_particles;

	part->texture = p_texture;

	//take the chance and request processing for them, at least once until they become visible again
	RSG::particles_storage->particles_request_process(p_particles);
}

void RendererCanvasCull::canvas_item_add_multimesh(RID p_item, RID p_mesh, RID p_texture) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandMultiMesh *mm = canvas_item->alloc_command<Item::CommandMultiMesh>();
	ERR_FAIL_NULL(mm);
	mm->multimesh = p_mesh;

	mm->texture = p_texture;
}

void RendererCanvasCull::canvas_item_add_clip_ignore(RID p_item, bool p_ignore) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandClipIgnore *ci = canvas_item->alloc_command<Item::CommandClipIgnore>();
	ERR_FAIL_NULL(ci);
	ci->ignore = p_ignore;
}

void RendererCanvasCull::canvas_item_add_animation_slice(RID p_item, double p_animation_length, double p_slice_begin, double p_slice_end, double p_offset) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	Item::CommandAnimationSlice *as = canvas_item->alloc_command<Item::CommandAnimationSlice>();
	ERR_FAIL_NULL(as);
	as->animation_length = p_animation_length;
	as->slice_begin = p_slice_begin;
	as->slice_end = p_slice_end;
	as->offset = p_offset;
}

void RendererCanvasCull::canvas_item_set_sort_children_by_y(RID p_item, bool p_enable) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->sort_y = p_enable;

	_mark_ysort_dirty(canvas_item, canvas_item_owner);
}

void RendererCanvasCull::canvas_item_set_z_index(RID p_item, int p_z) {
	ERR_FAIL_COND(p_z < RS::CANVAS_ITEM_Z_MIN || p_z > RS::CANVAS_ITEM_Z_MAX);

	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->z_index = p_z;
}

void RendererCanvasCull::canvas_item_set_z_as_relative_to_parent(RID p_item, bool p_enable) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->z_relative = p_enable;
}

void RendererCanvasCull::canvas_item_attach_skeleton(RID p_item, RID p_skeleton) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);
	if (canvas_item->skeleton == p_skeleton) {
		return;
	}
	canvas_item->skeleton = p_skeleton;

	Item::Command *c = canvas_item->commands;

	while (c) {
		if (c->type == Item::Command::TYPE_MESH) {
			Item::CommandMesh *cm = static_cast<Item::CommandMesh *>(c);
			if (canvas_item->skeleton.is_valid()) {
				if (cm->mesh_instance.is_null()) {
					cm->mesh_instance = RSG::mesh_storage->mesh_instance_create(cm->mesh);
				}
				RSG::mesh_storage->mesh_instance_set_skeleton(cm->mesh_instance, canvas_item->skeleton);
			} else {
				if (cm->mesh_instance.is_valid()) {
					RSG::mesh_storage->mesh_instance_free(cm->mesh_instance);
					cm->mesh_instance = RID();
				}
			}
		}
		c = c->next;
	}
}

void RendererCanvasCull::canvas_item_set_copy_to_backbuffer(RID p_item, bool p_enable, const Rect2 &p_rect) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);
	if (p_enable && (canvas_item->copy_back_buffer == nullptr)) {
		canvas_item->copy_back_buffer = memnew(RendererCanvasRender::Item::CopyBackBuffer);
	}
	if (!p_enable && (canvas_item->copy_back_buffer != nullptr)) {
		memdelete(canvas_item->copy_back_buffer);
		canvas_item->copy_back_buffer = nullptr;
	}

	if (p_enable) {
		canvas_item->copy_back_buffer->rect = p_rect;
		canvas_item->copy_back_buffer->full = p_rect == Rect2();
	}
}

void RendererCanvasCull::canvas_item_clear(RID p_item) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->clear();
#ifdef DEBUG_ENABLED
	if (debug_redraw) {
		canvas_item->debug_redraw_time = debug_redraw_time;
	}
#endif
}

void RendererCanvasCull::canvas_item_set_draw_index(RID p_item, int p_index) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->index = p_index;

	if (canvas_item_owner.owns(canvas_item->parent)) {
		Item *canvas_item_parent = canvas_item_owner.get_or_null(canvas_item->parent);
		canvas_item_parent->children_order_dirty = true;
		return;
	}

	Canvas *canvas = canvas_owner.get_or_null(canvas_item->parent);
	if (canvas) {
		canvas->children_order_dirty = true;
		return;
	}
}

void RendererCanvasCull::canvas_item_set_material(RID p_item, RID p_material) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->material = p_material;
}

void RendererCanvasCull::canvas_item_set_use_parent_material(RID p_item, bool p_enable) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	canvas_item->use_parent_material = p_enable;
}

void RendererCanvasCull::canvas_item_set_visibility_notifier(RID p_item, bool p_enable, const Rect2 &p_area, const Callable &p_enter_callable, const Callable &p_exit_callable) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	if (p_enable) {
		if (!canvas_item->visibility_notifier) {
			canvas_item->visibility_notifier = visibility_notifier_allocator.alloc();
		}
		canvas_item->visibility_notifier->area = p_area;
		canvas_item->visibility_notifier->enter_callable = p_enter_callable;
		canvas_item->visibility_notifier->exit_callable = p_exit_callable;

	} else {
		if (canvas_item->visibility_notifier) {
			visibility_notifier_allocator.free(canvas_item->visibility_notifier);
			canvas_item->visibility_notifier = nullptr;
		}
	}
}

void RendererCanvasCull::canvas_item_set_debug_redraw(bool p_enabled) {
	debug_redraw = p_enabled;
	RSG::canvas_render->set_debug_redraw(p_enabled, debug_redraw_time, debug_redraw_color);
}

bool RendererCanvasCull::canvas_item_get_debug_redraw() const {
	return debug_redraw;
}

void RendererCanvasCull::canvas_item_set_canvas_group_mode(RID p_item, RS::CanvasGroupMode p_mode, float p_clear_margin, bool p_fit_empty, float p_fit_margin, bool p_blur_mipmaps) {
	Item *canvas_item = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(canvas_item);

	if (p_mode == RS::CANVAS_GROUP_MODE_DISABLED) {
		if (canvas_item->canvas_group != nullptr) {
			memdelete(canvas_item->canvas_group);
			canvas_item->canvas_group = nullptr;
		}
	} else {
		if (canvas_item->canvas_group == nullptr) {
			canvas_item->canvas_group = memnew(RendererCanvasRender::Item::CanvasGroup);
		}
		canvas_item->canvas_group->mode = p_mode;
		canvas_item->canvas_group->fit_empty = p_fit_empty;
		canvas_item->canvas_group->fit_margin = p_fit_margin;
		canvas_item->canvas_group->blur_mipmaps = p_blur_mipmaps;
		canvas_item->canvas_group->clear_margin = p_clear_margin;
	}
}

RID RendererCanvasCull::canvas_light_allocate() {
	return canvas_light_owner.allocate_rid();
}
void RendererCanvasCull::canvas_light_initialize(RID p_rid) {
	canvas_light_owner.initialize_rid(p_rid);
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_rid);
	clight->light_internal = RSG::canvas_render->light_create();
}

void RendererCanvasCull::canvas_light_set_mode(RID p_light, RS::CanvasLightMode p_mode) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	if (clight->mode == p_mode) {
		return;
	}

	RID canvas = clight->canvas;

	if (canvas.is_valid()) {
		canvas_light_attach_to_canvas(p_light, RID());
	}

	clight->mode = p_mode;

	if (canvas.is_valid()) {
		canvas_light_attach_to_canvas(p_light, canvas);
	}
}

void RendererCanvasCull::canvas_light_attach_to_canvas(RID p_light, RID p_canvas) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	if (clight->canvas.is_valid()) {
		Canvas *canvas = canvas_owner.get_or_null(clight->canvas);
		if (clight->mode == RS::CANVAS_LIGHT_MODE_POINT) {
			canvas->lights.erase(clight);
		} else {
			canvas->directional_lights.erase(clight);
		}
	}

	if (!canvas_owner.owns(p_canvas)) {
		p_canvas = RID();
	}

	clight->canvas = p_canvas;

	if (clight->canvas.is_valid()) {
		Canvas *canvas = canvas_owner.get_or_null(clight->canvas);
		if (clight->mode == RS::CANVAS_LIGHT_MODE_POINT) {
			canvas->lights.insert(clight);
		} else {
			canvas->directional_lights.insert(clight);
		}
	}
}

void RendererCanvasCull::canvas_light_set_enabled(RID p_light, bool p_enabled) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->enabled = p_enabled;
}

void RendererCanvasCull::canvas_light_set_texture_scale(RID p_light, float p_scale) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->scale = p_scale;
}

void RendererCanvasCull::canvas_light_set_transform(RID p_light, const Transform2D &p_transform) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->xform = p_transform;
}

void RendererCanvasCull::canvas_light_set_texture(RID p_light, RID p_texture) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	if (clight->texture == p_texture) {
		return;
	}
	clight->texture = p_texture;
	clight->version++;
	RSG::canvas_render->light_set_texture(clight->light_internal, p_texture);
}

void RendererCanvasCull::canvas_light_set_texture_offset(RID p_light, const Vector2 &p_offset) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->texture_offset = p_offset;
}

void RendererCanvasCull::canvas_light_set_color(RID p_light, const Color &p_color) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->color = p_color;
}

void RendererCanvasCull::canvas_light_set_height(RID p_light, float p_height) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->height = p_height;
}

void RendererCanvasCull::canvas_light_set_energy(RID p_light, float p_energy) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->energy = p_energy;
}

void RendererCanvasCull::canvas_light_set_z_range(RID p_light, int p_min_z, int p_max_z) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->z_min = p_min_z;
	clight->z_max = p_max_z;
}

void RendererCanvasCull::canvas_light_set_layer_range(RID p_light, int p_min_layer, int p_max_layer) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->layer_max = p_max_layer;
	clight->layer_min = p_min_layer;
}

void RendererCanvasCull::canvas_light_set_item_cull_mask(RID p_light, int p_mask) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->item_mask = p_mask;
}

void RendererCanvasCull::canvas_light_set_item_shadow_cull_mask(RID p_light, int p_mask) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->item_shadow_mask = p_mask;
}

void RendererCanvasCull::canvas_light_set_directional_distance(RID p_light, float p_distance) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->directional_distance = p_distance;
}

void RendererCanvasCull::canvas_light_set_blend_mode(RID p_light, RS::CanvasLightBlendMode p_mode) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->blend_mode = p_mode;
}

void RendererCanvasCull::canvas_light_set_shadow_enabled(RID p_light, bool p_enabled) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	if (clight->use_shadow == p_enabled) {
		return;
	}
	clight->use_shadow = p_enabled;
	clight->version++;
	RSG::canvas_render->light_set_use_shadow(clight->light_internal, clight->use_shadow);
}

void RendererCanvasCull::canvas_light_set_shadow_filter(RID p_light, RS::CanvasLightShadowFilter p_filter) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->shadow_filter = p_filter;
}

void RendererCanvasCull::canvas_light_set_shadow_color(RID p_light, const Color &p_color) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);

	clight->shadow_color = p_color;
}

void RendererCanvasCull::canvas_light_set_shadow_smooth(RID p_light, float p_smooth) {
	RendererCanvasRender::Light *clight = canvas_light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(clight);
	clight->shadow_smooth = p_smooth;
}

RID RendererCanvasCull::canvas_light_occluder_allocate() {
	return canvas_light_occluder_owner.allocate_rid();
}
void RendererCanvasCull::canvas_light_occluder_initialize(RID p_rid) {
	return canvas_light_occluder_owner.initialize_rid(p_rid);
}

void RendererCanvasCull::canvas_light_occluder_attach_to_canvas(RID p_occluder, RID p_canvas) {
	RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_occluder);
	ERR_FAIL_NULL(occluder);

	if (occluder->canvas.is_valid()) {
		Canvas *canvas = canvas_owner.get_or_null(occluder->canvas);
		canvas->occluders.erase(occluder);
	}

	if (!canvas_owner.owns(p_canvas)) {
		p_canvas = RID();
	}

	occluder->canvas = p_canvas;

	if (occluder->canvas.is_valid()) {
		Canvas *canvas = canvas_owner.get_or_null(occluder->canvas);
		canvas->occluders.insert(occluder);
	}
}

void RendererCanvasCull::canvas_light_occluder_set_enabled(RID p_occluder, bool p_enabled) {
	RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_occluder);
	ERR_FAIL_NULL(occluder);

	occluder->enabled = p_enabled;
}

void RendererCanvasCull::canvas_light_occluder_set_polygon(RID p_occluder, RID p_polygon) {
	RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_occluder);
	ERR_FAIL_NULL(occluder);

	if (occluder->polygon.is_valid()) {
		LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(occluder->polygon);
		if (occluder_poly) {
			occluder_poly->owners.erase(occluder);
		}
	}

	occluder->polygon = p_polygon;
	occluder->occluder = RID();

	if (occluder->polygon.is_valid()) {
		LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(p_polygon);
		if (!occluder_poly) {
			occluder->polygon = RID();
			ERR_FAIL_NULL(occluder_poly);
		} else {
			occluder_poly->owners.insert(occluder);
			occluder->occluder = occluder_poly->occluder;
			occluder->aabb_cache = occluder_poly->aabb;
			occluder->cull_cache = occluder_poly->cull_mode;
		}
	}
}

void RendererCanvasCull::canvas_light_occluder_set_as_sdf_collision(RID p_occluder, bool p_enable) {
	RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_occluder);
	ERR_FAIL_NULL(occluder);
}

void RendererCanvasCull::canvas_light_occluder_set_transform(RID p_occluder, const Transform2D &p_xform) {
	RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_occluder);
	ERR_FAIL_NULL(occluder);

	occluder->xform = p_xform;
}

void RendererCanvasCull::canvas_light_occluder_set_light_mask(RID p_occluder, int p_mask) {
	RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_occluder);
	ERR_FAIL_NULL(occluder);

	occluder->light_mask = p_mask;
}

RID RendererCanvasCull::canvas_occluder_polygon_allocate() {
	return canvas_light_occluder_polygon_owner.allocate_rid();
}
void RendererCanvasCull::canvas_occluder_polygon_initialize(RID p_rid) {
	canvas_light_occluder_polygon_owner.initialize_rid(p_rid);
	LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(p_rid);
	occluder_poly->occluder = RSG::canvas_render->occluder_polygon_create();
}

void RendererCanvasCull::canvas_occluder_polygon_set_shape(RID p_occluder_polygon, const Vector<Vector2> &p_shape, bool p_closed) {
	LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(p_occluder_polygon);
	ERR_FAIL_NULL(occluder_poly);

	uint32_t pc = p_shape.size();
	ERR_FAIL_COND(pc < 2);

	occluder_poly->aabb = Rect2();
	const Vector2 *r = p_shape.ptr();
	for (uint32_t i = 0; i < pc; i++) {
		if (i == 0) {
			occluder_poly->aabb.position = r[i];
		} else {
			occluder_poly->aabb.expand_to(r[i]);
		}
	}

	RSG::canvas_render->occluder_polygon_set_shape(occluder_poly->occluder, p_shape, p_closed);

	for (RendererCanvasRender::LightOccluderInstance *E : occluder_poly->owners) {
		E->aabb_cache = occluder_poly->aabb;
	}
}

void RendererCanvasCull::canvas_occluder_polygon_set_cull_mode(RID p_occluder_polygon, RS::CanvasOccluderPolygonCullMode p_mode) {
	LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(p_occluder_polygon);
	ERR_FAIL_NULL(occluder_poly);
	occluder_poly->cull_mode = p_mode;
	RSG::canvas_render->occluder_polygon_set_cull_mode(occluder_poly->occluder, p_mode);
	for (RendererCanvasRender::LightOccluderInstance *E : occluder_poly->owners) {
		E->cull_cache = p_mode;
	}
}

void RendererCanvasCull::canvas_set_shadow_texture_size(int p_size) {
	RSG::canvas_render->set_shadow_texture_size(p_size);
}

RID RendererCanvasCull::canvas_texture_allocate() {
	return RSG::texture_storage->canvas_texture_allocate();
}
void RendererCanvasCull::canvas_texture_initialize(RID p_rid) {
	RSG::texture_storage->canvas_texture_initialize(p_rid);
}

void RendererCanvasCull::canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) {
	RSG::texture_storage->canvas_texture_set_channel(p_canvas_texture, p_channel, p_texture);
}

void RendererCanvasCull::canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_base_color, float p_shininess) {
	RSG::texture_storage->canvas_texture_set_shading_parameters(p_canvas_texture, p_base_color, p_shininess);
}

void RendererCanvasCull::canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter) {
	RSG::texture_storage->canvas_texture_set_texture_filter(p_canvas_texture, p_filter);
}

void RendererCanvasCull::canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat) {
	RSG::texture_storage->canvas_texture_set_texture_repeat(p_canvas_texture, p_repeat);
}

void RendererCanvasCull::canvas_item_set_default_texture_filter(RID p_item, RS::CanvasItemTextureFilter p_filter) {
	Item *ci = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(ci);
	ci->texture_filter = p_filter;
}
void RendererCanvasCull::canvas_item_set_default_texture_repeat(RID p_item, RS::CanvasItemTextureRepeat p_repeat) {
	Item *ci = canvas_item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(ci);
	ci->texture_repeat = p_repeat;
}

void RendererCanvasCull::update_visibility_notifiers() {
	SelfList<Item::VisibilityNotifierData> *E = visibility_notifier_list.first();
	while (E) {
		SelfList<Item::VisibilityNotifierData> *N = E->next();

		Item::VisibilityNotifierData *visibility_notifier = E->self();
		if (visibility_notifier->just_visible) {
			visibility_notifier->just_visible = false;

			if (!visibility_notifier->enter_callable.is_null()) {
				if (RSG::threaded) {
					visibility_notifier->enter_callable.call_deferred();
				} else {
					visibility_notifier->enter_callable.call();
				}
			}
		} else {
			if (visibility_notifier->visible_in_frame != RSG::rasterizer->get_frame_number()) {
				visibility_notifier_list.remove(E);

				if (!visibility_notifier->exit_callable.is_null()) {
					if (RSG::threaded) {
						visibility_notifier->exit_callable.call_deferred();
					} else {
						visibility_notifier->exit_callable.call();
					}
				}
			}
		}

		E = N;
	}
}

bool RendererCanvasCull::free(RID p_rid) {
	if (canvas_owner.owns(p_rid)) {
		Canvas *canvas = canvas_owner.get_or_null(p_rid);
		ERR_FAIL_NULL_V(canvas, false);

		while (canvas->viewports.size()) {
			RendererViewport::Viewport *vp = RSG::viewport->viewport_owner.get_or_null(*canvas->viewports.begin());
			ERR_FAIL_NULL_V(vp, true);

			HashMap<RID, RendererViewport::Viewport::CanvasData>::Iterator E = vp->canvas_map.find(p_rid);
			ERR_FAIL_COND_V(!E, true);
			vp->canvas_map.erase(p_rid);

			canvas->viewports.erase(*canvas->viewports.begin());
		}

		for (int i = 0; i < canvas->child_items.size(); i++) {
			canvas->child_items[i].item->parent = RID();
		}

		for (RendererCanvasRender::Light *E : canvas->lights) {
			E->canvas = RID();
		}

		for (RendererCanvasRender::LightOccluderInstance *E : canvas->occluders) {
			E->canvas = RID();
		}

		canvas_owner.free(p_rid);

	} else if (canvas_item_owner.owns(p_rid)) {
		Item *canvas_item = canvas_item_owner.get_or_null(p_rid);
		ERR_FAIL_NULL_V(canvas_item, true);

		if (canvas_item->parent.is_valid()) {
			if (canvas_owner.owns(canvas_item->parent)) {
				Canvas *canvas = canvas_owner.get_or_null(canvas_item->parent);
				canvas->erase_item(canvas_item);
			} else if (canvas_item_owner.owns(canvas_item->parent)) {
				Item *item_owner = canvas_item_owner.get_or_null(canvas_item->parent);
				item_owner->child_items.erase(canvas_item);

				if (item_owner->sort_y) {
					_mark_ysort_dirty(item_owner, canvas_item_owner);
				}
			}
		}

		for (int i = 0; i < canvas_item->child_items.size(); i++) {
			canvas_item->child_items[i]->parent = RID();
		}

		if (canvas_item->visibility_notifier != nullptr) {
			visibility_notifier_allocator.free(canvas_item->visibility_notifier);
		}

		/*
		if (canvas_item->material) {
			canvas_item->material->owners.erase(canvas_item);
		}
		*/

		if (canvas_item->canvas_group != nullptr) {
			memdelete(canvas_item->canvas_group);
			canvas_item->canvas_group = nullptr;
		}

		canvas_item_owner.free(p_rid);

	} else if (canvas_light_owner.owns(p_rid)) {
		RendererCanvasRender::Light *canvas_light = canvas_light_owner.get_or_null(p_rid);
		ERR_FAIL_NULL_V(canvas_light, true);

		if (canvas_light->canvas.is_valid()) {
			Canvas *canvas = canvas_owner.get_or_null(canvas_light->canvas);
			if (canvas) {
				canvas->lights.erase(canvas_light);
			}
		}

		RSG::canvas_render->free(canvas_light->light_internal);

		canvas_light_owner.free(p_rid);

	} else if (canvas_light_occluder_owner.owns(p_rid)) {
		RendererCanvasRender::LightOccluderInstance *occluder = canvas_light_occluder_owner.get_or_null(p_rid);
		ERR_FAIL_NULL_V(occluder, true);

		if (occluder->polygon.is_valid()) {
			LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(occluder->polygon);
			if (occluder_poly) {
				occluder_poly->owners.erase(occluder);
			}
		}

		if (occluder->canvas.is_valid() && canvas_owner.owns(occluder->canvas)) {
			Canvas *canvas = canvas_owner.get_or_null(occluder->canvas);
			canvas->occluders.erase(occluder);
		}

		canvas_light_occluder_owner.free(p_rid);

	} else if (canvas_light_occluder_polygon_owner.owns(p_rid)) {
		LightOccluderPolygon *occluder_poly = canvas_light_occluder_polygon_owner.get_or_null(p_rid);
		ERR_FAIL_NULL_V(occluder_poly, true);
		RSG::canvas_render->free(occluder_poly->occluder);

		while (occluder_poly->owners.size()) {
			(*occluder_poly->owners.begin())->polygon = RID();
			occluder_poly->owners.remove(occluder_poly->owners.begin());
		}

		canvas_light_occluder_polygon_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

RendererCanvasCull::RendererCanvasCull() {
	z_list = (RendererCanvasRender::Item **)memalloc(z_range * sizeof(RendererCanvasRender::Item *));
	z_last_list = (RendererCanvasRender::Item **)memalloc(z_range * sizeof(RendererCanvasRender::Item *));

	disable_scale = false;

	debug_redraw_time = GLOBAL_DEF("debug/canvas_items/debug_redraw_time", 1.0);
	debug_redraw_color = GLOBAL_DEF("debug/canvas_items/debug_redraw_color", Color(1.0, 0.2, 0.2, 0.5));
}

RendererCanvasCull::~RendererCanvasCull() {
	memfree(z_list);
	memfree(z_last_list);
}
