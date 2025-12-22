/**************************************************************************/
/*  container_editor_plugin.cpp                                           */
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

#include "container_editor_plugin.h"

#include "core/math/math_funcs.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/box_container.h"
#include "scene/gui/center_container.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/scroll_container.h"

ContainerEditorPlugin::ContainerType ContainerEditorPlugin::_get_container_type(Object *p_object) const {
	if (Object::cast_to<GridContainer>(p_object)) {
		return CONTAINER_TYPE_GRID;
	} else if (Object::cast_to<AspectRatioContainer>(p_object)) {
		return CONTAINER_TYPE_ASPECT_RATIO;
	} else if (Object::cast_to<CenterContainer>(p_object)) {
		return CONTAINER_TYPE_CENTER;
	} else if (Object::cast_to<ScrollContainer>(p_object)) {
		return CONTAINER_TYPE_SCROLL;
	} else if (Object::cast_to<MarginContainer>(p_object)) {
		return CONTAINER_TYPE_MARGIN;
	} else if (Object::cast_to<SplitContainer>(p_object)) {
		return CONTAINER_TYPE_SPLIT;
	} else if (Object::cast_to<FlowContainer>(p_object)) {
		return CONTAINER_TYPE_FLOW;
	} else if (Object::cast_to<BoxContainer>(p_object)) {
		return CONTAINER_TYPE_BOX;
	}
	return CONTAINER_TYPE_NONE;
}

void ContainerEditorPlugin::edit(Object *p_object) {
	if (container) {
		container->disconnect(SNAME("draw"), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
		container->disconnect(SNAME("sort_children"), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
		container->disconnect(SceneStringName(minimum_size_changed), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	}

	container = Object::cast_to<Container>(p_object);
	type = _get_container_type(p_object);

	if (container) {
		container->connect(SNAME("draw"), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
		container->connect(SNAME("sort_children"), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
		// `minimum_size_changed` is required for BoxContainer because it indirectly resorts its children instead of calling queue_resort when vertical is changed.
		container->connect(SceneStringName(minimum_size_changed), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	}
	CanvasItemEditor::get_singleton()->update_viewport();
}

bool ContainerEditorPlugin::handles(Object *p_object) const {
	return _get_container_type(p_object) != CONTAINER_TYPE_NONE;
}

void ContainerEditorPlugin::forward_canvas_draw_over_viewport(Control *p_viewport_control) {
	if (!container || !container->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * container->get_screen_transform();

	// NOTE: This color is copied from Camera2DEditor::forward_canvas_draw_over_viewport.
	// We may want to unify them somehow in the future.
	Color border_color = Color(1, 1, 0.25, 0.63);

	int border_width = Math::round(1 * EDSCALE);

	Rect2 rect = container->_edit_get_rect();

	switch (type) {
		case CONTAINER_TYPE_NONE:
			break;
		case CONTAINER_TYPE_ASPECT_RATIO: {
			AspectRatioContainer *aspect_ratio_container = Object::cast_to<AspectRatioContainer>(container);
			Size2 child_size = Size2(aspect_ratio_container->get_ratio(), 1.0);
			float scale_factor = 1.0;

			switch (aspect_ratio_container->get_stretch_mode()) {
				case AspectRatioContainer::STRETCH_WIDTH_CONTROLS_HEIGHT: {
					scale_factor = rect.size.x / child_size.x;
				} break;
				case AspectRatioContainer::STRETCH_HEIGHT_CONTROLS_WIDTH: {
					scale_factor = rect.size.y / child_size.y;
				} break;
				case AspectRatioContainer::STRETCH_FIT: {
					scale_factor = MIN(rect.size.x / child_size.x, rect.size.y / child_size.y);
				} break;
				case AspectRatioContainer::STRETCH_COVER: {
					scale_factor = MAX(rect.size.x / child_size.x, rect.size.y / child_size.y);
				} break;
			}
			child_size *= scale_factor;

			float align_x = 0.5;
			switch (aspect_ratio_container->get_alignment_horizontal()) {
				case AspectRatioContainer::ALIGNMENT_BEGIN: {
					align_x = 0.0;
				} break;
				case AspectRatioContainer::ALIGNMENT_CENTER: {
					align_x = 0.5;
				} break;
				case AspectRatioContainer::ALIGNMENT_END: {
					align_x = 1.0;
				} break;
			}
			float align_y = 0.5;
			switch (aspect_ratio_container->get_alignment_vertical()) {
				case AspectRatioContainer::ALIGNMENT_BEGIN: {
					align_y = 0.0;
				} break;
				case AspectRatioContainer::ALIGNMENT_CENTER: {
					align_y = 0.5;
				} break;
				case AspectRatioContainer::ALIGNMENT_END: {
					align_y = 1.0;
				} break;
			}
			Vector2 offset = (rect.size - child_size) * Vector2(align_x, align_y);
			Rect2 inner_rect = aspect_ratio_container->is_layout_rtl() ? Rect2(Vector2(rect.size.x - offset.x - child_size.x, offset.y), child_size) : Rect2(offset, child_size);

			p_viewport_control->draw_rect(xform.xform(inner_rect), border_color, false, border_width);
		} break;
		case CONTAINER_TYPE_GRID: {
			GridContainer *grid_container = Object::cast_to<GridContainer>(container);
			for (int cell_index = 0; cell_index < grid_container->cell_sizes.size(); cell_index++) {
				int32_t cell_offset = grid_container->cell_sizes[cell_index];
				bool draw_row_lines = cell_index >= grid_container->get_columns() - 1;
				Vector2 p1, p2;
				if (draw_row_lines) {
					p1 = rect.position + Vector2(0.0, cell_offset);
					p2 = rect.position + Vector2(rect.size.x, cell_offset);
				} else {
					p1 = rect.position + Vector2(cell_offset, 0.0);
					p2 = rect.position + Vector2(cell_offset, rect.size.y);
				}
				p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
			}
		} break;
		case CONTAINER_TYPE_CENTER: {
			Vector2 p1, p2;
			Vector2 center = rect.get_center();
			double cross_width = border_width * 5.0;
			p1 = center + Vector2(-cross_width, 0.0);
			p2 = center + Vector2(+cross_width, 0.0);
			p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);

			p1 = center + Vector2(0.0, -cross_width);
			p2 = center + Vector2(0.0, +cross_width);
			p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
		} break;
		case CONTAINER_TYPE_SCROLL: {
		} break;
		case CONTAINER_TYPE_MARGIN: {
			MarginContainer *margin_container = Object::cast_to<MarginContainer>(container);
			int margin_left = margin_container->get_margin_size(SIDE_LEFT);
			int margin_top = margin_container->get_margin_size(SIDE_TOP);
			int margin_right = margin_container->get_margin_size(SIDE_RIGHT);
			int margin_bottom = margin_container->get_margin_size(SIDE_BOTTOM);

			Vector2 p1, p2;

			// Calculate left margin line.
			p1 = rect.position + Vector2(margin_left, margin_top);
			p2 = rect.position + Vector2(margin_left, rect.size.y - margin_bottom);
			p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);

			// Calculate top margin line.
			p1 = rect.position + Vector2(margin_left, margin_top);
			p2 = rect.position + Vector2(rect.size.x - margin_right, margin_top);
			p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);

			// Calculate right margin line.
			p1 = rect.position + Vector2(rect.size.x - margin_right, margin_top);
			p2 = rect.position + Vector2(rect.size.x - margin_right, rect.size.y - margin_bottom);
			p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);

			// Calculate bottom margin line.
			p1 = rect.position + Vector2(margin_left, rect.size.y - margin_bottom);
			p2 = rect.position + Vector2(rect.size.x - margin_right, rect.size.y - margin_bottom);
			p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
		} break;
		case CONTAINER_TYPE_SPLIT: {
			SplitContainer *split_container = Object::cast_to<SplitContainer>(container);
			Vector2 p1, p2;
			double half_dragger_width = split_container->_get_separation() / 2.0;
			for (int dragger_offset : split_container->dragger_positions) {
				double centered_dragger_offset = dragger_offset + half_dragger_width;
				if (split_container->is_vertical()) {
					p1 = rect.position + Vector2(0.0, centered_dragger_offset);
					p2 = rect.position + Vector2(rect.size.x, centered_dragger_offset);
				} else {
					p1 = rect.position + Vector2(centered_dragger_offset, 0.0);
					p2 = rect.position + Vector2(centered_dragger_offset, rect.size.y);
				}
				p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
			}
		} break;
		case CONTAINER_TYPE_FLOW: {
			FlowContainer *flow_container = Object::cast_to<FlowContainer>(container);
			Vector2 p1, p2;
			// Separation along the axis elements overflow across.
			double cross_half_separation = (flow_container->is_vertical() ? flow_container->theme_cache.h_separation : flow_container->theme_cache.v_separation) / 2.0;
			// Separation across the axis elements flow across.
			double half_separation = (flow_container->is_vertical() ? flow_container->theme_cache.v_separation : flow_container->theme_cache.h_separation) / 2.0;
			double last_cross_pos = 0.0;
			for (int cross_index = 0; cross_index < flow_container->cell_sizes.size(); cross_index++) {
				bool is_last_cross_row = cross_index >= flow_container->cell_sizes.size() - 1;
				Vector<int32_t> cross_cell_sizes = flow_container->cell_sizes[cross_index];
				// Subtract the separation because overflow cell size is aligned to the top of the i-th row.
				// `cross_y` is the last item because `cell_sizes` is laid out like [...flow main axis, overflow cross axis].
				// Separation must be doubled to align it to the end of the previous item on the last row.
				double cross_pos = cross_cell_sizes.get(cross_cell_sizes.size() - 1) - cross_half_separation * (is_last_cross_row ? 2.0 : 1.0);
				for (int main_index = 0; main_index < cross_cell_sizes.size() - 2; main_index++) {
					double cell_pos = cross_cell_sizes[main_index] - half_separation;
					if (flow_container->is_vertical()) {
						p1 = rect.position + Vector2(last_cross_pos, cell_pos);
						p2 = rect.position + Vector2(cross_pos, cell_pos);
					} else {
						p1 = rect.position + Vector2(cell_pos, last_cross_pos);
						p2 = rect.position + Vector2(cell_pos, cross_pos);
					}
					p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
				}
				// Omit the last cross separator, all it does is add visual noise.
				if (!is_last_cross_row) {
					if (flow_container->is_vertical()) {
						p1 = rect.position + Vector2(cross_pos, 0.0);
						p2 = rect.position + Vector2(cross_pos, rect.size.y);
					} else {
						p1 = rect.position + Vector2(0.0, cross_pos);
						p2 = rect.position + Vector2(rect.size.x, cross_pos);
					}
					p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
					last_cross_pos = cross_pos;
				}
			}
		} break;
		case CONTAINER_TYPE_BOX: {
			BoxContainer *box_container = Object::cast_to<BoxContainer>(container);
			double half_separation = box_container->theme_cache.separation / 2.0;
			// Skip the first position index because it'll just be 0.
			for (uint32_t position_index = 1; position_index < box_container->cell_positions.size(); position_index++) {
				Vector2 p1, p2;
				int size = box_container->cell_positions[position_index];
				double cell_pos = size - half_separation;
				if (box_container->is_vertical()) {
					p1 = rect.position + Vector2(0.0, cell_pos);
					p2 = rect.position + Vector2(rect.size.x, cell_pos);
				} else {
					p1 = rect.position + Vector2(cell_pos, 0.0);
					p2 = rect.position + Vector2(cell_pos, rect.size.y);
				}
				p_viewport_control->draw_line(xform.xform(p1), xform.xform(p2), border_color, border_width);
			}
		} break;
	}
}
