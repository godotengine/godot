/*************************************************************************/
/*  canvas_item_arrange_plugin.cpp                                       */
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

#include "editor/plugins/canvas_item_arrange_panel.h"

List<CanvasItem *> ArrangePanel::get_selected_nodes() {
	List<CanvasItem *> selected;
	List<Node *> selection = EditorNode::get_singleton()->get_editor_selection()->get_full_selected_node_list();
	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
		CanvasItem *node = Object::cast_to<CanvasItem>(E->get());
		if (node && !node->has_meta("_edit_lock_")) {
			selected.push_back(node);
		}
	}
	return selected;
}

void ArrangePanel::arrange_nodes(int p_action) {
	List<CanvasItem *> selected = get_selected_nodes();

	if (selected.size() < 2) {
		return;
	}

	undo_redo->create_action(TTR("Arrange CanvasItem nodes."), UndoRedo::MERGE_DISABLE);
	switch (p_action) {
		case ALIGN_LEFT: {
			selected.sort_custom<SortLeft>();

			CanvasItem *target_node = selected.front()->get();
			Rect2 rect_node = target_node->_edit_get_rect();
			Point2 pos_local = rect_node.position;
			Point2 pos_global = target_node->get_global_transform().xform(pos_local);
			real_t target_x = pos_global.x;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.x = target_x;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);
			}
		} break;
		case ALIGN_RIGHT: {
			selected.sort_custom<SortRight>();

			CanvasItem *target_node = selected.back()->get();
			Rect2 rect_node = target_node->_edit_get_rect();
			Point2 pos_local = rect_node.position + rect_node.size;
			Point2 pos_global = target_node->get_global_transform().xform(pos_local);
			real_t target_x = pos_global.x;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.x = target_x;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.size.x - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x - rect.size.x;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);
			}
		} break;
		case ALIGN_CENTER_VERTICAL: {
			selected.sort_custom<SortLeft>();

			CanvasItem *left_node = selected.front()->get();
			Rect2 rect_node = left_node->_edit_get_rect();
			Point2 pos_local = left_node->_edit_get_rect().position;
			Point2 pos_global = left_node->get_global_transform().xform(pos_local);
			real_t target_x = pos_global.x;

			selected.sort_custom<SortRight>();

			CanvasItem *right_node = selected.back()->get();
			rect_node = right_node->_edit_get_rect();
			pos_local = right_node->_edit_get_rect().position + right_node->_edit_get_rect().size;
			pos_global = right_node->get_global_transform().xform(pos_local);
			target_x = (target_x + pos_global.x) / 2;

			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.x = target_x;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.size.x / 2 - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x - rect.size.x / 2;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);
			}
		} break;
		case DISTRIBUTE_LEFT: {
			selected.sort_custom<SortLeft>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_x = first_global.x;

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_x = last_global.x;

			real_t gap_x = (last_x - first_x) / (selected.size() - 1);
			int index = 0;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.x = first_x + gap_x * index;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				index++;
			}
		} break;
		case DISTRIBUTE_RIGHT: {
			selected.sort_custom<SortRight>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position + first_rect.size;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_x = first_global.x;

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position + last_rect.size;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_x = last_global.x;

			real_t gap_x = (last_x - first_x) / (selected.size() - 1);
			int index = 0;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.x = first_x + gap_x * index;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.size.x - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x - rect.size.x;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				index++;
			}
		} break;
		case DISTRIBUTE_CENTER_HORIZONTAL: {
			selected.sort_custom<SortLeft>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position;
			first_local.x += first_rect.size.x / 2;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_x = first_global.x;

			selected.sort_custom<SortRight>();

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position;
			last_local.x += last_rect.size.x / 2;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_x = last_global.x;

			real_t gap_x = (last_x - first_x) / (selected.size() - 1);
			int index = 0;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.x = first_x + gap_x * index;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.size.x / 2 - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x - rect.size.x / 2;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				index++;
			}
		} break;
		case DISTRIBUTE_GAP_HORIZONTAL: {
			selected.sort_custom<SortLeft>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_x = first_global.x;

			selected.sort_custom<SortRight>();

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position + last_rect.size;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_x = last_global.x;

			real_t max_x = last_x - first_x;
			real_t total_x = 0;

			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 size = node->get_global_transform().xform(node->_edit_get_rect().position + node->_edit_get_rect().size) - node->get_global_transform().xform(node->_edit_get_rect().position);
				total_x += size.x;
			}

			real_t gap_x = (max_x - total_x) / (selected.size() - 1);
			first_node = selected.front()->get();
			Vector2 to = first_node->get_global_transform().xform(first_node->_edit_get_rect().position + first_node->_edit_get_rect().size);

			for (List<CanvasItem *>::Element *E = selected.front()->next(); E; E = E->next()) {
				CanvasItem *node = E->get();

				to.x += gap_x;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.x -= to.x - prev.position.x;
				Rect2 rect = node->_edit_get_rect();
				rect.position.x = to.x;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				to = node->get_global_transform().xform(to + node->_edit_get_rect().size);
			}
		} break;
		case ALIGN_TOP: {
			selected.sort_custom<SortTop>();

			CanvasItem *target_node = selected.front()->get();
			Rect2 rect_node = target_node->_edit_get_rect();
			Point2 pos_local = rect_node.position;
			Point2 pos_global = target_node->get_global_transform().xform(pos_local);
			real_t target_y = pos_global.y;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.y = target_y;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);
			}
		} break;
		case ALIGN_BOTTOM: {
			selected.sort_custom<SortBottom>();

			CanvasItem *target_node = selected.back()->get();
			Rect2 rect_node = target_node->_edit_get_rect();
			Point2 pos_local = rect_node.position + rect_node.size;
			Point2 pos_global = target_node->get_global_transform().xform(pos_local);
			real_t target_y = pos_global.y;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.y = target_y;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.size.y - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y - rect.size.y;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);
			}
		} break;
		case ALIGN_CENTER_HORIZONTAL: {
			selected.sort_custom<SortTop>();

			CanvasItem *top_node = selected.front()->get();
			Rect2 rect_node = top_node->_edit_get_rect();
			Point2 pos_local = top_node->_edit_get_rect().position;
			Point2 pos_global = top_node->get_global_transform().xform(pos_local);
			real_t target_y = pos_global.y;

			selected.sort_custom<SortBottom>();

			CanvasItem *bottom_node = selected.back()->get();
			rect_node = bottom_node->_edit_get_rect();
			pos_local = bottom_node->_edit_get_rect().position + bottom_node->_edit_get_rect().size;
			pos_global = bottom_node->get_global_transform().xform(pos_local);
			target_y = (target_y + pos_global.y) / 2;

			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.y = target_y;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.size.y / 2 - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y - rect.size.y / 2;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);
			}
		} break;
		case DISTRIBUTE_TOP: {
			selected.sort_custom<SortTop>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_y = first_global.y;

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_y = last_global.y;

			real_t gap_y = (last_y - first_y) / (selected.size() - 1);
			int index = 0;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.y = first_y + gap_y * index;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				index++;
			}
		} break;
		case DISTRIBUTE_BOTTOM: {
			selected.sort_custom<SortBottom>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position + first_rect.size;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_y = first_global.y;

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position + last_rect.size;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_y = last_global.y;

			real_t gap_y = (last_y - first_y) / (selected.size() - 1);
			int index = 0;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.y = first_y + gap_y * index;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.size.y - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y - rect.size.y;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				index++;
			}
		} break;
		case DISTRIBUTE_CENTER_VERTICAL: {
			selected.sort_custom<SortTop>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position;
			first_local.y += first_rect.size.y / 2;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_y = first_global.y;

			selected.sort_custom<SortBottom>();

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position;
			last_local.y += last_rect.size.y / 2;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_y = last_global.y;

			real_t gap_y = (last_y - first_y) / (selected.size() - 1);
			int index = 0;
			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 to = node->get_global_transform().get_origin();
				to.y = first_y + gap_y * index;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.size.y / 2 - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y - rect.size.y / 2;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				index++;
			}
		} break;
		case DISTRIBUTE_GAP_VERTICAL: {
			selected.sort_custom<SortTop>();

			CanvasItem *first_node = selected.front()->get();
			Rect2 first_rect = first_node->_edit_get_rect();
			Point2 first_local = first_rect.position;
			Point2 first_global = first_node->get_global_transform().xform(first_local);
			real_t first_y = first_global.y;

			selected.sort_custom<SortBottom>();

			CanvasItem *last_node = selected.back()->get();
			Rect2 last_rect = last_node->_edit_get_rect();
			Point2 last_local = last_rect.position + last_rect.size;
			Point2 last_global = last_node->get_global_transform().xform(last_local);
			real_t last_y = last_global.y;

			real_t max_y = last_y - first_y;
			real_t total_y = 0;

			for (List<CanvasItem *>::Element *E = selected.front(); E; E = E->next()) {
				CanvasItem *node = E->get();
				Vector2 size = node->get_global_transform().xform(node->_edit_get_rect().position + node->_edit_get_rect().size) - node->get_global_transform().xform(node->_edit_get_rect().position);
				total_y += size.y;
			}

			real_t gap_y = (max_y - total_y) / (selected.size() - 1);
			first_node = selected.front()->get();
			Vector2 to = first_node->get_global_transform().xform(first_node->_edit_get_rect().position + first_node->_edit_get_rect().size);

			for (List<CanvasItem *>::Element *E = selected.front()->next(); E; E = E->next()) {
				CanvasItem *node = E->get();

				to.y += gap_y;
				to = node->make_canvas_position_local(to);

				Rect2 prev = node->_edit_get_rect();
				prev.position.y -= to.y - prev.position.y;
				Rect2 rect = node->_edit_get_rect();
				rect.position.y = to.y;

				undo_redo->add_do_method(node, "_edit_set_rect", rect);
				undo_redo->add_undo_method(node, "_edit_set_rect", prev);

				to = node->get_global_transform().xform(to + node->_edit_get_rect().size);
			}
		} break;
	}

	undo_redo->commit_action();
}

Button *ArrangePanel::create_button(String p_tooltip, Action p_action) {
	Vector2 button_size(32 * EDSCALE, 32 * EDSCALE);
	Button *btn = memnew(Button);
	btn->set_tooltip(TTR(p_tooltip));
	btn->connect("pressed", this, "arrange_nodes", varray(p_action));
	return btn;
}

void ArrangePanel::_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid() && b->is_pressed()) {
		if (!Rect2(Point2(), get_size()).has_point(b->get_position())) {
			queue_delete();
		}
	}
}

bool ArrangePanel::has_point(const Point2 &p_point) const {
	return true;
}

void ArrangePanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_gui_input"), &ArrangePanel::_gui_input);
	ClassDB::bind_method(D_METHOD("arrange_nodes"), &ArrangePanel::arrange_nodes);
}

void ArrangePanel::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		btn_align_left->set_icon(get_icon("ArrangeAlignLeft", "EditorIcons"));
		btn_align_center_vertical->set_icon(get_icon("ArrangeAlignCenterVertical", "EditorIcons"));
		btn_align_right->set_icon(get_icon("ArrangeAlignRight", "EditorIcons"));
		btn_align_top->set_icon(get_icon("ArrangeAlignTop", "EditorIcons"));
		btn_align_center_horizontal->set_icon(get_icon("ArrangeAlignCenterHorizontal", "EditorIcons"));
		btn_align_bottom->set_icon(get_icon("ArrangeAlignBottom", "EditorIcons"));

		btn_dist_left->set_icon(get_icon("ArrangeDistLeft", "EditorIcons"));
		btn_dist_center_horizon->set_icon(get_icon("ArrangeDistCenterHorizontal", "EditorIcons"));
		btn_dist_right->set_icon(get_icon("ArrangeDistRight", "EditorIcons"));
		btn_dist_gap_horizontal->set_icon(get_icon("ArrangeDistGapHorizontal", "EditorIcons"));
		btn_dist_top->set_icon(get_icon("ArrangeDistTop", "EditorIcons"));
		btn_dist_center_vertical->set_icon(get_icon("ArrangeDistCenterVertical", "EditorIcons"));
		btn_dist_bottom->set_icon(get_icon("ArrangeDistBottom", "EditorIcons"));
		btn_dist_gap_vertical->set_icon(get_icon("ArrangeDistGapVertical", "EditorIcons"));

	} else if (p_what == NOTIFICATION_DRAW) {
		draw_style_box(get_stylebox("panel", "PopupMenu"), Rect2(Vector2(), get_size()));
	}
}

ArrangePanel::ArrangePanel() {
	undo_redo = EditorNode::get_singleton()->get_undo_redo();

	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	GridContainer *grid_align = memnew(GridContainer);
	grid_align->set_columns(3);
	vbox->add_child(grid_align);

	btn_align_left = create_button("Align left", ALIGN_LEFT);
	grid_align->add_child(btn_align_left);

	btn_align_center_vertical = create_button("Align center vertically", ALIGN_CENTER_VERTICAL);
	grid_align->add_child(btn_align_center_vertical);

	btn_align_right = create_button("Align right", ALIGN_RIGHT);
	grid_align->add_child(btn_align_right);

	btn_align_top = create_button("Align top", ALIGN_TOP);
	grid_align->add_child(btn_align_top);

	btn_align_center_horizontal = create_button("Align center horizontally", ALIGN_CENTER_HORIZONTAL);
	grid_align->add_child(btn_align_center_horizontal);

	btn_align_bottom = create_button("Align bottom", ALIGN_BOTTOM);
	grid_align->add_child(btn_align_bottom);

	HSeparator *split = memnew(HSeparator);
	vbox->add_child(split);

	GridContainer *grid_distribute = memnew(GridContainer);
	grid_distribute->set_columns(4);
	vbox->add_child(grid_distribute);

	btn_dist_left = create_button("Distribute left edges", DISTRIBUTE_LEFT);
	grid_distribute->add_child(btn_dist_left);

	btn_dist_center_horizon = create_button("Distribute center horizontally", DISTRIBUTE_CENTER_HORIZONTAL);
	grid_distribute->add_child(btn_dist_center_horizon);

	btn_dist_right = create_button("Distribute right edges", DISTRIBUTE_RIGHT);
	grid_distribute->add_child(btn_dist_right);

	btn_dist_gap_horizontal = create_button("Distribute gap horizontally", DISTRIBUTE_GAP_HORIZONTAL);
	grid_distribute->add_child(btn_dist_gap_horizontal);

	btn_dist_top = create_button("Distribute top edges", DISTRIBUTE_TOP);
	grid_distribute->add_child(btn_dist_top);

	btn_dist_center_vertical = create_button("Distribute center vertically", DISTRIBUTE_CENTER_VERTICAL);
	grid_distribute->add_child(btn_dist_center_vertical);

	btn_dist_bottom = create_button("Distribute bottom edges", DISTRIBUTE_BOTTOM);
	grid_distribute->add_child(btn_dist_bottom);

	btn_dist_gap_vertical = create_button("Distribute gap vertically", DISTRIBUTE_GAP_VERTICAL);
	grid_distribute->add_child(btn_dist_gap_vertical);
}
