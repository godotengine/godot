/*************************************************************************/
/*  canvas_item_arrange_plugin.h                                         */
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

#ifndef CANVASITEM_ARRANGE_PLUGIN_H
#define CANVASITEM_ARRANGE_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"

class ArrangePanel : public PanelContainer {
	GDCLASS(ArrangePanel, PanelContainer);

	enum Action {
		ALIGN_LEFT,
		ALIGN_RIGHT,
		ALIGN_TOP,
		ALIGN_BOTTOM,
		ALIGN_CENTER_HORIZONTAL,
		ALIGN_CENTER_VERTICAL,
		DISTRIBUTE_LEFT,
		DISTRIBUTE_RIGHT,
		DISTRIBUTE_TOP,
		DISTRIBUTE_BOTTOM,
		DISTRIBUTE_CENTER_HORIZONTAL,
		DISTRIBUTE_CENTER_VERTICAL,
		DISTRIBUTE_GAP_HORIZONTAL,
		DISTRIBUTE_GAP_VERTICAL
	};

	struct SortLeft {
		bool operator()(const CanvasItem *p_a, const CanvasItem *p_b) const {
			Vector2 pos_a = p_a->get_global_transform().xform(p_a->_edit_get_rect().position);
			Vector2 pos_b = p_b->get_global_transform().xform(p_b->_edit_get_rect().position);
			return pos_a.x < pos_b.x;
		}
	};

	struct SortRight {
		bool operator()(const CanvasItem *p_a, const CanvasItem *p_b) const {
			Vector2 pos_a = p_a->get_global_transform().xform(p_a->_edit_get_rect().position + p_a->_edit_get_rect().size);
			Vector2 pos_b = p_b->get_global_transform().xform(p_b->_edit_get_rect().position + p_b->_edit_get_rect().size);
			return pos_a.x < pos_b.x;
		}
	};

	struct SortTop {
		bool operator()(const CanvasItem *p_a, const CanvasItem *p_b) const {
			Vector2 pos_a = p_a->get_global_transform().xform(p_a->_edit_get_rect().position);
			Vector2 pos_b = p_b->get_global_transform().xform(p_b->_edit_get_rect().position);
			return pos_a.y < pos_b.y;
		}
	};

	struct SortBottom {
		bool operator()(const CanvasItem *p_a, const CanvasItem *p_b) const {
			Vector2 pos_a = p_a->get_global_transform().xform(p_a->_edit_get_rect().position + p_a->_edit_get_rect().size);
			Vector2 pos_b = p_b->get_global_transform().xform(p_b->_edit_get_rect().position + p_b->_edit_get_rect().size);
			return pos_a.y < pos_b.y;
		}
	};

	UndoRedo *undo_redo;

	Button *btn_align_left;
	Button *btn_align_right;
	Button *btn_align_center_horizontal;
	Button *btn_align_top;
	Button *btn_align_bottom;
	Button *btn_align_center_vertical;
	Button *btn_dist_left;
	Button *btn_dist_right;
	Button *btn_dist_top;
	Button *btn_dist_center_vertical;
	Button *btn_dist_bottom;
	Button *btn_dist_center_horizon;
	Button *btn_dist_gap_horizontal;
	Button *btn_dist_gap_vertical;

	void arrange_nodes(int p_action);
	Button *create_button(String p_tooltip, Action p_action);
	void _gui_input(const Ref<InputEvent> &p_event);

protected:
	virtual bool has_point(const Point2 &p_point) const;
	static void _bind_methods();
	void _notification(int p_what);

public:
	List<CanvasItem *> get_selected_nodes();
	ArrangePanel();
};

#endif // CANVASITEM_ARRANGE_PLUGIN_H
