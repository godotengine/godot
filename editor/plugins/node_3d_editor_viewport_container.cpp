/*************************************************************************/
/*  node_3d_editor_viewport_container.cpp                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "node_3d_editor_viewport_container.h"

#include "node_3d_editor_viewport.h"

void Node3DEditorViewportContainer::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			Vector2 size = get_size();

			int h_sep = get_theme_constant(SNAME("separation"), SNAME("HSplitContainer"));
			int v_sep = get_theme_constant(SNAME("separation"), SNAME("VSplitContainer"));

			int mid_w = size.width * ratio_h;
			int mid_h = size.height * ratio_v;

			dragging_h = mb->get_position().x > (mid_w - h_sep / 2) && mb->get_position().x < (mid_w + h_sep / 2);
			dragging_v = mb->get_position().y > (mid_h - v_sep / 2) && mb->get_position().y < (mid_h + v_sep / 2);

			drag_begin_pos = mb->get_position();
			drag_begin_ratio.x = ratio_h;
			drag_begin_ratio.y = ratio_v;

			switch (view) {
				case VIEW_USE_1_VIEWPORT: {
					dragging_h = false;
					dragging_v = false;

				} break;
				case VIEW_USE_2_VIEWPORTS: {
					dragging_h = false;

				} break;
				case VIEW_USE_2_VIEWPORTS_ALT: {
					dragging_v = false;

				} break;
				case VIEW_USE_3_VIEWPORTS:
				case VIEW_USE_3_VIEWPORTS_ALT:
				case VIEW_USE_4_VIEWPORTS: {
					// Do nothing.

				} break;
			}
		} else {
			dragging_h = false;
			dragging_v = false;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (view == VIEW_USE_3_VIEWPORTS || view == VIEW_USE_3_VIEWPORTS_ALT || view == VIEW_USE_4_VIEWPORTS) {
			Vector2 size = get_size();

			int h_sep = get_theme_constant(SNAME("separation"), SNAME("HSplitContainer"));
			int v_sep = get_theme_constant(SNAME("separation"), SNAME("VSplitContainer"));

			int mid_w = size.width * ratio_h;
			int mid_h = size.height * ratio_v;

			bool was_hovering_h = hovering_h;
			bool was_hovering_v = hovering_v;
			hovering_h = mm->get_position().x > (mid_w - h_sep / 2) && mm->get_position().x < (mid_w + h_sep / 2);
			hovering_v = mm->get_position().y > (mid_h - v_sep / 2) && mm->get_position().y < (mid_h + v_sep / 2);

			if (was_hovering_h != hovering_h || was_hovering_v != hovering_v) {
				queue_redraw();
			}
		}

		if (dragging_h) {
			real_t new_ratio = drag_begin_ratio.x + (mm->get_position().x - drag_begin_pos.x) / get_size().width;
			new_ratio = CLAMP(new_ratio, 40 / get_size().width, (get_size().width - 40) / get_size().width);
			ratio_h = new_ratio;
			queue_sort();
			queue_redraw();
		}
		if (dragging_v) {
			real_t new_ratio = drag_begin_ratio.y + (mm->get_position().y - drag_begin_pos.y) / get_size().height;
			new_ratio = CLAMP(new_ratio, 40 / get_size().height, (get_size().height - 40) / get_size().height);
			ratio_v = new_ratio;
			queue_sort();
			queue_redraw();
		}
	}
}

void Node3DEditorViewportContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_MOUSE_ENTER:
		case NOTIFICATION_MOUSE_EXIT: {
			mouseover = (p_what == NOTIFICATION_MOUSE_ENTER);
			queue_redraw();
		} break;

		case NOTIFICATION_DRAW: {
			if (mouseover) {
				Ref<Texture2D> h_grabber = get_theme_icon(SNAME("grabber"), SNAME("HSplitContainer"));
				Ref<Texture2D> v_grabber = get_theme_icon(SNAME("grabber"), SNAME("VSplitContainer"));

				Ref<Texture2D> hdiag_grabber = get_theme_icon(SNAME("GuiViewportHdiagsplitter"), SNAME("EditorIcons"));
				Ref<Texture2D> vdiag_grabber = get_theme_icon(SNAME("GuiViewportVdiagsplitter"), SNAME("EditorIcons"));
				Ref<Texture2D> vh_grabber = get_theme_icon(SNAME("GuiViewportVhsplitter"), SNAME("EditorIcons"));

				Vector2 size = get_size();

				int h_sep = get_theme_constant(SNAME("separation"), SNAME("HSplitContainer"));

				int v_sep = get_theme_constant(SNAME("separation"), SNAME("VSplitContainer"));

				int mid_w = size.width * ratio_h;
				int mid_h = size.height * ratio_v;

				int size_left = mid_w - h_sep / 2;
				int size_bottom = size.height - mid_h - v_sep / 2;

				switch (view) {
					case VIEW_USE_1_VIEWPORT: {
						// Nothing to show.

					} break;
					case VIEW_USE_2_VIEWPORTS: {
						draw_texture(v_grabber, Vector2((size.width - v_grabber->get_width()) / 2, mid_h - v_grabber->get_height() / 2));
						set_default_cursor_shape(CURSOR_VSPLIT);

					} break;
					case VIEW_USE_2_VIEWPORTS_ALT: {
						draw_texture(h_grabber, Vector2(mid_w - h_grabber->get_width() / 2, (size.height - h_grabber->get_height()) / 2));
						set_default_cursor_shape(CURSOR_HSPLIT);

					} break;
					case VIEW_USE_3_VIEWPORTS: {
						if ((hovering_v && hovering_h && !dragging_v && !dragging_h) || (dragging_v && dragging_h)) {
							draw_texture(hdiag_grabber, Vector2(mid_w - hdiag_grabber->get_width() / 2, mid_h - v_grabber->get_height() / 4));
							set_default_cursor_shape(CURSOR_DRAG);
						} else if ((hovering_v && !dragging_h) || dragging_v) {
							draw_texture(v_grabber, Vector2((size.width - v_grabber->get_width()) / 2, mid_h - v_grabber->get_height() / 2));
							set_default_cursor_shape(CURSOR_VSPLIT);
						} else if (hovering_h || dragging_h) {
							draw_texture(h_grabber, Vector2(mid_w - h_grabber->get_width() / 2, mid_h + v_grabber->get_height() / 2 + (size_bottom - h_grabber->get_height()) / 2));
							set_default_cursor_shape(CURSOR_HSPLIT);
						}

					} break;
					case VIEW_USE_3_VIEWPORTS_ALT: {
						if ((hovering_v && hovering_h && !dragging_v && !dragging_h) || (dragging_v && dragging_h)) {
							draw_texture(vdiag_grabber, Vector2(mid_w - vdiag_grabber->get_width() + v_grabber->get_height() / 4, mid_h - vdiag_grabber->get_height() / 2));
							set_default_cursor_shape(CURSOR_DRAG);
						} else if ((hovering_v && !dragging_h) || dragging_v) {
							draw_texture(v_grabber, Vector2((size_left - v_grabber->get_width()) / 2, mid_h - v_grabber->get_height() / 2));
							set_default_cursor_shape(CURSOR_VSPLIT);
						} else if (hovering_h || dragging_h) {
							draw_texture(h_grabber, Vector2(mid_w - h_grabber->get_width() / 2, (size.height - h_grabber->get_height()) / 2));
							set_default_cursor_shape(CURSOR_HSPLIT);
						}

					} break;
					case VIEW_USE_4_VIEWPORTS: {
						Vector2 half(mid_w, mid_h);
						if ((hovering_v && hovering_h && !dragging_v && !dragging_h) || (dragging_v && dragging_h)) {
							draw_texture(vh_grabber, half - vh_grabber->get_size() / 2.0);
							set_default_cursor_shape(CURSOR_DRAG);
						} else if ((hovering_v && !dragging_h) || dragging_v) {
							draw_texture(v_grabber, half - v_grabber->get_size() / 2.0);
							set_default_cursor_shape(CURSOR_VSPLIT);
						} else if (hovering_h || dragging_h) {
							draw_texture(h_grabber, half - h_grabber->get_size() / 2.0);
							set_default_cursor_shape(CURSOR_HSPLIT);
						}

					} break;
				}
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			Node3DEditorViewport *viewports[4];
			int vc = 0;
			for (int i = 0; i < get_child_count(); i++) {
				viewports[vc] = Object::cast_to<Node3DEditorViewport>(get_child(i));
				if (viewports[vc]) {
					vc++;
				}
			}

			ERR_FAIL_COND(vc != 4);

			Size2 size = get_size();

			if (size.x < 10 || size.y < 10) {
				for (int i = 0; i < 4; i++) {
					viewports[i]->hide();
				}
				return;
			}
			int h_sep = get_theme_constant(SNAME("separation"), SNAME("HSplitContainer"));

			int v_sep = get_theme_constant(SNAME("separation"), SNAME("VSplitContainer"));

			int mid_w = size.width * ratio_h;
			int mid_h = size.height * ratio_v;

			int size_left = mid_w - h_sep / 2;
			int size_right = size.width - mid_w - h_sep / 2;

			int size_top = mid_h - v_sep / 2;
			int size_bottom = size.height - mid_h - v_sep / 2;

			switch (view) {
				case VIEW_USE_1_VIEWPORT: {
					viewports[0]->show();
					for (int i = 1; i < 4; i++) {
						viewports[i]->hide();
					}

					fit_child_in_rect(viewports[0], Rect2(Vector2(), size));

				} break;
				case VIEW_USE_2_VIEWPORTS: {
					for (int i = 0; i < 4; i++) {
						if (i == 1 || i == 3) {
							viewports[i]->hide();
						} else {
							viewports[i]->show();
						}
					}

					fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size.width, size_top)));
					fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size.width, size_bottom)));

				} break;
				case VIEW_USE_2_VIEWPORTS_ALT: {
					for (int i = 0; i < 4; i++) {
						if (i == 1 || i == 3) {
							viewports[i]->hide();
						} else {
							viewports[i]->show();
						}
					}
					fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size_left, size.height)));
					fit_child_in_rect(viewports[2], Rect2(Vector2(mid_w + h_sep / 2, 0), Vector2(size_right, size.height)));

				} break;
				case VIEW_USE_3_VIEWPORTS: {
					for (int i = 0; i < 4; i++) {
						if (i == 1) {
							viewports[i]->hide();
						} else {
							viewports[i]->show();
						}
					}

					fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size.width, size_top)));
					fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size_left, size_bottom)));
					fit_child_in_rect(viewports[3], Rect2(Vector2(mid_w + h_sep / 2, mid_h + v_sep / 2), Vector2(size_right, size_bottom)));

				} break;
				case VIEW_USE_3_VIEWPORTS_ALT: {
					for (int i = 0; i < 4; i++) {
						if (i == 1) {
							viewports[i]->hide();
						} else {
							viewports[i]->show();
						}
					}

					fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size_left, size_top)));
					fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size_left, size_bottom)));
					fit_child_in_rect(viewports[3], Rect2(Vector2(mid_w + h_sep / 2, 0), Vector2(size_right, size.height)));

				} break;
				case VIEW_USE_4_VIEWPORTS: {
					for (int i = 0; i < 4; i++) {
						viewports[i]->show();
					}

					fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size_left, size_top)));
					fit_child_in_rect(viewports[1], Rect2(Vector2(mid_w + h_sep / 2, 0), Vector2(size_right, size_top)));
					fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size_left, size_bottom)));
					fit_child_in_rect(viewports[3], Rect2(Vector2(mid_w + h_sep / 2, mid_h + v_sep / 2), Vector2(size_right, size_bottom)));

				} break;
			}
		} break;
	}
}

void Node3DEditorViewportContainer::set_view(View p_view) {
	view = p_view;
	queue_sort();
}

Node3DEditorViewportContainer::View Node3DEditorViewportContainer::get_view() {
	return view;
}

Node3DEditorViewportContainer::Node3DEditorViewportContainer() {
	set_clip_contents(true);
	view = VIEW_USE_1_VIEWPORT;
	mouseover = false;
	ratio_h = 0.5;
	ratio_v = 0.5;
	hovering_v = false;
	hovering_h = false;
	dragging_v = false;
	dragging_h = false;
}
