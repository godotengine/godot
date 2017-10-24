/*************************************************************************/
/*  texture_region_editor_plugin.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Author: Mariano Suligoy                                               */
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
#include "texture_region_editor_plugin.h"

#include "core/core_string_names.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "scene/gui/check_box.h"

void draw_margin_line(Control *edit_draw, Vector2 from, Vector2 to) {
	Vector2 line = (to - from).normalized() * 10;
	while ((to - from).length_squared() > 200) {
		edit_draw->draw_line(from, from + line, EditorNode::get_singleton()->get_theme_base()->get_color("mono_color", "Editor"), 2);
		from += line * 2;
	}
}

void TextureRegionEditor::_region_draw() {
	Ref<Texture> base_tex = NULL;
	if (node_sprite)
		base_tex = node_sprite->get_texture();
	else if (node_ninepatch)
		base_tex = node_ninepatch->get_texture();
	else if (obj_styleBox.is_valid())
		base_tex = obj_styleBox->get_texture();
	else if (atlas_tex.is_valid())
		base_tex = atlas_tex->get_atlas();
	if (base_tex.is_null())
		return;

	Transform2D mtx;
	mtx.elements[2] = -draw_ofs;
	mtx.scale_basis(Vector2(draw_zoom, draw_zoom));

	VS::get_singleton()->canvas_item_add_set_transform(edit_draw->get_canvas_item(), mtx);
	edit_draw->draw_texture(base_tex, Point2());
	VS::get_singleton()->canvas_item_add_set_transform(edit_draw->get_canvas_item(), Transform2D());

	if (snap_mode == SNAP_GRID) {
		Color grid_color = get_color("grid_major_color", "Editor");
		Size2 s = edit_draw->get_size();
		int last_cell = 0;

		if (snap_step.x != 0) {
			if (snap_separation.x == 0)
				for (int i = 0; i < s.width; i++) {
					int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(i, 0)).x - snap_offset.x) / snap_step.x));
					if (i == 0)
						last_cell = cell;
					if (last_cell != cell)
						edit_draw->draw_line(Point2(i, 0), Point2(i, s.height), grid_color);
					last_cell = cell;
				}
			else
				for (int i = 0; i < s.width; i++) {
					int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(i, 0)).x - snap_offset.x) / (snap_step.x + snap_separation.x)));
					if (i == 0)
						last_cell = cell;
					if (last_cell != cell)
						edit_draw->draw_rect(Rect2(i - snap_separation.x * draw_zoom, 0, snap_separation.x * draw_zoom, s.height), grid_color);
					last_cell = cell;
				}
		}

		if (snap_step.y != 0) {
			if (snap_separation.y == 0)
				for (int i = 0; i < s.height; i++) {
					int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(0, i)).y - snap_offset.y) / snap_step.y));
					if (i == 0)
						last_cell = cell;
					if (last_cell != cell)
						edit_draw->draw_line(Point2(0, i), Point2(s.width, i), grid_color);
					last_cell = cell;
				}
			else
				for (int i = 0; i < s.height; i++) {
					int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(0, i)).y - snap_offset.y) / (snap_step.y + snap_separation.y)));
					if (i == 0)
						last_cell = cell;
					if (last_cell != cell)
						edit_draw->draw_rect(Rect2(0, i - snap_separation.y * draw_zoom, s.width, snap_separation.y * draw_zoom), grid_color);
					last_cell = cell;
				}
		}
	} else if (snap_mode == SNAP_AUTOSLICE) {
		for (List<Rect2>::Element *E = autoslice_cache.front(); E; E = E->next()) {
			Rect2 r = E->get();
			Vector2 endpoints[4] = {
				mtx.basis_xform(r.position),
				mtx.basis_xform(r.position + Vector2(r.size.x, 0)),
				mtx.basis_xform(r.position + r.size),
				mtx.basis_xform(r.position + Vector2(0, r.size.y))
			};
			for (int i = 0; i < 4; i++) {
				int next = (i + 1) % 4;
				edit_draw->draw_line(endpoints[i] - draw_ofs, endpoints[next] - draw_ofs, Color(0.3, 0.7, 1, 1), 2);
			}
		}
	}

	Ref<Texture> select_handle = get_icon("EditorHandle", "EditorIcons");

	Rect2 scroll_rect(Point2(), mtx.basis_xform(base_tex->get_size()));
	scroll_rect.expand_to(mtx.basis_xform(edit_draw->get_size()));

	Vector2 endpoints[4] = {
		mtx.basis_xform(rect.position),
		mtx.basis_xform(rect.position + Vector2(rect.size.x, 0)),
		mtx.basis_xform(rect.position + rect.size),
		mtx.basis_xform(rect.position + Vector2(0, rect.size.y))
	};
	Color color = get_color("mono_color", "Editor");
	for (int i = 0; i < 4; i++) {

		int prev = (i + 3) % 4;
		int next = (i + 1) % 4;

		Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
		ofs *= 1.4144 * (select_handle->get_size().width / 2);

		edit_draw->draw_line(endpoints[i] - draw_ofs, endpoints[next] - draw_ofs, color, 2);

		if (snap_mode != SNAP_AUTOSLICE)
			edit_draw->draw_texture(select_handle, (endpoints[i] + ofs - (select_handle->get_size() / 2)).floor() - draw_ofs);

		ofs = (endpoints[next] - endpoints[i]) / 2;
		ofs += (endpoints[next] - endpoints[i]).tangent().normalized() * (select_handle->get_size().width / 2);

		if (snap_mode != SNAP_AUTOSLICE)
			edit_draw->draw_texture(select_handle, (endpoints[i] + ofs - (select_handle->get_size() / 2)).floor() - draw_ofs);

		scroll_rect.expand_to(endpoints[i]);
	}

	scroll_rect = scroll_rect.grow(200);
	updating_scroll = true;
	hscroll->set_min(scroll_rect.position.x);
	hscroll->set_max(scroll_rect.position.x + scroll_rect.size.x);
	hscroll->set_page(edit_draw->get_size().x);
	hscroll->set_value(draw_ofs.x);
	hscroll->set_step(0.001);

	vscroll->set_min(scroll_rect.position.y);
	vscroll->set_max(scroll_rect.position.y + scroll_rect.size.y);
	vscroll->set_page(edit_draw->get_size().y);
	vscroll->set_value(draw_ofs.y);
	vscroll->set_step(0.001);
	updating_scroll = false;

	float margins[4];
	if (node_ninepatch || obj_styleBox.is_valid()) {
		if (node_ninepatch) {
			margins[0] = node_ninepatch->get_patch_margin(MARGIN_TOP);
			margins[1] = node_ninepatch->get_patch_margin(MARGIN_BOTTOM);
			margins[2] = node_ninepatch->get_patch_margin(MARGIN_LEFT);
			margins[3] = node_ninepatch->get_patch_margin(MARGIN_RIGHT);
		} else if (obj_styleBox.is_valid()) {
			margins[0] = obj_styleBox->get_margin_size(MARGIN_TOP);
			margins[1] = obj_styleBox->get_margin_size(MARGIN_BOTTOM);
			margins[2] = obj_styleBox->get_margin_size(MARGIN_LEFT);
			margins[3] = obj_styleBox->get_margin_size(MARGIN_RIGHT);
		}
		Vector2 pos[4] = {
			mtx.basis_xform(Vector2(0, margins[0])) + Vector2(0, endpoints[0].y - draw_ofs.y),
			-mtx.basis_xform(Vector2(0, margins[1])) + Vector2(0, endpoints[2].y - draw_ofs.y),
			mtx.basis_xform(Vector2(margins[2], 0)) + Vector2(endpoints[0].x - draw_ofs.x, 0),
			-mtx.basis_xform(Vector2(margins[3], 0)) + Vector2(endpoints[2].x - draw_ofs.x, 0)
		};

		draw_margin_line(edit_draw, pos[0], pos[0] + Vector2(edit_draw->get_size().x, 0));
		draw_margin_line(edit_draw, pos[1], pos[1] + Vector2(edit_draw->get_size().x, 0));
		draw_margin_line(edit_draw, pos[2], pos[2] + Vector2(0, edit_draw->get_size().y));
		draw_margin_line(edit_draw, pos[3], pos[3] + Vector2(0, edit_draw->get_size().y));
	}
}

void TextureRegionEditor::_region_input(const Ref<InputEvent> &p_input) {
	Transform2D mtx;
	mtx.elements[2] = -draw_ofs;
	mtx.scale_basis(Vector2(draw_zoom, draw_zoom));

	Vector2 endpoints[8] = {
		mtx.xform(rect.position) + Vector2(-4, -4),
		mtx.xform(rect.position + Vector2(rect.size.x / 2, 0)) + Vector2(0, -4),
		mtx.xform(rect.position + Vector2(rect.size.x, 0)) + Vector2(4, -4),
		mtx.xform(rect.position + Vector2(rect.size.x, rect.size.y / 2)) + Vector2(4, 0),
		mtx.xform(rect.position + rect.size) + Vector2(4, 4),
		mtx.xform(rect.position + Vector2(rect.size.x / 2, rect.size.y)) + Vector2(0, 4),
		mtx.xform(rect.position + Vector2(0, rect.size.y)) + Vector2(-4, 4),
		mtx.xform(rect.position + Vector2(0, rect.size.y / 2)) + Vector2(-4, 0)
	};

	Ref<InputEventMouseButton> mb = p_input;
	if (mb.is_valid()) {

		if (mb->get_button_index() == BUTTON_LEFT) {

			if (mb->is_pressed()) {
				if (node_ninepatch || obj_styleBox.is_valid()) {
					edited_margin = -1;
					float margins[4];
					if (node_ninepatch) {
						margins[0] = node_ninepatch->get_patch_margin(MARGIN_TOP);
						margins[1] = node_ninepatch->get_patch_margin(MARGIN_BOTTOM);
						margins[2] = node_ninepatch->get_patch_margin(MARGIN_LEFT);
						margins[3] = node_ninepatch->get_patch_margin(MARGIN_RIGHT);
					} else if (obj_styleBox.is_valid()) {
						margins[0] = obj_styleBox->get_margin_size(MARGIN_TOP);
						margins[1] = obj_styleBox->get_margin_size(MARGIN_BOTTOM);
						margins[2] = obj_styleBox->get_margin_size(MARGIN_LEFT);
						margins[3] = obj_styleBox->get_margin_size(MARGIN_RIGHT);
					}
					Vector2 pos[4] = {
						mtx.basis_xform(rect.position + Vector2(0, margins[0])) - draw_ofs,
						mtx.basis_xform(rect.position + rect.size - Vector2(0, margins[1])) - draw_ofs,
						mtx.basis_xform(rect.position + Vector2(margins[2], 0)) - draw_ofs,
						mtx.basis_xform(rect.position + rect.size - Vector2(margins[3], 0)) - draw_ofs
					};
					if (Math::abs(mb->get_position().y - pos[0].y) < 8) {
						edited_margin = 0;
						prev_margin = margins[0];
					} else if (Math::abs(mb->get_position().y - pos[1].y) < 8) {
						edited_margin = 1;
						prev_margin = margins[1];
					} else if (Math::abs(mb->get_position().x - pos[2].x) < 8) {
						edited_margin = 2;
						prev_margin = margins[2];
					} else if (Math::abs(mb->get_position().x - pos[3].x) < 8) {
						edited_margin = 3;
						prev_margin = margins[3];
					}
					if (edited_margin >= 0) {
						drag_from = Vector2(mb->get_position().x, mb->get_position().y);
						drag = true;
					}
				}
				if (edited_margin < 0 && snap_mode == SNAP_AUTOSLICE) {
					Vector2 point = mtx.affine_inverse().xform(Vector2(mb->get_position().x, mb->get_position().y));
					for (List<Rect2>::Element *E = autoslice_cache.front(); E; E = E->next()) {
						if (E->get().has_point(point)) {
							rect = E->get();
							if (Input::get_singleton()->is_key_pressed(KEY_CONTROL) && !(Input::get_singleton()->is_key_pressed(KEY_SHIFT | KEY_ALT))) {
								Rect2 r;
								if (node_sprite)
									r = node_sprite->get_region_rect();
								else if (node_ninepatch)
									r = node_ninepatch->get_region_rect();
								else if (obj_styleBox.is_valid())
									r = obj_styleBox->get_region_rect();
								else if (atlas_tex.is_valid())
									r = atlas_tex->get_region();
								rect.expand_to(r.position);
								rect.expand_to(r.position + r.size);
							}
							undo_redo->create_action(TTR("Set Region Rect"));
							if (node_sprite) {
								undo_redo->add_do_method(node_sprite, "set_region_rect", rect);
								undo_redo->add_undo_method(node_sprite, "set_region_rect", node_sprite->get_region_rect());
							} else if (node_ninepatch) {
								undo_redo->add_do_method(node_ninepatch, "set_region_rect", rect);
								undo_redo->add_undo_method(node_ninepatch, "set_region_rect", node_ninepatch->get_region_rect());
							} else if (obj_styleBox.is_valid()) {
								undo_redo->add_do_method(obj_styleBox.ptr(), "set_region_rect", rect);
								undo_redo->add_undo_method(obj_styleBox.ptr(), "set_region_rect", obj_styleBox->get_region_rect());
							} else if (atlas_tex.is_valid()) {
								undo_redo->add_do_method(atlas_tex.ptr(), "set_region", rect);
								undo_redo->add_undo_method(atlas_tex.ptr(), "set_region", atlas_tex->get_region());
							}
							undo_redo->add_do_method(edit_draw, "update");
							undo_redo->add_undo_method(edit_draw, "update");
							undo_redo->commit_action();
							break;
						}
					}
				} else if (edited_margin < 0) {
					drag_from = mtx.affine_inverse().xform(Vector2(mb->get_position().x, mb->get_position().y));
					if (snap_mode == SNAP_PIXEL)
						drag_from = drag_from.snapped(Vector2(1, 1));
					else if (snap_mode == SNAP_GRID)
						drag_from = snap_point(drag_from);
					drag = true;
					if (node_sprite)
						rect_prev = node_sprite->get_region_rect();
					else if (node_ninepatch)
						rect_prev = node_ninepatch->get_region_rect();
					else if (obj_styleBox.is_valid())
						rect_prev = obj_styleBox->get_region_rect();
					else if (atlas_tex.is_valid())
						rect_prev = atlas_tex->get_region();

					for (int i = 0; i < 8; i++) {
						Vector2 tuv = endpoints[i];
						if (tuv.distance_to(Vector2(mb->get_position().x, mb->get_position().y)) < 8) {
							drag_index = i;
						}
					}

					if (drag_index == -1) {
						creating = true;
						rect = Rect2(drag_from, Size2());
					}
				}

			} else if (drag) {
				if (edited_margin >= 0) {
					undo_redo->create_action("Set Margin");
					static Margin m[4] = { MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT };
					if (node_ninepatch) {
						undo_redo->add_do_method(node_ninepatch, "set_patch_margin", m[edited_margin], node_ninepatch->get_patch_margin(m[edited_margin]));
						undo_redo->add_undo_method(node_ninepatch, "set_patch_margin", m[edited_margin], prev_margin);
					} else if (obj_styleBox.is_valid()) {
						undo_redo->add_do_method(obj_styleBox.ptr(), "set_margin_size", m[edited_margin], obj_styleBox->get_margin_size(m[edited_margin]));
						undo_redo->add_undo_method(obj_styleBox.ptr(), "set_margin_size", m[edited_margin], prev_margin);
						obj_styleBox->emit_signal(CoreStringNames::get_singleton()->changed);
					}
					edited_margin = -1;
				} else {
					undo_redo->create_action("Set Region Rect");
					if (node_sprite) {
						undo_redo->add_do_method(node_sprite, "set_region_rect", node_sprite->get_region_rect());
						undo_redo->add_undo_method(node_sprite, "set_region_rect", rect_prev);
					} else if (atlas_tex.is_valid()) {
						undo_redo->add_do_method(atlas_tex.ptr(), "set_region", atlas_tex->get_region());
						undo_redo->add_undo_method(atlas_tex.ptr(), "set_region", rect_prev);
					} else if (node_ninepatch) {
						// FIXME: Is this intentional?
					} else if (node_ninepatch) {
						undo_redo->add_do_method(node_ninepatch, "set_region_rect", node_ninepatch->get_region_rect());
						undo_redo->add_undo_method(node_ninepatch, "set_region_rect", rect_prev);
					} else if (obj_styleBox.is_valid()) {
						undo_redo->add_do_method(obj_styleBox.ptr(), "set_region_rect", obj_styleBox->get_region_rect());
						undo_redo->add_undo_method(obj_styleBox.ptr(), "set_region_rect", rect_prev);
					}
					drag_index = -1;
				}
				undo_redo->add_do_method(edit_draw, "update");
				undo_redo->add_undo_method(edit_draw, "update");
				undo_redo->commit_action();
				drag = false;
				creating = false;
			}

		} else if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {

			if (drag) {
				drag = false;
				if (edited_margin >= 0) {
					static Margin m[4] = { MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT };
					if (node_ninepatch)
						node_ninepatch->set_patch_margin(m[edited_margin], prev_margin);
					if (obj_styleBox.is_valid())
						obj_styleBox->set_margin_size(m[edited_margin], prev_margin);
					edited_margin = -1;
				} else {
					apply_rect(rect_prev);
					rect = rect_prev;
					edit_draw->update();
					drag_index = -1;
				}
			}
		} else if (mb->get_button_index() == BUTTON_WHEEL_UP && mb->is_pressed()) {
			_zoom_in();
		} else if (mb->get_button_index() == BUTTON_WHEEL_DOWN && mb->is_pressed()) {
			_zoom_out();
		}
	}

	Ref<InputEventMouseMotion> mm = p_input;

	if (mm.is_valid()) {

		if (mm->get_button_mask() & BUTTON_MASK_MIDDLE || Input::get_singleton()->is_key_pressed(KEY_SPACE)) {

			Vector2 draged(mm->get_relative().x, mm->get_relative().y);
			hscroll->set_value(hscroll->get_value() - draged.x);
			vscroll->set_value(vscroll->get_value() - draged.y);

		} else if (drag) {

			if (edited_margin >= 0) {
				float new_margin = 0;
				if (edited_margin == 0)
					new_margin = prev_margin + (mm->get_position().y - drag_from.y) / draw_zoom;
				else if (edited_margin == 1)
					new_margin = prev_margin - (mm->get_position().y - drag_from.y) / draw_zoom;
				else if (edited_margin == 2)
					new_margin = prev_margin + (mm->get_position().x - drag_from.x) / draw_zoom;
				else if (edited_margin == 3)
					new_margin = prev_margin - (mm->get_position().x - drag_from.x) / draw_zoom;
				else
					ERR_PRINT("Unexpected edited_margin");

				if (new_margin < 0)
					new_margin = 0;
				static Margin m[4] = { MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT };
				if (node_ninepatch)
					node_ninepatch->set_patch_margin(m[edited_margin], new_margin);
				if (obj_styleBox.is_valid())
					obj_styleBox->set_margin_size(m[edited_margin], new_margin);
			} else {
				Vector2 new_pos = mtx.affine_inverse().xform(mm->get_position());
				if (snap_mode == SNAP_PIXEL)
					new_pos = new_pos.snapped(Vector2(1, 1));
				else if (snap_mode == SNAP_GRID)
					new_pos = snap_point(new_pos);

				if (creating) {
					rect = Rect2(drag_from, Size2());
					rect.expand_to(new_pos);
					apply_rect(rect);
					edit_draw->update();
					return;
				}

				switch (drag_index) {
					case 0: {
						Vector2 p = rect_prev.position + rect_prev.size;
						rect = Rect2(p, Size2());
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 1: {
						Vector2 p = rect_prev.position + Vector2(0, rect_prev.size.y);
						rect = Rect2(p, Size2(rect_prev.size.x, 0));
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 2: {
						Vector2 p = rect_prev.position + Vector2(0, rect_prev.size.y);
						rect = Rect2(p, Size2());
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 3: {
						Vector2 p = rect_prev.position;
						rect = Rect2(p, Size2(0, rect_prev.size.y));
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 4: {
						Vector2 p = rect_prev.position;
						rect = Rect2(p, Size2());
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 5: {
						Vector2 p = rect_prev.position;
						rect = Rect2(p, Size2(rect_prev.size.x, 0));
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 6: {
						Vector2 p = rect_prev.position + Vector2(rect_prev.size.x, 0);
						rect = Rect2(p, Size2());
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
					case 7: {
						Vector2 p = rect_prev.position + Vector2(rect_prev.size.x, 0);
						rect = Rect2(p, Size2(0, rect_prev.size.y));
						rect.expand_to(new_pos);
						apply_rect(rect);
					} break;
				}
			}
			edit_draw->update();
		}
	}
}

void TextureRegionEditor::_scroll_changed(float) {
	if (updating_scroll)
		return;

	draw_ofs.x = hscroll->get_value();
	draw_ofs.y = vscroll->get_value();
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_mode(int p_mode) {
	snap_mode_button->get_popup()->set_item_checked(snap_mode, false);
	snap_mode = p_mode;
	snap_mode_button->set_text(snap_mode_button->get_popup()->get_item_text(p_mode));
	snap_mode_button->get_popup()->set_item_checked(snap_mode, true);

	if (snap_mode == SNAP_GRID)
		hb_grid->show();
	else
		hb_grid->hide();

	edit_draw->update();
}

void TextureRegionEditor::_set_snap_off_x(float p_val) {
	snap_offset.x = p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_off_y(float p_val) {
	snap_offset.y = p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_step_x(float p_val) {
	snap_step.x = p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_step_y(float p_val) {
	snap_step.y = p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_sep_x(float p_val) {
	snap_separation.x = p_val;
	edit_draw->update();
}

void TextureRegionEditor::_set_snap_sep_y(float p_val) {
	snap_separation.y = p_val;
	edit_draw->update();
}

void TextureRegionEditor::_zoom_in() {
	if (draw_zoom < 8) {
		draw_zoom *= 2;
		edit_draw->update();
	}
}

void TextureRegionEditor::_zoom_reset() {
	if (draw_zoom == 1) return;
	draw_zoom = 1;
	edit_draw->update();
}

void TextureRegionEditor::_zoom_out() {
	if (draw_zoom > 0.25) {
		draw_zoom /= 2;
		edit_draw->update();
	}
}

void TextureRegionEditor::apply_rect(const Rect2 &rect) {
	if (node_sprite)
		node_sprite->set_region_rect(rect);
	else if (node_ninepatch)
		node_ninepatch->set_region_rect(rect);
	else if (obj_styleBox.is_valid())
		obj_styleBox->set_region_rect(rect);
	else if (atlas_tex.is_valid())
		atlas_tex->set_region(rect);
}

void TextureRegionEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			zoom_out->set_icon(get_icon("ZoomLess", "EditorIcons"));
			zoom_reset->set_icon(get_icon("ZoomReset", "EditorIcons"));
			zoom_in->set_icon(get_icon("ZoomMore", "EditorIcons"));
			icon_zoom->set_texture(get_icon("Zoom", "EditorIcons"));
		} break;
	}
}

void TextureRegionEditor::_node_removed(Object *p_obj) {
	if (p_obj == node_sprite || p_obj == node_ninepatch || p_obj == obj_styleBox.ptr() || p_obj == atlas_tex.ptr()) {
		node_ninepatch = NULL;
		node_sprite = NULL;
		obj_styleBox = Ref<StyleBox>(NULL);
		atlas_tex = Ref<AtlasTexture>(NULL);
		hide();
	}
}

void TextureRegionEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_edit_region"), &TextureRegionEditor::_edit_region);
	ClassDB::bind_method(D_METHOD("_region_draw"), &TextureRegionEditor::_region_draw);
	ClassDB::bind_method(D_METHOD("_region_input"), &TextureRegionEditor::_region_input);
	ClassDB::bind_method(D_METHOD("_scroll_changed"), &TextureRegionEditor::_scroll_changed);
	ClassDB::bind_method(D_METHOD("_node_removed"), &TextureRegionEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_set_snap_mode"), &TextureRegionEditor::_set_snap_mode);
	ClassDB::bind_method(D_METHOD("_set_snap_off_x"), &TextureRegionEditor::_set_snap_off_x);
	ClassDB::bind_method(D_METHOD("_set_snap_off_y"), &TextureRegionEditor::_set_snap_off_y);
	ClassDB::bind_method(D_METHOD("_set_snap_step_x"), &TextureRegionEditor::_set_snap_step_x);
	ClassDB::bind_method(D_METHOD("_set_snap_step_y"), &TextureRegionEditor::_set_snap_step_y);
	ClassDB::bind_method(D_METHOD("_set_snap_sep_x"), &TextureRegionEditor::_set_snap_sep_x);
	ClassDB::bind_method(D_METHOD("_set_snap_sep_y"), &TextureRegionEditor::_set_snap_sep_y);
	ClassDB::bind_method(D_METHOD("_zoom_in"), &TextureRegionEditor::_zoom_in);
	ClassDB::bind_method(D_METHOD("_zoom_reset"), &TextureRegionEditor::_zoom_reset);
	ClassDB::bind_method(D_METHOD("_zoom_out"), &TextureRegionEditor::_zoom_out);
}

void TextureRegionEditor::edit(Object *p_obj) {
	if (node_sprite)
		node_sprite->remove_change_receptor(this);
	if (node_ninepatch)
		node_ninepatch->remove_change_receptor(this);
	if (obj_styleBox.is_valid())
		obj_styleBox->remove_change_receptor(this);
	if (atlas_tex.is_valid())
		atlas_tex->remove_change_receptor(this);
	if (p_obj) {
		node_sprite = Object::cast_to<Sprite>(p_obj);
		node_ninepatch = Object::cast_to<NinePatchRect>(p_obj);
		if (Object::cast_to<StyleBoxTexture>(p_obj))
			obj_styleBox = Ref<StyleBoxTexture>(Object::cast_to<StyleBoxTexture>(p_obj));
		if (Object::cast_to<AtlasTexture>(p_obj))
			atlas_tex = Ref<AtlasTexture>(Object::cast_to<AtlasTexture>(p_obj));
		p_obj->add_change_receptor(this);
		_edit_region();
	} else {
		node_sprite = NULL;
		node_ninepatch = NULL;
		obj_styleBox = Ref<StyleBoxTexture>(NULL);
		atlas_tex = Ref<AtlasTexture>(NULL);
	}
	edit_draw->update();
}

void TextureRegionEditor::_changed_callback(Object *p_changed, const char *p_prop) {

	if (!is_visible())
		return;
	if (p_prop == StringName("atlas") || p_prop == StringName("texture"))
		_edit_region();
}

void TextureRegionEditor::_edit_region() {
	Ref<Texture> texture = NULL;
	if (node_sprite)
		texture = node_sprite->get_texture();
	else if (node_ninepatch)
		texture = node_ninepatch->get_texture();
	else if (obj_styleBox.is_valid())
		texture = obj_styleBox->get_texture();
	else if (atlas_tex.is_valid())
		texture = atlas_tex->get_atlas();

	if (texture.is_null()) {
		return;
	}

	autoslice_cache.clear();
	Ref<Image> i;
	i.instance();
	if (i->load(texture->get_path()) == OK) {
		BitMap bm;
		bm.create_from_image_alpha(i);
		for (int y = 0; y < i->get_height(); y++) {
			for (int x = 0; x < i->get_width(); x++) {
				if (bm.get_bit(Point2(x, y))) {
					bool found = false;
					for (List<Rect2>::Element *E = autoslice_cache.front(); E; E = E->next()) {
						Rect2 grown = E->get().grow(1.5);
						if (grown.has_point(Point2(x, y))) {
							E->get().expand_to(Point2(x, y));
							E->get().expand_to(Point2(x + 1, y + 1));
							x = E->get().position.x + E->get().size.x - 1;
							bool merged = true;
							while (merged) {
								merged = false;
								bool queue_erase = false;
								for (List<Rect2>::Element *F = autoslice_cache.front(); F; F = F->next()) {
									if (queue_erase) {
										autoslice_cache.erase(F->prev());
										queue_erase = false;
									}
									if (F == E)
										continue;
									if (E->get().grow(1).intersects(F->get())) {
										E->get().expand_to(F->get().position);
										E->get().expand_to(F->get().position + F->get().size);
										if (F->prev()) {
											F = F->prev();
											autoslice_cache.erase(F->next());
										} else {
											queue_erase = true;
											//Can't delete the first rect in the list.
										}
										merged = true;
									}
								}
							}
							found = true;
							break;
						}
					}
					if (!found) {
						Rect2 new_rect(x, y, 1, 1);
						autoslice_cache.push_back(new_rect);
					}
				}
			}
		}
	}

	if (node_sprite)
		rect = node_sprite->get_region_rect();
	else if (node_ninepatch)
		rect = node_ninepatch->get_region_rect();
	else if (obj_styleBox.is_valid())
		rect = obj_styleBox->get_region_rect();
	else if (atlas_tex.is_valid())
		rect = atlas_tex->get_region();

	edit_draw->update();
}

Vector2 TextureRegionEditor::snap_point(Vector2 p_target) const {
	if (snap_mode == SNAP_GRID) {
		p_target.x = Math::snap_scalar_seperation(snap_offset.x, snap_step.x, p_target.x, snap_separation.x);
		p_target.y = Math::snap_scalar_seperation(snap_offset.y, snap_step.y, p_target.y, snap_separation.y);
	}

	return p_target;
}

TextureRegionEditor::TextureRegionEditor(EditorNode *p_editor) {
	node_sprite = NULL;
	node_ninepatch = NULL;
	obj_styleBox = Ref<StyleBoxTexture>(NULL);
	atlas_tex = Ref<AtlasTexture>(NULL);
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	snap_step = Vector2(10, 10);
	snap_separation = Vector2(0, 0);
	edited_margin = -1;
	drag_index = -1;
	drag = false;

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);
	main_vb->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	HBoxContainer *hb_tools = memnew(HBoxContainer);
	main_vb->add_child(hb_tools);

	hb_tools->add_child(memnew(Label(TTR("Snap Mode:"))));

	snap_mode_button = memnew(MenuButton);
	hb_tools->add_child(snap_mode_button);
	snap_mode_button->set_text(TTR("<None>"));
	PopupMenu *p = snap_mode_button->get_popup();
	p->set_hide_on_checkable_item_selection(false);
	p->add_item(TTR("<None>"), 0);
	p->add_item(TTR("Pixel Snap"), 1);
	p->add_item(TTR("Grid Snap"), 2);
	p->add_item(TTR("Auto Slice"), 3);
	for (int i = 0; i < 4; i++)
		p->set_item_as_checkable(i, true);
	p->set_item_checked(0, true);
	p->connect("id_pressed", this, "_set_snap_mode");
	hb_grid = memnew(HBoxContainer);
	hb_tools->add_child(hb_grid);
	hb_grid->add_child(memnew(VSeparator));

	hb_grid->add_child(memnew(Label(TTR("Offset:"))));

	sb_off_x = memnew(SpinBox);
	sb_off_x->set_min(-256);
	sb_off_x->set_max(256);
	sb_off_x->set_step(1);
	sb_off_x->set_value(snap_offset.x);
	sb_off_x->set_suffix("px");
	sb_off_x->connect("value_changed", this, "_set_snap_off_x");
	hb_grid->add_child(sb_off_x);

	sb_off_y = memnew(SpinBox);
	sb_off_y->set_min(-256);
	sb_off_y->set_max(256);
	sb_off_y->set_step(1);
	sb_off_y->set_value(snap_offset.y);
	sb_off_y->set_suffix("px");
	sb_off_y->connect("value_changed", this, "_set_snap_off_y");
	hb_grid->add_child(sb_off_y);

	hb_grid->add_child(memnew(VSeparator));
	hb_grid->add_child(memnew(Label(TTR("Step:"))));

	sb_step_x = memnew(SpinBox);
	sb_step_x->set_min(-256);
	sb_step_x->set_max(256);
	sb_step_x->set_step(1);
	sb_step_x->set_value(snap_step.x);
	sb_step_x->set_suffix("px");
	sb_step_x->connect("value_changed", this, "_set_snap_step_x");
	hb_grid->add_child(sb_step_x);

	sb_step_y = memnew(SpinBox);
	sb_step_y->set_min(-256);
	sb_step_y->set_max(256);
	sb_step_y->set_step(1);
	sb_step_y->set_value(snap_step.y);
	sb_step_y->set_suffix("px");
	sb_step_y->connect("value_changed", this, "_set_snap_step_y");
	hb_grid->add_child(sb_step_y);

	hb_grid->add_child(memnew(VSeparator));
	hb_grid->add_child(memnew(Label(TTR("Separation:"))));

	sb_sep_x = memnew(SpinBox);
	sb_sep_x->set_min(0);
	sb_sep_x->set_max(256);
	sb_sep_x->set_step(1);
	sb_sep_x->set_value(snap_separation.x);
	sb_sep_x->set_suffix("px");
	sb_sep_x->connect("value_changed", this, "_set_snap_sep_x");
	hb_grid->add_child(sb_sep_x);

	sb_sep_y = memnew(SpinBox);
	sb_sep_y->set_min(0);
	sb_sep_y->set_max(256);
	sb_sep_y->set_step(1);
	sb_sep_y->set_value(snap_separation.y);
	sb_sep_y->set_suffix("px");
	sb_sep_y->connect("value_changed", this, "_set_snap_sep_y");
	hb_grid->add_child(sb_sep_y);

	hb_grid->hide();

	HBoxContainer *main_hb = memnew(HBoxContainer);
	main_vb->add_child(main_hb);
	edit_draw = memnew(Control);
	main_hb->add_child(edit_draw);
	main_hb->set_v_size_flags(SIZE_EXPAND_FILL);
	edit_draw->set_h_size_flags(SIZE_EXPAND_FILL);

	Control *separator = memnew(Control);
	separator->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb_tools->add_child(separator);

	icon_zoom = memnew(TextureRect);
	hb_tools->add_child(icon_zoom);

	zoom_out = memnew(ToolButton);
	zoom_out->connect("pressed", this, "_zoom_out");
	hb_tools->add_child(zoom_out);

	zoom_reset = memnew(ToolButton);
	zoom_reset->connect("pressed", this, "_zoom_reset");
	hb_tools->add_child(zoom_reset);

	zoom_in = memnew(ToolButton);
	zoom_in->connect("pressed", this, "_zoom_in");
	hb_tools->add_child(zoom_in);

	vscroll = memnew(VScrollBar);
	main_hb->add_child(vscroll);
	vscroll->connect("value_changed", this, "_scroll_changed");
	hscroll = memnew(HScrollBar);
	main_vb->add_child(hscroll);
	hscroll->connect("value_changed", this, "_scroll_changed");

	edit_draw->connect("draw", this, "_region_draw");
	edit_draw->connect("gui_input", this, "_region_input");
	draw_zoom = 1.0;
	updating_scroll = false;

	edit_draw->set_clip_contents(true);
}

void TextureRegionEditorPlugin::edit(Object *p_object) {
	region_editor->edit(p_object);
}

bool TextureRegionEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Sprite") || p_object->is_class("NinePatchRect") || p_object->is_class("StyleBoxTexture") || p_object->is_class("AtlasTexture");
}

void TextureRegionEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		region_button->show();
		if (region_button->is_pressed())
			region_editor->show();
	} else {
		region_button->hide();
		region_editor->edit(NULL);
		region_editor->hide();
	}
}

Dictionary TextureRegionEditorPlugin::get_state() const {

	Dictionary state;
	state["zoom"] = region_editor->draw_zoom;
	state["snap_offset"] = region_editor->snap_offset;
	state["snap_step"] = region_editor->snap_step;
	state["snap_separation"] = region_editor->snap_separation;
	state["snap_mode"] = region_editor->snap_mode;
	return state;
}

void TextureRegionEditorPlugin::set_state(const Dictionary &p_state) {

	Dictionary state = p_state;
	if (state.has("zoom")) {
		region_editor->draw_zoom = p_state["zoom"];
	}

	if (state.has("snap_step")) {
		Vector2 s = state["snap_step"];
		region_editor->sb_step_x->set_value(s.x);
		region_editor->sb_step_y->set_value(s.y);
		region_editor->snap_step = s;
	}

	if (state.has("snap_offset")) {
		Vector2 ofs = state["snap_offset"];
		region_editor->sb_off_x->set_value(ofs.x);
		region_editor->sb_off_y->set_value(ofs.y);
		region_editor->snap_offset = ofs;
	}

	if (state.has("snap_separation")) {
		Vector2 sep = state["snap_separation"];
		region_editor->sb_sep_x->set_value(sep.x);
		region_editor->sb_sep_y->set_value(sep.y);
		region_editor->snap_separation = sep;
	}

	if (state.has("snap_mode")) {
		region_editor->_set_snap_mode(state["snap_mode"]);
	}
}

TextureRegionEditorPlugin::TextureRegionEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	region_editor = memnew(TextureRegionEditor(p_node));

	region_button = p_node->add_bottom_panel_item(TTR("Texture Region"), region_editor);
	region_button->set_tooltip(TTR("Texture Region Editor"));

	region_editor->set_custom_minimum_size(Size2(0, 200));
	region_editor->hide();
	region_button->hide();
}
