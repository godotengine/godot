/*************************************************************************/
/*  tree_item.cpp                                                        */
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

#include "tree_item_cell.h"

#include "scene/gui/tree2.h"
#include "scene/gui/tree_item.h"
#include "scene/resources/text_line.h"
#include "scene/resources/texture.h"

void TreeItemCell::_changed_notify() {
	dirty = true;
	tree->update();
}

void TreeItemCell::draw_icon(const RID &p_where, const Point2 &p_pos, const Size2 &p_size, const Color &p_color) const {
	if (icon.is_null()) {
		return;
	}

	Size2i dsize = (p_size == Size2()) ? icon->get_size() : p_size;

	if (icon_region == Rect2i()) {
		icon->draw_rect_region(p_where, Rect2(p_pos, dsize), Rect2(Point2(), icon->get_size()), p_color);
	} else {
		icon->draw_rect_region(p_where, Rect2(p_pos, dsize), icon_region, p_color);
	}
}

Size2 TreeItemCell::get_icon_size() const {
	if (icon.is_null()) {
		return Size2();
	}
	if (icon_region == Rect2i()) {
		return icon->get_size();
	} else {
		return icon_region.size;
	}
}

int TreeItemCell::get_height() {
	if (dirty) {
		update();
	}

	int height = 0;
	for (int i = 0; i < buttons.size(); i++) {
		Size2i s = tree->cache.button_pressed->get_minimum_size();
		s += buttons[i].texture->get_size();
		height = MAX(height, s.height);
	}


	return height;
}

void TreeItemCell::draw(Vector2i p_pos, int p_label_h, Point2 p_draw_ofs, int &r_skip, int &r_offset) {
	if (r_skip > 0) {
		r_skip--;
		return; // false
	}

	int w = tree->get_column_width(column); // TODO: cache?
	if (column == 0) {
		w -= r_offset;

		if (w <= 0) {
			r_offset = tree->get_column_width(0);
			return;
		}
	} else {
		r_offset += tree->cache.hseparation;
		w -= tree->cache.hseparation;
	}

	if (expand_right) {
		int plus = 1;
		while (column + plus < tree->columns.size() && !editable) {// && p_item->cells[i + plus].mode == TreeItem::CELL_MODE_STRING && p_item->cells[i + plus].text.is_empty() && p_item->cells[i + plus].icon.is_null()) {
			w += tree->get_column_width(column + plus);
			plus++;
			r_skip++;
		}
	}

	// if (!tree->rtl && !buttons.is_empty()) {
	// 	int button_w = 0;
	// 	for (int j = buttons.size() - 1; j >= 0; j--) {
	// 		Ref<Texture2D> b = buttons[j].texture;
	// 		button_w += b->get_size().width + cache.button_pressed->get_minimum_size().width + cache.button_margin;
	// 	}

	// 	int total_ofs = ofs - cache.offset.x;

	// 	if (total_ofs + w > p_draw_size.width) {
	// 		w = MAX(button_w, p_draw_size.width - total_ofs);
	// 	}
	// }

	RID ci = tree->get_canvas_item();

	int bw = 0;
	for (int i = buttons.size() - 1; i >= 0; i--) {
		Ref<Texture2D> b = buttons[i].texture;
		Size2 s = b->get_size() + tree->cache.button_pressed->get_minimum_size();

		Point2i o = Point2i(r_offset + w - s.width, p_pos.y) - tree->cache.offset + p_draw_ofs;

		// if (cache.click_type == Cache::CLICK_BUTTON && cache.click_item == p_item && cache.click_column == i && cache.click_index == i && !p_item->cells[i].buttons[i].disabled) {
		// 	// Being pressed.
		// 	Point2 od = o;
		// 	if (rtl) {
		// 		od.x = get_size().width - od.x - s.x;
		// 	}
		// 	cache.button_pressed->draw(get_canvas_item(), Rect2(od.x, od.y, s.width, MAX(s.height, label_h)));
		// }

		o.y += (p_label_h - s.height) / 2;
		o += tree->cache.button_pressed->get_offset();

		// if (rtl) {
		// 	o.x = get_size().width - o.x - b->get_width();
		// }

		b->draw(ci, o, buttons[i].disabled ? Color(1, 1, 1, 0.5) : buttons[i].color);
		w -= s.width + tree->cache.button_margin;
		bw += s.width + tree->cache.button_margin;
	}
}

void TreeItemCell::add_button(const Ref<Texture2D> &p_button, int p_id, bool p_disabled, const String &p_tooltip) {
	ERR_FAIL_COND(p_button.is_null());
	TreeItemCell::CellButton button;
	button.texture = p_button;
	if (p_id < 0) {
		p_id = buttons.size();
	}
	button.id = p_id;
	button.disabled = p_disabled;
	button.tooltip = p_tooltip;
	buttons.push_back(button);
	// cached_minimum_size_dirty = true;

	_changed_notify();
}

void TreeItemCell::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_button", "button", "id", "disabled", "tooltip"), &TreeItemCell::add_button, DEFVAL(-1), DEFVAL(false), DEFVAL(""));
}

TreeItemCell::TreeItemCell(TreeItem2 *p_item, int p_column) {
	tree_item = p_item;
	tree = tree_item->tree;
	column = p_column;
}

void TreeItemCellText::update() {
	String valtext;

	text_buf->clear();
	// if (p_item->cells[p_col].mode == TreeItem::CELL_MODE_RANGE) {
	// 	if (!p_item->cells[p_col].text.is_empty()) {
	// 		if (!p_item->cells[p_col].editable) {
	// 			return;
	// 		}

	// 		int option = (int)p_item->cells[p_col].val;

	// 		valtext = RTR("(Other)");
	// 		Vector<String> strings = p_item->cells[p_col].text.split(",");
	// 		for (int j = 0; j < strings.size(); j++) {
	// 			int value = j;
	// 			if (!strings[j].get_slicec(':', 1).is_empty()) {
	// 				value = strings[j].get_slicec(':', 1).to_int();
	// 			}
	// 			if (option == value) {
	// 				valtext = strings[j].get_slicec(':', 0);
	// 				break;
	// 			}
	// 		}

	// 	} else {
	// 		valtext = String::num(p_item->cells[p_col].val, Math::range_step_decimals(p_item->cells[p_col].step));
	// 	}
	// } else {
	valtext = text;
	// }

	// if (!p_item->cells[p_col].suffix.is_empty()) {
	// 	valtext += " " + p_item->cells[p_col].suffix;
	// }

	// if (p_item->cells[p_col].text_direction == Control::TEXT_DIRECTION_INHERITED) {
	// 	p_item->cells.write[p_col].text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	// } else {
	// 	p_item->cells.write[p_col].text_buf->set_direction((TextServer::Direction)p_item->cells[p_col].text_direction);
	// }

	Ref<Font> font;
	// if (p_item->cells[p_col].custom_font.is_valid()) {
	// 	font = p_item->cells[p_col].custom_font;
	// } else {
	font = tree->cache.font;
	// }

	int font_size;
	// if (p_item->cells[p_col].custom_font_size > 0) {
	// 	font_size = p_item->cells[p_col].custom_font_size;
	// } else {
	font_size = tree->cache.font_size;
	// }
	text_buf->add_string(valtext, font, font_size); //, p_item->cells[p_col].language);
	// TS->shaped_text_set_bidi_override(p_item->cells[p_col].text_buf->get_rid(), structured_text_parser(p_item->cells[p_col].st_parser, p_item->cells[p_col].st_args, valtext));
	dirty = false;
}

int TreeItemCellText::get_height() {
	int height = TreeItemCell::get_height();
	height = MAX(height, text_buf->get_size().y);
	return height;
}

void TreeItemCellText::draw(Point2i p_pos, int p_label_h, Point2 p_draw_ofs, int &r_skip, int &r_offset) {
	TreeItemCell::draw(p_pos, p_label_h, p_draw_ofs, r_skip, r_offset);

	int w = 0;
	RID ci = tree->get_canvas_item();

	Rect2i item_rect = Rect2i(Point2i(r_offset, p_pos.y) - tree->cache.offset/* + p_draw_ofs*/, Size2i(w, p_label_h));
	Rect2i cell_rect = item_rect;
	if (column != 0) {
		cell_rect.position.x -= tree->cache.hseparation;
		cell_rect.size.x += tree->cache.hseparation;
	}

	if (tree->cache.draw_guides) {
		Rect2 r = cell_rect;
		// if (rtl) {
		// 	r.position.x = tree->get_size().width - r.position.x - r.size.x;
		// }
		RenderingServer::get_singleton()->canvas_item_add_line(tree->get_canvas_item(), Point2i(r.position.x, r.position.y + r.size.height), r.position + r.size, tree->cache.guide_color, 1);
	}

	if (column == 0) {
		// if (cells[0].selected && select_mode == SELECT_ROW) {
		// 	Rect2i row_rect = Rect2i(Point2i(tree->cache.bg->get_margin(SIDE_LEFT), item_rect.position.y), Size2i(get_size().width - tree->cache.bg->get_minimum_size().width, item_rect.size.y));
		// 	//Rect2 r = Rect2i(row_rect.pos,row_rect.size);
		// 	//r.grow(tree->cache.selected->get_margin(SIDE_LEFT));
		// 	if (rtl) {
		// 		row_rect.position.x = get_size().width - row_rect.position.x - row_rect.size.x;
		// 	}
		// 	if (has_focus()) {
		// 		tree->cache.selected_focus->draw(ci, row_rect);
		// 	} else {
		// 		tree->cache.selected->draw(ci, row_rect);
		// 	}
		// }
	}

	if (/*(select_mode == SELECT_ROW && selected_item == p_item) || */selected || !tree_item->has_meta("__focus_rect")) {
		Rect2i r = cell_rect;

		set_meta("__focus_rect", Rect2(r.position, r.size));

		// if (rtl) {
		// 	r.position.x = tree->get_size().width - r.position.x - r.size.x;
		// }

		if (selected) {
			if (tree->has_focus()) {
				tree->cache.selected_focus->draw(ci, r);
			} else {
				tree->cache.selected->draw(ci, r);
			}
		}
	}

	if (custom_bg_color) {
		Rect2 r = cell_rect;
		if (column == 0) {
			// r.position.x = p_draw_ofs.x;
			r.size.x = w + r_offset;
		} else {
			r.position.x -= tree->cache.hseparation;
			r.size.x += tree->cache.hseparation;
		}
		// if (rtl) {
		// 	r.position.x = tree->get_size().width - r.position.x - r.size.x;
		// }

		if (custom_bg_outline) {
			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, r.size.x, 1), bg_color);
			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y + r.size.y - 1, r.size.x, 1), bg_color);
			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, 1, r.size.y), bg_color);
			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x + r.size.x - 1, r.position.y, 1, r.size.y), bg_color);
		} else {
			RenderingServer::get_singleton()->canvas_item_add_rect(ci, r, bg_color);
		}
	}

	// if (drop_mode_flags && drop_mode_over) {
	// 	Rect2 r = cell_rect;
	// 	if (rtl) {
	// 		r.position.x = get_size().width - r.position.x - r.size.x;
	// 	}
	// 	if (drop_mode_over == p_item) {
	// 		if (drop_mode_section == 0 || drop_mode_section == -1) {
	// 			// Line above.
	// 			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, r.size.x, 1), tree->cache.drop_position_color);
	// 		}
	// 		if (drop_mode_section == 0) {
	// 			// Side lines.
	// 			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, 1, r.size.y), tree->cache.drop_position_color);
	// 			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x + r.size.x - 1, r.position.y, 1, r.size.y), tree->cache.drop_position_color);
	// 		}
	// 		if (drop_mode_section == 0 || (drop_mode_section == 1 && (!get_first_child() || is_collapsed()))) {
	// 			// Line below.
	// 			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y + r.size.y, r.size.x, 1), tree->cache.drop_position_color);
	// 		}
	// 	} else if (drop_mode_over == get_parent()) {
	// 		if (drop_mode_section == 1 && !get_prev() /* && !drop_mode_over->is_collapsed() */) { // The drop_mode_over shouldn't ever be collapsed in here, otherwise we would be drawing a child of a collapsed item.
	// 			// Line above.
	// 			RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, r.size.x, 1), tree->cache.drop_position_color);
	// 		}
	// 	}
	// }

	Color col = custom_text_color ? text_color : tree->get_theme_color(selected ? "font_selected_color" : "font_color");
	Color font_outline_color = tree->cache.font_outline_color;
	int outline_size = tree->cache.font_outline_size;
	Color icon_col = icon_color;

	if (dirty) {
		update();
	}

	// if (tree->rtl) {
	// 	item_rect.position.x = tree->get_size().width - item_rect.position.x - item_rect.size.x;
	// }

	Point2i text_pos = item_rect.position;
	text_pos.y += Math::floor((item_rect.size.y - text_buf->get_size().y) / 2);
	int text_width = text_buf->get_size().x;

	// switch (cells[i]->mode) {
	// 	case TreeItem::CELL_MODE_STRING: {
			draw_item_rect(item_rect, col, icon_col, outline_size, font_outline_color);
	// 	} break;
	// 	case TreeItem::CELL_MODE_CHECK: {
	// 		Ref<Texture2D> checked = tree->cache.checked;
	// 		Ref<Texture2D> unchecked = tree->cache.unchecked;
	// 		Ref<Texture2D> indeterminate = tree->cache.indeterminate;
	// 		Point2i check_ofs = item_rect.position;
	// 		check_ofs.y += Math::floor((real_t)(item_rect.size.y - checked->get_height()) / 2);

	// 		if (cells[i]->indeterminate) {
	// 			indeterminate->draw(ci, check_ofs);
	// 		} else if (cells[i]->checked) {
	// 			checked->draw(ci, check_ofs);
	// 		} else {
	// 			unchecked->draw(ci, check_ofs);
	// 		}

	// 		int check_w = checked->get_width() + tree->cache.hseparation;

	// 		text_pos.x += check_w;

	// 		item_rect.size.x -= check_w;
	// 		item_rect.position.x += check_w;

	// 		draw_item_rect(cells.write[i], item_rect, col, icon_col, outline_size, font_outline_color);

	// 	} break;
	// 	case TreeItem::CELL_MODE_RANGE: {
	// 		if (!cells[i]->text.is_empty()) {
	// 			if (!cells[i]->editable) {
	// 				break;
	// 			}

	// 			Ref<Texture2D> downarrow = tree->cache.select_arrow;
	// 			int cell_width = item_rect.size.x - downarrow->get_width();

	// 			cells.write[i].text_buf->set_width(cell_width);
	// 			if (rtl) {
	// 				if (outline_size > 0 && font_outline_color.a > 0) {
	// 					cells[i]->text_buf->draw_outline(ci, text_pos + Vector2(cell_width - text_width, 0), outline_size, font_outline_color);
	// 				}
	// 				cells[i]->text_buf->draw(ci, text_pos + Vector2(cell_width - text_width, 0), col);
	// 			} else {
	// 				if (outline_size > 0 && font_outline_color.a > 0) {
	// 					cells[i]->text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
	// 				}
	// 				cells[i]->text_buf->draw(ci, text_pos, col);
	// 			}

	// 			Point2i arrow_pos = item_rect.position;
	// 			arrow_pos.x += item_rect.size.x - downarrow->get_width();
	// 			arrow_pos.y += Math::floor(((item_rect.size.y - downarrow->get_height())) / 2.0);

	// 			downarrow->draw(ci, arrow_pos);
	// 		} else {
	// 			Ref<Texture2D> updown = tree->cache.updown;

	// 			int cell_width = item_rect.size.x - updown->get_width();

	// 			if (rtl) {
	// 				if (outline_size > 0 && font_outline_color.a > 0) {
	// 					cells[i]->text_buf->draw_outline(ci, text_pos + Vector2(cell_width - text_width, 0), outline_size, font_outline_color);
	// 				}
	// 				cells[i]->text_buf->draw(ci, text_pos + Vector2(cell_width - text_width, 0), col);
	// 			} else {
	// 				if (outline_size > 0 && font_outline_color.a > 0) {
	// 					cells[i]->text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
	// 				}
	// 				cells[i]->text_buf->draw(ci, text_pos, col);
	// 			}

	// 			if (!cells[i]->editable) {
	// 				break;
	// 			}

	// 			Point2i updown_pos = item_rect.position;
	// 			updown_pos.x += item_rect.size.x - updown->get_width();
	// 			updown_pos.y += Math::floor(((item_rect.size.y - updown->get_height())) / 2.0);

	// 			updown->draw(ci, updown_pos);
	// 		}

	// 	} break;
	// 	case TreeItem::CELL_MODE_ICON: {
	// 		if (cells[i]->icon.is_null()) {
	// 			break;
	// 		}
	// 		Size2i icon_size = cells[i]->get_icon_size();
	// 		if (cells[i]->icon_max_w > 0 && icon_size.width > cells[i]->icon_max_w) {
	// 			icon_size.height = icon_size.height * cells[i]->icon_max_w / icon_size.width;
	// 			icon_size.width = cells[i]->icon_max_w;
	// 		}

	// 		Point2i icon_ofs = (item_rect.size - icon_size) / 2;
	// 		icon_ofs += item_rect.position;

	// 		draw_texture_rect(cells[i]->icon, Rect2(icon_ofs, icon_size), false, icon_col);

	// 	} break;
	// 	case TreeItem::CELL_MODE_CUSTOM: {
	// 		if (cells[i]->custom_draw_obj.is_valid()) {
	// 			Object *cdo = ObjectDB::get_instance(cells[i]->custom_draw_obj);
	// 			if (cdo) {
	// 				cdo->call(cells[i]->custom_draw_callback, p_item, Rect2(item_rect));
	// 			}
	// 		}

	// 		if (!cells[i]->editable) {
	// 			draw_item_rect(cells.write[i], item_rect, col, icon_col, outline_size, font_outline_color);
	// 			break;
	// 		}

	// 		Ref<Texture2D> downarrow = tree->cache.select_arrow;

	// 		Rect2i ir = item_rect;

	// 		Point2i arrow_pos = item_rect.position;
	// 		arrow_pos.x += item_rect.size.x - downarrow->get_width();
	// 		arrow_pos.y += Math::floor(((item_rect.size.y - downarrow->get_height())) / 2.0);
	// 		ir.size.width -= downarrow->get_width();

	// 		if (cells[i]->custom_button) {
	// 			if (tree->cache.hover_item == p_item && tree->cache.hover_cell == i) {
	// 				if (Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT)) {
	// 					draw_style_box(tree->cache.custom_button_pressed, ir);
	// 				} else {
	// 					draw_style_box(tree->cache.custom_button_hover, ir);
	// 					col = tree->cache.custom_button_font_highlight;
	// 				}
	// 			} else {
	// 				draw_style_box(tree->cache.custom_button, ir);
	// 			}
	// 			ir.size -= tree->cache.custom_button->get_minimum_size();
	// 			ir.position += tree->cache.custom_button->get_offset();
	// 		}

	// 		draw_item_rect(cells.write[i], ir, col, icon_col, outline_size, font_outline_color);

	// 		downarrow->draw(ci, arrow_pos);

	// 	} break;
	// }

	if (column == 0) {
		r_offset = tree->get_column_width(0);
	} else {
		// r_offset += w + bw;
	}

	// if (select_mode == SELECT_MULTI && selected_item == p_item && selected_col == i) {
		// if (is_layout_rtl()) {
		// 	cell_rect.position.x = get_size().width - cell_rect.position.x - cell_rect.size.x;
		// }
		// if (tree->has_focus()) {
		// 	tree->cache.cursor->draw(ci, cell_rect);
		// } else {
		// 	tree->cache.cursor_unfocus->draw(ci, cell_rect);
		// }
	// }
}

void TreeItemCellText::draw_item_rect(const Rect2i &p_rect, const Color &p_color, const Color &p_icon_color, int p_ol_size, const Color &p_ol_color) {
	ERR_FAIL_COND(tree->cache.font.is_null());

	Rect2i rect = p_rect;
	Size2 ts = text_buf->get_size();
	bool rtl = tree->is_layout_rtl();

	int w = 0;
	if (!icon.is_null()) {
		Size2i bmsize = get_icon_size();
		if (icon_max_w > 0 && bmsize.width > icon_max_w) {
			bmsize.width = icon_max_w;
		}
		w += bmsize.width + tree->cache.hseparation;
		if (rect.size.width > 0 && (w + ts.width) > rect.size.width) {
			ts.width = rect.size.width - w;
		}
	}
	w += ts.width;

	switch (text_alignment) {
		case HORIZONTAL_ALIGNMENT_FILL:
		case HORIZONTAL_ALIGNMENT_LEFT: {
			if (rtl) {
				rect.position.x += MAX(0, (rect.size.width - w));
			}
		} break;
		case HORIZONTAL_ALIGNMENT_CENTER:
			rect.position.x += MAX(0, (rect.size.width - w) / 2);
			break;
		case HORIZONTAL_ALIGNMENT_RIGHT:
			if (!rtl) {
				rect.position.x += MAX(0, (rect.size.width - w));
			}
			break;
	}

	RID ci = tree->get_canvas_item();

	// if (rtl) {
	// 	Point2 draw_pos = rect.position;
	// 	draw_pos.y += Math::floor((rect.size.y - text_buf->get_size().y) / 2.0);
	// 	text_buf->set_width(MAX(0, rect.size.width));
	// 	if (p_ol_size > 0 && p_ol_color.a > 0) {
	// 		text_buf->draw_outline(ci, draw_pos, p_ol_size, p_ol_color);
	// 	}
	// 	text_buf->draw(ci, draw_pos, p_color);
	// 	rect.position.x += ts.width + cache.hseparation;
	// 	rect.size.x -= ts.width + cache.hseparation;
	// }

	if (!icon.is_null()) {
		Size2i bmsize = get_icon_size();

		if (icon_max_w > 0 && bmsize.width > icon_max_w) {
			bmsize.height = bmsize.height * icon_max_w / bmsize.width;
			bmsize.width = icon_max_w;
		}

		draw_icon(ci, rect.position + Size2i(0, Math::floor((real_t)(rect.size.y - bmsize.y) / 2)), bmsize, p_icon_color);
		rect.position.x += bmsize.x + tree->cache.hseparation;
		rect.size.x -= bmsize.x + tree->cache.hseparation;
	}

	// if (!rtl) {
		Point2 draw_pos = rect.position;
		draw_pos.y += Math::floor((rect.size.y - text_buf->get_size().y) / 2.0);
		text_buf->set_width(MAX(0, rect.size.width));
		if (p_ol_size > 0 && p_ol_color.a > 0) {
			text_buf->draw_outline(ci, draw_pos, p_ol_size, p_ol_color);
		}
		text_buf->draw(ci, draw_pos, p_color);
	// }
}

void TreeItemCellText::set_text(const String &p_text) {
	text = p_text;
}

String TreeItemCellText::get_text() const {
	return text;
}


void TreeItemCellText::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &TreeItemCellText::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &TreeItemCellText::get_text);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "text"), "set_text", "get_text");
}

TreeItemCellText::TreeItemCellText(TreeItem2 *p_item, int p_column) :
		TreeItemCell(p_item, p_column) {
	text_buf.instantiate();
}
