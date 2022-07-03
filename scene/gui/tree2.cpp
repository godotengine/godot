/*************************************************************************/
/*  tree.cpp                                                             */
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

#include "tree2.h"

#include "scene/gui/scroll_bar.h"
#include "scene/gui/tree_item.h"
#include "scene/gui/tree_item_cell.h"
// #include "scene/resources/text_line.h"

TreeItem2 *Tree2::create_item(TreeItem2 *p_parent, int p_idx) {
	ERR_FAIL_COND_V(blocked > 0, nullptr);

	TreeItem2 *ti = nullptr;

	if (p_parent) {
		ERR_FAIL_COND_V_MSG(p_parent->tree != this, nullptr, "A different tree owns the given parent");
		ti = p_parent->create_child(p_idx);
	} else {
		if (!root) {
			// No root exists, make the given item the new root.
			ti = memnew(TreeItem2(this));
			ti->cells.resize(columns.size());
			ti->is_root = true;
			root = ti;
		} else {
			// Root exists, append or insert to root.
			ti = create_item(root, p_idx);
		}
	}

	return ti;
}

void Tree2::update_cache() {
	cache.font = get_theme_font(SNAME("font"));
	cache.font_size = get_theme_font_size(SNAME("font_size"));
	cache.tb_font = get_theme_font(SNAME("title_button_font"));
	cache.tb_font_size = get_theme_font_size(SNAME("title_button_font_size"));
	cache.bg = get_theme_stylebox(SNAME("bg"));
	cache.selected = get_theme_stylebox(SNAME("selected"));
	cache.selected_focus = get_theme_stylebox(SNAME("selected_focus"));
	cache.cursor = get_theme_stylebox(SNAME("cursor"));
	cache.cursor_unfocus = get_theme_stylebox(SNAME("cursor_unfocused"));
	cache.button_pressed = get_theme_stylebox(SNAME("button_pressed"));

	cache.checked = get_theme_icon(SNAME("checked"));
	cache.unchecked = get_theme_icon(SNAME("unchecked"));
	cache.indeterminate = get_theme_icon(SNAME("indeterminate"));
	if (is_layout_rtl()) {
		cache.arrow_collapsed = get_theme_icon(SNAME("arrow_collapsed_mirrored"));
	} else {
		cache.arrow_collapsed = get_theme_icon(SNAME("arrow_collapsed"));
	}
	cache.arrow = get_theme_icon(SNAME("arrow"));
	cache.select_arrow = get_theme_icon(SNAME("select_arrow"));
	cache.updown = get_theme_icon(SNAME("updown"));

	cache.custom_button = get_theme_stylebox(SNAME("custom_button"));
	cache.custom_button_hover = get_theme_stylebox(SNAME("custom_button_hover"));
	cache.custom_button_pressed = get_theme_stylebox(SNAME("custom_button_pressed"));
	cache.custom_button_font_highlight = get_theme_color(SNAME("custom_button_font_highlight"));

	cache.font_color = get_theme_color(SNAME("font_color"));
	cache.font_selected_color = get_theme_color(SNAME("font_selected_color"));
	cache.drop_position_color = get_theme_color(SNAME("drop_position_color"));
	cache.hseparation = get_theme_constant(SNAME("h_separation"));
	cache.vseparation = get_theme_constant(SNAME("v_separation"));
	cache.item_margin = get_theme_constant(SNAME("item_margin"));
	cache.button_margin = get_theme_constant(SNAME("button_margin"));

	cache.font_outline_color = get_theme_color(SNAME("font_outline_color"));
	cache.font_outline_size = get_theme_constant(SNAME("outline_size"));

	cache.draw_guides = get_theme_constant(SNAME("draw_guides"));
	cache.guide_color = get_theme_color(SNAME("guide_color"));
	cache.draw_relationship_lines = get_theme_constant(SNAME("draw_relationship_lines"));
	cache.relationship_line_width = get_theme_constant(SNAME("relationship_line_width"));
	cache.parent_hl_line_width = get_theme_constant(SNAME("parent_hl_line_width"));
	cache.children_hl_line_width = get_theme_constant(SNAME("children_hl_line_width"));
	cache.parent_hl_line_margin = get_theme_constant(SNAME("parent_hl_line_margin"));
	cache.relationship_line_color = get_theme_color(SNAME("relationship_line_color"));
	cache.parent_hl_line_color = get_theme_color(SNAME("parent_hl_line_color"));
	cache.children_hl_line_color = get_theme_color(SNAME("children_hl_line_color"));

	cache.scroll_border = get_theme_constant(SNAME("scroll_border"));
	cache.scroll_speed = get_theme_constant(SNAME("scroll_speed"));

	cache.title_button = get_theme_stylebox(SNAME("title_button_normal"));
	cache.title_button_pressed = get_theme_stylebox(SNAME("title_button_pressed"));
	cache.title_button_hover = get_theme_stylebox(SNAME("title_button_hover"));
	cache.title_button_color = get_theme_color(SNAME("title_button_color"));

	cache.base_scale = get_theme_default_base_scale();

	v_scroll->set_custom_step(cache.font->get_height(cache.font_size));
}

void Tree2::update_scrollbars() {
	Size2 size = get_size();
	int tbh;
	if (show_column_titles) {
		tbh = get_title_button_height();
	} else {
		tbh = 0;
	}

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_begin(Point2(size.width - vmin.width, cache.bg->get_margin(SIDE_TOP)));
	v_scroll->set_end(Point2(size.width, size.height - cache.bg->get_margin(SIDE_TOP) - cache.bg->get_margin(SIDE_BOTTOM)));

	h_scroll->set_begin(Point2(0, size.height - hmin.height));
	h_scroll->set_end(Point2(size.width - vmin.width, size.height));

	Size2 internal_min_size = get_internal_min_size();

	bool display_vscroll = internal_min_size.height + cache.bg->get_margin(SIDE_TOP) > size.height;
	bool display_hscroll = internal_min_size.width + cache.bg->get_margin(SIDE_LEFT) > size.width;
	for (int i = 0; i < 2; i++) {
		// Check twice, as both values are dependent on each other.
		if (display_hscroll) {
			display_vscroll = internal_min_size.height + cache.bg->get_margin(SIDE_TOP) + hmin.height > size.height;
		}
		if (display_vscroll) {
			display_hscroll = internal_min_size.width + cache.bg->get_margin(SIDE_LEFT) + vmin.width > size.width;
		}
	}

	if (display_vscroll) {
		v_scroll->show();
		v_scroll->set_max(internal_min_size.height);
		v_scroll->set_page(size.height - hmin.height - tbh);
		cache.offset.y = v_scroll->get_value();
	} else {
		v_scroll->hide();
		cache.offset.y = 0;
	}

	if (display_hscroll) {
		h_scroll->show();
		h_scroll->set_max(internal_min_size.width);
		h_scroll->set_page(size.width - vmin.width);
		cache.offset.x = h_scroll->get_value();
	} else {
		h_scroll->hide();
		cache.offset.x = 0;
	}
}

Size2 Tree2::get_internal_min_size() const {
	Size2i size = cache.bg->get_offset();
	if (root) {
		size.height += root->get_height();
	}
	for (int i = 0; i < columns.size(); i++) {
		size.width += get_column_minimum_width(i);
	}

	return size;
}

int Tree2::get_title_button_height() const {
	ERR_FAIL_COND_V(cache.font.is_null() || cache.title_button.is_null(), 0);
	int h = 0;
	if (show_column_titles) {
		for (int i = 0; i < columns.size(); i++) {
			h = MAX(h, columns[i].text_buf->get_size().y + cache.title_button->get_minimum_size().height);
		}
	}
	return h;
}

void Tree2::set_columns(int p_columns) {
	ERR_FAIL_COND(p_columns < 1);
	ERR_FAIL_COND(blocked > 0);
	columns.resize(p_columns);

	// if (root) {
	// 	propagate_set_columns(root);
	// }
	// if (selected_col >= p_columns) {
	// 	selected_col = p_columns - 1;
	// }
	update();
}

void Tree2::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			update_scrollbars();
			RID ci = get_canvas_item();

			Ref<StyleBox> bg = cache.bg;
			Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
			int outline_size = get_theme_constant(SNAME("outline_size"));

			Point2 draw_ofs;
			draw_ofs += bg->get_offset();
			Size2 draw_size = get_size() - bg->get_minimum_size();
			if (h_scroll->is_visible()) {
				draw_size.width -= h_scroll->get_minimum_size().width;
			}

			bg->draw(ci, Rect2(Point2(), get_size()));

			int tbh = get_title_button_height();

			draw_ofs.y += tbh;
			draw_size.y -= tbh;

			cache.rtl = is_layout_rtl();

			if (root && get_size().x > 0 && get_size().y > 0) {
				root->draw(Point2(), draw_ofs, draw_size);
			}

			if (show_column_titles) {
				// Title buttons.
				int ofs2 = cache.bg->get_margin(SIDE_LEFT);
				for (int i = 0; i < columns.size(); i++) {
					Ref<StyleBox> sb = (cache.click_type == Cache::CLICK_TITLE && cache.click_index == i) ? cache.title_button_pressed : ((cache.hover_type == Cache::CLICK_TITLE && cache.hover_index == i) ? cache.title_button_hover : cache.title_button);
					Ref<Font> f = cache.tb_font;
					Rect2 tbrect = Rect2(ofs2 - cache.offset.x, bg->get_margin(SIDE_TOP), get_column_width(i), tbh);
					if (cache.rtl) {
						tbrect.position.x = get_size().width - tbrect.size.x - tbrect.position.x;
					}
					sb->draw(ci, tbrect);
					ofs2 += tbrect.size.width;
					//text
					int clip_w = tbrect.size.width - sb->get_minimum_size().width;
					columns.write[i].text_buf->set_width(clip_w);

					Vector2 text_pos = tbrect.position + Point2i(sb->get_offset().x + (tbrect.size.width - columns[i].text_buf->get_size().x) / 2, (tbrect.size.height - columns[i].text_buf->get_size().y) / 2);
					if (outline_size > 0 && font_outline_color.a > 0) {
						columns[i].text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
					}
					columns[i].text_buf->draw(ci, text_pos, cache.title_button_color);
				}
			}

			// Draw the background focus outline last, so that it is drawn in front of the section headings.
			// Otherwise, section heading backgrounds can appear to be in front of the focus outline when scrolling.
			if (has_focus()) {
				RenderingServer::get_singleton()->canvas_item_add_clip_ignore(ci, true);
				const Ref<StyleBox> bg_focus = get_theme_stylebox(SNAME("bg_focus"));
				bg_focus->draw(ci, Rect2(Point2(), get_size()));
				RenderingServer::get_singleton()->canvas_item_add_clip_ignore(ci, false);
			}
		} break;

		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			update_cache();
			// _update_all();
		} break;
	}
}

void Tree2::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_columns", "columns"), &Tree2::set_columns);
	ClassDB::bind_method(D_METHOD("create_item", "parent", "idx"), &Tree2::create_item, DEFVAL(Variant()), DEFVAL(-1));
}

int Tree2::get_column_width(int p_column) const {
	ERR_FAIL_INDEX_V(p_column, columns.size(), -1);

	int column_width = get_column_minimum_width(p_column);

	if (columns[p_column].expand) {
		int expand_area = get_size().width;

		Ref<StyleBox> bg = cache.bg;

		if (bg.is_valid()) {
			expand_area -= bg->get_margin(SIDE_LEFT) + bg->get_margin(SIDE_RIGHT);
		}

		// if (v_scroll->is_visible_in_tree()) {
		// 	expand_area -= v_scroll->get_combined_minimum_size().width;
		// }

		int expanding_total = 0;

		for (int i = 0; i < columns.size(); i++) {
			expand_area -= get_column_minimum_width(i);
			if (columns[i].expand) {
				expanding_total += columns[i].expand_ratio;
			}
		}

		if (expand_area >= expanding_total && expanding_total > 0) {
			column_width += expand_area * columns[p_column].expand_ratio / expanding_total;
		}
	}

	return column_width;
}

int Tree2::get_column_minimum_width(int p_column) const {
	ERR_FAIL_INDEX_V(p_column, columns.size(), -1);

	// Use the custom minimum width.
	int min_width = columns[p_column].custom_min_width;

	// Check if the visible title of the column is wider.
	if (show_column_titles) {
		min_width = MAX(cache.font->get_string_size(columns[p_column].title, HORIZONTAL_ALIGNMENT_LEFT, -1, cache.font_size).width + cache.bg->get_margin(SIDE_LEFT) + cache.bg->get_margin(SIDE_RIGHT), min_width);
	}

	// if (!columns[p_column].clip_content) {
	// 	int depth = 0;
	// 	TreeItem2 *next;
	// 	for (TreeItem2 *item = get_root(); item; item = next) {
	// 		next = item->get_next_visible();
	// 		// Compute the depth in tree.
	// 		if (next && p_column == 0) {
	// 			if (next->get_parent() == item) {
	// 				depth += 1;
	// 			} else {
	// 				TreeItem2 *common_parent = item->get_parent();
	// 				while (common_parent != next->get_parent() && common_parent) {
	// 					common_parent = common_parent->get_parent();
	// 					depth -= 1;
	// 				}
	// 			}
	// 		}

	// 		// Get the item minimum size.
	// 		Size2 item_size = item->get_minimum_size(p_column);
	// 		if (p_column == 0) {
	// 			item_size.width += cache.item_margin * depth;
	// 		} else {
	// 			item_size.width += cache.hseparation;
	// 		}

	// 		// Check if the item is wider.
	// 		min_width = MAX(min_width, item_size.width);
	// 	}
	// }

	return min_width;
}

Tree2::Tree2() {
	columns.resize(1);

	h_scroll = memnew(HScrollBar);
	v_scroll = memnew(VScrollBar);
}
