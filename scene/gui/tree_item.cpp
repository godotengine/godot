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

#include "tree_item.h"

#include "scene/gui/tree2.h"
#include "scene/gui/tree_item_cell.h"

TreeItem2 *TreeItem2::create_child(int p_idx) {
	TreeItem2 *ti = memnew(TreeItem2(tree));
	if (tree) {
		ti->cells.resize(tree->columns.size());
		tree->update();
	}

	TreeItem2 *l_prev = nullptr;
	TreeItem2 *c = first_child;
	int idx = 0;

	while (c) {
		if (idx++ == p_idx) {
			c->prev = ti;
			ti->next = c;
			break;
		}
		l_prev = c;
		c = c->next;
	}

	if (l_prev) {
		l_prev->next = ti;
		ti->prev = l_prev;
		if (!children_cache.is_empty()) {
			if (ti->next) {
				children_cache.insert(p_idx, ti);
			} else {
				children_cache.append(ti);
			}
		}
	} else {
		first_child = ti;
		if (!children_cache.is_empty()) {
			children_cache.insert(0, ti);
		}
	}

	ti->parent = this;

	return ti;
}

int TreeItem2::draw(const Point2i &p_pos, const Point2 &p_draw_ofs, const Size2 &p_draw_size) {
	if (p_pos.y - tree->cache.offset.y > (p_draw_size.height)) {
		return -1; // Draw no more!
	}

	if (!visible) {
		return 0;
	}

	RID ci = tree->get_canvas_item();

	int htotal = 0;

	int label_h = get_height();
	bool rtl = tree->cache.rtl;

	label_h += tree->cache.vseparation;
	bool skip = (is_root && tree->hide_root);

	if (!skip && (p_pos.y + label_h - tree->cache.offset.y) > 0) {
		int ofs = p_pos.x + ((disable_folding || tree->hide_folding) ? tree->cache.hseparation : tree->cache.item_margin);
		int skip2 = 0;
		for (int i = 0; i < cells.size(); i++) {
			cells.write[i]->draw(p_pos, label_h, p_draw_ofs, skip2, ofs);
		}

		if (!disable_folding && !tree->hide_folding && first_child && get_visible_child_count() > 0) { // Has visible children, draw the guide box.
			Ref<Texture2D> arrow;
			if (collapsed) {
				arrow = tree->cache.arrow_collapsed;
			} else {
				arrow = tree->cache.arrow;
			}

			Point2 apos = p_pos + Point2i(0, (label_h - arrow->get_height()) / 2) - tree->cache.offset + p_draw_ofs;
			apos.x += tree->cache.item_margin - arrow->get_width();

			if (rtl) {
				apos.x = tree->get_size().width - apos.x - arrow->get_width();
			}
			arrow->draw(ci, apos);
		}
	}

	Point2 children_pos = p_pos;

	if (!skip) {
		children_pos.x += tree->cache.item_margin;
		htotal += label_h;
		children_pos.y += htotal;
	}

	if (!collapsed) {
		TreeItem2 *c = first_child;

		int base_ofs = children_pos.y - tree->cache.offset.y + p_draw_ofs.y;
		int prev_ofs = base_ofs;
		int prev_hl_ofs = base_ofs;

		while (c) {
			int child_h = -1;
			if (htotal >= 0) {
				child_h = c->draw(children_pos, p_draw_ofs, p_draw_size);
			}

			// Draw relationship lines.
			// if (tree->cache.draw_relationship_lines > 0 && (!hide_root || c->parent != root) && c->is_visible()) {
			// 	int root_ofs = children_pos.x + ((disable_folding || hide_folding) ? tree->cache.hseparation : tree->cache.item_margin);
			// 	int parent_ofs = p_pos.x + tree->cache.item_margin;
			// 	Point2i root_pos = Point2i(root_ofs, children_pos.y + label_h / 2) - tree->cache.offset + p_draw_ofs;

			// 	if (c->get_visible_child_count() > 0) {
			// 		root_pos -= Point2i(tree->cache.arrow->get_width(), 0);
			// 	}

			// 	float line_width = tree->cache.relationship_line_width * Math::round(tree->cache.base_scale);
			// 	float parent_line_width = tree->cache.parent_hl_line_width * Math::round(tree->cache.base_scale);
			// 	float children_line_width = tree->cache.children_hl_line_width * Math::round(tree->cache.base_scale);

			// 	Point2i parent_pos = Point2i(parent_ofs - tree->cache.arrow->get_width() / 2, p_pos.y + label_h / 2 + tree->cache.arrow->get_height() / 2) - tree->cache.offset + p_draw_ofs;

			// 	int more_prev_ofs = 0;

			// 	if (root_pos.y + line_width >= 0) {
			// 		if (rtl) {
			// 			root_pos.x = get_size().width - root_pos.x;
			// 			parent_pos.x = get_size().width - parent_pos.x;
			// 		}

			// 		// Order of parts on this bend: the horizontal line first, then the vertical line.
			// 		if (_is_branch_selected(c)) {
			// 			// If this item or one of its children is selected, we draw the line using parent highlight style.
			// 			if (htotal >= 0) {
			// 				RenderingServer::get_singleton()->canvas_item_add_line(ci, root_pos, Point2i(parent_pos.x + Math::floor(parent_line_width / 2), root_pos.y), tree->cache.parent_hl_line_color, parent_line_width);
			// 			}
			// 			RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2i(parent_pos.x, root_pos.y + Math::floor(parent_line_width / 2)), Point2i(parent_pos.x, prev_hl_ofs), tree->cache.parent_hl_line_color, parent_line_width);

			// 			more_prev_ofs = tree->cache.parent_hl_line_margin;
			// 			prev_hl_ofs = root_pos.y + Math::floor(parent_line_width / 2);
			// 		} else if (is_selected(0)) {
			// 			// If parent item is selected (but this item is not), we draw the line using children highlight style.
			// 			// Siblings of the selected branch can be drawn with a slight offset and their vertical line must appear as highlighted.
			// 			if (_is_sibling_branch_selected(c)) {
			// 				if (htotal >= 0) {
			// 					RenderingServer::get_singleton()->canvas_item_add_line(ci, root_pos, Point2i(parent_pos.x + Math::floor(parent_line_width / 2), root_pos.y), tree->cache.children_hl_line_color, children_line_width);
			// 				}
			// 				RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2i(parent_pos.x, root_pos.y + Math::floor(parent_line_width / 2)), Point2i(parent_pos.x, prev_hl_ofs), tree->cache.parent_hl_line_color, parent_line_width);

			// 				prev_hl_ofs = root_pos.y + Math::floor(parent_line_width / 2);
			// 			} else {
			// 				if (htotal >= 0) {
			// 					RenderingServer::get_singleton()->canvas_item_add_line(ci, root_pos, Point2i(parent_pos.x + Math::floor(children_line_width / 2), root_pos.y), tree->cache.children_hl_line_color, children_line_width);
			// 				}
			// 				RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2i(parent_pos.x, root_pos.y + Math::floor(children_line_width / 2)), Point2i(parent_pos.x, prev_ofs + Math::floor(children_line_width / 2)), tree->cache.children_hl_line_color, children_line_width);
			// 			}
			// 		} else {
			// 			// If nothing of the above is true, we draw the line using normal style.
			// 			// Siblings of the selected branch can be drawn with a slight offset and their vertical line must appear as highlighted.
			// 			if (_is_sibling_branch_selected(c)) {
			// 				if (htotal >= 0) {
			// 					RenderingServer::get_singleton()->canvas_item_add_line(ci, root_pos, Point2i(parent_pos.x + tree->cache.parent_hl_line_margin, root_pos.y), tree->cache.relationship_line_color, line_width);
			// 				}
			// 				RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2i(parent_pos.x, root_pos.y + Math::floor(parent_line_width / 2)), Point2i(parent_pos.x, prev_hl_ofs), tree->cache.parent_hl_line_color, parent_line_width);

			// 				prev_hl_ofs = root_pos.y + Math::floor(parent_line_width / 2);
			// 			} else {
			// 				if (htotal >= 0) {
			// 					RenderingServer::get_singleton()->canvas_item_add_line(ci, root_pos, Point2i(parent_pos.x + Math::floor(line_width / 2), root_pos.y), tree->cache.relationship_line_color, line_width);
			// 				}
			// 				RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2i(parent_pos.x, root_pos.y + Math::floor(line_width / 2)), Point2i(parent_pos.x, prev_ofs + Math::floor(line_width / 2)), tree->cache.relationship_line_color, line_width);
			// 			}
			// 		}
			// 	}

			// 	prev_ofs = root_pos.y + more_prev_ofs;
			// }

			if (child_h < 0) {
				if (htotal == -1) {
					break; // Last loop done, stop.
				}

				if (tree->cache.draw_relationship_lines == 0) {
					return -1; // No need to draw anymore, full stop.
				}

				htotal = -1;
				children_pos.y = tree->cache.offset.y + p_draw_size.height;
			} else {
				htotal += child_h;
				children_pos.y += child_h;
			}
			c = c->next;
		}
	}

	return htotal;
}

int TreeItem2::get_height() {
	if ((is_root && tree->hide_root) || !visible) {
		return 0;
	}

	int height = 0;

	for (int i = 0; i < cells.size(); i++) {
		height = MAX(height, cells[i]->get_height());

	// 	switch (cells[i]->mode) {
	// 		case TreeItem::CELL_MODE_CHECK: {
	// 			int check_icon_h = tree->cache.checked->get_height();
	// 			if (height < check_icon_h) {
	// 				height = check_icon_h;
	// 			}
	// 			[[fallthrough]];
	// 		}
	// 		case TreeItem::CELL_MODE_STRING:
	// 		case TreeItem::CELL_MODE_CUSTOM:
	// 		case TreeItem::CELL_MODE_ICON: {
	// 			Ref<Texture2D> icon = cells[i]->icon;
	// 			if (!icon.is_null()) {
	// 				Size2i s = cells[i]->get_icon_size();
	// 				if (cells[i]->icon_max_w > 0 && s.width > cells[i]->icon_max_w) {
	// 					s.height = s.height * cells[i]->icon_max_w / s.width;
	// 				}
	// 				if (s.height > height) {
	// 					height = s.height;
	// 				}
	// 			}
	// 			if (cells[i]->mode == TreeItem::CELL_MODE_CUSTOM && cells[i]->custom_button) {
	// 				height += tree->cache.custom_button->get_minimum_size().height;
	// 			}

	// 		} break;
	// 		default: {
	// 		}
	// 	}
	}
	height = MAX(height, custom_min_height);
	height += tree->cache.vseparation;

	return height;
}

void TreeItem2::refresh_cells() {
	for (int i = cells.size(); i < tree->columns.size(); i++) {
		TreeItemCell *cell = memnew(TreeItemCellText(this, i));
		cells.append(cell);
	}
	for (int i = cells.size(); i > tree->columns.size(); i--) {
		memfree(cells[i - 1]);
	}
	cells.resize(tree->columns.size());
}

void TreeItem2::create_children_cache() {
	if (children_cache.is_empty()) {
		TreeItem2 *c = first_child;
		while (c) {
			children_cache.append(c);
			c = c->next;
		}
	}
}

int TreeItem2::get_custom_minimum_height() const {
	return custom_min_height;
}

TreeItemCell *TreeItem2::get_cell(int p_idx) const {
	return cells[p_idx];
}

int TreeItem2::get_visible_child_count() {
	create_children_cache();
	int visible_count = 0;
	for (const TreeItem2 *E : children_cache) {
		if (E->visible) {
			visible_count += 1;
		}
	}
	return visible_count;
}

void TreeItem2::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_cell", "idx"), &TreeItem2::get_cell);
}

TreeItem2::TreeItem2(Tree2 *p_tree) {
	tree = p_tree;
	refresh_cells();
}