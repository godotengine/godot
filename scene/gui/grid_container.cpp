/**************************************************************************/
/*  grid_container.cpp                                                    */
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

#include "grid_container.h"

#include "core/object/class_db.h"
#include "core/templates/rb_map.h"
#include "core/templates/rb_set.h"
#include "scene/theme/theme_db.h"

void GridContainer::_resort() {
	RBMap<int, int> col_minw; // Max of min_width of all controls in each col (indexed by col).
	RBMap<int, int> row_minh; // Max of min_height of all controls in each row (indexed by row).
	RBMap<int, int> col_maxw; // Min positive max_width of all controls in each col (indexed by col).
	RBMap<int, int> row_maxh; // Min positive max_height of all controls in each row (indexed by row).
	RBMap<int, int> col_fixed_size; // Final fixed width for non-expanded columns.
	RBMap<int, int> row_fixed_size; // Final fixed height for non-expanded rows.
	RBSet<int> col_expanded; // Columns which have the SIZE_EXPAND flag set.
	RBSet<int> row_expanded; // Rows which have the SIZE_EXPAND flag set.

	// Compute the per-column/per-row data.
	int valid_controls_index = 0;
	Size2i size = Size2i(get_size());
	Size2i combined_max_size = Size2i(get_combined_maximum_size());

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i));
		if (!c) {
			continue;
		}

		int row = valid_controls_index / columns;
		int col = valid_controls_index % columns;
		valid_controls_index++;

		if (is_propagating_maximum_size()) {
			c->set_parent_maximum_size_cache(combined_max_size);
		}
		Size2i ms = c->get_combined_minimum_size();
		Size2 max_size = c->get_combined_maximum_size();
		if (col_minw.has(col)) {
			col_minw[col] = MAX(col_minw[col], ms.width);
		} else {
			col_minw[col] = ms.width;
		}
		if (row_minh.has(row)) {
			row_minh[row] = MAX(row_minh[row], ms.height);
		} else {
			row_minh[row] = ms.height;
		}

		int max_width = int(max_size.width) >= 0 ? int(max_size.width) : (combined_max_size.width >= 0 ? combined_max_size.width : size.width);
		if (col_maxw.has(col)) {
			col_maxw[col] = MAX(col_maxw[col], max_width);
		} else {
			col_maxw[col] = max_width;
		}

		int max_height = int(max_size.height) >= 0 ? int(max_size.height) : (combined_max_size.height >= 0 ? combined_max_size.height : size.height);
		if (row_maxh.has(row)) {
			row_maxh[row] = MAX(row_maxh[row], max_height);
		} else {
			row_maxh[row] = max_height;
		}

		if (c->get_h_size_flags().has_flag(SIZE_EXPAND)) {
			col_expanded.insert(col);
		}
		if (c->get_v_size_flags().has_flag(SIZE_EXPAND)) {
			row_expanded.insert(row);
		}
	}

	for (const KeyValue<int, int> &E : col_maxw) {
		if (col_maxw[E.key] < col_minw[E.key]) {
			col_maxw[E.key] = col_minw[E.key];
		}
	}

	for (const KeyValue<int, int> &E : row_maxh) {
		if (row_maxh[E.key] < row_minh[E.key]) {
			row_maxh[E.key] = row_minh[E.key];
		}
	}

	int max_col = MIN(valid_controls_index, columns);
	int max_row = std::ceil((float)valid_controls_index / (float)columns);

	// Consider all empty columns expanded.
	for (int i = valid_controls_index; i < columns; i++) {
		col_expanded.insert(i);
	}

	// Evaluate the remaining space for expanded columns/rows.
	Size2 remaining_space = get_size();
	for (const KeyValue<int, int> &E : col_minw) {
		if (!col_expanded.has(E.key)) {
			remaining_space.width -= E.value;
		}
	}

	for (const KeyValue<int, int> &E : row_minh) {
		if (!row_expanded.has(E.key)) {
			remaining_space.height -= E.value;
		}
	}
	remaining_space.height -= theme_cache.v_separation * MAX(max_row - 1, 0);
	remaining_space.width -= theme_cache.h_separation * MAX(max_col - 1, 0);

	bool can_fit = false;
	while (!can_fit && col_expanded.size() > 0) {
		// Check if all minwidth constraints are OK if we use the remaining space.
		can_fit = true;
		int max_index = col_expanded.front()->get();
		for (const int &E : col_expanded) {
			if (col_minw[E] > col_minw[max_index]) {
				max_index = E;
			}
			if (can_fit && (remaining_space.width / col_expanded.size()) < col_minw[E]) {
				can_fit = false;
			}
		}

		// If not, the column with maximum minwidth is not expanded.
		if (!can_fit) {
			col_expanded.erase(max_index);
			remaining_space.width -= col_minw[max_index];
			col_fixed_size[max_index] = col_minw[max_index];
		}
	}

	can_fit = false;
	while (!can_fit && row_expanded.size() > 0) {
		// Check if all minheight constraints are OK if we use the remaining space.
		can_fit = true;
		int max_index = row_expanded.front()->get();
		for (const int &E : row_expanded) {
			if (row_minh[E] > row_minh[max_index]) {
				max_index = E;
			}
			if (can_fit && (remaining_space.height / row_expanded.size()) < row_minh[E]) {
				can_fit = false;
			}
		}

		// If not, the row with maximum minheight is not expanded.
		if (!can_fit) {
			row_expanded.erase(max_index);
			remaining_space.height -= row_minh[max_index];
			row_fixed_size[max_index] = row_minh[max_index];
		}
	}

	bool max_fit = false;
	while (!max_fit && col_expanded.size() > 0) {
		max_fit = true;
		int capped_col = -1;
		for (const int &E : col_expanded) {
			if (col_maxw.has(E) && (remaining_space.width / col_expanded.size()) > col_maxw[E]) {
				capped_col = E;
				max_fit = false;
				break;
			}
		}

		if (capped_col >= 0) {
			col_expanded.erase(capped_col);
			remaining_space.width -= col_maxw[capped_col];
			col_fixed_size[capped_col] = col_maxw[capped_col];
		}
	}

	max_fit = false;
	while (!max_fit && row_expanded.size() > 0) {
		max_fit = true;
		int capped_row = -1;
		for (const int &E : row_expanded) {
			if (row_maxh.has(E) && (remaining_space.height / row_expanded.size()) > row_maxh[E]) {
				capped_row = E;
				max_fit = false;
				break;
			}
		}

		if (capped_row >= 0) {
			row_expanded.erase(capped_row);
			remaining_space.height -= row_maxh[capped_row];
			row_fixed_size[capped_row] = row_maxh[capped_row];
		}
	}

	// Finally, fit the nodes.
	int col_remaining_pixel = 0;
	int col_expand = 0;
	if (col_expanded.size() > 0) {
		col_expand = remaining_space.width / col_expanded.size();
		col_remaining_pixel = remaining_space.width - col_expanded.size() * col_expand;
	}

	int row_remaining_pixel = 0;
	int row_expand = 0;
	if (row_expanded.size() > 0) {
		row_expand = remaining_space.height / row_expanded.size();
		row_remaining_pixel = remaining_space.height - row_expanded.size() * row_expand;
	}

	bool rtl = is_layout_rtl();

	int col_ofs = 0;
	int row_ofs = 0;

	// Calculate the index of rows and columns that receive the remaining pixel.
	int col_remaining_pixel_index = 0;
	for (int i = 0; i < max_col; i++) {
		if (col_remaining_pixel == 0) {
			break;
		}
		if (col_expanded.has(i)) {
			col_remaining_pixel_index = i + 1;
			col_remaining_pixel--;
		}
	}
	int row_remaining_pixel_index = 0;
	for (int i = 0; i < max_row; i++) {
		if (row_remaining_pixel == 0) {
			break;
		}
		if (row_expanded.has(i)) {
			row_remaining_pixel_index = i + 1;
			row_remaining_pixel--;
		}
	}

	int accumulated_width = 0;
	int accumulated_height = 0;
	valid_controls_index = 0;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i));
		if (!c) {
			continue;
		}
		int row = valid_controls_index / columns;
		int col = valid_controls_index % columns;
		valid_controls_index++;

		if (col == 0) {
			if (rtl) {
				col_ofs = get_size().width;
			} else {
				col_ofs = 0;
			}
			if (row > 0) {
				row_ofs += (row_expanded.has(row - 1) ? row_expand : row_minh[row - 1]) + theme_cache.v_separation;

				if (row_expanded.has(row - 1) && row - 1 < row_remaining_pixel_index) {
					// Apply the remaining pixel of the previous row.
					row_ofs++;
				}
			}
		}

		Size2 s(
				col_expanded.has(col) ? col_expand : (col_fixed_size.has(col) ? col_fixed_size[col] : col_minw[col]),
				row_expanded.has(row) ? row_expand : (row_fixed_size.has(row) ? row_fixed_size[row] : row_minh[row]));

		// Add the remaining pixel to the expanding columns and rows, starting from left and top.
		if (col_expanded.has(col) && col < col_remaining_pixel_index) {
			s.x++;
		}
		if (row_expanded.has(row) && row < row_remaining_pixel_index) {
			s.y++;
		}

		if (is_propagating_maximum_size()) {
			Size2 ms = combined_max_size;
			if (ms.width >= 0) {
				ms.width -= accumulated_width;
				ms.width = MAX(ms.width, 0);
			}
			if (ms.height >= 0) {
				ms.height -= accumulated_height;
				ms.height = MAX(ms.height, 0);
			}
			c->set_parent_maximum_size_cache(ms);
		}

		if (rtl) {
			Point2 p(col_ofs - s.width, row_ofs);
			fit_child_in_rect(c, Rect2(p, s));
			col_ofs -= s.width + theme_cache.h_separation;
		} else {
			Point2 p(col_ofs, row_ofs);
			fit_child_in_rect(c, Rect2(p, s));
			col_ofs += s.width + theme_cache.h_separation;
		}

		if (col == columns - 1) {
			accumulated_height += s.height + theme_cache.v_separation;
			accumulated_width = 0;
		} else {
			accumulated_width += s.width + theme_cache.h_separation;
		}
	}
}

void GridContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
			update_minimum_size();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			update_minimum_size();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_sort();
		} break;
	}
}

void GridContainer::set_columns(int p_columns) {
	ERR_FAIL_COND(p_columns < 1);

	if (columns == p_columns) {
		return;
	}

	columns = p_columns;
	queue_sort();
	update_minimum_size();
}

int GridContainer::get_columns() const {
	return columns;
}

int GridContainer::get_h_separation() const {
	return theme_cache.h_separation;
}

void GridContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_columns", "columns"), &GridContainer::set_columns);
	ClassDB::bind_method(D_METHOD("get_columns"), &GridContainer::get_columns);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "columns", PROPERTY_HINT_RANGE, "1,1024,1"), "set_columns", "get_columns");

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GridContainer, h_separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GridContainer, v_separation);
}

Size2 GridContainer::get_minimum_size() const {
	RBMap<int, int> col_minw;
	RBMap<int, int> row_minh;

	int max_row = 0;
	int max_col = 0;

	int valid_controls_index = 0;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i), SortableVisibilityMode::VISIBLE);
		if (!c) {
			continue;
		}
		int row = valid_controls_index / columns;
		int col = valid_controls_index % columns;
		valid_controls_index++;

		Size2i ms = c->get_bound_minimum_size();
		if (col_minw.has(col)) {
			col_minw[col] = MAX(col_minw[col], ms.width);
		} else {
			col_minw[col] = ms.width;
		}

		if (row_minh.has(row)) {
			row_minh[row] = MAX(row_minh[row], ms.height);
		} else {
			row_minh[row] = ms.height;
		}
		max_col = MAX(col, max_col);
		max_row = MAX(row, max_row);
	}

	Size2 ms;

	for (const KeyValue<int, int> &E : col_minw) {
		ms.width += E.value;
	}

	for (const KeyValue<int, int> &E : row_minh) {
		ms.height += E.value;
	}

	ms.height += theme_cache.v_separation * max_row;
	ms.width += theme_cache.h_separation * max_col;

	return ms;
}
