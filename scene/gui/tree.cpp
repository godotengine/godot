/*************************************************************************/
/*  tree.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "tree.h"
#include <limits.h>

#include "math_funcs.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "project_settings.h"
#include "scene/main/viewport.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

void TreeItem::move_to_top() {

	if (!parent || parent->childs == this)
		return; //already on top
	TreeItem *prev = get_prev();
	prev->next = next;
	next = parent->childs;
	parent->childs = this;
}

void TreeItem::move_to_bottom() {
	if (!parent || !next)
		return;

	TreeItem *prev = get_prev();
	TreeItem *last = next;
	while (last->next)
		last = last->next;

	if (prev) {
		prev->next = next;
	} else {
		parent->childs = next;
	}
	last->next = this;
	next = NULL;
}

Size2 TreeItem::Cell::get_icon_size() const {

	if (icon.is_null())
		return Size2();
	if (icon_region == Rect2i())
		return icon->get_size();
	else
		return icon_region.size;
}

void TreeItem::Cell::draw_icon(const RID &p_where, const Point2 &p_pos, const Size2 &p_size, const Color &p_color) const {

	if (icon.is_null())
		return;

	Size2i dsize = (p_size == Size2()) ? icon->get_size() : p_size;

	if (icon_region == Rect2i()) {

		icon->draw_rect_region(p_where, Rect2(p_pos, dsize), Rect2(Point2(), icon->get_size()), p_color);
	} else {

		icon->draw_rect_region(p_where, Rect2(p_pos, dsize), icon_region, p_color);
	}
}

void TreeItem::_changed_notify(int p_cell) {

	tree->item_changed(p_cell, this);
}

void TreeItem::_changed_notify() {

	tree->item_changed(-1, this);
}

void TreeItem::_cell_selected(int p_cell) {

	tree->item_selected(p_cell, this);
}

void TreeItem::_cell_deselected(int p_cell) {

	tree->item_deselected(p_cell, this);
}

/* cell mode */
void TreeItem::set_cell_mode(int p_column, TreeCellMode p_mode) {

	ERR_FAIL_INDEX(p_column, cells.size());
	Cell &c = cells[p_column];
	c.mode = p_mode;
	c.min = 0;
	c.max = 100;
	c.step = 1;
	c.val = 0;
	c.checked = false;
	c.icon = Ref<Texture>();
	c.text = "";
	c.icon_max_w = 0;
	_changed_notify(p_column);
}

TreeItem::TreeCellMode TreeItem::get_cell_mode(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), TreeItem::CELL_MODE_STRING);
	return cells[p_column].mode;
}

/* check mode */
void TreeItem::set_checked(int p_column, bool p_checked) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].checked = p_checked;
	_changed_notify(p_column);
}

bool TreeItem::is_checked(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].checked;
}

void TreeItem::set_text(int p_column, String p_text) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].text = p_text;

	if (cells[p_column].mode == TreeItem::CELL_MODE_RANGE || cells[p_column].mode == TreeItem::CELL_MODE_RANGE_EXPRESSION) {

		Vector<String> strings = p_text.split(",");
		cells[p_column].min = INT_MAX;
		cells[p_column].max = INT_MIN;
		for (int i = 0; i < strings.size(); i++) {
			int value = i;
			if (!strings[i].get_slicec(':', 1).empty()) {
				value = strings[i].get_slicec(':', 1).to_int();
			}
			cells[p_column].min = MIN(cells[p_column].min, value);
			cells[p_column].max = MAX(cells[p_column].max, value);
		}
		cells[p_column].step = 0;
	}
	_changed_notify(p_column);
}

String TreeItem::get_text(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), "");
	return cells[p_column].text;
}

void TreeItem::set_suffix(int p_column, String p_suffix) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].suffix = p_suffix;

	_changed_notify(p_column);
}

String TreeItem::get_suffix(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), "");
	return cells[p_column].suffix;
}

void TreeItem::set_icon(int p_column, const Ref<Texture> &p_icon) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].icon = p_icon;
	_changed_notify(p_column);
}

Ref<Texture> TreeItem::get_icon(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), Ref<Texture>());
	return cells[p_column].icon;
}

void TreeItem::set_icon_region(int p_column, const Rect2 &p_icon_region) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].icon_region = p_icon_region;
	_changed_notify(p_column);
}

Rect2 TreeItem::get_icon_region(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), Rect2());
	return cells[p_column].icon_region;
}

void TreeItem::set_icon_color(int p_column, const Color &p_icon_color) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].icon_color = p_icon_color;
	_changed_notify(p_column);
}

Color TreeItem::get_icon_color(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), Color());
	return cells[p_column].icon_color;
}

void TreeItem::set_icon_max_width(int p_column, int p_max) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].icon_max_w = p_max;
	_changed_notify(p_column);
}

int TreeItem::get_icon_max_width(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), 0);
	return cells[p_column].icon_max_w;
}

/* range works for mode number or mode combo */
void TreeItem::set_range(int p_column, double p_value) {

	ERR_FAIL_INDEX(p_column, cells.size());
	if (cells[p_column].step > 0)
		p_value = Math::stepify(p_value, cells[p_column].step);
	if (p_value < cells[p_column].min)
		p_value = cells[p_column].min;
	if (p_value > cells[p_column].max)
		p_value = cells[p_column].max;

	cells[p_column].val = p_value;
	_changed_notify(p_column);
}

double TreeItem::get_range(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), 0);
	return cells[p_column].val;
}

bool TreeItem::is_range_exponential(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].expr;
}
void TreeItem::set_range_config(int p_column, double p_min, double p_max, double p_step, bool p_exp) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].min = p_min;
	cells[p_column].max = p_max;
	cells[p_column].step = p_step;
	cells[p_column].expr = p_exp;
	_changed_notify(p_column);
}

void TreeItem::get_range_config(int p_column, double &r_min, double &r_max, double &r_step) const {

	ERR_FAIL_INDEX(p_column, cells.size());
	r_min = cells[p_column].min;
	r_max = cells[p_column].max;
	r_step = cells[p_column].step;
}

void TreeItem::set_metadata(int p_column, const Variant &p_meta) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].meta = p_meta;
}

Variant TreeItem::get_metadata(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), Variant());

	return cells[p_column].meta;
}

void TreeItem::set_custom_draw(int p_column, Object *p_object, const StringName &p_callback) {

	ERR_FAIL_INDEX(p_column, cells.size());
	ERR_FAIL_NULL(p_object);

	cells[p_column].custom_draw_obj = p_object->get_instance_id();
	cells[p_column].custom_draw_callback = p_callback;
}

void TreeItem::set_collapsed(bool p_collapsed) {

	if (collapsed == p_collapsed)
		return;
	collapsed = p_collapsed;
	TreeItem *ci = tree->selected_item;
	if (ci) {

		while (ci && ci != this) {

			ci = ci->parent;
		}
		if (ci) { // collapsing cursor/selectd, move it!

			if (tree->select_mode == Tree::SELECT_MULTI) {

				tree->selected_item = this;
				emit_signal("cell_selected");
			} else {

				select(tree->selected_col);
			}

			tree->update();
		}
	}

	_changed_notify();
	if (tree)
		tree->emit_signal("item_collapsed", this);
}

bool TreeItem::is_collapsed() {

	return collapsed;
}

void TreeItem::set_custom_minimum_height(int p_height) {
	custom_min_height = p_height;
	_changed_notify();
}

int TreeItem::get_custom_minimum_height() const {
	return custom_min_height;
}

TreeItem *TreeItem::get_next() {

	return next;
}

TreeItem *TreeItem::get_prev() {

	if (!parent || parent->childs == this)
		return NULL;

	TreeItem *prev = parent->childs;
	while (prev && prev->next != this)
		prev = prev->next;

	return prev;
}

TreeItem *TreeItem::get_parent() {

	return parent;
}

TreeItem *TreeItem::get_children() {

	return childs;
}

TreeItem *TreeItem::get_prev_visible() {

	TreeItem *current = this;

	TreeItem *prev = current->get_prev();

	if (!prev) {

		current = current->parent;
		if (!current || (current == tree->root && tree->hide_root))
			return NULL;
	} else {

		current = prev;
		while (!current->collapsed && current->childs) {
			//go to the very end

			current = current->childs;
			while (current->next)
				current = current->next;
		}
	}

	return current;
}

TreeItem *TreeItem::get_next_visible() {

	TreeItem *current = this;

	if (!current->collapsed && current->childs) {

		current = current->childs;

	} else if (current->next) {

		current = current->next;
	} else {

		while (current && !current->next) {

			current = current->parent;
		}

		if (current == NULL)
			return NULL;
		else
			current = current->next;
	}

	return current;
}

void TreeItem::remove_child(TreeItem *p_item) {

	ERR_FAIL_NULL(p_item);
	TreeItem **c = &childs;

	while (*c) {

		if ((*c) == p_item) {

			TreeItem *aux = *c;

			*c = (*c)->next;

			aux->parent = NULL;
			return;
		}

		c = &(*c)->next;
	}

	ERR_FAIL();
}

void TreeItem::set_selectable(int p_column, bool p_selectable) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].selectable = p_selectable;
}

bool TreeItem::is_selectable(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].selectable;
}

bool TreeItem::is_selected(int p_column) {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].selectable && cells[p_column].selected;
}

void TreeItem::set_as_cursor(int p_column) {

	ERR_FAIL_INDEX(p_column, cells.size());
	if (!tree)
		return;
	if (tree->select_mode != Tree::SELECT_MULTI)
		return;
	tree->selected_item = this;
	tree->selected_col = p_column;
	tree->update();
}

void TreeItem::select(int p_column) {

	ERR_FAIL_INDEX(p_column, cells.size());
	_cell_selected(p_column);
}

void TreeItem::deselect(int p_column) {

	ERR_FAIL_INDEX(p_column, cells.size());
	_cell_deselected(p_column);
}

void TreeItem::add_button(int p_column, const Ref<Texture> &p_button, int p_id, bool p_disabled, const String &p_tooltip) {

	ERR_FAIL_INDEX(p_column, cells.size());
	ERR_FAIL_COND(!p_button.is_valid());
	TreeItem::Cell::Button button;
	button.texture = p_button;
	if (p_id < 0)
		p_id = cells[p_column].buttons.size();
	button.id = p_id;
	button.disabled = p_disabled;
	button.tooltip = p_tooltip;
	cells[p_column].buttons.push_back(button);
	_changed_notify(p_column);
}

int TreeItem::get_button_count(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), -1);
	return cells[p_column].buttons.size();
}
Ref<Texture> TreeItem::get_button(int p_column, int p_idx) const {
	ERR_FAIL_INDEX_V(p_column, cells.size(), Ref<Texture>());
	ERR_FAIL_INDEX_V(p_idx, cells[p_column].buttons.size(), Ref<Texture>());
	return cells[p_column].buttons[p_idx].texture;
}
int TreeItem::get_button_id(int p_column, int p_idx) const {
	ERR_FAIL_INDEX_V(p_column, cells.size(), -1);
	ERR_FAIL_INDEX_V(p_idx, cells[p_column].buttons.size(), -1);
	return cells[p_column].buttons[p_idx].id;
}
void TreeItem::erase_button(int p_column, int p_idx) {

	ERR_FAIL_INDEX(p_column, cells.size());
	ERR_FAIL_INDEX(p_idx, cells[p_column].buttons.size());
	cells[p_column].buttons.remove(p_idx);
	_changed_notify(p_column);
}

int TreeItem::get_button_by_id(int p_column, int p_id) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), -1);
	for (int i = 0; i < cells[p_column].buttons.size(); i++) {

		if (cells[p_column].buttons[i].id == p_id)
			return i;
	}

	return -1;
}

bool TreeItem::is_button_disabled(int p_column, int p_idx) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	ERR_FAIL_INDEX_V(p_idx, cells[p_column].buttons.size(), false);

	return cells[p_column].buttons[p_idx].disabled;
}
void TreeItem::set_button(int p_column, int p_idx, const Ref<Texture> &p_button) {

	ERR_FAIL_COND(p_button.is_null());
	ERR_FAIL_INDEX(p_column, cells.size());
	ERR_FAIL_INDEX(p_idx, cells[p_column].buttons.size());
	cells[p_column].buttons[p_idx].texture = p_button;
	_changed_notify(p_column);
}

void TreeItem::set_button_color(int p_column, int p_idx, const Color &p_color) {

	ERR_FAIL_INDEX(p_column, cells.size());
	ERR_FAIL_INDEX(p_idx, cells[p_column].buttons.size());
	cells[p_column].buttons[p_idx].color = p_color;
	_changed_notify(p_column);
}

void TreeItem::set_editable(int p_column, bool p_editable) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].editable = p_editable;
	_changed_notify(p_column);
}

bool TreeItem::is_editable(int p_column) {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].editable;
}

void TreeItem::set_custom_color(int p_column, const Color &p_color) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].custom_color = true;
	cells[p_column].color = p_color;
	_changed_notify(p_column);
}
Color TreeItem::get_custom_color(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), Color());
	if (!cells[p_column].custom_color)
		return Color();
	return cells[p_column].color;
}
void TreeItem::clear_custom_color(int p_column) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].custom_color = false;
	cells[p_column].color = Color();
	_changed_notify(p_column);
}

void TreeItem::set_tooltip(int p_column, const String &p_tooltip) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].tooltip = p_tooltip;
}

String TreeItem::get_tooltip(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), "");
	return cells[p_column].tooltip;
}

void TreeItem::set_custom_bg_color(int p_column, const Color &p_color, bool p_bg_outline) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].custom_bg_color = true;
	cells[p_column].custom_bg_outline = p_bg_outline;
	cells[p_column].bg_color = p_color;
	_changed_notify(p_column);
}

void TreeItem::clear_custom_bg_color(int p_column) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].custom_bg_color = false;
	cells[p_column].bg_color = Color();
	_changed_notify(p_column);
}

Color TreeItem::get_custom_bg_color(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), Color());
	if (!cells[p_column].custom_bg_color)
		return Color();
	return cells[p_column].bg_color;
}

void TreeItem::set_custom_as_button(int p_column, bool p_button) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].custom_button = p_button;
}

bool TreeItem::is_custom_set_as_button(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].custom_button;
}

void TreeItem::set_text_align(int p_column, TextAlign p_align) {
	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].text_align = p_align;
	_changed_notify(p_column);
}

TreeItem::TextAlign TreeItem::get_text_align(int p_column) const {
	ERR_FAIL_INDEX_V(p_column, cells.size(), ALIGN_LEFT);
	return cells[p_column].text_align;
}

void TreeItem::set_expand_right(int p_column, bool p_enable) {

	ERR_FAIL_INDEX(p_column, cells.size());
	cells[p_column].expand_right = p_enable;
	_changed_notify(p_column);
}

bool TreeItem::get_expand_right(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, cells.size(), false);
	return cells[p_column].expand_right;
}

void TreeItem::set_disable_folding(bool p_disable) {

	disable_folding = p_disable;
	_changed_notify(0);
}

bool TreeItem::is_folding_disabled() const {
	return disable_folding;
}

void TreeItem::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_cell_mode", "column", "mode"), &TreeItem::set_cell_mode);
	ClassDB::bind_method(D_METHOD("get_cell_mode", "column"), &TreeItem::get_cell_mode);

	ClassDB::bind_method(D_METHOD("set_checked", "column", "checked"), &TreeItem::set_checked);
	ClassDB::bind_method(D_METHOD("is_checked", "column"), &TreeItem::is_checked);

	ClassDB::bind_method(D_METHOD("set_text", "column", "text"), &TreeItem::set_text);
	ClassDB::bind_method(D_METHOD("get_text", "column"), &TreeItem::get_text);

	ClassDB::bind_method(D_METHOD("set_icon", "column", "texture"), &TreeItem::set_icon);
	ClassDB::bind_method(D_METHOD("get_icon", "column"), &TreeItem::get_icon);

	ClassDB::bind_method(D_METHOD("set_icon_region", "column", "region"), &TreeItem::set_icon_region);
	ClassDB::bind_method(D_METHOD("get_icon_region", "column"), &TreeItem::get_icon_region);

	ClassDB::bind_method(D_METHOD("set_icon_max_width", "column", "width"), &TreeItem::set_icon_max_width);
	ClassDB::bind_method(D_METHOD("get_icon_max_width", "column"), &TreeItem::get_icon_max_width);

	ClassDB::bind_method(D_METHOD("set_range", "column", "value"), &TreeItem::set_range);
	ClassDB::bind_method(D_METHOD("get_range", "column"), &TreeItem::get_range);
	ClassDB::bind_method(D_METHOD("set_range_config", "column", "min", "max", "step", "expr"), &TreeItem::set_range_config, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_range_config", "column"), &TreeItem::_get_range_config);

	ClassDB::bind_method(D_METHOD("set_metadata", "column", "meta"), &TreeItem::set_metadata);
	ClassDB::bind_method(D_METHOD("get_metadata", "column"), &TreeItem::get_metadata);

	ClassDB::bind_method(D_METHOD("set_custom_draw", "column", "object", "callback"), &TreeItem::set_custom_draw);

	ClassDB::bind_method(D_METHOD("set_collapsed", "enable"), &TreeItem::set_collapsed);
	ClassDB::bind_method(D_METHOD("is_collapsed"), &TreeItem::is_collapsed);

	ClassDB::bind_method(D_METHOD("set_custom_minimum_height", "height"), &TreeItem::set_custom_minimum_height);
	ClassDB::bind_method(D_METHOD("get_custom_minimum_height"), &TreeItem::get_custom_minimum_height);

	ClassDB::bind_method(D_METHOD("get_next"), &TreeItem::get_next);
	ClassDB::bind_method(D_METHOD("get_prev"), &TreeItem::get_prev);
	ClassDB::bind_method(D_METHOD("get_parent"), &TreeItem::get_parent);
	ClassDB::bind_method(D_METHOD("get_children"), &TreeItem::get_children);

	ClassDB::bind_method(D_METHOD("get_next_visible"), &TreeItem::get_next_visible);
	ClassDB::bind_method(D_METHOD("get_prev_visible"), &TreeItem::get_prev_visible);

	ClassDB::bind_method(D_METHOD("remove_child", "child"), &TreeItem::_remove_child);

	ClassDB::bind_method(D_METHOD("set_selectable", "column", "selectable"), &TreeItem::set_selectable);
	ClassDB::bind_method(D_METHOD("is_selectable", "column"), &TreeItem::is_selectable);

	ClassDB::bind_method(D_METHOD("is_selected", "column"), &TreeItem::is_selected);
	ClassDB::bind_method(D_METHOD("select", "column"), &TreeItem::select);
	ClassDB::bind_method(D_METHOD("deselect", "column"), &TreeItem::deselect);

	ClassDB::bind_method(D_METHOD("set_editable", "column", "enabled"), &TreeItem::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable", "column"), &TreeItem::is_editable);

	ClassDB::bind_method(D_METHOD("set_custom_color", "column", "color"), &TreeItem::set_custom_color);
	ClassDB::bind_method(D_METHOD("clear_custom_color", "column"), &TreeItem::clear_custom_color);

	ClassDB::bind_method(D_METHOD("set_custom_bg_color", "column", "color", "just_outline"), &TreeItem::set_custom_bg_color, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("clear_custom_bg_color", "column"), &TreeItem::clear_custom_bg_color);
	ClassDB::bind_method(D_METHOD("get_custom_bg_color", "column"), &TreeItem::get_custom_bg_color);

	ClassDB::bind_method(D_METHOD("set_custom_as_button", "column", "enable"), &TreeItem::set_custom_as_button);
	ClassDB::bind_method(D_METHOD("is_custom_set_as_button", "column"), &TreeItem::is_custom_set_as_button);

	ClassDB::bind_method(D_METHOD("add_button", "column", "button", "button_idx", "disabled", "tooltip"), &TreeItem::add_button, DEFVAL(-1), DEFVAL(false), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_button_count", "column"), &TreeItem::get_button_count);
	ClassDB::bind_method(D_METHOD("get_button", "column", "button_idx"), &TreeItem::get_button);
	ClassDB::bind_method(D_METHOD("set_button", "column", "button_idx", "button"), &TreeItem::set_button);
	ClassDB::bind_method(D_METHOD("erase_button", "column", "button_idx"), &TreeItem::erase_button);
	ClassDB::bind_method(D_METHOD("is_button_disabled", "column", "button_idx"), &TreeItem::is_button_disabled);

	ClassDB::bind_method(D_METHOD("set_expand_right", "column", "enable"), &TreeItem::set_expand_right);
	ClassDB::bind_method(D_METHOD("get_expand_right", "column"), &TreeItem::get_expand_right);

	ClassDB::bind_method(D_METHOD("set_tooltip", "column", "tooltip"), &TreeItem::set_tooltip);
	ClassDB::bind_method(D_METHOD("get_tooltip", "column"), &TreeItem::get_tooltip);
	ClassDB::bind_method(D_METHOD("set_text_align", "column", "text_align"), &TreeItem::set_text_align);
	ClassDB::bind_method(D_METHOD("get_text_align", "column"), &TreeItem::get_text_align);
	ClassDB::bind_method(D_METHOD("move_to_top"), &TreeItem::move_to_top);
	ClassDB::bind_method(D_METHOD("move_to_bottom"), &TreeItem::move_to_bottom);

	ClassDB::bind_method(D_METHOD("set_disable_folding", "disable"), &TreeItem::set_disable_folding);
	ClassDB::bind_method(D_METHOD("is_folding_disabled"), &TreeItem::is_folding_disabled);

	BIND_ENUM_CONSTANT(CELL_MODE_STRING);
	BIND_ENUM_CONSTANT(CELL_MODE_CHECK);
	BIND_ENUM_CONSTANT(CELL_MODE_RANGE);
	BIND_ENUM_CONSTANT(CELL_MODE_RANGE_EXPRESSION);
	BIND_ENUM_CONSTANT(CELL_MODE_ICON);
	BIND_ENUM_CONSTANT(CELL_MODE_CUSTOM);

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
}

void TreeItem::clear_children() {

	TreeItem *c = childs;
	while (c) {

		TreeItem *aux = c;
		c = c->get_next();
		aux->parent = 0; // so it wont try to recursively autoremove from me in here
		memdelete(aux);
	}

	childs = 0;
};

TreeItem::TreeItem(Tree *p_tree) {

	tree = p_tree;
	collapsed = false;
	disable_folding = false;
	custom_min_height = 0;

	parent = 0; // parent item
	next = 0; // next in list
	childs = 0; //child items
}

TreeItem::~TreeItem() {

	clear_children();

	if (parent)
		parent->remove_child(this);

	if (tree && tree->root == this) {

		tree->root = 0;
	}

	if (tree && tree->popup_edited_item == this) {
		tree->popup_edited_item = NULL;
		tree->pressing_for_editor = false;
	}

	if (tree && tree->cache.hover_item == this) {
		tree->cache.hover_item = NULL;
	}

	if (tree && tree->selected_item == this)
		tree->selected_item = NULL;

	if (tree && tree->drop_mode_over == this)
		tree->drop_mode_over = NULL;

	if (tree && tree->single_select_defer == this)
		tree->single_select_defer = NULL;

	if (tree && tree->edited_item == this) {
		tree->edited_item = NULL;
		tree->pressing_for_editor = false;
	}
}

/**********************************************/
/**********************************************/
/**********************************************/
/**********************************************/
/**********************************************/
/**********************************************/

void Tree::update_cache() {

	cache.font = get_font("font");
	cache.tb_font = get_font("title_button_font");
	cache.bg = get_stylebox("bg");
	cache.selected = get_stylebox("selected");
	cache.selected_focus = get_stylebox("selected_focus");
	cache.cursor = get_stylebox("cursor");
	cache.cursor_unfocus = get_stylebox("cursor_unfocused");
	cache.button_pressed = get_stylebox("button_pressed");

	cache.checked = get_icon("checked");
	cache.unchecked = get_icon("unchecked");
	cache.arrow_collapsed = get_icon("arrow_collapsed");
	cache.arrow = get_icon("arrow");
	cache.select_arrow = get_icon("select_arrow");
	cache.select_option = get_icon("select_option");
	cache.updown = get_icon("updown");

	cache.custom_button = get_stylebox("custom_button");
	cache.custom_button_hover = get_stylebox("custom_button_hover");
	cache.custom_button_pressed = get_stylebox("custom_button_pressed");
	cache.custom_button_font_highlight = get_color("custom_button_font_highlight");

	cache.font_color = get_color("font_color");
	cache.font_color_selected = get_color("font_color_selected");
	cache.guide_color = get_color("guide_color");
	cache.drop_position_color = get_color("drop_position_color");
	cache.hseparation = get_constant("hseparation");
	cache.vseparation = get_constant("vseparation");
	cache.item_margin = get_constant("item_margin");
	cache.button_margin = get_constant("button_margin");
	cache.guide_width = get_constant("guide_width");
	cache.draw_relationship_lines = get_constant("draw_relationship_lines");
	cache.relationship_line_color = get_color("relationship_line_color");
	cache.scroll_border = get_constant("scroll_border");
	cache.scroll_speed = get_constant("scroll_speed");

	cache.title_button = get_stylebox("title_button_normal");
	cache.title_button_pressed = get_stylebox("title_button_pressed");
	cache.title_button_hover = get_stylebox("title_button_hover");
	cache.title_button_color = get_color("title_button_color");

	v_scroll->set_custom_step(cache.font->get_height());
}

int Tree::compute_item_height(TreeItem *p_item) const {

	if (p_item == root && hide_root)
		return 0;

	int height = cache.font->get_height();

	for (int i = 0; i < columns.size(); i++) {

		for (int j = 0; j < p_item->cells[i].buttons.size(); j++) {

			Size2i s; // = cache.button_pressed->get_minimum_size();
			s += p_item->cells[i].buttons[j].texture->get_size();
			if (s.height > height)
				height = s.height;
		}

		switch (p_item->cells[i].mode) {

			case TreeItem::CELL_MODE_CHECK: {

				int check_icon_h = cache.checked->get_height();
				if (height < check_icon_h)
					height = check_icon_h;
			}
			case TreeItem::CELL_MODE_STRING:
			case TreeItem::CELL_MODE_CUSTOM:
			case TreeItem::CELL_MODE_ICON: {

				Ref<Texture> icon = p_item->cells[i].icon;
				if (!icon.is_null()) {

					Size2i s = p_item->cells[i].get_icon_size();
					if (p_item->cells[i].icon_max_w > 0 && s.width > p_item->cells[i].icon_max_w) {
						s.height = s.height * p_item->cells[i].icon_max_w / s.width;
					}
					if (s.height > height)
						height = s.height;
				}
				if (p_item->cells[i].mode == TreeItem::CELL_MODE_CUSTOM && p_item->cells[i].custom_button) {
					height += cache.custom_button->get_minimum_size().height;
				}

			} break;
			default: {}
		}
	}
	int item_min_height = p_item->get_custom_minimum_height();
	if (height < item_min_height)
		height = item_min_height;

	height += cache.vseparation;

	return height;
}

int Tree::get_item_height(TreeItem *p_item) const {

	int height = compute_item_height(p_item);
	height += cache.vseparation;

	if (!p_item->collapsed) { /* if not collapsed, check the childs */

		TreeItem *c = p_item->childs;

		while (c) {

			height += get_item_height(c);

			c = c->next;
		}
	}

	return height;
}

void Tree::draw_item_rect(const TreeItem::Cell &p_cell, const Rect2i &p_rect, const Color &p_color, const Color &p_icon_color) {

	Rect2i rect = p_rect;
	Ref<Font> font = cache.font;
	String text = p_cell.text;
	if (p_cell.suffix != String())
		text += " " + p_cell.suffix;

	int w = 0;
	if (!p_cell.icon.is_null()) {
		Size2i bmsize = p_cell.get_icon_size();

		if (p_cell.icon_max_w > 0 && bmsize.width > p_cell.icon_max_w) {
			bmsize.width = p_cell.icon_max_w;
		}
		w += bmsize.width + cache.hseparation;
	}
	w += font->get_string_size(text).width;

	switch (p_cell.text_align) {
		case TreeItem::ALIGN_LEFT:
			break; //do none
		case TreeItem::ALIGN_CENTER:
			rect.position.x = MAX(0, (rect.size.width - w) / 2);
			break; //do none
		case TreeItem::ALIGN_RIGHT:
			rect.position.x = MAX(0, (rect.size.width - w));
			break; //do none
	}

	RID ci = get_canvas_item();
	if (!p_cell.icon.is_null()) {
		Size2i bmsize = p_cell.get_icon_size();

		if (p_cell.icon_max_w > 0 && bmsize.width > p_cell.icon_max_w) {
			bmsize.height = bmsize.height * p_cell.icon_max_w / bmsize.width;
			bmsize.width = p_cell.icon_max_w;
		}

		p_cell.draw_icon(ci, rect.position + Size2i(0, Math::floor((real_t)(rect.size.y - bmsize.y) / 2)), bmsize, p_icon_color);
		rect.position.x += bmsize.x + cache.hseparation;
		rect.size.x -= bmsize.x + cache.hseparation;
	}

	rect.position.y += Math::floor((rect.size.y - font->get_height()) / 2.0) + font->get_ascent();
	font->draw(ci, rect.position, text, p_color, rect.size.x);
}

int Tree::draw_item(const Point2i &p_pos, const Point2 &p_draw_ofs, const Size2 &p_draw_size, TreeItem *p_item) {

	if (p_pos.y - cache.offset.y > (p_draw_size.height))
		return -1; //draw no more!

	RID ci = get_canvas_item();

	int htotal = 0;

	int label_h = compute_item_height(p_item);

	/* Calculate height of the label part */
	label_h += cache.vseparation;

	/* Draw label, if height fits */

	bool skip = (p_item == root && hide_root);

	if (!skip && (p_pos.y + label_h - cache.offset.y) > 0) {

		//draw separation.
		//if (p_item->get_parent()!=root || !hide_root)

		Ref<Font> font = cache.font;

		int font_ascent = font->get_ascent();

		int ofs = p_pos.x + ((p_item->disable_folding || hide_folding) ? cache.hseparation : cache.item_margin);
		int skip = 0;
		for (int i = 0; i < columns.size(); i++) {

			if (skip) {
				skip--;
				continue;
			}

			int w = get_column_width(i);

			if (i == 0) {

				w -= ofs;

				if (w <= 0) {

					ofs = get_column_width(0);
					continue;
				}
			} else {

				ofs += cache.hseparation;
				w -= cache.hseparation;
			}

			if (p_item->cells[i].expand_right) {

				int plus = 1;
				while (i + plus < columns.size() && !p_item->cells[i + plus].editable && p_item->cells[i + plus].mode == TreeItem::CELL_MODE_STRING && p_item->cells[i + plus].text == "" && p_item->cells[i + plus].icon.is_null()) {
					w += get_column_width(i + plus);
					plus++;
					skip++;
				}
			}

			int bw = 0;
			for (int j = p_item->cells[i].buttons.size() - 1; j >= 0; j--) {
				Ref<Texture> b = p_item->cells[i].buttons[j].texture;
				Size2 s = b->get_size() + cache.button_pressed->get_minimum_size();

				Point2i o = Point2i(ofs + w - s.width, p_pos.y) - cache.offset + p_draw_ofs;

				if (cache.click_type == Cache::CLICK_BUTTON && cache.click_item == p_item && cache.click_column == i && cache.click_index == j && !p_item->cells[i].buttons[j].disabled) {
					//being pressed
					cache.button_pressed->draw(get_canvas_item(), Rect2(o, s));
				}

				o.y += (label_h - s.height) / 2;
				o += cache.button_pressed->get_offset();

				b->draw(ci, o, p_item->cells[i].buttons[j].disabled ? Color(1, 1, 1, 0.5) : p_item->cells[i].buttons[j].color);
				w -= s.width + cache.button_margin;
				bw += s.width + cache.button_margin;
			}

			Rect2i item_rect = Rect2i(Point2i(ofs, p_pos.y) - cache.offset + p_draw_ofs, Size2i(w, label_h));
			Rect2i cell_rect = item_rect;
			if (i != 0) {
				cell_rect.position.x -= cache.hseparation;
				cell_rect.size.x += cache.hseparation;
			}

			VisualServer::get_singleton()->canvas_item_add_line(ci, Point2i(cell_rect.position.x, cell_rect.position.y + cell_rect.size.height), cell_rect.position + cell_rect.size, cache.guide_color, 1);

			if (i == 0) {

				if (p_item->cells[0].selected && select_mode == SELECT_ROW) {
					Rect2i row_rect = Rect2i(Point2i(cache.bg->get_margin(MARGIN_LEFT), item_rect.position.y), Size2i(get_size().width - cache.bg->get_minimum_size().width, item_rect.size.y));
					//Rect2 r = Rect2i(row_rect.pos,row_rect.size);
					//r.grow(cache.selected->get_margin(MARGIN_LEFT));
					if (has_focus())
						cache.selected_focus->draw(ci, row_rect);
					else
						cache.selected->draw(ci, row_rect);
				}
			}

			if (p_item->cells[i].selected && select_mode != SELECT_ROW) {

				Rect2i r(cell_rect.position, cell_rect.size);
				if (p_item->cells[i].text.size() > 0) {
					float icon_width = p_item->cells[i].get_icon_size().width;
					r.position.x += icon_width;
					r.size.x -= icon_width;
				}
				//r.grow(cache.selected->get_margin(MARGIN_LEFT));
				if (has_focus()) {
					cache.selected_focus->draw(ci, r);
					p_item->set_meta("__focus_rect", Rect2(r.position, r.size));
				} else {
					cache.selected->draw(ci, r);
				}
				if (text_editor->is_visible_in_tree()) {
					Vector2 ofs(0, (text_editor->get_size().height - r.size.height) / 2);
					text_editor->set_position(get_global_position() + r.position - ofs);
				}
			}

			if (p_item->cells[i].custom_bg_color) {

				Rect2 r = cell_rect;
				if (i == 0) {
					r.position.x = p_draw_ofs.x;
					r.size.x = w + ofs;
				} else {
					r.position.x -= cache.hseparation;
					r.size.x += cache.hseparation;
				}
				if (p_item->cells[i].custom_bg_outline) {
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, r.size.x, 1), p_item->cells[i].bg_color);
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y + r.size.y - 1, r.size.x, 1), p_item->cells[i].bg_color);
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, 1, r.size.y), p_item->cells[i].bg_color);
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x + r.size.x - 1, r.position.y, 1, r.size.y), p_item->cells[i].bg_color);
				} else {
					VisualServer::get_singleton()->canvas_item_add_rect(ci, r, p_item->cells[i].bg_color);
				}
			}

			if (drop_mode_flags && drop_mode_over == p_item) {

				Rect2 r = cell_rect;

				if (drop_mode_section == -1 || drop_mode_section == 0) {
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, r.size.x, 1), cache.drop_position_color);
				}

				if (drop_mode_section == 0) {
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y, 1, r.size.y), cache.drop_position_color);
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x + r.size.x - 1, r.position.y, 1, r.size.y), cache.drop_position_color);
				}

				if (drop_mode_section == 1 || drop_mode_section == 0) {
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(r.position.x, r.position.y + r.size.y, r.size.x, 1), cache.drop_position_color);
				}
			}

			Color col = p_item->cells[i].custom_color ? p_item->cells[i].color : get_color(p_item->cells[i].selected ? "font_color_selected" : "font_color");
			Color icon_col = p_item->cells[i].icon_color;

			Point2i text_pos = item_rect.position;
			text_pos.y += Math::floor((item_rect.size.y - font->get_height()) / 2) + font_ascent;

			switch (p_item->cells[i].mode) {

				case TreeItem::CELL_MODE_STRING: {

					draw_item_rect(p_item->cells[i], item_rect, col, icon_col);
				} break;
				case TreeItem::CELL_MODE_CHECK: {

					Ref<Texture> checked = cache.checked;
					Ref<Texture> unchecked = cache.unchecked;
					Point2i check_ofs = item_rect.position;
					check_ofs.y += Math::floor((real_t)(item_rect.size.y - checked->get_height()) / 2);

					if (p_item->cells[i].checked) {

						checked->draw(ci, check_ofs, icon_col);
					} else {
						unchecked->draw(ci, check_ofs, icon_col);
					}

					int check_w = checked->get_width() + cache.hseparation;

					text_pos.x += check_w;

					item_rect.size.x -= check_w;
					item_rect.position.x += check_w;

					draw_item_rect(p_item->cells[i], item_rect, col, icon_col);

					//font->draw( ci, text_pos, p_item->cells[i].text, col,item_rect.size.x-check_w );

				} break;
				case TreeItem::CELL_MODE_RANGE:
				case TreeItem::CELL_MODE_RANGE_EXPRESSION: {

					if (p_item->cells[i].text != "") {

						if (!p_item->cells[i].editable)
							break;

						int option = (int)p_item->cells[i].val;

						String s = RTR("(Other)");
						Vector<String> strings = p_item->cells[i].text.split(",");
						for (int i = 0; i < strings.size(); i++) {
							int value = i;
							if (!strings[i].get_slicec(':', 1).empty()) {
								value = strings[i].get_slicec(':', 1).to_int();
							}
							if (option == value) {
								s = strings[i].get_slicec(':', 0);
								break;
							}
						}

						if (p_item->cells[i].suffix != String())
							s += " " + p_item->cells[i].suffix;

						Ref<Texture> downarrow = cache.select_arrow;

						font->draw(ci, text_pos, s, col, item_rect.size.x - downarrow->get_width());

						//?
						Point2i arrow_pos = item_rect.position;
						arrow_pos.x += item_rect.size.x - downarrow->get_width();
						arrow_pos.y += Math::floor(((item_rect.size.y - downarrow->get_height())) / 2.0);

						downarrow->draw(ci, arrow_pos, icon_col);
					} else {

						Ref<Texture> updown = cache.updown;

						String valtext = String::num(p_item->cells[i].val, Math::step_decimals(p_item->cells[i].step));
						//String valtext = rtos( p_item->cells[i].val );

						if (p_item->cells[i].suffix != String())
							valtext += " " + p_item->cells[i].suffix;

						font->draw(ci, text_pos, valtext, col, item_rect.size.x - updown->get_width());

						if (!p_item->cells[i].editable)
							break;

						Point2i updown_pos = item_rect.position;
						updown_pos.x += item_rect.size.x - updown->get_width();
						updown_pos.y += Math::floor(((item_rect.size.y - updown->get_height())) / 2.0);

						updown->draw(ci, updown_pos, icon_col);
					}

				} break;
				case TreeItem::CELL_MODE_ICON: {

					if (p_item->cells[i].icon.is_null())
						break;
					Size2i icon_size = p_item->cells[i].get_icon_size();
					if (p_item->cells[i].icon_max_w > 0 && icon_size.width > p_item->cells[i].icon_max_w) {
						icon_size.height = icon_size.height * p_item->cells[i].icon_max_w / icon_size.width;
						icon_size.width = p_item->cells[i].icon_max_w;
					}

					Point2i icon_ofs = (item_rect.size - icon_size) / 2;
					icon_ofs += item_rect.position;

					draw_texture_rect(p_item->cells[i].icon, Rect2(icon_ofs, icon_size), false, icon_col);
					//p_item->cells[i].icon->draw(ci, icon_ofs);

				} break;
				case TreeItem::CELL_MODE_CUSTOM: {

					//int option = (int)p_item->cells[i].val;

					if (p_item->cells[i].custom_draw_obj) {

						Object *cdo = ObjectDB::get_instance(p_item->cells[i].custom_draw_obj);
						if (cdo)
							cdo->call(p_item->cells[i].custom_draw_callback, p_item, Rect2(item_rect));
					}

					if (!p_item->cells[i].editable) {

						draw_item_rect(p_item->cells[i], item_rect, col, icon_col);
						break;
					}

					Ref<Texture> downarrow = cache.select_arrow;

					Rect2i ir = item_rect;

					Point2i arrow_pos = item_rect.position;
					arrow_pos.x += item_rect.size.x - downarrow->get_width();
					arrow_pos.y += Math::floor(((item_rect.size.y - downarrow->get_height())) / 2.0);
					ir.size.width -= downarrow->get_width();

					if (p_item->cells[i].custom_button) {
						if (cache.hover_item == p_item && cache.hover_cell == i) {
							if (Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT)) {
								draw_style_box(cache.custom_button_pressed, ir);
							} else {
								draw_style_box(cache.custom_button_hover, ir);
								col = cache.custom_button_font_highlight;
							}
						} else {
							draw_style_box(cache.custom_button, ir);
						}
						ir.size -= cache.custom_button->get_minimum_size();
						ir.position += cache.custom_button->get_offset();
					}

					draw_item_rect(p_item->cells[i], ir, col, icon_col);

					downarrow->draw(ci, arrow_pos);

				} break;
			}

			if (i == 0) {

				ofs = get_column_width(0);
			} else {

				ofs += w + bw;
			}

			if (select_mode == SELECT_MULTI && selected_item == p_item && selected_col == i) {

				if (has_focus())
					cache.cursor->draw(ci, cell_rect);
				else
					cache.cursor_unfocus->draw(ci, cell_rect);
			}
		}

		if (!p_item->disable_folding && !hide_folding && p_item->childs) { //has childs, draw the guide box

			Ref<Texture> arrow;

			if (p_item->collapsed) {

				arrow = cache.arrow_collapsed;
			} else {
				arrow = cache.arrow;
			}

			arrow->draw(ci, p_pos + p_draw_ofs + Point2i(0, (label_h - arrow->get_height()) / 2) - cache.offset);
		}
		//separator
		//get_painter()->draw_fill_rect( Point2i(0,pos.y),Size2i(get_size().width,1),color( COLOR_TREE_GRID) );

		//pos=p_pos; //reset pos
	}

	Point2 children_pos = p_pos;

	if (!skip) {
		children_pos.x += cache.item_margin;
		htotal += label_h;
		children_pos.y += htotal;
	}

	if (!p_item->collapsed) { /* if not collapsed, check the childs */

		TreeItem *c = p_item->childs;

		while (c) {

			if (cache.draw_relationship_lines == 1) {
				int root_ofs = children_pos.x + ((p_item->disable_folding || hide_folding) ? cache.hseparation : cache.item_margin);
				int parent_ofs = p_pos.x + ((p_item->disable_folding || hide_folding) ? cache.hseparation : cache.item_margin);
				Point2i root_pos = Point2i(root_ofs, children_pos.y + label_h / 2) - cache.offset + p_draw_ofs;
				if (c->get_children() != NULL)
					root_pos -= Point2i(cache.arrow->get_width(), 0);

				float line_width = 1.0;
#ifdef TOOLS_ENABLED
				line_width *= EDSCALE;
#endif

				Point2i parent_pos = Point2i(parent_ofs - cache.arrow->get_width() / 2, p_pos.y + label_h / 2 + cache.arrow->get_height() / 2) - cache.offset + p_draw_ofs;
				VisualServer::get_singleton()->canvas_item_add_line(ci, root_pos, Point2i(parent_pos.x - Math::floor(line_width / 2), root_pos.y), cache.relationship_line_color, line_width);
				VisualServer::get_singleton()->canvas_item_add_line(ci, Point2i(parent_pos.x, root_pos.y), parent_pos, cache.relationship_line_color, line_width);
			}

			int child_h = draw_item(children_pos, p_draw_ofs, p_draw_size, c);

			if (child_h < 0 && cache.draw_relationship_lines == 0)
				return -1; // break, stop drawing, no need to anymore

			htotal += child_h;
			children_pos.y += child_h;
			c = c->next;
		}
	}

	return htotal;
}

int Tree::_count_selected_items(TreeItem *p_from) const {

	int count = 0;
	for (int i = 0; i < columns.size(); i++) {
		if (p_from->is_selected(i))
			count++;
	}

	if (p_from->get_children()) {
		count += _count_selected_items(p_from->get_children());
	}

	if (p_from->get_next()) {
		count += _count_selected_items(p_from->get_next());
	}

	return count;
}
void Tree::select_single_item(TreeItem *p_selected, TreeItem *p_current, int p_col, TreeItem *p_prev, bool *r_in_range, bool p_force_deselect) {

	TreeItem::Cell &selected_cell = p_selected->cells[p_col];

	bool switched = false;
	if (r_in_range && !*r_in_range && (p_current == p_selected || p_current == p_prev)) {
		*r_in_range = true;
		switched = true;
	}

	bool emitted_row = false;

	for (int i = 0; i < columns.size(); i++) {

		TreeItem::Cell &c = p_current->cells[i];

		if (!c.selectable)
			continue;

		if (select_mode == SELECT_ROW) {

			if (p_selected == p_current && (!c.selected || allow_reselect)) {
				c.selected = true;
				selected_item = p_selected;
				selected_col = 0;
				if (!emitted_row) {
					emit_signal("item_selected");
					emitted_row = true;
				}
				/*
					if (p_col==i)
						p_current->selected_signal.call(p_col);
					*/

			} else if (c.selected) {

				c.selected = false;
				//p_current->deselected_signal.call(p_col);
			}
		} else if (select_mode == SELECT_SINGLE || select_mode == SELECT_MULTI) {

			if (!r_in_range && &selected_cell == &c) {

				if (!selected_cell.selected || allow_reselect) {

					selected_cell.selected = true;

					selected_item = p_selected;
					selected_col = i;

					emit_signal("cell_selected");
					if (select_mode == SELECT_MULTI)
						emit_signal("multi_selected", p_current, i, true);
					else if (select_mode == SELECT_SINGLE)
						emit_signal("item_selected");

				} else if (select_mode == SELECT_MULTI && (selected_item != p_selected || selected_col != i)) {

					selected_item = p_selected;
					selected_col = i;
					emit_signal("cell_selected");
				}
			} else {

				if (r_in_range && *r_in_range && !p_force_deselect) {

					if (!c.selected && c.selectable) {
						c.selected = true;
						emit_signal("multi_selected", p_current, i, true);
					}

				} else if (!r_in_range || p_force_deselect) {
					if (select_mode == SELECT_MULTI && c.selected)
						emit_signal("multi_selected", p_current, i, false);
					c.selected = false;
				}
				//p_current->deselected_signal.call(p_col);
			}
		}
	}

	if (!switched && r_in_range && *r_in_range && (p_current == p_selected || p_current == p_prev)) {
		*r_in_range = false;
	}

	TreeItem *c = p_current->childs;

	while (c) {

		select_single_item(p_selected, c, p_col, p_prev, r_in_range, p_current->is_collapsed() || p_force_deselect);
		c = c->next;
	}
}

Rect2 Tree::search_item_rect(TreeItem *p_from, TreeItem *p_item) {

	return Rect2();
}

void Tree::_range_click_timeout() {

	if (range_item_last && !range_drag_enabled && Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT)) {

		Point2 pos = get_local_mouse_position() - cache.bg->get_offset();
		if (show_column_titles) {
			pos.y -= _get_title_button_height();

			if (pos.y < 0) {
				range_click_timer->stop();
				return;
			}
		}

		click_handled = false;
		Ref<InputEventMouseButton> mb;
		mb.instance();
		;

		blocked++;
		propagate_mouse_event(pos + cache.offset, 0, 0, false, root, BUTTON_LEFT, mb);
		blocked--;

		if (range_click_timer->is_one_shot()) {
			range_click_timer->set_wait_time(0.05);
			range_click_timer->set_one_shot(false);
			range_click_timer->start();
		}

		if (!click_handled)
			range_click_timer->stop();

	} else {
		range_click_timer->stop();
	}
}

int Tree::propagate_mouse_event(const Point2i &p_pos, int x_ofs, int y_ofs, bool p_doubleclick, TreeItem *p_item, int p_button, const Ref<InputEventWithModifiers> &p_mod) {

	int item_h = compute_item_height(p_item) + cache.vseparation;

	bool skip = (p_item == root && hide_root);

	if (!skip && p_pos.y < item_h) {
		// check event!

		if (range_click_timer->get_time_left() > 0 && p_item != range_item_last) {
			return -1;
		}

		if (!p_item->disable_folding && !hide_folding && (p_pos.x >= x_ofs && p_pos.x < (x_ofs + cache.item_margin))) {

			if (p_item->childs)
				p_item->set_collapsed(!p_item->is_collapsed());

			return -1; //handled!
		}

		int x = p_pos.x;
		/* find clicked column */
		int col = -1;
		int col_ofs = 0;
		int col_width = 0;
		for (int i = 0; i < columns.size(); i++) {

			col_width = get_column_width(i);

			if (p_item->cells[i].expand_right) {

				int plus = 1;
				while (i + plus < columns.size() && !p_item->cells[i + plus].editable && p_item->cells[i + plus].mode == TreeItem::CELL_MODE_STRING && p_item->cells[i + plus].text == "" && p_item->cells[i + plus].icon.is_null()) {
					col_width += cache.hseparation;
					col_width += get_column_width(i + plus);
					plus++;
				}
			}

			if (x > col_width) {
				col_ofs += col_width;
				x -= col_width;
				continue;
			}

			col = i;
			break;
		}

		if (col == -1)
			return -1;
		else if (col == 0) {
			int margin = x_ofs + cache.item_margin; //-cache.hseparation;
			//int lm = cache.bg->get_margin(MARGIN_LEFT);
			col_width -= margin;
			col_ofs += margin;
			x -= margin;
		} else {

			col_width -= cache.hseparation;
			x -= cache.hseparation;
		}

		if (!p_item->disable_folding && !hide_folding && !p_item->cells[col].editable && !p_item->cells[col].selectable && p_item->get_children()) {
			p_item->set_collapsed(!p_item->is_collapsed());
			return -1; //collapse/uncollapse because nothing can be done with item
		}

		TreeItem::Cell &c = p_item->cells[col];

		bool already_selected = c.selected;
		bool already_cursor = (p_item == selected_item) && col == selected_col;

		for (int j = c.buttons.size() - 1; j >= 0; j--) {
			Ref<Texture> b = c.buttons[j].texture;
			int w = b->get_size().width + cache.button_pressed->get_minimum_size().width;

			if (x > col_width - w) {
				if (c.buttons[j].disabled) {
					pressed_button = -1;
					cache.click_type = Cache::CLICK_NONE;
					return -1;
				}
				pressed_button = j;
				cache.click_type = Cache::CLICK_BUTTON;
				cache.click_index = j;
				cache.click_id = c.buttons[j].id;
				cache.click_item = p_item;
				cache.click_column = col;
				cache.click_pos = get_global_mouse_position() - get_global_position();
				update();
				//emit_signal("button_pressed");
				return -1;
			}
			col_width -= w + cache.button_margin;
		}

		if (p_button == BUTTON_LEFT || (p_button == BUTTON_RIGHT && allow_rmb_select)) {
			/* process selection */

			if (p_doubleclick && (!c.editable || c.mode == TreeItem::CELL_MODE_CUSTOM || c.mode == TreeItem::CELL_MODE_ICON /*|| c.mode==TreeItem::CELL_MODE_CHECK*/)) { //it' s confusing for check

				emit_signal("item_activated");
				incr_search.clear();
				return -1;
			}

			if (select_mode == SELECT_MULTI && p_mod->get_command() && c.selectable) {

				if (!c.selected || p_button == BUTTON_RIGHT) {

					p_item->select(col);
					emit_signal("multi_selected", p_item, col, true);
					if (p_button == BUTTON_RIGHT) {
						emit_signal("item_rmb_selected", get_local_mouse_position());
					}

					//p_item->selected_signal.call(col);
				} else {

					p_item->deselect(col);
					emit_signal("multi_selected", p_item, col, false);
					//p_item->deselected_signal.call(col);
				}

			} else {

				if (c.selectable) {

					if (select_mode == SELECT_MULTI && p_mod->get_shift() && selected_item && selected_item != p_item) {

						bool inrange = false;

						select_single_item(p_item, root, col, selected_item, &inrange);
						if (p_button == BUTTON_RIGHT) {
							emit_signal("item_rmb_selected", get_local_mouse_position());
						}
					} else {

						int icount = _count_selected_items(root);

						if (select_mode == SELECT_MULTI && icount > 1 && p_button != BUTTON_RIGHT) {
							single_select_defer = p_item;
							single_select_defer_column = col;
						} else {

							if (p_button != BUTTON_RIGHT || !c.selected) {
								select_single_item(p_item, root, col);
							}

							if (p_button == BUTTON_RIGHT) {
								emit_signal("item_rmb_selected", get_local_mouse_position());
							}
						}
					}

					/*
					if (!c.selected && select_mode==SELECT_MULTI) {
						emit_signal("multi_selected",p_item,col,true);
					}
					*/
					update();
				}
			}
		}

		if (!c.editable)
			return -1; // if cell is not editable, don't bother

		/* editing */

		bool bring_up_editor = allow_reselect ? (c.selected && already_selected) : c.selected;
		String editor_text = c.text;

		switch (c.mode) {

			case TreeItem::CELL_MODE_STRING: {
				//nothing in particular

				if (select_mode == SELECT_MULTI && (get_tree()->get_last_event_id() == focus_in_id || !already_cursor)) {
					bring_up_editor = false;
				}

			} break;
			case TreeItem::CELL_MODE_CHECK: {

				bring_up_editor = false; //checkboxes are not edited with editor
				if (force_edit_checkbox_only_on_checkbox) {
					if (x < cache.checked->get_width()) {
						p_item->set_checked(col, !c.checked);
						item_edited(col, p_item);
					}
				} else {
					p_item->set_checked(col, !c.checked);
					item_edited(col, p_item);
				}
				click_handled = true;
				//p_item->edited_signal.call(col);

			} break;
			case TreeItem::CELL_MODE_RANGE:
			case TreeItem::CELL_MODE_RANGE_EXPRESSION: {

				if (c.text != "") {
					//if (x >= (get_column_width(col)-item_h/2)) {

					popup_menu->clear();
					for (int i = 0; i < c.text.get_slice_count(","); i++) {

						String s = c.text.get_slicec(',', i);
						popup_menu->add_item(s.get_slicec(':', 0), s.get_slicec(':', 1).empty() ? i : s.get_slicec(':', 1).to_int());
					}

					popup_menu->set_size(Size2(col_width, 0));
					popup_menu->set_position(get_global_position() + Point2i(col_ofs, _get_title_button_height() + y_ofs + item_h) - cache.offset);
					popup_menu->popup();
					popup_edited_item = p_item;
					popup_edited_item_col = col;
					//}
					bring_up_editor = false;
				} else {

					if (x >= (col_width - item_h / 2)) {

						/* touching the combo */
						bool up = p_pos.y < (item_h / 2);

						if (p_button == BUTTON_LEFT) {

							if (range_click_timer->get_time_left() == 0) {

								range_item_last = p_item;
								range_up_last = up;

								range_click_timer->set_wait_time(0.6);
								range_click_timer->set_one_shot(true);
								range_click_timer->start();

							} else if (up != range_up_last) {

								return -1; // break. avoid changing direction on mouse held
							}

							p_item->set_range(col, c.val + (up ? 1.0 : -1.0) * c.step);

							item_edited(col, p_item);

						} else if (p_button == BUTTON_RIGHT) {

							p_item->set_range(col, (up ? c.max : c.min));
							item_edited(col, p_item);
						} else if (p_button == BUTTON_WHEEL_UP) {

							p_item->set_range(col, c.val + c.step);
							item_edited(col, p_item);
						} else if (p_button == BUTTON_WHEEL_DOWN) {

							p_item->set_range(col, c.val - c.step);
							item_edited(col, p_item);
						}

						//p_item->edited_signal.call(col);
						bring_up_editor = false;

					} else {

						editor_text = String::num(p_item->cells[col].val, Math::step_decimals(p_item->cells[col].step));
						if (select_mode == SELECT_MULTI && get_tree()->get_last_event_id() == focus_in_id)
							bring_up_editor = false;
					}
				}
				click_handled = true;

			} break;
			case TreeItem::CELL_MODE_ICON: {
				bring_up_editor = false;
			} break;
			case TreeItem::CELL_MODE_CUSTOM: {
				edited_item = p_item;
				edited_col = col;
				bool on_arrow = x > col_width - cache.select_arrow->get_width();

				bring_up_editor = false;

				custom_popup_rect = Rect2i(get_global_position() + Point2i(col_ofs, _get_title_button_height() + y_ofs + item_h - cache.offset.y), Size2(get_column_width(col), item_h));

				if (on_arrow || !p_item->cells[col].custom_button) {
					emit_signal("custom_popup_edited", ((bool)(x >= (col_width - item_h / 2))));
				}

				if (!p_item->cells[col].custom_button || !on_arrow) {
					item_edited(col, p_item, p_button == BUTTON_LEFT);
				}
				click_handled = true;
				return -1;
			} break;
		};

		if (!bring_up_editor || p_button != BUTTON_LEFT)
			return -1;

		click_handled = true;
		popup_edited_item = p_item;
		popup_edited_item_col = col;

		pressing_item_rect = Rect2(get_global_position() + Point2i(col_ofs, _get_title_button_height() + y_ofs) - cache.offset, Size2(col_width, item_h));
		pressing_for_editor_text = editor_text;
		pressing_for_editor = true;

		return -1; //select
	} else {

		Point2i new_pos = p_pos;

		if (!skip) {
			x_ofs += cache.item_margin;
			//new_pos.x-=cache.item_margin;
			y_ofs += item_h;
			new_pos.y -= item_h;
		}

		if (!p_item->collapsed) { /* if not collapsed, check the childs */

			TreeItem *c = p_item->childs;

			while (c) {

				int child_h = propagate_mouse_event(new_pos, x_ofs, y_ofs, p_doubleclick, c, p_button, p_mod);

				if (child_h < 0)
					return -1; // break, stop propagating, no need to anymore

				new_pos.y -= child_h;
				y_ofs += child_h;
				c = c->next;
				item_h += child_h;
			}
		}
	}

	return item_h; // nothing found
}

void Tree::_text_editor_modal_close() {

	if (Input::get_singleton()->is_key_pressed(KEY_ESCAPE) ||
			Input::get_singleton()->is_key_pressed(KEY_KP_ENTER) ||
			Input::get_singleton()->is_key_pressed(KEY_ENTER)) {

		return;
	}

	if (value_editor->has_point(value_editor->get_local_mouse_position()))
		return;

	text_editor_enter(text_editor->get_text());
}

void Tree::text_editor_enter(String p_text) {

	text_editor->hide();
	value_editor->hide();

	if (!popup_edited_item)
		return;

	if (popup_edited_item_col < 0 || popup_edited_item_col > columns.size())
		return;

	TreeItem::Cell &c = popup_edited_item->cells[popup_edited_item_col];
	switch (c.mode) {

		case TreeItem::CELL_MODE_STRING: {

			c.text = p_text;
			//popup_edited_item->edited_signal.call( popup_edited_item_col );
		} break;
		case TreeItem::CELL_MODE_RANGE: {

			c.val = p_text.to_double();
			if (c.step > 0)
				c.val = Math::stepify(c.val, c.step);
			if (c.val < c.min)
				c.val = c.min;
			else if (c.val > c.max)
				c.val = c.max;

			//popup_edited_item->edited_signal.call( popup_edited_item_col );
		} break;
		case TreeItem::CELL_MODE_RANGE_EXPRESSION: {

			if (evaluator)
				c.val = evaluator->eval(p_text);
			else
				c.val = p_text.to_double();

			if (c.step > 0)
				c.val = Math::stepify(c.val, c.step);
			if (c.val < c.min)
				c.val = c.min;
			else if (c.val > c.max)
				c.val = c.max;

		} break;
		default: { ERR_FAIL(); }
	}

	item_edited(popup_edited_item_col, popup_edited_item);
	update();
}

void Tree::value_editor_changed(double p_value) {

	if (updating_value_editor) {
		return;
	}
	if (!popup_edited_item) {
		return;
	}

	TreeItem::Cell &c = popup_edited_item->cells[popup_edited_item_col];
	c.val = p_value;
	item_edited(popup_edited_item_col, popup_edited_item);
	update();
}

void Tree::popup_select(int p_option) {

	if (!popup_edited_item)
		return;

	if (popup_edited_item_col < 0 || popup_edited_item_col > columns.size())
		return;

	popup_edited_item->cells[popup_edited_item_col].val = p_option;
	//popup_edited_item->edited_signal.call( popup_edited_item_col );
	update();
	item_edited(popup_edited_item_col, popup_edited_item);
}

void Tree::_gui_input(Ref<InputEvent> p_event) {

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {

		if (!k->is_pressed())
			return;
		if (k->get_command() || (k->get_shift() && k->get_unicode() == 0) || k->get_metakey())
			return;
		if (!root)
			return;

		if (hide_root && !root->get_next_visible())
			return;

		switch (k->get_scancode()) {
#define EXIT_BREAK                                 \
	{                                              \
		if (!cursor_can_exit_tree) accept_event(); \
		break;                                     \
	}
			case KEY_RIGHT: {
				bool dobreak = true;

				//TreeItem *next = NULL;
				if (!selected_item)
					break;
				if (select_mode == SELECT_ROW) {
					EXIT_BREAK;
				}
				if (selected_col > (columns.size() - 1)) {
					EXIT_BREAK;
				}
				if (k->get_alt()) {
					selected_item->set_collapsed(false);
					TreeItem *next = selected_item->get_children();
					while (next && next != selected_item->next) {
						next->set_collapsed(false);
						next = next->get_next_visible();
					}
				} else if (selected_col == (columns.size() - 1)) {
					if (selected_item->get_children() != NULL && selected_item->is_collapsed()) {
						selected_item->set_collapsed(false);
					} else {
						selected_col = 0;
						dobreak = false; // fall through to key_down
					}
				} else {
					if (select_mode == SELECT_MULTI) {
						selected_col++;
						emit_signal("cell_selected");
					} else {

						selected_item->select(selected_col + 1);
					}
				}
				update();
				ensure_cursor_is_visible();
				accept_event();
				if (dobreak) {
					break;
				}
			}
			case KEY_DOWN: {

				TreeItem *next = NULL;
				if (!selected_item) {

					next = hide_root ? root->get_next_visible() : root;
					selected_item = 0;
				} else {

					next = selected_item->get_next_visible();

					//if (diff < uint64_t(GLOBAL_DEF("gui/incr_search_max_interval_msec",2000))) {
					if (last_keypress != 0) {
						//incr search next
						int col;
						next = _search_item_text(next, incr_search, &col, true);
						if (!next) {
							accept_event();
							return;
						}
					}
				}

				if (select_mode == SELECT_MULTI) {

					if (!next)
						EXIT_BREAK;

					selected_item = next;
					emit_signal("cell_selected");
					update();
				} else {

					int col = selected_col < 0 ? 0 : selected_col;

					while (next && !next->cells[col].selectable)
						next = next->get_next_visible();
					if (!next)
						EXIT_BREAK; // do nothing..
					next->select(col);
				}

				ensure_cursor_is_visible();
				accept_event();

			} break;
			case KEY_LEFT: {
				bool dobreak = true;

				//TreeItem *next = NULL;
				if (!selected_item)
					break;
				if (select_mode == SELECT_ROW) {
					EXIT_BREAK;
				}
				if (selected_col < 0) {
					EXIT_BREAK;
				}
				if (k->get_alt()) {
					selected_item->set_collapsed(true);
					TreeItem *next = selected_item->get_children();
					while (next && next != selected_item->next) {
						next->set_collapsed(true);
						next = next->get_next_visible();
					}
				} else if (selected_col == 0) {
					if (selected_item->get_children() != NULL && !selected_item->is_collapsed()) {
						selected_item->set_collapsed(true);
					} else {
						if (columns.size() == 1) { // goto parent with one column
							TreeItem *parent = selected_item->get_parent();
							if (selected_item != get_root() && parent && parent->is_selectable(selected_col) && !(hide_root && parent == get_root())) {
								select_single_item(parent, get_root(), selected_col);
							}
						} else {
							selected_col = columns.size() - 1;
							dobreak = false; // fall through to key_up
						}
					}
				} else {
					if (select_mode == SELECT_MULTI) {
						selected_col--;
						emit_signal("cell_selected");
					} else {

						selected_item->select(selected_col - 1);
					}
				}
				update();
				accept_event();
				ensure_cursor_is_visible();

				if (dobreak) {
					break;
				}
			}
			case KEY_UP: {

				TreeItem *prev = NULL;
				if (!selected_item) {
					prev = get_last_item();
					selected_col = 0;
				} else {

					prev = selected_item->get_prev_visible();
					if (last_keypress != 0) {
						//incr search next
						int col;
						prev = _search_item_text(prev, incr_search, &col, true, true);
						if (!prev) {
							accept_event();
							return;
						}
					}
				}

				if (select_mode == SELECT_MULTI) {

					if (!prev)
						break;
					selected_item = prev;
					emit_signal("cell_selected");
					update();
				} else {

					int col = selected_col < 0 ? 0 : selected_col;
					while (prev && !prev->cells[col].selectable)
						prev = prev->get_prev_visible();
					if (!prev)
						break; // do nothing..
					prev->select(col);
				}

				ensure_cursor_is_visible();
				accept_event();

			} break;
			case KEY_PAGEDOWN: {

				TreeItem *next = NULL;
				if (!selected_item)
					break;
				next = selected_item;

				for (int i = 0; i < 10; i++) {

					TreeItem *_n = next->get_next_visible();
					if (_n) {
						next = _n;
					} else {

						break;
					}
				}
				if (next == selected_item)
					break;

				if (select_mode == SELECT_MULTI) {

					selected_item = next;
					emit_signal("cell_selected");
					update();
				} else {

					while (next && !next->cells[selected_col].selectable)
						next = next->get_next_visible();
					if (!next)
						EXIT_BREAK; // do nothing..
					next->select(selected_col);
				}

				ensure_cursor_is_visible();
			} break;
			case KEY_PAGEUP: {

				TreeItem *prev = NULL;
				if (!selected_item)
					break;
				prev = selected_item;

				for (int i = 0; i < 10; i++) {

					TreeItem *_n = prev->get_prev_visible();
					if (_n) {
						prev = _n;
					} else {

						break;
					}
				}
				if (prev == selected_item)
					break;

				if (select_mode == SELECT_MULTI) {

					selected_item = prev;
					emit_signal("cell_selected");
					update();
				} else {

					while (prev && !prev->cells[selected_col].selectable)
						prev = prev->get_prev_visible();
					if (!prev)
						EXIT_BREAK; // do nothing..
					prev->select(selected_col);
				}

				ensure_cursor_is_visible();

			} break;
			case KEY_F2:
			case KEY_ENTER:
			case KEY_KP_ENTER: {

				if (selected_item) {
					//bring up editor if possible
					if (!edit_selected()) {
						emit_signal("item_activated");
						incr_search.clear();
					}
				}
				accept_event();

			} break;
			case KEY_SPACE: {
				if (select_mode == SELECT_MULTI) {
					if (!selected_item)
						break;
					if (selected_item->is_selected(selected_col)) {
						selected_item->deselect(selected_col);
						emit_signal("multi_selected", selected_item, selected_col, false);
					} else if (selected_item->is_selectable(selected_col)) {
						selected_item->select(selected_col);
						emit_signal("multi_selected", selected_item, selected_col, true);
					}
				}
				accept_event();

			} break;
			default: {

				if (k->get_unicode() > 0) {

					_do_incr_search(String::chr(k->get_unicode()));
					accept_event();

					return;
				} else {
					if (k->get_scancode() != KEY_SHIFT)
						last_keypress = 0;
				}
			} break;

				last_keypress = 0;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		if (cache.font.is_null()) // avoid a strange case that may fuckup stuff
			update_cache();

		Ref<StyleBox> bg = cache.bg;

		Point2 pos = mm->get_position() - bg->get_offset();

		Cache::ClickType old_hover = cache.hover_type;
		int old_index = cache.hover_index;

		cache.hover_type = Cache::CLICK_NONE;
		cache.hover_index = 0;
		if (show_column_titles) {
			pos.y -= _get_title_button_height();
			if (pos.y < 0) {
				pos.x += cache.offset.x;
				int len = 0;
				for (int i = 0; i < columns.size(); i++) {

					len += get_column_width(i);
					if (pos.x < len) {

						cache.hover_type = Cache::CLICK_TITLE;
						cache.hover_index = i;
						update();
						break;
					}
				}
			}
		}

		if (root) {

			Point2 mpos = mm->get_position();
			mpos -= cache.bg->get_offset();
			mpos.y -= _get_title_button_height();
			if (mpos.y >= 0) {

				if (h_scroll->is_visible_in_tree())
					mpos.x += h_scroll->get_value();
				if (v_scroll->is_visible_in_tree())
					mpos.y += v_scroll->get_value();

				int col, h, section;
				TreeItem *it = _find_item_at_pos(root, mpos, col, h, section);

				if ((drop_mode_flags && it != drop_mode_over) || section != drop_mode_section) {
					drop_mode_over = it;
					drop_mode_section = section;
					update();
				}

				if (it != cache.hover_item || col != cache.hover_cell) {
					cache.hover_item = it;
					cache.hover_cell = col;
					update();
				}
			}
		}

		if (cache.hover_type != old_hover || cache.hover_index != old_index) {
			update();
		}

		if (pressing_for_editor && popup_edited_item && (popup_edited_item->get_cell_mode(popup_edited_item_col) == TreeItem::CELL_MODE_RANGE || popup_edited_item->get_cell_mode(popup_edited_item_col) == TreeItem::CELL_MODE_RANGE_EXPRESSION)) {
			//range drag

			if (!range_drag_enabled) {

				Vector2 cpos = mm->get_position();
				if (cpos.distance_to(pressing_pos) > 2) {
					range_drag_enabled = true;
					range_drag_capture_pos = cpos;
					range_drag_base = popup_edited_item->get_range(popup_edited_item_col);
					Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
				}
			} else {

				TreeItem::Cell &c = popup_edited_item->cells[popup_edited_item_col];
				float diff_y = -mm->get_relative().y;
				diff_y = Math::pow(ABS(diff_y), 1.8f) * SGN(diff_y);
				diff_y *= 0.1;
				range_drag_base = CLAMP(range_drag_base + c.step * diff_y, c.min, c.max);
				popup_edited_item->set_range(popup_edited_item_col, range_drag_base);
				item_edited(popup_edited_item_col, popup_edited_item);
			}
		}

		if (drag_touching && !drag_touching_deaccel) {

			drag_accum -= mm->get_relative().y;
			v_scroll->set_value(drag_from + drag_accum);
			drag_speed = -mm->get_speed().y;
		}
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (cache.font.is_null()) // avoid a strange case that may fuckup stuff
			update_cache();

		if (!b->is_pressed()) {

			if (b->get_button_index() == BUTTON_LEFT) {

				Point2 pos = b->get_position() - cache.bg->get_offset();
				if (show_column_titles) {
					pos.y -= _get_title_button_height();

					if (pos.y < 0) {
						pos.x += cache.offset.x;
						int len = 0;
						for (int i = 0; i < columns.size(); i++) {

							len += get_column_width(i);
							if (pos.x < len) {
								emit_signal("column_title_pressed", i);
								break;
							}
						}
					}
				}

				if (single_select_defer) {
					select_single_item(single_select_defer, root, single_select_defer_column);
					single_select_defer = NULL;
				}

				range_click_timer->stop();

				if (pressing_for_editor) {

					if (range_drag_enabled) {

						range_drag_enabled = false;
						Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
						warp_mouse(range_drag_capture_pos);
					} else {
						Rect2 rect = get_selected()->get_meta("__focus_rect");
						if (rect.has_point(Point2(b->get_position().x, b->get_position().y))) {
							edit_selected();
						} else {
							emit_signal("item_double_clicked");
						}
					}
					pressing_for_editor = false;
				}

				if (cache.click_type == Cache::CLICK_BUTTON) {
					// make sure in case of wrong reference after reconstructing whole TreeItems
					cache.click_item = get_item_at_position(cache.click_pos);
					emit_signal("button_pressed", cache.click_item, cache.click_column, cache.click_id);
				}
				cache.click_type = Cache::CLICK_NONE;
				cache.click_index = -1;
				cache.click_id = -1;
				cache.click_item = NULL;
				cache.click_column = 0;

				if (drag_touching) {

					if (drag_speed == 0) {
						drag_touching_deaccel = false;
						drag_touching = false;
						set_physics_process(false);
					} else {

						drag_touching_deaccel = true;
					}
				}
				update();
			}
			return;
		}

		if (range_drag_enabled)
			return;

		switch (b->get_button_index()) {
			case BUTTON_RIGHT:
			case BUTTON_LEFT: {
				Ref<StyleBox> bg = cache.bg;

				Point2 pos = b->get_position() - bg->get_offset();
				cache.click_type = Cache::CLICK_NONE;
				if (show_column_titles) {
					pos.y -= _get_title_button_height();

					if (pos.y < 0) {
						if (b->get_button_index() == BUTTON_LEFT) {
							pos.x += cache.offset.x;
							int len = 0;
							for (int i = 0; i < columns.size(); i++) {

								len += get_column_width(i);
								if (pos.x < len) {

									cache.click_type = Cache::CLICK_TITLE;
									cache.click_index = i;
									//cache.click_id=;
									update();
									break;
								}
							}
						}
						break;
					}
				}
				if (!root || (!root->get_children() && hide_root)) {
					if (b->get_button_index() == BUTTON_RIGHT && allow_rmb_select) {
						emit_signal("empty_tree_rmb_selected", get_local_mouse_position());
					}
					break;
				}

				click_handled = false;
				pressing_for_editor = false;

				blocked++;
				propagate_mouse_event(pos + cache.offset, 0, 0, b->is_doubleclick(), root, b->get_button_index(), b);
				blocked--;

				if (pressing_for_editor) {
					pressing_pos = b->get_position();
				}

				if (b->get_button_index() == BUTTON_RIGHT)
					break;

				if (drag_touching) {
					set_physics_process(false);
					drag_touching_deaccel = false;
					drag_touching = false;
					drag_speed = 0;
					drag_from = 0;
				}

				if (!click_handled) {
					drag_speed = 0;
					drag_accum = 0;
					//last_drag_accum=0;
					drag_from = v_scroll->get_value();
					drag_touching = OS::get_singleton()->has_touchscreen_ui_hint();
					drag_touching_deaccel = false;
					if (drag_touching) {
						set_physics_process(true);
					}

					if (b->get_button_index() == BUTTON_LEFT) {
						if (get_item_at_position(b->get_position()) == NULL && !b->get_shift() && !b->get_control() && !b->get_command())
							emit_signal("nothing_selected");
					}
				}

			} break;
			case BUTTON_WHEEL_UP: {

				v_scroll->set_value(v_scroll->get_value() - v_scroll->get_page() * b->get_factor() / 8);
			} break;
			case BUTTON_WHEEL_DOWN: {

				v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * b->get_factor() / 8);
			} break;
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {

		v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * pan_gesture->get_delta().y / 8);
	}
}

bool Tree::edit_selected() {

	TreeItem *s = get_selected();
	ERR_EXPLAIN("No item selected!");
	ERR_FAIL_COND_V(!s, false);
	ensure_cursor_is_visible();
	int col = get_selected_column();
	ERR_EXPLAIN("No item column selected!");
	ERR_FAIL_INDEX_V(col, columns.size(), false);

	if (!s->cells[col].editable)
		return false;

	Rect2 rect = s->get_meta("__focus_rect");

	popup_edited_item = s;
	popup_edited_item_col = col;

	TreeItem::Cell &c = s->cells[col];

	if (c.mode == TreeItem::CELL_MODE_CHECK) {

		s->set_checked(col, !c.checked);
		item_edited(col, s);
		return true;
	} else if (c.mode == TreeItem::CELL_MODE_CUSTOM) {

		edited_item = s;
		edited_col = col;
		custom_popup_rect = Rect2i(get_global_position() + rect.position, rect.size);
		emit_signal("custom_popup_edited", false);
		item_edited(col, s);

		return true;
	} else if ((c.mode == TreeItem::CELL_MODE_RANGE || c.mode == TreeItem::CELL_MODE_RANGE_EXPRESSION) && c.text != "") {

		popup_menu->clear();
		for (int i = 0; i < c.text.get_slice_count(","); i++) {

			String s = c.text.get_slicec(',', i);
			popup_menu->add_item(s.get_slicec(':', 0), s.get_slicec(':', 1).empty() ? i : s.get_slicec(':', 1).to_int());
		}

		popup_menu->set_size(Size2(rect.size.width, 0));
		popup_menu->set_position(get_global_position() + rect.position + Point2i(0, rect.size.height));
		popup_menu->popup();
		popup_edited_item = s;
		popup_edited_item_col = col;
		return true;

	} else if (c.mode == TreeItem::CELL_MODE_STRING || c.mode == TreeItem::CELL_MODE_RANGE || c.mode == TreeItem::CELL_MODE_RANGE_EXPRESSION) {

		Vector2 ofs(0, (text_editor->get_size().height - rect.size.height) / 2);
		Point2i textedpos = get_global_position() + rect.position - ofs;
		text_editor->set_position(textedpos);
		text_editor->set_size(rect.size);
		text_editor->clear();
		text_editor->set_text(c.mode == TreeItem::CELL_MODE_STRING ? c.text : String::num(c.val, Math::step_decimals(c.step)));
		text_editor->select_all();

		if (c.mode == TreeItem::CELL_MODE_RANGE || c.mode == TreeItem::CELL_MODE_RANGE_EXPRESSION) {

			value_editor->set_position(textedpos + Point2i(0, text_editor->get_size().height));
			value_editor->set_size(Size2(rect.size.width, 1));
			value_editor->show_modal();
			updating_value_editor = true;
			value_editor->set_min(c.min);
			value_editor->set_max(c.max);
			value_editor->set_step(c.step);
			value_editor->set_value(c.val);
			value_editor->set_exp_ratio(c.expr);
			updating_value_editor = false;
		}

		text_editor->show_modal();
		text_editor->grab_focus();
		return true;
	}

	return false;
}

Size2 Tree::get_internal_min_size() const {

	Size2i size = cache.bg->get_offset();
	if (root)
		size.height += get_item_height(root);
	for (int i = 0; i < columns.size(); i++) {

		size.width += columns[i].min_width;
	}

	return size;
}

void Tree::update_scrollbars() {

	Size2 size = get_size();
	int tbh;
	if (show_column_titles) {
		tbh = _get_title_button_height();
	} else {

		tbh = 0;
	}

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_begin(Point2(size.width - vmin.width, cache.bg->get_margin(MARGIN_TOP)));
	v_scroll->set_end(Point2(size.width, size.height - cache.bg->get_margin(MARGIN_TOP) - cache.bg->get_margin(MARGIN_BOTTOM)));

	h_scroll->set_begin(Point2(0, size.height - hmin.height));
	h_scroll->set_end(Point2(size.width - vmin.width, size.height));

	Size2 min = get_internal_min_size();

	if (min.height < size.height - hmin.height) {

		v_scroll->hide();
		cache.offset.y = 0;
	} else {

		v_scroll->show();
		v_scroll->set_max(min.height);
		v_scroll->set_page(size.height - hmin.height - tbh);
		cache.offset.y = v_scroll->get_value();
	}

	if (min.width < size.width - vmin.width) {

		h_scroll->hide();
		cache.offset.x = 0;
	} else {

		h_scroll->show();
		h_scroll->set_max(min.width);
		h_scroll->set_page(size.width - vmin.width);
		cache.offset.x = h_scroll->get_value();
	}
}

int Tree::_get_title_button_height() const {

	return show_column_titles ? cache.font->get_height() + cache.title_button->get_minimum_size().height : 0;
}

void Tree::_notification(int p_what) {

	if (p_what == NOTIFICATION_FOCUS_ENTER) {

		focus_in_id = get_tree()->get_last_event_id();
	}
	if (p_what == NOTIFICATION_MOUSE_EXIT) {

		if (cache.hover_type != Cache::CLICK_NONE) {
			cache.hover_type = Cache::CLICK_NONE;
			update();
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		drag_touching = false;
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		update_cache();
	}
	if (p_what == NOTIFICATION_DRAG_END) {

		drop_mode_flags = 0;
		scrolling = false;
		set_physics_process(false);
		update();
	}
	if (p_what == NOTIFICATION_DRAG_BEGIN) {

		single_select_defer = NULL;
		if (cache.scroll_speed > 0 && get_rect().has_point(get_viewport()->get_mouse_position() - get_global_position())) {
			scrolling = true;
			set_physics_process(true);
		}
	}
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {

		if (drag_touching) {

			if (drag_touching_deaccel) {

				float pos = v_scroll->get_value();
				pos += drag_speed * get_physics_process_delta_time();

				bool turnoff = false;
				if (pos < 0) {
					pos = 0;
					turnoff = true;
					set_physics_process(false);
					drag_touching = false;
					drag_touching_deaccel = false;
				}
				if (pos > (v_scroll->get_max() - v_scroll->get_page())) {
					pos = v_scroll->get_max() - v_scroll->get_page();
					turnoff = true;
				}

				v_scroll->set_value(pos);
				float sgn = drag_speed < 0 ? -1 : 1;
				float val = Math::abs(drag_speed);
				val -= 1000 * get_physics_process_delta_time();

				if (val < 0) {
					turnoff = true;
				}
				drag_speed = sgn * val;

				if (turnoff) {
					set_physics_process(false);
					drag_touching = false;
					drag_touching_deaccel = false;
				}

			} else {
			}
		}

		if (scrolling) {
			Point2 point = get_viewport()->get_mouse_position() - get_global_position();
			if (point.x < cache.scroll_border) {
				point.x -= cache.scroll_border;
			} else if (point.x > get_size().width - cache.scroll_border) {
				point.x -= get_size().width - cache.scroll_border;
			} else {
				point.x = 0;
			}
			if (point.y < cache.scroll_border) {
				point.y -= cache.scroll_border;
			} else if (point.y > get_size().height - cache.scroll_border) {
				point.y -= get_size().height - cache.scroll_border;
			} else {
				point.y = 0;
			}
			point *= cache.scroll_speed * get_physics_process_delta_time();
			point += get_scroll();
			h_scroll->set_value(point.x);
			v_scroll->set_value(point.y);
		}
	}

	if (p_what == NOTIFICATION_DRAW) {

		update_cache();
		update_scrollbars();
		RID ci = get_canvas_item();

		Ref<StyleBox> bg = cache.bg;
		Ref<StyleBox> bg_focus = get_stylebox("bg_focus");

		Point2 draw_ofs;
		draw_ofs += bg->get_offset();
		Size2 draw_size = get_size() - bg->get_minimum_size();

		bg->draw(ci, Rect2(Point2(), get_size()));
		if (has_focus()) {
			VisualServer::get_singleton()->canvas_item_add_clip_ignore(ci, true);
			bg_focus->draw(ci, Rect2(Point2(), get_size()));
			VisualServer::get_singleton()->canvas_item_add_clip_ignore(ci, false);
		}

		int tbh = _get_title_button_height();

		draw_ofs.y += tbh;
		draw_size.y -= tbh;

		if (root) {

			draw_item(Point2(), draw_ofs, draw_size, root);
		}

		int ofs = 0;

		for (int i = 0; i < (columns.size() - 1 - 1); i++) {

			ofs += get_column_width(i);
		}

		if (show_column_titles) {

			//title butons
			int ofs = cache.bg->get_margin(MARGIN_LEFT);
			for (int i = 0; i < columns.size(); i++) {

				Ref<StyleBox> sb = (cache.click_type == Cache::CLICK_TITLE && cache.click_index == i) ? cache.title_button_pressed : ((cache.hover_type == Cache::CLICK_TITLE && cache.hover_index == i) ? cache.title_button_hover : cache.title_button);
				Ref<Font> f = cache.tb_font;
				Rect2 tbrect = Rect2(ofs - cache.offset.x, bg->get_margin(MARGIN_TOP), get_column_width(i), tbh);
				sb->draw(ci, tbrect);
				ofs += tbrect.size.width;
				//text
				int clip_w = tbrect.size.width - sb->get_minimum_size().width;
				f->draw_halign(ci, tbrect.position + Point2i(sb->get_offset().x, (tbrect.size.height - f->get_height()) / 2 + f->get_ascent()), HALIGN_CENTER, clip_w, columns[i].title, cache.title_button_color);
			}
		}
	}

	if (p_what == NOTIFICATION_THEME_CHANGED) {
		update_cache();
	}
}

Size2 Tree::get_minimum_size() const {

	return Size2(1, 1);
}

TreeItem *Tree::create_item(TreeItem *p_parent) {

	ERR_FAIL_COND_V(blocked > 0, NULL);

	TreeItem *ti = memnew(TreeItem(this));

	ERR_FAIL_COND_V(!ti, NULL);
	ti->cells.resize(columns.size());

	if (p_parent) {

		/* Always append at the end */

		TreeItem *last = 0;
		TreeItem *c = p_parent->childs;

		while (c) {

			last = c;
			c = c->next;
		}

		if (last) {

			last->next = ti;
		} else {

			p_parent->childs = ti;
		}
		ti->parent = p_parent;

	} else {

		if (root)
			ti->childs = root;

		root = ti;
	}

	return ti;
}

TreeItem *Tree::get_root() {

	return root;
}
TreeItem *Tree::get_last_item() {

	TreeItem *last = root;

	while (last) {

		if (last->next)
			last = last->next;
		else if (last->childs)
			last = last->childs;
		else
			break;
	}

	return last;
}

void Tree::item_edited(int p_column, TreeItem *p_item, bool p_lmb) {

	edited_item = p_item;
	edited_col = p_column;
	if (p_lmb)
		emit_signal("item_edited");
	else
		emit_signal("item_rmb_edited");
}

void Tree::item_changed(int p_column, TreeItem *p_item) {

	update();
}

void Tree::item_selected(int p_column, TreeItem *p_item) {

	if (select_mode == SELECT_MULTI) {

		if (!p_item->cells[p_column].selectable)
			return;

		p_item->cells[p_column].selected = true;
		//emit_signal("multi_selected",p_item,p_column,true); - NO this is for TreeItem::select

	} else {

		select_single_item(p_item, root, p_column);
	}
	update();
}

void Tree::item_deselected(int p_column, TreeItem *p_item) {

	if (select_mode == SELECT_MULTI || select_mode == SELECT_SINGLE) {
		p_item->cells[p_column].selected = false;
	}
	update();
}

void Tree::set_select_mode(SelectMode p_mode) {

	select_mode = p_mode;
}

void Tree::deselect_all() {

	TreeItem *item = get_next_selected(get_root());
	while (item) {
		item->deselect(selected_col);
		item = get_next_selected(get_root());
	}

	selected_item = NULL;
	selected_col = -1;

	update();
}

bool Tree::is_anything_selected() {

	return (selected_item != NULL);
}

void Tree::clear() {

	if (blocked > 0) {

		ERR_FAIL_COND(blocked > 0);
	}

	if (pressing_for_editor) {
		if (range_drag_enabled) {
			range_drag_enabled = false;
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			warp_mouse(range_drag_capture_pos);
		}
		pressing_for_editor = false;
	}

	if (root) {
		memdelete(root);
		root = NULL;
	};

	selected_item = NULL;
	edited_item = NULL;
	popup_edited_item = NULL;

	update();
};

void Tree::set_hide_root(bool p_enabled) {

	hide_root = p_enabled;
	update();
}

void Tree::set_column_min_width(int p_column, int p_min_width) {

	ERR_FAIL_INDEX(p_column, columns.size());

	if (p_min_width < 1)
		return;
	columns[p_column].min_width = p_min_width;
	update();
}
void Tree::set_column_expand(int p_column, bool p_expand) {

	ERR_FAIL_INDEX(p_column, columns.size());

	columns[p_column].expand = p_expand;
	update();
}

TreeItem *Tree::get_selected() const {

	return selected_item;
}

int Tree::get_selected_column() const {

	return selected_col;
}

TreeItem *Tree::get_edited() const {

	return edited_item;
}

int Tree::get_edited_column() const {

	return edited_col;
}

TreeItem *Tree::get_next_selected(TreeItem *p_item) {

	/*
	if (!p_item)
		return NULL;
	*/
	if (!root)
		return NULL;

	while (true) {

		if (!p_item) {
			p_item = root;
		} else {

			if (p_item->childs) {

				p_item = p_item->childs;

			} else if (p_item->next) {

				p_item = p_item->next;
			} else {

				while (!p_item->next) {

					p_item = p_item->parent;
					if (p_item == NULL)
						return NULL;
				}

				p_item = p_item->next;
			}
		}

		for (int i = 0; i < columns.size(); i++)
			if (p_item->cells[i].selected)
				return p_item;
	}

	return NULL;
}

int Tree::get_column_width(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, columns.size(), -1);

	if (!columns[p_column].expand)
		return columns[p_column].min_width;

	Ref<StyleBox> bg = cache.bg;

	int expand_area = get_size().width - (bg->get_margin(MARGIN_LEFT) + bg->get_margin(MARGIN_RIGHT));

	if (v_scroll->is_visible_in_tree())
		expand_area -= v_scroll->get_combined_minimum_size().width;

	int expanding_columns = 0;
	int expanding_total = 0;

	for (int i = 0; i < columns.size(); i++) {

		if (!columns[i].expand) {
			expand_area -= columns[i].min_width;
		} else {
			expanding_total += columns[i].min_width;
			expanding_columns++;
		}
	}

	if (expand_area < expanding_total)
		return columns[p_column].min_width;

	ERR_FAIL_COND_V(expanding_columns == 0, -1); // shouldn't happen

	return expand_area * columns[p_column].min_width / expanding_total;
}

void Tree::propagate_set_columns(TreeItem *p_item) {

	p_item->cells.resize(columns.size());

	TreeItem *c = p_item->get_children();
	while (c) {

		propagate_set_columns(c);
		c = c->get_next();
	}
}

void Tree::set_columns(int p_columns) {

	ERR_FAIL_COND(p_columns < 1);
	ERR_FAIL_COND(blocked > 0);
	columns.resize(p_columns);

	if (root)
		propagate_set_columns(root);
	if (selected_col >= p_columns)
		selected_col = p_columns - 1;
	update();
}

int Tree::get_columns() const {

	return columns.size();
}

void Tree::_scroll_moved(float) {

	update();
}

Rect2 Tree::get_custom_popup_rect() const {

	return custom_popup_rect;
}

int Tree::get_item_offset(TreeItem *p_item) const {

	TreeItem *it = root;
	int ofs = _get_title_button_height();
	if (!it)
		return 0;

	while (true) {

		if (it == p_item)
			return ofs;

		ofs += compute_item_height(it) + cache.vseparation;

		if (it->childs && !it->collapsed) {

			it = it->childs;

		} else if (it->next) {

			it = it->next;
		} else {

			while (!it->next) {

				it = it->parent;
				if (it == NULL)
					return 0;
			}

			it = it->next;
		}
	}

	return -1; //not found
}

void Tree::ensure_cursor_is_visible() {

	if (!is_inside_tree())
		return;

	TreeItem *selected = get_selected();
	if (!selected)
		return;
	int ofs = get_item_offset(selected);
	if (ofs == -1)
		return;
	int h = compute_item_height(selected) + cache.vseparation;
	int screenh = get_size().height - h_scroll->get_combined_minimum_size().height;

	if (ofs + h > v_scroll->get_value() + screenh)
		v_scroll->call_deferred("set_value", ofs - screenh + h);
	else if (ofs < v_scroll->get_value())
		v_scroll->set_value(ofs);
}

int Tree::get_pressed_button() const {

	return pressed_button;
}

Rect2 Tree::get_item_rect(TreeItem *p_item, int p_column) const {

	ERR_FAIL_NULL_V(p_item, Rect2());
	ERR_FAIL_COND_V(p_item->tree != this, Rect2());
	if (p_column != -1) {
		ERR_FAIL_INDEX_V(p_column, columns.size(), Rect2());
	}

	int ofs = get_item_offset(p_item);
	int height = compute_item_height(p_item);
	Rect2 r;
	r.position.y = ofs;
	r.size.height = height;

	if (p_column == -1) {
		r.position.x = 0;
		r.size.x = get_size().width;
	} else {

		int accum = 0;
		for (int i = 0; i < p_column; i++) {
			accum += get_column_width(i);
		}
		r.position.x = accum;
		r.size.x = get_column_width(p_column);
	}

	return r;
}

void Tree::set_column_titles_visible(bool p_show) {

	show_column_titles = p_show;
	update();
}

bool Tree::are_column_titles_visible() const {

	return show_column_titles;
}

void Tree::set_column_title(int p_column, const String &p_title) {

	ERR_FAIL_INDEX(p_column, columns.size());
	columns[p_column].title = p_title;
	update();
}

String Tree::get_column_title(int p_column) const {

	ERR_FAIL_INDEX_V(p_column, columns.size(), "");
	return columns[p_column].title;
}

Point2 Tree::get_scroll() const {

	Point2 ofs;
	if (h_scroll->is_visible_in_tree())
		ofs.x = h_scroll->get_value();
	if (v_scroll->is_visible_in_tree())
		ofs.y = v_scroll->get_value();
	return ofs;
}

void Tree::scroll_to_item(TreeItem *p_item) {

	if (!is_visible_in_tree()) {

		// hack to work around crash in get_item_rect() if Tree is not in tree.
		return;
	}

	// make sure the scrollbar min and max are up to date with latest changes.
	update_scrollbars();

	const Rect2 r = get_item_rect(p_item);

	if (r.position.y < v_scroll->get_value()) {
		v_scroll->set_value(r.position.y);
	} else if (r.position.y + r.size.y + 2 * cache.vseparation > v_scroll->get_value() + get_size().y) {
		v_scroll->set_value(r.position.y + r.size.y + 2 * cache.vseparation - get_size().y);
	}
}

TreeItem *Tree::_search_item_text(TreeItem *p_at, const String &p_find, int *r_col, bool p_selectable, bool p_backwards) {

	while (p_at) {

		for (int i = 0; i < columns.size(); i++) {
			if (p_at->get_text(i).findn(p_find) == 0 && (!p_selectable || p_at->is_selectable(i))) {
				if (r_col)
					*r_col = i;
				return p_at;
			}
		}

		if (p_backwards)
			p_at = p_at->get_prev_visible();
		else
			p_at = p_at->get_next_visible();
	}

	return NULL;
}

TreeItem *Tree::search_item_text(const String &p_find, int *r_col, bool p_selectable) {

	if (!root)
		return NULL;

	return _search_item_text(root, p_find, r_col, p_selectable);
}

void Tree::_do_incr_search(const String &p_add) {

	uint64_t time = OS::get_singleton()->get_ticks_usec() / 1000; // convert to msec
	uint64_t diff = time - last_keypress;
	if (diff > uint64_t(GLOBAL_DEF("gui/timers/incremental_search_max_interval_msec", 2000)))
		incr_search = p_add;
	else
		incr_search += p_add;

	last_keypress = time;
	int col;
	TreeItem *item = search_item_text(incr_search, &col, true);
	if (!item)
		return;

	item->select(col);
	ensure_cursor_is_visible();
}

TreeItem *Tree::_find_item_at_pos(TreeItem *p_item, const Point2 &p_pos, int &r_column, int &h, int &section) const {

	Point2 pos = p_pos;

	if (root != p_item || !hide_root) {

		h = compute_item_height(p_item) + cache.vseparation;
		if (pos.y < h) {

			if (drop_mode_flags == DROP_MODE_ON_ITEM) {
				section = 0;
			} else if (drop_mode_flags == DROP_MODE_INBETWEEN) {
				section = pos.y < h / 2 ? -1 : 1;
			} else if (pos.y < h / 4) {
				section = -1;
			} else if (pos.y >= (h * 3 / 4)) {
				section = 1;
			} else {
				section = 0;
			}

			for (int i = 0; i < columns.size(); i++) {

				int w = get_column_width(i);
				if (pos.x < w) {
					r_column = i;

					return p_item;
				}
				pos.x -= w;
			}

			return NULL;
		} else {

			pos.y -= h;
		}
	} else {

		h = 0;
	}

	if (p_item->is_collapsed())
		return NULL; // do not try childs, it's collapsed

	TreeItem *n = p_item->get_children();
	while (n) {

		int ch;
		TreeItem *r = _find_item_at_pos(n, pos, r_column, ch, section);
		pos.y -= ch;
		h += ch;
		if (r)
			return r;
		n = n->get_next();
	}

	return NULL;
}

int Tree::get_column_at_position(const Point2 &p_pos) const {

	if (root) {

		Point2 pos = p_pos;
		pos -= cache.bg->get_offset();
		pos.y -= _get_title_button_height();
		if (pos.y < 0)
			return -1;

		if (h_scroll->is_visible_in_tree())
			pos.x += h_scroll->get_value();
		if (v_scroll->is_visible_in_tree())
			pos.y += v_scroll->get_value();

		int col, h, section;
		TreeItem *it = _find_item_at_pos(root, pos, col, h, section);

		if (it) {
			return col;
		}
	}

	return -1;
}

int Tree::get_drop_section_at_position(const Point2 &p_pos) const {

	if (root) {

		Point2 pos = p_pos;
		pos -= cache.bg->get_offset();
		pos.y -= _get_title_button_height();
		if (pos.y < 0)
			return -100;

		if (h_scroll->is_visible_in_tree())
			pos.x += h_scroll->get_value();
		if (v_scroll->is_visible_in_tree())
			pos.y += v_scroll->get_value();

		int col, h, section;
		TreeItem *it = _find_item_at_pos(root, pos, col, h, section);

		if (it) {
			return section;
		}
	}

	return -100;
}
TreeItem *Tree::get_item_at_position(const Point2 &p_pos) const {

	if (root) {

		Point2 pos = p_pos;
		pos -= cache.bg->get_offset();
		pos.y -= _get_title_button_height();
		if (pos.y < 0)
			return NULL;

		if (h_scroll->is_visible_in_tree())
			pos.x += h_scroll->get_value();
		if (v_scroll->is_visible_in_tree())
			pos.y += v_scroll->get_value();

		int col, h, section;
		TreeItem *it = _find_item_at_pos(root, pos, col, h, section);

		if (it) {

			return it;
		}
	}

	return NULL;
}

String Tree::get_tooltip(const Point2 &p_pos) const {

	if (root) {

		Point2 pos = p_pos;
		pos -= cache.bg->get_offset();
		pos.y -= _get_title_button_height();
		if (pos.y < 0)
			return Control::get_tooltip(p_pos);

		if (h_scroll->is_visible_in_tree())
			pos.x += h_scroll->get_value();
		if (v_scroll->is_visible_in_tree())
			pos.y += v_scroll->get_value();

		int col, h, section;
		TreeItem *it = _find_item_at_pos(root, pos, col, h, section);

		if (it) {

			TreeItem::Cell &c = it->cells[col];
			int col_width = get_column_width(col);
			for (int j = c.buttons.size() - 1; j >= 0; j--) {
				Ref<Texture> b = c.buttons[j].texture;
				Size2 size = b->get_size() + cache.button_pressed->get_minimum_size();
				if (pos.x > col_width - size.width) {
					String tooltip = c.buttons[j].tooltip;
					if (tooltip != "") {
						return tooltip;
					}
				}
				col_width -= size.width;
			}
			String ret;
			if (it->get_tooltip(col) == "")
				ret = it->get_text(col);
			else
				ret = it->get_tooltip(col);
			return ret;
		}
	}

	return Control::get_tooltip(p_pos);
}

void Tree::set_cursor_can_exit_tree(bool p_enable) {

	cursor_can_exit_tree = p_enable;
}

bool Tree::can_cursor_exit_tree() const {

	return cursor_can_exit_tree;
}

void Tree::set_hide_folding(bool p_hide) {
	hide_folding = p_hide;
	update();
}

bool Tree::is_folding_hidden() const {

	return hide_folding;
}

void Tree::set_value_evaluator(ValueEvaluator *p_evaluator) {
	evaluator = p_evaluator;
}

void Tree::set_drop_mode_flags(int p_flags) {
	if (drop_mode_flags == p_flags)
		return;
	drop_mode_flags = p_flags;
	if (drop_mode_flags == 0) {
		drop_mode_over = NULL;
	}

	update();
}

int Tree::get_drop_mode_flags() const {

	return drop_mode_flags;
}

void Tree::set_edit_checkbox_cell_only_when_checkbox_is_pressed(bool p_enable) {

	force_edit_checkbox_only_on_checkbox = p_enable;
}

bool Tree::get_edit_checkbox_cell_only_when_checkbox_is_pressed() const {

	return force_edit_checkbox_only_on_checkbox;
}

void Tree::set_allow_rmb_select(bool p_allow) {

	allow_rmb_select = p_allow;
}

bool Tree::get_allow_rmb_select() const {

	return allow_rmb_select;
}

void Tree::set_allow_reselect(bool p_allow) {
	allow_reselect = p_allow;
}

bool Tree::get_allow_reselect() const {

	return allow_reselect;
}

void Tree::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_range_click_timeout"), &Tree::_range_click_timeout);
	ClassDB::bind_method(D_METHOD("_gui_input"), &Tree::_gui_input);
	ClassDB::bind_method(D_METHOD("_popup_select"), &Tree::popup_select);
	ClassDB::bind_method(D_METHOD("_text_editor_enter"), &Tree::text_editor_enter);
	ClassDB::bind_method(D_METHOD("_text_editor_modal_close"), &Tree::_text_editor_modal_close);
	ClassDB::bind_method(D_METHOD("_value_editor_changed"), &Tree::value_editor_changed);
	ClassDB::bind_method(D_METHOD("_scroll_moved"), &Tree::_scroll_moved);

	ClassDB::bind_method(D_METHOD("clear"), &Tree::clear);
	ClassDB::bind_method(D_METHOD("create_item", "parent"), &Tree::_create_item, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("get_root"), &Tree::get_root);
	ClassDB::bind_method(D_METHOD("set_column_min_width", "column", "min_width"), &Tree::set_column_min_width);
	ClassDB::bind_method(D_METHOD("set_column_expand", "column", "expand"), &Tree::set_column_expand);
	ClassDB::bind_method(D_METHOD("get_column_width", "column"), &Tree::get_column_width);

	ClassDB::bind_method(D_METHOD("set_hide_root", "enable"), &Tree::set_hide_root);
	ClassDB::bind_method(D_METHOD("get_next_selected", "from"), &Tree::_get_next_selected);
	ClassDB::bind_method(D_METHOD("get_selected"), &Tree::get_selected);
	ClassDB::bind_method(D_METHOD("get_selected_column"), &Tree::get_selected_column);
	ClassDB::bind_method(D_METHOD("get_pressed_button"), &Tree::get_pressed_button);
	ClassDB::bind_method(D_METHOD("set_select_mode", "mode"), &Tree::set_select_mode);

	ClassDB::bind_method(D_METHOD("set_columns", "amount"), &Tree::set_columns);
	ClassDB::bind_method(D_METHOD("get_columns"), &Tree::get_columns);

	ClassDB::bind_method(D_METHOD("get_edited"), &Tree::get_edited);
	ClassDB::bind_method(D_METHOD("get_edited_column"), &Tree::get_edited_column);
	ClassDB::bind_method(D_METHOD("get_custom_popup_rect"), &Tree::get_custom_popup_rect);
	ClassDB::bind_method(D_METHOD("get_item_area_rect", "item", "column"), &Tree::_get_item_rect, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_item_at_position", "position"), &Tree::get_item_at_position);
	ClassDB::bind_method(D_METHOD("get_column_at_position", "position"), &Tree::get_column_at_position);
	ClassDB::bind_method(D_METHOD("get_drop_section_at_position", "position"), &Tree::get_drop_section_at_position);

	ClassDB::bind_method(D_METHOD("ensure_cursor_is_visible"), &Tree::ensure_cursor_is_visible);

	ClassDB::bind_method(D_METHOD("set_column_titles_visible", "visible"), &Tree::set_column_titles_visible);
	ClassDB::bind_method(D_METHOD("are_column_titles_visible"), &Tree::are_column_titles_visible);

	ClassDB::bind_method(D_METHOD("set_column_title", "column", "title"), &Tree::set_column_title);
	ClassDB::bind_method(D_METHOD("get_column_title", "column"), &Tree::get_column_title);
	ClassDB::bind_method(D_METHOD("get_scroll"), &Tree::get_scroll);

	ClassDB::bind_method(D_METHOD("set_hide_folding", "hide"), &Tree::set_hide_folding);
	ClassDB::bind_method(D_METHOD("is_folding_hidden"), &Tree::is_folding_hidden);

	ClassDB::bind_method(D_METHOD("set_drop_mode_flags", "flags"), &Tree::set_drop_mode_flags);
	ClassDB::bind_method(D_METHOD("get_drop_mode_flags"), &Tree::get_drop_mode_flags);

	ClassDB::bind_method(D_METHOD("set_allow_rmb_select", "allow"), &Tree::set_allow_rmb_select);
	ClassDB::bind_method(D_METHOD("get_allow_rmb_select"), &Tree::get_allow_rmb_select);

	ClassDB::bind_method(D_METHOD("set_allow_reselect", "allow"), &Tree::set_allow_reselect);
	ClassDB::bind_method(D_METHOD("get_allow_reselect"), &Tree::get_allow_reselect);

	ADD_SIGNAL(MethodInfo("item_selected"));
	ADD_SIGNAL(MethodInfo("cell_selected"));
	ADD_SIGNAL(MethodInfo("multi_selected", PropertyInfo(Variant::OBJECT, "item"), PropertyInfo(Variant::INT, "column"), PropertyInfo(Variant::BOOL, "selected")));
	ADD_SIGNAL(MethodInfo("item_rmb_selected", PropertyInfo(Variant::VECTOR2, "position")));
	ADD_SIGNAL(MethodInfo("empty_tree_rmb_selected", PropertyInfo(Variant::VECTOR2, "position")));
	ADD_SIGNAL(MethodInfo("item_edited"));
	ADD_SIGNAL(MethodInfo("item_rmb_edited"));
	ADD_SIGNAL(MethodInfo("item_custom_button_pressed"));
	ADD_SIGNAL(MethodInfo("item_double_clicked"));
	ADD_SIGNAL(MethodInfo("item_collapsed", PropertyInfo(Variant::OBJECT, "item")));
	//ADD_SIGNAL( MethodInfo("item_doubleclicked" ) );
	ADD_SIGNAL(MethodInfo("button_pressed", PropertyInfo(Variant::OBJECT, "item"), PropertyInfo(Variant::INT, "column"), PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("custom_popup_edited", PropertyInfo(Variant::BOOL, "arrow_clicked")));
	ADD_SIGNAL(MethodInfo("item_activated"));
	ADD_SIGNAL(MethodInfo("column_title_pressed", PropertyInfo(Variant::INT, "column")));
	ADD_SIGNAL(MethodInfo("nothing_selected"));

	BIND_ENUM_CONSTANT(SELECT_SINGLE);
	BIND_ENUM_CONSTANT(SELECT_ROW);
	BIND_ENUM_CONSTANT(SELECT_MULTI);

	BIND_ENUM_CONSTANT(DROP_MODE_DISABLED);
	BIND_ENUM_CONSTANT(DROP_MODE_ON_ITEM);
	BIND_ENUM_CONSTANT(DROP_MODE_INBETWEEN);
}

Tree::Tree() {

	selected_col = 0;
	columns.resize(1);
	selected_item = NULL;
	edited_item = NULL;
	selected_col = -1;
	edited_col = -1;

	hide_root = false;
	select_mode = SELECT_SINGLE;
	root = 0;
	popup_menu = NULL;
	popup_edited_item = NULL;
	text_editor = NULL;
	set_focus_mode(FOCUS_ALL);

	popup_menu = memnew(PopupMenu);
	popup_menu->hide();
	add_child(popup_menu);
	popup_menu->set_as_toplevel(true);
	text_editor = memnew(LineEdit);
	add_child(text_editor);
	text_editor->set_as_toplevel(true);
	text_editor->hide();
	value_editor = memnew(HSlider);
	add_child(value_editor);
	value_editor->set_as_toplevel(true);
	value_editor->hide();

	h_scroll = memnew(HScrollBar);
	v_scroll = memnew(VScrollBar);

	add_child(h_scroll);
	add_child(v_scroll);

	range_click_timer = memnew(Timer);
	range_click_timer->connect("timeout", this, "_range_click_timeout");
	add_child(range_click_timer);

	h_scroll->connect("value_changed", this, "_scroll_moved");
	v_scroll->connect("value_changed", this, "_scroll_moved");
	text_editor->connect("text_entered", this, "_text_editor_enter");
	text_editor->connect("modal_closed", this, "_text_editor_modal_close");
	popup_menu->connect("id_pressed", this, "_popup_select");
	value_editor->connect("value_changed", this, "_value_editor_changed");

	value_editor->set_as_toplevel(true);
	text_editor->set_as_toplevel(true);

	updating_value_editor = false;
	pressed_button = -1;
	show_column_titles = false;

	cache.click_type = Cache::CLICK_NONE;
	cache.hover_type = Cache::CLICK_NONE;
	cache.hover_index = -1;
	cache.click_index = -1;
	cache.click_id = -1;
	cache.click_item = NULL;
	cache.click_column = 0;
	last_keypress = 0;
	focus_in_id = 0;

	blocked = 0;

	cursor_can_exit_tree = true;
	set_mouse_filter(MOUSE_FILTER_STOP);

	drag_speed = 0;
	drag_touching = false;
	drag_touching_deaccel = false;
	pressing_for_editor = false;
	range_drag_enabled = false;

	hide_folding = false;

	evaluator = NULL;

	drop_mode_flags = 0;
	drop_mode_over = NULL;
	drop_mode_section = 0;
	single_select_defer = NULL;

	allow_rmb_select = false;
	force_edit_checkbox_only_on_checkbox = false;

	set_clip_contents(true);

	cache.hover_item = NULL;
	cache.hover_cell = -1;

	allow_reselect = false;
}

Tree::~Tree() {

	if (root) {
		memdelete(root);
	}
}
