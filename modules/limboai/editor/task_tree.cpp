/**
 * task_tree.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "task_tree.h"

#include "../bt/tasks/bt_comment.h"
#include "../bt/tasks/composites/bt_probability_selector.h"
#include "../util/limbo_compat.h"
#include "../util/limbo_utility.h"

#ifdef LIMBOAI_MODULE
#include "core/object/script_language.h"
#include "editor/themes/editor_scale.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/script.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

//**** TaskTree

TreeItem *TaskTree::_create_tree(const Ref<BTTask> &p_task, TreeItem *p_parent, int p_idx) {
	ERR_FAIL_COND_V(p_task.is_null(), nullptr);
	TreeItem *item = tree->create_item(p_parent, p_idx);
	item->set_metadata(0, p_task);
	for (int i = 0; i < p_task->get_child_count(); i++) {
		_create_tree(p_task->get_child(i), item);
	}
	_update_item(item);
	return item;
}

void TaskTree::_update_item(TreeItem *p_item) {
	if (p_item == nullptr) {
		return;
	}

	if (p_item->get_parent()) {
		Ref<BTProbabilitySelector> sel = p_item->get_parent()->get_metadata(0);
		if (sel.is_valid() && sel->has_probability(p_item->get_index())) {
			p_item->set_custom_draw_callback(0, callable_mp(this, &TaskTree::_draw_probability));
			p_item->set_cell_mode(0, TreeItem::CELL_MODE_CUSTOM);
		}
	}

	Ref<BTTask> task = p_item->get_metadata(0);
	ERR_FAIL_COND_MSG(!task.is_valid(), "Invalid task reference in metadata.");
	p_item->set_text(0, task->get_task_name());
	if (IS_CLASS(task, BTComment)) {
		p_item->set_custom_font(0, theme_cache.comment_font);
		p_item->set_custom_color(0, theme_cache.comment_color);
	} else if (task->get_custom_name().is_empty()) {
		p_item->set_custom_font(0, theme_cache.normal_name_font);
		p_item->clear_custom_color(0);
	} else {
		p_item->set_custom_font(0, theme_cache.custom_name_font);
		// p_item->set_custom_color(0, get_theme_color(SNAME("warning_color"), SNAME("Editor")));
	}
	String type_arg;
	if (task->get_script() != Variant()) {
		Ref<Script> sc = task->get_script();
		if (sc.is_valid()) {
			type_arg = sc->get_path();
		}
	}
	if (type_arg.is_empty()) {
		type_arg = task->get_class();
	}
	p_item->set_icon(0, LimboUtility::get_singleton()->get_task_icon(type_arg));
	p_item->set_icon_max_width(0, 16 * EDSCALE);
	p_item->set_editable(0, false);
	p_item->set_collapsed(task->is_displayed_collapsed());

	for (int i = 0; i < p_item->get_button_count(0); i++) {
		p_item->erase_button(0, i);
	}

	PackedStringArray warnings = task->get_configuration_warnings();
	String warning_text;
	for (int j = 0; j < warnings.size(); j++) {
		if (j > 0) {
			warning_text += "\n";
		}
		warning_text += warnings[j];
	}
	if (!warning_text.is_empty()) {
		p_item->add_button(0, theme_cache.task_warning_icon, 0, false, warning_text);
	}
}

void TaskTree::_update_tree() {
	Ref<BTTask> sel;
	if (tree->get_selected()) {
		sel = tree->get_selected()->get_metadata(0);
	}

	tree->clear();
	if (bt.is_null()) {
		return;
	}

	if (bt->get_root_task().is_valid()) {
		updating_tree = true;
		_create_tree(bt->get_root_task(), nullptr);
		updating_tree = false;
	}

	TreeItem *item = _find_item(sel);
	if (item) {
		item->select(0);
	}
}

TreeItem *TaskTree::_find_item(const Ref<BTTask> &p_task) const {
	if (p_task.is_null()) {
		return nullptr;
	}
	TreeItem *item = tree->get_root();
	List<TreeItem *> stack;
	while (item && item->get_metadata(0) != p_task) {
		if (item->get_child_count() > 0) {
			stack.push_back(item->get_first_child());
		}
		item = item->get_next();
		if (item == nullptr && !stack.is_empty()) {
			item = stack.front()->get();
			stack.pop_front();
		}
	}
	return item;
}

void TaskTree::_on_item_mouse_selected(const Vector2 &p_pos, MouseButton p_button_index) {
	if (p_button_index == LW_MBTN(LEFT)) {
		Rect2 rect = get_selected_probability_rect();
		if (rect != Rect2() && rect.has_point(p_pos)) {
			emit_signal(LW_NAME(probability_clicked));
		}
	} else if (p_button_index == LW_MBTN(RIGHT)) {
		emit_signal(LW_NAME(rmb_pressed), get_screen_position() + p_pos);
	}
}

void TaskTree::_on_item_selected() {
	Callable on_task_changed = callable_mp(this, &TaskTree::_on_task_changed);
	if (last_selected.is_valid()) {
		update_task(last_selected);
		if (last_selected->is_connected(LW_NAME(changed), on_task_changed)) {
			last_selected->disconnect(LW_NAME(changed), on_task_changed);
		}
	}
	last_selected = get_selected();
	last_selected->connect(LW_NAME(changed), on_task_changed);
	emit_signal(LW_NAME(task_selected), last_selected);
}

void TaskTree::_on_item_activated() {
	emit_signal(LW_NAME(task_activated));
}

void TaskTree::_on_item_collapsed(Object *p_obj) {
	if (updating_tree) {
		return;
	}

	TreeItem *item = Object::cast_to<TreeItem>(p_obj);
	if (!item) {
		return;
	}

	Ref<BTTask> task = item->get_metadata(0);
	ERR_FAIL_NULL(task);
	task->set_display_collapsed(item->is_collapsed());
}

void TaskTree::_on_task_changed() {
	_update_item(tree->get_selected());
}

void TaskTree::load_bt(const Ref<BehaviorTree> &p_behavior_tree) {
	ERR_FAIL_COND_MSG(p_behavior_tree.is_null(), "Tried to load a null tree.");

	Callable on_task_changed = callable_mp(this, &TaskTree::_on_task_changed);
	if (last_selected.is_valid() && last_selected->is_connected(LW_NAME(changed), on_task_changed)) {
		last_selected->disconnect(LW_NAME(changed), on_task_changed);
	}

	bt = p_behavior_tree;
	tree->clear();
	probability_rect_cache.clear();
	if (bt->get_root_task().is_valid()) {
		updating_tree = true;
		_create_tree(bt->get_root_task(), nullptr);
		updating_tree = false;
	}
}

void TaskTree::unload() {
	Callable on_task_changed = callable_mp(this, &TaskTree::_on_task_changed);
	if (last_selected.is_valid() && last_selected->is_connected(LW_NAME(changed), on_task_changed)) {
		last_selected->disconnect(LW_NAME(changed), on_task_changed);
	}

	bt.unref();
	tree->clear();
}

void TaskTree::update_task(const Ref<BTTask> &p_task) {
	ERR_FAIL_COND(p_task.is_null());
	TreeItem *item = _find_item(p_task);
	if (item) {
		_update_item(item);
	}
}

Ref<BTTask> TaskTree::get_selected() const {
	if (tree->get_selected()) {
		return tree->get_selected()->get_metadata(0);
	}
	return nullptr;
}

void TaskTree::deselect() {
	TreeItem *sel = tree->get_selected();
	if (sel) {
		sel->deselect(0);
	}
}

Rect2 TaskTree::get_selected_probability_rect() const {
	if (tree->get_selected() == nullptr) {
		return Rect2();
	}

	RECT_CACHE_KEY key = tree->get_selected()->get_instance_id();
	if (unlikely(!probability_rect_cache.has(key))) {
		return Rect2();
	} else {
		return probability_rect_cache[key];
	}
}

double TaskTree::get_selected_probability_weight() const {
	Ref<BTTask> selected = get_selected();
	ERR_FAIL_COND_V(selected.is_null(), 0.0);
	Ref<BTProbabilitySelector> probability_selector = selected->get_parent();
	ERR_FAIL_COND_V(probability_selector.is_null(), 0.0);
	return probability_selector->get_weight(selected->get_index());
}

double TaskTree::get_selected_probability_percent() const {
	Ref<BTTask> selected = get_selected();
	ERR_FAIL_COND_V(selected.is_null(), 0.0);
	Ref<BTProbabilitySelector> probability_selector = selected->get_parent();
	ERR_FAIL_COND_V(probability_selector.is_null(), 0.0);
	return probability_selector->get_probability(selected->get_index()) * 100.0;
}

bool TaskTree::selected_has_probability() const {
	bool result = false;
	Ref<BTTask> selected = get_selected();
	if (selected.is_valid() && !IS_CLASS(selected, BTComment)) {
		Ref<BTProbabilitySelector> probability_selector = selected->get_parent();
		result = probability_selector.is_valid();
	}
	return result;
}

Variant TaskTree::_get_drag_data_fw(const Point2 &p_point) {
	if (editable && tree->get_item_at_position(p_point)) {
		Dictionary drag_data;
		drag_data["type"] = "task";
		drag_data["task"] = tree->get_item_at_position(p_point)->get_metadata(0);
		tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN | Tree::DROP_MODE_ON_ITEM);
		return drag_data;
	}
	return Variant();
}

bool TaskTree::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data) const {
	if (!editable) {
		return false;
	}

	Dictionary d = p_data;
	if (!d.has("type") || !d.has("task")) {
		return false;
	}

	int section = tree->get_drop_section_at_position(p_point);
	TreeItem *item = tree->get_item_at_position(p_point);
	if (!item || section < -1) {
		return false;
	}

	if (!item->get_parent() && section != 0) { // before/after root item
		return false;
	}

	if (String(d["type"]) == "task") {
		Ref<BTTask> task = d["task"];
		const Ref<BTTask> to_task = item->get_metadata(0);
		if (task != to_task && !to_task->is_descendant_of(task)) {
			return true;
		}
	}

	return false;
}

void TaskTree::_drop_data_fw(const Point2 &p_point, const Variant &p_data) {
	Dictionary d = p_data;
	TreeItem *item = tree->get_item_at_position(p_point);
	if (item && d.has("task")) {
		Ref<BTTask> task = d["task"];
		emit_signal(LW_NAME(task_dragged), task, item->get_metadata(0), tree->get_drop_section_at_position(p_point));
	}
}

void TaskTree::_draw_probability(Object *item_obj, Rect2 rect) {
	TreeItem *item = Object::cast_to<TreeItem>(item_obj);
	if (!item) {
		return;
	}
	Ref<BTProbabilitySelector> sel = item->get_parent()->get_metadata(0);
	if (sel.is_null()) {
		return;
	}

	String text = rtos(Math::snapped(sel->get_probability(item->get_index()) * 100, 0.01)) + "%";
	Size2 text_size = theme_cache.probability_font->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.probability_font_size);

	Rect2 prob_rect = rect;
	prob_rect.position.x += theme_cache.name_font->get_string_size(item->get_text(0), HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.name_font_size).x;
	prob_rect.position.x += EDSCALE * 40.0;
	prob_rect.size.x = text_size.x + EDSCALE * 12;
	prob_rect.position.y += 4 * EDSCALE;
	prob_rect.size.y -= 8 * EDSCALE;
	probability_rect_cache[item->get_instance_id()] = prob_rect; // Cache rect for later click detection.

	theme_cache.probability_bg->draw(tree->get_canvas_item(), prob_rect);

	Point2 text_pos = prob_rect.position;
	text_pos.y += text_size.y + (prob_rect.size.y - text_size.y) * 0.5;
	text_pos.y -= theme_cache.probability_font->get_descent(theme_cache.probability_font_size);
	text_pos.y = Math::floor(text_pos.y);

	tree->draw_string(theme_cache.probability_font, text_pos, text, HORIZONTAL_ALIGNMENT_CENTER,
			prob_rect.size.x, theme_cache.probability_font_size, theme_cache.probability_font_color);
}

void TaskTree::_do_update_theme_item_cache() {
	theme_cache.name_font = get_theme_font(LW_NAME(font));
	theme_cache.custom_name_font = get_theme_font(LW_NAME(bold), LW_NAME(EditorFonts));
	theme_cache.comment_font = get_theme_font(LW_NAME(doc_italic), LW_NAME(EditorFonts));
	theme_cache.probability_font = get_theme_font(LW_NAME(font));

	theme_cache.name_font_size = get_theme_font_size(LW_NAME(font_size));
	theme_cache.probability_font_size = Math::floor(get_theme_font_size(LW_NAME(font_size)) * 0.9);

	theme_cache.task_warning_icon = get_theme_icon(LW_NAME(NodeWarning), LW_NAME(EditorIcons));

	theme_cache.comment_color = get_theme_color(LW_NAME(disabled_font_color), LW_NAME(Editor));
	theme_cache.probability_font_color = get_theme_color(LW_NAME(font_color), LW_NAME(Editor));

	theme_cache.probability_bg.instantiate();
	theme_cache.probability_bg->set_bg_color(get_theme_color(LW_NAME(accent_color), LW_NAME(Editor)) * Color(1, 1, 1, 0.25));
	theme_cache.probability_bg->set_corner_radius_all(12.0 * EDSCALE);
}

void TaskTree::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			tree->connect("item_mouse_selected", callable_mp(this, &TaskTree::_on_item_mouse_selected));
			// Note: CONNECT_DEFERRED is needed to avoid double updates with set_allow_reselect(true), which breaks folding/unfolding.
			tree->connect("item_selected", callable_mp(this, &TaskTree::_on_item_selected), CONNECT_DEFERRED);
			tree->connect("item_activated", callable_mp(this, &TaskTree::_on_item_activated));
			tree->connect("item_collapsed", callable_mp(this, &TaskTree::_on_item_collapsed));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();
			_update_tree();
		} break;
	}
}

void TaskTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_bt", "behavior_tree"), &TaskTree::load_bt);
	ClassDB::bind_method(D_METHOD("get_bt"), &TaskTree::get_bt);
	ClassDB::bind_method(D_METHOD("update_tree"), &TaskTree::update_tree);
	ClassDB::bind_method(D_METHOD("update_task", "task"), &TaskTree::update_task);
	ClassDB::bind_method(D_METHOD("get_selected"), &TaskTree::get_selected);
	ClassDB::bind_method(D_METHOD("deselect"), &TaskTree::deselect);

	ClassDB::bind_method(D_METHOD("_get_drag_data_fw"), &TaskTree::_get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("_can_drop_data_fw"), &TaskTree::_can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_drop_data_fw"), &TaskTree::_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_draw_probability"), &TaskTree::_draw_probability);

	ADD_SIGNAL(MethodInfo("rmb_pressed"));
	ADD_SIGNAL(MethodInfo("task_selected"));
	ADD_SIGNAL(MethodInfo("task_activated"));
	ADD_SIGNAL(MethodInfo("probability_clicked"));
	ADD_SIGNAL(MethodInfo("task_dragged",
			PropertyInfo(Variant::OBJECT, "task", PROPERTY_HINT_RESOURCE_TYPE, "BTTask"),
			PropertyInfo(Variant::OBJECT, "to_task", PROPERTY_HINT_RESOURCE_TYPE, "BTTask"),
			PropertyInfo(Variant::INT, "type")));
}

TaskTree::TaskTree() {
	editable = true;
	updating_tree = false;

	tree = memnew(Tree);
	add_child(tree);
	tree->set_columns(2);
	tree->set_column_expand(0, true);
	tree->set_column_expand(1, false);
	tree->set_anchor(SIDE_RIGHT, ANCHOR_END);
	tree->set_anchor(SIDE_BOTTOM, ANCHOR_END);
	tree->set_allow_rmb_select(true);
	tree->set_allow_reselect(true);

	tree->set_drag_forwarding(callable_mp(this, &TaskTree::_get_drag_data_fw), callable_mp(this, &TaskTree::_can_drop_data_fw), callable_mp(this, &TaskTree::_drop_data_fw));
}

TaskTree::~TaskTree() {
	Callable on_task_changed = callable_mp(this, &TaskTree::_on_task_changed);
	if (last_selected.is_valid() && last_selected->is_connected(LW_NAME(changed), on_task_changed)) {
		last_selected->disconnect(LW_NAME(changed), on_task_changed);
	}
}

#endif // ! TOOLS_ENABLED
