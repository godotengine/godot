/**
 * behavior_tree_view.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "behavior_tree_view.h"

#include "../../bt/tasks/bt_task.h"
#include "../../util/limbo_compat.h"
#include "../../util/limbo_utility.h"
#include "behavior_tree_data.h"

#ifdef LIMBOAI_MODULE
#include "core/math/color.h"
#include "core/math/math_defs.h"
#include "core/object/callable_method_pointer.h"
#include "core/os/time.h"
#include "core/typedefs.h"
#include "editor/themes/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/resources/style_box.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/time.hpp>
#endif // LIMBOAI_GDEXTENSION

inline static uint64_t item_get_task_id(TreeItem *p_item) {
	return p_item->get_metadata(0);
}

inline static BTTask::Status item_get_task_status(TreeItem *p_item) {
	return VariantCaster<BTTask::Status>::cast(p_item->get_metadata(1));
}

inline static String item_get_task_type(TreeItem *p_item) {
	return ((String)p_item->get_metadata(2)).get_slicec('|', 0);
}

inline static String item_get_task_script_path(TreeItem *p_item) {
	return ((String)p_item->get_metadata(2)).get_slicec('|', 1);
}

void BehaviorTreeView::_draw_running_status(Object *p_obj, Rect2 p_rect) {
	p_rect = p_rect.grow_side(SIDE_LEFT, p_rect.get_position().x);
	theme_cache.sbf_running->draw(tree->get_canvas_item(), p_rect);
}

void BehaviorTreeView::_draw_success_status(Object *p_obj, Rect2 p_rect) {
	p_rect = p_rect.grow_side(SIDE_LEFT, p_rect.get_position().x);
	theme_cache.sbf_success->draw(tree->get_canvas_item(), p_rect);
}

void BehaviorTreeView::_draw_failure_status(Object *p_obj, Rect2 p_rect) {
	p_rect = p_rect.grow_side(SIDE_LEFT, p_rect.get_position().x);
	theme_cache.sbf_failure->draw(tree->get_canvas_item(), p_rect);
}

void BehaviorTreeView::_item_collapsed(Object *p_obj) {
	TreeItem *item = Object::cast_to<TreeItem>(p_obj);
	if (!item) {
		return;
	}
	uint64_t id = item_get_task_id(item);
	bool collapsed = item->is_collapsed();
	if (!collapsed_ids.has(id) && collapsed) {
		collapsed_ids.push_back(item_get_task_id(item));
	} else if (collapsed_ids.has(id) && !collapsed) {
		collapsed_ids.erase(id);
	}
}

void BehaviorTreeView::_item_selected() {
	TreeItem *item = tree->get_selected();
	ERR_FAIL_NULL(item);
	emit_signal(LW_NAME(task_selected), item_get_task_type(item), item_get_task_script_path(item));
}

double BehaviorTreeView::_get_editor_scale() const {
	if (Engine::get_singleton()->is_editor_hint()) {
		return EDSCALE;
	} else {
		return 1.0;
	}
}

inline void _item_set_elapsed_time(TreeItem *p_item, double p_elapsed) {
	p_item->set_text(2, rtos(Math::snapped(p_elapsed, 0.01)).pad_decimals(2));
}

void BehaviorTreeView::update_tree(const Ref<BehaviorTreeData> &p_data) {
	ERR_FAIL_COND_MSG(p_data.is_null(), "Invalid data. View won't update.");
	update_data = p_data;
	update_pending = true;
	_notification(NOTIFICATION_PROCESS);
}

void BehaviorTreeView::_update_tree(const Ref<BehaviorTreeData> &p_data) {
	// Remember selected.
	uint64_t selected_id = 0;
	if (tree->get_selected()) {
		selected_id = item_get_task_id(tree->get_selected());
	}

	if (last_root_id != 0 && p_data->tasks.size() > 0 && last_root_id == (uint64_t)p_data->tasks.get(0).id) {
		// * Update tree.
		// ! Update routine is built on assumption that the behavior tree does NOT mutate. With little work it could detect mutations.

		TreeItem *item = tree->get_root();
		int idx = 0;
		while (item) {
			ERR_FAIL_COND(idx >= p_data->tasks.size());

			const BTTask::Status current_status = (BTTask::Status)p_data->tasks.get(idx).status;
			const BTTask::Status last_status = item_get_task_status(item);
			const bool status_changed = last_status != p_data->tasks.get(idx).status;

			if (status_changed) {
				item->set_metadata(1, current_status);
				if (current_status == BTTask::SUCCESS) {
					item->set_custom_draw(0, this, LW_NAME(_draw_success_status));
					item->set_icon(1, theme_cache.icon_success);
				} else if (current_status == BTTask::FAILURE) {
					item->set_custom_draw(0, this, LW_NAME(_draw_failure_status));
					item->set_icon(1, theme_cache.icon_failure);
				} else if (current_status == BTTask::RUNNING) {
					item->set_custom_draw(0, this, LW_NAME(_draw_running_status));
					item->set_icon(1, theme_cache.icon_running);
				} else {
					item->set_custom_draw(0, this, LW_NAME(_draw_fresh));
					item->set_icon(1, nullptr);
				}
			}

			if (status_changed || current_status == BTTask::RUNNING) {
				_item_set_elapsed_time(item, p_data->tasks.get(idx).elapsed_time);
			}

			if (item->get_first_child()) {
				item = item->get_first_child();
			} else if (item->get_next()) {
				item = item->get_next();
			} else {
				while (item) {
					item = item->get_parent();
					if (item && item->get_next()) {
						item = item->get_next();
						break;
					}
				}
			}

			idx += 1;
		}
		ERR_FAIL_COND(idx != p_data->tasks.size());
	} else {
		// * Create new tree.

		last_root_id = p_data->tasks.size() > 0 ? p_data->tasks.get(0).id : 0;

		tree->clear();
		TreeItem *parent = nullptr;
		List<Pair<TreeItem *, int>> parents;
		for (const BehaviorTreeData::TaskData &task_data : p_data->tasks) {
			// Figure out parent.
			parent = nullptr;
			if (parents.size()) {
				Pair<TreeItem *, int> &p = parents.get(0);
				parent = p.first;
				if (!(--p.second)) {
					// No children left, remove it.
					parents.pop_front();
				}
			}

			TreeItem *item = tree->create_item(parent);
			// Do this first because it resets properties of the cell...
			item->set_cell_mode(0, TreeItem::CELL_MODE_CUSTOM);
			item->set_cell_mode(1, TreeItem::CELL_MODE_ICON);

			item->set_metadata(0, task_data.id);
			item->set_metadata(1, task_data.status);
			item->set_metadata(2, task_data.type_name + String("|") + task_data.script_path);

			item->set_text(0, task_data.name);
			if (task_data.is_custom_name) {
				item->set_custom_font(0, theme_cache.font_custom_name);
			}

			item->set_text_alignment(2, HORIZONTAL_ALIGNMENT_RIGHT);
			_item_set_elapsed_time(item, task_data.elapsed_time);

			String cors = (task_data.script_path.is_empty()) ? task_data.type_name : task_data.script_path;
			item->set_icon(0, LimboUtility::get_singleton()->get_task_icon(cors));
			item->set_icon_max_width(0, 16 * _get_editor_scale()); // Force user icon size.

			if (task_data.status == BTTask::SUCCESS) {
				item->set_custom_draw(0, this, LW_NAME(_draw_success_status));
				item->set_icon(1, theme_cache.icon_success);
			} else if (task_data.status == BTTask::FAILURE) {
				item->set_custom_draw(0, this, LW_NAME(_draw_failure_status));
				item->set_icon(1, theme_cache.icon_failure);
			} else if (task_data.status == BTTask::RUNNING) {
				item->set_custom_draw(0, this, LW_NAME(_draw_running_status));
				item->set_icon(1, theme_cache.icon_running);
			}

			if (task_data.id == selected_id) {
				tree->set_selected(item, 0);
			}

			if (collapsed_ids.has(task_data.id)) {
				item->set_collapsed(true);
			}

			// Add in front of parents stack if children are expected.
			if (task_data.num_children) {
				parents.push_front(Pair<TreeItem *, int>(item, task_data.num_children));
			}
		}
	}
}

void BehaviorTreeView::clear() {
	tree->clear();
	collapsed_ids.clear();
	last_root_id = 0;
}

void BehaviorTreeView::_do_update_theme_item_cache() {
	theme_cache.icon_running = LimboUtility::get_singleton()->get_task_icon("LimboExtraClock");
	theme_cache.icon_success = LimboUtility::get_singleton()->get_task_icon("BTAlwaysSucceed");
	theme_cache.icon_failure = LimboUtility::get_singleton()->get_task_icon("BTAlwaysFail");

	theme_cache.font_custom_name = get_theme_font(LW_NAME(bold), LW_NAME(EditorFonts));

	Color running_border = Color::html("#fea900");
	Color running_fill = Color(running_border, 0.1);
	Color success_border = Color::html("#2fa139");
	Color success_fill = Color(success_border, 0.1);
	Color failure_border = Color::html("#cd3838");
	Color failure_fill = Color(failure_border, 0.1);

	theme_cache.sbf_running.instantiate();
	theme_cache.sbf_running->set_border_color(running_border);
	theme_cache.sbf_running->set_bg_color(running_fill);
	theme_cache.sbf_running->set_border_width(SIDE_LEFT, 4.0);
	theme_cache.sbf_running->set_border_width(SIDE_RIGHT, 4.0);

	theme_cache.sbf_success.instantiate();
	theme_cache.sbf_success->set_border_color(success_border);
	theme_cache.sbf_success->set_bg_color(success_fill);
	theme_cache.sbf_success->set_border_width(SIDE_LEFT, 4.0);
	theme_cache.sbf_success->set_border_width(SIDE_RIGHT, 4.0);

	theme_cache.sbf_failure.instantiate();
	theme_cache.sbf_failure->set_border_color(failure_border);
	theme_cache.sbf_failure->set_bg_color(failure_fill);
	theme_cache.sbf_failure->set_border_width(SIDE_LEFT, 4.0);
	theme_cache.sbf_failure->set_border_width(SIDE_RIGHT, 4.0);

	double extra_spacing = 0.0;
	if (Engine::get_singleton()->is_editor_hint()) {
		extra_spacing = EDITOR_GET("interface/theme/additional_spacing");
		extra_spacing *= 2.0;
	}
	tree->set_column_clip_content(0, true);
	tree->set_column_custom_minimum_width(1, 18 * _get_editor_scale());

	Ref<Font> font = tree->get_theme_font(LW_NAME(font));
	int font_size = tree->get_theme_font_size(LW_NAME(font_size));
	int timings_size = font->get_string_size("00.00", HORIZONTAL_ALIGNMENT_RIGHT, -1, font_size).x + 16 + extra_spacing;
	tree->set_column_custom_minimum_width(2, timings_size * _get_editor_scale());
}

void BehaviorTreeView::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			tree->connect(LW_NAME(item_collapsed), callable_mp(this, &BehaviorTreeView::_item_collapsed));
			tree->connect(LW_NAME(item_selected), callable_mp(this, &BehaviorTreeView::_item_selected));
		} break;
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process(is_visible_in_tree());
		} break;
		case NOTIFICATION_PROCESS: {
			int ticks_msec = Time::get_singleton()->get_ticks_msec();
			if (update_pending && (ticks_msec - last_update_msec) >= update_interval_msec) {
				_update_tree(update_data);
				update_pending = false;
				last_update_msec = ticks_msec;
			}
		} break;
	}
}

void BehaviorTreeView::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_draw_running_status"), &BehaviorTreeView::_draw_running_status);
	ClassDB::bind_method(D_METHOD("_draw_success_status"), &BehaviorTreeView::_draw_success_status);
	ClassDB::bind_method(D_METHOD("_draw_failure_status"), &BehaviorTreeView::_draw_failure_status);
	ClassDB::bind_method(D_METHOD("_item_collapsed"), &BehaviorTreeView::_item_collapsed);
	ClassDB::bind_method(D_METHOD("update_tree", "behavior_tree_data"), &BehaviorTreeView::update_tree);
	ClassDB::bind_method(D_METHOD("clear"), &BehaviorTreeView::clear);

	ClassDB::bind_method(D_METHOD("set_update_interval_msec", "interval_msec"), &BehaviorTreeView::set_update_interval_msec);
	ClassDB::bind_method(D_METHOD("get_update_interval_msec"), &BehaviorTreeView::get_update_interval_msec);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_interval_msec"), "set_update_interval_msec", "get_update_interval_msec");

	ADD_SIGNAL(MethodInfo("task_selected", PropertyInfo(Variant::STRING, "type_name"), PropertyInfo(Variant::STRING, "script_path")));
}

BehaviorTreeView::BehaviorTreeView() {
	tree = memnew(Tree);
	add_child(tree);
	tree->set_columns(3); // task | status icon | elapsed
	tree->set_column_expand(0, true);
	tree->set_column_expand(1, false);
	tree->set_column_expand(2, false);
	tree->set_anchor(SIDE_RIGHT, ANCHOR_END);
	tree->set_anchor(SIDE_BOTTOM, ANCHOR_END);
}

#endif // TOOLS_ENABLED
