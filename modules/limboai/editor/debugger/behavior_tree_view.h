/**
 * behavior_tree_view.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef BEHAVIOR_TREE_VIEW_H
#define BEHAVIOR_TREE_VIEW_H

#include "behavior_tree_data.h"

#ifdef LIMBOAI_MODULE
#include "scene/gui/control.h"
#include "scene/gui/tree.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/texture.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/style_box_flat.hpp>
#include <godot_cpp/classes/tree.hpp>
#endif // LIMBOAI_GDEXTENSION

class BehaviorTreeView : public Control {
	GDCLASS(BehaviorTreeView, Control);

private:
	Tree *tree;

	struct ThemeCache {
		Ref<StyleBoxFlat> sbf_running;
		Ref<StyleBoxFlat> sbf_success;
		Ref<StyleBoxFlat> sbf_failure;

		Ref<Texture2D> icon_running;
		Ref<Texture2D> icon_success;
		Ref<Texture2D> icon_failure;

		Ref<Font> font_custom_name;
	} theme_cache;

	Vector<uint64_t> collapsed_ids;
	uint64_t last_root_id = 0;

	int last_update_msec = 0;
	int update_interval_msec = 0;
	Ref<BehaviorTreeData> update_data;
	bool update_pending = false;

	void _draw_success_status(Object *p_obj, Rect2 p_rect);
	void _draw_running_status(Object *p_obj, Rect2 p_rect);
	void _draw_failure_status(Object *p_obj, Rect2 p_rect);
	void _draw_fresh(Object *p_obj, Rect2 p_rect) {}
	void _item_collapsed(Object *p_obj);
	double _get_editor_scale() const;

	void _update_tree(const Ref<BehaviorTreeData> &p_data);

protected:
	void _do_update_theme_item_cache();

	void _notification(int p_what);

	static void _bind_methods();

public:
	void clear();
	void update_tree(const Ref<BehaviorTreeData> &p_data);

	void set_update_interval_msec(int p_milliseconds) { update_interval_msec = p_milliseconds; }
	int get_update_interval_msec() const { return update_interval_msec; }

	BehaviorTreeView();
};

#endif // ! BEHAVIOR_TREE_VIEW_H

#endif // ! TOOLS_ENABLED
