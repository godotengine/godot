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

	Vector<int> collapsed_ids;

	void _draw_success_status(Object *p_obj, Rect2 p_rect);
	void _draw_running_status(Object *p_obj, Rect2 p_rect);
	void _draw_failure_status(Object *p_obj, Rect2 p_rect);
	void _item_collapsed(Object *p_obj);

protected:
	void _do_update_theme_item_cache();

	void _notification(int p_what);

	static void _bind_methods();

public:
	void update_tree(const BehaviorTreeData &p_data);
	void clear();

	BehaviorTreeView();
};

#endif // ! BEHAVIOR_TREE_VIEW_H

#endif // ! TOOLS_ENABLED
