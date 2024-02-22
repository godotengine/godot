/**
 * task_tree.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "../bt/behavior_tree.h"

#ifdef LIMBOAI_MODULE
#include "scene/gui/control.h"
#include "scene/gui/tree.h"
#include "scene/resources/style_box_flat.h"

#define RECT_CACHE_KEY ObjectID
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/style_box_flat.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/tree.hpp>
#include <godot_cpp/templates/hash_map.hpp>

#define RECT_CACHE_KEY uint64_t
#endif // LIMBOAI_GDEXTENSION

class TaskTree : public Control {
	GDCLASS(TaskTree, Control);

private:
	Tree *tree;
	Ref<BehaviorTree> bt;
	Ref<BTTask> last_selected;
	bool editable;
	bool updating_tree;
	HashMap<RECT_CACHE_KEY, Rect2> probability_rect_cache;

	struct ThemeCache {
		Ref<Font> comment_font;
		Ref<Font> name_font;
		Ref<Font> custom_name_font;
		Ref<Font> normal_name_font;
		Ref<Font> probability_font;

		double name_font_size = 18.0;
		double probability_font_size = 16.0;

		Ref<Texture2D> task_warning_icon;

		Color comment_color;
		Color probability_font_color;

		Ref<StyleBoxFlat> probability_bg;
	} theme_cache;

	TreeItem *_create_tree(const Ref<BTTask> &p_task, TreeItem *p_parent, int p_idx = -1);
	void _update_item(TreeItem *p_item);
	void _update_tree();
	TreeItem *_find_item(const Ref<BTTask> &p_task) const;

	void _on_item_selected();
	void _on_item_activated();
	void _on_item_collapsed(Object *p_obj);
	void _on_item_mouse_selected(const Vector2 &p_pos, MouseButton p_button_index);
	void _on_task_changed();

	Variant _get_drag_data_fw(const Point2 &p_point);
	bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data) const;
	void _drop_data_fw(const Point2 &p_point, const Variant &p_data);

	void _draw_probability(Object *item_obj, Rect2 rect);

protected:
	virtual void _do_update_theme_item_cache();

	void _notification(int p_what);
	static void _bind_methods();

public:
	void load_bt(const Ref<BehaviorTree> &p_behavior_tree);
	void unload();
	Ref<BehaviorTree> get_bt() const { return bt; }
	void update_tree() { _update_tree(); }
	void update_task(const Ref<BTTask> &p_task);
	Ref<BTTask> get_selected() const;
	void deselect();

	Rect2 get_selected_probability_rect() const;
	double get_selected_probability_weight() const;
	double get_selected_probability_percent() const;
	bool selected_has_probability() const;

	virtual bool editor_can_reload_from_file() { return false; }

	TaskTree();
	~TaskTree();
};

#endif // ! TOOLS_ENABLED
