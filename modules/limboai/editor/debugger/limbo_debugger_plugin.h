/**
 * limbo_debugger_plugin.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef LIMBO_DEBUGGER_PLUGIN_H
#define LIMBO_DEBUGGER_PLUGIN_H

#include "../../editor/debugger/behavior_tree_data.h"
#include "../../editor/debugger/behavior_tree_view.h"
#include "../../util/compat_window_wrapper.h"

#ifdef LIMBOAI_MODULE
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/typedefs.h"
#include "editor/gui/editor_spin_slider.h"
#include "editor/plugins/editor_debugger_plugin.h"
#include "editor/window_wrapper.h"
#include "scene/gui/box_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/split_container.h"
#include "scene/gui/texture_rect.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button.hpp>
#include <godot_cpp/classes/editor_debugger_plugin.hpp>
#include <godot_cpp/classes/editor_debugger_session.hpp>
#include <godot_cpp/classes/editor_spin_slider.hpp>
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/h_split_container.hpp>
#include <godot_cpp/classes/item_list.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/line_edit.hpp>
#include <godot_cpp/classes/panel_container.hpp>
#include <godot_cpp/classes/texture_rect.hpp>
#include <godot_cpp/classes/v_box_container.hpp>
#endif // LIMBOAI_GDEXTENSION

class LimboDebuggerTab : public PanelContainer {
	GDCLASS(LimboDebuggerTab, PanelContainer);

private:
	struct BTInstanceInfo {
		uint64_t instance_id = 0;
		String owner_node_path;
	};

	Vector<BTInstanceInfo> active_bt_instances;
	Ref<EditorDebuggerSession> session;
	VBoxContainer *root_vb = nullptr;
	HBoxContainer *toolbar = nullptr;
	HSplitContainer *hsc = nullptr;
	Label *info_message = nullptr;
	ItemList *bt_instance_list = nullptr;
	BehaviorTreeView *bt_view = nullptr;
	VBoxContainer *view_box = nullptr;
	HBoxContainer *alert_box = nullptr;
	TextureRect *alert_icon = nullptr;
	Label *alert_message = nullptr;
	LineEdit *filter_players = nullptr;
	Button *resource_header = nullptr;
	Button *make_floating = nullptr;
	EditorSpinSlider *update_interval = nullptr;
	CompatWindowWrapper *window_wrapper = nullptr;

	void _reset_controls();
	void _show_alert(const String &p_message);
	void _update_bt_instance_list(const Vector<BTInstanceInfo> &p_instances, const String &p_filter);
	void _bt_instance_selected(int p_idx);
	void _filter_changed(String p_text);
	void _window_visibility_changed(bool p_visible);
	void _resource_header_pressed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void start_session();
	void stop_session();
	void update_active_bt_instances(const Array &p_data);
	BehaviorTreeView *get_behavior_tree_view() const { return bt_view; }
	uint64_t get_selected_bt_instance_id();
	void update_behavior_tree(const Ref<BehaviorTreeData> &p_data);

	void setup(Ref<EditorDebuggerSession> p_session, CompatWindowWrapper *p_wrapper);
	LimboDebuggerTab();
};

class LimboDebuggerPlugin : public EditorDebuggerPlugin {
	GDCLASS(LimboDebuggerPlugin, EditorDebuggerPlugin);

private:
	static LimboDebuggerPlugin *singleton;

	HashMap<int, CompatWindowWrapper *> session_windows;

	void _window_visibility_changed(bool p_visible);

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ LimboDebuggerPlugin *get_singleton() { return singleton; }

#ifdef LIMBOAI_MODULE
	void setup_session(int p_session_id) override;
	bool has_capture(const String &p_capture) const override;
	bool capture(const String &p_message, const Array &p_data, int p_session_id) override;
#elif LIMBOAI_GDEXTENSION
	void _setup_session(int32_t p_idx) override;
	bool _has_capture(const String &p_capture) const override;
	bool _capture(const String &p_message, const Array &p_data, int32_t p_session_id) override;
#endif

	CompatWindowWrapper *get_first_session_window() const;
	int get_first_session_tab_index() const;

	LimboDebuggerPlugin();
	~LimboDebuggerPlugin();
};

#endif // LIMBO_DEBUGGER_PLUGIN_H

#endif // ! TOOLS_ENABLED
