/**************************************************************************/
/*  editor_scene_tabs.h                                                   */
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

#ifndef EDITOR_SCENE_TABS_H
#define EDITOR_SCENE_TABS_H

#include "core/io/config_file.h"
#include "editor/editor_nav_tabs.h"
#include "editor/editor_tab.h"
#include "scene/gui/margin_container.h"

class Button;
class HBoxContainer;
class Panel;
class PanelContainer;
class PopupMenu;
class TabBar;
class TextureRect;

enum class MainEditorTabContent {
	SCENES_ONLY,
	ALL
};

class EditorSceneTabs : public MarginContainer {
	GDCLASS(EditorSceneTabs, MarginContainer);

	static EditorSceneTabs *singleton;
	static MainEditorTabContent tab_content_setting;

	enum MenuOptions {
		CLOSE_OTHERS = 99990,
		CLOSE_RIGHT = 99991,
		CLOSE_ALL = 99992
	};

	PanelContainer *tabbar_panel = nullptr;
	HBoxContainer *tabbar_container = nullptr;

	TabBar *scene_tabs = nullptr;
	PopupMenu *scene_tabs_context_menu = nullptr;
	Button *scene_tab_add = nullptr;
	Control *scene_tab_add_ph = nullptr;

	Panel *tab_preview_panel = nullptr;
	TextureRect *tab_preview = nullptr;

	EditorNavTabs *nav_tabs = nullptr;

	void _scene_tab_changed(int p_tab);
	void _scene_tab_button_pressed(int p_tab);
	void _scene_tab_closed_pressed(int p_tab);
	void _scene_tab_hovered(int p_tab);
	void _scene_tab_exit();
	void _scene_tab_input(const Ref<InputEvent> &p_input);
	void _scene_tabs_resized();

	void _update_tab_titles();
	void _reposition_active_tab(int p_to_index);
	void _update_context_menu();
	void _disable_menu_option_if(int p_option, bool p_condition);

	void _tab_preview_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_udata);

	void _global_menu_scene(const Variant &p_tag);
	void _global_menu_new_window(const Variant &p_tag);

	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

	int _current_tab_index = -1;
	int _selected_index = -1;
	Vector<EditorTab *> _tabs;
	Vector<EditorTab *> _tabs_to_close;

	int _get_tab_index(const EditorTab *p_tab) const;
	int _get_tab_index_by_resource_path(const String &p_resource_path) const;
	int _get_tab_index_by_name(const String &p_name) const;
	int _get_most_recent_tab_index() const;
	void _set_current_tab_index(int p_tab);

	void _context_menu_id_pressed(int p_option);

	void _memdel_edit_tab_deferred(EditorTab *p_tab);

protected:
	void _notification(int p_what);
	virtual void unhandled_key_input(const Ref<InputEvent> &p_event) override;
	static void _bind_methods();

public:
	static EditorSceneTabs *get_singleton() { return singleton; }
	static MainEditorTabContent get_tab_content_setting() { return tab_content_setting; }

	EditorTab *add_tab();
	EditorTab *get_current_tab() const;
	int get_current_tab_index() const;
	void remove_tab(EditorTab *p_tab, bool p_user_removal);
	void select_tab(const EditorTab *p_tab);
	void select_tab_index(int p_index);
	EditorTab *get_tab_by_state(Variant p_state);

	void update_scene_tabs();
	void add_extra_button(Button *p_button);
	void cancel_close_process();
	void save_tabs_layout(Ref<ConfigFile> p_layout);
	void restore_tabs_layout(Ref<ConfigFile> p_layout);

	EditorSceneTabs();
	~EditorSceneTabs();
};

#endif // EDITOR_SCENE_TABS_H
