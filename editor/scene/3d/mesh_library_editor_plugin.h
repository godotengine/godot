/**************************************************************************/
/*  mesh_library_editor_plugin.h                                          */
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

#pragma once

#include "editor/docks/editor_dock.h"
#include "editor/plugins/editor_plugin.h"

class ConfirmationDialog;
class EditorFileDialog;
class EditorInspector;
class EditorZoomWidget;
class FilterLineEdit;
class HSplitContainer;
class ItemList;
class MenuButton;
class MeshInstance3D;
class MeshLibrary;
class Timer;

class MeshLibraryEditor : public EditorDock {
	GDCLASS(MeshLibraryEditor, EditorDock);

	class MeshLibraryItem : public RefCounted {
		GDCLASS(MeshLibraryItem, RefCounted);

		friend MeshLibraryEditor;

		Ref<MeshLibrary> mesh_library;
		int mesh_id = -1;

	protected:
		bool _set(const StringName &p_name, const Variant &p_value);
		bool _get(const StringName &p_name, Variant &r_ret) const;
		void _get_property_list(List<PropertyInfo> *p_list) const;
	};

	enum {
		MENU_OPTION_ADD_ITEM,
		MENU_OPTION_REMOVE_ITEM,
		MENU_OPTION_UPDATE_FROM_SCENE,
		MENU_OPTION_IMPORT_FROM_SCENE,
		MENU_OPTION_IMPORT_FROM_SCENE_APPLY_XFORMS
	};

	const float UPDATE_ITEMS_DELAY_TIMEOUT = 0.1;

	Ref<MeshLibrary> mesh_library;
	Ref<MeshLibraryItem> mesh_library_item; // Keep a reference to avoid crashes.
	int selected_item = -1;

	Button *add_item = nullptr;
	Button *remove_item = nullptr;
	MenuButton *import_scene = nullptr;
	FilterLineEdit *search_box = nullptr;
	EditorZoomWidget *zoom_widget = nullptr;
	HSplitContainer *item_split = nullptr;
	ItemList *mesh_items = nullptr;
	EditorInspector *inspector = nullptr;
	Label *empty_lib = nullptr;
	Timer *update_items_delay = nullptr;

	EditorFileDialog *file = nullptr;
	ConfirmationDialog *cd_update = nullptr;

	bool apply_xforms = false;
	bool import_update = false;

	void _update_mesh_items(bool p_reselect = true, Ref<MeshLibrary> p_lib_check = Ref<MeshLibrary>());
	void _update_resource_preview(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, int p_idx);

	void _select_item(int p_id, Ref<MeshLibrary> p_lib_check = Ref<MeshLibrary>());
	void _select_item_and_button(int p_id, Ref<MeshLibrary> p_lib_check = Ref<MeshLibrary>());
	void _select_prev_item_and_button(int p_id, Ref<MeshLibrary> p_lib_check = Ref<MeshLibrary>());

	void _mesh_items_cbk(int p_idx);
	void _mesh_items_input(const Ref<InputEvent> &p_event);

	void _menu_cbk(int p_option);
	void _menu_update_confirm(bool p_apply_xforms);

	void _import_scene_cbk(const String &p_str);
	static void _import_scene(Node *p_scene, Ref<MeshLibrary> p_library, bool p_merge, bool p_apply_xforms);
	static void _import_scene_parse_node(Ref<MeshLibrary> p_library, HashMap<int, MeshInstance3D *> &p_mesh_instances, Node *p_node, bool p_merge, bool p_apply_xforms);

	void _icon_size_changed(float p_value);

private:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void edit(const Ref<MeshLibrary> &p_mesh_library);
	static Error update_library_file(Node *p_base_scene, Ref<MeshLibrary> ml, bool p_merge = true, bool p_apply_xforms = false);

	MeshLibraryEditor();
};

class MeshLibraryEditorPlugin : public EditorPlugin {
	GDCLASS(MeshLibraryEditorPlugin, EditorPlugin);

	static inline MeshLibraryEditorPlugin *singleton = nullptr;

	MeshLibraryEditor *mesh_library_editor = nullptr;

public:
	_FORCE_INLINE_ static MeshLibraryEditorPlugin *get_singleton() { return singleton; }

	virtual void edit(Object *p_node) override;
	virtual bool handles(Object *p_node) const override;
	virtual void make_visible(bool p_visible) override;

	void open_editor();

	MeshLibraryEditorPlugin();
	~MeshLibraryEditorPlugin() { singleton = nullptr; }
};
