/*************************************************************************/
/*  gdnative_library_editor_plugin.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GDNATIVE_LIBRARY_EDITOR_PLUGIN_H
#define GDNATIVE_LIBRARY_EDITOR_PLUGIN_H

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "gdnative.h"

class GDNativeLibraryEditor : public Control {
	GDCLASS(GDNativeLibraryEditor, Control);

	struct NativePlatformConfig {
		String name;
		String library_extension;
		List<String> entries;
	};

	struct TargetConfig {
		String library;
		Array dependencies;
	};

	enum ItemButton {
		BUTTON_SELECT_LIBRARY,
		BUTTON_CLEAR_LIBRARY,
		BUTTON_SELECT_DEPENDENCES,
		BUTTON_CLEAR_DEPENDENCES,
		BUTTON_ERASE_ENTRY,
		BUTTON_MOVE_UP,
		BUTTON_MOVE_DOWN,
	};

	Tree *tree;
	MenuButton *filter;
	EditorFileDialog *file_dialog;
	ConfirmationDialog *new_architecture_dialog;
	LineEdit *new_architecture_input;
	Set<String> collapsed_items;

	String showing_platform;
	Ref<GDNativeLibrary> library;
	Map<String, NativePlatformConfig> platforms;
	Map<String, TargetConfig> entry_configs;

protected:
	static void _bind_methods();
	void _update_tree();
	void _on_item_button(Object *item, int column, int id);
	void _on_library_selected(const String &file);
	void _on_dependencies_selected(const PoolStringArray &files);
	void _on_filter_selected(int id);
	void _on_item_collapsed(Object *p_item);
	void _on_item_activated();
	void _on_create_new_entry();
	void _set_target_value(const String &section, const String &target, Variant file);
	void _erase_entry(const String &platform, const String &entry);
	void _move_entry(const String &platform, const String &entry, int dir);
	void _translate_to_config_file();

public:
	void edit(Ref<GDNativeLibrary> p_library);

	GDNativeLibraryEditor();
};

class GDNativeLibraryEditorPlugin : public EditorPlugin {
	GDCLASS(GDNativeLibraryEditorPlugin, EditorPlugin);

	GDNativeLibraryEditor *library_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "GDNativeLibrary"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	GDNativeLibraryEditorPlugin(EditorNode *p_node);
};
#endif
#endif // GDNATIVE_LIBRARY_EDITOR_PLUGIN_H
