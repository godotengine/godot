/**************************************************************************/
/*  resource_bundle_editor_plugin.h                                       */
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
#include "scene/gui/scroll_container.h"

class EditorFileDialog;
class Label;
class TabContainer;
class ResourceBundle;
class ResourceBundleTab;
class ResourceBundleTable;

class ResourceBundleEditor : public EditorDock {
	GDCLASS(ResourceBundleEditor, EditorDock);

	friend class ResourceBundleEditorPlugin;

	inline static ResourceBundleEditor *singleton = nullptr;

	Vector<ResourceBundleTab *> bundle_tabs;

	EditorFileDialog *file_dialog = nullptr;
	TabContainer *bundle_container = nullptr;
	Label *info_label = nullptr;

	void _update_info_label();

	void _bundle_tab_close_pressed(int p_tab);
	int _find_tab_index(const String &p_path) const;

	void _add_tab(const Ref<ResourceBundle> &p_bundle);
	void _add_or_focus_tab(const Ref<ResourceBundle> &p_bundle);
	void _remove_tab(const String &p_path);
	void _remove_tab(ResourceBundleTab *p_tab);

protected:
	void _notification(int p_what);

public:
	static ResourceBundleEditor *get_singleton() { return singleton; }

	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;

	bool make_bundle(const String &p_path);
	bool remove_bundle(const String &p_path);
	void edit(Object *p_object);

	ResourceBundleEditor();
};

class ResourceBundleTab : public MarginContainer {
	GDSOFTCLASS(ResourceBundleTab, MarginContainer);

	String tab_name;

	Ref<ResourceBundle> bundle;
	ResourceBundleTable *table = nullptr;

	Label *label = nullptr;

public:
	void set_tab_name(const String &p_name);
	String get_tab_name() const;

	void set_bundle(const Ref<ResourceBundle> &p_bundle);
	Ref<ResourceBundle> get_bundle() const;

	ResourceBundleTab();
};

class ResourceBundleTable : public ScrollContainer {
	GDSOFTCLASS(ResourceBundleTable, ScrollContainer);

public:
	ResourceBundleTable();
};

class ResourceBundleEditorPlugin : public EditorPlugin {
	GDCLASS(ResourceBundleEditorPlugin, EditorPlugin);

	ResourceBundleEditor *bundle_editor = nullptr;

public:
	virtual String get_plugin_name() const override { return "Bundle"; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ResourceBundleEditorPlugin();
};
