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
#include "scene/gui/box_container.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/scroll_container.h"

class EditorFileDialog;
class LineEdit;
class Button;
class PanelContainer;
class HBoxContainer;
class TabBar;
class ResourceBundle;
class ResourceBundleTab;
class ResourceBundleTable;

class ResourceBundleEditor : public EditorDock {
	GDCLASS(ResourceBundleEditor, EditorDock);

	friend class ResourceBundleEditorPlugin;

	inline static ResourceBundleEditor *singleton = nullptr;

	Vector<Ref<ResourceBundle>> bundles;

	EditorFileDialog *file_dialog = nullptr;

	VBoxContainer *tab_container = nullptr;

	PanelContainer *tabbar_panel = nullptr;
	HBoxContainer *tabbar_container = nullptr;

	TabBar *bundle_tabs = nullptr;
	Button *bundle_tab_add = nullptr;
	Control *bundle_tab_add_ph = nullptr;

	void _bundle_tabs_resized();
	void _add_tab(const String &p_bundle = "", const String &p_schema = "");

protected:
	void _notification(int p_what);

public:
	static ResourceBundleEditor *get_singleton() { return singleton; }

	bool make_bundle(const String &p_path);
	bool remove_bundle(const String &p_path);

	ResourceBundleEditor();
};

class ResourceBundleTab : public VBoxContainer {
	GDSOFTCLASS(ResourceBundleTab, VBoxContainer);

	ResourceBundleTable *table = nullptr;

public:
	ResourceBundleTab(const String &p_bundle, const String &p_schema);
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
