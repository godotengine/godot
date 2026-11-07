/**************************************************************************/
/*  import_dock.h                                                         */
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

#include "core/object/worker_thread_pool.h"
#include "editor/docks/editor_dock.h"
#include "editor/inspector/editor_inspector.h"

class ConfirmationDialog;
class MenuButton;
class OptionButton;
class ProgressBar;
class ResourceImporter;
class Timer;
class VBoxContainer;

class ImportDockParameters : public Object {
	GDCLASS(ImportDockParameters, Object);

public:
	HashMap<StringName, Variant> values;
	List<PropertyInfo> properties;
	Ref<ResourceImporter> importer;
	Vector<String> paths;
	HashSet<StringName> checked;
	bool checking = false;
	bool skip = false;
	String base_options_path;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void update();
};

class ImportDockLoader : public Object {
	GDCLASS(ImportDockLoader, Object);

private:
	WorkerThreadPool::TaskID task_id = WorkerThreadPool::INVALID_TASK_ID;
	SafeFlag exit_requested;

	void _process();

public:
	Vector<String> file_paths;

	HashMap<String, Dictionary> value_frequency;
	bool success = false;
	String importer_name;
	String options_base_path;
	Ref<ConfigFile> first_loaded_file;

	SafeNumeric<int> files_loaded;

	void run_blocking();

	void start();
	void stop();

	bool is_running();
	void tick();
	void wait();

	void reset();
};

class ImportDock : public EditorDock {
	GDCLASS(ImportDock, EditorDock);

	Label *imported = nullptr;
	OptionButton *import_as = nullptr;
	MenuButton *preset = nullptr;
	EditorInspector *import_opts = nullptr;

	ConfirmationDialog *reimport_confirm = nullptr;
	Label *cleanup_warning = nullptr;
	Label *label_warning = nullptr;
	Button *import = nullptr;
	List<String> need_cleanup;
	bool dirty = false;

	Control *advanced_spacer = nullptr;
	Button *advanced = nullptr;

	ImportDockParameters *params = nullptr;

	VBoxContainer *content = nullptr;
	Label *select_a_resource = nullptr;

	VBoxContainer *loader_vbox = nullptr;
	ProgressBar *loader_bar = nullptr;
	Label *loader_file_count_label = nullptr;
	Timer *loader_min_display_timer = nullptr;
	uint64_t loader_start_frame = 0;
	ImportDockLoader *loader = nullptr;

	bool _finish_set_edit(const Ref<ConfigFile> &p_config = Ref<ConfigFile>());
	void _preset_selected(int p_idx);
	void _importer_selected(int i_idx);
	void _update_options(const String &p_path, const Ref<ConfigFile> &p_config = Ref<ConfigFile>());
	void _update_preset_menu();
	void _add_keep_import_option(const String &p_importer_name);

	void _property_edited(const StringName &p_prop);
	void _property_toggled(const StringName &p_prop, bool p_checked);
	void _set_dirty(bool p_dirty);
	void _reimport_pressed();
	void _reimport_attempt();
	void _reimport_and_cleanup();
	void _reimport();

	void _advanced_options();
	enum {
		ITEM_SET_AS_DEFAULT = 100,
		ITEM_LOAD_DEFAULT,
		ITEM_CLEAR_DEFAULT,
	};

private:
	static ImportDock *singleton;

public:
	static ImportDock *get_singleton() { return singleton; }

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_edit_path(const String &p_path);
	void set_edit_multiple_paths(const Vector<String> &p_paths);
	void reimport_resources(const Vector<String> &p_paths);
	void initialize_import_options() const;
	void clear();

	ImportDock();
	~ImportDock();
};
