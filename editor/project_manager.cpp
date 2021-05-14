/*************************************************************************/
/*  project_manager.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "project_manager.h"

#include "core/io/config_file.h"
#include "core/io/resource_saver.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/translation.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "editor_themes.h"
#include "scene/gui/center_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"

// Used to test for GLES3 support.
#ifndef SERVER_ENABLED
#include "drivers/gles3/rasterizer_gles3.h"
#endif

static inline String get_project_key_from_path(const String &dir) {
	return dir.replace("/", "::");
}

class CreateProjectDialog : public ConfirmationDialog {
	GDCLASS(CreateProjectDialog, ConfirmationDialog);

private:
	Ref<ButtonGroup> renderer_button_group;

	// We don't need to store a reference to the gles3 button because if it's not
	// gles2 then it must be gles3.
	CheckBox *gles2_checkbox;

	LineEdit *project_name_line_edit;
	LineEdit *project_path_line_edit;
	Button *project_path_picker;
	CheckBox *use_custom_project_path_checkbox;

	// Used to warn the user of any invalid settings (such as an invalid project path).
	Label *error_label;

	String default_project_folder_path;

	void _use_custom_project_path_toggled(bool p_use_custom_project_path) {
		project_path_line_edit->set_editable(p_use_custom_project_path);
		project_path_picker->set_disabled(!p_use_custom_project_path);
		if (!p_use_custom_project_path) {
			_set_path(default_project_folder_path.plus_file(project_name_line_edit->get_text()));
		}
	}

	void _project_name_line_edit_changed(const String &p_new_text) {
		if (!use_custom_project_path_checkbox->is_pressed()) {
			_set_path(default_project_folder_path.plus_file(p_new_text));
		}
	}

	// Updates `project_path_line_edit` and validates the new path,
	// because calling `set_text()` doesn't emit `text_changed()`.
	void _set_path(const String &p_path) {
		project_path_line_edit->set_text(p_path);
		_validate_path(p_path);
	}

	void _validate_path(const String &p_path) {
		bool path_valid = true;

		DirAccessRef dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		String project_folder_path = p_path.left(p_path.find_last("/"));
		String project_folder_name = p_path.right(p_path.find_last("/") + 1);
		if (!dir_access->dir_exists(project_folder_path)) {
			error_label->set_text(vformat(TTR("Path does not exist: \"%s\""), project_folder_path));
			path_valid = false;
		}
		if (!project_folder_name.is_valid_filename()) {
			error_label->set_text(vformat(TTR("Invalid project folder name: \"%s\""), project_folder_name));
			path_valid = false;
		}

		// Make sure (if the project folder already exist),
		// that it doesn't already contain a Godot project.
		if (dir_access->dir_exists(p_path)) {
			PoolStringArray reserved_files;
			reserved_files.push_back("project.godot");
			reserved_files.push_back("default_env.tres");
			reserved_files.push_back("icon.png");
			for (int i = 0; i < reserved_files.size(); i++) {
				const String &reserved_file = reserved_files[i];
				if (dir_access->file_exists(p_path.plus_file(reserved_file))) {
					error_label->set_text(vformat(TTR("Folder contains files that would be overwritten: \"%s\""), reserved_file));
					path_valid = false;
					break;
				}
			}
		}

		error_label->set_modulate(Color(1.0f, 1.0f, 1.0f, path_valid ? 0.0f : 1.0f));
		// Disable the "Create & Edit" button until all errors have been resolved.
		get_ok()->set_disabled(!path_valid);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method("_project_name_line_edit_changed", &CreateProjectDialog::_project_name_line_edit_changed);
		ClassDB::bind_method("_use_custom_project_path_toggled", &CreateProjectDialog::_use_custom_project_path_toggled);
		ClassDB::bind_method("_set_path", &CreateProjectDialog::_set_path);
		ClassDB::bind_method("_validate_path", &CreateProjectDialog::_validate_path);
		ClassDB::bind_method("create_project_popup", &CreateProjectDialog::create_project_popup);
	}

public:
	String get_new_project_name() {
		return project_name_line_edit->get_text();
	}

	String get_new_project_path() {
		return project_path_line_edit->get_text();
	}

	bool should_new_project_use_gles2() {
		return renderer_button_group->get_pressed_button() == gles2_checkbox;
	}

	void create_project_popup() {
		popup_centered();

		project_name_line_edit->select_all();
		project_name_line_edit->grab_focus();
	}

	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			project_path_picker->set_icon(get_icon("Load", "EditorIcons"));
			error_label->add_color_override("font_color", error_label->get_color("error_color", "Editor"));
		}
	}

	CreateProjectDialog() {
		String default_project_name = TTR("New Godot Project");
		default_project_folder_path = EDITOR_GET("filesystem/directories/default_project_path");
		default_project_folder_path.replace("\\", "/");

		set_title(TTR("Create new project"));
		get_ok()->set_text(TTR("Create & Edit"));

		VBoxContainer *vbox = memnew(VBoxContainer);
		add_child(vbox);

		Label *project_name_label = memnew(Label);
		project_name_label->set_text(TTR("Project name:"));
		vbox->add_child(project_name_label);

		project_name_line_edit = memnew(LineEdit);
		project_name_line_edit->set_text(default_project_name);
		project_name_line_edit->connect("text_changed", this, "_project_name_line_edit_changed");
		vbox->add_child(project_name_line_edit);

		HBoxContainer *project_path_label_hbox = memnew(HBoxContainer);
		vbox->add_child(project_path_label_hbox);

		Label *project_path_label = memnew(Label);
		project_path_label->set_text(TTR("Project path:"));
		project_path_label_hbox->add_child(project_path_label);

		use_custom_project_path_checkbox = memnew(CheckBox);
		use_custom_project_path_checkbox->set_text(TTR("Use custom"));
		use_custom_project_path_checkbox->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_END);
		use_custom_project_path_checkbox->connect("toggled", this, "_use_custom_project_path_toggled");
		project_path_label_hbox->add_child(use_custom_project_path_checkbox);

		HBoxContainer *project_path_line_edit_hbox = memnew(HBoxContainer);
		vbox->add_child(project_path_line_edit_hbox);

		project_path_line_edit = memnew(LineEdit);
		// Text gets set farther down, after `error_label` is created.
		project_path_line_edit->set_editable(false);
		project_path_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
		project_path_line_edit->connect("text_changed", this, "_validate_path");
		project_path_line_edit_hbox->add_child(project_path_line_edit);

		FileDialog *project_path_file_dialog = memnew(FileDialog);
		project_path_file_dialog->set_access(FileDialog::ACCESS_FILESYSTEM);
		project_path_file_dialog->set_mode(FileDialog::MODE_OPEN_DIR);
		project_path_file_dialog->set_current_dir(default_project_folder_path);
		project_path_file_dialog->connect("dir_selected", this, "_set_path");
		add_child(project_path_file_dialog);

		project_path_picker = memnew(Button);
		project_path_picker->set_disabled(true);
		project_path_picker->connect("pressed", project_path_file_dialog, "popup_centered_ratio");
		project_path_line_edit_hbox->add_child(project_path_picker);

		error_label = memnew(Label);
		error_label->set_autowrap(true);
		error_label->set_align(Label::ALIGN_CENTER);
		error_label->set_modulate(Color(1.0f, 1.0f, 1.0f, 0.0f));
		vbox->add_child(error_label);

		// Validate, in case `default_project_folder_path` got set to something weird.
		// `error_label` has to be created before `_set_path()` can be called,
		// which is why it's here and not farther up.
		_set_path(default_project_folder_path.plus_file(default_project_name));

		Label *renderer_label = memnew(Label);
		renderer_label->set_text(TTR("Renderer:"));
		vbox->add_child(renderer_label);

		HBoxContainer *renderer_hbox = memnew(HBoxContainer);
		vbox->add_child(renderer_hbox);

		renderer_button_group.instance();

		VBoxContainer *gles3_vbox = memnew(VBoxContainer);
		gles3_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		renderer_hbox->add_child(gles3_vbox);

		CheckBox *gles3_checkbox = memnew(CheckBox);
		gles3_checkbox->set_text(TTR("OpenGL ES 3.0"));
		gles3_checkbox->set_button_group(renderer_button_group);
		gles3_vbox->add_child(gles3_checkbox);

		// Enable GLES3 by default as it's the default value for the project setting.
#ifndef SERVER_ENABLED
		bool gles3_viable = RasterizerGLES3::is_viable() == OK;

		if (gles3_viable) {
			gles3_checkbox->set_pressed(true);
		} else {
			// If GLES3 can't be used, don't let users shoot themselves in the foot.
			gles3_checkbox->set_disabled(true);
			Label *gles3_not_supported_label = memnew(Label);
			gles3_not_supported_label->set_text(TTR("Not supported by your GPU drivers."));
			gles3_vbox->add_child(gles3_not_supported_label);
		}
#endif

		Label *gles3_description = memnew(Label);
		gles3_description->set_text(TTR("Higher visual quality\nAll features available\nIncompatible with older hardware\nNot recommended for web games"));
		gles3_vbox->add_child(gles3_description);

		VSeparator *v_separator = memnew(VSeparator);
		renderer_hbox->add_child(v_separator);

		VBoxContainer *gles2_vbox = memnew(VBoxContainer);
		gles2_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		renderer_hbox->add_child(gles2_vbox);

		gles2_checkbox = memnew(CheckBox);
		gles2_checkbox->set_text(TTR("OpenGL ES 2.0"));
		gles2_checkbox->set_button_group(renderer_button_group);
		gles2_vbox->add_child(gles2_checkbox);

		Label *gles2_description = memnew(Label);
		gles2_description->set_text(TTR("Lower visual quality\nSome features not available\nWorks on most hardware\nRecommended for web games"));
		gles2_vbox->add_child(gles2_description);
	}
};

class ProjectListItemControl : public HBoxContainer {
	GDCLASS(ProjectListItemControl, HBoxContainer)

public:
	TextureButton *favorite_button;
	TextureRect *icon;
	bool icon_needs_reload;
	bool hover;

	ProjectListItemControl() {
		favorite_button = nullptr;
		icon = nullptr;
		icon_needs_reload = true;
		hover = false;

		set_focus_mode(FocusMode::FOCUS_ALL);
	}

	void set_is_favorite(bool fav) {
		favorite_button->set_modulate(fav ? Color(1, 1, 1, 1) : Color(1, 1, 1, 0.2));
	}

	//refactor me
	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_MOUSE_ENTER: {
				hover = true;
				update();
			} break;
			case NOTIFICATION_MOUSE_EXIT: {
				hover = false;
				update();
			} break;
			case NOTIFICATION_DRAW: {
				if (hover) {
					draw_style_box(get_stylebox("hover", "Tree"), Rect2(Point2(), get_size() - Size2(10, 0) * EDSCALE));
				}
			} break;
		}
	}
};

// TODO: is there a nicer way?
inline void sort(int &a, int &b) {
	if (a > b) {
		int temp = a;
		a = b;
		b = temp;
	}
}

class ProjectListFilter : public HBoxContainer {
	GDCLASS(ProjectListFilter, HBoxContainer);

public:
	enum FilterOption {
		FILTER_NAME,
		FILTER_PATH,
		FILTER_MODIFIED,
	};

private:
	friend class ProjectManager;

	OptionButton *filter_option;
	LineEdit *search_box;
	bool has_search_box;
	FilterOption _current_filter;

	void _search_text_changed(const String &p_newtext) {
		emit_signal("filter_changed");
	}

	void _filter_option_selected(int p_idx) {
		FilterOption selected = (FilterOption)(filter_option->get_selected());
		if (_current_filter != selected) {
			_current_filter = selected;
			emit_signal("filter_changed");
		}
	}

protected:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_ENTER_TREE && has_search_box) {
			search_box->set_right_icon(get_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);
		}
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("_search_text_changed"), &ProjectListFilter::_search_text_changed);
		ClassDB::bind_method(D_METHOD("_filter_option_selected"), &ProjectListFilter::_filter_option_selected);

		ADD_SIGNAL(MethodInfo("filter_changed"));
	}

public:
	void _setup_filters(Vector<String> options) {
		filter_option->clear();
		for (int i = 0; i < options.size(); i++) {
			filter_option->add_item(options[i]);
		}
	}

	void add_filter_option() {
		filter_option = memnew(OptionButton);
		filter_option->set_clip_text(true);
		filter_option->connect("item_selected", this, "_filter_option_selected");
		add_child(filter_option);
	}

	void add_search_box() {
		search_box = memnew(LineEdit);
		search_box->set_placeholder(TTR("Search"));
		search_box->set_tooltip(
				TTR("The search box filters projects by name and last path component.\nTo filter projects by name and full path, the query must contain at least one `/` character."));
		search_box->connect("text_changed", this, "_search_text_changed");
		search_box->set_h_size_flags(SIZE_EXPAND_FILL);
		add_child(search_box);

		has_search_box = true;
	}

	void set_filter_size(int h_size) {
		filter_option->set_custom_minimum_size(Size2(h_size * EDSCALE, 10 * EDSCALE));
	}

	String get_search_term() {
		return search_box->get_text().strip_edges();
	}

	FilterOption get_filter_option() {
		return _current_filter;
	}

	void set_filter_option(FilterOption option) {
		filter_option->select((int)option);
		_filter_option_selected(0);
	}

	ProjectListFilter() {
		_current_filter = FILTER_NAME;
		has_search_box = false;
	}

	void clear() {
		if (has_search_box) {
			search_box->clear();
		}
	}
};

class ProjectList : public ScrollContainer {
	GDCLASS(ProjectList, ScrollContainer)

public:
	static constexpr const char *SIGNAL_SELECTION_CHANGED = "selection_changed";
	static constexpr const char *SIGNAL_PROJECT_ASK_OPEN = "project_ask_opened";

	enum MenuOptions {
		GLOBAL_NEW_WINDOW,
		GLOBAL_OPEN_PROJECT
	};

	// Can often be passed by copy.
	struct Item {
		String project_key;
		String project_name = TTR("Unnamed Project");
		String description;
		String path;
		String icon;
		String main_scene;
		uint64_t last_modified = 0;
		bool favorite;
		bool grayed = false;
		bool missing = false;
		int version = 0;

		ProjectListItemControl *control = nullptr;

		Item(const String &p_property_key, bool p_favorite) {
			// Load item.
			favorite = p_favorite;
			path = EDITOR_GET(p_property_key);
			String conf = path.plus_file("project.godot");

			Ref<ConfigFile> cf = memnew(ConfigFile);
			Error cf_err = cf->load(conf);

			if (cf_err == OK) {
				String cf_project_name = static_cast<String>(cf->get_value("application", "config/name", ""));
				if (cf_project_name != "") {
					project_name = cf_project_name.xml_unescape();
				}
				version = (int)cf->get_value("", "config_version", 0);
			}

			if (version > ProjectSettings::CONFIG_VERSION) {
				// Comes from an incompatible (more recent) Godot version, gray it out.
				grayed = true;
			}

			description = cf->get_value("application", "config/description", "");
			icon = cf->get_value("application", "config/icon", "");
			main_scene = cf->get_value("application", "run/main_scene", "");

			if (FileAccess::exists(conf)) {
				last_modified = FileAccess::get_modified_time(conf);

				String fscache = path.plus_file(".fscache");
				if (FileAccess::exists(fscache)) {
					uint64_t cache_modified = FileAccess::get_modified_time(fscache);
					if (cache_modified > last_modified) {
						last_modified = cache_modified;
					}
				}
			} else {
				grayed = true;
				missing = true;
				// why getting called twice?
				print_line("Project is missing: " + conf);
			}

			project_key = p_property_key.get_slice("/", 1);
		}
		Item(){};
		_FORCE_INLINE_ bool operator==(const Item &l) const {
			return project_key == l.project_key;
		}
	};

	ProjectList() {
		_scroll_children_vbox = memnew(VBoxContainer);
		_scroll_children_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		add_child(_scroll_children_vbox);
	}

	// what do I do?
	void update_dock_menu() {
		OS::get_singleton()->global_menu_clear("_dock");

		int favs_added = 0;
		int total_added = 0;
		for (int i = 0; i < _projects.size(); ++i) {
			if (!_projects[i].grayed && !_projects[i].missing) {
				if (_projects[i].favorite) {
					favs_added++;
				} else {
					if (favs_added != 0) {
						OS::get_singleton()->global_menu_add_separator("_dock");
					}
					favs_added = 0;
				}
				OS::get_singleton()->global_menu_add_item("_dock", _projects[i].project_name + " ( " + _projects[i].path + " )", GLOBAL_OPEN_PROJECT, Variant(_projects[i].path.plus_file("project.godot")));
				total_added++;
			}
		}
		if (total_added != 0) {
			OS::get_singleton()->global_menu_add_separator("_dock");
		}
		OS::get_singleton()->global_menu_add_item("_dock", TTR("New Window"), GLOBAL_NEW_WINDOW, Variant());
	}

	// rename me
	void load_projects() {
		// This is a full, hard reload of the list. Don't call this unless really required, it's expensive.
		// If you have 150 projects, it may read through 150 files on your disk at once + load 150 icons.

		// Clear whole list.
		for (int i = 0; i < _projects.size(); ++i) {
			Item &project = _projects.write[i];
			CRASH_COND(project.control == nullptr);
			memdelete(project.control); // Why not queue_free()?
		}
		_projects.clear();
		_last_clicked = "";
		_selected_project_keys.clear();

		// Load data.
		// TODO Would be nice to change how projects and favourites are stored... it complicates things a bit.
		// Use a dictionary associating project path to metadata (like is_favorite).

		List<PropertyInfo> properties;
		EditorSettings::get_singleton()->get_property_list(&properties);

		Set<String> favorites;
		// Find favorites...
		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			String property_key = E->get().name;
			if (property_key.begins_with("favorite_projects/")) {
				favorites.insert(property_key);
			}
		}

		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			// This is actually something like "projects/C:::Documents::Godot::Projects::MyGame".
			String property_key = E->get().name;
			if (!property_key.begins_with("projects/")) {
				continue;
			}

			String project_key = property_key.get_slice("/", 1);
			bool favorite = favorites.has("favorite_projects/" + project_key);

			Item item = Item(property_key, favorite);
			_projects.push_back(item);
		}

		// Create controls.
		for (int i = 0; i < _projects.size(); ++i) {
			create_project_item_control(i);
		}

		sort_projects();
		set_v_scroll(0);
		update_icons_async();
		update_dock_menu();
	}

	void set_search_term(String p_search_term) {
		_search_term = p_search_term;
	}

	void set_order_option(ProjectListFilter::FilterOption p_option) {
		if (_order_option != p_option) {
			_order_option = p_option;
			EditorSettings::get_singleton()->set("project_manager/sorting_order", (int)_order_option);
			EditorSettings::get_singleton()->save();
		}
	}

	void sort_projects() {
		struct ProjectListComparator {
			ProjectListFilter::FilterOption order_option;

			// operator<
			_FORCE_INLINE_ bool operator()(const ProjectList::Item &a, const ProjectList::Item &b) const {
				if (a.favorite && !b.favorite) {
					return true;
				}
				if (b.favorite && !a.favorite) {
					return false;
				}
				switch (order_option) {
					case ProjectListFilter::FILTER_PATH:
						return a.project_key < b.project_key;
					case ProjectListFilter::FILTER_MODIFIED:
						return a.last_modified > b.last_modified;
					default:
						return a.project_name < b.project_name;
				}
			}
		};

		SortArray<Item, ProjectListComparator> sorter;
		sorter.compare.order_option = _order_option;
		sorter.sort(_projects.ptrw(), _projects.size());

		for (int i = 0; i < _projects.size(); ++i) {
			Item &item = _projects.write[i];

			bool visible = true;
			if (_search_term != "") {
				String search_path;
				if (_search_term.find("/") != -1) {
					// Search path will match the whole path.
					search_path = item.path;
				} else {
					// Search path will only match the last path component to make searching more strict.
					search_path = item.path.get_file();
				}

				// When searching, display projects whose name or path contain the search term.
				visible = item.project_name.findn(_search_term) != -1 || search_path.findn(_search_term) != -1;
			}

			item.control->set_visible(visible);
		}

		for (int i = 0; i < _projects.size(); ++i) {
			Item &item = _projects.write[i];
			item.control->get_parent()->move_child(item.control, i);
		}

		// Rewind the coroutine because order of projects changed.
		update_icons_async();
		update_dock_menu();
	}

	int get_project_count() const {
		return _projects.size();
	}

	void select_project(int p_index) {
		Vector<Item> previous_selected_items = get_selected_projects();
		_selected_project_keys.clear();

		for (int i = 0; i < previous_selected_items.size(); ++i) {
			previous_selected_items[i].control->update();
		}

		toggle_select(p_index);
	}

	void erase_selected_projects() {
		if (_selected_project_keys.size() == 0) {
			return;
		}

		for (int i = 0; i < _projects.size(); ++i) {
			Item &item = _projects.write[i];
			if (_selected_project_keys.has(item.project_key) && item.control->is_visible()) {
				EditorSettings::get_singleton()->erase("projects/" + item.project_key);
				EditorSettings::get_singleton()->erase("favorite_projects/" + item.project_key);

				memdelete(item.control);
				_projects.remove(i);
				--i;
			}
		}

		EditorSettings::get_singleton()->save();

		_selected_project_keys.clear();
		_last_clicked = "";

		update_dock_menu();
	}

	Vector<Item> get_selected_projects() {
		Vector<Item> items;
		if (_selected_project_keys.size() == 0) {
			return items;
		}
		items.resize(_selected_project_keys.size());
		int j = 0;
		for (int i = 0; i < _projects.size(); ++i) {
			const Item &item = _projects[i];
			if (_selected_project_keys.has(item.project_key)) {
				items.write[j++] = item;
			}
		}
		ERR_FAIL_COND_V(j != items.size(), items);
		return items;
	}

	const Set<String> &get_selected_project_keys() const {
		// Faster if that's all you need.
		return _selected_project_keys;
	}

	bool is_any_project_missing() const {
		for (int i = 0; i < _projects.size(); ++i) {
			if (_projects[i].missing) {
				return true;
			}
		}
		return false;
	}

	void erase_missing_projects() {
		if (_projects.empty()) {
			return;
		}

		int deleted_count = 0;

		for (int i = 0; i < _projects.size(); ++i) {
			const Item &item = _projects[i];

			if (item.missing) {
				remove_project(i, true);
				--i;
				++deleted_count;
			}
		}

		print_line("Removed " + itos(deleted_count) + " projects from the list, remaining " + itos(_projects.size()) + " projects");

		EditorSettings::get_singleton()->save();
	}

	int refresh_project(const String &dir_path) {
		// Reads editor settings and reloads information about a specific project.
		// If it wasn't loaded and should be in the list, it is added (i.e new project).
		// If it isn't in the list anymore, it is removed.
		// If it is in the list but doesn't exist anymore, it is marked as missing.

		String project_key = get_project_key_from_path(dir_path);

		// Read project manager settings.
		bool is_favorite = false;
		bool should_be_in_list = false;
		String property_key = "projects/" + project_key;
		{
			List<PropertyInfo> properties;
			EditorSettings::get_singleton()->get_property_list(&properties);
			String favorite_property_key = "favorite_projects/" + project_key;

			bool found = false;
			for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
				String prop = E->get().name;
				if (!found && prop == property_key) {
					found = true;
				} else if (!is_favorite && prop == favorite_property_key) {
					is_favorite = true;
				}
			}

			should_be_in_list = found;
		}

		bool was_selected = _selected_project_keys.has(project_key);

		// Remove item in any case.
		for (int i = 0; i < _projects.size(); ++i) {
			const Item &existing_item = _projects[i];
			if (existing_item.path == dir_path) {
				remove_project(i, false);
				break;
			}
		}

		int index = -1;
		if (should_be_in_list) {
			// Recreate it with updated info.
			Item item = Item(property_key, is_favorite);

			_projects.push_back(item);
			create_project_item_control(_projects.size() - 1);

			sort_projects();

			for (int i = 0; i < _projects.size(); ++i) {
				if (_projects[i].project_key == project_key) {
					if (was_selected) {
						select_project(i);
					}
					load_project_icon(i);

					index = i;
					break;
				}
			}
		}

		return index;
	}

private:
	static void _bind_methods() {
		ClassDB::bind_method("_panel_draw", &ProjectList::_panel_draw);
		ClassDB::bind_method("_panel_input", &ProjectList::_panel_input);
		ClassDB::bind_method("_favorite_pressed", &ProjectList::_favorite_pressed);
		ClassDB::bind_method("_show_project", &ProjectList::_show_project);

		ADD_SIGNAL(MethodInfo(SIGNAL_SELECTION_CHANGED));
		ADD_SIGNAL(MethodInfo(SIGNAL_PROJECT_ASK_OPEN));
	}

	void _notification(int p_what) {
		if (p_what == NOTIFICATION_PROCESS) {
			// Load icons as a coroutine to speed up launch when you have hundreds of projects.
			if (_icon_load_index < _projects.size()) {
				Item &item = _projects.write[_icon_load_index];
				if (item.control->icon_needs_reload) {
					load_project_icon(_icon_load_index);
				}
				_icon_load_index++;

			} else {
				set_process(false);
			}
		}
	}

	// Draws selected project highlight.
	// rename me?
	void _panel_draw(Node *p_hb) {
		Control *hb = Object::cast_to<Control>(p_hb);

		hb->draw_line(Point2(0, hb->get_size().y + 1), Point2(hb->get_size().x - 10, hb->get_size().y + 1), get_color("guide_color", "Tree"));

		String key = _projects[p_hb->get_index()].project_key;

		if (_selected_project_keys.has(key)) {
			hb->draw_style_box(get_stylebox("selected", "Tree"), Rect2(Point2(), hb->get_size() - Size2(10, 0) * EDSCALE));
		}
	}

	// Input for each item in the list.
	void _panel_input(const Ref<InputEvent> &p_ev, Node *p_hb) {
		Ref<InputEventMouseButton> mb = p_ev;
		int clicked_index = p_hb->get_index();
		const Item &clicked_project = _projects[clicked_index];

		if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
			if (mb->get_shift() && _selected_project_keys.size() > 0 && _last_clicked != "" && clicked_project.project_key != _last_clicked) {
				int anchor_index = -1;
				for (int i = 0; i < _projects.size(); ++i) {
					const Item &p = _projects[i];
					if (p.project_key == _last_clicked) {
						anchor_index = p.control->get_index();
						break;
					}
				}
				CRASH_COND(anchor_index == -1);
				select_range(anchor_index, clicked_index);

			} else if (mb->get_control()) {
				toggle_select(clicked_index);

			} else {
				_last_clicked = clicked_project.project_key;
				select_project(clicked_index);
			}

			emit_signal(SIGNAL_SELECTION_CHANGED);

			if (!mb->get_control() && mb->is_doubleclick()) {
				emit_signal(SIGNAL_PROJECT_ASK_OPEN);
			}
		}
	}

	void _favorite_pressed(Node *p_hb) {
		ProjectListItemControl *control = Object::cast_to<ProjectListItemControl>(p_hb);

		int index = control->get_index();
		Item item = _projects.write[index]; // Take copy.

		item.favorite = !item.favorite;

		if (item.favorite) {
			EditorSettings::get_singleton()->set("favorite_projects/" + item.project_key, item.path);
		} else {
			EditorSettings::get_singleton()->erase("favorite_projects/" + item.project_key);
		}
		EditorSettings::get_singleton()->save();

		_projects.write[index] = item;

		control->set_is_favorite(item.favorite);

		sort_projects();

		update_dock_menu();
	}

	void _show_project(const String &p_path) {
		OS::get_singleton()->shell_open(String("file://") + p_path);
	}

	void select_range(int p_begin, int p_end) {
		sort(p_begin, p_end);
		select_project(p_begin);
		for (int i = p_begin + 1; i <= p_end; ++i) {
			toggle_select(i);
		}
	}

	void toggle_select(int p_index) {
		Item &item = _projects.write[p_index];
		if (_selected_project_keys.has(item.project_key)) {
			_selected_project_keys.erase(item.project_key);
		} else {
			_selected_project_keys.insert(item.project_key);
		}
		item.control->update();
	}

	// REFACTOR ME
	void create_project_item_control(int p_index) {
		// Will be added last in the list, so make sure indexes match.
		ERR_FAIL_COND(p_index != _scroll_children_vbox->get_child_count());

		Item &item = _projects.write[p_index];
		ERR_FAIL_COND(item.control != nullptr); // Already created.

		Ref<Texture> favorite_icon = get_icon("Favorites", "EditorIcons");
		Color font_color = get_color("font_color", "Tree");

		ProjectListItemControl *item_control = memnew(ProjectListItemControl);
		item_control->connect("draw", this, "_panel_draw", varray(item_control));
		item_control->connect("gui_input", this, "_panel_input", varray(item_control));
		item_control->add_constant_override("separation", 10 * EDSCALE);
		item_control->set_tooltip(item.description);

		VBoxContainer *favorite_box = memnew(VBoxContainer);
		favorite_box->set_name("FavoriteBox");
		TextureButton *favorite = memnew(TextureButton);
		favorite->set_name("FavoriteButton");
		favorite->set_normal_texture(favorite_icon);
		// This makes the project's "hover" style display correctly when hovering the favorite icon.
		favorite->set_mouse_filter(MOUSE_FILTER_PASS);
		favorite->connect("pressed", this, "_favorite_pressed", varray(item_control));
		favorite_box->add_child(favorite);
		favorite_box->set_alignment(BoxContainer::ALIGN_CENTER);
		item_control->add_child(favorite_box);
		item_control->favorite_button = favorite;
		item_control->set_is_favorite(item.favorite);

		TextureRect *icon = memnew(TextureRect);
		// The project icon may not be loaded by the time the control is displayed,
		// so use a loading placeholder.
		icon->set_texture(get_icon("ProjectIconLoading", "EditorIcons"));
		icon->set_v_size_flags(SIZE_SHRINK_CENTER);
		if (item.missing) {
			icon->set_modulate(Color(1, 1, 1, 0.5));
		}
		item_control->add_child(icon);
		item_control->icon = icon;

		VBoxContainer *vbox = memnew(VBoxContainer);
		if (item.grayed) {
			vbox->set_modulate(Color(1, 1, 1, 0.5));
		}
		vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		item_control->add_child(vbox);
		Control *ec = memnew(Control); // rename me
		ec->set_custom_minimum_size(Size2(0, 1));
		ec->set_mouse_filter(MOUSE_FILTER_PASS);
		vbox->add_child(ec);
		Label *title = memnew(Label(!item.missing ? item.project_name : TTR("Missing Project")));
		title->add_font_override("font", get_font("title", "EditorFonts"));
		title->add_color_override("font_color", font_color);
		title->set_clip_text(true);
		vbox->add_child(title);

		HBoxContainer *path_hbox = memnew(HBoxContainer);
		path_hbox->set_h_size_flags(SIZE_EXPAND_FILL);
		vbox->add_child(path_hbox);

		Button *show = memnew(Button);
		// Display a folder icon if the project directory can be opened, or a "broken file" icon if it can't.
		show->set_icon(get_icon(!item.missing ? "Load" : "FileBroken", "EditorIcons"));
		if (!item.grayed) {
			// Don't make the icon less prominent if the parent is already grayed out.
			show->set_modulate(Color(1, 1, 1, 0.5));
		}
		path_hbox->add_child(show);

		if (!item.missing) {
			show->connect("pressed", this, "_show_project", varray(item.path));
			show->set_tooltip(TTR("Show in File Manager"));
		} else {
			show->set_tooltip(TTR("Error: Project is missing on the filesystem."));
		}

		Label *fpath = memnew(Label(item.path));
		path_hbox->add_child(fpath);
		fpath->set_h_size_flags(SIZE_EXPAND_FILL);
		fpath->set_modulate(Color(1, 1, 1, 0.5));
		fpath->add_color_override("font_color", font_color);
		fpath->set_clip_text(true);

		_scroll_children_vbox->add_child(item_control);
		item.control = item_control;
	}

	void remove_project(int p_index, bool p_update_settings) {
		const Item item = _projects[p_index]; // Take a copy.

		_selected_project_keys.erase(item.project_key);

		if (_last_clicked == item.project_key) {
			_last_clicked = "";
		}

		memdelete(item.control);
		_projects.remove(p_index);

		if (p_update_settings) {
			EditorSettings::get_singleton()->erase("projects/" + item.project_key);
			EditorSettings::get_singleton()->erase("favorite_projects/" + item.project_key);
			// Not actually saving the file, in case you are doing more changes to settings.
		}

		update_dock_menu();
	}

	void update_icons_async() {
		// Icons are loaded in `NOTIFCATION_PROCESS`.
		_icon_load_index = 0;
		set_process(true);
	}

	void load_project_icon(int p_index) {
		Item &item = _projects.write[p_index];

		Ref<Texture> default_icon = get_icon("DefaultProjectIcon", "EditorIcons");
		Ref<Texture> icon;
		if (item.icon != "") {
			Ref<Image> img;
			img.instance();
			Error err = img->load(item.icon.replace_first("res://", item.path + "/"));
			if (err == OK) {
				img->resize(default_icon->get_width(), default_icon->get_height(), Image::INTERPOLATE_LANCZOS);
				Ref<ImageTexture> it = memnew(ImageTexture);
				it->create_from_image(img);
				icon = it;
			}
		}
		if (icon.is_null()) {
			icon = default_icon;
		}

		item.control->icon->set_texture(icon);
		item.control->icon_needs_reload = false;
	}

	String _search_term;
	ProjectListFilter::FilterOption _order_option = ProjectListFilter::FILTER_MODIFIED;
	Set<String> _selected_project_keys;
	String _last_clicked; // Project key.
	VBoxContainer *_scroll_children_vbox;
	int _icon_load_index = 0;

	Vector<Item> _projects;
};

void ProjectManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Engine::get_singleton()->set_editor_hint(false);
		} break;
		case NOTIFICATION_RESIZED: {
			if (open_templates->is_visible()) {
				open_templates->popup_centered_minsize();
			}
		} break;
		case NOTIFICATION_READY: {
			if (_project_list->get_project_count() == 0 && StreamPeerSSL::is_available()) {
				open_templates->popup_centered_minsize();
			}

			if (_project_list->get_project_count() >= 1) {
				// Focus on the search box immediately to allow the user
				// to search without having to reach for their mouse.
				project_filter->search_box->grab_focus();
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_unhandled_input(is_visible_in_tree());
		} break;
		case NOTIFICATION_WM_QUIT_REQUEST: {
			_dim_window();
		} break;
		case NOTIFICATION_WM_ABOUT: {
			_show_about();
		} break;
	}
}

void ProjectManager::_dim_window() {
	// This method must be called before calling `get_tree()->quit()`.
	// Otherwise, its effect won't be visible.

	// Dim the project manager window while it's quitting to make it clearer that it's busy.
	// No transition is applied, as the effect needs to be visible immediately.
	float c = 0.5f;
	Color dim_color = Color(c, c, c);
	gui_base->set_modulate(dim_color);
}

void ProjectManager::_update_project_buttons() {
	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	bool empty_selection = selected_projects.empty();

	bool is_missing_project_selected = false;
	for (int i = 0; i < selected_projects.size(); ++i) {
		if (selected_projects[i].missing) {
			is_missing_project_selected = true;
			break;
		}
	}

	erase_button->set_disabled(empty_selection);
	open_button->set_disabled(empty_selection || is_missing_project_selected);
	rename_button->set_disabled(empty_selection || is_missing_project_selected);
	run_button->set_disabled(empty_selection || is_missing_project_selected);

	remove_missing_button->set_disabled(!_project_list->is_any_project_missing());
}

void ProjectManager::_unhandled_input(const Ref<InputEvent> &p_ev) {
	// refactor me
	Ref<InputEventKey> k = p_ev;
	if (!k.is_valid() || !k->is_pressed()) {
		return;
	}
	if (tabs->get_current_tab() != 0) {
		return;
	}

	// Pressing Command + Q quits the Project Manager.
	// This is handled by the platform implementation on macOS,
	// so only define the shortcut on other platforms.
#ifndef OSX_ENABLED
	if (k->get_scancode_with_modifiers() == (KEY_MASK_CMD | KEY_Q)) {
		_dim_window();
		get_tree()->quit();
	}
#endif

	bool scancode_handled = true;

	switch (k->get_scancode()) {
		case KEY_ENTER: {
			_open_selected_projects_ask();
		} break;
		case KEY_DELETE: {
			_erase_project_ask();
		} break;
		case KEY_HOME: {
			if (_project_list->get_project_count() > 0) {
				_project_list->select_project(0);
				_update_project_buttons();
			}
		} break;
		case KEY_END: {
			if (_project_list->get_project_count() > 0) {
				_project_list->select_project(_project_list->get_project_count() - 1);
				_update_project_buttons();
			}
		} break;
		case KEY_F: {
			if (k->get_command()) { // refactor me
				project_filter->search_box->grab_focus();
			} else {
				scancode_handled = false;
			}
		} break;
		default: {
			scancode_handled = false;
		} break;
	}

	if (scancode_handled) {
		accept_event();
	}
}

void ProjectManager::_load_recent_projects() {
	_project_list->set_order_option(project_order_filter->get_filter_option());
	_project_list->set_search_term(project_filter->get_search_term());
	_project_list->load_projects();

	_update_project_buttons();
}

void ProjectManager::_on_projects_updated() {
	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	int index = 0;
	for (int i = 0; i < selected_projects.size(); ++i) {
		index = _project_list->refresh_project(selected_projects[i].path);
	}

	_project_list->update_dock_menu();
}

void ProjectManager::_create_project(String project_name, String path, bool use_gles2) {
	// The path to the folder the new project's folder is in/will be created if it doesn't exist.
	String project_folder_path = path.left(path.find_last("/"));
	String project_folder_name = path.right(path.find_last("/") + 1);
	DirAccessRef dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	ERR_FAIL_COND_MSG(!dir_access->dir_exists(project_folder_path), project_folder_path + " not found.");
	if (!dir_access->dir_exists(path)) {
		// Create the project's root folder if it doesn't exist.
		// This method will only create one new folder
		dir_access->make_dir(path);
	}

	ProjectSettings::CustomMap initial_settings;
	if (use_gles2) {
		initial_settings["rendering/quality/driver/driver_name"] = "GLES2";
		initial_settings["rendering/vram_compression/import_etc2"] = false;
		initial_settings["rendering/vram_compression/import_etc"] = true;
	} else {
		initial_settings["rendering/quality/driver/driver_name"] = "GLES3";
	}
	initial_settings["application/config/name"] = project_name;
	initial_settings["application/config/icon"] = "res://icon.png";
	initial_settings["rendering/environment/default_environment"] = "res://default_env.tres";
	initial_settings["physics/common/enable_pause_aware_picking"] = true;

	ERR_FAIL_COND_MSG(ProjectSettings::get_singleton()->save_custom(path.plus_file("project.godot"), initial_settings, Vector<String>(), false) != OK, "Couldn't create project.godot in project path.");

	ResourceSaver::save(path.plus_file("icon.png"), create_unscaled_default_project_icon());

	FileAccess *f = FileAccess::open(path.plus_file("default_env.tres"), FileAccess::WRITE);
	ERR_FAIL_COND_MSG(!f, "Couldn't create project.godot in project path.");
	f->store_line("[gd_resource type=\"Environment\" load_steps=2 format=2]");
	f->store_line("[sub_resource type=\"ProceduralSky\" id=1]");
	f->store_line("[resource]");
	f->store_line("background_mode = 2");
	f->store_line("background_sky = SubResource( 1 )");
	memdelete(f);

	String project_key = get_project_key_from_path(path);
	EditorSettings::get_singleton()->set("projects/" + project_key, path);
	EditorSettings::get_singleton()->save();
}

// what am I for?
void ProjectManager::_create_project_confirmed() {
	_create_project(create_project_dialog->get_new_project_name(), create_project_dialog->get_new_project_path(), create_project_dialog->should_new_project_use_gles2());
	//project_filter->clear();
	int i = _project_list->refresh_project(create_project_dialog->get_new_project_path());
	_project_list->select_project(i);
	_open_selected_projects_ask();

	_project_list->update_dock_menu();
}

void ProjectManager::_confirm_update_settings() {
	_open_selected_projects();
}

// refactor me
void ProjectManager::_global_menu_action(const Variant &p_id, const Variant &p_meta) {
	int id = (int)p_id;
	if (id == ProjectList::GLOBAL_NEW_WINDOW) {
		List<String> args;
		args.push_back("-p");
		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		OS::get_singleton()->execute(exec, args, false, &pid);
	} else if (id == ProjectList::GLOBAL_OPEN_PROJECT) {
		String conf = (String)p_meta;

		if (conf != String()) {
			List<String> args;
			args.push_back(conf);
			String exec = OS::get_singleton()->get_executable_path();

			OS::ProcessID pid = 0;
			OS::get_singleton()->execute(exec, args, false, &pid);
		}
	}
}

void ProjectManager::_open_selected_projects() {
	// Show loading text to tell the user that the project manager is busy loading.
	// This is especially important for the HTML5 project manager.
	loading_label->set_modulate(Color(1, 1, 1));

	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	for (const Set<String>::Element *E = selected_list.front(); E; E = E->next()) {
		const String &selected = E->get();
		String path = EDITOR_GET("projects/" + selected);
		String conf = path.plus_file("project.godot");

		if (!FileAccess::exists(conf)) {
			dialog_error->set_text(vformat(TTR("Can't open project at '%s'."), path));
			dialog_error->popup_centered_minsize();
			return;
		}

		print_line("Editing project: " + path + " (" + selected + ")");

		List<String> args;

		args.push_back("--path");
		args.push_back(path);

		args.push_back("--editor");

		if (OS::get_singleton()->is_stdout_debug_enabled()) {
			args.push_back("--debug");
		}

		if (OS::get_singleton()->is_stdout_verbose()) {
			args.push_back("--verbose");
		}

		if (OS::get_singleton()->is_disable_crash_handler()) {
			args.push_back("--disable-crash-handler");
		}

		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		Error err = OS::get_singleton()->execute(exec, args, false, &pid);
		ERR_FAIL_COND(err);
	}

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_open_selected_projects_ask() {
	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() > 1) {
		multi_open_ask->popup_centered_minsize();
		return;
	}

	ProjectList::Item project = _project_list->get_selected_projects()[0];
	if (project.missing) {
		return;
	}

	// Update the project settings or don't open.
	String conf = project.path.plus_file("project.godot");
	int config_version = project.version;

	// Check if the config_version property was empty or 0.
	if (config_version == 0) {
		ask_update_settings->set_text(vformat(TTR("The following project settings file does not specify the version of Godot through which it was created.\n\n%s\n\nIf you proceed with opening it, it will be converted to Godot's current configuration file format.\nWarning: You won't be able to open the project with previous versions of the engine anymore."), conf));
		ask_update_settings->popup_centered_minsize();
		return;
	}
	// Check if we need to convert project settings from an earlier engine version.
	if (config_version < ProjectSettings::CONFIG_VERSION) {
		ask_update_settings->set_text(vformat(TTR("The following project settings file was generated by an older engine version, and needs to be converted for this version:\n\n%s\n\nDo you want to convert it?\nWarning: You won't be able to open the project with previous versions of the engine anymore."), conf));
		ask_update_settings->popup_centered_minsize();
		return;
	}
	// Check if the file was generated by a newer, incompatible engine version.
	if (config_version > ProjectSettings::CONFIG_VERSION) {
		dialog_error->set_text(vformat(TTR("Can't open project at '%s'.") + "\n" + TTR("The project settings were created by a newer engine version, whose settings are not compatible with this version."), project.path));
		dialog_error->popup_centered_minsize();
		return;
	}

	// Open if the project is up-to-date.
	_open_selected_projects();
}

void ProjectManager::_run_project_confirmed() {
	Vector<ProjectList::Item> selected_list = _project_list->get_selected_projects();

	for (int i = 0; i < selected_list.size(); ++i) {
		const String &selected_main = selected_list[i].main_scene;
		if (selected_main == "") {
			run_error_diag->set_text(TTR("Can't run project: no main scene defined.\nPlease edit the project and set the main scene in the Project Settings under the \"Application\" category."));
			run_error_diag->popup_centered();
			continue;
		}

		const String &selected = selected_list[i].project_key;
		String path = EDITOR_GET("projects/" + selected);

		if (!DirAccess::exists(path + "/.import")) {
			run_error_diag->set_text(TTR("Can't run project: Assets need to be imported.\nPlease edit the project to trigger the initial import."));
			run_error_diag->popup_centered();
			continue;
		}

		print_line("Running project: " + path + " (" + selected + ")");

		List<String> args;

		args.push_back("--path");
		args.push_back(path);

		if (OS::get_singleton()->is_disable_crash_handler()) {
			args.push_back("--disable-crash-handler");
		}

		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		Error err = OS::get_singleton()->execute(exec, args, false, &pid);
		ERR_FAIL_COND(err);
	}
}

// When you press the "Run" button.
void ProjectManager::_run_project_ask() {
	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	if (selected_list.size() > 1) {
		multi_run_ask->set_text(vformat(TTR("Are you sure to run %d projects at once?"), selected_list.size()));
		multi_run_ask->popup_centered_minsize();
	} else {
		_run_project_confirmed();
	}
}

void ProjectManager::_scan_dir(const String &path, List<String> *r_projects) {
	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error error = da->change_dir(path);
	ERR_FAIL_COND_MSG(error != OK, "Could not scan directory at: " + path);
	da->list_dir_begin();
	String n = da->get_next();
	while (n != String()) {
		if (da->current_is_dir() && !n.begins_with(".")) {
			_scan_dir(da->get_current_dir().plus_file(n), r_projects);
		} else if (n == "project.godot") {
			r_projects->push_back(da->get_current_dir());
		}
		n = da->get_next();
	}
	da->list_dir_end();
}

void ProjectManager::_scan_begin(const String &p_base) {
	print_line("Scanning projects at: " + p_base);
	List<String> projects;
	_scan_dir(p_base, &projects);
	print_line("Found " + itos(projects.size()) + " projects.");

	for (List<String>::Element *E = projects.front(); E; E = E->next()) {
		String proj = get_project_key_from_path(E->get());
		EditorSettings::get_singleton()->set("projects/" + proj, E->get());
	}

	EditorSettings::get_singleton()->save();
	_load_recent_projects();
}

void ProjectManager::_import_project() {
	/*npdialog->set_mode(ProjectDialog::MODE_IMPORT);
	npdialog->show_dialog();*/
}

void ProjectManager::_rename_project() { /*
	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0) {
		return;
	}

	for (Set<String>::Element *E = selected_list.front(); E; E = E->next()) {
		const String &selected = E->get();
		String path = EDITOR_GET("projects/" + selected);
		npdialog->set_project_path(path);
		npdialog->set_mode(ProjectDialog::MODE_RENAME);
		npdialog->show_dialog();
	}*/
}

void ProjectManager::_erase_project_confirmed() {
	_project_list->erase_selected_projects();
	_update_project_buttons();
}

void ProjectManager::_erase_missing_projects_confirmed() {
	_project_list->erase_missing_projects();
	_update_project_buttons();
}

void ProjectManager::_erase_project_ask() {
	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	String confirm_message;
	if (selected_list.size() >= 2) {
		confirm_message = vformat(TTR("Remove %d projects from the list?\nThe project folders' contents won't be modified."), selected_list.size());
	} else {
		confirm_message = TTR("Remove this project from the list?\nThe project folder's contents won't be modified.");
	}

	erase_ask->set_text(confirm_message);
	erase_ask->popup_centered_minsize();
}

void ProjectManager::_erase_missing_projects() {
	erase_missing_ask->set_text(TTR("Remove all missing projects from the list?\nThe project folders' contents won't be modified."));
	erase_missing_ask->popup_centered_minsize();
}

void ProjectManager::_show_about() {
	about->popup_centered(Size2(780, 500) * EDSCALE);
}

void ProjectManager::_language_selected(int p_id) {
	String lang = language_button->get_item_metadata(p_id);
	EditorSettings::get_singleton()->set("interface/editor/editor_language", lang);
	//language_button->set_text(lang);

	language_restart_ask->set_text(TTR("Language changed.\nThe interface will update after restarting the editor or project manager."));
	language_restart_ask->popup_centered();
}

void ProjectManager::_restart_confirmed() {
	List<String> args = OS::get_singleton()->get_cmdline_args();
	String exec = OS::get_singleton()->get_executable_path();
	OS::ProcessID pid = 0;
	Error err = OS::get_singleton()->execute(exec, args, false, &pid);
	ERR_FAIL_COND(err);

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_install_project_from_zip(const String &p_zip_path, const String &p_title) { /*
	npdialog->set_mode(ProjectDialog::MODE_INSTALL);
	npdialog->set_zip_path(p_zip_path);
	npdialog->set_zip_title(p_title);
	npdialog->show_dialog();
																								  */
}

void ProjectManager::_files_dropped(PoolStringArray p_files, int p_screen) {
	if (p_files.size() == 1 && p_files[0].ends_with(".zip")) {
		const String file = p_files[0].get_file();
		_install_project_from_zip(p_files[0], file.substr(0, file.length() - 4).capitalize());
		return;
	}
	Set<String> folders_set;
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	for (int i = 0; i < p_files.size(); i++) {
		String file = p_files[i];
		folders_set.insert(da->dir_exists(file) ? file : file.get_base_dir());
	}
	memdelete(da);
	if (folders_set.size() > 0) {
		PoolStringArray folders;
		for (Set<String>::Element *E = folders_set.front(); E; E = E->next()) {
			folders.append(E->get());
		}

		bool confirm = true;
		if (folders.size() == 1) {
			DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			if (dir->change_dir(folders[0]) == OK) {
				dir->list_dir_begin();
				String file = dir->get_next();
				while (confirm && file != String()) {
					if (!dir->current_is_dir() && file.ends_with("project.godot")) {
						confirm = false;
					}
					file = dir->get_next();
				}
				dir->list_dir_end();
			}
			memdelete(dir);
		}
		if (confirm) {
			multi_scan_ask->get_ok()->disconnect("pressed", this, "_scan_multiple_folders");
			multi_scan_ask->get_ok()->connect("pressed", this, "_scan_multiple_folders", varray(folders));
			multi_scan_ask->set_text(
					vformat(TTR("Are you sure to scan %s folders for existing Godot projects?\nThis could take a while."), folders.size()));
			multi_scan_ask->popup_centered_minsize();
		} else {
			_scan_multiple_folders(folders);
		}
	}
}

void ProjectManager::_scan_multiple_folders(PoolStringArray p_files) {
	for (int i = 0; i < p_files.size(); i++) {
		_scan_begin(p_files.get(i));
	}
}

void ProjectManager::_on_order_option_changed() {
	_project_list->set_order_option(project_order_filter->get_filter_option());
	_project_list->sort_projects();
}

void ProjectManager::_on_filter_option_changed() {
	_project_list->set_search_term(project_filter->get_search_term());
	_project_list->sort_projects();
}

void ProjectManager::_bind_methods() {
	ClassDB::bind_method("_open_selected_projects_ask", &ProjectManager::_open_selected_projects_ask);
	ClassDB::bind_method("_open_selected_projects", &ProjectManager::_open_selected_projects);
	ClassDB::bind_method(D_METHOD("_global_menu_action"), &ProjectManager::_global_menu_action, DEFVAL(Variant()));
	ClassDB::bind_method("_run_project_ask", &ProjectManager::_run_project_ask);
	ClassDB::bind_method("_run_project_confirmed", &ProjectManager::_run_project_confirmed);
	ClassDB::bind_method("_scan_begin", &ProjectManager::_scan_begin);
	ClassDB::bind_method("_import_project", &ProjectManager::_import_project);
	ClassDB::bind_method("_rename_project", &ProjectManager::_rename_project);
	ClassDB::bind_method("_erase_project_ask", &ProjectManager::_erase_project_ask);
	ClassDB::bind_method("_erase_missing_projects", &ProjectManager::_erase_missing_projects);
	ClassDB::bind_method("_erase_project_confirmed", &ProjectManager::_erase_project_confirmed);
	ClassDB::bind_method("_erase_missing_projects_confirmed", &ProjectManager::_erase_missing_projects_confirmed);
	ClassDB::bind_method("_show_about", &ProjectManager::_show_about);
	ClassDB::bind_method("_version_button_pressed", &ProjectManager::_version_button_pressed);
	ClassDB::bind_method("_language_selected", &ProjectManager::_language_selected);
	ClassDB::bind_method("_restart_confirmed", &ProjectManager::_restart_confirmed);
	ClassDB::bind_method("_on_order_option_changed", &ProjectManager::_on_order_option_changed);
	ClassDB::bind_method("_on_filter_option_changed", &ProjectManager::_on_filter_option_changed);
	ClassDB::bind_method("_on_projects_updated", &ProjectManager::_on_projects_updated);
	ClassDB::bind_method("_create_project_confirmed", &ProjectManager::_create_project_confirmed);
	ClassDB::bind_method("_unhandled_input", &ProjectManager::_unhandled_input);
	ClassDB::bind_method("_install_project_from_zip", &ProjectManager::_install_project_from_zip);
	ClassDB::bind_method("_files_dropped", &ProjectManager::_files_dropped);
	ClassDB::bind_method("_open_asset_library", &ProjectManager::_open_asset_library);
	ClassDB::bind_method("_confirm_update_settings", &ProjectManager::_confirm_update_settings);
	ClassDB::bind_method("_update_project_buttons", &ProjectManager::_update_project_buttons);
	ClassDB::bind_method(D_METHOD("_scan_multiple_folders", "files"), &ProjectManager::_scan_multiple_folders);
}

void ProjectManager::_open_asset_library() {
	// For if the user doesn't have any existing projects.
	asset_library->disable_community_support();
	tabs->set_current_tab(1);
}

void ProjectManager::_version_button_pressed() {
	OS::get_singleton()->set_clipboard(version_button->get_text());
}

ProjectManager::ProjectManager() {
	// Load settings.
	if (!EditorSettings::get_singleton()) {
		EditorSettings::create();
	}

	EditorSettings::get_singleton()->set_optimize_save(false); // Just write settings as they came.

	int display_scale = EDITOR_GET("interface/editor/display_scale");
	if (display_scale == 0) {
		// Try applying a suitable display scale automatically.
		editor_set_scale(EditorSettings::get_singleton()->get_auto_display_scale());
	} else if (display_scale >= 1 && display_scale <= 6) {
		// The display scale has been set to one of the preset options.
		editor_set_scale(0.5 + 0.25 * display_scale);
	} else {
		// The user is using a custom display scale.
		float custom_display_scale = EDITOR_GET("interface/editor/custom_display_scale");
		editor_set_scale(custom_display_scale);
	}

	// Define a minimum window size to prevent UI elements from overlapping or being cut off.
	OS::get_singleton()->set_min_window_size(Size2(750, 420) * EDSCALE);

	// TODO: Resize windows on hiDPI displays on Windows and Linux and remove the line below.
	OS::get_singleton()->set_window_size(OS::get_singleton()->get_window_size() * MAX(1, EDSCALE));

	FileDialog::set_default_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));

	set_anchors_and_margins_preset(PRESET_WIDE);
	set_theme(create_custom_theme());

	Panel *gui_base = memnew(Panel);
	add_child(gui_base);
	gui_base->set_anchors_and_margins_preset(PRESET_WIDE);
	gui_base->add_style_override("panel", gui_base->get_stylebox("Background", "EditorStyles"));

	VBoxContainer *vbox = memnew(VBoxContainer);
	gui_base->add_child(vbox);
	vbox->set_anchors_and_margins_preset(PRESET_WIDE, PRESET_MODE_MINSIZE, 8 * EDSCALE);

	// TRANSLATORS: This refers to the application where users manage their Godot projects.
	OS::get_singleton()->set_window_title(VERSION_NAME + String(" - ") + TTR("Project Manager"));

	Control *center_box = memnew(Control);
	center_box->set_v_size_flags(SIZE_EXPAND_FILL);
	vbox->add_child(center_box);

	tabs = memnew(TabContainer);
	center_box->add_child(tabs);
	tabs->set_anchors_and_margins_preset(PRESET_WIDE);
	tabs->set_tab_align(TabContainer::ALIGN_LEFT);

	HBoxContainer *tree_hbox = memnew(HBoxContainer);
	projects_hbox = tree_hbox;

	projects_hbox->set_name(TTR("Local Projects"));

	tabs->add_child(tree_hbox);

	VBoxContainer *search_buttons_vbox = memnew(VBoxContainer);
	tree_hbox->add_child(search_buttons_vbox);
	search_buttons_vbox->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *sort_filters = memnew(HBoxContainer);
	loading_label = memnew(Label(TTR("Loading, please wait...")));
	loading_label->add_font_override("font", get_font("bold", "EditorFonts"));
	loading_label->set_h_size_flags(SIZE_EXPAND_FILL);
	sort_filters->add_child(loading_label);
	// Hide the label but make it still take up space. This prevents reflows when showing the label.
	loading_label->set_modulate(Color(0, 0, 0, 0));

	Label *sort_label = memnew(Label);
	sort_label->set_text(TTR("Sort:"));
	sort_filters->add_child(sort_label);
	Vector<String> sort_filter_titles;
	sort_filter_titles.push_back(TTR("Name"));
	sort_filter_titles.push_back(TTR("Path"));
	sort_filter_titles.push_back(TTR("Last Modified"));
	project_order_filter = memnew(ProjectListFilter);
	project_order_filter->add_filter_option();
	project_order_filter->_setup_filters(sort_filter_titles);
	project_order_filter->set_filter_size(150);
	project_order_filter->set_custom_minimum_size(Size2(180, 10) * EDSCALE);
	project_order_filter->connect("filter_changed", this, "_on_order_option_changed");
	sort_filters->add_child(project_order_filter);

	int projects_sorting_order = (int)EDITOR_GET("project_manager/sorting_order");
	project_order_filter->set_filter_option((ProjectListFilter::FilterOption)projects_sorting_order);

	project_filter = memnew(ProjectListFilter);
	project_filter->add_search_box();
	project_filter->set_custom_minimum_size(Size2(280, 10) * EDSCALE);
	project_filter->connect("filter_changed", this, "_on_filter_option_changed");
	sort_filters->add_child(project_filter);

	search_buttons_vbox->add_child(sort_filters);

	PanelContainer *panel_container = memnew(PanelContainer);
	panel_container->add_style_override("panel", get_stylebox("bg", "Tree"));
	panel_container->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(panel_container);

	_project_list = memnew(ProjectList);
	_project_list->connect(ProjectList::SIGNAL_SELECTION_CHANGED, this, "_update_project_buttons");
	_project_list->connect(ProjectList::SIGNAL_PROJECT_ASK_OPEN, this, "_open_selected_projects_ask");
	search_buttons_vbox->add_child(_project_list);
	_project_list->set_enable_h_scroll(false);

	// Contains the Edit, Run, New Project, etc. buttons.
	VBoxContainer *buttons_vbox = memnew(VBoxContainer);
	buttons_vbox->set_custom_minimum_size(Size2(120, 120));
	tree_hbox->add_child(buttons_vbox);

	open_button = memnew(Button);
	open_button->set_text(TTR("Edit"));
	open_button->set_shortcut(ED_SHORTCUT("project_manager/edit_project", TTR("Edit Project"), KEY_MASK_CMD | KEY_E));
	open_button->connect("pressed", this, "_open_selected_projects_ask");
	buttons_vbox->add_child(open_button);

	run_button = memnew(Button);
	run_button->set_text(TTR("Run"));
	run_button->set_shortcut(ED_SHORTCUT("project_manager/run_project", TTR("Run Project"), KEY_MASK_CMD | KEY_R));
	run_button->connect("pressed", this, "_run_project_ask");
	buttons_vbox->add_child(run_button);

	buttons_vbox->add_child(memnew(HSeparator));

	create_project_dialog = memnew(CreateProjectDialog);
	create_project_dialog->connect("confirmed", this, "_create_project_confirmed");
	gui_base->add_child(create_project_dialog);

	Button *create_button = memnew(Button);
	create_button->set_text(TTR("New Project"));
	create_button->set_shortcut(ED_SHORTCUT("project_manager/new_project", TTR("New Project"), KEY_MASK_CMD | KEY_N));
	create_button->connect("pressed", create_project_dialog, "create_project_popup");
	buttons_vbox->add_child(create_button);

	Button *import_button = memnew(Button);
	import_button->set_text(TTR("Import"));
	import_button->set_shortcut(ED_SHORTCUT("project_manager/import_project", TTR("Import Project"), KEY_MASK_CMD | KEY_I));
	import_button->connect("pressed", this, "_import_project");
	buttons_vbox->add_child(import_button);

	scan_dir_file_dialog = memnew(FileDialog);
	scan_dir_file_dialog->set_access(FileDialog::ACCESS_FILESYSTEM);
	scan_dir_file_dialog->set_mode(FileDialog::MODE_OPEN_DIR);
	scan_dir_file_dialog->set_title(TTR("Select a Folder to Scan")); // Must be after `set_mode()` or it gets overridden.
	scan_dir_file_dialog->set_current_dir(EDITOR_GET("filesystem/directories/default_project_path"));
	scan_dir_file_dialog->connect("dir_selected", this, "_scan_begin");
	gui_base->add_child(scan_dir_file_dialog);

	Button *scan_button = memnew(Button);
	scan_button->set_text(TTR("Scan"));
	scan_button->set_shortcut(ED_SHORTCUT("project_manager/scan_projects", TTR("Scan Projects"), KEY_MASK_CMD | KEY_S));
	scan_button->connect("pressed", scan_dir_file_dialog, "popup_centered_ratio");
	buttons_vbox->add_child(scan_button);

	rename_button = memnew(Button);
	rename_button->set_text(TTR("Rename"));
	rename_button->set_shortcut(ED_SHORTCUT("project_manager/rename_project", TTR("Rename Project"), KEY_F2));
	rename_button->connect("pressed", this, "_rename_project");
	buttons_vbox->add_child(rename_button);

	buttons_vbox->add_child(memnew(HSeparator));

	erase_button = memnew(Button);
	erase_button->set_text(TTR("Remove"));
	erase_button->set_shortcut(ED_SHORTCUT("project_manager/remove_project", TTR("Remove Project"), KEY_DELETE));
	erase_button->connect("pressed", this, "_erase_project");
	buttons_vbox->add_child(erase_button);

	remove_missing_button = memnew(Button);
	remove_missing_button->set_text(TTR("Remove Missing"));
	remove_missing_button->connect("pressed", this, "_erase_missing_projects");
	buttons_vbox->add_child(remove_missing_button);

	buttons_vbox->add_spacer();

	about_button = memnew(Button);
	about_button->set_text(TTR("About"));
	about_button->connect("pressed", this, "_show_about");
	buttons_vbox->add_child(about_button);

	if (StreamPeerSSL::is_available()) {
		asset_library = memnew(EditorAssetLibrary(true));
		asset_library->set_name(TTR("Asset Library Projects"));
		asset_library->connect("install_asset", this, "_install_project_from_zip");
		tabs->add_child(asset_library);
	} else {
		WARN_PRINT("Asset Library not available, as it requires SSL to work.");
	}

	HBoxContainer *settings_hbox = memnew(HBoxContainer);
	settings_hbox->set_alignment(BoxContainer::ALIGN_END);
	settings_hbox->set_h_grow_direction(GROW_DIRECTION_BEGIN);

	// A VBoxContainer that contains a dummy Control node to adjust the LinkButton's vertical position.
	VBoxContainer *spacer_vbox = memnew(VBoxContainer);
	settings_hbox->add_child(spacer_vbox);

	Control *v_spacer = memnew(Control);
	spacer_vbox->add_child(v_spacer);

	version_button = memnew(LinkButton);
	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = " " + vformat("[%s]", hash.left(9));
	}
	version_button->set_text("v" VERSION_FULL_BUILD + hash);
	// Fade the version label to be less prominent, but still readable.
	version_button->set_self_modulate(Color(1, 1, 1, 0.6));
	version_button->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	version_button->set_tooltip(TTR("Click to copy."));
	version_button->connect("pressed", this, "_version_button_pressed");
	spacer_vbox->add_child(version_button);

	// Add a small horizontal spacer between the version and language buttons
	// to distinguish them.
	Control *h_spacer = memnew(Control);
	settings_hbox->add_child(h_spacer);

	language_button = memnew(OptionButton);
	language_button->set_flat(true);
	language_button->set_icon(get_icon("Environment", "EditorIcons"));
	language_button->set_focus_mode(FOCUS_NONE);

	Vector<String> editor_languages;
	List<PropertyInfo> editor_settings_properties;
	EditorSettings::get_singleton()->get_property_list(&editor_settings_properties);
	for (List<PropertyInfo>::Element *E = editor_settings_properties.front(); E; E = E->next()) {
		PropertyInfo &pi = E->get();
		if (pi.name == "interface/editor/editor_language") {
			editor_languages = pi.hint_string.split(",");
		}
	}
	String current_lang = EDITOR_GET("interface/editor/editor_language");
	for (int i = 0; i < editor_languages.size(); i++) {
		String lang = editor_languages[i];
		String lang_name = TranslationServer::get_singleton()->get_locale_name(lang);
		language_button->add_item(lang_name + " [" + lang + "]", i);
		language_button->set_item_metadata(i, lang);
		if (current_lang == lang) {
			language_button->select(i);
			language_button->set_text(lang);
		}
	}
	language_button->set_icon(get_icon("Environment", "EditorIcons"));
	language_button->connect("item_selected", this, "_language_selected");
	settings_hbox->add_child(language_button);

	center_box->add_child(settings_hbox);
	settings_hbox->set_anchors_and_margins_preset(PRESET_TOP_RIGHT);

	language_restart_ask = memnew(ConfirmationDialog);
	language_restart_ask->get_ok()->set_text(TTR("Restart Now"));
	language_restart_ask->get_ok()->connect("pressed", this, "_restart_confirmed");
	language_restart_ask->get_cancel()->set_text(TTR("Continue"));
	gui_base->add_child(language_restart_ask);

	erase_missing_ask = memnew(ConfirmationDialog);
	erase_missing_ask->get_ok()->set_text(TTR("Remove All"));
	erase_missing_ask->get_ok()->connect("pressed", this, "_erase_missing_projects_confirmed");
	gui_base->add_child(erase_missing_ask);

	erase_ask = memnew(ConfirmationDialog);
	erase_ask->get_ok()->set_text(TTR("Remove"));
	erase_ask->get_ok()->connect("pressed", this, "_erase_project_confirmed");
	gui_base->add_child(erase_ask);

	multi_open_ask = memnew(ConfirmationDialog);
	multi_open_ask->set_text(TTR("Are you sure to open more than one project?"));
	multi_open_ask->get_ok()->set_text(TTR("Edit"));
	multi_open_ask->get_ok()->connect("pressed", this, "_open_selected_projects");
	gui_base->add_child(multi_open_ask);

	multi_run_ask = memnew(ConfirmationDialog);
	multi_run_ask->get_ok()->set_text(TTR("Run"));
	multi_run_ask->get_ok()->connect("pressed", this, "_run_project_confirmed");
	gui_base->add_child(multi_run_ask);

	multi_scan_ask = memnew(ConfirmationDialog);
	multi_scan_ask->get_ok()->set_text(TTR("Scan"));
	gui_base->add_child(multi_scan_ask);

	ask_update_settings = memnew(ConfirmationDialog);
	ask_update_settings->get_ok()->connect("pressed", this, "_confirm_update_settings");
	gui_base->add_child(ask_update_settings);

	OS::get_singleton()->set_low_processor_usage_mode(true);

	_load_recent_projects();

	DirAccessRef dir_access = DirAccess::create(DirAccess::AccessType::ACCESS_FILESYSTEM);

	String default_project_path = EDITOR_GET("filesystem/directories/default_project_path");
	if (!dir_access->dir_exists(default_project_path)) {
		Error error = dir_access->make_dir_recursive(default_project_path);
		if (error != OK) {
			ERR_PRINT("Could not create default project directory at: " + default_project_path);
		}
	}

	String autoscan_path = EDITOR_GET("filesystem/directories/autoscan_project_path");
	if (autoscan_path != "") {
		if (dir_access->dir_exists(autoscan_path)) {
			_scan_begin(autoscan_path);
		}
	}

	SceneTree::get_singleton()->connect("files_dropped", this, "_files_dropped");
	SceneTree::get_singleton()->connect("global_menu_action", this, "_global_menu_action");

	run_error_diag = memnew(AcceptDialog);
	gui_base->add_child(run_error_diag);
	run_error_diag->set_title(TTR("Can't run project"));

	dialog_error = memnew(AcceptDialog);
	gui_base->add_child(dialog_error);

	open_templates = memnew(ConfirmationDialog);
	open_templates->set_text(TTR("You currently don't have any projects.\nWould you like to explore official example projects in the Asset Library?"));
	open_templates->get_ok()->set_text(TTR("Open Asset Library"));
	open_templates->connect("confirmed", this, "_open_asset_library");
	add_child(open_templates);

	about = memnew(EditorAbout);
	add_child(about);
}

ProjectManager::~ProjectManager() {
	if (EditorSettings::get_singleton()) {
		EditorSettings::destroy();
	}
}
