/**************************************************************************/
/*  file_dialog.h                                                         */
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

#include "scene/gui/dialogs.h"
#include "scene/property_list_helper.h"

class DirAccess;
class FlowContainer;
class GridContainer;
class HBoxContainer;
class ItemList;
class LineEdit;
class MenuButton;
class OptionButton;
class PopupMenu;
class VBoxContainer;
class VSeparator;

class FileDialog : public ConfirmationDialog {
	GDCLASS(FileDialog, ConfirmationDialog);

	inline static constexpr int MAX_RECENTS = 20;

	struct Option {
		String name;
		Vector<String> values;
		int default_idx = 0;
	};

	struct DirInfo {
		String name;
		uint64_t modified_time = 0;
		bool bundle = false;

		struct NameComparator {
			bool operator()(const DirInfo &p_a, const DirInfo &p_b) const {
				return FileNoCaseComparator()(p_a.name, p_b.name);
			}
		};

		struct TimeComparator {
			bool operator()(const DirInfo &p_a, const DirInfo &p_b) const {
				return p_a.modified_time > p_b.modified_time;
			}
		};
	};

	struct FileInfo {
		String name;
		String match_string;
		String type_sort;
		uint64_t modified_time = 0;

		struct NameComparator {
			bool operator()(const FileInfo &p_a, const FileInfo &p_b) const {
				return FileNoCaseComparator()(p_a.name, p_b.name);
			}
		};

		struct TypeComparator {
			bool operator()(const FileInfo &p_a, const FileInfo &p_b) const {
				return FileNoCaseComparator()(p_a.type_sort, p_b.type_sort);
			}
		};

		struct TimeComparator {
			bool operator()(const FileInfo &p_a, const FileInfo &p_b) const {
				return p_a.modified_time > p_b.modified_time;
			}
		};
	};

	enum class FileSortOption {
		NAME,
		NAME_REVERSE,
		TYPE,
		TYPE_REVERSE,
		MODIFIED_TIME,
		MODIFIED_TIME_REVERSE,
		MAX
	};

public:
	enum Access {
		ACCESS_RESOURCES,
		ACCESS_USERDATA,
		ACCESS_FILESYSTEM,
	};

	enum FileMode {
		FILE_MODE_OPEN_FILE,
		FILE_MODE_OPEN_FILES,
		FILE_MODE_OPEN_DIR,
		FILE_MODE_OPEN_ANY,
		FILE_MODE_SAVE_FILE,
	};

	enum DisplayMode {
		DISPLAY_THUMBNAILS,
		DISPLAY_LIST,
		DISPLAY_MAX
	};

	enum ItemMenu {
		ITEM_MENU_COPY_PATH,
		ITEM_MENU_DELETE,
		ITEM_MENU_REFRESH,
		ITEM_MENU_NEW_FOLDER,
		ITEM_MENU_SHOW_IN_EXPLORER,
		ITEM_MENU_SHOW_BUNDLE_CONTENT,
		// Not in the menu, only for shortcuts.
		ITEM_MENU_GO_UP,
		ITEM_MENU_TOGGLE_HIDDEN,
		ITEM_MENU_FIND,
		ITEM_MENU_FOCUS_PATH,
	};

	enum Customization {
		CUSTOMIZATION_HIDDEN_FILES,
		CUSTOMIZATION_CREATE_FOLDER,
		CUSTOMIZATION_FILE_FILTER,
		CUSTOMIZATION_FILE_SORT,
		CUSTOMIZATION_FAVORITES,
		CUSTOMIZATION_RECENT,
		CUSTOMIZATION_LAYOUT,
		CUSTOMIZATION_OVERWRITE_WARNING,
		CUSTOMIZATION_DELETE,
		CUSTOMIZATION_MAX
	};

	typedef void (*RegisterFunc)(FileDialog *);

	inline static Callable get_icon_callback;
	inline static Callable get_thumbnail_callback;

	inline static RegisterFunc register_func = nullptr;
	inline static RegisterFunc unregister_func = nullptr;

private:
	static inline PropertyListHelper base_property_helper;
	PropertyListHelper property_helper;

	inline static bool default_show_hidden_files = false;
	static inline DisplayMode default_display_mode = DISPLAY_THUMBNAILS;
	bool show_hidden_files = false;
	bool use_native_dialog = false;
	bool can_create_folders = true;
	bool customization_flags[CUSTOMIZATION_MAX]; // Initialized to true in the constructor.

	HashMap<ItemMenu, Ref<Shortcut>> action_shortcuts;

	inline static LocalVector<String> global_favorites;
	inline static LocalVector<String> global_recents;

	Access access = ACCESS_RESOURCES;
	FileMode mode = FILE_MODE_SAVE_FILE;
	DisplayMode display_mode = DISPLAY_THUMBNAILS;
	FileSortOption file_sort = FileSortOption::NAME;

	Vector<String> filters;
	Vector<String> processed_filters;

	Vector<Option> options;
	Dictionary selected_options;
	bool options_dirty = false;

	String file_name_filter;
	bool show_filename_filter = false;

	Vector<String> local_history;
	int local_history_pos = 0;

	bool mode_overrides_title = true;
	String root_subfolder;
	String root_prefix;
	String full_dir;

	bool is_invalidating = false;

	VBoxContainer *main_vbox = nullptr;

	Button *dir_prev = nullptr;
	Button *dir_next = nullptr;
	Button *dir_up = nullptr;

	HBoxContainer *drives_container = nullptr;
	OptionButton *drives = nullptr;
	LineEdit *directory_edit = nullptr;
	HBoxContainer *shortcuts_container = nullptr;

	Button *refresh_button = nullptr;
	Button *favorite_button = nullptr;
	HBoxContainer *make_dir_container = nullptr;
	Button *make_dir_button = nullptr;

	Button *show_hidden = nullptr;
	VSeparator *show_hidden_separator = nullptr;
	HBoxContainer *layout_container = nullptr;
	VSeparator *layout_separator = nullptr;
	Button *thumbnail_mode_button = nullptr;
	Button *list_mode_button = nullptr;
	Button *show_filename_filter_button = nullptr;
	MenuButton *file_sort_button = nullptr;

	VBoxContainer *favorite_vbox = nullptr;
	Button *fav_up_button = nullptr;
	Button *fav_down_button = nullptr;
	ItemList *favorite_list = nullptr;
	VBoxContainer *recent_vbox = nullptr;
	ItemList *recent_list = nullptr;

	ItemList *file_list = nullptr;
	Label *message = nullptr;
	PopupMenu *item_menu = nullptr;

	HBoxContainer *filename_filter_box = nullptr;
	LineEdit *filename_filter = nullptr;

	HBoxContainer *file_box = nullptr;
	LineEdit *filename_edit = nullptr;
	OptionButton *filter = nullptr;

	FlowContainer *flow_checkbox_options = nullptr;
	GridContainer *grid_select_options = nullptr;

	ConfirmationDialog *delete_dialog = nullptr;
	ConfirmationDialog *make_dir_dialog = nullptr;
	LineEdit *new_dir_name = nullptr;
	AcceptDialog *mkdirerr = nullptr;
	AcceptDialog *exterr = nullptr;
	ConfirmationDialog *confirm_save = nullptr;

	struct ThemeCache {
		int thumbnail_size = 64;

		Ref<Texture2D> parent_folder;
		Ref<Texture2D> forward_folder;
		Ref<Texture2D> back_folder;
		Ref<Texture2D> reload;
		Ref<Texture2D> toggle_hidden;
		Ref<Texture2D> toggle_filename_filter;
		Ref<Texture2D> thumbnail_mode;
		Ref<Texture2D> list_mode;
		Ref<Texture2D> folder;
		Ref<Texture2D> file;
		Ref<Texture2D> create_folder;
		Ref<Texture2D> sort;
		Ref<Texture2D> favorite;
		Ref<Texture2D> favorite_up;
		Ref<Texture2D> favorite_down;
		Ref<Texture2D> file_thumbnail;
		Ref<Texture2D> folder_thumbnail;

		Color folder_icon_color;
		Color file_icon_color;
		Color file_disabled_color;

		Color icon_normal_color;
		Color icon_hover_color;
		Color icon_focus_color;
		Color icon_pressed_color;
	} theme_cache;

	void update_dir();
	void update_file_name();
	void update_file_list();
	void update_filename_filter();
	void update_filename_filter_gui();
	void update_filters();
	void update_customization();

	void _empty_clicked(const Vector2 &p_pos, MouseButton p_button);
	void _item_clicked(int p_item, const Vector2 &p_pos, MouseButton p_button);
	void _popup_menu(const Vector2 &p_pos, int p_for_item);

	void _focus_file_text();

	int _get_selected_file_idx();
	String _get_item_path(int p_idx) const;
	void _file_list_multi_selected(int p_item, bool p_selected);
	void _file_list_selected(int p_item);
	void _file_list_item_activated(int p_item);

	void _select_drive(int p_idx);
	void _dir_submitted(String p_dir);
	void _action_pressed();
	void _save_confirm_pressed();
	void _cancel_pressed();
	void _filter_selected(int);
	void _filename_filter_changed();
	void _filename_filter_selected();
	void _file_list_select_first();
	void _delete_confirm();
	void _make_dir();
	void _make_dir_confirm();
	void _go_up();
	void _go_back();
	void _go_forward();
	void _push_history();

	void _change_dir(const String &p_new_dir);
	void _update_drives(bool p_select = true);
	void _sort_option_selected(int p_option);

	void _favorite_selected(int p_item);
	void _favorite_pressed();
	void _favorite_move_up();
	void _favorite_move_down();
	void _update_favorite_list();
	void _update_fav_buttons();

	void _recent_selected(int p_item);
	void _save_to_recent();
	void _update_recent_list();
	bool _path_matches_access(const String &p_path) const;

	void _invalidate();
	void _setup_button(Button *p_button, const Ref<Texture2D> &p_icon);
	void _update_make_dir_visible();

	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

	void _native_popup();
	void _native_dialog_cb(bool p_ok, const Vector<String> &p_files, int p_filter);
	void _native_dialog_cb_with_options(bool p_ok, const Vector<String> &p_files, int p_filter, const Dictionary &p_selected_options);

	bool _is_open_should_be_disabled();
	void _thumbnail_callback(const Ref<Texture2D> &p_texture, const String &p_path);

	TypedArray<Dictionary> _get_options() const;
	void _update_option_controls();
	void _option_changed_checkbox_toggled(bool p_pressed, const String &p_name);
	void _option_changed_item_selected(int p_idx, const String &p_name);

	virtual void _post_popup() override;

protected:
	Ref<DirAccess> dir_access;

	bool _can_use_native_popup() const;
	virtual void _item_menu_id_pressed(int p_option);
	virtual void _dir_contents_changed() {}

	virtual bool _should_use_native_popup() const;
	virtual bool _should_hide_file(const String &p_file) const { return false; }
	virtual Color _get_folder_color(const String &p_path) const { return theme_cache.folder_icon_color; }

	virtual void _popup_base(const Rect2i &p_screen_rect = Rect2i()) override;

	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value) { return property_helper.property_set_value(p_name, p_value); }
	bool _get(const StringName &p_name, Variant &r_ret) const { return property_helper.property_get_value(p_name, r_ret); }
	void _get_property_list(List<PropertyInfo> *p_list) const { property_helper.get_property_list(p_list); }
	bool _property_can_revert(const StringName &p_name) const { return property_helper.property_can_revert(p_name); }
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const { return property_helper.property_get_revert(p_name, r_property); }

	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	void _add_filter_bind_compat_111439(const String &p_filter, const String &p_description = "");

	static void _bind_compatibility_methods();
#endif

public:
	virtual void set_visible(bool p_visible) override;

	void popup_file_dialog();
	void clear_filters();
	void add_filter(const String &p_filter, const String &p_description = "", const String &p_mime = "");
	void set_filters(const Vector<String> &p_filters);
	Vector<String> get_filters() const;
	void clear_filename_filter();
	void set_filename_filter(const String &p_filename_filter);
	String get_filename_filter() const;

	Vector<String> get_selected_files() const;

	String get_current_dir() const;
	String get_current_file() const;
	String get_current_path() const;
	void set_current_dir(const String &p_dir);
	void set_current_file(const String &p_file);
	void set_current_path(const String &p_path);

	String get_option_name(int p_option) const;
	Vector<String> get_option_values(int p_option) const;
	int get_option_default(int p_option) const;
	void set_option_name(int p_option, const String &p_name);
	void set_option_values(int p_option, const Vector<String> &p_values);
	void set_option_default(int p_option, int p_index);

	void add_option(const String &p_name, const Vector<String> &p_values, int p_index);

	void set_option_count(int p_count);
	int get_option_count() const;

	Dictionary get_selected_options() const;

	void set_root_subfolder(const String &p_root);
	String get_root_subfolder() const;

	void set_mode_overrides_title(bool p_override);
	bool is_mode_overriding_title() const;

	void set_use_native_dialog(bool p_native);
	bool get_use_native_dialog() const;

	void set_file_mode(FileMode p_mode);
	FileMode get_file_mode() const;

	void set_display_mode(DisplayMode p_mode);
	DisplayMode get_display_mode() const;

	static void set_favorite_list(const PackedStringArray &p_favorites);
	static PackedStringArray get_favorite_list();

	static void set_recent_list(const PackedStringArray &p_recents);
	static PackedStringArray get_recent_list();

	void set_customization_flag_enabled(Customization p_flag, bool p_enabled);
	bool is_customization_flag_enabled(Customization p_flag) const;

	VBoxContainer *get_vbox() { return main_vbox; }
	LineEdit *get_line_edit() { return filename_edit; }
	ItemList *get_file_item_list() { return file_list; }

	void set_access(Access p_access);
	Access get_access() const;

	void set_show_hidden_files(bool p_show);
	bool is_showing_hidden_files() const;
	void set_show_filename_filter(bool p_show);
	bool get_show_filename_filter() const;

	static void set_default_show_hidden_files(bool p_show);
	static void set_default_display_mode(DisplayMode p_mode);

	static void set_get_icon_callback(const Callable &p_callback);
	static void set_get_thumbnail_callback(const Callable &p_callback);

	void invalidate();

	void deselect_all();

	FileDialog();
	~FileDialog();
};

VARIANT_ENUM_CAST(FileDialog::FileMode);
VARIANT_ENUM_CAST(FileDialog::Access);
VARIANT_ENUM_CAST(FileDialog::DisplayMode);
VARIANT_ENUM_CAST(FileDialog::Customization);
