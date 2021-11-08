/*************************************************************************/
/*  editor_inspector.h                                                   */
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

#ifndef EDITOR_INSPECTOR_H
#define EDITOR_INSPECTOR_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_rect.h"

class UndoRedo;

class EditorPropertyRevert {
public:
	static bool get_instantiated_node_original_property(Node *p_node, const StringName &p_prop, Variant &value, bool p_check_class_default = true);
	static bool is_node_property_different(Node *p_node, const Variant &p_current, const Variant &p_orig);
	static bool is_property_value_different(const Variant &p_a, const Variant &p_b);
	static Variant get_property_revert_value(Object *p_object, const StringName &p_property);

	static bool can_property_revert(Object *p_object, const StringName &p_property);
};

class EditorProperty : public Container {
	GDCLASS(EditorProperty, Container);

public:
	enum MenuItems {
		MENU_COPY_PROPERTY,
		MENU_PASTE_PROPERTY,
		MENU_COPY_PROPERTY_PATH,
		MENU_PIN_VALUE,
	};

private:
	String label;
	int text_size;
	friend class EditorInspector;
	Object *object;
	StringName property;

	int property_usage;

	bool read_only;
	bool checkable;
	bool checked;
	bool draw_warning;
	bool keying;
	bool deletable;

	Rect2 right_child_rect;
	Rect2 bottom_child_rect;

	Rect2 keying_rect;
	bool keying_hover = false;
	Rect2 revert_rect;
	bool revert_hover = false;
	Rect2 check_rect;
	bool check_hover = false;
	Rect2 delete_rect;
	bool delete_hover = false;

	bool can_revert;
	bool can_pin;
	bool pin_hidden;
	bool pinned;

	bool use_folding;
	bool draw_top_bg;

	void _update_popup();
	void _focusable_focused(int p_index);

	bool selectable;
	bool selected;
	int selected_focusable;

	float split_ratio;

	Vector<Control *> focusables;
	Control *label_reference;
	Control *bottom_editor;
	PopupMenu *menu;

	mutable String tooltip_text;

	Map<StringName, Variant> cache;

	GDVIRTUAL0(_update_property)
	void _update_pin_flags();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void _set_read_only(bool p_read_only);

	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void unhandled_key_input(const Ref<InputEvent> &p_event) override;
	const Color *_get_property_colors();

public:
	void emit_changed(const StringName &p_property, const Variant &p_value, const StringName &p_field = StringName(), bool p_changing = false);

	virtual Size2 get_minimum_size() const override;

	void set_label(const String &p_label);
	String get_label() const;

	void set_read_only(bool p_read_only);
	bool is_read_only() const;

	Object *get_edited_object();
	StringName get_edited_property();

	virtual void update_property();
	void update_revert_and_pin_status();

	virtual bool use_keying_next() const;

	void set_checkable(bool p_checkable);
	bool is_checkable() const;

	void set_checked(bool p_checked);
	bool is_checked() const;

	void set_draw_warning(bool p_draw_warning);
	bool is_draw_warning() const;

	void set_keying(bool p_keying);
	bool is_keying() const;

	void set_deletable(bool p_enable);
	bool is_deletable() const;
	void add_focusable(Control *p_control);
	void select(int p_focusable = -1);
	void deselect();
	bool is_selected() const;

	void set_label_reference(Control *p_control);
	void set_bottom_editor(Control *p_control);

	void set_use_folding(bool p_use_folding);
	bool is_using_folding() const;

	virtual void expand_all_folding();
	virtual void collapse_all_folding();

	virtual Variant get_drag_data(const Point2 &p_point) override;
	virtual void update_cache();
	virtual bool is_cache_valid() const;

	void set_selectable(bool p_selectable);
	bool is_selectable() const;

	void set_name_split_ratio(float p_ratio);
	float get_name_split_ratio() const;

	void set_object_and_property(Object *p_object, const StringName &p_property);
	virtual Control *make_custom_tooltip(const String &p_text) const override;

	String get_tooltip_text() const;

	void set_draw_top_bg(bool p_draw) { draw_top_bg = p_draw; }

	bool can_revert_to_default() const { return can_revert; }

	void menu_option(int p_option);

	EditorProperty();
};

class EditorInspectorPlugin : public RefCounted {
	GDCLASS(EditorInspectorPlugin, RefCounted);

	friend class EditorInspector;
	struct AddedEditor {
		Control *property_editor = nullptr;
		Vector<String> properties;
		String label;
	};

	List<AddedEditor> added_editors;

protected:
	static void _bind_methods();

	GDVIRTUAL1RC(bool, _can_handle, Variant)
	GDVIRTUAL0(_parse_begin)
	GDVIRTUAL2(_parse_category, Object *, String)
	GDVIRTUAL7R(bool, _parse_property, Object *, int, String, int, String, int, bool)
	GDVIRTUAL0(_parse_end)

public:
	void add_custom_control(Control *control);
	void add_property_editor(const String &p_for_property, Control *p_prop);
	void add_property_editor_for_multiple_properties(const String &p_label, const Vector<String> &p_properties, Control *p_prop);

	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
	virtual void parse_category(Object *p_object, const String &p_parse_category);
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide = false);
	virtual void parse_end();
};

class EditorInspectorCategory : public Control {
	GDCLASS(EditorInspectorCategory, Control);

	friend class EditorInspector;
	Ref<Texture2D> icon;
	String label;

	mutable String tooltip_text;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual Size2 get_minimum_size() const override;
	virtual Control *make_custom_tooltip(const String &p_text) const override;

	String get_tooltip_text() const;

	EditorInspectorCategory();
};

class EditorInspectorSection : public Container {
	GDCLASS(EditorInspectorSection, Container);

	String label;
	String section;
	bool vbox_added; // Optimization.
	Color bg_color;
	bool foldable;

	Timer *dropping_unfold_timer;
	bool dropping;

	void _test_unfold();

protected:
	Object *object;
	VBoxContainer *vbox;

	void _notification(int p_what);
	static void _bind_methods();
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

public:
	virtual Size2 get_minimum_size() const override;

	void setup(const String &p_section, const String &p_label, Object *p_object, const Color &p_bg_color, bool p_foldable);
	VBoxContainer *get_vbox();
	void unfold();
	void fold();

	EditorInspectorSection();
	~EditorInspectorSection();
};

class EditorInspectorArray : public EditorInspectorSection {
	GDCLASS(EditorInspectorArray, EditorInspectorSection);

	UndoRedo *undo_redo;

	enum Mode {
		MODE_NONE,
		MODE_USE_COUNT_PROPERTY,
		MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION,
	} mode;
	StringName count_property;
	StringName array_element_prefix;

	int count = 0;

	VBoxContainer *elements_vbox;

	Control *control_dropping;
	bool dropping = false;

	Button *add_button;

	AcceptDialog *resize_dialog;
	int new_size = 0;
	LineEdit *new_size_line_edit;

	// Pagination
	int page_lenght = 5;
	int page = 0;
	int max_page = 0;
	int begin_array_index = 0;
	int end_array_index = 0;
	HBoxContainer *hbox_pagination;
	Button *first_page_button;
	Button *prev_page_button;
	LineEdit *page_line_edit;
	Label *page_count_label;
	Button *next_page_button;
	Button *last_page_button;

	enum MenuOptions {
		OPTION_MOVE_UP = 0,
		OPTION_MOVE_DOWN,
		OPTION_NEW_BEFORE,
		OPTION_NEW_AFTER,
		OPTION_REMOVE,
		OPTION_CLEAR_ARRAY,
		OPTION_RESIZE_ARRAY,
	};
	int popup_array_index_pressed = -1;
	PopupMenu *rmb_popup;

	struct ArrayElement {
		PanelContainer *panel;
		MarginContainer *margin;
		HBoxContainer *hbox;
		TextureRect *move_texture_rect;
		VBoxContainer *vbox;
	};
	LocalVector<ArrayElement> array_elements;

	Ref<StyleBoxFlat> odd_style;
	Ref<StyleBoxFlat> even_style;

	int _get_array_count();
	void _add_button_pressed();

	void _first_page_button_pressed();
	void _prev_page_button_pressed();
	void _page_line_edit_text_submitted(String p_text);
	void _next_page_button_pressed();
	void _last_page_button_pressed();

	void _rmb_popup_id_pressed(int p_id);

	void _control_dropping_draw();

	void _vbox_visibility_changed();

	void _panel_draw(int p_index);
	void _panel_gui_input(Ref<InputEvent> p_event, int p_index);
	void _move_element(int p_element_index, int p_to_pos);
	void _clear_array();
	void _resize_array(int p_size);
	Array _extract_properties_as_array(const List<PropertyInfo> &p_list);
	int _drop_position() const;

	void _new_size_line_edit_text_changed(String p_text);
	void _new_size_line_edit_text_submitted(String p_text);
	void _resize_dialog_confirmed();

	void _update_elements_visibility();
	void _setup();

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_undo_redo(UndoRedo *p_undo_redo);

	void setup_with_move_element_function(Object *p_object, String p_label, const StringName &p_array_element_prefix, int p_page, const Color &p_bg_color, bool p_foldable);
	void setup_with_count_property(Object *p_object, String p_label, const StringName &p_count_property, const StringName &p_array_element_prefix, int p_page, const Color &p_bg_color, bool p_foldable);
	VBoxContainer *get_vbox(int p_index);

	EditorInspectorArray();
};

class EditorInspector : public ScrollContainer {
	GDCLASS(EditorInspector, ScrollContainer);

	UndoRedo *undo_redo;
	enum {
		MAX_PLUGINS = 1024
	};
	static Ref<EditorInspectorPlugin> inspector_plugins[MAX_PLUGINS];
	static int inspector_plugin_count;

	VBoxContainer *main_vbox;

	//map use to cache the instantiated editors
	Map<StringName, List<EditorProperty *>> editor_property_map;
	List<EditorInspectorSection *> sections;
	Set<StringName> pending;

	void _clear();
	Object *object;

	//

	LineEdit *search_box;
	bool show_categories;
	bool hide_script;
	bool use_doc_hints;
	bool capitalize_paths;
	bool use_filter;
	bool autoclear;
	bool use_folding;
	int changing;
	bool update_all_pending;
	bool read_only;
	bool keying;
	bool sub_inspector;
	bool wide_editors;
	bool deletable_properties;

	float refresh_countdown;
	bool update_tree_pending;
	StringName _prop_edited;
	StringName property_selected;
	int property_focusable;
	int update_scroll_request;

	Map<StringName, Map<StringName, String>> descr_cache;
	Map<StringName, String> class_descr_cache;
	Set<StringName> restart_request_props;

	Map<ObjectID, int> scroll_cache;

	String property_prefix; //used for sectioned inspector
	String object_class;
	Variant property_clipboard;

	bool restrict_to_basic = false;

	void _edit_set(const String &p_name, const Variant &p_value, bool p_refresh_all, const String &p_changed_field);

	void _property_changed(const String &p_path, const Variant &p_value, const String &p_name = "", bool p_changing = false, bool p_update_all = false);
	void _multiple_properties_changed(Vector<String> p_paths, Array p_values, bool p_changing = false);
	void _property_keyed(const String &p_path, bool p_advance);
	void _property_keyed_with_value(const String &p_path, const Variant &p_value, bool p_advance);
	void _property_deleted(const String &p_path);
	void _property_checked(const String &p_path, bool p_checked);
	void _property_pinned(const String &p_path, bool p_pinned);

	void _resource_selected(const String &p_path, RES p_resource);
	void _property_selected(const String &p_path, int p_focusable);
	void _object_id_selected(const String &p_path, ObjectID p_id);

	void _node_removed(Node *p_node);

	Map<StringName, int> per_array_page;
	void _page_change_request(int p_new_page, const StringName &p_array_prefix);

	void _changed_callback();
	void _edit_request_change(Object *p_object, const String &p_prop);

	void _filter_changed(const String &p_text);
	void _parse_added_editors(VBoxContainer *current_vbox, Ref<EditorInspectorPlugin> ped);

	void _vscroll_changed(double);

	void _feature_profile_changed();
	void _update_script_class_properties(const Object &p_object, List<PropertyInfo> &r_list) const;

	bool _is_property_disabled_by_feature_profile(const StringName &p_property);

	void _update_inspector_bg();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	static void add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	static void remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	static void cleanup_plugins();

	static EditorProperty *instantiate_property_editor(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide = false);

	void set_undo_redo(UndoRedo *p_undo_redo);

	String get_selected_path() const;

	void update_tree();
	void update_property(const String &p_prop);
	void edit(Object *p_object);
	Object *get_edited_object();

	void set_keying(bool p_active);
	void set_read_only(bool p_read_only);

	bool is_capitalize_paths_enabled() const;
	void set_enable_capitalize_paths(bool p_capitalize);
	void set_autoclear(bool p_enable);

	void set_show_categories(bool p_show);
	void set_use_doc_hints(bool p_enable);
	void set_hide_script(bool p_hide);

	void set_use_filter(bool p_use);
	void register_text_enter(Node *p_line_edit);

	void set_use_folding(bool p_enable);
	bool is_using_folding();

	void collapse_all_folding();
	void expand_all_folding();

	void set_scroll_offset(int p_offset);
	int get_scroll_offset() const;

	void set_property_prefix(const String &p_prefix);
	String get_property_prefix() const;

	void set_object_class(const String &p_class);
	String get_object_class() const;

	void set_use_wide_editors(bool p_enable);
	void set_sub_inspector(bool p_enable);
	bool is_sub_inspector() const { return sub_inspector; }

	void set_use_deletable_properties(bool p_enabled);

	void set_restrict_to_basic_settings(bool p_restrict);
	void set_property_clipboard(const Variant &p_value);
	Variant get_property_clipboard() const;

	EditorInspector();
};

#endif // INSPECTOR_H
