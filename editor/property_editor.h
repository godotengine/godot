/*************************************************************************/
/*  property_editor.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef PROPERTY_EDITOR_H
#define PROPERTY_EDITOR_H

#include "editor/editor_file_dialog.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class PropertyValueEvaluator;
class CreateDialog;
class PropertySelector;

class EditorResourceConversionPlugin : public Reference {

	GDCLASS(EditorResourceConversionPlugin, Reference)

protected:
	static void _bind_methods();

public:
	virtual String converts_to() const;
	virtual bool handles(const Ref<Resource> &p_resource) const;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource);
};

class CustomPropertyEditor : public Popup {

	GDCLASS(CustomPropertyEditor, Popup);

	enum {
		MAX_VALUE_EDITORS = 12,
		MAX_ACTION_BUTTONS = 5,
		OBJ_MENU_LOAD = 0,
		OBJ_MENU_EDIT = 1,
		OBJ_MENU_CLEAR = 2,
		OBJ_MENU_MAKE_UNIQUE = 3,
		OBJ_MENU_COPY = 4,
		OBJ_MENU_PASTE = 5,
		OBJ_MENU_NEW_SCRIPT = 6,
		OBJ_MENU_SHOW_IN_FILE_SYSTEM = 7,
		TYPE_BASE_ID = 100,
		CONVERT_BASE_ID = 1000
	};

	enum {
		EASING_LINEAR,
		EASING_EASE_IN,
		EASING_EASE_OUT,
		EASING_ZERO,
		EASING_IN_OUT,
		EASING_OUT_IN
	};

	PopupMenu *menu;
	SceneTreeDialog *scene_tree;
	EditorFileDialog *file;
	ConfirmationDialog *error;
	String name;
	Variant::Type type;
	Variant v;
	List<String> field_names;
	int hint;
	String hint_text;
	LineEdit *value_editor[MAX_VALUE_EDITORS];
	int focused_value_editor;
	Label *value_label[MAX_VALUE_EDITORS];
	HScrollBar *scroll[4];
	Button *action_buttons[MAX_ACTION_BUTTONS];
	MenuButton *type_button;
	Vector<String> inheritors_array;
	TextureRect *texture_preview;
	ColorPicker *color_picker;
	TextEdit *text_edit;
	bool read_only;
	bool picking_viewport;
	GridContainer *checks20gc;
	CheckBox *checks20[20];
	SpinBox *spinbox;
	HSlider *slider;

	Control *easing_draw;
	CreateDialog *create_dialog;
	PropertySelector *property_select;

	Object *owner;

	bool updating;

	PropertyValueEvaluator *evaluator;

	void _text_edit_changed();
	void _file_selected(String p_file);
	void _modified(String p_string);
	void _range_modified(double p_value);
	void _focus_enter();
	void _focus_exit();
	void _action_pressed(int p_which);
	void _type_create_selected(int p_idx);
	void _create_dialog_callback();
	void _create_selected_property(const String &p_prop);

	void _color_changed(const Color &p_color);
	void _draw_easing();
	void _menu_option(int p_which);

	void _drag_easing(const Ref<InputEvent> &p_ev);

	void _node_path_selected(NodePath p_path);
	void show_value_editors(int p_amount);
	void config_value_editors(int p_amount, int p_columns, int p_label_w, const List<String> &p_strings);
	void config_action_buttons(const List<String> &p_strings);

	void _emit_changed_whole_or_field();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void hide_menu();

	Variant get_variant() const;
	String get_name() const;

	void set_read_only(bool p_read_only) { read_only = p_read_only; }

	void set_value_evaluator(PropertyValueEvaluator *p_evaluator) { evaluator = p_evaluator; }

	bool edit(Object *p_owner, const String &p_name, Variant::Type p_type, const Variant &p_variant, int p_hint, String p_hint_text);

	CustomPropertyEditor();
};

class PropertyEditor : public Control {

	GDCLASS(PropertyEditor, Control);

	Tree *tree;
	Label *top_label;
	LineEdit *search_box;

	PropertyValueEvaluator *evaluator;

	Object *obj;

	StringName _prop_edited;

	bool capitalize_paths;
	bool changing;
	bool update_tree_pending;
	bool autoclear;
	bool keying;
	bool read_only;
	bool show_categories;
	bool show_type_icons;
	float refresh_countdown;
	bool use_doc_hints;
	bool use_filter;
	bool subsection_selectable;
	bool hide_script;
	bool use_folding;
	bool property_selectable;

	bool updating_folding;

	HashMap<String, String> pending;
	String selected_property;

	Map<StringName, Map<StringName, String> > descr_cache;
	Map<StringName, String> class_descr_cache;

	CustomPropertyEditor *custom_editor;

	void _resource_edit_request();
	void _custom_editor_edited();
	void _custom_editor_edited_field(const String &p_field_name);
	void _custom_editor_request(bool p_arrow);

	void _item_selected();
	void _item_rmb_edited();
	void _item_edited();
	TreeItem *get_parent_node(String p_path, HashMap<String, TreeItem *> &item_paths, TreeItem *root, TreeItem *category);

	void set_item_text(TreeItem *p_item, int p_type, const String &p_name, int p_hint = PROPERTY_HINT_NONE, const String &p_hint_text = "");

	TreeItem *find_item(TreeItem *p_item, const String &p_name);

	virtual void _changed_callback(Object *p_changed, const char *p_prop);
	virtual void _changed_callbacks(Object *p_changed, const String &p_prop);

	void _check_reload_status(const String &p_name, TreeItem *item);

	void _edit_button(Object *p_item, int p_column, int p_button);

	void _node_removed(Node *p_node);

	friend class ProjectExportDialog;
	void _edit_set(const String &p_name, const Variant &p_value, bool p_refresh_all = false, const String &p_changed_field = "");
	void _draw_flags(Object *p_object, const Rect2 &p_rect);

	bool _might_be_in_instance();
	bool _get_instanced_node_original_property(const StringName &p_prop, Variant &value);
	bool _is_property_different(const Variant &p_current, const Variant &p_orig, int p_usage = 0);

	void _refresh_item(TreeItem *p_item);
	void _set_range_def(Object *p_item, String prop, float p_frame);

	void _filter_changed(const String &p_text);

	void _mark_drop_fields(TreeItem *p_at);
	void _clear_drop_fields(TreeItem *p_at);

	bool _is_drop_valid(const Dictionary &p_drag_data, const Dictionary &p_item_data) const;
	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _resource_preview_done(const String &p_path, const Ref<Texture> &p_preview, Variant p_ud);
	void _draw_transparency(Object *t, const Rect2 &p_rect);
	void _item_folded(Object *item_obj);

	UndoRedo *undo_redo;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }

	String get_selected_path() const;

	Tree *get_scene_tree();
	Label *get_top_label();
	void hide_top_label();
	void update_tree();
	void update_property(const String &p_prop);

	void refresh();

	void edit(Object *p_object);

	void set_keying(bool p_active);
	void set_read_only(bool p_read_only) {
		read_only = p_read_only;
		custom_editor->set_read_only(p_read_only);
	}

	bool is_capitalize_paths_enabled() const;
	void set_enable_capitalize_paths(bool p_capitalize);
	void set_autoclear(bool p_enable);

	void set_show_categories(bool p_show);
	void set_use_doc_hints(bool p_enable) { use_doc_hints = p_enable; }
	void set_hide_script(bool p_hide) { hide_script = p_hide; }

	void set_use_filter(bool p_use);
	void register_text_enter(Node *p_line_edit);

	void set_subsection_selectable(bool p_selectable);
	void set_property_selectable(bool p_selectable);

	void set_use_folding(bool p_enable);
	PropertyEditor();
	~PropertyEditor();
};

class SectionedPropertyEditorFilter;

class SectionedPropertyEditor : public HBoxContainer {

	GDCLASS(SectionedPropertyEditor, HBoxContainer);

	ObjectID obj;

	Tree *sections;
	SectionedPropertyEditorFilter *filter;

	Map<String, TreeItem *> section_map;
	PropertyEditor *editor;
	LineEdit *search_box;

	static void _bind_methods();
	void _section_selected();

	void _search_changed(const String &p_what);

public:
	void register_search_box(LineEdit *p_box);
	PropertyEditor *get_property_editor();
	void edit(Object *p_object);
	String get_full_item_path(const String &p_item);

	void set_current_section(const String &p_section);
	String get_current_section() const;

	void update_category_list();

	SectionedPropertyEditor();
	~SectionedPropertyEditor();
};

class PropertyValueEvaluator : public ValueEvaluator {
	GDCLASS(PropertyValueEvaluator, ValueEvaluator);

	Object *obj;
	ScriptLanguage *script_language;
	String _build_script(const String &p_text);

	_FORCE_INLINE_ double _default_eval(const String &p_text) {
		return p_text.to_double();
	}

public:
	void edit(Object *p_obj);
	double eval(const String &p_text);

	PropertyValueEvaluator();
	~PropertyValueEvaluator();
};

#endif
