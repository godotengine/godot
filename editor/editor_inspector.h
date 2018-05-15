#ifndef EDITOR_INSPECTOR_H
#define EDITOR_INSPECTOR_H

#include "editor_data.h"
#include "scene/gui/scroll_container.h"

class EditorProperty : public Container {

	GDCLASS(EditorProperty, Container)
public:
	enum LabelLayout {
		LABEL_LAYOUT_LEFT,
		LABEL_LAYOUT_TOP,
	};

private:
	String label;
	int text_size;
	friend class EditorInspector;
	Object *object;
	StringName property;

	LabelLayout label_layout;

	int property_usage;

	bool read_only;
	bool checkable;
	bool checked;
	bool draw_red;
	bool keying;

	Rect2 keying_rect;
	bool keying_hover;
	Rect2 revert_rect;
	bool revert_hover;
	Rect2 check_rect;
	bool check_hover;

	bool can_revert;

	bool _might_be_in_instance();
	bool _is_property_different(const Variant &p_current, const Variant &p_orig, int p_usage);
	bool _is_instanced_node_with_original_property_different();
	bool _get_instanced_node_original_property(const StringName &p_prop, Variant &value);
	void _focusable_focused(int p_index);

	bool selected;
	int selected_focusable;

	Vector<Control *> focusables;
	Control *label_reference;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	void _gui_input(const Ref<InputEvent> &p_event);

public:
	virtual Size2 get_minimum_size() const;

	void set_label(const String &p_label);
	String get_label() const;

	void set_read_only(bool p_read_only);
	bool is_read_only() const;

	Object *get_edited_object();
	StringName get_edited_property();

	virtual void update_property();
	void update_reload_status();

	virtual bool use_keying_next() const;

	void set_checkable(bool p_checkable);
	bool is_checkable() const;

	void set_checked(bool p_checked);
	bool is_checked() const;

	void set_draw_red(bool p_draw_red);
	bool is_draw_red() const;

	void set_keying(bool p_keying);
	bool is_keying() const;

	void add_focusable(Control *p_control);
	void select(int p_focusable = -1);
	void deselect();
	bool is_selected() const;

	void set_label_reference(Control *p_control);

	virtual Variant get_drag_data(const Point2 &p_point);

	void set_label_layout(LabelLayout p_layout);
	EditorProperty();
};

class EditorInspectorPlugin : public Reference {
	GDCLASS(EditorInspectorPlugin, Reference)

	friend class EditorInspector;
	struct AddedEditor {
		Control *property_editor;
		Vector<String> properties;
		String label;
	};

	List<AddedEditor> added_editors;

protected:
	static void _bind_methods();

public:
	void add_custom_control(Control *control);
	void add_property_editor(const String &p_for_property, Control *p_prop);
	void add_property_editor_for_multiple_properties(const String &p_label, const Vector<String> &p_properties, Control *p_prop);

	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
	virtual bool parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage);
	virtual void parse_end();
};

class EditorInspectorCategory : public Control {
	GDCLASS(EditorInspectorCategory, Control);

	friend class EditorInspector;
	Ref<Texture> icon;
	String label;
	Color bg_color;

protected:
	void _notification(int p_what);

public:
	virtual Size2 get_minimum_size() const;

	EditorInspectorCategory();
};

class EditorInspectorSection : public Container {
	GDCLASS(EditorInspectorSection, Container);

	String label;
	String section;
	Object *object;
	VBoxContainer *vbox;
	Color bg_color;
	bool foldable;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _gui_input(const Ref<InputEvent> &p_event);

public:
	virtual Size2 get_minimum_size() const;

	void setup(const String &p_section, const String &p_label, Object *p_object, const Color &p_bg_color, bool p_foldable);
	VBoxContainer *get_vbox();
	void unfold();
	void fold();

	Object *get_edited_object();

	EditorInspectorSection();
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

	//map use to cache the instanced editors
	Map<StringName, List<EditorProperty *> > editor_property_map;
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

	int refresh_countdown;
	bool update_tree_pending;
	StringName _prop_edited;
	StringName property_selected;
	int property_focusable;

	Map<StringName, Map<StringName, String> > descr_cache;
	Map<StringName, String> class_descr_cache;

	void _edit_set(const String &p_name, const Variant &p_value, bool p_refresh_all, const String &p_changed_field);

	void _property_changed(const String &p_path, const Variant &p_value);
	void _multiple_properties_changed(Vector<String> p_paths, Array p_values);
	void _property_keyed(const String &p_path);
	void _property_checked(const String &p_path, bool p_checked);

	void _resource_selected(const String &p_path, RES p_resource);
	void _property_selected(const String &p_path, int p_focusable);
	void _object_id_selected(const String &p_path, ObjectID p_id);

	void _node_removed(Node *p_node);

	void _changed_callback(Object *p_changed, const char *p_prop);
	void _edit_request_change(Object *p_changed, const String &p_prop);

	void _filter_changed(const String &p_text);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	static void add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	static void remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	static void cleanup_plugins();

	void set_undo_redo(UndoRedo *p_undo_redo);

	String get_selected_path() const;

	void update_tree();
	void update_property(const String &p_prop);

	void refresh();

	void edit(Object *p_object);

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

	void set_subsection_selectable(bool p_selectable);
	void set_property_selectable(bool p_selectable);

	void set_use_folding(bool p_enable);

	void collapse_all_folding();
	void expand_all_folding();

	void set_scroll_offset(int p_offset);
	int get_scroll_offset() const;

	EditorInspector();
};

#endif // INSPECTOR_H
