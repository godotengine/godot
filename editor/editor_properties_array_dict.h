#ifndef EDITOR_PROPERTIES_ARRAY_DICT_H
#define EDITOR_PROPERTIES_ARRAY_DICT_H

#include "editor/editor_inspector.h"
#include "editor/editor_spin_slider.h"
#include "scene/gui/button.h"

class EditorPropertyArrayObject : public Reference {

	GDCLASS(EditorPropertyArrayObject, Reference);

	Variant array;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;

public:
	void set_array(const Variant &p_array);
	Variant get_array();

	EditorPropertyArrayObject();
};

class EditorPropertyDictionaryObject : public Reference {

	GDCLASS(EditorPropertyDictionaryObject, Reference);

	Variant new_item_key;
	Variant new_item_value;
	Dictionary dict;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;

public:
	void set_dict(const Dictionary &p_dict);
	Dictionary get_dict();

	void set_new_item_key(const Variant &p_new_item);
	Variant get_new_item_key();

	void set_new_item_value(const Variant &p_new_item);
	Variant get_new_item_value();

	EditorPropertyDictionaryObject();
};

class EditorPropertyArray : public EditorProperty {
	GDCLASS(EditorPropertyArray, EditorProperty)

	PopupMenu *change_type;
	bool updating;

	Ref<EditorPropertyArrayObject> object;
	int page_len;
	int page_idx;
	int changing_type_idx;
	Button *edit;
	VBoxContainer *vbox;
	EditorSpinSlider *length;
	EditorSpinSlider *page;
	HBoxContainer *page_hb;
	Variant::Type array_type;

	void _page_changed(double p_page);
	void _length_changed(double p_page);
	void _edit_pressed();
	void _property_changed(const String &p_prop, Variant p_value, bool changing = false);
	void _change_type(Object *p_button, int p_index);
	void _change_type_menu(int p_index);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void setup(Variant::Type p_array_type);
	virtual void update_property();
	EditorPropertyArray();
};

class EditorPropertyDictionary : public EditorProperty {
	GDCLASS(EditorPropertyDictionary, EditorProperty)

	PopupMenu *change_type;
	bool updating;

	Ref<EditorPropertyDictionaryObject> object;
	int page_len;
	int page_idx;
	int changing_type_idx;
	Button *edit;
	VBoxContainer *vbox;
	EditorSpinSlider *length;
	EditorSpinSlider *page;
	HBoxContainer *page_hb;

	void _page_changed(double p_page);
	void _edit_pressed();
	void _property_changed(const String &p_prop, Variant p_value, bool changing = false);
	void _change_type(Object *p_button, int p_index);
	void _change_type_menu(int p_index);

	void _add_key_value();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void update_property();
	EditorPropertyDictionary();
};

#endif // EDITOR_PROPERTIES_ARRAY_DICT_H
