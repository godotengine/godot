/*************************************************************************/
/*  editor_properties.h                                                  */
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

#ifndef EDITOR_PROPERTIES_H
#define EDITOR_PROPERTIES_H

#include "editor/create_dialog.h"
#include "editor/editor_inspector.h"
#include "editor/editor_resource_picker.h"
#include "editor/editor_spin_slider.h"
#include "editor/property_selector.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/line_edit.h"

class EditorPropertyNil : public EditorProperty {
	GDCLASS(EditorPropertyNil, EditorProperty);
	LineEdit *text;

public:
	virtual void update_property();
	EditorPropertyNil();
};

class EditorPropertyText : public EditorProperty {
	GDCLASS(EditorPropertyText, EditorProperty);
	LineEdit *text;

	bool updating;
	void _text_changed(const String &p_string);
	void _text_entered(const String &p_string);

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	void set_placeholder(const String &p_string);
	EditorPropertyText();
};

class EditorPropertyMultilineText : public EditorProperty {
	GDCLASS(EditorPropertyMultilineText, EditorProperty);
	TextEdit *text;

	AcceptDialog *big_text_dialog;
	TextEdit *big_text;
	Button *open_big_text;

	void _big_text_changed();
	void _text_changed();
	void _open_big_text();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	EditorPropertyMultilineText();
};

class EditorPropertyTextEnum : public EditorProperty {
	GDCLASS(EditorPropertyTextEnum, EditorProperty);

	HBoxContainer *default_layout;
	HBoxContainer *edit_custom_layout;

	OptionButton *option_button;
	Button *edit_button;

	LineEdit *custom_value_edit;
	Button *accept_button;
	Button *cancel_button;

	Vector<String> options;
	bool loose_mode = false;

	void _emit_changed_value(String p_string);
	void _option_selected(int p_which);

	void _edit_custom_value();
	void _custom_value_submitted(String p_value);
	void _custom_value_accepted();
	void _custom_value_cancelled();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void setup(const Vector<String> &p_options, bool p_loose_mode = false);
	virtual void update_property();
	EditorPropertyTextEnum();
};

class EditorPropertyPath : public EditorProperty {
	GDCLASS(EditorPropertyPath, EditorProperty);
	Vector<String> extensions;
	bool folder;
	bool global;
	bool save_mode;
	EditorFileDialog *dialog;
	LineEdit *path;
	Button *path_edit;

	void _path_selected(const String &p_path);
	void _path_pressed();
	void _path_focus_exited();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void setup(const Vector<String> &p_extensions, bool p_folder, bool p_global);
	void set_save_mode();
	virtual void update_property();
	EditorPropertyPath();
};

class EditorPropertyClassName : public EditorProperty {
	GDCLASS(EditorPropertyClassName, EditorProperty);

private:
	CreateDialog *dialog;
	Button *property;
	String selected_type;
	String base_type;
	void _property_selected();
	void _dialog_created();

protected:
	static void _bind_methods();

public:
	void setup(const String &p_base_type, const String &p_selected_type);
	virtual void update_property();
	EditorPropertyClassName();
};

class EditorPropertyMember : public EditorProperty {
	GDCLASS(EditorPropertyMember, EditorProperty);

public:
	enum Type {
		MEMBER_METHOD_OF_VARIANT_TYPE, ///< a method of a type
		MEMBER_METHOD_OF_BASE_TYPE, ///< a method of a base type
		MEMBER_METHOD_OF_INSTANCE, ///< a method of an instance
		MEMBER_METHOD_OF_SCRIPT, ///< a method of a script & base
		MEMBER_PROPERTY_OF_VARIANT_TYPE, ///< a property of a type
		MEMBER_PROPERTY_OF_BASE_TYPE, ///< a property of a base type
		MEMBER_PROPERTY_OF_INSTANCE, ///< a property of an instance
		MEMBER_PROPERTY_OF_SCRIPT, ///< a property of a script & base

	};

private:
	Type hint;
	PropertySelector *selector;
	Button *property;
	String hint_text;

	void _property_selected(const String &p_selected);
	void _property_select();

protected:
	static void _bind_methods();

public:
	void setup(Type p_hint, const String &p_hint_text);
	virtual void update_property();
	EditorPropertyMember();
};

class EditorPropertyCheck : public EditorProperty {
	GDCLASS(EditorPropertyCheck, EditorProperty);
	CheckBox *checkbox;

	void _checkbox_pressed();

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	EditorPropertyCheck();
};

class EditorPropertyEnum : public EditorProperty {
	GDCLASS(EditorPropertyEnum, EditorProperty);
	OptionButton *options;

	void _option_selected(int p_which);

protected:
	static void _bind_methods();

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property();
	void set_option_button_clip(bool p_enable);
	EditorPropertyEnum();
};

class EditorPropertyFlags : public EditorProperty {
	GDCLASS(EditorPropertyFlags, EditorProperty);
	VBoxContainer *vbox;
	Vector<CheckBox *> flags;
	Vector<int> flag_indices;

	void _flag_toggled();

protected:
	static void _bind_methods();

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property();
	EditorPropertyFlags();
};

class EditorPropertyLayersGrid;

class EditorPropertyLayers : public EditorProperty {
	GDCLASS(EditorPropertyLayers, EditorProperty);

public:
	enum LayerType {
		LAYER_PHYSICS_2D,
		LAYER_RENDER_2D,
		LAYER_PHYSICS_3D,
		LAYER_RENDER_3D,
	};

private:
	EditorPropertyLayersGrid *grid;
	void _grid_changed(uint32_t p_grid);
	LayerType layer_type;
	PopupMenu *layers;
	Button *button;

	void _button_pressed();
	void _menu_pressed(int p_menu);
	void _refresh_names();

protected:
	static void _bind_methods();

public:
	void setup(LayerType p_layer_type);
	virtual void update_property();
	EditorPropertyLayers();
};

class EditorPropertyInteger : public EditorProperty {
	GDCLASS(EditorPropertyInteger, EditorProperty);
	EditorSpinSlider *spin;
	bool setting;
	void _value_changed(int64_t p_val);

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(int64_t p_min, int64_t p_max, int64_t p_step, bool p_allow_greater, bool p_allow_lesser);
	EditorPropertyInteger();
};

class EditorPropertyObjectID : public EditorProperty {
	GDCLASS(EditorPropertyObjectID, EditorProperty);
	Button *edit;
	String base_type;
	void _edit_pressed();

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(const String &p_base_type);
	EditorPropertyObjectID();
};

class EditorPropertyFloat : public EditorProperty {
	GDCLASS(EditorPropertyFloat, EditorProperty);
	EditorSpinSlider *spin;
	bool setting;
	void _value_changed(double p_val);

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, bool p_exp_range, bool p_greater, bool p_lesser);
	EditorPropertyFloat();
};

class EditorPropertyEasing : public EditorProperty {
	GDCLASS(EditorPropertyEasing, EditorProperty);
	Control *easing_draw;
	PopupMenu *preset;
	EditorSpinSlider *spin;
	bool setting;

	bool dragging;
	bool full;
	bool flip;

	enum {
		EASING_ZERO,
		EASING_LINEAR,
		EASING_IN,
		EASING_OUT,
		EASING_IN_OUT,
		EASING_OUT_IN,
		EASING_MAX

	};

	void _drag_easing(const Ref<InputEvent> &p_ev);
	void _draw_easing();
	void _set_preset(int);

	void _setup_spin();
	void _spin_value_changed(double p_value);
	void _spin_focus_exited();

	void _notification(int p_what);

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(bool p_full, bool p_flip);
	EditorPropertyEasing();
};

class EditorPropertyVector2 : public EditorProperty {
	GDCLASS(EditorPropertyVector2, EditorProperty);
	EditorSpinSlider *spin[2];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyVector2();
};

class EditorPropertyRect2 : public EditorProperty {
	GDCLASS(EditorPropertyRect2, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyRect2();
};

class EditorPropertyVector3 : public EditorProperty {
	GDCLASS(EditorPropertyVector3, EditorProperty);
	EditorSpinSlider *spin[3];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyVector3();
};

class EditorPropertyPlane : public EditorProperty {
	GDCLASS(EditorPropertyPlane, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyPlane();
};

class EditorPropertyQuat : public EditorProperty {
	GDCLASS(EditorPropertyQuat, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyQuat();
};

class EditorPropertyAABB : public EditorProperty {
	GDCLASS(EditorPropertyAABB, EditorProperty);
	EditorSpinSlider *spin[6];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyAABB();
};

class EditorPropertyTransform2D : public EditorProperty {
	GDCLASS(EditorPropertyTransform2D, EditorProperty);
	EditorSpinSlider *spin[6];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyTransform2D();
};

class EditorPropertyBasis : public EditorProperty {
	GDCLASS(EditorPropertyBasis, EditorProperty);
	EditorSpinSlider *spin[9];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyBasis();
};

class EditorPropertyTransform : public EditorProperty {
	GDCLASS(EditorPropertyTransform, EditorProperty);
	EditorSpinSlider *spin[12];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider);
	EditorPropertyTransform();
};

class EditorPropertyColor : public EditorProperty {
	GDCLASS(EditorPropertyColor, EditorProperty);
	ColorPickerButton *picker;
	void _color_changed(const Color &p_color);
	void _popup_closed();
	void _picker_created();
	void _picker_opening();

	Color last_color;

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	void setup(bool p_show_alpha);
	EditorPropertyColor();
};

class EditorPropertyNodePath : public EditorProperty {
	GDCLASS(EditorPropertyNodePath, EditorProperty);
	Button *assign;
	Button *clear;
	SceneTreeDialog *scene_tree;
	NodePath base_hint;
	bool use_path_from_scene_root;

	Vector<StringName> valid_types;
	void _node_selected(const NodePath &p_path);
	void _node_assign();
	void _node_clear();

	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	bool is_drop_valid(const Dictionary &p_drag_data) const;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void update_property();
	void setup(const NodePath &p_base_hint, Vector<StringName> p_valid_types, bool p_use_path_from_scene_root = true);
	EditorPropertyNodePath();
};

class EditorPropertyRID : public EditorProperty {
	GDCLASS(EditorPropertyRID, EditorProperty);
	Label *label;

public:
	virtual void update_property();
	EditorPropertyRID();
};

class EditorPropertyResource : public EditorProperty {
	GDCLASS(EditorPropertyResource, EditorProperty);

	EditorResourcePicker *resource_picker = nullptr;
	SceneTreeDialog *scene_tree = nullptr;

	bool use_sub_inspector = false;
	EditorInspector *sub_inspector = nullptr;
	VBoxContainer *sub_inspector_vbox = nullptr;
	bool updating_theme = false;
	bool opened_editor = false;

	void _resource_selected(const RES &p_resource, bool p_edit);
	void _resource_changed(const RES &p_resource);

	void _viewport_selected(const NodePath &p_path);

	void _sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool);
	void _sub_inspector_resource_selected(const RES &p_resource, const String &p_property);
	void _sub_inspector_object_id_selected(int p_id);

	bool _can_use_sub_inspector(const RES &p_resource);
	void _open_editor_pressed();
	void _fold_other_editors(Object *p_self);
	void _update_property_bg();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void update_property();
	void setup(Object *p_object, const String &p_path, const String &p_base_type);

	void collapse_all_folding();
	void expand_all_folding();

	void set_use_sub_inspector(bool p_enable);

	EditorPropertyResource();
};

///////////////////////////////////////////////////
/// \brief The EditorInspectorDefaultPlugin class
///
class EditorInspectorDefaultPlugin : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorDefaultPlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
	virtual bool parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage);
	virtual void parse_end();
};

#endif // EDITOR_PROPERTIES_H
