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
#include "editor/editor_locale_dialog.h"
#include "editor/editor_resource_picker.h"
#include "editor/editor_spin_slider.h"
#include "editor/property_selector.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/line_edit.h"

class EditorPropertyNil : public EditorProperty {
	GDCLASS(EditorPropertyNil, EditorProperty);
	LineEdit *text = nullptr;

public:
	virtual void update_property() override;
	EditorPropertyNil();
};

class EditorPropertyText : public EditorProperty {
	GDCLASS(EditorPropertyText, EditorProperty);
	LineEdit *text;

	bool updating;
	bool string_name;
	void _text_changed(const String &p_string);
	void _text_submitted(const String &p_string);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	void set_string_name(bool p_enabled);
	virtual void update_property() override;
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
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
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
	bool string_name = false;
	bool loose_mode = false;

	void _emit_changed_value(String p_string);
	void _option_selected(int p_which);

	void _edit_custom_value();
	void _custom_value_submitted(String p_value);
	void _custom_value_accepted();
	void _custom_value_cancelled();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();
	void _notification(int p_what);

public:
	void setup(const Vector<String> &p_options, bool p_string_name = false, bool p_loose_mode = false);
	virtual void update_property() override;
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();
	void _notification(int p_what);

public:
	void setup(const Vector<String> &p_extensions, bool p_folder, bool p_global);
	void set_save_mode();
	virtual void update_property() override;
	EditorPropertyPath();
};

class EditorPropertyLocale : public EditorProperty {
	GDCLASS(EditorPropertyLocale, EditorProperty);
	EditorLocaleDialog *dialog;
	LineEdit *locale;
	Button *locale_edit;

	void _locale_selected(const String &p_locale);
	void _locale_pressed();
	void _locale_focus_exited();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void setup(const String &p_hit_string);
	virtual void update_property() override;
	EditorPropertyLocale();
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	void setup(const String &p_base_type, const String &p_selected_type);
	virtual void update_property() override;
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	void setup(Type p_hint, const String &p_hint_text);
	virtual void update_property() override;
	EditorPropertyMember();
};

class EditorPropertyCheck : public EditorProperty {
	GDCLASS(EditorPropertyCheck, EditorProperty);
	CheckBox *checkbox;

	void _checkbox_pressed();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	virtual void update_property() override;
	EditorPropertyCheck();
};

class EditorPropertyEnum : public EditorProperty {
	GDCLASS(EditorPropertyEnum, EditorProperty);
	OptionButton *options;

	void _option_selected(int p_which);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property() override;
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property() override;
	EditorPropertyFlags();
};

class EditorPropertyLayersGrid;

class EditorPropertyLayers : public EditorProperty {
	GDCLASS(EditorPropertyLayers, EditorProperty);

public:
	enum LayerType {
		LAYER_PHYSICS_2D,
		LAYER_RENDER_2D,
		LAYER_NAVIGATION_2D,
		LAYER_PHYSICS_3D,
		LAYER_RENDER_3D,
		LAYER_NAVIGATION_3D,
	};

private:
	EditorPropertyLayersGrid *grid;
	void _grid_changed(uint32_t p_grid);
	LayerType layer_type;
	PopupMenu *layers;
	Button *button;

	void _button_pressed();
	void _menu_pressed(int p_menu);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	void setup(LayerType p_layer_type);
	virtual void update_property() override;
	EditorPropertyLayers();
};

class EditorPropertyInteger : public EditorProperty {
	GDCLASS(EditorPropertyInteger, EditorProperty);
	EditorSpinSlider *spin;
	bool setting;
	void _value_changed(int64_t p_val);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(int64_t p_min, int64_t p_max, int64_t p_step, bool p_allow_greater, bool p_allow_lesser);
	EditorPropertyInteger();
};

class EditorPropertyObjectID : public EditorProperty {
	GDCLASS(EditorPropertyObjectID, EditorProperty);
	Button *edit;
	String base_type;
	void _edit_pressed();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(const String &p_base_type);
	EditorPropertyObjectID();
};

class EditorPropertyFloat : public EditorProperty {
	GDCLASS(EditorPropertyFloat, EditorProperty);
	EditorSpinSlider *spin;
	bool setting = false;
	bool angle_in_radians = false;
	void _value_changed(double p_val);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, bool p_exp_range, bool p_greater, bool p_lesser, const String &p_suffix = String(), bool p_angle_in_radians = false);
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(bool p_full, bool p_flip);
	EditorPropertyEasing();
};

class EditorPropertyVector2 : public EditorProperty {
	GDCLASS(EditorPropertyVector2, EditorProperty);
	EditorSpinSlider *spin[2];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyVector2(bool p_force_wide = false);
};

class EditorPropertyRect2 : public EditorProperty {
	GDCLASS(EditorPropertyRect2, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyRect2(bool p_force_wide = false);
};

class EditorPropertyVector3 : public EditorProperty {
	GDCLASS(EditorPropertyVector3, EditorProperty);
	EditorSpinSlider *spin[3];
	bool setting = false;
	bool angle_in_radians = false;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	virtual void update_using_vector(Vector3 p_vector);
	virtual Vector3 get_vector();
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String(), bool p_angle_in_radians = false);
	EditorPropertyVector3(bool p_force_wide = false);
};

class EditorPropertyVector2i : public EditorProperty {
	GDCLASS(EditorPropertyVector2i, EditorProperty);
	EditorSpinSlider *spin[2];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(int p_min, int p_max, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyVector2i(bool p_force_wide = false);
};

class EditorPropertyRect2i : public EditorProperty {
	GDCLASS(EditorPropertyRect2i, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(int p_min, int p_max, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyRect2i(bool p_force_wide = false);
};

class EditorPropertyVector3i : public EditorProperty {
	GDCLASS(EditorPropertyVector3i, EditorProperty);
	EditorSpinSlider *spin[3];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(int p_min, int p_max, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyVector3i(bool p_force_wide = false);
};

class EditorPropertyPlane : public EditorProperty {
	GDCLASS(EditorPropertyPlane, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyPlane(bool p_force_wide = false);
};

class EditorPropertyQuaternion : public EditorProperty {
	GDCLASS(EditorPropertyQuaternion, EditorProperty);
	EditorSpinSlider *spin[4];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyQuaternion();
};

class EditorPropertyAABB : public EditorProperty {
	GDCLASS(EditorPropertyAABB, EditorProperty);
	EditorSpinSlider *spin[6];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyAABB();
};

class EditorPropertyTransform2D : public EditorProperty {
	GDCLASS(EditorPropertyTransform2D, EditorProperty);
	EditorSpinSlider *spin[6];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyTransform2D(bool p_include_origin = true);
};

class EditorPropertyBasis : public EditorProperty {
	GDCLASS(EditorPropertyBasis, EditorProperty);
	EditorSpinSlider *spin[9];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyBasis();
};

class EditorPropertyTransform3D : public EditorProperty {
	GDCLASS(EditorPropertyTransform3D, EditorProperty);
	EditorSpinSlider *spin[12];
	bool setting;
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	virtual void update_using_transform(Transform3D p_transform);
	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());
	EditorPropertyTransform3D();
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();

public:
	virtual void update_property() override;
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
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const NodePath &p_base_hint, Vector<StringName> p_valid_types, bool p_use_path_from_scene_root = true);
	EditorPropertyNodePath();
};

class EditorPropertyRID : public EditorProperty {
	GDCLASS(EditorPropertyRID, EditorProperty);
	Label *label;

public:
	virtual void update_property() override;
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

	void _sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool p_advance);
	void _sub_inspector_resource_selected(const RES &p_resource, const String &p_property);
	void _sub_inspector_object_id_selected(int p_id);

	void _open_editor_pressed();
	void _fold_other_editors(Object *p_self);
	void _update_property_bg();
	void _update_preferred_shader();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(Object *p_object, const String &p_path, const String &p_base_type);

	void collapse_all_folding() override;
	void expand_all_folding() override;

	void set_use_sub_inspector(bool p_enable);

	EditorPropertyResource();
};

///////////////////////////////////////////////////
/// \brief The EditorInspectorDefaultPlugin class
///
class EditorInspectorDefaultPlugin : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorDefaultPlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide = false) override;

	static EditorProperty *get_editor_for_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide = false);
};

#endif // EDITOR_PROPERTIES_H
