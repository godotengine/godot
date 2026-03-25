/**************************************************************************/
/*  editor_properties.h                                                   */
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

#include "scene/gui/container.h"

class AcceptDialog;
class BoxContainer;
class Button;
class CheckBox;
class ColorPickerButton;
class ConfirmationDialog;
class CreateDialog;
class EditorFileDialog;
class EditorInspector;
class EditorLocaleDialog;
class EditorResourcePicker;
class EditorSpinSlider;
class EditorVariantTypePopupMenu;
class HBoxContainer;
class LineEdit;
class MenuButton;
class OptionButton;
class PopupMenu;
class SceneTreeDialog;
class TextEdit;
class TextureButton;
class VBoxContainer;

struct EditorPropertyRangeHint {
	bool or_greater = true;
	bool or_less = true;
	double min = 0.0;
	double max = 0.0;
	double step = 1.0;
	String suffix;
	bool exp_range = false;
	bool prefer_slider = false;
	bool hide_control = true;
	bool radians_as_degrees = false;
};

class EditorPropertyRevert {
public:
	static Variant get_property_revert_value(Object *p_object, const StringName &p_property, bool *r_is_valid);
	static bool can_property_revert(Object *p_object, const StringName &p_property, const Variant *p_custom_current_value = nullptr);
};

class EditorProperty : public Container {
	GDCLASS(EditorProperty, Container);

	friend class EditorInspector;

	struct ThemeCache {
		Ref<Font> font;

		Ref<StyleBox> background;
		Ref<StyleBox> background_selected;
		Ref<StyleBox> child_background;
		Ref<StyleBox> hover;
		Ref<StyleBox> sub_inspector_background[17];

		Ref<Texture2D> key_icon;
		Ref<Texture2D> key_next_icon;
		Ref<Texture2D> delete_icon;
		Ref<Texture2D> checked_icon;
		Ref<Texture2D> unchecked_icon;
		Ref<Texture2D> revert_icon;
		Ref<Texture2D> pin_icon;
		Ref<Texture2D> copy_icon;
		Ref<Texture2D> copy_node_path_icon;
		Ref<Texture2D> paste_icon;
		Ref<Texture2D> unfavorite_icon;
		Ref<Texture2D> favorite_icon;
		Ref<Texture2D> override_icon;
		Ref<Texture2D> remove_icon;
		Ref<Texture2D> help_icon;

		int font_size = 0;
		int font_offset = 0;
		int horizontal_separation = 0;
		int vertical_separation = 0;
		int padding = 0;
		int inspector_property_height = 0;

		Color property_color;
		Color readonly_property_color;
		Color warning_color;
		Color readonly_warning_color;
		Color property_color_x;
		Color property_color_y;
		Color property_color_z;
		Color property_color_w;
		Color sub_inspector_property_color;
	} theme_cache;

public:
	enum MenuItems {
		MENU_COPY_VALUE,
		MENU_PASTE_VALUE,
		MENU_COPY_PROPERTY_PATH,
		MENU_OVERRIDE_FOR_PROJECT,
		MENU_FAVORITE_PROPERTY,
		MENU_PIN_VALUE,
		MENU_DELETE,
		MENU_REVERT_VALUE,
		MENU_OPEN_DOCUMENTATION,
	};

	enum ColorationMode {
		COLORATION_CONTAINER_RESOURCE,
		COLORATION_RESOURCE,
		COLORATION_EXTERNAL,
	};

	enum InlineControlSide {
		INLINE_CONTROL_LEFT,
		INLINE_CONTROL_RIGHT
	};

private:
	String label;
	int text_size;
	friend class EditorInspector;
	Object *object = nullptr;
	StringName property;
	String property_path;
	String doc_path;
	bool internal = false;
	bool has_doc_tooltip = false;

	int property_usage;

	bool draw_label = true;
	bool draw_background = true;
	bool read_only = false;
	bool checkable = false;
	bool checked = false;
	bool draw_warning = false;
	bool draw_prop_warning = false;
	bool keying = false;
	bool deletable = false;
	bool label_overlayed = false;

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

	bool can_revert = false;
	bool can_pin = false;
	bool pin_hidden = false;
	bool pinned = false;

	bool can_favorite = false;
	bool favorited = false;

	bool use_folding = false;
	bool draw_top_bg = true;

	int sub_inspector_color_level = -1;

	void _update_popup();
	void _focusable_focused(int p_index);
	int _get_v_separation() const { return bottom_editor ? 0 : theme_cache.vertical_separation; }

	bool selectable = true;
	bool selected = false;
	int selected_focusable;
	bool deferred_drag_mode = false;

	float split_ratio;

	Vector<Control *> focusables;
	Control *label_reference = nullptr;
	Control *bottom_editor = nullptr;
	PopupMenu *menu = nullptr;
	HBoxContainer *left_container = nullptr;
	HBoxContainer *right_container = nullptr;

	HashMap<StringName, Variant> cache;

	GDVIRTUAL0(_update_property)
	GDVIRTUAL1(_set_read_only, bool)

	void _update_flags();

protected:
	bool has_borders = false;
	bool can_override = false;

	void _notification(int p_what);
	static void _bind_methods();
	virtual void _set_read_only(bool p_read_only);

	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;
	const Color *_get_property_colors();

	virtual Variant _get_cache_value(const StringName &p_prop, bool &r_valid) const;
	virtual StringName _get_revert_property() const;

	void _update_property_bg();

	void _accessibility_action_menu(const Variant &p_data);
	void _accessibility_action_click(const Variant &p_data);

public:
	void emit_changed(const StringName &p_property, const Variant &p_value, const StringName &p_field = StringName(), bool p_changing = false);

	String get_tooltip_string(const String &p_string) const;

	virtual Size2 get_minimum_size() const override;

	void set_label(const String &p_label);
	String get_label() const;

	void set_read_only(bool p_read_only);
	bool is_read_only() const;

	void set_draw_label(bool p_draw_label);
	bool is_draw_label() const;

	void set_draw_background(bool p_draw_background);
	bool is_draw_background() const;

	Object *get_edited_object();
	StringName get_edited_property() const;
	inline Variant get_edited_property_value() const {
		ERR_FAIL_NULL_V(object, Variant());
		return object->get(property);
	}
	Variant get_edited_property_display_value() const;
	EditorInspector *get_parent_inspector() const;

	void set_doc_path(const String &p_doc_path);
	void set_internal(bool p_internal);

	virtual void update_property();
	void update_editor_property_status();

	virtual bool use_keying_next() const;

	void set_checkable(bool p_checkable);
	bool is_checkable() const;

	void set_checked(bool p_checked);
	bool is_checked() const;

	void set_draw_warning(bool p_draw_warning);
	bool is_draw_warning() const;

	virtual void set_keying(bool p_keying);
	bool is_keying() const;

	virtual bool is_colored(ColorationMode p_mode) { return false; }

	void set_deletable(bool p_enable);
	bool is_deletable() const;
	void add_focusable(Control *p_control);
	void grab_focus(int p_focusable = -1);
	void select(int p_focusable = -1);
	void deselect();
	bool is_selected() const;

	void add_inline_control(Control *p_control, InlineControlSide p_side);
	HBoxContainer *get_inline_container(InlineControlSide p_side);
	void set_label_overlayed(bool p_overlay);

	void set_label_reference(Control *p_control);
	void set_bottom_editor(Control *p_control);

	void set_use_folding(bool p_use_folding);
	bool is_using_folding() const;

	virtual void expand_all_folding();
	virtual void collapse_all_folding();
	virtual void expand_revertable();

	virtual Variant get_drag_data(const Point2 &p_point) override;
	virtual void update_cache();
	virtual bool is_cache_valid() const;

	virtual void set_deferred_drag_mode_enabled(bool p_enabled = true);
	bool is_deferred_drag_mode_enabled() const;

	void set_selectable(bool p_selectable);
	bool is_selectable() const;

	void set_name_split_ratio(float p_ratio);
	float get_name_split_ratio() const;

	void set_favoritable(bool p_favoritable);
	bool is_favoritable() const;

	void set_object_and_property(Object *p_object, const StringName &p_property);
	virtual Control *make_custom_tooltip(const String &p_text) const override;

	void set_draw_top_bg(bool p_draw) { draw_top_bg = p_draw; }

	bool can_revert_to_default() const { return can_revert; }

	void menu_option(int p_option);

	EditorProperty();
};

class EditorPropertyNil : public EditorProperty {
	GDCLASS(EditorPropertyNil, EditorProperty);
	LineEdit *text = nullptr;

public:
	virtual void update_property() override;
	EditorPropertyNil();
};

class EditorPropertyVariant : public EditorProperty {
	GDCLASS(EditorPropertyVariant, EditorProperty);

	HBoxContainer *content = nullptr;
	EditorProperty *sub_property = nullptr;
	Button *edit_button = nullptr;
	EditorVariantTypePopupMenu *change_type = nullptr;

	Variant::Type current_type = Variant::VARIANT_MAX;
	Variant::Type new_type = Variant::VARIANT_MAX;

	void _change_type(int p_to_type);
	void _popup_edit_menu();
	void _object_id_selected(const StringName &p_property, ObjectID p_id);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	EditorPropertyVariant();
};

class EditorPropertyText : public EditorProperty {
	GDCLASS(EditorPropertyText, EditorProperty);
	LineEdit *text = nullptr;

	bool monospaced = false;
	bool updating = false;
	bool string_name = false;
	void _text_changed(const String &p_string);
	void _text_submitted(const String &p_string);
	void _update_theme();

protected:
	void _notification(int p_what);

	virtual void _set_read_only(bool p_read_only) override;

public:
	void set_string_name(bool p_enabled);
	virtual void update_property() override;
	void set_placeholder(const String &p_string);
	void set_secret(bool p_enabled);
	void set_monospaced(bool p_monospaced);
	EditorPropertyText();
};

class EditorPropertyMultilineText : public EditorProperty {
	GDCLASS(EditorPropertyMultilineText, EditorProperty);

	TextEdit *text = nullptr;

	AcceptDialog *big_text_dialog = nullptr;
	TextEdit *big_text = nullptr;
	Button *open_big_text = nullptr;

	bool expression = false;
	bool monospaced = false;
	bool wrap_lines = true;

	void _big_text_changed();
	void _text_changed();
	void _open_big_text();
	void _update_theme();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;

	void set_monospaced(bool p_monospaced);
	bool get_monospaced();

	void set_wrap_lines(bool p_wrap_lines);
	bool get_wrap_lines();

	EditorPropertyMultilineText(bool p_expression = false);
};

class EditorPropertyTextEnum : public EditorProperty {
	GDCLASS(EditorPropertyTextEnum, EditorProperty);

	HBoxContainer *default_layout = nullptr;
	HBoxContainer *edit_custom_layout = nullptr;

	OptionButton *option_button = nullptr;
	Button *edit_button = nullptr;

	LineEdit *custom_value_edit = nullptr;
	Button *accept_button = nullptr;
	Button *cancel_button = nullptr;

	Vector<String> options;
	Vector<String> option_names;
	bool string_name = false;
	bool loose_mode = false;

	void _emit_changed_value(const String &p_string);
	void _option_selected(int p_which);

	void _edit_custom_value();
	void _custom_value_submitted(const String &p_value);
	void _custom_value_accepted();
	void _custom_value_canceled();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	void setup(const Vector<String> &p_options, const Vector<String> &p_option_names = {}, bool p_string_name = false, bool p_loose_mode = false);
	virtual void update_property() override;
	EditorPropertyTextEnum();
};

class EditorPropertyPath : public EditorProperty {
	GDCLASS(EditorPropertyPath, EditorProperty);
	Vector<String> extensions;
	bool folder = false;
	bool global = false;
	bool save_mode = false;
	bool enable_uid = false;
	bool display_uid = false;

	EditorFileDialog *dialog = nullptr;
	LineEdit *path = nullptr;
	Button *toggle_uid = nullptr;
	Button *path_edit = nullptr;

	String _get_path_text(bool p_allow_uid = false);

	void _path_selected(const String &p_path);
	void _path_pressed();
	void _path_focus_exited();
	void _toggle_uid_display();
	void _update_uid_icon();
	void _drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	LineEdit *get_path_edit() const { return path; }

	void setup(const Vector<String> &p_extensions, bool p_folder, bool p_global, bool p_enable_uid);
	void set_save_mode();
	virtual void update_property() override;
	EditorPropertyPath();
};

class EditorPropertyLocale : public EditorProperty {
	GDCLASS(EditorPropertyLocale, EditorProperty);
	EditorLocaleDialog *dialog = nullptr;
	LineEdit *locale = nullptr;
	Button *locale_edit = nullptr;

	void _locale_selected(const String &p_locale);
	void _locale_pressed();
	void _locale_focus_exited();

protected:
	void _notification(int p_what);

public:
	void setup(const String &p_hit_string);
	virtual void update_property() override;
	EditorPropertyLocale();
};

class EditorPropertyClassName : public EditorProperty {
	GDCLASS(EditorPropertyClassName, EditorProperty);

private:
	CreateDialog *dialog = nullptr;
	Button *property = nullptr;
	String selected_type;
	String base_type;
	void _property_selected();
	void _dialog_created();

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	void setup(const String &p_base_type, const String &p_selected_type);
	virtual void update_property() override;
	EditorPropertyClassName();
};

class EditorPropertyCheck : public EditorProperty {
	GDCLASS(EditorPropertyCheck, EditorProperty);
	CheckBox *checkbox = nullptr;

	void _checkbox_pressed();

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	virtual void update_property() override;
	EditorPropertyCheck();
};

class EditorPropertyEnum : public EditorProperty {
	GDCLASS(EditorPropertyEnum, EditorProperty);
	OptionButton *options = nullptr;

	void _option_selected(int p_which);

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property() override;
	void set_option_button_clip(bool p_enable);
	OptionButton *get_option_button(); // Hack to allow setting icons.
	EditorPropertyEnum();
};

class EditorPropertyFlags : public EditorProperty {
	GDCLASS(EditorPropertyFlags, EditorProperty);
	VBoxContainer *vbox = nullptr;
	Vector<CheckBox *> flags;
	Vector<uint32_t> flag_values;

	void _flag_toggled(int p_index);

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property() override;
	EditorPropertyFlags();
};

///////////////////// LAYERS /////////////////////////

class EditorPropertyLayersGrid : public Control {
	GDCLASS(EditorPropertyLayersGrid, Control);

private:
	Vector<Rect2> flag_rects;
	Rect2 expand_rect;
	bool expand_hovered = false;
	bool expanded = false;
	int expansion_rows = 0;
	const uint32_t HOVERED_INDEX_NONE = UINT32_MAX;
	uint32_t hovered_index = HOVERED_INDEX_NONE;
	bool dragging = false;
	bool dragging_value_to_set = false;
	bool read_only = false;
	int renamed_layer_index = -1;
	PopupMenu *layer_rename = nullptr;
	ConfirmationDialog *rename_dialog = nullptr;
	LineEdit *rename_dialog_text = nullptr;

	void _rename_pressed(int p_menu);
	void _rename_operation_confirm();
	void _update_hovered(const Vector2 &p_position);
	void _on_hover_exit();
	void _update_flag(bool p_replace);
	Size2 get_grid_size() const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	uint32_t value = 0;
	int layer_group_size = 0;
	uint32_t layer_count = 0;
	Vector<String> names;
	Vector<String> tooltips;

	void set_read_only(bool p_read_only);
	virtual Size2 get_minimum_size() const override;
	virtual String get_tooltip(const Point2 &p_pos) const override;
	void gui_input(const Ref<InputEvent> &p_ev) override;
	void set_flag(uint32_t p_flag);
	EditorPropertyLayersGrid();
};

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
		LAYER_AVOIDANCE,
	};

private:
	EditorPropertyLayersGrid *grid = nullptr;
	void _grid_changed(uint32_t p_grid);
	String basename;
	LayerType layer_type;
	PopupMenu *layers = nullptr;
	TextureButton *button = nullptr;

	void _button_pressed();
	void _menu_pressed(int p_menu);
	void _refresh_names();

protected:
	void _notification(int p_what);
	virtual void _set_read_only(bool p_read_only) override;

public:
	void setup(LayerType p_layer_type);
	void set_layer_name(int p_index, const String &p_name);
	String get_layer_name(int p_index) const;
	virtual void update_property() override;
	EditorPropertyLayers();
};

class EditorPropertyInteger : public EditorProperty {
	GDCLASS(EditorPropertyInteger, EditorProperty);
	EditorSpinSlider *spin = nullptr;
	void _value_changed(int64_t p_val);

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	virtual void set_deferred_drag_mode_enabled(bool p_enabled = true) override;
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyInteger();
};

class EditorPropertyObjectID : public EditorProperty {
	GDCLASS(EditorPropertyObjectID, EditorProperty);
	Button *edit = nullptr;
	String base_type;

	ObjectID _get_object_id() const;
	void _edit_pressed();

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	virtual void update_property() override;
	void setup(const String &p_base_type);
	EditorPropertyObjectID();
};

class EditorPropertySignal : public EditorProperty {
	GDCLASS(EditorPropertySignal, EditorProperty);
	Button *edit = nullptr;
	String base_type;
	void _edit_pressed();

public:
	virtual void update_property() override;
	EditorPropertySignal();
};

class EditorPropertyCallable : public EditorProperty {
	GDCLASS(EditorPropertyCallable, EditorProperty);
	Button *edit = nullptr;
	String base_type;

public:
	virtual void update_property() override;
	EditorPropertyCallable();
};

class EditorPropertyFloat : public EditorProperty {
	GDCLASS(EditorPropertyFloat, EditorProperty);
	EditorSpinSlider *spin = nullptr;
	bool radians_as_degrees = false;
	void _value_changed(double p_val);

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	virtual void set_deferred_drag_mode_enabled(bool p_enabled = true) override;
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyFloat();
};

class EditorPropertyEasing : public EditorProperty {
	GDCLASS(EditorPropertyEasing, EditorProperty);
	Control *easing_draw = nullptr;
	PopupMenu *preset = nullptr;
	EditorSpinSlider *spin = nullptr;

	bool dragging = false;
	bool full = false;
	bool flip = false;
	bool positive_only = false;

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

public:
	virtual void update_property() override;
	void setup(bool p_positive_only, bool p_flip);
	EditorPropertyEasing();
};

class EditorPropertyRect2 : public EditorProperty {
	GDCLASS(EditorPropertyRect2, EditorProperty);
	EditorSpinSlider *spin[4];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyRect2(bool p_force_wide = false);
};

class EditorPropertyRect2i : public EditorProperty {
	GDCLASS(EditorPropertyRect2i, EditorProperty);
	EditorSpinSlider *spin[4];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyRect2i(bool p_force_wide = false);
};

class EditorPropertyPlane : public EditorProperty {
	GDCLASS(EditorPropertyPlane, EditorProperty);
	EditorSpinSlider *spin[4];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyPlane(bool p_force_wide = false);
};

class EditorPropertyQuaternion : public EditorProperty {
	GDCLASS(EditorPropertyQuaternion, EditorProperty);
	BoxContainer *default_layout = nullptr;
	EditorSpinSlider *spin[4];

	Button *warning = nullptr;
	AcceptDialog *warning_dialog = nullptr;

	Label *euler_label = nullptr;
	VBoxContainer *edit_custom_bc = nullptr;
	EditorSpinSlider *euler[3];
	Button *edit_button = nullptr;

	Vector3 edit_euler;

	void _value_changed(double p_val, const String &p_name);
	void _edit_custom_value();
	void _custom_value_changed(double p_val);
	void _warning_pressed();

	bool is_grabbing_euler();

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint, bool p_hide_editor = false);
	EditorPropertyQuaternion();
};

class EditorPropertyAABB : public EditorProperty {
	GDCLASS(EditorPropertyAABB, EditorProperty);
	EditorSpinSlider *spin[6];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyAABB();
};

class EditorPropertyTransform2D : public EditorProperty {
	GDCLASS(EditorPropertyTransform2D, EditorProperty);
	EditorSpinSlider *spin[6];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyTransform2D(bool p_include_origin = true);
};

class EditorPropertyBasis : public EditorProperty {
	GDCLASS(EditorPropertyBasis, EditorProperty);
	EditorSpinSlider *spin[9];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyBasis();
};

class EditorPropertyTransform3D : public EditorProperty {
	GDCLASS(EditorPropertyTransform3D, EditorProperty);
	EditorSpinSlider *spin[12];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	virtual void update_using_transform(Transform3D p_transform);
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyTransform3D();
};

class EditorPropertyProjection : public EditorProperty {
	GDCLASS(EditorPropertyProjection, EditorProperty);
	EditorSpinSlider *spin[16];
	void _value_changed(double p_val, const String &p_name);

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	virtual void update_using_transform(Projection p_transform);
	void setup(const EditorPropertyRangeHint &p_range_hint);
	EditorPropertyProjection();
};

class EditorPropertyColor : public EditorProperty {
	GDCLASS(EditorPropertyColor, EditorProperty);
	ColorPickerButton *picker = nullptr;
	void _color_changed(const Color &p_color);
	void _picker_created();
	void _popup_opening();
	void _popup_closed();

	Color last_color;
	bool live_changes_enabled = true;
	bool was_checked = false;

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	virtual void update_property() override;
	void setup(bool p_show_alpha);
	void set_live_changes_enabled(bool p_enabled);
	EditorPropertyColor();
};

class EditorPropertyNodePath : public EditorProperty {
	GDCLASS(EditorPropertyNodePath, EditorProperty);

	enum {
		ACTION_CLEAR,
		ACTION_COPY,
		ACTION_EDIT,
		ACTION_SELECT,
	};

	Button *assign = nullptr;
	MenuButton *menu = nullptr;
	LineEdit *edit = nullptr;

	SceneTreeDialog *scene_tree = nullptr;
	bool use_path_from_scene_root = false;
	bool editing_node = false;
	bool dropping = false;

	Vector<StringName> valid_types;
	void _node_selected(const NodePath &p_path, bool p_absolute = true);
	void _node_assign();
	void _assign_draw();
	Node *get_base_node();
	void _update_menu();
	void _menu_option(int p_idx);
	void _accept_text();
	void _text_submitted(const String &p_text);
	const NodePath _get_node_path() const;

	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	bool is_drop_valid(const Dictionary &p_drag_data) const;

	virtual Variant _get_cache_value(const StringName &p_prop, bool &r_valid) const override;

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(const Vector<StringName> &p_valid_types, bool p_use_path_from_scene_root = true, bool p_editing_node = false);
	EditorPropertyNodePath();
};

class EditorPropertyRID : public EditorProperty {
	GDCLASS(EditorPropertyRID, EditorProperty);
	Label *label = nullptr;

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
	bool opened_editor = false;
	bool use_filter = false;

	void _resource_selected(const Ref<Resource> &p_resource, bool p_inspect);
	void _resource_changed(const Ref<Resource> &p_resource);

	Node *_get_base_node();
	void _viewport_selected(const NodePath &p_path);

	void _sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool p_advance);
	void _sub_inspector_resource_selected(const Ref<Resource> &p_resource, const String &p_property);
	void _sub_inspector_object_id_selected(int p_id);

	void _open_editor_pressed();
	void _update_preferred_shader();
	bool _should_stop_editing() const;

protected:
	virtual void _set_read_only(bool p_read_only) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	void setup(Object *p_object, const String &p_path, const String &p_base_type);
	EditorResourcePicker *get_resource_picker() const { return resource_picker; }

	void collapse_all_folding() override;
	void expand_all_folding() override;
	void expand_revertable() override;

	void set_use_sub_inspector(bool p_enable);
	void set_use_filter(bool p_use);
	void fold_resource();

	virtual void set_keying(bool p_keying) override;

	virtual bool is_colored(ColorationMode p_mode) override;

	EditorPropertyResource();
};
