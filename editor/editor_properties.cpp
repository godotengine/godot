/*************************************************************************/
/*  editor_properties.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_properties.h"
#include "editor/editor_resource_preview.h"
#include "editor_node.h"
#include "editor_properties_array_dict.h"
#include "scene/main/viewport.h"

///////////////////// NULL /////////////////////////

void EditorPropertyNil::update_property() {
}

EditorPropertyNil::EditorPropertyNil() {
	Label *label = memnew(Label);
	label->set_text("[null]");
	add_child(label);
}

///////////////////// TEXT /////////////////////////

void EditorPropertyText::_text_entered(const String &p_string) {
	if (updating)
		return;

	if (text->has_focus()) {
		text->release_focus();
		_text_changed(p_string);
	}
}

void EditorPropertyText::_text_changed(const String &p_string) {
	if (updating)
		return;

	emit_changed(get_edited_property(), p_string, "", true);
}

void EditorPropertyText::update_property() {
	String s = get_edited_object()->get(get_edited_property());
	updating = true;
	text->set_text(s);
	text->set_editable(!is_read_only());
	updating = false;
}

void EditorPropertyText::set_placeholder(const String &p_string) {
	text->set_placeholder(p_string);
}

void EditorPropertyText::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_text_changed", "txt"), &EditorPropertyText::_text_changed);
	ClassDB::bind_method(D_METHOD("_text_entered", "txt"), &EditorPropertyText::_text_entered);
}

EditorPropertyText::EditorPropertyText() {
	text = memnew(LineEdit);
	add_child(text);
	add_focusable(text);
	text->connect("text_changed", this, "_text_changed");
	text->connect("text_entered", this, "_text_entered");

	updating = false;
}

///////////////////// MULTILINE TEXT /////////////////////////

void EditorPropertyMultilineText::_big_text_changed() {
	text->set_text(big_text->get_text());
	emit_changed(get_edited_property(), big_text->get_text(), "", true);
}

void EditorPropertyMultilineText::_text_changed() {
	emit_changed(get_edited_property(), text->get_text(), "", true);
}

void EditorPropertyMultilineText::_open_big_text() {

	if (!big_text_dialog) {
		big_text = memnew(TextEdit);
		big_text->connect("text_changed", this, "_big_text_changed");
		big_text->set_wrap_enabled(true);
		big_text_dialog = memnew(AcceptDialog);
		big_text_dialog->add_child(big_text);
		big_text_dialog->set_title("Edit Text:");
		add_child(big_text_dialog);
	}

	big_text->set_text(text->get_text());
	big_text_dialog->popup_centered_ratio();
}

void EditorPropertyMultilineText::update_property() {
	String t = get_edited_object()->get(get_edited_property());
	text->set_text(t);
	if (big_text && big_text->is_visible_in_tree()) {
		big_text->set_text(t);
	}
}

void EditorPropertyMultilineText::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			Ref<Texture> df = get_icon("DistractionFree", "EditorIcons");
			open_big_text->set_icon(df);
			Ref<Font> font = get_font("font", "Label");
			text->set_custom_minimum_size(Vector2(0, font->get_height() * 6));

		} break;
	}
}

void EditorPropertyMultilineText::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_text_changed"), &EditorPropertyMultilineText::_text_changed);
	ClassDB::bind_method(D_METHOD("_big_text_changed"), &EditorPropertyMultilineText::_big_text_changed);
	ClassDB::bind_method(D_METHOD("_open_big_text"), &EditorPropertyMultilineText::_open_big_text);
}

EditorPropertyMultilineText::EditorPropertyMultilineText() {
	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	set_bottom_editor(hb);
	text = memnew(TextEdit);
	text->connect("text_changed", this, "_text_changed");
	text->set_wrap_enabled(true);
	add_focusable(text);
	hb->add_child(text);
	text->set_h_size_flags(SIZE_EXPAND_FILL);
	open_big_text = memnew(ToolButton);
	open_big_text->connect("pressed", this, "_open_big_text");
	hb->add_child(open_big_text);
	big_text_dialog = NULL;
	big_text = NULL;
}

///////////////////// TEXT ENUM /////////////////////////

void EditorPropertyTextEnum::_option_selected(int p_which) {

	emit_changed(get_edited_property(), options->get_item_text(p_which));
}

void EditorPropertyTextEnum::update_property() {

	String which = get_edited_object()->get(get_edited_property());
	for (int i = 0; i < options->get_item_count(); i++) {
		String t = options->get_item_text(i);
		if (t == which) {
			options->select(i);
			return;
		}
	}
}

void EditorPropertyTextEnum::setup(const Vector<String> &p_options) {
	for (int i = 0; i < p_options.size(); i++) {
		options->add_item(p_options[i], i);
	}
}

void EditorPropertyTextEnum::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_option_selected"), &EditorPropertyTextEnum::_option_selected);
}

EditorPropertyTextEnum::EditorPropertyTextEnum() {
	options = memnew(OptionButton);
	options->set_clip_text(true);
	options->set_flat(true);

	add_child(options);
	add_focusable(options);
	options->connect("item_selected", this, "_option_selected");
}
///////////////////// PATH /////////////////////////

void EditorPropertyPath::_path_selected(const String &p_path) {

	emit_changed(get_edited_property(), p_path);
	update_property();
}
void EditorPropertyPath::_path_pressed() {

	if (!dialog) {
		dialog = memnew(EditorFileDialog);
		dialog->connect("file_selected", this, "_path_selected");
		dialog->connect("dir_selected", this, "_path_selected");
		add_child(dialog);
	}

	String full_path = get_edited_object()->get(get_edited_property());

	dialog->clear_filters();

	if (global) {
		dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	} else {
		dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
	}

	if (folder) {
		dialog->set_mode(EditorFileDialog::MODE_OPEN_DIR);
		dialog->set_current_dir(full_path);
	} else {
		dialog->set_mode(save_mode ? EditorFileDialog::MODE_SAVE_FILE : EditorFileDialog::MODE_OPEN_FILE);
		for (int i = 0; i < extensions.size(); i++) {
			String e = extensions[i].strip_edges();
			if (e != String()) {
				dialog->add_filter(extensions[i].strip_edges());
			}
		}
		dialog->set_current_path(full_path);
	}

	dialog->popup_centered_ratio();
}

void EditorPropertyPath::update_property() {

	String full_path = get_edited_object()->get(get_edited_property());
	path->set_text(full_path);
	path->set_tooltip(full_path);
}

void EditorPropertyPath::setup(const Vector<String> &p_extensions, bool p_folder, bool p_global) {

	extensions = p_extensions;
	folder = p_folder;
	global = p_global;
}

void EditorPropertyPath::set_save_mode() {

	save_mode = true;
}

void EditorPropertyPath::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		path_edit->set_icon(get_icon("Folder", "EditorIcons"));
	}
}

void EditorPropertyPath::_path_focus_exited() {

	_path_selected(path->get_text());
}

void EditorPropertyPath::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_path_pressed"), &EditorPropertyPath::_path_pressed);
	ClassDB::bind_method(D_METHOD("_path_selected"), &EditorPropertyPath::_path_selected);
	ClassDB::bind_method(D_METHOD("_path_focus_exited"), &EditorPropertyPath::_path_focus_exited);
}

EditorPropertyPath::EditorPropertyPath() {
	HBoxContainer *path_hb = memnew(HBoxContainer);
	add_child(path_hb);
	path = memnew(LineEdit);
	path_hb->add_child(path);
	path->connect("text_entered", this, "_path_selected");
	path->connect("focus_exited", this, "_path_focus_exited");
	path->set_h_size_flags(SIZE_EXPAND_FILL);

	path_edit = memnew(Button);
	path_edit->set_clip_text(true);
	path_hb->add_child(path_edit);
	add_focusable(path);
	dialog = NULL;
	path_edit->connect("pressed", this, "_path_pressed");
	folder = false;
	global = false;
	save_mode = false;
}

///////////////////// CLASS NAME /////////////////////////

void EditorPropertyClassName::setup(const String &p_base_type, const String &p_selected_type) {

	base_type = p_base_type;
	dialog->set_base_type(base_type);
	selected_type = p_selected_type;
	property->set_text(selected_type);
}

void EditorPropertyClassName::update_property() {

	String s = get_edited_object()->get(get_edited_property());
	property->set_text(s);
	selected_type = s;
}

void EditorPropertyClassName::_property_selected() {
	dialog->popup_create(true);
}

void EditorPropertyClassName::_dialog_created() {
	selected_type = dialog->get_selected_type();
	emit_changed(get_edited_property(), selected_type);
	update_property();
}

void EditorPropertyClassName::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_dialog_created"), &EditorPropertyClassName::_dialog_created);
	ClassDB::bind_method(D_METHOD("_property_selected"), &EditorPropertyClassName::_property_selected);
}

EditorPropertyClassName::EditorPropertyClassName() {
	property = memnew(Button);
	property->set_clip_text(true);
	add_child(property);
	add_focusable(property);
	property->set_text(selected_type);
	property->connect("pressed", this, "_property_selected");
	dialog = memnew(CreateDialog);
	dialog->set_base_type(base_type);
	dialog->connect("create", this, "_dialog_created");
	add_child(dialog);
}

///////////////////// MEMBER /////////////////////////

void EditorPropertyMember::_property_selected(const String &p_selected) {

	emit_changed(get_edited_property(), p_selected);
	update_property();
}

void EditorPropertyMember::_property_select() {

	if (!selector) {
		selector = memnew(PropertySelector);
		selector->connect("selected", this, "_property_selected");
		add_child(selector);
	}

	String current = get_edited_object()->get(get_edited_property());

	if (hint == MEMBER_METHOD_OF_VARIANT_TYPE) {

		Variant::Type type = Variant::NIL;
		for (int i = 0; i < Variant::VARIANT_MAX; i++) {
			if (hint_text == Variant::get_type_name(Variant::Type(i))) {
				type = Variant::Type(i);
			}
		}
		if (type)
			selector->select_method_from_basic_type(type, current);

	} else if (hint == MEMBER_METHOD_OF_BASE_TYPE) {

		selector->select_method_from_base_type(hint_text, current);

	} else if (hint == MEMBER_METHOD_OF_INSTANCE) {

		Object *instance = ObjectDB::get_instance(hint_text.to_int64());
		if (instance)
			selector->select_method_from_instance(instance, current);

	} else if (hint == MEMBER_METHOD_OF_SCRIPT) {

		Object *obj = ObjectDB::get_instance(hint_text.to_int64());
		if (Object::cast_to<Script>(obj)) {
			selector->select_method_from_script(Object::cast_to<Script>(obj), current);
		}

	} else if (hint == MEMBER_PROPERTY_OF_VARIANT_TYPE) {

		Variant::Type type = Variant::NIL;
		String tname = hint_text;
		if (tname.find(".") != -1)
			tname = tname.get_slice(".", 0);
		for (int i = 0; i < Variant::VARIANT_MAX; i++) {
			if (tname == Variant::get_type_name(Variant::Type(i))) {
				type = Variant::Type(Variant::Type(i));
			}
		}

		if (type != Variant::NIL)
			selector->select_property_from_basic_type(type, current);

	} else if (hint == MEMBER_PROPERTY_OF_BASE_TYPE) {

		selector->select_property_from_base_type(hint_text, current);

	} else if (hint == MEMBER_PROPERTY_OF_INSTANCE) {

		Object *instance = ObjectDB::get_instance(hint_text.to_int64());
		if (instance)
			selector->select_property_from_instance(instance, current);

	} else if (hint == MEMBER_PROPERTY_OF_SCRIPT) {

		Object *obj = ObjectDB::get_instance(hint_text.to_int64());
		if (Object::cast_to<Script>(obj)) {
			selector->select_property_from_script(Object::cast_to<Script>(obj), current);
		}
	}
}

void EditorPropertyMember::setup(Type p_hint, const String &p_hint_text) {
	hint = p_hint;
	hint_text = p_hint_text;
}

void EditorPropertyMember::update_property() {

	String full_path = get_edited_object()->get(get_edited_property());
	property->set_text(full_path);
}

void EditorPropertyMember::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_property_selected"), &EditorPropertyMember::_property_selected);
	ClassDB::bind_method(D_METHOD("_property_select"), &EditorPropertyMember::_property_select);
}

EditorPropertyMember::EditorPropertyMember() {
	selector = NULL;
	property = memnew(Button);
	property->set_clip_text(true);
	add_child(property);
	add_focusable(property);
	property->connect("pressed", this, "_property_select");
}

///////////////////// CHECK /////////////////////////
void EditorPropertyCheck::_checkbox_pressed() {

	emit_changed(get_edited_property(), checkbox->is_pressed());
}

void EditorPropertyCheck::update_property() {
	bool c = get_edited_object()->get(get_edited_property());
	checkbox->set_pressed(c);
	checkbox->set_disabled(is_read_only());
}

void EditorPropertyCheck::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_checkbox_pressed"), &EditorPropertyCheck::_checkbox_pressed);
}

EditorPropertyCheck::EditorPropertyCheck() {
	checkbox = memnew(CheckBox);
	checkbox->set_text(TTR("On"));
	add_child(checkbox);
	add_focusable(checkbox);
	checkbox->connect("pressed", this, "_checkbox_pressed");
}

///////////////////// ENUM /////////////////////////

void EditorPropertyEnum::_option_selected(int p_which) {

	int val = options->get_item_metadata(p_which);
	emit_changed(get_edited_property(), val);
}

void EditorPropertyEnum::update_property() {

	int which = get_edited_object()->get(get_edited_property());

	for (int i = 0; i < options->get_item_count(); i++) {
		if (which == (int)options->get_item_metadata(i)) {
			options->select(i);
			return;
		}
	}
}

void EditorPropertyEnum::setup(const Vector<String> &p_options) {

	int current_val = 0;
	for (int i = 0; i < p_options.size(); i++) {
		Vector<String> text_split = p_options[i].split(":");
		if (text_split.size() != 1)
			current_val = text_split[1].to_int();
		options->add_item(text_split[0]);
		options->set_item_metadata(i, current_val);
		current_val += 1;
	}
}

void EditorPropertyEnum::set_option_button_clip(bool p_enable) {
	options->set_clip_text(p_enable);
}

void EditorPropertyEnum::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_option_selected"), &EditorPropertyEnum::_option_selected);
}

EditorPropertyEnum::EditorPropertyEnum() {
	options = memnew(OptionButton);
	options->set_clip_text(true);
	options->set_flat(true);
	add_child(options);
	add_focusable(options);
	options->connect("item_selected", this, "_option_selected");
}

///////////////////// FLAGS /////////////////////////

void EditorPropertyFlags::_flag_toggled() {

	uint32_t value = 0;
	for (int i = 0; i < flags.size(); i++) {
		if (flags[i]->is_pressed()) {
			uint32_t val = 1;
			val <<= flag_indices[i];
			value |= val;
		}
	}

	emit_changed(get_edited_property(), value);
}

void EditorPropertyFlags::update_property() {

	uint32_t value = get_edited_object()->get(get_edited_property());

	for (int i = 0; i < flags.size(); i++) {
		uint32_t val = 1;
		val <<= flag_indices[i];
		if (value & val) {

			flags[i]->set_pressed(true);
		} else {
			flags[i]->set_pressed(false);
		}
	}
}

void EditorPropertyFlags::setup(const Vector<String> &p_options) {
	ERR_FAIL_COND(flags.size());

	bool first = true;
	for (int i = 0; i < p_options.size(); i++) {
		String option = p_options[i].strip_edges();
		if (option != "") {
			CheckBox *cb = memnew(CheckBox);
			cb->set_text(option);
			cb->set_clip_text(true);
			cb->connect("pressed", this, "_flag_toggled");
			add_focusable(cb);
			vbox->add_child(cb);
			flags.push_back(cb);
			flag_indices.push_back(i);
			if (first) {
				set_label_reference(cb);
				first = false;
			}
		}
	}
}

void EditorPropertyFlags::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_flag_toggled"), &EditorPropertyFlags::_flag_toggled);
}

EditorPropertyFlags::EditorPropertyFlags() {

	vbox = memnew(VBoxContainer);
	add_child(vbox);
}

///////////////////// LAYERS /////////////////////////

class EditorPropertyLayersGrid : public Control {
	GDCLASS(EditorPropertyLayersGrid, Control)
public:
	uint32_t value;
	Vector<Rect2> flag_rects;
	Vector<String> names;
	Vector<String> tooltips;

	virtual Size2 get_minimum_size() const {
		Ref<Font> font = get_font("font", "Label");
		return Vector2(0, font->get_height() * 2);
	}

	virtual String get_tooltip(const Point2 &p_pos) const {
		for (int i = 0; i < flag_rects.size(); i++) {
			if (i < tooltips.size() && flag_rects[i].has_point(p_pos)) {
				return tooltips[i];
			}
		}
		return String();
	}
	void _gui_input(const Ref<InputEvent> &p_ev) {
		Ref<InputEventMouseButton> mb = p_ev;
		if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT && mb->is_pressed()) {
			for (int i = 0; i < flag_rects.size(); i++) {
				if (flag_rects[i].has_point(mb->get_position())) {
					//toggle
					if (value & (1 << i)) {
						value &= ~(1 << i);
					} else {
						value |= (1 << i);
					}
					emit_signal("flag_changed", value);
					update();
				}
			}
		}
	}

	void _notification(int p_what) {
		if (p_what == NOTIFICATION_DRAW) {

			Rect2 rect;
			rect.size = get_size();
			flag_rects.clear();

			int bsize = (rect.size.height * 80 / 100) / 2;

			int h = bsize * 2 + 1;
			int vofs = (rect.size.height - h) / 2;

			Color color = get_color("highlight_color", "Editor");
			for (int i = 0; i < 2; i++) {

				Point2 ofs(4, vofs);
				if (i == 1)
					ofs.y += bsize + 1;

				ofs += rect.position;
				for (int j = 0; j < 10; j++) {

					Point2 o = ofs + Point2(j * (bsize + 1), 0);
					if (j >= 5)
						o.x += 1;

					uint32_t idx = i * 10 + j;
					bool on = value & (1 << idx);
					Rect2 rect = Rect2(o, Size2(bsize, bsize));
					color.a = on ? 0.6 : 0.2;
					draw_rect(rect, color);
					flag_rects.push_back(rect);
				}
			}
		}
	}

	void set_flag(uint32_t p_flag) {
		value = p_flag;
		update();
	}

	static void _bind_methods() {

		ClassDB::bind_method(D_METHOD("_gui_input"), &EditorPropertyLayersGrid::_gui_input);
		ADD_SIGNAL(MethodInfo("flag_changed", PropertyInfo(Variant::INT, "flag")));
	}

	EditorPropertyLayersGrid() {
		value = 0;
	}
};
void EditorPropertyLayers::_grid_changed(uint32_t p_grid) {

	emit_changed(get_edited_property(), p_grid);
}

void EditorPropertyLayers::update_property() {

	uint32_t value = get_edited_object()->get(get_edited_property());

	grid->set_flag(value);
}

void EditorPropertyLayers::setup(LayerType p_layer_type) {

	String basename;
	switch (p_layer_type) {
		case LAYER_RENDER_2D:
			basename = "layer_names/2d_render";
			break;
		case LAYER_PHYSICS_2D:
			basename = "layer_names/2d_physics";
			break;
		case LAYER_RENDER_3D:
			basename = "layer_names/3d_render";
			break;
		case LAYER_PHYSICS_3D:
			basename = "layer_names/3d_physics";
			break;
	}

	Vector<String> names;
	Vector<String> tooltips;
	for (int i = 0; i < 20; i++) {
		String name;

		if (ProjectSettings::get_singleton()->has_setting(basename + "/layer_" + itos(i + 1))) {
			name = ProjectSettings::get_singleton()->get(basename + "/layer_" + itos(i + 1));
		}

		if (name == "") {
			name = TTR("Layer") + " " + itos(i + 1);
		}

		names.push_back(name);
		tooltips.push_back(name + "\n" + vformat(TTR("Bit %d, value %d"), i, 1 << i));
	}

	grid->names = names;
	grid->tooltips = tooltips;
}

void EditorPropertyLayers::_button_pressed() {

	layers->clear();
	for (int i = 0; i < 20; i++) {
		if (i == 5 || i == 10 || i == 15) {
			layers->add_separator();
		}
		layers->add_check_item(grid->names[i], i);
		int idx = layers->get_item_index(i);
		layers->set_item_checked(idx, grid->value & (1 << i));
	}

	Rect2 gp = button->get_global_rect();
	layers->set_as_minsize();
	Vector2 popup_pos = gp.position - Vector2(layers->get_combined_minimum_size().x, 0);
	layers->set_global_position(popup_pos);
	layers->popup();
}

void EditorPropertyLayers::_menu_pressed(int p_menu) {
	if (grid->value & (1 << p_menu)) {
		grid->value &= ~(1 << p_menu);
	} else {
		grid->value |= (1 << p_menu);
	}
	grid->update();
	layers->set_item_checked(layers->get_item_index(p_menu), grid->value & (1 << p_menu));
	_grid_changed(grid->value);
}

void EditorPropertyLayers::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_grid_changed"), &EditorPropertyLayers::_grid_changed);
	ClassDB::bind_method(D_METHOD("_button_pressed"), &EditorPropertyLayers::_button_pressed);
	ClassDB::bind_method(D_METHOD("_menu_pressed"), &EditorPropertyLayers::_menu_pressed);
}

EditorPropertyLayers::EditorPropertyLayers() {

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	grid = memnew(EditorPropertyLayersGrid);
	grid->connect("flag_changed", this, "_grid_changed");
	grid->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(grid);
	button = memnew(Button);
	button->set_text("..");
	button->connect("pressed", this, "_button_pressed");
	hb->add_child(button);
	set_bottom_editor(hb);
	layers = memnew(PopupMenu);
	add_child(layers);
	layers->set_hide_on_checkable_item_selection(false);
	layers->connect("id_pressed", this, "_menu_pressed");
}

///////////////////// INT /////////////////////////

void EditorPropertyInteger::_value_changed(double val) {
	if (setting)
		return;
	emit_changed(get_edited_property(), int(val));
}

void EditorPropertyInteger::update_property() {
	int val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin->set_value(val);
	setting = false;
}

void EditorPropertyInteger::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyInteger::_value_changed);
}

void EditorPropertyInteger::setup(int p_min, int p_max, int p_step, bool p_allow_greater, bool p_allow_lesser) {
	spin->set_min(p_min);
	spin->set_max(p_max);
	spin->set_step(p_step);
	spin->set_allow_greater(p_allow_greater);
	spin->set_allow_lesser(p_allow_lesser);
}

EditorPropertyInteger::EditorPropertyInteger() {
	spin = memnew(EditorSpinSlider);
	spin->set_flat(true);
	add_child(spin);
	add_focusable(spin);
	spin->connect("value_changed", this, "_value_changed");
	setting = false;
}

///////////////////// OBJECT ID /////////////////////////

void EditorPropertyObjectID::_edit_pressed() {

	emit_signal("object_id_selected", get_edited_property(), get_edited_object()->get(get_edited_property()));
}

void EditorPropertyObjectID::update_property() {
	String type = base_type;
	if (type == "")
		type = "Object";

	ObjectID id = get_edited_object()->get(get_edited_property());
	if (id != 0) {
		edit->set_text(type + " ID: " + itos(id));
		edit->set_disabled(false);
		edit->set_icon(EditorNode::get_singleton()->get_class_icon(type));
	} else {
		edit->set_text(TTR("[Empty]"));
		edit->set_disabled(true);
		edit->set_icon(Ref<Texture>());
	}
}

void EditorPropertyObjectID::setup(const String &p_base_type) {
	base_type = p_base_type;
}

void EditorPropertyObjectID::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_edit_pressed"), &EditorPropertyObjectID::_edit_pressed);
}

EditorPropertyObjectID::EditorPropertyObjectID() {
	edit = memnew(Button);
	add_child(edit);
	add_focusable(edit);
	edit->connect("pressed", this, "_edit_pressed");
}

///////////////////// FLOAT /////////////////////////

void EditorPropertyFloat::_value_changed(double val) {
	if (setting)
		return;

	emit_changed(get_edited_property(), val);
}

void EditorPropertyFloat::update_property() {
	double val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin->set_value(val);
	setting = false;
}

void EditorPropertyFloat::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyFloat::_value_changed);
}

void EditorPropertyFloat::setup(double p_min, double p_max, double p_step, bool p_no_slider, bool p_exp_range, bool p_greater, bool p_lesser) {

	spin->set_min(p_min);
	spin->set_max(p_max);
	spin->set_step(p_step);
	spin->set_hide_slider(p_no_slider);
	spin->set_exp_ratio(p_exp_range);
	spin->set_allow_greater(p_greater);
	spin->set_allow_lesser(p_lesser);
}

EditorPropertyFloat::EditorPropertyFloat() {
	spin = memnew(EditorSpinSlider);
	spin->set_flat(true);
	add_child(spin);
	add_focusable(spin);
	spin->connect("value_changed", this, "_value_changed");
	setting = false;
}

///////////////////// EASING /////////////////////////

void EditorPropertyEasing::_drag_easing(const Ref<InputEvent> &p_ev) {

	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) {
		preset->set_global_position(easing_draw->get_global_transform().xform(mb->get_position()));
		preset->popup();
	}

	Ref<InputEventMouseMotion> mm = p_ev;

	if (mm.is_valid() && mm->get_button_mask() & BUTTON_MASK_LEFT) {

		float rel = mm->get_relative().x;
		if (rel == 0)
			return;

		if (flip)
			rel = -rel;

		float val = get_edited_object()->get(get_edited_property());
		if (val == 0)
			return;
		bool sg = val < 0;
		val = Math::absf(val);

		val = Math::log(val) / Math::log((float)2.0);
		//logspace
		val += rel * 0.05;

		val = Math::pow(2.0f, val);
		if (sg)
			val = -val;

		emit_changed(get_edited_property(), val);
		easing_draw->update();
	}
}

void EditorPropertyEasing::_draw_easing() {

	RID ci = easing_draw->get_canvas_item();

	Size2 s = easing_draw->get_size();
	Rect2 r(Point2(), s);
	r = r.grow(3);
	//get_stylebox("normal", "LineEdit")->draw(ci, r);

	int points = 48;

	float prev = 1.0;
	float exp = get_edited_object()->get(get_edited_property());

	Ref<Font> f = get_font("font", "Label");
	Color color = get_color("font_color", "Label");

	Vector<Point2> lines;
	for (int i = 1; i <= points; i++) {

		float ifl = i / float(points);
		float iflp = (i - 1) / float(points);

		float h = 1.0 - Math::ease(ifl, exp);

		if (flip) {
			ifl = 1.0 - ifl;
			iflp = 1.0 - iflp;
		}

		lines.push_back(Point2(ifl * s.width, h * s.height));
		lines.push_back(Point2(iflp * s.width, prev * s.height));
		prev = h;
	}

	easing_draw->draw_multiline(lines, color, 1.0, true);
	f->draw(ci, Point2(10, 10 + f->get_ascent()), String::num(exp, 2), color);
}

void EditorPropertyEasing::update_property() {
	easing_draw->update();
}

void EditorPropertyEasing::_set_preset(int p_preset) {
	static const float preset_value[EASING_MAX] = { 0.0, 1.0, 2.0, 0.5, -2.0, -0.5 };

	emit_changed(get_edited_property(), preset_value[p_preset]);
	easing_draw->update();
}

void EditorPropertyEasing::setup(bool p_full, bool p_flip) {

	flip = p_flip;
	full = p_full;
}

void EditorPropertyEasing::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			preset->clear();
			preset->add_icon_item(get_icon("CurveConstant", "EditorIcons"), "Zero", EASING_ZERO);
			preset->add_icon_item(get_icon("CurveLinear", "EditorIcons"), "Linear", EASING_LINEAR);
			preset->add_icon_item(get_icon("CurveIn", "EditorIcons"), "In", EASING_IN);
			preset->add_icon_item(get_icon("CurveOut", "EditorIcons"), "Out", EASING_OUT);
			if (full) {
				preset->add_icon_item(get_icon("CurveInOut", "EditorIcons"), "In-Out", EASING_IN_OUT);
				preset->add_icon_item(get_icon("CurveOutIn", "EditorIcons"), "Out-In", EASING_OUT_IN);
			}
			easing_draw->set_custom_minimum_size(Size2(0, get_font("font", "Label")->get_height() * 2));
		} break;
		case NOTIFICATION_RESIZED: {

		} break;
	}
}

void EditorPropertyEasing::_bind_methods() {

	ClassDB::bind_method("_draw_easing", &EditorPropertyEasing::_draw_easing);
	ClassDB::bind_method("_drag_easing", &EditorPropertyEasing::_drag_easing);
	ClassDB::bind_method("_set_preset", &EditorPropertyEasing::_set_preset);
}

EditorPropertyEasing::EditorPropertyEasing() {

	easing_draw = memnew(Control);
	easing_draw->connect("draw", this, "_draw_easing");
	easing_draw->connect("gui_input", this, "_drag_easing");
	easing_draw->set_default_cursor_shape(Control::CURSOR_MOVE);
	add_child(easing_draw);

	preset = memnew(PopupMenu);
	add_child(preset);
	preset->connect("id_pressed", this, "_set_preset");

	flip = false;
	full = false;
}

///////////////////// VECTOR2 /////////////////////////

void EditorPropertyVector2::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Vector2 v2;
	v2.x = spin[0]->get_value();
	v2.y = spin[1]->get_value();
	emit_changed(get_edited_property(), v2, p_name);
}

void EditorPropertyVector2::update_property() {
	Vector2 val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.x);
	spin[1]->set_value(val.y);
	setting = false;
}

void EditorPropertyVector2::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 2; i++) {

			Color c = base;
			c.set_hsv(float(i) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}

void EditorPropertyVector2::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyVector2::_value_changed);
}

void EditorPropertyVector2::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 2; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyVector2::EditorPropertyVector2() {
	bool horizontal = EDITOR_GET("interface/inspector/horizontal_vector2_editing");

	BoxContainer *bc;

	if (horizontal) {
		bc = memnew(HBoxContainer);
		add_child(bc);
		set_bottom_editor(bc);
	} else {
		bc = memnew(VBoxContainer);
		add_child(bc);
	}

	static const char *desc[2] = { "x", "y" };
	for (int i = 0; i < 2; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_flat(true);
		spin[i]->set_label(desc[i]);
		bc->add_child(spin[i]);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
		if (horizontal) {
			spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		}
	}

	if (!horizontal) {
		set_label_reference(spin[0]); //show text and buttons around this
	}
	setting = false;
}

///////////////////// RECT2 /////////////////////////

void EditorPropertyRect2::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Rect2 r2;
	r2.position.x = spin[0]->get_value();
	r2.position.y = spin[1]->get_value();
	r2.size.x = spin[2]->get_value();
	r2.size.y = spin[3]->get_value();
	emit_changed(get_edited_property(), r2, p_name);
}

void EditorPropertyRect2::update_property() {
	Rect2 val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.position.x);
	spin[1]->set_value(val.position.y);
	spin[2]->set_value(val.size.x);
	spin[3]->set_value(val.size.y);
	setting = false;
}
void EditorPropertyRect2::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 4; i++) {

			Color c = base;
			c.set_hsv(float(i % 2) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyRect2::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyRect2::_value_changed);
}

void EditorPropertyRect2::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 4; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyRect2::EditorPropertyRect2() {

	bool horizontal = EDITOR_GET("interface/inspector/horizontal_vector_types_editing");

	BoxContainer *bc;

	if (horizontal) {
		bc = memnew(HBoxContainer);
		add_child(bc);
		set_bottom_editor(bc);
	} else {
		bc = memnew(VBoxContainer);
		add_child(bc);
	}

	static const char *desc[4] = { "x", "y", "w", "h" };
	for (int i = 0; i < 4; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_label(desc[i]);
		spin[i]->set_flat(true);
		bc->add_child(spin[i]);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
		if (horizontal) {
			spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		}
	}

	if (!horizontal) {
		set_label_reference(spin[0]); //show text and buttons around this
	}
	setting = false;
}

///////////////////// VECTOR3 /////////////////////////

void EditorPropertyVector3::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Vector3 v3;
	v3.x = spin[0]->get_value();
	v3.y = spin[1]->get_value();
	v3.z = spin[2]->get_value();
	emit_changed(get_edited_property(), v3, p_name);
}

void EditorPropertyVector3::update_property() {
	Vector3 val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.x);
	spin[1]->set_value(val.y);
	spin[2]->set_value(val.z);
	setting = false;
}
void EditorPropertyVector3::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 3; i++) {

			Color c = base;
			c.set_hsv(float(i) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyVector3::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyVector3::_value_changed);
}

void EditorPropertyVector3::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 3; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyVector3::EditorPropertyVector3() {
	bool horizontal = EDITOR_GET("interface/inspector/horizontal_vector_types_editing");

	BoxContainer *bc;

	if (horizontal) {
		bc = memnew(HBoxContainer);
		add_child(bc);
		set_bottom_editor(bc);
	} else {
		bc = memnew(VBoxContainer);
		add_child(bc);
	}

	static const char *desc[3] = { "x", "y", "z" };
	for (int i = 0; i < 3; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_flat(true);
		spin[i]->set_label(desc[i]);
		bc->add_child(spin[i]);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
		if (horizontal) {
			spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		}
	}

	if (!horizontal) {
		set_label_reference(spin[0]); //show text and buttons around this
	}
	setting = false;
}
///////////////////// PLANE /////////////////////////

void EditorPropertyPlane::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Plane p;
	p.normal.x = spin[0]->get_value();
	p.normal.y = spin[1]->get_value();
	p.normal.z = spin[2]->get_value();
	p.d = spin[3]->get_value();
	emit_changed(get_edited_property(), p, p_name);
}

void EditorPropertyPlane::update_property() {
	Plane val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.normal.x);
	spin[1]->set_value(val.normal.y);
	spin[2]->set_value(val.normal.z);
	spin[3]->set_value(val.d);
	setting = false;
}
void EditorPropertyPlane::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 3; i++) {

			Color c = base;
			c.set_hsv(float(i) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyPlane::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyPlane::_value_changed);
}

void EditorPropertyPlane::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 4; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyPlane::EditorPropertyPlane() {

	bool horizontal = EDITOR_GET("interface/inspector/horizontal_vector_types_editing");

	BoxContainer *bc;

	if (horizontal) {
		bc = memnew(HBoxContainer);
		add_child(bc);
		set_bottom_editor(bc);
	} else {
		bc = memnew(VBoxContainer);
		add_child(bc);
	}

	static const char *desc[4] = { "x", "y", "z", "d" };
	for (int i = 0; i < 4; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_flat(true);
		spin[i]->set_label(desc[i]);
		bc->add_child(spin[i]);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
		if (horizontal) {
			spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		}
	}

	if (!horizontal) {
		set_label_reference(spin[0]); //show text and buttons around this
	}
	setting = false;
}

///////////////////// QUAT /////////////////////////

void EditorPropertyQuat::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Quat p;
	p.x = spin[0]->get_value();
	p.y = spin[1]->get_value();
	p.z = spin[2]->get_value();
	p.w = spin[3]->get_value();
	emit_changed(get_edited_property(), p, p_name);
}

void EditorPropertyQuat::update_property() {
	Quat val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.x);
	spin[1]->set_value(val.y);
	spin[2]->set_value(val.z);
	spin[3]->set_value(val.w);
	setting = false;
}
void EditorPropertyQuat::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 3; i++) {

			Color c = base;
			c.set_hsv(float(i) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyQuat::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyQuat::_value_changed);
}

void EditorPropertyQuat::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 4; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyQuat::EditorPropertyQuat() {
	bool horizontal = EDITOR_GET("interface/inspector/horizontal_vector_types_editing");

	BoxContainer *bc;

	if (horizontal) {
		bc = memnew(HBoxContainer);
		add_child(bc);
		set_bottom_editor(bc);
	} else {
		bc = memnew(VBoxContainer);
		add_child(bc);
	}

	static const char *desc[4] = { "x", "y", "z", "w" };
	for (int i = 0; i < 4; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_flat(true);
		spin[i]->set_label(desc[i]);
		bc->add_child(spin[i]);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
		if (horizontal) {
			spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		}
	}

	if (!horizontal) {
		set_label_reference(spin[0]); //show text and buttons around this
	}
	setting = false;
}

///////////////////// AABB /////////////////////////

void EditorPropertyAABB::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	AABB p;
	p.position.x = spin[0]->get_value();
	p.position.y = spin[1]->get_value();
	p.position.z = spin[2]->get_value();
	p.size.x = spin[3]->get_value();
	p.size.y = spin[4]->get_value();
	p.size.z = spin[5]->get_value();

	emit_changed(get_edited_property(), p, p_name);
}

void EditorPropertyAABB::update_property() {
	AABB val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.position.x);
	spin[1]->set_value(val.position.y);
	spin[2]->set_value(val.position.z);
	spin[3]->set_value(val.size.x);
	spin[4]->set_value(val.size.y);
	spin[5]->set_value(val.size.z);

	setting = false;
}
void EditorPropertyAABB::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 6; i++) {

			Color c = base;
			c.set_hsv(float(i % 3) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyAABB::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyAABB::_value_changed);
}

void EditorPropertyAABB::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 6; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyAABB::EditorPropertyAABB() {
	GridContainer *g = memnew(GridContainer);
	g->set_columns(3);
	add_child(g);

	static const char *desc[6] = { "x", "y", "z", "w", "h", "d" };
	for (int i = 0; i < 6; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_label(desc[i]);
		spin[i]->set_flat(true);

		g->add_child(spin[i]);
		spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
	}
	set_bottom_editor(g);
	setting = false;
}

///////////////////// TRANSFORM2D /////////////////////////

void EditorPropertyTransform2D::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Transform2D p;
	p[0][0] = spin[0]->get_value();
	p[0][1] = spin[1]->get_value();
	p[1][0] = spin[2]->get_value();
	p[1][1] = spin[3]->get_value();
	p[2][0] = spin[4]->get_value();
	p[2][1] = spin[5]->get_value();

	emit_changed(get_edited_property(), p, p_name);
}

void EditorPropertyTransform2D::update_property() {
	Transform2D val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val[0][0]);
	spin[1]->set_value(val[0][1]);
	spin[2]->set_value(val[1][0]);
	spin[3]->set_value(val[1][1]);
	spin[4]->set_value(val[2][0]);
	spin[5]->set_value(val[2][1]);

	setting = false;
}
void EditorPropertyTransform2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 6; i++) {

			Color c = base;
			c.set_hsv(float(i % 2) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyTransform2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyTransform2D::_value_changed);
}

void EditorPropertyTransform2D::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 6; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyTransform2D::EditorPropertyTransform2D() {
	GridContainer *g = memnew(GridContainer);
	g->set_columns(2);
	add_child(g);

	static const char *desc[6] = { "x", "y", "x", "y", "x", "y" };
	for (int i = 0; i < 6; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_label(desc[i]);
		spin[i]->set_flat(true);
		g->add_child(spin[i]);
		spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
	}
	set_bottom_editor(g);
	setting = false;
}

///////////////////// BASIS /////////////////////////

void EditorPropertyBasis::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Basis p;
	p[0][0] = spin[0]->get_value();
	p[1][0] = spin[1]->get_value();
	p[2][0] = spin[2]->get_value();
	p[0][1] = spin[3]->get_value();
	p[1][1] = spin[4]->get_value();
	p[2][1] = spin[5]->get_value();
	p[0][2] = spin[6]->get_value();
	p[1][2] = spin[7]->get_value();
	p[2][2] = spin[8]->get_value();

	emit_changed(get_edited_property(), p, p_name);
}

void EditorPropertyBasis::update_property() {
	Basis val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val[0][0]);
	spin[1]->set_value(val[1][0]);
	spin[2]->set_value(val[2][0]);
	spin[3]->set_value(val[0][1]);
	spin[4]->set_value(val[1][1]);
	spin[5]->set_value(val[2][1]);
	spin[6]->set_value(val[0][2]);
	spin[7]->set_value(val[1][2]);
	spin[8]->set_value(val[2][2]);

	setting = false;
}
void EditorPropertyBasis::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 9; i++) {

			Color c = base;
			c.set_hsv(float(i % 3) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyBasis::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyBasis::_value_changed);
}

void EditorPropertyBasis::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 9; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyBasis::EditorPropertyBasis() {
	GridContainer *g = memnew(GridContainer);
	g->set_columns(3);
	add_child(g);

	static const char *desc[9] = { "x", "y", "z", "x", "y", "z", "x", "y", "z" };
	for (int i = 0; i < 9; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_label(desc[i]);
		spin[i]->set_flat(true);
		g->add_child(spin[i]);
		spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
	}
	set_bottom_editor(g);
	setting = false;
}

///////////////////// TRANSFORM /////////////////////////

void EditorPropertyTransform::_value_changed(double val, const String &p_name) {
	if (setting)
		return;

	Transform p;
	p.basis[0][0] = spin[0]->get_value();
	p.basis[1][0] = spin[1]->get_value();
	p.basis[2][0] = spin[2]->get_value();
	p.basis[0][1] = spin[3]->get_value();
	p.basis[1][1] = spin[4]->get_value();
	p.basis[2][1] = spin[5]->get_value();
	p.basis[0][2] = spin[6]->get_value();
	p.basis[1][2] = spin[7]->get_value();
	p.basis[2][2] = spin[8]->get_value();
	p.origin[0] = spin[9]->get_value();
	p.origin[1] = spin[10]->get_value();
	p.origin[2] = spin[11]->get_value();

	emit_changed(get_edited_property(), p, p_name);
}

void EditorPropertyTransform::update_property() {
	Transform val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin[0]->set_value(val.basis[0][0]);
	spin[1]->set_value(val.basis[1][0]);
	spin[2]->set_value(val.basis[2][0]);
	spin[3]->set_value(val.basis[0][1]);
	spin[4]->set_value(val.basis[1][1]);
	spin[5]->set_value(val.basis[2][1]);
	spin[6]->set_value(val.basis[0][2]);
	spin[7]->set_value(val.basis[1][2]);
	spin[8]->set_value(val.basis[2][2]);
	spin[9]->set_value(val.origin[0]);
	spin[10]->set_value(val.origin[1]);
	spin[11]->set_value(val.origin[2]);

	setting = false;
}
void EditorPropertyTransform::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Color base = get_color("accent_color", "Editor");
		for (int i = 0; i < 12; i++) {

			Color c = base;
			c.set_hsv(float(i % 3) / 3.0 + 0.05, c.get_s() * 0.75, c.get_v());
			spin[i]->set_custom_label_color(true, c);
		}
	}
}
void EditorPropertyTransform::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_value_changed"), &EditorPropertyTransform::_value_changed);
}

void EditorPropertyTransform::setup(double p_min, double p_max, double p_step, bool p_no_slider) {
	for (int i = 0; i < 12; i++) {
		spin[i]->set_min(p_min);
		spin[i]->set_max(p_max);
		spin[i]->set_step(p_step);
		spin[i]->set_hide_slider(p_no_slider);
		spin[i]->set_allow_greater(true);
		spin[i]->set_allow_lesser(true);
	}
}

EditorPropertyTransform::EditorPropertyTransform() {
	GridContainer *g = memnew(GridContainer);
	g->set_columns(3);
	add_child(g);

	static const char *desc[12] = { "x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z" };
	for (int i = 0; i < 12; i++) {
		spin[i] = memnew(EditorSpinSlider);
		spin[i]->set_label(desc[i]);
		spin[i]->set_flat(true);
		g->add_child(spin[i]);
		spin[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		add_focusable(spin[i]);
		spin[i]->connect("value_changed", this, "_value_changed", varray(desc[i]));
	}
	set_bottom_editor(g);
	setting = false;
}

////////////// COLOR PICKER //////////////////////

void EditorPropertyColor::_color_changed(const Color &p_color) {

	emit_changed(get_edited_property(), p_color, "", true);
}

void EditorPropertyColor::_popup_closed() {

	emit_changed(get_edited_property(), picker->get_pick_color(), "", false);
}

void EditorPropertyColor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_color_changed"), &EditorPropertyColor::_color_changed);
	ClassDB::bind_method(D_METHOD("_popup_closed"), &EditorPropertyColor::_popup_closed);
}

void EditorPropertyColor::update_property() {

	picker->set_pick_color(get_edited_object()->get(get_edited_property()));
}

void EditorPropertyColor::setup(bool p_show_alpha) {
	picker->set_edit_alpha(p_show_alpha);
}

EditorPropertyColor::EditorPropertyColor() {

	picker = memnew(ColorPickerButton);
	add_child(picker);
	picker->set_flat(true);
	picker->connect("color_changed", this, "_color_changed");
	picker->connect("popup_closed", this, "_popup_closed");
}

////////////// NODE PATH //////////////////////

void EditorPropertyNodePath::_node_selected(const NodePath &p_path) {

	NodePath path = p_path;
	Node *base_node = NULL;

	if (!use_path_from_scene_root) {
		base_node = Object::cast_to<Node>(get_edited_object());

		if (!base_node) {
			//try a base node within history
			if (EditorNode::get_singleton()->get_editor_history()->get_path_size() > 0) {
				Object *base = ObjectDB::get_instance(EditorNode::get_singleton()->get_editor_history()->get_path_object(0));
				if (base) {
					base_node = Object::cast_to<Node>(base);
				}
			}
		}
	}

	if (!base_node && get_edited_object()->has_method("get_root_path")) {
		base_node = get_edited_object()->call("get_root_path");
	}

	if (!base_node && Object::cast_to<Reference>(get_edited_object())) {
		Node *to_node = get_node(p_path);
		ERR_FAIL_COND(!to_node);
		path = get_tree()->get_edited_scene_root()->get_path_to(to_node);
	}

	if (base_node) { // for AnimationTrackKeyEdit
		path = base_node->get_path().rel_path_to(p_path);
	}
	emit_changed(get_edited_property(), path);
	update_property();
}

void EditorPropertyNodePath::_node_assign() {
	if (!scene_tree) {
		scene_tree = memnew(SceneTreeDialog);
		scene_tree->get_scene_tree()->set_show_enabled_subscene(true);
		scene_tree->get_scene_tree()->set_valid_types(valid_types);
		add_child(scene_tree);
		scene_tree->connect("selected", this, "_node_selected");
	}
	scene_tree->popup_centered_ratio();
}

void EditorPropertyNodePath::_node_clear() {

	emit_changed(get_edited_property(), NodePath());
	update_property();
}

void EditorPropertyNodePath::update_property() {

	NodePath p = get_edited_object()->get(get_edited_property());

	assign->set_tooltip(p);
	if (p == NodePath()) {
		assign->set_icon(Ref<Texture>());
		assign->set_text(TTR("Assign..."));
		assign->set_flat(false);
		return;
	}
	assign->set_flat(true);

	Node *base_node = NULL;
	if (base_hint != NodePath()) {
		if (get_tree()->get_root()->has_node(base_hint)) {
			base_node = get_tree()->get_root()->get_node(base_hint);
		}
	} else {
		base_node = Object::cast_to<Node>(get_edited_object());
	}

	if (!base_node || !base_node->has_node(p)) {
		assign->set_icon(Ref<Texture>());
		assign->set_text(p);
		return;
	}

	Node *target_node = base_node->get_node(p);
	ERR_FAIL_COND(!target_node);

	if (String(target_node->get_name()).find("@") != -1) {
		assign->set_icon(Ref<Texture>());
		assign->set_text(p);
		return;
	}

	assign->set_text(target_node->get_name());
	assign->set_icon(EditorNode::get_singleton()->get_object_icon(target_node, "Node"));
}

void EditorPropertyNodePath::setup(const NodePath &p_base_hint, Vector<StringName> p_valid_types, bool p_use_path_from_scene_root) {

	base_hint = p_base_hint;
	valid_types = p_valid_types;
	use_path_from_scene_root = p_use_path_from_scene_root;
}

void EditorPropertyNodePath::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Ref<Texture> t = get_icon("Clear", "EditorIcons");
		clear->set_icon(t);
	}
}

void EditorPropertyNodePath::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_node_selected"), &EditorPropertyNodePath::_node_selected);
	ClassDB::bind_method(D_METHOD("_node_assign"), &EditorPropertyNodePath::_node_assign);
	ClassDB::bind_method(D_METHOD("_node_clear"), &EditorPropertyNodePath::_node_clear);
}

EditorPropertyNodePath::EditorPropertyNodePath() {

	HBoxContainer *hbc = memnew(HBoxContainer);
	add_child(hbc);
	assign = memnew(Button);
	assign->set_flat(true);
	assign->set_h_size_flags(SIZE_EXPAND_FILL);
	assign->set_clip_text(true);
	assign->connect("pressed", this, "_node_assign");
	hbc->add_child(assign);

	clear = memnew(Button);
	clear->set_flat(true);
	clear->connect("pressed", this, "_node_clear");
	hbc->add_child(clear);
	use_path_from_scene_root = false;

	scene_tree = NULL; //do not allocate unnecessarily
}

///////////////////// RID /////////////////////////

void EditorPropertyRID::update_property() {
	RID rid = get_edited_object()->get(get_edited_property());
	if (rid.is_valid()) {
		int id = rid.get_id();
		label->set_text("RID: " + itos(id));
	} else {
		label->set_text(TTR("Invalid RID"));
	}
}

EditorPropertyRID::EditorPropertyRID() {
	label = memnew(Label);
	add_child(label);
}

////////////// RESOURCE //////////////////////

void EditorPropertyResource::_file_selected(const String &p_path) {

	RES res = ResourceLoader::load(p_path);

	List<PropertyInfo> prop_list;
	get_edited_object()->get_property_list(&prop_list);
	String property_types;

	for (List<PropertyInfo>::Element *E = prop_list.front(); E; E = E->next()) {
		if (E->get().name == get_edited_property() && (E->get().hint & PROPERTY_HINT_RESOURCE_TYPE)) {
			property_types = E->get().hint_string;
		}
	}
	if (!property_types.empty()) {
		bool any_type_matches = false;
		const Vector<String> split_property_types = property_types.split(",");
		for (int i = 0; i < split_property_types.size(); ++i) {
			if (res->is_class(split_property_types[i])) {
				any_type_matches = true;
				break;
			}
		}

		if (!any_type_matches)
			EditorNode::get_singleton()->show_warning(vformat(TTR("The selected resource (%s) does not match any type expected for this property (%s)."), res->get_class(), property_types));
	}

	emit_changed(get_edited_property(), res);
	update_property();
}

void EditorPropertyResource::_menu_option(int p_which) {

	//	scene_tree->popup_centered_ratio();
	switch (p_which) {
		case OBJ_MENU_LOAD: {

			if (!file) {
				file = memnew(EditorFileDialog);
				file->connect("file_selected", this, "_file_selected");
				add_child(file);
			}
			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			String type = base_type;

			List<String> extensions;
			for (int i = 0; i < type.get_slice_count(","); i++) {

				ResourceLoader::get_recognized_extensions_for_type(type.get_slice(",", i), &extensions);
			}

			Set<String> valid_extensions;
			for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
				valid_extensions.insert(E->get());
			}

			file->clear_filters();
			for (Set<String>::Element *E = valid_extensions.front(); E; E = E->next()) {

				file->add_filter("*." + E->get() + " ; " + E->get().to_upper());
			}

			file->popup_centered_ratio();
		} break;

		case OBJ_MENU_EDIT: {

			RES res = get_edited_object()->get(get_edited_property());

			if (!res.is_null()) {

				emit_signal("resource_selected", get_edited_property(), res);
			}
		} break;
		case OBJ_MENU_CLEAR: {

			emit_changed(get_edited_property(), RES());
			update_property();

		} break;

		case OBJ_MENU_MAKE_UNIQUE: {

			RES res_orig = get_edited_object()->get(get_edited_property());
			if (res_orig.is_null())
				return;

			List<PropertyInfo> property_list;
			res_orig->get_property_list(&property_list);
			List<Pair<String, Variant> > propvalues;

			for (List<PropertyInfo>::Element *E = property_list.front(); E; E = E->next()) {

				Pair<String, Variant> p;
				PropertyInfo &pi = E->get();
				if (pi.usage & PROPERTY_USAGE_STORAGE) {

					p.first = pi.name;
					p.second = res_orig->get(pi.name);
				}

				propvalues.push_back(p);
			}

			String orig_type = res_orig->get_class();

			Object *inst = ClassDB::instance(orig_type);

			Ref<Resource> res = Ref<Resource>(Object::cast_to<Resource>(inst));

			ERR_FAIL_COND(res.is_null());

			for (List<Pair<String, Variant> >::Element *E = propvalues.front(); E; E = E->next()) {

				Pair<String, Variant> &p = E->get();
				res->set(p.first, p.second);
			}

			emit_changed(get_edited_property(), res);
			update_property();

		} break;

		case OBJ_MENU_SAVE: {
			RES res = get_edited_object()->get(get_edited_property());
			if (res.is_null())
				return;
			EditorNode::get_singleton()->save_resource(res);
		} break;

		case OBJ_MENU_COPY: {
			RES res = get_edited_object()->get(get_edited_property());

			EditorSettings::get_singleton()->set_resource_clipboard(res);

		} break;
		case OBJ_MENU_PASTE: {

			RES res = EditorSettings::get_singleton()->get_resource_clipboard();
			emit_changed(get_edited_property(), res);
			update_property();

		} break;
		case OBJ_MENU_NEW_SCRIPT: {

			if (Object::cast_to<Node>(get_edited_object())) {
				EditorNode::get_singleton()->get_scene_tree_dock()->open_script_dialog(Object::cast_to<Node>(get_edited_object()));
			}

		} break;
		case OBJ_MENU_SHOW_IN_FILE_SYSTEM: {
			RES res = get_edited_object()->get(get_edited_property());

			FileSystemDock *file_system_dock = EditorNode::get_singleton()->get_filesystem_dock();
			file_system_dock->navigate_to_path(res->get_path());
			// Ensure that the FileSystem dock is visible.
			TabContainer *tab_container = (TabContainer *)file_system_dock->get_parent_control();
			tab_container->set_current_tab(file_system_dock->get_position_in_parent());
		} break;
		default: {

			RES res = get_edited_object()->get(get_edited_property());

			if (p_which >= CONVERT_BASE_ID) {

				int to_type = p_which - CONVERT_BASE_ID;

				Vector<Ref<EditorResourceConversionPlugin> > conversions = EditorNode::get_singleton()->find_resource_conversion_plugin(res);

				ERR_FAIL_INDEX(to_type, conversions.size());

				Ref<Resource> new_res = conversions[to_type]->convert(res);

				emit_changed(get_edited_property(), new_res);
				update_property();
				break;
			}
			ERR_FAIL_COND(inheritors_array.empty());

			String intype = inheritors_array[p_which - TYPE_BASE_ID];

			if (intype == "ViewportTexture") {

				Resource *r = Object::cast_to<Resource>(get_edited_object());
				if (r && r->get_path().is_resource_file()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't create a ViewportTexture on resources saved as a file.\nResource needs to belong to a scene."));
					return;
				}

				if (r && !r->is_local_to_scene()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't create a ViewportTexture on this resource because it's not set as local to scene.\nPlease switch on the 'local to scene' property on it (and all resources containing it up to a node)."));
					return;
				}

				if (!scene_tree) {
					scene_tree = memnew(SceneTreeDialog);
					Vector<StringName> valid_types;
					valid_types.push_back("Viewport");
					scene_tree->get_scene_tree()->set_valid_types(valid_types);
					scene_tree->get_scene_tree()->set_show_enabled_subscene(true);
					add_child(scene_tree);
					scene_tree->connect("selected", this, "_viewport_selected");
					scene_tree->set_title(TTR("Pick a Viewport"));
				}
				scene_tree->popup_centered_ratio();

				return;
			}

			Object *obj = ClassDB::instance(intype);

			if (!obj) {
				obj = EditorNode::get_editor_data().instance_custom_type(intype, "Resource");
			}

			ERR_BREAK(!obj);
			Resource *resp = Object::cast_to<Resource>(obj);
			ERR_BREAK(!resp);
			if (get_edited_object() && base_type != String() && base_type == "Script") {
				//make visual script the right type
				resp->call("set_instance_base_type", get_edited_object()->get_class());
			}

			res = Ref<Resource>(resp);
			emit_changed(get_edited_property(), res);
			update_property();

		} break;
	}
}

void EditorPropertyResource::_resource_preview(const String &p_path, const Ref<Texture> &p_preview, const Ref<Texture> &p_small_preview, ObjectID p_obj) {

	RES p = get_edited_object()->get(get_edited_property());
	if (p.is_valid() && p->get_instance_id() == p_obj) {
		String type = p->get_class_name();

		if (ClassDB::is_parent_class(type, "Script")) {
			assign->set_text(p->get_path().get_file());
			return;
		}

		if (p_preview.is_valid()) {
			preview->set_margin(MARGIN_LEFT, assign->get_icon()->get_width() + assign->get_stylebox("normal")->get_default_margin(MARGIN_LEFT) + get_constant("hseparation", "Button"));
			if (type == "GradientTexture") {
				preview->set_stretch_mode(TextureRect::STRETCH_SCALE);
				assign->set_custom_minimum_size(Size2(1, 1));
			} else {
				preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
				int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
				thumbnail_size *= EDSCALE;
				assign->set_custom_minimum_size(Size2(1, thumbnail_size));
			}
			preview->set_texture(p_preview);
			assign->set_text("");
		}
	}
}

void EditorPropertyResource::_update_menu_items() {

	//////////////////// UPDATE MENU //////////////////////////
	RES res = get_edited_object()->get(get_edited_property());

	menu->clear();

	if (get_edited_property() == "script" && base_type == "Script" && Object::cast_to<Node>(get_edited_object())) {
		menu->add_icon_item(get_icon("Script", "EditorIcons"), TTR("New Script"), OBJ_MENU_NEW_SCRIPT);
		menu->add_separator();
	} else if (base_type != "") {
		int idx = 0;

		Vector<EditorData::CustomType> custom_resources;

		if (EditorNode::get_editor_data().get_custom_types().has("Resource")) {
			custom_resources = EditorNode::get_editor_data().get_custom_types()["Resource"];
		}

		for (int i = 0; i < base_type.get_slice_count(","); i++) {

			String base = base_type.get_slice(",", i);

			Set<String> valid_inheritors;
			valid_inheritors.insert(base);
			List<StringName> inheritors;
			ClassDB::get_inheriters_from_class(base.strip_edges(), &inheritors);

			for (int i = 0; i < custom_resources.size(); i++) {
				inheritors.push_back(custom_resources[i].name);
			}

			List<StringName>::Element *E = inheritors.front();
			while (E) {
				valid_inheritors.insert(E->get());
				E = E->next();
			}

			for (Set<String>::Element *E = valid_inheritors.front(); E; E = E->next()) {
				String t = E->get();

				bool is_custom_resource = false;
				Ref<Texture> icon;
				if (!custom_resources.empty()) {
					for (int i = 0; i < custom_resources.size(); i++) {
						if (custom_resources[i].name == t) {
							is_custom_resource = true;
							if (custom_resources[i].icon.is_valid())
								icon = custom_resources[i].icon;
							break;
						}
					}
				}

				if (!is_custom_resource && !ClassDB::can_instance(t))
					continue;

				inheritors_array.push_back(t);

				int id = TYPE_BASE_ID + idx;

				if (!icon.is_valid() && has_icon(t, "EditorIcons")) {
					icon = get_icon(t, "EditorIcons");
				}

				if (icon.is_valid()) {

					menu->add_icon_item(icon, vformat(TTR("New %s"), t), id);
				} else {

					menu->add_item(vformat(TTR("New %s"), t), id);
				}

				idx++;
			}
		}

		if (menu->get_item_count())
			menu->add_separator();
	}

	menu->add_icon_item(get_icon("Load", "EditorIcons"), TTR("Load"), OBJ_MENU_LOAD);

	if (!res.is_null()) {

		menu->add_icon_item(get_icon("Edit", "EditorIcons"), TTR("Edit"), OBJ_MENU_EDIT);
		menu->add_icon_item(get_icon("Clear", "EditorIcons"), TTR("Clear"), OBJ_MENU_CLEAR);
		menu->add_icon_item(get_icon("Duplicate", "EditorIcons"), TTR("Make Unique"), OBJ_MENU_MAKE_UNIQUE);
		menu->add_icon_item(get_icon("Save", "EditorIcons"), TTR("Save"), OBJ_MENU_SAVE);
		RES r = res;
		if (r.is_valid() && r->get_path().is_resource_file()) {
			menu->add_separator();
			menu->add_item(TTR("Show in FileSystem"), OBJ_MENU_SHOW_IN_FILE_SYSTEM);
		}
	} else {
	}

	RES cb = EditorSettings::get_singleton()->get_resource_clipboard();
	bool paste_valid = false;
	if (cb.is_valid()) {
		if (base_type == "")
			paste_valid = true;
		else
			for (int i = 0; i < base_type.get_slice_count(","); i++)
				if (ClassDB::is_parent_class(cb->get_class(), base_type.get_slice(",", i))) {
					paste_valid = true;
					break;
				}
	}

	if (!res.is_null() || paste_valid) {
		menu->add_separator();

		if (!res.is_null()) {

			menu->add_item(TTR("Copy"), OBJ_MENU_COPY);
		}

		if (paste_valid) {

			menu->add_item(TTR("Paste"), OBJ_MENU_PASTE);
		}
	}

	if (!res.is_null()) {

		Vector<Ref<EditorResourceConversionPlugin> > conversions = EditorNode::get_singleton()->find_resource_conversion_plugin(res);
		if (conversions.size()) {
			menu->add_separator();
		}
		for (int i = 0; i < conversions.size(); i++) {
			String what = conversions[i]->converts_to();
			Ref<Texture> icon;
			if (has_icon(what, "EditorIcons")) {

				icon = get_icon(what, "EditorIcons");
			} else {

				icon = get_icon(what, "Resource");
			}

			menu->add_icon_item(icon, vformat(TTR("Convert To %s"), what), CONVERT_BASE_ID + i);
		}
	}
}

void EditorPropertyResource::_update_menu() {

	_update_menu_items();

	Rect2 gt = edit->get_global_rect();
	menu->set_as_minsize();
	int ms = menu->get_combined_minimum_size().width;
	Vector2 popup_pos = gt.position + gt.size - Vector2(ms, 0);
	menu->set_global_position(popup_pos);
	menu->popup();
}

void EditorPropertyResource::_sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool) {

	emit_signal("property_keyed_with_value", String(get_edited_property()) + ":" + p_property, p_value, false);
}

void EditorPropertyResource::_sub_inspector_resource_selected(const RES &p_resource, const String &p_property) {

	emit_signal("resource_selected", String(get_edited_property()) + ":" + p_property, p_resource);
}

void EditorPropertyResource::_sub_inspector_object_id_selected(int p_id) {

	emit_signal("object_id_selected", get_edited_property(), p_id);
}

void EditorPropertyResource::_button_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) {
			_update_menu_items();
			Vector2 pos = mb->get_global_position();
			//pos = assign->get_global_transform().xform(pos);
			menu->set_as_minsize();
			menu->set_global_position(pos);
			menu->popup();
		}
	}
}

void EditorPropertyResource::_open_editor_pressed() {
	RES res = get_edited_object()->get(get_edited_property());
	if (res.is_valid()) {
		EditorNode::get_singleton()->edit_item(res.ptr());
	}
}

void EditorPropertyResource::update_property() {

	RES res = get_edited_object()->get(get_edited_property());

	if (use_sub_inspector) {

		if (res.is_valid() != assign->is_toggle_mode()) {
			assign->set_toggle_mode(res.is_valid());
		}
#ifdef TOOLS_ENABLED
		if (res.is_valid() && get_edited_object()->editor_is_section_unfolded(get_edited_property())) {

			if (!sub_inspector) {
				sub_inspector = memnew(EditorInspector);
				sub_inspector->set_enable_v_scroll(false);
				sub_inspector->set_use_doc_hints(true);

				sub_inspector->set_use_sub_inspector_bg(true);
				sub_inspector->set_enable_capitalize_paths(true);

				sub_inspector->connect("property_keyed", this, "_sub_inspector_property_keyed");
				sub_inspector->connect("resource_selected", this, "_sub_inspector_resource_selected");
				sub_inspector->connect("object_id_selected", this, "_sub_inspector_object_id_selected");
				sub_inspector->set_keying(is_keying());
				sub_inspector->set_read_only(is_read_only());
				sub_inspector->set_use_folding(is_using_folding());
				sub_inspector->set_undo_redo(EditorNode::get_singleton()->get_undo_redo());

				sub_inspector_vbox = memnew(VBoxContainer);
				add_child(sub_inspector_vbox);
				set_bottom_editor(sub_inspector_vbox);

				sub_inspector_vbox->add_child(sub_inspector);
				assign->set_pressed(true);

				bool use_editor = false;
				for (int i = 0; i < EditorNode::get_singleton()->get_editor_data().get_editor_plugin_count(); i++) {
					EditorPlugin *ep = EditorNode::get_singleton()->get_editor_data().get_editor_plugin(i);
					if (ep->handles(res.ptr())) {
						use_editor = true;
					}
				}

				if (use_editor) {
					Button *open_in_editor = memnew(Button);
					open_in_editor->set_text(TTR("Open Editor"));
					open_in_editor->set_icon(get_icon("Edit", "EditorIcons"));
					sub_inspector_vbox->add_child(open_in_editor);
					open_in_editor->connect("pressed", this, "_open_editor_pressed");
					open_in_editor->set_h_size_flags(SIZE_SHRINK_CENTER);
				}
			}

			if (res.ptr() != sub_inspector->get_edited_object()) {
				sub_inspector->edit(res.ptr());
			}

		} else {
			if (sub_inspector) {
				set_bottom_editor(NULL);
				memdelete(sub_inspector_vbox);
				sub_inspector = NULL;
				sub_inspector_vbox = NULL;
			}
		}
#endif
	}

	preview->set_texture(Ref<Texture>());
	if (res == RES()) {
		assign->set_icon(Ref<Texture>());
		assign->set_text(TTR("[empty]"));
	} else {

		assign->set_icon(EditorNode::get_singleton()->get_object_icon(res.operator->(), "Node"));

		if (res->get_name() != String()) {
			assign->set_text(res->get_name());
		} else if (res->get_path().is_resource_file()) {
			assign->set_text(res->get_path().get_file());
			assign->set_tooltip(res->get_path());
		} else {
			assign->set_text(res->get_class());
		}

		if (res->get_path().is_resource_file()) {
			assign->set_tooltip(res->get_path());
		}

		//preview will override the above, so called at the end
		EditorResourcePreview::get_singleton()->queue_edited_resource_preview(res, this, "_resource_preview", res->get_instance_id());
	}
}

void EditorPropertyResource::_resource_selected() {
	RES res = get_edited_object()->get(get_edited_property());

	if (res.is_null()) {
		_update_menu();
		return;
	}

	if (use_sub_inspector) {

		get_edited_object()->editor_set_section_unfold(get_edited_property(), assign->is_pressed());
		update_property();
	} else {

		emit_signal("resource_selected", get_edited_property(), res);
	}
}

void EditorPropertyResource::setup(const String &p_base_type) {
	base_type = p_base_type;
}

void EditorPropertyResource::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		Ref<Texture> t = get_icon("select_arrow", "Tree");
		edit->set_icon(t);
	}

	if (p_what == NOTIFICATION_DRAG_BEGIN) {

		if (is_visible_in_tree()) {
			if (_is_drop_valid(get_viewport()->gui_get_drag_data())) {
				dropping = true;
				assign->update();
			}
		}
	}

	if (p_what == NOTIFICATION_DRAG_END) {
		if (dropping) {
			dropping = false;
			assign->update();
		}
	}
}

void EditorPropertyResource::_viewport_selected(const NodePath &p_path) {

	Node *to_node = get_node(p_path);
	if (!Object::cast_to<Viewport>(to_node)) {
		EditorNode::get_singleton()->show_warning(TTR("Selected node is not a Viewport!"));
		return;
	}

	Ref<ViewportTexture> vt;
	vt.instance();
	vt->set_viewport_path_in_scene(get_tree()->get_edited_scene_root()->get_path_to(to_node));
	vt->setup_local_to_scene();

	emit_changed(get_edited_property(), vt);
	update_property();
}

void EditorPropertyResource::collapse_all_folding() {
	if (sub_inspector) {
		sub_inspector->collapse_all_folding();
	}
}

void EditorPropertyResource::expand_all_folding() {

	if (sub_inspector) {
		sub_inspector->expand_all_folding();
	}
}

void EditorPropertyResource::_button_draw() {

	if (dropping) {
		Color color = get_color("accent_color", "Editor");
		assign->draw_rect(Rect2(Point2(), assign->get_size()), color, false);
	}
}

Variant EditorPropertyResource::get_drag_data_fw(const Point2 &p_point, Control *p_from) {

	RES res = get_edited_object()->get(get_edited_property());
	if (res.is_valid()) {

		return EditorNode::get_singleton()->drag_resource(res, p_from);
	}

	return Variant();
}

bool EditorPropertyResource::_is_drop_valid(const Dictionary &p_drag_data) const {

	String allowed_type = base_type;

	Dictionary drag_data = p_drag_data;
	if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
		Ref<Resource> res = drag_data["resource"];
		for (int i = 0; i < allowed_type.get_slice_count(","); i++) {
			String at = allowed_type.get_slice(",", i).strip_edges();
			if (res.is_valid() && ClassDB::is_parent_class(res->get_class(), at)) {
				return true;
			}
		}
	}

	if (drag_data.has("type") && String(drag_data["type"]) == "files") {

		Vector<String> files = drag_data["files"];

		if (files.size() == 1) {
			String file = files[0];
			String ftype = EditorFileSystem::get_singleton()->get_file_type(file);

			if (ftype != "") {

				for (int i = 0; i < allowed_type.get_slice_count(","); i++) {
					String at = allowed_type.get_slice(",", i).strip_edges();
					if (ClassDB::is_parent_class(ftype, at)) {
						return true;
					}
				}
			}
		}
	}

	return false;
}

bool EditorPropertyResource::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	return _is_drop_valid(p_data);
}
void EditorPropertyResource::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {

	ERR_FAIL_COND(!_is_drop_valid(p_data));

	Dictionary drag_data = p_data;
	if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
		Ref<Resource> res = drag_data["resource"];
		if (res.is_valid()) {
			emit_changed(get_edited_property(), res);
			update_property();
			return;
		}
	}

	if (drag_data.has("type") && String(drag_data["type"]) == "files") {

		Vector<String> files = drag_data["files"];

		if (files.size() == 1) {
			String file = files[0];
			RES res = ResourceLoader::load(file);
			if (res.is_valid()) {
				emit_changed(get_edited_property(), res);
				update_property();
				return;
			}
		}
	}
}

void EditorPropertyResource::set_use_sub_inspector(bool p_enable) {
	use_sub_inspector = p_enable;
}

void EditorPropertyResource::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_file_selected"), &EditorPropertyResource::_file_selected);
	ClassDB::bind_method(D_METHOD("_menu_option"), &EditorPropertyResource::_menu_option);
	ClassDB::bind_method(D_METHOD("_update_menu"), &EditorPropertyResource::_update_menu);
	ClassDB::bind_method(D_METHOD("_resource_preview"), &EditorPropertyResource::_resource_preview);
	ClassDB::bind_method(D_METHOD("_resource_selected"), &EditorPropertyResource::_resource_selected);
	ClassDB::bind_method(D_METHOD("_viewport_selected"), &EditorPropertyResource::_viewport_selected);
	ClassDB::bind_method(D_METHOD("_sub_inspector_property_keyed"), &EditorPropertyResource::_sub_inspector_property_keyed);
	ClassDB::bind_method(D_METHOD("_sub_inspector_resource_selected"), &EditorPropertyResource::_sub_inspector_resource_selected);
	ClassDB::bind_method(D_METHOD("_sub_inspector_object_id_selected"), &EditorPropertyResource::_sub_inspector_object_id_selected);
	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &EditorPropertyResource::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &EditorPropertyResource::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &EditorPropertyResource::drop_data_fw);
	ClassDB::bind_method(D_METHOD("_button_draw"), &EditorPropertyResource::_button_draw);
	ClassDB::bind_method(D_METHOD("_open_editor_pressed"), &EditorPropertyResource::_open_editor_pressed);
	ClassDB::bind_method(D_METHOD("_button_input"), &EditorPropertyResource::_button_input);
}

EditorPropertyResource::EditorPropertyResource() {

	sub_inspector = NULL;
	sub_inspector_vbox = NULL;
	use_sub_inspector = bool(EDITOR_GET("interface/inspector/open_resources_in_current_inspector"));

	HBoxContainer *hbc = memnew(HBoxContainer);
	add_child(hbc);
	assign = memnew(Button);
	assign->set_flat(true);
	assign->set_h_size_flags(SIZE_EXPAND_FILL);
	assign->set_clip_text(true);
	assign->connect("pressed", this, "_resource_selected");
	assign->set_drag_forwarding(this);
	assign->connect("draw", this, "_button_draw");
	hbc->add_child(assign);

	preview = memnew(TextureRect);
	preview->set_expand(true);
	preview->set_anchors_and_margins_preset(PRESET_WIDE);
	preview->set_margin(MARGIN_TOP, 1);
	preview->set_margin(MARGIN_BOTTOM, -1);
	preview->set_margin(MARGIN_RIGHT, -1);
	assign->add_child(preview);
	assign->connect("gui_input", this, "_button_input");

	menu = memnew(PopupMenu);
	add_child(menu);
	edit = memnew(Button);
	edit->set_flat(true);
	menu->connect("id_pressed", this, "_menu_option");
	edit->connect("pressed", this, "_update_menu");
	hbc->add_child(edit);
	edit->connect("gui_input", this, "_button_input");

	file = NULL;
	scene_tree = NULL;
	dropping = false;
}

////////////// DEFAULT PLUGIN //////////////////////

bool EditorInspectorDefaultPlugin::can_handle(Object *p_object) {
	return true; //can handle everything
}

void EditorInspectorDefaultPlugin::parse_begin(Object *p_object) {
	//do none
}

bool EditorInspectorDefaultPlugin::parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage) {

	switch (p_type) {

		// atomic types
		case Variant::NIL: {
			EditorPropertyNil *editor = memnew(EditorPropertyNil);
			add_property_editor(p_path, editor);
		} break;
		case Variant::BOOL: {
			EditorPropertyCheck *editor = memnew(EditorPropertyCheck);
			add_property_editor(p_path, editor);
		} break;
		case Variant::INT: {

			if (p_hint == PROPERTY_HINT_ENUM) {
				EditorPropertyEnum *editor = memnew(EditorPropertyEnum);
				Vector<String> options = p_hint_text.split(",");
				editor->setup(options);
				add_property_editor(p_path, editor);

			} else if (p_hint == PROPERTY_HINT_FLAGS) {
				EditorPropertyFlags *editor = memnew(EditorPropertyFlags);
				Vector<String> options = p_hint_text.split(",");
				editor->setup(options);
				add_property_editor(p_path, editor);

			} else if (p_hint == PROPERTY_HINT_LAYERS_2D_PHYSICS || p_hint == PROPERTY_HINT_LAYERS_2D_RENDER || p_hint == PROPERTY_HINT_LAYERS_3D_PHYSICS || p_hint == PROPERTY_HINT_LAYERS_3D_RENDER) {

				EditorPropertyLayers::LayerType lt = EditorPropertyLayers::LAYER_RENDER_2D;
				switch (p_hint) {
					case PROPERTY_HINT_LAYERS_2D_RENDER:
						lt = EditorPropertyLayers::LAYER_RENDER_2D;
						break;
					case PROPERTY_HINT_LAYERS_2D_PHYSICS:
						lt = EditorPropertyLayers::LAYER_PHYSICS_2D;
						break;
					case PROPERTY_HINT_LAYERS_3D_RENDER:
						lt = EditorPropertyLayers::LAYER_RENDER_3D;
						break;
					case PROPERTY_HINT_LAYERS_3D_PHYSICS:
						lt = EditorPropertyLayers::LAYER_PHYSICS_3D;
						break;
					default: {} //compiler could be smarter here and realize this can't happen
				}
				EditorPropertyLayers *editor = memnew(EditorPropertyLayers);
				editor->setup(lt);
				add_property_editor(p_path, editor);
			} else if (p_hint == PROPERTY_HINT_OBJECT_ID) {

				EditorPropertyObjectID *editor = memnew(EditorPropertyObjectID);
				editor->setup(p_hint_text);
				add_property_editor(p_path, editor);

			} else {
				EditorPropertyInteger *editor = memnew(EditorPropertyInteger);
				int min = 0, max = 65535, step = 1;
				bool greater = true, lesser = true;

				if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
					greater = false; //if using ranged, assume false by default
					lesser = false;
					min = p_hint_text.get_slice(",", 0).to_int();
					max = p_hint_text.get_slice(",", 1).to_int();

					if (p_hint_text.get_slice_count(",") >= 3) {
						step = p_hint_text.get_slice(",", 2).to_int();
					}

					for (int i = 2; i < p_hint_text.get_slice_count(","); i++) {
						String slice = p_hint_text.get_slice(",", i).strip_edges();
						if (slice == "or_greater") {
							greater = true;
						}
						if (slice == "or_lesser") {
							lesser = true;
						}
					}
				}

				editor->setup(min, max, step, greater, lesser);

				add_property_editor(p_path, editor);
			}
		} break;
		case Variant::REAL: {

			if (p_hint == PROPERTY_HINT_EXP_EASING) {
				EditorPropertyEasing *editor = memnew(EditorPropertyEasing);
				bool full = true;
				bool flip = false;
				Vector<String> hints = p_hint_text.split(",");
				for (int i = 0; i < hints.size(); i++) {
					String h = hints[i].strip_edges();
					if (h == "attenuation") {
						flip = true;
					}
					if (h == "inout") {
						full = true;
					}
				}

				editor->setup(full, flip);
				add_property_editor(p_path, editor);

			} else {
				EditorPropertyFloat *editor = memnew(EditorPropertyFloat);
				double min = -65535, max = 65535, step = 0.001;
				bool hide_slider = true;
				bool exp_range = false;
				bool greater = true, lesser = true;

				if ((p_hint == PROPERTY_HINT_RANGE || p_hint == PROPERTY_HINT_EXP_RANGE) && p_hint_text.get_slice_count(",") >= 2) {
					greater = false; //if using ranged, assume false by default
					lesser = false;
					min = p_hint_text.get_slice(",", 0).to_double();
					max = p_hint_text.get_slice(",", 1).to_double();
					if (p_hint_text.get_slice_count(",") >= 3) {
						step = p_hint_text.get_slice(",", 2).to_double();
					}
					hide_slider = false;
					exp_range = p_hint == PROPERTY_HINT_EXP_RANGE;
					for (int i = 2; i < p_hint_text.get_slice_count(","); i++) {
						String slice = p_hint_text.get_slice(",", i).strip_edges();
						if (slice == "or_greater") {
							greater = true;
						}
						if (slice == "or_lesser") {
							lesser = true;
						}
					}
				}

				editor->setup(min, max, step, hide_slider, exp_range, greater, lesser);

				add_property_editor(p_path, editor);
			}
		} break;
		case Variant::STRING: {

			if (p_hint == PROPERTY_HINT_ENUM) {
				EditorPropertyTextEnum *editor = memnew(EditorPropertyTextEnum);
				Vector<String> options = p_hint_text.split(",");
				editor->setup(options);
				add_property_editor(p_path, editor);
			} else if (p_hint == PROPERTY_HINT_MULTILINE_TEXT) {
				EditorPropertyMultilineText *editor = memnew(EditorPropertyMultilineText);
				add_property_editor(p_path, editor);
			} else if (p_hint == PROPERTY_HINT_TYPE_STRING) {
				EditorPropertyClassName *editor = memnew(EditorPropertyClassName);
				editor->setup("Object", p_hint_text);
				add_property_editor(p_path, editor);
			} else if (p_hint == PROPERTY_HINT_DIR || p_hint == PROPERTY_HINT_FILE || p_hint == PROPERTY_HINT_GLOBAL_DIR || p_hint == PROPERTY_HINT_GLOBAL_FILE) {

				Vector<String> extensions = p_hint_text.split(",");
				bool global = p_hint == PROPERTY_HINT_GLOBAL_DIR || p_hint == PROPERTY_HINT_GLOBAL_FILE;
				bool folder = p_hint == PROPERTY_HINT_DIR || p_hint == PROPERTY_HINT_GLOBAL_DIR;
				EditorPropertyPath *editor = memnew(EditorPropertyPath);
				editor->setup(extensions, folder, global);
				add_property_editor(p_path, editor);
			} else if (p_hint == PROPERTY_HINT_METHOD_OF_VARIANT_TYPE ||
					   p_hint == PROPERTY_HINT_METHOD_OF_BASE_TYPE ||
					   p_hint == PROPERTY_HINT_METHOD_OF_INSTANCE ||
					   p_hint == PROPERTY_HINT_METHOD_OF_SCRIPT ||
					   p_hint == PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE ||
					   p_hint == PROPERTY_HINT_PROPERTY_OF_BASE_TYPE ||
					   p_hint == PROPERTY_HINT_PROPERTY_OF_INSTANCE ||
					   p_hint == PROPERTY_HINT_PROPERTY_OF_SCRIPT) {

				EditorPropertyMember *editor = memnew(EditorPropertyMember);

				EditorPropertyMember::Type type = EditorPropertyMember::MEMBER_METHOD_OF_BASE_TYPE;
				switch (p_hint) {
					case PROPERTY_HINT_METHOD_OF_BASE_TYPE: type = EditorPropertyMember::MEMBER_METHOD_OF_BASE_TYPE; break;
					case PROPERTY_HINT_METHOD_OF_INSTANCE: type = EditorPropertyMember::MEMBER_METHOD_OF_INSTANCE; break;
					case PROPERTY_HINT_METHOD_OF_SCRIPT: type = EditorPropertyMember::MEMBER_METHOD_OF_SCRIPT; break;
					case PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE: type = EditorPropertyMember::MEMBER_PROPERTY_OF_VARIANT_TYPE; break;
					case PROPERTY_HINT_PROPERTY_OF_BASE_TYPE: type = EditorPropertyMember::MEMBER_PROPERTY_OF_BASE_TYPE; break;
					case PROPERTY_HINT_PROPERTY_OF_INSTANCE: type = EditorPropertyMember::MEMBER_PROPERTY_OF_INSTANCE; break;
					case PROPERTY_HINT_PROPERTY_OF_SCRIPT: type = EditorPropertyMember::MEMBER_PROPERTY_OF_SCRIPT; break;
					default: {}
				}
				editor->setup(type, p_hint_text);
				add_property_editor(p_path, editor);

			} else {

				EditorPropertyText *editor = memnew(EditorPropertyText);
				if (p_hint == PROPERTY_HINT_PLACEHOLDER_TEXT) {
					editor->set_placeholder(p_hint_text);
				}
				add_property_editor(p_path, editor);
			}
		} break;

			// math types

		case Variant::VECTOR2: {
			EditorPropertyVector2 *editor = memnew(EditorPropertyVector2);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);

		} break; // 5
		case Variant::RECT2: {
			EditorPropertyRect2 *editor = memnew(EditorPropertyRect2);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);
		} break;
		case Variant::VECTOR3: {
			EditorPropertyVector3 *editor = memnew(EditorPropertyVector3);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);

		} break;
		case Variant::TRANSFORM2D: {
			EditorPropertyTransform2D *editor = memnew(EditorPropertyTransform2D);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);

		} break;
		case Variant::PLANE: {
			EditorPropertyPlane *editor = memnew(EditorPropertyPlane);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);
		} break;
		case Variant::QUAT: {
			EditorPropertyQuat *editor = memnew(EditorPropertyQuat);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);
		} break; // 10
		case Variant::AABB: {
			EditorPropertyAABB *editor = memnew(EditorPropertyAABB);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);
		} break;
		case Variant::BASIS: {
			EditorPropertyBasis *editor = memnew(EditorPropertyBasis);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);
		} break;
		case Variant::TRANSFORM: {
			EditorPropertyTransform *editor = memnew(EditorPropertyTransform);
			double min = -65535, max = 65535, step = 0.001;
			bool hide_slider = true;

			if (p_hint == PROPERTY_HINT_RANGE && p_hint_text.get_slice_count(",") >= 2) {
				min = p_hint_text.get_slice(",", 0).to_double();
				max = p_hint_text.get_slice(",", 1).to_double();
				if (p_hint_text.get_slice_count(",") >= 3) {
					step = p_hint_text.get_slice(",", 2).to_double();
				}
				hide_slider = false;
			}

			editor->setup(min, max, step, hide_slider);
			add_property_editor(p_path, editor);

		} break;

		// misc types
		case Variant::COLOR: {
			EditorPropertyColor *editor = memnew(EditorPropertyColor);
			editor->setup(p_hint != PROPERTY_HINT_COLOR_NO_ALPHA);
			add_property_editor(p_path, editor);
		} break;
		case Variant::NODE_PATH: {

			EditorPropertyNodePath *editor = memnew(EditorPropertyNodePath);
			if (p_hint == PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE && p_hint_text != String()) {
				editor->setup(p_hint_text, Vector<StringName>(), (p_usage & PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT));
			}
			if (p_hint == PROPERTY_HINT_NODE_PATH_VALID_TYPES && p_hint_text != String()) {
				Vector<String> types = p_hint_text.split(",", false);
				Vector<StringName> sn = Variant(types); //convert via variant
				editor->setup(NodePath(), sn, (p_usage & PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT));
			}
			add_property_editor(p_path, editor);

		} break; // 15
		case Variant::_RID: {
			EditorPropertyRID *editor = memnew(EditorPropertyRID);
			add_property_editor(p_path, editor);
		} break;
		case Variant::OBJECT: {
			EditorPropertyResource *editor = memnew(EditorPropertyResource);
			editor->setup(p_hint == PROPERTY_HINT_RESOURCE_TYPE ? p_hint_text : "Resource");

			if (p_hint == PROPERTY_HINT_RESOURCE_TYPE) {
				String open_in_new = EDITOR_GET("interface/inspector/resources_types_to_open_in_new_inspector");
				for (int i = 0; i < open_in_new.get_slice_count(","); i++) {
					String type = open_in_new.get_slicec(',', i).strip_edges();
					for (int j = 0; j < p_hint_text.get_slice_count(","); j++) {
						String inherits = p_hint_text.get_slicec(',', j);

						if (ClassDB::is_parent_class(inherits, type)) {

							editor->set_use_sub_inspector(false);
						}
					}
				}
			}

			add_property_editor(p_path, editor);

		} break;
		case Variant::DICTIONARY: {
			EditorPropertyDictionary *editor = memnew(EditorPropertyDictionary);
			add_property_editor(p_path, editor);
		} break;
		case Variant::ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::ARRAY, p_hint_text);
			add_property_editor(p_path, editor);
		} break;
		case Variant::POOL_BYTE_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_BYTE_ARRAY);
			add_property_editor(p_path, editor);
		} break; // 20
		case Variant::POOL_INT_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_INT_ARRAY);
			add_property_editor(p_path, editor);
		} break;
		case Variant::POOL_REAL_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_REAL_ARRAY);
			add_property_editor(p_path, editor);
		} break;
		case Variant::POOL_STRING_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_STRING_ARRAY);
			add_property_editor(p_path, editor);
		} break;
		case Variant::POOL_VECTOR2_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_VECTOR2_ARRAY);
			add_property_editor(p_path, editor);
		} break;
		case Variant::POOL_VECTOR3_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_VECTOR3_ARRAY);
			add_property_editor(p_path, editor);
		} break; // 25
		case Variant::POOL_COLOR_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::POOL_COLOR_ARRAY);
			add_property_editor(p_path, editor);
		} break;
		default: {}
	}

	return false; //can be overridden, although it will most likely be last anyway
}

void EditorInspectorDefaultPlugin::parse_end() {
	//do none
}
