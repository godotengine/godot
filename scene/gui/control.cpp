/*************************************************************************/
/*  control.cpp                                                          */
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

#include "control.h"

#include "container.h"
#include "core/config/project_settings.h"
#include "core/math/geometry_2d.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"
#include "scene/theme/theme_db.h"
#include "servers/rendering_server.h"
#include "servers/text_server.h"

#ifdef TOOLS_ENABLED
#include "editor/plugins/control_editor_plugin.h"
#endif

// Editor plugin interoperability.

// TODO: Decouple controls from their editor plugin and get rid of this.
#ifdef TOOLS_ENABLED
Dictionary Control::_edit_get_state() const {
	Dictionary s;
	s["rotation"] = get_rotation();
	s["scale"] = get_scale();
	s["pivot"] = get_pivot_offset();

	Array anchors;
	anchors.push_back(get_anchor(SIDE_LEFT));
	anchors.push_back(get_anchor(SIDE_TOP));
	anchors.push_back(get_anchor(SIDE_RIGHT));
	anchors.push_back(get_anchor(SIDE_BOTTOM));
	s["anchors"] = anchors;

	Array offsets;
	offsets.push_back(get_offset(SIDE_LEFT));
	offsets.push_back(get_offset(SIDE_TOP));
	offsets.push_back(get_offset(SIDE_RIGHT));
	offsets.push_back(get_offset(SIDE_BOTTOM));
	s["offsets"] = offsets;

	s["layout_mode"] = _get_layout_mode();
	s["anchors_layout_preset"] = _get_anchors_layout_preset();

	return s;
}

void Control::_edit_set_state(const Dictionary &p_state) {
	ERR_FAIL_COND((p_state.size() <= 0) ||
			!p_state.has("rotation") || !p_state.has("scale") ||
			!p_state.has("pivot") || !p_state.has("anchors") || !p_state.has("offsets") ||
			!p_state.has("layout_mode") || !p_state.has("anchors_layout_preset"));
	Dictionary state = p_state;

	set_rotation(state["rotation"]);
	set_scale(state["scale"]);
	set_pivot_offset(state["pivot"]);

	Array anchors = state["anchors"];

	// If anchors are not in their default position, force the anchor layout mode in place of position.
	LayoutMode _layout = (LayoutMode)(int)state["layout_mode"];
	if (_layout == LayoutMode::LAYOUT_MODE_POSITION) {
		bool anchors_mode = ((real_t)anchors[0] != 0.0 || (real_t)anchors[1] != 0.0 || (real_t)anchors[2] != 0.0 || (real_t)anchors[3] != 0.0);
		if (anchors_mode) {
			_layout = LayoutMode::LAYOUT_MODE_ANCHORS;
		}
	}

	_set_layout_mode(_layout);
	if (_layout == LayoutMode::LAYOUT_MODE_ANCHORS) {
		_set_anchors_layout_preset((int)state["anchors_layout_preset"]);
	}

	data.anchor[SIDE_LEFT] = anchors[0];
	data.anchor[SIDE_TOP] = anchors[1];
	data.anchor[SIDE_RIGHT] = anchors[2];
	data.anchor[SIDE_BOTTOM] = anchors[3];

	Array offsets = state["offsets"];
	data.offset[SIDE_LEFT] = offsets[0];
	data.offset[SIDE_TOP] = offsets[1];
	data.offset[SIDE_RIGHT] = offsets[2];
	data.offset[SIDE_BOTTOM] = offsets[3];

	_size_changed();
}

void Control::_edit_set_position(const Point2 &p_position) {
	ERR_FAIL_COND_MSG(!Engine::get_singleton()->is_editor_hint(), "This function can only be used from editor plugins.");
	set_position(p_position, ControlEditorToolbar::get_singleton()->is_anchors_mode_enabled() && Object::cast_to<Control>(data.parent));
};

Point2 Control::_edit_get_position() const {
	return get_position();
};

void Control::_edit_set_scale(const Size2 &p_scale) {
	set_scale(p_scale);
}

Size2 Control::_edit_get_scale() const {
	return data.scale;
}

void Control::_edit_set_rect(const Rect2 &p_edit_rect) {
	ERR_FAIL_COND_MSG(!Engine::get_singleton()->is_editor_hint(), "This function can only be used from editor plugins.");
	set_position((get_position() + get_transform().basis_xform(p_edit_rect.position)).snapped(Vector2(1, 1)), ControlEditorToolbar::get_singleton()->is_anchors_mode_enabled());
	set_size(p_edit_rect.size.snapped(Vector2(1, 1)), ControlEditorToolbar::get_singleton()->is_anchors_mode_enabled());
}

Rect2 Control::_edit_get_rect() const {
	return Rect2(Point2(), get_size());
}

bool Control::_edit_use_rect() const {
	return true;
}

void Control::_edit_set_rotation(real_t p_rotation) {
	set_rotation(p_rotation);
}

real_t Control::_edit_get_rotation() const {
	return get_rotation();
}

bool Control::_edit_use_rotation() const {
	return true;
}

void Control::_edit_set_pivot(const Point2 &p_pivot) {
	Vector2 delta_pivot = p_pivot - get_pivot_offset();
	Vector2 move = Vector2((cos(data.rotation) - 1.0) * delta_pivot.x - sin(data.rotation) * delta_pivot.y, sin(data.rotation) * delta_pivot.x + (cos(data.rotation) - 1.0) * delta_pivot.y);
	set_position(get_position() + move);
	set_pivot_offset(p_pivot);
}

Point2 Control::_edit_get_pivot() const {
	return get_pivot_offset();
}

bool Control::_edit_use_pivot() const {
	return true;
}

Size2 Control::_edit_get_minimum_size() const {
	return get_combined_minimum_size();
}
#endif

// Editor integration.

void Control::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	Node::get_argument_options(p_function, p_idx, r_options);

	if (p_idx == 0) {
		List<StringName> sn;
		String pf = p_function;
		if (pf == "add_theme_color_override" || pf == "has_theme_color" || pf == "has_theme_color_override" || pf == "get_theme_color") {
			ThemeDB::get_singleton()->get_default_theme()->get_color_list(get_class(), &sn);
		} else if (pf == "add_theme_style_override" || pf == "has_theme_style" || pf == "has_theme_style_override" || pf == "get_theme_style") {
			ThemeDB::get_singleton()->get_default_theme()->get_stylebox_list(get_class(), &sn);
		} else if (pf == "add_theme_font_override" || pf == "has_theme_font" || pf == "has_theme_font_override" || pf == "get_theme_font") {
			ThemeDB::get_singleton()->get_default_theme()->get_font_list(get_class(), &sn);
		} else if (pf == "add_theme_font_size_override" || pf == "has_theme_font_size" || pf == "has_theme_font_size_override" || pf == "get_theme_font_size") {
			ThemeDB::get_singleton()->get_default_theme()->get_font_size_list(get_class(), &sn);
		} else if (pf == "add_theme_constant_override" || pf == "has_theme_constant" || pf == "has_theme_constant_override" || pf == "get_theme_constant") {
			ThemeDB::get_singleton()->get_default_theme()->get_constant_list(get_class(), &sn);
		}

		sn.sort_custom<StringName::AlphCompare>();
		for (const StringName &name : sn) {
			r_options->push_back(String(name).quote());
		}
	}
}

TypedArray<String> Control::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (data.mouse_filter == MOUSE_FILTER_IGNORE && !data.tooltip.is_empty()) {
		warnings.push_back(RTR("The Hint Tooltip won't be displayed as the control's Mouse Filter is set to \"Ignore\". To solve this, set the Mouse Filter to \"Stop\" or \"Pass\"."));
	}

	return warnings;
}

bool Control::is_text_field() const {
	return false;
}

// Dynamic properties.

String Control::properties_managed_by_container[] = {
	"offset_left",
	"offset_top",
	"offset_right",
	"offset_bottom",
	"anchor_left",
	"anchor_top",
	"anchor_right",
	"anchor_bottom",
	"position",
	"rotation",
	"scale",
	"size"
};

bool Control::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (!name.begins_with("theme_override")) {
		return false;
	}

	if (p_value.get_type() == Variant::NIL || (p_value.get_type() == Variant::OBJECT && (Object *)p_value == nullptr)) {
		if (name.begins_with("theme_override_icons/")) {
			String dname = name.get_slicec('/', 1);
			if (data.icon_override.has(dname)) {
				data.icon_override[dname]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
			}
			data.icon_override.erase(dname);
			_notify_theme_override_changed();
		} else if (name.begins_with("theme_override_styles/")) {
			String dname = name.get_slicec('/', 1);
			if (data.style_override.has(dname)) {
				data.style_override[dname]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
			}
			data.style_override.erase(dname);
			_notify_theme_override_changed();
		} else if (name.begins_with("theme_override_fonts/")) {
			String dname = name.get_slicec('/', 1);
			if (data.font_override.has(dname)) {
				data.font_override[dname]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
			}
			data.font_override.erase(dname);
			_notify_theme_override_changed();
		} else if (name.begins_with("theme_override_font_sizes/")) {
			String dname = name.get_slicec('/', 1);
			data.font_size_override.erase(dname);
			_notify_theme_override_changed();
		} else if (name.begins_with("theme_override_colors/")) {
			String dname = name.get_slicec('/', 1);
			data.color_override.erase(dname);
			_notify_theme_override_changed();
		} else if (name.begins_with("theme_override_constants/")) {
			String dname = name.get_slicec('/', 1);
			data.constant_override.erase(dname);
			_notify_theme_override_changed();
		} else {
			return false;
		}

	} else {
		if (name.begins_with("theme_override_icons/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_icon_override(dname, p_value);
		} else if (name.begins_with("theme_override_styles/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_style_override(dname, p_value);
		} else if (name.begins_with("theme_override_fonts/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_font_override(dname, p_value);
		} else if (name.begins_with("theme_override_font_sizes/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_font_size_override(dname, p_value);
		} else if (name.begins_with("theme_override_colors/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_color_override(dname, p_value);
		} else if (name.begins_with("theme_override_constants/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_constant_override(dname, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool Control::_get(const StringName &p_name, Variant &r_ret) const {
	String sname = p_name;
	if (!sname.begins_with("theme_override")) {
		return false;
	}

	if (sname.begins_with("theme_override_icons/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.icon_override.has(name) ? Variant(data.icon_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_styles/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.style_override.has(name) ? Variant(data.style_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_fonts/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.font_override.has(name) ? Variant(data.font_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_font_sizes/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.font_size_override.has(name) ? Variant(data.font_size_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_colors/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.color_override.has(name) ? Variant(data.color_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_constants/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.constant_override.has(name) ? Variant(data.constant_override[name]) : Variant();
	} else {
		return false;
	}

	return true;
}

void Control::_get_property_list(List<PropertyInfo> *p_list) const {
	Ref<Theme> theme = ThemeDB::get_singleton()->get_default_theme();

	p_list->push_back(PropertyInfo(Variant::NIL, TTRC("Theme Overrides"), PROPERTY_HINT_NONE, "theme_override_", PROPERTY_USAGE_GROUP));

	{
		List<StringName> names;
		theme->get_color_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.color_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::COLOR, "theme_override_colors/" + E, PROPERTY_HINT_NONE, "", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_constant_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.constant_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::INT, "theme_override_constants/" + E, PROPERTY_HINT_RANGE, "-16384,16384", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_font_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.font_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::OBJECT, "theme_override_fonts/" + E, PROPERTY_HINT_RESOURCE_TYPE, "Font", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_font_size_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.font_size_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::INT, "theme_override_font_sizes/" + E, PROPERTY_HINT_RANGE, "1,256,1,or_greater,suffix:px", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_icon_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.icon_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::OBJECT, "theme_override_icons/" + E, PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_stylebox_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.style_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::OBJECT, "theme_override_styles/" + E, PROPERTY_HINT_RESOURCE_TYPE, "StyleBox", usage));
		}
	}
}

void Control::_validate_property(PropertyInfo &p_property) const {
	// Update theme type variation options.
	if (p_property.name == "theme_type_variation") {
		List<StringName> names;

		// Only the default theme and the project theme are used for the list of options.
		// This is an imposed limitation to simplify the logic needed to leverage those options.
		ThemeDB::get_singleton()->get_default_theme()->get_type_variation_list(get_class_name(), &names);
		if (ThemeDB::get_singleton()->get_project_theme().is_valid()) {
			ThemeDB::get_singleton()->get_project_theme()->get_type_variation_list(get_class_name(), &names);
		}
		names.sort_custom<StringName::AlphCompare>();

		Vector<StringName> unique_names;
		String hint_string;
		for (const StringName &E : names) {
			// Skip duplicate values.
			if (unique_names.has(E)) {
				continue;
			}

			hint_string += String(E) + ",";
			unique_names.append(E);
		}

		p_property.hint_string = hint_string;
	}

	if (p_property.name == "mouse_force_pass_scroll_events") {
		// Disable force pass if the control is not stopping the event.
		if (data.mouse_filter != MOUSE_FILTER_STOP) {
			p_property.usage |= PROPERTY_USAGE_READ_ONLY;
		}
	}

	if (p_property.name == "scale") {
		p_property.hint = PROPERTY_HINT_LINK;
	}

	// Validate which positioning properties should be displayed depending on the parent and the layout mode.
	Node *parent_node = get_parent_control();
	if (!parent_node) {
		// If there is no parent, display both anchor and container options.

		// Set the layout mode to be disabled with the proper value.
		if (p_property.name == "layout_mode") {
			p_property.hint_string = "Position,Anchors,Container,Uncontrolled";
			p_property.usage |= PROPERTY_USAGE_READ_ONLY;
		}

		// Use the layout mode to display or hide advanced anchoring properties.
		bool use_custom_anchors = _get_anchors_layout_preset() == -1; // Custom "preset".
		if (!use_custom_anchors && (p_property.name.begins_with("anchor_") || p_property.name.begins_with("offset_") || p_property.name.begins_with("grow_"))) {
			p_property.usage ^= PROPERTY_USAGE_EDITOR;
		}
	} else if (Object::cast_to<Container>(parent_node)) {
		// If the parent is a container, display only container-related properties.
		if (p_property.name.begins_with("anchor_") || p_property.name.begins_with("offset_") || p_property.name.begins_with("grow_") || p_property.name == "anchors_preset" ||
				p_property.name == "position" || p_property.name == "rotation" || p_property.name == "scale" || p_property.name == "size" || p_property.name == "pivot_offset") {
			p_property.usage ^= PROPERTY_USAGE_EDITOR;

		} else if (p_property.name == "layout_mode") {
			// Set the layout mode to be disabled with the proper value.
			p_property.hint_string = "Position,Anchors,Container,Uncontrolled";
			p_property.usage |= PROPERTY_USAGE_READ_ONLY;
		} else if (p_property.name == "size_flags_horizontal" || p_property.name == "size_flags_vertical") {
			// Filter allowed size flags based on the parent container configuration.
			Container *parent_container = Object::cast_to<Container>(parent_node);
			Vector<int> size_flags;
			if (p_property.name == "size_flags_horizontal") {
				size_flags = parent_container->get_allowed_size_flags_horizontal();
			} else if (p_property.name == "size_flags_vertical") {
				size_flags = parent_container->get_allowed_size_flags_vertical();
			}

			// Enforce the order of the options, regardless of what the container provided.
			String hint_string;
			if (size_flags.has(SIZE_FILL)) {
				hint_string += "Fill:1";
			}
			if (size_flags.has(SIZE_EXPAND)) {
				if (!hint_string.is_empty()) {
					hint_string += ",";
				}
				hint_string += "Expand:2";
			}
			if (size_flags.has(SIZE_SHRINK_CENTER)) {
				if (!hint_string.is_empty()) {
					hint_string += ",";
				}
				hint_string += "Shrink Center:4";
			}
			if (size_flags.has(SIZE_SHRINK_END)) {
				if (!hint_string.is_empty()) {
					hint_string += ",";
				}
				hint_string += "Shrink End:8";
			}

			if (hint_string.is_empty()) {
				p_property.hint_string = "";
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			} else {
				p_property.hint_string = hint_string;
			}
		}
	} else {
		// If the parent is NOT a container or not a control at all, display only anchoring-related properties.
		if (p_property.name.begins_with("size_flags_")) {
			p_property.usage ^= PROPERTY_USAGE_EDITOR;

		} else if (p_property.name == "layout_mode") {
			// Set the layout mode to be enabled with proper options.
			p_property.hint_string = "Position,Anchors";
		}

		// Use the layout mode to display or hide advanced anchoring properties.
		bool use_anchors = _get_layout_mode() == LayoutMode::LAYOUT_MODE_ANCHORS;
		if (!use_anchors && p_property.name == "anchors_preset") {
			p_property.usage ^= PROPERTY_USAGE_EDITOR;
		}
		bool use_custom_anchors = use_anchors && _get_anchors_layout_preset() == -1; // Custom "preset".
		if (!use_custom_anchors && (p_property.name.begins_with("anchor_") || p_property.name.begins_with("offset_") || p_property.name.begins_with("grow_"))) {
			p_property.usage ^= PROPERTY_USAGE_EDITOR;
		}
	}

	// Disable the property if it's managed by the parent container.
	if (!Object::cast_to<Container>(parent_node)) {
		return;
	}
	bool property_is_managed_by_container = false;
	for (unsigned i = 0; i < properties_managed_by_container_count; i++) {
		property_is_managed_by_container = properties_managed_by_container[i] == p_property.name;
		if (property_is_managed_by_container) {
			break;
		}
	}
	if (property_is_managed_by_container) {
		p_property.usage |= PROPERTY_USAGE_READ_ONLY;
	}
}

bool Control::_property_can_revert(const StringName &p_name) const {
	if (p_name == "layout_mode" || p_name == "anchors_preset") {
		return true;
	}

	return false;
}

bool Control::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	if (p_name == "layout_mode") {
		r_property = _get_default_layout_mode();
		return true;
	} else if (p_name == "anchors_preset") {
		r_property = LayoutPreset::PRESET_TOP_LEFT;
		return true;
	}

	return false;
}

// Global relations.

bool Control::is_top_level_control() const {
	return is_inside_tree() && (!data.parent_canvas_item && !data.RI && is_set_as_top_level());
}

Control *Control::get_parent_control() const {
	return data.parent;
}

Window *Control::get_parent_window() const {
	return data.parent_window;
}

Control *Control::get_root_parent_control() const {
	const CanvasItem *ci = this;
	const Control *root = this;

	while (ci) {
		const Control *c = Object::cast_to<Control>(ci);
		if (c) {
			root = c;

			if (c->data.RI || c->is_top_level_control()) {
				break;
			}
		}

		ci = ci->get_parent_item();
	}

	return const_cast<Control *>(root);
}

Rect2 Control::get_parent_anchorable_rect() const {
	if (!is_inside_tree()) {
		return Rect2();
	}

	Rect2 parent_rect;
	if (data.parent_canvas_item) {
		parent_rect = data.parent_canvas_item->get_anchorable_rect();
	} else {
#ifdef TOOLS_ENABLED
		Node *edited_root = get_tree()->get_edited_scene_root();
		if (edited_root && (this == edited_root || edited_root->is_ancestor_of(this))) {
			parent_rect.size = Size2(ProjectSettings::get_singleton()->get("display/window/size/viewport_width"), ProjectSettings::get_singleton()->get("display/window/size/viewport_height"));
		} else {
			parent_rect = get_viewport()->get_visible_rect();
		}

#else
		parent_rect = get_viewport()->get_visible_rect();
#endif
	}

	return parent_rect;
}

Size2 Control::get_parent_area_size() const {
	return get_parent_anchorable_rect().size;
}

// Positioning and sizing.

Transform2D Control::_get_internal_transform() const {
	Transform2D rot_scale;
	rot_scale.set_rotation_and_scale(data.rotation, data.scale);
	Transform2D offset;
	offset.set_origin(-data.pivot_offset);

	return offset.affine_inverse() * (rot_scale * offset);
}

void Control::_update_canvas_item_transform() {
	Transform2D xform = _get_internal_transform();
	xform[2] += get_position();

	// We use a little workaround to avoid flickering when moving the pivot with _edit_set_pivot()
	if (is_inside_tree() && Math::abs(Math::sin(data.rotation * 4.0f)) < 0.00001f && get_viewport()->is_snap_controls_to_pixels_enabled()) {
		xform[2] = xform[2].round();
	}

	RenderingServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), xform);
}

Transform2D Control::get_transform() const {
	Transform2D xform = _get_internal_transform();
	xform[2] += get_position();
	return xform;
}

/// Anchors and offsets.

void Control::_set_anchor(Side p_side, real_t p_anchor) {
	set_anchor(p_side, p_anchor);
}

void Control::set_anchor(Side p_side, real_t p_anchor, bool p_keep_offset, bool p_push_opposite_anchor) {
	ERR_FAIL_INDEX((int)p_side, 4);

	Rect2 parent_rect = get_parent_anchorable_rect();
	real_t parent_range = (p_side == SIDE_LEFT || p_side == SIDE_RIGHT) ? parent_rect.size.x : parent_rect.size.y;
	real_t previous_pos = data.offset[p_side] + data.anchor[p_side] * parent_range;
	real_t previous_opposite_pos = data.offset[(p_side + 2) % 4] + data.anchor[(p_side + 2) % 4] * parent_range;

	data.anchor[p_side] = p_anchor;

	if (((p_side == SIDE_LEFT || p_side == SIDE_TOP) && data.anchor[p_side] > data.anchor[(p_side + 2) % 4]) ||
			((p_side == SIDE_RIGHT || p_side == SIDE_BOTTOM) && data.anchor[p_side] < data.anchor[(p_side + 2) % 4])) {
		if (p_push_opposite_anchor) {
			data.anchor[(p_side + 2) % 4] = data.anchor[p_side];
		} else {
			data.anchor[p_side] = data.anchor[(p_side + 2) % 4];
		}
	}

	if (!p_keep_offset) {
		data.offset[p_side] = previous_pos - data.anchor[p_side] * parent_range;
		if (p_push_opposite_anchor) {
			data.offset[(p_side + 2) % 4] = previous_opposite_pos - data.anchor[(p_side + 2) % 4] * parent_range;
		}
	}
	if (is_inside_tree()) {
		_size_changed();
	}

	queue_redraw();
}

real_t Control::get_anchor(Side p_side) const {
	ERR_FAIL_INDEX_V(int(p_side), 4, 0.0);

	return data.anchor[p_side];
}

void Control::set_offset(Side p_side, real_t p_value) {
	ERR_FAIL_INDEX((int)p_side, 4);
	if (data.offset[p_side] == p_value) {
		return;
	}

	data.offset[p_side] = p_value;
	_size_changed();
}

real_t Control::get_offset(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);

	return data.offset[p_side];
}

void Control::set_anchor_and_offset(Side p_side, real_t p_anchor, real_t p_pos, bool p_push_opposite_anchor) {
	set_anchor(p_side, p_anchor, false, p_push_opposite_anchor);
	set_offset(p_side, p_pos);
}

void Control::set_begin(const Size2 &p_point) {
	if (data.offset[0] == p_point.x && data.offset[1] == p_point.y) {
		return;
	}

	data.offset[0] = p_point.x;
	data.offset[1] = p_point.y;
	_size_changed();
}

Size2 Control::get_begin() const {
	return Size2(data.offset[0], data.offset[1]);
}

void Control::set_end(const Size2 &p_point) {
	if (data.offset[2] == p_point.x && data.offset[3] == p_point.y) {
		return;
	}

	data.offset[2] = p_point.x;
	data.offset[3] = p_point.y;
	_size_changed();
}

Size2 Control::get_end() const {
	return Size2(data.offset[2], data.offset[3]);
}

void Control::set_h_grow_direction(GrowDirection p_direction) {
	if (data.h_grow == p_direction) {
		return;
	}

	ERR_FAIL_INDEX((int)p_direction, 3);

	data.h_grow = p_direction;
	_size_changed();
}

Control::GrowDirection Control::get_h_grow_direction() const {
	return data.h_grow;
}

void Control::set_v_grow_direction(GrowDirection p_direction) {
	if (data.v_grow == p_direction) {
		return;
	}

	ERR_FAIL_INDEX((int)p_direction, 3);

	data.v_grow = p_direction;
	_size_changed();
}

Control::GrowDirection Control::get_v_grow_direction() const {
	return data.v_grow;
}

void Control::_compute_anchors(Rect2 p_rect, const real_t p_offsets[4], real_t (&r_anchors)[4]) {
	Size2 parent_rect_size = get_parent_anchorable_rect().size;
	ERR_FAIL_COND(parent_rect_size.x == 0.0);
	ERR_FAIL_COND(parent_rect_size.y == 0.0);

	real_t x = p_rect.position.x;
	if (is_layout_rtl()) {
		x = parent_rect_size.x - x - p_rect.size.x;
	}
	r_anchors[0] = (x - p_offsets[0]) / parent_rect_size.x;
	r_anchors[1] = (p_rect.position.y - p_offsets[1]) / parent_rect_size.y;
	r_anchors[2] = (x + p_rect.size.x - p_offsets[2]) / parent_rect_size.x;
	r_anchors[3] = (p_rect.position.y + p_rect.size.y - p_offsets[3]) / parent_rect_size.y;
}

void Control::_compute_offsets(Rect2 p_rect, const real_t p_anchors[4], real_t (&r_offsets)[4]) {
	Size2 parent_rect_size = get_parent_anchorable_rect().size;

	real_t x = p_rect.position.x;
	if (is_layout_rtl()) {
		x = parent_rect_size.x - x - p_rect.size.x;
	}
	r_offsets[0] = x - (p_anchors[0] * parent_rect_size.x);
	r_offsets[1] = p_rect.position.y - (p_anchors[1] * parent_rect_size.y);
	r_offsets[2] = x + p_rect.size.x - (p_anchors[2] * parent_rect_size.x);
	r_offsets[3] = p_rect.position.y + p_rect.size.y - (p_anchors[3] * parent_rect_size.y);
}

/// Presets and layout modes.

void Control::_set_layout_mode(LayoutMode p_mode) {
	bool list_changed = false;

	if (data.stored_layout_mode != p_mode) {
		list_changed = true;
		data.stored_layout_mode = p_mode;
	}

	if (data.stored_layout_mode == LayoutMode::LAYOUT_MODE_POSITION) {
		data.stored_use_custom_anchors = false;
		set_anchors_and_offsets_preset(LayoutPreset::PRESET_TOP_LEFT, LayoutPresetMode::PRESET_MODE_KEEP_SIZE);
		set_grow_direction_preset(LayoutPreset::PRESET_TOP_LEFT);
	}

	if (list_changed) {
		notify_property_list_changed();
	}
}

Control::LayoutMode Control::_get_layout_mode() const {
	Node *parent_node = get_parent_control();
	// In these modes the property is read-only.
	if (!parent_node) {
		return LayoutMode::LAYOUT_MODE_UNCONTROLLED;
	} else if (Object::cast_to<Container>(parent_node)) {
		return LayoutMode::LAYOUT_MODE_CONTAINER;
	}

	// If anchors are not in the top-left position, this is definitely in anchors mode.
	if (_get_anchors_layout_preset() != (int)LayoutPreset::PRESET_TOP_LEFT) {
		return LayoutMode::LAYOUT_MODE_ANCHORS;
	}

	// Otherwise fallback on what's stored.
	return data.stored_layout_mode;
}

Control::LayoutMode Control::_get_default_layout_mode() const {
	Node *parent_node = get_parent_control();
	// In these modes the property is read-only.
	if (!parent_node) {
		return LayoutMode::LAYOUT_MODE_UNCONTROLLED;
	} else if (Object::cast_to<Container>(parent_node)) {
		return LayoutMode::LAYOUT_MODE_CONTAINER;
	}

	// Otherwise fallback on the position mode.
	return LayoutMode::LAYOUT_MODE_POSITION;
}

void Control::_set_anchors_layout_preset(int p_preset) {
	bool list_changed = false;

	if (data.stored_layout_mode != LayoutMode::LAYOUT_MODE_ANCHORS) {
		list_changed = true;
		data.stored_layout_mode = LayoutMode::LAYOUT_MODE_ANCHORS;
	}

	if (p_preset == -1) {
		if (!data.stored_use_custom_anchors) {
			data.stored_use_custom_anchors = true;
			notify_property_list_changed();
		}
		return; // Keep settings as is.
	}

	if (data.stored_use_custom_anchors) {
		list_changed = true;
		data.stored_use_custom_anchors = false;
	}

	LayoutPreset preset = (LayoutPreset)p_preset;
	// Set correct anchors.
	set_anchors_preset(preset);

	// Select correct preset mode.
	switch (preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_TOP:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
			set_offsets_preset(preset, LayoutPresetMode::PRESET_MODE_KEEP_SIZE);
			break;
		case PRESET_LEFT_WIDE:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_offsets_preset(preset, LayoutPresetMode::PRESET_MODE_MINSIZE);
			break;
	}

	// Select correct grow directions.
	set_grow_direction_preset(preset);

	if (list_changed) {
		notify_property_list_changed();
	}
}

int Control::_get_anchors_layout_preset() const {
	// If the custom preset was selected by user, use it.
	if (data.stored_use_custom_anchors) {
		return -1;
	}

	// Check anchors to determine if the current state matches a preset, or not.

	float left = get_anchor(SIDE_LEFT);
	float right = get_anchor(SIDE_RIGHT);
	float top = get_anchor(SIDE_TOP);
	float bottom = get_anchor(SIDE_BOTTOM);

	if (left == ANCHOR_BEGIN && right == ANCHOR_BEGIN && top == ANCHOR_BEGIN && bottom == ANCHOR_BEGIN) {
		return (int)LayoutPreset::PRESET_TOP_LEFT;
	}
	if (left == ANCHOR_END && right == ANCHOR_END && top == ANCHOR_BEGIN && bottom == ANCHOR_BEGIN) {
		return (int)LayoutPreset::PRESET_TOP_RIGHT;
	}
	if (left == ANCHOR_BEGIN && right == ANCHOR_BEGIN && top == ANCHOR_END && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_BOTTOM_LEFT;
	}
	if (left == ANCHOR_END && right == ANCHOR_END && top == ANCHOR_END && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_BOTTOM_RIGHT;
	}

	if (left == ANCHOR_BEGIN && right == ANCHOR_BEGIN && top == 0.5 && bottom == 0.5) {
		return (int)LayoutPreset::PRESET_CENTER_LEFT;
	}
	if (left == ANCHOR_END && right == ANCHOR_END && top == 0.5 && bottom == 0.5) {
		return (int)LayoutPreset::PRESET_CENTER_RIGHT;
	}
	if (left == 0.5 && right == 0.5 && top == ANCHOR_BEGIN && bottom == ANCHOR_BEGIN) {
		return (int)LayoutPreset::PRESET_CENTER_TOP;
	}
	if (left == 0.5 && right == 0.5 && top == ANCHOR_END && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_CENTER_BOTTOM;
	}
	if (left == 0.5 && right == 0.5 && top == 0.5 && bottom == 0.5) {
		return (int)LayoutPreset::PRESET_CENTER;
	}

	if (left == ANCHOR_BEGIN && right == ANCHOR_BEGIN && top == ANCHOR_BEGIN && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_LEFT_WIDE;
	}
	if (left == ANCHOR_END && right == ANCHOR_END && top == ANCHOR_BEGIN && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_RIGHT_WIDE;
	}
	if (left == ANCHOR_BEGIN && right == ANCHOR_END && top == ANCHOR_BEGIN && bottom == ANCHOR_BEGIN) {
		return (int)LayoutPreset::PRESET_TOP_WIDE;
	}
	if (left == ANCHOR_BEGIN && right == ANCHOR_END && top == ANCHOR_END && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_BOTTOM_WIDE;
	}

	if (left == 0.5 && right == 0.5 && top == ANCHOR_BEGIN && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_VCENTER_WIDE;
	}
	if (left == ANCHOR_BEGIN && right == ANCHOR_END && top == 0.5 && bottom == 0.5) {
		return (int)LayoutPreset::PRESET_HCENTER_WIDE;
	}

	if (left == ANCHOR_BEGIN && right == ANCHOR_END && top == ANCHOR_BEGIN && bottom == ANCHOR_END) {
		return (int)LayoutPreset::PRESET_FULL_RECT;
	}

	// Does not match any preset, return "Custom".
	return -1;
}

void Control::set_anchors_preset(LayoutPreset p_preset, bool p_keep_offsets) {
	ERR_FAIL_INDEX((int)p_preset, 16);

	//Left
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_TOP_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_LEFT_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_anchor(SIDE_LEFT, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			set_anchor(SIDE_LEFT, 0.5, p_keep_offsets);
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			set_anchor(SIDE_LEFT, ANCHOR_END, p_keep_offsets);
			break;
	}

	// Top
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_TOP_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_anchor(SIDE_TOP, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			set_anchor(SIDE_TOP, 0.5, p_keep_offsets);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			set_anchor(SIDE_TOP, ANCHOR_END, p_keep_offsets);
			break;
	}

	// Right
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			set_anchor(SIDE_RIGHT, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			set_anchor(SIDE_RIGHT, 0.5, p_keep_offsets);
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_anchor(SIDE_RIGHT, ANCHOR_END, p_keep_offsets);
			break;
	}

	// Bottom
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			set_anchor(SIDE_BOTTOM, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			set_anchor(SIDE_BOTTOM, 0.5, p_keep_offsets);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_anchor(SIDE_BOTTOM, ANCHOR_END, p_keep_offsets);
			break;
	}
}

void Control::set_offsets_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode, int p_margin) {
	ERR_FAIL_INDEX((int)p_preset, 16);
	ERR_FAIL_INDEX((int)p_resize_mode, 4);

	// Calculate the size if the node is not resized
	Size2 min_size = get_minimum_size();
	Size2 new_size = get_size();
	if (p_resize_mode == PRESET_MODE_MINSIZE || p_resize_mode == PRESET_MODE_KEEP_HEIGHT) {
		new_size.x = min_size.x;
	}
	if (p_resize_mode == PRESET_MODE_MINSIZE || p_resize_mode == PRESET_MODE_KEEP_WIDTH) {
		new_size.y = min_size.y;
	}

	Rect2 parent_rect = get_parent_anchorable_rect();

	real_t x = parent_rect.size.x;
	if (is_layout_rtl()) {
		x = parent_rect.size.x - x - new_size.x;
	}
	//Left
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_TOP_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_LEFT_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			data.offset[0] = x * (0.0 - data.anchor[0]) + p_margin + parent_rect.position.x;
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			data.offset[0] = x * (0.5 - data.anchor[0]) - new_size.x / 2 + parent_rect.position.x;
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			data.offset[0] = x * (1.0 - data.anchor[0]) - new_size.x - p_margin + parent_rect.position.x;
			break;
	}

	// Top
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_TOP_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_FULL_RECT:
			data.offset[1] = parent_rect.size.y * (0.0 - data.anchor[1]) + p_margin + parent_rect.position.y;
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			data.offset[1] = parent_rect.size.y * (0.5 - data.anchor[1]) - new_size.y / 2 + parent_rect.position.y;
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			data.offset[1] = parent_rect.size.y * (1.0 - data.anchor[1]) - new_size.y - p_margin + parent_rect.position.y;
			break;
	}

	// Right
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			data.offset[2] = x * (0.0 - data.anchor[2]) + new_size.x + p_margin + parent_rect.position.x;
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			data.offset[2] = x * (0.5 - data.anchor[2]) + new_size.x / 2 + parent_rect.position.x;
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			data.offset[2] = x * (1.0 - data.anchor[2]) - p_margin + parent_rect.position.x;
			break;
	}

	// Bottom
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			data.offset[3] = parent_rect.size.y * (0.0 - data.anchor[3]) + new_size.y + p_margin + parent_rect.position.y;
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			data.offset[3] = parent_rect.size.y * (0.5 - data.anchor[3]) + new_size.y / 2 + parent_rect.position.y;
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_FULL_RECT:
			data.offset[3] = parent_rect.size.y * (1.0 - data.anchor[3]) - p_margin + parent_rect.position.y;
			break;
	}

	_size_changed();
}

void Control::set_anchors_and_offsets_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode, int p_margin) {
	set_anchors_preset(p_preset);
	set_offsets_preset(p_preset, p_resize_mode, p_margin);
}

void Control::set_grow_direction_preset(LayoutPreset p_preset) {
	// Select correct horizontal grow direction.
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			set_h_grow_direction(GrowDirection::GROW_DIRECTION_END);
			break;
		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			set_h_grow_direction(GrowDirection::GROW_DIRECTION_BEGIN);
			break;
		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_TOP_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_h_grow_direction(GrowDirection::GROW_DIRECTION_BOTH);
			break;
	}

	// Select correct vertical grow direction.
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			set_v_grow_direction(GrowDirection::GROW_DIRECTION_END);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			set_v_grow_direction(GrowDirection::GROW_DIRECTION_BEGIN);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_FULL_RECT:
			set_v_grow_direction(GrowDirection::GROW_DIRECTION_BOTH);
			break;
	}
}

/// Manual positioning.

void Control::_set_position(const Size2 &p_point) {
	set_position(p_point);
}

void Control::set_position(const Size2 &p_point, bool p_keep_offsets) {
	if (p_keep_offsets) {
		_compute_anchors(Rect2(p_point, data.size_cache), data.offset, data.anchor);
	} else {
		_compute_offsets(Rect2(p_point, data.size_cache), data.anchor, data.offset);
	}
	_size_changed();
}

Size2 Control::get_position() const {
	return data.pos_cache;
}

void Control::_set_global_position(const Point2 &p_point) {
	set_global_position(p_point);
}

void Control::set_global_position(const Point2 &p_point, bool p_keep_offsets) {
	Transform2D inv;

	if (data.parent_canvas_item) {
		inv = data.parent_canvas_item->get_global_transform().affine_inverse();
	}

	set_position(inv.xform(p_point), p_keep_offsets);
}

Point2 Control::get_global_position() const {
	return get_global_transform().get_origin();
}

Point2 Control::get_screen_position() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Point2());
	Point2 global_pos = get_global_transform_with_canvas().get_origin();
	Window *w = Object::cast_to<Window>(get_viewport());
	if (w && !w->is_embedding_subwindows()) {
		global_pos += w->get_position();
	}

	return global_pos;
}

void Control::_set_size(const Size2 &p_size) {
#ifdef DEBUG_ENABLED
	if (data.size_warning && (data.anchor[SIDE_LEFT] != data.anchor[SIDE_RIGHT] || data.anchor[SIDE_TOP] != data.anchor[SIDE_BOTTOM])) {
		WARN_PRINT("Nodes with non-equal opposite anchors will have their size overridden after _ready(). \nIf you want to set size, change the anchors or consider using set_deferred().");
	}
#endif
	set_size(p_size);
}

void Control::set_size(const Size2 &p_size, bool p_keep_offsets) {
	Size2 new_size = p_size;
	Size2 min = get_combined_minimum_size();
	if (new_size.x < min.x) {
		new_size.x = min.x;
	}
	if (new_size.y < min.y) {
		new_size.y = min.y;
	}

	if (p_keep_offsets) {
		_compute_anchors(Rect2(data.pos_cache, new_size), data.offset, data.anchor);
	} else {
		_compute_offsets(Rect2(data.pos_cache, new_size), data.anchor, data.offset);
	}
	_size_changed();
}

Size2 Control::get_size() const {
	return data.size_cache;
}

void Control::reset_size() {
	set_size(Size2());
}

void Control::set_rect(const Rect2 &p_rect) {
	for (int i = 0; i < 4; i++) {
		data.anchor[i] = ANCHOR_BEGIN;
	}

	_compute_offsets(p_rect, data.anchor, data.offset);
	if (is_inside_tree()) {
		_size_changed();
	}
}

Rect2 Control::get_rect() const {
	return Rect2(get_position(), get_size());
}

Rect2 Control::get_global_rect() const {
	return Rect2(get_global_position(), get_size());
}

Rect2 Control::get_screen_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());

	Rect2 r(get_global_position(), get_size());

	Window *w = Object::cast_to<Window>(get_viewport());
	if (w && !w->is_embedding_subwindows()) {
		r.position += w->get_position();
	}

	return r;
}

Rect2 Control::get_window_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	Rect2 gr = get_global_rect();
	gr.position += get_viewport()->get_visible_rect().position;
	return gr;
}

Rect2 Control::get_anchorable_rect() const {
	return Rect2(Point2(), get_size());
}

void Control::set_scale(const Vector2 &p_scale) {
	if (data.scale == p_scale) {
		return;
	}

	data.scale = p_scale;
	// Avoid having 0 scale values, can lead to errors in physics and rendering.
	if (data.scale.x == 0) {
		data.scale.x = CMP_EPSILON;
	}
	if (data.scale.y == 0) {
		data.scale.y = CMP_EPSILON;
	}
	queue_redraw();
	_notify_transform();
}

Vector2 Control::get_scale() const {
	return data.scale;
}

void Control::set_rotation(real_t p_radians) {
	if (data.rotation == p_radians) {
		return;
	}

	data.rotation = p_radians;
	queue_redraw();
	_notify_transform();
}

real_t Control::get_rotation() const {
	return data.rotation;
}

void Control::set_pivot_offset(const Vector2 &p_pivot) {
	if (data.pivot_offset == p_pivot) {
		return;
	}

	data.pivot_offset = p_pivot;
	queue_redraw();
	_notify_transform();
}

Vector2 Control::get_pivot_offset() const {
	return data.pivot_offset;
}

/// Sizes.

void Control::_update_minimum_size() {
	if (!is_inside_tree()) {
		return;
	}

	Size2 minsize = get_combined_minimum_size();
	data.updating_last_minimum_size = false;

	if (minsize != data.last_minimum_size) {
		data.last_minimum_size = minsize;
		_size_changed();
		emit_signal(SceneStringNames::get_singleton()->minimum_size_changed);
	}
}

void Control::update_minimum_size() {
	if (!is_inside_tree() || data.block_minimum_size_adjust) {
		return;
	}

	Control *invalidate = this;

	//invalidate cache upwards
	while (invalidate && invalidate->data.minimum_size_valid) {
		invalidate->data.minimum_size_valid = false;
		if (invalidate->is_set_as_top_level()) {
			break; // do not go further up
		}
		if (!invalidate->data.parent && get_parent()) {
			Window *parent_window = Object::cast_to<Window>(get_parent());
			if (parent_window && parent_window->is_wrapping_controls()) {
				parent_window->child_controls_changed();
			}
		}
		invalidate = invalidate->data.parent;
	}

	if (!is_visible_in_tree()) {
		return;
	}

	if (data.updating_last_minimum_size) {
		return;
	}

	data.updating_last_minimum_size = true;

	MessageQueue::get_singleton()->push_call(this, "_update_minimum_size");
}

void Control::set_block_minimum_size_adjust(bool p_block) {
	data.block_minimum_size_adjust = p_block;
}

bool Control::is_minimum_size_adjust_blocked() const {
	return data.block_minimum_size_adjust;
}

Size2 Control::get_minimum_size() const {
	Vector2 ms;
	if (GDVIRTUAL_CALL(_get_minimum_size, ms)) {
		return ms;
	}
	return Vector2();
}

void Control::set_custom_minimum_size(const Size2 &p_custom) {
	if (p_custom == data.custom_minimum_size) {
		return;
	}
	data.custom_minimum_size = p_custom;
	update_minimum_size();
}

Size2 Control::get_custom_minimum_size() const {
	return data.custom_minimum_size;
}

void Control::_update_minimum_size_cache() {
	Size2 minsize = get_minimum_size();
	minsize.x = MAX(minsize.x, data.custom_minimum_size.x);
	minsize.y = MAX(minsize.y, data.custom_minimum_size.y);

	bool size_changed = false;
	if (data.minimum_size_cache != minsize) {
		size_changed = true;
	}

	data.minimum_size_cache = minsize;
	data.minimum_size_valid = true;

	if (size_changed) {
		update_minimum_size();
	}
}

Size2 Control::get_combined_minimum_size() const {
	if (!data.minimum_size_valid) {
		const_cast<Control *>(this)->_update_minimum_size_cache();
	}
	return data.minimum_size_cache;
}

void Control::_size_changed() {
	Rect2 parent_rect = get_parent_anchorable_rect();

	real_t edge_pos[4];

	for (int i = 0; i < 4; i++) {
		real_t area = parent_rect.size[i & 1];
		edge_pos[i] = data.offset[i] + (data.anchor[i] * area);
	}

	Point2 new_pos_cache = Point2(edge_pos[0], edge_pos[1]);
	Size2 new_size_cache = Point2(edge_pos[2], edge_pos[3]) - new_pos_cache;

	Size2 minimum_size = get_combined_minimum_size();

	if (minimum_size.width > new_size_cache.width) {
		if (data.h_grow == GROW_DIRECTION_BEGIN) {
			new_pos_cache.x += new_size_cache.width - minimum_size.width;
		} else if (data.h_grow == GROW_DIRECTION_BOTH) {
			new_pos_cache.x += 0.5 * (new_size_cache.width - minimum_size.width);
		}

		new_size_cache.width = minimum_size.width;
	}

	if (is_layout_rtl()) {
		new_pos_cache.x = parent_rect.size.x - new_pos_cache.x - new_size_cache.x;
	}

	if (minimum_size.height > new_size_cache.height) {
		if (data.v_grow == GROW_DIRECTION_BEGIN) {
			new_pos_cache.y += new_size_cache.height - minimum_size.height;
		} else if (data.v_grow == GROW_DIRECTION_BOTH) {
			new_pos_cache.y += 0.5 * (new_size_cache.height - minimum_size.height);
		}

		new_size_cache.height = minimum_size.height;
	}

	bool pos_changed = new_pos_cache != data.pos_cache;
	bool size_changed = new_size_cache != data.size_cache;

	data.pos_cache = new_pos_cache;
	data.size_cache = new_size_cache;

	if (is_inside_tree()) {
		if (size_changed) {
			notification(NOTIFICATION_RESIZED);
		}
		if (pos_changed || size_changed) {
			item_rect_changed(size_changed);
			_notify_transform();
		}

		if (pos_changed && !size_changed) {
			_update_canvas_item_transform(); //move because it won't be updated
		}
	}
}

void Control::_clear_size_warning() {
	data.size_warning = false;
}

// Container sizing.

void Control::set_h_size_flags(int p_flags) {
	if (data.h_size_flags == p_flags) {
		return;
	}
	data.h_size_flags = p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

int Control::get_h_size_flags() const {
	return data.h_size_flags;
}

void Control::set_v_size_flags(int p_flags) {
	if (data.v_size_flags == p_flags) {
		return;
	}
	data.v_size_flags = p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

int Control::get_v_size_flags() const {
	return data.v_size_flags;
}

void Control::set_stretch_ratio(real_t p_ratio) {
	if (data.expand == p_ratio) {
		return;
	}

	data.expand = p_ratio;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

real_t Control::get_stretch_ratio() const {
	return data.expand;
}

// Input events.

void Control::_call_gui_input(const Ref<InputEvent> &p_event) {
	emit_signal(SceneStringNames::get_singleton()->gui_input, p_event); //signal should be first, so it's possible to override an event (and then accept it)
	if (!is_inside_tree() || get_viewport()->is_input_handled()) {
		return; //input was handled, abort
	}
	GDVIRTUAL_CALL(_gui_input, p_event);
	if (!is_inside_tree() || get_viewport()->is_input_handled()) {
		return; //input was handled, abort
	}
	gui_input(p_event);
}

void Control::gui_input(const Ref<InputEvent> &p_event) {
}

void Control::accept_event() {
	if (is_inside_tree()) {
		get_viewport()->_gui_accept_event();
	}
}

bool Control::has_point(const Point2 &p_point) const {
	bool ret;
	if (GDVIRTUAL_CALL(_has_point, p_point, ret)) {
		return ret;
	}
	return Rect2(Point2(), get_size()).has_point(p_point);
}

void Control::set_mouse_filter(MouseFilter p_filter) {
	ERR_FAIL_INDEX(p_filter, 3);
	data.mouse_filter = p_filter;
	notify_property_list_changed();
	update_configuration_warnings();
}

Control::MouseFilter Control::get_mouse_filter() const {
	return data.mouse_filter;
}

void Control::set_force_pass_scroll_events(bool p_force_pass_scroll_events) {
	data.force_pass_scroll_events = p_force_pass_scroll_events;
}

bool Control::is_force_pass_scroll_events() const {
	return data.force_pass_scroll_events;
}

void Control::warp_mouse(const Point2 &p_position) {
	ERR_FAIL_COND(!is_inside_tree());
	get_viewport()->warp_mouse(get_global_transform_with_canvas().xform(p_position));
}

// Drag and drop handling.

void Control::set_drag_forwarding(Object *p_target) {
	if (p_target) {
		data.drag_owner = p_target->get_instance_id();
	} else {
		data.drag_owner = ObjectID();
	}
}

Variant Control::get_drag_data(const Point2 &p_point) {
	if (data.drag_owner.is_valid()) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			return obj->call("_get_drag_data_fw", p_point, this);
		}
	}

	Variant dd;
	if (GDVIRTUAL_CALL(_get_drag_data, p_point, dd)) {
		return dd;
	}

	return Variant();
}

bool Control::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	if (data.drag_owner.is_valid()) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			return obj->call("_can_drop_data_fw", p_point, p_data, this);
		}
	}

	bool ret;
	if (GDVIRTUAL_CALL(_can_drop_data, p_point, p_data, ret)) {
		return ret;
	}
	return false;
}

void Control::drop_data(const Point2 &p_point, const Variant &p_data) {
	if (data.drag_owner.is_valid()) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			obj->call("_drop_data_fw", p_point, p_data, this);
			return;
		}
	}

	GDVIRTUAL_CALL(_drop_data, p_point, p_data);
}

void Control::force_drag(const Variant &p_data, Control *p_control) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(p_data.get_type() == Variant::NIL);

	get_viewport()->_gui_force_drag(this, p_data, p_control);
}

void Control::set_drag_preview(Control *p_control) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!get_viewport()->gui_is_dragging());
	get_viewport()->_gui_set_drag_preview(this, p_control);
}

bool Control::is_drag_successful() const {
	return is_inside_tree() && get_viewport()->gui_is_drag_successful();
}

// Focus.

void Control::set_focus_mode(FocusMode p_focus_mode) {
	ERR_FAIL_INDEX((int)p_focus_mode, 3);

	if (is_inside_tree() && p_focus_mode == FOCUS_NONE && data.focus_mode != FOCUS_NONE && has_focus()) {
		release_focus();
	}

	data.focus_mode = p_focus_mode;
}

Control::FocusMode Control::get_focus_mode() const {
	return data.focus_mode;
}

bool Control::has_focus() const {
	return is_inside_tree() && get_viewport()->_gui_control_has_focus(this);
}

void Control::grab_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	if (data.focus_mode == FOCUS_NONE) {
		WARN_PRINT("This control can't grab focus. Use set_focus_mode() to allow a control to get focus.");
		return;
	}

	get_viewport()->_gui_control_grab_focus(this);
}

void Control::grab_click_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	get_viewport()->_gui_grab_click_focus(this);
}

void Control::release_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	if (!has_focus()) {
		return;
	}

	get_viewport()->gui_release_focus();
}

static Control *_next_control(Control *p_from) {
	if (p_from->is_set_as_top_level()) {
		return nullptr; // Can't go above.
	}

	Control *parent = Object::cast_to<Control>(p_from->get_parent());

	if (!parent) {
		return nullptr;
	}

	int next = p_from->get_index();
	ERR_FAIL_INDEX_V(next, parent->get_child_count(), nullptr);
	for (int i = (next + 1); i < parent->get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(parent->get_child(i));
		if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
			continue;
		}

		return c;
	}

	// No next in parent, try the same in parent.
	return _next_control(parent);
}

Control *Control::find_next_valid_focus() const {
	Control *from = const_cast<Control *>(this);

	while (true) {
		// If the focus property is manually overwritten, attempt to use it.

		if (!data.focus_next.is_empty()) {
			Node *n = get_node(data.focus_next);
			Control *c;
			if (n) {
				c = Object::cast_to<Control>(n);
				ERR_FAIL_COND_V_MSG(!c, nullptr, "Next focus node is not a control: " + n->get_name() + ".");
			} else {
				return nullptr;
			}
			if (c->is_visible() && c->get_focus_mode() != FOCUS_NONE) {
				return c;
			}
		}

		// Find next child.

		Control *next_child = nullptr;

		for (int i = 0; i < from->get_child_count(); i++) {
			Control *c = Object::cast_to<Control>(from->get_child(i));
			if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
				continue;
			}

			next_child = c;
			break;
		}

		if (!next_child) {
			next_child = _next_control(from);
			if (!next_child) { // Nothing else. Go up and find either window or subwindow.
				next_child = const_cast<Control *>(this);
				while (next_child && !next_child->is_set_as_top_level()) {
					next_child = cast_to<Control>(next_child->get_parent());
				}

				if (!next_child) {
					next_child = const_cast<Control *>(this);
					while (next_child) {
						if (next_child->data.RI) {
							break;
						}
						next_child = next_child->get_parent_control();
					}
				}
			}
		}

		if (next_child == from || next_child == this) { // No next control.
			return (get_focus_mode() == FOCUS_ALL) ? next_child : nullptr;
		}
		if (next_child) {
			if (next_child->get_focus_mode() == FOCUS_ALL) {
				return next_child;
			}
			from = next_child;
		} else {
			break;
		}
	}

	return nullptr;
}

static Control *_prev_control(Control *p_from) {
	Control *child = nullptr;
	for (int i = p_from->get_child_count() - 1; i >= 0; i--) {
		Control *c = Object::cast_to<Control>(p_from->get_child(i));
		if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
			continue;
		}

		child = c;
		break;
	}

	if (!child) {
		return p_from;
	}

	// No prev in parent, try the same in parent.
	return _prev_control(child);
}

Control *Control::find_prev_valid_focus() const {
	Control *from = const_cast<Control *>(this);

	while (true) {
		// If the focus property is manually overwritten, attempt to use it.

		if (!data.focus_prev.is_empty()) {
			Node *n = get_node(data.focus_prev);
			Control *c;
			if (n) {
				c = Object::cast_to<Control>(n);
				ERR_FAIL_COND_V_MSG(!c, nullptr, "Previous focus node is not a control: " + n->get_name() + ".");
			} else {
				return nullptr;
			}
			if (c->is_visible() && c->get_focus_mode() != FOCUS_NONE) {
				return c;
			}
		}

		// Find prev child.

		Control *prev_child = nullptr;

		if (from->is_set_as_top_level() || !Object::cast_to<Control>(from->get_parent())) {
			// Find last of the children.

			prev_child = _prev_control(from);

		} else {
			for (int i = (from->get_index() - 1); i >= 0; i--) {
				Control *c = Object::cast_to<Control>(from->get_parent()->get_child(i));

				if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
					continue;
				}

				prev_child = c;
				break;
			}

			if (!prev_child) {
				prev_child = Object::cast_to<Control>(from->get_parent());
			} else {
				prev_child = _prev_control(prev_child);
			}
		}

		if (prev_child == from || prev_child == this) { // No prev control.
			return (get_focus_mode() == FOCUS_ALL) ? prev_child : nullptr;
		}

		if (prev_child->get_focus_mode() == FOCUS_ALL) {
			return prev_child;
		}

		from = prev_child;
	}

	return nullptr;
}

void Control::set_focus_neighbor(Side p_side, const NodePath &p_neighbor) {
	ERR_FAIL_INDEX((int)p_side, 4);
	data.focus_neighbor[p_side] = p_neighbor;
}

NodePath Control::get_focus_neighbor(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, NodePath());
	return data.focus_neighbor[p_side];
}

void Control::set_focus_next(const NodePath &p_next) {
	data.focus_next = p_next;
}

NodePath Control::get_focus_next() const {
	return data.focus_next;
}

void Control::set_focus_previous(const NodePath &p_prev) {
	data.focus_prev = p_prev;
}

NodePath Control::get_focus_previous() const {
	return data.focus_prev;
}

#define MAX_NEIGHBOR_SEARCH_COUNT 512

Control *Control::_get_focus_neighbor(Side p_side, int p_count) {
	ERR_FAIL_INDEX_V((int)p_side, 4, nullptr);

	if (p_count >= MAX_NEIGHBOR_SEARCH_COUNT) {
		return nullptr;
	}
	if (!data.focus_neighbor[p_side].is_empty()) {
		Control *c = nullptr;
		Node *n = get_node(data.focus_neighbor[p_side]);
		if (n) {
			c = Object::cast_to<Control>(n);
			ERR_FAIL_COND_V_MSG(!c, nullptr, "Neighbor focus node is not a control: " + n->get_name() + ".");
		} else {
			return nullptr;
		}
		bool valid = true;
		if (!c->is_visible()) {
			valid = false;
		}
		if (c->get_focus_mode() == FOCUS_NONE) {
			valid = false;
		}
		if (valid) {
			return c;
		}

		c = c->_get_focus_neighbor(p_side, p_count + 1);
		return c;
	}

	real_t dist = 1e7;
	Control *result = nullptr;

	Point2 points[4];

	Transform2D xform = get_global_transform();

	points[0] = xform.xform(Point2());
	points[1] = xform.xform(Point2(get_size().x, 0));
	points[2] = xform.xform(get_size());
	points[3] = xform.xform(Point2(0, get_size().y));

	const Vector2 dir[4] = {
		Vector2(-1, 0),
		Vector2(0, -1),
		Vector2(1, 0),
		Vector2(0, 1)
	};

	Vector2 vdir = dir[p_side];

	real_t maxd = -1e7;

	for (int i = 0; i < 4; i++) {
		real_t d = vdir.dot(points[i]);
		if (d > maxd) {
			maxd = d;
		}
	}

	Node *base = this;

	while (base) {
		Control *c = Object::cast_to<Control>(base);
		if (c) {
			if (c->data.RI) {
				break;
			}
		}
		base = base->get_parent();
	}

	if (!base) {
		return nullptr;
	}

	_window_find_focus_neighbor(vdir, base, points, maxd, dist, &result);

	return result;
}

void Control::_window_find_focus_neighbor(const Vector2 &p_dir, Node *p_at, const Point2 *p_points, real_t p_min, real_t &r_closest_dist, Control **r_closest) {
	if (Object::cast_to<Viewport>(p_at)) {
		return; //bye
	}

	Control *c = Object::cast_to<Control>(p_at);

	if (c && c != this && c->get_focus_mode() == FOCUS_ALL && c->is_visible_in_tree()) {
		Point2 points[4];

		Transform2D xform = c->get_global_transform();

		points[0] = xform.xform(Point2());
		points[1] = xform.xform(Point2(c->get_size().x, 0));
		points[2] = xform.xform(c->get_size());
		points[3] = xform.xform(Point2(0, c->get_size().y));

		real_t min = 1e7;

		for (int i = 0; i < 4; i++) {
			real_t d = p_dir.dot(points[i]);
			if (d < min) {
				min = d;
			}
		}

		if (min > (p_min - CMP_EPSILON)) {
			for (int i = 0; i < 4; i++) {
				Vector2 la = p_points[i];
				Vector2 lb = p_points[(i + 1) % 4];

				for (int j = 0; j < 4; j++) {
					Vector2 fa = points[j];
					Vector2 fb = points[(j + 1) % 4];

					Vector2 pa, pb;
					real_t d = Geometry2D::get_closest_points_between_segments(la, lb, fa, fb, pa, pb);
					//real_t d = Geometry2D::get_closest_distance_between_segments(Vector3(la.x,la.y,0),Vector3(lb.x,lb.y,0),Vector3(fa.x,fa.y,0),Vector3(fb.x,fb.y,0));
					if (d < r_closest_dist) {
						r_closest_dist = d;
						*r_closest = c;
					}
				}
			}
		}
	}

	for (int i = 0; i < p_at->get_child_count(); i++) {
		Node *child = p_at->get_child(i);
		Control *childc = Object::cast_to<Control>(child);
		if (childc && childc->data.RI) {
			continue; //subwindow, ignore
		}
		_window_find_focus_neighbor(p_dir, p_at->get_child(i), p_points, p_min, r_closest_dist, r_closest);
	}
}

// Rendering.

void Control::set_default_cursor_shape(CursorShape p_shape) {
	ERR_FAIL_INDEX(int(p_shape), CURSOR_MAX);

	data.default_cursor = p_shape;
}

Control::CursorShape Control::get_default_cursor_shape() const {
	return data.default_cursor;
}

Control::CursorShape Control::get_cursor_shape(const Point2 &p_pos) const {
	return data.default_cursor;
}

void Control::set_disable_visibility_clip(bool p_ignore) {
	if (data.disable_visibility_clip == p_ignore) {
		return;
	}
	data.disable_visibility_clip = p_ignore;
	queue_redraw();
}

bool Control::is_visibility_clip_disabled() const {
	return data.disable_visibility_clip;
}

void Control::set_clip_contents(bool p_clip) {
	if (data.clip_contents == p_clip) {
		return;
	}
	data.clip_contents = p_clip;
	queue_redraw();
}

bool Control::is_clipping_contents() {
	return data.clip_contents;
}

// Theming.

void Control::_propagate_theme_changed(Node *p_at, Control *p_owner, Window *p_owner_window, bool p_notify, bool p_assign) {
	Control *c = Object::cast_to<Control>(p_at);
	Window *w = c == nullptr ? Object::cast_to<Window>(p_at) : nullptr;

	if (!c && !w) {
		// Theme inheritance chains are broken by nodes that aren't Control or Window.
		return;
	}

	bool assign = p_assign;
	if (c) {
		if (c != p_owner && c->data.theme.is_valid()) {
			// Has a theme, so we don't want to change the theme owner,
			// but we still want to propagate in case this child has theme items
			// it inherits from the theme this node uses.
			// See https://github.com/godotengine/godot/issues/62844.
			assign = false;
		}

		if (assign) {
			c->data.theme_owner = p_owner;
			c->data.theme_owner_window = p_owner_window;
		}

		if (p_notify) {
			c->notification(Control::NOTIFICATION_THEME_CHANGED);
		}
	} else if (w) {
		if (w != p_owner_window && w->theme.is_valid()) {
			// Same as above.
			assign = false;
		}

		if (assign) {
			w->theme_owner = p_owner;
			w->theme_owner_window = p_owner_window;
		}

		if (p_notify) {
			w->notification(Window::NOTIFICATION_THEME_CHANGED);
		}
	}

	for (int i = 0; i < p_at->get_child_count(); i++) {
		_propagate_theme_changed(p_at->get_child(i), p_owner, p_owner_window, p_notify, assign);
	}
}

void Control::_theme_changed() {
	if (is_inside_tree()) {
		_propagate_theme_changed(this, this, nullptr, true, false);
	}
}

void Control::_notify_theme_override_changed() {
	if (!data.bulk_theme_override && is_inside_tree()) {
		notification(NOTIFICATION_THEME_CHANGED);
	}
}

void Control::_invalidate_theme_cache() {
	data.theme_icon_cache.clear();
	data.theme_style_cache.clear();
	data.theme_font_cache.clear();
	data.theme_font_size_cache.clear();
	data.theme_color_cache.clear();
	data.theme_constant_cache.clear();
}

void Control::_update_theme_item_cache() {
}

void Control::set_theme(const Ref<Theme> &p_theme) {
	if (data.theme == p_theme) {
		return;
	}

	if (data.theme.is_valid()) {
		data.theme->disconnect("changed", callable_mp(this, &Control::_theme_changed));
	}

	data.theme = p_theme;
	if (data.theme.is_valid()) {
		_propagate_theme_changed(this, this, nullptr, is_inside_tree(), true);
		data.theme->connect("changed", callable_mp(this, &Control::_theme_changed), CONNECT_DEFERRED);
		return;
	}

	Control *parent_c = Object::cast_to<Control>(get_parent());
	if (parent_c && (parent_c->data.theme_owner || parent_c->data.theme_owner_window)) {
		_propagate_theme_changed(this, parent_c->data.theme_owner, parent_c->data.theme_owner_window, is_inside_tree(), true);
		return;
	}

	Window *parent_w = cast_to<Window>(get_parent());
	if (parent_w && (parent_w->theme_owner || parent_w->theme_owner_window)) {
		_propagate_theme_changed(this, parent_w->theme_owner, parent_w->theme_owner_window, is_inside_tree(), true);
		return;
	}

	_propagate_theme_changed(this, nullptr, nullptr, is_inside_tree(), true);
}

Ref<Theme> Control::get_theme() const {
	return data.theme;
}

void Control::set_theme_type_variation(const StringName &p_theme_type) {
	if (data.theme_type_variation == p_theme_type) {
		return;
	}
	data.theme_type_variation = p_theme_type;
	if (is_inside_tree()) {
		notification(NOTIFICATION_THEME_CHANGED);
	}
}

StringName Control::get_theme_type_variation() const {
	return data.theme_type_variation;
}

/// Theme property lookup.

template <class T>
T Control::get_theme_item_in_types(Control *p_theme_owner, Window *p_theme_owner_window, Theme::DataType p_data_type, const StringName &p_name, List<StringName> p_theme_types) {
	ERR_FAIL_COND_V_MSG(p_theme_types.size() == 0, T(), "At least one theme type must be specified.");

	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		// For each theme resource check the theme types provided and see if p_name exists with any of them.
		for (const StringName &E : p_theme_types) {
			if (theme_owner && theme_owner->data.theme->has_theme_item(p_data_type, p_name, E)) {
				return theme_owner->data.theme->get_theme_item(p_data_type, p_name, E);
			}

			if (theme_owner_window && theme_owner_window->theme->has_theme_item(p_data_type, p_name, E)) {
				return theme_owner_window->theme->get_theme_item(p_data_type, p_name, E);
			}
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (ThemeDB::get_singleton()->get_project_theme().is_valid()) {
		for (const StringName &E : p_theme_types) {
			if (ThemeDB::get_singleton()->get_project_theme()->has_theme_item(p_data_type, p_name, E)) {
				return ThemeDB::get_singleton()->get_project_theme()->get_theme_item(p_data_type, p_name, E);
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	for (const StringName &E : p_theme_types) {
		if (ThemeDB::get_singleton()->get_default_theme()->has_theme_item(p_data_type, p_name, E)) {
			return ThemeDB::get_singleton()->get_default_theme()->get_theme_item(p_data_type, p_name, E);
		}
	}
	// If they don't exist, use any type to return the default/empty value.
	return ThemeDB::get_singleton()->get_default_theme()->get_theme_item(p_data_type, p_name, p_theme_types[0]);
}

bool Control::has_theme_item_in_types(Control *p_theme_owner, Window *p_theme_owner_window, Theme::DataType p_data_type, const StringName &p_name, List<StringName> p_theme_types) {
	ERR_FAIL_COND_V_MSG(p_theme_types.size() == 0, false, "At least one theme type must be specified.");

	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		// For each theme resource check the theme types provided and see if p_name exists with any of them.
		for (const StringName &E : p_theme_types) {
			if (theme_owner && theme_owner->data.theme->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}

			if (theme_owner_window && theme_owner_window->theme->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (ThemeDB::get_singleton()->get_project_theme().is_valid()) {
		for (const StringName &E : p_theme_types) {
			if (ThemeDB::get_singleton()->get_project_theme()->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	for (const StringName &E : p_theme_types) {
		if (ThemeDB::get_singleton()->get_default_theme()->has_theme_item(p_data_type, p_name, E)) {
			return true;
		}
	}
	return false;
}

void Control::_get_theme_type_dependencies(const StringName &p_theme_type, List<StringName> *p_list) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (ThemeDB::get_singleton()->get_project_theme().is_valid() && ThemeDB::get_singleton()->get_project_theme()->get_type_variation_base(data.theme_type_variation) != StringName()) {
			ThemeDB::get_singleton()->get_project_theme()->get_type_dependencies(get_class_name(), data.theme_type_variation, p_list);
		} else {
			ThemeDB::get_singleton()->get_default_theme()->get_type_dependencies(get_class_name(), data.theme_type_variation, p_list);
		}
	} else {
		ThemeDB::get_singleton()->get_default_theme()->get_type_dependencies(p_theme_type, StringName(), p_list);
	}
}

Ref<Texture2D> Control::get_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Ref<Texture2D> *tex = data.icon_override.getptr(p_name);
		if (tex) {
			return *tex;
		}
	}

	if (data.theme_icon_cache.has(p_theme_type) && data.theme_icon_cache[p_theme_type].has(p_name)) {
		return data.theme_icon_cache[p_theme_type][p_name];
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	Ref<Texture2D> icon = get_theme_item_in_types<Ref<Texture2D>>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_ICON, p_name, theme_types);
	data.theme_icon_cache[p_theme_type][p_name] = icon;
	return icon;
}

Ref<StyleBox> Control::get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Ref<StyleBox> *style = data.style_override.getptr(p_name);
		if (style) {
			return *style;
		}
	}

	if (data.theme_style_cache.has(p_theme_type) && data.theme_style_cache[p_theme_type].has(p_name)) {
		return data.theme_style_cache[p_theme_type][p_name];
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	Ref<StyleBox> style = get_theme_item_in_types<Ref<StyleBox>>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_STYLEBOX, p_name, theme_types);
	data.theme_style_cache[p_theme_type][p_name] = style;
	return style;
}

Ref<Font> Control::get_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Ref<Font> *font = data.font_override.getptr(p_name);
		if (font) {
			return *font;
		}
	}

	if (data.theme_font_cache.has(p_theme_type) && data.theme_font_cache[p_theme_type].has(p_name)) {
		return data.theme_font_cache[p_theme_type][p_name];
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	Ref<Font> font = get_theme_item_in_types<Ref<Font>>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT, p_name, theme_types);
	data.theme_font_cache[p_theme_type][p_name] = font;
	return font;
}

int Control::get_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const int *font_size = data.font_size_override.getptr(p_name);
		if (font_size && (*font_size) > 0) {
			return *font_size;
		}
	}

	if (data.theme_font_size_cache.has(p_theme_type) && data.theme_font_size_cache[p_theme_type].has(p_name)) {
		return data.theme_font_size_cache[p_theme_type][p_name];
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	int font_size = get_theme_item_in_types<int>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT_SIZE, p_name, theme_types);
	data.theme_font_size_cache[p_theme_type][p_name] = font_size;
	return font_size;
}

Color Control::get_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Color *color = data.color_override.getptr(p_name);
		if (color) {
			return *color;
		}
	}

	if (data.theme_color_cache.has(p_theme_type) && data.theme_color_cache[p_theme_type].has(p_name)) {
		return data.theme_color_cache[p_theme_type][p_name];
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	Color color = get_theme_item_in_types<Color>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_COLOR, p_name, theme_types);
	data.theme_color_cache[p_theme_type][p_name] = color;
	return color;
}

int Control::get_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const int *constant = data.constant_override.getptr(p_name);
		if (constant) {
			return *constant;
		}
	}

	if (data.theme_constant_cache.has(p_theme_type) && data.theme_constant_cache[p_theme_type].has(p_name)) {
		return data.theme_constant_cache[p_theme_type][p_name];
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	int constant = get_theme_item_in_types<int>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_CONSTANT, p_name, theme_types);
	data.theme_constant_cache[p_theme_type][p_name] = constant;
	return constant;
}

bool Control::has_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_icon_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_ICON, p_name, theme_types);
}

bool Control::has_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_stylebox_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_STYLEBOX, p_name, theme_types);
}

bool Control::has_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_font_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT, p_name, theme_types);
}

bool Control::has_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_font_size_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT_SIZE, p_name, theme_types);
}

bool Control::has_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_color_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_COLOR, p_name, theme_types);
}

bool Control::has_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_constant_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_CONSTANT, p_name, theme_types);
}

/// Local property overrides.

void Control::add_theme_icon_override(const StringName &p_name, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_COND(!p_icon.is_valid());

	if (data.icon_override.has(p_name)) {
		data.icon_override[p_name]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	data.icon_override[p_name] = p_icon;
	data.icon_override[p_name]->connect("changed", callable_mp(this, &Control::_notify_theme_override_changed), CONNECT_REFERENCE_COUNTED);
	_notify_theme_override_changed();
}

void Control::add_theme_style_override(const StringName &p_name, const Ref<StyleBox> &p_style) {
	ERR_FAIL_COND(!p_style.is_valid());

	if (data.style_override.has(p_name)) {
		data.style_override[p_name]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	data.style_override[p_name] = p_style;
	data.style_override[p_name]->connect("changed", callable_mp(this, &Control::_notify_theme_override_changed), CONNECT_REFERENCE_COUNTED);
	_notify_theme_override_changed();
}

void Control::add_theme_font_override(const StringName &p_name, const Ref<Font> &p_font) {
	ERR_FAIL_COND(!p_font.is_valid());

	if (data.font_override.has(p_name)) {
		data.font_override[p_name]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	data.font_override[p_name] = p_font;
	data.font_override[p_name]->connect("changed", callable_mp(this, &Control::_notify_theme_override_changed), CONNECT_REFERENCE_COUNTED);
	_notify_theme_override_changed();
}

void Control::add_theme_font_size_override(const StringName &p_name, int p_font_size) {
	data.font_size_override[p_name] = p_font_size;
	_notify_theme_override_changed();
}

void Control::add_theme_color_override(const StringName &p_name, const Color &p_color) {
	data.color_override[p_name] = p_color;
	_notify_theme_override_changed();
}

void Control::add_theme_constant_override(const StringName &p_name, int p_constant) {
	data.constant_override[p_name] = p_constant;
	_notify_theme_override_changed();
}

void Control::remove_theme_icon_override(const StringName &p_name) {
	if (data.icon_override.has(p_name)) {
		data.icon_override[p_name]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	data.icon_override.erase(p_name);
	_notify_theme_override_changed();
}

void Control::remove_theme_style_override(const StringName &p_name) {
	if (data.style_override.has(p_name)) {
		data.style_override[p_name]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	data.style_override.erase(p_name);
	_notify_theme_override_changed();
}

void Control::remove_theme_font_override(const StringName &p_name) {
	if (data.font_override.has(p_name)) {
		data.font_override[p_name]->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	data.font_override.erase(p_name);
	_notify_theme_override_changed();
}

void Control::remove_theme_font_size_override(const StringName &p_name) {
	data.font_size_override.erase(p_name);
	_notify_theme_override_changed();
}

void Control::remove_theme_color_override(const StringName &p_name) {
	data.color_override.erase(p_name);
	_notify_theme_override_changed();
}

void Control::remove_theme_constant_override(const StringName &p_name) {
	data.constant_override.erase(p_name);
	_notify_theme_override_changed();
}

bool Control::has_theme_icon_override(const StringName &p_name) const {
	const Ref<Texture2D> *tex = data.icon_override.getptr(p_name);
	return tex != nullptr;
}

bool Control::has_theme_stylebox_override(const StringName &p_name) const {
	const Ref<StyleBox> *style = data.style_override.getptr(p_name);
	return style != nullptr;
}

bool Control::has_theme_font_override(const StringName &p_name) const {
	const Ref<Font> *font = data.font_override.getptr(p_name);
	return font != nullptr;
}

bool Control::has_theme_font_size_override(const StringName &p_name) const {
	const int *font_size = data.font_size_override.getptr(p_name);
	return font_size != nullptr;
}

bool Control::has_theme_color_override(const StringName &p_name) const {
	const Color *color = data.color_override.getptr(p_name);
	return color != nullptr;
}

bool Control::has_theme_constant_override(const StringName &p_name) const {
	const int *constant = data.constant_override.getptr(p_name);
	return constant != nullptr;
}

/// Default theme properties.

float Control::fetch_theme_default_base_scale(Control *p_theme_owner, Window *p_theme_owner_window) {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		if (theme_owner && theme_owner->data.theme->has_default_base_scale()) {
			return theme_owner->data.theme->get_default_base_scale();
		}

		if (theme_owner_window && theme_owner_window->theme->has_default_base_scale()) {
			return theme_owner_window->theme->get_default_base_scale();
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (ThemeDB::get_singleton()->get_project_theme().is_valid()) {
		if (ThemeDB::get_singleton()->get_project_theme()->has_default_base_scale()) {
			return ThemeDB::get_singleton()->get_project_theme()->get_default_base_scale();
		}
	}

	// Lastly, fall back on the default Theme.
	if (ThemeDB::get_singleton()->get_default_theme()->has_default_base_scale()) {
		return ThemeDB::get_singleton()->get_default_theme()->get_default_base_scale();
	}
	return ThemeDB::get_singleton()->get_fallback_base_scale();
}

float Control::get_theme_default_base_scale() const {
	return fetch_theme_default_base_scale(data.theme_owner, data.theme_owner_window);
}

Ref<Font> Control::fetch_theme_default_font(Control *p_theme_owner, Window *p_theme_owner_window) {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		if (theme_owner && theme_owner->data.theme->has_default_font()) {
			return theme_owner->data.theme->get_default_font();
		}

		if (theme_owner_window && theme_owner_window->theme->has_default_font()) {
			return theme_owner_window->theme->get_default_font();
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (ThemeDB::get_singleton()->get_project_theme().is_valid()) {
		if (ThemeDB::get_singleton()->get_project_theme()->has_default_font()) {
			return ThemeDB::get_singleton()->get_project_theme()->get_default_font();
		}
	}

	// Lastly, fall back on the default Theme.
	if (ThemeDB::get_singleton()->get_default_theme()->has_default_font()) {
		return ThemeDB::get_singleton()->get_default_theme()->get_default_font();
	}
	return ThemeDB::get_singleton()->get_fallback_font();
}

Ref<Font> Control::get_theme_default_font() const {
	return fetch_theme_default_font(data.theme_owner, data.theme_owner_window);
}

int Control::fetch_theme_default_font_size(Control *p_theme_owner, Window *p_theme_owner_window) {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		if (theme_owner && theme_owner->data.theme->has_default_font_size()) {
			return theme_owner->data.theme->get_default_font_size();
		}

		if (theme_owner_window && theme_owner_window->theme->has_default_font_size()) {
			return theme_owner_window->theme->get_default_font_size();
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (ThemeDB::get_singleton()->get_project_theme().is_valid()) {
		if (ThemeDB::get_singleton()->get_project_theme()->has_default_font_size()) {
			return ThemeDB::get_singleton()->get_project_theme()->get_default_font_size();
		}
	}

	// Lastly, fall back on the default Theme.
	if (ThemeDB::get_singleton()->get_default_theme()->has_default_font_size()) {
		return ThemeDB::get_singleton()->get_default_theme()->get_default_font_size();
	}
	return ThemeDB::get_singleton()->get_fallback_font_size();
}

int Control::get_theme_default_font_size() const {
	return fetch_theme_default_font_size(data.theme_owner, data.theme_owner_window);
}

/// Bulk actions.

void Control::begin_bulk_theme_override() {
	data.bulk_theme_override = true;
}

void Control::end_bulk_theme_override() {
	ERR_FAIL_COND(!data.bulk_theme_override);

	data.bulk_theme_override = false;
	_notify_theme_override_changed();
}

// Internationalization.

TypedArray<Vector2i> Control::structured_text_parser(TextServer::StructuredTextParser p_parser_type, const Array &p_args, const String &p_text) const {
	if (p_parser_type == TextServer::STRUCTURED_TEXT_CUSTOM) {
		TypedArray<Vector2i> ret;
		if (GDVIRTUAL_CALL(_structured_text_parser, p_args, p_text, ret)) {
			return ret;
		} else {
			return TypedArray<Vector2i>();
		}
	} else {
		return TS->parse_structured_text(p_parser_type, p_args, p_text);
	}
}

void Control::set_layout_direction(Control::LayoutDirection p_direction) {
	if (data.layout_dir == p_direction) {
		return;
	}
	ERR_FAIL_INDEX((int)p_direction, 4);

	data.layout_dir = p_direction;
	data.is_rtl_dirty = true;

	propagate_notification(NOTIFICATION_LAYOUT_DIRECTION_CHANGED);
}

Control::LayoutDirection Control::get_layout_direction() const {
	return data.layout_dir;
}

bool Control::is_layout_rtl() const {
	if (data.is_rtl_dirty) {
		const_cast<Control *>(this)->data.is_rtl_dirty = false;
		if (data.layout_dir == LAYOUT_DIRECTION_INHERITED) {
			Window *parent_window = get_parent_window();
			Control *parent_control = get_parent_control();
			if (parent_control) {
				const_cast<Control *>(this)->data.is_rtl = parent_control->is_layout_rtl();
			} else if (parent_window) {
				const_cast<Control *>(this)->data.is_rtl = parent_window->is_layout_rtl();
			} else {
				if (GLOBAL_GET(SNAME("internationalization/rendering/force_right_to_left_layout_direction"))) {
					const_cast<Control *>(this)->data.is_rtl = true;
				} else {
					String locale = TranslationServer::get_singleton()->get_tool_locale();
					const_cast<Control *>(this)->data.is_rtl = TS->is_locale_right_to_left(locale);
				}
			}
		} else if (data.layout_dir == LAYOUT_DIRECTION_LOCALE) {
			if (GLOBAL_GET(SNAME("internationalization/rendering/force_right_to_left_layout_direction"))) {
				const_cast<Control *>(this)->data.is_rtl = true;
			} else {
				String locale = TranslationServer::get_singleton()->get_tool_locale();
				const_cast<Control *>(this)->data.is_rtl = TS->is_locale_right_to_left(locale);
			}
		} else {
			const_cast<Control *>(this)->data.is_rtl = (data.layout_dir == LAYOUT_DIRECTION_RTL);
		}
	}
	return data.is_rtl;
}

void Control::set_auto_translate(bool p_enable) {
	if (p_enable == data.auto_translate) {
		return;
	}

	data.auto_translate = p_enable;

	notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
}

bool Control::is_auto_translating() const {
	return data.auto_translate;
}

// Extra properties.

void Control::set_tooltip_text(const String &p_hint) {
	data.tooltip = p_hint;
	update_configuration_warnings();
}

String Control::get_tooltip_text() const {
	return data.tooltip;
}

String Control::get_tooltip(const Point2 &p_pos) const {
	return data.tooltip;
}

Control *Control::make_custom_tooltip(const String &p_text) const {
	Object *ret = nullptr;
	if (GDVIRTUAL_CALL(_make_custom_tooltip, p_text, ret)) {
		return Object::cast_to<Control>(ret);
	}
	return nullptr;
}

// Base object overrides.

void Control::add_child_notify(Node *p_child) {
	// We propagate when this node uses a custom theme, so it can pass it on to its children.
	if (data.theme_owner || data.theme_owner_window) {
		// `p_notify` is false here as `NOTIFICATION_THEME_CHANGED` will be handled by `NOTIFICATION_ENTER_TREE`.
		_propagate_theme_changed(p_child, data.theme_owner, data.theme_owner_window, false, true);
	}
}

void Control::remove_child_notify(Node *p_child) {
	// If the removed child isn't inheriting any theme items through this node, then there's no need to propagate.
	if (data.theme_owner || data.theme_owner_window) {
		_propagate_theme_changed(p_child, nullptr, nullptr, false, true);
	}
}

void Control::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_POSTINITIALIZE: {
			_invalidate_theme_cache();
			_update_theme_item_cache();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			// Need to defer here, because theme owner information might be set in
			// add_child_notify, which doesn't get called until right after this.
			call_deferred(SNAME("notification"), NOTIFICATION_THEME_CHANGED);
		} break;

		case NOTIFICATION_POST_ENTER_TREE: {
			data.minimum_size_valid = false;
			data.is_rtl_dirty = true;
			_size_changed();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			release_focus();
			get_viewport()->_gui_remove_control(this);
		} break;

		case NOTIFICATION_READY: {
#ifdef DEBUG_ENABLED
			connect("ready", callable_mp(this, &Control::_clear_size_warning), CONNECT_DEFERRED | CONNECT_ONESHOT);
#endif
		} break;

		case NOTIFICATION_ENTER_CANVAS: {
			data.parent = Object::cast_to<Control>(get_parent());
			data.parent_window = Object::cast_to<Window>(get_parent());
			data.is_rtl_dirty = true;

			CanvasItem *node = this;
			bool has_parent_control = false;

			while (!node->is_set_as_top_level()) {
				CanvasItem *parent = Object::cast_to<CanvasItem>(node->get_parent());
				if (!parent) {
					break;
				}

				Control *parent_control = Object::cast_to<Control>(parent);
				if (parent_control) {
					has_parent_control = true;
					break;
				}

				node = parent;
			}

			if (has_parent_control) {
				// Do nothing, has a parent control.
			} else {
				// Is a regular root control or top_level.
				Viewport *viewport = get_viewport();
				ERR_FAIL_COND(!viewport);
				data.RI = viewport->_gui_add_root_control(this);
			}

			data.parent_canvas_item = get_parent_item();

			if (data.parent_canvas_item) {
				data.parent_canvas_item->connect("item_rect_changed", callable_mp(this, &Control::_size_changed));
			} else {
				// Connect viewport.
				Viewport *viewport = get_viewport();
				ERR_FAIL_COND(!viewport);
				viewport->connect("size_changed", callable_mp(this, &Control::_size_changed));
			}
		} break;

		case NOTIFICATION_EXIT_CANVAS: {
			if (data.parent_canvas_item) {
				data.parent_canvas_item->disconnect("item_rect_changed", callable_mp(this, &Control::_size_changed));
				data.parent_canvas_item = nullptr;
			} else if (!is_set_as_top_level()) {
				//disconnect viewport
				Viewport *viewport = get_viewport();
				ERR_FAIL_COND(!viewport);
				viewport->disconnect("size_changed", callable_mp(this, &Control::_size_changed));
			}

			if (data.RI) {
				get_viewport()->_gui_remove_root_control(data.RI);
				data.RI = nullptr;
			}

			data.parent = nullptr;
			data.parent_canvas_item = nullptr;
			data.parent_window = nullptr;
			data.is_rtl_dirty = true;
		} break;

		case NOTIFICATION_MOVED_IN_PARENT: {
			// some parents need to know the order of the children to draw (like TabContainer)
			// update if necessary
			if (data.parent) {
				data.parent->queue_redraw();
			}
			queue_redraw();

			if (data.RI) {
				get_viewport()->_gui_set_root_order_dirty();
			}
		} break;

		case NOTIFICATION_RESIZED: {
			emit_signal(SceneStringNames::get_singleton()->resized);
		} break;

		case NOTIFICATION_DRAW: {
			_update_canvas_item_transform();
			RenderingServer::get_singleton()->canvas_item_set_custom_rect(get_canvas_item(), !data.disable_visibility_clip, Rect2(Point2(), get_size()));
			RenderingServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), data.clip_contents);
		} break;

		case NOTIFICATION_MOUSE_ENTER: {
			emit_signal(SceneStringNames::get_singleton()->mouse_entered);
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			emit_signal(SceneStringNames::get_singleton()->mouse_exited);
		} break;

		case NOTIFICATION_FOCUS_ENTER: {
			emit_signal(SceneStringNames::get_singleton()->focus_entered);
			queue_redraw();
		} break;

		case NOTIFICATION_FOCUS_EXIT: {
			emit_signal(SceneStringNames::get_singleton()->focus_exited);
			queue_redraw();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			emit_signal(SceneStringNames::get_singleton()->theme_changed);
			_invalidate_theme_cache();
			_update_theme_item_cache();
			update_minimum_size();
			queue_redraw();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible_in_tree()) {
				if (get_viewport() != nullptr) {
					get_viewport()->_gui_hide_control(this);
				}
			} else {
				data.minimum_size_valid = false;
				_update_minimum_size();
				_size_changed();
			}
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			if (is_inside_tree()) {
				data.is_rtl_dirty = true;
				_invalidate_theme_cache();
				_update_theme_item_cache();
				_size_changed();
			}
		} break;
	}
}

void Control::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_minimum_size"), &Control::_update_minimum_size);

	ClassDB::bind_method(D_METHOD("accept_event"), &Control::accept_event);
	ClassDB::bind_method(D_METHOD("get_minimum_size"), &Control::get_minimum_size);
	ClassDB::bind_method(D_METHOD("get_combined_minimum_size"), &Control::get_combined_minimum_size);

	ClassDB::bind_method(D_METHOD("_set_layout_mode", "mode"), &Control::_set_layout_mode);
	ClassDB::bind_method(D_METHOD("_get_layout_mode"), &Control::_get_layout_mode);
	ClassDB::bind_method(D_METHOD("_set_anchors_layout_preset", "preset"), &Control::_set_anchors_layout_preset);
	ClassDB::bind_method(D_METHOD("_get_anchors_layout_preset"), &Control::_get_anchors_layout_preset);
	ClassDB::bind_method(D_METHOD("set_anchors_preset", "preset", "keep_offsets"), &Control::set_anchors_preset, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_offsets_preset", "preset", "resize_mode", "margin"), &Control::set_offsets_preset, DEFVAL(PRESET_MODE_MINSIZE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_anchors_and_offsets_preset", "preset", "resize_mode", "margin"), &Control::set_anchors_and_offsets_preset, DEFVAL(PRESET_MODE_MINSIZE), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("_set_anchor", "side", "anchor"), &Control::_set_anchor);
	ClassDB::bind_method(D_METHOD("set_anchor", "side", "anchor", "keep_offset", "push_opposite_anchor"), &Control::set_anchor, DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_anchor", "side"), &Control::get_anchor);
	ClassDB::bind_method(D_METHOD("set_offset", "side", "offset"), &Control::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset", "offset"), &Control::get_offset);
	ClassDB::bind_method(D_METHOD("set_anchor_and_offset", "side", "anchor", "offset", "push_opposite_anchor"), &Control::set_anchor_and_offset, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("set_begin", "position"), &Control::set_begin);
	ClassDB::bind_method(D_METHOD("set_end", "position"), &Control::set_end);
	ClassDB::bind_method(D_METHOD("set_position", "position", "keep_offsets"), &Control::set_position, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_set_position", "position"), &Control::_set_position);
	ClassDB::bind_method(D_METHOD("set_size", "size", "keep_offsets"), &Control::set_size, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("reset_size"), &Control::reset_size);
	ClassDB::bind_method(D_METHOD("_set_size", "size"), &Control::_set_size);
	ClassDB::bind_method(D_METHOD("set_custom_minimum_size", "size"), &Control::set_custom_minimum_size);
	ClassDB::bind_method(D_METHOD("set_global_position", "position", "keep_offsets"), &Control::set_global_position, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_set_global_position", "position"), &Control::_set_global_position);
	ClassDB::bind_method(D_METHOD("set_rotation", "radians"), &Control::set_rotation);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Control::set_scale);
	ClassDB::bind_method(D_METHOD("set_pivot_offset", "pivot_offset"), &Control::set_pivot_offset);
	ClassDB::bind_method(D_METHOD("get_begin"), &Control::get_begin);
	ClassDB::bind_method(D_METHOD("get_end"), &Control::get_end);
	ClassDB::bind_method(D_METHOD("get_position"), &Control::get_position);
	ClassDB::bind_method(D_METHOD("get_size"), &Control::get_size);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Control::get_rotation);
	ClassDB::bind_method(D_METHOD("get_scale"), &Control::get_scale);
	ClassDB::bind_method(D_METHOD("get_pivot_offset"), &Control::get_pivot_offset);
	ClassDB::bind_method(D_METHOD("get_custom_minimum_size"), &Control::get_custom_minimum_size);
	ClassDB::bind_method(D_METHOD("get_parent_area_size"), &Control::get_parent_area_size);
	ClassDB::bind_method(D_METHOD("get_global_position"), &Control::get_global_position);
	ClassDB::bind_method(D_METHOD("get_screen_position"), &Control::get_screen_position);
	ClassDB::bind_method(D_METHOD("get_rect"), &Control::get_rect);
	ClassDB::bind_method(D_METHOD("get_global_rect"), &Control::get_global_rect);
	ClassDB::bind_method(D_METHOD("set_focus_mode", "mode"), &Control::set_focus_mode);
	ClassDB::bind_method(D_METHOD("get_focus_mode"), &Control::get_focus_mode);
	ClassDB::bind_method(D_METHOD("has_focus"), &Control::has_focus);
	ClassDB::bind_method(D_METHOD("grab_focus"), &Control::grab_focus);
	ClassDB::bind_method(D_METHOD("release_focus"), &Control::release_focus);
	ClassDB::bind_method(D_METHOD("find_prev_valid_focus"), &Control::find_prev_valid_focus);
	ClassDB::bind_method(D_METHOD("find_next_valid_focus"), &Control::find_next_valid_focus);

	ClassDB::bind_method(D_METHOD("set_h_size_flags", "flags"), &Control::set_h_size_flags);
	ClassDB::bind_method(D_METHOD("get_h_size_flags"), &Control::get_h_size_flags);

	ClassDB::bind_method(D_METHOD("set_stretch_ratio", "ratio"), &Control::set_stretch_ratio);
	ClassDB::bind_method(D_METHOD("get_stretch_ratio"), &Control::get_stretch_ratio);

	ClassDB::bind_method(D_METHOD("set_v_size_flags", "flags"), &Control::set_v_size_flags);
	ClassDB::bind_method(D_METHOD("get_v_size_flags"), &Control::get_v_size_flags);

	ClassDB::bind_method(D_METHOD("set_theme", "theme"), &Control::set_theme);
	ClassDB::bind_method(D_METHOD("get_theme"), &Control::get_theme);

	ClassDB::bind_method(D_METHOD("set_theme_type_variation", "theme_type"), &Control::set_theme_type_variation);
	ClassDB::bind_method(D_METHOD("get_theme_type_variation"), &Control::get_theme_type_variation);

	ClassDB::bind_method(D_METHOD("begin_bulk_theme_override"), &Control::begin_bulk_theme_override);
	ClassDB::bind_method(D_METHOD("end_bulk_theme_override"), &Control::end_bulk_theme_override);

	ClassDB::bind_method(D_METHOD("add_theme_icon_override", "name", "texture"), &Control::add_theme_icon_override);
	ClassDB::bind_method(D_METHOD("add_theme_stylebox_override", "name", "stylebox"), &Control::add_theme_style_override);
	ClassDB::bind_method(D_METHOD("add_theme_font_override", "name", "font"), &Control::add_theme_font_override);
	ClassDB::bind_method(D_METHOD("add_theme_font_size_override", "name", "font_size"), &Control::add_theme_font_size_override);
	ClassDB::bind_method(D_METHOD("add_theme_color_override", "name", "color"), &Control::add_theme_color_override);
	ClassDB::bind_method(D_METHOD("add_theme_constant_override", "name", "constant"), &Control::add_theme_constant_override);

	ClassDB::bind_method(D_METHOD("remove_theme_icon_override", "name"), &Control::remove_theme_icon_override);
	ClassDB::bind_method(D_METHOD("remove_theme_stylebox_override", "name"), &Control::remove_theme_style_override);
	ClassDB::bind_method(D_METHOD("remove_theme_font_override", "name"), &Control::remove_theme_font_override);
	ClassDB::bind_method(D_METHOD("remove_theme_font_size_override", "name"), &Control::remove_theme_font_size_override);
	ClassDB::bind_method(D_METHOD("remove_theme_color_override", "name"), &Control::remove_theme_color_override);
	ClassDB::bind_method(D_METHOD("remove_theme_constant_override", "name"), &Control::remove_theme_constant_override);

	ClassDB::bind_method(D_METHOD("get_theme_icon", "name", "theme_type"), &Control::get_theme_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_stylebox", "name", "theme_type"), &Control::get_theme_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_font", "name", "theme_type"), &Control::get_theme_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_font_size", "name", "theme_type"), &Control::get_theme_font_size, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_color", "name", "theme_type"), &Control::get_theme_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_constant", "name", "theme_type"), &Control::get_theme_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("has_theme_icon_override", "name"), &Control::has_theme_icon_override);
	ClassDB::bind_method(D_METHOD("has_theme_stylebox_override", "name"), &Control::has_theme_stylebox_override);
	ClassDB::bind_method(D_METHOD("has_theme_font_override", "name"), &Control::has_theme_font_override);
	ClassDB::bind_method(D_METHOD("has_theme_font_size_override", "name"), &Control::has_theme_font_size_override);
	ClassDB::bind_method(D_METHOD("has_theme_color_override", "name"), &Control::has_theme_color_override);
	ClassDB::bind_method(D_METHOD("has_theme_constant_override", "name"), &Control::has_theme_constant_override);

	ClassDB::bind_method(D_METHOD("has_theme_icon", "name", "theme_type"), &Control::has_theme_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_stylebox", "name", "theme_type"), &Control::has_theme_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_font", "name", "theme_type"), &Control::has_theme_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_font_size", "name", "theme_type"), &Control::has_theme_font_size, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_color", "name", "theme_type"), &Control::has_theme_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_constant", "name", "theme_type"), &Control::has_theme_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("get_theme_default_base_scale"), &Control::get_theme_default_base_scale);
	ClassDB::bind_method(D_METHOD("get_theme_default_font"), &Control::get_theme_default_font);
	ClassDB::bind_method(D_METHOD("get_theme_default_font_size"), &Control::get_theme_default_font_size);

	ClassDB::bind_method(D_METHOD("get_parent_control"), &Control::get_parent_control);

	ClassDB::bind_method(D_METHOD("set_h_grow_direction", "direction"), &Control::set_h_grow_direction);
	ClassDB::bind_method(D_METHOD("get_h_grow_direction"), &Control::get_h_grow_direction);

	ClassDB::bind_method(D_METHOD("set_v_grow_direction", "direction"), &Control::set_v_grow_direction);
	ClassDB::bind_method(D_METHOD("get_v_grow_direction"), &Control::get_v_grow_direction);

	ClassDB::bind_method(D_METHOD("set_tooltip_text", "hint"), &Control::set_tooltip_text);
	ClassDB::bind_method(D_METHOD("get_tooltip_text"), &Control::get_tooltip_text);
	ClassDB::bind_method(D_METHOD("get_tooltip", "at_position"), &Control::get_tooltip, DEFVAL(Point2()));

	ClassDB::bind_method(D_METHOD("set_default_cursor_shape", "shape"), &Control::set_default_cursor_shape);
	ClassDB::bind_method(D_METHOD("get_default_cursor_shape"), &Control::get_default_cursor_shape);
	ClassDB::bind_method(D_METHOD("get_cursor_shape", "position"), &Control::get_cursor_shape, DEFVAL(Point2()));

	ClassDB::bind_method(D_METHOD("set_focus_neighbor", "side", "neighbor"), &Control::set_focus_neighbor);
	ClassDB::bind_method(D_METHOD("get_focus_neighbor", "side"), &Control::get_focus_neighbor);

	ClassDB::bind_method(D_METHOD("set_focus_next", "next"), &Control::set_focus_next);
	ClassDB::bind_method(D_METHOD("get_focus_next"), &Control::get_focus_next);

	ClassDB::bind_method(D_METHOD("set_focus_previous", "previous"), &Control::set_focus_previous);
	ClassDB::bind_method(D_METHOD("get_focus_previous"), &Control::get_focus_previous);

	ClassDB::bind_method(D_METHOD("force_drag", "data", "preview"), &Control::force_drag);

	ClassDB::bind_method(D_METHOD("set_mouse_filter", "filter"), &Control::set_mouse_filter);
	ClassDB::bind_method(D_METHOD("get_mouse_filter"), &Control::get_mouse_filter);

	ClassDB::bind_method(D_METHOD("set_force_pass_scroll_events", "force_pass_scroll_events"), &Control::set_force_pass_scroll_events);
	ClassDB::bind_method(D_METHOD("is_force_pass_scroll_events"), &Control::is_force_pass_scroll_events);

	ClassDB::bind_method(D_METHOD("set_clip_contents", "enable"), &Control::set_clip_contents);
	ClassDB::bind_method(D_METHOD("is_clipping_contents"), &Control::is_clipping_contents);

	ClassDB::bind_method(D_METHOD("grab_click_focus"), &Control::grab_click_focus);

	ClassDB::bind_method(D_METHOD("set_drag_forwarding", "target"), &Control::set_drag_forwarding);
	ClassDB::bind_method(D_METHOD("set_drag_preview", "control"), &Control::set_drag_preview);
	ClassDB::bind_method(D_METHOD("is_drag_successful"), &Control::is_drag_successful);

	ClassDB::bind_method(D_METHOD("warp_mouse", "position"), &Control::warp_mouse);

	ClassDB::bind_method(D_METHOD("update_minimum_size"), &Control::update_minimum_size);

	ClassDB::bind_method(D_METHOD("set_layout_direction", "direction"), &Control::set_layout_direction);
	ClassDB::bind_method(D_METHOD("get_layout_direction"), &Control::get_layout_direction);
	ClassDB::bind_method(D_METHOD("is_layout_rtl"), &Control::is_layout_rtl);

	ClassDB::bind_method(D_METHOD("set_auto_translate", "enable"), &Control::set_auto_translate);
	ClassDB::bind_method(D_METHOD("is_auto_translating"), &Control::is_auto_translating);

	ADD_GROUP("Layout", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_contents"), "set_clip_contents", "is_clipping_contents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "custom_minimum_size", PROPERTY_HINT_NONE, "suffix:px"), "set_custom_minimum_size", "get_custom_minimum_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layout_direction", PROPERTY_HINT_ENUM, "Inherited,Locale,Left-to-Right,Right-to-Left"), "set_layout_direction", "get_layout_direction");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layout_mode", PROPERTY_HINT_ENUM, "Position,Anchors,Container,Uncontrolled", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "_set_layout_mode", "_get_layout_mode");
	ADD_PROPERTY_DEFAULT("layout_mode", LayoutMode::LAYOUT_MODE_POSITION);

	const String anchors_presets_options = "Custom:-1,PresetFullRect:15,"
										   "PresetTopLeft:0,PresetTopRight:1,PresetBottomRight:3,PresetBottomLeft:2,"
										   "PresetCenterLeft:4,PresetCenterTop:5,PresetCenterRight:6,PresetCenterBottom:7,PresetCenter:8,"
										   "PresetLeftWide:9,PresetTopWide:10,PresetRightWide:11,PresetBottomWide:12,PresetVCenterWide:13,PresetHCenterWide:14";

	ADD_PROPERTY(PropertyInfo(Variant::INT, "anchors_preset", PROPERTY_HINT_ENUM, anchors_presets_options, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "_set_anchors_layout_preset", "_get_anchors_layout_preset");
	ADD_PROPERTY_DEFAULT("anchors_preset", -1);

	ADD_SUBGROUP_INDENT("Anchor Points", "anchor_", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_left", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_top", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_right", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_bottom", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_BOTTOM);

	ADD_SUBGROUP_INDENT("Anchor Offsets", "offset_", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_left", PROPERTY_HINT_RANGE, "-4096,4096,suffix:px"), "set_offset", "get_offset", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_top", PROPERTY_HINT_RANGE, "-4096,4096,suffix:px"), "set_offset", "get_offset", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_right", PROPERTY_HINT_RANGE, "-4096,4096,suffix:px"), "set_offset", "get_offset", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_bottom", PROPERTY_HINT_RANGE, "-4096,4096,suffix:px"), "set_offset", "get_offset", SIDE_BOTTOM);

	ADD_SUBGROUP_INDENT("Grow Direction", "grow_", 1);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "grow_horizontal", PROPERTY_HINT_ENUM, "Left,Right,Both"), "set_h_grow_direction", "get_h_grow_direction");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "grow_vertical", PROPERTY_HINT_ENUM, "Top,Bottom,Both"), "set_v_grow_direction", "get_v_grow_direction");

	ADD_SUBGROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_NONE, "suffix:px", PROPERTY_USAGE_EDITOR), "_set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position", PROPERTY_HINT_NONE, "suffix:px", PROPERTY_USAGE_EDITOR), "_set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "global_position", PROPERTY_HINT_NONE, "suffix:px", PROPERTY_USAGE_NONE), "_set_global_position", "get_global_position");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation", PROPERTY_HINT_RANGE, "-360,360,0.1,or_lesser,or_greater,radians"), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scale"), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "pivot_offset", PROPERTY_HINT_NONE, "suffix:px"), "set_pivot_offset", "get_pivot_offset");

	ADD_SUBGROUP("Container Sizing", "size_flags_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size_flags_horizontal", PROPERTY_HINT_FLAGS, "Fill:1,Expand:2,Shrink Center:4,Shrink End:8"), "set_h_size_flags", "get_h_size_flags");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size_flags_vertical", PROPERTY_HINT_FLAGS, "Fill:1,Expand:2,Shrink Center:4,Shrink End:8"), "set_v_size_flags", "get_v_size_flags");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "size_flags_stretch_ratio", PROPERTY_HINT_RANGE, "0,20,0.01,or_greater"), "set_stretch_ratio", "get_stretch_ratio");

	ADD_GROUP("Auto Translate", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_translate"), "set_auto_translate", "is_auto_translating");

	ADD_GROUP("Tooltip", "tooltip_");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tooltip_text", PROPERTY_HINT_MULTILINE_TEXT), "set_tooltip_text", "get_tooltip_text");

	ADD_GROUP("Focus", "focus_");
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_left", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_top", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_right", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_bottom", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_BOTTOM);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "focus_next", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_next", "get_focus_next");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "focus_previous", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_previous", "get_focus_previous");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "focus_mode", PROPERTY_HINT_ENUM, "None,Click,All"), "set_focus_mode", "get_focus_mode");

	ADD_GROUP("Mouse", "mouse_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_filter", PROPERTY_HINT_ENUM, "Stop,Pass,Ignore"), "set_mouse_filter", "get_mouse_filter");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mouse_force_pass_scroll_events"), "set_force_pass_scroll_events", "is_force_pass_scroll_events");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_default_cursor_shape", PROPERTY_HINT_ENUM, "Arrow,I-Beam,Pointing Hand,Cross,Wait,Busy,Drag,Can Drop,Forbidden,Vertical Resize,Horizontal Resize,Secondary Diagonal Resize,Main Diagonal Resize,Move,Vertical Split,Horizontal Split,Help"), "set_default_cursor_shape", "get_default_cursor_shape");

	ADD_GROUP("Theme", "theme_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "theme", PROPERTY_HINT_RESOURCE_TYPE, "Theme"), "set_theme", "get_theme");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "theme_type_variation", PROPERTY_HINT_ENUM_SUGGESTION), "set_theme_type_variation", "get_theme_type_variation");

	BIND_ENUM_CONSTANT(FOCUS_NONE);
	BIND_ENUM_CONSTANT(FOCUS_CLICK);
	BIND_ENUM_CONSTANT(FOCUS_ALL);

	BIND_CONSTANT(NOTIFICATION_RESIZED);
	BIND_CONSTANT(NOTIFICATION_MOUSE_ENTER);
	BIND_CONSTANT(NOTIFICATION_MOUSE_EXIT);
	BIND_CONSTANT(NOTIFICATION_FOCUS_ENTER);
	BIND_CONSTANT(NOTIFICATION_FOCUS_EXIT);
	BIND_CONSTANT(NOTIFICATION_THEME_CHANGED);
	BIND_CONSTANT(NOTIFICATION_SCROLL_BEGIN);
	BIND_CONSTANT(NOTIFICATION_SCROLL_END);
	BIND_CONSTANT(NOTIFICATION_LAYOUT_DIRECTION_CHANGED);

	BIND_ENUM_CONSTANT(CURSOR_ARROW);
	BIND_ENUM_CONSTANT(CURSOR_IBEAM);
	BIND_ENUM_CONSTANT(CURSOR_POINTING_HAND);
	BIND_ENUM_CONSTANT(CURSOR_CROSS);
	BIND_ENUM_CONSTANT(CURSOR_WAIT);
	BIND_ENUM_CONSTANT(CURSOR_BUSY);
	BIND_ENUM_CONSTANT(CURSOR_DRAG);
	BIND_ENUM_CONSTANT(CURSOR_CAN_DROP);
	BIND_ENUM_CONSTANT(CURSOR_FORBIDDEN);
	BIND_ENUM_CONSTANT(CURSOR_VSIZE);
	BIND_ENUM_CONSTANT(CURSOR_HSIZE);
	BIND_ENUM_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_MOVE);
	BIND_ENUM_CONSTANT(CURSOR_VSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HELP);

	BIND_ENUM_CONSTANT(PRESET_TOP_LEFT);
	BIND_ENUM_CONSTANT(PRESET_TOP_RIGHT);
	BIND_ENUM_CONSTANT(PRESET_BOTTOM_LEFT);
	BIND_ENUM_CONSTANT(PRESET_BOTTOM_RIGHT);
	BIND_ENUM_CONSTANT(PRESET_CENTER_LEFT);
	BIND_ENUM_CONSTANT(PRESET_CENTER_TOP);
	BIND_ENUM_CONSTANT(PRESET_CENTER_RIGHT);
	BIND_ENUM_CONSTANT(PRESET_CENTER_BOTTOM);
	BIND_ENUM_CONSTANT(PRESET_CENTER);
	BIND_ENUM_CONSTANT(PRESET_LEFT_WIDE);
	BIND_ENUM_CONSTANT(PRESET_TOP_WIDE);
	BIND_ENUM_CONSTANT(PRESET_RIGHT_WIDE);
	BIND_ENUM_CONSTANT(PRESET_BOTTOM_WIDE);
	BIND_ENUM_CONSTANT(PRESET_VCENTER_WIDE);
	BIND_ENUM_CONSTANT(PRESET_HCENTER_WIDE);
	BIND_ENUM_CONSTANT(PRESET_FULL_RECT);

	BIND_ENUM_CONSTANT(PRESET_MODE_MINSIZE);
	BIND_ENUM_CONSTANT(PRESET_MODE_KEEP_WIDTH);
	BIND_ENUM_CONSTANT(PRESET_MODE_KEEP_HEIGHT);
	BIND_ENUM_CONSTANT(PRESET_MODE_KEEP_SIZE);

	BIND_ENUM_CONSTANT(SIZE_SHRINK_BEGIN);
	BIND_ENUM_CONSTANT(SIZE_FILL);
	BIND_ENUM_CONSTANT(SIZE_EXPAND);
	BIND_ENUM_CONSTANT(SIZE_EXPAND_FILL);
	BIND_ENUM_CONSTANT(SIZE_SHRINK_CENTER);
	BIND_ENUM_CONSTANT(SIZE_SHRINK_END);

	BIND_ENUM_CONSTANT(MOUSE_FILTER_STOP);
	BIND_ENUM_CONSTANT(MOUSE_FILTER_PASS);
	BIND_ENUM_CONSTANT(MOUSE_FILTER_IGNORE);

	BIND_ENUM_CONSTANT(GROW_DIRECTION_BEGIN);
	BIND_ENUM_CONSTANT(GROW_DIRECTION_END);
	BIND_ENUM_CONSTANT(GROW_DIRECTION_BOTH);

	BIND_ENUM_CONSTANT(ANCHOR_BEGIN);
	BIND_ENUM_CONSTANT(ANCHOR_END);

	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_INHERITED);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_LOCALE);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_LTR);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_RTL);

	BIND_ENUM_CONSTANT(TEXT_DIRECTION_INHERITED);
	BIND_ENUM_CONSTANT(TEXT_DIRECTION_AUTO);
	BIND_ENUM_CONSTANT(TEXT_DIRECTION_LTR);
	BIND_ENUM_CONSTANT(TEXT_DIRECTION_RTL);

	ADD_SIGNAL(MethodInfo("resized"));
	ADD_SIGNAL(MethodInfo("gui_input", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));
	ADD_SIGNAL(MethodInfo("focus_entered"));
	ADD_SIGNAL(MethodInfo("focus_exited"));
	ADD_SIGNAL(MethodInfo("size_flags_changed"));
	ADD_SIGNAL(MethodInfo("minimum_size_changed"));
	ADD_SIGNAL(MethodInfo("theme_changed"));

	GDVIRTUAL_BIND(_has_point, "position");
	GDVIRTUAL_BIND(_structured_text_parser, "args", "text");
	GDVIRTUAL_BIND(_get_minimum_size);

	GDVIRTUAL_BIND(_get_drag_data, "at_position");
	GDVIRTUAL_BIND(_can_drop_data, "at_position", "data");
	GDVIRTUAL_BIND(_drop_data, "at_position", "data");
	GDVIRTUAL_BIND(_make_custom_tooltip, "for_text");

	GDVIRTUAL_BIND(_gui_input, "event");
}

Control::~Control() {
	// Resources need to be disconnected.
	for (KeyValue<StringName, Ref<Texture2D>> &E : data.icon_override) {
		E.value->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}
	for (KeyValue<StringName, Ref<StyleBox>> &E : data.style_override) {
		E.value->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}
	for (KeyValue<StringName, Ref<Font>> &E : data.font_override) {
		E.value->disconnect("changed", callable_mp(this, &Control::_notify_theme_override_changed));
	}

	// Then override maps can be simply cleared.
	data.icon_override.clear();
	data.style_override.clear();
	data.font_override.clear();
	data.font_size_override.clear();
	data.color_override.clear();
	data.constant_override.clear();
}
