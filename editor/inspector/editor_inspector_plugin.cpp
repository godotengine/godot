/**************************************************************************/
/*  editor_inspector_plugin.cpp                                           */
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

#include "editor_inspector_plugin.h"
#include "editor_inspector_plugin.compat.inc"

#include "core/config/project_settings.h"
#include "core/input/input_map.h"
#include "core/object/class_db.h"
#include "editor/inspector/editor_properties.h"
#include "editor/inspector/editor_properties_array_dict.h"
#include "editor/inspector/editor_properties_vector.h"
#include "editor/settings/editor_settings.h"

void EditorInspectorPlugin::add_custom_control(Control *control) {
	AddedEditor ae;
	ae.property_editor = control;
	added_editors.push_back(ae);
}

void EditorInspectorPlugin::add_property_editor(const String &p_for_property, Control *p_prop, bool p_add_to_end, const String &p_label) {
	AddedEditor ae;
	ae.properties.push_back(p_for_property);
	ae.property_editor = p_prop;
	ae.add_to_end = p_add_to_end;
	ae.label = p_label;
	added_editors.push_back(ae);
}

void EditorInspectorPlugin::add_property_editor_for_multiple_properties(const String &p_label, const Vector<String> &p_properties, Control *p_prop) {
	AddedEditor ae;
	ae.properties = p_properties;
	ae.property_editor = p_prop;
	ae.label = p_label;
	added_editors.push_back(ae);
}

bool EditorInspectorPlugin::can_handle(Object *p_object) {
	bool success = false;
	GDVIRTUAL_CALL(_can_handle, p_object, success);
	return success;
}

void EditorInspectorPlugin::parse_begin(Object *p_object) {
	GDVIRTUAL_CALL(_parse_begin, p_object);
}

void EditorInspectorPlugin::parse_category(Object *p_object, const String &p_category) {
	GDVIRTUAL_CALL(_parse_category, p_object, p_category);
}

void EditorInspectorPlugin::parse_group(Object *p_object, const String &p_group) {
	GDVIRTUAL_CALL(_parse_group, p_object, p_group);
}

bool EditorInspectorPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	bool ret = false;
	GDVIRTUAL_CALL(_parse_property, p_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide, ret);
	return ret;
}

void EditorInspectorPlugin::parse_end(Object *p_object) {
	GDVIRTUAL_CALL(_parse_end, p_object);
}

void EditorInspectorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_custom_control", "control"), &EditorInspectorPlugin::add_custom_control);
	ClassDB::bind_method(D_METHOD("add_property_editor", "property", "editor", "add_to_end", "label"), &EditorInspectorPlugin::add_property_editor, DEFVAL(false), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("add_property_editor_for_multiple_properties", "label", "properties", "editor"), &EditorInspectorPlugin::add_property_editor_for_multiple_properties);

	GDVIRTUAL_BIND(_can_handle, "object")
	GDVIRTUAL_BIND(_parse_begin, "object")
	GDVIRTUAL_BIND(_parse_category, "object", "category")
	GDVIRTUAL_BIND(_parse_group, "object", "group")
	GDVIRTUAL_BIND(_parse_property, "object", "type", "name", "hint_type", "hint_string", "usage_flags", "wide");
	GDVIRTUAL_BIND(_parse_end, "object")
}

////////////// DEFAULT PLUGIN //////////////////////

bool EditorInspectorDefaultPlugin::can_handle(Object *p_object) {
	return true; // Can handle everything.
}

bool EditorInspectorDefaultPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	Control *editor = EditorInspectorDefaultPlugin::get_editor_for_property(p_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
	if (editor) {
		add_property_editor(p_path, editor);
	}
	return false;
}

static EditorPropertyRangeHint _parse_range_hint(PropertyHint p_hint, const String &p_hint_text, double p_default_step, bool is_int = false) {
	EditorPropertyRangeHint hint;
	hint.step = p_default_step;
	if (is_int) {
		hint.hide_control = false; // Always show controls for ints, unless specified in hint range.
	}
	Vector<String> slices = p_hint_text.split(",");
	if (p_hint == PROPERTY_HINT_RANGE) {
		ERR_FAIL_COND_V_MSG(slices.size() < 2, hint,
				vformat("Invalid PROPERTY_HINT_RANGE with hint \"%s\": Missing required min and/or max values.", p_hint_text));

		hint.or_greater = false; // If using ranged, assume false by default.
		hint.or_less = false;

		hint.min = slices[0].to_float();
		hint.max = slices[1].to_float();

		if (slices.size() >= 3 && slices[2].is_valid_float()) {
			// Step is optional, could be something else if not a number.
			hint.step = slices[2].to_float();
		}
		hint.hide_control = false;
		for (int i = 2; i < slices.size(); i++) {
			String slice = slices[i].strip_edges();
			if (slice == "or_greater") {
				hint.or_greater = true;
			} else if (slice == "or_less") {
				hint.or_less = true;
			} else if (slice == "prefer_slider") {
				hint.prefer_slider = true;
			} else if (slice == "hide_control") {
				hint.hide_control = true;
#ifndef DISABLE_DEPRECATED
			} else if (slice == "hide_slider") {
				hint.hide_control = true;
#endif
			} else if (slice == "exp") {
				hint.exp_range = true;
			}
		}
	}
	bool degrees = false;
	for (int i = 0; i < slices.size(); i++) {
		String slice = slices[i].strip_edges();
		if (slice == "radians_as_degrees"
#ifndef DISABLE_DEPRECATED
				|| slice == "radians"
#endif // DISABLE_DEPRECATED
		) {
			hint.radians_as_degrees = true;
		} else if (slice == "degrees") {
			degrees = true;
		} else if (slice.begins_with("suffix:")) {
			hint.suffix = " " + slice.replace_first("suffix:", "").strip_edges();
		}
	}

	if ((hint.radians_as_degrees || degrees) && hint.suffix.is_empty()) {
		hint.suffix = U"\u00B0";
	}

	ERR_FAIL_COND_V_MSG(hint.step == 0, hint,
			vformat("Invalid PROPERTY_HINT_RANGE with hint \"%s\": Step cannot be 0.", p_hint_text));

	return hint;
}

static EditorProperty *get_input_action_editor(const String &p_hint_text, bool is_string_name) {
	// TODO: Should probably use a better editor GUI with a search bar.
	// Said GUI could also handle showing builtin options, requiring 1 less hint.
	EditorPropertyTextEnum *editor = memnew(EditorPropertyTextEnum);
	Vector<String> options;
	Vector<String> builtin_options;
	List<PropertyInfo> pinfo;
	ProjectSettings::get_singleton()->get_property_list(&pinfo);
	Vector<String> hints = p_hint_text.remove_char(' ').split(",", false);

	HashMap<String, List<Ref<InputEvent>>> builtins(InputMap::get_singleton()->get_builtins());
	bool show_builtin = hints.has("show_builtin");

	for (const PropertyInfo &pi : pinfo) {
		if (!pi.name.begins_with("input/")) {
			continue;
		}

		const String action_name = pi.name.get_slicec('/', 1);
		if (builtins.has(action_name)) {
			if (show_builtin) {
				builtin_options.append(action_name);
			}
		} else {
			options.append(action_name);
		}
	}
	options.append_array(builtin_options);
	editor->setup(options, Vector<String>(), is_string_name, hints.has("loose_mode"));
	return editor;
}

EditorProperty *EditorInspectorDefaultPlugin::get_editor_for_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	double default_float_step = EDITOR_GET("interface/inspector/default_float_step");

	switch (p_type) {
		// atomic types
		case Variant::NIL: {
			if (p_usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
				return memnew(EditorPropertyVariant);
			} else {
				return memnew(EditorPropertyNil);
			}
		} break;
		case Variant::BOOL: {
			EditorPropertyCheck *editor = memnew(EditorPropertyCheck);
			return editor;
		} break;
		case Variant::INT: {
			if (p_hint == PROPERTY_HINT_ENUM) {
				EditorPropertyEnum *editor = memnew(EditorPropertyEnum);
				Vector<String> options = p_hint_text.split(",");
				editor->setup(options);
				return editor;

			} else if (p_hint == PROPERTY_HINT_FLAGS) {
				EditorPropertyFlags *editor = memnew(EditorPropertyFlags);
				Vector<String> options = p_hint_text.split(",");
				editor->setup(options);
				return editor;

			} else if (p_hint == PROPERTY_HINT_LAYERS_2D_PHYSICS ||
					p_hint == PROPERTY_HINT_LAYERS_2D_RENDER ||
					p_hint == PROPERTY_HINT_LAYERS_2D_NAVIGATION ||
					p_hint == PROPERTY_HINT_LAYERS_3D_PHYSICS ||
					p_hint == PROPERTY_HINT_LAYERS_3D_RENDER ||
					p_hint == PROPERTY_HINT_LAYERS_3D_NAVIGATION ||
					p_hint == PROPERTY_HINT_LAYERS_AVOIDANCE) {
				EditorPropertyLayers::LayerType lt = EditorPropertyLayers::LAYER_RENDER_2D;
				switch (p_hint) {
					case PROPERTY_HINT_LAYERS_2D_RENDER:
						lt = EditorPropertyLayers::LAYER_RENDER_2D;
						break;
					case PROPERTY_HINT_LAYERS_2D_PHYSICS:
						lt = EditorPropertyLayers::LAYER_PHYSICS_2D;
						break;
					case PROPERTY_HINT_LAYERS_2D_NAVIGATION:
						lt = EditorPropertyLayers::LAYER_NAVIGATION_2D;
						break;
					case PROPERTY_HINT_LAYERS_3D_RENDER:
						lt = EditorPropertyLayers::LAYER_RENDER_3D;
						break;
					case PROPERTY_HINT_LAYERS_3D_PHYSICS:
						lt = EditorPropertyLayers::LAYER_PHYSICS_3D;
						break;
					case PROPERTY_HINT_LAYERS_3D_NAVIGATION:
						lt = EditorPropertyLayers::LAYER_NAVIGATION_3D;
						break;
					case PROPERTY_HINT_LAYERS_AVOIDANCE:
						lt = EditorPropertyLayers::LAYER_AVOIDANCE;
						break;
					default: {
					} //compiler could be smarter here and realize this can't happen
				}
				EditorPropertyLayers *editor = memnew(EditorPropertyLayers);
				editor->setup(lt);
				return editor;
			} else if (p_hint == PROPERTY_HINT_OBJECT_ID) {
				EditorPropertyObjectID *editor = memnew(EditorPropertyObjectID);
				editor->setup(p_hint_text);
				return editor;

			} else {
				EditorPropertyInteger *editor = memnew(EditorPropertyInteger);
				editor->setup(_parse_range_hint(p_hint, p_hint_text, 1, true));
				return editor;
			}
		} break;
		case Variant::FLOAT: {
			if (p_hint == PROPERTY_HINT_EXP_EASING) {
				EditorPropertyEasing *editor = memnew(EditorPropertyEasing);
				bool positive_only = false;
				bool flip = false;
				const Vector<String> hints = p_hint_text.split(",");
				for (int i = 0; i < hints.size(); i++) {
					const String hint = hints[i].strip_edges();
					if (hint == "attenuation") {
						flip = true;
					}
					if (hint == "positive_only") {
						positive_only = true;
					}
				}

				editor->setup(positive_only, flip);
				return editor;

			} else {
				EditorPropertyFloat *editor = memnew(EditorPropertyFloat);
				editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
				return editor;
			}
		} break;
		case Variant::STRING: {
			if (p_hint == PROPERTY_HINT_ENUM || p_hint == PROPERTY_HINT_ENUM_SUGGESTION) {
				EditorPropertyTextEnum *editor = memnew(EditorPropertyTextEnum);
				Vector<String> options;
				Vector<String> option_names;
				if (p_hint_text.begins_with(";")) {
					// This is not supported officially. Only for `interface/editor/localization/editor_language`.
					for (const String &option : p_hint_text.split(";", false)) {
						options.append(option.get_slicec('/', 0));
						option_names.append(option.get_slicec('/', 1));
					}
				} else {
					options = p_hint_text.split(",", false);
				}
				editor->setup(options, option_names, false, (p_hint == PROPERTY_HINT_ENUM_SUGGESTION));
				return editor;
			} else if (p_hint == PROPERTY_HINT_INPUT_NAME) {
				return get_input_action_editor(p_hint_text, false);
			} else if (p_hint == PROPERTY_HINT_MULTILINE_TEXT) {
				Vector<String> options = p_hint_text.split(",", false);
				EditorPropertyMultilineText *editor = memnew(EditorPropertyMultilineText(false));
				if (options.has("monospace")) {
					editor->set_monospaced(true);
				}
				if (options.has("no_wrap")) {
					editor->set_wrap_lines(false);
				}
				return editor;
			} else if (p_hint == PROPERTY_HINT_EXPRESSION) {
				EditorPropertyMultilineText *editor = memnew(EditorPropertyMultilineText(true));
				return editor;
			} else if (p_hint == PROPERTY_HINT_TYPE_STRING) {
				EditorPropertyClassName *editor = memnew(EditorPropertyClassName);
				editor->setup(p_hint_text, p_hint_text);
				return editor;
			} else if (p_hint == PROPERTY_HINT_LOCALE_ID) {
				EditorPropertyLocale *editor = memnew(EditorPropertyLocale);
				editor->setup(p_hint_text);
				return editor;
			} else if (p_hint == PROPERTY_HINT_DIR || p_hint == PROPERTY_HINT_FILE || p_hint == PROPERTY_HINT_SAVE_FILE || p_hint == PROPERTY_HINT_GLOBAL_SAVE_FILE || p_hint == PROPERTY_HINT_GLOBAL_DIR || p_hint == PROPERTY_HINT_GLOBAL_FILE || p_hint == PROPERTY_HINT_FILE_PATH) {
				Vector<String> extensions = p_hint_text.split(",");
				bool global = p_hint == PROPERTY_HINT_GLOBAL_DIR || p_hint == PROPERTY_HINT_GLOBAL_FILE || p_hint == PROPERTY_HINT_GLOBAL_SAVE_FILE;
				bool folder = p_hint == PROPERTY_HINT_DIR || p_hint == PROPERTY_HINT_GLOBAL_DIR;
				bool save = p_hint == PROPERTY_HINT_SAVE_FILE || p_hint == PROPERTY_HINT_GLOBAL_SAVE_FILE;
				bool enable_uid = p_hint == PROPERTY_HINT_FILE;
				EditorPropertyPath *editor = memnew(EditorPropertyPath);
				editor->setup(extensions, folder, global, enable_uid);
				if (save) {
					editor->set_save_mode();
				}
				return editor;
			} else {
				EditorPropertyText *editor = memnew(EditorPropertyText);

				Vector<String> hints = p_hint_text.split(",");
				if (hints.has("monospace")) {
					editor->set_monospaced(true);
				}

				if (p_hint == PROPERTY_HINT_PLACEHOLDER_TEXT) {
					editor->set_placeholder(p_hint_text);
				} else if (p_hint == PROPERTY_HINT_PASSWORD) {
					editor->set_secret(true);
					editor->set_placeholder(p_hint_text);
				}
				return editor;
			}
		} break;

			// math types

		case Variant::VECTOR2: {
			EditorPropertyVector2 *editor = memnew(EditorPropertyVector2(p_wide));
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step), p_hint == PROPERTY_HINT_LINK);
			return editor;

		} break;
		case Variant::VECTOR2I: {
			EditorPropertyVector2i *editor = memnew(EditorPropertyVector2i(p_wide));
			EditorPropertyRangeHint hint = _parse_range_hint(p_hint, p_hint_text, 1, true);
			hint.step = Math::round(hint.step);
			editor->setup(hint, p_hint == PROPERTY_HINT_LINK, true);
			return editor;

		} break;
		case Variant::RECT2: {
			EditorPropertyRect2 *editor = memnew(EditorPropertyRect2(p_wide));
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;
		} break;
		case Variant::RECT2I: {
			EditorPropertyRect2i *editor = memnew(EditorPropertyRect2i(p_wide));
			editor->setup(_parse_range_hint(p_hint, p_hint_text, 1, true));
			return editor;
		} break;
		case Variant::VECTOR3: {
			EditorPropertyVector3 *editor = memnew(EditorPropertyVector3(p_wide));
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step), p_hint == PROPERTY_HINT_LINK);
			return editor;

		} break;
		case Variant::VECTOR3I: {
			EditorPropertyVector3i *editor = memnew(EditorPropertyVector3i(p_wide));
			EditorPropertyRangeHint hint = _parse_range_hint(p_hint, p_hint_text, 1, true);
			hint.step = Math::round(hint.step);
			editor->setup(hint, p_hint == PROPERTY_HINT_LINK, true);
			return editor;

		} break;
		case Variant::VECTOR4: {
			EditorPropertyVector4 *editor = memnew(EditorPropertyVector4);
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step), p_hint == PROPERTY_HINT_LINK);
			return editor;

		} break;
		case Variant::VECTOR4I: {
			EditorPropertyVector4i *editor = memnew(EditorPropertyVector4i);
			EditorPropertyRangeHint hint = _parse_range_hint(p_hint, p_hint_text, 1, true);
			hint.step = Math::round(hint.step);
			editor->setup(hint, p_hint == PROPERTY_HINT_LINK, true);
			return editor;

		} break;
		case Variant::TRANSFORM2D: {
			EditorPropertyTransform2D *editor = memnew(EditorPropertyTransform2D);
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;
		} break;
		case Variant::PLANE: {
			EditorPropertyPlane *editor = memnew(EditorPropertyPlane(p_wide));
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;
		} break;
		case Variant::QUATERNION: {
			EditorPropertyQuaternion *editor = memnew(EditorPropertyQuaternion);
			// Quaternions are almost never used for human-readable values that need stepifying,
			// so we should be more precise with their step, as much as the float precision allows.
#ifdef REAL_T_IS_DOUBLE
			constexpr double QUATERNION_STEP = 1e-14;
#else
			constexpr double QUATERNION_STEP = 1e-6;
#endif
			editor->setup(_parse_range_hint(p_hint, p_hint_text, QUATERNION_STEP), p_hint == PROPERTY_HINT_HIDE_QUATERNION_EDIT);
			return editor;
		} break;
		case Variant::AABB: {
			EditorPropertyAABB *editor = memnew(EditorPropertyAABB);
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;
		} break;
		case Variant::BASIS: {
			EditorPropertyBasis *editor = memnew(EditorPropertyBasis);
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;
		} break;
		case Variant::TRANSFORM3D: {
			EditorPropertyTransform3D *editor = memnew(EditorPropertyTransform3D);
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;

		} break;
		case Variant::PROJECTION: {
			EditorPropertyProjection *editor = memnew(EditorPropertyProjection);
			editor->setup(_parse_range_hint(p_hint, p_hint_text, default_float_step));
			return editor;

		} break;

		// misc types
		case Variant::COLOR: {
			EditorPropertyColor *editor = memnew(EditorPropertyColor);
			editor->setup(p_hint != PROPERTY_HINT_COLOR_NO_ALPHA);
			return editor;
		} break;
		case Variant::STRING_NAME: {
			if (p_hint == PROPERTY_HINT_ENUM || p_hint == PROPERTY_HINT_ENUM_SUGGESTION) {
				EditorPropertyTextEnum *editor = memnew(EditorPropertyTextEnum);
				Vector<String> options = p_hint_text.split(",", false);
				editor->setup(options, Vector<String>(), true, (p_hint == PROPERTY_HINT_ENUM_SUGGESTION));
				return editor;
			} else if (p_hint == PROPERTY_HINT_INPUT_NAME) {
				return get_input_action_editor(p_hint_text, true);
			} else {
				EditorPropertyText *editor = memnew(EditorPropertyText);
				if (p_hint == PROPERTY_HINT_PLACEHOLDER_TEXT) {
					editor->set_placeholder(p_hint_text);
				} else if (p_hint == PROPERTY_HINT_PASSWORD) {
					editor->set_secret(true);
					editor->set_placeholder(p_hint_text);
				}
				editor->set_string_name(true);
				return editor;
			}
		} break;
		case Variant::NODE_PATH: {
			EditorPropertyNodePath *editor = memnew(EditorPropertyNodePath);
			if (p_hint == PROPERTY_HINT_NODE_PATH_VALID_TYPES && !p_hint_text.is_empty()) {
				Vector<String> types = p_hint_text.split(",", false);
				Vector<StringName> sn = Variant(types); //convert via variant
				editor->setup(sn, (p_usage & PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT));
			}
			return editor;

		} break;
		case Variant::RID: {
			EditorPropertyRID *editor = memnew(EditorPropertyRID);
			return editor;
		} break;
		case Variant::OBJECT: {
			if (p_hint == PROPERTY_HINT_NODE_TYPE) {
				EditorPropertyNodePath *editor = memnew(EditorPropertyNodePath);
				Vector<String> types = p_hint_text.split(",", false);
				Vector<StringName> sn = Variant(types); //convert via variant
				editor->setup(sn, false, true);
				return editor;
			} else {
				EditorPropertyResource *editor = memnew(EditorPropertyResource);
				editor->setup(p_object, p_path, p_hint == PROPERTY_HINT_RESOURCE_TYPE ? p_hint_text : "Resource");

				if (p_hint == PROPERTY_HINT_RESOURCE_TYPE) {
					const PackedStringArray open_in_new_inspector = EDITOR_GET("interface/inspector/resources_to_open_in_new_inspector");

					for (const String &type : open_in_new_inspector) {
						for (int j = 0; j < p_hint_text.get_slice_count(","); j++) {
							const String inherits = p_hint_text.get_slicec(',', j);
							if (ClassDB::is_parent_class(inherits, type)) {
								editor->set_use_sub_inspector(false);
							}
						}
					}
				}

				return editor;
			}

		} break;
		case Variant::CALLABLE: {
			EditorPropertyCallable *editor = memnew(EditorPropertyCallable);
			return editor;
		} break;
		case Variant::SIGNAL: {
			EditorPropertySignal *editor = memnew(EditorPropertySignal);
			return editor;
		} break;
		case Variant::DICTIONARY: {
			if (p_hint == PROPERTY_HINT_LOCALIZABLE_STRING) {
				EditorPropertyLocalizableString *editor = memnew(EditorPropertyLocalizableString);
				return editor;
			} else {
				EditorPropertyDictionary *editor = memnew(EditorPropertyDictionary);
				editor->setup(p_hint, p_hint_text);
				return editor;
			}
		} break;
		case Variant::ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_BYTE_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_BYTE_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_INT32_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_INT32_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_INT64_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_INT64_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_FLOAT32_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_FLOAT64_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_STRING_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_STRING_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_VECTOR2_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_VECTOR2_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_VECTOR3_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_COLOR_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_COLOR_ARRAY, p_hint_text);
			return editor;
		} break;
		case Variant::PACKED_VECTOR4_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(Variant::PACKED_VECTOR4_ARRAY, p_hint_text);
			return editor;
		} break;
		default: {
		}
	}

	return nullptr;
}
