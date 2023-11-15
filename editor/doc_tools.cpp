/**************************************************************************/
/*  doc_tools.cpp                                                         */
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

#include "doc_tools.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/core_constants.h"
#include "core/io/compression.h"
#include "core/io/dir_access.h"
#include "core/io/marshalls.h"
#include "core/io/resource_importer.h"
#include "core/object/script_language.h"
#include "core/string/translation.h"
#include "editor/editor_settings.h"
#include "editor/export/editor_export.h"
#include "scene/resources/theme.h"
#include "scene/theme/theme_db.h"

// Used for a hack preserving Mono properties on non-Mono builds.
#include "modules/modules_enabled.gen.h" // For mono.

static String _get_indent(const String &p_text) {
	String indent;
	bool has_text = false;
	int line_start = 0;

	for (int i = 0; i < p_text.length(); i++) {
		const char32_t c = p_text[i];
		if (c == '\n') {
			line_start = i + 1;
		} else if (c > 32) {
			has_text = true;
			indent = p_text.substr(line_start, i - line_start);
			break; // Indentation of the first line that has text.
		}
	}
	if (!has_text) {
		return p_text;
	}
	return indent;
}

static String _translate_doc_string(const String &p_text) {
	const String indent = _get_indent(p_text);
	const String message = p_text.dedent().strip_edges();
	const String translated = TranslationServer::get_singleton()->doc_translate(message, "");
	// No need to restore stripped edges because they'll be stripped again later.
	return translated.indent(indent);
}

void DocTools::merge_from(const DocTools &p_data) {
	for (KeyValue<String, DocData::ClassDoc> &E : class_list) {
		DocData::ClassDoc &c = E.value;

		if (!p_data.class_list.has(c.name)) {
			continue;
		}

		const DocData::ClassDoc &cf = p_data.class_list[c.name];

		c.is_deprecated = cf.is_deprecated;
		c.is_experimental = cf.is_experimental;

		c.description = cf.description;
		c.brief_description = cf.brief_description;
		c.tutorials = cf.tutorials;

		for (int i = 0; i < c.constructors.size(); i++) {
			DocData::MethodDoc &m = c.constructors.write[i];

			for (int j = 0; j < cf.constructors.size(); j++) {
				if (cf.constructors[j].name != m.name) {
					continue;
				}

				{
					// Since constructors can repeat, we need to check the type of
					// the arguments so we make sure they are different.
					if (cf.constructors[j].arguments.size() != m.arguments.size()) {
						continue;
					}
					int arg_count = cf.constructors[j].arguments.size();
					Vector<bool> arg_used;
					arg_used.resize(arg_count);
					for (int l = 0; l < arg_count; ++l) {
						arg_used.write[l] = false;
					}
					// also there is no guarantee that argument ordering will match, so we
					// have to check one by one so we make sure we have an exact match
					for (int k = 0; k < arg_count; ++k) {
						for (int l = 0; l < arg_count; ++l) {
							if (cf.constructors[j].arguments[k].type == m.arguments[l].type && !arg_used[l]) {
								arg_used.write[l] = true;
								break;
							}
						}
					}
					bool not_the_same = false;
					for (int l = 0; l < arg_count; ++l) {
						if (!arg_used[l]) { // at least one of the arguments was different
							not_the_same = true;
						}
					}
					if (not_the_same) {
						continue;
					}
				}

				const DocData::MethodDoc &mf = cf.constructors[j];

				m.description = mf.description;
				m.is_deprecated = mf.is_deprecated;
				m.is_experimental = mf.is_experimental;
				break;
			}
		}

		for (int i = 0; i < c.methods.size(); i++) {
			DocData::MethodDoc &m = c.methods.write[i];

			for (int j = 0; j < cf.methods.size(); j++) {
				if (cf.methods[j].name != m.name) {
					continue;
				}

				const DocData::MethodDoc &mf = cf.methods[j];

				m.description = mf.description;
				m.is_deprecated = mf.is_deprecated;
				m.is_experimental = mf.is_experimental;
				break;
			}
		}

		for (int i = 0; i < c.signals.size(); i++) {
			DocData::MethodDoc &m = c.signals.write[i];

			for (int j = 0; j < cf.signals.size(); j++) {
				if (cf.signals[j].name != m.name) {
					continue;
				}
				const DocData::MethodDoc &mf = cf.signals[j];

				m.description = mf.description;
				m.is_deprecated = mf.is_deprecated;
				m.is_experimental = mf.is_experimental;
				break;
			}
		}

		for (int i = 0; i < c.constants.size(); i++) {
			DocData::ConstantDoc &m = c.constants.write[i];

			for (int j = 0; j < cf.constants.size(); j++) {
				if (cf.constants[j].name != m.name) {
					continue;
				}
				const DocData::ConstantDoc &mf = cf.constants[j];

				m.description = mf.description;
				m.is_deprecated = mf.is_deprecated;
				m.is_experimental = mf.is_experimental;
				break;
			}
		}

		for (int i = 0; i < c.annotations.size(); i++) {
			DocData::MethodDoc &m = c.annotations.write[i];

			for (int j = 0; j < cf.annotations.size(); j++) {
				if (cf.annotations[j].name != m.name) {
					continue;
				}
				const DocData::MethodDoc &mf = cf.annotations[j];

				m.description = mf.description;
				m.is_deprecated = mf.is_deprecated;
				m.is_experimental = mf.is_experimental;
				break;
			}
		}

		for (int i = 0; i < c.properties.size(); i++) {
			DocData::PropertyDoc &p = c.properties.write[i];

			for (int j = 0; j < cf.properties.size(); j++) {
				if (cf.properties[j].name != p.name) {
					continue;
				}
				const DocData::PropertyDoc &pf = cf.properties[j];

				p.description = pf.description;
				p.is_deprecated = pf.is_deprecated;
				p.is_experimental = pf.is_experimental;
				break;
			}
		}

		for (int i = 0; i < c.theme_properties.size(); i++) {
			DocData::ThemeItemDoc &ti = c.theme_properties.write[i];

			for (int j = 0; j < cf.theme_properties.size(); j++) {
				if (cf.theme_properties[j].name != ti.name || cf.theme_properties[j].data_type != ti.data_type) {
					continue;
				}
				const DocData::ThemeItemDoc &pf = cf.theme_properties[j];

				ti.description = pf.description;
				break;
			}
		}

		for (int i = 0; i < c.operators.size(); i++) {
			DocData::MethodDoc &m = c.operators.write[i];

			for (int j = 0; j < cf.operators.size(); j++) {
				if (cf.operators[j].name != m.name) {
					continue;
				}

				{
					// Since operators can repeat, we need to check the type of
					// the arguments so we make sure they are different.
					if (cf.operators[j].arguments.size() != m.arguments.size()) {
						continue;
					}
					int arg_count = cf.operators[j].arguments.size();
					Vector<bool> arg_used;
					arg_used.resize(arg_count);
					for (int l = 0; l < arg_count; ++l) {
						arg_used.write[l] = false;
					}
					// also there is no guarantee that argument ordering will match, so we
					// have to check one by one so we make sure we have an exact match
					for (int k = 0; k < arg_count; ++k) {
						for (int l = 0; l < arg_count; ++l) {
							if (cf.operators[j].arguments[k].type == m.arguments[l].type && !arg_used[l]) {
								arg_used.write[l] = true;
								break;
							}
						}
					}
					bool not_the_same = false;
					for (int l = 0; l < arg_count; ++l) {
						if (!arg_used[l]) { // at least one of the arguments was different
							not_the_same = true;
						}
					}
					if (not_the_same) {
						continue;
					}
				}

				const DocData::MethodDoc &mf = cf.operators[j];

				m.description = mf.description;
				m.is_deprecated = mf.is_deprecated;
				m.is_experimental = mf.is_experimental;
				break;
			}
		}

#ifndef MODULE_MONO_ENABLED
		// The Mono module defines some properties that we want to keep when
		// re-generating docs with a non-Mono build, to prevent pointless diffs
		// (and loss of descriptions) depending on the config of the doc writer.
		// We use a horrible hack to force keeping the relevant properties,
		// hardcoded below. At least it's an ad hoc hack... ¯\_(ツ)_/¯
		// Don't show this to your kids.
		if (c.name == "@GlobalScope") {
			// Retrieve GodotSharp singleton.
			for (int j = 0; j < cf.properties.size(); j++) {
				if (cf.properties[j].name == "GodotSharp") {
					c.properties.push_back(cf.properties[j]);
				}
			}
		}
#endif
	}
}

void DocTools::remove_from(const DocTools &p_data) {
	for (const KeyValue<String, DocData::ClassDoc> &E : p_data.class_list) {
		if (class_list.has(E.key)) {
			class_list.erase(E.key);
		}
	}
}

void DocTools::add_doc(const DocData::ClassDoc &p_class_doc) {
	ERR_FAIL_COND(p_class_doc.name.is_empty());
	class_list[p_class_doc.name] = p_class_doc;
}

void DocTools::remove_doc(const String &p_class_name) {
	ERR_FAIL_COND(p_class_name.is_empty() || !class_list.has(p_class_name));
	class_list.erase(p_class_name);
}

bool DocTools::has_doc(const String &p_class_name) {
	if (p_class_name.is_empty()) {
		return false;
	}
	return class_list.has(p_class_name);
}

static Variant get_documentation_default_value(const StringName &p_class_name, const StringName &p_property_name, bool &r_default_value_valid) {
	Variant default_value = Variant();
	r_default_value_valid = false;

	if (ClassDB::can_instantiate(p_class_name) && !ClassDB::is_virtual(p_class_name)) { // Keep this condition in sync with ClassDB::class_get_default_property_value.
		default_value = ClassDB::class_get_default_property_value(p_class_name, p_property_name, &r_default_value_valid);
	} else {
		// Cannot get default value of classes that can't be instantiated
		List<StringName> inheriting_classes;
		ClassDB::get_direct_inheriters_from_class(p_class_name, &inheriting_classes);
		for (List<StringName>::Element *E2 = inheriting_classes.front(); E2; E2 = E2->next()) {
			if (ClassDB::can_instantiate(E2->get())) {
				default_value = ClassDB::class_get_default_property_value(E2->get(), p_property_name, &r_default_value_valid);
				if (r_default_value_valid) {
					break;
				}
			}
		}
	}

	return default_value;
}

void DocTools::generate(BitField<GenerateFlags> p_flags) {
	// This may involve instantiating classes that are only usable from the main thread
	// (which is in fact the case of the core API).
	ERR_FAIL_COND(!Thread::is_main_thread());

	// Add ClassDB-exposed classes.
	{
		List<StringName> classes;
		if (p_flags.has_flag(GENERATE_FLAG_EXTENSION_CLASSES_ONLY)) {
			ClassDB::get_extensions_class_list(&classes);
		} else {
			ClassDB::get_class_list(&classes);
			// Move ProjectSettings, so that other classes can register properties there.
			classes.move_to_back(classes.find("ProjectSettings"));
		}

		bool skip_setter_getter_methods = true;

		// Populate documentation data for each exposed class.
		while (classes.size()) {
			const String &name = classes.front()->get();
			if (!ClassDB::is_class_exposed(name)) {
				print_verbose(vformat("Class '%s' is not exposed, skipping.", name));
				classes.pop_front();
				continue;
			}

			String cname = name;
			// Property setters and getters do not get exposed as individual methods.
			HashSet<StringName> setters_getters;

			class_list[cname] = DocData::ClassDoc();
			DocData::ClassDoc &c = class_list[cname];
			c.name = cname;
			c.inherits = ClassDB::get_parent_class(name);

			List<PropertyInfo> properties;
			List<PropertyInfo> own_properties;

			// Special cases for editor/project settings, and ResourceImporter classes,
			// we have to rely on Object's property list to get settings and import options.
			// Otherwise we just use ClassDB's property list (pure registered properties).

			bool properties_from_instance = true; // To skip `script`, etc.
			bool import_option = false; // Special case for default value.
			HashMap<StringName, Variant> import_options_default;
			if (name == "EditorSettings") {
				// We don't create the full blown EditorSettings (+ config file) with `create()`,
				// instead we just make a local instance to get default values.
				Ref<EditorSettings> edset = memnew(EditorSettings);
				edset->get_property_list(&properties);
				own_properties = properties;
			} else if (name == "ProjectSettings") {
				ProjectSettings::get_singleton()->get_property_list(&properties);
				own_properties = properties;
			} else if (ClassDB::is_parent_class(name, "ResourceImporter") && name != "EditorImportPlugin" && ClassDB::can_instantiate(name)) {
				import_option = true;
				ResourceImporter *resimp = Object::cast_to<ResourceImporter>(ClassDB::instantiate(name));
				List<ResourceImporter::ImportOption> options;
				resimp->get_import_options("", &options);
				for (int i = 0; i < options.size(); i++) {
					const PropertyInfo &prop = options[i].option;
					properties.push_back(prop);
					import_options_default[prop.name] = options[i].default_value;
				}
				own_properties = properties;
				memdelete(resimp);
			} else if (name.begins_with("EditorExportPlatform") && ClassDB::can_instantiate(name)) {
				properties_from_instance = false;
				Ref<EditorExportPlatform> platform = Object::cast_to<EditorExportPlatform>(ClassDB::instantiate(name));
				if (platform.is_valid()) {
					List<EditorExportPlatform::ExportOption> options;
					platform->get_export_options(&options);
					for (const EditorExportPlatform::ExportOption &E : options) {
						properties.push_back(E.option);
					}
					own_properties = properties;
				}
			} else {
				properties_from_instance = false;
				ClassDB::get_property_list(name, &properties);
				ClassDB::get_property_list(name, &own_properties, true);
			}

			// Sort is still needed here to handle inherited properties, even though it is done below, do not remove.
			properties.sort();
			own_properties.sort();

			List<PropertyInfo>::Element *EO = own_properties.front();
			for (const PropertyInfo &E : properties) {
				bool inherited = true;
				if (EO && EO->get() == E) {
					inherited = false;
					EO = EO->next();
				}

				if (properties_from_instance) {
					if (E.name == "resource_local_to_scene" || E.name == "resource_name" || E.name == "resource_path" || E.name == "script") {
						// Don't include spurious properties from Object property list.
						continue;
					}
				}

				if (E.usage & PROPERTY_USAGE_GROUP || E.usage & PROPERTY_USAGE_SUBGROUP || E.usage & PROPERTY_USAGE_CATEGORY || E.usage & PROPERTY_USAGE_INTERNAL || (E.type == Variant::NIL && E.usage & PROPERTY_USAGE_ARRAY)) {
					continue;
				}

				DocData::PropertyDoc prop;
				prop.name = E.name;
				prop.overridden = inherited;

				if (inherited) {
					String parent = ClassDB::get_parent_class(c.name);
					while (!ClassDB::has_property(parent, prop.name, true)) {
						parent = ClassDB::get_parent_class(parent);
					}
					prop.overrides = parent;
				}

				bool default_value_valid = false;
				Variant default_value;

				if (name == "ProjectSettings") {
					// Special case for project settings, so that settings are not taken from the current project's settings
					if (!ProjectSettings::get_singleton()->is_builtin_setting(E.name)) {
						continue;
					}
					if (E.usage & PROPERTY_USAGE_EDITOR) {
						if (!ProjectSettings::get_singleton()->get_ignore_value_in_docs(E.name)) {
							default_value = ProjectSettings::get_singleton()->property_get_revert(E.name);
							default_value_valid = true;
						}
					}
				} else if (name == "EditorSettings") {
					// Special case for editor settings, to prevent hardware or OS specific settings to affect the result.
				} else if (import_option) {
					default_value = import_options_default[E.name];
					default_value_valid = true;
				} else {
					default_value = get_documentation_default_value(name, E.name, default_value_valid);
					if (inherited) {
						bool base_default_value_valid = false;
						Variant base_default_value = get_documentation_default_value(ClassDB::get_parent_class(name), E.name, base_default_value_valid);
						if (!default_value_valid || !base_default_value_valid || default_value == base_default_value) {
							continue;
						}
					}
				}

				if (default_value_valid && default_value.get_type() != Variant::OBJECT) {
					prop.default_value = DocData::get_default_value_string(default_value);
				}

				StringName setter = ClassDB::get_property_setter(name, E.name);
				StringName getter = ClassDB::get_property_getter(name, E.name);

				prop.setter = setter;
				prop.getter = getter;

				bool found_type = false;
				if (getter != StringName()) {
					MethodBind *mb = ClassDB::get_method(name, getter);
					if (mb) {
						PropertyInfo retinfo = mb->get_return_info();

						found_type = true;
						if (retinfo.type == Variant::INT && retinfo.usage & (PROPERTY_USAGE_CLASS_IS_ENUM | PROPERTY_USAGE_CLASS_IS_BITFIELD)) {
							prop.enumeration = retinfo.class_name;
							prop.is_bitfield = retinfo.usage & PROPERTY_USAGE_CLASS_IS_BITFIELD;
							prop.type = "int";
						} else if (retinfo.class_name != StringName()) {
							prop.type = retinfo.class_name;
						} else if (retinfo.type == Variant::ARRAY && retinfo.hint == PROPERTY_HINT_ARRAY_TYPE) {
							prop.type = retinfo.hint_string + "[]";
						} else if (retinfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
							prop.type = retinfo.hint_string;
						} else if (retinfo.type == Variant::NIL && retinfo.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
							prop.type = "Variant";
						} else if (retinfo.type == Variant::NIL) {
							prop.type = "void";
						} else {
							prop.type = Variant::get_type_name(retinfo.type);
						}
					}

					setters_getters.insert(getter);
				}

				if (setter != StringName()) {
					setters_getters.insert(setter);
				}

				if (!found_type) {
					if (E.type == Variant::OBJECT && E.hint == PROPERTY_HINT_RESOURCE_TYPE) {
						prop.type = E.hint_string;
					} else {
						prop.type = Variant::get_type_name(E.type);
					}
				}

				c.properties.push_back(prop);
			}

			c.properties.sort();

			List<MethodInfo> method_list;
			ClassDB::get_method_list(name, &method_list, true);

			for (const MethodInfo &E : method_list) {
				if (E.name.is_empty() || (E.name[0] == '_' && !(E.flags & METHOD_FLAG_VIRTUAL))) {
					continue; //hidden, don't count
				}

				if (skip_setter_getter_methods && setters_getters.has(E.name)) {
					// Don't skip parametric setters and getters, i.e. method which require
					// one or more parameters to define what property should be set or retrieved.
					// E.g. CPUParticles3D::set_param(Parameter param, float value).
					if (E.arguments.size() == 0 /* getter */ || (E.arguments.size() == 1 && E.return_val.type == Variant::NIL /* setter */)) {
						continue;
					}
				}

				DocData::MethodDoc method;
				DocData::method_doc_from_methodinfo(method, E, "");

				Vector<Error> errs = ClassDB::get_method_error_return_values(name, E.name);
				if (errs.size()) {
					if (!errs.has(OK)) {
						errs.insert(0, OK);
					}
					for (int i = 0; i < errs.size(); i++) {
						if (!method.errors_returned.has(errs[i])) {
							method.errors_returned.push_back(errs[i]);
						}
					}
				}

				c.methods.push_back(method);
			}

			c.methods.sort();

			List<MethodInfo> signal_list;
			ClassDB::get_signal_list(name, &signal_list, true);

			if (signal_list.size()) {
				for (List<MethodInfo>::Element *EV = signal_list.front(); EV; EV = EV->next()) {
					DocData::MethodDoc signal;
					signal.name = EV->get().name;
					for (int i = 0; i < EV->get().arguments.size(); i++) {
						const PropertyInfo &arginfo = EV->get().arguments[i];
						DocData::ArgumentDoc argument;
						DocData::argument_doc_from_arginfo(argument, arginfo);

						signal.arguments.push_back(argument);
					}

					c.signals.push_back(signal);
				}
			}

			List<String> constant_list;
			ClassDB::get_integer_constant_list(name, &constant_list, true);

			for (const String &E : constant_list) {
				DocData::ConstantDoc constant;
				constant.name = E;
				constant.value = itos(ClassDB::get_integer_constant(name, E));
				constant.is_value_valid = true;
				constant.enumeration = ClassDB::get_integer_constant_enum(name, E);
				constant.is_bitfield = ClassDB::is_enum_bitfield(name, constant.enumeration);
				c.constants.push_back(constant);
			}

			// Theme items.
			{
				List<ThemeDB::ThemeItemBind> theme_items;
				ThemeDB::get_singleton()->get_class_own_items(cname, &theme_items);
				Ref<Theme> default_theme = ThemeDB::get_singleton()->get_default_theme();

				for (const ThemeDB::ThemeItemBind &theme_item : theme_items) {
					DocData::ThemeItemDoc tid;
					tid.name = theme_item.item_name;

					switch (theme_item.data_type) {
						case Theme::DATA_TYPE_COLOR:
							tid.type = "Color";
							tid.data_type = "color";
							break;
						case Theme::DATA_TYPE_CONSTANT:
							tid.type = "int";
							tid.data_type = "constant";
							break;
						case Theme::DATA_TYPE_FONT:
							tid.type = "Font";
							tid.data_type = "font";
							break;
						case Theme::DATA_TYPE_FONT_SIZE:
							tid.type = "int";
							tid.data_type = "font_size";
							break;
						case Theme::DATA_TYPE_ICON:
							tid.type = "Texture2D";
							tid.data_type = "icon";
							break;
						case Theme::DATA_TYPE_STYLEBOX:
							tid.type = "StyleBox";
							tid.data_type = "style";
							break;
						case Theme::DATA_TYPE_MAX:
							break; // Can't happen, but silences warning.
					}

					if (theme_item.data_type == Theme::DATA_TYPE_COLOR || theme_item.data_type == Theme::DATA_TYPE_CONSTANT) {
						tid.default_value = DocData::get_default_value_string(default_theme->get_theme_item(theme_item.data_type, theme_item.item_name, cname));
					}

					c.theme_properties.push_back(tid);
				}

				c.theme_properties.sort();
			}

			classes.pop_front();
		}
	}

	if (p_flags.has_flag(GENERATE_FLAG_SKIP_BASIC_TYPES)) {
		return;
	}

	// Add a dummy Variant entry.
	{
		// This allows us to document the concept of Variant even though
		// it's not a ClassDB-exposed class.
		class_list["Variant"] = DocData::ClassDoc();
		class_list["Variant"].name = "Variant";
	}

	// Add Variant data types.
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i == Variant::NIL) {
			continue; // Not exposed outside of 'null', should not be in class list.
		}
		if (i == Variant::OBJECT) {
			continue; // Use the core type instead.
		}

		String cname = Variant::get_type_name(Variant::Type(i));

		class_list[cname] = DocData::ClassDoc();
		DocData::ClassDoc &c = class_list[cname];
		c.name = cname;

		Callable::CallError cerror;
		Variant v;
		Variant::construct(Variant::Type(i), v, nullptr, 0, cerror);

		List<MethodInfo> method_list;
		v.get_method_list(&method_list);
		Variant::get_constructor_list(Variant::Type(i), &method_list);

		for (int j = 0; j < Variant::OP_AND; j++) { // Showing above 'and' is pretty confusing and there are a lot of variations.
			for (int k = 0; k < Variant::VARIANT_MAX; k++) {
				// Prevent generating for comparison with null.
				if (Variant::Type(k) == Variant::NIL && (Variant::Operator(j) == Variant::OP_EQUAL || Variant::Operator(j) == Variant::OP_NOT_EQUAL)) {
					continue;
				}

				Variant::Type rt = Variant::get_operator_return_type(Variant::Operator(j), Variant::Type(i), Variant::Type(k));
				if (rt != Variant::NIL) { // Has operator.
					// Skip String % operator as it's registered separately for each Variant arg type,
					// we'll add it manually below.
					if ((i == Variant::STRING || i == Variant::STRING_NAME) && Variant::Operator(j) == Variant::OP_MODULE) {
						continue;
					}
					MethodInfo mi;
					mi.name = "operator " + Variant::get_operator_name(Variant::Operator(j));
					mi.return_val.type = rt;
					if (k != Variant::NIL) {
						PropertyInfo arg;
						arg.name = "right";
						arg.type = Variant::Type(k);
						mi.arguments.push_back(arg);
					}
					method_list.push_back(mi);
				}
			}
		}

		if (i == Variant::STRING || i == Variant::STRING_NAME) {
			// We skipped % operator above, and we register it manually once for Variant arg type here.
			MethodInfo mi;
			mi.name = "operator %";
			mi.return_val.type = Variant::STRING;

			PropertyInfo arg;
			arg.name = "right";
			arg.type = Variant::NIL;
			arg.usage = PROPERTY_USAGE_NIL_IS_VARIANT;
			mi.arguments.push_back(arg);

			method_list.push_back(mi);
		}

		if (Variant::is_keyed(Variant::Type(i))) {
			MethodInfo mi;
			mi.name = "operator []";
			mi.return_val.type = Variant::NIL;
			mi.return_val.usage = PROPERTY_USAGE_NIL_IS_VARIANT;

			PropertyInfo arg;
			arg.name = "key";
			arg.type = Variant::NIL;
			arg.usage = PROPERTY_USAGE_NIL_IS_VARIANT;
			mi.arguments.push_back(arg);

			method_list.push_back(mi);
		} else if (Variant::has_indexing(Variant::Type(i))) {
			MethodInfo mi;
			mi.name = "operator []";
			mi.return_val.type = Variant::get_indexed_element_type(Variant::Type(i));
			mi.return_val.usage = Variant::get_indexed_element_usage(Variant::Type(i));
			PropertyInfo arg;
			arg.name = "index";
			arg.type = Variant::INT;
			mi.arguments.push_back(arg);

			method_list.push_back(mi);
		}

		for (const MethodInfo &mi : method_list) {
			DocData::MethodDoc method;

			method.name = mi.name;

			for (int j = 0; j < mi.arguments.size(); j++) {
				PropertyInfo arginfo = mi.arguments[j];
				DocData::ArgumentDoc ad;
				DocData::argument_doc_from_arginfo(ad, mi.arguments[j]);
				ad.name = arginfo.name;

				int darg_idx = mi.default_arguments.size() - mi.arguments.size() + j;
				if (darg_idx >= 0) {
					ad.default_value = DocData::get_default_value_string(mi.default_arguments[darg_idx]);
				}

				method.arguments.push_back(ad);
			}

			DocData::return_doc_from_retinfo(method, mi.return_val);

			if (mi.flags & METHOD_FLAG_VARARG) {
				if (!method.qualifiers.is_empty()) {
					method.qualifiers += " ";
				}
				method.qualifiers += "vararg";
			}

			if (mi.flags & METHOD_FLAG_CONST) {
				if (!method.qualifiers.is_empty()) {
					method.qualifiers += " ";
				}
				method.qualifiers += "const";
			}

			if (mi.flags & METHOD_FLAG_STATIC) {
				if (!method.qualifiers.is_empty()) {
					method.qualifiers += " ";
				}
				method.qualifiers += "static";
			}

			if (method.name == cname) {
				c.constructors.push_back(method);
			} else if (method.name.begins_with("operator")) {
				c.operators.push_back(method);
			} else {
				c.methods.push_back(method);
			}
		}

		c.methods.sort();

		List<PropertyInfo> properties;
		v.get_property_list(&properties);
		for (const PropertyInfo &pi : properties) {
			DocData::PropertyDoc property;
			property.name = pi.name;
			property.type = Variant::get_type_name(pi.type);
			property.default_value = DocData::get_default_value_string(v.get(pi.name));

			c.properties.push_back(property);
		}

		List<StringName> constants;
		Variant::get_constants_for_type(Variant::Type(i), &constants);

		for (const StringName &E : constants) {
			DocData::ConstantDoc constant;
			constant.name = E;
			Variant value = Variant::get_constant_value(Variant::Type(i), E);
			constant.value = value.get_type() == Variant::INT ? itos(value) : value.get_construct_string().replace("\n", " ");
			constant.is_value_valid = true;
			c.constants.push_back(constant);
		}
	}

	// Add global API (servers, engine singletons, global constants) and Variant utility functions.
	{
		String cname = "@GlobalScope";
		class_list[cname] = DocData::ClassDoc();
		DocData::ClassDoc &c = class_list[cname];
		c.name = cname;

		// Global constants.
		for (int i = 0; i < CoreConstants::get_global_constant_count(); i++) {
			DocData::ConstantDoc cd;
			cd.name = CoreConstants::get_global_constant_name(i);
			cd.is_bitfield = CoreConstants::is_global_constant_bitfield(i);
			if (!CoreConstants::get_ignore_value_in_docs(i)) {
				cd.value = itos(CoreConstants::get_global_constant_value(i));
				cd.is_value_valid = true;
			} else {
				cd.is_value_valid = false;
			}
			cd.enumeration = CoreConstants::get_global_constant_enum(i);
			c.constants.push_back(cd);
		}

		// Servers/engine singletons.
		List<Engine::Singleton> singletons;
		Engine::get_singleton()->get_singletons(&singletons);

		// FIXME: this is kind of hackish...
		for (const Engine::Singleton &s : singletons) {
			DocData::PropertyDoc pd;
			if (!s.ptr) {
				continue;
			}
			pd.name = s.name;
			pd.type = s.ptr->get_class();
			while (String(ClassDB::get_parent_class(pd.type)) != "Object") {
				pd.type = ClassDB::get_parent_class(pd.type);
			}
			c.properties.push_back(pd);
		}

		// Variant utility functions.
		List<StringName> utility_functions;
		Variant::get_utility_function_list(&utility_functions);
		utility_functions.sort_custom<StringName::AlphCompare>();
		for (const StringName &E : utility_functions) {
			DocData::MethodDoc md;
			md.name = E;
			// Utility function's return type.
			if (Variant::has_utility_function_return_value(E)) {
				PropertyInfo pi;
				pi.type = Variant::get_utility_function_return_type(E);
				if (pi.type == Variant::NIL) {
					pi.usage = PROPERTY_USAGE_NIL_IS_VARIANT;
				}
				DocData::ArgumentDoc ad;
				DocData::argument_doc_from_arginfo(ad, pi);
				md.return_type = ad.type;
			}

			// Utility function's arguments.
			if (Variant::is_utility_function_vararg(E)) {
				md.qualifiers = "vararg";
			} else {
				for (int i = 0; i < Variant::get_utility_function_argument_count(E); i++) {
					PropertyInfo pi;
					pi.type = Variant::get_utility_function_argument_type(E, i);
					pi.name = Variant::get_utility_function_argument_name(E, i);
					if (pi.type == Variant::NIL) {
						pi.usage = PROPERTY_USAGE_NIL_IS_VARIANT;
					}
					DocData::ArgumentDoc ad;
					DocData::argument_doc_from_arginfo(ad, pi);
					md.arguments.push_back(ad);
				}
			}

			c.methods.push_back(md);
		}
	}

	// Add scripting language built-ins.
	{
		// We only add a doc entry for languages which actually define any built-in
		// methods, constants, or annotations.
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptLanguage *lang = ScriptServer::get_language(i);
			String cname = "@" + lang->get_name();
			DocData::ClassDoc c;
			c.name = cname;

			// Get functions.
			List<MethodInfo> minfo;
			lang->get_public_functions(&minfo);

			for (const MethodInfo &mi : minfo) {
				DocData::MethodDoc md;
				md.name = mi.name;

				if (mi.flags & METHOD_FLAG_VARARG) {
					if (!md.qualifiers.is_empty()) {
						md.qualifiers += " ";
					}
					md.qualifiers += "vararg";
				}

				DocData::return_doc_from_retinfo(md, mi.return_val);

				for (int j = 0; j < mi.arguments.size(); j++) {
					DocData::ArgumentDoc ad;
					DocData::argument_doc_from_arginfo(ad, mi.arguments[j]);

					int darg_idx = j - (mi.arguments.size() - mi.default_arguments.size());
					if (darg_idx >= 0) {
						ad.default_value = DocData::get_default_value_string(mi.default_arguments[darg_idx]);
					}

					md.arguments.push_back(ad);
				}

				c.methods.push_back(md);
			}

			// Get constants.
			List<Pair<String, Variant>> cinfo;
			lang->get_public_constants(&cinfo);

			for (const Pair<String, Variant> &E : cinfo) {
				DocData::ConstantDoc cd;
				cd.name = E.first;
				cd.value = E.second;
				cd.is_value_valid = true;
				c.constants.push_back(cd);
			}

			// Get annotations.
			List<MethodInfo> ainfo;
			lang->get_public_annotations(&ainfo);

			for (const MethodInfo &ai : ainfo) {
				DocData::MethodDoc atd;
				atd.name = ai.name;

				if (ai.flags & METHOD_FLAG_VARARG) {
					if (!atd.qualifiers.is_empty()) {
						atd.qualifiers += " ";
					}
					atd.qualifiers += "vararg";
				}

				DocData::return_doc_from_retinfo(atd, ai.return_val);

				for (int j = 0; j < ai.arguments.size(); j++) {
					DocData::ArgumentDoc ad;
					DocData::argument_doc_from_arginfo(ad, ai.arguments[j]);

					int darg_idx = j - (ai.arguments.size() - ai.default_arguments.size());
					if (darg_idx >= 0) {
						ad.default_value = DocData::get_default_value_string(ai.default_arguments[darg_idx]);
					}

					atd.arguments.push_back(ad);
				}

				c.annotations.push_back(atd);
			}

			// Skip adding the lang if it doesn't expose anything (e.g. C#).
			if (c.methods.is_empty() && c.constants.is_empty() && c.annotations.is_empty()) {
				continue;
			}

			class_list[cname] = c;
		}
	}
}

static Error _parse_methods(Ref<XMLParser> &parser, Vector<DocData::MethodDoc> &methods) {
	String section = parser->get_node_name();
	String element = section.substr(0, section.length() - 1);

	while (parser->read() == OK) {
		if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
			if (parser->get_node_name() == element) {
				DocData::MethodDoc method;
				ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
				method.name = parser->get_named_attribute_value("name");
				if (parser->has_attribute("qualifiers")) {
					method.qualifiers = parser->get_named_attribute_value("qualifiers");
				}
				if (parser->has_attribute("is_deprecated")) {
					method.is_deprecated = parser->get_named_attribute_value("is_deprecated").to_lower() == "true";
				}
				if (parser->has_attribute("is_experimental")) {
					method.is_experimental = parser->get_named_attribute_value("is_experimental").to_lower() == "true";
				}

				while (parser->read() == OK) {
					if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
						String name = parser->get_node_name();
						if (name == "return") {
							ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
							method.return_type = parser->get_named_attribute_value("type");
							if (parser->has_attribute("enum")) {
								method.return_enum = parser->get_named_attribute_value("enum");
								if (parser->has_attribute("is_bitfield")) {
									method.return_is_bitfield = parser->get_named_attribute_value("is_bitfield").to_lower() == "true";
								}
							}
						} else if (name == "returns_error") {
							ERR_FAIL_COND_V(!parser->has_attribute("number"), ERR_FILE_CORRUPT);
							method.errors_returned.push_back(parser->get_named_attribute_value("number").to_int());
						} else if (name == "param") {
							DocData::ArgumentDoc argument;
							ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
							argument.name = parser->get_named_attribute_value("name");
							ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
							argument.type = parser->get_named_attribute_value("type");
							if (parser->has_attribute("enum")) {
								argument.enumeration = parser->get_named_attribute_value("enum");
								if (parser->has_attribute("is_bitfield")) {
									argument.is_bitfield = parser->get_named_attribute_value("is_bitfield").to_lower() == "true";
								}
							}

							method.arguments.push_back(argument);

						} else if (name == "description") {
							parser->read();
							if (parser->get_node_type() == XMLParser::NODE_TEXT) {
								method.description = parser->get_node_data();
							}
						}

					} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == element) {
						break;
					}
				}

				methods.push_back(method);

			} else {
				ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Invalid tag in doc file: " + parser->get_node_name() + ", expected " + element + ".");
			}

		} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == section) {
			break;
		}
	}

	return OK;
}

Error DocTools::load_classes(const String &p_dir) {
	Error err;
	Ref<DirAccess> da = DirAccess::open(p_dir, &err);
	if (da.is_null()) {
		return err;
	}

	da->list_dir_begin();
	String path;
	path = da->get_next();
	while (!path.is_empty()) {
		if (!da->current_is_dir() && path.ends_with("xml")) {
			Ref<XMLParser> parser = memnew(XMLParser);
			Error err2 = parser->open(p_dir.path_join(path));
			if (err2) {
				return err2;
			}

			_load(parser);
		}
		path = da->get_next();
	}

	da->list_dir_end();

	return OK;
}

Error DocTools::erase_classes(const String &p_dir) {
	Error err;
	Ref<DirAccess> da = DirAccess::open(p_dir, &err);
	if (da.is_null()) {
		return err;
	}

	List<String> to_erase;

	da->list_dir_begin();
	String path;
	path = da->get_next();
	while (!path.is_empty()) {
		if (!da->current_is_dir() && path.ends_with("xml")) {
			to_erase.push_back(path);
		}
		path = da->get_next();
	}
	da->list_dir_end();

	while (to_erase.size()) {
		da->remove(to_erase.front()->get());
		to_erase.pop_front();
	}

	return OK;
}

Error DocTools::_load(Ref<XMLParser> parser) {
	Error err = OK;

	while ((err = parser->read()) == OK) {
		if (parser->get_node_type() == XMLParser::NODE_ELEMENT && parser->get_node_name() == "?xml") {
			parser->skip_section();
		}

		if (parser->get_node_type() != XMLParser::NODE_ELEMENT) {
			continue; //no idea what this may be, but skipping anyway
		}

		ERR_FAIL_COND_V(parser->get_node_name() != "class", ERR_FILE_CORRUPT);

		ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
		String name = parser->get_named_attribute_value("name");
		class_list[name] = DocData::ClassDoc();
		DocData::ClassDoc &c = class_list[name];

		c.name = name;
		if (parser->has_attribute("inherits")) {
			c.inherits = parser->get_named_attribute_value("inherits");
		}

		if (parser->has_attribute("is_deprecated")) {
			c.is_deprecated = parser->get_named_attribute_value("is_deprecated").to_lower() == "true";
		}

		if (parser->has_attribute("is_experimental")) {
			c.is_experimental = parser->get_named_attribute_value("is_experimental").to_lower() == "true";
		}

		while (parser->read() == OK) {
			if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
				String name2 = parser->get_node_name();

				if (name2 == "brief_description") {
					parser->read();
					if (parser->get_node_type() == XMLParser::NODE_TEXT) {
						c.brief_description = parser->get_node_data();
					}

				} else if (name2 == "description") {
					parser->read();
					if (parser->get_node_type() == XMLParser::NODE_TEXT) {
						c.description = parser->get_node_data();
					}
				} else if (name2 == "tutorials") {
					while (parser->read() == OK) {
						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
							String name3 = parser->get_node_name();

							if (name3 == "link") {
								DocData::TutorialDoc tutorial;
								if (parser->has_attribute("title")) {
									tutorial.title = parser->get_named_attribute_value("title");
								}
								parser->read();
								if (parser->get_node_type() == XMLParser::NODE_TEXT) {
									tutorial.link = parser->get_node_data().strip_edges();
									c.tutorials.push_back(tutorial);
								}
							} else {
								ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Invalid tag in doc file: " + name3 + ".");
							}
						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "tutorials") {
							break; // End of <tutorials>.
						}
					}
				} else if (name2 == "constructors") {
					Error err2 = _parse_methods(parser, c.constructors);
					ERR_FAIL_COND_V(err2, err2);
				} else if (name2 == "methods") {
					Error err2 = _parse_methods(parser, c.methods);
					ERR_FAIL_COND_V(err2, err2);
				} else if (name2 == "operators") {
					Error err2 = _parse_methods(parser, c.operators);
					ERR_FAIL_COND_V(err2, err2);
				} else if (name2 == "signals") {
					Error err2 = _parse_methods(parser, c.signals);
					ERR_FAIL_COND_V(err2, err2);
				} else if (name2 == "annotations") {
					Error err2 = _parse_methods(parser, c.annotations);
					ERR_FAIL_COND_V(err2, err2);
				} else if (name2 == "members") {
					while (parser->read() == OK) {
						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
							String name3 = parser->get_node_name();

							if (name3 == "member") {
								DocData::PropertyDoc prop2;

								ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
								prop2.name = parser->get_named_attribute_value("name");
								ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
								prop2.type = parser->get_named_attribute_value("type");
								if (parser->has_attribute("setter")) {
									prop2.setter = parser->get_named_attribute_value("setter");
								}
								if (parser->has_attribute("getter")) {
									prop2.getter = parser->get_named_attribute_value("getter");
								}
								if (parser->has_attribute("enum")) {
									prop2.enumeration = parser->get_named_attribute_value("enum");
									if (parser->has_attribute("is_bitfield")) {
										prop2.is_bitfield = parser->get_named_attribute_value("is_bitfield").to_lower() == "true";
									}
								}
								if (parser->has_attribute("is_deprecated")) {
									prop2.is_deprecated = parser->get_named_attribute_value("is_deprecated").to_lower() == "true";
								}
								if (parser->has_attribute("is_experimental")) {
									prop2.is_experimental = parser->get_named_attribute_value("is_experimental").to_lower() == "true";
								}
								if (!parser->is_empty()) {
									parser->read();
									if (parser->get_node_type() == XMLParser::NODE_TEXT) {
										prop2.description = parser->get_node_data();
									}
								}
								c.properties.push_back(prop2);
							} else {
								ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Invalid tag in doc file: " + name3 + ".");
							}

						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "members") {
							break; // End of <members>.
						}
					}

				} else if (name2 == "theme_items") {
					while (parser->read() == OK) {
						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
							String name3 = parser->get_node_name();

							if (name3 == "theme_item") {
								DocData::ThemeItemDoc prop2;

								ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
								prop2.name = parser->get_named_attribute_value("name");
								ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
								prop2.type = parser->get_named_attribute_value("type");
								ERR_FAIL_COND_V(!parser->has_attribute("data_type"), ERR_FILE_CORRUPT);
								prop2.data_type = parser->get_named_attribute_value("data_type");
								if (!parser->is_empty()) {
									parser->read();
									if (parser->get_node_type() == XMLParser::NODE_TEXT) {
										prop2.description = parser->get_node_data();
									}
								}
								c.theme_properties.push_back(prop2);
							} else {
								ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Invalid tag in doc file: " + name3 + ".");
							}

						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "theme_items") {
							break; // End of <theme_items>.
						}
					}

				} else if (name2 == "constants") {
					while (parser->read() == OK) {
						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
							String name3 = parser->get_node_name();

							if (name3 == "constant") {
								DocData::ConstantDoc constant2;
								ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
								constant2.name = parser->get_named_attribute_value("name");
								ERR_FAIL_COND_V(!parser->has_attribute("value"), ERR_FILE_CORRUPT);
								constant2.value = parser->get_named_attribute_value("value");
								constant2.is_value_valid = true;
								if (parser->has_attribute("enum")) {
									constant2.enumeration = parser->get_named_attribute_value("enum");
									if (parser->has_attribute("is_bitfield")) {
										constant2.is_bitfield = parser->get_named_attribute_value("is_bitfield").to_lower() == "true";
									}
								}
								if (parser->has_attribute("is_deprecated")) {
									constant2.is_deprecated = parser->get_named_attribute_value("is_deprecated").to_lower() == "true";
								}
								if (parser->has_attribute("is_experimental")) {
									constant2.is_experimental = parser->get_named_attribute_value("is_experimental").to_lower() == "true";
								}
								if (!parser->is_empty()) {
									parser->read();
									if (parser->get_node_type() == XMLParser::NODE_TEXT) {
										constant2.description = parser->get_node_data();
									}
								}
								c.constants.push_back(constant2);
							} else {
								ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Invalid tag in doc file: " + name3 + ".");
							}

						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "constants") {
							break; // End of <constants>.
						}
					}

				} else {
					ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Invalid tag in doc file: " + name2 + ".");
				}

			} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "class") {
				break; // End of <class>.
			}
		}
	}

	return OK;
}

static void _write_string(Ref<FileAccess> f, int p_tablevel, const String &p_string) {
	if (p_string.is_empty()) {
		return;
	}
	String tab = String("\t").repeat(p_tablevel);
	f->store_string(tab + p_string + "\n");
}

static void _write_method_doc(Ref<FileAccess> f, const String &p_name, Vector<DocData::MethodDoc> &p_method_docs) {
	if (!p_method_docs.is_empty()) {
		p_method_docs.sort();
		_write_string(f, 1, "<" + p_name + "s>");
		for (int i = 0; i < p_method_docs.size(); i++) {
			const DocData::MethodDoc &m = p_method_docs[i];

			String qualifiers;
			if (!m.qualifiers.is_empty()) {
				qualifiers += " qualifiers=\"" + m.qualifiers.xml_escape() + "\"";
			}

			String additional_attributes;
			if (m.is_deprecated) {
				additional_attributes += " is_deprecated=\"true\"";
			}
			if (m.is_experimental) {
				additional_attributes += " is_experimental=\"true\"";
			}

			_write_string(f, 2, "<" + p_name + " name=\"" + m.name.xml_escape() + "\"" + qualifiers + additional_attributes + ">");

			if (!m.return_type.is_empty()) {
				String enum_text;
				if (!m.return_enum.is_empty()) {
					enum_text = " enum=\"" + m.return_enum + "\"";
					if (m.return_is_bitfield) {
						enum_text += " is_bitfield=\"true\"";
					}
				}
				_write_string(f, 3, "<return type=\"" + m.return_type.xml_escape(true) + "\"" + enum_text + " />");
			}
			if (m.errors_returned.size() > 0) {
				for (int j = 0; j < m.errors_returned.size(); j++) {
					_write_string(f, 3, "<returns_error number=\"" + itos(m.errors_returned[j]) + "\"/>");
				}
			}

			for (int j = 0; j < m.arguments.size(); j++) {
				const DocData::ArgumentDoc &a = m.arguments[j];

				String enum_text;
				if (!a.enumeration.is_empty()) {
					enum_text = " enum=\"" + a.enumeration + "\"";
					if (a.is_bitfield) {
						enum_text += " is_bitfield=\"true\"";
					}
				}

				if (!a.default_value.is_empty()) {
					_write_string(f, 3, "<param index=\"" + itos(j) + "\" name=\"" + a.name.xml_escape() + "\" type=\"" + a.type.xml_escape(true) + "\"" + enum_text + " default=\"" + a.default_value.xml_escape(true) + "\" />");
				} else {
					_write_string(f, 3, "<param index=\"" + itos(j) + "\" name=\"" + a.name.xml_escape() + "\" type=\"" + a.type.xml_escape(true) + "\"" + enum_text + " />");
				}
			}

			_write_string(f, 3, "<description>");
			_write_string(f, 4, _translate_doc_string(m.description).strip_edges().xml_escape());
			_write_string(f, 3, "</description>");

			_write_string(f, 2, "</" + p_name + ">");
		}

		_write_string(f, 1, "</" + p_name + "s>");
	}
}

Error DocTools::save_classes(const String &p_default_path, const HashMap<String, String> &p_class_path, bool p_include_xml_schema) {
	for (KeyValue<String, DocData::ClassDoc> &E : class_list) {
		DocData::ClassDoc &c = E.value;

		String save_path;
		if (p_class_path.has(c.name)) {
			save_path = p_class_path[c.name];
		} else {
			save_path = p_default_path;
		}

		Error err;
		String save_file = save_path.path_join(c.name.replace("\"", "").replace("/", "--") + ".xml");
		Ref<FileAccess> f = FileAccess::open(save_file, FileAccess::WRITE, &err);

		ERR_CONTINUE_MSG(err != OK, "Can't write doc file: " + save_file + ".");

		_write_string(f, 0, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>");

		String header = "<class name=\"" + c.name.xml_escape(true) + "\"";
		if (!c.inherits.is_empty()) {
			header += " inherits=\"" + c.inherits.xml_escape(true) + "\"";
			if (c.is_deprecated) {
				header += " is_deprecated=\"true\"";
			}
			if (c.is_experimental) {
				header += " is_experimental=\"true\"";
			}
		}
		if (p_include_xml_schema) {
			// Reference the XML schema so editors can provide error checking.
			// Modules are nested deep, so change the path to reference the same schema everywhere.
			const String schema_path = save_path.find("modules/") != -1 ? "../../../doc/class.xsd" : "../class.xsd";
			header += vformat(
					R"( xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="%s")",
					schema_path);
		}
		header += ">";
		_write_string(f, 0, header);

		_write_string(f, 1, "<brief_description>");
		_write_string(f, 2, _translate_doc_string(c.brief_description).strip_edges().xml_escape());
		_write_string(f, 1, "</brief_description>");

		_write_string(f, 1, "<description>");
		_write_string(f, 2, _translate_doc_string(c.description).strip_edges().xml_escape());
		_write_string(f, 1, "</description>");

		_write_string(f, 1, "<tutorials>");
		for (int i = 0; i < c.tutorials.size(); i++) {
			DocData::TutorialDoc tutorial = c.tutorials.get(i);
			String title_attribute = (!tutorial.title.is_empty()) ? " title=\"" + _translate_doc_string(tutorial.title).xml_escape() + "\"" : "";
			_write_string(f, 2, "<link" + title_attribute + ">" + tutorial.link.xml_escape() + "</link>");
		}
		_write_string(f, 1, "</tutorials>");

		_write_method_doc(f, "constructor", c.constructors);

		_write_method_doc(f, "method", c.methods);

		if (!c.properties.is_empty()) {
			_write_string(f, 1, "<members>");

			c.properties.sort();

			for (int i = 0; i < c.properties.size(); i++) {
				String additional_attributes;
				if (!c.properties[i].enumeration.is_empty()) {
					additional_attributes += " enum=\"" + c.properties[i].enumeration + "\"";
					if (c.properties[i].is_bitfield) {
						additional_attributes += " is_bitfield=\"true\"";
					}
				}
				if (!c.properties[i].default_value.is_empty()) {
					additional_attributes += " default=\"" + c.properties[i].default_value.xml_escape(true) + "\"";
				}
				if (c.properties[i].is_deprecated) {
					additional_attributes += " is_deprecated=\"true\"";
				}
				if (c.properties[i].is_experimental) {
					additional_attributes += " is_experimental=\"true\"";
				}

				const DocData::PropertyDoc &p = c.properties[i];

				if (c.properties[i].overridden) {
					_write_string(f, 2, "<member name=\"" + p.name + "\" type=\"" + p.type.xml_escape(true) + "\" setter=\"" + p.setter + "\" getter=\"" + p.getter + "\" overrides=\"" + p.overrides + "\"" + additional_attributes + " />");
				} else {
					_write_string(f, 2, "<member name=\"" + p.name + "\" type=\"" + p.type.xml_escape(true) + "\" setter=\"" + p.setter + "\" getter=\"" + p.getter + "\"" + additional_attributes + ">");
					_write_string(f, 3, _translate_doc_string(p.description).strip_edges().xml_escape());
					_write_string(f, 2, "</member>");
				}
			}
			_write_string(f, 1, "</members>");
		}

		_write_method_doc(f, "signal", c.signals);

		if (!c.constants.is_empty()) {
			_write_string(f, 1, "<constants>");
			for (int i = 0; i < c.constants.size(); i++) {
				const DocData::ConstantDoc &k = c.constants[i];

				String additional_attributes;
				if (c.constants[i].is_deprecated) {
					additional_attributes += " is_deprecated=\"true\"";
				}
				if (c.constants[i].is_experimental) {
					additional_attributes += " is_experimental=\"true\"";
				}

				if (k.is_value_valid) {
					if (!k.enumeration.is_empty()) {
						if (k.is_bitfield) {
							_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"" + k.value.xml_escape(true) + "\" enum=\"" + k.enumeration + "\" is_bitfield=\"true\"" + additional_attributes + ">");
						} else {
							_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"" + k.value.xml_escape(true) + "\" enum=\"" + k.enumeration + "\"" + additional_attributes + ">");
						}
					} else {
						_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"" + k.value.xml_escape(true) + "\"" + additional_attributes + ">");
					}
				} else {
					if (!k.enumeration.is_empty()) {
						_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"platform-dependent\" enum=\"" + k.enumeration + "\"" + additional_attributes + ">");
					} else {
						_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"platform-dependent\"" + additional_attributes + ">");
					}
				}
				_write_string(f, 3, _translate_doc_string(k.description).strip_edges().xml_escape());
				_write_string(f, 2, "</constant>");
			}

			_write_string(f, 1, "</constants>");
		}

		_write_method_doc(f, "annotation", c.annotations);

		if (!c.theme_properties.is_empty()) {
			c.theme_properties.sort();

			_write_string(f, 1, "<theme_items>");
			for (int i = 0; i < c.theme_properties.size(); i++) {
				const DocData::ThemeItemDoc &ti = c.theme_properties[i];

				if (!ti.default_value.is_empty()) {
					_write_string(f, 2, "<theme_item name=\"" + ti.name + "\" data_type=\"" + ti.data_type + "\" type=\"" + ti.type + "\" default=\"" + ti.default_value.xml_escape(true) + "\">");
				} else {
					_write_string(f, 2, "<theme_item name=\"" + ti.name + "\" data_type=\"" + ti.data_type + "\" type=\"" + ti.type + "\">");
				}

				_write_string(f, 3, _translate_doc_string(ti.description).strip_edges().xml_escape());

				_write_string(f, 2, "</theme_item>");
			}
			_write_string(f, 1, "</theme_items>");
		}

		_write_method_doc(f, "operator", c.operators);

		_write_string(f, 0, "</class>");
	}

	return OK;
}

Error DocTools::load_compressed(const uint8_t *p_data, int p_compressed_size, int p_uncompressed_size) {
	Vector<uint8_t> data;
	data.resize(p_uncompressed_size);
	int ret = Compression::decompress(data.ptrw(), p_uncompressed_size, p_data, p_compressed_size, Compression::MODE_DEFLATE);
	ERR_FAIL_COND_V_MSG(ret == -1, ERR_FILE_CORRUPT, "Compressed file is corrupt.");
	class_list.clear();

	Ref<XMLParser> parser = memnew(XMLParser);
	Error err = parser->open_buffer(data);
	if (err) {
		return err;
	}

	_load(parser);

	return OK;
}
