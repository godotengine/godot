/**************************************************************************/
/*  class_db_api_json.cpp                                                 */
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

#include "class_db_api_json.h"

#ifdef DEBUG_METHODS_ENABLED

#include "core/io/json.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "core/version.h"

void class_db_api_to_json(const String &p_output_file, ClassDB::APIType p_api) {
	Dictionary classes_dict;

	List<StringName> names;

	const StringName *k = NULL;

	while ((k = ClassDB::classes.next(k))) {
		names.push_back(*k);
	}
	//must be alphabetically sorted for hash to compute
	names.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		ClassDB::ClassInfo *t = ClassDB::classes.getptr(E->get());
		ERR_FAIL_COND(!t);
		if (t->api != p_api || !t->exposed)
			continue;

		Dictionary class_dict;
		classes_dict[t->name] = class_dict;

		class_dict["inherits"] = t->inherits;

		{ //methods

			List<StringName> snames;

			k = NULL;

			while ((k = t->method_map.next(k))) {
				String name = k->operator String();

				ERR_CONTINUE(name.empty());

				if (name[0] == '_')
					continue; // Ignore non-virtual methods that start with an underscore

				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			Array methods;

			for (List<StringName>::Element *F = snames.front(); F; F = F->next()) {
				Dictionary method_dict;
				methods.push_back(method_dict);

				MethodBind *mb = t->method_map[F->get()];
				method_dict["name"] = mb->get_name();
				method_dict["argument_count"] = mb->get_argument_count();
				method_dict["return_type"] = mb->get_argument_type(-1);

				Array arguments;
				method_dict["arguments"] = arguments;

				for (int i = 0; i < mb->get_argument_count(); i++) {
					Dictionary argument_dict;
					arguments.push_back(argument_dict);
					const PropertyInfo info = mb->get_argument_info(i);
					argument_dict["type"] = info.type;
					argument_dict["name"] = info.name;
					argument_dict["hint"] = info.hint;
					argument_dict["hint_string"] = info.hint_string;
				}

				method_dict["default_argument_count"] = mb->get_default_argument_count();

				Array default_arguments;
				method_dict["default_arguments"] = default_arguments;

				for (int i = 0; i < mb->get_default_argument_count(); i++) {
					Dictionary default_argument_dict;
					default_arguments.push_back(default_argument_dict);
					//hash should not change, i hope for tis
					Variant da = mb->get_default_argument(i);
					default_argument_dict["value"] = da;
				}

				method_dict["hint_flags"] = mb->get_hint_flags();
			}

			if (!methods.empty()) {
				class_dict["methods"] = methods;
			}
		}

		{ //constants

			List<StringName> snames;

			k = NULL;

			while ((k = t->constant_map.next(k))) {
				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			Array constants;

			for (List<StringName>::Element *F = snames.front(); F; F = F->next()) {
				Dictionary constant_dict;
				constants.push_back(constant_dict);

				constant_dict["name"] = F->get();
				constant_dict["value"] = t->constant_map[F->get()];
			}

			if (!constants.empty()) {
				class_dict["constants"] = constants;
			}
		}

		{ //signals

			List<StringName> snames;

			k = NULL;

			while ((k = t->signal_map.next(k))) {
				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			Array signals;

			for (List<StringName>::Element *F = snames.front(); F; F = F->next()) {
				Dictionary signal_dict;
				signals.push_back(signal_dict);

				MethodInfo &mi = t->signal_map[F->get()];
				signal_dict["name"] = F->get();

				Array arguments;
				signal_dict["arguments"] = arguments;
				for (int i = 0; i < mi.arguments.size(); i++) {
					Dictionary argument_dict;
					arguments.push_back(argument_dict);
					argument_dict["type"] = mi.arguments[i].type;
				}
			}

			if (!signals.empty()) {
				class_dict["signals"] = signals;
			}
		}

		{ //properties

			List<StringName> snames;

			k = NULL;

			while ((k = t->property_setget.next(k))) {
				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			Array properties;

			for (List<StringName>::Element *F = snames.front(); F; F = F->next()) {
				Dictionary property_dict;
				properties.push_back(property_dict);

				ClassDB::PropertySetGet *psg = t->property_setget.getptr(F->get());

				property_dict["name"] = F->get();
				property_dict["setter"] = psg->setter;
				property_dict["getter"] = psg->getter;
			}

			if (!properties.empty()) {
				class_dict["property_setget"] = properties;
			}
		}

		Array property_list;

		//property list
		for (List<PropertyInfo>::Element *F = t->property_list.front(); F; F = F->next()) {
			Dictionary property_dict;
			property_list.push_back(property_dict);

			property_dict["name"] = F->get().name;
			property_dict["type"] = F->get().type;
			property_dict["hint"] = F->get().hint;
			property_dict["hint_string"] = F->get().hint_string;
			property_dict["usage"] = F->get().usage;
		}

		if (!property_list.empty()) {
			class_dict["property_list"] = property_list;
		}
	}

	FileAccessRef f = FileAccess::open(p_output_file, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(!f, "Cannot open file '" + p_output_file + "'.");
	f->store_string(JSON::print(classes_dict, /*indent: */ "\t"));
	f->close();

	print_line(String() + "ClassDB API JSON written to: " + ProjectSettings::get_singleton()->globalize_path(p_output_file));
}

#endif // DEBUG_METHODS_ENABLED
