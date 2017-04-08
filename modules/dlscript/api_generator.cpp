#include "api_generator.h"

#ifdef TOOLS_ENABLED

#include "class_db.h"
#include "core/global_config.h"
#include "os/file_access.h"

// helper stuff

static Error save_file(const String &p_path, const List<String> &p_content) {

	FileAccessRef file = FileAccess::open(p_path, FileAccess::WRITE);

	ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);

	for (const List<String>::Element *e = p_content.front(); e != NULL; e = e->next()) {
		file->store_string(e->get());
	}

	file->close();

	return OK;
}

// helper stuff end

struct MethodAPI {
	String method_name;
	String return_type;

	List<String> argument_types;
	List<String> argument_names;

	Map<int, Variant> default_arguments;

	int argument_count;
	bool has_varargs;
	bool is_editor;
	bool is_noscript;
	bool is_const;
	bool is_reverse;
	bool is_virtual;
	bool is_from_script;
};

struct PropertyAPI {
	String name;
	String getter;
	String setter;
	String type;
};

struct ConstantAPI {
	String constant_name;
	int constant_value;
};

struct SignalAPI {
	String name;
	List<String> argument_types;
	List<String> argument_names;
	Map<int, Variant> default_arguments;
};

struct ClassAPI {
	String class_name;
	String super_class_name;

	ClassDB::APIType api_type;

	bool is_singleton;
	bool is_instanciable;
	// @Unclear
	bool is_creatable;
	// @Unclear
	bool memory_own;

	List<MethodAPI> methods;
	List<PropertyAPI> properties;
	List<ConstantAPI> constants;
	List<SignalAPI> signals_;
};

/*
 * Reads the entire Godot API to a list
 */
List<ClassAPI> generate_c_api_classes() {

	List<ClassAPI> api;

	List<StringName> classes;
	ClassDB::get_class_list(&classes);

	for (List<StringName>::Element *e = classes.front(); e != NULL; e = e->next()) {
		StringName class_name = e->get();

		ClassAPI class_api;
		class_api.api_type = ClassDB::get_api_type(e->get());
		class_api.class_name = class_name;
		class_api.super_class_name = ClassDB::get_parent_class(class_name);
		{
			String name = class_name;
			if (name.begins_with("_")) {
				name.remove(0);
			}
			class_api.is_singleton = GlobalConfig::get_singleton()->has_singleton(name);
		}
		class_api.is_instanciable = !class_api.is_singleton && ClassDB::can_instance(class_name);

		{
			bool is_reference = false;
			List<StringName> inheriters;
			ClassDB::get_inheriters_from_class("Reference", &inheriters);
			is_reference = !!inheriters.find(class_name);
			// @Unclear
			class_api.memory_own = !class_api.is_singleton && is_reference;
		}

		// constants
		{
			List<String> constant;
			ClassDB::get_integer_constant_list(class_name, &constant, true);
			for (List<String>::Element *c = constant.front(); c != NULL; c = c->next()) {
				ConstantAPI constant_api;
				constant_api.constant_name = c->get();
				constant_api.constant_value = ClassDB::get_integer_constant(class_name, c->get());

				class_api.constants.push_back(constant_api);
			}
		}

		// signals
		{
			List<MethodInfo> signals_;
			ClassDB::get_signal_list(class_name, &signals_, true);

			for (int i = 0; i < signals_.size(); i++) {
				SignalAPI signal;

				MethodInfo method_info = signals_[i];
				signal.name = method_info.name;

				for (int j = 0; j < method_info.arguments.size(); j++) {
					PropertyInfo argument = method_info.arguments[j];
					String type;
					String name = argument.name;

					if (argument.name.find(":") != -1) {
						type = argument.name.get_slice(":", 1);
						name = argument.name.get_slice(":", 0);
					} else if (argument.hint == PROPERTY_HINT_RESOURCE_TYPE) {
						type = argument.hint_string;
					} else if (argument.type == Variant::NIL) {
						type = "Variant";
					} else {
						type = Variant::get_type_name(argument.type);
					}

					signal.argument_names.push_back(name);
					signal.argument_types.push_back(type);
				}

				Vector<Variant> default_arguments = method_info.default_arguments;

				int default_start = signal.argument_names.size() - default_arguments.size();

				for (int j = 0; j < default_arguments.size(); j++) {
					signal.default_arguments[default_start + j] = default_arguments[j];
				}

				class_api.signals_.push_back(signal);
			}
		}

		//properties
		{
			List<PropertyInfo> properties;
			ClassDB::get_property_list(class_name, &properties, true);

			for (List<PropertyInfo>::Element *p = properties.front(); p != NULL; p = p->next()) {
				PropertyAPI property_api;

				property_api.name = p->get().name;
				property_api.getter = ClassDB::get_property_getter(class_name, p->get().name);
				property_api.setter = ClassDB::get_property_setter(class_name, p->get().name);

				if (p->get().name.find(":") != -1) {
					property_api.type = p->get().name.get_slice(":", 1);
					property_api.name = p->get().name.get_slice(":", 0);
				} else if (p->get().hint == PROPERTY_HINT_RESOURCE_TYPE) {
					property_api.type = p->get().hint_string;
				} else if (p->get().type == Variant::NIL) {
					property_api.type = "Variant";
				} else {
					property_api.type = Variant::get_type_name(p->get().type);
				}

				if (!property_api.setter.empty() || !property_api.getter.empty()) {
					class_api.properties.push_back(property_api);
				}
			}
		}

		//methods
		{
			List<MethodInfo> methods;
			ClassDB::get_method_list(class_name, &methods, true);

			for (List<MethodInfo>::Element *m = methods.front(); m != NULL; m = m->next()) {
				MethodAPI method_api;
				MethodBind *method_bind = ClassDB::get_method(class_name, m->get().name);
				MethodInfo &method_info = m->get();

				//method name
				method_api.method_name = m->get().name;
				//method return type
				if (method_bind && method_bind->get_return_type() != StringName()) {
					method_api.return_type = method_bind->get_return_type();
				} else if (method_api.method_name.find(":") != -1) {
					method_api.return_type = method_api.method_name.get_slice(":", 1);
					method_api.method_name = method_api.method_name.get_slice(":", 0);
				} else if (m->get().return_val.type != Variant::NIL) {
					method_api.return_type = m->get().return_val.hint == PROPERTY_HINT_RESOURCE_TYPE ? m->get().return_val.hint_string : Variant::get_type_name(m->get().return_val.type);
				} else {
					method_api.return_type = "void";
				}

				method_api.argument_count = method_info.arguments.size();
				method_api.has_varargs = method_bind && method_bind->is_vararg();

				// Method flags
				if (method_info.flags) {
					const uint32_t flags = method_info.flags;
					method_api.is_editor = flags & METHOD_FLAG_EDITOR;
					method_api.is_noscript = flags & METHOD_FLAG_NOSCRIPT;
					method_api.is_const = flags & METHOD_FLAG_CONST;
					method_api.is_reverse = flags & METHOD_FLAG_REVERSE;
					method_api.is_virtual = flags & METHOD_FLAG_VIRTUAL;
					method_api.is_from_script = flags & METHOD_FLAG_FROM_SCRIPT;
				}

				method_api.is_virtual = method_api.is_virtual || method_api.method_name[0] == '_';

				// method argument name and type

				for (int i = 0; i < method_api.argument_count; i++) {
					String arg_name;
					String arg_type;
					PropertyInfo arg_info = method_info.arguments[i];

					arg_name = arg_info.name;

					if (arg_info.name.find(":") != -1) {
						arg_type = arg_info.name.get_slice(":", 1);
						arg_name = arg_info.name.get_slice(":", 0);
					} else if (arg_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
						arg_type = arg_info.hint_string;
					} else if (arg_info.type == Variant::NIL) {
						arg_type = "Variant";
					} else {
						arg_type = Variant::get_type_name(arg_info.type);
					}

					method_api.argument_names.push_back(arg_name);
					method_api.argument_types.push_back(arg_type);

					if (method_bind && method_bind->has_default_argument(i)) {
						method_api.default_arguments[i] = method_bind->get_default_argument(i);
					}
				}

				class_api.methods.push_back(method_api);
			}
		}

		api.push_back(class_api);
	}

	return api;
}

/*
 * Generates the JSON source from the API in p_api
 */
static List<String> generate_c_api_json(const List<ClassAPI> &p_api) {

	// I'm sorry for the \t mess

	List<String> source;

	source.push_back("[\n");

	for (const List<ClassAPI>::Element *c = p_api.front(); c != NULL; c = c->next()) {
		ClassAPI api = c->get();

		source.push_back("\t{\n");

		source.push_back("\t\t\"name\": \"" + api.class_name + "\",\n");
		source.push_back("\t\t\"base_class\": \"" + api.super_class_name + "\",\n");
		source.push_back(String("\t\t\"api_type\": \"") + (api.api_type == ClassDB::API_CORE ? "core" : (api.api_type == ClassDB::API_EDITOR ? "tools" : "none")) + "\",\n");
		source.push_back(String("\t\t\"singleton\": ") + (api.is_singleton ? "true" : "false") + ",\n");
		source.push_back(String("\t\t\"instanciable\": ") + (api.is_instanciable ? "true" : "false") + ",\n");
		// @Unclear
		// source.push_back(String("\t\t\"createable\": ") + (api.is_creatable ? "true" : "false") + ",\n");

		source.push_back("\t\t\"constants\": {\n");
		for (List<ConstantAPI>::Element *e = api.constants.front(); e; e = e->next()) {
			source.push_back("\t\t\t\"" + e->get().constant_name + "\": " + String::num_int64(e->get().constant_value) + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t},\n");

		source.push_back("\t\t\"properties\": [\n");
		for (List<PropertyAPI>::Element *e = api.properties.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().name + "\",\n");
			source.push_back("\t\t\t\t\"type\": \"" + e->get().type + "\",\n");
			source.push_back("\t\t\t\t\"getter\": \"" + e->get().getter + "\",\n");
			source.push_back("\t\t\t\t\"setter\": \"" + e->get().setter + "\"\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t],\n");

		source.push_back("\t\t\"signals\": [\n");
		for (List<SignalAPI>::Element *e = api.signals_.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().name + "\",\n");
			source.push_back("\t\t\t\t\"arguments\": [\n");
			for (int i = 0; i < e->get().argument_names.size(); i++) {
				source.push_back("\t\t\t\t\t{\n");
				source.push_back("\t\t\t\t\t\t\"name\": \"" + e->get().argument_names[i] + "\",\n");
				source.push_back("\t\t\t\t\t\t\"type\": \"" + e->get().argument_types[i] + "\",\n");
				source.push_back("\t\t\t\t\t\t\"default_value\": \"" + (e->get().default_arguments.has(i) ? (String)e->get().default_arguments[i] : "") + "\"\n");
				source.push_back(String("\t\t\t\t\t}") + ((i < e->get().argument_names.size() - 1) ? "," : "") + "\n");
			}
			source.push_back("\t\t\t\t]\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t],\n");

		source.push_back("\t\t\"methods\": [\n");
		for (List<MethodAPI>::Element *e = api.methods.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().method_name + "\",\n");
			source.push_back("\t\t\t\t\"return_type\": \"" + e->get().return_type + "\",\n");
			source.push_back(String("\t\t\t\t\"is_editor\": ") + (e->get().is_editor ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_noscript\": ") + (e->get().is_noscript ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_const\": ") + (e->get().is_const ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_reverse\": ") + (e->get().is_reverse ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_virtual\": ") + (e->get().is_virtual ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"has_varargs\": ") + (e->get().has_varargs ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_from_script\": ") + (e->get().is_from_script ? "true" : "false") + ",\n");
			source.push_back("\t\t\t\t\"arguments\": [\n");
			for (int i = 0; i < e->get().argument_names.size(); i++) {
				source.push_back("\t\t\t\t\t{\n");
				source.push_back("\t\t\t\t\t\t\"name\": \"" + e->get().argument_names[i] + "\",\n");
				source.push_back("\t\t\t\t\t\t\"type\": \"" + e->get().argument_types[i] + "\",\n");
				source.push_back("\t\t\t\t\t\t\"default_value\": \"" + (e->get().default_arguments.has(i) ? (String)e->get().default_arguments[i] : "") + "\"\n");
				source.push_back(String("\t\t\t\t\t}") + ((i < e->get().argument_names.size() - 1) ? "," : "") + "\n");
			}
			source.push_back("\t\t\t\t]\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t]\n");

		source.push_back(String("\t}") + (c->next() ? "," : "") + "\n");
	}

	source.push_back("]");

	return source;
}

//

#endif

/*
 * Saves the whole Godot API to a JSON file located at
 *  p_path
 */
Error generate_c_api(const String &p_path) {

#ifndef TOOLS_ENABLED
	return ERR_BUG;
#else

	List<ClassAPI> api = generate_c_api_classes();

	List<String> json_source = generate_c_api_json(api);

	return save_file(p_path, json_source);
#endif
}
