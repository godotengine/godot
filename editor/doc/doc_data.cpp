/*************************************************************************/
/*  doc_data.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "doc_data.h"

#include "engine.h"
#include "global_constants.h"
#include "io/compression.h"
#include "io/marshalls.h"
#include "os/dir_access.h"
#include "project_settings.h"
#include "scene/resources/theme.h"
#include "script_language.h"
#include "version.h"

void DocData::merge_from(const DocData &p_data) {

	for (Map<String, ClassDoc>::Element *E = class_list.front(); E; E = E->next()) {

		ClassDoc &c = E->get();

		if (!p_data.class_list.has(c.name))
			continue;

		const ClassDoc &cf = p_data.class_list[c.name];

		c.description = cf.description;
		c.brief_description = cf.brief_description;
		c.tutorials = cf.tutorials;
		c.demos = cf.demos;

		for (int i = 0; i < c.methods.size(); i++) {

			MethodDoc &m = c.methods[i];

			for (int j = 0; j < cf.methods.size(); j++) {

				if (cf.methods[j].name != m.name)
					continue;
				if (cf.methods[j].arguments.size() != m.arguments.size())
					continue;
				// since polymorphic functions are allowed we need to check the type of
				// the arguments so we make sure they are different.
				int arg_count = cf.methods[j].arguments.size();
				Vector<bool> arg_used;
				arg_used.resize(arg_count);
				for (int l = 0; l < arg_count; ++l)
					arg_used[l] = false;
				// also there is no guarantee that argument ordering will match, so we
				// have to check one by one so we make sure we have an exact match
				for (int k = 0; k < arg_count; ++k) {
					for (int l = 0; l < arg_count; ++l)
						if (cf.methods[j].arguments[k].type == m.arguments[l].type && !arg_used[l]) {
							arg_used[l] = true;
							break;
						}
				}
				bool not_the_same = false;
				for (int l = 0; l < arg_count; ++l)
					if (!arg_used[l]) // at least one of the arguments was different
						not_the_same = true;
				if (not_the_same)
					continue;

				const MethodDoc &mf = cf.methods[j];

				m.description = mf.description;
				break;
			}
		}

		for (int i = 0; i < c.signals.size(); i++) {

			MethodDoc &m = c.signals[i];

			for (int j = 0; j < cf.signals.size(); j++) {

				if (cf.signals[j].name != m.name)
					continue;
				const MethodDoc &mf = cf.signals[j];

				m.description = mf.description;
				break;
			}
		}

		for (int i = 0; i < c.constants.size(); i++) {

			ConstantDoc &m = c.constants[i];

			for (int j = 0; j < cf.constants.size(); j++) {

				if (cf.constants[j].name != m.name)
					continue;
				const ConstantDoc &mf = cf.constants[j];

				m.description = mf.description;
				break;
			}
		}

		for (int i = 0; i < c.properties.size(); i++) {

			PropertyDoc &p = c.properties[i];

			for (int j = 0; j < cf.properties.size(); j++) {

				if (cf.properties[j].name != p.name)
					continue;
				const PropertyDoc &pf = cf.properties[j];

				p.description = pf.description;
				p.setter = pf.setter;
				p.getter = pf.getter;

				break;
			}
		}

		for (int i = 0; i < c.theme_properties.size(); i++) {

			PropertyDoc &p = c.theme_properties[i];

			for (int j = 0; j < cf.theme_properties.size(); j++) {

				if (cf.theme_properties[j].name != p.name)
					continue;
				const PropertyDoc &pf = cf.theme_properties[j];

				p.description = pf.description;
				break;
			}
		}
	}
}

void DocData::remove_from(const DocData &p_data) {
	for (Map<String, ClassDoc>::Element *E = p_data.class_list.front(); E; E = E->next()) {
		if (class_list.has(E->key()))
			class_list.erase(E->key());
	}
}

static void return_doc_from_retinfo(DocData::MethodDoc &p_method, const PropertyInfo &p_retinfo) {

	if (p_retinfo.type == Variant::INT && p_retinfo.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
		p_method.return_enum = p_retinfo.class_name;
		if (p_method.return_enum.begins_with("_")) //proxy class
			p_method.return_enum = p_method.return_enum.substr(1, p_method.return_enum.length());
		p_method.return_type = "int";
	} else if (p_retinfo.class_name != StringName()) {
		p_method.return_type = p_retinfo.class_name;
	} else if (p_retinfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		p_method.return_type = p_retinfo.hint_string;
	} else if (p_retinfo.type == Variant::NIL && p_retinfo.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
		p_method.return_type = "Variant";
	} else if (p_retinfo.type == Variant::NIL) {
		p_method.return_type = "void";
	} else {
		p_method.return_type = Variant::get_type_name(p_retinfo.type);
	}
}

static void argument_doc_from_arginfo(DocData::ArgumentDoc &p_argument, const PropertyInfo &p_arginfo) {

	p_argument.name = p_arginfo.name;

	if (p_arginfo.type == Variant::INT && p_arginfo.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
		p_argument.enumeration = p_arginfo.class_name;
		if (p_argument.enumeration.begins_with("_")) //proxy class
			p_argument.enumeration = p_argument.enumeration.substr(1, p_argument.enumeration.length());
		p_argument.type = "int";
	} else if (p_arginfo.class_name != StringName()) {
		p_argument.type = p_arginfo.class_name;
	} else if (p_arginfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		p_argument.type = p_arginfo.hint_string;
	} else if (p_arginfo.type == Variant::NIL) {
		// Parameters cannot be void, so PROPERTY_USAGE_NIL_IS_VARIANT is not necessary
		p_argument.type = "Variant";
	} else {
		p_argument.type = Variant::get_type_name(p_arginfo.type);
	}
}

void DocData::generate(bool p_basic_types) {

	List<StringName> classes;
	ClassDB::get_class_list(&classes);
	classes.sort_custom<StringName::AlphCompare>();

	bool skip_setter_getter_methods = true;

	while (classes.size()) {

		Set<StringName> setters_getters;

		String name = classes.front()->get();
		String cname = name;
		if (cname.begins_with("_")) //proxy class
			cname = cname.substr(1, name.length());

		class_list[cname] = ClassDoc();
		ClassDoc &c = class_list[cname];
		c.name = cname;
		c.inherits = ClassDB::get_parent_class(name);
		c.category = ClassDB::get_category(name);

		List<PropertyInfo> properties;
		ClassDB::get_property_list(name, &properties, true);

		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			if (E->get().usage & PROPERTY_USAGE_GROUP || E->get().usage & PROPERTY_USAGE_CATEGORY || E->get().usage & PROPERTY_USAGE_INTERNAL)
				continue;

			PropertyDoc prop;
			StringName setter = ClassDB::get_property_setter(name, E->get().name);
			StringName getter = ClassDB::get_property_getter(name, E->get().name);

			prop.name = E->get().name;
			prop.setter = setter;
			prop.getter = getter;

			bool found_type = false;
			if (getter != StringName()) {
				MethodBind *mb = ClassDB::get_method(name, getter);
				if (mb) {
					PropertyInfo retinfo = mb->get_return_info();

					found_type = true;
					if (retinfo.type == Variant::INT && retinfo.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
						prop.enumeration = retinfo.class_name;
						prop.type = "int";
					} else if (retinfo.class_name != StringName()) {
						prop.type = retinfo.class_name;
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

				if (E->get().type == Variant::OBJECT && E->get().hint == PROPERTY_HINT_RESOURCE_TYPE)
					prop.type = E->get().hint_string;
				else
					prop.type = Variant::get_type_name(E->get().type);
			}

			c.properties.push_back(prop);
		}

		List<MethodInfo> method_list;
		ClassDB::get_method_list(name, &method_list, true);
		method_list.sort();

		for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {

			if (E->get().name == "" || (E->get().name[0] == '_' && !(E->get().flags & METHOD_FLAG_VIRTUAL)))
				continue; //hidden, don't count

			if (skip_setter_getter_methods && setters_getters.has(E->get().name) && E->get().name.find("/") == -1)
				continue;

			MethodDoc method;

			method.name = E->get().name;

			if (E->get().flags & METHOD_FLAG_VIRTUAL)
				method.qualifiers = "virtual";

			if (E->get().flags & METHOD_FLAG_CONST) {
				if (method.qualifiers != "")
					method.qualifiers += " ";
				method.qualifiers += "const";
			} else if (E->get().flags & METHOD_FLAG_VARARG) {
				if (method.qualifiers != "")
					method.qualifiers += " ";
				method.qualifiers += "vararg";
			}

			for (int i = -1; i < E->get().arguments.size(); i++) {

				if (i == -1) {
#ifdef DEBUG_METHODS_ENABLED
					return_doc_from_retinfo(method, E->get().return_val);
#endif
				} else {

					const PropertyInfo &arginfo = E->get().arguments[i];

					ArgumentDoc argument;

					argument_doc_from_arginfo(argument, arginfo);

					int darg_idx = i - (E->get().arguments.size() - E->get().default_arguments.size());

					if (darg_idx >= 0) {
						Variant default_arg = E->get().default_arguments[darg_idx];
						argument.default_value = default_arg.get_construct_string();
					}

					method.arguments.push_back(argument);
				}

				/*
				String hint;
				switch(arginfo.hint) {
					case PROPERTY_HINT_DIR: hint="A directory."; break;
					case PROPERTY_HINT_RANGE: hint="Range - min: "+arginfo.hint_string.get_slice(",",0)+" max: "+arginfo.hint_string.get_slice(",",1)+" step: "+arginfo.hint_string.get_slice(",",2); break;
					case PROPERTY_HINT_ENUM: hint="Values: "; for(int j=0;j<arginfo.hint_string.get_slice_count(",");j++) { if (j>0) hint+=", "; hint+=arginfo.hint_string.get_slice(",",j)+"="+itos(j); } break;
					case PROPERTY_HINT_LENGTH: hint="Length: "+arginfo.hint_string; break;
					case PROPERTY_HINT_FLAGS: hint="Values: "; for(int j=0;j<arginfo.hint_string.get_slice_count(",");j++) { if (j>0) hint+=", "; hint+=arginfo.hint_string.get_slice(",",j)+"="+itos(1<<j); } break;
					case PROPERTY_HINT_FILE: hint="A file:"; break;
					//case PROPERTY_HINT_RESOURCE_TYPE: hint="Type: "+arginfo.hint_string; break;
				};
				if (hint!="")
					_write_string(f,4,hint);
*/
			}

			c.methods.push_back(method);
		}

		List<MethodInfo> signal_list;
		ClassDB::get_signal_list(name, &signal_list, true);

		if (signal_list.size()) {

			for (List<MethodInfo>::Element *EV = signal_list.front(); EV; EV = EV->next()) {

				MethodDoc signal;
				signal.name = EV->get().name;
				for (int i = 0; i < EV->get().arguments.size(); i++) {

					PropertyInfo arginfo = EV->get().arguments[i];
					ArgumentDoc argument;
					argument.name = arginfo.name;
					argument.type = Variant::get_type_name(arginfo.type);
					signal.arguments.push_back(argument);
				}

				c.signals.push_back(signal);
			}
		}

		List<String> constant_list;
		ClassDB::get_integer_constant_list(name, &constant_list, true);

		for (List<String>::Element *E = constant_list.front(); E; E = E->next()) {

			ConstantDoc constant;
			constant.name = E->get();
			constant.value = itos(ClassDB::get_integer_constant(name, E->get()));
			constant.enumeration = ClassDB::get_integer_constant_enum(name, E->get());
			c.constants.push_back(constant);
		}

		//theme stuff

		{
			List<StringName> l;
			Theme::get_default()->get_constant_list(cname, &l);
			for (List<StringName>::Element *E = l.front(); E; E = E->next()) {

				PropertyDoc pd;
				pd.name = E->get();
				pd.type = "int";
				c.theme_properties.push_back(pd);
			}

			l.clear();
			Theme::get_default()->get_color_list(cname, &l);
			for (List<StringName>::Element *E = l.front(); E; E = E->next()) {

				PropertyDoc pd;
				pd.name = E->get();
				pd.type = "Color";
				c.theme_properties.push_back(pd);
			}

			l.clear();
			Theme::get_default()->get_icon_list(cname, &l);
			for (List<StringName>::Element *E = l.front(); E; E = E->next()) {

				PropertyDoc pd;
				pd.name = E->get();
				pd.type = "Texture";
				c.theme_properties.push_back(pd);
			}
			l.clear();
			Theme::get_default()->get_font_list(cname, &l);
			for (List<StringName>::Element *E = l.front(); E; E = E->next()) {

				PropertyDoc pd;
				pd.name = E->get();
				pd.type = "Font";
				c.theme_properties.push_back(pd);
			}
			l.clear();
			Theme::get_default()->get_stylebox_list(cname, &l);
			for (List<StringName>::Element *E = l.front(); E; E = E->next()) {

				PropertyDoc pd;
				pd.name = E->get();
				pd.type = "StyleBox";
				c.theme_properties.push_back(pd);
			}
		}

		classes.pop_front();
	}

	{
		//so it can be documented that it does not exist
		class_list["Variant"] = ClassDoc();
		class_list["Variant"].name = "Variant";
	}

	if (!p_basic_types)
		return;

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {

		if (i == Variant::OBJECT)
			continue; //use the core type instead

		String cname = Variant::get_type_name(Variant::Type(i));

		class_list[cname] = ClassDoc();
		ClassDoc &c = class_list[cname];
		c.name = cname;
		c.category = "Built-In Types";

		Variant::CallError cerror;
		Variant v = Variant::construct(Variant::Type(i), NULL, 0, cerror);

		List<MethodInfo> method_list;
		v.get_method_list(&method_list);
		method_list.sort();
		Variant::get_constructor_list(Variant::Type(i), &method_list);

		for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {

			MethodInfo &mi = E->get();
			MethodDoc method;

			method.name = mi.name;

			for (int i = 0; i < mi.arguments.size(); i++) {

				PropertyInfo arginfo = mi.arguments[i];

				ArgumentDoc ad;
				ad.name = arginfo.name;

				if (arginfo.type == Variant::NIL)
					ad.type = "var";
				else
					ad.type = Variant::get_type_name(arginfo.type);

				int defarg = mi.default_arguments.size() - mi.arguments.size() + i;
				if (defarg >= 0)
					ad.default_value = mi.default_arguments[defarg];

				method.arguments.push_back(ad);
			}

			if (mi.return_val.type == Variant::NIL) {
				if (mi.return_val.name != "")
					method.return_type = "var";
			} else {
				method.return_type = Variant::get_type_name(mi.return_val.type);
			}

			c.methods.push_back(method);
		}

		List<PropertyInfo> properties;
		v.get_property_list(&properties);
		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {

			PropertyInfo pi = E->get();
			PropertyDoc property;
			property.name = pi.name;
			property.type = Variant::get_type_name(pi.type);

			c.properties.push_back(property);
		}

		List<StringName> constants;
		Variant::get_numeric_constants_for_type(Variant::Type(i), &constants);

		for (List<StringName>::Element *E = constants.front(); E; E = E->next()) {

			ConstantDoc constant;
			constant.name = E->get();
			constant.value = itos(Variant::get_numeric_constant_value(Variant::Type(i), E->get()));
			c.constants.push_back(constant);
		}
	}

	//built in constants and functions

	{

		String cname = "@GlobalScope";
		class_list[cname] = ClassDoc();
		ClassDoc &c = class_list[cname];
		c.name = cname;

		for (int i = 0; i < GlobalConstants::get_global_constant_count(); i++) {

			ConstantDoc cd;
			cd.name = GlobalConstants::get_global_constant_name(i);
			cd.value = itos(GlobalConstants::get_global_constant_value(i));
			cd.enumeration = GlobalConstants::get_global_constant_enum(i);
			c.constants.push_back(cd);
		}

		List<Engine::Singleton> singletons;
		Engine::get_singleton()->get_singletons(&singletons);

		//servers (this is kind of hackish)
		for (List<Engine::Singleton>::Element *E = singletons.front(); E; E = E->next()) {

			PropertyDoc pd;
			Engine::Singleton &s = E->get();
			pd.name = s.name;
			pd.type = s.ptr->get_class();
			while (String(ClassDB::get_parent_class(pd.type)) != "Object")
				pd.type = ClassDB::get_parent_class(pd.type);
			if (pd.type.begins_with("_"))
				pd.type = pd.type.substr(1, pd.type.length());
			c.properties.push_back(pd);
		}
	}

	//built in script reference

	{

		for (int i = 0; i < ScriptServer::get_language_count(); i++) {

			ScriptLanguage *lang = ScriptServer::get_language(i);
			String cname = "@" + lang->get_name();
			class_list[cname] = ClassDoc();
			ClassDoc &c = class_list[cname];
			c.name = cname;

			List<MethodInfo> minfo;

			lang->get_public_functions(&minfo);

			for (List<MethodInfo>::Element *E = minfo.front(); E; E = E->next()) {

				MethodInfo &mi = E->get();
				MethodDoc md;
				md.name = mi.name;

				if (mi.flags & METHOD_FLAG_VARARG) {
					if (md.qualifiers != "")
						md.qualifiers += " ";
					md.qualifiers += "vararg";
				}

				return_doc_from_retinfo(md, mi.return_val);

				for (int i = 0; i < mi.arguments.size(); i++) {

					ArgumentDoc ad;
					argument_doc_from_arginfo(ad, mi.arguments[i]);
					md.arguments.push_back(ad);
				}

				c.methods.push_back(md);
			}

			List<Pair<String, Variant> > cinfo;
			lang->get_public_constants(&cinfo);

			for (List<Pair<String, Variant> >::Element *E = cinfo.front(); E; E = E->next()) {

				ConstantDoc cd;
				cd.name = E->get().first;
				cd.value = E->get().second;
				c.constants.push_back(cd);
			}
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
				method.name = parser->get_attribute_value("name");
				if (parser->has_attribute("qualifiers"))
					method.qualifiers = parser->get_attribute_value("qualifiers");

				while (parser->read() == OK) {

					if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {

						String name = parser->get_node_name();
						if (name == "return") {

							ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
							method.return_type = parser->get_attribute_value("type");
							if (parser->has_attribute("enum")) {
								method.return_enum = parser->get_attribute_value("enum");
							}
						} else if (name == "argument") {

							DocData::ArgumentDoc argument;
							ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
							argument.name = parser->get_attribute_value("name");
							ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
							argument.type = parser->get_attribute_value("type");
							if (parser->has_attribute("enum")) {
								argument.enumeration = parser->get_attribute_value("enum");
							}

							method.arguments.push_back(argument);

						} else if (name == "description") {

							parser->read();
							if (parser->get_node_type() == XMLParser::NODE_TEXT)
								method.description = parser->get_node_data();
						}

					} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == element)
						break;
				}

				methods.push_back(method);

			} else {
				ERR_EXPLAIN("Invalid tag in doc file: " + parser->get_node_name());
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

		} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == section)
			break;
	}

	return OK;
}

Error DocData::load_classes(const String &p_dir) {

	Error err;
	DirAccessRef da = DirAccess::open(p_dir, &err);
	if (!da) {
		return err;
	}

	da->list_dir_begin();
	String path;
	bool isdir;
	path = da->get_next(&isdir);
	while (path != String()) {
		if (!isdir && path.ends_with("xml")) {
			Ref<XMLParser> parser = memnew(XMLParser);
			Error err = parser->open(p_dir.plus_file(path));
			if (err)
				return err;

			_load(parser);
		}
		path = da->get_next(&isdir);
	}

	da->list_dir_end();

	return OK;
}
Error DocData::erase_classes(const String &p_dir) {

	Error err;
	DirAccessRef da = DirAccess::open(p_dir, &err);
	if (!da) {
		return err;
	}

	List<String> to_erase;

	da->list_dir_begin();
	String path;
	bool isdir;
	path = da->get_next(&isdir);
	while (path != String()) {
		if (!isdir && path.ends_with("xml")) {
			to_erase.push_back(path);
		}
		path = da->get_next(&isdir);
	}
	da->list_dir_end();

	while (to_erase.size()) {
		da->remove(to_erase.front()->get());
		to_erase.pop_front();
	}

	return OK;
}
Error DocData::_load(Ref<XMLParser> parser) {

	Error err = OK;

	while ((err = parser->read()) == OK) {

		if (parser->get_node_type() == XMLParser::NODE_ELEMENT && parser->get_node_name() == "?xml") {
			parser->skip_section();
		}

		if (parser->get_node_type() != XMLParser::NODE_ELEMENT)
			continue; //no idea what this may be, but skipping anyway

		ERR_FAIL_COND_V(parser->get_node_name() != "class", ERR_FILE_CORRUPT);

		ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
		String name = parser->get_attribute_value("name");
		class_list[name] = ClassDoc();
		ClassDoc &c = class_list[name];

		//print_line("class: "+name);
		c.name = name;
		if (parser->has_attribute("inherits"))
			c.inherits = parser->get_attribute_value("inherits");
		if (parser->has_attribute("category"))
			c.category = parser->get_attribute_value("category");

		while (parser->read() == OK) {

			if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {

				String name = parser->get_node_name();

				if (name == "brief_description") {

					parser->read();
					if (parser->get_node_type() == XMLParser::NODE_TEXT)
						c.brief_description = parser->get_node_data();

				} else if (name == "description") {
					parser->read();
					if (parser->get_node_type() == XMLParser::NODE_TEXT)
						c.description = parser->get_node_data();
				} else if (name == "tutorials") {
					parser->read();
					if (parser->get_node_type() == XMLParser::NODE_TEXT)
						c.tutorials = parser->get_node_data();
				} else if (name == "demos") {
					parser->read();
					if (parser->get_node_type() == XMLParser::NODE_TEXT)
						c.demos = parser->get_node_data();
				} else if (name == "methods") {

					Error err = _parse_methods(parser, c.methods);
					ERR_FAIL_COND_V(err, err);

				} else if (name == "signals") {

					Error err = _parse_methods(parser, c.signals);
					ERR_FAIL_COND_V(err, err);
				} else if (name == "members") {

					while (parser->read() == OK) {

						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {

							String name = parser->get_node_name();

							if (name == "member") {

								PropertyDoc prop;

								ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
								prop.name = parser->get_attribute_value("name");
								ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
								prop.type = parser->get_attribute_value("type");
								if (parser->has_attribute("setter"))
									prop.setter = parser->get_attribute_value("setter");
								if (parser->has_attribute("getter"))
									prop.getter = parser->get_attribute_value("getter");
								if (parser->has_attribute("enum"))
									prop.enumeration = parser->get_attribute_value("enum");
								parser->read();
								if (parser->get_node_type() == XMLParser::NODE_TEXT)
									prop.description = parser->get_node_data();
								c.properties.push_back(prop);
							} else {
								ERR_EXPLAIN("Invalid tag in doc file: " + name);
								ERR_FAIL_V(ERR_FILE_CORRUPT);
							}

						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "members")
							break; //end of <constants>
					}

				} else if (name == "theme_items") {

					while (parser->read() == OK) {

						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {

							String name = parser->get_node_name();

							if (name == "theme_item") {

								PropertyDoc prop;

								ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
								prop.name = parser->get_attribute_value("name");
								ERR_FAIL_COND_V(!parser->has_attribute("type"), ERR_FILE_CORRUPT);
								prop.type = parser->get_attribute_value("type");
								parser->read();
								if (parser->get_node_type() == XMLParser::NODE_TEXT)
									prop.description = parser->get_node_data();
								c.theme_properties.push_back(prop);
							} else {
								ERR_EXPLAIN("Invalid tag in doc file: " + name);
								ERR_FAIL_V(ERR_FILE_CORRUPT);
							}

						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "theme_items")
							break; //end of <constants>
					}

				} else if (name == "constants") {

					while (parser->read() == OK) {

						if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {

							String name = parser->get_node_name();

							if (name == "constant") {

								ConstantDoc constant;
								ERR_FAIL_COND_V(!parser->has_attribute("name"), ERR_FILE_CORRUPT);
								constant.name = parser->get_attribute_value("name");
								ERR_FAIL_COND_V(!parser->has_attribute("value"), ERR_FILE_CORRUPT);
								constant.value = parser->get_attribute_value("value");
								if (parser->has_attribute("enum")) {
									constant.enumeration = parser->get_attribute_value("enum");
								}
								parser->read();
								if (parser->get_node_type() == XMLParser::NODE_TEXT)
									constant.description = parser->get_node_data();
								c.constants.push_back(constant);
							} else {
								ERR_EXPLAIN("Invalid tag in doc file: " + name);
								ERR_FAIL_V(ERR_FILE_CORRUPT);
							}

						} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "constants")
							break; //end of <constants>
					}

				} else {

					ERR_EXPLAIN("Invalid tag in doc file: " + name);
					ERR_FAIL_V(ERR_FILE_CORRUPT);
				}

			} else if (parser->get_node_type() == XMLParser::NODE_ELEMENT_END && parser->get_node_name() == "class")
				break; //end of <asset>
		}
	}

	return OK;
}

static void _write_string(FileAccess *f, int p_tablevel, const String &p_string) {

	if (p_string == "")
		return;
	String tab;
	for (int i = 0; i < p_tablevel; i++)
		tab += "\t";
	f->store_string(tab + p_string + "\n");
}

Error DocData::save_classes(const String &p_default_path, const Map<String, String> &p_class_path) {

	for (Map<String, ClassDoc>::Element *E = class_list.front(); E; E = E->next()) {

		ClassDoc &c = E->get();

		String save_path;
		if (p_class_path.has(c.name)) {
			save_path = p_class_path[c.name];
		} else {
			save_path = p_default_path;
		}

		Error err;
		String save_file = save_path.plus_file(c.name + ".xml");
		FileAccessRef f = FileAccess::open(save_file, FileAccess::WRITE, &err);
		if (err) {
			ERR_EXPLAIN("Can't write doc file: " + save_file);

			ERR_FAIL_V(err);
		}

		_write_string(f, 0, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>");

		String header = "<class name=\"" + c.name + "\"";
		if (c.inherits != "")
			header += " inherits=\"" + c.inherits + "\"";

		String category = c.category;
		if (c.category == "")
			category = "Core";
		header += " category=\"" + category + "\"";
		header += String(" version=\"") + itos(VERSION_MAJOR) + "." + itos(VERSION_MINOR) + "-" + VERSION_STATUS + "\"";
		header += ">";
		_write_string(f, 0, header);
		_write_string(f, 1, "<brief_description>");
		_write_string(f, 2, c.brief_description.strip_edges().xml_escape());
		_write_string(f, 1, "</brief_description>");
		_write_string(f, 1, "<description>");
		_write_string(f, 2, c.description.strip_edges().xml_escape());
		_write_string(f, 1, "</description>");
		_write_string(f, 1, "<tutorials>");
		_write_string(f, 2, c.tutorials.strip_edges().xml_escape());
		_write_string(f, 1, "</tutorials>");
		_write_string(f, 1, "<demos>");
		_write_string(f, 2, c.demos.strip_edges().xml_escape());
		_write_string(f, 1, "</demos>");
		_write_string(f, 1, "<methods>");

		c.methods.sort();

		for (int i = 0; i < c.methods.size(); i++) {

			MethodDoc &m = c.methods[i];

			String qualifiers;
			if (m.qualifiers != "")
				qualifiers += " qualifiers=\"" + m.qualifiers.xml_escape() + "\"";

			_write_string(f, 2, "<method name=\"" + m.name + "\"" + qualifiers + ">");

			if (m.return_type != "") {

				String enum_text;
				if (m.return_enum != String()) {
					enum_text = " enum=\"" + m.return_enum + "\"";
				}
				_write_string(f, 3, "<return type=\"" + m.return_type + "\"" + enum_text + ">");
				_write_string(f, 3, "</return>");
			}

			for (int j = 0; j < m.arguments.size(); j++) {

				ArgumentDoc &a = m.arguments[j];

				String enum_text;
				if (a.enumeration != String()) {
					enum_text = " enum=\"" + a.enumeration + "\"";
				}

				if (a.default_value != "")
					_write_string(f, 3, "<argument index=\"" + itos(j) + "\" name=\"" + a.name.xml_escape() + "\" type=\"" + a.type.xml_escape() + "\"" + enum_text + " default=\"" + a.default_value.xml_escape(true) + "\">");
				else
					_write_string(f, 3, "<argument index=\"" + itos(j) + "\" name=\"" + a.name.xml_escape() + "\" type=\"" + a.type.xml_escape() + "\"" + enum_text + ">");

				_write_string(f, 3, "</argument>");
			}

			_write_string(f, 3, "<description>");
			_write_string(f, 4, m.description.strip_edges().xml_escape());
			_write_string(f, 3, "</description>");

			_write_string(f, 2, "</method>");
		}

		_write_string(f, 1, "</methods>");

		if (c.properties.size()) {
			_write_string(f, 1, "<members>");

			c.properties.sort();

			for (int i = 0; i < c.properties.size(); i++) {

				String enum_text;
				if (c.properties[i].enumeration != String()) {
					enum_text = " enum=\"" + c.properties[i].enumeration + "\"";
				}
				PropertyDoc &p = c.properties[i];
				_write_string(f, 2, "<member name=\"" + p.name + "\" type=\"" + p.type + "\" setter=\"" + p.setter + "\" getter=\"" + p.getter + "\"" + enum_text + ">");
				_write_string(f, 3, p.description.strip_edges().xml_escape());
				_write_string(f, 2, "</member>");
			}
			_write_string(f, 1, "</members>");
		}

		if (c.signals.size()) {

			c.signals.sort();

			_write_string(f, 1, "<signals>");
			for (int i = 0; i < c.signals.size(); i++) {

				MethodDoc &m = c.signals[i];
				_write_string(f, 2, "<signal name=\"" + m.name + "\">");
				for (int j = 0; j < m.arguments.size(); j++) {

					ArgumentDoc &a = m.arguments[j];
					_write_string(f, 3, "<argument index=\"" + itos(j) + "\" name=\"" + a.name.xml_escape() + "\" type=\"" + a.type.xml_escape() + "\">");
					_write_string(f, 3, "</argument>");
				}

				_write_string(f, 3, "<description>");
				_write_string(f, 4, m.description.strip_edges().xml_escape());
				_write_string(f, 3, "</description>");

				_write_string(f, 2, "</signal>");
			}

			_write_string(f, 1, "</signals>");
		}

		_write_string(f, 1, "<constants>");

		for (int i = 0; i < c.constants.size(); i++) {

			ConstantDoc &k = c.constants[i];
			if (k.enumeration != String()) {
				_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"" + k.value + "\" enum=\"" + k.enumeration + "\">");
			} else {
				_write_string(f, 2, "<constant name=\"" + k.name + "\" value=\"" + k.value + "\">");
			}
			_write_string(f, 3, k.description.strip_edges().xml_escape());
			_write_string(f, 2, "</constant>");
		}

		_write_string(f, 1, "</constants>");

		if (c.theme_properties.size()) {

			c.theme_properties.sort();

			_write_string(f, 1, "<theme_items>");
			for (int i = 0; i < c.theme_properties.size(); i++) {

				PropertyDoc &p = c.theme_properties[i];
				_write_string(f, 2, "<theme_item name=\"" + p.name + "\" type=\"" + p.type + "\">");
				_write_string(f, 2, "</theme_item>");
			}
			_write_string(f, 1, "</theme_items>");
		}

		_write_string(f, 0, "</class>");
	}

	return OK;
}

Error DocData::load_compressed(const uint8_t *p_data, int p_compressed_size, int p_uncompressed_size) {

	Vector<uint8_t> data;
	data.resize(p_uncompressed_size);
	Compression::decompress(data.ptrw(), p_uncompressed_size, p_data, p_compressed_size, Compression::MODE_DEFLATE);
	class_list.clear();

	Ref<XMLParser> parser = memnew(XMLParser);
	Error err = parser->open_buffer(data);
	if (err)
		return err;

	_load(parser);

	return OK;
}
