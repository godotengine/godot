/*************************************************************************/
/*  doc_dump.cpp                                                         */
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
#include "doc_dump.h"

#include "os/file_access.h"
#include "scene/main/node.h"
#include "version.h"

static void _write_string(FileAccess *f, int p_tablevel, const String &p_string) {

	String tab;
	for (int i = 0; i < p_tablevel; i++)
		tab += "\t";
	f->store_string(tab + p_string + "\n");
}

struct _ConstantSort {

	String name;
	int value;
	bool operator<(const _ConstantSort &p_c) const {

		String left_a = name.find("_") == -1 ? name : name.substr(0, name.find("_"));
		String left_b = p_c.name.find("_") == -1 ? p_c.name : p_c.name.substr(0, p_c.name.find("_"));
		if (left_a == left_b)
			return value < p_c.value;
		else
			return left_a < left_b;
	}
};

static String _escape_string(const String &p_str) {

	String ret = p_str;
	ret = ret.replace("&", "&amp;");
	ret = ret.replace("<", "&gt;");
	ret = ret.replace(">", "&lt;");
	ret = ret.replace("'", "&apos;");
	ret = ret.replace("\"", "&quot;");
	for (char i = 1; i < 32; i++) {

		char chr[2] = { i, 0 };
		ret = ret.replace(chr, "&#" + String::num(i) + ";");
	}
	ret = ret.utf8();
	return ret;
}
void DocDump::dump(const String &p_file) {

	List<StringName> class_list;
	ClassDB::get_class_list(&class_list);

	class_list.sort_custom<StringName::AlphCompare>();

	FileAccess *f = FileAccess::open(p_file, FileAccess::WRITE);

	_write_string(f, 0, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>");
	_write_string(f, 0, String("<doc version=\"") + itos(VERSION_MAJOR) + "." + itos(VERSION_MINOR) + "-" + VERSION_STATUS + "\" name=\"Engine Types\">");

	while (class_list.size()) {

		String name = class_list.front()->get();

		String header = "<class name=\"" + name + "\"";
		String inherits = ClassDB::get_parent_class(name);
		if (inherits != "")
			header += " inherits=\"" + inherits + "\"";
		String category = ClassDB::get_category(name);
		if (category == "")
			category = "Core";
		header += " category=\"" + category + "\"";
		header += ">";
		_write_string(f, 0, header);
		_write_string(f, 1, "<brief_description>");
		_write_string(f, 1, "</brief_description>");
		_write_string(f, 1, "<description>");
		_write_string(f, 1, "</description>");
		_write_string(f, 1, "<methods>");

		List<MethodInfo> method_list;
		ClassDB::get_method_list(name, &method_list, true);
		method_list.sort();

		for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {
			if (E->get().name == "" || E->get().name[0] == '_')
				continue; //hiden

			MethodBind *m = ClassDB::get_method(name, E->get().name);

			String qualifiers;
			if (E->get().flags & METHOD_FLAG_CONST)
				qualifiers += "qualifiers=\"const\"";

			_write_string(f, 2, "<method name=\"" + _escape_string(E->get().name) + "\" " + qualifiers + " >");

			for (int i = -1; i < E->get().arguments.size(); i++) {

				PropertyInfo arginfo;

				if (i == -1) {

					arginfo = E->get().return_val;
					String type_name = (arginfo.hint == PROPERTY_HINT_RESOURCE_TYPE) ? arginfo.hint_string : Variant::get_type_name(arginfo.type);

					if (arginfo.type == Variant::NIL)
						continue;
					_write_string(f, 3, "<return type=\"" + type_name + "\">");
				} else {

					arginfo = E->get().arguments[i];

					String type_name;

					if (arginfo.hint == PROPERTY_HINT_RESOURCE_TYPE)
						type_name = arginfo.hint_string;
					else if (arginfo.type == Variant::NIL)
						type_name = "var";
					else
						type_name = Variant::get_type_name(arginfo.type);

					if (m && m->has_default_argument(i)) {
						Variant default_arg = m->get_default_argument(i);
						String default_arg_text = String(_escape_string(m->get_default_argument(i)));

						switch (default_arg.get_type()) {

							case Variant::NIL:
								default_arg_text = "NULL";
								break;
							// atomic types
							case Variant::BOOL:
								if (bool(default_arg))
									default_arg_text = "true";
								else
									default_arg_text = "false";
								break;
							case Variant::INT:
							case Variant::REAL:
								//keep it
								break;
							case Variant::STRING:
							case Variant::NODE_PATH:
								default_arg_text = "\"" + default_arg_text + "\"";
								break;
							case Variant::TRANSFORM:
								if (default_arg.operator Transform() == Transform()) {
									default_arg_text = "";
								}

								default_arg_text = Variant::get_type_name(default_arg.get_type()) + "(" + default_arg_text + ")";
								break;

							case Variant::VECTOR2:
							case Variant::RECT2:
							case Variant::VECTOR3:
							case Variant::PLANE:
							case Variant::QUAT:
							case Variant::AABB:
							case Variant::BASIS:
							case Variant::COLOR:
							case Variant::POOL_BYTE_ARRAY:
							case Variant::POOL_INT_ARRAY:
							case Variant::POOL_REAL_ARRAY:
							case Variant::POOL_STRING_ARRAY:
							case Variant::POOL_VECTOR3_ARRAY:
							case Variant::POOL_COLOR_ARRAY:
								default_arg_text = Variant::get_type_name(default_arg.get_type()) + "(" + default_arg_text + ")";
								break;
							case Variant::OBJECT:
							case Variant::DICTIONARY: // 20
							case Variant::ARRAY:
							case Variant::_RID:

							default: {}
						}

						_write_string(f, 3, "<argument index=\"" + itos(i) + "\" name=\"" + _escape_string(arginfo.name) + "\" type=\"" + type_name + "\" default=\"" + _escape_string(default_arg_text) + "\">");
					} else
						_write_string(f, 3, "<argument index=\"" + itos(i) + "\" name=\"" + arginfo.name + "\" type=\"" + type_name + "\">");
				}

				String hint;
				switch (arginfo.hint) {
					case PROPERTY_HINT_DIR: hint = "A directory."; break;
					case PROPERTY_HINT_RANGE: hint = "Range - min: " + arginfo.hint_string.get_slice(",", 0) + " max: " + arginfo.hint_string.get_slice(",", 1) + " step: " + arginfo.hint_string.get_slice(",", 2); break;
					case PROPERTY_HINT_ENUM:
						hint = "Values: ";
						for (int j = 0; j < arginfo.hint_string.get_slice_count(","); j++) {
							if (j > 0) hint += ", ";
							hint += arginfo.hint_string.get_slice(",", j) + "=" + itos(j);
						}
						break;
					case PROPERTY_HINT_LENGTH: hint = "Length: " + arginfo.hint_string; break;
					case PROPERTY_HINT_FLAGS:
						hint = "Values: ";
						for (int j = 0; j < arginfo.hint_string.get_slice_count(","); j++) {
							if (j > 0) hint += ", ";
							hint += arginfo.hint_string.get_slice(",", j) + "=" + itos(1 << j);
						}
						break;
					case PROPERTY_HINT_FILE: hint = "A file:"; break;
					default: {}
						//case PROPERTY_HINT_RESOURCE_TYPE: hint="Type: "+arginfo.hint_string; break;
				};
				if (hint != "")
					_write_string(f, 4, hint);

				_write_string(f, 3, (i == -1) ? "</return>" : "</argument>");
			}

			_write_string(f, 3, "<description>");
			_write_string(f, 3, "</description>");

			_write_string(f, 2, "</method>");
		}

		_write_string(f, 1, "</methods>");

		List<MethodInfo> signal_list;
		ClassDB::get_signal_list(name, &signal_list, true);

		if (signal_list.size()) {

			_write_string(f, 1, "<signals>");
			for (List<MethodInfo>::Element *EV = signal_list.front(); EV; EV = EV->next()) {

				_write_string(f, 2, "<signal name=\"" + EV->get().name + "\">");
				for (int i = 0; i < EV->get().arguments.size(); i++) {
					PropertyInfo arginfo = EV->get().arguments[i];
					_write_string(f, 3, "<argument index=\"" + itos(i) + "\" name=\"" + arginfo.name + "\" type=\"" + Variant::get_type_name(arginfo.type) + "\">");
					_write_string(f, 3, "</argument>");
				}
				_write_string(f, 3, "<description>");
				_write_string(f, 3, "</description>");

				_write_string(f, 2, "</signal>");
			}

			_write_string(f, 1, "</signals>");
		}

		_write_string(f, 1, "<constants>");

		List<String> constant_list;
		ClassDB::get_integer_constant_list(name, &constant_list, true);

		/* constants are sorted in a special way */

		List<_ConstantSort> constant_sort;

		for (List<String>::Element *E = constant_list.front(); E; E = E->next()) {
			_ConstantSort cs;
			cs.name = E->get();
			cs.value = ClassDB::get_integer_constant(name, E->get());
			constant_sort.push_back(cs);
		}

		constant_sort.sort();

		for (List<_ConstantSort>::Element *E = constant_sort.front(); E; E = E->next()) {

			_write_string(f, 2, "<constant name=\"" + E->get().name + "\" value=\"" + itos(E->get().value) + "\">");
			_write_string(f, 2, "</constant>");
		}

		_write_string(f, 1, "</constants>");
		_write_string(f, 0, "</class>");

		class_list.erase(name);
	}

	_write_string(f, 0, "</doc>");
	f->close();
	memdelete(f);
}
