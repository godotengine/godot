/*************************************************************************/
/*  bindings_generator.cpp                                               */
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

#include "bindings_generator.h"

#ifdef DEBUG_METHODS_ENABLED

#include "core/engine.h"
#include "core/global_constants.h"
#include "core/io/compression.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/string_builder.h"
#include "core/ucaps.h"

#include "../glue/cs_compressed.gen.h"
#include "../glue/cs_glue_version.gen.h"
#include "../godotsharp_defs.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../utils/path_utils.h"
#include "../utils/string_utils.h"
#include "csharp_project.h"

#define CS_INDENT "    " // 4 whitespaces

#define INDENT1 CS_INDENT
#define INDENT2 INDENT1 INDENT1
#define INDENT3 INDENT2 INDENT1
#define INDENT4 INDENT3 INDENT1
#define INDENT5 INDENT4 INDENT1

#define MEMBER_BEGIN "\n" INDENT2

#define OPEN_BLOCK "{\n"
#define CLOSE_BLOCK "}\n"

#define OPEN_BLOCK_L2 INDENT2 OPEN_BLOCK INDENT3
#define OPEN_BLOCK_L3 INDENT3 OPEN_BLOCK INDENT4
#define OPEN_BLOCK_L4 INDENT4 OPEN_BLOCK INDENT5
#define CLOSE_BLOCK_L2 INDENT2 CLOSE_BLOCK
#define CLOSE_BLOCK_L3 INDENT3 CLOSE_BLOCK
#define CLOSE_BLOCK_L4 INDENT4 CLOSE_BLOCK

#define CS_FIELD_MEMORYOWN "memoryOwn"
#define CS_PARAM_METHODBIND "method"
#define CS_PARAM_INSTANCE "ptr"
#define CS_SMETHOD_GETINSTANCE "GetPtr"
#define CS_METHOD_CALL "Call"

#define GLUE_HEADER_FILE "glue_header.h"
#define ICALL_PREFIX "godot_icall_"
#define SINGLETON_ICALL_SUFFIX "_get_singleton"
#define ICALL_GET_METHODBIND ICALL_PREFIX "Object_ClassDB_get_method"

#define C_LOCAL_RET "ret"
#define C_LOCAL_VARARG_RET "vararg_ret"
#define C_LOCAL_PTRCALL_ARGS "call_args"
#define C_MACRO_OBJECT_CONSTRUCT "GODOTSHARP_INSTANCE_OBJECT"

#define C_NS_MONOUTILS "GDMonoUtils"
#define C_NS_MONOINTERNALS "GDMonoInternals"
#define C_METHOD_TIE_MANAGED_TO_UNMANAGED C_NS_MONOINTERNALS "::tie_managed_to_unmanaged"
#define C_METHOD_UNMANAGED_GET_MANAGED C_NS_MONOUTILS "::unmanaged_get_managed"

#define C_NS_MONOMARSHAL "GDMonoMarshal"
#define C_METHOD_MANAGED_TO_VARIANT C_NS_MONOMARSHAL "::mono_object_to_variant"
#define C_METHOD_MANAGED_FROM_VARIANT C_NS_MONOMARSHAL "::variant_to_mono_object"
#define C_METHOD_MONOSTR_TO_GODOT C_NS_MONOMARSHAL "::mono_string_to_godot"
#define C_METHOD_MONOSTR_FROM_GODOT C_NS_MONOMARSHAL "::mono_string_from_godot"
#define C_METHOD_MONOARRAY_TO(m_type) C_NS_MONOMARSHAL "::mono_array_to_" #m_type
#define C_METHOD_MONOARRAY_FROM(m_type) C_NS_MONOMARSHAL "::" #m_type "_to_mono_array"

#define BINDINGS_GENERATOR_VERSION UINT32_C(8)

const char *BindingsGenerator::TypeInterface::DEFAULT_VARARG_C_IN("\t%0 %1_in = %1;\n");

bool BindingsGenerator::verbose_output = false;

BindingsGenerator *BindingsGenerator::singleton = NULL;

static String fix_doc_description(const String &p_bbcode) {

	// This seems to be the correct way to do this. It's the same EditorHelp does.

	return p_bbcode.dedent()
			.replace("\t", "")
			.replace("\r", "")
			.strip_edges();
}

static String snake_to_pascal_case(const String &p_identifier, bool p_input_is_upper = false) {

	String ret;
	Vector<String> parts = p_identifier.split("_", true);

	for (int i = 0; i < parts.size(); i++) {
		String part = parts[i];

		if (part.length()) {
			part[0] = _find_upper(part[0]);
			if (p_input_is_upper) {
				for (int j = 1; j < part.length(); j++)
					part[j] = _find_lower(part[j]);
			}
			ret += part;
		} else {
			if (i == 0 || i == (parts.size() - 1)) {
				// Preserve underscores at the beginning and end
				ret += "_";
			} else {
				// Preserve contiguous underscores
				if (parts[i - 1].length()) {
					ret += "__";
				} else {
					ret += "_";
				}
			}
		}
	}

	return ret;
}

static String snake_to_camel_case(const String &p_identifier, bool p_input_is_upper = false) {

	String ret;
	Vector<String> parts = p_identifier.split("_", true);

	for (int i = 0; i < parts.size(); i++) {
		String part = parts[i];

		if (part.length()) {
			if (i != 0) {
				part[0] = _find_upper(part[0]);
			}
			if (p_input_is_upper) {
				for (int j = i != 0 ? 1 : 0; j < part.length(); j++)
					part[j] = _find_lower(part[j]);
			}
			ret += part;
		} else {
			if (i == 0 || i == (parts.size() - 1)) {
				// Preserve underscores at the beginning and end
				ret += "_";
			} else {
				// Preserve contiguous underscores
				if (parts[i - 1].length()) {
					ret += "__";
				} else {
					ret += "_";
				}
			}
		}
	}

	return ret;
}

String BindingsGenerator::bbcode_to_xml(const String &p_bbcode, const TypeInterface *p_itype) {

	// Based on the version in EditorHelp

	if (p_bbcode.empty())
		return String();

	DocData *doc = EditorHelp::get_doc_data();

	String bbcode = p_bbcode;

	StringBuilder xml_output;

	xml_output.append("<para>");

	List<String> tag_stack;
	bool code_tag = false;

	int pos = 0;
	while (pos < bbcode.length()) {
		int brk_pos = bbcode.find("[", pos);

		if (brk_pos < 0)
			brk_pos = bbcode.length();

		if (brk_pos > pos) {
			String text = bbcode.substr(pos, brk_pos - pos);
			if (code_tag || tag_stack.size() > 0) {
				xml_output.append(text.xml_escape());
			} else {
				Vector<String> lines = text.split("\n");
				for (int i = 0; i < lines.size(); i++) {
					if (i != 0)
						xml_output.append("<para>");

					xml_output.append(lines[i].xml_escape());

					if (i != lines.size() - 1)
						xml_output.append("</para>\n");
				}
			}
		}

		if (brk_pos == bbcode.length())
			break; // nothing else to add

		int brk_end = bbcode.find("]", brk_pos + 1);

		if (brk_end == -1) {
			String text = bbcode.substr(brk_pos, bbcode.length() - brk_pos);
			if (code_tag || tag_stack.size() > 0) {
				xml_output.append(text.xml_escape());
			} else {
				Vector<String> lines = text.split("\n");
				for (int i = 0; i < lines.size(); i++) {
					if (i != 0)
						xml_output.append("<para>");

					xml_output.append(lines[i].xml_escape());

					if (i != lines.size() - 1)
						xml_output.append("</para>\n");
				}
			}

			break;
		}

		String tag = bbcode.substr(brk_pos + 1, brk_end - brk_pos - 1);

		if (tag.begins_with("/")) {
			bool tag_ok = tag_stack.size() && tag_stack.front()->get() == tag.substr(1, tag.length());

			if (!tag_ok) {
				xml_output.append("[");
				pos = brk_pos + 1;
				continue;
			}

			tag_stack.pop_front();
			pos = brk_end + 1;
			code_tag = false;

			if (tag == "/url") {
				xml_output.append("</a>");
			} else if (tag == "/code") {
				xml_output.append("</c>");
			} else if (tag == "/codeblock") {
				xml_output.append("</code>");
			}
		} else if (code_tag) {
			xml_output.append("[");
			pos = brk_pos + 1;
		} else if (tag.begins_with("method ") || tag.begins_with("member ") || tag.begins_with("signal ") || tag.begins_with("enum ") || tag.begins_with("constant ")) {
			String link_target = tag.substr(tag.find(" ") + 1, tag.length());
			String link_tag = tag.substr(0, tag.find(" "));

			Vector<String> link_target_parts = link_target.split(".");

			if (link_target_parts.size() <= 0 || link_target_parts.size() > 2) {
				ERR_PRINTS("Invalid reference format: " + tag);

				xml_output.append("<c>");
				xml_output.append(tag);
				xml_output.append("</c>");

				pos = brk_end + 1;
				continue;
			}

			const TypeInterface *target_itype;
			StringName target_cname;

			if (link_target_parts.size() == 2) {
				target_itype = _get_type_or_null(TypeReference(link_target_parts[0]));
				if (!target_itype) {
					target_itype = _get_type_or_null(TypeReference("_" + link_target_parts[0]));
				}
				target_cname = link_target_parts[1];
			} else {
				target_itype = p_itype;
				target_cname = link_target_parts[0];
			}

			if (link_tag == "method") {
				if (!target_itype || !target_itype->is_object_type) {
					if (OS::get_singleton()->is_stdout_verbose()) {
						if (target_itype) {
							OS::get_singleton()->print("Cannot resolve method reference for non-Godot.Object type in documentation: %s\n", link_target.utf8().get_data());
						} else {
							OS::get_singleton()->print("Cannot resolve type from method reference in documentation: %s\n", link_target.utf8().get_data());
						}
					}

					// TODO Map what we can
					xml_output.append("<c>");
					xml_output.append(link_target);
					xml_output.append("</c>");
				} else {
					const MethodInterface *target_imethod = target_itype->find_method_by_name(target_cname);

					if (target_imethod) {
						xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
						xml_output.append(target_itype->proxy_name);
						xml_output.append(".");
						xml_output.append(target_imethod->proxy_name);
						xml_output.append("\"/>");
					}
				}
			} else if (link_tag == "member") {
				if (!target_itype || !target_itype->is_object_type) {
					if (OS::get_singleton()->is_stdout_verbose()) {
						if (target_itype) {
							OS::get_singleton()->print("Cannot resolve member reference for non-Godot.Object type in documentation: %s\n", link_target.utf8().get_data());
						} else {
							OS::get_singleton()->print("Cannot resolve type from member reference in documentation: %s\n", link_target.utf8().get_data());
						}
					}

					// TODO Map what we can
					xml_output.append("<c>");
					xml_output.append(link_target);
					xml_output.append("</c>");
				} else {
					const PropertyInterface *target_iprop = target_itype->find_property_by_name(target_cname);

					if (target_iprop) {
						xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
						xml_output.append(target_itype->proxy_name);
						xml_output.append(".");
						xml_output.append(target_iprop->proxy_name);
						xml_output.append("\"/>");
					}
				}
			} else if (link_tag == "signal") {
				// We do not declare signals in any way in C#, so there is nothing to reference
				xml_output.append("<c>");
				xml_output.append(link_target);
				xml_output.append("</c>");
			} else if (link_tag == "enum") {
				StringName search_cname = !target_itype ? target_cname :
														  StringName(target_itype->name + "." + (String)target_cname);

				const Map<StringName, TypeInterface>::Element *enum_match = enum_types.find(search_cname);

				if (!enum_match && search_cname != target_cname) {
					enum_match = enum_types.find(target_cname);
				}

				if (enum_match) {
					const TypeInterface &target_enum_itype = enum_match->value();

					xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
					xml_output.append(target_enum_itype.proxy_name); // Includes nesting class if any
					xml_output.append("\"/>");
				} else {
					ERR_PRINTS("Cannot resolve enum reference in documentation: " + link_target);

					xml_output.append("<c>");
					xml_output.append(link_target);
					xml_output.append("</c>");
				}
			} else if (link_tag == "const") {
				if (!target_itype || !target_itype->is_object_type) {
					if (OS::get_singleton()->is_stdout_verbose()) {
						if (target_itype) {
							OS::get_singleton()->print("Cannot resolve constant reference for non-Godot.Object type in documentation: %s\n", link_target.utf8().get_data());
						} else {
							OS::get_singleton()->print("Cannot resolve type from constant reference in documentation: %s\n", link_target.utf8().get_data());
						}
					}

					// TODO Map what we can
					xml_output.append("<c>");
					xml_output.append(link_target);
					xml_output.append("</c>");
				} else if (!target_itype && target_cname == name_cache.type_at_GlobalScope) {
					String target_name = (String)target_cname;

					// Try to find as a global constant
					const ConstantInterface *target_iconst = find_constant_by_name(target_name, global_constants);

					if (target_iconst) {
						// Found global constant
						xml_output.append("<see cref=\"" BINDINGS_NAMESPACE "." BINDINGS_GLOBAL_SCOPE_CLASS ".");
						xml_output.append(target_iconst->proxy_name);
						xml_output.append("\"/>");
					} else {
						// Try to find as global enum constant
						const EnumInterface *target_ienum = NULL;

						for (const List<EnumInterface>::Element *E = global_enums.front(); E; E = E->next()) {
							target_ienum = &E->get();
							target_iconst = find_constant_by_name(target_name, target_ienum->constants);
							if (target_iconst)
								break;
						}

						if (target_iconst) {
							xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
							xml_output.append(target_ienum->cname);
							xml_output.append(".");
							xml_output.append(target_iconst->proxy_name);
							xml_output.append("\"/>");
						} else {
							ERR_PRINTS("Cannot resolve global constant reference in documentation: " + link_target);

							xml_output.append("<c>");
							xml_output.append(link_target);
							xml_output.append("</c>");
						}
					}
				} else {
					String target_name = (String)target_cname;

					// Try to find the constant in the current class
					const ConstantInterface *target_iconst = find_constant_by_name(target_name, target_itype->constants);

					if (target_iconst) {
						// Found constant in current class
						xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
						xml_output.append(target_itype->proxy_name);
						xml_output.append(".");
						xml_output.append(target_iconst->proxy_name);
						xml_output.append("\"/>");
					} else {
						// Try to find as enum constant in the current class
						const EnumInterface *target_ienum = NULL;

						for (const List<EnumInterface>::Element *E = target_itype->enums.front(); E; E = E->next()) {
							target_ienum = &E->get();
							target_iconst = find_constant_by_name(target_name, target_ienum->constants);
							if (target_iconst)
								break;
						}

						if (target_iconst) {
							xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
							xml_output.append(target_itype->proxy_name);
							xml_output.append(".");
							xml_output.append(target_ienum->cname);
							xml_output.append(".");
							xml_output.append(target_iconst->proxy_name);
							xml_output.append("\"/>");
						} else {
							ERR_PRINTS("Cannot resolve constant reference in documentation: " + link_target);

							xml_output.append("<c>");
							xml_output.append(link_target);
							xml_output.append("</c>");
						}
					}
				}
			}

			pos = brk_end + 1;
		} else if (doc->class_list.has(tag)) {
			if (tag == "Array" || tag == "Dictionary") {
				xml_output.append("<see cref=\"" BINDINGS_NAMESPACE_COLLECTIONS ".");
				xml_output.append(tag);
				xml_output.append("\"/>");
			} else if (tag == "bool" || tag == "int") {
				xml_output.append("<see cref=\"");
				xml_output.append(tag);
				xml_output.append("\"/>");
			} else if (tag == "float") {
				xml_output.append("<see cref=\""
#ifdef REAL_T_IS_DOUBLE
								  "double"
#else
								  "float"
#endif
								  "\"/>");
			} else if (tag == "Variant") {
				// We use System.Object for Variant, so there is no Variant type in C#
				xml_output.append("<c>Variant</c>");
			} else if (tag == "String") {
				xml_output.append("<see cref=\"string\"/>");
			} else if (tag == "Nil") {
				xml_output.append("<see langword=\"null\"/>");
			} else if (tag.begins_with("@")) {
				// @GlobalScope, @GDScript, etc
				xml_output.append("<c>");
				xml_output.append(tag);
				xml_output.append("</c>");
			} else if (tag == "PoolByteArray") {
				xml_output.append("<see cref=\"byte\"/>");
			} else if (tag == "PoolIntArray") {
				xml_output.append("<see cref=\"int\"/>");
			} else if (tag == "PoolRealArray") {
#ifdef REAL_T_IS_DOUBLE
				xml_output.append("<see cref=\"double\"/>");
#else
				xml_output.append("<see cref=\"float\"/>");
#endif
			} else if (tag == "PoolStringArray") {
				xml_output.append("<see cref=\"string\"/>");
			} else if (tag == "PoolVector2Array") {
				xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".Vector2\"/>");
			} else if (tag == "PoolVector3Array") {
				xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".Vector3\"/>");
			} else if (tag == "PoolColorArray") {
				xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".Color\"/>");
			} else {
				const TypeInterface *target_itype = _get_type_or_null(TypeReference(tag));

				if (!target_itype) {
					target_itype = _get_type_or_null(TypeReference("_" + tag));
				}

				if (target_itype) {
					xml_output.append("<see cref=\"" BINDINGS_NAMESPACE ".");
					xml_output.append(target_itype->proxy_name);
					xml_output.append("\"/>");
				} else {
					ERR_PRINTS("Cannot resolve type reference in documentation: " + tag);

					xml_output.append("<c>");
					xml_output.append(tag);
					xml_output.append("</c>");
				}
			}

			pos = brk_end + 1;
		} else if (tag == "b") {
			// bold is not supported in xml comments
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {
			// italics is not supported in xml comments
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code") {
			xml_output.append("<c>");

			code_tag = true;
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "codeblock") {
			xml_output.append("<code>");

			code_tag = true;
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "center") {
			// center is alignment not supported in xml comments
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "br") {
			xml_output.append("\n"); // FIXME: Should use <para> instead. Luckily this tag isn't used for now.
			pos = brk_end + 1;
		} else if (tag == "u") {
			// underline is not supported in xml comments
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "s") {
			// strikethrough is not supported in xml comments
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "url") {
			int end = bbcode.find("[", brk_end);
			if (end == -1)
				end = bbcode.length();
			String url = bbcode.substr(brk_end + 1, end - brk_end - 1);
			xml_output.append("<a href=\"");
			xml_output.append(url);
			xml_output.append("\">");
			xml_output.append(url);

			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("url=")) {
			String url = tag.substr(4, tag.length());
			xml_output.append("<a href=\"");
			xml_output.append(url);
			xml_output.append("\">");

			pos = brk_end + 1;
			tag_stack.push_front("url");
		} else if (tag == "img") {
			int end = bbcode.find("[", brk_end);
			if (end == -1)
				end = bbcode.length();
			String image = bbcode.substr(brk_end + 1, end - brk_end - 1);

			// Not supported. Just append the bbcode.
			xml_output.append("[img]");
			xml_output.append(image);
			xml_output.append("[/img]");

			pos = end;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("color=")) {
			// Not supported.
			pos = brk_end + 1;
			tag_stack.push_front("color");
		} else if (tag.begins_with("font=")) {
			// Not supported.
			pos = brk_end + 1;
			tag_stack.push_front("font");
		} else {
			xml_output.append("["); // ignore
			pos = brk_pos + 1;
		}
	}

	xml_output.append("</para>");

	return xml_output.as_string();
}

int BindingsGenerator::_determine_enum_prefix(const EnumInterface &p_ienum) {

	CRASH_COND(p_ienum.constants.empty());

	const ConstantInterface &front_iconstant = p_ienum.constants.front()->get();
	Vector<String> front_parts = front_iconstant.name.split("_", /* p_allow_empty: */ true);
	int candidate_len = front_parts.size() - 1;

	if (candidate_len == 0)
		return 0;

	for (const List<ConstantInterface>::Element *E = p_ienum.constants.front()->next(); E; E = E->next()) {
		const ConstantInterface &iconstant = E->get();

		Vector<String> parts = iconstant.name.split("_", /* p_allow_empty: */ true);

		int i;
		for (i = 0; i < candidate_len && i < parts.size(); i++) {
			if (front_parts[i] != parts[i]) {
				// HARDCODED: Some Flag enums have the prefix 'FLAG_' for everything except 'FLAGS_DEFAULT' (same for 'METHOD_FLAG_' and'METHOD_FLAGS_DEFAULT').
				bool hardcoded_exc = (i == candidate_len - 1 && ((front_parts[i] == "FLAGS" && parts[i] == "FLAG") || (front_parts[i] == "FLAG" && parts[i] == "FLAGS")));
				if (!hardcoded_exc)
					break;
			}
		}
		candidate_len = i;

		if (candidate_len == 0)
			return 0;
	}

	return candidate_len;
}

void BindingsGenerator::_apply_prefix_to_enum_constants(BindingsGenerator::EnumInterface &p_ienum, int p_prefix_length) {

	if (p_prefix_length > 0) {
		for (List<ConstantInterface>::Element *E = p_ienum.constants.front(); E; E = E->next()) {
			int curr_prefix_length = p_prefix_length;

			ConstantInterface &curr_const = E->get();

			String constant_name = curr_const.name;

			Vector<String> parts = constant_name.split("_", /* p_allow_empty: */ true);

			if (parts.size() <= curr_prefix_length)
				continue;

			if (parts[curr_prefix_length][0] >= '0' && parts[curr_prefix_length][0] <= '9') {
				// The name of enum constants may begin with a numeric digit when strip from the enum prefix,
				// so we make the prefix for this constant one word shorter in those cases.
				for (curr_prefix_length = curr_prefix_length - 1; curr_prefix_length > 0; curr_prefix_length--) {
					if (parts[curr_prefix_length][0] < '0' || parts[curr_prefix_length][0] > '9')
						break;
				}
			}

			constant_name = "";
			for (int i = curr_prefix_length; i < parts.size(); i++) {
				if (i > curr_prefix_length)
					constant_name += "_";
				constant_name += parts[i];
			}

			curr_const.proxy_name = snake_to_pascal_case(constant_name, true);
		}
	}
}

void BindingsGenerator::_generate_method_icalls(const TypeInterface &p_itype) {

	for (const List<MethodInterface>::Element *E = p_itype.methods.front(); E; E = E->next()) {
		const MethodInterface &imethod = E->get();

		if (imethod.is_virtual)
			continue;

		const TypeInterface *return_type = _get_type_or_placeholder(imethod.return_type);

		String im_sig = "IntPtr " CS_PARAM_METHODBIND ", ";
		String im_unique_sig = imethod.return_type.cname.operator String() + ",IntPtr,IntPtr";

		im_sig += "IntPtr " CS_PARAM_INSTANCE;

		// Get arguments information
		int i = 0;
		for (const List<ArgumentInterface>::Element *F = imethod.arguments.front(); F; F = F->next()) {
			const TypeInterface *arg_type = _get_type_or_placeholder(F->get().type);

			im_sig += ", ";
			im_sig += arg_type->im_type_in;
			im_sig += " arg";
			im_sig += itos(i + 1);

			im_unique_sig += ",";
			im_unique_sig += get_unique_sig(*arg_type);

			i++;
		}

		// godot_icall_{argc}_{icallcount}
		String icall_method = ICALL_PREFIX;
		icall_method += itos(imethod.arguments.size());
		icall_method += "_";
		icall_method += itos(method_icalls.size());

		InternalCall im_icall = InternalCall(p_itype.api_type, icall_method, return_type->im_type_out, im_sig, im_unique_sig);

		List<InternalCall>::Element *match = method_icalls.find(im_icall);

		if (match) {
			if (p_itype.api_type != ClassDB::API_EDITOR)
				match->get().editor_only = false;
			method_icalls_map.insert(&E->get(), &match->get());
		} else {
			List<InternalCall>::Element *added = method_icalls.push_back(im_icall);
			method_icalls_map.insert(&E->get(), &added->get());
		}
	}
}

void BindingsGenerator::_generate_global_constants(List<String> &p_output) {

	// Constants (in partial GD class)

	p_output.push_back("\n#pragma warning disable CS1591 // Disable warning: "
					   "'Missing XML comment for publicly visible type or member'\n");

	p_output.push_back("namespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);
	p_output.push_back(INDENT1 "public static partial class " BINDINGS_GLOBAL_SCOPE_CLASS "\n" INDENT1 "{");

	for (const List<ConstantInterface>::Element *E = global_constants.front(); E; E = E->next()) {
		const ConstantInterface &iconstant = E->get();

		if (iconstant.const_doc && iconstant.const_doc->description.size()) {
			String xml_summary = bbcode_to_xml(fix_doc_description(iconstant.const_doc->description), NULL);
			Vector<String> summary_lines = xml_summary.split("\n");

			if (summary_lines.size()) {
				p_output.push_back(MEMBER_BEGIN "/// <summary>\n");

				for (int i = 0; i < summary_lines.size(); i++) {
					p_output.push_back(INDENT2 "/// ");
					p_output.push_back(summary_lines[i]);
					p_output.push_back("\n");
				}

				p_output.push_back(INDENT2 "/// </summary>");
			}
		}

		p_output.push_back(MEMBER_BEGIN "public const int ");
		p_output.push_back(iconstant.proxy_name);
		p_output.push_back(" = ");
		p_output.push_back(itos(iconstant.value));
		p_output.push_back(";");
	}

	if (!global_constants.empty())
		p_output.push_back("\n");

	p_output.push_back(INDENT1 CLOSE_BLOCK); // end of GD class

	// Enums

	for (List<EnumInterface>::Element *E = global_enums.front(); E; E = E->next()) {
		const EnumInterface &ienum = E->get();

		CRASH_COND(ienum.constants.empty());

		String enum_proxy_name = ienum.cname.operator String();

		bool enum_in_static_class = false;

		if (enum_proxy_name.find(".") > 0) {
			enum_in_static_class = true;
			String enum_class_name = enum_proxy_name.get_slicec('.', 0);
			enum_proxy_name = enum_proxy_name.get_slicec('.', 1);

			CRASH_COND(enum_class_name != "Variant"); // Hard-coded...

			if (verbose_output) {
				WARN_PRINTS("Declaring global enum `" + enum_proxy_name + "` inside static class `" + enum_class_name + "`");
			}

			p_output.push_back("\n" INDENT1 "public static partial class ");
			p_output.push_back(enum_class_name);
			p_output.push_back("\n" INDENT1 OPEN_BLOCK);
		}

		p_output.push_back("\n" INDENT1 "public enum ");
		p_output.push_back(enum_proxy_name);
		p_output.push_back("\n" INDENT1 OPEN_BLOCK);

		for (const List<ConstantInterface>::Element *F = ienum.constants.front(); F; F = F->next()) {
			const ConstantInterface &iconstant = F->get();

			if (iconstant.const_doc && iconstant.const_doc->description.size()) {
				String xml_summary = bbcode_to_xml(fix_doc_description(iconstant.const_doc->description), NULL);
				Vector<String> summary_lines = xml_summary.split("\n");

				if (summary_lines.size()) {
					p_output.push_back(INDENT2 "/// <summary>\n");

					for (int i = 0; i < summary_lines.size(); i++) {
						p_output.push_back(INDENT2 "/// ");
						p_output.push_back(summary_lines[i]);
						p_output.push_back("\n");
					}

					p_output.push_back(INDENT2 "/// </summary>\n");
				}
			}

			p_output.push_back(INDENT2);
			p_output.push_back(iconstant.proxy_name);
			p_output.push_back(" = ");
			p_output.push_back(itos(iconstant.value));
			p_output.push_back(F != ienum.constants.back() ? ",\n" : "\n");
		}

		p_output.push_back(INDENT1 CLOSE_BLOCK);

		if (enum_in_static_class)
			p_output.push_back(INDENT1 CLOSE_BLOCK);
	}

	p_output.push_back(CLOSE_BLOCK); // end of namespace

	p_output.push_back("\n#pragma warning restore CS1591\n");
}

Error BindingsGenerator::generate_cs_core_project(const String &p_solution_dir, DotNetSolution &r_solution, bool p_verbose_output) {

	verbose_output = p_verbose_output;

	String proj_dir = p_solution_dir.plus_file(CORE_API_ASSEMBLY_NAME);

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);

	if (!DirAccess::exists(proj_dir)) {
		Error err = da->make_dir_recursive(proj_dir);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	da->change_dir(proj_dir);
	da->make_dir("Core");
	da->make_dir("ObjectType");

	String core_dir = path_join(proj_dir, "Core");
	String obj_type_dir = path_join(proj_dir, "ObjectType");

	Vector<String> compile_items;

	// Generate source file for global scope constants and enums
	{
		List<String> constants_source;
		_generate_global_constants(constants_source);
		String output_file = path_join(core_dir, BINDINGS_GLOBAL_SCOPE_CLASS "_constants.cs");
		Error save_err = _save_file(output_file, constants_source);
		if (save_err != OK)
			return save_err;

		compile_items.push_back(output_file);
	}

	for (OrderedHashMap<StringName, TypeInterface>::Element E = obj_types.front(); E; E = E.next()) {
		const TypeInterface &itype = E.get();

		if (itype.api_type == ClassDB::API_EDITOR)
			continue;

		String output_file = path_join(obj_type_dir, itype.proxy_name + ".cs");
		Error err = _generate_cs_type(itype, output_file);

		if (err == ERR_SKIP)
			continue;

		if (err != OK)
			return err;

		compile_items.push_back(output_file);
	}

	// Generate sources from compressed files

	Map<String, CompressedFile> compressed_files;
	get_compressed_files(compressed_files);

	for (Map<String, CompressedFile>::Element *E = compressed_files.front(); E; E = E->next()) {
		const String &file_name = E->key();
		const CompressedFile &file_data = E->value();

		String output_file = path_join(core_dir, file_name);

		Vector<uint8_t> data;
		data.resize(file_data.uncompressed_size);
		Compression::decompress(data.ptrw(), file_data.uncompressed_size, file_data.data, file_data.compressed_size, Compression::MODE_DEFLATE);

		String output_dir = output_file.get_base_dir();

		if (!DirAccess::exists(output_dir)) {
			Error err = da->make_dir_recursive(ProjectSettings::get_singleton()->globalize_path(output_dir));
			ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
		}

		FileAccessRef file = FileAccess::open(output_file, FileAccess::WRITE);
		ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);
		file->store_buffer(data.ptr(), data.size());
		file->close();

		compile_items.push_back(output_file);
	}

	List<String> cs_icalls_content;

	cs_icalls_content.push_back("using System;\n"
								"using System.Runtime.CompilerServices;\n"
								"\n");
	cs_icalls_content.push_back("namespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);
	cs_icalls_content.push_back(INDENT1 "internal static class " BINDINGS_CLASS_NATIVECALLS "\n" INDENT1 OPEN_BLOCK);

	cs_icalls_content.push_back(MEMBER_BEGIN "internal static ulong godot_api_hash = ");
	cs_icalls_content.push_back(String::num_uint64(GDMono::get_singleton()->get_api_core_hash()) + ";\n");
	cs_icalls_content.push_back(MEMBER_BEGIN "internal static uint bindings_version = ");
	cs_icalls_content.push_back(String::num_uint64(BINDINGS_GENERATOR_VERSION) + ";\n");
	cs_icalls_content.push_back(MEMBER_BEGIN "internal static uint cs_glue_version = ");
	cs_icalls_content.push_back(String::num_uint64(CS_GLUE_VERSION) + ";\n");

#define ADD_INTERNAL_CALL(m_icall)                                                                  \
	if (!m_icall.editor_only) {                                                                     \
		cs_icalls_content.push_back(MEMBER_BEGIN "[MethodImpl(MethodImplOptions.InternalCall)]\n"); \
		cs_icalls_content.push_back(INDENT2 "internal extern static ");                             \
		cs_icalls_content.push_back(m_icall.im_type_out + " ");                                     \
		cs_icalls_content.push_back(m_icall.name + "(");                                            \
		cs_icalls_content.push_back(m_icall.im_sig + ");\n");                                       \
	}

	for (const List<InternalCall>::Element *E = core_custom_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());
	for (const List<InternalCall>::Element *E = method_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());

#undef ADD_INTERNAL_CALL

	cs_icalls_content.push_back(INDENT1 CLOSE_BLOCK CLOSE_BLOCK);

	String internal_methods_file = path_join(core_dir, BINDINGS_CLASS_NATIVECALLS ".cs");

	Error err = _save_file(internal_methods_file, cs_icalls_content);
	if (err != OK)
		return err;

	compile_items.push_back(internal_methods_file);

	String guid = CSharpProject::generate_core_api_project(proj_dir, compile_items);

	DotNetSolution::ProjectInfo proj_info;
	proj_info.guid = guid;
	proj_info.relpath = String(CORE_API_ASSEMBLY_NAME).plus_file(CORE_API_ASSEMBLY_NAME ".csproj");
	proj_info.configs.push_back("Debug");
	proj_info.configs.push_back("Release");

	r_solution.add_new_project(CORE_API_ASSEMBLY_NAME, proj_info);

	if (verbose_output)
		OS::get_singleton()->print("The solution and C# project for the Core API was generated successfully\n");

	return OK;
}

Error BindingsGenerator::generate_cs_editor_project(const String &p_solution_dir, DotNetSolution &r_solution, bool p_verbose_output) {

	verbose_output = p_verbose_output;

	String proj_dir = p_solution_dir.plus_file(EDITOR_API_ASSEMBLY_NAME);

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);

	if (!DirAccess::exists(proj_dir)) {
		Error err = da->make_dir_recursive(proj_dir);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	da->change_dir(proj_dir);
	da->make_dir("Core");
	da->make_dir("ObjectType");

	String core_dir = path_join(proj_dir, "Core");
	String obj_type_dir = path_join(proj_dir, "ObjectType");

	Vector<String> compile_items;

	for (OrderedHashMap<StringName, TypeInterface>::Element E = obj_types.front(); E; E = E.next()) {
		const TypeInterface &itype = E.get();

		if (itype.api_type != ClassDB::API_EDITOR)
			continue;

		String output_file = path_join(obj_type_dir, itype.proxy_name + ".cs");
		Error err = _generate_cs_type(itype, output_file);

		if (err == ERR_SKIP)
			continue;

		if (err != OK)
			return err;

		compile_items.push_back(output_file);
	}

	List<String> cs_icalls_content;

	cs_icalls_content.push_back("using System;\n"
								"using System.Runtime.CompilerServices;\n"
								"\n");
	cs_icalls_content.push_back("namespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);
	cs_icalls_content.push_back(INDENT1 "internal static class " BINDINGS_CLASS_NATIVECALLS_EDITOR "\n" INDENT1 OPEN_BLOCK);

	cs_icalls_content.push_back(INDENT2 "internal static ulong godot_api_hash = ");
	cs_icalls_content.push_back(String::num_uint64(GDMono::get_singleton()->get_api_editor_hash()) + ";\n");
	cs_icalls_content.push_back(INDENT2 "internal static uint bindings_version = ");
	cs_icalls_content.push_back(String::num_uint64(BINDINGS_GENERATOR_VERSION) + ";\n");
	cs_icalls_content.push_back(INDENT2 "internal static uint cs_glue_version = ");
	cs_icalls_content.push_back(String::num_uint64(CS_GLUE_VERSION) + ";\n");
	cs_icalls_content.push_back("\n");

#define ADD_INTERNAL_CALL(m_icall)                                                             \
	if (m_icall.editor_only) {                                                                 \
		cs_icalls_content.push_back(INDENT2 "[MethodImpl(MethodImplOptions.InternalCall)]\n"); \
		cs_icalls_content.push_back(INDENT2 "internal extern static ");                        \
		cs_icalls_content.push_back(m_icall.im_type_out + " ");                                \
		cs_icalls_content.push_back(m_icall.name + "(");                                       \
		cs_icalls_content.push_back(m_icall.im_sig + ");\n");                                  \
	}

	for (const List<InternalCall>::Element *E = editor_custom_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());
	for (const List<InternalCall>::Element *E = method_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());

#undef ADD_INTERNAL_CALL

	cs_icalls_content.push_back(INDENT1 CLOSE_BLOCK CLOSE_BLOCK);

	String internal_methods_file = path_join(core_dir, BINDINGS_CLASS_NATIVECALLS_EDITOR ".cs");

	Error err = _save_file(internal_methods_file, cs_icalls_content);
	if (err != OK)
		return err;

	compile_items.push_back(internal_methods_file);

	String guid = CSharpProject::generate_editor_api_project(proj_dir, "../" CORE_API_ASSEMBLY_NAME "/" CORE_API_ASSEMBLY_NAME ".csproj", compile_items);

	DotNetSolution::ProjectInfo proj_info;
	proj_info.guid = guid;
	proj_info.relpath = String(EDITOR_API_ASSEMBLY_NAME).plus_file(EDITOR_API_ASSEMBLY_NAME ".csproj");
	proj_info.configs.push_back("Debug");
	proj_info.configs.push_back("Release");

	r_solution.add_new_project(EDITOR_API_ASSEMBLY_NAME, proj_info);

	if (verbose_output)
		OS::get_singleton()->print("The solution and C# project for the Editor API was generated successfully\n");

	return OK;
}

Error BindingsGenerator::generate_cs_api(const String &p_output_dir, bool p_verbose_output) {

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);

	if (!DirAccess::exists(p_output_dir)) {
		Error err = da->make_dir_recursive(p_output_dir);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	DotNetSolution solution(API_SOLUTION_NAME);

	if (!solution.set_path(p_output_dir))
		return ERR_FILE_NOT_FOUND;

	Error proj_err;

	proj_err = generate_cs_core_project(p_output_dir, solution, p_verbose_output);
	if (proj_err != OK) {
		ERR_PRINT("Generation of the Core API C# project failed");
		return proj_err;
	}

	proj_err = generate_cs_editor_project(p_output_dir, solution, p_verbose_output);
	if (proj_err != OK) {
		ERR_PRINT("Generation of the Editor API C# project failed");
		return proj_err;
	}

	Error sln_error = solution.save();
	if (sln_error != OK) {
		ERR_PRINT("Failed to save API solution");
		return sln_error;
	}

	return OK;
}

// FIXME: There are some members that hide other inherited members.
// - In the case of both members being the same kind, the new one must be declared
// explicitly as `new` to avoid the warning (and we must print a message about it).
// - In the case of both members being of a different kind, then the new one must
// be renamed to avoid the name collision (and we must print a warning about it).
// - Csc warning e.g.:
// ObjectType/LineEdit.cs(140,38): warning CS0108: 'LineEdit.FocusMode' hides inherited member 'Control.FocusMode'. Use the new keyword if hiding was intended.
Error BindingsGenerator::_generate_cs_type(const TypeInterface &itype, const String &p_output_file) {

	CRASH_COND(!itype.is_object_type);

	bool is_derived_type = itype.base_name != StringName();

	if (!is_derived_type) {
		// Some Godot.Object assertions
		CRASH_COND(itype.cname != name_cache.type_Object);
		CRASH_COND(!itype.is_instantiable);
		CRASH_COND(itype.api_type != ClassDB::API_CORE);
		CRASH_COND(itype.is_reference);
		CRASH_COND(itype.is_singleton);
	}

	List<InternalCall> &custom_icalls = itype.api_type == ClassDB::API_EDITOR ? editor_custom_icalls : core_custom_icalls;

	if (verbose_output)
		OS::get_singleton()->print("Generating %s.cs...\n", itype.proxy_name.utf8().get_data());

	String ctor_method(ICALL_PREFIX + itype.proxy_name + "_Ctor"); // Used only for derived types

	List<String> output;

	output.push_back("using System;\n"); // IntPtr
	output.push_back("using System.Diagnostics;\n"); // DebuggerBrowsable

	output.push_back("\n"
					 "#pragma warning disable CS1591 // Disable warning: "
					 "'Missing XML comment for publicly visible type or member'\n"
					 "#pragma warning disable CS1573 // Disable warning: "
					 "'Parameter has no matching param tag in the XML comment'\n");

	output.push_back("\nnamespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);

	const DocData::ClassDoc *class_doc = itype.class_doc;

	if (class_doc && class_doc->description.size()) {
		String xml_summary = bbcode_to_xml(fix_doc_description(class_doc->description), &itype);
		Vector<String> summary_lines = xml_summary.split("\n");

		if (summary_lines.size()) {
			output.push_back(INDENT1 "/// <summary>\n");

			for (int i = 0; i < summary_lines.size(); i++) {
				output.push_back(INDENT1 "/// ");
				output.push_back(summary_lines[i]);
				output.push_back("\n");
			}

			output.push_back(INDENT1 "/// </summary>\n");
		}
	}

	output.push_back(INDENT1 "public ");
	if (itype.is_singleton) {
		output.push_back("static partial class ");
	} else {
		output.push_back(itype.is_instantiable ? "partial class " : "abstract partial class ");
	}
	output.push_back(itype.proxy_name);

	if (itype.is_singleton) {
		output.push_back("\n");
	} else if (is_derived_type) {
		if (obj_types.has(itype.base_name)) {
			output.push_back(" : ");
			output.push_back(obj_types[itype.base_name].proxy_name);
			output.push_back("\n");
		} else {
			ERR_PRINTS("Base type '" + itype.base_name.operator String() + "' does not exist, for class " + itype.name);
			return ERR_INVALID_DATA;
		}
	}

	output.push_back(INDENT1 "{");

	if (class_doc) {

		// Add constants

		for (const List<ConstantInterface>::Element *E = itype.constants.front(); E; E = E->next()) {
			const ConstantInterface &iconstant = E->get();

			if (iconstant.const_doc && iconstant.const_doc->description.size()) {
				String xml_summary = bbcode_to_xml(fix_doc_description(iconstant.const_doc->description), &itype);
				Vector<String> summary_lines = xml_summary.split("\n");

				if (summary_lines.size()) {
					output.push_back(MEMBER_BEGIN "/// <summary>\n");

					for (int i = 0; i < summary_lines.size(); i++) {
						output.push_back(INDENT2 "/// ");
						output.push_back(summary_lines[i]);
						output.push_back("\n");
					}

					output.push_back(INDENT2 "/// </summary>");
				}
			}

			output.push_back(MEMBER_BEGIN "public const int ");
			output.push_back(iconstant.proxy_name);
			output.push_back(" = ");
			output.push_back(itos(iconstant.value));
			output.push_back(";");
		}

		if (itype.constants.size())
			output.push_back("\n");

		// Add enums

		for (const List<EnumInterface>::Element *E = itype.enums.front(); E; E = E->next()) {
			const EnumInterface &ienum = E->get();

			ERR_FAIL_COND_V(ienum.constants.empty(), ERR_BUG);

			output.push_back(MEMBER_BEGIN "public enum ");
			output.push_back(ienum.cname.operator String());
			output.push_back(MEMBER_BEGIN OPEN_BLOCK);

			for (const List<ConstantInterface>::Element *F = ienum.constants.front(); F; F = F->next()) {
				const ConstantInterface &iconstant = F->get();

				if (iconstant.const_doc && iconstant.const_doc->description.size()) {
					String xml_summary = bbcode_to_xml(fix_doc_description(iconstant.const_doc->description), &itype);
					Vector<String> summary_lines = xml_summary.split("\n");

					if (summary_lines.size()) {
						output.push_back(INDENT3 "/// <summary>\n");

						for (int i = 0; i < summary_lines.size(); i++) {
							output.push_back(INDENT3 "/// ");
							output.push_back(summary_lines[i]);
							output.push_back("\n");
						}

						output.push_back(INDENT3 "/// </summary>\n");
					}
				}

				output.push_back(INDENT3);
				output.push_back(iconstant.proxy_name);
				output.push_back(" = ");
				output.push_back(itos(iconstant.value));
				output.push_back(F != ienum.constants.back() ? ",\n" : "\n");
			}

			output.push_back(INDENT2 CLOSE_BLOCK);
		}

		// Add properties

		for (const List<PropertyInterface>::Element *E = itype.properties.front(); E; E = E->next()) {
			const PropertyInterface &iprop = E->get();
			Error prop_err = _generate_cs_property(itype, iprop, output);
			if (prop_err != OK) {
				ERR_EXPLAIN("Failed to generate property '" + iprop.cname.operator String() +
							"' for class '" + itype.name + "'");
				ERR_FAIL_V(prop_err);
			}
		}
	}

	// TODO: BINDINGS_NATIVE_NAME_FIELD should be StringName, once we support it in C#

	if (itype.is_singleton) {
		// Add the type name and the singleton pointer as static fields

		output.push_back(MEMBER_BEGIN "private static Godot.Object singleton;\n");
		output.push_back(MEMBER_BEGIN "public static Godot.Object Singleton\n" INDENT2 "{\n" INDENT3
									  "get\n" INDENT3 "{\n" INDENT4 "if (singleton == null)\n" INDENT5
									  "singleton = Engine.GetSingleton(" BINDINGS_NATIVE_NAME_FIELD ");\n" INDENT4
									  "return singleton;\n" INDENT3 "}\n" INDENT2 "}\n");

		output.push_back(MEMBER_BEGIN "private const string " BINDINGS_NATIVE_NAME_FIELD " = \"");
		output.push_back(itype.name);
		output.push_back("\";\n");

		output.push_back(INDENT2 "internal static IntPtr " BINDINGS_PTR_FIELD " = ");
		output.push_back(itype.api_type == ClassDB::API_EDITOR ? BINDINGS_CLASS_NATIVECALLS_EDITOR : BINDINGS_CLASS_NATIVECALLS);
		output.push_back("." ICALL_PREFIX);
		output.push_back(itype.name);
		output.push_back(SINGLETON_ICALL_SUFFIX "();\n");
	} else if (is_derived_type) {
		// Add member fields

		output.push_back(MEMBER_BEGIN "private const string " BINDINGS_NATIVE_NAME_FIELD " = \"");
		output.push_back(itype.name);
		output.push_back("\";\n");

		// Add default constructor
		if (itype.is_instantiable) {
			output.push_back(MEMBER_BEGIN "public ");
			output.push_back(itype.proxy_name);
			output.push_back("() : this(");
			output.push_back(itype.memory_own ? "true" : "false");

			// The default constructor may also be called by the engine when instancing existing native objects
			// The engine will initialize the pointer field of the managed side before calling the constructor
			// This is why we only allocate a new native object from the constructor if the pointer field is not set
			output.push_back(")\n" OPEN_BLOCK_L2 "if (" BINDINGS_PTR_FIELD " == IntPtr.Zero)\n" INDENT4 BINDINGS_PTR_FIELD " = ");
			output.push_back(itype.api_type == ClassDB::API_EDITOR ? BINDINGS_CLASS_NATIVECALLS_EDITOR : BINDINGS_CLASS_NATIVECALLS);
			output.push_back("." + ctor_method);
			output.push_back("(this);\n" CLOSE_BLOCK_L2);
		} else {
			// Hide the constructor
			output.push_back(MEMBER_BEGIN "internal ");
			output.push_back(itype.proxy_name);
			output.push_back("() {}\n");
		}

		// Add.. em.. trick constructor. Sort of.
		output.push_back(MEMBER_BEGIN "internal ");
		output.push_back(itype.proxy_name);
		output.push_back("(bool " CS_FIELD_MEMORYOWN ") : base(" CS_FIELD_MEMORYOWN ") {}\n");
	}

	int method_bind_count = 0;
	for (const List<MethodInterface>::Element *E = itype.methods.front(); E; E = E->next()) {
		const MethodInterface &imethod = E->get();
		Error method_err = _generate_cs_method(itype, imethod, method_bind_count, output);
		if (method_err != OK) {
			ERR_EXPLAIN("Failed to generate method '" + imethod.name + "' for class '" + itype.name + "'");
			ERR_FAIL_V(method_err);
		}
	}

	if (itype.is_singleton) {
		InternalCall singleton_icall = InternalCall(itype.api_type, ICALL_PREFIX + itype.name + SINGLETON_ICALL_SUFFIX, "IntPtr");

		if (!find_icall_by_name(singleton_icall.name, custom_icalls))
			custom_icalls.push_back(singleton_icall);
	}

	if (is_derived_type && itype.is_instantiable) {
		InternalCall ctor_icall = InternalCall(itype.api_type, ctor_method, "IntPtr", itype.proxy_name + " obj");

		if (!find_icall_by_name(ctor_icall.name, custom_icalls))
			custom_icalls.push_back(ctor_icall);
	}

	output.push_back(INDENT1 CLOSE_BLOCK /* class */
					CLOSE_BLOCK /* namespace */);

	output.push_back("\n"
					 "#pragma warning restore CS1591\n"
					 "#pragma warning restore CS1573\n");

	return _save_file(p_output_file, output);
}

Error BindingsGenerator::_generate_cs_property(const BindingsGenerator::TypeInterface &p_itype, const PropertyInterface &p_iprop, List<String> &p_output) {

	const MethodInterface *setter = p_itype.find_method_by_name(p_iprop.setter);

	// Search it in base types too
	const TypeInterface *current_type = &p_itype;
	while (!setter && current_type->base_name != StringName()) {
		OrderedHashMap<StringName, TypeInterface>::Element base_match = obj_types.find(current_type->base_name);
		ERR_FAIL_COND_V(!base_match, ERR_BUG);
		current_type = &base_match.get();
		setter = current_type->find_method_by_name(p_iprop.setter);
	}

	const MethodInterface *getter = p_itype.find_method_by_name(p_iprop.getter);

	// Search it in base types too
	current_type = &p_itype;
	while (!getter && current_type->base_name != StringName()) {
		OrderedHashMap<StringName, TypeInterface>::Element base_match = obj_types.find(current_type->base_name);
		ERR_FAIL_COND_V(!base_match, ERR_BUG);
		current_type = &base_match.get();
		getter = current_type->find_method_by_name(p_iprop.getter);
	}

	ERR_FAIL_COND_V(!setter && !getter, ERR_BUG);

	if (setter) {
		int setter_argc = p_iprop.index != -1 ? 2 : 1;
		ERR_FAIL_COND_V(setter->arguments.size() != setter_argc, ERR_BUG);
	}

	if (getter) {
		int getter_argc = p_iprop.index != -1 ? 1 : 0;
		ERR_FAIL_COND_V(getter->arguments.size() != getter_argc, ERR_BUG);
	}

	if (getter && setter) {
		ERR_FAIL_COND_V(getter->return_type.cname != setter->arguments.back()->get().type.cname, ERR_BUG);
	}

	const TypeReference &proptype_name = getter ? getter->return_type : setter->arguments.back()->get().type;

	const TypeInterface *prop_itype = _get_type_or_null(proptype_name);
	ERR_FAIL_NULL_V(prop_itype, ERR_BUG); // Property type not found

	if (p_iprop.prop_doc && p_iprop.prop_doc->description.size()) {
		String xml_summary = bbcode_to_xml(fix_doc_description(p_iprop.prop_doc->description), &p_itype);
		Vector<String> summary_lines = xml_summary.split("\n");

		if (summary_lines.size()) {
			p_output.push_back(MEMBER_BEGIN "/// <summary>\n");

			for (int i = 0; i < summary_lines.size(); i++) {
				p_output.push_back(INDENT2 "/// ");
				p_output.push_back(summary_lines[i]);
				p_output.push_back("\n");
			}

			p_output.push_back(INDENT2 "/// </summary>");
		}
	}

	p_output.push_back(MEMBER_BEGIN "public ");

	if (p_itype.is_singleton)
		p_output.push_back("static ");

	p_output.push_back(prop_itype->cs_type);
	p_output.push_back(" ");
	p_output.push_back(p_iprop.proxy_name);
	p_output.push_back("\n" INDENT2 OPEN_BLOCK);

	if (getter) {
		p_output.push_back(INDENT3 "get\n" OPEN_BLOCK_L3);
		p_output.push_back("return ");
		p_output.push_back(getter->proxy_name + "(");
		if (p_iprop.index != -1) {
			const ArgumentInterface &idx_arg = getter->arguments.front()->get();
			if (idx_arg.type.cname != name_cache.type_int) {
				// Assume the index parameter is an enum
				const TypeInterface *idx_arg_type = _get_type_or_null(idx_arg.type);
				CRASH_COND(idx_arg_type == NULL);
				p_output.push_back("(" + idx_arg_type->proxy_name + ")" + itos(p_iprop.index));
			} else {
				p_output.push_back(itos(p_iprop.index));
			}
		}
		p_output.push_back(");\n" CLOSE_BLOCK_L3);
	}

	if (setter) {
		p_output.push_back(INDENT3 "set\n" OPEN_BLOCK_L3);
		p_output.push_back(setter->proxy_name + "(");
		if (p_iprop.index != -1) {
			const ArgumentInterface &idx_arg = setter->arguments.front()->get();
			if (idx_arg.type.cname != name_cache.type_int) {
				// Assume the index parameter is an enum
				const TypeInterface *idx_arg_type = _get_type_or_null(idx_arg.type);
				CRASH_COND(idx_arg_type == NULL);
				p_output.push_back("(" + idx_arg_type->proxy_name + ")" + itos(p_iprop.index) + ", ");
			} else {
				p_output.push_back(itos(p_iprop.index) + ", ");
			}
		}
		p_output.push_back("value);\n" CLOSE_BLOCK_L3);
	}

	p_output.push_back(CLOSE_BLOCK_L2);

	return OK;
}

Error BindingsGenerator::_generate_cs_method(const BindingsGenerator::TypeInterface &p_itype, const BindingsGenerator::MethodInterface &p_imethod, int &p_method_bind_count, List<String> &p_output) {

	const TypeInterface *return_type = _get_type_or_placeholder(p_imethod.return_type);

	String method_bind_field = "method_bind_" + itos(p_method_bind_count);

	String arguments_sig;
	String cs_in_statements;

	String icall_params = method_bind_field + ", ";
	icall_params += sformat(p_itype.cs_in, "this");

	List<String> default_args_doc;

	// Retrieve information from the arguments
	for (const List<ArgumentInterface>::Element *F = p_imethod.arguments.front(); F; F = F->next()) {
		const ArgumentInterface &iarg = F->get();
		const TypeInterface *arg_type = _get_type_or_placeholder(iarg.type);

		// Add the current arguments to the signature
		// If the argument has a default value which is not a constant, we will make it Nullable
		{
			if (F != p_imethod.arguments.front())
				arguments_sig += ", ";

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				arguments_sig += "Nullable<";

			arguments_sig += arg_type->cs_type;

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				arguments_sig += "> ";
			else
				arguments_sig += " ";

			arguments_sig += iarg.name;

			if (iarg.default_argument.size()) {
				if (iarg.def_param_mode != ArgumentInterface::CONSTANT)
					arguments_sig += " = null";
				else
					arguments_sig += " = " + sformat(iarg.default_argument, arg_type->cs_type);
			}
		}

		icall_params += ", ";

		if (iarg.default_argument.size() && iarg.def_param_mode != ArgumentInterface::CONSTANT) {
			// The default value of an argument must be constant. Otherwise we make it Nullable and do the following:
			// Type arg_in = arg.HasValue ? arg.Value : <non-const default value>;
			String arg_in = iarg.name;
			arg_in += "_in";

			cs_in_statements += arg_type->cs_type;
			cs_in_statements += " ";
			cs_in_statements += arg_in;
			cs_in_statements += " = ";
			cs_in_statements += iarg.name;

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				cs_in_statements += ".HasValue ? ";
			else
				cs_in_statements += " != null ? ";

			cs_in_statements += iarg.name;

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				cs_in_statements += ".Value : ";
			else
				cs_in_statements += " : ";

			String def_arg = sformat(iarg.default_argument, arg_type->cs_type);

			cs_in_statements += def_arg;
			cs_in_statements += ";\n" INDENT3;

			icall_params += arg_type->cs_in.empty() ? arg_in : sformat(arg_type->cs_in, arg_in);

			// Apparently the name attribute must not include the @
			String param_tag_name = iarg.name.begins_with("@") ? iarg.name.substr(1, iarg.name.length()) : iarg.name;

			default_args_doc.push_back(INDENT2 "/// <param name=\"" + param_tag_name + "\">If the parameter is null, then the default value is " + def_arg + "</param>\n");
		} else {
			icall_params += arg_type->cs_in.empty() ? iarg.name : sformat(arg_type->cs_in, iarg.name);
		}
	}

	// Generate method
	{
		if (!p_imethod.is_virtual && !p_imethod.requires_object_call) {
			p_output.push_back(MEMBER_BEGIN "[DebuggerBrowsable(DebuggerBrowsableState.Never)]" MEMBER_BEGIN "private static IntPtr ");
			p_output.push_back(method_bind_field + " = Object." ICALL_GET_METHODBIND "(" BINDINGS_NATIVE_NAME_FIELD ", \"");
			p_output.push_back(p_imethod.name);
			p_output.push_back("\");\n");
		}

		if (p_imethod.method_doc && p_imethod.method_doc->description.size()) {
			String xml_summary = bbcode_to_xml(fix_doc_description(p_imethod.method_doc->description), &p_itype);
			Vector<String> summary_lines = xml_summary.split("\n");

			if (summary_lines.size() || default_args_doc.size()) {
				p_output.push_back(MEMBER_BEGIN "/// <summary>\n");

				for (int i = 0; i < summary_lines.size(); i++) {
					p_output.push_back(INDENT2 "/// ");
					p_output.push_back(summary_lines[i]);
					p_output.push_back("\n");
				}

				for (List<String>::Element *E = default_args_doc.front(); E; E = E->next()) {
					p_output.push_back(E->get());
				}

				p_output.push_back(INDENT2 "/// </summary>");
			}
		}

		if (!p_imethod.is_internal) {
			p_output.push_back(MEMBER_BEGIN "[GodotMethod(\"");
			p_output.push_back(p_imethod.name);
			p_output.push_back("\")]");
		}

		p_output.push_back(MEMBER_BEGIN);
		p_output.push_back(p_imethod.is_internal ? "internal " : "public ");

		if (p_itype.is_singleton) {
			p_output.push_back("static ");
		} else if (p_imethod.is_virtual) {
			p_output.push_back("virtual ");
		}

		p_output.push_back(return_type->cs_type + " ");
		p_output.push_back(p_imethod.proxy_name + "(");
		p_output.push_back(arguments_sig + ")\n" OPEN_BLOCK_L2);

		if (p_imethod.is_virtual) {
			// Godot virtual method must be overridden, therefore we return a default value by default.

			if (return_type->cname == name_cache.type_void) {
				p_output.push_back("return;\n" CLOSE_BLOCK_L2);
			} else {
				p_output.push_back("return default(");
				p_output.push_back(return_type->cs_type);
				p_output.push_back(");\n" CLOSE_BLOCK_L2);
			}

			return OK; // Won't increment method bind count
		}

		if (p_imethod.requires_object_call) {
			// Fallback to Godot's object.Call(string, params)

			p_output.push_back(CS_METHOD_CALL "(\"");
			p_output.push_back(p_imethod.name);
			p_output.push_back("\"");

			for (const List<ArgumentInterface>::Element *F = p_imethod.arguments.front(); F; F = F->next()) {
				p_output.push_back(", ");
				p_output.push_back(F->get().name);
			}

			p_output.push_back(");\n" CLOSE_BLOCK_L2);

			return OK; // Won't increment method bind count
		}

		const Map<const MethodInterface *, const InternalCall *>::Element *match = method_icalls_map.find(&p_imethod);
		ERR_FAIL_NULL_V(match, ERR_BUG);

		const InternalCall *im_icall = match->value();

		String im_call = im_icall->editor_only ? BINDINGS_CLASS_NATIVECALLS_EDITOR : BINDINGS_CLASS_NATIVECALLS;
		im_call += "." + im_icall->name + "(" + icall_params + ")";

		if (p_imethod.arguments.size())
			p_output.push_back(cs_in_statements);

		if (return_type->cname == name_cache.type_void) {
			p_output.push_back(im_call + ";\n");
		} else if (return_type->cs_out.empty()) {
			p_output.push_back("return " + im_call + ";\n");
		} else {
			p_output.push_back(sformat(return_type->cs_out, im_call, return_type->cs_type, return_type->im_type_out));
			p_output.push_back("\n");
		}

		p_output.push_back(CLOSE_BLOCK_L2);
	}

	p_method_bind_count++;
	return OK;
}

Error BindingsGenerator::generate_glue(const String &p_output_dir) {

	verbose_output = true;

	bool dir_exists = DirAccess::exists(p_output_dir);
	ERR_EXPLAIN("The output directory does not exist.");
	ERR_FAIL_COND_V(!dir_exists, ERR_FILE_BAD_PATH);

	List<String> output;

	output.push_back("/* THIS FILE IS GENERATED DO NOT EDIT */\n");
	output.push_back("#include \"" GLUE_HEADER_FILE "\"\n");
	output.push_back("\n#ifdef MONO_GLUE_ENABLED\n");

	generated_icall_funcs.clear();

	for (OrderedHashMap<StringName, TypeInterface>::Element type_elem = obj_types.front(); type_elem; type_elem = type_elem.next()) {
		const TypeInterface &itype = type_elem.get();

		bool is_derived_type = itype.base_name != StringName();

		if (!is_derived_type) {
			// Some Object assertions
			CRASH_COND(itype.cname != name_cache.type_Object);
			CRASH_COND(!itype.is_instantiable);
			CRASH_COND(itype.api_type != ClassDB::API_CORE);
			CRASH_COND(itype.is_reference);
			CRASH_COND(itype.is_singleton);
		}

		List<InternalCall> &custom_icalls = itype.api_type == ClassDB::API_EDITOR ? editor_custom_icalls : core_custom_icalls;

		OS::get_singleton()->print("Generating %s...\n", itype.name.utf8().get_data());

		String ctor_method(ICALL_PREFIX + itype.proxy_name + "_Ctor"); // Used only for derived types

		for (const List<MethodInterface>::Element *E = itype.methods.front(); E; E = E->next()) {
			const MethodInterface &imethod = E->get();
			Error method_err = _generate_glue_method(itype, imethod, output);
			if (method_err != OK) {
				ERR_EXPLAIN("Failed to generate method '" + imethod.name + "' for class '" + itype.name + "'");
				ERR_FAIL_V(method_err);
			}
		}

		if (itype.is_singleton) {
			String singleton_icall_name = ICALL_PREFIX + itype.name + SINGLETON_ICALL_SUFFIX;
			InternalCall singleton_icall = InternalCall(itype.api_type, singleton_icall_name, "IntPtr");

			if (!find_icall_by_name(singleton_icall.name, custom_icalls))
				custom_icalls.push_back(singleton_icall);

			output.push_back("Object* ");
			output.push_back(singleton_icall_name);
			output.push_back("() " OPEN_BLOCK "\treturn Engine::get_singleton()->get_singleton_object(\"");
			output.push_back(itype.proxy_name);
			output.push_back("\");\n" CLOSE_BLOCK "\n");
		}

		if (is_derived_type && itype.is_instantiable) {
			InternalCall ctor_icall = InternalCall(itype.api_type, ctor_method, "IntPtr", itype.proxy_name + " obj");

			if (!find_icall_by_name(ctor_icall.name, custom_icalls))
				custom_icalls.push_back(ctor_icall);

			output.push_back("Object* ");
			output.push_back(ctor_method);
			output.push_back("(MonoObject* obj) " OPEN_BLOCK
							 "\t" C_MACRO_OBJECT_CONSTRUCT "(instance, \"");
			output.push_back(itype.name);
			output.push_back("\");\n"
							 "\t" C_METHOD_TIE_MANAGED_TO_UNMANAGED "(obj, instance);\n"
							 "\treturn instance;\n" CLOSE_BLOCK "\n");
		}
	}

	output.push_back("namespace GodotSharpBindings\n" OPEN_BLOCK "\n");

	output.push_back("uint64_t get_core_api_hash() { return ");
	output.push_back(String::num_uint64(GDMono::get_singleton()->get_api_core_hash()) + "U; }\n");

	output.push_back("#ifdef TOOLS_ENABLED\n"
					 "uint64_t get_editor_api_hash() { return ");
	output.push_back(String::num_uint64(GDMono::get_singleton()->get_api_editor_hash()) + "U; }\n");
	output.push_back("#endif // TOOLS_ENABLED\n");

	output.push_back("uint32_t get_bindings_version() { return ");
	output.push_back(String::num_uint64(BINDINGS_GENERATOR_VERSION) + "; }\n");

	output.push_back("\nvoid register_generated_icalls() " OPEN_BLOCK);
	output.push_back("\tgodot_register_glue_header_icalls();\n");

#define ADD_INTERNAL_CALL_REGISTRATION(m_icall)                                                                 \
	{                                                                                                           \
		output.push_back("\tmono_add_internal_call(");                                                          \
		output.push_back("\"" BINDINGS_NAMESPACE ".");                                                          \
		output.push_back(m_icall.editor_only ? BINDINGS_CLASS_NATIVECALLS_EDITOR : BINDINGS_CLASS_NATIVECALLS); \
		output.push_back("::");                                                                                 \
		output.push_back(m_icall.name);                                                                         \
		output.push_back("\", (void*)");                                                                        \
		output.push_back(m_icall.name);                                                                         \
		output.push_back(");\n");                                                                               \
	}

	bool tools_sequence = false;
	for (const List<InternalCall>::Element *E = core_custom_icalls.front(); E; E = E->next()) {

		if (tools_sequence) {
			if (!E->get().editor_only) {
				tools_sequence = false;
				output.push_back("#endif\n");
			}
		} else {
			if (E->get().editor_only) {
				output.push_back("#ifdef TOOLS_ENABLED\n");
				tools_sequence = true;
			}
		}

		ADD_INTERNAL_CALL_REGISTRATION(E->get());
	}

	if (tools_sequence) {
		tools_sequence = false;
		output.push_back("#endif\n");
	}

	output.push_back("#ifdef TOOLS_ENABLED\n");
	for (const List<InternalCall>::Element *E = editor_custom_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL_REGISTRATION(E->get());
	output.push_back("#endif // TOOLS_ENABLED\n");

	for (const List<InternalCall>::Element *E = method_icalls.front(); E; E = E->next()) {
		if (tools_sequence) {
			if (!E->get().editor_only) {
				tools_sequence = false;
				output.push_back("#endif\n");
			}
		} else {
			if (E->get().editor_only) {
				output.push_back("#ifdef TOOLS_ENABLED\n");
				tools_sequence = true;
			}
		}

		ADD_INTERNAL_CALL_REGISTRATION(E->get());
	}

	if (tools_sequence) {
		tools_sequence = false;
		output.push_back("#endif\n");
	}

#undef ADD_INTERNAL_CALL_REGISTRATION

	output.push_back(CLOSE_BLOCK "\n} // namespace GodotSharpBindings\n");

	output.push_back("\n#endif // MONO_GLUE_ENABLED\n");

	Error save_err = _save_file(path_join(p_output_dir, "mono_glue.gen.cpp"), output);
	if (save_err != OK)
		return save_err;

	OS::get_singleton()->print("Mono glue generated successfully\n");

	return OK;
}

uint32_t BindingsGenerator::get_version() {
	return BINDINGS_GENERATOR_VERSION;
}

Error BindingsGenerator::_save_file(const String &p_path, const List<String> &p_content) {

	FileAccessRef file = FileAccess::open(p_path, FileAccess::WRITE);

	ERR_EXPLAIN("Cannot open file: " + p_path);
	ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);

	for (const List<String>::Element *E = p_content.front(); E; E = E->next()) {
		file->store_string(E->get());
	}

	file->close();

	return OK;
}

Error BindingsGenerator::_generate_glue_method(const BindingsGenerator::TypeInterface &p_itype, const BindingsGenerator::MethodInterface &p_imethod, List<String> &p_output) {

	if (p_imethod.is_virtual)
		return OK; // Ignore

	bool ret_void = p_imethod.return_type.cname == name_cache.type_void;

	const TypeInterface *return_type = _get_type_or_placeholder(p_imethod.return_type);

	String argc_str = itos(p_imethod.arguments.size());

	String c_func_sig = "MethodBind* " CS_PARAM_METHODBIND ", " + p_itype.c_type_in + " " CS_PARAM_INSTANCE;
	String c_in_statements;
	String c_args_var_content;

	// Get arguments information
	int i = 0;
	for (const List<ArgumentInterface>::Element *F = p_imethod.arguments.front(); F; F = F->next()) {
		const ArgumentInterface &iarg = F->get();
		const TypeInterface *arg_type = _get_type_or_placeholder(iarg.type);

		String c_param_name = "arg" + itos(i + 1);

		if (p_imethod.is_vararg) {
			if (i < p_imethod.arguments.size() - 1) {
				c_in_statements += sformat(arg_type->c_in.size() ? arg_type->c_in : TypeInterface::DEFAULT_VARARG_C_IN, "Variant", c_param_name);
				c_in_statements += "\t" C_LOCAL_PTRCALL_ARGS ".set(";
				c_in_statements += itos(i);
				c_in_statements += sformat(", &%s_in);\n", c_param_name);
			}
		} else {
			if (i > 0)
				c_args_var_content += ", ";
			if (arg_type->c_in.size())
				c_in_statements += sformat(arg_type->c_in, arg_type->c_type, c_param_name);
			c_args_var_content += sformat(arg_type->c_arg_in, c_param_name);
		}

		c_func_sig += ", ";
		c_func_sig += arg_type->c_type_in;
		c_func_sig += " ";
		c_func_sig += c_param_name;

		i++;
	}

	const Map<const MethodInterface *, const InternalCall *>::Element *match = method_icalls_map.find(&p_imethod);
	ERR_FAIL_NULL_V(match, ERR_BUG);

	const InternalCall *im_icall = match->value();
	String icall_method = im_icall->name;

	if (!generated_icall_funcs.find(im_icall)) {
		generated_icall_funcs.push_back(im_icall);

		if (im_icall->editor_only)
			p_output.push_back("#ifdef TOOLS_ENABLED\n");

		// Generate icall function

		p_output.push_back(ret_void ? "void " : return_type->c_type_out + " ");
		p_output.push_back(icall_method);
		p_output.push_back("(");
		p_output.push_back(c_func_sig);
		p_output.push_back(") " OPEN_BLOCK);

		String fail_ret = ret_void ? "" : ", " + (return_type->c_type_out.ends_with("*") ? "NULL" : return_type->c_type_out + "()");

		if (!ret_void) {
			String ptrcall_return_type;
			String initialization;

			if (p_imethod.is_vararg && return_type->cname != name_cache.type_Variant) {
				// VarArg methods always return Variant, but there are some cases in which MethodInfo provides
				// a specific return type. We trust this information is valid. We need a temporary local to keep
				// the Variant alive until the method returns. Otherwise, if the returned Variant holds a RefPtr,
				// it could be deleted too early. This is the case with GDScript.new() which returns OBJECT.
				// Alternatively, we could just return Variant, but that would result in a worse API.
				p_output.push_back("\tVariant " C_LOCAL_VARARG_RET ";\n");
			}

			if (return_type->is_object_type) {
				ptrcall_return_type = return_type->is_reference ? "Ref<Reference>" : return_type->c_type;
				initialization = return_type->is_reference ? "" : " = NULL";
			} else {
				ptrcall_return_type = return_type->c_type;
			}

			p_output.push_back("\t" + ptrcall_return_type);
			p_output.push_back(" " C_LOCAL_RET);
			p_output.push_back(initialization + ";\n");
			p_output.push_back("\tERR_FAIL_NULL_V(" CS_PARAM_INSTANCE);
			p_output.push_back(fail_ret);
			p_output.push_back(");\n");
		} else {
			p_output.push_back("\tERR_FAIL_NULL(" CS_PARAM_INSTANCE ");\n");
		}

		if (p_imethod.arguments.size()) {
			if (p_imethod.is_vararg) {
				String err_fail_macro = ret_void ? "ERR_FAIL_COND" : "ERR_FAIL_COND_V";
				String vararg_arg = "arg" + argc_str;
				String real_argc_str = itos(p_imethod.arguments.size() - 1); // Arguments count without vararg

				p_output.push_back("\tint vararg_length = mono_array_length(");
				p_output.push_back(vararg_arg);
				p_output.push_back(");\n\tint total_length = ");
				p_output.push_back(real_argc_str);
				p_output.push_back(" + vararg_length;\n"
								   "\tArgumentsVector<Variant> varargs(vararg_length);\n"
								   "\tArgumentsVector<const Variant *> " C_LOCAL_PTRCALL_ARGS "(total_length);\n");
				p_output.push_back(c_in_statements);
				p_output.push_back("\tfor (int i = 0; i < vararg_length; i++) " OPEN_BLOCK
								   "\t\tMonoObject* elem = mono_array_get(");
				p_output.push_back(vararg_arg);
				p_output.push_back(", MonoObject*, i);\n"
								   "\t\tvarargs.set(i, GDMonoMarshal::mono_object_to_variant(elem));\n"
								   "\t\t" C_LOCAL_PTRCALL_ARGS ".set(");
				p_output.push_back(real_argc_str);
				p_output.push_back(" + i, &varargs.get(i));\n\t" CLOSE_BLOCK);
			} else {
				p_output.push_back(c_in_statements);
				p_output.push_back("\tconst void* " C_LOCAL_PTRCALL_ARGS "[");
				p_output.push_back(argc_str + "] = { ");
				p_output.push_back(c_args_var_content + " };\n");
			}
		}

		if (p_imethod.is_vararg) {
			p_output.push_back("\tVariant::CallError vcall_error;\n\t");

			if (!ret_void) {
				// See the comment on the C_LOCAL_VARARG_RET declaration
				if (return_type->cname != name_cache.type_Variant) {
					p_output.push_back(C_LOCAL_VARARG_RET " = ");
				} else {
					p_output.push_back(C_LOCAL_RET " = ");
				}
			}

			p_output.push_back(CS_PARAM_METHODBIND "->call(" CS_PARAM_INSTANCE ", ");
			p_output.push_back(p_imethod.arguments.size() ? C_LOCAL_PTRCALL_ARGS ".ptr()" : "NULL");
			p_output.push_back(", total_length, vcall_error);\n");

			// See the comment on the C_LOCAL_VARARG_RET declaration
			if (return_type->cname != name_cache.type_Variant) {
				p_output.push_back("\t" C_LOCAL_RET " = " C_LOCAL_VARARG_RET ";\n");
			}
		} else {
			p_output.push_back("\t" CS_PARAM_METHODBIND "->ptrcall(" CS_PARAM_INSTANCE ", ");
			p_output.push_back(p_imethod.arguments.size() ? C_LOCAL_PTRCALL_ARGS ", " : "NULL, ");
			p_output.push_back(!ret_void ? "&" C_LOCAL_RET ");\n" : "NULL);\n");
		}

		if (!ret_void) {
			if (return_type->c_out.empty())
				p_output.push_back("\treturn " C_LOCAL_RET ";\n");
			else
				p_output.push_back(sformat(return_type->c_out, return_type->c_type_out, C_LOCAL_RET, return_type->name));
		}

		p_output.push_back(CLOSE_BLOCK "\n");

		if (im_icall->editor_only)
			p_output.push_back("#endif // TOOLS_ENABLED\n");
	}

	return OK;
}

const BindingsGenerator::TypeInterface *BindingsGenerator::_get_type_or_null(const TypeReference &p_typeref) {

	const Map<StringName, TypeInterface>::Element *builtin_type_match = builtin_types.find(p_typeref.cname);

	if (builtin_type_match)
		return &builtin_type_match->get();

	const OrderedHashMap<StringName, TypeInterface>::Element obj_type_match = obj_types.find(p_typeref.cname);

	if (obj_type_match)
		return &obj_type_match.get();

	if (p_typeref.is_enum) {
		const Map<StringName, TypeInterface>::Element *enum_match = enum_types.find(p_typeref.cname);

		if (enum_match)
			return &enum_match->get();

		// Enum not found. Most likely because none of its constants were bound, so it's empty. That's fine. Use int instead.
		const Map<StringName, TypeInterface>::Element *int_match = builtin_types.find(name_cache.type_int);
		ERR_FAIL_NULL_V(int_match, NULL);
		return &int_match->get();
	}

	return NULL;
}

const BindingsGenerator::TypeInterface *BindingsGenerator::_get_type_or_placeholder(const TypeReference &p_typeref) {

	const TypeInterface *found = _get_type_or_null(p_typeref);

	if (found)
		return found;

	ERR_PRINTS(String() + "Type not found. Creating placeholder: " + p_typeref.cname.operator String());

	const Map<StringName, TypeInterface>::Element *match = placeholder_types.find(p_typeref.cname);

	if (match)
		return &match->get();

	TypeInterface placeholder;
	TypeInterface::create_placeholder_type(placeholder, p_typeref.cname);

	return &placeholder_types.insert(placeholder.cname, placeholder)->get();
}

void BindingsGenerator::_populate_object_type_interfaces() {

	obj_types.clear();

	List<StringName> class_list;
	ClassDB::get_class_list(&class_list);
	class_list.sort_custom<StringName::AlphCompare>();

	while (class_list.size()) {
		StringName type_cname = class_list.front()->get();

		ClassDB::APIType api_type = ClassDB::get_api_type(type_cname);

		if (api_type == ClassDB::API_NONE) {
			class_list.pop_front();
			continue;
		}

		if (!ClassDB::is_class_exposed(type_cname)) {
			if (verbose_output)
				WARN_PRINTS("Ignoring type " + type_cname.operator String() + " because it's not exposed");
			class_list.pop_front();
			continue;
		}

		if (!ClassDB::is_class_enabled(type_cname)) {
			if (verbose_output)
				WARN_PRINTS("Ignoring type " + type_cname.operator String() + " because it's not enabled");
			class_list.pop_front();
			continue;
		}

		ClassDB::ClassInfo *class_info = ClassDB::classes.getptr(type_cname);

		TypeInterface itype = TypeInterface::create_object_type(type_cname, api_type);

		itype.base_name = ClassDB::get_parent_class(type_cname);
		itype.is_singleton = Engine::get_singleton()->has_singleton(itype.proxy_name);
		itype.is_instantiable = ClassDB::can_instance(type_cname) && !itype.is_singleton;
		itype.is_reference = ClassDB::is_parent_class(type_cname, name_cache.type_Reference);
		itype.memory_own = itype.is_reference;

		itype.c_out = "\treturn ";
		itype.c_out += C_METHOD_UNMANAGED_GET_MANAGED;
		itype.c_out += itype.is_reference ? "(%1.ptr());\n" : "(%1);\n";

		itype.cs_in = itype.is_singleton ? BINDINGS_PTR_FIELD : "Object." CS_SMETHOD_GETINSTANCE "(%0)";

		itype.c_type = "Object*";
		itype.c_type_in = itype.c_type;
		itype.c_type_out = "MonoObject*";
		itype.cs_type = itype.proxy_name;
		itype.im_type_in = "IntPtr";
		itype.im_type_out = itype.proxy_name;

		List<PropertyInfo> property_list;
		ClassDB::get_property_list(type_cname, &property_list, true);

		// Populate properties

		for (const List<PropertyInfo>::Element *E = property_list.front(); E; E = E->next()) {
			const PropertyInfo &property = E->get();

			if (property.usage & PROPERTY_USAGE_GROUP || property.usage & PROPERTY_USAGE_CATEGORY)
				continue;

			PropertyInterface iprop;
			iprop.cname = property.name;
			iprop.setter = ClassDB::get_property_setter(type_cname, iprop.cname);
			iprop.getter = ClassDB::get_property_getter(type_cname, iprop.cname);

			bool valid = false;
			iprop.index = ClassDB::get_property_index(type_cname, iprop.cname, &valid);
			ERR_FAIL_COND(!valid);

			iprop.proxy_name = escape_csharp_keyword(snake_to_pascal_case(iprop.cname));

			// Prevent property and enclosing type from sharing the same name
			if (iprop.proxy_name == itype.proxy_name) {
				if (verbose_output) {
					WARN_PRINTS("Name of property `" + iprop.proxy_name + "` is ambiguous with the name of its class `" +
								itype.proxy_name + "`. Renaming property to `" + iprop.proxy_name + "_`");
				}

				iprop.proxy_name += "_";
			}

			iprop.proxy_name = iprop.proxy_name.replace("/", "__"); // Some members have a slash...

			iprop.prop_doc = NULL;

			for (int i = 0; i < itype.class_doc->properties.size(); i++) {
				const DocData::PropertyDoc &prop_doc = itype.class_doc->properties[i];

				if (prop_doc.name == iprop.cname) {
					iprop.prop_doc = &prop_doc;
					break;
				}
			}

			itype.properties.push_back(iprop);
		}

		// Populate methods

		List<MethodInfo> virtual_method_list;
		ClassDB::get_virtual_methods(type_cname, &virtual_method_list, true);

		List<MethodInfo> method_list;
		ClassDB::get_method_list(type_cname, &method_list, true);
		method_list.sort();

		for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {
			const MethodInfo &method_info = E->get();

			int argc = method_info.arguments.size();

			if (method_info.name.empty())
				continue;

			MethodInterface imethod;
			imethod.name = method_info.name;
			imethod.cname = imethod.name;

			if (method_info.flags & METHOD_FLAG_VIRTUAL)
				imethod.is_virtual = true;

			PropertyInfo return_info = method_info.return_val;

			MethodBind *m = imethod.is_virtual ? NULL : ClassDB::get_method(type_cname, method_info.name);

			imethod.is_vararg = m && m->is_vararg();

			if (!m && !imethod.is_virtual) {
				if (virtual_method_list.find(method_info)) {
					// A virtual method without the virtual flag. This is a special case.

					// There is no method bind, so let's fallback to Godot's object.Call(string, params)
					imethod.requires_object_call = true;

					// The method Object.free is registered as a virtual method, but without the virtual flag.
					// This is because this method is not supposed to be overridden, but called.
					// We assume the return type is void.
					imethod.return_type.cname = name_cache.type_void;

					// Actually, more methods like this may be added in the future,
					// which could actually will return something different.
					// Let's put this to notify us if that ever happens.
					if (itype.cname != name_cache.type_Object || imethod.name != "free") {
						if (verbose_output) {
							WARN_PRINTS("Notification: New unexpected virtual non-overridable method found.\n"
										"We only expected Object.free, but found " +
										itype.name + "." + imethod.name);
						}
					}
				} else {
					ERR_PRINTS("Missing MethodBind for non-virtual method: " + itype.name + "." + imethod.name);
				}
			} else if (return_info.type == Variant::INT && return_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
				imethod.return_type.cname = return_info.class_name;
				imethod.return_type.is_enum = true;
			} else if (return_info.class_name != StringName()) {
				imethod.return_type.cname = return_info.class_name;
			} else if (return_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
				imethod.return_type.cname = return_info.hint_string;
			} else if (return_info.type == Variant::NIL && return_info.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
				imethod.return_type.cname = name_cache.type_Variant;
			} else if (return_info.type == Variant::NIL) {
				imethod.return_type.cname = name_cache.type_void;
			} else {
				imethod.return_type.cname = Variant::get_type_name(return_info.type);
			}

			for (int i = 0; i < argc; i++) {
				PropertyInfo arginfo = method_info.arguments[i];

				ArgumentInterface iarg;
				iarg.name = arginfo.name;

				if (arginfo.type == Variant::INT && arginfo.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
					iarg.type.cname = arginfo.class_name;
					iarg.type.is_enum = true;
				} else if (arginfo.class_name != StringName()) {
					iarg.type.cname = arginfo.class_name;
				} else if (arginfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
					iarg.type.cname = arginfo.hint_string;
				} else if (arginfo.type == Variant::NIL) {
					iarg.type.cname = name_cache.type_Variant;
				} else {
					iarg.type.cname = Variant::get_type_name(arginfo.type);
				}

				iarg.name = escape_csharp_keyword(snake_to_camel_case(iarg.name));

				if (m && m->has_default_argument(i)) {
					_default_argument_from_variant(m->get_default_argument(i), iarg);
				}

				imethod.add_argument(iarg);
			}

			if (imethod.is_vararg) {
				ArgumentInterface ivararg;
				ivararg.type.cname = name_cache.type_VarArg;
				ivararg.name = "@args";
				imethod.add_argument(ivararg);
			}

			imethod.proxy_name = escape_csharp_keyword(snake_to_pascal_case(imethod.name));

			// Prevent naming the property and its enclosing type from sharing the same name
			if (imethod.proxy_name == itype.proxy_name) {
				if (verbose_output) {
					WARN_PRINTS("Name of method `" + imethod.proxy_name + "` is ambiguous with the name of its class `" +
								itype.proxy_name + "`. Renaming method to `" + imethod.proxy_name + "_`");
				}

				imethod.proxy_name += "_";
			}

			if (itype.class_doc) {
				for (int i = 0; i < itype.class_doc->methods.size(); i++) {
					if (itype.class_doc->methods[i].name == imethod.name) {
						imethod.method_doc = &itype.class_doc->methods[i];
						break;
					}
				}
			}

			if (!imethod.is_virtual && imethod.name[0] == '_') {
				for (const List<PropertyInterface>::Element *F = itype.properties.front(); F; F = F->next()) {
					const PropertyInterface &iprop = F->get();

					if (iprop.setter == imethod.name || iprop.getter == imethod.name) {
						imethod.is_internal = true;
						itype.methods.push_back(imethod);
						break;
					}
				}
			} else {
				itype.methods.push_back(imethod);
			}
		}

		// Populate enums and constants

		List<String> constant_list;
		ClassDB::get_integer_constant_list(type_cname, &constant_list, true);

		const HashMap<StringName, List<StringName> > &enum_map = class_info->enum_map;
		const StringName *k = NULL;

		while ((k = enum_map.next(k))) {
			StringName enum_proxy_cname = *k;
			String enum_proxy_name = enum_proxy_cname.operator String();
			if (itype.find_property_by_proxy_name(enum_proxy_cname)) {
				// We have several conflicts between enums and PascalCase properties,
				// so we append 'Enum' to the enum name in those cases.
				enum_proxy_name += "Enum";
				enum_proxy_cname = StringName(enum_proxy_name);
			}
			EnumInterface ienum(enum_proxy_cname);
			const List<StringName> &constants = enum_map.get(*k);
			for (const List<StringName>::Element *E = constants.front(); E; E = E->next()) {
				const StringName &constant_cname = E->get();
				String constant_name = constant_cname.operator String();
				int *value = class_info->constant_map.getptr(constant_cname);
				ERR_FAIL_NULL(value);
				constant_list.erase(constant_name);

				ConstantInterface iconstant(constant_name, snake_to_pascal_case(constant_name, true), *value);

				iconstant.const_doc = NULL;
				for (int i = 0; i < itype.class_doc->constants.size(); i++) {
					const DocData::ConstantDoc &const_doc = itype.class_doc->constants[i];

					if (const_doc.name == iconstant.name) {
						iconstant.const_doc = &const_doc;
						break;
					}
				}

				ienum.constants.push_back(iconstant);
			}

			int prefix_length = _determine_enum_prefix(ienum);

			_apply_prefix_to_enum_constants(ienum, prefix_length);

			itype.enums.push_back(ienum);

			TypeInterface enum_itype;
			enum_itype.is_enum = true;
			enum_itype.name = itype.name + "." + String(*k);
			enum_itype.cname = StringName(enum_itype.name);
			enum_itype.proxy_name = itype.proxy_name + "." + enum_proxy_name;
			TypeInterface::postsetup_enum_type(enum_itype);
			enum_types.insert(enum_itype.cname, enum_itype);
		}

		for (const List<String>::Element *E = constant_list.front(); E; E = E->next()) {
			const String &constant_name = E->get();
			int *value = class_info->constant_map.getptr(StringName(E->get()));
			ERR_FAIL_NULL(value);

			ConstantInterface iconstant(constant_name, snake_to_pascal_case(constant_name, true), *value);

			iconstant.const_doc = NULL;
			for (int i = 0; i < itype.class_doc->constants.size(); i++) {
				const DocData::ConstantDoc &const_doc = itype.class_doc->constants[i];

				if (const_doc.name == iconstant.name) {
					iconstant.const_doc = &const_doc;
					break;
				}
			}

			itype.constants.push_back(iconstant);
		}

		obj_types.insert(itype.cname, itype);

		class_list.pop_front();
	}
}

void BindingsGenerator::_default_argument_from_variant(const Variant &p_val, ArgumentInterface &r_iarg) {

	r_iarg.default_argument = p_val;

	switch (p_val.get_type()) {
		case Variant::NIL:
			if (ClassDB::class_exists(r_iarg.type.cname)) {
				// Object type
				r_iarg.default_argument = "null";
			} else {
				// Variant
				r_iarg.default_argument = "null";
			}
			break;
		// Atomic types
		case Variant::BOOL:
			r_iarg.default_argument = bool(p_val) ? "true" : "false";
			break;
		case Variant::INT:
			if (r_iarg.type.cname != name_cache.type_int) {
				r_iarg.default_argument = "(%s)" + r_iarg.default_argument;
			}
			break;
		case Variant::REAL:
#ifndef REAL_T_IS_DOUBLE
			r_iarg.default_argument += "f";
#endif
			break;
		case Variant::STRING:
		case Variant::NODE_PATH:
			r_iarg.default_argument = "\"" + r_iarg.default_argument + "\"";
			break;
		case Variant::TRANSFORM:
			if (p_val.operator Transform() == Transform())
				r_iarg.default_argument.clear();
			r_iarg.default_argument = "new %s(" + r_iarg.default_argument + ")";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		case Variant::PLANE:
		case Variant::AABB:
		case Variant::COLOR:
			r_iarg.default_argument = "new Color(1, 1, 1, 1)";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
			r_iarg.default_argument = "new %s" + r_iarg.default_argument;
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		case Variant::OBJECT:
			if (p_val.is_zero()) {
				r_iarg.default_argument = "null";
				break;
			}
			FALLTHROUGH;
		case Variant::DICTIONARY:
		case Variant::_RID:
			r_iarg.default_argument = "new %s()";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_REF;
			break;
		case Variant::ARRAY:
		case Variant::POOL_BYTE_ARRAY:
		case Variant::POOL_INT_ARRAY:
		case Variant::POOL_REAL_ARRAY:
		case Variant::POOL_STRING_ARRAY:
		case Variant::POOL_VECTOR2_ARRAY:
		case Variant::POOL_VECTOR3_ARRAY:
		case Variant::POOL_COLOR_ARRAY:
			r_iarg.default_argument = "new %s {}";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_REF;
			break;
		case Variant::TRANSFORM2D:
		case Variant::BASIS:
		case Variant::QUAT:
			r_iarg.default_argument = Variant::get_type_name(p_val.get_type()) + ".Identity";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		default: {
		}
	}

	if (r_iarg.def_param_mode == ArgumentInterface::CONSTANT && r_iarg.type.cname == name_cache.type_Variant && r_iarg.default_argument != "null")
		r_iarg.def_param_mode = ArgumentInterface::NULLABLE_REF;
}

void BindingsGenerator::_populate_builtin_type_interfaces() {

	builtin_types.clear();

	TypeInterface itype;

#define INSERT_STRUCT_TYPE(m_type, m_type_in)                          \
	{                                                                  \
		itype = TypeInterface::create_value_type(String(#m_type));     \
		itype.c_in = "\t%0 %1_in = MARSHALLED_IN(" #m_type ", %1);\n"; \
		itype.c_out = "\treturn MARSHALLED_OUT(" #m_type ", %1);\n";   \
		itype.c_arg_in = "&%s_in";                                     \
		itype.c_type_in = "GDMonoMarshal::M_" #m_type "*";             \
		itype.c_type_out = "GDMonoMarshal::M_" #m_type;                \
		itype.cs_in = "ref %s";                                        \
		itype.cs_out = "return (%1)%0;";                               \
		itype.im_type_out = itype.cs_type;                             \
		builtin_types.insert(itype.cname, itype);                      \
	}

	INSERT_STRUCT_TYPE(Vector2, "real_t*")
	INSERT_STRUCT_TYPE(Rect2, "real_t*")
	INSERT_STRUCT_TYPE(Transform2D, "real_t*")
	INSERT_STRUCT_TYPE(Vector3, "real_t*")
	INSERT_STRUCT_TYPE(Basis, "real_t*")
	INSERT_STRUCT_TYPE(Quat, "real_t*")
	INSERT_STRUCT_TYPE(Transform, "real_t*")
	INSERT_STRUCT_TYPE(AABB, "real_t*")
	INSERT_STRUCT_TYPE(Color, "real_t*")
	INSERT_STRUCT_TYPE(Plane, "real_t*")

#undef INSERT_STRUCT_TYPE

	// bool
	itype = TypeInterface::create_value_type(String("bool"));

	{
		// MonoBoolean <---> bool
		itype.c_in = "\t%0 %1_in = (%0)%1;\n";
		itype.c_out = "\treturn (%0)%1;\n";
		itype.c_type = "bool";
		itype.c_type_in = "MonoBoolean";
		itype.c_type_out = itype.c_type_in;
		itype.c_arg_in = "&%s_in";
	}
	itype.im_type_in = itype.name;
	itype.im_type_out = itype.name;
	builtin_types.insert(itype.cname, itype);

	// int
	// C interface is the same as that of enums. Remember to apply any
	// changes done here to TypeInterface::postsetup_enum_type as well
	itype = TypeInterface::create_value_type(String("int"));
	itype.c_arg_in = "&%s_in";
	{
		// The expected types for parameters and return value in ptrcall are 'int64_t' or 'uint64_t'.
		itype.c_in = "\t%0 %1_in = (%0)%1;\n";
		itype.c_out = "\treturn (%0)%1;\n";
		itype.c_type = "int64_t";
	}
	itype.c_type_in = "int32_t";
	itype.c_type_out = itype.c_type_in;
	itype.im_type_in = itype.name;
	itype.im_type_out = itype.name;
	builtin_types.insert(itype.cname, itype);

	// real_t
	itype = TypeInterface();
	itype.name = "float"; // The name is always "float" in Variant, even with REAL_T_IS_DOUBLE.
	itype.cname = itype.name;
#ifdef REAL_T_IS_DOUBLE
	itype.proxy_name = "double";
#else
	itype.proxy_name = "float";
#endif
	{
		// The expected type for parameters and return value in ptrcall is 'double'.
		itype.c_in = "\t%0 %1_in = (%0)%1;\n";
		itype.c_out = "\treturn (%0)%1;\n";
		itype.c_type = "double";
		itype.c_type_in = "real_t";
		itype.c_type_out = "real_t";
		itype.c_arg_in = "&%s_in";
	}
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.cname, itype);

	// String
	itype = TypeInterface();
	itype.name = "String";
	itype.cname = itype.name;
	itype.proxy_name = "string";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MONOSTR_TO_GODOT "(%1);\n";
	itype.c_out = "\treturn " C_METHOD_MONOSTR_FROM_GODOT "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = itype.name;
	itype.c_type_in = "MonoString*";
	itype.c_type_out = "MonoString*";
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.cname, itype);

	// NodePath
	itype = TypeInterface();
	itype.name = "NodePath";
	itype.cname = itype.name;
	itype.proxy_name = "NodePath";
	itype.c_out = "\treturn memnew(NodePath(%1));\n";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type + "*";
	itype.c_type_out = itype.c_type + "*";
	itype.cs_type = itype.proxy_name;
	itype.cs_in = "NodePath." CS_SMETHOD_GETINSTANCE "(%0)";
	itype.cs_out = "return new %1(%0);";
	itype.im_type_in = "IntPtr";
	itype.im_type_out = "IntPtr";
	builtin_types.insert(itype.cname, itype);

	// RID
	itype = TypeInterface();
	itype.name = "RID";
	itype.cname = itype.name;
	itype.proxy_name = "RID";
	itype.c_out = "\treturn memnew(RID(%1));\n";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type + "*";
	itype.c_type_out = itype.c_type + "*";
	itype.cs_type = itype.proxy_name;
	itype.cs_in = "RID." CS_SMETHOD_GETINSTANCE "(%0)";
	itype.cs_out = "return new %1(%0);";
	itype.im_type_in = "IntPtr";
	itype.im_type_out = "IntPtr";
	builtin_types.insert(itype.cname, itype);

	// Variant
	itype = TypeInterface();
	itype.name = "Variant";
	itype.cname = itype.name;
	itype.proxy_name = "object";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MANAGED_TO_VARIANT "(%1);\n";
	itype.c_out = "\treturn " C_METHOD_MANAGED_FROM_VARIANT "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = itype.name;
	itype.c_type_in = "MonoObject*";
	itype.c_type_out = "MonoObject*";
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = "object";
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.cname, itype);

	// VarArg (fictitious type to represent variable arguments)
	itype = TypeInterface();
	itype.name = "VarArg";
	itype.cname = itype.name;
	itype.proxy_name = "object[]";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MONOARRAY_TO(Array) "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = "Array";
	itype.c_type_in = "MonoArray*";
	itype.cs_type = "params object[]";
	itype.im_type_in = "object[]";
	builtin_types.insert(itype.cname, itype);

#define INSERT_ARRAY_FULL(m_name, m_type, m_proxy_t)                          \
	{                                                                         \
		itype = TypeInterface();                                              \
		itype.name = #m_name;                                                 \
		itype.cname = itype.name;                                             \
		itype.proxy_name = #m_proxy_t "[]";                                   \
		itype.c_in = "\t%0 %1_in = " C_METHOD_MONOARRAY_TO(m_type) "(%1);\n"; \
		itype.c_out = "\treturn " C_METHOD_MONOARRAY_FROM(m_type) "(%1);\n";  \
		itype.c_arg_in = "&%s_in";                                            \
		itype.c_type = #m_type;                                               \
		itype.c_type_in = "MonoArray*";                                       \
		itype.c_type_out = "MonoArray*";                                      \
		itype.cs_type = itype.proxy_name;                                     \
		itype.im_type_in = itype.proxy_name;                                  \
		itype.im_type_out = itype.proxy_name;                                 \
		builtin_types.insert(itype.name, itype);                              \
	}

#define INSERT_ARRAY(m_type, m_proxy_t) INSERT_ARRAY_FULL(m_type, m_type, m_proxy_t)

	INSERT_ARRAY(PoolIntArray, int);
	INSERT_ARRAY_FULL(PoolByteArray, PoolByteArray, byte);

#ifdef REAL_T_IS_DOUBLE
	INSERT_ARRAY(PoolRealArray, double);
#else
	INSERT_ARRAY(PoolRealArray, float);
#endif

	INSERT_ARRAY(PoolStringArray, string);

	INSERT_ARRAY(PoolColorArray, Color);
	INSERT_ARRAY(PoolVector2Array, Vector2);
	INSERT_ARRAY(PoolVector3Array, Vector3);

#undef INSERT_ARRAY

	// Array
	itype = TypeInterface();
	itype.name = "Array";
	itype.cname = itype.name;
	itype.proxy_name = itype.name;
	itype.c_out = "\treturn memnew(Array(%1));\n";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type + "*";
	itype.c_type_out = itype.c_type + "*";
	itype.cs_type = BINDINGS_NAMESPACE_COLLECTIONS "." + itype.proxy_name;
	itype.cs_in = "%0." CS_SMETHOD_GETINSTANCE "()";
	itype.cs_out = "return new " + itype.cs_type + "(%0);";
	itype.im_type_in = "IntPtr";
	itype.im_type_out = "IntPtr";
	builtin_types.insert(itype.cname, itype);

	// Dictionary
	itype = TypeInterface();
	itype.name = "Dictionary";
	itype.cname = itype.name;
	itype.proxy_name = itype.name;
	itype.c_out = "\treturn memnew(Dictionary(%1));\n";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type + "*";
	itype.c_type_out = itype.c_type + "*";
	itype.cs_type = BINDINGS_NAMESPACE_COLLECTIONS "." + itype.proxy_name;
	itype.cs_in = "%0." CS_SMETHOD_GETINSTANCE "()";
	itype.cs_out = "return new " + itype.cs_type + "(%0);";
	itype.im_type_in = "IntPtr";
	itype.im_type_out = "IntPtr";
	builtin_types.insert(itype.cname, itype);

	// void (fictitious type to represent the return type of methods that do not return anything)
	itype = TypeInterface();
	itype.name = "void";
	itype.cname = itype.name;
	itype.proxy_name = itype.name;
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type;
	itype.c_type_out = itype.c_type;
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.cname, itype);
}

void BindingsGenerator::_populate_global_constants() {

	int global_constants_count = GlobalConstants::get_global_constant_count();

	if (global_constants_count > 0) {
		Map<String, DocData::ClassDoc>::Element *match = EditorHelp::get_doc_data()->class_list.find("@GlobalScope");

		ERR_EXPLAIN("Could not find `@GlobalScope` in DocData");
		CRASH_COND(!match);

		const DocData::ClassDoc &global_scope_doc = match->value();

		for (int i = 0; i < global_constants_count; i++) {

			String constant_name = GlobalConstants::get_global_constant_name(i);

			const DocData::ConstantDoc *const_doc = NULL;
			for (int j = 0; j < global_scope_doc.constants.size(); j++) {
				const DocData::ConstantDoc &curr_const_doc = global_scope_doc.constants[j];

				if (curr_const_doc.name == constant_name) {
					const_doc = &curr_const_doc;
					break;
				}
			}

			int constant_value = GlobalConstants::get_global_constant_value(i);
			StringName enum_name = GlobalConstants::get_global_constant_enum(i);

			ConstantInterface iconstant(constant_name, snake_to_pascal_case(constant_name, true), constant_value);
			iconstant.const_doc = const_doc;

			if (enum_name != StringName()) {
				EnumInterface ienum(enum_name);
				List<EnumInterface>::Element *enum_match = global_enums.find(ienum);
				if (enum_match) {
					enum_match->get().constants.push_back(iconstant);
				} else {
					ienum.constants.push_back(iconstant);
					global_enums.push_back(ienum);
				}
			} else {
				global_constants.push_back(iconstant);
			}
		}

		for (List<EnumInterface>::Element *E = global_enums.front(); E; E = E->next()) {
			EnumInterface &ienum = E->get();

			TypeInterface enum_itype;
			enum_itype.is_enum = true;
			enum_itype.name = ienum.cname.operator String();
			enum_itype.cname = ienum.cname;
			enum_itype.proxy_name = enum_itype.name;
			TypeInterface::postsetup_enum_type(enum_itype);
			enum_types.insert(enum_itype.cname, enum_itype);

			int prefix_length = _determine_enum_prefix(ienum);

			// HARDCODED: The Error enum have the prefix 'ERR_' for everything except 'OK' and 'FAILED'.
			if (ienum.cname == name_cache.enum_Error) {
				if (prefix_length > 0) { // Just in case it ever changes
					ERR_PRINTS("Prefix for enum 'Error' is not empty");
				}

				prefix_length = 1; // 'ERR_'
			}

			_apply_prefix_to_enum_constants(ienum, prefix_length);
		}
	}

	// HARDCODED
	List<StringName> hardcoded_enums;
	hardcoded_enums.push_back("Vector3.Axis");
	for (List<StringName>::Element *E = hardcoded_enums.front(); E; E = E->next()) {
		// These enums are not generated and must be written manually (e.g.: Vector3.Axis)
		// Here, we assume core types do not begin with underscore
		TypeInterface enum_itype;
		enum_itype.is_enum = true;
		enum_itype.name = E->get().operator String();
		enum_itype.cname = E->get();
		enum_itype.proxy_name = enum_itype.name;
		TypeInterface::postsetup_enum_type(enum_itype);
		enum_types.insert(enum_itype.cname, enum_itype);
	}
}

void BindingsGenerator::initialize() {

	EditorHelp::generate_doc();

	enum_types.clear();

	_populate_object_type_interfaces();
	_populate_builtin_type_interfaces();

	_populate_global_constants();

	// Generate internal calls (after populating type interfaces and global constants)

	core_custom_icalls.clear();
	editor_custom_icalls.clear();

	for (OrderedHashMap<StringName, TypeInterface>::Element E = obj_types.front(); E; E = E.next())
		_generate_method_icalls(E.get());
}

void BindingsGenerator::handle_cmdline_args(const List<String> &p_cmdline_args) {

	const int NUM_OPTIONS = 2;
	int options_left = NUM_OPTIONS;

	String mono_glue_option = "--generate-mono-glue";
	String cs_api_option = "--generate-cs-api";

	verbose_output = true;

	const List<String>::Element *elem = p_cmdline_args.front();

	while (elem && options_left) {

		if (elem->get() == mono_glue_option) {

			const List<String>::Element *path_elem = elem->next();

			if (path_elem) {
				if (get_singleton()->generate_glue(path_elem->get()) != OK)
					ERR_PRINTS(mono_glue_option + ": Failed to generate mono glue");
				elem = elem->next();
			} else {
				ERR_PRINTS(mono_glue_option + ": No output directory specified");
			}

			--options_left;

		} else if (elem->get() == cs_api_option) {

			const List<String>::Element *path_elem = elem->next();

			if (path_elem) {
				if (get_singleton()->generate_cs_api(path_elem->get()) != OK)
					ERR_PRINTS(cs_api_option + ": Failed to generate the C# API");
				elem = elem->next();
			} else {
				ERR_PRINTS(cs_api_option + ": No output directory specified");
			}

			--options_left;
		}

		elem = elem->next();
	}

	verbose_output = false;

	if (options_left != NUM_OPTIONS)
		::exit(0);
}

#endif
