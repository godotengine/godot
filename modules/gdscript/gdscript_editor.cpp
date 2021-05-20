/*************************************************************************/
/*  gdscript_editor.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdscript.h"

#include "core/config/engine.h"
#include "core/core_constants.h"
#include "core/os/file_access.h"
#include "gdscript_analyzer.h"
#include "gdscript_compiler.h"
#include "gdscript_parser.h"
#include "gdscript_tokenizer.h"
#include "gdscript_utility_functions.h"

#ifdef TOOLS_ENABLED
#include "core/config/project_settings.h"
#include "editor/editor_file_system.h"
#include "editor/editor_settings.h"
#endif

void GDScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {
	p_delimiters->push_back("#");
}

void GDScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {
	p_delimiters->push_back("\" \"");
	p_delimiters->push_back("' '");
	p_delimiters->push_back("\"\"\" \"\"\"");
}

String GDScriptLanguage::_get_processed_template(const String &p_template, const String &p_base_class_name) const {
	String processed_template = p_template;

#ifdef TOOLS_ENABLED
	if (EDITOR_DEF("text_editor/completion/add_type_hints", false)) {
		processed_template = processed_template.replace("%INT_TYPE%", ": int");
		processed_template = processed_template.replace("%STRING_TYPE%", ": String");
		processed_template = processed_template.replace("%FLOAT_TYPE%", ": float");
		processed_template = processed_template.replace("%VOID_RETURN%", " -> void");
	} else {
		processed_template = processed_template.replace("%INT_TYPE%", "");
		processed_template = processed_template.replace("%STRING_TYPE%", "");
		processed_template = processed_template.replace("%FLOAT_TYPE%", "");
		processed_template = processed_template.replace("%VOID_RETURN%", "");
	}
#else
	processed_template = processed_template.replace("%INT_TYPE%", "");
	processed_template = processed_template.replace("%STRING_TYPE%", "");
	processed_template = processed_template.replace("%FLOAT_TYPE%", "");
	processed_template = processed_template.replace("%VOID_RETURN%", "");
#endif

	processed_template = processed_template.replace("%BASE%", p_base_class_name);
	processed_template = processed_template.replace("%TS%", _get_indentation());

	return processed_template;
}

Ref<Script> GDScriptLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {
	String _template = "extends %BASE%\n"
					   "\n"
					   "\n"
					   "# Declare member variables here. Examples:\n"
					   "# var a%INT_TYPE% = 2\n"
					   "# var b%STRING_TYPE% = \"text\"\n"
					   "\n"
					   "\n"
					   "# Called when the node enters the scene tree for the first time.\n"
					   "func _ready()%VOID_RETURN%:\n"
					   "%TS%pass # Replace with function body.\n"
					   "\n"
					   "\n"
					   "# Called every frame. 'delta' is the elapsed time since the previous frame.\n"
					   "#func _process(delta%FLOAT_TYPE%)%VOID_RETURN%:\n"
					   "#%TS%pass\n";

	_template = _get_processed_template(_template, p_base_class_name);

	Ref<GDScript> script;
	script.instance();
	script->set_source_code(_template);

	return script;
}

bool GDScriptLanguage::is_using_templates() {
	return true;
}

void GDScriptLanguage::make_template(const String &p_class_name, const String &p_base_class_name, Ref<Script> &p_script) {
	String _template = _get_processed_template(p_script->get_source_code(), p_base_class_name);
	p_script->set_source_code(_template);
}

static void get_function_names_recursively(const GDScriptParser::ClassNode *p_class, const String &p_prefix, Map<int, String> &r_funcs) {
	for (int i = 0; i < p_class->members.size(); i++) {
		if (p_class->members[i].type == GDScriptParser::ClassNode::Member::FUNCTION) {
			const GDScriptParser::FunctionNode *function = p_class->members[i].function;
			r_funcs[function->start_line] = p_prefix.is_empty() ? String(function->identifier->name) : p_prefix + "." + String(function->identifier->name);
		} else if (p_class->members[i].type == GDScriptParser::ClassNode::Member::CLASS) {
			String new_prefix = p_class->members[i].m_class->identifier->name;
			get_function_names_recursively(p_class->members[i].m_class, p_prefix.is_empty() ? new_prefix : p_prefix + "." + new_prefix, r_funcs);
		}
	}
}

bool GDScriptLanguage::validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions, List<ScriptLanguage::Warning> *r_warnings, Set<int> *r_safe_lines) const {
	GDScriptParser parser;
	GDScriptAnalyzer analyzer(&parser);

	Error err = parser.parse(p_script, p_path, false);
	if (err == OK) {
		err = analyzer.analyze();
	}
#ifdef DEBUG_ENABLED
	if (r_warnings) {
		for (const List<GDScriptWarning>::Element *E = parser.get_warnings().front(); E; E = E->next()) {
			const GDScriptWarning &warn = E->get();
			ScriptLanguage::Warning w;
			w.start_line = warn.start_line;
			w.end_line = warn.end_line;
			w.leftmost_column = warn.leftmost_column;
			w.rightmost_column = warn.rightmost_column;
			w.code = (int)warn.code;
			w.string_code = GDScriptWarning::get_name_from_code(warn.code);
			w.message = warn.get_message();
			r_warnings->push_back(w);
		}
	}
#endif
	if (err) {
		GDScriptParser::ParserError parse_error = parser.get_errors().front()->get();
		r_line_error = parse_error.line;
		r_col_error = parse_error.column;
		r_test_error = parse_error.message;
		return false;
	} else {
		const GDScriptParser::ClassNode *cl = parser.get_tree();
		Map<int, String> funcs;

		get_function_names_recursively(cl, "", funcs);

		for (Map<int, String>::Element *E = funcs.front(); E; E = E->next()) {
			r_functions->push_back(E->get() + ":" + itos(E->key()));
		}
	}

#ifdef DEBUG_ENABLED
	if (r_safe_lines) {
		const Set<int> &unsafe_lines = parser.get_unsafe_lines();
		for (int i = 1; i <= parser.get_last_line_number(); i++) {
			if (!unsafe_lines.has(i)) {
				r_safe_lines->insert(i);
			}
		}
	}
#endif

	return true;
}

bool GDScriptLanguage::has_named_classes() const {
	return false;
}

bool GDScriptLanguage::supports_builtin_mode() const {
	return true;
}

bool GDScriptLanguage::supports_documentation() const {
	return true;
}

int GDScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	GDScriptTokenizer tokenizer;
	tokenizer.set_source_code(p_code);
	int indent = 0;
	GDScriptTokenizer::Token current = tokenizer.scan();
	while (current.type != GDScriptTokenizer::Token::TK_EOF && current.type != GDScriptTokenizer::Token::ERROR) {
		if (current.type == GDScriptTokenizer::Token::INDENT) {
			indent++;
		} else if (current.type == GDScriptTokenizer::Token::DEDENT) {
			indent--;
		}
		if (indent == 0 && current.type == GDScriptTokenizer::Token::FUNC) {
			current = tokenizer.scan();
			if (current.is_identifier()) {
				String identifier = current.get_identifier();
				if (identifier == p_function) {
					return current.start_line;
				}
			}
		}
		current = tokenizer.scan();
	}
	return -1;
}

Script *GDScriptLanguage::create_script() const {
	return memnew(GDScript);
}

/* DEBUGGER FUNCTIONS */

bool GDScriptLanguage::debug_break_parse(const String &p_file, int p_line, const String &p_error) {
	//break because of parse error

	if (EngineDebugger::is_active() && Thread::get_caller_id() == Thread::get_main_id()) {
		_debug_parse_err_line = p_line;
		_debug_parse_err_file = p_file;
		_debug_error = p_error;
		EngineDebugger::get_script_debugger()->debug(this, false, true);
		return true;
	} else {
		return false;
	}
}

bool GDScriptLanguage::debug_break(const String &p_error, bool p_allow_continue) {
	if (EngineDebugger::is_active() && Thread::get_caller_id() == Thread::get_main_id()) {
		_debug_parse_err_line = -1;
		_debug_parse_err_file = "";
		_debug_error = p_error;
		bool is_error_breakpoint = p_error != "Breakpoint";
		EngineDebugger::get_script_debugger()->debug(this, p_allow_continue, is_error_breakpoint);
		return true;
	} else {
		return false;
	}
}

String GDScriptLanguage::debug_get_error() const {
	return _debug_error;
}

int GDScriptLanguage::debug_get_stack_level_count() const {
	if (_debug_parse_err_line >= 0) {
		return 1;
	}

	return _debug_call_stack_pos;
}

int GDScriptLanguage::debug_get_stack_level_line(int p_level) const {
	if (_debug_parse_err_line >= 0) {
		return _debug_parse_err_line;
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, -1);

	int l = _debug_call_stack_pos - p_level - 1;

	return *(_call_stack[l].line);
}

String GDScriptLanguage::debug_get_stack_level_function(int p_level) const {
	if (_debug_parse_err_line >= 0) {
		return "";
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, "");
	int l = _debug_call_stack_pos - p_level - 1;
	return _call_stack[l].function->get_name();
}

String GDScriptLanguage::debug_get_stack_level_source(int p_level) const {
	if (_debug_parse_err_line >= 0) {
		return _debug_parse_err_file;
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, "");
	int l = _debug_call_stack_pos - p_level - 1;
	return _call_stack[l].function->get_source();
}

void GDScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_debug_parse_err_line >= 0) {
		return;
	}

	ERR_FAIL_INDEX(p_level, _debug_call_stack_pos);
	int l = _debug_call_stack_pos - p_level - 1;

	GDScriptFunction *f = _call_stack[l].function;

	List<Pair<StringName, int>> locals;

	f->debug_get_stack_member_state(*_call_stack[l].line, &locals);
	for (List<Pair<StringName, int>>::Element *E = locals.front(); E; E = E->next()) {
		p_locals->push_back(E->get().first);
		p_values->push_back(_call_stack[l].stack[E->get().second]);
	}
}

void GDScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_debug_parse_err_line >= 0) {
		return;
	}

	ERR_FAIL_INDEX(p_level, _debug_call_stack_pos);
	int l = _debug_call_stack_pos - p_level - 1;

	GDScriptInstance *instance = _call_stack[l].instance;

	if (!instance) {
		return;
	}

	Ref<GDScript> script = instance->get_script();
	ERR_FAIL_COND(script.is_null());

	const Map<StringName, GDScript::MemberInfo> &mi = script->debug_get_member_indices();

	for (const Map<StringName, GDScript::MemberInfo>::Element *E = mi.front(); E; E = E->next()) {
		p_members->push_back(E->key());
		p_values->push_back(instance->debug_get_member_by_index(E->get().index));
	}
}

ScriptInstance *GDScriptLanguage::debug_get_stack_level_instance(int p_level) {
	if (_debug_parse_err_line >= 0) {
		return nullptr;
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, nullptr);

	int l = _debug_call_stack_pos - p_level - 1;
	ScriptInstance *instance = _call_stack[l].instance;

	return instance;
}

void GDScriptLanguage::debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	const Map<StringName, int> &name_idx = GDScriptLanguage::get_singleton()->get_global_map();
	const Variant *globals = GDScriptLanguage::get_singleton()->get_global_array();

	List<Pair<String, Variant>> cinfo;
	get_public_constants(&cinfo);

	for (const Map<StringName, int>::Element *E = name_idx.front(); E; E = E->next()) {
		if (ClassDB::class_exists(E->key()) || Engine::get_singleton()->has_singleton(E->key())) {
			continue;
		}

		bool is_script_constant = false;
		for (List<Pair<String, Variant>>::Element *CE = cinfo.front(); CE; CE = CE->next()) {
			if (CE->get().first == E->key()) {
				is_script_constant = true;
				break;
			}
		}
		if (is_script_constant) {
			continue;
		}

		const Variant &var = globals[E->value()];
		if (Object *obj = var) {
			if (Object::cast_to<GDScriptNativeClass>(obj)) {
				continue;
			}
		}

		bool skip = false;
		for (int i = 0; i < CoreConstants::get_global_constant_count(); i++) {
			if (E->key() == CoreConstants::get_global_constant_name(i)) {
				skip = true;
				break;
			}
		}
		if (skip) {
			continue;
		}

		p_globals->push_back(E->key());
		p_values->push_back(var);
	}
}

String GDScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return "";
}

void GDScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gd");
}

void GDScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
	List<StringName> functions;
	GDScriptUtilityFunctions::get_function_list(&functions);

	for (const List<StringName>::Element *E = functions.front(); E; E = E->next()) {
		p_functions->push_back(GDScriptUtilityFunctions::get_function_info(E->get()));
	}

	// Not really "functions", but show in documentation.
	{
		MethodInfo mi;
		mi.name = "preload";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "path"));
		mi.return_val = PropertyInfo(Variant::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, "Resource");
		p_functions->push_back(mi);
	}
	{
		MethodInfo mi;
		mi.name = "assert";
		mi.return_val.type = Variant::NIL;
		mi.arguments.push_back(PropertyInfo(Variant::BOOL, "condition"));
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "message"));
		mi.default_arguments.push_back(String());
		p_functions->push_back(mi);
	}
}

void GDScriptLanguage::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
	Pair<String, Variant> pi;
	pi.first = "PI";
	pi.second = Math_PI;
	p_constants->push_back(pi);

	Pair<String, Variant> tau;
	tau.first = "TAU";
	tau.second = Math_TAU;
	p_constants->push_back(tau);

	Pair<String, Variant> infinity;
	infinity.first = "INF";
	infinity.second = Math_INF;
	p_constants->push_back(infinity);

	Pair<String, Variant> nan;
	nan.first = "NAN";
	nan.second = Math_NAN;
	p_constants->push_back(nan);
}

String GDScriptLanguage::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
#ifdef TOOLS_ENABLED
	bool th = EditorSettings::get_singleton()->get_setting("text_editor/completion/add_type_hints");
#else
	bool th = false;
#endif

	String s = "func " + p_name + "(";
	if (p_args.size()) {
		for (int i = 0; i < p_args.size(); i++) {
			if (i > 0) {
				s += ", ";
			}
			s += p_args[i].get_slice(":", 0);
			if (th) {
				String type = p_args[i].get_slice(":", 1);
				if (!type.is_empty() && type != "var") {
					s += ": " + type;
				}
			}
		}
	}
	s += String(")") + (th ? " -> void" : "") + ":\n" + _get_indentation() + "pass # Replace with function body.\n";

	return s;
}

//////// COMPLETION //////////

#ifdef TOOLS_ENABLED

#define COMPLETION_RECURSION_LIMIT 200

struct GDScriptCompletionIdentifier {
	GDScriptParser::DataType type;
	String enumeration;
	Variant value;
	const GDScriptParser::ExpressionNode *assigned_expression = nullptr;
};

static String _get_visual_datatype(const PropertyInfo &p_info, bool p_is_arg = true) {
	if (p_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
		String enum_name = p_info.class_name;
		if (enum_name.find(".") == -1) {
			return enum_name;
		}
		return enum_name.get_slice(".", 1);
	}

	String n = p_info.name;
	int idx = n.find(":");
	if (idx != -1) {
		return n.substr(idx + 1, n.length());
	}

	if (p_info.type == Variant::OBJECT) {
		if (p_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
			return p_info.hint_string;
		} else {
			return p_info.class_name.operator String();
		}
	}
	if (p_info.type == Variant::NIL) {
		if (p_is_arg || (p_info.usage & PROPERTY_USAGE_NIL_IS_VARIANT)) {
			return "Variant";
		} else {
			return "void";
		}
	}

	return Variant::get_type_name(p_info.type);
}

static String _make_arguments_hint(const MethodInfo &p_info, int p_arg_idx, bool p_is_annotation = false) {
	String arghint;
	if (!p_is_annotation) {
		arghint += _get_visual_datatype(p_info.return_val, false) + " ";
	}
	arghint += p_info.name + "(";

	int def_args = p_info.arguments.size() - p_info.default_arguments.size();
	int i = 0;
	for (const List<PropertyInfo>::Element *E = p_info.arguments.front(); E; E = E->next()) {
		if (i > 0) {
			arghint += ", ";
		}

		if (i == p_arg_idx) {
			arghint += String::chr(0xFFFF);
		}
		arghint += E->get().name + ": " + _get_visual_datatype(E->get(), true);

		if (i - def_args >= 0) {
			arghint += String(" = ") + p_info.default_arguments[i - def_args].get_construct_string();
		}

		if (i == p_arg_idx) {
			arghint += String::chr(0xFFFF);
		}

		i++;
	}

	if (p_info.flags & METHOD_FLAG_VARARG) {
		if (p_info.arguments.size() > 0) {
			arghint += ", ";
		}
		if (p_arg_idx >= p_info.arguments.size()) {
			arghint += String::chr(0xFFFF);
		}
		arghint += "...";
		if (p_arg_idx >= p_info.arguments.size()) {
			arghint += String::chr(0xFFFF);
		}
	}

	arghint += ")";

	return arghint;
}

static String _make_arguments_hint(const GDScriptParser::FunctionNode *p_function, int p_arg_idx) {
	String arghint = p_function->get_datatype().to_string() + " " + p_function->identifier->name.operator String() + "(";

	for (int i = 0; i < p_function->parameters.size(); i++) {
		if (i > 0) {
			arghint += ", ";
		}

		if (i == p_arg_idx) {
			arghint += String::chr(0xFFFF);
		}
		const GDScriptParser::ParameterNode *par = p_function->parameters[i];
		arghint += par->identifier->name.operator String() + ": " + par->get_datatype().to_string();

		if (par->default_value) {
			String def_val = "<unknown>";
			if (par->default_value->type == GDScriptParser::Node::LITERAL) {
				const GDScriptParser::LiteralNode *literal = static_cast<const GDScriptParser::LiteralNode *>(par->default_value);
				def_val = literal->value.get_construct_string();
			} else if (par->default_value->type == GDScriptParser::Node::IDENTIFIER) {
				const GDScriptParser::IdentifierNode *id = static_cast<const GDScriptParser::IdentifierNode *>(par->default_value);
				def_val = id->name.operator String();
			}
			arghint += " = " + def_val;
		}
		if (i == p_arg_idx) {
			arghint += String::chr(0xFFFF);
		}
	}

	arghint += ")";

	return arghint;
}

static void _get_directory_contents(EditorFileSystemDirectory *p_dir, Map<String, ScriptCodeCompletionOption> &r_list) {
	const String quote_style = EDITOR_DEF("text_editor/completion/use_single_quotes", false) ? "'" : "\"";

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		ScriptCodeCompletionOption option(p_dir->get_file_path(i), ScriptCodeCompletionOption::KIND_FILE_PATH);
		option.insert_text = quote_style + option.display + quote_style;
		r_list.insert(option.display, option);
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_get_directory_contents(p_dir->get_subdir(i), r_list);
	}
}

static void _find_annotation_arguments(const GDScriptParser::AnnotationNode *p_annotation, int p_argument, const String p_quote_style, Map<String, ScriptCodeCompletionOption> &r_result) {
	if (p_annotation->name == "@export_range" || p_annotation->name == "@export_exp_range") {
		if (p_argument == 3 || p_argument == 4) {
			// Slider hint.
			ScriptCodeCompletionOption slider1("or_greater", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			slider1.insert_text = p_quote_style + slider1.display + p_quote_style;
			r_result.insert(slider1.display, slider1);
			ScriptCodeCompletionOption slider2("or_lesser", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			slider2.insert_text = p_quote_style + slider2.display + p_quote_style;
			r_result.insert(slider2.display, slider2);
		}
	} else if (p_annotation->name == "@export_exp_easing") {
		if (p_argument == 0 || p_argument == 1) {
			// Easing hint.
			ScriptCodeCompletionOption hint1("attenuation", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			hint1.insert_text = p_quote_style + hint1.display + p_quote_style;
			r_result.insert(hint1.display, hint1);
			ScriptCodeCompletionOption hint2("inout", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			hint2.insert_text = p_quote_style + hint2.display + p_quote_style;
			r_result.insert(hint2.display, hint2);
		}
	} else if (p_annotation->name == "@export_node_path") {
		ScriptCodeCompletionOption node("Node", ScriptCodeCompletionOption::KIND_CLASS);
		r_result.insert(node.display, node);
		List<StringName> node_types;
		ClassDB::get_inheriters_from_class("Node", &node_types);
		for (const List<StringName>::Element *E = node_types.front(); E != nullptr; E = E->next()) {
			if (!ClassDB::is_class_exposed(E->get())) {
				continue;
			}
			ScriptCodeCompletionOption option(E->get(), ScriptCodeCompletionOption::KIND_CLASS);
			r_result.insert(option.display, option);
		}
	}
}

static void _list_available_types(bool p_inherit_only, GDScriptParser::CompletionContext &p_context, Map<String, ScriptCodeCompletionOption> &r_result) {
	List<StringName> native_types;
	ClassDB::get_class_list(&native_types);
	for (const List<StringName>::Element *E = native_types.front(); E != nullptr; E = E->next()) {
		if (ClassDB::is_class_exposed(E->get()) && !Engine::get_singleton()->has_singleton(E->get())) {
			ScriptCodeCompletionOption option(E->get(), ScriptCodeCompletionOption::KIND_CLASS);
			r_result.insert(option.display, option);
		}
	}

	if (p_context.current_class) {
		if (!p_inherit_only && p_context.current_class->base_type.is_set()) {
			// Native enums from base class
			List<StringName> enums;
			ClassDB::get_enum_list(p_context.current_class->base_type.native_type, &enums);
			for (const List<StringName>::Element *E = enums.front(); E != nullptr; E = E->next()) {
				ScriptCodeCompletionOption option(E->get(), ScriptCodeCompletionOption::KIND_ENUM);
				r_result.insert(option.display, option);
			}
		}
		// Check current class for potential types
		const GDScriptParser::ClassNode *current = p_context.current_class;
		while (current) {
			for (int i = 0; i < current->members.size(); i++) {
				const GDScriptParser::ClassNode::Member &member = current->members[i];
				switch (member.type) {
					case GDScriptParser::ClassNode::Member::CLASS: {
						ScriptCodeCompletionOption option(member.m_class->identifier->name, ScriptCodeCompletionOption::KIND_CLASS);
						r_result.insert(option.display, option);
					} break;
					case GDScriptParser::ClassNode::Member::ENUM: {
						if (!p_inherit_only) {
							ScriptCodeCompletionOption option(member.m_enum->identifier->name, ScriptCodeCompletionOption::KIND_ENUM);
							r_result.insert(option.display, option);
						}
					} break;
					case GDScriptParser::ClassNode::Member::CONSTANT: {
						if (member.constant->get_datatype().is_meta_type && p_context.current_class->outer != nullptr) {
							ScriptCodeCompletionOption option(member.constant->identifier->name, ScriptCodeCompletionOption::KIND_CLASS);
							r_result.insert(option.display, option);
						}
					} break;
					default:
						break;
				}
			}
			current = current->outer;
		}
	}

	// Global scripts
	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);
	for (const List<StringName>::Element *E = global_classes.front(); E != nullptr; E = E->next()) {
		ScriptCodeCompletionOption option(E->get(), ScriptCodeCompletionOption::KIND_CLASS);
		r_result.insert(option.display, option);
	}

	// Autoload singletons
	Map<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();
	for (const Map<StringName, ProjectSettings::AutoloadInfo>::Element *E = autoloads.front(); E != nullptr; E = E->next()) {
		const ProjectSettings::AutoloadInfo &info = E->get();
		if (!info.is_singleton || info.path.get_extension().to_lower() != "gd") {
			continue;
		}
		ScriptCodeCompletionOption option(info.name, ScriptCodeCompletionOption::KIND_CLASS);
		r_result.insert(option.display, option);
	}
}

static void _find_identifiers_in_suite(const GDScriptParser::SuiteNode *p_suite, Map<String, ScriptCodeCompletionOption> &r_result) {
	for (int i = 0; i < p_suite->locals.size(); i++) {
		ScriptCodeCompletionOption option;
		if (p_suite->locals[i].type == GDScriptParser::SuiteNode::Local::CONSTANT) {
			option = ScriptCodeCompletionOption(p_suite->locals[i].name, ScriptCodeCompletionOption::KIND_CONSTANT);
			option.default_value = p_suite->locals[i].constant->initializer->reduced_value;
		} else {
			option = ScriptCodeCompletionOption(p_suite->locals[i].name, ScriptCodeCompletionOption::KIND_VARIABLE);
		}
		r_result.insert(option.display, option);
	}
	if (p_suite->parent_block) {
		_find_identifiers_in_suite(p_suite->parent_block, r_result);
	}
}

static void _find_identifiers_in_base(const GDScriptCompletionIdentifier &p_base, bool p_only_functions, Map<String, ScriptCodeCompletionOption> &r_result, int p_recursion_depth);

static void _find_identifiers_in_class(const GDScriptParser::ClassNode *p_class, bool p_only_functions, bool p_static, bool p_parent_only, Map<String, ScriptCodeCompletionOption> &r_result, int p_recursion_depth) {
	ERR_FAIL_COND(p_recursion_depth > COMPLETION_RECURSION_LIMIT);

	if (!p_parent_only) {
		bool outer = false;
		const GDScriptParser::ClassNode *clss = p_class;
		while (clss) {
			for (int i = 0; i < clss->members.size(); i++) {
				const GDScriptParser::ClassNode::Member &member = clss->members[i];
				ScriptCodeCompletionOption option;
				switch (member.type) {
					case GDScriptParser::ClassNode::Member::VARIABLE:
						if (p_only_functions || outer || (p_static)) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.variable->identifier->name, ScriptCodeCompletionOption::KIND_MEMBER);
						break;
					case GDScriptParser::ClassNode::Member::CONSTANT:
						if (p_only_functions) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.constant->identifier->name, ScriptCodeCompletionOption::KIND_CONSTANT);
						if (member.constant->initializer) {
							option.default_value = member.constant->initializer->reduced_value;
						}
						break;
					case GDScriptParser::ClassNode::Member::CLASS:
						if (p_only_functions) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.m_class->identifier->name, ScriptCodeCompletionOption::KIND_CLASS);
						break;
					case GDScriptParser::ClassNode::Member::ENUM_VALUE:
						if (p_only_functions) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.enum_value.identifier->name, ScriptCodeCompletionOption::KIND_CONSTANT);
						break;
					case GDScriptParser::ClassNode::Member::ENUM:
						if (p_only_functions) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.m_enum->identifier->name, ScriptCodeCompletionOption::KIND_ENUM);
						break;
					case GDScriptParser::ClassNode::Member::FUNCTION:
						if (outer || (p_static && !member.function->is_static) || member.function->identifier->name.operator String().begins_with("@")) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.function->identifier->name, ScriptCodeCompletionOption::KIND_FUNCTION);
						if (member.function->parameters.size() > 0) {
							option.insert_text += "(";
						} else {
							option.insert_text += "()";
						}
						break;
					case GDScriptParser::ClassNode::Member::SIGNAL:
						if (p_only_functions || outer) {
							continue;
						}
						option = ScriptCodeCompletionOption(member.signal->identifier->name, ScriptCodeCompletionOption::KIND_SIGNAL);
						break;
					case GDScriptParser::ClassNode::Member::UNDEFINED:
						break;
				}
				r_result.insert(option.display, option);
			}
			outer = true;
			clss = clss->outer;
		}
	}

	// Parents.
	GDScriptCompletionIdentifier base_type;
	base_type.type = p_class->base_type;
	base_type.type.is_meta_type = p_static;

	_find_identifiers_in_base(base_type, p_only_functions, r_result, p_recursion_depth + 1);
}

static void _find_identifiers_in_base(const GDScriptCompletionIdentifier &p_base, bool p_only_functions, Map<String, ScriptCodeCompletionOption> &r_result, int p_recursion_depth) {
	ERR_FAIL_COND(p_recursion_depth > COMPLETION_RECURSION_LIMIT);

	GDScriptParser::DataType base_type = p_base.type;
	bool _static = base_type.is_meta_type;

	if (_static && base_type.kind != GDScriptParser::DataType::BUILTIN) {
		ScriptCodeCompletionOption option("new", ScriptCodeCompletionOption::KIND_FUNCTION);
		option.insert_text += "(";
		r_result.insert(option.display, option);
	}

	while (!base_type.has_no_type()) {
		switch (base_type.kind) {
			case GDScriptParser::DataType::CLASS: {
				_find_identifiers_in_class(base_type.class_type, p_only_functions, _static, false, r_result, p_recursion_depth + 1);
				// This already finds all parent identifiers, so we are done.
				base_type = GDScriptParser::DataType();
			} break;
			case GDScriptParser::DataType::SCRIPT: {
				Ref<Script> scr = base_type.script_type;
				if (scr.is_valid()) {
					if (!p_only_functions) {
						if (!_static) {
							List<PropertyInfo> members;
							scr->get_script_property_list(&members);
							for (List<PropertyInfo>::Element *E = members.front(); E; E = E->next()) {
								ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_MEMBER);
								r_result.insert(option.display, option);
							}
						}
						Map<StringName, Variant> constants;
						scr->get_constants(&constants);
						for (Map<StringName, Variant>::Element *E = constants.front(); E; E = E->next()) {
							ScriptCodeCompletionOption option(E->key().operator String(), ScriptCodeCompletionOption::KIND_CONSTANT);
							r_result.insert(option.display, option);
						}

						List<MethodInfo> signals;
						scr->get_script_signal_list(&signals);
						for (List<MethodInfo>::Element *E = signals.front(); E; E = E->next()) {
							ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_SIGNAL);
							r_result.insert(option.display, option);
						}
					}

					List<MethodInfo> methods;
					scr->get_script_method_list(&methods);
					for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
						if (E->get().name.begins_with("@")) {
							continue;
						}
						ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_FUNCTION);
						if (E->get().arguments.size()) {
							option.insert_text += "(";
						} else {
							option.insert_text += "()";
						}
						r_result.insert(option.display, option);
					}

					Ref<Script> base_script = scr->get_base_script();
					if (base_script.is_valid()) {
						base_type.script_type = base_script;
					} else {
						base_type.kind = GDScriptParser::DataType::NATIVE;
						base_type.native_type = scr->get_instance_base_type();
					}
				} else {
					return;
				}
			} break;
			case GDScriptParser::DataType::NATIVE: {
				StringName type = GDScriptParser::get_real_class_name(base_type.native_type);
				if (!ClassDB::class_exists(type)) {
					return;
				}

				if (!p_only_functions) {
					List<String> constants;
					ClassDB::get_integer_constant_list(type, &constants);
					for (List<String>::Element *E = constants.front(); E; E = E->next()) {
						ScriptCodeCompletionOption option(E->get(), ScriptCodeCompletionOption::KIND_CONSTANT);
						r_result.insert(option.display, option);
					}

					if (!_static || Engine::get_singleton()->has_singleton(type)) {
						List<PropertyInfo> pinfo;
						ClassDB::get_property_list(type, &pinfo);
						for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
							if (E->get().usage & (PROPERTY_USAGE_GROUP | PROPERTY_USAGE_CATEGORY)) {
								continue;
							}
							if (E->get().name.find("/") != -1) {
								continue;
							}
							ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_MEMBER);
							r_result.insert(option.display, option);
						}
					}
				}

				if (!_static || Engine::get_singleton()->has_singleton(type)) {
					List<MethodInfo> methods;
					bool is_autocompleting_getters = GLOBAL_GET("debug/gdscript/completion/autocomplete_setters_and_getters").booleanize();
					ClassDB::get_method_list(type, &methods, false, !is_autocompleting_getters);
					for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
						if (E->get().name.begins_with("_")) {
							continue;
						}
						ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_FUNCTION);
						if (E->get().arguments.size()) {
							option.insert_text += "(";
						} else {
							option.insert_text += "()";
						}
						r_result.insert(option.display, option);
					}
				}

				return;
			} break;
			case GDScriptParser::DataType::BUILTIN: {
				Callable::CallError err;
				Variant tmp;
				Variant::construct(base_type.builtin_type, tmp, nullptr, 0, err);
				if (err.error != Callable::CallError::CALL_OK) {
					return;
				}

				if (!p_only_functions) {
					List<PropertyInfo> members;
					if (p_base.value.get_type() != Variant::NIL) {
						p_base.value.get_property_list(&members);
					} else {
						tmp.get_property_list(&members);
					}

					for (List<PropertyInfo>::Element *E = members.front(); E; E = E->next()) {
						if (String(E->get().name).find("/") == -1) {
							ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_MEMBER);
							r_result.insert(option.display, option);
						}
					}
				}

				List<MethodInfo> methods;
				tmp.get_method_list(&methods);
				for (const List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
					ScriptCodeCompletionOption option(E->get().name, ScriptCodeCompletionOption::KIND_FUNCTION);
					if (E->get().arguments.size()) {
						option.insert_text += "(";
					} else {
						option.insert_text += "()";
					}
					r_result.insert(option.display, option);
				}

				return;
			} break;
			default: {
				return;
			} break;
		}
	}
}

static void _find_identifiers(GDScriptParser::CompletionContext &p_context, bool p_only_functions, Map<String, ScriptCodeCompletionOption> &r_result, int p_recursion_depth) {
	if (!p_only_functions && p_context.current_suite) {
		// This includes function parameters, since they are also locals.
		_find_identifiers_in_suite(p_context.current_suite, r_result);
	}

	if (p_context.current_class) {
		_find_identifiers_in_class(p_context.current_class, p_only_functions, (!p_context.current_function || p_context.current_function->is_static), false, r_result, p_recursion_depth + 1);
	}

	List<StringName> functions;
	GDScriptUtilityFunctions::get_function_list(&functions);

	for (const List<StringName>::Element *E = functions.front(); E; E = E->next()) {
		MethodInfo function = GDScriptUtilityFunctions::get_function_info(E->get());
		ScriptCodeCompletionOption option(String(E->get()), ScriptCodeCompletionOption::KIND_FUNCTION);
		if (function.arguments.size() || (function.flags & METHOD_FLAG_VARARG)) {
			option.insert_text += "(";
		} else {
			option.insert_text += "()";
		}
		r_result.insert(option.display, option);
	}

	if (p_only_functions) {
		return;
	}

	static const char *_type_names[Variant::VARIANT_MAX] = {
		"null", "bool", "int", "float", "String", "StringName", "Vector2", "Vector2i", "Rect2", "Rect2i", "Vector3", "Vector3i", "Transform2D", "Plane", "Quat", "AABB", "Basis", "Transform",
		"Color", "NodePath", "RID", "Signal", "Callable", "Object", "Dictionary", "Array", "PackedByteArray", "PackedInt32Array", "PackedInt64Array", "PackedFloat32Array", "PackedFloat64Array", "PackedStringArray",
		"PackedVector2Array", "PackedVector3Array", "PackedColorArray"
	};
	static_assert((sizeof(_type_names) / sizeof(*_type_names)) == Variant::VARIANT_MAX, "Completion for builtin types is incomplete");

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		ScriptCodeCompletionOption option(_type_names[i], ScriptCodeCompletionOption::KIND_CLASS);
		r_result.insert(option.display, option);
	}

	static const char *_keywords[] = {
		"false", "PI", "TAU", "INF", "NAN", "self", "true", "breakpoint", "tool", "super",
		"break", "continue", "pass", "return",
		nullptr
	};

	const char **kw = _keywords;
	while (*kw) {
		ScriptCodeCompletionOption option(*kw, ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
		r_result.insert(option.display, option);
		kw++;
	}

	static const char *_keywords_with_space[] = {
		"and", "in", "not", "or", "as", "class", "extends", "is", "func", "signal", "await",
		"const", "enum", "static", "var", "if", "elif", "else", "for", "match", "while",
		nullptr
	};

	const char **kws = _keywords_with_space;
	while (*kws) {
		ScriptCodeCompletionOption option(*kws, ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
		option.insert_text += " ";
		r_result.insert(option.display, option);
		kws++;
	}

	static const char *_keywords_with_args[] = {
		"assert", "preload",
		nullptr
	};

	const char **kwa = _keywords_with_args;
	while (*kwa) {
		ScriptCodeCompletionOption option(*kwa, ScriptCodeCompletionOption::KIND_FUNCTION);
		option.insert_text += "(";
		r_result.insert(option.display, option);
		kwa++;
	}

	Map<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();
	for (const Map<StringName, ProjectSettings::AutoloadInfo>::Element *E = autoloads.front(); E != nullptr; E = E->next()) {
		if (!E->value().is_singleton) {
			continue;
		}
		ScriptCodeCompletionOption option(E->key(), ScriptCodeCompletionOption::KIND_CONSTANT);
		r_result.insert(option.display, option);
	}

	// Native classes and global constants.
	for (const Map<StringName, int>::Element *E = GDScriptLanguage::get_singleton()->get_global_map().front(); E; E = E->next()) {
		ScriptCodeCompletionOption option;
		if (ClassDB::class_exists(E->key()) || Engine::get_singleton()->has_singleton(E->key())) {
			option = ScriptCodeCompletionOption(E->key().operator String(), ScriptCodeCompletionOption::KIND_CLASS);
		} else {
			option = ScriptCodeCompletionOption(E->key().operator String(), ScriptCodeCompletionOption::KIND_CONSTANT);
		}
		r_result.insert(option.display, option);
	}
}

static GDScriptCompletionIdentifier _type_from_variant(const Variant &p_value) {
	GDScriptCompletionIdentifier ci;
	ci.value = p_value;
	ci.type.is_constant = true;
	ci.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	ci.type.kind = GDScriptParser::DataType::BUILTIN;
	ci.type.builtin_type = p_value.get_type();

	if (ci.type.builtin_type == Variant::OBJECT) {
		Object *obj = p_value.operator Object *();
		if (!obj) {
			return ci;
		}
		ci.type.native_type = obj->get_class_name();
		Ref<Script> scr = p_value;
		if (scr.is_valid()) {
			ci.type.is_meta_type = true;
		} else {
			ci.type.is_meta_type = false;
			scr = obj->get_script();
		}
		if (scr.is_valid()) {
			ci.type.script_type = scr;
			ci.type.kind = GDScriptParser::DataType::SCRIPT;
			ci.type.native_type = scr->get_instance_base_type();
		} else {
			ci.type.kind = GDScriptParser::DataType::NATIVE;
		}
	}

	return ci;
}

static GDScriptCompletionIdentifier _type_from_property(const PropertyInfo &p_property) {
	GDScriptCompletionIdentifier ci;

	if (p_property.type == Variant::NIL) {
		// Variant
		return ci;
	}

	if (p_property.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
		ci.enumeration = p_property.class_name;
	}

	ci.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	ci.type.builtin_type = p_property.type;
	if (p_property.type == Variant::OBJECT) {
		ci.type.kind = GDScriptParser::DataType::NATIVE;
		ci.type.native_type = p_property.class_name == StringName() ? "Object" : p_property.class_name;
	} else {
		ci.type.kind = GDScriptParser::DataType::BUILTIN;
	}
	return ci;
}

static bool _guess_identifier_type(GDScriptParser::CompletionContext &p_context, const StringName &p_identifier, GDScriptCompletionIdentifier &r_type);
static bool _guess_identifier_type_from_base(GDScriptParser::CompletionContext &p_context, const GDScriptCompletionIdentifier &p_base, const StringName &p_identifier, GDScriptCompletionIdentifier &r_type);
static bool _guess_method_return_type_from_base(GDScriptParser::CompletionContext &p_context, const GDScriptCompletionIdentifier &p_base, const StringName &p_method, GDScriptCompletionIdentifier &r_type);

static bool _guess_expression_type(GDScriptParser::CompletionContext &p_context, const GDScriptParser::ExpressionNode *p_expression, GDScriptCompletionIdentifier &r_type) {
	bool found = false;

	if (p_expression->is_constant) {
		// Already has a value, so just use that.
		r_type = _type_from_variant(p_expression->reduced_value);
		found = true;
	} else {
		switch (p_expression->type) {
			case GDScriptParser::Node::LITERAL: {
				const GDScriptParser::LiteralNode *literal = static_cast<const GDScriptParser::LiteralNode *>(p_expression);
				r_type = _type_from_variant(literal->value);
				found = true;
			} break;
			case GDScriptParser::Node::SELF: {
				if (p_context.current_class) {
					r_type.type.kind = GDScriptParser::DataType::CLASS;
					r_type.type.type_source = GDScriptParser::DataType::INFERRED;
					r_type.type.is_constant = true;
					r_type.type.class_type = p_context.current_class;
					r_type.value = p_context.base;
					found = true;
				}
			} break;
			case GDScriptParser::Node::IDENTIFIER: {
				const GDScriptParser::IdentifierNode *id = static_cast<const GDScriptParser::IdentifierNode *>(p_expression);
				found = _guess_identifier_type(p_context, id->name, r_type);
			} break;
			case GDScriptParser::Node::DICTIONARY: {
				// Try to recreate the dictionary.
				const GDScriptParser::DictionaryNode *dn = static_cast<const GDScriptParser::DictionaryNode *>(p_expression);
				Dictionary d;
				bool full = true;
				for (int i = 0; i < dn->elements.size(); i++) {
					GDScriptCompletionIdentifier key;
					if (_guess_expression_type(p_context, dn->elements[i].key, key)) {
						if (!key.type.is_constant) {
							full = false;
							break;
						}
						GDScriptCompletionIdentifier value;
						if (_guess_expression_type(p_context, dn->elements[i].value, value)) {
							if (!value.type.is_constant) {
								full = false;
								break;
							}
							d[key.value] = value.value;
						} else {
							full = false;
							break;
						}
					} else {
						full = false;
						break;
					}
				}
				if (full) {
					r_type.value = d;
					r_type.type.is_constant = true;
				}
				r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				r_type.type.kind = GDScriptParser::DataType::BUILTIN;
				r_type.type.builtin_type = Variant::DICTIONARY;
				found = true;
			} break;
			case GDScriptParser::Node::ARRAY: {
				// Try to recreate the array
				const GDScriptParser::ArrayNode *an = static_cast<const GDScriptParser::ArrayNode *>(p_expression);
				Array a;
				bool full = true;
				a.resize(an->elements.size());
				for (int i = 0; i < an->elements.size(); i++) {
					GDScriptCompletionIdentifier value;
					if (_guess_expression_type(p_context, an->elements[i], value)) {
						if (value.type.is_constant) {
							a[i] = value.value;
						} else {
							full = false;
							break;
						}
					} else {
						full = false;
						break;
					}
				}
				if (full) {
					// If not fully constant, setting this value is detrimental to the inference.
					r_type.value = a;
				}
				r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				r_type.type.kind = GDScriptParser::DataType::BUILTIN;
				r_type.type.builtin_type = Variant::ARRAY;
				found = true;
			} break;
			case GDScriptParser::Node::CAST: {
				const GDScriptParser::CastNode *cn = static_cast<const GDScriptParser::CastNode *>(p_expression);
				GDScriptCompletionIdentifier value;
				if (_guess_expression_type(p_context, cn->operand, r_type)) {
					r_type.type = cn->get_datatype();
					found = true;
				}
			} break;
			case GDScriptParser::Node::CALL: {
				const GDScriptParser::CallNode *call = static_cast<const GDScriptParser::CallNode *>(p_expression);
				if (GDScriptParser::get_builtin_type(call->function_name) < Variant::VARIANT_MAX) {
					r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
					r_type.type.kind = GDScriptParser::DataType::BUILTIN;
					r_type.type.builtin_type = GDScriptParser::get_builtin_type(call->function_name);
					found = true;
					break;
				} else if (GDScriptUtilityFunctions::function_exists(call->function_name)) {
					MethodInfo mi = GDScriptUtilityFunctions::get_function_info(call->function_name);
					r_type = _type_from_property(mi.return_val);
					found = true;
					break;
				} else {
					GDScriptParser::CompletionContext c = p_context;
					c.current_line = call->start_line;

					GDScriptParser::Node::Type callee_type = call->get_callee_type();

					GDScriptCompletionIdentifier base;
					if (callee_type == GDScriptParser::Node::IDENTIFIER || call->is_super) {
						// Simple call, so base is 'self'.
						if (p_context.current_class) {
							base.type.kind = GDScriptParser::DataType::CLASS;
							base.type.type_source = GDScriptParser::DataType::INFERRED;
							base.type.is_constant = true;
							base.type.class_type = p_context.current_class;
							base.value = p_context.base;
						} else {
							break;
						}
					} else if (callee_type == GDScriptParser::Node::SUBSCRIPT && static_cast<const GDScriptParser::SubscriptNode *>(call->callee)->is_attribute) {
						if (!_guess_expression_type(c, static_cast<const GDScriptParser::SubscriptNode *>(call->callee)->base, base)) {
							found = false;
							break;
						}
					} else {
						break;
					}

					// Try call if constant methods with constant arguments
					if (base.type.is_constant && base.value.get_type() == Variant::OBJECT) {
						GDScriptParser::DataType native_type = base.type;

						while (native_type.kind == GDScriptParser::DataType::CLASS) {
							native_type = native_type.class_type->base_type;
						}

						while (native_type.kind == GDScriptParser::DataType::SCRIPT) {
							if (native_type.script_type.is_valid()) {
								Ref<Script> parent = native_type.script_type->get_base_script();
								if (parent.is_valid()) {
									native_type.script_type = parent;
								} else {
									native_type.kind = GDScriptParser::DataType::NATIVE;
									native_type.native_type = native_type.script_type->get_instance_base_type();
									if (!ClassDB::class_exists(native_type.native_type)) {
										native_type.native_type = String("_") + native_type.native_type;
										if (!ClassDB::class_exists(native_type.native_type)) {
											native_type.kind = GDScriptParser::DataType::UNRESOLVED;
										}
									}
								}
							}
						}

						if (native_type.kind == GDScriptParser::DataType::NATIVE) {
							MethodBind *mb = ClassDB::get_method(native_type.native_type, call->function_name);
							if (mb && mb->is_const()) {
								bool all_is_const = true;
								Vector<Variant> args;
								GDScriptParser::CompletionContext c2 = p_context;
								c2.current_line = call->start_line;
								for (int i = 0; all_is_const && i < call->arguments.size(); i++) {
									GDScriptCompletionIdentifier arg;

									if (!call->arguments[i]->is_constant) {
										all_is_const = false;
									}
								}

								Object *baseptr = base.value;

								if (all_is_const && String(call->function_name) == "get_node" && ClassDB::is_parent_class(native_type.native_type, "Node") && args.size()) {
									String arg1 = args[0];
									if (arg1.begins_with("/root/")) {
										String which = arg1.get_slice("/", 2);
										if (which != "") {
											// Try singletons first
											if (GDScriptLanguage::get_singleton()->get_named_globals_map().has(which)) {
												r_type = _type_from_variant(GDScriptLanguage::get_singleton()->get_named_globals_map()[which]);
												found = true;
											} else {
												Map<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();

												for (Map<StringName, ProjectSettings::AutoloadInfo>::Element *E = autoloads.front(); E; E = E->next()) {
													String name = E->key();
													if (name == which) {
														String script = E->value().path;

														if (!script.begins_with("res://")) {
															script = "res://" + script;
														}

														if (!script.ends_with(".gd")) {
															//not a script, try find the script anyway,
															//may have some success
															script = script.get_basename() + ".gd";
														}

														if (FileAccess::exists(script)) {
															Error err = OK;
															Ref<GDScriptParserRef> parser = GDScriptCache::get_parser(script, GDScriptParserRef::INTERFACE_SOLVED, err);
															if (err == OK) {
																r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
																r_type.type.script_path = script;
																r_type.type.class_type = parser->get_parser()->get_tree();
																r_type.type.is_constant = false;
																r_type.type.kind = GDScriptParser::DataType::CLASS;
																r_type.value = Variant();
																p_context.dependent_parsers.push_back(parser);
																found = true;
															}
														}
														break;
													}
												}
											}
										}
									}
								}

								if (!found && all_is_const && baseptr) {
									Vector<const Variant *> argptr;
									for (int i = 0; i < args.size(); i++) {
										argptr.push_back(&args[i]);
									}

									Callable::CallError ce;
									Variant ret = mb->call(baseptr, (const Variant **)argptr.ptr(), argptr.size(), ce);

									if (ce.error == Callable::CallError::CALL_OK && ret.get_type() != Variant::NIL) {
										if (ret.get_type() != Variant::OBJECT || ret.operator Object *() != nullptr) {
											r_type = _type_from_variant(ret);
											found = true;
										}
									}
								}
							}
						}
					}

					if (!found) {
						found = _guess_method_return_type_from_base(c, base, call->function_name, r_type);
					}
				}
			} break;
			case GDScriptParser::Node::SUBSCRIPT: {
				const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(p_expression);
				if (subscript->is_attribute) {
					GDScriptParser::CompletionContext c = p_context;
					c.current_line = subscript->start_line;

					GDScriptCompletionIdentifier base;
					if (!_guess_expression_type(c, subscript->base, base)) {
						found = false;
						break;
					}

					if (base.value.get_type() == Variant::DICTIONARY && base.value.operator Dictionary().has(String(subscript->attribute->name))) {
						Variant value = base.value.operator Dictionary()[String(subscript->attribute->name)];
						r_type = _type_from_variant(value);
						found = true;
						break;
					}

					const GDScriptParser::DictionaryNode *dn = nullptr;
					if (subscript->base->type == GDScriptParser::Node::DICTIONARY) {
						dn = static_cast<const GDScriptParser::DictionaryNode *>(subscript->base);
					} else if (base.assigned_expression && base.assigned_expression->type == GDScriptParser::Node::DICTIONARY) {
						dn = static_cast<const GDScriptParser::DictionaryNode *>(base.assigned_expression);
					}

					if (dn) {
						for (int i = 0; i < dn->elements.size(); i++) {
							GDScriptCompletionIdentifier key;
							if (!_guess_expression_type(c, dn->elements[i].key, key)) {
								continue;
							}
							if (key.value == String(subscript->attribute->name)) {
								r_type.assigned_expression = dn->elements[i].value;
								found = _guess_expression_type(c, dn->elements[i].value, r_type);
								break;
							}
						}
					}

					if (!found) {
						found = _guess_identifier_type_from_base(c, base, subscript->attribute->name, r_type);
					}
				} else {
					if (subscript->index == nullptr) {
						found = false;
						break;
					}

					GDScriptParser::CompletionContext c = p_context;
					c.current_line = subscript->start_line;

					GDScriptCompletionIdentifier base;
					if (!_guess_expression_type(c, subscript->base, base)) {
						found = false;
						break;
					}

					GDScriptCompletionIdentifier index;
					if (!_guess_expression_type(c, subscript->index, index)) {
						found = false;
						break;
					}

					if (base.value.in(index.value)) {
						Variant value = base.value.get(index.value);
						r_type = _type_from_variant(value);
						found = true;
						break;
					}

					// Look if it is a dictionary node.
					const GDScriptParser::DictionaryNode *dn = nullptr;
					if (subscript->base->type == GDScriptParser::Node::DICTIONARY) {
						dn = static_cast<const GDScriptParser::DictionaryNode *>(subscript->base);
					} else if (base.assigned_expression && base.assigned_expression->type == GDScriptParser::Node::DICTIONARY) {
						dn = static_cast<const GDScriptParser::DictionaryNode *>(base.assigned_expression);
					}

					if (dn) {
						for (int i = 0; i < dn->elements.size(); i++) {
							GDScriptCompletionIdentifier key;
							if (!_guess_expression_type(c, dn->elements[i].key, key)) {
								continue;
							}
							if (key.value == index.value) {
								r_type.assigned_expression = dn->elements[i].value;
								found = _guess_expression_type(p_context, dn->elements[i].value, r_type);
								break;
							}
						}
					}

					// Look if it is an array node.
					if (!found && index.value.is_num()) {
						int idx = index.value;
						const GDScriptParser::ArrayNode *an = nullptr;
						if (subscript->base->type == GDScriptParser::Node::ARRAY) {
							an = static_cast<const GDScriptParser::ArrayNode *>(subscript->base);
						} else if (base.assigned_expression && base.assigned_expression->type == GDScriptParser::Node::ARRAY) {
							an = static_cast<const GDScriptParser::ArrayNode *>(base.assigned_expression);
						}

						if (an && idx >= 0 && an->elements.size() > idx) {
							r_type.assigned_expression = an->elements[idx];
							found = _guess_expression_type(c, an->elements[idx], r_type);
							break;
						}
					}

					// Look for valid indexing in other types
					if (!found && (index.value.get_type() == Variant::STRING || index.value.get_type() == Variant::NODE_PATH)) {
						StringName id = index.value;
						found = _guess_identifier_type_from_base(c, base, id, r_type);
					} else if (!found && index.type.kind == GDScriptParser::DataType::BUILTIN) {
						Callable::CallError err;
						Variant base_val;
						Variant::construct(base.type.builtin_type, base_val, nullptr, 0, err);
						bool valid = false;
						Variant res = base_val.get(index.value, &valid);
						if (valid) {
							r_type = _type_from_variant(res);
							r_type.value = Variant();
							r_type.type.is_constant = false;
							found = true;
						}
					}
				}
			} break;
			case GDScriptParser::Node::BINARY_OPERATOR: {
				const GDScriptParser::BinaryOpNode *op = static_cast<const GDScriptParser::BinaryOpNode *>(p_expression);

				if (op->variant_op == Variant::OP_MAX) {
					break;
				}

				GDScriptParser::CompletionContext context = p_context;
				context.current_line = op->start_line;

				GDScriptCompletionIdentifier p1;
				GDScriptCompletionIdentifier p2;

				if (!_guess_expression_type(context, op->left_operand, p1)) {
					found = false;
					break;
				}

				if (!_guess_expression_type(context, op->right_operand, p2)) {
					found = false;
					break;
				}

				Callable::CallError ce;
				bool v1_use_value = p1.value.get_type() != Variant::NIL && p1.value.get_type() != Variant::OBJECT;
				Variant d1;
				Variant::construct(p1.type.builtin_type, d1, nullptr, 0, ce);
				Variant d2;
				Variant::construct(p2.type.builtin_type, d2, nullptr, 0, ce);

				Variant v1 = (v1_use_value) ? p1.value : d1;
				bool v2_use_value = p2.value.get_type() != Variant::NIL && p2.value.get_type() != Variant::OBJECT;
				Variant v2 = (v2_use_value) ? p2.value : d2;
				// avoid potential invalid ops
				if ((op->variant_op == Variant::OP_DIVIDE || op->variant_op == Variant::OP_MODULE) && v2.get_type() == Variant::INT) {
					v2 = 1;
					v2_use_value = false;
				}
				if (op->variant_op == Variant::OP_DIVIDE && v2.get_type() == Variant::FLOAT) {
					v2 = 1.0;
					v2_use_value = false;
				}

				Variant res;
				bool valid;
				Variant::evaluate(op->variant_op, v1, v2, res, valid);
				if (!valid) {
					found = false;
					break;
				}
				r_type = _type_from_variant(res);
				if (!v1_use_value || !v2_use_value) {
					r_type.value = Variant();
					r_type.type.is_constant = false;
				}

				found = true;
			} break;
			default:
				break;
		}
	}

	// It may have found a null, but that's never useful
	if (found && r_type.type.kind == GDScriptParser::DataType::BUILTIN && r_type.type.builtin_type == Variant::NIL) {
		found = false;
	}

	// Check type hint last. For collections we want chance to get the actual value first
	// This way we can detect types from the content of dictionaries and arrays
	if (!found && p_expression->get_datatype().is_hard_type()) {
		r_type.type = p_expression->get_datatype();
		if (!r_type.assigned_expression) {
			r_type.assigned_expression = p_expression;
		}
		found = true;
	}

	return found;
}

static bool _guess_identifier_type(GDScriptParser::CompletionContext &p_context, const StringName &p_identifier, GDScriptCompletionIdentifier &r_type) {
	// Look in blocks first.
	int last_assign_line = -1;
	const GDScriptParser::ExpressionNode *last_assigned_expression = nullptr;
	GDScriptParser::DataType id_type;
	GDScriptParser::SuiteNode *suite = p_context.current_suite;
	bool is_function_parameter = false;

	if (suite) {
		if (suite->has_local(p_identifier)) {
			const GDScriptParser::SuiteNode::Local &local = suite->get_local(p_identifier);

			id_type = local.get_datatype();

			// Check initializer as the first assignment.
			switch (local.type) {
				case GDScriptParser::SuiteNode::Local::VARIABLE:
					if (local.variable->initializer) {
						last_assign_line = local.variable->initializer->end_line;
						last_assigned_expression = local.variable->initializer;
					}
					break;
				case GDScriptParser::SuiteNode::Local::CONSTANT:
					if (local.constant->initializer) {
						last_assign_line = local.constant->initializer->end_line;
						last_assigned_expression = local.constant->initializer;
					}
					break;
				case GDScriptParser::SuiteNode::Local::PARAMETER:
					if (local.parameter->default_value) {
						last_assign_line = local.parameter->default_value->end_line;
						last_assigned_expression = local.parameter->default_value;
					}
					is_function_parameter = true;
					break;
				default:
					break;
			}
		}
	}

	while (suite) {
		for (int i = 0; i < suite->statements.size(); i++) {
			if (suite->statements[i]->start_line > p_context.current_line) {
				break;
			}

			switch (suite->statements[i]->type) {
				case GDScriptParser::Node::ASSIGNMENT: {
					const GDScriptParser::AssignmentNode *assign = static_cast<const GDScriptParser::AssignmentNode *>(suite->statements[i]);
					if (assign->end_line > last_assign_line && assign->assignee && assign->assigned_value && assign->assignee->type == GDScriptParser::Node::IDENTIFIER) {
						const GDScriptParser::IdentifierNode *id = static_cast<const GDScriptParser::IdentifierNode *>(assign->assignee);
						if (id->name == p_identifier) {
							last_assign_line = assign->assigned_value->end_line;
							last_assigned_expression = assign->assigned_value;
						}
					}
				} break;
				default:
					// TODO: Check sub blocks (control flow statements) as they might also reassign stuff.
					break;
			}
		}

		if (suite->parent_if && suite->parent_if->condition && suite->parent_if->condition->type == GDScriptParser::Node::BINARY_OPERATOR && static_cast<const GDScriptParser::BinaryOpNode *>(suite->parent_if->condition)->operation == GDScriptParser::BinaryOpNode::OP_TYPE_TEST) {
			// Operator `is` used, check if identifier is in there! this helps resolve in blocks that are (if (identifier is value)): which are very common..
			// Super dirty hack, but very useful.
			// Credit: Zylann.
			// TODO: this could be hacked to detect ANDed conditions too...
			const GDScriptParser::BinaryOpNode *op = static_cast<const GDScriptParser::BinaryOpNode *>(suite->parent_if->condition);
			if (op->left_operand && op->right_operand && op->left_operand->type == GDScriptParser::Node::IDENTIFIER && static_cast<const GDScriptParser::IdentifierNode *>(op->left_operand)->name == p_identifier) {
				// Bingo.
				GDScriptParser::CompletionContext c = p_context;
				c.current_line = op->left_operand->start_line;
				c.current_suite = suite;
				GDScriptCompletionIdentifier is_type;
				if (_guess_expression_type(c, op->right_operand, is_type)) {
					id_type = is_type.type;
					id_type.is_meta_type = false;
					if (last_assign_line < c.current_line) {
						// Override last assignment.
						last_assign_line = c.current_line;
						last_assigned_expression = nullptr;
					}
				}
			}
		}

		suite = suite->parent_block;
	}

	if (last_assigned_expression && last_assign_line != p_context.current_line) {
		GDScriptParser::CompletionContext c = p_context;
		c.current_line = last_assign_line;
		r_type.assigned_expression = last_assigned_expression;
		if (_guess_expression_type(c, last_assigned_expression, r_type)) {
			return true;
		}
	}

	if (is_function_parameter && p_context.current_function && p_context.current_class) {
		// Check if it's override of native function, then we can assume the type from the signature.
		GDScriptParser::DataType base_type = p_context.current_class->base_type;
		while (base_type.is_set()) {
			switch (base_type.kind) {
				case GDScriptParser::DataType::CLASS:
					if (base_type.class_type->has_function(p_context.current_function->identifier->name)) {
						GDScriptParser::FunctionNode *parent_function = base_type.class_type->get_member(p_context.current_function->identifier->name).function;
						const GDScriptParser::ParameterNode *parameter = parent_function->parameters[parent_function->parameters_indices[p_identifier]];
						if ((!id_type.is_set() || id_type.is_variant()) && parameter->get_datatype().is_hard_type()) {
							id_type = parameter->get_datatype();
						}
						if (parameter->default_value) {
							GDScriptParser::CompletionContext c = p_context;
							c.current_function = parent_function;
							c.current_class = base_type.class_type;
							c.base = nullptr;
							if (_guess_expression_type(c, parameter->default_value, r_type)) {
								return true;
							}
						}
					}
					base_type = base_type.class_type->base_type;
					break;
				case GDScriptParser::DataType::NATIVE: {
					if (id_type.is_set() && !id_type.is_variant()) {
						base_type = GDScriptParser::DataType();
						break;
					}
					StringName real_native = GDScriptParser::get_real_class_name(base_type.native_type);
					MethodInfo info;
					if (ClassDB::get_method_info(real_native, p_context.current_function->identifier->name, &info)) {
						for (const List<PropertyInfo>::Element *E = info.arguments.front(); E; E = E->next()) {
							if (E->get().name == p_identifier) {
								r_type = _type_from_property(E->get());
								return true;
							}
						}
					}
					base_type = GDScriptParser::DataType();
				} break;
				default:
					break;
			}
		}
	}

	if (id_type.is_set() && !id_type.is_variant()) {
		r_type.type = id_type;
		return true;
	}

	// Check current class (including inheritance).
	if (p_context.current_class) {
		GDScriptCompletionIdentifier base;
		base.value = p_context.base;
		base.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
		base.type.kind = GDScriptParser::DataType::CLASS;
		base.type.class_type = p_context.current_class;
		base.type.is_meta_type = p_context.current_function && p_context.current_function->is_static;

		if (_guess_identifier_type_from_base(p_context, base, p_identifier, r_type)) {
			return true;
		}
	}

	// Check global scripts.
	if (ScriptServer::is_global_class(p_identifier)) {
		String script = ScriptServer::get_global_class_path(p_identifier);
		if (script.to_lower().ends_with(".gd")) {
			Error err = OK;
			Ref<GDScriptParserRef> parser = GDScriptCache::get_parser(script, GDScriptParserRef::INTERFACE_SOLVED, err);
			if (err == OK) {
				r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				r_type.type.script_path = script;
				r_type.type.class_type = parser->get_parser()->get_tree();
				r_type.type.is_constant = false;
				r_type.type.kind = GDScriptParser::DataType::CLASS;
				r_type.value = Variant();
				p_context.dependent_parsers.push_back(parser);
				return true;
			}
		} else {
			Ref<Script> scr = ResourceLoader::load(ScriptServer::get_global_class_path(p_identifier));
			if (scr.is_valid()) {
				r_type = _type_from_variant(scr);
				r_type.type.is_meta_type = true;
				return true;
			}
		}
		return false;
	}

	// Check autoloads.
	if (ProjectSettings::get_singleton()->has_autoload(p_identifier)) {
		r_type = _type_from_variant(GDScriptLanguage::get_singleton()->get_named_globals_map()[p_identifier]);
		return true;
	}

	// Check ClassDB.
	StringName class_name = GDScriptParser::get_real_class_name(p_identifier);
	if (ClassDB::class_exists(class_name) && ClassDB::is_class_exposed(class_name)) {
		r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
		r_type.type.kind = GDScriptParser::DataType::NATIVE;
		r_type.type.native_type = p_identifier;
		r_type.type.is_constant = true;
		r_type.type.is_meta_type = !Engine::get_singleton()->has_singleton(class_name);
		r_type.value = Variant();
	}

	return false;
}

static bool _guess_identifier_type_from_base(GDScriptParser::CompletionContext &p_context, const GDScriptCompletionIdentifier &p_base, const StringName &p_identifier, GDScriptCompletionIdentifier &r_type) {
	GDScriptParser::DataType base_type = p_base.type;
	bool is_static = base_type.is_meta_type;
	while (base_type.is_set()) {
		switch (base_type.kind) {
			case GDScriptParser::DataType::CLASS:
				if (base_type.class_type->has_member(p_identifier)) {
					const GDScriptParser::ClassNode::Member &member = base_type.class_type->get_member(p_identifier);
					switch (member.type) {
						case GDScriptParser::ClassNode::Member::CONSTANT:
							r_type.type = member.constant->get_datatype();
							if (member.constant->initializer && member.constant->initializer->is_constant) {
								r_type.value = member.constant->initializer->reduced_value;
							}
							return true;
						case GDScriptParser::ClassNode::Member::VARIABLE:
							if (!is_static) {
								if (member.variable->initializer) {
									const GDScriptParser::ExpressionNode *init = member.variable->initializer;
									if (init->is_constant) {
										r_type.value = init->reduced_value;
										r_type = _type_from_variant(init->reduced_value);
										return true;
									} else if (init->start_line == p_context.current_line) {
										return false;
									} else if (_guess_expression_type(p_context, init, r_type)) {
										return true;
									} else if (init->get_datatype().is_set() && !init->get_datatype().is_variant()) {
										r_type.type = init->get_datatype();
										return true;
									}
								} else if (member.variable->get_datatype().is_set() && !member.variable->get_datatype().is_variant()) {
									r_type.type = member.variable->get_datatype();
									return true;
								}
							}
							// TODO: Check assignments in constructor.
							return false;
						case GDScriptParser::ClassNode::Member::ENUM:
							r_type.type = member.m_enum->get_datatype();
							r_type.enumeration = member.m_enum->identifier->name;
							return true;
						case GDScriptParser::ClassNode::Member::ENUM_VALUE:
							r_type = _type_from_variant(member.enum_value.value);
							return true;
						case GDScriptParser::ClassNode::Member::SIGNAL:
							r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
							r_type.type.kind = GDScriptParser::DataType::BUILTIN;
							r_type.type.builtin_type = Variant::SIGNAL;
							return true;
						case GDScriptParser::ClassNode::Member::FUNCTION:
							if (is_static && !member.function->is_static) {
								return false;
							}
							r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
							r_type.type.kind = GDScriptParser::DataType::BUILTIN;
							r_type.type.builtin_type = Variant::CALLABLE;
							return true;
						case GDScriptParser::ClassNode::Member::CLASS:
							r_type.type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
							r_type.type.kind = GDScriptParser::DataType::CLASS;
							r_type.type.class_type = member.m_class;
							return true;
						case GDScriptParser::ClassNode::Member::UNDEFINED:
							return false; // Unreachable.
					}
					return false;
				}
				base_type = base_type.class_type->base_type;
				break;
			case GDScriptParser::DataType::SCRIPT: {
				Ref<Script> scr = base_type.script_type;
				if (scr.is_valid()) {
					Map<StringName, Variant> constants;
					scr->get_constants(&constants);
					if (constants.has(p_identifier)) {
						r_type = _type_from_variant(constants[p_identifier]);
						return true;
					}

					if (!is_static) {
						List<PropertyInfo> members;
						scr->get_script_property_list(&members);
						for (const List<PropertyInfo>::Element *E = members.front(); E; E = E->next()) {
							const PropertyInfo &prop = E->get();
							if (prop.name == p_identifier) {
								r_type = _type_from_property(prop);
								return true;
							}
						}
					}
					Ref<Script> parent = scr->get_base_script();
					if (parent.is_valid()) {
						base_type.script_type = parent;
					} else {
						base_type.kind = GDScriptParser::DataType::NATIVE;
						base_type.native_type = scr->get_instance_base_type();
					}
				} else {
					return false;
				}
			} break;
			case GDScriptParser::DataType::NATIVE: {
				StringName class_name = GDScriptParser::get_real_class_name(base_type.native_type);
				if (!ClassDB::class_exists(class_name)) {
					return false;
				}

				// Skip constants since they're all integers. Type does not matter because int has no members.

				PropertyInfo prop;
				if (ClassDB::get_property_info(class_name, p_identifier, &prop)) {
					StringName getter = ClassDB::get_property_getter(class_name, p_identifier);
					if (getter != StringName()) {
						MethodBind *g = ClassDB::get_method(class_name, getter);
						if (g) {
							r_type = _type_from_property(g->get_return_info());
							return true;
						}
					} else {
						r_type = _type_from_property(prop);
						return true;
					}
				}
				return false;
			} break;
			case GDScriptParser::DataType::BUILTIN: {
				Callable::CallError err;
				Variant tmp;
				Variant::construct(base_type.builtin_type, tmp, nullptr, 0, err);

				if (err.error != Callable::CallError::CALL_OK) {
					return false;
				}
				bool valid = false;
				Variant res = tmp.get(p_identifier, &valid);
				if (valid) {
					r_type = _type_from_variant(res);
					r_type.value = Variant();
					r_type.type.is_constant = false;
					return true;
				}
				return false;
			} break;
			default: {
				return false;
			} break;
		}
	}
	return false;
}

static void _find_last_return_in_block(GDScriptParser::CompletionContext &p_context, int &r_last_return_line, const GDScriptParser::ExpressionNode **r_last_returned_value) {
	if (!p_context.current_suite) {
		return;
	}

	for (int i = 0; i < p_context.current_suite->statements.size(); i++) {
		if (p_context.current_suite->statements[i]->start_line < r_last_return_line) {
			break;
		}

		GDScriptParser::CompletionContext c = p_context;
		switch (p_context.current_suite->statements[i]->type) {
			case GDScriptParser::Node::FOR:
				c.current_suite = static_cast<const GDScriptParser::ForNode *>(p_context.current_suite->statements[i])->loop;
				_find_last_return_in_block(c, r_last_return_line, r_last_returned_value);
				break;
			case GDScriptParser::Node::WHILE:
				c.current_suite = static_cast<const GDScriptParser::WhileNode *>(p_context.current_suite->statements[i])->loop;
				_find_last_return_in_block(c, r_last_return_line, r_last_returned_value);
				break;
			case GDScriptParser::Node::IF: {
				const GDScriptParser::IfNode *_if = static_cast<const GDScriptParser::IfNode *>(p_context.current_suite->statements[i]);
				c.current_suite = _if->true_block;
				_find_last_return_in_block(c, r_last_return_line, r_last_returned_value);
				if (_if->false_block) {
					c.current_suite = _if->false_block;
					_find_last_return_in_block(c, r_last_return_line, r_last_returned_value);
				}
			} break;
			case GDScriptParser::Node::MATCH: {
				const GDScriptParser::MatchNode *match = static_cast<const GDScriptParser::MatchNode *>(p_context.current_suite->statements[i]);
				for (int j = 0; j < match->branches.size(); j++) {
					c.current_suite = match->branches[j]->block;
					_find_last_return_in_block(c, r_last_return_line, r_last_returned_value);
				}
			} break;
			case GDScriptParser::Node::RETURN: {
				const GDScriptParser::ReturnNode *ret = static_cast<const GDScriptParser::ReturnNode *>(p_context.current_suite->statements[i]);
				if (ret->return_value) {
					if (ret->start_line > r_last_return_line) {
						r_last_return_line = ret->start_line;
						*r_last_returned_value = ret->return_value;
					}
				}
			} break;
			default:
				break;
		}
	}
}

static bool _guess_method_return_type_from_base(GDScriptParser::CompletionContext &p_context, const GDScriptCompletionIdentifier &p_base, const StringName &p_method, GDScriptCompletionIdentifier &r_type) {
	GDScriptParser::DataType base_type = p_base.type;
	bool is_static = base_type.is_meta_type;

	if (is_static && p_method == "new") {
		r_type.type = base_type;
		r_type.type.is_meta_type = false;
		r_type.type.is_constant = false;
		return true;
	}

	while (base_type.is_set() && !base_type.is_variant()) {
		switch (base_type.kind) {
			case GDScriptParser::DataType::CLASS:
				if (base_type.class_type->has_function(p_method)) {
					const GDScriptParser::FunctionNode *method = base_type.class_type->get_member(p_method).function;
					if (!is_static || method->is_static) {
						int last_return_line = -1;
						const GDScriptParser::ExpressionNode *last_returned_value = nullptr;
						GDScriptParser::CompletionContext c = p_context;
						c.current_class = base_type.class_type;
						c.current_function = const_cast<GDScriptParser::FunctionNode *>(method);
						c.current_suite = method->body;

						_find_last_return_in_block(c, last_return_line, &last_returned_value);
						if (last_returned_value) {
							c.current_line = c.current_suite->end_line;
							if (_guess_expression_type(c, last_returned_value, r_type)) {
								return true;
							}
							if (method->get_datatype().is_set() && !method->get_datatype().is_variant()) {
								r_type.type = method->get_datatype();
								return true;
							}
						}
					}
				}
				base_type = base_type.class_type->base_type;
				break;
			case GDScriptParser::DataType::SCRIPT: {
				Ref<Script> scr = base_type.script_type;
				if (scr.is_valid()) {
					List<MethodInfo> methods;
					scr->get_script_method_list(&methods);
					for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
						MethodInfo &mi = E->get();
						if (mi.name == p_method) {
							r_type = _type_from_property(mi.return_val);
							return true;
						}
					}
					Ref<Script> base_script = scr->get_base_script();
					if (base_script.is_valid()) {
						base_type.script_type = base_script;
					} else {
						base_type.kind = GDScriptParser::DataType::NATIVE;
						base_type.native_type = scr->get_instance_base_type();
					}
				} else {
					return false;
				}
			} break;
			case GDScriptParser::DataType::NATIVE: {
				StringName native = GDScriptParser::get_real_class_name(base_type.native_type);
				if (!ClassDB::class_exists(native)) {
					return false;
				}
				MethodBind *mb = ClassDB::get_method(native, p_method);
				if (mb) {
					r_type = _type_from_property(mb->get_return_info());
					return true;
				}
				return false;
			} break;
			case GDScriptParser::DataType::BUILTIN: {
				Callable::CallError err;
				Variant tmp;
				Variant::construct(base_type.builtin_type, tmp, nullptr, 0, err);
				if (err.error != Callable::CallError::CALL_OK) {
					return false;
				}

				List<MethodInfo> methods;
				tmp.get_method_list(&methods);

				for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
					MethodInfo &mi = E->get();
					if (mi.name == p_method) {
						r_type = _type_from_property(mi.return_val);
						return true;
					}
				}
				return false;
			} break;
			default: {
				return false;
			}
		}
	}

	return false;
}

static void _find_enumeration_candidates(GDScriptParser::CompletionContext &p_context, const String &p_enum_hint, Map<String, ScriptCodeCompletionOption> &r_result) {
	if (p_enum_hint.find(".") == -1) {
		// Global constant or in the current class.
		StringName current_enum = p_enum_hint;
		if (p_context.current_class && p_context.current_class->has_member(current_enum) && p_context.current_class->get_member(current_enum).type == GDScriptParser::ClassNode::Member::ENUM) {
			const GDScriptParser::EnumNode *_enum = p_context.current_class->get_member(current_enum).m_enum;
			for (int i = 0; i < _enum->values.size(); i++) {
				ScriptCodeCompletionOption option(_enum->values[i].identifier->name, ScriptCodeCompletionOption::KIND_ENUM);
				r_result.insert(option.display, option);
			}
		} else {
			for (int i = 0; i < CoreConstants::get_global_constant_count(); i++) {
				if (CoreConstants::get_global_constant_enum(i) == current_enum) {
					ScriptCodeCompletionOption option(CoreConstants::get_global_constant_name(i), ScriptCodeCompletionOption::KIND_ENUM);
					r_result.insert(option.display, option);
				}
			}
		}
	} else {
		String class_name = p_enum_hint.get_slice(".", 0);
		String enum_name = p_enum_hint.get_slice(".", 1);

		if (!ClassDB::class_exists(class_name)) {
			return;
		}

		List<StringName> enum_constants;
		ClassDB::get_enum_constants(class_name, enum_name, &enum_constants);
		for (List<StringName>::Element *E = enum_constants.front(); E; E = E->next()) {
			String candidate = class_name + "." + E->get();
			ScriptCodeCompletionOption option(candidate, ScriptCodeCompletionOption::KIND_ENUM);
			r_result.insert(option.display, option);
		}
	}
}

static void _find_call_arguments(GDScriptParser::CompletionContext &p_context, const GDScriptCompletionIdentifier &p_base, const StringName &p_method, int p_argidx, bool p_static, Map<String, ScriptCodeCompletionOption> &r_result, String &r_arghint) {
	Variant base = p_base.value;
	GDScriptParser::DataType base_type = p_base.type;

	const String quote_style = EDITOR_DEF("text_editor/completion/use_single_quotes", false) ? "'" : "\"";

	while (base_type.is_set() && !base_type.is_variant()) {
		switch (base_type.kind) {
			case GDScriptParser::DataType::CLASS: {
				if (base_type.class_type->has_member(p_method)) {
					const GDScriptParser::ClassNode::Member &member = base_type.class_type->get_member(p_method);

					if (member.type == GDScriptParser::ClassNode::Member::FUNCTION) {
						r_arghint = _make_arguments_hint(member.function, p_argidx);
						return;
					}
				}

				base_type = base_type.class_type->base_type;
			} break;
			case GDScriptParser::DataType::NATIVE: {
				StringName class_name = GDScriptParser::get_real_class_name(base_type.native_type);
				if (!ClassDB::class_exists(class_name)) {
					base_type.kind = GDScriptParser::DataType::UNRESOLVED;
					break;
				}

				MethodInfo info;
				int method_args = 0;

				if (ClassDB::get_method_info(class_name, p_method, &info)) {
					method_args = info.arguments.size();
					if (base.get_type() == Variant::OBJECT) {
						Object *obj = base.operator Object *();
						if (obj) {
							List<String> options;
							obj->get_argument_options(p_method, p_argidx, &options);
							for (List<String>::Element *F = options.front(); F; F = F->next()) {
								ScriptCodeCompletionOption option(F->get(), ScriptCodeCompletionOption::KIND_FUNCTION);
								r_result.insert(option.display, option);
							}
						}
					}

					if (p_argidx < method_args) {
						PropertyInfo arg_info = info.arguments[p_argidx];
						if (arg_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
							_find_enumeration_candidates(p_context, arg_info.class_name, r_result);
						}
					}

					r_arghint = _make_arguments_hint(info, p_argidx);
				}

				if (p_argidx == 0 && ClassDB::is_parent_class(class_name, "Node") && (p_method == "get_node" || p_method == "has_node")) {
					// Get autoloads
					List<PropertyInfo> props;
					ProjectSettings::get_singleton()->get_property_list(&props);

					for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
						String s = E->get().name;
						if (!s.begins_with("autoload/")) {
							continue;
						}
						String name = s.get_slice("/", 1);
						ScriptCodeCompletionOption option("/root/" + name, ScriptCodeCompletionOption::KIND_NODE_PATH);
						option.insert_text = quote_style + option.display + quote_style;
						r_result.insert(option.display, option);
					}
				}

				if (p_argidx == 0 && method_args > 0 && ClassDB::is_parent_class(class_name, "InputEvent") && p_method.operator String().find("action") != -1) {
					// Get input actions
					List<PropertyInfo> props;
					ProjectSettings::get_singleton()->get_property_list(&props);
					for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
						String s = E->get().name;
						if (!s.begins_with("input/")) {
							continue;
						}
						String name = s.get_slice("/", 1);
						ScriptCodeCompletionOption option(name, ScriptCodeCompletionOption::KIND_CONSTANT);
						option.insert_text = quote_style + option.display + quote_style;
						r_result.insert(option.display, option);
					}
				}

				base_type.kind = GDScriptParser::DataType::UNRESOLVED;
			} break;
			case GDScriptParser::DataType::BUILTIN: {
				if (base.get_type() == Variant::NIL) {
					Callable::CallError err;
					Variant::construct(base_type.builtin_type, base, nullptr, 0, err);
					if (err.error != Callable::CallError::CALL_OK) {
						return;
					}
				}

				List<MethodInfo> methods;
				base.get_method_list(&methods);
				for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
					if (E->get().name == p_method) {
						r_arghint = _make_arguments_hint(E->get(), p_argidx);
						return;
					}
				}

				base_type.kind = GDScriptParser::DataType::UNRESOLVED;
			} break;
			default: {
				base_type.kind = GDScriptParser::DataType::UNRESOLVED;
			} break;
		}
	}
}

static void _find_call_arguments(GDScriptParser::CompletionContext &p_context, const GDScriptParser::Node *p_call, int p_argidx, Map<String, ScriptCodeCompletionOption> &r_result, bool &r_forced, String &r_arghint) {
	const String quote_style = EDITOR_DEF("text_editor/completion/use_single_quotes", false) ? "'" : "\"";

	if (p_call->type == GDScriptParser::Node::PRELOAD) {
		if (p_argidx == 0 && bool(EditorSettings::get_singleton()->get("text_editor/completion/complete_file_paths"))) {
			_get_directory_contents(EditorFileSystem::get_singleton()->get_filesystem(), r_result);
		}

		MethodInfo mi(PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource"), "preload", PropertyInfo(Variant::STRING, "path"));
		r_arghint = _make_arguments_hint(mi, p_argidx);
		return;
	} else if (p_call->type != GDScriptParser::Node::CALL) {
		return;
	}

	Variant base;
	GDScriptParser::DataType base_type;
	bool _static = false;
	const GDScriptParser::CallNode *call = static_cast<const GDScriptParser::CallNode *>(p_call);
	GDScriptParser::Node::Type callee_type = call->get_callee_type();

	GDScriptCompletionIdentifier connect_base;

	if (GDScriptUtilityFunctions::function_exists(call->function_name)) {
		MethodInfo info = GDScriptUtilityFunctions::get_function_info(call->function_name);
		r_arghint = _make_arguments_hint(info, p_argidx);
		return;
	} else if (GDScriptParser::get_builtin_type(call->function_name) < Variant::VARIANT_MAX) {
		// Complete constructor
		List<MethodInfo> constructors;
		Variant::get_constructor_list(GDScriptParser::get_builtin_type(call->function_name), &constructors);

		int i = 0;
		for (List<MethodInfo>::Element *E = constructors.front(); E; E = E->next()) {
			if (p_argidx >= E->get().arguments.size()) {
				continue;
			}
			if (i > 0) {
				r_arghint += "\n";
			}
			r_arghint += _make_arguments_hint(E->get(), p_argidx);
			i++;
		}
		return;
	} else if (call->is_super || callee_type == GDScriptParser::Node::IDENTIFIER) {
		base = p_context.base;

		if (p_context.current_class) {
			base_type = p_context.current_class->get_datatype();
			_static = !p_context.current_function || p_context.current_function->is_static;
		}
	} else if (callee_type == GDScriptParser::Node::SUBSCRIPT) {
		const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(call->callee);

		if (subscript->is_attribute) {
			GDScriptCompletionIdentifier ci;
			if (_guess_expression_type(p_context, subscript->base, ci)) {
				base_type = ci.type;
				base = ci.value;
			} else {
				return;
			}

			_static = base_type.is_meta_type;
		}
	} else {
		return;
	}

	GDScriptCompletionIdentifier ci;
	ci.type = base_type;
	ci.value = base;
	_find_call_arguments(p_context, ci, call->function_name, p_argidx, _static, r_result, r_arghint);

	r_forced = r_result.size() > 0;
}

Error GDScriptLanguage::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptCodeCompletionOption> *r_options, bool &r_forced, String &r_call_hint) {
	const String quote_style = EDITOR_DEF("text_editor/completion/use_single_quotes", false) ? "'" : "\"";

	GDScriptParser parser;
	GDScriptAnalyzer analyzer(&parser);

	parser.parse(p_code, p_path, true);
	analyzer.analyze();

	r_forced = false;
	Map<String, ScriptCodeCompletionOption> options;

	GDScriptParser::CompletionContext completion_context = parser.get_completion_context();
	completion_context.base = p_owner;
	bool is_function = false;

	switch (completion_context.type) {
		case GDScriptParser::COMPLETION_NONE:
			break;
		case GDScriptParser::COMPLETION_ANNOTATION: {
			List<MethodInfo> annotations;
			parser.get_annotation_list(&annotations);
			for (const List<MethodInfo>::Element *E = annotations.front(); E != nullptr; E = E->next()) {
				ScriptCodeCompletionOption option(E->get().name.substr(1), ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
				if (E->get().arguments.size() > 0) {
					option.insert_text += "(";
				}
				options.insert(option.display, option);
			}
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_ANNOTATION_ARGUMENTS: {
			if (completion_context.node == nullptr || completion_context.node->type != GDScriptParser::Node::ANNOTATION) {
				break;
			}
			const GDScriptParser::AnnotationNode *annotation = static_cast<const GDScriptParser::AnnotationNode *>(completion_context.node);
			_find_annotation_arguments(annotation, completion_context.current_argument, quote_style, options);
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_BUILT_IN_TYPE_CONSTANT: {
			List<StringName> constants;
			Variant::get_constants_for_type(completion_context.builtin_type, &constants);
			for (const List<StringName>::Element *E = constants.front(); E != nullptr; E = E->next()) {
				ScriptCodeCompletionOption option(E->get(), ScriptCodeCompletionOption::KIND_CONSTANT);
				bool valid = false;
				Variant default_value = Variant::get_constant_value(completion_context.builtin_type, E->get(), &valid);
				if (valid) {
					option.default_value = default_value;
				}
				options.insert(option.display, option);
			}
		} break;
		case GDScriptParser::COMPLETION_INHERIT_TYPE: {
			_list_available_types(true, completion_context, options);
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_TYPE_NAME_OR_VOID: {
			ScriptCodeCompletionOption option("void", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			options.insert(option.display, option);
		}
			[[fallthrough]];
		case GDScriptParser::COMPLETION_TYPE_NAME: {
			_list_available_types(false, completion_context, options);
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_PROPERTY_DECLARATION_OR_TYPE: {
			_list_available_types(false, completion_context, options);
			ScriptCodeCompletionOption get("get", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			options.insert(get.display, get);
			ScriptCodeCompletionOption set("set", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			options.insert(set.display, set);
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_PROPERTY_DECLARATION: {
			ScriptCodeCompletionOption get("get", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			options.insert(get.display, get);
			ScriptCodeCompletionOption set("set", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			options.insert(set.display, set);
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_PROPERTY_METHOD: {
			if (!completion_context.current_class) {
				break;
			}
			for (int i = 0; i < completion_context.current_class->members.size(); i++) {
				const GDScriptParser::ClassNode::Member &member = completion_context.current_class->members[i];
				if (member.type != GDScriptParser::ClassNode::Member::FUNCTION) {
					continue;
				}
				if (member.function->is_static) {
					continue;
				}
				ScriptCodeCompletionOption option(member.function->identifier->name, ScriptCodeCompletionOption::KIND_FUNCTION);
				options.insert(option.display, option);
			}
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_ASSIGN: {
			GDScriptCompletionIdentifier type;
			if (!completion_context.node || completion_context.node->type != GDScriptParser::Node::ASSIGNMENT) {
				break;
			}
			if (!_guess_expression_type(completion_context, static_cast<const GDScriptParser::AssignmentNode *>(completion_context.node)->assignee, type)) {
				_find_identifiers(completion_context, false, options, 0);
				r_forced = true;
				break;
			}

			if (!type.enumeration.is_empty()) {
				_find_enumeration_candidates(completion_context, type.enumeration, options);
				r_forced = options.size() > 0;
			} else {
				_find_identifiers(completion_context, false, options, 0);
				r_forced = true;
			}
		} break;
		case GDScriptParser::COMPLETION_METHOD:
			is_function = true;
			[[fallthrough]];
		case GDScriptParser::COMPLETION_IDENTIFIER: {
			_find_identifiers(completion_context, is_function, options, 0);
		} break;
		case GDScriptParser::COMPLETION_ATTRIBUTE_METHOD:
			is_function = true;
			[[fallthrough]];
		case GDScriptParser::COMPLETION_ATTRIBUTE: {
			r_forced = true;
			const GDScriptParser::SubscriptNode *attr = static_cast<const GDScriptParser::SubscriptNode *>(completion_context.node);
			if (attr->base) {
				GDScriptCompletionIdentifier base;
				if (!_guess_expression_type(completion_context, attr->base, base)) {
					break;
				}

				_find_identifiers_in_base(base, is_function, options, 0);
			}
		} break;
		case GDScriptParser::COMPLETION_SUBSCRIPT: {
			const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(completion_context.node);
			GDScriptCompletionIdentifier base;
			if (!_guess_expression_type(completion_context, subscript->base, base)) {
				break;
			}

			GDScriptParser::CompletionContext c = completion_context;
			c.current_function = nullptr;
			c.current_suite = nullptr;
			c.base = base.value.get_type() == Variant::OBJECT ? base.value.operator Object *() : nullptr;
			if (base.type.kind == GDScriptParser::DataType::CLASS) {
				c.current_class = base.type.class_type;
			} else {
				c.current_class = nullptr;
			}

			_find_identifiers_in_base(base, false, options, 0);
		} break;
		case GDScriptParser::COMPLETION_TYPE_ATTRIBUTE: {
			if (!completion_context.current_class) {
				break;
			}
			const GDScriptParser::TypeNode *type = static_cast<const GDScriptParser::TypeNode *>(completion_context.node);
			bool found = true;
			GDScriptCompletionIdentifier base;
			base.type.kind = GDScriptParser::DataType::CLASS;
			base.type.type_source = GDScriptParser::DataType::INFERRED;
			base.type.is_constant = true;
			base.type.class_type = completion_context.current_class;
			base.value = completion_context.base;

			for (int i = 0; i < completion_context.current_argument; i++) {
				GDScriptCompletionIdentifier ci;
				if (!_guess_identifier_type_from_base(completion_context, base, type->type_chain[i]->name, ci)) {
					found = false;
					break;
				}
				base = ci;
			}

			// TODO: Improve this to only list types.
			if (found) {
				_find_identifiers_in_base(base, false, options, 0);
			}
			r_forced = true;
		} break;
		case GDScriptParser::COMPLETION_RESOURCE_PATH: {
			if (EditorSettings::get_singleton()->get("text_editor/completion/complete_file_paths")) {
				_get_directory_contents(EditorFileSystem::get_singleton()->get_filesystem(), options);
				r_forced = true;
			}
		} break;
		case GDScriptParser::COMPLETION_CALL_ARGUMENTS: {
			if (!completion_context.node) {
				break;
			}
			_find_call_arguments(completion_context, completion_context.node, completion_context.current_argument, options, r_forced, r_call_hint);
		} break;
		case GDScriptParser::COMPLETION_OVERRIDE_METHOD: {
			GDScriptParser::DataType native_type = completion_context.current_class->base_type;
			while (native_type.is_set() && native_type.kind != GDScriptParser::DataType::NATIVE) {
				switch (native_type.kind) {
					case GDScriptParser::DataType::CLASS: {
						native_type = native_type.class_type->base_type;
					} break;
					default: {
						native_type.kind = GDScriptParser::DataType::UNRESOLVED;
					} break;
				}
			}

			if (!native_type.is_set()) {
				break;
			}

			StringName class_name = GDScriptParser::get_real_class_name(native_type.native_type);
			if (!ClassDB::class_exists(class_name)) {
				break;
			}

			bool use_type_hint = EditorSettings::get_singleton()->get_setting("text_editor/completion/add_type_hints").operator bool();

			List<MethodInfo> virtual_methods;
			ClassDB::get_virtual_methods(class_name, &virtual_methods);
			for (List<MethodInfo>::Element *E = virtual_methods.front(); E; E = E->next()) {
				MethodInfo &mi = E->get();
				String method_hint = mi.name;
				if (method_hint.find(":") != -1) {
					method_hint = method_hint.get_slice(":", 0);
				}
				method_hint += "(";

				if (mi.arguments.size()) {
					for (int i = 0; i < mi.arguments.size(); i++) {
						if (i > 0) {
							method_hint += ", ";
						}
						String arg = mi.arguments[i].name;
						if (arg.find(":") != -1) {
							arg = arg.substr(0, arg.find(":"));
						}
						method_hint += arg;
						if (use_type_hint && mi.arguments[i].type != Variant::NIL) {
							method_hint += ": ";
							if (mi.arguments[i].type == Variant::OBJECT && mi.arguments[i].class_name != StringName()) {
								method_hint += mi.arguments[i].class_name.operator String();
							} else {
								method_hint += Variant::get_type_name(mi.arguments[i].type);
							}
						}
					}
				}
				method_hint += ")";
				if (use_type_hint && (mi.return_val.type != Variant::NIL || !(mi.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT))) {
					method_hint += " -> ";
					if (mi.return_val.type == Variant::NIL) {
						method_hint += "void";
					} else if (mi.return_val.type == Variant::OBJECT && mi.return_val.class_name != StringName()) {
						method_hint += mi.return_val.class_name.operator String();
					} else {
						method_hint += Variant::get_type_name(mi.return_val.type);
					}
				}
				method_hint += ":";

				ScriptCodeCompletionOption option(method_hint, ScriptCodeCompletionOption::KIND_FUNCTION);
				options.insert(option.display, option);
			}
		} break;
		case GDScriptParser::COMPLETION_GET_NODE: {
			if (p_owner) {
				List<String> opts;
				p_owner->get_argument_options("get_node", 0, &opts);

				for (List<String>::Element *E = opts.front(); E; E = E->next()) {
					String opt = E->get().strip_edges();
					if (opt.is_quoted()) {
						r_forced = true;
						String idopt = opt.unquote();
						if (idopt.replace("/", "_").is_valid_identifier()) {
							ScriptCodeCompletionOption option(idopt, ScriptCodeCompletionOption::KIND_NODE_PATH);
							options.insert(option.display, option);
						} else {
							ScriptCodeCompletionOption option(opt, ScriptCodeCompletionOption::KIND_NODE_PATH);
							options.insert(option.display, option);
						}
					}
				}

				// Get autoloads.
				Map<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();

				for (Map<StringName, ProjectSettings::AutoloadInfo>::Element *E = autoloads.front(); E; E = E->next()) {
					String name = E->key();
					ScriptCodeCompletionOption option(quote_style + "/root/" + name + quote_style, ScriptCodeCompletionOption::KIND_NODE_PATH);
					options.insert(option.display, option);
				}
			}
		} break;
		case GDScriptParser::COMPLETION_SUPER_METHOD: {
			if (!completion_context.current_class) {
				break;
			}
			_find_identifiers_in_class(completion_context.current_class, true, false, true, options, 0);
		} break;
	}

	for (Map<String, ScriptCodeCompletionOption>::Element *E = options.front(); E; E = E->next()) {
		r_options->push_back(E->get());
	}

	return OK;
}

#else

Error GDScriptLanguage::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptCodeCompletionOption> *r_options, bool &r_forced, String &r_call_hint) {
	return OK;
}

#endif

//////// END COMPLETION //////////

String GDScriptLanguage::_get_indentation() const {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		bool use_space_indentation = EDITOR_DEF("text_editor/indent/type", false);

		if (use_space_indentation) {
			int indent_size = EDITOR_DEF("text_editor/indent/size", 4);

			String space_indent = "";
			for (int i = 0; i < indent_size; i++) {
				space_indent += " ";
			}
			return space_indent;
		}
	}
#endif
	return "\t";
}

void GDScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
	String indent = _get_indentation();

	Vector<String> lines = p_code.split("\n");
	List<int> indent_stack;

	for (int i = 0; i < lines.size(); i++) {
		String l = lines[i];
		int tc = 0;
		for (int j = 0; j < l.length(); j++) {
			if (l[j] == ' ' || l[j] == '\t') {
				tc++;
			} else {
				break;
			}
		}

		String st = l.substr(tc, l.length()).strip_edges();
		if (st == "" || st.begins_with("#")) {
			continue; //ignore!
		}

		int ilevel = 0;
		if (indent_stack.size()) {
			ilevel = indent_stack.back()->get();
		}

		if (tc > ilevel) {
			indent_stack.push_back(tc);
		} else if (tc < ilevel) {
			while (indent_stack.size() && indent_stack.back()->get() > tc) {
				indent_stack.pop_back();
			}

			if (indent_stack.size() && indent_stack.back()->get() != tc) {
				indent_stack.push_back(tc); //this is not right but gets the job done
			}
		}

		if (i >= p_from_line) {
			l = "";
			for (int j = 0; j < indent_stack.size(); j++) {
				l += indent;
			}
			l += st;

		} else if (i > p_to_line) {
			break;
		}

		lines.write[i] = l;
	}

	p_code = "";
	for (int i = 0; i < lines.size(); i++) {
		if (i > 0) {
			p_code += "\n";
		}
		p_code += lines[i];
	}
}

#ifdef TOOLS_ENABLED

static Error _lookup_symbol_from_base(const GDScriptParser::DataType &p_base, const String &p_symbol, bool p_is_function, GDScriptLanguage::LookupResult &r_result) {
	GDScriptParser::DataType base_type = p_base;

	while (base_type.is_set()) {
		switch (base_type.kind) {
			case GDScriptParser::DataType::CLASS: {
				if (base_type.class_type) {
					if (base_type.class_type->has_member(p_symbol)) {
						r_result.type = ScriptLanguage::LookupResult::RESULT_SCRIPT_LOCATION;
						r_result.location = base_type.class_type->get_member(p_symbol).get_line();
						return OK;
					}
					base_type = base_type.class_type->base_type;
				}
			} break;
			case GDScriptParser::DataType::SCRIPT: {
				Ref<Script> scr = base_type.script_type;
				if (scr.is_valid()) {
					int line = scr->get_member_line(p_symbol);
					if (line >= 0) {
						r_result.type = ScriptLanguage::LookupResult::RESULT_SCRIPT_LOCATION;
						r_result.location = line;
						r_result.script = scr;
						return OK;
					}
					Ref<Script> base_script = scr->get_base_script();
					if (base_script.is_valid()) {
						base_type.script_type = base_script;
					} else {
						base_type.kind = GDScriptParser::DataType::NATIVE;
						base_type.native_type = scr->get_instance_base_type();
					}
				} else {
					base_type.kind = GDScriptParser::DataType::UNRESOLVED;
				}
			} break;
			case GDScriptParser::DataType::NATIVE: {
				StringName class_name = GDScriptParser::get_real_class_name(base_type.native_type);
				if (!ClassDB::class_exists(class_name)) {
					base_type.kind = GDScriptParser::DataType::UNRESOLVED;
					break;
				}

				if (ClassDB::has_method(class_name, p_symbol, true)) {
					r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_METHOD;
					r_result.class_name = base_type.native_type;
					r_result.class_member = p_symbol;
					return OK;
				}

				List<MethodInfo> virtual_methods;
				ClassDB::get_virtual_methods(class_name, &virtual_methods, true);
				for (List<MethodInfo>::Element *E = virtual_methods.front(); E; E = E->next()) {
					if (E->get().name == p_symbol) {
						r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_METHOD;
						r_result.class_name = base_type.native_type;
						r_result.class_member = p_symbol;
						return OK;
					}
				}

				StringName enum_name = ClassDB::get_integer_constant_enum(class_name, p_symbol, true);
				if (enum_name != StringName()) {
					r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_ENUM;
					r_result.class_name = base_type.native_type;
					r_result.class_member = enum_name;
					return OK;
				}

				List<String> constants;
				ClassDB::get_integer_constant_list(class_name, &constants, true);
				for (List<String>::Element *E = constants.front(); E; E = E->next()) {
					if (E->get() == p_symbol) {
						r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_CONSTANT;
						r_result.class_name = base_type.native_type;
						r_result.class_member = p_symbol;
						return OK;
					}
				}

				if (ClassDB::has_property(class_name, p_symbol, true)) {
					r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_PROPERTY;
					r_result.class_name = base_type.native_type;
					r_result.class_member = p_symbol;
					return OK;
				}

				StringName parent = ClassDB::get_parent_class(class_name);
				if (parent != StringName()) {
					if (String(parent).begins_with("_")) {
						base_type.native_type = String(parent).substr(1);
					} else {
						base_type.native_type = parent;
					}
				} else {
					base_type.kind = GDScriptParser::DataType::UNRESOLVED;
				}
			} break;
			case GDScriptParser::DataType::BUILTIN: {
				base_type.kind = GDScriptParser::DataType::UNRESOLVED;

				if (Variant::has_constant(base_type.builtin_type, p_symbol)) {
					r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_CONSTANT;
					r_result.class_name = Variant::get_type_name(base_type.builtin_type);
					r_result.class_member = p_symbol;
					return OK;
				}

				Variant v;
				REF v_ref;
				if (base_type.builtin_type == Variant::OBJECT) {
					v_ref.instance();
					v = v_ref;
				} else {
					Callable::CallError err;
					Variant::construct(base_type.builtin_type, v, nullptr, 0, err);
					if (err.error != Callable::CallError::CALL_OK) {
						break;
					}
				}

				if (v.has_method(p_symbol)) {
					r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_METHOD;
					r_result.class_name = Variant::get_type_name(base_type.builtin_type);
					r_result.class_member = p_symbol;
					return OK;
				}

				bool valid = false;
				v.get(p_symbol, &valid);
				if (valid) {
					r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_PROPERTY;
					r_result.class_name = Variant::get_type_name(base_type.builtin_type);
					r_result.class_member = p_symbol;
					return OK;
				}
			} break;
			default: {
				base_type.kind = GDScriptParser::DataType::UNRESOLVED;
			} break;
		}
	}

	return ERR_CANT_RESOLVE;
}

Error GDScriptLanguage::lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner, LookupResult &r_result) {
	//before parsing, try the usual stuff
	if (ClassDB::class_exists(p_symbol)) {
		r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS;
		r_result.class_name = p_symbol;
		return OK;
	} else {
		String under_prefix = "_" + p_symbol;
		if (ClassDB::class_exists(under_prefix)) {
			r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS;
			r_result.class_name = p_symbol;
			return OK;
		}
	}

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		Variant::Type t = Variant::Type(i);
		if (Variant::get_type_name(t) == p_symbol) {
			r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS;
			r_result.class_name = Variant::get_type_name(t);
			return OK;
		}
	}

	if (GDScriptUtilityFunctions::function_exists(p_symbol)) {
		r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_METHOD;
		r_result.class_name = "@GDScript";
		r_result.class_member = p_symbol;
		return OK;
	}

	if ("PI" == p_symbol || "TAU" == p_symbol || "INF" == p_symbol || "NAN" == p_symbol) {
		r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_CONSTANT;
		r_result.class_name = "@GDScript";
		r_result.class_member = p_symbol;
		return OK;
	}

	GDScriptParser parser;
	parser.parse(p_code, p_path, true);
	GDScriptAnalyzer analyzer(&parser);
	analyzer.analyze();

	GDScriptParser::CompletionContext context = parser.get_completion_context();

	if (context.current_class && context.current_class->extends.size() > 0) {
		bool success = false;
		ClassDB::get_integer_constant(context.current_class->extends[0], p_symbol, &success);
		if (success) {
			r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_CONSTANT;
			r_result.class_name = context.current_class->extends[0];
			r_result.class_member = p_symbol;
			return OK;
		}
	}

	bool is_function = false;

	switch (context.type) {
		case GDScriptParser::COMPLETION_BUILT_IN_TYPE_CONSTANT: {
			r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_CONSTANT;
			r_result.class_name = Variant::get_type_name(context.builtin_type);
			r_result.class_member = p_symbol;
			return OK;
		} break;
		case GDScriptParser::COMPLETION_SUPER_METHOD:
		case GDScriptParser::COMPLETION_METHOD: {
			is_function = true;
			[[fallthrough]];
		}
		case GDScriptParser::COMPLETION_CALL_ARGUMENTS:
		case GDScriptParser::COMPLETION_IDENTIFIER: {
			GDScriptParser::DataType base_type;
			if (context.current_class) {
				if (context.type != GDScriptParser::COMPLETION_SUPER_METHOD) {
					base_type = context.current_class->get_datatype();
				} else {
					base_type = context.current_class->base_type;
				}
			} else {
				break;
			}

			if (!is_function && context.current_suite) {
				// Lookup local variables.
				const GDScriptParser::SuiteNode *suite = context.current_suite;
				while (suite) {
					if (suite->has_local(p_symbol)) {
						r_result.type = ScriptLanguage::LookupResult::RESULT_SCRIPT_LOCATION;
						r_result.location = suite->get_local(p_symbol).start_line;
						return OK;
					}
					suite = suite->parent_block;
				}
			}

			if (_lookup_symbol_from_base(base_type, p_symbol, is_function, r_result) == OK) {
				return OK;
			}

			if (!is_function) {
				// Guess in autoloads as singletons.
				if (ProjectSettings::get_singleton()->has_autoload(p_symbol)) {
					const ProjectSettings::AutoloadInfo &singleton = ProjectSettings::get_singleton()->get_autoload(p_symbol);
					if (singleton.is_singleton) {
						String script = singleton.path;
						if (!script.ends_with(".gd")) {
							// Not a script, try find the script anyway,
							// may have some success.
							script = script.get_basename() + ".gd";
						}

						if (FileAccess::exists(script)) {
							r_result.type = ScriptLanguage::LookupResult::RESULT_SCRIPT_LOCATION;
							r_result.location = 0;
							r_result.script = ResourceLoader::load(script);
							return OK;
						}
					}
				}

				// Global.
				Map<StringName, int> classes = GDScriptLanguage::get_singleton()->get_global_map();
				if (classes.has(p_symbol)) {
					Variant value = GDScriptLanguage::get_singleton()->get_global_array()[classes[p_symbol]];
					if (value.get_type() == Variant::OBJECT) {
						Object *obj = value;
						if (obj) {
							if (Object::cast_to<GDScriptNativeClass>(obj)) {
								r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS;
								r_result.class_name = Object::cast_to<GDScriptNativeClass>(obj)->get_name();
							} else {
								r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS;
								r_result.class_name = obj->get_class();
							}

							// proxy class remove the underscore.
							if (r_result.class_name.begins_with("_")) {
								r_result.class_name = r_result.class_name.substr(1);
							}
							return OK;
						}
					} else {
						/*
						// Because get_integer_constant_enum and get_integer_constant don't work on @GlobalScope
						// We cannot determine the exact nature of the identifier here
						// Otherwise these codes would work
						StringName enumName = ClassDB::get_integer_constant_enum("@GlobalScope", p_symbol, true);
						if (enumName != nullptr) {
							r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_ENUM;
							r_result.class_name = "@GlobalScope";
							r_result.class_member = enumName;
							return OK;
						}
						else {
							r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_CONSTANT;
							r_result.class_name = "@GlobalScope";
							r_result.class_member = p_symbol;
							return OK;
						}*/
						r_result.type = ScriptLanguage::LookupResult::RESULT_CLASS_TBD_GLOBALSCOPE;
						r_result.class_name = "@GlobalScope";
						r_result.class_member = p_symbol;
						return OK;
					}
				}
			}
		} break;
		case GDScriptParser::COMPLETION_ATTRIBUTE_METHOD: {
			is_function = true;
			[[fallthrough]];
		}
		case GDScriptParser::COMPLETION_ATTRIBUTE: {
			if (context.node->type != GDScriptParser::Node::SUBSCRIPT) {
				break;
			}
			const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(context.node);
			if (!subscript->is_attribute) {
				break;
			}
			GDScriptCompletionIdentifier base;
			if (!_guess_expression_type(context, subscript->base, base)) {
				break;
			}

			if (_lookup_symbol_from_base(base.type, p_symbol, is_function, r_result) == OK) {
				return OK;
			}
		} break;
		case GDScriptParser::COMPLETION_OVERRIDE_METHOD: {
			GDScriptParser::DataType base_type = context.current_class->base_type;

			if (_lookup_symbol_from_base(base_type, p_symbol, true, r_result) == OK) {
				return OK;
			}
		} break;
		case GDScriptParser::COMPLETION_TYPE_NAME_OR_VOID:
		case GDScriptParser::COMPLETION_TYPE_NAME: {
			GDScriptParser::DataType base_type = context.current_class->get_datatype();

			if (_lookup_symbol_from_base(base_type, p_symbol, false, r_result) == OK) {
				return OK;
			}
		} break;
		default: {
		}
	}

	return ERR_CANT_RESOLVE;
}

#endif
