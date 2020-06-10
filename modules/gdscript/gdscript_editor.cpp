/*************************************************************************/
/*  gdscript_editor.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/engine.h"
#include "core/global_constants.h"
#include "core/os/file_access.h"
#include "gdscript_analyzer.h"
#include "gdscript_compiler.h"
#include "gdscript_parser.h"
#include "gdscript_tokenizer.h"

#ifdef TOOLS_ENABLED
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
			r_funcs[function->start_line] = p_prefix.empty() ? String(function->identifier->name) : p_prefix + "." + String(function->identifier->name);
		} else if (p_class->members[i].type == GDScriptParser::ClassNode::Member::CLASS) {
			String new_prefix = p_class->members[i].m_class->identifier->name;
			get_function_names_recursively(p_class->members[i].m_class, p_prefix.empty() ? new_prefix : p_prefix + "." + new_prefix, r_funcs);
		}
	}
}

bool GDScriptLanguage::validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions, List<ScriptLanguage::Warning> *r_warnings, Set<int> *r_safe_lines) const {
	GDScriptParser parser;
	GDScriptAnalyzer analyzer(&parser);

	Error err = parser.parse(p_script, p_path, false);
#ifdef DEBUG_ENABLED
	// FIXME: Warnings.
	// if (r_warnings) {
	// 	for (const List<GDScriptWarning>::Element *E = parser.get_warnings().front(); E; E = E->next()) {
	// 		const GDScriptWarning &warn = E->get();
	// 		ScriptLanguage::Warning w;
	// 		w.line = warn.line;
	// 		w.code = (int)warn.code;
	// 		w.string_code = GDScriptWarning::get_name_from_code(warn.code);
	// 		w.message = warn.get_message();
	// 		r_warnings->push_back(w);
	// 	}
	// }
#endif
	if (err == OK) {
		err = analyzer.analyze();
	}
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

	return true;
}

bool GDScriptLanguage::has_named_classes() const {
	return false;
}

bool GDScriptLanguage::supports_builtin_mode() const {
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
		for (int i = 0; i < GlobalConstants::get_global_constant_count(); i++) {
			if (E->key() == GlobalConstants::get_global_constant_name(i)) {
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
	for (int i = 0; i < GDScriptFunctions::FUNC_MAX; i++) {
		p_functions->push_back(GDScriptFunctions::get_info(GDScriptFunctions::Function(i)));
	}

	//not really "functions", but..
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
				if (!type.empty() && type != "var") {
					s += ": " + type;
				}
			}
		}
	}
	s += String(")") + (th ? " -> void" : "") + ":\n" + _get_indentation() + "pass # Replace with function body.\n";

	return s;
}

//////// COMPLETION //////////

// FIXME: Readd completion
Error GDScriptLanguage::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptCodeCompletionOption> *r_options, bool &r_forced, String &r_call_hint) {
	return OK;
}

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

Error GDScriptLanguage::lookup_code(const String &p_code, const String &p_symbol, const String &p_path, Object *p_owner, LookupResult &r_result) {
	// FIXME: Implement lookup.
	return ERR_CANT_RESOLVE;
}

#endif
