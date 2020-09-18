/*************************************************************************/
/*  gdscript_parser.cpp                                                  */
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

#include "gdscript_parser.h"

#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "gdscript.h"

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#include "core/string_builder.h"
#endif // DEBUG_ENABLED

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

static HashMap<StringName, Variant::Type> builtin_types;
Variant::Type GDScriptParser::get_builtin_type(const StringName &p_type) {
	if (builtin_types.empty()) {
		builtin_types["bool"] = Variant::BOOL;
		builtin_types["int"] = Variant::INT;
		builtin_types["float"] = Variant::FLOAT;
		builtin_types["String"] = Variant::STRING;
		builtin_types["Vector2"] = Variant::VECTOR2;
		builtin_types["Vector2i"] = Variant::VECTOR2I;
		builtin_types["Rect2"] = Variant::RECT2;
		builtin_types["Rect2i"] = Variant::RECT2I;
		builtin_types["Transform2D"] = Variant::TRANSFORM2D;
		builtin_types["Vector3"] = Variant::VECTOR3;
		builtin_types["Vector3i"] = Variant::VECTOR3I;
		builtin_types["AABB"] = Variant::AABB;
		builtin_types["Plane"] = Variant::PLANE;
		builtin_types["Quat"] = Variant::QUAT;
		builtin_types["Basis"] = Variant::BASIS;
		builtin_types["Transform"] = Variant::TRANSFORM;
		builtin_types["Color"] = Variant::COLOR;
		builtin_types["RID"] = Variant::_RID;
		builtin_types["Object"] = Variant::OBJECT;
		builtin_types["StringName"] = Variant::STRING_NAME;
		builtin_types["NodePath"] = Variant::NODE_PATH;
		builtin_types["Dictionary"] = Variant::DICTIONARY;
		builtin_types["Callable"] = Variant::CALLABLE;
		builtin_types["Signal"] = Variant::SIGNAL;
		builtin_types["Array"] = Variant::ARRAY;
		builtin_types["PackedByteArray"] = Variant::PACKED_BYTE_ARRAY;
		builtin_types["PackedInt32Array"] = Variant::PACKED_INT32_ARRAY;
		builtin_types["PackedInt64Array"] = Variant::PACKED_INT64_ARRAY;
		builtin_types["PackedFloat32Array"] = Variant::PACKED_FLOAT32_ARRAY;
		builtin_types["PackedFloat64Array"] = Variant::PACKED_FLOAT64_ARRAY;
		builtin_types["PackedStringArray"] = Variant::PACKED_STRING_ARRAY;
		builtin_types["PackedVector2Array"] = Variant::PACKED_VECTOR2_ARRAY;
		builtin_types["PackedVector3Array"] = Variant::PACKED_VECTOR3_ARRAY;
		builtin_types["PackedColorArray"] = Variant::PACKED_COLOR_ARRAY;
		// NIL is not here, hence the -1.
		if (builtin_types.size() != Variant::VARIANT_MAX - 1) {
			ERR_PRINT("Outdated parser: amount of built-in types don't match the amount of types in Variant.");
		}
	}

	if (builtin_types.has(p_type)) {
		return builtin_types[p_type];
	}
	return Variant::VARIANT_MAX;
}

void GDScriptParser::cleanup() {
	builtin_types.clear();
}

GDScriptFunctions::Function GDScriptParser::get_builtin_function(const StringName &p_name) {
	for (int i = 0; i < GDScriptFunctions::FUNC_MAX; i++) {
		if (p_name == GDScriptFunctions::get_func_name(GDScriptFunctions::Function(i))) {
			return GDScriptFunctions::Function(i);
		}
	}
	return GDScriptFunctions::FUNC_MAX;
}

void GDScriptParser::get_annotation_list(List<MethodInfo> *r_annotations) const {
	List<StringName> keys;
	valid_annotations.get_key_list(&keys);
	for (const List<StringName>::Element *E = keys.front(); E != nullptr; E = E->next()) {
		r_annotations->push_back(valid_annotations[E->get()].info);
	}
}

GDScriptParser::GDScriptParser() {
	// Register valid annotations.
	// TODO: Should this be static?
	// TODO: Validate applicable types (e.g. a VARIABLE annotation that only applies to string variables).
	register_annotation(MethodInfo("@tool"), AnnotationInfo::SCRIPT, &GDScriptParser::tool_annotation);
	register_annotation(MethodInfo("@icon", { Variant::STRING, "icon_path" }), AnnotationInfo::SCRIPT, &GDScriptParser::icon_annotation);
	register_annotation(MethodInfo("@onready"), AnnotationInfo::VARIABLE, &GDScriptParser::onready_annotation);
	// Export annotations.
	register_annotation(MethodInfo("@export"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_TYPE_STRING, Variant::NIL>);
	register_annotation(MethodInfo("@export_enum", { Variant::STRING, "names" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_ENUM, Variant::INT>, 0, true);
	register_annotation(MethodInfo("@export_file", { Variant::STRING, "filter" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_FILE, Variant::STRING>, 1, true);
	register_annotation(MethodInfo("@export_dir"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_DIR, Variant::STRING>);
	register_annotation(MethodInfo("@export_global_file", { Variant::STRING, "filter" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_GLOBAL_FILE, Variant::STRING>, 1, true);
	register_annotation(MethodInfo("@export_global_dir"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_GLOBAL_DIR, Variant::STRING>);
	register_annotation(MethodInfo("@export_multiline"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_MULTILINE_TEXT, Variant::STRING>);
	register_annotation(MethodInfo("@export_placeholder"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_PLACEHOLDER_TEXT, Variant::STRING>);
	register_annotation(MethodInfo("@export_range", { Variant::FLOAT, "min" }, { Variant::FLOAT, "max" }, { Variant::FLOAT, "step" }, { Variant::STRING, "slider1" }, { Variant::STRING, "slider2" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_RANGE, Variant::FLOAT>, 3);
	register_annotation(MethodInfo("@export_exp_range", { Variant::FLOAT, "min" }, { Variant::FLOAT, "max" }, { Variant::FLOAT, "step" }, { Variant::STRING, "slider1" }, { Variant::STRING, "slider2" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_EXP_RANGE, Variant::FLOAT>, 3);
	register_annotation(MethodInfo("@export_exp_easing", { Variant::STRING, "hint1" }, { Variant::STRING, "hint2" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_EXP_EASING, Variant::FLOAT>, 2);
	register_annotation(MethodInfo("@export_color_no_alpha"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_COLOR_NO_ALPHA, Variant::COLOR>);
	register_annotation(MethodInfo("@export_node_path", { Variant::STRING, "type" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_NODE_PATH_VALID_TYPES, Variant::NODE_PATH>, 1, true);
	register_annotation(MethodInfo("@export_flags", { Variant::STRING, "names" }), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_FLAGS, Variant::INT>, 0, true);
	register_annotation(MethodInfo("@export_flags_2d_render"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_LAYERS_2D_RENDER, Variant::INT>);
	register_annotation(MethodInfo("@export_flags_2d_physics"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_LAYERS_2D_PHYSICS, Variant::INT>);
	register_annotation(MethodInfo("@export_flags_3d_render"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_LAYERS_3D_RENDER, Variant::INT>);
	register_annotation(MethodInfo("@export_flags_3d_physics"), AnnotationInfo::VARIABLE, &GDScriptParser::export_annotations<PROPERTY_HINT_LAYERS_3D_PHYSICS, Variant::INT>);
	// Networking.
	register_annotation(MethodInfo("@remote"), AnnotationInfo::VARIABLE | AnnotationInfo::FUNCTION, &GDScriptParser::network_annotations<MultiplayerAPI::RPC_MODE_REMOTE>);
	register_annotation(MethodInfo("@master"), AnnotationInfo::VARIABLE | AnnotationInfo::FUNCTION, &GDScriptParser::network_annotations<MultiplayerAPI::RPC_MODE_MASTER>);
	register_annotation(MethodInfo("@puppet"), AnnotationInfo::VARIABLE | AnnotationInfo::FUNCTION, &GDScriptParser::network_annotations<MultiplayerAPI::RPC_MODE_PUPPET>);
	register_annotation(MethodInfo("@remotesync"), AnnotationInfo::VARIABLE | AnnotationInfo::FUNCTION, &GDScriptParser::network_annotations<MultiplayerAPI::RPC_MODE_REMOTESYNC>);
	register_annotation(MethodInfo("@mastersync"), AnnotationInfo::VARIABLE | AnnotationInfo::FUNCTION, &GDScriptParser::network_annotations<MultiplayerAPI::RPC_MODE_MASTERSYNC>);
	register_annotation(MethodInfo("@puppetsync"), AnnotationInfo::VARIABLE | AnnotationInfo::FUNCTION, &GDScriptParser::network_annotations<MultiplayerAPI::RPC_MODE_PUPPETSYNC>);
	// TODO: Warning annotations.
}

GDScriptParser::~GDScriptParser() {
	clear();
}

void GDScriptParser::clear() {
	while (list != nullptr) {
		Node *element = list;
		list = list->next;
		memdelete(element);
	}

	head = nullptr;
	list = nullptr;
	_is_tool = false;
	for_completion = false;
	errors.clear();
	multiline_stack.clear();
}

void GDScriptParser::push_error(const String &p_message, const Node *p_origin) {
	// TODO: Improve error reporting by pointing at source code.
	// TODO: Errors might point at more than one place at once (e.g. show previous declaration).
	panic_mode = true;
	// TODO: Improve positional information.
	if (p_origin == nullptr) {
		errors.push_back({ p_message, current.start_line, current.start_column });
	} else {
		errors.push_back({ p_message, p_origin->start_line, p_origin->leftmost_column });
	}
}

#ifdef DEBUG_ENABLED
void GDScriptParser::push_warning(const Node *p_source, GDScriptWarning::Code p_code, const String &p_symbol1, const String &p_symbol2, const String &p_symbol3, const String &p_symbol4) {
	Vector<String> symbols;
	if (!p_symbol1.empty()) {
		symbols.push_back(p_symbol1);
	}
	if (!p_symbol2.empty()) {
		symbols.push_back(p_symbol2);
	}
	if (!p_symbol3.empty()) {
		symbols.push_back(p_symbol3);
	}
	if (!p_symbol4.empty()) {
		symbols.push_back(p_symbol4);
	}
	push_warning(p_source, p_code, symbols);
}

void GDScriptParser::push_warning(const Node *p_source, GDScriptWarning::Code p_code, const Vector<String> &p_symbols) {
	if (is_ignoring_warnings) {
		return;
	}
	if (GLOBAL_GET("debug/gdscript/warnings/exclude_addons").booleanize() && script_path.begins_with("res://addons/")) {
		return;
	}

	String warn_name = GDScriptWarning::get_name_from_code((GDScriptWarning::Code)p_code).to_lower();
	if (ignored_warnings.has(warn_name)) {
		return;
	}
	if (!GLOBAL_GET("debug/gdscript/warnings/" + warn_name)) {
		return;
	}

	GDScriptWarning warning;
	warning.code = p_code;
	warning.symbols = p_symbols;
	warning.start_line = p_source->start_line;
	warning.end_line = p_source->end_line;
	warning.leftmost_column = p_source->leftmost_column;
	warning.rightmost_column = p_source->rightmost_column;

	List<GDScriptWarning>::Element *before = nullptr;
	for (List<GDScriptWarning>::Element *E = warnings.front(); E != nullptr; E = E->next()) {
		if (E->get().start_line > warning.start_line) {
			break;
		}
		before = E;
	}
	if (before) {
		warnings.insert_after(before, warning);
	} else {
		warnings.push_front(warning);
	}
}
#endif

void GDScriptParser::make_completion_context(CompletionType p_type, Node *p_node, int p_argument, bool p_force) {
	if (!for_completion || (!p_force && completion_context.type != COMPLETION_NONE)) {
		return;
	}
	if (previous.cursor_place != GDScriptTokenizer::CURSOR_MIDDLE && previous.cursor_place != GDScriptTokenizer::CURSOR_END && current.cursor_place == GDScriptTokenizer::CURSOR_NONE) {
		return;
	}
	CompletionContext context;
	context.type = p_type;
	context.current_class = current_class;
	context.current_function = current_function;
	context.current_suite = current_suite;
	context.current_line = tokenizer.get_cursor_line();
	context.current_argument = p_argument;
	context.node = p_node;
	completion_context = context;
}

void GDScriptParser::make_completion_context(CompletionType p_type, Variant::Type p_builtin_type, bool p_force) {
	if (!for_completion || (!p_force && completion_context.type != COMPLETION_NONE)) {
		return;
	}
	if (previous.cursor_place != GDScriptTokenizer::CURSOR_MIDDLE && previous.cursor_place != GDScriptTokenizer::CURSOR_END && current.cursor_place == GDScriptTokenizer::CURSOR_NONE) {
		return;
	}
	CompletionContext context;
	context.type = p_type;
	context.current_class = current_class;
	context.current_function = current_function;
	context.current_suite = current_suite;
	context.current_line = tokenizer.get_cursor_line();
	context.builtin_type = p_builtin_type;
	completion_context = context;
}

void GDScriptParser::push_completion_call(Node *p_call) {
	if (!for_completion) {
		return;
	}
	CompletionCall call;
	call.call = p_call;
	call.argument = 0;
	completion_call_stack.push_back(call);
	if (previous.cursor_place == GDScriptTokenizer::CURSOR_MIDDLE || previous.cursor_place == GDScriptTokenizer::CURSOR_END || current.cursor_place == GDScriptTokenizer::CURSOR_BEGINNING) {
		completion_call = call;
	}
}

void GDScriptParser::pop_completion_call() {
	if (!for_completion) {
		return;
	}
	ERR_FAIL_COND_MSG(completion_call_stack.empty(), "Trying to pop empty completion call stack");
	completion_call_stack.pop_back();
}

void GDScriptParser::set_last_completion_call_arg(int p_argument) {
	if (!for_completion || passed_cursor) {
		return;
	}
	ERR_FAIL_COND_MSG(completion_call_stack.empty(), "Trying to set argument on empty completion call stack");
	completion_call_stack.back()->get().argument = p_argument;
}

Error GDScriptParser::parse(const String &p_source_code, const String &p_script_path, bool p_for_completion) {
	clear();

	String source = p_source_code;
	int cursor_line = -1;
	int cursor_column = -1;
	for_completion = p_for_completion;

	int tab_size = 4;
#ifdef TOOLS_ENABLED
	if (EditorSettings::get_singleton()) {
		tab_size = EditorSettings::get_singleton()->get_setting("text_editor/indent/size");
	}
#endif // TOOLS_ENABLED

	if (p_for_completion) {
		// Remove cursor sentinel char.
		const Vector<String> lines = p_source_code.split("\n");
		cursor_line = 1;
		cursor_column = 1;
		for (int i = 0; i < lines.size(); i++) {
			bool found = false;
			const String &line = lines[i];
			for (int j = 0; j < line.size(); j++) {
				if (line[j] == char32_t(0xFFFF)) {
					found = true;
					break;
				} else if (line[j] == '\t') {
					cursor_column += tab_size - 1;
				}
				cursor_column++;
			}
			if (found) {
				break;
			}
			cursor_line++;
			cursor_column = 1;
		}

		source = source.replace_first(String::chr(0xFFFF), String());
	}

	tokenizer.set_source_code(source);
	tokenizer.set_cursor_position(cursor_line, cursor_column);
	script_path = p_script_path;
	current = tokenizer.scan();
	// Avoid error as the first token.
	while (current.type == GDScriptTokenizer::Token::ERROR) {
		push_error(current.literal);
		current = tokenizer.scan();
	}

	push_multiline(false); // Keep one for the whole parsing.
	parse_program();
	pop_multiline();

#ifdef DEBUG_ENABLED
	if (multiline_stack.size() > 0) {
		ERR_PRINT("Parser bug: Imbalanced multiline stack.");
	}
#endif

	if (errors.empty()) {
		return OK;
	} else {
		return ERR_PARSE_ERROR;
	}
}

GDScriptTokenizer::Token GDScriptParser::advance() {
	if (current.type == GDScriptTokenizer::Token::TK_EOF) {
		ERR_FAIL_COND_V_MSG(current.type == GDScriptTokenizer::Token::TK_EOF, current, "GDScript parser bug: Trying to advance past the end of stream.");
	}
	if (for_completion && !completion_call_stack.empty()) {
		if (completion_call.call == nullptr && tokenizer.is_past_cursor()) {
			completion_call = completion_call_stack.back()->get();
			passed_cursor = true;
		}
	}
	previous = current;
	current = tokenizer.scan();
	while (current.type == GDScriptTokenizer::Token::ERROR) {
		push_error(current.literal);
		current = tokenizer.scan();
	}
	return previous;
}

bool GDScriptParser::match(GDScriptTokenizer::Token::Type p_token_type) {
	if (!check(p_token_type)) {
		return false;
	}
	advance();
	return true;
}

bool GDScriptParser::check(GDScriptTokenizer::Token::Type p_token_type) {
	if (p_token_type == GDScriptTokenizer::Token::IDENTIFIER) {
		return current.is_identifier();
	}
	return current.type == p_token_type;
}

bool GDScriptParser::consume(GDScriptTokenizer::Token::Type p_token_type, const String &p_error_message) {
	if (match(p_token_type)) {
		return true;
	}
	push_error(p_error_message);
	return false;
}

bool GDScriptParser::is_at_end() {
	return check(GDScriptTokenizer::Token::TK_EOF);
}

void GDScriptParser::synchronize() {
	panic_mode = false;
	while (!is_at_end()) {
		if (previous.type == GDScriptTokenizer::Token::NEWLINE || previous.type == GDScriptTokenizer::Token::SEMICOLON) {
			return;
		}

		switch (current.type) {
			case GDScriptTokenizer::Token::CLASS:
			case GDScriptTokenizer::Token::FUNC:
			case GDScriptTokenizer::Token::STATIC:
			case GDScriptTokenizer::Token::VAR:
			case GDScriptTokenizer::Token::CONST:
			case GDScriptTokenizer::Token::SIGNAL:
			//case GDScriptTokenizer::Token::IF: // Can also be inside expressions.
			case GDScriptTokenizer::Token::FOR:
			case GDScriptTokenizer::Token::WHILE:
			case GDScriptTokenizer::Token::MATCH:
			case GDScriptTokenizer::Token::RETURN:
			case GDScriptTokenizer::Token::ANNOTATION:
				return;
			default:
				// Do nothing.
				break;
		}

		advance();
	}
}

void GDScriptParser::push_multiline(bool p_state) {
	multiline_stack.push_back(p_state);
	tokenizer.set_multiline_mode(p_state);
	if (p_state) {
		// Consume potential whitespace tokens already waiting in line.
		while (current.type == GDScriptTokenizer::Token::NEWLINE || current.type == GDScriptTokenizer::Token::INDENT || current.type == GDScriptTokenizer::Token::DEDENT) {
			current = tokenizer.scan(); // Don't call advance() here, as we don't want to change the previous token.
		}
	}
}

void GDScriptParser::pop_multiline() {
	ERR_FAIL_COND_MSG(multiline_stack.size() == 0, "Parser bug: trying to pop from multiline stack without available value.");
	multiline_stack.pop_back();
	tokenizer.set_multiline_mode(multiline_stack.size() > 0 ? multiline_stack.back()->get() : false);
}

bool GDScriptParser::is_statement_end() {
	return check(GDScriptTokenizer::Token::NEWLINE) || check(GDScriptTokenizer::Token::SEMICOLON) || check(GDScriptTokenizer::Token::TK_EOF);
}

void GDScriptParser::end_statement(const String &p_context) {
	bool found = false;
	while (is_statement_end() && !is_at_end()) {
		// Remove sequential newlines/semicolons.
		found = true;
		advance();
	}
	if (!found && !is_at_end()) {
		push_error(vformat(R"(Expected end of statement after %s, found "%s" instead.)", p_context, current.get_name()));
	}
}

void GDScriptParser::parse_program() {
	head = alloc_node<ClassNode>();
	current_class = head;

	if (match(GDScriptTokenizer::Token::ANNOTATION)) {
		// Check for @tool annotation.
		AnnotationNode *annotation = parse_annotation(AnnotationInfo::SCRIPT | AnnotationInfo::CLASS_LEVEL);
		if (annotation != nullptr) {
			if (annotation->name == "@tool") {
				// TODO: don't allow @tool anywhere else. (Should all script annotations be the first thing?).
				_is_tool = true;
				if (previous.type != GDScriptTokenizer::Token::NEWLINE) {
					push_error(R"(Expected newline after "@tool" annotation.)");
				}
				// @tool annotation has no specific target.
				annotation->apply(this, nullptr);
			} else {
				annotation_stack.push_back(annotation);
			}
		}
	}

	for (bool should_break = false; !should_break;) {
		// Order here doesn't matter, but there should be only one of each at most.
		switch (current.type) {
			case GDScriptTokenizer::Token::CLASS_NAME:
				if (!annotation_stack.empty()) {
					push_error(R"("class_name" should be used before annotations.)");
				}
				advance();
				if (head->identifier != nullptr) {
					push_error(R"("class_name" can only be used once.)");
				} else {
					parse_class_name();
				}
				break;
			case GDScriptTokenizer::Token::EXTENDS:
				if (!annotation_stack.empty()) {
					push_error(R"("extends" should be used before annotations.)");
				}
				advance();
				if (head->extends_used) {
					push_error(R"("extends" can only be used once.)");
				} else {
					parse_extends();
					end_statement("superclass");
				}
				break;
			default:
				should_break = true;
				break;
		}

		if (panic_mode) {
			synchronize();
		}
	}

	if (match(GDScriptTokenizer::Token::ANNOTATION)) {
		// Check for @icon annotation.
		AnnotationNode *annotation = parse_annotation(AnnotationInfo::SCRIPT | AnnotationInfo::CLASS_LEVEL);
		if (annotation != nullptr) {
			if (annotation->name == "@icon") {
				if (previous.type != GDScriptTokenizer::Token::NEWLINE) {
					push_error(R"(Expected newline after "@icon" annotation.)");
				}
				annotation->apply(this, head);
			} else {
				annotation_stack.push_back(annotation);
			}
		}
	}

	parse_class_body();

	if (!check(GDScriptTokenizer::Token::TK_EOF)) {
		push_error("Expected end of file.");
	}

	clear_unused_annotations();
}

GDScriptParser::ClassNode *GDScriptParser::parse_class() {
	ClassNode *n_class = alloc_node<ClassNode>();

	ClassNode *previous_class = current_class;
	current_class = n_class;
	n_class->outer = previous_class;

	if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected identifier for the class name after "class".)")) {
		n_class->identifier = parse_identifier();
	}

	if (match(GDScriptTokenizer::Token::EXTENDS)) {
		parse_extends();
	}

	consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after class declaration.)");
	consume(GDScriptTokenizer::Token::NEWLINE, R"(Expected newline after class declaration.)");

	if (!consume(GDScriptTokenizer::Token::INDENT, R"(Expected indented block after class declaration.)")) {
		current_class = previous_class;
		return n_class;
	}

	if (match(GDScriptTokenizer::Token::EXTENDS)) {
		if (n_class->extends_used) {
			push_error(R"(Cannot use "extends" more than once in the same class.)");
		}
		parse_extends();
		end_statement("superclass");
	}

	parse_class_body();

	consume(GDScriptTokenizer::Token::DEDENT, R"(Missing unindent at the end of the class body.)");

	current_class = previous_class;
	return n_class;
}

void GDScriptParser::parse_class_name() {
	if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected identifier for the global class name after "class_name".)")) {
		current_class->identifier = parse_identifier();
	}

	// TODO: Move this to annotation
	if (match(GDScriptTokenizer::Token::COMMA)) {
		// Icon path.
		if (consume(GDScriptTokenizer::Token::LITERAL, R"(Expected class icon path string after ",".)")) {
			if (previous.literal.get_type() != Variant::STRING) {
				push_error(vformat(R"(Only strings can be used for the class icon path, found "%s" instead.)", Variant::get_type_name(previous.literal.get_type())));
			}
			current_class->icon_path = previous.literal;
		}
	}

	if (match(GDScriptTokenizer::Token::EXTENDS)) {
		// Allow extends on the same line.
		parse_extends();
		end_statement("superclass");
	} else {
		end_statement("class_name statement");
	}
}

void GDScriptParser::parse_extends() {
	current_class->extends_used = true;

	int chain_index = 0;

	if (match(GDScriptTokenizer::Token::LITERAL)) {
		if (previous.literal.get_type() != Variant::STRING) {
			push_error(vformat(R"(Only strings or identifiers can be used after "extends", found "%s" instead.)", Variant::get_type_name(previous.literal.get_type())));
		}
		current_class->extends_path = previous.literal;

		if (!match(GDScriptTokenizer::Token::PERIOD)) {
			return;
		}
	}

	make_completion_context(COMPLETION_INHERIT_TYPE, current_class, chain_index++);

	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected superclass name after "extends".)")) {
		return;
	}
	current_class->extends.push_back(previous.literal);

	while (match(GDScriptTokenizer::Token::PERIOD)) {
		make_completion_context(COMPLETION_INHERIT_TYPE, current_class, chain_index++);
		if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected superclass name after ".".)")) {
			return;
		}
		current_class->extends.push_back(previous.literal);
	}
}

template <class T>
void GDScriptParser::parse_class_member(T *(GDScriptParser::*p_parse_function)(), AnnotationInfo::TargetKind p_target, const String &p_member_kind) {
	advance();
	T *member = (this->*p_parse_function)();
	if (member == nullptr) {
		return;
	}
	// Consume annotations.
	while (!annotation_stack.empty()) {
		AnnotationNode *last_annotation = annotation_stack.back()->get();
		if (last_annotation->applies_to(p_target)) {
			last_annotation->apply(this, member);
			member->annotations.push_front(last_annotation);
			annotation_stack.pop_back();
		} else {
			push_error(vformat(R"(Annotation "%s" cannot be applied to a %s.)", last_annotation->name, p_member_kind));
			clear_unused_annotations();
			return;
		}
	}
	if (member->identifier != nullptr) {
		// Enums may be unnamed.
		// TODO: Consider names in outer scope too, for constants and classes (and static functions?)
		if (current_class->members_indices.has(member->identifier->name)) {
			push_error(vformat(R"(%s "%s" has the same name as a previously declared %s.)", p_member_kind.capitalize(), member->identifier->name, current_class->get_member(member->identifier->name).get_type_name()), member->identifier);
		} else {
			current_class->add_member(member);
		}
	}
}

void GDScriptParser::parse_class_body() {
	bool class_end = false;
	while (!class_end && !is_at_end()) {
		switch (current.type) {
			case GDScriptTokenizer::Token::VAR:
				parse_class_member(&GDScriptParser::parse_variable, AnnotationInfo::VARIABLE, "variable");
				break;
			case GDScriptTokenizer::Token::CONST:
				parse_class_member(&GDScriptParser::parse_constant, AnnotationInfo::CONSTANT, "constant");
				break;
			case GDScriptTokenizer::Token::SIGNAL:
				parse_class_member(&GDScriptParser::parse_signal, AnnotationInfo::SIGNAL, "signal");
				break;
			case GDScriptTokenizer::Token::STATIC:
			case GDScriptTokenizer::Token::FUNC:
				parse_class_member(&GDScriptParser::parse_function, AnnotationInfo::FUNCTION, "function");
				break;
			case GDScriptTokenizer::Token::CLASS:
				parse_class_member(&GDScriptParser::parse_class, AnnotationInfo::CLASS, "class");
				break;
			case GDScriptTokenizer::Token::ENUM:
				parse_class_member(&GDScriptParser::parse_enum, AnnotationInfo::NONE, "enum");
				break;
			case GDScriptTokenizer::Token::ANNOTATION: {
				advance();
				AnnotationNode *annotation = parse_annotation(AnnotationInfo::CLASS_LEVEL);
				if (annotation != nullptr) {
					annotation_stack.push_back(annotation);
				}
				break;
			}
			case GDScriptTokenizer::Token::PASS:
				advance();
				end_statement(R"("pass")");
				break;
			case GDScriptTokenizer::Token::DEDENT:
				class_end = true;
				break;
			default:
				push_error(vformat(R"(Unexpected "%s" in class body.)", current.get_name()));
				advance();
				break;
		}
		if (panic_mode) {
			synchronize();
		}
	}
}

GDScriptParser::VariableNode *GDScriptParser::parse_variable() {
	return parse_variable(true);
}

GDScriptParser::VariableNode *GDScriptParser::parse_variable(bool p_allow_property) {
	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected variable name after "var".)")) {
		return nullptr;
	}

	VariableNode *variable = alloc_node<VariableNode>();
	variable->identifier = parse_identifier();

	if (match(GDScriptTokenizer::Token::COLON)) {
		if (check(GDScriptTokenizer::Token::NEWLINE)) {
			if (p_allow_property) {
				advance();

				return parse_property(variable, true);
			} else {
				push_error(R"(Expected type after ":")");
				return nullptr;
			}
		} else if (check((GDScriptTokenizer::Token::EQUAL))) {
			// Infer type.
			variable->infer_datatype = true;
		} else {
			if (p_allow_property) {
				make_completion_context(COMPLETION_PROPERTY_DECLARATION_OR_TYPE, variable);
				if (check(GDScriptTokenizer::Token::IDENTIFIER)) {
					// Check if get or set.
					if (current.get_identifier() == "get" || current.get_identifier() == "set") {
						return parse_property(variable, false);
					}
				}
			}

			// Parse type.
			variable->datatype_specifier = parse_type();
		}
	}

	if (match(GDScriptTokenizer::Token::EQUAL)) {
		// Initializer.
		variable->initializer = parse_expression(false);
		variable->assignments++;
	}

	if (p_allow_property && match(GDScriptTokenizer::Token::COLON)) {
		if (match(GDScriptTokenizer::Token::NEWLINE)) {
			return parse_property(variable, true);
		} else {
			return parse_property(variable, false);
		}
	}

	end_statement("variable declaration");

	variable->export_info.name = variable->identifier->name;

	return variable;
}

GDScriptParser::VariableNode *GDScriptParser::parse_property(VariableNode *p_variable, bool p_need_indent) {
	if (p_need_indent) {
		if (!consume(GDScriptTokenizer::Token::INDENT, R"(Expected indented block for property after ":".)")) {
			return nullptr;
		}
	}

	VariableNode *property = p_variable;

	make_completion_context(COMPLETION_PROPERTY_DECLARATION, property);

	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected "get" or "set" for property declaration.)")) {
		return nullptr;
	}

	IdentifierNode *function = parse_identifier();

	if (check(GDScriptTokenizer::Token::EQUAL)) {
		p_variable->property = VariableNode::PROP_SETGET;
	} else {
		p_variable->property = VariableNode::PROP_INLINE;
		if (!p_need_indent) {
			push_error("Property with inline code must go to an indented block.");
		}
	}

	bool getter_used = false;
	bool setter_used = false;

	// Run with a loop because order doesn't matter.
	for (int i = 0; i < 2; i++) {
		if (function->name == "set") {
			if (setter_used) {
				push_error(R"(Properties can only have one setter.)");
			} else {
				parse_property_setter(property);
				setter_used = true;
			}
		} else if (function->name == "get") {
			if (getter_used) {
				push_error(R"(Properties can only have one getter.)");
			} else {
				parse_property_getter(property);
				getter_used = true;
			}
		} else {
			// TODO: Update message to only have the missing one if it's the case.
			push_error(R"(Expected "get" or "set" for property declaration.)");
		}

		if (i == 0 && p_variable->property == VariableNode::PROP_SETGET) {
			if (match(GDScriptTokenizer::Token::COMMA)) {
				// Consume potential newline.
				if (match(GDScriptTokenizer::Token::NEWLINE)) {
					if (!p_need_indent) {
						push_error(R"(Inline setter/getter setting cannot span across multiple lines (use "\\"" if needed).)");
					}
				}
			} else {
				break;
			}
		}

		if (!match(GDScriptTokenizer::Token::IDENTIFIER)) {
			break;
		}
		function = parse_identifier();
	}

	if (p_variable->property == VariableNode::PROP_SETGET) {
		end_statement("property declaration");
	}

	if (p_need_indent) {
		consume(GDScriptTokenizer::Token::DEDENT, R"(Expected end of indented block for property.)");
	}
	return property;
}

void GDScriptParser::parse_property_setter(VariableNode *p_variable) {
	switch (p_variable->property) {
		case VariableNode::PROP_INLINE:
			consume(GDScriptTokenizer::Token::PARENTHESIS_OPEN, R"(Expected "(" after "set".)");
			if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected parameter name after "(".)")) {
				p_variable->setter_parameter = parse_identifier();
			}
			consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected ")" after parameter name.)*");
			consume(GDScriptTokenizer::Token::COLON, R"*(Expected ":" after ")".)*");

			p_variable->setter = parse_suite("setter definition");
			break;

		case VariableNode::PROP_SETGET:
			consume(GDScriptTokenizer::Token::EQUAL, R"(Expected "=" after "set")");
			make_completion_context(COMPLETION_PROPERTY_METHOD, p_variable);
			if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected setter function name after "=".)")) {
				p_variable->setter_pointer = parse_identifier();
			}
			break;
		case VariableNode::PROP_NONE:
			break; // Unreachable.
	}
}

void GDScriptParser::parse_property_getter(VariableNode *p_variable) {
	switch (p_variable->property) {
		case VariableNode::PROP_INLINE:
			consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after "get".)");

			p_variable->getter = parse_suite("getter definition");
			break;
		case VariableNode::PROP_SETGET:
			consume(GDScriptTokenizer::Token::EQUAL, R"(Expected "=" after "get")");
			make_completion_context(COMPLETION_PROPERTY_METHOD, p_variable);
			if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected getter function name after "=".)")) {
				p_variable->getter_pointer = parse_identifier();
			}
			break;
		case VariableNode::PROP_NONE:
			break; // Unreachable.
	}
}

GDScriptParser::ConstantNode *GDScriptParser::parse_constant() {
	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected constant name after "const".)")) {
		return nullptr;
	}

	ConstantNode *constant = alloc_node<ConstantNode>();
	constant->identifier = parse_identifier();

	if (match(GDScriptTokenizer::Token::COLON)) {
		if (check((GDScriptTokenizer::Token::EQUAL))) {
			// Infer type.
			constant->infer_datatype = true;
		} else {
			// Parse type.
			constant->datatype_specifier = parse_type();
		}
	}

	if (consume(GDScriptTokenizer::Token::EQUAL, R"(Expected initializer after constant name.)")) {
		// Initializer.
		constant->initializer = parse_expression(false);

		if (constant->initializer == nullptr) {
			push_error(R"(Expected initializer expression for constant.)");
			return nullptr;
		}
	}

	end_statement("constant declaration");

	return constant;
}

GDScriptParser::ParameterNode *GDScriptParser::parse_parameter() {
	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected parameter name.)")) {
		return nullptr;
	}

	ParameterNode *parameter = alloc_node<ParameterNode>();
	parameter->identifier = parse_identifier();

	if (match(GDScriptTokenizer::Token::COLON)) {
		if (check((GDScriptTokenizer::Token::EQUAL))) {
			// Infer type.
			parameter->infer_datatype = true;
		} else {
			// Parse type.
			make_completion_context(COMPLETION_TYPE_NAME, parameter);
			parameter->datatype_specifier = parse_type();
		}
	}

	if (match(GDScriptTokenizer::Token::EQUAL)) {
		// Default value.
		parameter->default_value = parse_expression(false);
	}

	return parameter;
}

GDScriptParser::SignalNode *GDScriptParser::parse_signal() {
	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected signal name after "signal".)")) {
		return nullptr;
	}

	SignalNode *signal = alloc_node<SignalNode>();
	signal->identifier = parse_identifier();

	if (match(GDScriptTokenizer::Token::PARENTHESIS_OPEN)) {
		do {
			if (check(GDScriptTokenizer::Token::PARENTHESIS_CLOSE)) {
				// Allow for trailing comma.
				break;
			}

			ParameterNode *parameter = parse_parameter();
			if (parameter == nullptr) {
				push_error("Expected signal parameter name.");
				break;
			}
			if (parameter->default_value != nullptr) {
				push_error(R"(Signal parameters cannot have a default value.)");
			}
			if (signal->parameters_indices.has(parameter->identifier->name)) {
				push_error(vformat(R"(Parameter with name "%s" was already declared for this signal.)", parameter->identifier->name));
			} else {
				signal->parameters_indices[parameter->identifier->name] = signal->parameters.size();
				signal->parameters.push_back(parameter);
			}
		} while (match(GDScriptTokenizer::Token::COMMA) && !is_at_end());

		consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected closing ")" after signal parameters.)*");
	}

	end_statement("signal declaration");

	return signal;
}

GDScriptParser::EnumNode *GDScriptParser::parse_enum() {
	EnumNode *enum_node = alloc_node<EnumNode>();
	bool named = false;

	if (check(GDScriptTokenizer::Token::IDENTIFIER)) {
		advance();
		enum_node->identifier = parse_identifier();
		named = true;
	}

	push_multiline(true);
	consume(GDScriptTokenizer::Token::BRACE_OPEN, vformat(R"(Expected "{" after %s.)", named ? "enum name" : R"("enum")"));

	HashMap<StringName, int> elements;

	do {
		if (check(GDScriptTokenizer::Token::BRACE_CLOSE)) {
			break; // Allow trailing comma.
		}
		if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected identifier for enum key.)")) {
			EnumNode::Value item;
			item.identifier = parse_identifier();
			item.parent_enum = enum_node;
			item.line = previous.start_line;
			item.leftmost_column = previous.leftmost_column;

			if (elements.has(item.identifier->name)) {
				push_error(vformat(R"(Name "%s" was already in this enum (at line %d).)", item.identifier->name, elements[item.identifier->name]), item.identifier);
			} else if (!named) {
				// TODO: Abstract this recursive member check.
				ClassNode *parent = current_class;
				while (parent != nullptr) {
					if (parent->members_indices.has(item.identifier->name)) {
						push_error(vformat(R"(Name "%s" is already used as a class %s.)", item.identifier->name, parent->get_member(item.identifier->name).get_type_name()));
						break;
					}
					parent = parent->outer;
				}
			}

			elements[item.identifier->name] = item.line;

			if (match(GDScriptTokenizer::Token::EQUAL)) {
				ExpressionNode *value = parse_expression(false);
				if (value == nullptr) {
					push_error(R"(Expected expression value after "=".)");
				}
				item.custom_value = value;
			}
			item.rightmost_column = previous.rightmost_column;

			item.index = enum_node->values.size();
			enum_node->values.push_back(item);
			if (!named) {
				// Add as member of current class.
				current_class->add_member(item);
			}
		}
	} while (match(GDScriptTokenizer::Token::COMMA));

	pop_multiline();
	consume(GDScriptTokenizer::Token::BRACE_CLOSE, R"(Expected closing "}" for enum.)");

	end_statement("enum");

	return enum_node;
}

GDScriptParser::FunctionNode *GDScriptParser::parse_function() {
	bool _static = false;
	if (previous.type == GDScriptTokenizer::Token::STATIC) {
		// TODO: Improve message if user uses "static" with "var" or "const"
		if (!consume(GDScriptTokenizer::Token::FUNC, R"(Expected "func" after "static".)")) {
			return nullptr;
		}
		_static = true;
	}

	FunctionNode *function = alloc_node<FunctionNode>();
	make_completion_context(COMPLETION_OVERRIDE_METHOD, function);

	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected function name after "func".)")) {
		return nullptr;
	}

	FunctionNode *previous_function = current_function;
	current_function = function;

	function->identifier = parse_identifier();
	function->is_static = _static;

	push_multiline(true);
	consume(GDScriptTokenizer::Token::PARENTHESIS_OPEN, R"(Expected opening "(" after function name.)");

	SuiteNode *body = alloc_node<SuiteNode>();
	SuiteNode *previous_suite = current_suite;
	current_suite = body;

	if (!check(GDScriptTokenizer::Token::PARENTHESIS_CLOSE) && !is_at_end()) {
		bool default_used = false;
		do {
			if (check(GDScriptTokenizer::Token::PARENTHESIS_CLOSE)) {
				// Allow for trailing comma.
				break;
			}
			ParameterNode *parameter = parse_parameter();
			if (parameter == nullptr) {
				break;
			}
			if (parameter->default_value != nullptr) {
				default_used = true;
			} else {
				if (default_used) {
					push_error("Cannot have a mandatory parameters after optional parameters.");
					continue;
				}
			}
			if (function->parameters_indices.has(parameter->identifier->name)) {
				push_error(vformat(R"(Parameter with name "%s" was already declared for this function.)", parameter->identifier->name));
			} else {
				function->parameters_indices[parameter->identifier->name] = function->parameters.size();
				function->parameters.push_back(parameter);
				body->add_local(parameter);
			}
		} while (match(GDScriptTokenizer::Token::COMMA));
	}

	pop_multiline();
	consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected closing ")" after function parameters.)*");

	if (match(GDScriptTokenizer::Token::FORWARD_ARROW)) {
		make_completion_context(COMPLETION_TYPE_NAME_OR_VOID, function);
		function->return_type = parse_type(true);
		if (function->return_type == nullptr) {
			push_error(R"(Expected return type or "void" after "->".)");
		}
	}

	// TODO: Improve token consumption so it synchronizes to a statement boundary. This way we can get into the function body with unrecognized tokens.
	consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after function declaration.)");

	current_suite = previous_suite;
	function->body = parse_suite("function declaration", body);

	current_function = previous_function;
	return function;
}

GDScriptParser::AnnotationNode *GDScriptParser::parse_annotation(uint32_t p_valid_targets) {
	AnnotationNode *annotation = alloc_node<AnnotationNode>();

	annotation->name = previous.literal;

	make_completion_context(COMPLETION_ANNOTATION, annotation);

	bool valid = true;

	if (!valid_annotations.has(annotation->name)) {
		push_error(vformat(R"(Unrecognized annotation: "%s".)", annotation->name));
		valid = false;
	}

	annotation->info = &valid_annotations[annotation->name];

	if (!annotation->applies_to(p_valid_targets)) {
		push_error(vformat(R"(Annotation "%s" is not allowed in this level.)", annotation->name));
		valid = false;
	}

	if (match(GDScriptTokenizer::Token::PARENTHESIS_OPEN)) {
		// Arguments.
		push_completion_call(annotation);
		make_completion_context(COMPLETION_ANNOTATION_ARGUMENTS, annotation, 0, true);
		if (!check(GDScriptTokenizer::Token::PARENTHESIS_CLOSE) && !is_at_end()) {
			int argument_index = 0;
			do {
				make_completion_context(COMPLETION_ANNOTATION_ARGUMENTS, annotation, argument_index, true);
				set_last_completion_call_arg(argument_index++);
				ExpressionNode *argument = parse_expression(false);
				if (argument == nullptr) {
					valid = false;
					continue;
				}
				annotation->arguments.push_back(argument);
			} while (match(GDScriptTokenizer::Token::COMMA));

			consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected ")" after annotation arguments.)*");
		}
		pop_completion_call();
	}

	match(GDScriptTokenizer::Token::NEWLINE); // Newline after annotation is optional.

	if (valid) {
		valid = validate_annotation_arguments(annotation);
	}

	return valid ? annotation : nullptr;
}

void GDScriptParser::clear_unused_annotations() {
	for (const List<AnnotationNode *>::Element *E = annotation_stack.front(); E != nullptr; E = E->next()) {
		AnnotationNode *annotation = E->get();
		push_error(vformat(R"(Annotation "%s" does not precedes a valid target, so it will have no effect.)", annotation->name), annotation);
	}

	annotation_stack.clear();
}

bool GDScriptParser::register_annotation(const MethodInfo &p_info, uint32_t p_target_kinds, AnnotationAction p_apply, int p_optional_arguments, bool p_is_vararg) {
	ERR_FAIL_COND_V_MSG(valid_annotations.has(p_info.name), false, vformat(R"(Annotation "%s" already registered.)", p_info.name));

	AnnotationInfo new_annotation;
	new_annotation.info = p_info;
	new_annotation.info.default_arguments.resize(p_optional_arguments);
	if (p_is_vararg) {
		new_annotation.info.flags |= METHOD_FLAG_VARARG;
	}
	new_annotation.apply = p_apply;
	new_annotation.target_kind = p_target_kinds;

	valid_annotations[p_info.name] = new_annotation;
	return true;
}

GDScriptParser::SuiteNode *GDScriptParser::parse_suite(const String &p_context, SuiteNode *p_suite) {
	SuiteNode *suite = p_suite != nullptr ? p_suite : alloc_node<SuiteNode>();
	suite->parent_block = current_suite;
	current_suite = suite;

	bool multiline = false;

	if (check(GDScriptTokenizer::Token::NEWLINE)) {
		multiline = true;
	}

	if (multiline) {
		consume(GDScriptTokenizer::Token::NEWLINE, vformat(R"(Expected newline after %s.)", p_context));

		if (!consume(GDScriptTokenizer::Token::INDENT, vformat(R"(Expected indented block after %s.)", p_context))) {
			current_suite = suite->parent_block;
			return suite;
		}
	}

	do {
		Node *statement = parse_statement();
		if (statement == nullptr) {
			continue;
		}
		suite->statements.push_back(statement);

		// Register locals.
		switch (statement->type) {
			case Node::VARIABLE: {
				VariableNode *variable = static_cast<VariableNode *>(statement);
				const SuiteNode::Local &local = current_suite->get_local(variable->identifier->name);
				if (local.type != SuiteNode::Local::UNDEFINED) {
					push_error(vformat(R"(There is already a %s named "%s" declared in this scope.)", local.get_name(), variable->identifier->name));
				}
				current_suite->add_local(variable);
				break;
			}
			case Node::CONSTANT: {
				ConstantNode *constant = static_cast<ConstantNode *>(statement);
				const SuiteNode::Local &local = current_suite->get_local(constant->identifier->name);
				if (local.type != SuiteNode::Local::UNDEFINED) {
					String name;
					if (local.type == SuiteNode::Local::CONSTANT) {
						name = "constant";
					} else {
						name = "variable";
					}
					push_error(vformat(R"(There is already a %s named "%s" declared in this scope.)", name, constant->identifier->name));
				}
				current_suite->add_local(constant);
				break;
			}
			default:
				break;
		}

	} while (multiline && !check(GDScriptTokenizer::Token::DEDENT) && !is_at_end());

	if (multiline) {
		consume(GDScriptTokenizer::Token::DEDENT, vformat(R"(Missing unindent at the end of %s.)", p_context));
	}

	current_suite = suite->parent_block;
	return suite;
}

GDScriptParser::Node *GDScriptParser::parse_statement() {
	Node *result = nullptr;
#ifdef DEBUG_ENABLED
	bool unreachable = current_suite->has_return && !current_suite->has_unreachable_code;
#endif

	switch (current.type) {
		case GDScriptTokenizer::Token::PASS:
			advance();
			result = alloc_node<PassNode>();
			end_statement(R"("pass")");
			break;
		case GDScriptTokenizer::Token::VAR:
			advance();
			result = parse_variable();
			break;
		case GDScriptTokenizer::Token::CONST:
			advance();
			result = parse_constant();
			break;
		case GDScriptTokenizer::Token::IF:
			advance();
			result = parse_if();
			break;
		case GDScriptTokenizer::Token::FOR:
			advance();
			result = parse_for();
			break;
		case GDScriptTokenizer::Token::WHILE:
			advance();
			result = parse_while();
			break;
		case GDScriptTokenizer::Token::MATCH:
			advance();
			result = parse_match();
			break;
		case GDScriptTokenizer::Token::BREAK:
			advance();
			result = parse_break();
			break;
		case GDScriptTokenizer::Token::CONTINUE:
			advance();
			result = parse_continue();
			break;
		case GDScriptTokenizer::Token::RETURN: {
			advance();
			ReturnNode *n_return = alloc_node<ReturnNode>();
			if (!is_statement_end()) {
				if (current_function && current_function->identifier->name == GDScriptLanguage::get_singleton()->strings._init) {
					push_error(R"(Constructor cannot return a value.)");
				}
				n_return->return_value = parse_expression(false);
			}
			result = n_return;

			current_suite->has_return = true;

			end_statement("return statement");
			break;
		}
		case GDScriptTokenizer::Token::BREAKPOINT:
			advance();
			result = alloc_node<BreakpointNode>();
			end_statement(R"("breakpoint")");
			break;
		case GDScriptTokenizer::Token::ASSERT:
			advance();
			result = parse_assert();
			break;
		case GDScriptTokenizer::Token::ANNOTATION: {
			advance();
			AnnotationNode *annotation = parse_annotation(AnnotationInfo::STATEMENT);
			if (annotation != nullptr) {
				annotation_stack.push_back(annotation);
			}
			break;
		}
		default: {
			// Expression statement.
			ExpressionNode *expression = parse_expression(true); // Allow assignment here.
			if (expression == nullptr) {
				push_error(vformat(R"(Expected statement, found "%s" instead.)", previous.get_name()));
			}
			end_statement("expression");
			result = expression;

#ifdef DEBUG_ENABLED
			if (expression != nullptr) {
				switch (expression->type) {
					case Node::CALL:
					case Node::ASSIGNMENT:
					case Node::AWAIT:
						// Fine.
						break;
					default:
						push_warning(expression, GDScriptWarning::STANDALONE_EXPRESSION);
				}
			}
#endif
			break;
		}
	}

#ifdef DEBUG_ENABLED
	if (unreachable) {
		current_suite->has_unreachable_code = true;
		push_warning(result, GDScriptWarning::UNREACHABLE_CODE, current_function->identifier->name);
	}
#endif

	if (panic_mode) {
		synchronize();
	}

	return result;
}

GDScriptParser::AssertNode *GDScriptParser::parse_assert() {
	// TODO: Add assert message.
	AssertNode *assert = alloc_node<AssertNode>();

	consume(GDScriptTokenizer::Token::PARENTHESIS_OPEN, R"(Expected "(" after "assert".)");
	assert->condition = parse_expression(false);
	if (assert->condition == nullptr) {
		push_error("Expected expression to assert.");
		return nullptr;
	}

	if (match(GDScriptTokenizer::Token::COMMA)) {
		// Error message.
		if (consume(GDScriptTokenizer::Token::LITERAL, R"(Expected error message for assert after ",".)")) {
			assert->message = parse_literal();
			if (assert->message->value.get_type() != Variant::STRING) {
				push_error(R"(Expected string for assert error message.)");
			}
		} else {
			return nullptr;
		}
	}

	consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected ")" after assert expression.)*");

	end_statement(R"("assert")");

	return assert;
}

GDScriptParser::BreakNode *GDScriptParser::parse_break() {
	if (!can_break) {
		push_error(R"(Cannot use "break" outside of a loop.)");
	}
	end_statement(R"("break")");
	return alloc_node<BreakNode>();
}

GDScriptParser::ContinueNode *GDScriptParser::parse_continue() {
	if (!can_continue) {
		push_error(R"(Cannot use "continue" outside of a loop or pattern matching block.)");
	}
	current_suite->has_continue = true;
	end_statement(R"("continue")");
	ContinueNode *cont = alloc_node<ContinueNode>();
	cont->is_for_match = is_continue_match;
	return cont;
}

GDScriptParser::ForNode *GDScriptParser::parse_for() {
	ForNode *n_for = alloc_node<ForNode>();

	if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected loop variable name after "for".)")) {
		n_for->variable = parse_identifier();
	}

	consume(GDScriptTokenizer::Token::IN, R"(Expected "in" after "for" variable name.)");

	n_for->list = parse_expression(false);

	consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after "for" condition.)");

	// Save break/continue state.
	bool could_break = can_break;
	bool could_continue = can_continue;
	bool was_continue_match = is_continue_match;

	// Allow break/continue.
	can_break = true;
	can_continue = true;
	is_continue_match = false;

	SuiteNode *suite = alloc_node<SuiteNode>();
	if (n_for->variable) {
		suite->add_local(SuiteNode::Local(n_for->variable));
	}
	suite->parent_for = n_for;

	n_for->loop = parse_suite(R"("for" block)", suite);

	// Reset break/continue state.
	can_break = could_break;
	can_continue = could_continue;
	is_continue_match = was_continue_match;

	return n_for;
}

GDScriptParser::IfNode *GDScriptParser::parse_if(const String &p_token) {
	IfNode *n_if = alloc_node<IfNode>();

	n_if->condition = parse_expression(false);
	if (n_if->condition == nullptr) {
		push_error(vformat(R"(Expected conditional expression after "%s".)", p_token));
	}

	consume(GDScriptTokenizer::Token::COLON, vformat(R"(Expected ":" after "%s" condition.)", p_token));

	n_if->true_block = parse_suite(vformat(R"("%s" block)", p_token));
	n_if->true_block->parent_if = n_if;

	if (n_if->true_block->has_continue) {
		current_suite->has_continue = true;
	}

	if (match(GDScriptTokenizer::Token::ELIF)) {
		IfNode *elif = parse_if("elif");

		SuiteNode *else_block = alloc_node<SuiteNode>();
		else_block->statements.push_back(elif);
		n_if->false_block = else_block;
	} else if (match(GDScriptTokenizer::Token::ELSE)) {
		consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after "else".)");
		n_if->false_block = parse_suite(R"("else" block)");
	}

	if (n_if->false_block != nullptr && n_if->false_block->has_return && n_if->true_block->has_return) {
		current_suite->has_return = true;
	}
	if (n_if->false_block != nullptr && n_if->false_block->has_continue) {
		current_suite->has_continue = true;
	}

	return n_if;
}

GDScriptParser::MatchNode *GDScriptParser::parse_match() {
	MatchNode *match = alloc_node<MatchNode>();

	match->test = parse_expression(false);
	if (match->test == nullptr) {
		push_error(R"(Expected expression to test after "match".)");
	}

	consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after "match" expression.)");
	consume(GDScriptTokenizer::Token::NEWLINE, R"(Expected a newline after "match" statement.)");

	if (!consume(GDScriptTokenizer::Token::INDENT, R"(Expected an indented block after "match" statement.)")) {
		return match;
	}

#ifdef DEBUG_ENABLED
	bool all_have_return = true;
	bool have_wildcard = false;
	bool wildcard_has_return = false;
	bool have_wildcard_without_continue = false;
#endif

	while (!check(GDScriptTokenizer::Token::DEDENT) && !is_at_end()) {
		MatchBranchNode *branch = parse_match_branch();
		if (branch == nullptr) {
			continue;
		}

#ifdef DEBUG_ENABLED
		if (have_wildcard_without_continue) {
			push_warning(branch->patterns[0], GDScriptWarning::UNREACHABLE_PATTERN);
		}

		if (branch->has_wildcard) {
			have_wildcard = true;
			if (branch->block->has_return) {
				wildcard_has_return = true;
			}
			if (!branch->block->has_continue) {
				have_wildcard_without_continue = true;
			}
		}
		if (!branch->block->has_return) {
			all_have_return = false;
		}
#endif
		match->branches.push_back(branch);
	}

	consume(GDScriptTokenizer::Token::DEDENT, R"(Expected an indented block after "match" statement.)");

#ifdef DEBUG_ENABLED
	if (wildcard_has_return || (all_have_return && have_wildcard)) {
		current_suite->has_return = true;
	}
#endif

	return match;
}

GDScriptParser::MatchBranchNode *GDScriptParser::parse_match_branch() {
	MatchBranchNode *branch = alloc_node<MatchBranchNode>();

	bool has_bind = false;

	do {
		PatternNode *pattern = parse_match_pattern();
		if (pattern == nullptr) {
			continue;
		}
		if (pattern->pattern_type == PatternNode::PT_BIND) {
			has_bind = true;
		}
		if (branch->patterns.size() > 0 && has_bind) {
			push_error(R"(Cannot use a variable bind with multiple patterns.)");
		}
		if (pattern->pattern_type == PatternNode::PT_REST) {
			push_error(R"(Rest pattern can only be used inside array and dictionary patterns.)");
		} else if (pattern->pattern_type == PatternNode::PT_BIND || pattern->pattern_type == PatternNode::PT_WILDCARD) {
			branch->has_wildcard = true;
		}
		branch->patterns.push_back(pattern);
	} while (match(GDScriptTokenizer::Token::COMMA));

	if (branch->patterns.empty()) {
		push_error(R"(No pattern found for "match" branch.)");
	}

	consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after "match" patterns.)");

	// Save continue state.
	bool could_continue = can_continue;
	bool was_continue_match = is_continue_match;
	// Allow continue for match.
	can_continue = true;
	is_continue_match = true;

	SuiteNode *suite = alloc_node<SuiteNode>();
	if (branch->patterns.size() > 0) {
		List<StringName> binds;
		branch->patterns[0]->binds.get_key_list(&binds);

		for (List<StringName>::Element *E = binds.front(); E != nullptr; E = E->next()) {
			SuiteNode::Local local(branch->patterns[0]->binds[E->get()]);
			suite->add_local(local);
		}
	}

	branch->block = parse_suite("match pattern block", suite);

	// Restore continue state.
	can_continue = could_continue;
	is_continue_match = was_continue_match;

	return branch;
}

GDScriptParser::PatternNode *GDScriptParser::parse_match_pattern(PatternNode *p_root_pattern) {
	PatternNode *pattern = alloc_node<PatternNode>();

	switch (current.type) {
		case GDScriptTokenizer::Token::LITERAL:
			advance();
			pattern->pattern_type = PatternNode::PT_LITERAL;
			pattern->literal = parse_literal();
			if (pattern->literal == nullptr) {
				// Error happened.
				return nullptr;
			}
			break;
		case GDScriptTokenizer::Token::VAR: {
			// Bind.
			advance();
			if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected bind name after "var".)")) {
				return nullptr;
			}
			pattern->pattern_type = PatternNode::PT_BIND;
			pattern->bind = parse_identifier();

			PatternNode *root_pattern = p_root_pattern == nullptr ? pattern : p_root_pattern;

			if (p_root_pattern != nullptr) {
				if (p_root_pattern->has_bind(pattern->bind->name)) {
					push_error(vformat(R"(Bind variable name "%s" was already used in this pattern.)", pattern->bind->name));
					return nullptr;
				}
			}

			if (current_suite->has_local(pattern->bind->name)) {
				push_error(vformat(R"(There's already a %s named "%s" in this scope.)", current_suite->get_local(pattern->bind->name).get_name(), pattern->bind->name));
				return nullptr;
			}

			root_pattern->binds[pattern->bind->name] = pattern->bind;

		} break;
		case GDScriptTokenizer::Token::UNDERSCORE:
			// Wildcard.
			advance();
			pattern->pattern_type = PatternNode::PT_WILDCARD;
			break;
		case GDScriptTokenizer::Token::PERIOD_PERIOD:
			// Rest.
			advance();
			pattern->pattern_type = PatternNode::PT_REST;
			break;
		case GDScriptTokenizer::Token::BRACKET_OPEN: {
			// Array.
			advance();
			pattern->pattern_type = PatternNode::PT_ARRAY;

			if (!check(GDScriptTokenizer::Token::BRACKET_CLOSE)) {
				do {
					PatternNode *sub_pattern = parse_match_pattern(p_root_pattern != nullptr ? p_root_pattern : pattern);
					if (sub_pattern == nullptr) {
						continue;
					}
					if (pattern->rest_used) {
						push_error(R"(The ".." pattern must be the last element in the pattern array.)");
					} else if (sub_pattern->pattern_type == PatternNode::PT_REST) {
						pattern->rest_used = true;
					}
					pattern->array.push_back(sub_pattern);
				} while (match(GDScriptTokenizer::Token::COMMA));
			}
			consume(GDScriptTokenizer::Token::BRACKET_CLOSE, R"(Expected "]" to close the array pattern.)");
			break;
		}
		case GDScriptTokenizer::Token::BRACE_OPEN: {
			// Dictionary.
			advance();
			pattern->pattern_type = PatternNode::PT_DICTIONARY;

			if (!check(GDScriptTokenizer::Token::BRACE_CLOSE) && !is_at_end()) {
				do {
					if (match(GDScriptTokenizer::Token::PERIOD_PERIOD)) {
						// Rest.
						if (pattern->rest_used) {
							push_error(R"(The ".." pattern must be the last element in the pattern dictionary.)");
						} else {
							PatternNode *sub_pattern = alloc_node<PatternNode>();
							sub_pattern->pattern_type = PatternNode::PT_REST;
							pattern->dictionary.push_back({ nullptr, sub_pattern });
							pattern->rest_used = true;
						}
					} else {
						ExpressionNode *key = parse_expression(false);
						if (key == nullptr) {
							push_error(R"(Expected expression as key for dictionary pattern.)");
						}
						if (match(GDScriptTokenizer::Token::COLON)) {
							// Value pattern.
							PatternNode *sub_pattern = parse_match_pattern(p_root_pattern != nullptr ? p_root_pattern : pattern);
							if (sub_pattern == nullptr) {
								continue;
							}
							if (pattern->rest_used) {
								push_error(R"(The ".." pattern must be the last element in the pattern dictionary.)");
							} else if (sub_pattern->pattern_type == PatternNode::PT_REST) {
								push_error(R"(The ".." pattern cannot be used as a value.)");
							} else {
								pattern->dictionary.push_back({ key, sub_pattern });
							}
						} else {
							// Key match only.
							pattern->dictionary.push_back({ key, nullptr });
						}
					}
				} while (match(GDScriptTokenizer::Token::COMMA));
			}
			consume(GDScriptTokenizer::Token::BRACE_CLOSE, R"(Expected "}" to close the dictionary pattern.)");
			break;
		}
		default: {
			// Expression.
			ExpressionNode *expression = parse_expression(false);
			if (expression == nullptr) {
				push_error(R"(Expected expression for match pattern.)");
			} else {
				pattern->pattern_type = PatternNode::PT_EXPRESSION;
				pattern->expression = expression;
			}
			break;
		}
	}

	return pattern;
}

bool GDScriptParser::PatternNode::has_bind(const StringName &p_name) {
	return binds.has(p_name);
}

GDScriptParser::IdentifierNode *GDScriptParser::PatternNode::get_bind(const StringName &p_name) {
	return binds[p_name];
}

GDScriptParser::WhileNode *GDScriptParser::parse_while() {
	WhileNode *n_while = alloc_node<WhileNode>();

	n_while->condition = parse_expression(false);
	if (n_while->condition == nullptr) {
		push_error(R"(Expected conditional expression after "while".)");
	}

	consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after "while" condition.)");

	// Save break/continue state.
	bool could_break = can_break;
	bool could_continue = can_continue;
	bool was_continue_match = is_continue_match;

	// Allow break/continue.
	can_break = true;
	can_continue = true;
	is_continue_match = false;

	n_while->loop = parse_suite(R"("while" block)");

	// Reset break/continue state.
	can_break = could_break;
	can_continue = could_continue;
	is_continue_match = was_continue_match;

	return n_while;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_precedence(Precedence p_precedence, bool p_can_assign, bool p_stop_on_assign) {
	// Switch multiline mode on for grouping tokens.
	// Do this early to avoid the tokenizer generating whitespace tokens.
	switch (current.type) {
		case GDScriptTokenizer::Token::PARENTHESIS_OPEN:
		case GDScriptTokenizer::Token::BRACE_OPEN:
		case GDScriptTokenizer::Token::BRACKET_OPEN:
			push_multiline(true);
			break;
		default:
			break; // Nothing to do.
	}

	// Completion can appear whenever an expression is expected.
	make_completion_context(COMPLETION_IDENTIFIER, nullptr);

	GDScriptTokenizer::Token token = advance();
	ParseFunction prefix_rule = get_rule(token.type)->prefix;

	if (prefix_rule == nullptr) {
		// Expected expression. Let the caller give the proper error message.
		return nullptr;
	}

	ExpressionNode *previous_operand = (this->*prefix_rule)(nullptr, p_can_assign);

	while (p_precedence <= get_rule(current.type)->precedence) {
		if (p_stop_on_assign && current.type == GDScriptTokenizer::Token::EQUAL) {
			return previous_operand;
		}
		// Also switch multiline mode on here for infix operators.
		switch (current.type) {
			// case GDScriptTokenizer::Token::BRACE_OPEN: // Not an infix operator.
			case GDScriptTokenizer::Token::PARENTHESIS_OPEN:
			case GDScriptTokenizer::Token::BRACKET_OPEN:
				push_multiline(true);
				break;
			default:
				break; // Nothing to do.
		}
		token = advance();
		ParseFunction infix_rule = get_rule(token.type)->infix;
		previous_operand = (this->*infix_rule)(previous_operand, p_can_assign);
	}

	return previous_operand;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_expression(bool p_can_assign, bool p_stop_on_assign) {
	return parse_precedence(PREC_ASSIGNMENT, p_can_assign, p_stop_on_assign);
}

GDScriptParser::IdentifierNode *GDScriptParser::parse_identifier() {
	return static_cast<IdentifierNode *>(parse_identifier(nullptr, false));
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_identifier(ExpressionNode *p_previous_operand, bool p_can_assign) {
	if (!previous.is_identifier()) {
		ERR_FAIL_V_MSG(nullptr, "Parser bug: parsing literal node without literal token.");
	}
	IdentifierNode *identifier = alloc_node<IdentifierNode>();
	identifier->name = previous.get_identifier();

	if (current_suite != nullptr && current_suite->has_local(identifier->name)) {
		const SuiteNode::Local &declaration = current_suite->get_local(identifier->name);
		switch (declaration.type) {
			case SuiteNode::Local::CONSTANT:
				identifier->source = IdentifierNode::LOCAL_CONSTANT;
				identifier->constant_source = declaration.constant;
				declaration.constant->usages++;
				break;
			case SuiteNode::Local::VARIABLE:
				identifier->source = IdentifierNode::LOCAL_VARIABLE;
				identifier->variable_source = declaration.variable;
				declaration.variable->usages++;
				break;
			case SuiteNode::Local::PARAMETER:
				identifier->source = IdentifierNode::FUNCTION_PARAMETER;
				identifier->parameter_source = declaration.parameter;
				declaration.parameter->usages++;
				break;
			case SuiteNode::Local::FOR_VARIABLE:
				identifier->source = IdentifierNode::LOCAL_ITERATOR;
				identifier->bind_source = declaration.bind;
				declaration.bind->usages++;
				break;
			case SuiteNode::Local::PATTERN_BIND:
				identifier->source = IdentifierNode::LOCAL_BIND;
				identifier->bind_source = declaration.bind;
				declaration.bind->usages++;
				break;
			case SuiteNode::Local::UNDEFINED:
				ERR_FAIL_V_MSG(nullptr, "Undefined local found.");
		}
	}

	return identifier;
}

GDScriptParser::LiteralNode *GDScriptParser::parse_literal() {
	return static_cast<LiteralNode *>(parse_literal(nullptr, false));
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_literal(ExpressionNode *p_previous_operand, bool p_can_assign) {
	if (previous.type != GDScriptTokenizer::Token::LITERAL) {
		push_error("Parser bug: parsing literal node without literal token.");
		ERR_FAIL_V_MSG(nullptr, "Parser bug: parsing literal node without literal token.");
	}

	LiteralNode *literal = alloc_node<LiteralNode>();
	literal->value = previous.literal;
	return literal;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_self(ExpressionNode *p_previous_operand, bool p_can_assign) {
	if (current_function && current_function->is_static) {
		push_error(R"(Cannot use "self" inside a static function.)");
	}
	SelfNode *self = alloc_node<SelfNode>();
	self->current_class = current_class;
	return self;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_builtin_constant(ExpressionNode *p_previous_operand, bool p_can_assign) {
	GDScriptTokenizer::Token::Type op_type = previous.type;
	LiteralNode *constant = alloc_node<LiteralNode>();

	switch (op_type) {
		case GDScriptTokenizer::Token::CONST_PI:
			constant->value = Math_PI;
			break;
		case GDScriptTokenizer::Token::CONST_TAU:
			constant->value = Math_TAU;
			break;
		case GDScriptTokenizer::Token::CONST_INF:
			constant->value = Math_INF;
			break;
		case GDScriptTokenizer::Token::CONST_NAN:
			constant->value = Math_NAN;
			break;
		default:
			return nullptr; // Unreachable.
	}

	return constant;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_unary_operator(ExpressionNode *p_previous_operand, bool p_can_assign) {
	GDScriptTokenizer::Token::Type op_type = previous.type;
	UnaryOpNode *operation = alloc_node<UnaryOpNode>();

	switch (op_type) {
		case GDScriptTokenizer::Token::MINUS:
			operation->operation = UnaryOpNode::OP_NEGATIVE;
			operation->variant_op = Variant::OP_NEGATE;
			operation->operand = parse_precedence(PREC_SIGN, false);
			break;
		case GDScriptTokenizer::Token::PLUS:
			operation->operation = UnaryOpNode::OP_POSITIVE;
			operation->variant_op = Variant::OP_POSITIVE;
			operation->operand = parse_precedence(PREC_SIGN, false);
			break;
		case GDScriptTokenizer::Token::TILDE:
			operation->operation = UnaryOpNode::OP_COMPLEMENT;
			operation->variant_op = Variant::OP_BIT_NEGATE;
			operation->operand = parse_precedence(PREC_BIT_NOT, false);
			break;
		case GDScriptTokenizer::Token::NOT:
		case GDScriptTokenizer::Token::BANG:
			operation->operation = UnaryOpNode::OP_LOGIC_NOT;
			operation->variant_op = Variant::OP_NOT;
			operation->operand = parse_precedence(PREC_LOGIC_NOT, false);
			break;
		default:
			return nullptr; // Unreachable.
	}

	return operation;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_binary_operator(ExpressionNode *p_previous_operand, bool p_can_assign) {
	GDScriptTokenizer::Token op = previous;
	BinaryOpNode *operation = alloc_node<BinaryOpNode>();

	Precedence precedence = (Precedence)(get_rule(op.type)->precedence + 1);
	operation->left_operand = p_previous_operand;
	operation->right_operand = parse_precedence(precedence, false);

	if (operation->right_operand == nullptr) {
		push_error(vformat(R"(Expected expression after "%s" operator.")", op.get_name()));
	}

	// TODO: Also for unary, ternary, and assignment.
	switch (op.type) {
		case GDScriptTokenizer::Token::PLUS:
			operation->operation = BinaryOpNode::OP_ADDITION;
			operation->variant_op = Variant::OP_ADD;
			break;
		case GDScriptTokenizer::Token::MINUS:
			operation->operation = BinaryOpNode::OP_SUBTRACTION;
			operation->variant_op = Variant::OP_SUBTRACT;
			break;
		case GDScriptTokenizer::Token::STAR:
			operation->operation = BinaryOpNode::OP_MULTIPLICATION;
			operation->variant_op = Variant::OP_MULTIPLY;
			break;
		case GDScriptTokenizer::Token::SLASH:
			operation->operation = BinaryOpNode::OP_DIVISION;
			operation->variant_op = Variant::OP_DIVIDE;
			break;
		case GDScriptTokenizer::Token::PERCENT:
			operation->operation = BinaryOpNode::OP_MODULO;
			operation->variant_op = Variant::OP_MODULE;
			break;
		case GDScriptTokenizer::Token::LESS_LESS:
			operation->operation = BinaryOpNode::OP_BIT_LEFT_SHIFT;
			operation->variant_op = Variant::OP_SHIFT_LEFT;
			break;
		case GDScriptTokenizer::Token::GREATER_GREATER:
			operation->operation = BinaryOpNode::OP_BIT_RIGHT_SHIFT;
			operation->variant_op = Variant::OP_SHIFT_RIGHT;
			break;
		case GDScriptTokenizer::Token::AMPERSAND:
			operation->operation = BinaryOpNode::OP_BIT_AND;
			operation->variant_op = Variant::OP_BIT_AND;
			break;
		case GDScriptTokenizer::Token::PIPE:
			operation->operation = BinaryOpNode::OP_BIT_OR;
			operation->variant_op = Variant::OP_BIT_OR;
			break;
		case GDScriptTokenizer::Token::CARET:
			operation->operation = BinaryOpNode::OP_BIT_XOR;
			operation->variant_op = Variant::OP_BIT_XOR;
			break;
		case GDScriptTokenizer::Token::AND:
		case GDScriptTokenizer::Token::AMPERSAND_AMPERSAND:
			operation->operation = BinaryOpNode::OP_LOGIC_AND;
			operation->variant_op = Variant::OP_AND;
			break;
		case GDScriptTokenizer::Token::OR:
		case GDScriptTokenizer::Token::PIPE_PIPE:
			operation->operation = BinaryOpNode::OP_LOGIC_OR;
			operation->variant_op = Variant::OP_OR;
			break;
		case GDScriptTokenizer::Token::IS:
			operation->operation = BinaryOpNode::OP_TYPE_TEST;
			break;
		case GDScriptTokenizer::Token::IN:
			operation->operation = BinaryOpNode::OP_CONTENT_TEST;
			operation->variant_op = Variant::OP_IN;
			break;
		case GDScriptTokenizer::Token::EQUAL_EQUAL:
			operation->operation = BinaryOpNode::OP_COMP_EQUAL;
			operation->variant_op = Variant::OP_EQUAL;
			break;
		case GDScriptTokenizer::Token::BANG_EQUAL:
			operation->operation = BinaryOpNode::OP_COMP_NOT_EQUAL;
			operation->variant_op = Variant::OP_NOT_EQUAL;
			break;
		case GDScriptTokenizer::Token::LESS:
			operation->operation = BinaryOpNode::OP_COMP_LESS;
			operation->variant_op = Variant::OP_LESS;
			break;
		case GDScriptTokenizer::Token::LESS_EQUAL:
			operation->operation = BinaryOpNode::OP_COMP_LESS_EQUAL;
			operation->variant_op = Variant::OP_LESS_EQUAL;
			break;
		case GDScriptTokenizer::Token::GREATER:
			operation->operation = BinaryOpNode::OP_COMP_GREATER;
			operation->variant_op = Variant::OP_GREATER;
			break;
		case GDScriptTokenizer::Token::GREATER_EQUAL:
			operation->operation = BinaryOpNode::OP_COMP_GREATER_EQUAL;
			operation->variant_op = Variant::OP_GREATER_EQUAL;
			break;
		default:
			return nullptr; // Unreachable.
	}

	return operation;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_ternary_operator(ExpressionNode *p_previous_operand, bool p_can_assign) {
	// Only one ternary operation exists, so no abstraction here.
	TernaryOpNode *operation = alloc_node<TernaryOpNode>();
	operation->true_expr = p_previous_operand;

	operation->condition = parse_precedence(PREC_TERNARY, false);

	if (operation->condition == nullptr) {
		push_error(R"(Expected expression as ternary condition after "if".)");
	}

	consume(GDScriptTokenizer::Token::ELSE, R"(Expected "else" after ternary operator condition.)");

	operation->false_expr = parse_precedence(PREC_TERNARY, false);

	return operation;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_assignment(ExpressionNode *p_previous_operand, bool p_can_assign) {
	if (!p_can_assign) {
		push_error("Assignment is not allowed inside an expression.");
		return parse_expression(false); // Return the following expression.
	}

#ifdef DEBUG_ENABLED
	VariableNode *source_variable = nullptr;
#endif

	switch (p_previous_operand->type) {
		case Node::IDENTIFIER: {
#ifdef DEBUG_ENABLED
			// Get source to store assignment count.
			// Also remove one usage since assignment isn't usage.
			IdentifierNode *id = static_cast<IdentifierNode *>(p_previous_operand);
			switch (id->source) {
				case IdentifierNode::LOCAL_VARIABLE:

					source_variable = id->variable_source;
					id->variable_source->usages--;
					break;
				case IdentifierNode::LOCAL_CONSTANT:
					id->constant_source->usages--;
					break;
				case IdentifierNode::FUNCTION_PARAMETER:
					id->parameter_source->usages--;
					break;
				case IdentifierNode::LOCAL_ITERATOR:
				case IdentifierNode::LOCAL_BIND:
					id->bind_source->usages--;
					break;
				default:
					break;
			}
#endif
		} break;
		case Node::SUBSCRIPT:
			// Okay.
			break;
		default:
			push_error(R"(Only identifier, attribute access, and subscription access can be used as assignment target.)");
			return parse_expression(false); // Return the following expression.
	}

	AssignmentNode *assignment = alloc_node<AssignmentNode>();
	make_completion_context(COMPLETION_ASSIGN, assignment);
#ifdef DEBUG_ENABLED
	bool has_operator = true;
#endif
	switch (previous.type) {
		case GDScriptTokenizer::Token::EQUAL:
			assignment->operation = AssignmentNode::OP_NONE;
			assignment->variant_op = Variant::OP_MAX;
#ifdef DEBUG_ENABLED
			has_operator = false;
#endif
			break;
		case GDScriptTokenizer::Token::PLUS_EQUAL:
			assignment->operation = AssignmentNode::OP_ADDITION;
			assignment->variant_op = Variant::OP_ADD;
			break;
		case GDScriptTokenizer::Token::MINUS_EQUAL:
			assignment->operation = AssignmentNode::OP_SUBTRACTION;
			assignment->variant_op = Variant::OP_SUBTRACT;
			break;
		case GDScriptTokenizer::Token::STAR_EQUAL:
			assignment->operation = AssignmentNode::OP_MULTIPLICATION;
			assignment->variant_op = Variant::OP_MULTIPLY;
			break;
		case GDScriptTokenizer::Token::SLASH_EQUAL:
			assignment->operation = AssignmentNode::OP_DIVISION;
			assignment->variant_op = Variant::OP_DIVIDE;
			break;
		case GDScriptTokenizer::Token::PERCENT_EQUAL:
			assignment->operation = AssignmentNode::OP_MODULO;
			assignment->variant_op = Variant::OP_MODULE;
			break;
		case GDScriptTokenizer::Token::LESS_LESS_EQUAL:
			assignment->operation = AssignmentNode::OP_BIT_SHIFT_LEFT;
			assignment->variant_op = Variant::OP_SHIFT_LEFT;
			break;
		case GDScriptTokenizer::Token::GREATER_GREATER_EQUAL:
			assignment->operation = AssignmentNode::OP_BIT_SHIFT_RIGHT;
			assignment->variant_op = Variant::OP_SHIFT_RIGHT;
			break;
		case GDScriptTokenizer::Token::AMPERSAND_EQUAL:
			assignment->operation = AssignmentNode::OP_BIT_AND;
			assignment->variant_op = Variant::OP_BIT_AND;
			break;
		case GDScriptTokenizer::Token::PIPE_EQUAL:
			assignment->operation = AssignmentNode::OP_BIT_OR;
			assignment->variant_op = Variant::OP_BIT_OR;
			break;
		case GDScriptTokenizer::Token::CARET_EQUAL:
			assignment->operation = AssignmentNode::OP_BIT_XOR;
			assignment->variant_op = Variant::OP_BIT_XOR;
			break;
		default:
			break; // Unreachable.
	}
	assignment->assignee = p_previous_operand;
	assignment->assigned_value = parse_expression(false);

#ifdef DEBUG_ENABLED
	if (has_operator && source_variable != nullptr && source_variable->assignments == 0) {
		push_warning(assignment, GDScriptWarning::UNASSIGNED_VARIABLE_OP_ASSIGN, source_variable->identifier->name);
	}
#endif

	return assignment;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_await(ExpressionNode *p_previous_operand, bool p_can_assign) {
	AwaitNode *await = alloc_node<AwaitNode>();
	await->to_await = parse_precedence(PREC_AWAIT, false);

	current_function->is_coroutine = true;

	return await;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_array(ExpressionNode *p_previous_operand, bool p_can_assign) {
	ArrayNode *array = alloc_node<ArrayNode>();

	if (!check(GDScriptTokenizer::Token::BRACKET_CLOSE)) {
		do {
			if (check(GDScriptTokenizer::Token::BRACKET_CLOSE)) {
				// Allow for trailing comma.
				break;
			}

			ExpressionNode *element = parse_expression(false);
			if (element == nullptr) {
				push_error(R"(Expected expression as array element.)");
			} else {
				array->elements.push_back(element);
			}
		} while (match(GDScriptTokenizer::Token::COMMA) && !is_at_end());
	}
	pop_multiline();
	consume(GDScriptTokenizer::Token::BRACKET_CLOSE, R"(Expected closing "]" after array elements.)");

	return array;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_dictionary(ExpressionNode *p_previous_operand, bool p_can_assign) {
	DictionaryNode *dictionary = alloc_node<DictionaryNode>();

	bool decided_style = false;
	if (!check(GDScriptTokenizer::Token::BRACE_CLOSE)) {
		do {
			if (check(GDScriptTokenizer::Token::BRACE_CLOSE)) {
				// Allow for trailing comma.
				break;
			}

			// Key.
			ExpressionNode *key = parse_expression(false, true); // Stop on "=" so we can check for Lua table style.

			if (key == nullptr) {
				push_error(R"(Expected expression as dictionary key.)");
			}

			if (!decided_style) {
				switch (current.type) {
					case GDScriptTokenizer::Token::COLON:
						dictionary->style = DictionaryNode::PYTHON_DICT;
						break;
					case GDScriptTokenizer::Token::EQUAL:
						dictionary->style = DictionaryNode::LUA_TABLE;
						break;
					default:
						push_error(R"(Expected ":" or "=" after dictionary key.)");
						break;
				}
				decided_style = true;
			}

			switch (dictionary->style) {
				case DictionaryNode::LUA_TABLE:
					if (key != nullptr && key->type != Node::IDENTIFIER) {
						push_error("Expected identifier as dictionary key.");
					}
					if (!match(GDScriptTokenizer::Token::EQUAL)) {
						if (match(GDScriptTokenizer::Token::COLON)) {
							push_error(R"(Expected "=" after dictionary key. Mixing dictionary styles is not allowed.)");
							advance(); // Consume wrong separator anyway.
						} else {
							push_error(R"(Expected "=" after dictionary key.)");
						}
					}
					break;
				case DictionaryNode::PYTHON_DICT:
					if (!match(GDScriptTokenizer::Token::COLON)) {
						if (match(GDScriptTokenizer::Token::EQUAL)) {
							push_error(R"(Expected ":" after dictionary key. Mixing dictionary styles is not allowed.)");
							advance(); // Consume wrong separator anyway.
						} else {
							push_error(R"(Expected ":" after dictionary key.)");
						}
					}
					break;
			}

			// Value.
			ExpressionNode *value = parse_expression(false);
			if (value == nullptr) {
				push_error(R"(Expected expression as dictionary value.)");
			}

			if (key != nullptr && value != nullptr) {
				dictionary->elements.push_back({ key, value });
			}
		} while (match(GDScriptTokenizer::Token::COMMA) && !is_at_end());
	}
	pop_multiline();
	consume(GDScriptTokenizer::Token::BRACE_CLOSE, R"(Expected closing "}" after dictionary elements.)");

	return dictionary;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_grouping(ExpressionNode *p_previous_operand, bool p_can_assign) {
	ExpressionNode *grouped = parse_expression(false);
	pop_multiline();
	if (grouped == nullptr) {
		push_error(R"(Expected grouping expression.)");
	} else {
		consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected closing ")" after grouping expression.)*");
	}
	return grouped;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_attribute(ExpressionNode *p_previous_operand, bool p_can_assign) {
	SubscriptNode *attribute = alloc_node<SubscriptNode>();

	if (for_completion) {
		bool is_builtin = false;
		if (p_previous_operand->type == Node::IDENTIFIER) {
			const IdentifierNode *id = static_cast<const IdentifierNode *>(p_previous_operand);
			Variant::Type builtin_type = get_builtin_type(id->name);
			if (builtin_type < Variant::VARIANT_MAX) {
				make_completion_context(COMPLETION_BUILT_IN_TYPE_CONSTANT, builtin_type, true);
				is_builtin = true;
			}
		}
		if (!is_builtin) {
			make_completion_context(COMPLETION_ATTRIBUTE, attribute, -1, true);
		}
	}

	attribute->is_attribute = true;
	attribute->base = p_previous_operand;

	if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected identifier after "." for attribute access.)")) {
		return attribute;
	}
	attribute->attribute = parse_identifier();

	return attribute;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_subscript(ExpressionNode *p_previous_operand, bool p_can_assign) {
	SubscriptNode *subscript = alloc_node<SubscriptNode>();

	make_completion_context(COMPLETION_SUBSCRIPT, subscript);

	subscript->base = p_previous_operand;
	subscript->index = parse_expression(false);

	pop_multiline();
	consume(GDScriptTokenizer::Token::BRACKET_CLOSE, R"(Expected "]" after subscription index.)");

	return subscript;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_cast(ExpressionNode *p_previous_operand, bool p_can_assign) {
	CastNode *cast = alloc_node<CastNode>();

	cast->operand = p_previous_operand;
	cast->cast_type = parse_type();

	if (cast->cast_type == nullptr) {
		push_error(R"(Expected type specifier after "as".)");
		return p_previous_operand;
	}

	return cast;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_call(ExpressionNode *p_previous_operand, bool p_can_assign) {
	CallNode *call = alloc_node<CallNode>();

	if (previous.type == GDScriptTokenizer::Token::SUPER) {
		// Super call.
		call->is_super = true;
		push_multiline(true);
		if (match(GDScriptTokenizer::Token::PARENTHESIS_OPEN)) {
			// Implicit call to the parent method of the same name.
			if (current_function == nullptr) {
				push_error(R"(Cannot use implicit "super" call outside of a function.)");
				pop_multiline();
				return nullptr;
			}
			call->function_name = current_function->identifier->name;
		} else {
			consume(GDScriptTokenizer::Token::PERIOD, R"(Expected "." or "(" after "super".)");
			make_completion_context(COMPLETION_SUPER_METHOD, call, true);
			if (!consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected function name after ".".)")) {
				pop_multiline();
				return nullptr;
			}
			IdentifierNode *identifier = parse_identifier();
			call->callee = identifier;
			call->function_name = identifier->name;
			consume(GDScriptTokenizer::Token::PARENTHESIS_OPEN, R"(Expected "(" after function name.)");
		}
	} else {
		call->callee = p_previous_operand;

		if (call->callee == nullptr) {
			push_error(R"*(Cannot call on an expression. Use ".call()" if it's a Callable.)*");
		} else if (call->callee->type == Node::IDENTIFIER) {
			call->function_name = static_cast<IdentifierNode *>(call->callee)->name;
			make_completion_context(COMPLETION_METHOD, call->callee);
		} else if (call->callee->type == Node::SUBSCRIPT) {
			SubscriptNode *attribute = static_cast<SubscriptNode *>(call->callee);
			if (attribute->is_attribute) {
				if (attribute->attribute) {
					call->function_name = attribute->attribute->name;
				}
				make_completion_context(COMPLETION_ATTRIBUTE_METHOD, call->callee);
			} else {
				// TODO: The analyzer can see if this is actually a Callable and give better error message.
				push_error(R"*(Cannot call on an expression. Use ".call()" if it's a Callable.)*");
			}
		} else {
			push_error(R"*(Cannot call on an expression. Use ".call()" if it's a Callable.)*");
		}
	}

	if (!check(GDScriptTokenizer::Token::PARENTHESIS_CLOSE)) {
		// Arguments.
		push_completion_call(call);
		make_completion_context(COMPLETION_CALL_ARGUMENTS, call, 0, true);
		int argument_index = 0;
		do {
			make_completion_context(COMPLETION_CALL_ARGUMENTS, call, argument_index++, true);
			if (check(GDScriptTokenizer::Token::PARENTHESIS_CLOSE)) {
				// Allow for trailing comma.
				break;
			}
			ExpressionNode *argument = parse_expression(false);
			if (argument == nullptr) {
				push_error(R"(Expected expression as the function argument.)");
			} else {
				call->arguments.push_back(argument);
			}
		} while (match(GDScriptTokenizer::Token::COMMA));
		pop_completion_call();
	}

	pop_multiline();
	consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected closing ")" after call arguments.)*");

	return call;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_get_node(ExpressionNode *p_previous_operand, bool p_can_assign) {
	if (match(GDScriptTokenizer::Token::LITERAL)) {
		if (previous.literal.get_type() != Variant::STRING) {
			push_error(R"(Expect node path as string or identifier after "$".)");
			return nullptr;
		}
		GetNodeNode *get_node = alloc_node<GetNodeNode>();
		make_completion_context(COMPLETION_GET_NODE, get_node);
		get_node->string = parse_literal();
		return get_node;
	} else if (current.is_node_name()) {
		GetNodeNode *get_node = alloc_node<GetNodeNode>();
		int chain_position = 0;
		do {
			make_completion_context(COMPLETION_GET_NODE, get_node, chain_position++);
			if (!current.is_node_name()) {
				push_error(R"(Expect node path after "/".)");
				return nullptr;
			}
			advance();
			IdentifierNode *identifier = alloc_node<IdentifierNode>();
			identifier->name = previous.get_identifier();
			get_node->chain.push_back(identifier);
		} while (match(GDScriptTokenizer::Token::SLASH));
		return get_node;
	} else {
		push_error(R"(Expect node path as string or identifier after "$".)");
		return nullptr;
	}
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_preload(ExpressionNode *p_previous_operand, bool p_can_assign) {
	PreloadNode *preload = alloc_node<PreloadNode>();
	preload->resolved_path = "<missing path>";

	push_multiline(true);
	consume(GDScriptTokenizer::Token::PARENTHESIS_OPEN, R"(Expected "(" after "preload".)");

	make_completion_context(COMPLETION_RESOURCE_PATH, preload);
	push_completion_call(preload);

	preload->path = parse_expression(false);

	if (preload->path == nullptr) {
		push_error(R"(Expected resource path after "(".)");
	}

	pop_completion_call();

	pop_multiline();
	consume(GDScriptTokenizer::Token::PARENTHESIS_CLOSE, R"*(Expected ")" after preload path.)*");

	return preload;
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_invalid_token(ExpressionNode *p_previous_operand, bool p_can_assign) {
	// Just for better error messages.
	GDScriptTokenizer::Token::Type invalid = previous.type;

	switch (invalid) {
		case GDScriptTokenizer::Token::QUESTION_MARK:
			push_error(R"(Unexpected "?" in source. If you want a ternary operator, use "truthy_value if true_condition else falsy_value".)");
			break;
		default:
			return nullptr; // Unreachable.
	}

	// Return the previous expression.
	return p_previous_operand;
}

GDScriptParser::TypeNode *GDScriptParser::parse_type(bool p_allow_void) {
	TypeNode *type = alloc_node<TypeNode>();
	make_completion_context(p_allow_void ? COMPLETION_TYPE_NAME_OR_VOID : COMPLETION_TYPE_NAME, type);
	if (!match(GDScriptTokenizer::Token::IDENTIFIER)) {
		if (match(GDScriptTokenizer::Token::VOID)) {
			if (p_allow_void) {
				TypeNode *void_type = alloc_node<TypeNode>();
				return void_type;
			} else {
				push_error(R"("void" is only allowed for a function return type.)");
			}
		}
		// Leave error message to the caller who knows the context.
		return nullptr;
	}

	IdentifierNode *type_element = parse_identifier();

	type->type_chain.push_back(type_element);

	int chain_index = 1;
	while (match(GDScriptTokenizer::Token::PERIOD)) {
		make_completion_context(COMPLETION_TYPE_ATTRIBUTE, type, chain_index++);
		if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected inner type name after ".".)")) {
			type_element = parse_identifier();
			type->type_chain.push_back(type_element);
		}
	}

	return type;
}

GDScriptParser::ParseRule *GDScriptParser::get_rule(GDScriptTokenizer::Token::Type p_token_type) {
	// Function table for expression parsing.
	// clang-format destroys the alignment here, so turn off for the table.
	/* clang-format off */
	static ParseRule rules[] = {
		// PREFIX                                           INFIX                                           PRECEDENCE (for infix)
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // EMPTY,
		// Basic
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // ANNOTATION,
		{ &GDScriptParser::parse_identifier,             	nullptr,                                        PREC_NONE }, // IDENTIFIER,
		{ &GDScriptParser::parse_literal,                	nullptr,                                        PREC_NONE }, // LITERAL,
		// Comparison
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_COMPARISON }, // LESS,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_COMPARISON }, // LESS_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_COMPARISON }, // GREATER,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_COMPARISON }, // GREATER_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_COMPARISON }, // EQUAL_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_COMPARISON }, // BANG_EQUAL,
		// Logical
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_LOGIC_AND }, // AND,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_LOGIC_OR }, // OR,
		{ &GDScriptParser::parse_unary_operator,         	nullptr,                                        PREC_NONE }, // NOT,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,			PREC_LOGIC_AND }, // AMPERSAND_AMPERSAND,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,			PREC_LOGIC_OR }, // PIPE_PIPE,
		{ &GDScriptParser::parse_unary_operator,			nullptr,                                        PREC_NONE }, // BANG,
		// Bitwise
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_BIT_AND }, // AMPERSAND,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_BIT_OR }, // PIPE,
		{ &GDScriptParser::parse_unary_operator,         	nullptr,                                        PREC_NONE }, // TILDE,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_BIT_XOR }, // CARET,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_BIT_SHIFT }, // LESS_LESS,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_BIT_SHIFT }, // GREATER_GREATER,
		// Math
		{ &GDScriptParser::parse_unary_operator,         	&GDScriptParser::parse_binary_operator,      	PREC_ADDITION }, // PLUS,
		{ &GDScriptParser::parse_unary_operator,         	&GDScriptParser::parse_binary_operator,      	PREC_SUBTRACTION }, // MINUS,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_FACTOR }, // STAR,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_FACTOR }, // SLASH,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_FACTOR }, // PERCENT,
		// Assignment
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // PLUS_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // MINUS_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // STAR_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // SLASH_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // PERCENT_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // LESS_LESS_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // GREATER_GREATER_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // AMPERSAND_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // PIPE_EQUAL,
		{ nullptr,                                          &GDScriptParser::parse_assignment,           	PREC_ASSIGNMENT }, // CARET_EQUAL,
		// Control flow
		{ nullptr,                                          &GDScriptParser::parse_ternary_operator,     	PREC_TERNARY }, // IF,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // ELIF,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // ELSE,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // FOR,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // WHILE,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // BREAK,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // CONTINUE,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // PASS,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // RETURN,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // MATCH,
		// Keywords
		{ nullptr,                                          &GDScriptParser::parse_cast,                 	PREC_CAST }, // AS,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // ASSERT,
		{ &GDScriptParser::parse_await,                  	nullptr,                                        PREC_NONE }, // AWAIT,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // BREAKPOINT,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // CLASS,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // CLASS_NAME,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // CONST,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // ENUM,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // EXTENDS,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // FUNC,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_CONTENT_TEST }, // IN,
		{ nullptr,                                          &GDScriptParser::parse_binary_operator,      	PREC_TYPE_TEST }, // IS,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // NAMESPACE,
		{ &GDScriptParser::parse_preload,					nullptr,                                        PREC_NONE }, // PRELOAD,
		{ &GDScriptParser::parse_self,                   	nullptr,                                        PREC_NONE }, // SELF,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // SIGNAL,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // STATIC,
		{ &GDScriptParser::parse_call,						nullptr,                                        PREC_NONE }, // SUPER,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // TRAIT,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // VAR,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // VOID,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // YIELD,
		// Punctuation
		{ &GDScriptParser::parse_array,                  	&GDScriptParser::parse_subscript,            	PREC_SUBSCRIPT }, // BRACKET_OPEN,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // BRACKET_CLOSE,
		{ &GDScriptParser::parse_dictionary,             	nullptr,                                        PREC_NONE }, // BRACE_OPEN,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // BRACE_CLOSE,
		{ &GDScriptParser::parse_grouping,               	&GDScriptParser::parse_call,                 	PREC_CALL }, // PARENTHESIS_OPEN,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // PARENTHESIS_CLOSE,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // COMMA,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // SEMICOLON,
		{ nullptr,                                          &GDScriptParser::parse_attribute,            	PREC_ATTRIBUTE }, // PERIOD,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // PERIOD_PERIOD,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // COLON,
		{ &GDScriptParser::parse_get_node,               	nullptr,                                        PREC_NONE }, // DOLLAR,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // FORWARD_ARROW,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // UNDERSCORE,
		// Whitespace
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // NEWLINE,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // INDENT,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // DEDENT,
		// Constants
		{ &GDScriptParser::parse_builtin_constant,			nullptr,                                        PREC_NONE }, // CONST_PI,
		{ &GDScriptParser::parse_builtin_constant,			nullptr,                                        PREC_NONE }, // CONST_TAU,
		{ &GDScriptParser::parse_builtin_constant,			nullptr,                                        PREC_NONE }, // CONST_INF,
		{ &GDScriptParser::parse_builtin_constant,			nullptr,                                        PREC_NONE }, // CONST_NAN,
		// Error message improvement
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // VCS_CONFLICT_MARKER,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // BACKTICK,
		{ nullptr,                                          &GDScriptParser::parse_invalid_token,        	PREC_CAST }, // QUESTION_MARK,
		// Special
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // ERROR,
		{ nullptr,                                          nullptr,                                        PREC_NONE }, // TK_EOF,
	};
	/* clang-format on */
	// Avoid desync.
	static_assert(sizeof(rules) / sizeof(rules[0]) == GDScriptTokenizer::Token::TK_MAX, "Amount of parse rules don't match the amount of token types.");

	// Let's assume this this never invalid, since nothing generates a TK_MAX.
	return &rules[p_token_type];
}

bool GDScriptParser::SuiteNode::has_local(const StringName &p_name) const {
	if (locals_indices.has(p_name)) {
		return true;
	}
	if (parent_block != nullptr) {
		return parent_block->has_local(p_name);
	}
	return false;
}

const GDScriptParser::SuiteNode::Local &GDScriptParser::SuiteNode::get_local(const StringName &p_name) const {
	if (locals_indices.has(p_name)) {
		return locals[locals_indices[p_name]];
	}
	if (parent_block != nullptr) {
		return parent_block->get_local(p_name);
	}
	return empty;
}

bool GDScriptParser::AnnotationNode::apply(GDScriptParser *p_this, Node *p_target) const {
	return (p_this->*(p_this->valid_annotations[name].apply))(this, p_target);
}

bool GDScriptParser::AnnotationNode::applies_to(uint32_t p_target_kinds) const {
	return (info->target_kind & p_target_kinds) > 0;
}

bool GDScriptParser::validate_annotation_arguments(AnnotationNode *p_annotation) {
	ERR_FAIL_COND_V_MSG(!valid_annotations.has(p_annotation->name), false, vformat(R"(Annotation "%s" not found to validate.)", p_annotation->name));

	const MethodInfo &info = valid_annotations[p_annotation->name].info;

	if (((info.flags & METHOD_FLAG_VARARG) == 0) && p_annotation->arguments.size() > info.arguments.size()) {
		push_error(vformat(R"(Annotation "%s" requires at most %d arguments, but %d were given.)", p_annotation->name, info.arguments.size(), p_annotation->arguments.size()));
		return false;
	}

	if (p_annotation->arguments.size() < info.arguments.size() - info.default_arguments.size()) {
		push_error(vformat(R"(Annotation "%s" requires at least %d arguments, but %d were given.)", p_annotation->name, info.arguments.size() - info.default_arguments.size(), p_annotation->arguments.size()));
		return false;
	}

	const List<PropertyInfo>::Element *E = info.arguments.front();
	for (int i = 0; i < p_annotation->arguments.size(); i++) {
		ExpressionNode *argument = p_annotation->arguments[i];
		const PropertyInfo &parameter = E->get();

		if (E->next() != nullptr) {
			E = E->next();
		}

		switch (parameter.type) {
			case Variant::STRING:
			case Variant::STRING_NAME:
			case Variant::NODE_PATH:
				// Allow "quote-less strings", as long as they are recognized as identifiers.
				if (argument->type == Node::IDENTIFIER) {
					IdentifierNode *string = static_cast<IdentifierNode *>(argument);
					Callable::CallError error;
					Vector<Variant> args = varray(string->name);
					const Variant *name = args.ptr();
					p_annotation->resolved_arguments.push_back(Variant::construct(parameter.type, &(name), 1, error));
					if (error.error != Callable::CallError::CALL_OK) {
						push_error(vformat(R"(Expected %s as argument %d of annotation "%s").)", Variant::get_type_name(parameter.type), i + 1, p_annotation->name));
						p_annotation->resolved_arguments.remove(p_annotation->resolved_arguments.size() - 1);
						return false;
					}
					break;
				}
				[[fallthrough]];
			default: {
				if (argument->type != Node::LITERAL) {
					push_error(vformat(R"(Expected %s as argument %d of annotation "%s").)", Variant::get_type_name(parameter.type), i + 1, p_annotation->name));
					return false;
				}

				Variant value = static_cast<LiteralNode *>(argument)->value;
				if (!Variant::can_convert_strict(value.get_type(), parameter.type)) {
					push_error(vformat(R"(Expected %s as argument %d of annotation "%s").)", Variant::get_type_name(parameter.type), i + 1, p_annotation->name));
					return false;
				}
				Callable::CallError error;
				const Variant *args = &value;
				p_annotation->resolved_arguments.push_back(Variant::construct(parameter.type, &(args), 1, error));
				if (error.error != Callable::CallError::CALL_OK) {
					push_error(vformat(R"(Expected %s as argument %d of annotation "%s").)", Variant::get_type_name(parameter.type), i + 1, p_annotation->name));
					p_annotation->resolved_arguments.remove(p_annotation->resolved_arguments.size() - 1);
					return false;
				}
				break;
			}
		}
	}

	return true;
}

bool GDScriptParser::tool_annotation(const AnnotationNode *p_annotation, Node *p_node) {
	this->_is_tool = true;
	return true;
}

bool GDScriptParser::icon_annotation(const AnnotationNode *p_annotation, Node *p_node) {
	ERR_FAIL_COND_V_MSG(p_node->type != Node::CLASS, false, R"("@icon" annotation can only be applied to classes.)");
	ClassNode *p_class = static_cast<ClassNode *>(p_node);
	p_class->icon_path = p_annotation->resolved_arguments[0];
	return true;
}

bool GDScriptParser::onready_annotation(const AnnotationNode *p_annotation, Node *p_node) {
	ERR_FAIL_COND_V_MSG(p_node->type != Node::VARIABLE, false, R"("@onready" annotation can only be applied to class variables.)");

	VariableNode *variable = static_cast<VariableNode *>(p_node);
	if (variable->onready) {
		push_error(R"("@onready" annotation can only be used once per variable.)");
		return false;
	}
	variable->onready = true;
	current_class->onready_used = true;
	return true;
}

template <PropertyHint t_hint, Variant::Type t_type>
bool GDScriptParser::export_annotations(const AnnotationNode *p_annotation, Node *p_node) {
	ERR_FAIL_COND_V_MSG(p_node->type != Node::VARIABLE, false, vformat(R"("%s" annotation can only be applied to variables.)", p_annotation->name));

	VariableNode *variable = static_cast<VariableNode *>(p_node);
	if (variable->exported) {
		push_error(vformat(R"(Annotation "%s" cannot be used with another "@export" annotation.)", p_annotation->name), p_annotation);
		return false;
	}

	variable->exported = true;
	// TODO: Improving setting type, especially for range hints, which can be int or float.
	variable->export_info.type = t_type;
	variable->export_info.hint = t_hint;

	if (p_annotation->name == "@export") {
		if (variable->datatype_specifier == nullptr) {
			if (variable->initializer == nullptr) {
				push_error(R"(Cannot use "@export" annotation with variable without type or initializer, since type can't be inferred.)", p_annotation);
				return false;
			}
			if (variable->initializer->type != Node::LITERAL) {
				push_error(R"(To use "@export" annotation with type-less variable, the default value must be a literal.)", p_annotation);
				return false;
			}
			variable->export_info.type = static_cast<LiteralNode *>(variable->initializer)->value.get_type();
		} // else: Actual type will be set by the analyzer, which can infer the proper type.
	}

	String hint_string;
	for (int i = 0; i < p_annotation->resolved_arguments.size(); i++) {
		if (i > 0) {
			hint_string += ",";
		}
		hint_string += String(p_annotation->resolved_arguments[i]);
	}

	variable->export_info.hint_string = hint_string;

	return true;
}

bool GDScriptParser::warning_annotations(const AnnotationNode *p_annotation, Node *p_node) {
	ERR_FAIL_V_MSG(false, "Not implemented.");
}

template <MultiplayerAPI::RPCMode t_mode>
bool GDScriptParser::network_annotations(const AnnotationNode *p_annotation, Node *p_node) {
	ERR_FAIL_COND_V_MSG(p_node->type != Node::VARIABLE && p_node->type != Node::FUNCTION, false, vformat(R"("%s" annotation can only be applied to variables and functions.)", p_annotation->name));

	switch (p_node->type) {
		case Node::VARIABLE: {
			VariableNode *variable = static_cast<VariableNode *>(p_node);
			if (variable->rpc_mode != MultiplayerAPI::RPC_MODE_DISABLED) {
				push_error(R"(RPC annotations can only be used once per variable.)", p_annotation);
			}
			variable->rpc_mode = t_mode;
			break;
		}
		case Node::FUNCTION: {
			FunctionNode *function = static_cast<FunctionNode *>(p_node);
			if (function->rpc_mode != MultiplayerAPI::RPC_MODE_DISABLED) {
				push_error(R"(RPC annotations can only be used once per function.)", p_annotation);
			}
			function->rpc_mode = t_mode;
			break;
		}
		default:
			return false; // Unreachable.
	}

	return true;
}

GDScriptParser::DataType GDScriptParser::SuiteNode::Local::get_datatype() const {
	switch (type) {
		case CONSTANT:
			return constant->get_datatype();
		case VARIABLE:
			return variable->get_datatype();
		case PARAMETER:
			return parameter->get_datatype();
		case FOR_VARIABLE:
		case PATTERN_BIND:
			return bind->get_datatype();
		case UNDEFINED:
			return DataType();
	}
	return DataType();
}

String GDScriptParser::SuiteNode::Local::get_name() const {
	String name;
	switch (type) {
		case SuiteNode::Local::PARAMETER:
			name = "parameter";
			break;
		case SuiteNode::Local::CONSTANT:
			name = "constant";
			break;
		case SuiteNode::Local::VARIABLE:
			name = "variable";
			break;
		case SuiteNode::Local::FOR_VARIABLE:
			name = "for loop iterator";
			break;
		case SuiteNode::Local::PATTERN_BIND:
			name = "pattern_bind";
			break;
		case SuiteNode::Local::UNDEFINED:
			name = "<undefined>";
			break;
	}
	return name;
}

String GDScriptParser::DataType::to_string() const {
	switch (kind) {
		case VARIANT:
			return "Variant";
		case BUILTIN:
			if (builtin_type == Variant::NIL) {
				return "null";
			}
			return Variant::get_type_name(builtin_type);
		case NATIVE:
			if (is_meta_type) {
				return GDScriptNativeClass::get_class_static();
			}
			return native_type.operator String();
		case CLASS:
			if (is_meta_type) {
				return GDScript::get_class_static();
			}
			if (class_type->identifier != nullptr) {
				return class_type->identifier->name.operator String();
			}
			return class_type->fqcn;
		case SCRIPT: {
			if (is_meta_type) {
				return script_type->get_class_name().operator String();
			}
			String name = script_type->get_name();
			if (!name.empty()) {
				return name;
			}
			name = script_path;
			if (!name.empty()) {
				return name;
			}
			return native_type.operator String();
		}
		case ENUM:
			return enum_type.operator String() + " (enum)";
		case ENUM_VALUE:
			return enum_type.operator String() + " (enum value)";
		case UNRESOLVED:
			return "<unresolved type>";
	}

	ERR_FAIL_V_MSG("<unresolved type", "Kind set outside the enum range.");
}

/*---------- PRETTY PRINT FOR DEBUG ----------*/

#ifdef DEBUG_ENABLED

void GDScriptParser::TreePrinter::increase_indent() {
	indent_level++;
	indent = "";
	for (int i = 0; i < indent_level * 4; i++) {
		if (i % 4 == 0) {
			indent += "|";
		} else {
			indent += " ";
		}
	}
}

void GDScriptParser::TreePrinter::decrease_indent() {
	indent_level--;
	indent = "";
	for (int i = 0; i < indent_level * 4; i++) {
		if (i % 4 == 0) {
			indent += "|";
		} else {
			indent += " ";
		}
	}
}

void GDScriptParser::TreePrinter::push_line(const String &p_line) {
	if (!p_line.empty()) {
		push_text(p_line);
	}
	printed += "\n";
	pending_indent = true;
}

void GDScriptParser::TreePrinter::push_text(const String &p_text) {
	if (pending_indent) {
		printed += indent;
		pending_indent = false;
	}
	printed += p_text;
}

void GDScriptParser::TreePrinter::print_annotation(AnnotationNode *p_annotation) {
	push_text(p_annotation->name);
	push_text(" (");
	for (int i = 0; i < p_annotation->arguments.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_annotation->arguments[i]);
	}
	push_line(")");
}

void GDScriptParser::TreePrinter::print_array(ArrayNode *p_array) {
	push_text("[ ");
	for (int i = 0; i < p_array->elements.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_array->elements[i]);
	}
	push_text(" ]");
}

void GDScriptParser::TreePrinter::print_assert(AssertNode *p_assert) {
	push_text("Assert ( ");
	print_expression(p_assert->condition);
	push_line(" )");
}

void GDScriptParser::TreePrinter::print_assignment(AssignmentNode *p_assignment) {
	switch (p_assignment->assignee->type) {
		case Node::IDENTIFIER:
			print_identifier(static_cast<IdentifierNode *>(p_assignment->assignee));
			break;
		case Node::SUBSCRIPT:
			print_subscript(static_cast<SubscriptNode *>(p_assignment->assignee));
			break;
		default:
			break; // Unreachable.
	}

	push_text(" ");
	switch (p_assignment->operation) {
		case AssignmentNode::OP_ADDITION:
			push_text("+");
			break;
		case AssignmentNode::OP_SUBTRACTION:
			push_text("-");
			break;
		case AssignmentNode::OP_MULTIPLICATION:
			push_text("*");
			break;
		case AssignmentNode::OP_DIVISION:
			push_text("/");
			break;
		case AssignmentNode::OP_MODULO:
			push_text("%");
			break;
		case AssignmentNode::OP_BIT_SHIFT_LEFT:
			push_text("<<");
			break;
		case AssignmentNode::OP_BIT_SHIFT_RIGHT:
			push_text(">>");
			break;
		case AssignmentNode::OP_BIT_AND:
			push_text("&");
			break;
		case AssignmentNode::OP_BIT_OR:
			push_text("|");
			break;
		case AssignmentNode::OP_BIT_XOR:
			push_text("^");
			break;
		case AssignmentNode::OP_NONE:
			break;
	}
	push_text("= ");
	print_expression(p_assignment->assigned_value);
	push_line();
}

void GDScriptParser::TreePrinter::print_await(AwaitNode *p_await) {
	push_text("Await ");
	print_expression(p_await->to_await);
}

void GDScriptParser::TreePrinter::print_binary_op(BinaryOpNode *p_binary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	print_expression(p_binary_op->left_operand);
	switch (p_binary_op->operation) {
		case BinaryOpNode::OP_ADDITION:
			push_text(" + ");
			break;
		case BinaryOpNode::OP_SUBTRACTION:
			push_text(" - ");
			break;
		case BinaryOpNode::OP_MULTIPLICATION:
			push_text(" * ");
			break;
		case BinaryOpNode::OP_DIVISION:
			push_text(" / ");
			break;
		case BinaryOpNode::OP_MODULO:
			push_text(" % ");
			break;
		case BinaryOpNode::OP_BIT_LEFT_SHIFT:
			push_text(" << ");
			break;
		case BinaryOpNode::OP_BIT_RIGHT_SHIFT:
			push_text(" >> ");
			break;
		case BinaryOpNode::OP_BIT_AND:
			push_text(" & ");
			break;
		case BinaryOpNode::OP_BIT_OR:
			push_text(" | ");
			break;
		case BinaryOpNode::OP_BIT_XOR:
			push_text(" ^ ");
			break;
		case BinaryOpNode::OP_LOGIC_AND:
			push_text(" AND ");
			break;
		case BinaryOpNode::OP_LOGIC_OR:
			push_text(" OR ");
			break;
		case BinaryOpNode::OP_TYPE_TEST:
			push_text(" IS ");
			break;
		case BinaryOpNode::OP_CONTENT_TEST:
			push_text(" IN ");
			break;
		case BinaryOpNode::OP_COMP_EQUAL:
			push_text(" == ");
			break;
		case BinaryOpNode::OP_COMP_NOT_EQUAL:
			push_text(" != ");
			break;
		case BinaryOpNode::OP_COMP_LESS:
			push_text(" < ");
			break;
		case BinaryOpNode::OP_COMP_LESS_EQUAL:
			push_text(" <= ");
			break;
		case BinaryOpNode::OP_COMP_GREATER:
			push_text(" > ");
			break;
		case BinaryOpNode::OP_COMP_GREATER_EQUAL:
			push_text(" >= ");
			break;
	}
	print_expression(p_binary_op->right_operand);
	// Surround in parenthesis for disambiguation.
	push_text(")");
}

void GDScriptParser::TreePrinter::print_call(CallNode *p_call) {
	if (p_call->is_super) {
		push_text("super");
		if (p_call->callee != nullptr) {
			push_text(".");
			print_expression(p_call->callee);
		}
	} else {
		print_expression(p_call->callee);
	}
	push_text("( ");
	for (int i = 0; i < p_call->arguments.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_call->arguments[i]);
	}
	push_text(" )");
}

void GDScriptParser::TreePrinter::print_cast(CastNode *p_cast) {
	print_expression(p_cast->operand);
	push_text(" AS ");
	print_type(p_cast->cast_type);
}

void GDScriptParser::TreePrinter::print_class(ClassNode *p_class) {
	push_text("Class ");
	if (p_class->identifier == nullptr) {
		push_text("<unnamed>");
	} else {
		print_identifier(p_class->identifier);
	}

	if (p_class->extends_used) {
		bool first = true;
		push_text(" Extends ");
		if (!p_class->extends_path.empty()) {
			push_text(vformat(R"("%s")", p_class->extends_path));
			first = false;
		}
		for (int i = 0; i < p_class->extends.size(); i++) {
			if (!first) {
				push_text(".");
			} else {
				first = false;
			}
			push_text(p_class->extends[i]);
		}
	}

	push_line(" :");

	increase_indent();

	for (int i = 0; i < p_class->members.size(); i++) {
		const ClassNode::Member &m = p_class->members[i];

		switch (m.type) {
			case ClassNode::Member::CLASS:
				print_class(m.m_class);
				break;
			case ClassNode::Member::VARIABLE:
				print_variable(m.variable);
				break;
			case ClassNode::Member::CONSTANT:
				print_constant(m.constant);
				break;
			case ClassNode::Member::SIGNAL:
				print_signal(m.signal);
				break;
			case ClassNode::Member::FUNCTION:
				print_function(m.function);
				break;
			case ClassNode::Member::ENUM:
				print_enum(m.m_enum);
				break;
			case ClassNode::Member::ENUM_VALUE:
				break; // Nothing. Will be printed by enum.
			case ClassNode::Member::UNDEFINED:
				push_line("<unknown member>");
				break;
		}
	}

	decrease_indent();
}

void GDScriptParser::TreePrinter::print_constant(ConstantNode *p_constant) {
	push_text("Constant ");
	print_identifier(p_constant->identifier);

	increase_indent();

	push_line();
	push_text("= ");
	if (p_constant->initializer == nullptr) {
		push_text("<missing value>");
	} else {
		print_expression(p_constant->initializer);
	}
	decrease_indent();
	push_line();
}

void GDScriptParser::TreePrinter::print_dictionary(DictionaryNode *p_dictionary) {
	push_line("{");
	increase_indent();
	for (int i = 0; i < p_dictionary->elements.size(); i++) {
		print_expression(p_dictionary->elements[i].key);
		if (p_dictionary->style == DictionaryNode::PYTHON_DICT) {
			push_text(" : ");
		} else {
			push_text(" = ");
		}
		print_expression(p_dictionary->elements[i].value);
		push_line(" ,");
	}
	decrease_indent();
	push_text("}");
}

void GDScriptParser::TreePrinter::print_expression(ExpressionNode *p_expression) {
	switch (p_expression->type) {
		case Node::ARRAY:
			print_array(static_cast<ArrayNode *>(p_expression));
			break;
		case Node::ASSIGNMENT:
			print_assignment(static_cast<AssignmentNode *>(p_expression));
			break;
		case Node::AWAIT:
			print_await(static_cast<AwaitNode *>(p_expression));
			break;
		case Node::BINARY_OPERATOR:
			print_binary_op(static_cast<BinaryOpNode *>(p_expression));
			break;
		case Node::CALL:
			print_call(static_cast<CallNode *>(p_expression));
			break;
		case Node::CAST:
			print_cast(static_cast<CastNode *>(p_expression));
			break;
		case Node::DICTIONARY:
			print_dictionary(static_cast<DictionaryNode *>(p_expression));
			break;
		case Node::GET_NODE:
			print_get_node(static_cast<GetNodeNode *>(p_expression));
			break;
		case Node::IDENTIFIER:
			print_identifier(static_cast<IdentifierNode *>(p_expression));
			break;
		case Node::LITERAL:
			print_literal(static_cast<LiteralNode *>(p_expression));
			break;
		case Node::PRELOAD:
			print_preload(static_cast<PreloadNode *>(p_expression));
			break;
		case Node::SELF:
			print_self(static_cast<SelfNode *>(p_expression));
			break;
		case Node::SUBSCRIPT:
			print_subscript(static_cast<SubscriptNode *>(p_expression));
			break;
		case Node::TERNARY_OPERATOR:
			print_ternary_op(static_cast<TernaryOpNode *>(p_expression));
			break;
		case Node::UNARY_OPERATOR:
			print_unary_op(static_cast<UnaryOpNode *>(p_expression));
			break;
		default:
			push_text(vformat("<unknown expression %d>", p_expression->type));
			break;
	}
}

void GDScriptParser::TreePrinter::print_enum(EnumNode *p_enum) {
	push_text("Enum ");
	if (p_enum->identifier != nullptr) {
		print_identifier(p_enum->identifier);
	} else {
		push_text("<unnamed>");
	}

	push_line(" {");
	increase_indent();
	for (int i = 0; i < p_enum->values.size(); i++) {
		const EnumNode::Value &item = p_enum->values[i];
		print_identifier(item.identifier);
		push_text(" = ");
		push_text(itos(item.value));
		push_line(" ,");
	}
	decrease_indent();
	push_line("}");
}

void GDScriptParser::TreePrinter::print_for(ForNode *p_for) {
	push_text("For ");
	print_identifier(p_for->variable);
	push_text(" IN ");
	print_expression(p_for->list);
	push_line(" :");

	increase_indent();

	print_suite(p_for->loop);

	decrease_indent();
}

void GDScriptParser::TreePrinter::print_function(FunctionNode *p_function) {
	for (const List<AnnotationNode *>::Element *E = p_function->annotations.front(); E != nullptr; E = E->next()) {
		print_annotation(E->get());
	}
	push_text("Function ");
	print_identifier(p_function->identifier);
	push_text("( ");
	for (int i = 0; i < p_function->parameters.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_parameter(p_function->parameters[i]);
	}
	push_line(" ) :");
	increase_indent();
	print_suite(p_function->body);
	decrease_indent();
}

void GDScriptParser::TreePrinter::print_get_node(GetNodeNode *p_get_node) {
	push_text("$");
	if (p_get_node->string != nullptr) {
		print_literal(p_get_node->string);
	} else {
		for (int i = 0; i < p_get_node->chain.size(); i++) {
			if (i > 0) {
				push_text("/");
			}
			print_identifier(p_get_node->chain[i]);
		}
	}
}

void GDScriptParser::TreePrinter::print_identifier(IdentifierNode *p_identifier) {
	push_text(p_identifier->name);
}

void GDScriptParser::TreePrinter::print_if(IfNode *p_if, bool p_is_elif) {
	if (p_is_elif) {
		push_text("Elif ");
	} else {
		push_text("If ");
	}
	print_expression(p_if->condition);
	push_line(" :");

	increase_indent();
	print_suite(p_if->true_block);
	decrease_indent();

	// FIXME: Properly detect "elif" blocks.
	if (p_if->false_block != nullptr) {
		push_line("Else :");
		increase_indent();
		print_suite(p_if->false_block);
		decrease_indent();
	}
}

void GDScriptParser::TreePrinter::print_literal(LiteralNode *p_literal) {
	// Prefix for string types.
	switch (p_literal->value.get_type()) {
		case Variant::NODE_PATH:
			push_text("^\"");
			break;
		case Variant::STRING:
			push_text("\"");
			break;
		case Variant::STRING_NAME:
			push_text("&\"");
			break;
		default:
			break;
	}
	push_text(p_literal->value);
	// Suffix for string types.
	switch (p_literal->value.get_type()) {
		case Variant::NODE_PATH:
		case Variant::STRING:
		case Variant::STRING_NAME:
			push_text("\"");
			break;
		default:
			break;
	}
}

void GDScriptParser::TreePrinter::print_match(MatchNode *p_match) {
	push_text("Match ");
	print_expression(p_match->test);
	push_line(" :");

	increase_indent();
	for (int i = 0; i < p_match->branches.size(); i++) {
		print_match_branch(p_match->branches[i]);
	}
	decrease_indent();
}

void GDScriptParser::TreePrinter::print_match_branch(MatchBranchNode *p_match_branch) {
	for (int i = 0; i < p_match_branch->patterns.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_match_pattern(p_match_branch->patterns[i]);
	}

	push_line(" :");

	increase_indent();
	print_suite(p_match_branch->block);
	decrease_indent();
}

void GDScriptParser::TreePrinter::print_match_pattern(PatternNode *p_match_pattern) {
	switch (p_match_pattern->pattern_type) {
		case PatternNode::PT_LITERAL:
			print_literal(p_match_pattern->literal);
			break;
		case PatternNode::PT_WILDCARD:
			push_text("_");
			break;
		case PatternNode::PT_REST:
			push_text("..");
			break;
		case PatternNode::PT_BIND:
			push_text("Var ");
			print_identifier(p_match_pattern->bind);
			break;
		case PatternNode::PT_EXPRESSION:
			print_expression(p_match_pattern->expression);
			break;
		case PatternNode::PT_ARRAY:
			push_text("[ ");
			for (int i = 0; i < p_match_pattern->array.size(); i++) {
				if (i > 0) {
					push_text(" , ");
				}
				print_match_pattern(p_match_pattern->array[i]);
			}
			push_text(" ]");
			break;
		case PatternNode::PT_DICTIONARY:
			push_text("{ ");
			for (int i = 0; i < p_match_pattern->dictionary.size(); i++) {
				if (i > 0) {
					push_text(" , ");
				}
				if (p_match_pattern->dictionary[i].key != nullptr) {
					// Key can be null for rest pattern.
					print_expression(p_match_pattern->dictionary[i].key);
					push_text(" : ");
				}
				print_match_pattern(p_match_pattern->dictionary[i].value_pattern);
			}
			push_text(" }");
			break;
	}
}

void GDScriptParser::TreePrinter::print_parameter(ParameterNode *p_parameter) {
	print_identifier(p_parameter->identifier);
	if (p_parameter->datatype_specifier != nullptr) {
		push_text(" : ");
		print_type(p_parameter->datatype_specifier);
	}
	if (p_parameter->default_value != nullptr) {
		push_text(" = ");
		print_expression(p_parameter->default_value);
	}
}

void GDScriptParser::TreePrinter::print_preload(PreloadNode *p_preload) {
	push_text(R"(Preload ( ")");
	push_text(p_preload->resolved_path);
	push_text(R"(" )");
}

void GDScriptParser::TreePrinter::print_return(ReturnNode *p_return) {
	push_text("Return");
	if (p_return->return_value != nullptr) {
		push_text(" ");
		print_expression(p_return->return_value);
	}
	push_line();
}

void GDScriptParser::TreePrinter::print_self(SelfNode *p_self) {
	push_text("Self(");
	if (p_self->current_class->identifier != nullptr) {
		print_identifier(p_self->current_class->identifier);
	} else {
		push_text("<main class>");
	}
	push_text(")");
}

void GDScriptParser::TreePrinter::print_signal(SignalNode *p_signal) {
	push_text("Signal ");
	print_identifier(p_signal->identifier);
	push_text("( ");
	for (int i = 0; i < p_signal->parameters.size(); i++) {
		print_parameter(p_signal->parameters[i]);
	}
	push_line(" )");
}

void GDScriptParser::TreePrinter::print_subscript(SubscriptNode *p_subscript) {
	print_expression(p_subscript->base);
	if (p_subscript->is_attribute) {
		push_text(".");
		print_identifier(p_subscript->attribute);
	} else {
		push_text("[ ");
		print_expression(p_subscript->index);
		push_text(" ]");
	}
}

void GDScriptParser::TreePrinter::print_statement(Node *p_statement) {
	switch (p_statement->type) {
		case Node::ASSERT:
			print_assert(static_cast<AssertNode *>(p_statement));
			break;
		case Node::VARIABLE:
			print_variable(static_cast<VariableNode *>(p_statement));
			break;
		case Node::CONSTANT:
			print_constant(static_cast<ConstantNode *>(p_statement));
			break;
		case Node::IF:
			print_if(static_cast<IfNode *>(p_statement));
			break;
		case Node::FOR:
			print_for(static_cast<ForNode *>(p_statement));
			break;
		case Node::WHILE:
			print_while(static_cast<WhileNode *>(p_statement));
			break;
		case Node::MATCH:
			print_match(static_cast<MatchNode *>(p_statement));
			break;
		case Node::RETURN:
			print_return(static_cast<ReturnNode *>(p_statement));
			break;
		case Node::BREAK:
			push_line("Break");
			break;
		case Node::CONTINUE:
			push_line("Continue");
			break;
		case Node::PASS:
			push_line("Pass");
			break;
		case Node::BREAKPOINT:
			push_line("Breakpoint");
			break;
		case Node::ASSIGNMENT:
			print_assignment(static_cast<AssignmentNode *>(p_statement));
			break;
		default:
			if (p_statement->is_expression()) {
				print_expression(static_cast<ExpressionNode *>(p_statement));
				push_line();
			} else {
				push_line(vformat("<unknown statement %d>", p_statement->type));
			}
			break;
	}
}

void GDScriptParser::TreePrinter::print_suite(SuiteNode *p_suite) {
	for (int i = 0; i < p_suite->statements.size(); i++) {
		print_statement(p_suite->statements[i]);
	}
}

void GDScriptParser::TreePrinter::print_ternary_op(TernaryOpNode *p_ternary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	print_expression(p_ternary_op->true_expr);
	push_text(") IF (");
	print_expression(p_ternary_op->condition);
	push_text(") ELSE (");
	print_expression(p_ternary_op->false_expr);
	push_text(")");
}

void GDScriptParser::TreePrinter::print_type(TypeNode *p_type) {
	if (p_type->type_chain.empty()) {
		push_text("Void");
	} else {
		for (int i = 0; i < p_type->type_chain.size(); i++) {
			if (i > 0) {
				push_text(".");
			}
			print_identifier(p_type->type_chain[i]);
		}
	}
}

void GDScriptParser::TreePrinter::print_unary_op(UnaryOpNode *p_unary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	switch (p_unary_op->operation) {
		case UnaryOpNode::OP_POSITIVE:
			push_text("+");
			break;
		case UnaryOpNode::OP_NEGATIVE:
			push_text("-");
			break;
		case UnaryOpNode::OP_LOGIC_NOT:
			push_text("NOT");
			break;
		case UnaryOpNode::OP_COMPLEMENT:
			push_text("~");
			break;
	}
	print_expression(p_unary_op->operand);
	// Surround in parenthesis for disambiguation.
	push_text(")");
}

void GDScriptParser::TreePrinter::print_variable(VariableNode *p_variable) {
	for (const List<AnnotationNode *>::Element *E = p_variable->annotations.front(); E != nullptr; E = E->next()) {
		print_annotation(E->get());
	}

	push_text("Variable ");
	print_identifier(p_variable->identifier);

	push_text(" : ");
	if (p_variable->datatype_specifier != nullptr) {
		print_type(p_variable->datatype_specifier);
	} else if (p_variable->infer_datatype) {
		push_text("<inferred type>");
	} else {
		push_text("Variant");
	}

	increase_indent();

	push_line();
	push_text("= ");
	if (p_variable->initializer == nullptr) {
		push_text("<default value>");
	} else {
		print_expression(p_variable->initializer);
	}
	push_line();

	if (p_variable->property != VariableNode::PROP_NONE) {
		if (p_variable->getter != nullptr) {
			push_text("Get");
			if (p_variable->property == VariableNode::PROP_INLINE) {
				push_line(":");
				increase_indent();
				print_suite(p_variable->getter);
				decrease_indent();
			} else {
				push_line(" =");
				increase_indent();
				print_identifier(p_variable->getter_pointer);
				push_line();
				decrease_indent();
			}
		}
		if (p_variable->setter != nullptr) {
			push_text("Set (");
			if (p_variable->property == VariableNode::PROP_INLINE) {
				if (p_variable->setter_parameter != nullptr) {
					print_identifier(p_variable->setter_parameter);
				} else {
					push_text("<missing>");
				}
				push_line("):");
				increase_indent();
				print_suite(p_variable->setter);
				decrease_indent();
			} else {
				push_line(" =");
				increase_indent();
				print_identifier(p_variable->setter_pointer);
				push_line();
				decrease_indent();
			}
		}
	}

	decrease_indent();
	push_line();
}

void GDScriptParser::TreePrinter::print_while(WhileNode *p_while) {
	push_text("While ");
	print_expression(p_while->condition);
	push_line(" :");

	increase_indent();
	print_suite(p_while->loop);
	decrease_indent();
}

void GDScriptParser::TreePrinter::print_tree(const GDScriptParser &p_parser) {
	ERR_FAIL_COND_MSG(p_parser.get_tree() == nullptr, "Parse the code before printing the parse tree.");

	if (p_parser.is_tool()) {
		push_line("@tool");
	}
	if (!p_parser.get_tree()->icon_path.empty()) {
		push_text(R"(@icon (")");
		push_text(p_parser.get_tree()->icon_path);
		push_line("\")");
	}
	print_class(p_parser.get_tree());

	print_line(printed);
}

#endif // DEBUG_ENABLED
