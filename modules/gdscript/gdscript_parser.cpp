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

<<<<<<< HEAD
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
=======
void GDScriptParser::_set_end_statement_error(String p_name) {
	String error_msg;
	if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER) {
		error_msg = vformat("Expected end of statement (\"%s\"), got %s (\"%s\") instead.", p_name, tokenizer->get_token_name(tokenizer->get_token()), tokenizer->get_token_identifier());
	} else {
		error_msg = vformat("Expected end of statement (\"%s\"), got %s instead.", p_name, tokenizer->get_token_name(tokenizer->get_token()));
	}
	_set_error(error_msg);
}

bool GDScriptParser::_enter_indent_block(BlockNode *p_block) {
>>>>>>> audio-bus-effect-fixed

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
		if (consume(GDScriptTokenizer::Token::IDENTIFIER, R"(Expected identifer for enum key.)")) {
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
			push_error(R"(Expect node path as string or identifer after "$".)");
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
		push_error(R"(Expect node path as string or identifer after "$".)");
		return nullptr;
	}
}

GDScriptParser::ExpressionNode *GDScriptParser::parse_preload(ExpressionNode *p_previous_operand, bool p_can_assign) {
	PreloadNode *preload = alloc_node<PreloadNode>();
	preload->resolved_path = "<missing path>";

	push_multiline(true);
	consume(GDScriptTokenizer::Token::PARENTHESIS_OPEN, R"(Expected "(" after "preload".)");

<<<<<<< HEAD
	make_completion_context(COMPLETION_RESOURCE_PATH, preload);
	push_completion_call(preload);
=======
bool GDScriptParser::_reduce_export_var_type(Variant &p_value, int p_line) {

	if (p_value.get_type() == Variant::ARRAY) {
		Array arr = p_value;
		for (int i = 0; i < arr.size(); i++) {
			if (!_reduce_export_var_type(arr[i], p_line)) return false;
		}
		return true;
	}

	if (p_value.get_type() == Variant::DICTIONARY) {
		Dictionary dict = p_value;
		for (int i = 0; i < dict.size(); i++) {
			Variant value = dict.get_value_at_index(i);
			if (!_reduce_export_var_type(value, p_line)) return false;
		}
		return true;
	}

	// validate type
	DataType type = _type_from_variant(p_value);
	if (type.kind == DataType::BUILTIN) {
		return true;
	} else if (type.kind == DataType::NATIVE) {
		if (ClassDB::is_parent_class(type.native_type, "Resource")) {
			return true;
		}
	}
	_set_error("Invalid export type. Only built-in and native resource types can be exported.", p_line);
	return false;
}

bool GDScriptParser::_recover_from_completion() {
>>>>>>> audio-bus-effect-fixed

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

<<<<<<< HEAD
	ERR_FAIL_V_MSG("<unresolved type", "Kind set outside the enum range.");
=======
	for (int i = 0; i < p_match_statement->branches.size(); i++) {

		PatternBranchNode *branch = p_match_statement->branches[i];

		MatchNode::CompiledPatternBranch compiled_branch;
		compiled_branch.compiled_pattern = NULL;

		Map<StringName, Node *> binding;

		for (int j = 0; j < branch->patterns.size(); j++) {
			PatternNode *pattern = branch->patterns[j];
			_mark_line_as_safe(pattern->line);

			Map<StringName, Node *> bindings;
			Node *resulting_node = NULL;
			_generate_pattern(pattern, id, resulting_node, bindings);

			if (!resulting_node) {
				return;
			}

			if (!binding.empty() && !bindings.empty()) {
				_set_error("Multipatterns can't contain bindings");
				return;
			} else {
				binding = bindings;
			}

			// Result is always a boolean
			DataType resulting_node_type;
			resulting_node_type.has_type = true;
			resulting_node_type.is_constant = true;
			resulting_node_type.kind = DataType::BUILTIN;
			resulting_node_type.builtin_type = Variant::BOOL;
			resulting_node->set_datatype(resulting_node_type);

			if (compiled_branch.compiled_pattern) {
				OperatorNode *or_node = alloc_node<OperatorNode>();
				or_node->op = OperatorNode::OP_OR;
				or_node->arguments.push_back(compiled_branch.compiled_pattern);
				or_node->arguments.push_back(resulting_node);

				compiled_branch.compiled_pattern = or_node;
			} else {
				// single pattern | first one
				compiled_branch.compiled_pattern = resulting_node;
			}
		}

		// prepare the body ...hehe
		for (Map<StringName, Node *>::Element *e = binding.front(); e; e = e->next()) {
			if (!branch->body->variables.has(e->key())) {
				_set_error("Parser bug: missing pattern bind variable.", branch->line);
				ERR_FAIL();
			}

			LocalVarNode *local_var = branch->body->variables[e->key()];
			local_var->assign = e->value();
			local_var->set_datatype(local_var->assign->get_datatype());
			local_var->assignments++;

			IdentifierNode *id2 = alloc_node<IdentifierNode>();
			id2->name = local_var->name;
			id2->datatype = local_var->datatype;
			id2->declared_block = branch->body;
			id2->set_datatype(local_var->assign->get_datatype());

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = OperatorNode::OP_ASSIGN;
			op->arguments.push_back(id2);
			op->arguments.push_back(local_var->assign);
			local_var->assign_op = op;

			branch->body->statements.push_front(op);
			branch->body->statements.push_front(local_var);
		}

		compiled_branch.body = branch->body;

		p_match_statement->compiled_pattern_branches.push_back(compiled_branch);
	}
}

void GDScriptParser::_parse_block(BlockNode *p_block, bool p_static) {

	IndentLevel current_level = indent_level.back()->get();

#ifdef DEBUG_ENABLED

	pending_newline = -1; // reset for the new block

	NewLineNode *nl = alloc_node<NewLineNode>();

	nl->line = tokenizer->get_token_line();
	p_block->statements.push_back(nl);
#endif

	bool is_first_line = true;

	while (true) {
		if (!is_first_line && indent_level.back()->prev() && indent_level.back()->prev()->get().indent == current_level.indent) {
			if (indent_level.back()->prev()->get().is_mixed(current_level)) {
				_set_error("Mixed tabs and spaces in indentation.");
				return;
			}
			// pythonic single-line expression, don't parse future lines
			indent_level.pop_back();
			p_block->end_line = tokenizer->get_token_line();
			return;
		}
		is_first_line = false;

		GDScriptTokenizer::Token token = tokenizer->get_token();
		if (error_set)
			return;

		if (current_level.indent > indent_level.back()->get().indent) {
			p_block->end_line = tokenizer->get_token_line();
			return; //go back a level
		}

		if (pending_newline != -1) {

			NewLineNode *nl2 = alloc_node<NewLineNode>();
			nl2->line = pending_newline;
			p_block->statements.push_back(nl2);
			pending_newline = -1;
		}

#ifdef DEBUG_ENABLED
		switch (token) {
			case GDScriptTokenizer::TK_EOF:
			case GDScriptTokenizer::TK_ERROR:
			case GDScriptTokenizer::TK_NEWLINE:
			case GDScriptTokenizer::TK_CF_PASS: {
				// will check later
			} break;
			default: {
				if (p_block->has_return && !current_function->has_unreachable_code) {
					_add_warning(GDScriptWarning::UNREACHABLE_CODE, -1, current_function->name.operator String());
					current_function->has_unreachable_code = true;
				}
			} break;
		}
#endif // DEBUG_ENABLED
		switch (token) {
			case GDScriptTokenizer::TK_EOF:
				p_block->end_line = tokenizer->get_token_line();
			case GDScriptTokenizer::TK_ERROR: {
				return; //go back

				//end of file!

			} break;
			case GDScriptTokenizer::TK_NEWLINE: {

				int line = tokenizer->get_token_line();

				if (!_parse_newline()) {
					if (!error_set) {
						p_block->end_line = tokenizer->get_token_line();
						pending_newline = p_block->end_line;
					}
					return;
				}

				_mark_line_as_safe(line);
				NewLineNode *nl2 = alloc_node<NewLineNode>();
				nl2->line = line;
				p_block->statements.push_back(nl2);

			} break;
			case GDScriptTokenizer::TK_CF_PASS: {
				if (tokenizer->get_token(1) != GDScriptTokenizer::TK_SEMICOLON && tokenizer->get_token(1) != GDScriptTokenizer::TK_NEWLINE && tokenizer->get_token(1) != GDScriptTokenizer::TK_EOF) {

					_set_error("Expected \";\" or a line break.");
					return;
				}
				_mark_line_as_safe(tokenizer->get_token_line());
				tokenizer->advance();
				if (tokenizer->get_token() == GDScriptTokenizer::TK_SEMICOLON) {
					// Ignore semicolon after 'pass'.
					tokenizer->advance();
				}
			} break;
			case GDScriptTokenizer::TK_PR_VAR: {
				// Variable declaration and (eventual) initialization.

				tokenizer->advance();
				int var_line = tokenizer->get_token_line();
				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Expected an identifier for the local variable name.");
					return;
				}
				StringName n = tokenizer->get_token_literal();
				tokenizer->advance();
				if (current_function) {
					for (int i = 0; i < current_function->arguments.size(); i++) {
						if (n == current_function->arguments[i]) {
							_set_error("Variable \"" + String(n) + "\" already defined in the scope (at line " + itos(current_function->line) + ").");
							return;
						}
					}
				}
				BlockNode *check_block = p_block;
				while (check_block) {
					if (check_block->variables.has(n)) {
						_set_error("Variable \"" + String(n) + "\" already defined in the scope (at line " + itos(check_block->variables[n]->line) + ").");
						return;
					}
					check_block = check_block->parent_block;
				}

				//must know when the local variable is declared
				LocalVarNode *lv = alloc_node<LocalVarNode>();
				lv->name = n;
				lv->line = var_line;
				p_block->statements.push_back(lv);

				Node *assigned = NULL;

				if (tokenizer->get_token() == GDScriptTokenizer::TK_COLON) {
					if (tokenizer->get_token(1) == GDScriptTokenizer::TK_OP_ASSIGN) {
						lv->datatype = DataType();
#ifdef DEBUG_ENABLED
						lv->datatype.infer_type = true;
#endif
						tokenizer->advance();
					} else if (!_parse_type(lv->datatype)) {
						_set_error("Expected a type for the variable.");
						return;
					}
				}

				if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_ASSIGN) {

					tokenizer->advance();
					Node *subexpr = _parse_and_reduce_expression(p_block, p_static);
					if (!subexpr) {
						if (_recover_from_completion()) {
							break;
						}
						return;
					}

					lv->assignments++;
					assigned = subexpr;
				} else {

					assigned = _get_default_value_for_type(lv->datatype, var_line);
				}
				//must be added later, to avoid self-referencing.
				p_block->variables.insert(n, lv);

				IdentifierNode *id = alloc_node<IdentifierNode>();
				id->name = n;
				id->declared_block = p_block;
				id->line = var_line;

				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = OperatorNode::OP_ASSIGN;
				op->arguments.push_back(id);
				op->arguments.push_back(assigned);
				op->line = var_line;
				p_block->statements.push_back(op);
				lv->assign_op = op;
				lv->assign = assigned;

				if (!_end_statement()) {
					_set_end_statement_error("var");
					return;
				}

			} break;
			case GDScriptTokenizer::TK_CF_IF: {

				tokenizer->advance();

				Node *condition = _parse_and_reduce_expression(p_block, p_static);
				if (!condition) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				ControlFlowNode *cf_if = alloc_node<ControlFlowNode>();

				cf_if->cf_type = ControlFlowNode::CF_IF;
				cf_if->arguments.push_back(condition);

				cf_if->body = alloc_node<BlockNode>();
				cf_if->body->parent_block = p_block;
				cf_if->body->if_condition = condition; //helps code completion

				p_block->sub_blocks.push_back(cf_if->body);

				if (!_enter_indent_block(cf_if->body)) {
					_set_error("Expected an indented block after \"if\".");
					p_block->end_line = tokenizer->get_token_line();
					return;
				}

				current_block = cf_if->body;
				_parse_block(cf_if->body, p_static);
				current_block = p_block;

				if (error_set)
					return;
				p_block->statements.push_back(cf_if);

				bool all_have_return = cf_if->body->has_return;
				bool have_else = false;

				while (true) {

					while (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE && _parse_newline())
						;

					if (indent_level.back()->get().indent < current_level.indent) { //not at current indent level
						p_block->end_line = tokenizer->get_token_line();
						return;
					}

					if (tokenizer->get_token() == GDScriptTokenizer::TK_CF_ELIF) {

						if (indent_level.back()->get().indent > current_level.indent) {

							_set_error("Invalid indentation.");
							return;
						}

						tokenizer->advance();

						cf_if->body_else = alloc_node<BlockNode>();
						cf_if->body_else->parent_block = p_block;
						p_block->sub_blocks.push_back(cf_if->body_else);

						ControlFlowNode *cf_else = alloc_node<ControlFlowNode>();
						cf_else->cf_type = ControlFlowNode::CF_IF;

						//condition
						Node *condition2 = _parse_and_reduce_expression(p_block, p_static);
						if (!condition2) {
							if (_recover_from_completion()) {
								break;
							}
							return;
						}
						cf_else->arguments.push_back(condition2);
						cf_else->cf_type = ControlFlowNode::CF_IF;

						cf_if->body_else->statements.push_back(cf_else);
						cf_if = cf_else;
						cf_if->body = alloc_node<BlockNode>();
						cf_if->body->parent_block = p_block;
						p_block->sub_blocks.push_back(cf_if->body);

						if (!_enter_indent_block(cf_if->body)) {
							_set_error("Expected an indented block after \"elif\".");
							p_block->end_line = tokenizer->get_token_line();
							return;
						}

						current_block = cf_else->body;
						_parse_block(cf_else->body, p_static);
						current_block = p_block;
						if (error_set)
							return;

						all_have_return = all_have_return && cf_else->body->has_return;

					} else if (tokenizer->get_token() == GDScriptTokenizer::TK_CF_ELSE) {

						if (indent_level.back()->get().indent > current_level.indent) {
							_set_error("Invalid indentation.");
							return;
						}

						tokenizer->advance();
						cf_if->body_else = alloc_node<BlockNode>();
						cf_if->body_else->parent_block = p_block;
						p_block->sub_blocks.push_back(cf_if->body_else);

						if (!_enter_indent_block(cf_if->body_else)) {
							_set_error("Expected an indented block after \"else\".");
							p_block->end_line = tokenizer->get_token_line();
							return;
						}
						current_block = cf_if->body_else;
						_parse_block(cf_if->body_else, p_static);
						current_block = p_block;
						if (error_set)
							return;

						all_have_return = all_have_return && cf_if->body_else->has_return;
						have_else = true;

						break; //after else, exit

					} else
						break;
				}

				cf_if->body->has_return = all_have_return;
				// If there's no else block, path out of the if might not have a return
				p_block->has_return = all_have_return && have_else;

			} break;
			case GDScriptTokenizer::TK_CF_WHILE: {

				tokenizer->advance();
				Node *condition2 = _parse_and_reduce_expression(p_block, p_static);
				if (!condition2) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				ControlFlowNode *cf_while = alloc_node<ControlFlowNode>();

				cf_while->cf_type = ControlFlowNode::CF_WHILE;
				cf_while->arguments.push_back(condition2);

				cf_while->body = alloc_node<BlockNode>();
				cf_while->body->parent_block = p_block;
				cf_while->body->can_break = true;
				cf_while->body->can_continue = true;
				p_block->sub_blocks.push_back(cf_while->body);

				if (!_enter_indent_block(cf_while->body)) {
					_set_error("Expected an indented block after \"while\".");
					p_block->end_line = tokenizer->get_token_line();
					return;
				}

				current_block = cf_while->body;
				_parse_block(cf_while->body, p_static);
				current_block = p_block;
				if (error_set)
					return;
				p_block->statements.push_back(cf_while);
			} break;
			case GDScriptTokenizer::TK_CF_FOR: {

				tokenizer->advance();

				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Identifier expected after \"for\".");
				}

				IdentifierNode *id = alloc_node<IdentifierNode>();
				id->name = tokenizer->get_token_identifier();
#ifdef DEBUG_ENABLED
				for (int j = 0; j < current_class->variables.size(); j++) {
					if (current_class->variables[j].identifier == id->name) {
						_add_warning(GDScriptWarning::SHADOWED_VARIABLE, id->line, id->name, itos(current_class->variables[j].line));
					}
				}
#endif // DEBUG_ENABLED

				BlockNode *check_block = p_block;
				while (check_block) {
					if (check_block->variables.has(id->name)) {
						_set_error("Variable \"" + String(id->name) + "\" already defined in the scope (at line " + itos(check_block->variables[id->name]->line) + ").");
						return;
					}
					check_block = check_block->parent_block;
				}

				tokenizer->advance();

				if (tokenizer->get_token() != GDScriptTokenizer::TK_OP_IN) {
					_set_error("\"in\" expected after identifier.");
					return;
				}

				tokenizer->advance();

				Node *container = _parse_and_reduce_expression(p_block, p_static);
				if (!container) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				DataType iter_type;

				if (container->type == Node::TYPE_OPERATOR) {

					OperatorNode *op = static_cast<OperatorNode *>(container);
					if (op->op == OperatorNode::OP_CALL && op->arguments[0]->type == Node::TYPE_BUILT_IN_FUNCTION && static_cast<BuiltInFunctionNode *>(op->arguments[0])->function == GDScriptFunctions::GEN_RANGE) {
						//iterating a range, so see if range() can be optimized without allocating memory, by replacing it by vectors (which can work as iterable too!)

						Vector<Node *> args;
						Vector<double> constants;

						bool constant = true;

						for (int i = 1; i < op->arguments.size(); i++) {
							args.push_back(op->arguments[i]);
							if (op->arguments[i]->type == Node::TYPE_CONSTANT) {
								ConstantNode *c = static_cast<ConstantNode *>(op->arguments[i]);
								if (c->value.get_type() == Variant::REAL || c->value.get_type() == Variant::INT) {
									constants.push_back(c->value);
								} else {
									constant = false;
								}
							} else {
								constant = false;
							}
						}

						if (args.size() > 0 && args.size() < 4) {

							if (constant) {

								ConstantNode *cn = alloc_node<ConstantNode>();
								switch (args.size()) {
									case 1: cn->value = (int)constants[0]; break;
									case 2: cn->value = Vector2(constants[0], constants[1]); break;
									case 3: cn->value = Vector3(constants[0], constants[1], constants[2]); break;
								}
								cn->datatype = _type_from_variant(cn->value);
								container = cn;
							} else {
								OperatorNode *on = alloc_node<OperatorNode>();
								on->op = OperatorNode::OP_CALL;

								TypeNode *tn = alloc_node<TypeNode>();
								on->arguments.push_back(tn);

								switch (args.size()) {
									case 1: tn->vtype = Variant::INT; break;
									case 2: tn->vtype = Variant::VECTOR2; break;
									case 3: tn->vtype = Variant::VECTOR3; break;
								}

								for (int i = 0; i < args.size(); i++) {
									on->arguments.push_back(args[i]);
								}

								container = on;
							}
						}

						iter_type.has_type = true;
						iter_type.kind = DataType::BUILTIN;
						iter_type.builtin_type = Variant::INT;
					}
				}

				ControlFlowNode *cf_for = alloc_node<ControlFlowNode>();

				cf_for->cf_type = ControlFlowNode::CF_FOR;
				cf_for->arguments.push_back(id);
				cf_for->arguments.push_back(container);

				cf_for->body = alloc_node<BlockNode>();
				cf_for->body->parent_block = p_block;
				cf_for->body->can_break = true;
				cf_for->body->can_continue = true;
				p_block->sub_blocks.push_back(cf_for->body);

				if (!_enter_indent_block(cf_for->body)) {
					_set_error("Expected indented block after \"for\".");
					p_block->end_line = tokenizer->get_token_line();
					return;
				}

				current_block = cf_for->body;

				// this is for checking variable for redefining
				// inside this _parse_block
				LocalVarNode *lv = alloc_node<LocalVarNode>();
				lv->name = id->name;
				lv->line = id->line;
				lv->assignments++;
				id->declared_block = cf_for->body;
				lv->set_datatype(iter_type);
				id->set_datatype(iter_type);
				cf_for->body->variables.insert(id->name, lv);
				_parse_block(cf_for->body, p_static);
				current_block = p_block;

				if (error_set)
					return;
				p_block->statements.push_back(cf_for);
			} break;
			case GDScriptTokenizer::TK_CF_CONTINUE: {
				BlockNode *upper_block = p_block;
				bool is_continue_valid = false;
				while (upper_block) {
					if (upper_block->can_continue) {
						is_continue_valid = true;
						break;
					}
					upper_block = upper_block->parent_block;
				}

				if (!is_continue_valid) {
					_set_error("Unexpected keyword \"continue\" outside a loop.");
					return;
				}

				_mark_line_as_safe(tokenizer->get_token_line());
				tokenizer->advance();
				ControlFlowNode *cf_continue = alloc_node<ControlFlowNode>();
				cf_continue->cf_type = ControlFlowNode::CF_CONTINUE;
				p_block->statements.push_back(cf_continue);
				if (!_end_statement()) {
					_set_end_statement_error("continue");
					return;
				}
			} break;
			case GDScriptTokenizer::TK_CF_BREAK: {
				BlockNode *upper_block = p_block;
				bool is_break_valid = false;
				while (upper_block) {
					if (upper_block->can_break) {
						is_break_valid = true;
						break;
					}
					upper_block = upper_block->parent_block;
				}

				if (!is_break_valid) {
					_set_error("Unexpected keyword \"break\" outside a loop.");
					return;
				}

				_mark_line_as_safe(tokenizer->get_token_line());
				tokenizer->advance();
				ControlFlowNode *cf_break = alloc_node<ControlFlowNode>();
				cf_break->cf_type = ControlFlowNode::CF_BREAK;
				p_block->statements.push_back(cf_break);
				if (!_end_statement()) {
					_set_end_statement_error("break");
					return;
				}
			} break;
			case GDScriptTokenizer::TK_CF_RETURN: {

				tokenizer->advance();
				ControlFlowNode *cf_return = alloc_node<ControlFlowNode>();
				cf_return->cf_type = ControlFlowNode::CF_RETURN;
				cf_return->line = tokenizer->get_token_line(-1);

				if (tokenizer->get_token() == GDScriptTokenizer::TK_SEMICOLON || tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE || tokenizer->get_token() == GDScriptTokenizer::TK_EOF) {
					//expect end of statement
					p_block->statements.push_back(cf_return);
					if (!_end_statement()) {
						return;
					}
				} else {
					//expect expression
					Node *retexpr = _parse_and_reduce_expression(p_block, p_static);
					if (!retexpr) {
						if (_recover_from_completion()) {
							break;
						}
						return;
					}
					cf_return->arguments.push_back(retexpr);
					p_block->statements.push_back(cf_return);
					if (!_end_statement()) {
						_set_end_statement_error("return");
						return;
					}
				}
				p_block->has_return = true;

			} break;
			case GDScriptTokenizer::TK_CF_MATCH: {

				tokenizer->advance();

				MatchNode *match_node = alloc_node<MatchNode>();

				Node *val_to_match = _parse_and_reduce_expression(p_block, p_static);

				if (!val_to_match) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				match_node->val_to_match = val_to_match;

				if (!_enter_indent_block()) {
					_set_error("Expected indented pattern matching block after \"match\".");
					return;
				}

				BlockNode *compiled_branches = alloc_node<BlockNode>();
				compiled_branches->parent_block = p_block;
				compiled_branches->parent_class = p_block->parent_class;
				compiled_branches->can_continue = true;

				p_block->sub_blocks.push_back(compiled_branches);

				_parse_pattern_block(compiled_branches, match_node->branches, p_static);

				if (error_set) return;

				ControlFlowNode *match_cf_node = alloc_node<ControlFlowNode>();
				match_cf_node->cf_type = ControlFlowNode::CF_MATCH;
				match_cf_node->match = match_node;
				match_cf_node->body = compiled_branches;

				p_block->has_return = p_block->has_return || compiled_branches->has_return;
				p_block->statements.push_back(match_cf_node);

				_end_statement();
			} break;
			case GDScriptTokenizer::TK_PR_ASSERT: {

				tokenizer->advance();

				if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_OPEN) {
					_set_error("Expected '(' after assert");
					return;
				}

				int assert_line = tokenizer->get_token_line();

				tokenizer->advance();

				Vector<Node *> args;
				const bool result = _parse_arguments(p_block, args, p_static);
				if (!result) {
					return;
				}

				if (args.empty() || args.size() > 2) {
					_set_error("Wrong number of arguments, expected 1 or 2", assert_line);
					return;
				}

				AssertNode *an = alloc_node<AssertNode>();
				an->condition = _reduce_expression(args[0], p_static);
				an->line = assert_line;

				if (args.size() == 2) {
					an->message = _reduce_expression(args[1], p_static);
				} else {
					ConstantNode *message_node = alloc_node<ConstantNode>();
					message_node->value = String();
					an->message = message_node;
				}

				p_block->statements.push_back(an);

				if (!_end_statement()) {
					_set_end_statement_error("assert");
					return;
				}
			} break;
			case GDScriptTokenizer::TK_PR_BREAKPOINT: {

				tokenizer->advance();
				BreakpointNode *bn = alloc_node<BreakpointNode>();
				p_block->statements.push_back(bn);

				if (!_end_statement()) {
					_set_end_statement_error("breakpoint");
					return;
				}
			} break;
			default: {

				Node *expression = _parse_and_reduce_expression(p_block, p_static, false, true);
				if (!expression) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}
				p_block->statements.push_back(expression);
				if (!_end_statement()) {
					// Attempt to guess a better error message if the user "retypes" a variable
					if (tokenizer->get_token() == GDScriptTokenizer::TK_COLON && tokenizer->get_token(1) == GDScriptTokenizer::TK_OP_ASSIGN) {
						_set_error("Unexpected ':=', use '=' instead. Expected end of statement after expression.");
					} else {
						_set_error(vformat("Expected end of statement after expression, got %s instead.", tokenizer->get_token_name(tokenizer->get_token())));
					}
					return;
				}

			} break;
		}
	}
}

bool GDScriptParser::_parse_newline() {

	if (tokenizer->get_token(1) != GDScriptTokenizer::TK_EOF && tokenizer->get_token(1) != GDScriptTokenizer::TK_NEWLINE) {

		IndentLevel current_level = indent_level.back()->get();
		int indent = tokenizer->get_token_line_indent();
		int tabs = tokenizer->get_token_line_tab_indent();
		IndentLevel new_level(indent, tabs);

		if (new_level.is_mixed(current_level)) {
			_set_error("Mixed tabs and spaces in indentation.");
			return false;
		}

		if (indent > current_level.indent) {
			_set_error("Unexpected indentation.");
			return false;
		}

		if (indent < current_level.indent) {

			while (indent < current_level.indent) {

				//exit block
				if (indent_level.size() == 1) {
					_set_error("Invalid indentation. Bug?");
					return false;
				}

				indent_level.pop_back();

				if (indent_level.back()->get().indent < indent) {

					_set_error("Unindent does not match any outer indentation level.");
					return false;
				}

				if (indent_level.back()->get().is_mixed(current_level)) {
					_set_error("Mixed tabs and spaces in indentation.");
					return false;
				}

				current_level = indent_level.back()->get();
			}

			tokenizer->advance();
			return false;
		}
	}

	tokenizer->advance();
	return true;
}

void GDScriptParser::_parse_extends(ClassNode *p_class) {

	if (p_class->extends_used) {

		_set_error("\"extends\" can only be present once per script.");
		return;
	}

	if (!p_class->constant_expressions.empty() || !p_class->subclasses.empty() || !p_class->functions.empty() || !p_class->variables.empty()) {

		_set_error("\"extends\" must be used before anything else.");
		return;
	}

	p_class->extends_used = true;

	tokenizer->advance();

	if (tokenizer->get_token() == GDScriptTokenizer::TK_BUILT_IN_TYPE && tokenizer->get_token_type() == Variant::OBJECT) {
		p_class->extends_class.push_back(Variant::get_type_name(Variant::OBJECT));
		tokenizer->advance();
		return;
	}

	// see if inheritance happens from a file
	if (tokenizer->get_token() == GDScriptTokenizer::TK_CONSTANT) {

		Variant constant = tokenizer->get_token_constant();
		if (constant.get_type() != Variant::STRING) {

			_set_error("\"extends\" constant must be a string.");
			return;
		}

		p_class->extends_file = constant;
		tokenizer->advance();

		// Add parent script as a dependency
		String parent = constant;
		if (parent.is_rel_path()) {
			parent = base_path.plus_file(parent).simplify_path();
		}
		dependencies.push_back(parent);

		if (tokenizer->get_token() != GDScriptTokenizer::TK_PERIOD) {
			return;
		} else
			tokenizer->advance();
	}

	while (true) {

		switch (tokenizer->get_token()) {

			case GDScriptTokenizer::TK_IDENTIFIER: {
				StringName identifier = tokenizer->get_token_identifier();
				p_class->extends_class.push_back(identifier);
			} break;

			case GDScriptTokenizer::TK_PERIOD:
				break;

			default: {

				_set_error("Invalid \"extends\" syntax, expected string constant (path) and/or identifier (parent class).");
				return;
			}
		}

		tokenizer->advance(1);

		switch (tokenizer->get_token()) {

			case GDScriptTokenizer::TK_IDENTIFIER:
			case GDScriptTokenizer::TK_PERIOD:
				continue;
			case GDScriptTokenizer::TK_CURSOR:
				completion_type = COMPLETION_EXTENDS;
				completion_class = current_class;
				completion_function = current_function;
				completion_line = tokenizer->get_token_line();
				completion_block = current_block;
				completion_ident_is_call = false;
				completion_found = true;
				return;
			default:
				return;
		}
	}
}

void GDScriptParser::_parse_class(ClassNode *p_class) {

	IndentLevel current_level = indent_level.back()->get();

	while (true) {

		GDScriptTokenizer::Token token = tokenizer->get_token();
		if (error_set)
			return;

		if (current_level.indent > indent_level.back()->get().indent) {
			p_class->end_line = tokenizer->get_token_line();
			return; //go back a level
		}

		switch (token) {

			case GDScriptTokenizer::TK_CURSOR: {
				tokenizer->advance();
			} break;
			case GDScriptTokenizer::TK_EOF:
				p_class->end_line = tokenizer->get_token_line();
			case GDScriptTokenizer::TK_ERROR: {
				return; //go back
				//end of file!
			} break;
			case GDScriptTokenizer::TK_NEWLINE: {
				if (!_parse_newline()) {
					if (!error_set) {
						p_class->end_line = tokenizer->get_token_line();
					}
					return;
				}
			} break;
			case GDScriptTokenizer::TK_PR_EXTENDS: {

				_mark_line_as_safe(tokenizer->get_token_line());
				_parse_extends(p_class);
				if (error_set)
					return;
				if (!_end_statement()) {
					_set_end_statement_error("extends");
					return;
				}

			} break;
			case GDScriptTokenizer::TK_PR_CLASS_NAME: {

				_mark_line_as_safe(tokenizer->get_token_line());
				if (p_class->owner) {
					_set_error("\"class_name\" is only valid for the main class namespace.");
					return;
				}
				if (self_path.begins_with("res://") && self_path.find("::") != -1) {
					_set_error("\"class_name\" isn't allowed in built-in scripts.");
					return;
				}
				if (tokenizer->get_token(1) != GDScriptTokenizer::TK_IDENTIFIER) {

					_set_error("\"class_name\" syntax: \"class_name <UniqueName>\"");
					return;
				}
				if (p_class->classname_used) {
					_set_error("\"class_name\" can only be present once per script.");
					return;
				}

				p_class->classname_used = true;

				p_class->name = tokenizer->get_token_identifier(1);

				if (self_path != String() && ScriptServer::is_global_class(p_class->name) && ScriptServer::get_global_class_path(p_class->name) != self_path) {
					_set_error("Unique global class \"" + p_class->name + "\" already exists at path: " + ScriptServer::get_global_class_path(p_class->name));
					return;
				}

				if (ClassDB::class_exists(p_class->name)) {
					_set_error("The class \"" + p_class->name + "\" shadows a native class.");
					return;
				}

				if (p_class->classname_used && ProjectSettings::get_singleton()->has_setting("autoload/" + p_class->name)) {
					const String autoload_path = ProjectSettings::get_singleton()->get_setting("autoload/" + p_class->name);
					if (autoload_path.begins_with("*")) {
						// It's a singleton, and not just a regular AutoLoad script.
						_set_error("The class \"" + p_class->name + "\" conflicts with the AutoLoad singleton of the same name, and is therefore redundant. Remove the class_name declaration to fix this error.");
					}
					return;
				}

				tokenizer->advance(2);

				if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
					tokenizer->advance();

					if ((tokenizer->get_token() == GDScriptTokenizer::TK_CONSTANT && tokenizer->get_token_constant().get_type() == Variant::STRING)) {
#ifdef TOOLS_ENABLED
						if (Engine::get_singleton()->is_editor_hint()) {
							Variant constant = tokenizer->get_token_constant();
							String icon_path = constant.operator String();

							String abs_icon_path = icon_path.is_rel_path() ? self_path.get_base_dir().plus_file(icon_path).simplify_path() : icon_path;
							if (!FileAccess::exists(abs_icon_path)) {
								_set_error("No class icon found at: " + abs_icon_path);
								return;
							}

							p_class->icon_path = icon_path;
						}
#endif

						tokenizer->advance();
					} else {
						_set_error("The optional parameter after \"class_name\" must be a string constant file path to an icon.");
						return;
					}

				} else if (tokenizer->get_token() == GDScriptTokenizer::TK_CONSTANT) {
					_set_error("The class icon must be separated by a comma.");
					return;
				}

			} break;
			case GDScriptTokenizer::TK_PR_TOOL: {

				if (p_class->tool) {

					_set_error("The \"tool\" keyword can only be present once per script.");
					return;
				}

				p_class->tool = true;
				tokenizer->advance();

			} break;
			case GDScriptTokenizer::TK_PR_CLASS: {
				//class inside class :D

				StringName name;

				if (tokenizer->get_token(1) != GDScriptTokenizer::TK_IDENTIFIER) {

					_set_error("\"class\" syntax: \"class <Name>:\" or \"class <Name> extends <BaseClass>:\"");
					return;
				}
				name = tokenizer->get_token_identifier(1);
				tokenizer->advance(2);

				// Check if name is shadowing something else
				if (ClassDB::class_exists(name) || ClassDB::class_exists("_" + name.operator String())) {
					_set_error("The class \"" + String(name) + "\" shadows a native class.");
					return;
				}
				if (ScriptServer::is_global_class(name)) {
					_set_error("Can't override name of the unique global class \"" + name + "\". It already exists at: " + ScriptServer::get_global_class_path(p_class->name));
					return;
				}
				ClassNode *outer_class = p_class;
				while (outer_class) {
					for (int i = 0; i < outer_class->subclasses.size(); i++) {
						if (outer_class->subclasses[i]->name == name) {
							_set_error("Another class named \"" + String(name) + "\" already exists in this scope (at line " + itos(outer_class->subclasses[i]->line) + ").");
							return;
						}
					}
					if (outer_class->constant_expressions.has(name)) {
						_set_error("A constant named \"" + String(name) + "\" already exists in the outer class scope (at line" + itos(outer_class->constant_expressions[name].expression->line) + ").");
						return;
					}
					for (int i = 0; i < outer_class->variables.size(); i++) {
						if (outer_class->variables[i].identifier == name) {
							_set_error("A variable named \"" + String(name) + "\" already exists in the outer class scope (at line " + itos(outer_class->variables[i].line) + ").");
							return;
						}
					}

					outer_class = outer_class->owner;
				}

				ClassNode *newclass = alloc_node<ClassNode>();
				newclass->initializer = alloc_node<BlockNode>();
				newclass->initializer->parent_class = newclass;
				newclass->ready = alloc_node<BlockNode>();
				newclass->ready->parent_class = newclass;
				newclass->name = name;
				newclass->owner = p_class;

				p_class->subclasses.push_back(newclass);

				if (tokenizer->get_token() == GDScriptTokenizer::TK_PR_EXTENDS) {

					_parse_extends(newclass);
					if (error_set)
						return;
				}

				if (!_enter_indent_block()) {

					_set_error("Indented block expected.");
					return;
				}
				current_class = newclass;
				_parse_class(newclass);
				current_class = p_class;

			} break;
			/* this is for functions....
			case GDScriptTokenizer::TK_CF_PASS: {

				tokenizer->advance(1);
			} break;
			*/
			case GDScriptTokenizer::TK_PR_STATIC: {
				tokenizer->advance();
				if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {

					_set_error("Expected \"func\".");
					return;
				}

				FALLTHROUGH;
			}
			case GDScriptTokenizer::TK_PR_FUNCTION: {

				bool _static = false;
				pending_newline = -1;

				if (tokenizer->get_token(-1) == GDScriptTokenizer::TK_PR_STATIC) {

					_static = true;
				}

				tokenizer->advance();
				StringName name;

				if (_get_completable_identifier(COMPLETION_VIRTUAL_FUNC, name)) {
				}

				if (name == StringName()) {

					_set_error("Expected an identifier after \"func\" (syntax: \"func <identifier>([arguments]):\").");
					return;
				}

				for (int i = 0; i < p_class->functions.size(); i++) {
					if (p_class->functions[i]->name == name) {
						_set_error("The function \"" + String(name) + "\" already exists in this class (at line " + itos(p_class->functions[i]->line) + ").");
					}
				}
				for (int i = 0; i < p_class->static_functions.size(); i++) {
					if (p_class->static_functions[i]->name == name) {
						_set_error("The function \"" + String(name) + "\" already exists in this class (at line " + itos(p_class->static_functions[i]->line) + ").");
					}
				}

#ifdef DEBUG_ENABLED
				if (p_class->constant_expressions.has(name)) {
					_add_warning(GDScriptWarning::FUNCTION_CONFLICTS_CONSTANT, -1, name);
				}
				for (int i = 0; i < p_class->variables.size(); i++) {
					if (p_class->variables[i].identifier == name) {
						_add_warning(GDScriptWarning::FUNCTION_CONFLICTS_VARIABLE, -1, name);
					}
				}
				for (int i = 0; i < p_class->subclasses.size(); i++) {
					if (p_class->subclasses[i]->name == name) {
						_add_warning(GDScriptWarning::FUNCTION_CONFLICTS_CONSTANT, -1, name);
					}
				}
#endif // DEBUG_ENABLED

				if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_OPEN) {

					_set_error("Expected \"(\" after the identifier (syntax: \"func <identifier>([arguments]):\" ).");
					return;
				}

				tokenizer->advance();

				Vector<StringName> arguments;
				Vector<DataType> argument_types;
				Vector<Node *> default_values;
#ifdef DEBUG_ENABLED
				Vector<int> arguments_usage;
#endif // DEBUG_ENABLED

				int fnline = tokenizer->get_token_line();

				if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
					//has arguments
					bool defaulting = false;
					while (true) {

						if (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE) {
							tokenizer->advance();
							continue;
						}

						if (tokenizer->get_token() == GDScriptTokenizer::TK_PR_VAR) {

							tokenizer->advance(); //var before the identifier is allowed
						}

						if (!tokenizer->is_token_literal(0, true)) {

							_set_error("Expected an identifier for an argument.");
							return;
						}

						StringName argname = tokenizer->get_token_identifier();
						for (int i = 0; i < arguments.size(); i++) {
							if (arguments[i] == argname) {
								_set_error("The argument name \"" + String(argname) + "\" is defined multiple times.");
								return;
							}
						}
						arguments.push_back(argname);
#ifdef DEBUG_ENABLED
						arguments_usage.push_back(0);
#endif // DEBUG_ENABLED

						tokenizer->advance();

						DataType argtype;
						if (tokenizer->get_token() == GDScriptTokenizer::TK_COLON) {
							if (tokenizer->get_token(1) == GDScriptTokenizer::TK_OP_ASSIGN) {
								argtype.infer_type = true;
								tokenizer->advance();
							} else if (!_parse_type(argtype)) {
								_set_error("Expected a type for an argument.");
								return;
							}
						}
						argument_types.push_back(argtype);

						if (defaulting && tokenizer->get_token() != GDScriptTokenizer::TK_OP_ASSIGN) {

							_set_error("Default parameter expected.");
							return;
						}

						//tokenizer->advance();

						if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_ASSIGN) {
							defaulting = true;
							tokenizer->advance(1);
							Node *defval = _parse_and_reduce_expression(p_class, _static);
							if (!defval || error_set)
								return;

							OperatorNode *on = alloc_node<OperatorNode>();
							on->op = OperatorNode::OP_ASSIGN;
							on->line = fnline;

							IdentifierNode *in = alloc_node<IdentifierNode>();
							in->name = argname;
							in->line = fnline;

							on->arguments.push_back(in);
							on->arguments.push_back(defval);
							/* no ..
							if (defval->type!=Node::TYPE_CONSTANT) {

								_set_error("default argument must be constant");
							}
							*/
							default_values.push_back(on);
						}

						while (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE) {
							tokenizer->advance();
						}

						if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
							tokenizer->advance();
							continue;
						} else if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {

							_set_error("Expected \",\" or \")\".");
							return;
						}

						break;
					}
				}

				tokenizer->advance();

				BlockNode *block = alloc_node<BlockNode>();
				block->parent_class = p_class;

				FunctionNode *function = alloc_node<FunctionNode>();
				function->name = name;
				function->arguments = arguments;
				function->argument_types = argument_types;
				function->default_values = default_values;
				function->_static = _static;
				function->line = fnline;
#ifdef DEBUG_ENABLED
				function->arguments_usage = arguments_usage;
#endif // DEBUG_ENABLED
				function->rpc_mode = rpc_mode;
				rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;

				if (name == "_init") {

					if (_static) {
						_set_error("The constructor cannot be static.");
						return;
					}

					if (p_class->extends_used) {

						OperatorNode *cparent = alloc_node<OperatorNode>();
						cparent->op = OperatorNode::OP_PARENT_CALL;
						block->statements.push_back(cparent);

						IdentifierNode *id = alloc_node<IdentifierNode>();
						id->name = "_init";
						cparent->arguments.push_back(id);

						if (tokenizer->get_token() == GDScriptTokenizer::TK_PERIOD) {
							tokenizer->advance();
							if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_OPEN) {
								_set_error("Expected \"(\" for parent constructor arguments.");
								return;
							}
							tokenizer->advance();

							if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
								//has arguments
								parenthesis++;
								while (true) {

									current_function = function;
									Node *arg = _parse_and_reduce_expression(p_class, _static);
									if (!arg) {
										return;
									}
									current_function = NULL;
									cparent->arguments.push_back(arg);

									if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
										tokenizer->advance();
										continue;
									} else if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {

										_set_error("Expected \",\" or \")\".");
										return;
									}

									break;
								}
								parenthesis--;
							}

							tokenizer->advance();
						}
					} else {

						if (tokenizer->get_token() == GDScriptTokenizer::TK_PERIOD) {

							_set_error("Parent constructor call found for a class without inheritance.");
							return;
						}
					}
				}

				DataType return_type;
				if (tokenizer->get_token() == GDScriptTokenizer::TK_FORWARD_ARROW) {

					if (!_parse_type(return_type, true)) {
						_set_error("Expected a return type for the function.");
						return;
					}
				}

				if (!_enter_indent_block(block)) {

					_set_error(vformat("Indented block expected after declaration of \"%s\" function.", function->name));
					return;
				}

				function->return_type = return_type;

				if (_static)
					p_class->static_functions.push_back(function);
				else
					p_class->functions.push_back(function);

				current_function = function;
				function->body = block;
				current_block = block;
				_parse_block(block, _static);
				current_block = NULL;

				//arguments
			} break;
			case GDScriptTokenizer::TK_PR_SIGNAL: {
				tokenizer->advance();

				if (!tokenizer->is_token_literal()) {
					_set_error("Expected an identifier after \"signal\".");
					return;
				}

				ClassNode::Signal sig;
				sig.name = tokenizer->get_token_identifier();
				sig.emissions = 0;
				sig.line = tokenizer->get_token_line();

				for (int i = 0; i < current_class->_signals.size(); i++) {
					if (current_class->_signals[i].name == sig.name) {
						_set_error("The signal \"" + sig.name + "\" already exists in this class (at line: " + itos(current_class->_signals[i].line) + ").");
						return;
					}
				}

				tokenizer->advance();

				if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_OPEN) {
					tokenizer->advance();
					while (true) {
						if (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE) {
							tokenizer->advance();
							continue;
						}

						if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
							tokenizer->advance();
							break;
						}

						if (!tokenizer->is_token_literal(0, true)) {
							_set_error("Expected an identifier in a \"signal\" argument.");
							return;
						}

						sig.arguments.push_back(tokenizer->get_token_identifier());
						tokenizer->advance();

						while (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE) {
							tokenizer->advance();
						}

						if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
							tokenizer->advance();
						} else if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
							_set_error("Expected \",\" or \")\" after a \"signal\" parameter identifier.");
							return;
						}
					}
				}

				p_class->_signals.push_back(sig);

				if (!_end_statement()) {
					_set_end_statement_error("signal");
					return;
				}
			} break;
			case GDScriptTokenizer::TK_PR_EXPORT: {

				tokenizer->advance();

				if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_OPEN) {

#define _ADVANCE_AND_CONSUME_NEWLINES \
	do {                              \
		tokenizer->advance();         \
	} while (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE)

					_ADVANCE_AND_CONSUME_NEWLINES;
					parenthesis++;

					String hint_prefix = "";
					bool is_arrayed = false;

					while (tokenizer->get_token() == GDScriptTokenizer::TK_BUILT_IN_TYPE &&
							tokenizer->get_token_type() == Variant::ARRAY &&
							tokenizer->get_token(1) == GDScriptTokenizer::TK_COMMA) {
						tokenizer->advance(); // Array
						tokenizer->advance(); // Comma
						if (is_arrayed) {
							hint_prefix += itos(Variant::ARRAY) + ":";
						} else {
							is_arrayed = true;
						}
					}

					if (tokenizer->get_token() == GDScriptTokenizer::TK_BUILT_IN_TYPE) {

						Variant::Type type = tokenizer->get_token_type();
						if (type == Variant::NIL) {
							_set_error("Can't export null type.");
							return;
						}
						if (type == Variant::OBJECT) {
							_set_error("Can't export raw object type.");
							return;
						}
						current_export.type = type;
						current_export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
						_ADVANCE_AND_CONSUME_NEWLINES;

						if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
							// hint expected next!
							_ADVANCE_AND_CONSUME_NEWLINES;

							switch (type) {

								case Variant::INT: {

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "FLAGS") {

										_ADVANCE_AND_CONSUME_NEWLINES;

										if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											WARN_DEPRECATED_MSG("Exporting bit flags hint requires string constants.");
											break;
										}
										if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {
											_set_error("Expected \",\" in the bit flags hint.");
											return;
										}

										current_export.hint = PROPERTY_HINT_FLAGS;
										_ADVANCE_AND_CONSUME_NEWLINES;

										bool first = true;
										while (true) {

											if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {
												current_export = PropertyInfo();
												_set_error("Expected a string constant in the named bit flags hint.");
												return;
											}

											String c = tokenizer->get_token_constant();
											if (!first)
												current_export.hint_string += ",";
											else
												first = false;

											current_export.hint_string += c.xml_escape();

											_ADVANCE_AND_CONSUME_NEWLINES;
											if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
												break;

											if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {
												current_export = PropertyInfo();
												_set_error("Expected \")\" or \",\" in the named bit flags hint.");
												return;
											}
											_ADVANCE_AND_CONSUME_NEWLINES;
										}

										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "LAYERS_2D_RENDER") {

										_ADVANCE_AND_CONSUME_NEWLINES;
										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the layers 2D render hint.");
											return;
										}
										current_export.hint = PROPERTY_HINT_LAYERS_2D_RENDER;
										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "LAYERS_2D_PHYSICS") {

										_ADVANCE_AND_CONSUME_NEWLINES;
										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the layers 2D physics hint.");
											return;
										}
										current_export.hint = PROPERTY_HINT_LAYERS_2D_PHYSICS;
										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "LAYERS_3D_RENDER") {

										_ADVANCE_AND_CONSUME_NEWLINES;
										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the layers 3D render hint.");
											return;
										}
										current_export.hint = PROPERTY_HINT_LAYERS_3D_RENDER;
										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "LAYERS_3D_PHYSICS") {

										_ADVANCE_AND_CONSUME_NEWLINES;
										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the layers 3D physics hint.");
											return;
										}
										current_export.hint = PROPERTY_HINT_LAYERS_3D_PHYSICS;
										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_CONSTANT && tokenizer->get_token_constant().get_type() == Variant::STRING) {
										//enumeration
										current_export.hint = PROPERTY_HINT_ENUM;
										bool first = true;
										while (true) {

											if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {

												current_export = PropertyInfo();
												_set_error("Expected a string constant in the enumeration hint.");
												return;
											}

											String c = tokenizer->get_token_constant();
											if (!first)
												current_export.hint_string += ",";
											else
												first = false;

											current_export.hint_string += c.xml_escape();

											_ADVANCE_AND_CONSUME_NEWLINES;
											if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
												break;

											if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {
												current_export = PropertyInfo();
												_set_error("Expected \")\" or \",\" in the enumeration hint.");
												return;
											}

											_ADVANCE_AND_CONSUME_NEWLINES;
										}

										break;
									}

									FALLTHROUGH;
								}
								case Variant::REAL: {

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "EASE") {
										current_export.hint = PROPERTY_HINT_EXP_EASING;
										_ADVANCE_AND_CONSUME_NEWLINES;
										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the hint.");
											return;
										}
										break;
									}

									// range
									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "EXP") {

										current_export.hint = PROPERTY_HINT_EXP_RANGE;
										_ADVANCE_AND_CONSUME_NEWLINES;

										if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
											break;
										else if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {
											_set_error("Expected \")\" or \",\" in the exponential range hint.");
											return;
										}
										_ADVANCE_AND_CONSUME_NEWLINES;
									} else
										current_export.hint = PROPERTY_HINT_RANGE;

									float sign = 1.0;

									if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_SUB) {
										sign = -1;
										_ADVANCE_AND_CONSUME_NEWLINES;
									}
									if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || !tokenizer->get_token_constant().is_num()) {

										current_export = PropertyInfo();
										_set_error("Expected a range in the numeric hint.");
										return;
									}

									current_export.hint_string = rtos(sign * double(tokenizer->get_token_constant()));
									_ADVANCE_AND_CONSUME_NEWLINES;

									if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
										current_export.hint_string = "0," + current_export.hint_string;
										break;
									}

									if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {

										current_export = PropertyInfo();
										_set_error("Expected \",\" or \")\" in the numeric range hint.");
										return;
									}

									_ADVANCE_AND_CONSUME_NEWLINES;

									sign = 1.0;
									if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_SUB) {
										sign = -1;
										_ADVANCE_AND_CONSUME_NEWLINES;
									}

									if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || !tokenizer->get_token_constant().is_num()) {

										current_export = PropertyInfo();
										_set_error("Expected a number as upper bound in the numeric range hint.");
										return;
									}

									current_export.hint_string += "," + rtos(sign * double(tokenizer->get_token_constant()));
									_ADVANCE_AND_CONSUME_NEWLINES;

									if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
										break;

									if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {

										current_export = PropertyInfo();
										_set_error("Expected \",\" or \")\" in the numeric range hint.");
										return;
									}

									_ADVANCE_AND_CONSUME_NEWLINES;
									sign = 1.0;
									if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_SUB) {
										sign = -1;
										_ADVANCE_AND_CONSUME_NEWLINES;
									}

									if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || !tokenizer->get_token_constant().is_num()) {

										current_export = PropertyInfo();
										_set_error("Expected a number as step in the numeric range hint.");
										return;
									}

									current_export.hint_string += "," + rtos(sign * double(tokenizer->get_token_constant()));
									_ADVANCE_AND_CONSUME_NEWLINES;

								} break;
								case Variant::STRING: {

									if (tokenizer->get_token() == GDScriptTokenizer::TK_CONSTANT && tokenizer->get_token_constant().get_type() == Variant::STRING) {
										//enumeration
										current_export.hint = PROPERTY_HINT_ENUM;
										bool first = true;
										while (true) {

											if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {

												current_export = PropertyInfo();
												_set_error("Expected a string constant in the enumeration hint.");
												return;
											}

											String c = tokenizer->get_token_constant();
											if (!first)
												current_export.hint_string += ",";
											else
												first = false;

											current_export.hint_string += c.xml_escape();
											_ADVANCE_AND_CONSUME_NEWLINES;
											if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
												break;

											if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {
												current_export = PropertyInfo();
												_set_error("Expected \")\" or \",\" in the enumeration hint.");
												return;
											}
											_ADVANCE_AND_CONSUME_NEWLINES;
										}

										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "DIR") {

										_ADVANCE_AND_CONSUME_NEWLINES;

										if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
											current_export.hint = PROPERTY_HINT_DIR;
										else if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {

											_ADVANCE_AND_CONSUME_NEWLINES;

											if (tokenizer->get_token() != GDScriptTokenizer::TK_IDENTIFIER || !(tokenizer->get_token_identifier() == "GLOBAL")) {
												_set_error("Expected \"GLOBAL\" after comma in the directory hint.");
												return;
											}
											if (!p_class->tool) {
												_set_error("Global filesystem hints may only be used in tool scripts.");
												return;
											}
											current_export.hint = PROPERTY_HINT_GLOBAL_DIR;
											_ADVANCE_AND_CONSUME_NEWLINES;

											if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
												_set_error("Expected \")\" in the hint.");
												return;
											}
										} else {
											_set_error("Expected \")\" or \",\" in the hint.");
											return;
										}
										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "FILE") {

										current_export.hint = PROPERTY_HINT_FILE;
										_ADVANCE_AND_CONSUME_NEWLINES;

										if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {

											_ADVANCE_AND_CONSUME_NEWLINES;

											if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "GLOBAL") {

												if (!p_class->tool) {
													_set_error("Global filesystem hints may only be used in tool scripts.");
													return;
												}
												current_export.hint = PROPERTY_HINT_GLOBAL_FILE;
												_ADVANCE_AND_CONSUME_NEWLINES;

												if (tokenizer->get_token() == GDScriptTokenizer::TK_PARENTHESIS_CLOSE)
													break;
												else if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA)
													_ADVANCE_AND_CONSUME_NEWLINES;
												else {
													_set_error("Expected \")\" or \",\" in the hint.");
													return;
												}
											}

											if (tokenizer->get_token() != GDScriptTokenizer::TK_CONSTANT || tokenizer->get_token_constant().get_type() != Variant::STRING) {

												if (current_export.hint == PROPERTY_HINT_GLOBAL_FILE)
													_set_error("Expected string constant with filter.");
												else
													_set_error("Expected \"GLOBAL\" or string constant with filter.");
												return;
											}
											current_export.hint_string = tokenizer->get_token_constant();
											_ADVANCE_AND_CONSUME_NEWLINES;
										}

										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the hint.");
											return;
										}
										break;
									}

									if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "MULTILINE") {

										current_export.hint = PROPERTY_HINT_MULTILINE_TEXT;
										_ADVANCE_AND_CONSUME_NEWLINES;
										if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
											_set_error("Expected \")\" in the hint.");
											return;
										}
										break;
									}
								} break;
								case Variant::COLOR: {

									if (tokenizer->get_token() != GDScriptTokenizer::TK_IDENTIFIER) {

										current_export = PropertyInfo();
										_set_error("Color type hint expects RGB or RGBA as hints.");
										return;
									}

									String identifier = tokenizer->get_token_identifier();
									if (identifier == "RGB") {
										current_export.hint = PROPERTY_HINT_COLOR_NO_ALPHA;
									} else if (identifier == "RGBA") {
										//none
									} else {
										current_export = PropertyInfo();
										_set_error("Color type hint expects RGB or RGBA as hints.");
										return;
									}
									_ADVANCE_AND_CONSUME_NEWLINES;

								} break;
								default: {

									current_export = PropertyInfo();
									_set_error("Type \"" + Variant::get_type_name(type) + "\" can't take hints.");
									return;
								} break;
							}
						}

					} else {

						parenthesis++;
						Node *subexpr = _parse_and_reduce_expression(p_class, true, true);
						if (!subexpr) {
							if (_recover_from_completion()) {
								break;
							}
							return;
						}
						parenthesis--;

						if (subexpr->type != Node::TYPE_CONSTANT) {
							current_export = PropertyInfo();
							_set_error("Expected a constant expression.");
						}

						Variant constant = static_cast<ConstantNode *>(subexpr)->value;

						if (constant.get_type() == Variant::OBJECT) {
							GDScriptNativeClass *native_class = Object::cast_to<GDScriptNativeClass>(constant);

							if (native_class && ClassDB::is_parent_class(native_class->get_name(), "Resource")) {
								current_export.type = Variant::OBJECT;
								current_export.hint = PROPERTY_HINT_RESOURCE_TYPE;
								current_export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;

								current_export.hint_string = native_class->get_name();
								current_export.class_name = native_class->get_name();

							} else {
								current_export = PropertyInfo();
								_set_error("The export hint isn't a resource type.");
							}
						} else if (constant.get_type() == Variant::DICTIONARY) {
							// Enumeration
							bool is_flags = false;

							if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
								_ADVANCE_AND_CONSUME_NEWLINES;

								if (tokenizer->get_token() == GDScriptTokenizer::TK_IDENTIFIER && tokenizer->get_token_identifier() == "FLAGS") {
									is_flags = true;
									_ADVANCE_AND_CONSUME_NEWLINES;
								} else {
									current_export = PropertyInfo();
									_set_error("Expected \"FLAGS\" after comma.");
								}
							}

							current_export.type = Variant::INT;
							current_export.hint = is_flags ? PROPERTY_HINT_FLAGS : PROPERTY_HINT_ENUM;
							current_export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
							Dictionary enum_values = constant;

							List<Variant> keys;
							enum_values.get_key_list(&keys);

							bool first = true;
							for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
								if (enum_values[E->get()].get_type() == Variant::INT) {
									if (!first)
										current_export.hint_string += ",";
									else
										first = false;

									current_export.hint_string += E->get().operator String().camelcase_to_underscore(true).capitalize().xml_escape();
									if (!is_flags) {
										current_export.hint_string += ":";
										current_export.hint_string += enum_values[E->get()].operator String().xml_escape();
									}
								}
							}
						} else {
							current_export = PropertyInfo();
							_set_error("Expected type for export.");
							return;
						}
					}

					if (tokenizer->get_token() != GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {

						current_export = PropertyInfo();
						_set_error("Expected \")\" or \",\" after the export hint.");
						return;
					}

					tokenizer->advance();
					parenthesis--;

					if (is_arrayed) {
						hint_prefix += itos(current_export.type);
						if (current_export.hint) {
							hint_prefix += "/" + itos(current_export.hint);
						}
						current_export.hint_string = hint_prefix + ":" + current_export.hint_string;
						current_export.hint = PROPERTY_HINT_TYPE_STRING;
						current_export.type = Variant::ARRAY;
					}
#undef _ADVANCE_AND_CONSUME_NEWLINES
				}

				if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_ONREADY && tokenizer->get_token() != GDScriptTokenizer::TK_PR_REMOTE && tokenizer->get_token() != GDScriptTokenizer::TK_PR_MASTER && tokenizer->get_token() != GDScriptTokenizer::TK_PR_PUPPET && tokenizer->get_token() != GDScriptTokenizer::TK_PR_SYNC && tokenizer->get_token() != GDScriptTokenizer::TK_PR_REMOTESYNC && tokenizer->get_token() != GDScriptTokenizer::TK_PR_MASTERSYNC && tokenizer->get_token() != GDScriptTokenizer::TK_PR_PUPPETSYNC && tokenizer->get_token() != GDScriptTokenizer::TK_PR_SLAVE) {

					current_export = PropertyInfo();
					_set_error("Expected \"var\", \"onready\", \"remote\", \"master\", \"puppet\", \"sync\", \"remotesync\", \"mastersync\", \"puppetsync\".");
					return;
				}

				continue;
			} break;
			case GDScriptTokenizer::TK_PR_ONREADY: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR) {
					_set_error("Expected \"var\".");
					return;
				}

				continue;
			} break;
			case GDScriptTokenizer::TK_PR_REMOTE: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (current_export.type) {
					if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR) {
						_set_error("Expected \"var\".");
						return;
					}

				} else {
					if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {
						_set_error("Expected \"var\" or \"func\".");
						return;
					}
				}
				rpc_mode = MultiplayerAPI::RPC_MODE_REMOTE;

				continue;
			} break;
			case GDScriptTokenizer::TK_PR_MASTER: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (current_export.type) {
					if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR) {
						_set_error("Expected \"var\".");
						return;
					}

				} else {
					if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {
						_set_error("Expected \"var\" or \"func\".");
						return;
					}
				}

				rpc_mode = MultiplayerAPI::RPC_MODE_MASTER;
				continue;
			} break;
			case GDScriptTokenizer::TK_PR_SLAVE:
#ifdef DEBUG_ENABLED
				_add_warning(GDScriptWarning::DEPRECATED_KEYWORD, tokenizer->get_token_line(), "slave", "puppet");
#endif
				FALLTHROUGH;
			case GDScriptTokenizer::TK_PR_PUPPET: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (current_export.type) {
					if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR) {
						_set_error("Expected \"var\".");
						return;
					}

				} else {
					if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {
						_set_error("Expected \"var\" or \"func\".");
						return;
					}
				}

				rpc_mode = MultiplayerAPI::RPC_MODE_PUPPET;
				continue;
			} break;
			case GDScriptTokenizer::TK_PR_REMOTESYNC:
			case GDScriptTokenizer::TK_PR_SYNC: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {
					if (current_export.type)
						_set_error("Expected \"var\".");
					else
						_set_error("Expected \"var\" or \"func\".");
					return;
				}

				rpc_mode = MultiplayerAPI::RPC_MODE_REMOTESYNC;
				continue;
			} break;
			case GDScriptTokenizer::TK_PR_MASTERSYNC: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {
					if (current_export.type)
						_set_error("Expected \"var\".");
					else
						_set_error("Expected \"var\" or \"func\".");
					return;
				}

				rpc_mode = MultiplayerAPI::RPC_MODE_MASTERSYNC;
				continue;
			} break;
			case GDScriptTokenizer::TK_PR_PUPPETSYNC: {

				//may be fallthrough from export, ignore if so
				tokenizer->advance();
				if (tokenizer->get_token() != GDScriptTokenizer::TK_PR_VAR && tokenizer->get_token() != GDScriptTokenizer::TK_PR_FUNCTION) {
					if (current_export.type)
						_set_error("Expected \"var\".");
					else
						_set_error("Expected \"var\" or \"func\".");
					return;
				}

				rpc_mode = MultiplayerAPI::RPC_MODE_PUPPETSYNC;
				continue;
			} break;
			case GDScriptTokenizer::TK_PR_VAR: {
				// variable declaration and (eventual) initialization

				ClassNode::Member member;

				bool autoexport = tokenizer->get_token(-1) == GDScriptTokenizer::TK_PR_EXPORT;
				if (current_export.type != Variant::NIL) {
					member._export = current_export;
					current_export = PropertyInfo();
				}

				bool onready = tokenizer->get_token(-1) == GDScriptTokenizer::TK_PR_ONREADY;

				tokenizer->advance();
				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Expected an identifier for the member variable name.");
					return;
				}

				member.identifier = tokenizer->get_token_literal();
				member.expression = NULL;
				member._export.name = member.identifier;
				member.line = tokenizer->get_token_line();
				member.usages = 0;
				member.rpc_mode = rpc_mode;

				if (current_class->constant_expressions.has(member.identifier)) {
					_set_error("A constant named \"" + String(member.identifier) + "\" already exists in this class (at line: " +
							   itos(current_class->constant_expressions[member.identifier].expression->line) + ").");
					return;
				}

				for (int i = 0; i < current_class->variables.size(); i++) {
					if (current_class->variables[i].identifier == member.identifier) {
						_set_error("Variable \"" + String(member.identifier) + "\" already exists in this class (at line: " +
								   itos(current_class->variables[i].line) + ").");
						return;
					}
				}

				for (int i = 0; i < current_class->subclasses.size(); i++) {
					if (current_class->subclasses[i]->name == member.identifier) {
						_set_error("A class named \"" + String(member.identifier) + "\" already exists in this class (at line " + itos(current_class->subclasses[i]->line) + ").");
						return;
					}
				}
#ifdef DEBUG_ENABLED
				for (int i = 0; i < current_class->functions.size(); i++) {
					if (current_class->functions[i]->name == member.identifier) {
						_add_warning(GDScriptWarning::VARIABLE_CONFLICTS_FUNCTION, member.line, member.identifier);
						break;
					}
				}
				for (int i = 0; i < current_class->static_functions.size(); i++) {
					if (current_class->static_functions[i]->name == member.identifier) {
						_add_warning(GDScriptWarning::VARIABLE_CONFLICTS_FUNCTION, member.line, member.identifier);
						break;
					}
				}
#endif // DEBUG_ENABLED
				tokenizer->advance();

				rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;

				if (tokenizer->get_token() == GDScriptTokenizer::TK_COLON) {
					if (tokenizer->get_token(1) == GDScriptTokenizer::TK_OP_ASSIGN) {
						member.data_type = DataType();
#ifdef DEBUG_ENABLED
						member.data_type.infer_type = true;
#endif
						tokenizer->advance();
					} else if (!_parse_type(member.data_type)) {
						_set_error("Expected a type for the class variable.");
						return;
					}
				}

				if (autoexport && member.data_type.has_type) {
					if (member.data_type.kind == DataType::BUILTIN) {
						member._export.type = member.data_type.builtin_type;
					} else if (member.data_type.kind == DataType::NATIVE) {
						if (ClassDB::is_parent_class(member.data_type.native_type, "Resource")) {
							member._export.type = Variant::OBJECT;
							member._export.hint = PROPERTY_HINT_RESOURCE_TYPE;
							member._export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
							member._export.hint_string = member.data_type.native_type;
							member._export.class_name = member.data_type.native_type;
						} else {
							_set_error("Invalid export type. Only built-in and native resource types can be exported.", member.line);
							return;
						}

					} else {
						_set_error("Invalid export type. Only built-in and native resource types can be exported.", member.line);
						return;
					}
				}

#ifdef TOOLS_ENABLED
				Variant::CallError ce;
				member.default_value = Variant::construct(member._export.type, NULL, 0, ce);
#endif

				if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_ASSIGN) {

#ifdef DEBUG_ENABLED
					int line = tokenizer->get_token_line();
#endif
					tokenizer->advance();

					Node *subexpr = _parse_and_reduce_expression(p_class, false, autoexport || member._export.type != Variant::NIL);
					if (!subexpr) {
						if (_recover_from_completion()) {
							break;
						}
						return;
					}

					//discourage common error
					if (!onready && subexpr->type == Node::TYPE_OPERATOR) {

						OperatorNode *op = static_cast<OperatorNode *>(subexpr);
						if (op->op == OperatorNode::OP_CALL && op->arguments[0]->type == Node::TYPE_SELF && op->arguments[1]->type == Node::TYPE_IDENTIFIER) {
							IdentifierNode *id = static_cast<IdentifierNode *>(op->arguments[1]);
							if (id->name == "get_node") {

								_set_error("Use \"onready var " + String(member.identifier) + " = get_node(...)\" instead.");
								return;
							}
						}
					}

					member.expression = subexpr;

					if (autoexport && !member.data_type.has_type) {

						if (subexpr->type != Node::TYPE_CONSTANT) {

							_set_error("Type-less export needs a constant expression assigned to infer type.");
							return;
						}

						ConstantNode *cn = static_cast<ConstantNode *>(subexpr);
						if (cn->value.get_type() == Variant::NIL) {

							_set_error("Can't accept a null constant expression for inferring export type.");
							return;
						}

						if (!_reduce_export_var_type(cn->value, member.line)) return;

						member._export.type = cn->value.get_type();
						member._export.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
						if (cn->value.get_type() == Variant::OBJECT) {
							Object *obj = cn->value;
							Resource *res = Object::cast_to<Resource>(obj);
							if (res == NULL) {
								_set_error("The exported constant isn't a type or resource.");
								return;
							}
							member._export.hint = PROPERTY_HINT_RESOURCE_TYPE;
							member._export.hint_string = res->get_class();
						}
					}
#ifdef TOOLS_ENABLED
					if (subexpr->type == Node::TYPE_CONSTANT && (member._export.type != Variant::NIL || member.data_type.has_type)) {

						ConstantNode *cn = static_cast<ConstantNode *>(subexpr);
						if (cn->value.get_type() != Variant::NIL) {
							if (member._export.type != Variant::NIL && cn->value.get_type() != member._export.type) {
								if (Variant::can_convert(cn->value.get_type(), member._export.type)) {
									Variant::CallError err;
									const Variant *args = &cn->value;
									cn->value = Variant::construct(member._export.type, &args, 1, err);
								} else {
									_set_error("Can't convert the provided value to the export type.");
									return;
								}
							}
							member.default_value = cn->value;
						}
					}
#endif

					IdentifierNode *id = alloc_node<IdentifierNode>();
					id->name = member.identifier;
					id->datatype = member.data_type;

					OperatorNode *op = alloc_node<OperatorNode>();
					op->op = OperatorNode::OP_INIT_ASSIGN;
					op->arguments.push_back(id);
					op->arguments.push_back(subexpr);

#ifdef DEBUG_ENABLED
					NewLineNode *nl2 = alloc_node<NewLineNode>();
					nl2->line = line;
					if (onready)
						p_class->ready->statements.push_back(nl2);
					else
						p_class->initializer->statements.push_back(nl2);
#endif
					if (onready)
						p_class->ready->statements.push_back(op);
					else
						p_class->initializer->statements.push_back(op);

					member.initial_assignment = op;

				} else {

					if (autoexport && !member.data_type.has_type) {
						_set_error("Type-less export needs a constant expression assigned to infer type.");
						return;
					}

					Node *expr;

					if (member.data_type.has_type) {
						expr = _get_default_value_for_type(member.data_type);
					} else {
						DataType exported_type;
						exported_type.has_type = true;
						exported_type.kind = DataType::BUILTIN;
						exported_type.builtin_type = member._export.type;
						expr = _get_default_value_for_type(exported_type);
					}

					IdentifierNode *id = alloc_node<IdentifierNode>();
					id->name = member.identifier;
					id->datatype = member.data_type;

					OperatorNode *op = alloc_node<OperatorNode>();
					op->op = OperatorNode::OP_INIT_ASSIGN;
					op->arguments.push_back(id);
					op->arguments.push_back(expr);

					p_class->initializer->statements.push_back(op);

					member.initial_assignment = op;
				}

				if (tokenizer->get_token() == GDScriptTokenizer::TK_PR_SETGET) {

					tokenizer->advance();

					if (tokenizer->get_token() != GDScriptTokenizer::TK_COMMA) {
						//just comma means using only getter
						if (!tokenizer->is_token_literal()) {
							_set_error("Expected an identifier for the setter function after \"setget\".");
						}

						member.setter = tokenizer->get_token_literal();

						tokenizer->advance();
					}

					if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
						//there is a getter
						tokenizer->advance();

						if (!tokenizer->is_token_literal()) {
							_set_error("Expected an identifier for the getter function after \",\".");
						}

						member.getter = tokenizer->get_token_literal();
						tokenizer->advance();
					}
				}

				p_class->variables.push_back(member);

				if (!_end_statement()) {
					_set_end_statement_error("var");
					return;
				}
			} break;
			case GDScriptTokenizer::TK_PR_CONST: {
				// constant declaration and initialization

				ClassNode::Constant constant;

				tokenizer->advance();
				if (!tokenizer->is_token_literal(0, true)) {

					_set_error("Expected an identifier for the constant.");
					return;
				}

				StringName const_id = tokenizer->get_token_literal();
				int line = tokenizer->get_token_line();

				if (current_class->constant_expressions.has(const_id)) {
					_set_error("Constant \"" + String(const_id) + "\" already exists in this class (at line " +
							   itos(current_class->constant_expressions[const_id].expression->line) + ").");
					return;
				}

				for (int i = 0; i < current_class->variables.size(); i++) {
					if (current_class->variables[i].identifier == const_id) {
						_set_error("A variable named \"" + String(const_id) + "\" already exists in this class (at line " +
								   itos(current_class->variables[i].line) + ").");
						return;
					}
				}

				for (int i = 0; i < current_class->subclasses.size(); i++) {
					if (current_class->subclasses[i]->name == const_id) {
						_set_error("A class named \"" + String(const_id) + "\" already exists in this class (at line " + itos(current_class->subclasses[i]->line) + ").");
						return;
					}
				}

				tokenizer->advance();

				if (tokenizer->get_token() == GDScriptTokenizer::TK_COLON) {
					if (tokenizer->get_token(1) == GDScriptTokenizer::TK_OP_ASSIGN) {
						constant.type = DataType();
#ifdef DEBUG_ENABLED
						constant.type.infer_type = true;
#endif
						tokenizer->advance();
					} else if (!_parse_type(constant.type)) {
						_set_error("Expected a type for the class constant.");
						return;
					}
				}

				if (tokenizer->get_token() != GDScriptTokenizer::TK_OP_ASSIGN) {
					_set_error("Constants must be assigned immediately.");
					return;
				}

				tokenizer->advance();

				Node *subexpr = _parse_and_reduce_expression(p_class, true, true);
				if (!subexpr) {
					if (_recover_from_completion()) {
						break;
					}
					return;
				}

				if (subexpr->type != Node::TYPE_CONSTANT) {
					_set_error("Expected a constant expression.", line);
					return;
				}
				subexpr->line = line;
				constant.expression = subexpr;

				p_class->constant_expressions.insert(const_id, constant);

				if (!_end_statement()) {
					_set_end_statement_error("const");
					return;
				}

			} break;
			case GDScriptTokenizer::TK_PR_ENUM: {
				//multiple constant declarations..

				int last_assign = -1; // Incremented by 1 right before the assignment.
				String enum_name;
				Dictionary enum_dict;
				int enum_start_line = tokenizer->get_token_line();

				tokenizer->advance();
				if (tokenizer->is_token_literal(0, true)) {
					enum_name = tokenizer->get_token_literal();

					if (current_class->constant_expressions.has(enum_name)) {
						_set_error("A constant named \"" + String(enum_name) + "\" already exists in this class (at line " +
								   itos(current_class->constant_expressions[enum_name].expression->line) + ").");
						return;
					}

					for (int i = 0; i < current_class->variables.size(); i++) {
						if (current_class->variables[i].identifier == enum_name) {
							_set_error("A variable named \"" + String(enum_name) + "\" already exists in this class (at line " +
									   itos(current_class->variables[i].line) + ").");
							return;
						}
					}

					for (int i = 0; i < current_class->subclasses.size(); i++) {
						if (current_class->subclasses[i]->name == enum_name) {
							_set_error("A class named \"" + String(enum_name) + "\" already exists in this class (at line " + itos(current_class->subclasses[i]->line) + ").");
							return;
						}
					}

					tokenizer->advance();
				}
				if (tokenizer->get_token() != GDScriptTokenizer::TK_CURLY_BRACKET_OPEN) {
					_set_error("Expected \"{\" in the enum declaration.");
					return;
				}
				tokenizer->advance();

				while (true) {
					if (tokenizer->get_token() == GDScriptTokenizer::TK_NEWLINE) {

						tokenizer->advance(); // Ignore newlines
					} else if (tokenizer->get_token() == GDScriptTokenizer::TK_CURLY_BRACKET_CLOSE) {

						tokenizer->advance();
						break; // End of enum
					} else if (!tokenizer->is_token_literal(0, true)) {

						if (tokenizer->get_token() == GDScriptTokenizer::TK_EOF) {
							_set_error("Unexpected end of file.");
						} else {
							_set_error(String("Unexpected ") + GDScriptTokenizer::get_token_name(tokenizer->get_token()) + ", expected an identifier.");
						}

						return;
					} else { // tokenizer->is_token_literal(0, true)
						StringName const_id = tokenizer->get_token_literal();

						tokenizer->advance();

						ConstantNode *enum_value_expr;

						if (tokenizer->get_token() == GDScriptTokenizer::TK_OP_ASSIGN) {
							tokenizer->advance();

							Node *subexpr = _parse_and_reduce_expression(p_class, true, true);
							if (!subexpr) {
								if (_recover_from_completion()) {
									break;
								}
								return;
							}

							if (subexpr->type != Node::TYPE_CONSTANT) {
								_set_error("Expected a constant expression.");
								return;
							}

							enum_value_expr = static_cast<ConstantNode *>(subexpr);

							if (enum_value_expr->value.get_type() != Variant::INT) {
								_set_error("Expected an integer value for \"enum\".");
								return;
							}

							last_assign = enum_value_expr->value;

						} else {
							last_assign = last_assign + 1;
							enum_value_expr = alloc_node<ConstantNode>();
							enum_value_expr->value = last_assign;
							enum_value_expr->datatype = _type_from_variant(enum_value_expr->value);
						}

						if (tokenizer->get_token() == GDScriptTokenizer::TK_COMMA) {
							tokenizer->advance();
						} else if (tokenizer->is_token_literal(0, true)) {
							_set_error("Unexpected identifier.");
							return;
						}

						if (enum_name != "") {
							enum_dict[const_id] = enum_value_expr->value;
						} else {
							if (current_class->constant_expressions.has(const_id)) {
								_set_error("A constant named \"" + String(const_id) + "\" already exists in this class (at line " +
										   itos(current_class->constant_expressions[const_id].expression->line) + ").");
								return;
							}

							for (int i = 0; i < current_class->variables.size(); i++) {
								if (current_class->variables[i].identifier == const_id) {
									_set_error("A variable named \"" + String(const_id) + "\" already exists in this class (at line " +
											   itos(current_class->variables[i].line) + ").");
									return;
								}
							}

							for (int i = 0; i < current_class->subclasses.size(); i++) {
								if (current_class->subclasses[i]->name == const_id) {
									_set_error("A class named \"" + String(const_id) + "\" already exists in this class (at line " + itos(current_class->subclasses[i]->line) + ").");
									return;
								}
							}

							ClassNode::Constant constant;
							constant.type.has_type = true;
							constant.type.kind = DataType::BUILTIN;
							constant.type.builtin_type = Variant::INT;
							constant.expression = enum_value_expr;
							p_class->constant_expressions.insert(const_id, constant);
						}
					}
				}

				if (enum_name != "") {
					ClassNode::Constant enum_constant;
					ConstantNode *cn = alloc_node<ConstantNode>();
					cn->value = enum_dict;
					cn->datatype = _type_from_variant(cn->value);
					cn->line = enum_start_line;

					enum_constant.expression = cn;
					enum_constant.type = cn->datatype;
					p_class->constant_expressions.insert(enum_name, enum_constant);
				}

				if (!_end_statement()) {
					_set_end_statement_error("enum");
					return;
				}

			} break;

			case GDScriptTokenizer::TK_CONSTANT: {
				if (tokenizer->get_token_constant().get_type() == Variant::STRING) {
					tokenizer->advance();
					// Ignore
				} else {
					_set_error(String() + "Unexpected constant of type: " + Variant::get_type_name(tokenizer->get_token_constant().get_type()));
					return;
				}
			} break;

			case GDScriptTokenizer::TK_CF_PASS: {
				tokenizer->advance();
			} break;

			default: {

				if (token == GDScriptTokenizer::TK_IDENTIFIER) {
					completion_type = COMPLETION_IDENTIFIER;
					completion_class = current_class;
					completion_function = current_function;
					completion_line = tokenizer->get_token_line();
					completion_block = current_block;
					completion_ident_is_call = false;
					completion_found = true;
				}

				_set_error(String() + "Unexpected token: " + tokenizer->get_token_name(tokenizer->get_token()) + ":" + tokenizer->get_token_identifier());
				return;

			} break;
		}
	}
}

void GDScriptParser::_determine_inheritance(ClassNode *p_class, bool p_recursive) {

	if (p_class->base_type.has_type) {
		// Already determined
	} else if (p_class->extends_used) {
		//do inheritance
		String path = p_class->extends_file;

		Ref<GDScript> script;
		StringName native;
		ClassNode *base_class = NULL;

		if (path != "") {
			//path (and optionally subclasses)

			if (path.is_rel_path()) {

				String base = base_path;

				if (base == "" || base.is_rel_path()) {
					_set_error("Couldn't resolve relative path for the parent class: " + path, p_class->line);
					return;
				}
				path = base.plus_file(path).simplify_path();
			}
			script = ResourceLoader::load(path);
			if (script.is_null()) {
				_set_error("Couldn't load the base class: " + path, p_class->line);
				return;
			}
			if (!script->is_valid()) {

				_set_error("Script isn't fully loaded (cyclic preload?): " + path, p_class->line);
				return;
			}

			if (p_class->extends_class.size()) {

				for (int i = 0; i < p_class->extends_class.size(); i++) {

					String sub = p_class->extends_class[i];
					if (script->get_subclasses().has(sub)) {

						Ref<Script> subclass = script->get_subclasses()[sub]; //avoid reference from disappearing
						script = subclass;
					} else {

						_set_error("Couldn't find the subclass: " + sub, p_class->line);
						return;
					}
				}
			}

		} else {

			if (p_class->extends_class.size() == 0) {
				_set_error("Parser bug: undecidable inheritance.", p_class->line);
				ERR_FAIL();
			}
			//look around for the subclasses

			int extend_iter = 1;
			String base = p_class->extends_class[0];
			ClassNode *p = p_class->owner;
			Ref<GDScript> base_script;

			if (ScriptServer::is_global_class(base)) {
				base_script = ResourceLoader::load(ScriptServer::get_global_class_path(base));
				if (!base_script.is_valid()) {
					_set_error("The class \"" + base + "\" couldn't be fully loaded (script error or cyclic dependency).", p_class->line);
					return;
				}
				p = NULL;
			} else {
				List<PropertyInfo> props;
				ProjectSettings::get_singleton()->get_property_list(&props);
				for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
					String s = E->get().name;
					if (!s.begins_with("autoload/")) {
						continue;
					}
					String name = s.get_slice("/", 1);
					if (name == base) {
						String singleton_path = ProjectSettings::get_singleton()->get(s);
						if (singleton_path.begins_with("*")) {
							singleton_path = singleton_path.right(1);
						}
						if (!singleton_path.begins_with("res://")) {
							singleton_path = "res://" + singleton_path;
						}
						base_script = ResourceLoader::load(singleton_path);
						if (!base_script.is_valid()) {
							_set_error("Class '" + base + "' could not be fully loaded (script error or cyclic inheritance).", p_class->line);
							return;
						}
						p = NULL;
					}
				}
			}

			while (p) {

				bool found = false;

				for (int i = 0; i < p->subclasses.size(); i++) {
					if (p->subclasses[i]->name == base) {
						ClassNode *test = p->subclasses[i];
						while (test) {
							if (test == p_class) {
								_set_error("Cyclic inheritance.", test->line);
								return;
							}
							if (test->base_type.kind == DataType::CLASS) {
								test = test->base_type.class_type;
							} else {
								break;
							}
						}
						found = true;
						if (extend_iter < p_class->extends_class.size()) {
							// Keep looking at current classes if possible
							base = p_class->extends_class[extend_iter++];
							p = p->subclasses[i];
						} else {
							base_class = p->subclasses[i];
						}
						break;
					}
				}

				if (base_class) break;
				if (found) continue;

				if (p->constant_expressions.has(base)) {
					if (p->constant_expressions[base].expression->type != Node::TYPE_CONSTANT) {
						_set_error("Couldn't resolve the constant \"" + base + "\".", p_class->line);
						return;
					}
					const ConstantNode *cn = static_cast<const ConstantNode *>(p->constant_expressions[base].expression);
					base_script = cn->value;
					if (base_script.is_null()) {
						_set_error("Constant isn't a class: " + base, p_class->line);
						return;
					}
					break;
				}

				p = p->owner;
			}

			if (base_script.is_valid()) {

				String ident = base;
				Ref<GDScript> find_subclass = base_script;

				for (int i = extend_iter; i < p_class->extends_class.size(); i++) {

					String subclass = p_class->extends_class[i];

					ident += ("." + subclass);

					if (find_subclass->get_subclasses().has(subclass)) {

						find_subclass = find_subclass->get_subclasses()[subclass];
					} else if (find_subclass->get_constants().has(subclass)) {

						Ref<GDScript> new_base_class = find_subclass->get_constants()[subclass];
						if (new_base_class.is_null()) {
							_set_error("Constant isn't a class: " + ident, p_class->line);
							return;
						}
						find_subclass = new_base_class;
					} else {

						_set_error("Couldn't find the subclass: " + ident, p_class->line);
						return;
					}
				}

				script = find_subclass;

			} else if (!base_class) {

				if (p_class->extends_class.size() > 1) {

					_set_error("Invalid inheritance (unknown class + subclasses).", p_class->line);
					return;
				}
				//if not found, try engine classes
				if (!GDScriptLanguage::get_singleton()->get_global_map().has(base)) {

					_set_error("Unknown class: \"" + base + "\"", p_class->line);
					return;
				}

				native = base;
			}
		}

		if (base_class) {
			p_class->base_type.has_type = true;
			p_class->base_type.kind = DataType::CLASS;
			p_class->base_type.class_type = base_class;
		} else if (script.is_valid()) {
			p_class->base_type.has_type = true;
			p_class->base_type.kind = DataType::GDSCRIPT;
			p_class->base_type.script_type = script;
			p_class->base_type.native_type = script->get_instance_base_type();
		} else if (native != StringName()) {
			p_class->base_type.has_type = true;
			p_class->base_type.kind = DataType::NATIVE;
			p_class->base_type.native_type = native;
		} else {
			_set_error("Couldn't determine inheritance.", p_class->line);
			return;
		}

	} else {
		// without extends, implicitly extend Reference
		p_class->base_type.has_type = true;
		p_class->base_type.kind = DataType::NATIVE;
		p_class->base_type.native_type = "Reference";
	}

	if (p_recursive) {
		// Recursively determine subclasses
		for (int i = 0; i < p_class->subclasses.size(); i++) {
			_determine_inheritance(p_class->subclasses[i], p_recursive);
		}
	}
}

String GDScriptParser::DataType::to_string() const {
	if (!has_type) return "var";
	switch (kind) {
		case BUILTIN: {
			if (builtin_type == Variant::NIL) return "null";
			return Variant::get_type_name(builtin_type);
		} break;
		case NATIVE: {
			if (is_meta_type) {
				return "GDScriptNativeClass";
			}
			return native_type.operator String();
		} break;

		case GDSCRIPT: {
			Ref<GDScript> gds = script_type;
			const String &gds_class = gds->get_script_class_name();
			if (!gds_class.empty()) {
				return gds_class;
			}
			FALLTHROUGH;
		}
		case SCRIPT: {
			if (is_meta_type) {
				return script_type->get_class_name().operator String();
			}
			String name = script_type->get_name();
			if (name != String()) {
				return name;
			}
			name = script_type->get_path().get_file();
			if (name != String()) {
				return name;
			}
			return native_type.operator String();
		} break;
		case CLASS: {
			ERR_FAIL_COND_V(!class_type, String());
			if (is_meta_type) {
				return "GDScript";
			}
			if (class_type->name == StringName()) {
				return "self";
			}
			return class_type->name.operator String();
		} break;
		case UNRESOLVED: {
		} break;
	}

	return "Unresolved";
}

bool GDScriptParser::_parse_type(DataType &r_type, bool p_can_be_void) {
	tokenizer->advance();
	r_type.has_type = true;

	bool finished = false;
	bool can_index = false;
	String full_name;

	if (tokenizer->get_token() == GDScriptTokenizer::TK_CURSOR) {
		completion_cursor = StringName();
		completion_type = COMPLETION_TYPE_HINT;
		completion_class = current_class;
		completion_function = current_function;
		completion_line = tokenizer->get_token_line();
		completion_argument = 0;
		completion_block = current_block;
		completion_found = true;
		completion_ident_is_call = p_can_be_void;
		tokenizer->advance();
	}

	switch (tokenizer->get_token()) {
		case GDScriptTokenizer::TK_PR_VOID: {
			if (!p_can_be_void) {
				return false;
			}
			r_type.kind = DataType::BUILTIN;
			r_type.builtin_type = Variant::NIL;
		} break;
		case GDScriptTokenizer::TK_BUILT_IN_TYPE: {
			r_type.builtin_type = tokenizer->get_token_type();
			if (tokenizer->get_token_type() == Variant::OBJECT) {
				r_type.kind = DataType::NATIVE;
				r_type.native_type = "Object";
			} else {
				r_type.kind = DataType::BUILTIN;
			}
		} break;
		case GDScriptTokenizer::TK_IDENTIFIER: {
			r_type.native_type = tokenizer->get_token_identifier();
			if (ClassDB::class_exists(r_type.native_type) || ClassDB::class_exists("_" + r_type.native_type.operator String())) {
				r_type.kind = DataType::NATIVE;
			} else {
				r_type.kind = DataType::UNRESOLVED;
				can_index = true;
				full_name = r_type.native_type;
			}
		} break;
		default: {
			return false;
		}
	}

	tokenizer->advance();

	if (tokenizer->get_token() == GDScriptTokenizer::TK_CURSOR) {
		completion_cursor = r_type.native_type;
		completion_type = COMPLETION_TYPE_HINT;
		completion_class = current_class;
		completion_function = current_function;
		completion_line = tokenizer->get_token_line();
		completion_argument = 0;
		completion_block = current_block;
		completion_found = true;
		completion_ident_is_call = p_can_be_void;
		tokenizer->advance();
	}

	if (can_index) {
		while (!finished) {
			switch (tokenizer->get_token()) {
				case GDScriptTokenizer::TK_PERIOD: {
					if (!can_index) {
						_set_error("Unexpected \".\".");
						return false;
					}
					can_index = false;
					tokenizer->advance();
				} break;
				case GDScriptTokenizer::TK_IDENTIFIER: {
					if (can_index) {
						_set_error("Unexpected identifier.");
						return false;
					}

					StringName id;
					bool has_completion = _get_completable_identifier(COMPLETION_TYPE_HINT_INDEX, id);
					if (id == StringName()) {
						id = "@temp";
					}

					full_name += "." + id.operator String();
					can_index = true;
					if (has_completion) {
						completion_cursor = full_name;
					}
				} break;
				default: {
					finished = true;
				} break;
			}
		}

		if (tokenizer->get_token(-1) == GDScriptTokenizer::TK_PERIOD) {
			_set_error("Expected a subclass identifier.");
			return false;
		}

		r_type.native_type = full_name;
	}

	return true;
}

GDScriptParser::DataType GDScriptParser::_resolve_type(const DataType &p_source, int p_line) {
	if (!p_source.has_type) return p_source;
	if (p_source.kind != DataType::UNRESOLVED) return p_source;

	Vector<String> full_name = p_source.native_type.operator String().split(".", false);
	int name_part = 0;

	DataType result;
	result.has_type = true;

	while (name_part < full_name.size()) {

		bool found = false;
		StringName id = full_name[name_part];
		DataType base_type = result;

		ClassNode *p = NULL;
		if (name_part == 0) {
			if (ScriptServer::is_global_class(id)) {
				String script_path = ScriptServer::get_global_class_path(id);
				if (script_path == self_path) {
					result.kind = DataType::CLASS;
					result.class_type = static_cast<ClassNode *>(head);
				} else {
					Ref<Script> script = ResourceLoader::load(script_path);
					Ref<GDScript> gds = script;
					if (gds.is_valid()) {
						if (!gds->is_valid()) {
							_set_error("The class \"" + id + "\" couldn't be fully loaded (script error or cyclic dependency).", p_line);
							return DataType();
						}
						result.kind = DataType::GDSCRIPT;
						result.script_type = gds;
					} else if (script.is_valid()) {
						result.kind = DataType::SCRIPT;
						result.script_type = script;
					} else {
						_set_error("The class \"" + id + "\" was found in global scope, but its script couldn't be loaded.", p_line);
						return DataType();
					}
				}
				name_part++;
				continue;
			}
			List<PropertyInfo> props;
			ProjectSettings::get_singleton()->get_property_list(&props);
			String singleton_path;
			for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
				String s = E->get().name;
				if (!s.begins_with("autoload/")) {
					continue;
				}
				String name = s.get_slice("/", 1);
				if (name == id) {
					singleton_path = ProjectSettings::get_singleton()->get(s);
					if (singleton_path.begins_with("*")) {
						singleton_path = singleton_path.right(1);
					}
					if (!singleton_path.begins_with("res://")) {
						singleton_path = "res://" + singleton_path;
					}
					break;
				}
			}
			if (!singleton_path.empty()) {
				Ref<Script> script = ResourceLoader::load(singleton_path);
				Ref<GDScript> gds = script;
				if (gds.is_valid()) {
					if (!gds->is_valid()) {
						_set_error("Class '" + id + "' could not be fully loaded (script error or cyclic inheritance).", p_line);
						return DataType();
					}
					result.kind = DataType::GDSCRIPT;
					result.script_type = gds;
				} else if (script.is_valid()) {
					result.kind = DataType::SCRIPT;
					result.script_type = script;
				} else {
					_set_error("Couldn't fully load singleton script '" + id + "' (possible cyclic reference or parse error).", p_line);
					return DataType();
				}
				name_part++;
				continue;
			}

			p = current_class;
		} else if (base_type.kind == DataType::CLASS) {
			p = base_type.class_type;
		}
		while (p) {
			if (p->constant_expressions.has(id)) {
				if (p->constant_expressions[id].expression->type != Node::TYPE_CONSTANT) {
					_set_error("Parser bug: unresolved constant.", p_line);
					ERR_FAIL_V(result);
				}
				const ConstantNode *cn = static_cast<const ConstantNode *>(p->constant_expressions[id].expression);
				Ref<GDScript> gds = cn->value;
				if (gds.is_valid()) {
					result.kind = DataType::GDSCRIPT;
					result.script_type = gds;
					found = true;
				} else {
					Ref<Script> scr = cn->value;
					if (scr.is_valid()) {
						result.kind = DataType::SCRIPT;
						result.script_type = scr;
						found = true;
					}
				}
				break;
			}

			// Inner classes
			ClassNode *outer_class = p;
			while (outer_class) {
				if (outer_class->name == id) {
					found = true;
					result.kind = DataType::CLASS;
					result.class_type = outer_class;
					break;
				}
				for (int i = 0; i < outer_class->subclasses.size(); i++) {
					if (outer_class->subclasses[i] == p) {
						continue;
					}
					if (outer_class->subclasses[i]->name == id) {
						found = true;
						result.kind = DataType::CLASS;
						result.class_type = outer_class->subclasses[i];
						break;
					}
				}
				if (found) {
					break;
				}
				outer_class = outer_class->owner;
			}

			if (!found && p->base_type.kind == DataType::CLASS) {
				p = p->base_type.class_type;
			} else {
				base_type = p->base_type;
				break;
			}
		}

		// Still look for class constants in parent scripts
		if (!found && (base_type.kind == DataType::GDSCRIPT || base_type.kind == DataType::SCRIPT)) {
			Ref<Script> scr = base_type.script_type;
			ERR_FAIL_COND_V(scr.is_null(), result);
			while (scr.is_valid()) {
				Map<StringName, Variant> constants;
				scr->get_constants(&constants);

				if (constants.has(id)) {
					Ref<GDScript> gds = constants[id];

					if (gds.is_valid()) {
						result.kind = DataType::GDSCRIPT;
						result.script_type = gds;
						found = true;
					} else {
						Ref<Script> scr2 = constants[id];
						if (scr2.is_valid()) {
							result.kind = DataType::SCRIPT;
							result.script_type = scr2;
							found = true;
						}
					}
				}
				if (found) {
					break;
				} else {
					scr = scr->get_base_script();
				}
			}
		}

		if (!found && !for_completion) {
			String base;
			if (name_part == 0) {
				base = "self";
			} else {
				base = result.to_string();
			}
			_set_error("The identifier \"" + String(id) + "\" isn't a valid type (not a script or class), or couldn't be found on base \"" +
							   base + "\".",
					p_line);
			return DataType();
		}

		name_part++;
	}

	return result;
}

GDScriptParser::DataType GDScriptParser::_type_from_variant(const Variant &p_value) const {
	DataType result;
	result.has_type = true;
	result.is_constant = true;
	result.kind = DataType::BUILTIN;
	result.builtin_type = p_value.get_type();

	if (result.builtin_type == Variant::OBJECT) {
		Object *obj = p_value.operator Object *();
		if (!obj) {
			return DataType();
		}
		result.native_type = obj->get_class_name();
		Ref<Script> scr = p_value;
		if (scr.is_valid()) {
			result.is_meta_type = true;
		} else {
			result.is_meta_type = false;
			scr = obj->get_script();
		}
		if (scr.is_valid()) {
			result.script_type = scr;
			Ref<GDScript> gds = scr;
			if (gds.is_valid()) {
				result.kind = DataType::GDSCRIPT;
			} else {
				result.kind = DataType::SCRIPT;
			}
			result.native_type = scr->get_instance_base_type();
		} else {
			result.kind = DataType::NATIVE;
		}
	}

	return result;
}

GDScriptParser::DataType GDScriptParser::_type_from_property(const PropertyInfo &p_property, bool p_nil_is_variant) const {
	DataType ret;
	if (p_property.type == Variant::NIL && (p_nil_is_variant || (p_property.usage & PROPERTY_USAGE_NIL_IS_VARIANT))) {
		// Variant
		return ret;
	}
	ret.has_type = true;
	ret.builtin_type = p_property.type;
	if (p_property.type == Variant::OBJECT) {
		ret.kind = DataType::NATIVE;
		ret.native_type = p_property.class_name == StringName() ? "Object" : p_property.class_name;
	} else {
		ret.kind = DataType::BUILTIN;
	}
	return ret;
}

GDScriptParser::DataType GDScriptParser::_type_from_gdtype(const GDScriptDataType &p_gdtype) const {
	DataType result;
	if (!p_gdtype.has_type) {
		return result;
	}

	result.has_type = true;
	result.builtin_type = p_gdtype.builtin_type;
	result.native_type = p_gdtype.native_type;
	result.script_type = p_gdtype.script_type;

	switch (p_gdtype.kind) {
		case GDScriptDataType::UNINITIALIZED: {
			ERR_PRINT("Uninitialized datatype. Please report a bug.");
		} break;
		case GDScriptDataType::BUILTIN: {
			result.kind = DataType::BUILTIN;
		} break;
		case GDScriptDataType::NATIVE: {
			result.kind = DataType::NATIVE;
		} break;
		case GDScriptDataType::GDSCRIPT: {
			result.kind = DataType::GDSCRIPT;
		} break;
		case GDScriptDataType::SCRIPT: {
			result.kind = DataType::SCRIPT;
		} break;
	}
	return result;
}

GDScriptParser::DataType GDScriptParser::_get_operation_type(const Variant::Operator p_op, const DataType &p_a, const DataType &p_b, bool &r_valid) const {
	if (!p_a.has_type || !p_b.has_type) {
		r_valid = true;
		return DataType();
	}

	Variant::Type a_type = p_a.kind == DataType::BUILTIN ? p_a.builtin_type : Variant::OBJECT;
	Variant::Type b_type = p_b.kind == DataType::BUILTIN ? p_b.builtin_type : Variant::OBJECT;

	Variant a;
	REF a_ref;
	if (a_type == Variant::OBJECT) {
		a_ref.instance();
		a = a_ref;
	} else {
		Variant::CallError err;
		a = Variant::construct(a_type, NULL, 0, err);
		if (err.error != Variant::CallError::CALL_OK) {
			r_valid = false;
			return DataType();
		}
	}
	Variant b;
	REF b_ref;
	if (b_type == Variant::OBJECT) {
		b_ref.instance();
		b = b_ref;
	} else {
		Variant::CallError err;
		b = Variant::construct(b_type, NULL, 0, err);
		if (err.error != Variant::CallError::CALL_OK) {
			r_valid = false;
			return DataType();
		}
	}

	// Avoid division by zero
	if (a_type == Variant::INT || a_type == Variant::REAL) {
		Variant::evaluate(Variant::OP_ADD, a, 1, a, r_valid);
	}
	if (b_type == Variant::INT || b_type == Variant::REAL) {
		Variant::evaluate(Variant::OP_ADD, b, 1, b, r_valid);
	}
	if (a_type == Variant::STRING && b_type != Variant::ARRAY) {
		a = "%s"; // Work around for formatting operator (%)
	}

	Variant ret;
	Variant::evaluate(p_op, a, b, ret, r_valid);

	if (r_valid) {
		return _type_from_variant(ret);
	}

	return DataType();
}

Variant::Operator GDScriptParser::_get_variant_operation(const OperatorNode::Operator &p_op) const {
	switch (p_op) {
		case OperatorNode::OP_NEG: {
			return Variant::OP_NEGATE;
		} break;
		case OperatorNode::OP_POS: {
			return Variant::OP_POSITIVE;
		} break;
		case OperatorNode::OP_NOT: {
			return Variant::OP_NOT;
		} break;
		case OperatorNode::OP_BIT_INVERT: {
			return Variant::OP_BIT_NEGATE;
		} break;
		case OperatorNode::OP_IN: {
			return Variant::OP_IN;
		} break;
		case OperatorNode::OP_EQUAL: {
			return Variant::OP_EQUAL;
		} break;
		case OperatorNode::OP_NOT_EQUAL: {
			return Variant::OP_NOT_EQUAL;
		} break;
		case OperatorNode::OP_LESS: {
			return Variant::OP_LESS;
		} break;
		case OperatorNode::OP_LESS_EQUAL: {
			return Variant::OP_LESS_EQUAL;
		} break;
		case OperatorNode::OP_GREATER: {
			return Variant::OP_GREATER;
		} break;
		case OperatorNode::OP_GREATER_EQUAL: {
			return Variant::OP_GREATER_EQUAL;
		} break;
		case OperatorNode::OP_AND: {
			return Variant::OP_AND;
		} break;
		case OperatorNode::OP_OR: {
			return Variant::OP_OR;
		} break;
		case OperatorNode::OP_ASSIGN_ADD:
		case OperatorNode::OP_ADD: {
			return Variant::OP_ADD;
		} break;
		case OperatorNode::OP_ASSIGN_SUB:
		case OperatorNode::OP_SUB: {
			return Variant::OP_SUBTRACT;
		} break;
		case OperatorNode::OP_ASSIGN_MUL:
		case OperatorNode::OP_MUL: {
			return Variant::OP_MULTIPLY;
		} break;
		case OperatorNode::OP_ASSIGN_DIV:
		case OperatorNode::OP_DIV: {
			return Variant::OP_DIVIDE;
		} break;
		case OperatorNode::OP_ASSIGN_MOD:
		case OperatorNode::OP_MOD: {
			return Variant::OP_MODULE;
		} break;
		case OperatorNode::OP_ASSIGN_BIT_AND:
		case OperatorNode::OP_BIT_AND: {
			return Variant::OP_BIT_AND;
		} break;
		case OperatorNode::OP_ASSIGN_BIT_OR:
		case OperatorNode::OP_BIT_OR: {
			return Variant::OP_BIT_OR;
		} break;
		case OperatorNode::OP_ASSIGN_BIT_XOR:
		case OperatorNode::OP_BIT_XOR: {
			return Variant::OP_BIT_XOR;
		} break;
		case OperatorNode::OP_ASSIGN_SHIFT_LEFT:
		case OperatorNode::OP_SHIFT_LEFT: {
			return Variant::OP_SHIFT_LEFT;
		}
		case OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
		case OperatorNode::OP_SHIFT_RIGHT: {
			return Variant::OP_SHIFT_RIGHT;
		}
		default: {
			return Variant::OP_MAX;
		} break;
	}
}

bool GDScriptParser::_is_type_compatible(const DataType &p_container, const DataType &p_expression, bool p_allow_implicit_conversion) const {
	// Ignore for completion
	if (!check_types || for_completion) {
		return true;
	}
	// Can't test if not all have type
	if (!p_container.has_type || !p_expression.has_type) {
		return true;
	}

	// Should never get here unresolved
	ERR_FAIL_COND_V(p_container.kind == DataType::UNRESOLVED, false);
	ERR_FAIL_COND_V(p_expression.kind == DataType::UNRESOLVED, false);

	if (p_container.kind == DataType::BUILTIN && p_expression.kind == DataType::BUILTIN) {
		bool valid = p_container.builtin_type == p_expression.builtin_type;
		if (p_allow_implicit_conversion) {
			valid = valid || Variant::can_convert_strict(p_expression.builtin_type, p_container.builtin_type);
		}
		return valid;
	}

	if (p_container.kind == DataType::BUILTIN && p_container.builtin_type == Variant::OBJECT) {
		// Object built-in is a special case, it's compatible with any object and with null
		if (p_expression.kind == DataType::BUILTIN) {
			return p_expression.builtin_type == Variant::NIL;
		}
		// If it's not a built-in, must be an object
		return true;
	}

	if (p_container.kind == DataType::BUILTIN || (p_expression.kind == DataType::BUILTIN && p_expression.builtin_type != Variant::NIL)) {
		// Can't mix built-ins with objects
		return false;
	}

	// From now on everything is objects, check polymorphism
	// The container must be the same class or a superclass of the expression

	if (p_expression.kind == DataType::BUILTIN && p_expression.builtin_type == Variant::NIL) {
		// Null can be assigned to object types
		return true;
	}

	StringName expr_native;
	Ref<Script> expr_script;
	ClassNode *expr_class = NULL;

	switch (p_expression.kind) {
		case DataType::NATIVE: {
			if (p_container.kind != DataType::NATIVE) {
				// Non-native type can't be a superclass of a native type
				return false;
			}
			if (p_expression.is_meta_type) {
				expr_native = GDScriptNativeClass::get_class_static();
			} else {
				expr_native = p_expression.native_type;
			}
		} break;
		case DataType::SCRIPT:
		case DataType::GDSCRIPT: {
			if (p_container.kind == DataType::CLASS) {
				// This cannot be resolved without cyclic dependencies, so just bail out
				return false;
			}
			if (p_expression.is_meta_type) {
				expr_native = p_expression.script_type->get_class_name();
			} else {
				expr_script = p_expression.script_type;
				expr_native = expr_script->get_instance_base_type();
			}
		} break;
		case DataType::CLASS: {
			if (p_expression.is_meta_type) {
				expr_native = GDScript::get_class_static();
			} else {
				expr_class = p_expression.class_type;
				ClassNode *base = expr_class;
				while (base->base_type.kind == DataType::CLASS) {
					base = base->base_type.class_type;
				}
				expr_native = base->base_type.native_type;
				expr_script = base->base_type.script_type;
			}
		} break;
		case DataType::BUILTIN: // Already handled above
		case DataType::UNRESOLVED: // Not allowed, see above
			break;
	}

	// Some classes are prefixed with `_` internally
	if (!ClassDB::class_exists(expr_native)) {
		expr_native = "_" + expr_native;
	}

	switch (p_container.kind) {
		case DataType::NATIVE: {
			if (p_container.is_meta_type) {
				return ClassDB::is_parent_class(expr_native, GDScriptNativeClass::get_class_static());
			} else {
				StringName container_native = ClassDB::class_exists(p_container.native_type) ? p_container.native_type : StringName("_" + p_container.native_type);
				return ClassDB::is_parent_class(expr_native, container_native);
			}
		} break;
		case DataType::SCRIPT:
		case DataType::GDSCRIPT: {
			if (p_container.is_meta_type) {
				return ClassDB::is_parent_class(expr_native, GDScript::get_class_static());
			}
			if (expr_class == head && p_container.script_type->get_path() == self_path) {
				// Special case: container is self script and expression is self
				return true;
			}
			while (expr_script.is_valid()) {
				if (expr_script == p_container.script_type) {
					return true;
				}
				expr_script = expr_script->get_base_script();
			}
			return false;
		} break;
		case DataType::CLASS: {
			if (p_container.is_meta_type) {
				return ClassDB::is_parent_class(expr_native, GDScript::get_class_static());
			}
			if (p_container.class_type == head && expr_script.is_valid() && expr_script->get_path() == self_path) {
				// Special case: container is self and expression is self script
				return true;
			}
			while (expr_class) {
				if (expr_class == p_container.class_type) {
					return true;
				}
				expr_class = expr_class->base_type.class_type;
			}
			return false;
		} break;
		case DataType::BUILTIN: // Already handled above
		case DataType::UNRESOLVED: // Not allowed, see above
			break;
	}

	return false;
}

GDScriptParser::Node *GDScriptParser::_get_default_value_for_type(const DataType &p_type, int p_line) {
	Node *result;

	if (p_type.has_type && p_type.kind == DataType::BUILTIN && p_type.builtin_type != Variant::NIL && p_type.builtin_type != Variant::OBJECT) {
		if (p_type.builtin_type == Variant::ARRAY) {
			result = alloc_node<ArrayNode>();
		} else if (p_type.builtin_type == Variant::DICTIONARY) {
			result = alloc_node<DictionaryNode>();
		} else {
			ConstantNode *c = alloc_node<ConstantNode>();
			Variant::CallError err;
			c->value = Variant::construct(p_type.builtin_type, NULL, 0, err);
			result = c;
		}
	} else {
		ConstantNode *c = alloc_node<ConstantNode>();
		c->value = Variant();
		result = c;
	}

	result->line = p_line;

	return result;
}

GDScriptParser::DataType GDScriptParser::_reduce_node_type(Node *p_node) {
#ifdef DEBUG_ENABLED
	if (p_node->get_datatype().has_type && p_node->type != Node::TYPE_ARRAY && p_node->type != Node::TYPE_DICTIONARY) {
#else
	if (p_node->get_datatype().has_type) {
#endif
		return p_node->get_datatype();
	}

	DataType node_type;

	switch (p_node->type) {
		case Node::TYPE_CONSTANT: {
			node_type = _type_from_variant(static_cast<ConstantNode *>(p_node)->value);
		} break;
		case Node::TYPE_TYPE: {
			TypeNode *tn = static_cast<TypeNode *>(p_node);
			node_type.has_type = true;
			node_type.is_meta_type = true;
			node_type.kind = DataType::BUILTIN;
			node_type.builtin_type = tn->vtype;
		} break;
		case Node::TYPE_ARRAY: {
			node_type.has_type = true;
			node_type.kind = DataType::BUILTIN;
			node_type.builtin_type = Variant::ARRAY;
#ifdef DEBUG_ENABLED
			// Check stuff inside the array
			ArrayNode *an = static_cast<ArrayNode *>(p_node);
			for (int i = 0; i < an->elements.size(); i++) {
				_reduce_node_type(an->elements[i]);
			}
#endif // DEBUG_ENABLED
		} break;
		case Node::TYPE_DICTIONARY: {
			node_type.has_type = true;
			node_type.kind = DataType::BUILTIN;
			node_type.builtin_type = Variant::DICTIONARY;
#ifdef DEBUG_ENABLED
			// Check stuff inside the dictionarty
			DictionaryNode *dn = static_cast<DictionaryNode *>(p_node);
			for (int i = 0; i < dn->elements.size(); i++) {
				_reduce_node_type(dn->elements[i].key);
				_reduce_node_type(dn->elements[i].value);
			}
#endif // DEBUG_ENABLED
		} break;
		case Node::TYPE_SELF: {
			node_type.has_type = true;
			node_type.kind = DataType::CLASS;
			node_type.class_type = current_class;
			node_type.is_constant = true;
		} break;
		case Node::TYPE_IDENTIFIER: {
			IdentifierNode *id = static_cast<IdentifierNode *>(p_node);
			if (id->declared_block) {
				node_type = id->declared_block->variables[id->name]->get_datatype();
				id->declared_block->variables[id->name]->usages += 1;
			} else if (id->name == "#match_value") {
				// It's a special id just for the match statetement, ignore
				break;
			} else if (current_function && current_function->arguments.find(id->name) >= 0) {
				int idx = current_function->arguments.find(id->name);
				node_type = current_function->argument_types[idx];
			} else {
				node_type = _reduce_identifier_type(NULL, id->name, id->line, false);
			}
		} break;
		case Node::TYPE_CAST: {
			CastNode *cn = static_cast<CastNode *>(p_node);

			DataType source_type = _reduce_node_type(cn->source_node);
			cn->cast_type = _resolve_type(cn->cast_type, cn->line);
			if (source_type.has_type) {

				bool valid = false;
				if (check_types) {
					if (cn->cast_type.kind == DataType::BUILTIN && source_type.kind == DataType::BUILTIN) {
						valid = Variant::can_convert(source_type.builtin_type, cn->cast_type.builtin_type);
					}
					if (cn->cast_type.kind != DataType::BUILTIN && source_type.kind != DataType::BUILTIN) {
						valid = _is_type_compatible(cn->cast_type, source_type) || _is_type_compatible(source_type, cn->cast_type);
					}

					if (!valid) {
						_set_error("Invalid cast. Cannot convert from \"" + source_type.to_string() +
										   "\" to \"" + cn->cast_type.to_string() + "\".",
								cn->line);
						return DataType();
					}
				}
			} else {
#ifdef DEBUG_ENABLED
				_add_warning(GDScriptWarning::UNSAFE_CAST, cn->line, cn->cast_type.to_string());
#endif // DEBUG_ENABLED
				_mark_line_as_unsafe(cn->line);
			}

			node_type = cn->cast_type;

		} break;
		case Node::TYPE_OPERATOR: {
			OperatorNode *op = static_cast<OperatorNode *>(p_node);

			switch (op->op) {
				case OperatorNode::OP_CALL:
				case OperatorNode::OP_PARENT_CALL: {
					node_type = _reduce_function_call_type(op);
				} break;
				case OperatorNode::OP_YIELD: {
					if (op->arguments.size() == 2) {
						DataType base_type = _reduce_node_type(op->arguments[0]);
						DataType signal_type = _reduce_node_type(op->arguments[1]);
						// TODO: Check if signal exists when it's a constant
						if (base_type.has_type && base_type.kind == DataType::BUILTIN && base_type.builtin_type != Variant::NIL && base_type.builtin_type != Variant::OBJECT) {
							_set_error("The first argument of \"yield()\" must be an object.", op->line);
							return DataType();
						}
						if (signal_type.has_type && (signal_type.kind != DataType::BUILTIN || signal_type.builtin_type != Variant::STRING)) {
							_set_error("The second argument of \"yield()\" must be a string.", op->line);
							return DataType();
						}
					}
					// yield can return anything
					node_type.has_type = false;
				} break;
				case OperatorNode::OP_IS:
				case OperatorNode::OP_IS_BUILTIN: {

					if (op->arguments.size() != 2) {
						_set_error("Parser bug: binary operation without 2 arguments.", op->line);
						ERR_FAIL_V(DataType());
					}

					DataType value_type = _reduce_node_type(op->arguments[0]);
					DataType type_type = _reduce_node_type(op->arguments[1]);

					if (check_types && type_type.has_type) {
						if (!type_type.is_meta_type && (type_type.kind != DataType::NATIVE || !ClassDB::is_parent_class(type_type.native_type, "Script"))) {
							_set_error("Invalid \"is\" test: the right operand isn't a type (neither a native type nor a script).", op->line);
							return DataType();
						}
						type_type.is_meta_type = false; // Test the actual type
						if (!_is_type_compatible(type_type, value_type) && !_is_type_compatible(value_type, type_type)) {
							if (op->op == OperatorNode::OP_IS) {
								_set_error("A value of type \"" + value_type.to_string() + "\" will never be an instance of \"" + type_type.to_string() + "\".", op->line);
							} else {
								_set_error("A value of type \"" + value_type.to_string() + "\" will never be of type \"" + type_type.to_string() + "\".", op->line);
							}
							return DataType();
						}
					}

					node_type.has_type = true;
					node_type.is_constant = true;
					node_type.is_meta_type = false;
					node_type.kind = DataType::BUILTIN;
					node_type.builtin_type = Variant::BOOL;
				} break;
				// Unary operators
				case OperatorNode::OP_NEG:
				case OperatorNode::OP_POS:
				case OperatorNode::OP_NOT:
				case OperatorNode::OP_BIT_INVERT: {

					DataType argument_type = _reduce_node_type(op->arguments[0]);
					if (!argument_type.has_type) {
						break;
					}

					Variant::Operator var_op = _get_variant_operation(op->op);
					bool valid = false;
					node_type = _get_operation_type(var_op, argument_type, argument_type, valid);

					if (check_types && !valid) {
						_set_error("Invalid operand type (\"" + argument_type.to_string() +
										   "\") to unary operator \"" + Variant::get_operator_name(var_op) + "\".",
								op->line, op->column);
						return DataType();
					}

				} break;
				// Binary operators
				case OperatorNode::OP_IN:
				case OperatorNode::OP_EQUAL:
				case OperatorNode::OP_NOT_EQUAL:
				case OperatorNode::OP_LESS:
				case OperatorNode::OP_LESS_EQUAL:
				case OperatorNode::OP_GREATER:
				case OperatorNode::OP_GREATER_EQUAL:
				case OperatorNode::OP_AND:
				case OperatorNode::OP_OR:
				case OperatorNode::OP_ADD:
				case OperatorNode::OP_SUB:
				case OperatorNode::OP_MUL:
				case OperatorNode::OP_DIV:
				case OperatorNode::OP_MOD:
				case OperatorNode::OP_SHIFT_LEFT:
				case OperatorNode::OP_SHIFT_RIGHT:
				case OperatorNode::OP_BIT_AND:
				case OperatorNode::OP_BIT_OR:
				case OperatorNode::OP_BIT_XOR: {

					if (op->arguments.size() != 2) {
						_set_error("Parser bug: binary operation without 2 arguments.", op->line);
						ERR_FAIL_V(DataType());
					}

					DataType argument_a_type = _reduce_node_type(op->arguments[0]);
					DataType argument_b_type = _reduce_node_type(op->arguments[1]);
					if (!argument_a_type.has_type || !argument_b_type.has_type) {
						_mark_line_as_unsafe(op->line);
						break;
					}

					Variant::Operator var_op = _get_variant_operation(op->op);
					bool valid = false;
					node_type = _get_operation_type(var_op, argument_a_type, argument_b_type, valid);

					if (check_types && !valid) {
						_set_error("Invalid operand types (\"" + argument_a_type.to_string() + "\" and \"" +
										   argument_b_type.to_string() + "\") to operator \"" + Variant::get_operator_name(var_op) + "\".",
								op->line, op->column);
						return DataType();
					}
#ifdef DEBUG_ENABLED
					if (var_op == Variant::OP_DIVIDE && argument_a_type.kind == DataType::BUILTIN && argument_a_type.builtin_type == Variant::INT &&
							argument_b_type.kind == DataType::BUILTIN && argument_b_type.builtin_type == Variant::INT) {
						_add_warning(GDScriptWarning::INTEGER_DIVISION, op->line);
					}
#endif // DEBUG_ENABLED

				} break;
				// Ternary operators
				case OperatorNode::OP_TERNARY_IF: {
					if (op->arguments.size() != 3) {
						_set_error("Parser bug: ternary operation without 3 arguments.");
						ERR_FAIL_V(DataType());
					}

					DataType true_type = _reduce_node_type(op->arguments[1]);
					DataType false_type = _reduce_node_type(op->arguments[2]);
					// Check arguments[0] errors.
					_reduce_node_type(op->arguments[0]);

					// If types are equal, then the expression is of the same type
					// If they are compatible, return the broader type
					if (true_type == false_type || _is_type_compatible(true_type, false_type)) {
						node_type = true_type;
					} else if (_is_type_compatible(false_type, true_type)) {
						node_type = false_type;
					} else {
#ifdef DEBUG_ENABLED
						_add_warning(GDScriptWarning::INCOMPATIBLE_TERNARY, op->line);
#endif // DEBUG_ENABLED
					}
				} break;
				// Assignment should never happen within an expression
				case OperatorNode::OP_ASSIGN:
				case OperatorNode::OP_ASSIGN_ADD:
				case OperatorNode::OP_ASSIGN_SUB:
				case OperatorNode::OP_ASSIGN_MUL:
				case OperatorNode::OP_ASSIGN_DIV:
				case OperatorNode::OP_ASSIGN_MOD:
				case OperatorNode::OP_ASSIGN_SHIFT_LEFT:
				case OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
				case OperatorNode::OP_ASSIGN_BIT_AND:
				case OperatorNode::OP_ASSIGN_BIT_OR:
				case OperatorNode::OP_ASSIGN_BIT_XOR:
				case OperatorNode::OP_INIT_ASSIGN: {

					_set_error("Assignment inside an expression isn't allowed (parser bug?).", op->line);
					return DataType();

				} break;
				case OperatorNode::OP_INDEX_NAMED: {
					if (op->arguments.size() != 2) {
						_set_error("Parser bug: named index with invalid arguments.", op->line);
						ERR_FAIL_V(DataType());
					}
					if (op->arguments[1]->type != Node::TYPE_IDENTIFIER) {
						_set_error("Parser bug: named index without identifier argument.", op->line);
						ERR_FAIL_V(DataType());
					}

					DataType base_type = _reduce_node_type(op->arguments[0]);
					IdentifierNode *member_id = static_cast<IdentifierNode *>(op->arguments[1]);

					if (base_type.has_type) {
						if (check_types && base_type.kind == DataType::BUILTIN) {
							// Variant type, just test if it's possible
							DataType result;
							switch (base_type.builtin_type) {
								case Variant::NIL:
								case Variant::DICTIONARY: {
									result.has_type = false;
								} break;
								default: {
									Variant::CallError err;
									Variant temp = Variant::construct(base_type.builtin_type, NULL, 0, err);

									bool valid = false;
									Variant res = temp.get(member_id->name.operator String(), &valid);

									if (valid) {
										result = _type_from_variant(res);
									} else if (check_types) {
										_set_error("Can't get index \"" + String(member_id->name.operator String()) + "\" on base \"" +
														   base_type.to_string() + "\".",
												op->line);
										return DataType();
									}
								} break;
							}
							result.is_constant = false;
							node_type = result;
						} else {
							node_type = _reduce_identifier_type(&base_type, member_id->name, op->line, true);
#ifdef DEBUG_ENABLED
							if (!node_type.has_type) {
								_mark_line_as_unsafe(op->line);
								_add_warning(GDScriptWarning::UNSAFE_PROPERTY_ACCESS, op->line, member_id->name.operator String(), base_type.to_string());
							}
#endif // DEBUG_ENABLED
						}
					} else {
						_mark_line_as_unsafe(op->line);
					}
					if (error_set) {
						return DataType();
					}
				} break;
				case OperatorNode::OP_INDEX: {

					if (op->arguments[1]->type == Node::TYPE_CONSTANT) {
						ConstantNode *cn = static_cast<ConstantNode *>(op->arguments[1]);
						if (cn->value.get_type() == Variant::STRING) {
							// Treat this as named indexing

							IdentifierNode *id = alloc_node<IdentifierNode>();
							id->name = cn->value.operator StringName();
							id->datatype = cn->datatype;

							op->op = OperatorNode::OP_INDEX_NAMED;
							op->arguments.write[1] = id;

							return _reduce_node_type(op);
						}
					}

					DataType base_type = _reduce_node_type(op->arguments[0]);
					DataType index_type = _reduce_node_type(op->arguments[1]);

					if (!base_type.has_type) {
						_mark_line_as_unsafe(op->line);
						break;
					}

					if (check_types && index_type.has_type) {
						if (base_type.kind == DataType::BUILTIN) {
							// Check if indexing is valid
							bool error = index_type.kind != DataType::BUILTIN && base_type.builtin_type != Variant::DICTIONARY;
							if (!error) {
								switch (base_type.builtin_type) {
									// Expect int or real as index
									case Variant::POOL_BYTE_ARRAY:
									case Variant::POOL_COLOR_ARRAY:
									case Variant::POOL_INT_ARRAY:
									case Variant::POOL_REAL_ARRAY:
									case Variant::POOL_STRING_ARRAY:
									case Variant::POOL_VECTOR2_ARRAY:
									case Variant::POOL_VECTOR3_ARRAY:
									case Variant::ARRAY:
									case Variant::STRING: {
										error = index_type.builtin_type != Variant::INT && index_type.builtin_type != Variant::REAL;
									} break;
									// Expect String only
									case Variant::RECT2:
									case Variant::PLANE:
									case Variant::QUAT:
									case Variant::AABB:
									case Variant::OBJECT: {
										error = index_type.builtin_type != Variant::STRING;
									} break;
									// Expect String or number
									case Variant::VECTOR2:
									case Variant::VECTOR3:
									case Variant::TRANSFORM2D:
									case Variant::BASIS:
									case Variant::TRANSFORM: {
										error = index_type.builtin_type != Variant::INT && index_type.builtin_type != Variant::REAL &&
												index_type.builtin_type != Variant::STRING;
									} break;
									// Expect String or int
									case Variant::COLOR: {
										error = index_type.builtin_type != Variant::INT && index_type.builtin_type != Variant::STRING;
									} break;
									default: {
									}
								}
							}
							if (error) {
								_set_error("Invalid index type (" + index_type.to_string() + ") for base \"" + base_type.to_string() + "\".",
										op->line);
								return DataType();
							}

							if (op->arguments[1]->type == GDScriptParser::Node::TYPE_CONSTANT) {
								ConstantNode *cn = static_cast<ConstantNode *>(op->arguments[1]);
								// Index is a constant, just try it if possible
								switch (base_type.builtin_type) {
									// Arrays/string have variable indexing, can't test directly
									case Variant::STRING:
									case Variant::ARRAY:
									case Variant::DICTIONARY:
									case Variant::POOL_BYTE_ARRAY:
									case Variant::POOL_COLOR_ARRAY:
									case Variant::POOL_INT_ARRAY:
									case Variant::POOL_REAL_ARRAY:
									case Variant::POOL_STRING_ARRAY:
									case Variant::POOL_VECTOR2_ARRAY:
									case Variant::POOL_VECTOR3_ARRAY: {
										break;
									}
									default: {
										Variant::CallError err;
										Variant temp = Variant::construct(base_type.builtin_type, NULL, 0, err);

										bool valid = false;
										Variant res = temp.get(cn->value, &valid);

										if (valid) {
											node_type = _type_from_variant(res);
											node_type.is_constant = false;
										} else if (check_types) {
											_set_error("Can't get index \"" + String(cn->value) + "\" on base \"" +
															   base_type.to_string() + "\".",
													op->line);
											return DataType();
										}
									} break;
								}
							} else {
								_mark_line_as_unsafe(op->line);
							}
						} else if (!for_completion && (index_type.kind != DataType::BUILTIN || index_type.builtin_type != Variant::STRING)) {
							_set_error("Only strings can be used as an index in the base type \"" + base_type.to_string() + "\".", op->line);
							return DataType();
						}
					}
					if (check_types && !node_type.has_type && base_type.kind == DataType::BUILTIN) {
						// Can infer indexing type for some variant types
						DataType result;
						result.has_type = true;
						result.kind = DataType::BUILTIN;
						switch (base_type.builtin_type) {
							// Can't index at all
							case Variant::NIL:
							case Variant::BOOL:
							case Variant::INT:
							case Variant::REAL:
							case Variant::NODE_PATH:
							case Variant::_RID: {
								_set_error("Can't index on a value of type \"" + base_type.to_string() + "\".", op->line);
								return DataType();
							} break;
								// Return int
							case Variant::POOL_BYTE_ARRAY:
							case Variant::POOL_INT_ARRAY: {
								result.builtin_type = Variant::INT;
							} break;
								// Return real
							case Variant::POOL_REAL_ARRAY:
							case Variant::VECTOR2:
							case Variant::VECTOR3:
							case Variant::QUAT: {
								result.builtin_type = Variant::REAL;
							} break;
								// Return color
							case Variant::POOL_COLOR_ARRAY: {
								result.builtin_type = Variant::COLOR;
							} break;
								// Return string
							case Variant::POOL_STRING_ARRAY:
							case Variant::STRING: {
								result.builtin_type = Variant::STRING;
							} break;
								// Return Vector2
							case Variant::POOL_VECTOR2_ARRAY:
							case Variant::TRANSFORM2D:
							case Variant::RECT2: {
								result.builtin_type = Variant::VECTOR2;
							} break;
								// Return Vector3
							case Variant::POOL_VECTOR3_ARRAY:
							case Variant::AABB:
							case Variant::BASIS: {
								result.builtin_type = Variant::VECTOR3;
							} break;
								// Depends on the index
							case Variant::TRANSFORM:
							case Variant::PLANE:
							case Variant::COLOR:
							default: {
								result.has_type = false;
							} break;
						}
						node_type = result;
					}
				} break;
				default: {
					_set_error("Parser bug: unhandled operation.", op->line);
					ERR_FAIL_V(DataType());
				}
			}
		} break;
		default: {
		}
	}

	node_type = _resolve_type(node_type, p_node->line);
	p_node->set_datatype(node_type);
	return node_type;
}

bool GDScriptParser::_get_function_signature(DataType &p_base_type, const StringName &p_function, DataType &r_return_type, List<DataType> &r_arg_types, int &r_default_arg_count, bool &r_static, bool &r_vararg) const {

	r_static = false;
	r_default_arg_count = 0;

	DataType original_type = p_base_type;
	ClassNode *base = NULL;
	FunctionNode *callee = NULL;

	if (p_base_type.kind == DataType::CLASS) {
		base = p_base_type.class_type;
	}

	// Look up the current file (parse tree)
	while (!callee && base) {
		for (int i = 0; i < base->static_functions.size(); i++) {
			FunctionNode *func = base->static_functions[i];
			if (p_function == func->name) {
				r_static = true;
				callee = func;
				break;
			}
		}
		if (!callee && !p_base_type.is_meta_type) {
			for (int i = 0; i < base->functions.size(); i++) {
				FunctionNode *func = base->functions[i];
				if (p_function == func->name) {
					callee = func;
					break;
				}
			}
		}
		p_base_type = base->base_type;
		if (p_base_type.kind == DataType::CLASS) {
			base = p_base_type.class_type;
		} else {
			break;
		}
	}

	if (callee) {
		r_return_type = callee->get_datatype();
		for (int i = 0; i < callee->argument_types.size(); i++) {
			r_arg_types.push_back(callee->argument_types[i]);
		}
		r_default_arg_count = callee->default_values.size();
		return true;
	}

	// Nothing in current file, check parent script
	Ref<GDScript> base_gdscript;
	Ref<Script> base_script;
	StringName native;
	if (p_base_type.kind == DataType::GDSCRIPT) {
		base_gdscript = p_base_type.script_type;
		if (base_gdscript.is_null() || !base_gdscript->is_valid()) {
			// GDScript wasn't properly compíled, don't bother trying
			return false;
		}
	} else if (p_base_type.kind == DataType::SCRIPT) {
		base_script = p_base_type.script_type;
	} else if (p_base_type.kind == DataType::NATIVE) {
		native = p_base_type.native_type;
	}

	while (base_gdscript.is_valid()) {
		native = base_gdscript->get_instance_base_type();

		Map<StringName, GDScriptFunction *> funcs = base_gdscript->get_member_functions();

		if (funcs.has(p_function)) {
			GDScriptFunction *f = funcs[p_function];
			r_static = f->is_static();
			r_default_arg_count = f->get_default_argument_count();
			r_return_type = _type_from_gdtype(f->get_return_type());
			for (int i = 0; i < f->get_argument_count(); i++) {
				r_arg_types.push_back(_type_from_gdtype(f->get_argument_type(i)));
			}
			return true;
		}

		base_gdscript = base_gdscript->get_base_script();
	}

	while (base_script.is_valid()) {
		native = base_script->get_instance_base_type();
		MethodInfo mi = base_script->get_method_info(p_function);

		if (!(mi == MethodInfo())) {
			r_return_type = _type_from_property(mi.return_val, false);
			r_default_arg_count = mi.default_arguments.size();
			for (List<PropertyInfo>::Element *E = mi.arguments.front(); E; E = E->next()) {
				r_arg_types.push_back(_type_from_property(E->get()));
			}
			return true;
		}
		base_script = base_script->get_base_script();
	}

	if (native == StringName()) {
		// Empty native class, might happen in some Script implementations
		// Just ignore it
		return false;
	}

#ifdef DEBUG_METHODS_ENABLED

	// Only native remains
	if (!ClassDB::class_exists(native)) {
		native = "_" + native.operator String();
	}
	if (!ClassDB::class_exists(native)) {
		if (!check_types) return false;
		ERR_FAIL_V_MSG(false, "Parser bug: Class '" + String(native) + "' not found.");
	}

	MethodBind *method = ClassDB::get_method(native, p_function);

	if (!method) {
		// Try virtual methods
		List<MethodInfo> virtuals;
		ClassDB::get_virtual_methods(native, &virtuals);

		for (const List<MethodInfo>::Element *E = virtuals.front(); E; E = E->next()) {
			const MethodInfo &mi = E->get();
			if (mi.name == p_function) {
				r_default_arg_count = mi.default_arguments.size();
				for (const List<PropertyInfo>::Element *pi = mi.arguments.front(); pi; pi = pi->next()) {
					r_arg_types.push_back(_type_from_property(pi->get()));
				}
				r_return_type = _type_from_property(mi.return_val, false);
				r_vararg = mi.flags & METHOD_FLAG_VARARG;
				return true;
			}
		}

		// If the base is a script, it might be trying to access members of the Script class itself
		if (original_type.is_meta_type && !(p_function == "new") && (original_type.kind == DataType::SCRIPT || original_type.kind == DataType::GDSCRIPT)) {
			method = ClassDB::get_method(original_type.script_type->get_class_name(), p_function);

			if (method) {
				r_static = true;
			} else {
				// Try virtual methods of the script type
				virtuals.clear();
				ClassDB::get_virtual_methods(original_type.script_type->get_class_name(), &virtuals);
				for (const List<MethodInfo>::Element *E = virtuals.front(); E; E = E->next()) {
					const MethodInfo &mi = E->get();
					if (mi.name == p_function) {
						r_default_arg_count = mi.default_arguments.size();
						for (const List<PropertyInfo>::Element *pi = mi.arguments.front(); pi; pi = pi->next()) {
							r_arg_types.push_back(_type_from_property(pi->get()));
						}
						r_return_type = _type_from_property(mi.return_val, false);
						r_static = true;
						r_vararg = mi.flags & METHOD_FLAG_VARARG;
						return true;
					}
				}
				return false;
			}
		} else {
			return false;
		}
	}

	r_default_arg_count = method->get_default_argument_count();
	if (method->get_name() == "get_script") {
		r_return_type = DataType(); // Variant for now and let runtime decide.
	} else {
		r_return_type = _type_from_property(method->get_return_info(), false);
	}
	r_vararg = method->is_vararg();

	for (int i = 0; i < method->get_argument_count(); i++) {
		r_arg_types.push_back(_type_from_property(method->get_argument_info(i)));
	}
	return true;
#else
	return false;
#endif
}

GDScriptParser::DataType GDScriptParser::_reduce_function_call_type(const OperatorNode *p_call) {
	if (p_call->arguments.size() < 1) {
		_set_error("Parser bug: function call without enough arguments.", p_call->line);
		ERR_FAIL_V(DataType());
	}

	DataType return_type;
	List<DataType> arg_types;
	int default_args_count = 0;
	int arg_count = p_call->arguments.size();
	String callee_name;
	bool is_vararg = false;

	switch (p_call->arguments[0]->type) {
		case GDScriptParser::Node::TYPE_TYPE: {
			// Built-in constructor, special case
			TypeNode *tn = static_cast<TypeNode *>(p_call->arguments[0]);

			Vector<DataType> par_types;
			par_types.resize(p_call->arguments.size() - 1);
			for (int i = 1; i < p_call->arguments.size(); i++) {
				par_types.write[i - 1] = _reduce_node_type(p_call->arguments[i]);
			}

			if (error_set) return DataType();

			// Special case: check copy constructor. Those are defined implicitly in Variant.
			if (par_types.size() == 1) {
				if (!par_types[0].has_type || (par_types[0].kind == DataType::BUILTIN && par_types[0].builtin_type == tn->vtype)) {
					DataType result;
					result.has_type = true;
					result.kind = DataType::BUILTIN;
					result.builtin_type = tn->vtype;
					return result;
				}
			}

			bool match = false;
			List<MethodInfo> constructors;
			Variant::get_constructor_list(tn->vtype, &constructors);
			PropertyInfo return_type2;

			for (List<MethodInfo>::Element *E = constructors.front(); E; E = E->next()) {
				MethodInfo &mi = E->get();

				if (p_call->arguments.size() - 1 < mi.arguments.size() - mi.default_arguments.size()) {
					continue;
				}
				if (p_call->arguments.size() - 1 > mi.arguments.size()) {
					continue;
				}

				bool types_match = true;
				for (int i = 0; i < par_types.size(); i++) {
					DataType arg_type;
					if (mi.arguments[i].type != Variant::NIL) {
						arg_type.has_type = true;
						arg_type.kind = mi.arguments[i].type == Variant::OBJECT ? DataType::NATIVE : DataType::BUILTIN;
						arg_type.builtin_type = mi.arguments[i].type;
						arg_type.native_type = mi.arguments[i].class_name;
					}

					if (!_is_type_compatible(arg_type, par_types[i], true)) {
						types_match = false;
						break;
					} else {
#ifdef DEBUG_ENABLED
						if (arg_type.kind == DataType::BUILTIN && arg_type.builtin_type == Variant::INT && par_types[i].kind == DataType::BUILTIN && par_types[i].builtin_type == Variant::REAL) {
							_add_warning(GDScriptWarning::NARROWING_CONVERSION, p_call->line, Variant::get_type_name(tn->vtype));
						}
						if (par_types[i].may_yield && p_call->arguments[i + 1]->type == Node::TYPE_OPERATOR) {
							_add_warning(GDScriptWarning::FUNCTION_MAY_YIELD, p_call->line, _find_function_name(static_cast<OperatorNode *>(p_call->arguments[i + 1])));
						}
#endif // DEBUG_ENABLED
					}
				}

				if (types_match) {
					match = true;
					return_type2 = mi.return_val;
					break;
				}
			}

			if (match) {
				return _type_from_property(return_type2, false);
			} else if (check_types) {
				String err = "No constructor of '";
				err += Variant::get_type_name(tn->vtype);
				err += "' matches the signature '";
				err += Variant::get_type_name(tn->vtype) + "(";
				for (int i = 0; i < par_types.size(); i++) {
					if (i > 0) err += ", ";
					err += par_types[i].to_string();
				}
				err += ")'.";
				_set_error(err, p_call->line, p_call->column);
				return DataType();
			}
			return DataType();
		} break;
		case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION: {
			BuiltInFunctionNode *func = static_cast<BuiltInFunctionNode *>(p_call->arguments[0]);
			MethodInfo mi = GDScriptFunctions::get_info(func->function);

			return_type = _type_from_property(mi.return_val, false);

			// Check all arguments beforehand to solve warnings
			for (int i = 1; i < p_call->arguments.size(); i++) {
				_reduce_node_type(p_call->arguments[i]);
			}

			// Check arguments

			is_vararg = mi.flags & METHOD_FLAG_VARARG;

			default_args_count = mi.default_arguments.size();
			callee_name = mi.name;
			arg_count -= 1;

			// Check each argument type
			for (List<PropertyInfo>::Element *E = mi.arguments.front(); E; E = E->next()) {
				arg_types.push_back(_type_from_property(E->get()));
			}
		} break;
		default: {
			if (p_call->op == OperatorNode::OP_CALL && p_call->arguments.size() < 2) {
				_set_error("Parser bug: self method call without enough arguments.", p_call->line);
				ERR_FAIL_V(DataType());
			}

			int arg_id = p_call->op == OperatorNode::OP_CALL ? 1 : 0;

			if (p_call->arguments[arg_id]->type != Node::TYPE_IDENTIFIER) {
				_set_error("Parser bug: invalid function call argument.", p_call->line);
				ERR_FAIL_V(DataType());
			}

			// Check all arguments beforehand to solve warnings
			for (int i = arg_id + 1; i < p_call->arguments.size(); i++) {
				_reduce_node_type(p_call->arguments[i]);
			}

			IdentifierNode *func_id = static_cast<IdentifierNode *>(p_call->arguments[arg_id]);
			callee_name = func_id->name;
			arg_count -= 1 + arg_id;

			DataType base_type;
			if (p_call->op == OperatorNode::OP_PARENT_CALL) {
				base_type = current_class->base_type;
			} else {
				base_type = _reduce_node_type(p_call->arguments[0]);
			}

			if (!base_type.has_type || (base_type.kind == DataType::BUILTIN && base_type.builtin_type == Variant::NIL)) {
				_mark_line_as_unsafe(p_call->line);
				return DataType();
			}

			if (base_type.kind == DataType::BUILTIN) {
				Variant::CallError err;
				Variant tmp = Variant::construct(base_type.builtin_type, NULL, 0, err);

				if (check_types) {
					if (!tmp.has_method(callee_name)) {
						_set_error("The method \"" + callee_name + "\" isn't declared on base \"" + base_type.to_string() + "\".", p_call->line);
						return DataType();
					}

					default_args_count = Variant::get_method_default_arguments(base_type.builtin_type, callee_name).size();
					const Vector<Variant::Type> &var_arg_types = Variant::get_method_argument_types(base_type.builtin_type, callee_name);

					for (int i = 0; i < var_arg_types.size(); i++) {
						DataType argtype;
						if (var_arg_types[i] != Variant::NIL) {
							argtype.has_type = true;
							argtype.kind = DataType::BUILTIN;
							argtype.builtin_type = var_arg_types[i];
						}
						arg_types.push_back(argtype);
					}
				}

				bool rets = false;
				return_type.has_type = true;
				return_type.kind = DataType::BUILTIN;
				return_type.builtin_type = Variant::get_method_return_type(base_type.builtin_type, callee_name, &rets);
				// If the method returns, but it might return any type, (Variant::NIL), pretend we don't know the type.
				// At least make sure we know that it returns
				if (rets && return_type.builtin_type == Variant::NIL) {
					return_type.has_type = false;
				}
				break;
			}

			DataType original_type = base_type;
			bool is_initializer = callee_name == "new";
			bool is_get_script = p_call->arguments[0]->type == Node::TYPE_SELF && callee_name == "get_script";
			bool is_static = false;
			bool valid = false;

			if (is_initializer && original_type.is_meta_type) {
				// Try to check it as initializer
				base_type = original_type;
				callee_name = "_init";
				base_type.is_meta_type = false;

				valid = _get_function_signature(base_type, callee_name, return_type, arg_types,
						default_args_count, is_static, is_vararg);

				return_type = original_type;
				return_type.is_meta_type = false;

				valid = true; // There's always an initializer, we can assume this is true
			}

			if (is_get_script) {
				// get_script() can be considered a meta-type.
				return_type.kind = DataType::CLASS;
				return_type.class_type = static_cast<ClassNode *>(head);
				return_type.is_meta_type = true;
				valid = true;
			}

			if (!valid) {
				base_type = original_type;
				return_type = DataType();
				valid = _get_function_signature(base_type, callee_name, return_type, arg_types,
						default_args_count, is_static, is_vararg);
			}

			if (!valid) {
#ifdef DEBUG_ENABLED
				if (p_call->arguments[0]->type == Node::TYPE_SELF) {
					_set_error("The method \"" + callee_name + "\" isn't declared in the current class.", p_call->line);
					return DataType();
				}
				DataType tmp_type;
				valid = _get_member_type(original_type, func_id->name, tmp_type);
				if (valid) {
					if (tmp_type.is_constant) {
						_add_warning(GDScriptWarning::CONSTANT_USED_AS_FUNCTION, p_call->line, callee_name, original_type.to_string());
					} else {
						_add_warning(GDScriptWarning::PROPERTY_USED_AS_FUNCTION, p_call->line, callee_name, original_type.to_string());
					}
				}
				_add_warning(GDScriptWarning::UNSAFE_METHOD_ACCESS, p_call->line, callee_name, original_type.to_string());
				_mark_line_as_unsafe(p_call->line);
#endif // DEBUG_ENABLED
				return DataType();
			}

#ifdef DEBUG_ENABLED
			if (current_function && !for_completion && !is_static && p_call->arguments[0]->type == Node::TYPE_SELF && current_function->_static) {
				_set_error("Can't call non-static function from a static function.", p_call->line);
				return DataType();
			}

			if (check_types && !is_static && !is_initializer && base_type.is_meta_type) {
				_set_error("Non-static function \"" + String(callee_name) + "\" can only be called from an instance.", p_call->line);
				return DataType();
			}

			// Check signal emission for warnings
			if (callee_name == "emit_signal" && p_call->op == OperatorNode::OP_CALL && p_call->arguments[0]->type == Node::TYPE_SELF && p_call->arguments.size() >= 3 && p_call->arguments[2]->type == Node::TYPE_CONSTANT) {
				ConstantNode *sig = static_cast<ConstantNode *>(p_call->arguments[2]);
				String emitted = sig->value.get_type() == Variant::STRING ? sig->value.operator String() : "";
				for (int i = 0; i < current_class->_signals.size(); i++) {
					if (current_class->_signals[i].name == emitted) {
						current_class->_signals.write[i].emissions += 1;
						break;
					}
				}
			}
#endif // DEBUG_ENABLED
		} break;
	}

#ifdef DEBUG_ENABLED
	if (!check_types) {
		return return_type;
	}

	if (arg_count < arg_types.size() - default_args_count) {
		_set_error("Too few arguments for \"" + callee_name + "()\" call. Expected at least " + itos(arg_types.size() - default_args_count) + ".", p_call->line);
		return return_type;
	}
	if (!is_vararg && arg_count > arg_types.size()) {
		_set_error("Too many arguments for \"" + callee_name + "()\" call. Expected at most " + itos(arg_types.size()) + ".", p_call->line);
		return return_type;
	}

	int arg_diff = p_call->arguments.size() - arg_count;
	for (int i = arg_diff; i < p_call->arguments.size(); i++) {
		DataType par_type = _reduce_node_type(p_call->arguments[i]);

		if ((i - arg_diff) >= arg_types.size()) {
			continue;
		}

		DataType arg_type = arg_types[i - arg_diff];

		if (!par_type.has_type) {
			_mark_line_as_unsafe(p_call->line);
			if (par_type.may_yield && p_call->arguments[i]->type == Node::TYPE_OPERATOR) {
				_add_warning(GDScriptWarning::FUNCTION_MAY_YIELD, p_call->line, _find_function_name(static_cast<OperatorNode *>(p_call->arguments[i])));
			}
		} else if (!_is_type_compatible(arg_types[i - arg_diff], par_type, true)) {
			// Supertypes are acceptable for dynamic compliance
			if (!_is_type_compatible(par_type, arg_types[i - arg_diff])) {
				_set_error("At \"" + callee_name + "()\" call, argument " + itos(i - arg_diff + 1) + ". The passed argument's type (" +
								   par_type.to_string() + ") doesn't match the function's expected argument type (" +
								   arg_types[i - arg_diff].to_string() + ").",
						p_call->line);
				return DataType();
			} else {
				_mark_line_as_unsafe(p_call->line);
			}
		} else {
			if (arg_type.kind == DataType::BUILTIN && arg_type.builtin_type == Variant::INT && par_type.kind == DataType::BUILTIN && par_type.builtin_type == Variant::REAL) {
				_add_warning(GDScriptWarning::NARROWING_CONVERSION, p_call->line, callee_name);
			}
		}
	}

#endif // DEBUG_ENABLED

	return return_type;
}

bool GDScriptParser::_get_member_type(const DataType &p_base_type, const StringName &p_member, DataType &r_member_type, bool *r_is_const) const {
	DataType base_type = p_base_type;

	// Check classes in current file
	ClassNode *base = NULL;
	if (base_type.kind == DataType::CLASS) {
		base = base_type.class_type;
	}

	while (base) {
		if (base->constant_expressions.has(p_member)) {
			if (r_is_const)
				*r_is_const = true;
			r_member_type = base->constant_expressions[p_member].expression->get_datatype();
			return true;
		}

		if (!base_type.is_meta_type) {
			for (int i = 0; i < base->variables.size(); i++) {
				if (base->variables[i].identifier == p_member) {
					r_member_type = base->variables[i].data_type;
					base->variables.write[i].usages += 1;
					return true;
				}
			}
		} else {
			for (int i = 0; i < base->subclasses.size(); i++) {
				ClassNode *c = base->subclasses[i];
				if (c->name == p_member) {
					DataType class_type;
					class_type.has_type = true;
					class_type.is_constant = true;
					class_type.is_meta_type = true;
					class_type.kind = DataType::CLASS;
					class_type.class_type = c;
					r_member_type = class_type;
					return true;
				}
			}
		}

		base_type = base->base_type;
		if (base_type.kind == DataType::CLASS) {
			base = base_type.class_type;
		} else {
			break;
		}
	}

	Ref<GDScript> gds;
	if (base_type.kind == DataType::GDSCRIPT) {
		gds = base_type.script_type;
		if (gds.is_null() || !gds->is_valid()) {
			// GDScript wasn't properly compíled, don't bother trying
			return false;
		}
	}

	Ref<Script> scr;
	if (base_type.kind == DataType::SCRIPT) {
		scr = base_type.script_type;
	}

	StringName native;
	if (base_type.kind == DataType::NATIVE) {
		native = base_type.native_type;
	}

	// Check GDScripts
	while (gds.is_valid()) {
		if (gds->get_constants().has(p_member)) {
			Variant c = gds->get_constants()[p_member];
			r_member_type = _type_from_variant(c);
			return true;
		}

		if (!base_type.is_meta_type) {
			if (gds->get_members().has(p_member)) {
				r_member_type = _type_from_gdtype(gds->get_member_type(p_member));
				return true;
			}
		}

		native = gds->get_instance_base_type();
		if (gds->get_base_script().is_valid()) {
			gds = gds->get_base_script();
			scr = gds->get_base_script();
			bool is_meta = base_type.is_meta_type;
			base_type = _type_from_variant(scr.operator Variant());
			base_type.is_meta_type = is_meta;
		} else {
			break;
		}
	}

#define IS_USAGE_MEMBER(m_usage) (!(m_usage & (PROPERTY_USAGE_GROUP | PROPERTY_USAGE_CATEGORY)))

	// Check other script types
	while (scr.is_valid()) {
		Map<StringName, Variant> constants;
		scr->get_constants(&constants);
		if (constants.has(p_member)) {
			r_member_type = _type_from_variant(constants[p_member]);
			return true;
		}

		List<PropertyInfo> properties;
		scr->get_script_property_list(&properties);
		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			if (E->get().name == p_member && IS_USAGE_MEMBER(E->get().usage)) {
				r_member_type = _type_from_property(E->get());
				return true;
			}
		}

		base_type = _type_from_variant(scr.operator Variant());
		native = scr->get_instance_base_type();
		scr = scr->get_base_script();
	}

	if (native == StringName()) {
		// Empty native class, might happen in some Script implementations
		// Just ignore it
		return false;
	}

	// Check ClassDB
	if (!ClassDB::class_exists(native)) {
		native = "_" + native.operator String();
	}
	if (!ClassDB::class_exists(native)) {
		if (!check_types) return false;
		ERR_FAIL_V_MSG(false, "Parser bug: Class \"" + String(native) + "\" not found.");
	}

	bool valid = false;
	ClassDB::get_integer_constant(native, p_member, &valid);
	if (valid) {
		DataType ct;
		ct.has_type = true;
		ct.is_constant = true;
		ct.kind = DataType::BUILTIN;
		ct.builtin_type = Variant::INT;
		r_member_type = ct;
		return true;
	}

	if (!base_type.is_meta_type) {
		List<PropertyInfo> properties;
		ClassDB::get_property_list(native, &properties);
		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			if (E->get().name == p_member && IS_USAGE_MEMBER(E->get().usage)) {
				// Check if a getter exists
				StringName getter_name = ClassDB::get_property_getter(native, p_member);
				if (getter_name != StringName()) {
					// Use the getter return type
#ifdef DEBUG_METHODS_ENABLED
					MethodBind *getter_method = ClassDB::get_method(native, getter_name);
					if (getter_method) {
						r_member_type = _type_from_property(getter_method->get_return_info());
					} else {
						r_member_type = DataType();
					}
#else
					r_member_type = DataType();
#endif
				} else {
					r_member_type = _type_from_property(E->get());
				}
				return true;
			}
		}
	}

	// If the base is a script, it might be trying to access members of the Script class itself
	if (p_base_type.is_meta_type && (p_base_type.kind == DataType::SCRIPT || p_base_type.kind == DataType::GDSCRIPT)) {
		native = p_base_type.script_type->get_class_name();
		ClassDB::get_integer_constant(native, p_member, &valid);
		if (valid) {
			DataType ct;
			ct.has_type = true;
			ct.is_constant = true;
			ct.kind = DataType::BUILTIN;
			ct.builtin_type = Variant::INT;
			r_member_type = ct;
			return true;
		}

		List<PropertyInfo> properties;
		ClassDB::get_property_list(native, &properties);
		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			if (E->get().name == p_member && IS_USAGE_MEMBER(E->get().usage)) {
				// Check if a getter exists
				StringName getter_name = ClassDB::get_property_getter(native, p_member);
				if (getter_name != StringName()) {
					// Use the getter return type
#ifdef DEBUG_METHODS_ENABLED
					MethodBind *getter_method = ClassDB::get_method(native, getter_name);
					if (getter_method) {
						r_member_type = _type_from_property(getter_method->get_return_info());
					} else {
						r_member_type = DataType();
					}
#else
					r_member_type = DataType();
#endif
				} else {
					r_member_type = _type_from_property(E->get());
				}
				return true;
			}
		}
	}
#undef IS_USAGE_MEMBER

	return false;
}

GDScriptParser::DataType GDScriptParser::_reduce_identifier_type(const DataType *p_base_type, const StringName &p_identifier, int p_line, bool p_is_indexing) {

	if (p_base_type && !p_base_type->has_type) {
		return DataType();
	}

	DataType base_type;
	DataType member_type;

	if (!p_base_type) {
		base_type.has_type = true;
		base_type.is_constant = true;
		base_type.kind = DataType::CLASS;
		base_type.class_type = current_class;
	} else {
		base_type = DataType(*p_base_type);
	}

	bool is_const = false;
	if (_get_member_type(base_type, p_identifier, member_type, &is_const)) {
		if (!p_base_type && current_function && current_function->_static && !is_const) {
			_set_error("Can't access member variable (\"" + p_identifier.operator String() + "\") from a static function.", p_line);
			return DataType();
		}
		return member_type;
	}

	if (p_is_indexing) {
		// Don't look for globals since this is an indexed identifier
		return DataType();
	}

	if (!p_base_type) {
		// Possibly this is a global, check before failing

		if (ClassDB::class_exists(p_identifier) || ClassDB::class_exists("_" + p_identifier.operator String())) {
			DataType result;
			result.has_type = true;
			result.is_constant = true;
			result.is_meta_type = true;
			if (Engine::get_singleton()->has_singleton(p_identifier) || Engine::get_singleton()->has_singleton("_" + p_identifier.operator String())) {
				result.is_meta_type = false;
			}
			result.kind = DataType::NATIVE;
			result.native_type = p_identifier;
			return result;
		}

		ClassNode *outer_class = current_class;
		while (outer_class) {
			if (outer_class->name == p_identifier) {
				DataType result;
				result.has_type = true;
				result.is_constant = true;
				result.is_meta_type = true;
				result.kind = DataType::CLASS;
				result.class_type = outer_class;
				return result;
			}
			if (outer_class->constant_expressions.has(p_identifier)) {
				return outer_class->constant_expressions[p_identifier].type;
			}
			for (int i = 0; i < outer_class->subclasses.size(); i++) {
				if (outer_class->subclasses[i] == current_class) {
					continue;
				}
				if (outer_class->subclasses[i]->name == p_identifier) {
					DataType result;
					result.has_type = true;
					result.is_constant = true;
					result.is_meta_type = true;
					result.kind = DataType::CLASS;
					result.class_type = outer_class->subclasses[i];
					return result;
				}
			}
			outer_class = outer_class->owner;
		}

		if (ScriptServer::is_global_class(p_identifier)) {
			Ref<Script> scr = ResourceLoader::load(ScriptServer::get_global_class_path(p_identifier));
			if (scr.is_valid()) {
				DataType result;
				result.has_type = true;
				result.script_type = scr;
				result.is_constant = true;
				result.is_meta_type = true;
				Ref<GDScript> gds = scr;
				if (gds.is_valid()) {
					if (!gds->is_valid()) {
						_set_error("The class \"" + p_identifier + "\" couldn't be fully loaded (script error or cyclic dependency).");
						return DataType();
					}
					result.kind = DataType::GDSCRIPT;
				} else {
					result.kind = DataType::SCRIPT;
				}
				return result;
			}
			_set_error("The class \"" + p_identifier + "\" was found in global scope, but its script couldn't be loaded.");
			return DataType();
		}

		if (GDScriptLanguage::get_singleton()->get_global_map().has(p_identifier)) {
			int idx = GDScriptLanguage::get_singleton()->get_global_map()[p_identifier];
			Variant g = GDScriptLanguage::get_singleton()->get_global_array()[idx];
			return _type_from_variant(g);
		}

		if (GDScriptLanguage::get_singleton()->get_named_globals_map().has(p_identifier)) {
			Variant g = GDScriptLanguage::get_singleton()->get_named_globals_map()[p_identifier];
			return _type_from_variant(g);
		}

		// Non-tool singletons aren't loaded, check project settings
		List<PropertyInfo> props;
		ProjectSettings::get_singleton()->get_property_list(&props);

		for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
			String s = E->get().name;
			if (!s.begins_with("autoload/")) {
				continue;
			}
			String name = s.get_slice("/", 1);
			if (name == p_identifier) {
				String script = ProjectSettings::get_singleton()->get(s);
				if (script.begins_with("*")) {
					script = script.right(1);
				}
				if (!script.begins_with("res://")) {
					script = "res://" + script;
				}
				Ref<Script> singleton = ResourceLoader::load(script);
				if (singleton.is_valid()) {
					DataType result;
					result.has_type = true;
					result.is_constant = true;
					result.script_type = singleton;

					Ref<GDScript> gds = singleton;
					if (gds.is_valid()) {
						if (!gds->is_valid()) {
							_set_error("Couldn't fully load the singleton script \"" + p_identifier + "\" (possible cyclic reference or parse error).", p_line);
							return DataType();
						}
						result.kind = DataType::GDSCRIPT;
					} else {
						result.kind = DataType::SCRIPT;
					}
				}
			}
		}

		// This means looking in the current class, which type is always known
		_set_error("The identifier \"" + p_identifier.operator String() + "\" isn't declared in the current scope.", p_line);
	}

#ifdef DEBUG_ENABLED
	{
		DataType tmp_type;
		List<DataType> arg_types;
		int argcount;
		bool _static;
		bool vararg;
		if (_get_function_signature(base_type, p_identifier, tmp_type, arg_types, argcount, _static, vararg)) {
			_add_warning(GDScriptWarning::FUNCTION_USED_AS_PROPERTY, p_line, p_identifier.operator String(), base_type.to_string());
		}
	}
#endif // DEBUG_ENABLED

	_mark_line_as_unsafe(p_line);
	return DataType();
}

void GDScriptParser::_check_class_level_types(ClassNode *p_class) {

	// Names of internal object properties that we check to avoid overriding them.
	// "__meta__" could also be in here, but since it doesn't really affect object metadata,
	// it is okay to override it on script.
	StringName script_name = CoreStringNames::get_singleton()->_script;

	_mark_line_as_safe(p_class->line);

	// Constants
	for (Map<StringName, ClassNode::Constant>::Element *E = p_class->constant_expressions.front(); E; E = E->next()) {
		ClassNode::Constant &c = E->get();
		_mark_line_as_safe(c.expression->line);
		DataType cont = _resolve_type(c.type, c.expression->line);
		DataType expr = _resolve_type(c.expression->get_datatype(), c.expression->line);

		if (check_types && !_is_type_compatible(cont, expr)) {
			_set_error("The constant value type (" + expr.to_string() + ") isn't compatible with declared type (" + cont.to_string() + ").",
					c.expression->line);
			return;
		}

		expr.is_constant = true;
		c.type = expr;
		c.expression->set_datatype(expr);

		DataType tmp;
		const StringName &constant_name = E->key();
		if (constant_name == script_name || _get_member_type(p_class->base_type, constant_name, tmp)) {
			_set_error("The member \"" + String(constant_name) + "\" already exists in a parent class.", c.expression->line);
			return;
		}
	}

	// Function declarations
	for (int i = 0; i < p_class->static_functions.size(); i++) {
		_check_function_types(p_class->static_functions[i]);
		if (error_set) return;
	}

	for (int i = 0; i < p_class->functions.size(); i++) {
		_check_function_types(p_class->functions[i]);
		if (error_set) return;
	}

	// Class variables
	for (int i = 0; i < p_class->variables.size(); i++) {
		ClassNode::Member &v = p_class->variables.write[i];

		DataType tmp;
		if (v.identifier == script_name || _get_member_type(p_class->base_type, v.identifier, tmp)) {
			_set_error("The member \"" + String(v.identifier) + "\" already exists in a parent class.", v.line);
			return;
		}

		_mark_line_as_safe(v.line);
		v.data_type = _resolve_type(v.data_type, v.line);
		v.initial_assignment->arguments[0]->set_datatype(v.data_type);

		if (v.expression) {
			DataType expr_type = _reduce_node_type(v.expression);

			if (check_types && !_is_type_compatible(v.data_type, expr_type)) {
				// Try supertype test
				if (_is_type_compatible(expr_type, v.data_type)) {
					_mark_line_as_unsafe(v.line);
				} else {
					// Try with implicit conversion
					if (v.data_type.kind != DataType::BUILTIN || !_is_type_compatible(v.data_type, expr_type, true)) {
						_set_error("The assigned expression's type (" + expr_type.to_string() + ") doesn't match the variable's type (" +
										   v.data_type.to_string() + ").",
								v.line);
						return;
					}

					// Replace assignment with implicit conversion
					BuiltInFunctionNode *convert = alloc_node<BuiltInFunctionNode>();
					convert->line = v.line;
					convert->function = GDScriptFunctions::TYPE_CONVERT;

					ConstantNode *tgt_type = alloc_node<ConstantNode>();
					tgt_type->line = v.line;
					tgt_type->value = (int)v.data_type.builtin_type;

					OperatorNode *convert_call = alloc_node<OperatorNode>();
					convert_call->line = v.line;
					convert_call->op = OperatorNode::OP_CALL;
					convert_call->arguments.push_back(convert);
					convert_call->arguments.push_back(v.expression);
					convert_call->arguments.push_back(tgt_type);

					v.expression = convert_call;
					v.initial_assignment->arguments.write[1] = convert_call;
				}
			}

			if (v.data_type.infer_type) {
				if (!expr_type.has_type) {
					_set_error("The assigned value doesn't have a set type; the variable type can't be inferred.", v.line);
					return;
				}
				if (expr_type.kind == DataType::BUILTIN && expr_type.builtin_type == Variant::NIL) {
					_set_error("The variable type cannot be inferred because its value is \"null\".", v.line);
					return;
				}
				v.data_type = expr_type;
				v.data_type.is_constant = false;
			}
		}

		// Check export hint
		if (v.data_type.has_type && v._export.type != Variant::NIL) {
			DataType export_type = _type_from_property(v._export);
			if (!_is_type_compatible(v.data_type, export_type, true)) {
				_set_error("The export hint's type (" + export_type.to_string() + ") doesn't match the variable's type (" +
								   v.data_type.to_string() + ").",
						v.line);
				return;
			}
		}

		// Setter and getter
		if (v.setter == StringName() && v.getter == StringName()) continue;

		bool found_getter = false;
		bool found_setter = false;
		for (int j = 0; j < p_class->functions.size(); j++) {
			if (v.setter == p_class->functions[j]->name) {
				found_setter = true;
				FunctionNode *setter = p_class->functions[j];

				if (setter->get_required_argument_count() != 1 &&
						!(setter->get_required_argument_count() == 0 && setter->default_values.size() > 0)) {
					_set_error("The setter function needs to receive exactly 1 argument. See \"" + setter->name +
									   "()\" definition at line " + itos(setter->line) + ".",
							v.line);
					return;
				}
				if (!_is_type_compatible(v.data_type, setter->argument_types[0])) {
					_set_error("The setter argument's type (" + setter->argument_types[0].to_string() +
									   ") doesn't match the variable's type (" + v.data_type.to_string() + "). See \"" +
									   setter->name + "()\" definition at line " + itos(setter->line) + ".",
							v.line);
					return;
				}
				continue;
			}
			if (v.getter == p_class->functions[j]->name) {
				found_getter = true;
				FunctionNode *getter = p_class->functions[j];

				if (getter->get_required_argument_count() != 0) {
					_set_error("The getter function can't receive arguments. See \"" + getter->name +
									   "()\" definition at line " + itos(getter->line) + ".",
							v.line);
					return;
				}
				if (!_is_type_compatible(v.data_type, getter->get_datatype())) {
					_set_error("The getter return type (" + getter->get_datatype().to_string() +
									   ") doesn't match the variable's type (" + v.data_type.to_string() +
									   "). See \"" + getter->name + "()\" definition at line " + itos(getter->line) + ".",
							v.line);
					return;
				}
			}
			if (found_getter && found_setter) break;
		}

		if ((found_getter || v.getter == StringName()) && (found_setter || v.setter == StringName())) continue;

		// Check for static functions
		for (int j = 0; j < p_class->static_functions.size(); j++) {
			if (v.setter == p_class->static_functions[j]->name) {
				FunctionNode *setter = p_class->static_functions[j];
				_set_error("The setter can't be a static function. See \"" + setter->name + "()\" definition at line " + itos(setter->line) + ".", v.line);
				return;
			}
			if (v.getter == p_class->static_functions[j]->name) {
				FunctionNode *getter = p_class->static_functions[j];
				_set_error("The getter can't be a static function. See \"" + getter->name + "()\" definition at line " + itos(getter->line) + ".", v.line);
				return;
			}
		}

		if (!found_setter && v.setter != StringName()) {
			_set_error("The setter function isn't defined.", v.line);
			return;
		}

		if (!found_getter && v.getter != StringName()) {
			_set_error("The getter function isn't defined.", v.line);
			return;
		}
	}

	// Signals
	DataType base = p_class->base_type;

	while (base.kind == DataType::CLASS) {
		ClassNode *base_class = base.class_type;
		for (int i = 0; i < p_class->_signals.size(); i++) {
			for (int j = 0; j < base_class->_signals.size(); j++) {
				if (p_class->_signals[i].name == base_class->_signals[j].name) {
					_set_error("The signal \"" + p_class->_signals[i].name + "\" already exists in a parent class.", p_class->_signals[i].line);
					return;
				}
			}
		}
		base = base_class->base_type;
	}

	StringName native;
	if (base.kind == DataType::GDSCRIPT || base.kind == DataType::SCRIPT) {
		Ref<Script> scr = base.script_type;
		if (scr.is_valid() && scr->is_valid()) {
			native = scr->get_instance_base_type();
			for (int i = 0; i < p_class->_signals.size(); i++) {
				if (scr->has_script_signal(p_class->_signals[i].name)) {
					_set_error("The signal \"" + p_class->_signals[i].name + "\" already exists in a parent class.", p_class->_signals[i].line);
					return;
				}
			}
		}
	} else if (base.kind == DataType::NATIVE) {
		native = base.native_type;
	}

	if (native != StringName()) {
		for (int i = 0; i < p_class->_signals.size(); i++) {
			if (ClassDB::has_signal(native, p_class->_signals[i].name)) {
				_set_error("The signal \"" + p_class->_signals[i].name + "\" already exists in a parent class.", p_class->_signals[i].line);
				return;
			}
		}
	}

	// Inner classes
	for (int i = 0; i < p_class->subclasses.size(); i++) {
		current_class = p_class->subclasses[i];
		_check_class_level_types(current_class);
		if (error_set) return;
		current_class = p_class;
	}
}

void GDScriptParser::_check_function_types(FunctionNode *p_function) {

	p_function->return_type = _resolve_type(p_function->return_type, p_function->line);

	// Arguments
	int defaults_ofs = p_function->arguments.size() - p_function->default_values.size();
	for (int i = 0; i < p_function->arguments.size(); i++) {
		if (i < defaults_ofs) {
			p_function->argument_types.write[i] = _resolve_type(p_function->argument_types[i], p_function->line);
		} else {
			if (p_function->default_values[i - defaults_ofs]->type != Node::TYPE_OPERATOR) {
				_set_error("Parser bug: invalid argument default value.", p_function->line, p_function->column);
				return;
			}

			OperatorNode *op = static_cast<OperatorNode *>(p_function->default_values[i - defaults_ofs]);

			if (op->op != OperatorNode::OP_ASSIGN || op->arguments.size() != 2) {
				_set_error("Parser bug: invalid argument default value operation.", p_function->line);
				return;
			}

			DataType def_type = _reduce_node_type(op->arguments[1]);

			if (p_function->argument_types[i].infer_type) {
				def_type.is_constant = false;
				p_function->argument_types.write[i] = def_type;
			} else {
				p_function->argument_types.write[i] = _resolve_type(p_function->argument_types[i], p_function->line);

				if (!_is_type_compatible(p_function->argument_types[i], def_type, true)) {
					String arg_name = p_function->arguments[i];
					_set_error("Value type (" + def_type.to_string() + ") doesn't match the type of argument '" +
									   arg_name + "' (" + p_function->argument_types[i].to_string() + ").",
							p_function->line);
				}
			}
		}
#ifdef DEBUG_ENABLED
		if (p_function->arguments_usage[i] == 0 && !p_function->arguments[i].operator String().begins_with("_")) {
			_add_warning(GDScriptWarning::UNUSED_ARGUMENT, p_function->line, p_function->name, p_function->arguments[i].operator String());
		}
		for (int j = 0; j < current_class->variables.size(); j++) {
			if (current_class->variables[j].identifier == p_function->arguments[i]) {
				_add_warning(GDScriptWarning::SHADOWED_VARIABLE, p_function->line, p_function->arguments[i], itos(current_class->variables[j].line));
			}
		}
#endif // DEBUG_ENABLED
	}

	if (!(p_function->name == "_init")) {
		// Signature for the initializer may vary
#ifdef DEBUG_ENABLED
		DataType return_type;
		List<DataType> arg_types;
		int default_arg_count = 0;
		bool _static = false;
		bool vararg = false;

		DataType base_type = current_class->base_type;
		if (_get_function_signature(base_type, p_function->name, return_type, arg_types, default_arg_count, _static, vararg)) {
			bool valid = _static == p_function->_static;
			valid = valid && return_type == p_function->return_type;
			int argsize_diff = p_function->arguments.size() - arg_types.size();
			valid = valid && argsize_diff >= 0;
			valid = valid && p_function->default_values.size() >= default_arg_count + argsize_diff;
			int i = 0;
			for (List<DataType>::Element *E = arg_types.front(); valid && E; E = E->next()) {
				valid = valid && E->get() == p_function->argument_types[i++];
			}

			if (!valid) {
				String parent_signature = return_type.has_type ? return_type.to_string() : "Variant";
				if (parent_signature == "null") {
					parent_signature = "void";
				}
				parent_signature += " " + p_function->name + "(";
				if (arg_types.size()) {
					int j = 0;
					for (List<DataType>::Element *E = arg_types.front(); E; E = E->next()) {
						if (E != arg_types.front()) {
							parent_signature += ", ";
						}
						String arg = E->get().to_string();
						if (arg == "null" || arg == "var") {
							arg = "Variant";
						}
						parent_signature += arg;
						if (j == arg_types.size() - default_arg_count) {
							parent_signature += "=default";
						}

						j++;
					}
				}
				parent_signature += ")";
				_set_error("The function signature doesn't match the parent. Parent signature is: \"" + parent_signature + "\".", p_function->line);
				return;
			}
		}
#endif // DEBUG_ENABLED
	} else {
		if (p_function->return_type.has_type && (p_function->return_type.kind != DataType::BUILTIN || p_function->return_type.builtin_type != Variant::NIL)) {
			_set_error("The constructor can't return a value.", p_function->line);
			return;
		}
	}

	if (p_function->return_type.has_type && (p_function->return_type.kind != DataType::BUILTIN || p_function->return_type.builtin_type != Variant::NIL)) {
		if (!p_function->body->has_return) {
			_set_error("A non-void function must return a value in all possible paths.", p_function->line);
			return;
		}
	}

	if (p_function->has_yield) {
		// yield() will make the function return a GDScriptFunctionState, so the type is ambiguous
		p_function->return_type.has_type = false;
		p_function->return_type.may_yield = true;
	}
}

void GDScriptParser::_check_class_blocks_types(ClassNode *p_class) {

	// Function blocks
	for (int i = 0; i < p_class->static_functions.size(); i++) {
		current_function = p_class->static_functions[i];
		current_block = current_function->body;
		_mark_line_as_safe(current_function->line);
		_check_block_types(current_block);
		current_block = NULL;
		current_function = NULL;
		if (error_set) return;
	}

	for (int i = 0; i < p_class->functions.size(); i++) {
		current_function = p_class->functions[i];
		current_block = current_function->body;
		_mark_line_as_safe(current_function->line);
		_check_block_types(current_block);
		current_block = NULL;
		current_function = NULL;
		if (error_set) return;
	}

#ifdef DEBUG_ENABLED
	// Warnings
	for (int i = 0; i < p_class->variables.size(); i++) {
		if (p_class->variables[i].usages == 0) {
			_add_warning(GDScriptWarning::UNUSED_CLASS_VARIABLE, p_class->variables[i].line, p_class->variables[i].identifier);
		}
	}
	for (int i = 0; i < p_class->_signals.size(); i++) {
		if (p_class->_signals[i].emissions == 0) {
			_add_warning(GDScriptWarning::UNUSED_SIGNAL, p_class->_signals[i].line, p_class->_signals[i].name);
		}
	}
#endif // DEBUG_ENABLED

	// Inner classes
	for (int i = 0; i < p_class->subclasses.size(); i++) {
		current_class = p_class->subclasses[i];
		_check_class_blocks_types(current_class);
		if (error_set) return;
		current_class = p_class;
	}
}

#ifdef DEBUG_ENABLED
static String _find_function_name(const GDScriptParser::OperatorNode *p_call) {
	switch (p_call->arguments[0]->type) {
		case GDScriptParser::Node::TYPE_TYPE: {
			return Variant::get_type_name(static_cast<GDScriptParser::TypeNode *>(p_call->arguments[0])->vtype);
		} break;
		case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION: {
			return GDScriptFunctions::get_func_name(static_cast<GDScriptParser::BuiltInFunctionNode *>(p_call->arguments[0])->function);
		} break;
		default: {
			int id_index = p_call->op == GDScriptParser::OperatorNode::OP_PARENT_CALL ? 0 : 1;
			if (p_call->arguments.size() > id_index && p_call->arguments[id_index]->type == GDScriptParser::Node::TYPE_IDENTIFIER) {
				return static_cast<GDScriptParser::IdentifierNode *>(p_call->arguments[id_index])->name;
			}
		} break;
	}
	return String();
}
#endif // DEBUG_ENABLED

void GDScriptParser::_check_block_types(BlockNode *p_block) {

	Node *last_var_assign = NULL;

	// Check each statement
	for (List<Node *>::Element *E = p_block->statements.front(); E; E = E->next()) {
		Node *statement = E->get();
		switch (statement->type) {
			case Node::TYPE_NEWLINE:
			case Node::TYPE_BREAKPOINT: {
				// Nothing to do
			} break;
			case Node::TYPE_ASSERT: {
				AssertNode *an = static_cast<AssertNode *>(statement);
				_mark_line_as_safe(an->line);
				_reduce_node_type(an->condition);
				_reduce_node_type(an->message);
			} break;
			case Node::TYPE_LOCAL_VAR: {
				LocalVarNode *lv = static_cast<LocalVarNode *>(statement);
				lv->datatype = _resolve_type(lv->datatype, lv->line);
				_mark_line_as_safe(lv->line);

				last_var_assign = lv->assign;
				if (lv->assign) {
					lv->assign_op->arguments[0]->set_datatype(lv->datatype);
					DataType assign_type = _reduce_node_type(lv->assign);
#ifdef DEBUG_ENABLED
					if (assign_type.has_type && assign_type.kind == DataType::BUILTIN && assign_type.builtin_type == Variant::NIL) {
						if (lv->assign->type == Node::TYPE_OPERATOR) {
							OperatorNode *call = static_cast<OperatorNode *>(lv->assign);
							if (call->op == OperatorNode::OP_CALL || call->op == OperatorNode::OP_PARENT_CALL) {
								_add_warning(GDScriptWarning::VOID_ASSIGNMENT, lv->line, _find_function_name(call));
							}
						}
					}
					if (lv->datatype.has_type && assign_type.may_yield && lv->assign->type == Node::TYPE_OPERATOR) {
						_add_warning(GDScriptWarning::FUNCTION_MAY_YIELD, lv->line, _find_function_name(static_cast<OperatorNode *>(lv->assign)));
					}
					for (int i = 0; i < current_class->variables.size(); i++) {
						if (current_class->variables[i].identifier == lv->name) {
							_add_warning(GDScriptWarning::SHADOWED_VARIABLE, lv->line, lv->name, itos(current_class->variables[i].line));
						}
					}
#endif // DEBUG_ENABLED

					if (!_is_type_compatible(lv->datatype, assign_type)) {
						// Try supertype test
						if (_is_type_compatible(assign_type, lv->datatype)) {
							_mark_line_as_unsafe(lv->line);
						} else {
							// Try implicit conversion
							if (lv->datatype.kind != DataType::BUILTIN || !_is_type_compatible(lv->datatype, assign_type, true)) {
								_set_error("The assigned value type (" + assign_type.to_string() + ") doesn't match the variable's type (" +
												   lv->datatype.to_string() + ").",
										lv->line);
								return;
							}
							// Replace assignment with implicit conversion
							BuiltInFunctionNode *convert = alloc_node<BuiltInFunctionNode>();
							convert->line = lv->line;
							convert->function = GDScriptFunctions::TYPE_CONVERT;

							ConstantNode *tgt_type = alloc_node<ConstantNode>();
							tgt_type->line = lv->line;
							tgt_type->value = (int)lv->datatype.builtin_type;

							OperatorNode *convert_call = alloc_node<OperatorNode>();
							convert_call->line = lv->line;
							convert_call->op = OperatorNode::OP_CALL;
							convert_call->arguments.push_back(convert);
							convert_call->arguments.push_back(lv->assign);
							convert_call->arguments.push_back(tgt_type);

							lv->assign = convert_call;
							lv->assign_op->arguments.write[1] = convert_call;
#ifdef DEBUG_ENABLED
							if (lv->datatype.builtin_type == Variant::INT && assign_type.builtin_type == Variant::REAL) {
								_add_warning(GDScriptWarning::NARROWING_CONVERSION, lv->line);
							}
#endif // DEBUG_ENABLED
						}
					}
					if (lv->datatype.infer_type) {
						if (!assign_type.has_type) {
							_set_error("The assigned value doesn't have a set type; the variable type can't be inferred.", lv->line);
							return;
						}
						if (assign_type.kind == DataType::BUILTIN && assign_type.builtin_type == Variant::NIL) {
							_set_error("The variable type cannot be inferred because its value is \"null\".", lv->line);
							return;
						}
						lv->datatype = assign_type;
						lv->datatype.is_constant = false;
					}
					if (lv->datatype.has_type && !assign_type.has_type) {
						_mark_line_as_unsafe(lv->line);
					}
				}
			} break;
			case Node::TYPE_OPERATOR: {
				OperatorNode *op = static_cast<OperatorNode *>(statement);

				switch (op->op) {
					case OperatorNode::OP_ASSIGN:
					case OperatorNode::OP_ASSIGN_ADD:
					case OperatorNode::OP_ASSIGN_SUB:
					case OperatorNode::OP_ASSIGN_MUL:
					case OperatorNode::OP_ASSIGN_DIV:
					case OperatorNode::OP_ASSIGN_MOD:
					case OperatorNode::OP_ASSIGN_SHIFT_LEFT:
					case OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
					case OperatorNode::OP_ASSIGN_BIT_AND:
					case OperatorNode::OP_ASSIGN_BIT_OR:
					case OperatorNode::OP_ASSIGN_BIT_XOR: {
						if (op->arguments.size() < 2) {
							_set_error("Parser bug: operation without enough arguments.", op->line, op->column);
							return;
						}

						if (op->arguments[1] == last_var_assign) {
							// Assignment was already checked
							break;
						}

						_mark_line_as_safe(op->line);

						DataType lh_type = _reduce_node_type(op->arguments[0]);

						if (error_set) {
							return;
						}

						if (check_types) {
							if (!lh_type.has_type) {
								if (op->arguments[0]->type == Node::TYPE_OPERATOR) {
									_mark_line_as_unsafe(op->line);
								}
							}
							if (lh_type.is_constant) {
								_set_error("Can't assign a new value to a constant.", op->line);
								return;
							}
						}

						DataType rh_type;
						if (op->op != OperatorNode::OP_ASSIGN) {
							// Validate operation
							DataType arg_type = _reduce_node_type(op->arguments[1]);
							if (!arg_type.has_type) {
								_mark_line_as_unsafe(op->line);
								break;
							}

							Variant::Operator oper = _get_variant_operation(op->op);
							bool valid = false;
							rh_type = _get_operation_type(oper, lh_type, arg_type, valid);

							if (check_types && !valid) {
								_set_error("Invalid operand types (\"" + lh_type.to_string() + "\" and \"" + arg_type.to_string() +
												   "\") to assignment operator \"" + Variant::get_operator_name(oper) + "\".",
										op->line);
								return;
							}
						} else {
							rh_type = _reduce_node_type(op->arguments[1]);
						}
#ifdef DEBUG_ENABLED
						if (rh_type.has_type && rh_type.kind == DataType::BUILTIN && rh_type.builtin_type == Variant::NIL) {
							if (op->arguments[1]->type == Node::TYPE_OPERATOR) {
								OperatorNode *call = static_cast<OperatorNode *>(op->arguments[1]);
								if (call->op == OperatorNode::OP_CALL || call->op == OperatorNode::OP_PARENT_CALL) {
									_add_warning(GDScriptWarning::VOID_ASSIGNMENT, op->line, _find_function_name(call));
								}
							}
						}
						if (lh_type.has_type && rh_type.may_yield && op->arguments[1]->type == Node::TYPE_OPERATOR) {
							_add_warning(GDScriptWarning::FUNCTION_MAY_YIELD, op->line, _find_function_name(static_cast<OperatorNode *>(op->arguments[1])));
						}

#endif // DEBUG_ENABLED
						bool type_match = lh_type.has_type && rh_type.has_type;
						if (check_types && !_is_type_compatible(lh_type, rh_type)) {
							type_match = false;

							// Try supertype test
							if (_is_type_compatible(rh_type, lh_type)) {
								_mark_line_as_unsafe(op->line);
							} else {
								// Try implicit conversion
								if (lh_type.kind != DataType::BUILTIN || !_is_type_compatible(lh_type, rh_type, true)) {
									_set_error("The assigned value's type (" + rh_type.to_string() + ") doesn't match the variable's type (" +
													   lh_type.to_string() + ").",
											op->line);
									return;
								}
								if (op->op == OperatorNode::OP_ASSIGN) {
									// Replace assignment with implicit conversion
									BuiltInFunctionNode *convert = alloc_node<BuiltInFunctionNode>();
									convert->line = op->line;
									convert->function = GDScriptFunctions::TYPE_CONVERT;

									ConstantNode *tgt_type = alloc_node<ConstantNode>();
									tgt_type->line = op->line;
									tgt_type->value = (int)lh_type.builtin_type;

									OperatorNode *convert_call = alloc_node<OperatorNode>();
									convert_call->line = op->line;
									convert_call->op = OperatorNode::OP_CALL;
									convert_call->arguments.push_back(convert);
									convert_call->arguments.push_back(op->arguments[1]);
									convert_call->arguments.push_back(tgt_type);

									op->arguments.write[1] = convert_call;

									type_match = true; // Since we are converting, the type is matching
								}
#ifdef DEBUG_ENABLED
								if (lh_type.builtin_type == Variant::INT && rh_type.builtin_type == Variant::REAL) {
									_add_warning(GDScriptWarning::NARROWING_CONVERSION, op->line);
								}
#endif // DEBUG_ENABLED
							}
						}
#ifdef DEBUG_ENABLED
						if (!rh_type.has_type && (op->op != OperatorNode::OP_ASSIGN || lh_type.has_type || op->arguments[0]->type == Node::TYPE_OPERATOR)) {
							_mark_line_as_unsafe(op->line);
						}
#endif // DEBUG_ENABLED
						op->datatype.has_type = type_match;
					} break;
					case OperatorNode::OP_CALL:
					case OperatorNode::OP_PARENT_CALL: {
						_mark_line_as_safe(op->line);
						DataType func_type = _reduce_function_call_type(op);
#ifdef DEBUG_ENABLED
						if (func_type.has_type && (func_type.kind != DataType::BUILTIN || func_type.builtin_type != Variant::NIL)) {
							// Figure out function name for warning
							String func_name = _find_function_name(op);
							if (func_name.empty()) {
								func_name = "<undetected name>";
							}
							_add_warning(GDScriptWarning::RETURN_VALUE_DISCARDED, op->line, func_name);
						}
#endif // DEBUG_ENABLED
						if (error_set) return;
					} break;
					case OperatorNode::OP_YIELD: {
						_mark_line_as_safe(op->line);
						_reduce_node_type(op);
					} break;
					default: {
						_mark_line_as_safe(op->line);
						_reduce_node_type(op); // Test for safety anyway
#ifdef DEBUG_ENABLED
						if (op->op == OperatorNode::OP_TERNARY_IF) {
							_add_warning(GDScriptWarning::STANDALONE_TERNARY, statement->line);
						} else {
							_add_warning(GDScriptWarning::STANDALONE_EXPRESSION, statement->line);
						}
#endif // DEBUG_ENABLED
					}
				}
			} break;
			case Node::TYPE_CONTROL_FLOW: {
				ControlFlowNode *cf = static_cast<ControlFlowNode *>(statement);
				_mark_line_as_safe(cf->line);

				switch (cf->cf_type) {
					case ControlFlowNode::CF_RETURN: {
						DataType function_type = current_function->get_datatype();

						DataType ret_type;
						if (cf->arguments.size() > 0) {
							ret_type = _reduce_node_type(cf->arguments[0]);
							if (error_set) {
								return;
							}
						}

						if (!function_type.has_type) break;

						if (function_type.kind == DataType::BUILTIN && function_type.builtin_type == Variant::NIL) {
							// Return void, should not have arguments
							if (cf->arguments.size() > 0) {
								_set_error("A void function cannot return a value.", cf->line, cf->column);
								return;
							}
						} else {
							// Return something, cannot be empty
							if (cf->arguments.size() == 0) {
								_set_error("A non-void function must return a value.", cf->line, cf->column);
								return;
							}

							if (!_is_type_compatible(function_type, ret_type)) {
								_set_error("The returned value type (" + ret_type.to_string() + ") doesn't match the function return type (" +
												   function_type.to_string() + ").",
										cf->line, cf->column);
								return;
							}
						}
					} break;
					case ControlFlowNode::CF_MATCH: {
						MatchNode *match_node = cf->match;
						_transform_match_statment(match_node);
					} break;
					default: {
						if (cf->body_else) {
							_mark_line_as_safe(cf->body_else->line);
						}
						for (int i = 0; i < cf->arguments.size(); i++) {
							_reduce_node_type(cf->arguments[i]);
						}
					} break;
				}
			} break;
			case Node::TYPE_CONSTANT: {
				ConstantNode *cn = static_cast<ConstantNode *>(statement);
				// Strings are fine since they can be multiline comments
				if (cn->value.get_type() == Variant::STRING) {
					break;
				}
				FALLTHROUGH;
			}
			default: {
				_mark_line_as_safe(statement->line);
				_reduce_node_type(statement); // Test for safety anyway
#ifdef DEBUG_ENABLED
				_add_warning(GDScriptWarning::STANDALONE_EXPRESSION, statement->line);
#endif // DEBUG_ENABLED
			}
		}
	}

	// Parse sub blocks
	for (int i = 0; i < p_block->sub_blocks.size(); i++) {
		current_block = p_block->sub_blocks[i];
		_check_block_types(current_block);
		current_block = p_block;
		if (error_set) return;
	}

#ifdef DEBUG_ENABLED
	// Warnings check
	for (Map<StringName, LocalVarNode *>::Element *E = p_block->variables.front(); E; E = E->next()) {
		LocalVarNode *lv = E->get();
		if (!lv->name.operator String().begins_with("_")) {
			if (lv->usages == 0) {
				_add_warning(GDScriptWarning::UNUSED_VARIABLE, lv->line, lv->name);
			} else if (lv->assignments == 0) {
				_add_warning(GDScriptWarning::UNASSIGNED_VARIABLE, lv->line, lv->name);
			}
		}
	}
#endif // DEBUG_ENABLED
>>>>>>> audio-bus-effect-fixed
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

<<<<<<< HEAD
void GDScriptParser::TreePrinter::print_call(CallNode *p_call) {
	if (p_call->is_super) {
		push_text("super");
		if (p_call->callee != nullptr) {
			push_text(".");
			print_expression(p_call->callee);
		}
	} else {
		print_expression(p_call->callee);
=======
	bool for_completion_error_set = false;
	if (error_set && for_completion) {
		for_completion_error_set = true;
		error_set = false;
	}

	if (error_set) {
		return ERR_PARSE_ERROR;
>>>>>>> audio-bus-effect-fixed
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

<<<<<<< HEAD
	increase_indent();
=======
	if (for_completion_error_set) {
		error_set = true;
	}

	if (error_set) {
		return ERR_PARSE_ERROR;
	}
>>>>>>> audio-bus-effect-fixed

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
