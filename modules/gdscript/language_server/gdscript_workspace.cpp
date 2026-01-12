/**************************************************************************/
/*  gdscript_workspace.cpp                                                */
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

#include "gdscript_workspace.h"

#include "../gdscript.h"
#include "../gdscript_parser.h"
#include "gdscript_language_protocol.h"

#include "core/config/project_settings.h"
#include "core/object/script_language.h"
#include "editor/doc/doc_tools.h"
#include "editor/doc/editor_help.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/settings/editor_settings.h"
#include "scene/resources/packed_scene.h"

void GDScriptWorkspace::_bind_methods() {
	ClassDB::bind_method(D_METHOD("apply_new_signal"), &GDScriptWorkspace::apply_new_signal);
	ClassDB::bind_method(D_METHOD("get_file_path", "uri"), &GDScriptWorkspace::get_file_path);
	ClassDB::bind_method(D_METHOD("get_file_uri", "path"), &GDScriptWorkspace::get_file_uri);
	ClassDB::bind_method(D_METHOD("publish_diagnostics", "path"), &GDScriptWorkspace::publish_diagnostics);
	ClassDB::bind_method(D_METHOD("generate_script_api", "path"), &GDScriptWorkspace::generate_script_api);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("didDeleteFiles"), &GDScriptWorkspace::didDeleteFiles);
	ClassDB::bind_method(D_METHOD("parse_script", "path", "content"), &GDScriptWorkspace::parse_script);
	ClassDB::bind_method(D_METHOD("parse_local_script", "path"), &GDScriptWorkspace::parse_script);
#endif
}

void GDScriptWorkspace::apply_new_signal(Object *obj, String function, PackedStringArray args) {
	Ref<Script> scr = obj->get_script();

	if (scr->get_language()->get_name() != "GDScript") {
		return;
	}

	String function_signature = "func " + function;
	String source = scr->get_source_code();

	if (source.contains(function_signature)) {
		return;
	}

	int first_class = source.find("\nclass ");
	int start_line = 0;
	if (first_class != -1) {
		start_line = source.substr(0, first_class).split("\n").size();
	} else {
		start_line = source.split("\n").size();
	}

	String function_body = "\n\n" + function_signature + "(";
	for (int i = 0; i < args.size(); ++i) {
		function_body += args[i];
		if (i < args.size() - 1) {
			function_body += ", ";
		}
	}
	function_body += ")";
	if (EditorSettings::get_singleton()->get_setting("text_editor/completion/add_type_hints")) {
		function_body += " -> void";
	}
	function_body += ":\n\tpass # Replace with function body.\n";

	LSP::TextEdit text_edit;

	if (first_class != -1) {
		function_body += "\n\n";
	}
	text_edit.range.end.line = text_edit.range.start.line = start_line;

	text_edit.newText = function_body;

	String uri = get_file_uri(scr->get_path());

	LSP::ApplyWorkspaceEditParams params;
	params.edit.add_edit(uri, text_edit);

	GDScriptLanguageProtocol::get_singleton()->request_client("workspace/applyEdit", params.to_json());
}

const LSP::DocumentSymbol *GDScriptWorkspace::get_native_symbol(const String &p_class, const String &p_member) const {
	StringName class_name = p_class;
	StringName empty;

	while (class_name != empty) {
		if (HashMap<StringName, LSP::DocumentSymbol>::ConstIterator E = native_symbols.find(class_name)) {
			const LSP::DocumentSymbol &class_symbol = E->value;

			if (p_member.is_empty()) {
				return &class_symbol;
			} else {
				for (int i = 0; i < class_symbol.children.size(); i++) {
					const LSP::DocumentSymbol &symbol = class_symbol.children[i];
					if (symbol.name == p_member) {
						return &symbol;
					}
				}
			}
		}
		// Might contain pseudo classes like @GDScript that only exist in documentation.
		if (ClassDB::class_exists(class_name)) {
			class_name = ClassDB::get_parent_class(class_name);
		} else {
			break;
		}
	}

	return nullptr;
}

const LSP::DocumentSymbol *GDScriptWorkspace::get_script_symbol(const String &p_path) const {
	ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(p_path);
	if (parser) {
		return &(parser->get_symbols());
	}
	return nullptr;
}

const LSP::DocumentSymbol *GDScriptWorkspace::get_parameter_symbol(const LSP::DocumentSymbol *p_parent, const String &symbol_identifier) {
	for (int i = 0; i < p_parent->children.size(); ++i) {
		const LSP::DocumentSymbol *parameter_symbol = &p_parent->children[i];
		if (!parameter_symbol->detail.is_empty() && parameter_symbol->name == symbol_identifier) {
			return parameter_symbol;
		}
	}

	return nullptr;
}

const LSP::DocumentSymbol *GDScriptWorkspace::get_local_symbol_at(const ExtendGDScriptParser *p_parser, const String &p_symbol_identifier, const LSP::Position p_position) {
	// Go down and pick closest `DocumentSymbol` with `p_symbol_identifier`.

	const LSP::DocumentSymbol *current = &p_parser->get_symbols();
	const LSP::DocumentSymbol *best_match = nullptr;

	while (current) {
		if (current->name == p_symbol_identifier) {
			if (current->selectionRange.contains(p_position)) {
				// Exact match: pos is ON symbol decl identifier.
				return current;
			}

			best_match = current;
		}

		const LSP::DocumentSymbol *parent = current;
		current = nullptr;
		for (const LSP::DocumentSymbol &child : parent->children) {
			if (child.range.contains(p_position)) {
				current = &child;
				break;
			}
		}
	}

	return best_match;
}

void GDScriptWorkspace::reload_all_workspace_scripts() {
	List<String> paths;
	list_script_files("res://", paths);
	for (const String &path : paths) {
		ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(path);
		if (parser == nullptr || parser->parse_result != OK) {
			String err_msg = "LSP: Failed to parse script: " + path;
			if (parser) {
				err_msg += "\n" + parser->get_errors().front()->get().message;
			}
			ERR_PRINT(err_msg);
		}
	}
}

void GDScriptWorkspace::list_script_files(const String &p_root_dir, List<String> &r_files) {
	Error err;
	Ref<DirAccess> dir = DirAccess::open(p_root_dir, &err);
	if (OK != err) {
		return;
	}

	// Ignore scripts in directories with a .gdignore file.
	if (dir->file_exists(".gdignore")) {
		return;
	}

	dir->list_dir_begin();
	String file_name = dir->get_next();
	while (file_name.length()) {
		if (dir->current_is_dir() && file_name != "." && file_name != ".." && file_name != "./") {
			list_script_files(p_root_dir.path_join(file_name), r_files);
		} else if (file_name.ends_with(".gd")) {
			String script_file = p_root_dir.path_join(file_name);
			r_files.push_back(script_file);
		}
		file_name = dir->get_next();
	}
}

#define HANDLE_DOC(m_string) ((is_native ? DTR(m_string) : (m_string)).strip_edges())

Error GDScriptWorkspace::initialize() {
	if (initialized) {
		return OK;
	}

	DocTools *doc = EditorHelp::get_doc_data();
	for (const KeyValue<String, DocData::ClassDoc> &E : doc->class_list) {
		const DocData::ClassDoc &class_data = E.value;
		const bool is_native = !class_data.is_script_doc;
		LSP::DocumentSymbol class_symbol;
		String class_name = E.key;
		class_symbol.name = class_name;
		class_symbol.native_class = class_name;
		class_symbol.kind = LSP::SymbolKind::Class;
		class_symbol.detail = String("<Native> class ") + class_name;
		if (!class_data.inherits.is_empty()) {
			class_symbol.detail += " extends " + class_data.inherits;
		}
		class_symbol.documentation = HANDLE_DOC(class_data.brief_description) + "\n" + HANDLE_DOC(class_data.description);

		for (int i = 0; i < class_data.constants.size(); i++) {
			const DocData::ConstantDoc &const_data = class_data.constants[i];
			LSP::DocumentSymbol symbol;
			symbol.name = const_data.name;
			symbol.native_class = class_name;
			symbol.kind = LSP::SymbolKind::Constant;
			symbol.detail = "const " + class_name + "." + const_data.name;
			if (const_data.enumeration.length()) {
				symbol.detail += ": " + const_data.enumeration;
			}
			symbol.detail += " = " + const_data.value;
			symbol.documentation = HANDLE_DOC(const_data.description);
			class_symbol.children.push_back(symbol);
		}

		for (int i = 0; i < class_data.properties.size(); i++) {
			const DocData::PropertyDoc &data = class_data.properties[i];
			LSP::DocumentSymbol symbol;
			symbol.name = data.name;
			symbol.native_class = class_name;
			symbol.kind = LSP::SymbolKind::Property;
			symbol.detail = "var " + class_name + "." + data.name;
			if (data.enumeration.length()) {
				symbol.detail += ": " + data.enumeration;
			} else {
				symbol.detail += ": " + data.type;
			}
			symbol.documentation = HANDLE_DOC(data.description);
			class_symbol.children.push_back(symbol);
		}

		for (int i = 0; i < class_data.theme_properties.size(); i++) {
			const DocData::ThemeItemDoc &data = class_data.theme_properties[i];
			LSP::DocumentSymbol symbol;
			symbol.name = data.name;
			symbol.native_class = class_name;
			symbol.kind = LSP::SymbolKind::Property;
			symbol.detail = "<Theme> var " + class_name + "." + data.name + ": " + data.type;
			symbol.documentation = HANDLE_DOC(data.description);
			class_symbol.children.push_back(symbol);
		}

		Vector<DocData::MethodDoc> method_likes;
		method_likes.append_array(class_data.methods);
		method_likes.append_array(class_data.annotations);
		const int constructors_start_idx = method_likes.size();
		method_likes.append_array(class_data.constructors);
		const int operator_start_idx = method_likes.size();
		method_likes.append_array(class_data.operators);
		const int signal_start_idx = method_likes.size();
		method_likes.append_array(class_data.signals);

		for (int i = 0; i < method_likes.size(); i++) {
			const DocData::MethodDoc &data = method_likes[i];

			LSP::DocumentSymbol symbol;
			symbol.name = data.name;
			symbol.native_class = class_name;

			if (i >= signal_start_idx) {
				symbol.kind = LSP::SymbolKind::Event;
			} else if (i >= operator_start_idx) {
				symbol.kind = LSP::SymbolKind::Operator;
			} else if (i >= constructors_start_idx) {
				symbol.kind = LSP::SymbolKind::Constructor;
			} else {
				symbol.kind = LSP::SymbolKind::Method;
			}

			String params = "";
			bool arg_default_value_started = false;
			for (int j = 0; j < data.arguments.size(); j++) {
				const DocData::ArgumentDoc &arg = data.arguments[j];

				LSP::DocumentSymbol symbol_arg;
				symbol_arg.name = arg.name;
				symbol_arg.kind = LSP::SymbolKind::Variable;
				symbol_arg.detail = arg.type;

				if (!arg_default_value_started && !arg.default_value.is_empty()) {
					arg_default_value_started = true;
				}
				String arg_str = arg.name + ": " + arg.type;
				if (arg_default_value_started) {
					arg_str += " = " + arg.default_value;
				}
				if (j < data.arguments.size() - 1) {
					arg_str += ", ";
				}
				params += arg_str;

				symbol.children.push_back(symbol_arg);
			}
			if (data.qualifiers.contains("vararg")) {
				params += params.is_empty() ? "..." : ", ...";
			}

			String return_type = data.return_type;
			if (return_type.is_empty()) {
				return_type = "void";
			}
			symbol.detail = "func " + class_name + "." + data.name + "(" + params + ") -> " + return_type;
			symbol.documentation = HANDLE_DOC(data.description);
			class_symbol.children.push_back(symbol);
		}

		native_symbols.insert(class_name, class_symbol);
	}

	reload_all_workspace_scripts();

	if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
		for (const KeyValue<StringName, LSP::DocumentSymbol> &E : native_symbols) {
			ClassMembers members;
			const LSP::DocumentSymbol &class_symbol = E.value;
			for (int i = 0; i < class_symbol.children.size(); i++) {
				const LSP::DocumentSymbol &symbol = class_symbol.children[i];
				members.insert(symbol.name, &symbol);
			}
			native_members.insert(E.key, members);
		}
	}

	EditorNode *editor_node = EditorNode::get_singleton();
	editor_node->connect("script_add_function_request", callable_mp(this, &GDScriptWorkspace::apply_new_signal));

	return OK;
}

static bool is_valid_rename_target(const LSP::DocumentSymbol *p_symbol) {
	// Must be valid symbol.
	if (!p_symbol) {
		return false;
	}

	// Cannot rename builtin.
	if (!p_symbol->native_class.is_empty()) {
		return false;
	}

	// Source must be available.
	if (p_symbol->script_path.is_empty()) {
		return false;
	}

	return true;
}

Dictionary GDScriptWorkspace::rename(const LSP::TextDocumentPositionParams &p_doc_pos, const String &new_name) {
	LSP::WorkspaceEdit edit;

	const LSP::DocumentSymbol *reference_symbol = resolve_symbol(p_doc_pos);
	if (is_valid_rename_target(reference_symbol)) {
		Vector<LSP::Location> usages = find_all_usages(*reference_symbol);
		for (int i = 0; i < usages.size(); ++i) {
			LSP::Location loc = usages[i];

			edit.add_change(loc.uri, loc.range.start.line, loc.range.start.character, loc.range.end.character, new_name);
		}
	}

	return edit.to_json();
}

bool GDScriptWorkspace::can_rename(const LSP::TextDocumentPositionParams &p_doc_pos, LSP::DocumentSymbol &r_symbol, LSP::Range &r_range) {
	const LSP::DocumentSymbol *reference_symbol = resolve_symbol(p_doc_pos);
	if (!is_valid_rename_target(reference_symbol)) {
		return false;
	}

	String path = get_file_path(p_doc_pos.textDocument.uri);
	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(path);
	if (parser) {
		_ALLOW_DISCARD_ parser->get_identifier_under_position(p_doc_pos.position, r_range);
		r_symbol = *reference_symbol;
		return true;
	}

	return false;
}

Vector<LSP::Location> GDScriptWorkspace::find_usages_in_file(const LSP::DocumentSymbol &p_symbol, const String &p_file_path) {
	Vector<LSP::Location> usages;

	String identifier = p_symbol.name;
	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(p_file_path);
	if (parser) {
		const PackedStringArray &content = parser->get_lines();
		for (int i = 0; i < content.size(); ++i) {
			String line = content[i];

			int character = line.find(identifier);
			while (character > -1) {
				LSP::TextDocumentPositionParams params;

				LSP::TextDocumentIdentifier text_doc;
				text_doc.uri = get_file_uri(p_file_path);

				params.textDocument = text_doc;
				params.position.line = i;
				params.position.character = character;

				const LSP::DocumentSymbol *other_symbol = resolve_symbol(params);

				if (other_symbol == &p_symbol) {
					LSP::Location loc;
					loc.uri = text_doc.uri;
					loc.range.start = params.position;
					loc.range.end.line = params.position.line;
					loc.range.end.character = params.position.character + identifier.length();
					usages.append(loc);
				}

				character = line.find(identifier, character + 1);
			}
		}
	}

	return usages;
}

Vector<LSP::Location> GDScriptWorkspace::find_all_usages(const LSP::DocumentSymbol &p_symbol) {
	if (p_symbol.local) {
		// Only search in current document.
		return find_usages_in_file(p_symbol, p_symbol.script_path);
	}
	// Search in all documents.
	List<String> paths;
	list_script_files("res://", paths);

	Vector<LSP::Location> usages;
	for (const String &path : paths) {
		usages.append_array(find_usages_in_file(p_symbol, path));
	}
	return usages;
}

String GDScriptWorkspace::get_file_path(const String &p_uri) {
	int port;
	String scheme;
	String host;
	String encoded_path;
	String fragment;

	// Don't use the returned error, the result isn't OK for URIs that are not valid web URLs.
	p_uri.parse_url(scheme, host, port, encoded_path, fragment);

	// TODO: Make the parsing RFC-3986 compliant.
	ERR_FAIL_COND_V_MSG(scheme != "file" && scheme != "file:" && scheme != "file://", String(), "LSP: The language server only supports the file protocol: " + p_uri);

	// Treat host like authority for now and ignore the port. It's an edge case for invalid file URI's anyway.
	ERR_FAIL_COND_V_MSG(host != "" && host != "localhost", String(), "LSP: The language server does not support nonlocal files: " + p_uri);

	// If query or fragment are present, the URI is not a valid file URI as per RFC-8089.
	// We currently don't handle the query and it will be part of the path. However,
	// this should not be a problem for a correct file URI.
	ERR_FAIL_COND_V_MSG(fragment != "", String(), "LSP: Received malformed file URI: " + p_uri);

	String canonical_res = ProjectSettings::get_singleton()->get_resource_path();
	String simple_path = encoded_path.uri_file_decode().simplify_path();

	// First try known paths that point to res://, to reduce file system interaction.
	bool res_adjusted = false;
	for (const String &res_path : absolute_res_paths) {
		if (simple_path.begins_with(res_path)) {
			res_adjusted = true;
			simple_path = "res://" + simple_path.substr(res_path.size());
			break;
		}
	}

	// Traverse the path and compare each directory with res://
	if (!res_adjusted) {
		Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

		int offset = 0;
		while (offset <= simple_path.length()) {
			offset = simple_path.find_char('/', offset);
			if (offset == -1) {
				offset = simple_path.length();
			}

			String part = simple_path.substr(0, offset);

			if (!part.is_empty()) {
				bool is_equal = dir->is_equivalent(canonical_res, part);

				if (is_equal) {
					absolute_res_paths.insert(part);
					res_adjusted = true;
					simple_path = "res://" + simple_path.substr(offset + 1);
					break;
				}
			}

			offset += 1;
		}

		// Could not resolve the path to the project.
		if (!res_adjusted) {
			return simple_path;
		}
	}

	// Resolve the file inside of the project using EditorFileSystem.
	EditorFileSystemDirectory *editor_dir;
	int file_idx;
	editor_dir = EditorFileSystem::get_singleton()->find_file(simple_path, &file_idx);
	if (editor_dir) {
		return editor_dir->get_file_path(file_idx);
	}

	return simple_path;
}

String GDScriptWorkspace::get_file_uri(const String &p_path) const {
	String path = ProjectSettings::get_singleton()->globalize_path(p_path).lstrip("/");
	LocalVector<String> encoded_parts;
	for (const String &part : path.split("/")) {
		encoded_parts.push_back(part.uri_encode());
	}

	// Always return file URI's with authority part (encoding drive letters with leading slash), to maintain compat with RFC-1738 which required it.
	return "file:///" + String("/").join(Vector<String>(encoded_parts));
}

void GDScriptWorkspace::publish_diagnostics(const String &p_path) {
	Dictionary params;
	Array errors;

	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(p_path);
	if (parser) {
		const Vector<LSP::Diagnostic> &list = parser->get_diagnostics();
		errors.resize(list.size());
		for (int i = 0; i < list.size(); ++i) {
			errors[i] = list[i].to_json();
		}
	}
	params["diagnostics"] = errors;
	params["uri"] = get_file_uri(p_path);
	GDScriptLanguageProtocol::get_singleton()->notify_client("textDocument/publishDiagnostics", params);
}

void GDScriptWorkspace::_get_owners(EditorFileSystemDirectory *efsd, String p_path, List<String> &owners) {
	if (!efsd) {
		return;
	}

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_get_owners(efsd->get_subdir(i), p_path, owners);
	}

	for (int i = 0; i < efsd->get_file_count(); i++) {
		Vector<String> deps = efsd->get_file_deps(i);
		bool found = false;
		for (int j = 0; j < deps.size(); j++) {
			if (deps[j] == p_path) {
				found = true;
				break;
			}
		}
		if (!found) {
			continue;
		}

		owners.push_back(efsd->get_file_path(i));
	}
}

Node *GDScriptWorkspace::_get_owner_scene_node(String p_path) {
	Node *owner_scene_node = nullptr;
	List<String> owners;

	_get_owners(EditorFileSystem::get_singleton()->get_filesystem(), p_path, owners);

	for (const String &owner : owners) {
		NodePath owner_path = owner;
		Ref<Resource> owner_res = ResourceLoader::load(String(owner_path));
		if (Object::cast_to<PackedScene>(owner_res.ptr())) {
			Ref<PackedScene> owner_packed_scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*owner_res));
			owner_scene_node = owner_packed_scene->instantiate();
			break;
		}
	}

	return owner_scene_node;
}

void GDScriptWorkspace::completion(const LSP::CompletionParams &p_params, List<ScriptLanguage::CodeCompletionOption> *r_options) {
	String path = get_file_path(p_params.textDocument.uri);
	String call_hint;
	bool forced = false;

	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(path);
	if (parser) {
		Node *owner_scene_node = _get_owner_scene_node(path);

		Array stack;
		Node *current = nullptr;
		if (owner_scene_node != nullptr) {
			stack.push_back(owner_scene_node);

			while (!stack.is_empty()) {
				current = Object::cast_to<Node>(stack.pop_back());
				Ref<GDScript> scr = current->get_script();
				if (scr.is_valid() && GDScript::is_canonically_equal_paths(scr->get_path(), path)) {
					break;
				}
				for (int i = 0; i < current->get_child_count(); ++i) {
					stack.push_back(current->get_child(i));
				}
			}

			Ref<GDScript> scr = current->get_script();
			if (scr.is_null() || !GDScript::is_canonically_equal_paths(scr->get_path(), path)) {
				current = owner_scene_node;
			}
		}

		String code = parser->get_text_for_completion(p_params.position);
		GDScriptLanguage::get_singleton()->complete_code(code, path, current, r_options, forced, call_hint);
		if (owner_scene_node) {
			memdelete(owner_scene_node);
		}
	}
}

const LSP::DocumentSymbol *GDScriptWorkspace::resolve_symbol(const LSP::TextDocumentPositionParams &p_doc_pos, const String &p_symbol_name, bool p_func_required) {
	const LSP::DocumentSymbol *symbol = nullptr;

	String path = get_file_path(p_doc_pos.textDocument.uri);

	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(path);
	if (parser) {
		String symbol_identifier = p_symbol_name;
		if (symbol_identifier.get_slice_count("(") > 0) {
			symbol_identifier = symbol_identifier.get_slicec('(', 0);
		}

		LSP::Position pos = p_doc_pos.position;
		if (symbol_identifier.is_empty()) {
			LSP::Range range;
			symbol_identifier = parser->get_identifier_under_position(p_doc_pos.position, range);
			pos.character = range.end.character;
		}

		if (!symbol_identifier.is_empty()) {
			if (ScriptServer::is_global_class(symbol_identifier)) {
				String class_path = ScriptServer::get_global_class_path(symbol_identifier);
				symbol = get_script_symbol(class_path);

			} else {
				ScriptLanguage::LookupResult ret;
				if (symbol_identifier == "new" && parser->get_lines()[p_doc_pos.position.line].remove_chars(" \t").contains("new(")) {
					symbol_identifier = "_init";
				}
				if (OK == GDScriptLanguage::get_singleton()->lookup_code(parser->get_text_for_lookup_symbol(pos, symbol_identifier, p_func_required), symbol_identifier, path, nullptr, ret)) {
					if (ret.location >= 0) {
						String target_script_path = path;
						if (ret.script.is_valid()) {
							target_script_path = ret.script->get_path();
						} else if (!ret.script_path.is_empty()) {
							target_script_path = ret.script_path;
						}

						const ExtendGDScriptParser *target_parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(target_script_path);
						if (target_parser) {
							symbol = target_parser->get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(ret.location), symbol_identifier);

							if (symbol) {
								switch (symbol->kind) {
									case LSP::SymbolKind::Function: {
										if (symbol->name != symbol_identifier) {
											symbol = get_parameter_symbol(symbol, symbol_identifier);
										}
									} break;
								}
							}
						}
					} else {
						String member = ret.class_member;
						if (member.is_empty() && symbol_identifier != ret.class_name) {
							member = symbol_identifier;
						}
						symbol = get_native_symbol(ret.class_name, member);
					}
				} else {
					symbol = get_local_symbol_at(parser, symbol_identifier, p_doc_pos.position);
					if (!symbol) {
						symbol = parser->get_member_symbol(symbol_identifier);
					}
				}
			}
		}
	}

	return symbol;
}

const LSP::DocumentSymbol *GDScriptWorkspace::resolve_native_symbol(const LSP::NativeSymbolInspectParams &p_params) {
	if (HashMap<StringName, LSP::DocumentSymbol>::Iterator E = native_symbols.find(p_params.native_class)) {
		const LSP::DocumentSymbol &symbol = E->value;
		if (p_params.symbol_name.is_empty() || p_params.symbol_name == symbol.name) {
			return &symbol;
		}

		for (int i = 0; i < symbol.children.size(); ++i) {
			if (symbol.children[i].name == p_params.symbol_name) {
				return &(symbol.children[i]);
			}
		}
	}

	return nullptr;
}

void GDScriptWorkspace::resolve_document_links(const String &p_uri, List<LSP::DocumentLink> &r_list) {
	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(get_file_path(p_uri));
	if (parser && parser->parse_result == Error::OK) {
		const List<LSP::DocumentLink> &links = parser->get_document_links();
		for (const LSP::DocumentLink &E : links) {
			r_list.push_back(E);
		}
	}
}

Dictionary GDScriptWorkspace::generate_script_api(const String &p_path) {
	Dictionary api;

	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(p_path);
	if (parser) {
		api = parser->generate_api();
	}
	return api;
}

Error GDScriptWorkspace::resolve_signature(const LSP::TextDocumentPositionParams &p_doc_pos, LSP::SignatureHelp &r_signature) {
	const ExtendGDScriptParser *parser = GDScriptLanguageProtocol::get_singleton()->get_parse_result(get_file_path(p_doc_pos.textDocument.uri));
	if (parser) {
		LSP::TextDocumentPositionParams text_pos;
		text_pos.textDocument = p_doc_pos.textDocument;

		if (parser->get_left_function_call(p_doc_pos.position, text_pos.position, r_signature.activeParameter) == OK) {
			List<const LSP::DocumentSymbol *> symbols;

			if (const LSP::DocumentSymbol *symbol = resolve_symbol(text_pos)) {
				symbols.push_back(symbol);
			} else if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
				GDScriptLanguageProtocol::get_singleton()->resolve_related_symbols(text_pos, symbols);
			}

			for (const LSP::DocumentSymbol *const &symbol : symbols) {
				if (symbol->kind == LSP::SymbolKind::Method || symbol->kind == LSP::SymbolKind::Function) {
					LSP::SignatureInformation signature_info;
					signature_info.label = symbol->detail;
					signature_info.documentation = symbol->render();

					for (int i = 0; i < symbol->children.size(); i++) {
						const LSP::DocumentSymbol &arg = symbol->children[i];
						LSP::ParameterInformation arg_info;
						arg_info.label = arg.name;
						signature_info.parameters.push_back(arg_info);
					}
					r_signature.signatures.push_back(signature_info);
					break;
				}
			}

			if (r_signature.signatures.size()) {
				return OK;
			}
		}
	}
	return ERR_METHOD_NOT_FOUND;
}

GDScriptWorkspace::GDScriptWorkspace() {}

GDScriptWorkspace::~GDScriptWorkspace() {}
