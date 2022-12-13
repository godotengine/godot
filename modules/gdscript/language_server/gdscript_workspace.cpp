/*************************************************************************/
/*  gdscript_workspace.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdscript_workspace.h"

#include "../gdscript.h"
#include "../gdscript_parser.h"
#include "core/project_settings.h"
#include "core/script_language.h"
#include "editor/editor_file_system.h"
#include "editor/editor_help.h"
#include "editor/editor_node.h"
#include "gdscript_language_protocol.h"
#include "scene/resources/packed_scene.h"

void GDScriptWorkspace::_bind_methods() {
	ClassDB::bind_method(D_METHOD("didDeleteFiles"), &GDScriptWorkspace::did_delete_files);
	ClassDB::bind_method(D_METHOD("symbol"), &GDScriptWorkspace::symbol);
	ClassDB::bind_method(D_METHOD("parse_script", "path", "content"), &GDScriptWorkspace::parse_script);
	ClassDB::bind_method(D_METHOD("parse_local_script", "path"), &GDScriptWorkspace::parse_local_script);
	ClassDB::bind_method(D_METHOD("get_file_path", "uri"), &GDScriptWorkspace::get_file_path);
	ClassDB::bind_method(D_METHOD("get_file_uri", "path"), &GDScriptWorkspace::get_file_uri);
	ClassDB::bind_method(D_METHOD("publish_diagnostics", "path"), &GDScriptWorkspace::publish_diagnostics);
	ClassDB::bind_method(D_METHOD("generate_script_api", "path"), &GDScriptWorkspace::generate_script_api);
	ClassDB::bind_method(D_METHOD("apply_new_signal", "obj", "function", "args"), &GDScriptWorkspace::apply_new_signal);
}

void GDScriptWorkspace::apply_new_signal(Object *obj, String function, PoolStringArray args) {
	Ref<Script> script = obj->get_script();

	if (script->get_language()->get_name() != "GDScript") {
		return;
	}

	String function_signature = "func " + function;
	String source = script->get_source_code();

	if (source.find(function_signature) != -1) {
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

	lsp::TextEdit text_edit;

	if (first_class != -1) {
		function_body += "\n\n";
	}
	text_edit.range.end.line = text_edit.range.start.line = start_line;

	text_edit.newText = function_body;

	String uri = get_file_uri(script->get_path());

	lsp::ApplyWorkspaceEditParams params;
	params.edit.add_edit(uri, text_edit);

	GDScriptLanguageProtocol::get_singleton()->request_client("workspace/applyEdit", params.to_json());
}

void GDScriptWorkspace::did_delete_files(const Dictionary &p_params) {
	Array files = p_params["files"];
	for (int i = 0; i < files.size(); ++i) {
		Dictionary file = files[i];
		String uri = file["uri"];
		String path = get_file_path(uri);
		parse_script(path, "");
	}
}

void GDScriptWorkspace::remove_cache_parser(const String &p_path) {
	Map<String, ExtendGDScriptParser *>::Element *parser = parse_results.find(p_path);
	Map<String, ExtendGDScriptParser *>::Element *script = scripts.find(p_path);
	if (parser && script) {
		if (script->get() && script->get() == parser->get()) {
			memdelete(script->get());
		} else {
			memdelete(script->get());
			memdelete(parser->get());
		}
		parse_results.erase(p_path);
		scripts.erase(p_path);
	} else if (parser) {
		memdelete(parser->get());
		parse_results.erase(p_path);
	} else if (script) {
		memdelete(script->get());
		scripts.erase(p_path);
	}
}

const lsp::DocumentSymbol *GDScriptWorkspace::get_native_symbol(const String &p_class, const String &p_member) const {
	StringName class_name = p_class;
	StringName empty;

	while (class_name != empty) {
		if (const Map<StringName, lsp::DocumentSymbol>::Element *E = native_symbols.find(class_name)) {
			const lsp::DocumentSymbol &class_symbol = E->value();

			if (p_member.empty()) {
				return &class_symbol;
			} else {
				for (int i = 0; i < class_symbol.children.size(); i++) {
					const lsp::DocumentSymbol &symbol = class_symbol.children[i];
					if (symbol.name == p_member) {
						return &symbol;
					}
				}
			}
		}
		class_name = ClassDB::get_parent_class(class_name);
	}

	return nullptr;
}

const lsp::DocumentSymbol *GDScriptWorkspace::get_script_symbol(const String &p_path) const {
	const Map<String, ExtendGDScriptParser *>::Element *S = scripts.find(p_path);
	if (S) {
		return &(S->get()->get_symbols());
	}
	return nullptr;
}

const lsp::DocumentSymbol *GDScriptWorkspace::get_parameter_symbol(const lsp::DocumentSymbol *p_parent, const String &symbol_identifier) {
	for (int i = 0; i < p_parent->children.size(); ++i) {
		const lsp::DocumentSymbol *parameter_symbol = &p_parent->children[i];
		if (!parameter_symbol->detail.empty() && parameter_symbol->name == symbol_identifier) {
			return parameter_symbol;
		}
	}

	return nullptr;
}

const lsp::DocumentSymbol *GDScriptWorkspace::get_local_symbol(const ExtendGDScriptParser *p_parser, const String &p_symbol_identifier) {
	const lsp::DocumentSymbol *class_symbol = &p_parser->get_symbols();

	for (int i = 0; i < class_symbol->children.size(); ++i) {
		if (class_symbol->children[i].kind == lsp::SymbolKind::Function || class_symbol->children[i].kind == lsp::SymbolKind::Class) {
			const lsp::DocumentSymbol *function_symbol = &class_symbol->children[i];

			for (int l = 0; l < function_symbol->children.size(); ++l) {
				const lsp::DocumentSymbol *local = &function_symbol->children[l];
				if (!local->detail.empty() && local->name == p_symbol_identifier) {
					return local;
				}
			}
		}
	}

	return nullptr;
}

void GDScriptWorkspace::reload_all_workspace_scripts() {
	List<String> paths;
	list_script_files("res://", paths);
	for (List<String>::Element *E = paths.front(); E; E = E->next()) {
		const String &path = E->get();
		Error err;
		String content = FileAccess::get_file_as_string(path, &err);
		ERR_CONTINUE(err != OK);
		err = parse_script(path, content);

		if (err != OK) {
			Map<String, ExtendGDScriptParser *>::Element *S = parse_results.find(path);
			String err_msg = "Failed parse script " + path;
			if (S) {
				err_msg += "\n" + S->get()->get_error();
			}
			ERR_CONTINUE_MSG(err != OK, err_msg);
		}
	}
}

void GDScriptWorkspace::list_script_files(const String &p_root_dir, List<String> &r_files) {
	Error err;
	DirAccessRef dir = DirAccess::open(p_root_dir, &err);
	if (OK == err) {
		dir->list_dir_begin();
		String file_name = dir->get_next();
		while (file_name.length()) {
			if (dir->current_is_dir() && file_name != "." && file_name != ".." && file_name != "./") {
				list_script_files(p_root_dir.plus_file(file_name), r_files);
			} else if (file_name.ends_with(".gd")) {
				String script_file = p_root_dir.plus_file(file_name);
				r_files.push_back(script_file);
			}
			file_name = dir->get_next();
		}
	}
}

ExtendGDScriptParser *GDScriptWorkspace::get_parse_successed_script(const String &p_path) {
	const Map<String, ExtendGDScriptParser *>::Element *S = scripts.find(p_path);
	if (!S) {
		parse_local_script(p_path);
		S = scripts.find(p_path);
	}
	if (S) {
		return S->get();
	}
	return nullptr;
}

ExtendGDScriptParser *GDScriptWorkspace::get_parse_result(const String &p_path) {
	const Map<String, ExtendGDScriptParser *>::Element *S = parse_results.find(p_path);
	if (!S) {
		parse_local_script(p_path);
		S = parse_results.find(p_path);
	}
	if (S) {
		return S->get();
	}
	return nullptr;
}

Array GDScriptWorkspace::symbol(const Dictionary &p_params) {
	String query = p_params["query"];
	Array arr;
	if (!query.empty()) {
		for (Map<String, ExtendGDScriptParser *>::Element *E = scripts.front(); E; E = E->next()) {
			Vector<lsp::DocumentedSymbolInformation> script_symbols;
			E->get()->get_symbols().symbol_tree_as_list(E->key(), script_symbols);
			for (int i = 0; i < script_symbols.size(); ++i) {
				if (query.is_subsequence_ofi(script_symbols[i].name)) {
					lsp::DocumentedSymbolInformation symbol = script_symbols[i];
					symbol.location.uri = get_file_uri(symbol.location.uri);
					arr.push_back(symbol.to_json());
				}
			}
		}
	}
	return arr;
}

Error GDScriptWorkspace::initialize() {
	if (initialized) {
		return OK;
	}

	DocData *doc = EditorHelp::get_doc_data();
	for (Map<String, DocData::ClassDoc>::Element *E = doc->class_list.front(); E; E = E->next()) {
		const DocData::ClassDoc &class_data = E->value();
		lsp::DocumentSymbol class_symbol;
		String class_name = E->key();
		class_symbol.name = class_name;
		class_symbol.native_class = class_name;
		class_symbol.kind = lsp::SymbolKind::Class;
		class_symbol.detail = String("<Native> class ") + class_name;
		if (!class_data.inherits.empty()) {
			class_symbol.detail += " extends " + class_data.inherits;
		}
		class_symbol.documentation = class_data.brief_description + "\n" + class_data.description;

		for (int i = 0; i < class_data.constants.size(); i++) {
			const DocData::ConstantDoc &const_data = class_data.constants[i];
			lsp::DocumentSymbol symbol;
			symbol.name = const_data.name;
			symbol.native_class = class_name;
			symbol.kind = lsp::SymbolKind::Constant;
			symbol.detail = "const " + class_name + "." + const_data.name;
			if (const_data.enumeration.length()) {
				symbol.detail += ": " + const_data.enumeration;
			}
			symbol.detail += " = " + const_data.value;
			symbol.documentation = const_data.description;
			class_symbol.children.push_back(symbol);
		}

		for (int i = 0; i < class_data.properties.size(); i++) {
			const DocData::PropertyDoc &data = class_data.properties[i];
			lsp::DocumentSymbol symbol;
			symbol.name = data.name;
			symbol.native_class = class_name;
			symbol.kind = lsp::SymbolKind::Property;
			symbol.detail = "var " + class_name + "." + data.name;
			if (data.enumeration.length()) {
				symbol.detail += ": " + data.enumeration;
			} else {
				symbol.detail += ": " + data.type;
			}
			symbol.documentation = data.description;
			class_symbol.children.push_back(symbol);
		}

		for (int i = 0; i < class_data.theme_properties.size(); i++) {
			const DocData::ThemeItemDoc &data = class_data.theme_properties[i];
			lsp::DocumentSymbol symbol;
			symbol.name = data.name;
			symbol.native_class = class_name;
			symbol.kind = lsp::SymbolKind::Property;
			symbol.detail = "<Theme> var " + class_name + "." + data.name + ": " + data.type;
			symbol.documentation = data.description;
			class_symbol.children.push_back(symbol);
		}

		Vector<DocData::MethodDoc> methods_signals;
		methods_signals.append_array(class_data.methods);
		const int signal_start_idx = methods_signals.size();
		methods_signals.append_array(class_data.signals);

		for (int i = 0; i < methods_signals.size(); i++) {
			const DocData::MethodDoc &data = methods_signals[i];

			lsp::DocumentSymbol symbol;
			symbol.name = data.name;
			symbol.native_class = class_name;
			symbol.kind = i >= signal_start_idx ? lsp::SymbolKind::Event : lsp::SymbolKind::Method;

			String params = "";
			bool arg_default_value_started = false;
			for (int j = 0; j < data.arguments.size(); j++) {
				const DocData::ArgumentDoc &arg = data.arguments[j];

				lsp::DocumentSymbol symbol_arg;
				symbol_arg.name = arg.name;
				symbol_arg.kind = lsp::SymbolKind::Variable;
				symbol_arg.detail = arg.type;

				if (!arg_default_value_started && !arg.default_value.empty()) {
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
			if (data.qualifiers.find("vararg") != -1) {
				params += params.empty() ? "..." : ", ...";
			}

			String return_type = data.return_type;
			if (return_type.empty()) {
				return_type = "void";
			}
			symbol.detail = "func " + class_name + "." + data.name + "(" + params + ") -> " + return_type;
			symbol.documentation = data.description;
			class_symbol.children.push_back(symbol);
		}

		native_symbols.insert(class_name, class_symbol);
	}

	reload_all_workspace_scripts();

	if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
		for (Map<StringName, lsp::DocumentSymbol>::Element *E = native_symbols.front(); E; E = E->next()) {
			ClassMembers members;
			const lsp::DocumentSymbol &class_symbol = E->get();
			for (int i = 0; i < class_symbol.children.size(); i++) {
				const lsp::DocumentSymbol &symbol = class_symbol.children[i];
				members.set(symbol.name, &symbol);
			}
			native_members.set(E->key(), members);
		}

		// cache member completions
		for (Map<String, ExtendGDScriptParser *>::Element *S = scripts.front(); S; S = S->next()) {
			S->get()->get_member_completions();
		}
	}

	EditorNode *editor_node = EditorNode::get_singleton();
	editor_node->connect("script_add_function_request", this, "apply_new_signal");

	return OK;
}

Error GDScriptWorkspace::parse_script(const String &p_path, const String &p_content) {
	ExtendGDScriptParser *parser = memnew(ExtendGDScriptParser);
	Error err = parser->parse(p_content, p_path);
	Map<String, ExtendGDScriptParser *>::Element *last_parser = parse_results.find(p_path);
	Map<String, ExtendGDScriptParser *>::Element *last_script = scripts.find(p_path);

	if (err == OK) {
		remove_cache_parser(p_path);
		parse_results[p_path] = parser;
		scripts[p_path] = parser;

	} else {
		if (last_parser && last_script && last_parser->get() != last_script->get()) {
			memdelete(last_parser->get());
		}
		parse_results[p_path] = parser;
	}

	publish_diagnostics(p_path);

	return err;
}

Dictionary GDScriptWorkspace::rename(const lsp::TextDocumentPositionParams &p_doc_pos, const String &new_name) {
	Error err;
	String path = get_file_path(p_doc_pos.textDocument.uri);

	lsp::WorkspaceEdit edit;

	List<String> paths;
	list_script_files("res://", paths);

	const lsp::DocumentSymbol *reference_symbol = resolve_symbol(p_doc_pos);
	if (reference_symbol) {
		String identifier = reference_symbol->name;

		for (List<String>::Element *PE = paths.front(); PE; PE = PE->next()) {
			Vector<String> content = FileAccess::get_file_as_string(PE->get(), &err).split("\n");
			for (int i = 0; i < content.size(); ++i) {
				String line = content[i];

				int character = line.find(identifier);
				while (character > -1) {
					lsp::TextDocumentPositionParams params;

					lsp::TextDocumentIdentifier text_doc;
					text_doc.uri = get_file_uri(PE->get());

					params.textDocument = text_doc;
					params.position.line = i;
					params.position.character = character;

					const lsp::DocumentSymbol *other_symbol = resolve_symbol(params);

					if (other_symbol == reference_symbol) {
						edit.add_change(text_doc.uri, i, character, character + identifier.length(), new_name);
					}

					character = line.find(identifier, character + 1);
				}
			}
		}
	}

	return edit.to_json();
}

Error GDScriptWorkspace::parse_local_script(const String &p_path) {
	Error err;
	String content = FileAccess::get_file_as_string(p_path, &err);
	if (err == OK) {
		err = parse_script(p_path, content);
	}
	return err;
}

String GDScriptWorkspace::get_file_path(const String &p_uri) const {
	String path = p_uri.http_unescape();
	String base_uri = root_uri.http_unescape();
	path = path.replacen(base_uri + "/", "res://");
	return path;
}

String GDScriptWorkspace::get_file_uri(const String &p_path) const {
	String uri = p_path;
	uri = uri.replace("res://", root_uri + "/");
	return uri;
}

void GDScriptWorkspace::publish_diagnostics(const String &p_path) {
	Dictionary params;
	Array errors;
	const Map<String, ExtendGDScriptParser *>::Element *ele = parse_results.find(p_path);
	if (ele) {
		const Vector<lsp::Diagnostic> &list = ele->get()->get_diagnostics();
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

	for (int i = 0; i < owners.size(); i++) {
		NodePath owner_path = owners[i];
		RES owner_res = ResourceLoader::load(owner_path);
		if (Object::cast_to<PackedScene>(owner_res.ptr())) {
			Ref<PackedScene> owner_packed_scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*owner_res));
			owner_scene_node = owner_packed_scene->instance();
			break;
		}
	}

	return owner_scene_node;
}

void GDScriptWorkspace::completion(const lsp::CompletionParams &p_params, List<ScriptCodeCompletionOption> *r_options) {
	String path = get_file_path(p_params.textDocument.uri);
	String call_hint;
	bool forced = false;

	if (const ExtendGDScriptParser *parser = get_parse_result(path)) {
		Node *owner_scene_node = _get_owner_scene_node(path);

		Array stack;
		Node *current = nullptr;

		if (owner_scene_node) {
			stack.push_back(owner_scene_node);
			while (!stack.empty()) {
				current = stack.pop_back();
				Ref<GDScript> script = current->get_script();
				if (script.is_valid() && script->get_path() == path) {
					break;
				}
				for (int i = 0; i < current->get_child_count(); ++i) {
					stack.push_back(current->get_child(i));
				}
			}

			Ref<GDScript> script = current->get_script();
			if (!script.is_valid() || script->get_path() != path) {
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

const lsp::DocumentSymbol *GDScriptWorkspace::resolve_symbol(const lsp::TextDocumentPositionParams &p_doc_pos, const String &p_symbol_name, bool p_func_required) {
	const lsp::DocumentSymbol *symbol = nullptr;

	String path = get_file_path(p_doc_pos.textDocument.uri);
	if (const ExtendGDScriptParser *parser = get_parse_result(path)) {
		String symbol_identifier = p_symbol_name;
		Vector<String> identifier_parts = symbol_identifier.split("(");
		if (identifier_parts.size()) {
			symbol_identifier = identifier_parts[0];
		}

		lsp::Position pos = p_doc_pos.position;
		if (symbol_identifier.empty()) {
			Vector2i offset;
			symbol_identifier = parser->get_identifier_under_position(p_doc_pos.position, offset);
			pos.character += offset.y;
		}

		if (!symbol_identifier.empty()) {
			if (ScriptServer::is_global_class(symbol_identifier)) {
				String class_path = ScriptServer::get_global_class_path(symbol_identifier);
				symbol = get_script_symbol(class_path);

			} else {
				ScriptLanguage::LookupResult ret;
				if (symbol_identifier == "new" && parser->get_lines()[p_doc_pos.position.line].replace(" ", "").replace("\t", "").find("new(") > -1) {
					symbol_identifier = "_init";
				}
				if (OK == GDScriptLanguage::get_singleton()->lookup_code(parser->get_text_for_lookup_symbol(pos, symbol_identifier, p_func_required), symbol_identifier, path, nullptr, ret)) {
					if (ret.type == ScriptLanguage::LookupResult::RESULT_SCRIPT_LOCATION) {
						String target_script_path = path;
						if (!ret.script.is_null()) {
							target_script_path = ret.script->get_path();
						}

						if (const ExtendGDScriptParser *target_parser = get_parse_result(target_script_path)) {
							symbol = target_parser->get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(ret.location));

							if (symbol && symbol->kind == lsp::SymbolKind::Function && symbol->name != symbol_identifier) {
								symbol = get_parameter_symbol(symbol, symbol_identifier);
							}
						}

					} else {
						String member = ret.class_member;
						if (member.empty() && symbol_identifier != ret.class_name) {
							member = symbol_identifier;
						}
						symbol = get_native_symbol(ret.class_name, member);
					}
				} else {
					symbol = parser->get_member_symbol(symbol_identifier);

					if (!symbol) {
						symbol = get_local_symbol(parser, symbol_identifier);
					}
				}
			}
		}
	}

	return symbol;
}

void GDScriptWorkspace::resolve_related_symbols(const lsp::TextDocumentPositionParams &p_doc_pos, List<const lsp::DocumentSymbol *> &r_list) {
	String path = get_file_path(p_doc_pos.textDocument.uri);
	if (const ExtendGDScriptParser *parser = get_parse_result(path)) {
		String symbol_identifier;
		Vector2i offset;
		symbol_identifier = parser->get_identifier_under_position(p_doc_pos.position, offset);

		const StringName *class_ptr = native_members.next(nullptr);
		while (class_ptr) {
			const ClassMembers &members = native_members.get(*class_ptr);
			if (const lsp::DocumentSymbol *const *symbol = members.getptr(symbol_identifier)) {
				r_list.push_back(*symbol);
			}
			class_ptr = native_members.next(class_ptr);
		}

		for (Map<String, ExtendGDScriptParser *>::Element *E = scripts.front(); E; E = E->next()) {
			const ExtendGDScriptParser *script = E->get();
			const ClassMembers &members = script->get_members();
			if (const lsp::DocumentSymbol *const *symbol = members.getptr(symbol_identifier)) {
				r_list.push_back(*symbol);
			}

			const HashMap<String, ClassMembers> &inner_classes = script->get_inner_classes();
			const String *_class = inner_classes.next(nullptr);
			while (_class) {
				const ClassMembers *inner_class = inner_classes.getptr(*_class);
				if (const lsp::DocumentSymbol *const *symbol = inner_class->getptr(symbol_identifier)) {
					r_list.push_back(*symbol);
				}

				_class = inner_classes.next(_class);
			}
		}
	}
}

const lsp::DocumentSymbol *GDScriptWorkspace::resolve_native_symbol(const lsp::NativeSymbolInspectParams &p_params) {
	if (Map<StringName, lsp::DocumentSymbol>::Element *E = native_symbols.find(p_params.native_class)) {
		const lsp::DocumentSymbol &symbol = E->get();
		if (p_params.symbol_name.empty() || p_params.symbol_name == symbol.name) {
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

void GDScriptWorkspace::resolve_document_links(const String &p_uri, List<lsp::DocumentLink> &r_list) {
	if (const ExtendGDScriptParser *parser = get_parse_successed_script(get_file_path(p_uri))) {
		const List<lsp::DocumentLink> &links = parser->get_document_links();
		for (const List<lsp::DocumentLink>::Element *E = links.front(); E; E = E->next()) {
			r_list.push_back(E->get());
		}
	}
}

Dictionary GDScriptWorkspace::generate_script_api(const String &p_path) {
	Dictionary api;
	if (const ExtendGDScriptParser *parser = get_parse_successed_script(p_path)) {
		api = parser->generate_api();
	}
	return api;
}

Error GDScriptWorkspace::resolve_signature(const lsp::TextDocumentPositionParams &p_doc_pos, lsp::SignatureHelp &r_signature) {
	if (const ExtendGDScriptParser *parser = get_parse_result(get_file_path(p_doc_pos.textDocument.uri))) {
		lsp::TextDocumentPositionParams text_pos;
		text_pos.textDocument = p_doc_pos.textDocument;

		if (parser->get_left_function_call(p_doc_pos.position, text_pos.position, r_signature.activeParameter) == OK) {
			List<const lsp::DocumentSymbol *> symbols;

			if (const lsp::DocumentSymbol *symbol = resolve_symbol(text_pos)) {
				symbols.push_back(symbol);
			} else if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
				GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_related_symbols(text_pos, symbols);
			}

			for (List<const lsp::DocumentSymbol *>::Element *E = symbols.front(); E; E = E->next()) {
				const lsp::DocumentSymbol *symbol = E->get();
				if (symbol->kind == lsp::SymbolKind::Method || symbol->kind == lsp::SymbolKind::Function) {
					lsp::SignatureInformation signature_info;
					signature_info.label = symbol->detail;
					signature_info.documentation = symbol->render();

					for (int i = 0; i < symbol->children.size(); i++) {
						const lsp::DocumentSymbol &arg = symbol->children[i];
						lsp::ParameterInformation arg_info;
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

GDScriptWorkspace::GDScriptWorkspace() {
	ProjectSettings::get_singleton()->get_resource_path();
}

GDScriptWorkspace::~GDScriptWorkspace() {
	Set<String> cached_parsers;

	for (Map<String, ExtendGDScriptParser *>::Element *E = parse_results.front(); E; E = E->next()) {
		cached_parsers.insert(E->key());
	}

	for (Map<String, ExtendGDScriptParser *>::Element *E = scripts.front(); E; E = E->next()) {
		cached_parsers.insert(E->key());
	}

	for (Set<String>::Element *E = cached_parsers.front(); E; E = E->next()) {
		remove_cache_parser(E->get());
	}
}
