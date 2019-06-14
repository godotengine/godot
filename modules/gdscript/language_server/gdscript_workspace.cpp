/*************************************************************************/
/*  gdscript_workspace.cpp                                               */
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

#include "gdscript_workspace.h"
#include "../gdscript.h"
#include "../gdscript_parser.h"
#include "core/project_settings.h"
#include "gdscript_language_protocol.h"

void GDScriptWorkspace::_bind_methods() {
	ClassDB::bind_method(D_METHOD("symbol"), &GDScriptWorkspace::symbol);
}

void GDScriptWorkspace::remove_cache_parser(const String &p_path) {
	Map<String, ExtendGDScriptParser *>::Element *parser = parse_results.find(p_path);
	Map<String, ExtendGDScriptParser *>::Element *script = scripts.find(p_path);
	if (parser && script) {
		if (script->get() && script->get() == script->get()) {
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

Array GDScriptWorkspace::symbol(const Dictionary &p_params) {
	String query = p_params["query"];
	Array arr;
	if (!query.empty()) {
		for (Map<String, ExtendGDScriptParser *>::Element *E = scripts.front(); E; E = E->next()) {
			Vector<lsp::SymbolInformation> script_symbols;
			E->get()->get_symbols().symbol_tree_as_list(E->key(), script_symbols);
			for (int i = 0; i < script_symbols.size(); ++i) {
				if (query.is_subsequence_ofi(script_symbols[i].name)) {
					arr.push_back(script_symbols[i].to_json());
				}
			}
		}
	}
	return arr;
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

String GDScriptWorkspace::get_file_path(const String &p_uri) const {
	String path = p_uri.replace("file://", "").http_unescape();
	path = path.replace(root + "/", "res://");
	return ProjectSettings::get_singleton()->localize_path(path);
}

String GDScriptWorkspace::get_file_uri(const String &p_path) const {
	String path = ProjectSettings::get_singleton()->globalize_path(p_path);
	return "file://" + path;
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

void GDScriptWorkspace::completion(const lsp::CompletionParams &p_params, List<ScriptCodeCompletionOption> *r_options) {
	String path = get_file_path(p_params.textDocument.uri);
	String call_hint;
	bool forced = false;
	if (Map<String, ExtendGDScriptParser *>::Element *E = parse_results.find(path)) {
		String code = E->get()->get_text_for_completion(p_params.position);
		GDScriptLanguage::get_singleton()->complete_code(code, path, NULL, r_options, forced, call_hint);
	}
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
