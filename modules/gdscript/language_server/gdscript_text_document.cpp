/*************************************************************************/
/*  gdscript_text_document.cpp                                           */
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

#include "gdscript_text_document.h"

#include "../gdscript.h"
#include "core/os/os.h"
#include "editor/editor_settings.h"
#include "editor/plugins/script_text_editor.h"
#include "gdscript_extend_parser.h"
#include "gdscript_language_protocol.h"
#include "servers/display_server.h"

void GDScriptTextDocument::_bind_methods() {
	ClassDB::bind_method(D_METHOD("didOpen"), &GDScriptTextDocument::didOpen);
	ClassDB::bind_method(D_METHOD("didClose"), &GDScriptTextDocument::didClose);
	ClassDB::bind_method(D_METHOD("didChange"), &GDScriptTextDocument::didChange);
	ClassDB::bind_method(D_METHOD("didSave"), &GDScriptTextDocument::didSave);
	ClassDB::bind_method(D_METHOD("nativeSymbol"), &GDScriptTextDocument::nativeSymbol);
	ClassDB::bind_method(D_METHOD("documentSymbol"), &GDScriptTextDocument::documentSymbol);
	ClassDB::bind_method(D_METHOD("completion"), &GDScriptTextDocument::completion);
	ClassDB::bind_method(D_METHOD("resolve"), &GDScriptTextDocument::resolve);
	ClassDB::bind_method(D_METHOD("rename"), &GDScriptTextDocument::rename);
	ClassDB::bind_method(D_METHOD("foldingRange"), &GDScriptTextDocument::foldingRange);
	ClassDB::bind_method(D_METHOD("codeLens"), &GDScriptTextDocument::codeLens);
	ClassDB::bind_method(D_METHOD("documentLink"), &GDScriptTextDocument::documentLink);
	ClassDB::bind_method(D_METHOD("colorPresentation"), &GDScriptTextDocument::colorPresentation);
	ClassDB::bind_method(D_METHOD("hover"), &GDScriptTextDocument::hover);
	ClassDB::bind_method(D_METHOD("definition"), &GDScriptTextDocument::definition);
	ClassDB::bind_method(D_METHOD("declaration"), &GDScriptTextDocument::declaration);
	ClassDB::bind_method(D_METHOD("signatureHelp"), &GDScriptTextDocument::signatureHelp);
	ClassDB::bind_method(D_METHOD("show_native_symbol_in_editor"), &GDScriptTextDocument::show_native_symbol_in_editor);
}

void GDScriptTextDocument::didOpen(const Variant &p_param) {
	lsp::TextDocumentItem doc = load_document_item(p_param);
	sync_script_content(doc.uri, doc.text);
}

void GDScriptTextDocument::didClose(const Variant &p_param) {
	// Left empty on purpose. Godot does nothing special on closing a document,
	// but it satisfies LSP clients that require didClose be implemented.
}

void GDScriptTextDocument::didChange(const Variant &p_param) {
	lsp::TextDocumentItem doc = load_document_item(p_param);
	Dictionary dict = p_param;
	Array contentChanges = dict["contentChanges"];
	for (int i = 0; i < contentChanges.size(); ++i) {
		lsp::TextDocumentContentChangeEvent evt;
		evt.load(contentChanges[i]);
		doc.text = evt.text;
	}
	sync_script_content(doc.uri, doc.text);
}

void GDScriptTextDocument::didSave(const Variant &p_param) {
	lsp::TextDocumentItem doc = load_document_item(p_param);
	Dictionary dict = p_param;
	String text = dict["text"];

	sync_script_content(doc.uri, text);

	/*String path = GDScriptLanguageProtocol::get_singleton()->get_workspace()->get_file_path(doc.uri);

	Ref<GDScript> script = ResourceLoader::load(path);
	script->load_source_code(path);
	script->reload(true);*/
}

lsp::TextDocumentItem GDScriptTextDocument::load_document_item(const Variant &p_param) {
	lsp::TextDocumentItem doc;
	Dictionary params = p_param;
	doc.load(params["textDocument"]);
	return doc;
}

void GDScriptTextDocument::notify_client_show_symbol(const lsp::DocumentSymbol *symbol) {
	ERR_FAIL_NULL(symbol);
	GDScriptLanguageProtocol::get_singleton()->notify_client("gdscript/show_native_symbol", symbol->to_json(true));
}

void GDScriptTextDocument::initialize() {
	if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
		const HashMap<StringName, ClassMembers> &native_members = GDScriptLanguageProtocol::get_singleton()->get_workspace()->native_members;

		const StringName *class_ptr = native_members.next(nullptr);
		while (class_ptr) {
			const ClassMembers &members = native_members.get(*class_ptr);

			const String *name = members.next(nullptr);
			while (name) {
				const lsp::DocumentSymbol *symbol = members.get(*name);
				lsp::CompletionItem item = symbol->make_completion_item();
				item.data = JOIN_SYMBOLS(String(*class_ptr), *name);
				native_member_completions.push_back(item.to_json());

				name = members.next(name);
			}

			class_ptr = native_members.next(class_ptr);
		}
	}
}

Variant GDScriptTextDocument::nativeSymbol(const Dictionary &p_params) {
	Variant ret;

	lsp::NativeSymbolInspectParams params;
	params.load(p_params);

	if (const lsp::DocumentSymbol *symbol = GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_native_symbol(params)) {
		ret = symbol->to_json(true);
		notify_client_show_symbol(symbol);
	}

	return ret;
}

Array GDScriptTextDocument::documentSymbol(const Dictionary &p_params) {
	Dictionary params = p_params["textDocument"];
	String uri = params["uri"];
	String path = GDScriptLanguageProtocol::get_singleton()->get_workspace()->get_file_path(uri);
	Array arr;
	if (const Map<String, ExtendGDScriptParser *>::Element *parser = GDScriptLanguageProtocol::get_singleton()->get_workspace()->scripts.find(path)) {
		Vector<lsp::DocumentedSymbolInformation> list;
		parser->get()->get_symbols().symbol_tree_as_list(uri, list);
		for (int i = 0; i < list.size(); i++) {
			arr.push_back(list[i].to_json());
		}
	}
	return arr;
}

Array GDScriptTextDocument::completion(const Dictionary &p_params) {
	Array arr;

	lsp::CompletionParams params;
	params.load(p_params);
	Dictionary request_data = params.to_json();

	List<ScriptCodeCompletionOption> options;
	GDScriptLanguageProtocol::get_singleton()->get_workspace()->completion(params, &options);

	if (!options.is_empty()) {
		int i = 0;
		arr.resize(options.size());

		for (const ScriptCodeCompletionOption &option : options) {
			lsp::CompletionItem item;
			item.label = option.display;
			item.data = request_data;

			switch (option.kind) {
				case ScriptCodeCompletionOption::KIND_ENUM:
					item.kind = lsp::CompletionItemKind::Enum;
					break;
				case ScriptCodeCompletionOption::KIND_CLASS:
					item.kind = lsp::CompletionItemKind::Class;
					break;
				case ScriptCodeCompletionOption::KIND_MEMBER:
					item.kind = lsp::CompletionItemKind::Property;
					break;
				case ScriptCodeCompletionOption::KIND_FUNCTION:
					item.kind = lsp::CompletionItemKind::Method;
					break;
				case ScriptCodeCompletionOption::KIND_SIGNAL:
					item.kind = lsp::CompletionItemKind::Event;
					break;
				case ScriptCodeCompletionOption::KIND_CONSTANT:
					item.kind = lsp::CompletionItemKind::Constant;
					break;
				case ScriptCodeCompletionOption::KIND_VARIABLE:
					item.kind = lsp::CompletionItemKind::Variable;
					break;
				case ScriptCodeCompletionOption::KIND_FILE_PATH:
					item.kind = lsp::CompletionItemKind::File;
					break;
				case ScriptCodeCompletionOption::KIND_NODE_PATH:
					item.kind = lsp::CompletionItemKind::Snippet;
					break;
				case ScriptCodeCompletionOption::KIND_PLAIN_TEXT:
					item.kind = lsp::CompletionItemKind::Text;
					break;
			}

			arr[i] = item.to_json();
			i++;
		}
	} else if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
		arr = native_member_completions.duplicate();

		for (KeyValue<String, ExtendGDScriptParser *> &E : GDScriptLanguageProtocol::get_singleton()->get_workspace()->scripts) {
			ExtendGDScriptParser *script = E.value;
			const Array &items = script->get_member_completions();

			const int start_size = arr.size();
			arr.resize(start_size + items.size());
			for (int i = start_size; i < arr.size(); i++) {
				arr[i] = items[i - start_size];
			}
		}
	}
	return arr;
}

Dictionary GDScriptTextDocument::rename(const Dictionary &p_params) {
	lsp::TextDocumentPositionParams params;
	params.load(p_params);
	String new_name = p_params["newName"];

	return GDScriptLanguageProtocol::get_singleton()->get_workspace()->rename(params, new_name);
}

Dictionary GDScriptTextDocument::resolve(const Dictionary &p_params) {
	lsp::CompletionItem item;
	item.load(p_params);

	lsp::CompletionParams params;
	Variant data = p_params["data"];

	const lsp::DocumentSymbol *symbol = nullptr;

	if (data.get_type() == Variant::DICTIONARY) {
		params.load(p_params["data"]);
		symbol = GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_symbol(params, item.label, item.kind == lsp::CompletionItemKind::Method || item.kind == lsp::CompletionItemKind::Function);

	} else if (data.get_type() == Variant::STRING) {
		String query = data;

		Vector<String> param_symbols = query.split(SYMBOL_SEPERATOR, false);

		if (param_symbols.size() >= 2) {
			String class_ = param_symbols[0];
			StringName class_name = class_;
			String member_name = param_symbols[param_symbols.size() - 1];
			String inner_class_name;
			if (param_symbols.size() >= 3) {
				inner_class_name = param_symbols[1];
			}

			if (const ClassMembers *members = GDScriptLanguageProtocol::get_singleton()->get_workspace()->native_members.getptr(class_name)) {
				if (const lsp::DocumentSymbol *const *member = members->getptr(member_name)) {
					symbol = *member;
				}
			}

			if (!symbol) {
				if (const Map<String, ExtendGDScriptParser *>::Element *E = GDScriptLanguageProtocol::get_singleton()->get_workspace()->scripts.find(class_name)) {
					symbol = E->get()->get_member_symbol(member_name, inner_class_name);
				}
			}
		}
	}

	if (symbol) {
		item.documentation = symbol->render();
	}

	if ((item.kind == lsp::CompletionItemKind::Method || item.kind == lsp::CompletionItemKind::Function) && !item.label.ends_with("):")) {
		item.insertText = item.label + "(";
		if (symbol && symbol->children.is_empty()) {
			item.insertText += ")";
		}
	} else if (item.kind == lsp::CompletionItemKind::Event) {
		if (params.context.triggerKind == lsp::CompletionTriggerKind::TriggerCharacter && (params.context.triggerCharacter == "(")) {
			const String quote_style = EDITOR_GET("text_editor/completion/use_single_quotes") ? "'" : "\"";
			item.insertText = item.label.quote(quote_style);
		}
	}

	return item.to_json(true);
}

Array GDScriptTextDocument::foldingRange(const Dictionary &p_params) {
	Array arr;
	return arr;
}

Array GDScriptTextDocument::codeLens(const Dictionary &p_params) {
	Array arr;
	return arr;
}

Array GDScriptTextDocument::documentLink(const Dictionary &p_params) {
	Array ret;

	lsp::DocumentLinkParams params;
	params.load(p_params);

	List<lsp::DocumentLink> links;
	GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_document_links(params.textDocument.uri, links);
	for (const lsp::DocumentLink &E : links) {
		ret.push_back(E.to_json());
	}
	return ret;
}

Array GDScriptTextDocument::colorPresentation(const Dictionary &p_params) {
	Array arr;
	return arr;
}

Variant GDScriptTextDocument::hover(const Dictionary &p_params) {
	lsp::TextDocumentPositionParams params;
	params.load(p_params);

	const lsp::DocumentSymbol *symbol = GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_symbol(params);
	if (symbol) {
		lsp::Hover hover;
		hover.contents = symbol->render();
		hover.range.start = params.position;
		hover.range.end = params.position;
		return hover.to_json();

	} else if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
		Dictionary ret;
		Array contents;
		List<const lsp::DocumentSymbol *> list;
		GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_related_symbols(params, list);
		for (const lsp::DocumentSymbol *&E : list) {
			if (const lsp::DocumentSymbol *s = E) {
				contents.push_back(s->render().value);
			}
		}
		ret["contents"] = contents;
		return ret;
	}

	return Variant();
}

Array GDScriptTextDocument::definition(const Dictionary &p_params) {
	lsp::TextDocumentPositionParams params;
	params.load(p_params);
	List<const lsp::DocumentSymbol *> symbols;
	Array arr = this->find_symbols(params, symbols);
	return arr;
}

Variant GDScriptTextDocument::declaration(const Dictionary &p_params) {
	lsp::TextDocumentPositionParams params;
	params.load(p_params);
	List<const lsp::DocumentSymbol *> symbols;
	Array arr = this->find_symbols(params, symbols);
	if (arr.is_empty() && !symbols.is_empty() && !symbols.front()->get()->native_class.is_empty()) { // Find a native symbol
		const lsp::DocumentSymbol *symbol = symbols.front()->get();
		if (GDScriptLanguageProtocol::get_singleton()->is_goto_native_symbols_enabled()) {
			String id;
			switch (symbol->kind) {
				case lsp::SymbolKind::Class:
					id = "class_name:" + symbol->name;
					break;
				case lsp::SymbolKind::Constant:
					id = "class_constant:" + symbol->native_class + ":" + symbol->name;
					break;
				case lsp::SymbolKind::Property:
				case lsp::SymbolKind::Variable:
					id = "class_property:" + symbol->native_class + ":" + symbol->name;
					break;
				case lsp::SymbolKind::Enum:
					id = "class_enum:" + symbol->native_class + ":" + symbol->name;
					break;
				case lsp::SymbolKind::Method:
				case lsp::SymbolKind::Function:
					id = "class_method:" + symbol->native_class + ":" + symbol->name;
					break;
				default:
					id = "class_global:" + symbol->native_class + ":" + symbol->name;
					break;
			}
			call_deferred(SNAME("show_native_symbol_in_editor"), id);
		} else {
			notify_client_show_symbol(symbol);
		}
	}
	return arr;
}

Variant GDScriptTextDocument::signatureHelp(const Dictionary &p_params) {
	Variant ret;

	lsp::TextDocumentPositionParams params;
	params.load(p_params);

	lsp::SignatureHelp s;
	if (OK == GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_signature(params, s)) {
		ret = s.to_json();
	}

	return ret;
}

GDScriptTextDocument::GDScriptTextDocument() {
	file_checker = FileAccess::create(FileAccess::ACCESS_RESOURCES);
}

GDScriptTextDocument::~GDScriptTextDocument() {
	memdelete(file_checker);
}

void GDScriptTextDocument::sync_script_content(const String &p_path, const String &p_content) {
	String path = GDScriptLanguageProtocol::get_singleton()->get_workspace()->get_file_path(p_path);
	GDScriptLanguageProtocol::get_singleton()->get_workspace()->parse_script(path, p_content);

	EditorFileSystem::get_singleton()->update_file(path);
	Error error;
	Ref<GDScript> script = ResourceLoader::load(path, "", ResourceFormatLoader::CACHE_MODE_REUSE, &error);
	if (error == OK) {
		if (script->load_source_code(path) == OK) {
			script->reload(true);
		}
	}
}

void GDScriptTextDocument::show_native_symbol_in_editor(const String &p_symbol_id) {
	ScriptEditor::get_singleton()->call_deferred(SNAME("_help_class_goto"), p_symbol_id);

	DisplayServer::get_singleton()->window_move_to_foreground();
}

Array GDScriptTextDocument::find_symbols(const lsp::TextDocumentPositionParams &p_location, List<const lsp::DocumentSymbol *> &r_list) {
	Array arr;
	const lsp::DocumentSymbol *symbol = GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_symbol(p_location);
	if (symbol) {
		lsp::Location location;
		location.uri = symbol->uri;
		location.range = symbol->range;
		const String &path = GDScriptLanguageProtocol::get_singleton()->get_workspace()->get_file_path(symbol->uri);
		if (file_checker->file_exists(path)) {
			arr.push_back(location.to_json());
		}
		r_list.push_back(symbol);
	} else if (GDScriptLanguageProtocol::get_singleton()->is_smart_resolve_enabled()) {
		List<const lsp::DocumentSymbol *> list;
		GDScriptLanguageProtocol::get_singleton()->get_workspace()->resolve_related_symbols(p_location, list);
		for (const lsp::DocumentSymbol *&E : list) {
			if (const lsp::DocumentSymbol *s = E) {
				if (!s->uri.is_empty()) {
					lsp::Location location;
					location.uri = s->uri;
					location.range = s->range;
					arr.push_back(location.to_json());
					r_list.push_back(s);
				}
			}
		}
	}
	return arr;
}
