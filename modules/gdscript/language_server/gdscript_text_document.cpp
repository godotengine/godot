/*************************************************************************/
/*  gdscript_text_document.cpp                                           */
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

#include "gdscript_text_document.h"
#include "../gdscript.h"
#include "gdscript_language_protocol.h"

void GDScriptTextDocument::_bind_methods() {
	ClassDB::bind_method(D_METHOD("didOpen"), &GDScriptTextDocument::didOpen);
	ClassDB::bind_method(D_METHOD("didChange"), &GDScriptTextDocument::didChange);
	ClassDB::bind_method(D_METHOD("documentSymbol"), &GDScriptTextDocument::documentSymbol);
	ClassDB::bind_method(D_METHOD("completion"), &GDScriptTextDocument::completion);
	ClassDB::bind_method(D_METHOD("foldingRange"), &GDScriptTextDocument::foldingRange);
	ClassDB::bind_method(D_METHOD("codeLens"), &GDScriptTextDocument::codeLens);
	ClassDB::bind_method(D_METHOD("documentLink"), &GDScriptTextDocument::documentLink);
	ClassDB::bind_method(D_METHOD("colorPresentation"), &GDScriptTextDocument::colorPresentation);
	ClassDB::bind_method(D_METHOD("hover"), &GDScriptTextDocument::hover);
}

void GDScriptTextDocument::didOpen(const Variant &p_param) {
	lsp::TextDocumentItem doc = load_document_item(p_param);
	sync_script_content(doc.uri, doc.text);
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

lsp::TextDocumentItem GDScriptTextDocument::load_document_item(const Variant &p_param) {
	lsp::TextDocumentItem doc;
	Dictionary params = p_param;
	doc.load(params["textDocument"]);
	print_line(doc.text);
	return doc;
}

Array GDScriptTextDocument::documentSymbol(const Dictionary &p_params) {
	Dictionary params = p_params["textDocument"];
	String uri = params["uri"];
	String path = GDScriptLanguageProtocol::get_singleton()->get_workspace().get_file_path(uri);
	Array arr;
	if (const Map<String, ExtendGDScriptParser *>::Element *parser = GDScriptLanguageProtocol::get_singleton()->get_workspace().scripts.find(path)) {
		Vector<lsp::SymbolInformation> list;
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
	GDScriptLanguageProtocol::get_singleton()->get_workspace().completion(params, &options);

	for (const List<ScriptCodeCompletionOption>::Element *E = options.front(); E; E = E->next()) {
		const ScriptCodeCompletionOption &option = E->get();
		lsp::CompletionItem item;
		item.label = option.display;
		item.insertText = option.insert_text;
		item.data = request_data;

		if (params.context.triggerKind == lsp::CompletionTriggerKind::TriggerCharacter && (params.context.triggerCharacter == "'" || params.context.triggerCharacter == "\"") && (option.insert_text.begins_with("'") || option.insert_text.begins_with("\""))) {
			item.insertText = option.insert_text.substr(1, option.insert_text.length() - 2);
		}

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

		arr.push_back(item.to_json());
	}

	return arr;
}

Array GDScriptTextDocument::foldingRange(const Dictionary &p_params) {
	Dictionary params = p_params["textDocument"];
	String path = params["uri"];
	Array arr;
	return arr;
}

Array GDScriptTextDocument::codeLens(const Dictionary &p_params) {
	Array arr;
	return arr;
}

Variant GDScriptTextDocument::documentLink(const Dictionary &p_params) {
	Variant ret;
	return ret;
}

Array GDScriptTextDocument::colorPresentation(const Dictionary &p_params) {
	Array arr;
	return arr;
}

Variant GDScriptTextDocument::hover(const Dictionary &p_params) {
	Variant ret;
	return ret;
}

void GDScriptTextDocument::sync_script_content(const String &p_uri, const String &p_content) {
	String path = GDScriptLanguageProtocol::get_singleton()->get_workspace().get_file_path(p_uri);
	GDScriptLanguageProtocol::get_singleton()->get_workspace().parse_script(path, p_content);
}
