/**************************************************************************/
/*  compilation_unit.cpp                                                  */
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

#include "compilation_unit.h"

#include "../gdscript_analyzer.h"
#include "../gdscript_cache.h"

#include "core/io/file_access.h"
#include "core/io/resource_loader.h"

GDScriptParser *GDScriptParserRef::get_parser() {
	if (parser == nullptr) {
		parser = memnew(GDScriptParser);
	}
	return parser;
}

GDScriptAnalyzer *GDScriptParserRef::get_analyzer() {
	if (analyzer == nullptr) {
		analyzer = memnew(GDScriptAnalyzer(unit, get_parser()));
	}
	return analyzer;
}

Error GDScriptParserRef::raise_status(Status p_new_status) {
	ERR_FAIL_COND_V(parser == nullptr && status != EMPTY, ERR_BUG);

	if (p_new_status < status) {
		return OK;
	}

	while (result == OK && p_new_status > status) {
		switch (status) {
			case EMPTY: {
				status = PARSED;
				const String remapped_path = ResourceLoader::path_remap(path);
				if (remapped_path.has_extension("gdc")) {
					Vector<uint8_t> tokens = GDScriptCache::get_binary_tokens(remapped_path);
					result = get_parser()->parse_binary(tokens, path);
				} else {
					String source = GDScriptCache::get_source_code(remapped_path);
					result = get_parser()->parse(source, path, false);
				}
			} break;
			case PARSED: {
				status = INHERITANCE_SOLVED;
				result = get_analyzer()->resolve_inheritance();
			} break;
			case INHERITANCE_SOLVED: {
				status = INTERFACE_SOLVED;
				result = get_analyzer()->resolve_interface();
			} break;
			case INTERFACE_SOLVED: {
				status = FULLY_SOLVED;
				result = get_analyzer()->resolve_body();
			} break;
			case FULLY_SOLVED: {
				return result;
			}
		}
	}

	return result;
}

GDScriptParserRef::~GDScriptParserRef() {
	memdelete(parser);
	memdelete(analyzer);
}

GDScriptParserRef *GDScriptCompilationUnit::get_depended_parser_for(const String &p_path, const String &p_owner) {
	GDScriptParserRef *ref = nullptr;
	if (parsers.has(p_path)) {
		ref = parsers[p_path];
	} else {
		const String remapped_path = ResourceLoader::path_remap(p_path);
		if (!FileAccess::exists(remapped_path)) {
			return nullptr;
		}
		ref = memnew(GDScriptParserRef(*this));
		ref->path = GDScript::canonicalize_path(p_path); // TODO: Normalize further.
		ref->raise_status(GDScriptParserRef::EMPTY);

		if (ref != nullptr) {
			parsers[p_path] = ref;
			if (!GDScript::is_canonically_equal_paths(p_owner, p_path)) {
				dependencies[p_owner].insert(p_path);
				GDScriptCache::register_dependency(p_path, p_owner);
			}
		}
	}

	return ref;
}

GDScriptParserRef *GDScriptCompilationUnit::find_parser_ref_for_class(const GDScriptParser::ClassNode *p_class) {
	for (KeyValue<String, GDScriptParserRef *> E : parsers) {
		if (E.value->get_parser()->has_class(p_class)) {
			return E.value;
		}
	}
	return nullptr;
}

const HashMap<String, GDScriptParserRef *> GDScriptCompilationUnit::get_depended_parsers(const String &p_owner) {
	HashMap<String, GDScriptParserRef *> res;
	for (const String &dep : dependencies[p_owner]) {
		res[dep] = parsers[dep];
	}
	return res;
}

GDScriptParserRef *GDScriptCompilationUnit::get_parser(const String &p_path) {
	GDScriptParserRef **ptr = parsers.getptr(p_path);
	if (ptr != nullptr) {
		return *ptr;
	}
	return nullptr;
}

GDScriptCompilationUnit::~GDScriptCompilationUnit() {
	for (const KeyValue<String, GDScriptParserRef *> &E : parsers) {
		memdelete(E.value);
	}
}
