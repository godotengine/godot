/**************************************************************************/
/*  compilation_unit.h                                                    */
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

#pragma once

#include "../gdscript_cache.h"
#include "../gdscript_parser.h"

#include "core/string/ustring.h"

// TODO: Better name.
class GDScriptParserRef final {
	GDScriptCompilationUnit &unit;

public:
	enum Status {
		EMPTY,
		PARSED,
		INHERITANCE_SOLVED,
		INTERFACE_SOLVED,
		FULLY_SOLVED,
	};

private:
	GDScriptParser *parser = nullptr;
	GDScriptAnalyzer *analyzer = nullptr;
	Status status = EMPTY;
	Error result = OK;
	String path;

	friend class GDScript;
	friend class GDScriptCompilationUnit;

public:
	Status get_status() const { return status; }
	String get_path() const { return path; }
	GDScriptParser *get_parser();
	GDScriptAnalyzer *get_analyzer();
	Error raise_status(Status p_new_status);

	GDScriptParserRef(GDScriptCompilationUnit &p_unit) : unit(p_unit) {}
	~GDScriptParserRef();
};

/**
 * Owner of resources related to parsing, analyzing and compiling of scripts.
 *
 * Not thread safe.
 */
class GDScriptCompilationUnit final {
	HashMap<String, GDScriptParserRef *> parsers;
	HashMap<String, HashSet<String>> dependencies;

public:
	GDScriptParserRef *get_depended_parser_for(const String &p_path, const String &p_owner);
	GDScriptParserRef *find_parser_ref_for_class(const GDScriptParser::ClassNode *p_class);
	const HashMap<String, GDScriptParserRef *> get_depended_parsers(const String &p_owner);

	/// Only meant as workaround for a completion usecase, use `get_depended_parser_for` instead.
	GDScriptParserRef *get_parser(const String &p_path);

	~GDScriptCompilationUnit();
};
