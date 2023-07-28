/**************************************************************************/
/*  test_lsp.h                                                            */
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

#ifndef TEST_LSP_H
#define TEST_LSP_H

#include "tests/test_macros.h"

#include "../language_server/gdscript_extend_parser.h"
#include "../language_server/gdscript_language_protocol.h"
#include "../language_server/gdscript_workspace.h"
#include "../language_server/godot_lsp.h"

#include "core/io/dir_access.h"
#include "core/io/file_access_pack.h"
#include "core/os/os.h"
#include "editor/editor_help.h"
#include "editor/editor_node.h"

#include "thirdparty/doctest/doctest.h"

template <>
struct doctest::StringMaker<lsp::Position> {
	static doctest::String convert(const lsp::Position &p_val) {
		return vformat("(%d,%d)", p_val.line, p_val.character).utf8().get_data();
	}
};
template <>
struct doctest::StringMaker<lsp::Range> {
	static doctest::String convert(const lsp::Range &p_val) {
		return vformat("[(%d,%d):(%d,%d)]", p_val.start.line, p_val.start.character, p_val.end.line, p_val.end.character).utf8().get_data();
	}
};

namespace GDScriptTests {

const String root = "modules/gdscript/tests/lsp_project/";

// `memdelete` returned `GDScriptLanguageProtocol` after use!
GDScriptLanguageProtocol *initialize(const String &root) {
	Error err = OK;
	Ref<DirAccess> dir(DirAccess::open(root, &err));
	REQUIRE_MESSAGE(err == OK, "Could not open specified root directory");
	String absolute_root = dir->get_current_dir();
	init_language(absolute_root);

	auto proto = memnew(GDScriptLanguageProtocol);

	auto workspace = GDScriptLanguageProtocol::get_singleton()->get_workspace();
	//TODO: adjust? escape `:` but not `/`
	workspace->root = absolute_root;
	workspace->root_uri = "file:///" + absolute_root.lstrip("/");

	return proto;
}

lsp::Position pos(const int line, const int character) {
	lsp::Position p;
	p.line = line;
	p.character = character;
	return p;
}
lsp::Range range(const lsp::Position start, const lsp::Position end) {
	lsp::Range r;
	r.start = start;
	r.end = end;
	return r;
}

lsp::TextDocumentPositionParams posIn(const lsp::DocumentUri &uri, const lsp::Position pos) {
	lsp::TextDocumentPositionParams params;
	params.textDocument.uri = uri;
	params.position = pos;
	return params;
}

// Notes for code examples:
// * Tab: →
// * cursor is BETWEEN chars.
//     Markers:
//     * `|`: inline, marks cursor position: `va|r` -> cursor between `a`&`r`
//     * `^`: below, marks character position -> cursor before:
//            `var`
//               ^
//            -> character on `r` -> cursor between `a`&`r`
//     * Note: both marker examples: `character = 2`
// * Line & Char: both 0-based (LSP)
//   * Godot: both 1-based
TEST_SUITE("[Modules][GDScript][LSP]") {
	TEST_CASE("[workspace][resolve_symbol]") {
		auto proto = initialize(root);
		REQUIRE(proto);
		auto workspace = GDScriptLanguageProtocol::get_singleton()->get_workspace();

		SUBCASE("[local][parameter]") {
			String uri = workspace->get_file_uri("res://local_variables.gd");
			// `arg` in `func return_arg(arg: int):`
			String expected_name = "arg";
			lsp::Range expected_range = range(pos(12, 16), pos(12, 19));

			SUBCASE("can resolve parameter at declaration") {
				// `func return_arg(a|rg: int) -> int:`
				lsp::Position p = pos(12, 17);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				// with `: int`:
				// auto expected_range = range(pos(12, 16), pos(12, 24));
				CHECK_EQ(symbol->range, expected_range);
			}
			SUBCASE("can resolve parameter at usage") {
				// `→a|rg += 2`
				lsp::Position p = pos(13, 2);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
		}

		SUBCASE("[public][variable]") {
			String uri = workspace->get_file_uri("res://local_variables.gd");
			String expected_name = "member";
			lsp::Range expected_range = range(pos(3, 4), pos(3, 10));

			SUBCASE("can resolve variable at declaration") {
				// `var m|ember := 2`
				lsp::Position p = pos(3, 5);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
			SUBCASE("can resolve variable at usage (get)") {
				// `→var test := memb|er + 42`
				lsp::Position p = pos(6, 17);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
			SUBCASE("can resolve variable at usage (set)") {
				// `→me|mber += 5`
				lsp::Position p = pos(8, 3);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
		}

		SUBCASE("[local][variable]") {
			String uri = workspace->get_file_uri("res://local_variables.gd");
			String expected_name = "test";
			lsp::Range expected_range = range(pos(6, 5), pos(6, 9));

			SUBCASE("can resolve variable at declaration") {
				// `→var t|est := ...`
				lsp::Position p = pos(6, 6);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
			SUBCASE("can resolve variable at usage (set)") {
				// `→tes|t = return_arg(test)`
				lsp::Position p = pos(9, 4);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
			SUBCASE("can resolve variable at usage (get)") {
				// `→test = return_arg(t|est)`
				lsp::Position p = pos(9, 20);
				lsp::TextDocumentPositionParams params = posIn(uri, p);
				const lsp::DocumentSymbol *symbol = workspace->resolve_symbol(params);
				REQUIRE(symbol);

				CHECK_EQ(symbol->uri, uri);
				CHECK_EQ(symbol->name, expected_name);
				CHECK_EQ(symbol->range, expected_range);
			}
		}

		memdelete(proto);
	}
}

} // namespace GDScriptTests

#endif // TEST_LSP_H
