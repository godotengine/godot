/**************************************************************************/
/*  test_gdscript_formatter.h                                             */
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

#ifndef TEST_GDSCRIPT_FORMATTER_H
#define TEST_GDSCRIPT_FORMATTER_H

#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "modules/gdscript/gdscript_format.h"
#include "modules/gdscript/gdscript_parser.h"
#include "tests/test_macros.h"

namespace GDScriptTests {

#define CHECK_FORMAT(code, pre_formatted)                                                    \
	do {                                                                                     \
		GDScriptFormat formatter;                                                            \
		formatter.indent_in_multiline_block = 1;                                             \
		String output;                                                                       \
		Error err = formatter.format(code, output);                                          \
		Vector<String> error_messages;                                                       \
		if (err != OK) {                                                                     \
			for (GDScriptParser::ParserError parser_error : formatter.get_parser_errors()) { \
				error_messages.push_back(                                                    \
						vformat(                                                             \
								"Parse Error: %s (%s:%s)",                                   \
								parser_error.message,                                        \
								parser_error.line,                                           \
								parser_error.column));                                       \
			}                                                                                \
		}                                                                                    \
		CHECK_MESSAGE(                                                                       \
				err == OK,                                                                   \
				vformat(                                                                     \
						"The formatter returned errors (%s): \n%s",                          \
						error_messages.size(),                                               \
						String("\n").join(error_messages)));                                 \
		CHECK_EQ(                                                                            \
				vformat(CHECK_FORMAT_FORMAT, output),                                        \
				vformat(CHECK_FORMAT_FORMAT, pre_formatted));                                \
	} while (false);
#define CHECK_FORMAT_FORMAT "\n---\n%s\n---\n"

// The GDSCRIPT macro does two things:
// 1. Removes the first \n in the string (all GDSCRIPT macro calls must begin with a \n)
// 2. <*noop*> is there to fix issues with {root}/misc/scripts/header_guards.sh that refuses
//      consecutive two line breaks. <*noop*> is replaced by nothing, which leaves the \n
//      right after intact.
#define GDSCRIPT(code) \
	String(code)       \
			.substr(1) \
			.replacen("<*noop*>", "")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][ClassMembers]") {
	TEST_CASE("Should output a variable with a property that has a setter and getter, inline") {
		const String code =
				GDSCRIPT(R"(
var my_property := 0:
	get:
		return my_property
	set(value):
		my_property = value
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_property := 0:
	set(value):
		my_property = value
	get:
		return my_property
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a property that has a setter with a spare line after the property for readability") {
		const String code =
				GDSCRIPT(R"(
var my_property := 0:
	set(value):
		my_property = value
var some_variable = 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_property := 0:
	set(value):
		my_property = value

var some_variable = 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that takes a casted value") {
		const String code =
				GDSCRIPT(R"(
var my_casted_variable := my_uncasted_variable as CastedType
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that takes a casted value's output") {
		const String code =
				GDSCRIPT(R"(
var my_casted_variable := (my_uncasted_variable as CastedType).result
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_casted_variable := (my_uncasted_variable as CastedType).result
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class with a variable") {
		const String code =
				GDSCRIPT(R"(
var my_variable
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a datatype but no value") {
		const String code =
				GDSCRIPT(R"(
var my_variable: MyDataType
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class with a variable with self") {
		const String code =
				GDSCRIPT(R"(
var my_variable = self
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that has a basic export annotation") {
		const String code =
				GDSCRIPT(R"(
@export var my_variable
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that has multiple annotations on one line when they fit") {
		SUBCASE("They fit") {
			const String code =
					GDSCRIPT(R"(
@onready @export var my_variable
)");
			const String pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("They don't fit") {
			const String code =
					GDSCRIPT(R"(
@onready @export var my_variable_with_a_very_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_name
)");
			const String pre_formatted =
					GDSCRIPT(R"(
@onready
@export
var my_variable_with_a_very_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_name
)");

			CHECK_FORMAT(code, pre_formatted);
		}
	}

	TEST_CASE("Should output a variable that has a export annotation with parameters") {
		const String code =
				GDSCRIPT(R"(
@export_range(0, 20) var my_variable
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that has a export annotation with parameters that causes an annotation split") {
		const String code =
				GDSCRIPT(R"(
@export_enum("One thing leads", "to another, which causes", "a split to occur", "even if this is not", "a valid enum") var my_variable
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@export_enum("One thing leads", "to another, which causes", "a split to occur", "even if this is not", "a valid enum")
var my_variable
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with an array being accessed by index") {
		const String code =
				GDSCRIPT(R"(
var my_variable := presences[0]
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a dictionary being accessed by index") {
		const String code =
				GDSCRIPT(R"(
var my_variable := presences["MatchId"]
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a dictionary being accessed by attribute") {
		const String code =
				GDSCRIPT(R"(
var my_variable := output.format
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a dictionary being accessed by function that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := output[get_formatting_index_based_on_data("localhost", 8080, "development_branch")]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := output[
	get_formatting_index_based_on_data("localhost", 8080, "development_branch")
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that uses a simple if/else ternary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 5 if true else 8
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that uses a if/else ternary that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "a long string goes here to force a wrap" if true else "lorem ipsum 3.145967 robot meme"
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := (
	"a long string goes here to force a wrap" if true
	else "lorem ipsum 3.145967 robot meme"
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that uses a if/else ternary that wraps due to length with a function that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "a long string goes here to force a wrap" if true else some_function("lorem ipsum", 3.145967, "robot memory of some length", "formidable length of string")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := (
	"a long string goes here to force a wrap" if true
	else some_function(
		"lorem ipsum", 3.145967, "robot memory of some length", "formidable length of string"
	)
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that holds a negated integer") {
		const String code =
				GDSCRIPT(R"(
var my_variable := -2
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that holds an inverted truth") {
		const String code =
				GDSCRIPT(R"(
var my_variable := not my_condition
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a get node path") {
		const String code =
				GDSCRIPT(R"(
var my_variable := $Node
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a get node chain path") {
		const String code =
				GDSCRIPT(R"(
var my_variable := $NodeA/NodeB/NodeC
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a get node string") {
		const String code =
				GDSCRIPT(R"(
var my_variable := $"../Parent/NodeB"
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a preload") {
		const String code =
				GDSCRIPT(R"(
var my_variable := preload("res://Player.tscn")
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a preload that wraps") {
		const String code =
				GDSCRIPT(R"(
var my_variable := preload("res://A/Deep/Folder/Hierarchy/To/Encourage/Wrapping/PlayerSceneWithALongName.tscn")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := preload(
	"res://A/Deep/Folder/Hierarchy/To/Encourage/Wrapping/PlayerSceneWithALongName.tscn"
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a basic operation") {
		SUBCASE("+") {
			String code =
					GDSCRIPT(R"(
var my_variable := 0+1
)");
			String pre_formatted =
					GDSCRIPT(R"(
var my_variable := 0 + 1
)");

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("-") {
			String code =
					GDSCRIPT(R"(
var my_variable := 0-1
)");
			String pre_formatted =
					GDSCRIPT(R"(
var my_variable := 0 - 1
)");

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("*") {
			String code =
					GDSCRIPT(R"(
var my_variable := 0*1
)");
			String pre_formatted =
					GDSCRIPT(R"(
var my_variable := 0 * 1
)");

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("/") {
			String code =
					GDSCRIPT(R"(
var my_variable := 0/1
)");
			String pre_formatted =
					GDSCRIPT(R"(
var my_variable := 0 / 1
)");

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("%") {
			String code =
					GDSCRIPT(R"(
var my_variable := 0%1
)");
			String pre_formatted =
					GDSCRIPT(R"(
var my_variable := 0 % 1
)");
			CHECK_FORMAT(code, pre_formatted);
		}
	}

	TEST_CASE("Should output nested binary operation") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 0+0+1
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := 0 + 0 + 1
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output double nested binary operation") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 0+1+0+1
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := 0 + 1 + 0 + 1
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a binary operation") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 0+1
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := 0 + 1
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with string concat") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "Hello"+"World !"
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := "Hello" + "World !"
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with string concat that is wrapped due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas et neque sodales, tempor ex sit amet, venenatis elit." + "Etiam ultrices enim id venenatis tempor. Quisque dictum ligula vel felis vestibulum, eget eleifend sem suscipit."
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := (
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas et neque sodales, tempor ex sit amet, venenatis elit."
	+ "Etiam ultrices enim id venenatis tempor. Quisque dictum ligula vel felis vestibulum, eget eleifend sem suscipit."
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with double string concat that is wrapped due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas et neque sodales, tempor ex sit amet, venenatis elit." + "Etiam ultrices enim id venenatis tempor. Quisque dictum ligula vel felis vestibulum, eget eleifend sem suscipit." + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas et neque sodales, tempor ex sit amet, venenatis elit." + "Etiam ultrices enim id venenatis tempor. Quisque dictum ligula vel felis vestibulum, eget eleifend sem suscipit."
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := (
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas et neque sodales, tempor ex sit amet, venenatis elit."
	+ "Etiam ultrices enim id venenatis tempor. Quisque dictum ligula vel felis vestibulum, eget eleifend sem suscipit."
	+ "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas et neque sodales, tempor ex sit amet, venenatis elit."
	+ "Etiam ultrices enim id venenatis tempor. Quisque dictum ligula vel felis vestibulum, eget eleifend sem suscipit."
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that is broken due to length with double string concat that is not wrapped due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum "
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := (
	"Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum" + "Lorem ipsum "
)
)");
		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a properly formatted binary division operation with order of operations preserved") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 4 / (1 + 1)
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a preserved formatted operation") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 4 / 1 + 1
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a formatted operation for complex operation") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 3+(6*(11+1-4))/8*2
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := 3 + 6 * (11 + 1 - 4) / 8 * 2
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a formatted operation for multiplication") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 4*(1+1)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := 4 * (1 + 1)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should be able to store a variable with array") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [0,1,2,3,4,5,6,7,8,9]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should be able to store a variable with array that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := ["Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit."]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should be able to store a variable with array with subarray that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [["Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit.","Lorem ipsum dolor sit amet, consectetur adipiscing elit."]]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	[
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
	],
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should be able to store a variable with array that breaks, of arrays that do not break") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should maintain type information for typed arrays") {
		const String code =
				GDSCRIPT(R"(
extends Node

@onready var children: Array[Node] = get_children()
)");
		const String pre_formatted =
				GDSCRIPT(R"(
extends Node
<*noop*>
<*noop*>
@onready var children: Array[Node] = get_children()
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple one line dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {"string key":"string value"}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {"string key": "string value"}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple one line Lua dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {string_key="string value"}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {string_key = "string value"}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a dictionary wrapped due to length") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {"string key 1":"string value", "string key 2":"string value", "string key 3":"string value", "string key 4":"string value", "string key 5":"string value"}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {
	"string key 1": "string value",
	"string key 2": "string value",
	"string key 3": "string value",
	"string key 4": "string value",
	"string key 5": "string value",
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a dictionary that wraps an element due to length") {
		const String code =
				GDSCRIPT(R"(
var test := {"test":"a long concat expression"+some_function_call("with a lot of", "very long parameters", "that should be wrapped", "due to its severely extended length")}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var test := {
	"test": (
		"a long concat expression"
		+ some_function_call(
			"with a lot of",
			"very long parameters",
			"that should be wrapped",
			"due to its severely extended length"
		)
	),
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with an assigned value") {
		const String code =
				GDSCRIPT(R"(
var my_variable = 0
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with inferred datatype") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 0
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with an explicit datatype") {
		const String code =
				GDSCRIPT(R"(
var my_variable: int = 0
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a string literal") {
		const String code =
				GDSCRIPT(R"(
var my_variable := "Hello, my friends!"
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a string literal that contains a quotation mark") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 'Hello, my "friends"!'
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a decimal") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 0.0
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a decimal value") {
		const String code =
				GDSCRIPT(R"(
var my_variable := 0.25
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should be able to refer to another variable by identifier") {
		const String code =
				GDSCRIPT(R"(
var other_variable_name
var my_variable := other_variable_name
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should do binary operation inside a simple function call") {
		const String code =
				GDSCRIPT(R"(
var my_variable := a_math_function(20+5)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := a_math_function(20 + 5)
)");
		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with multiple arguments") {
		const String code =
				GDSCRIPT(R"(
var my_variable:=Vector2(300,47)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := Vector2(300, 47)
)");
		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that does a function call") {
		const String code = GDSCRIPT(R"(
var cell_position := world_to_map_split(300, 47)
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should be able to store a variable with a function call that breaks, of parameters that do not break") {
		const String code =
				GDSCRIPT(R"(
var wrapped_text := wrap_some_text("Lorem ipsum","Lorem ipsum","Lorem ipsum","Lorem ipsum","Lorem ips")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var wrapped_text := wrap_some_text(
	"Lorem ipsum", "Lorem ipsum", "Lorem ipsum", "Lorem ipsum", "Lorem ips"
)
)");
		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("First function after a non-function should be separated by newlines") {
		const String code =
				GDSCRIPT(R"(
var my_variable = 0
func _ready():
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable = 0
<*noop*>
<*noop*>
func _ready():
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class constant") {
		const String code =
				GDSCRIPT(R"(
const MY_CONST = 50
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should keep built-in constants as named constants") {
		const String code =
				GDSCRIPT(R"(
const TAU_COPY = TAU
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Sequential constants should follow one another, then line break for the next element type") {
		const String code =
				GDSCRIPT(R"(
const MY_CONST_A := 5
const MY_CONST_B = 0
var my_variable := 10
)");
		const String pre_formatted =
				GDSCRIPT(R"(
const MY_CONST_A := 5
const MY_CONST_B = 0

var my_variable := 10
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output user-inputted incidental new lines in classes") {
		const String code = GDSCRIPT(R"(
const CONSTANT_A := 0
const CONSTANT_B := 1

const CONSTANT_GROUP_A := 0
const CONSTANT_GROUP_B := 1
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a get_node statement with a nodepath string") {
		const String code =
				GDSCRIPT(R"(
@onready var node = get_node(^"Node")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready var node = get_node(^"Node")
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a get_node statement with a stringname string") {
		const String code =
				GDSCRIPT(R"(
@onready var node = get_node(&"Node")
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a signal declaration") {
		const String code =
				GDSCRIPT(R"(
signal signal_happened
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a signal declaration with parameters") {
		const String code =
				GDSCRIPT(R"(
signal signal_happened(a, b,c)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
signal signal_happened(a, b, c)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a signal declaration with parameters that wrap due to length") {
		const String code =
				GDSCRIPT(R"(
signal signal_happened(a_long_list_of_long_parameters_a, a_long_list_of_long_parameters_b,a_long_list_of_long_parameters_c,a_long_list_of_long_parameters_d)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
signal signal_happened(
	a_long_list_of_long_parameters_a,
	a_long_list_of_long_parameters_b,
	a_long_list_of_long_parameters_c,
	a_long_list_of_long_parameters_d
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a signal declaration with explicit empty parameter list being maintained") {
		const String code =
				GDSCRIPT(R"(
signal my_signal()
)");
		const String pre_formatted =
				GDSCRIPT(R"(
signal my_signal()
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple named enum") {
		const String code =
				GDSCRIPT(R"(
enum MyEnum { A, B, C }
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an enum with a value") {
		const String code =
				GDSCRIPT(R"(
enum MyEnum { A, B, C = 5, D }
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a named enum that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
enum MyNamedEnumWithALongName { ENUM_A_WITH_A_NAME, ENUM_A_WITH_B_NAME, ENUM_A_WITH_C_NAME, ENUM_A_WITH_D_NAME, ENUM_A_WITH_E_NAME, ENUM_A_WITH_F_NAME }
)");
		const String pre_formatted =
				GDSCRIPT(R"(
enum MyNamedEnumWithALongName {
	ENUM_A_WITH_A_NAME,
	ENUM_A_WITH_B_NAME,
	ENUM_A_WITH_C_NAME,
	ENUM_A_WITH_D_NAME,
	ENUM_A_WITH_E_NAME,
	ENUM_A_WITH_F_NAME,
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a named enum that wraps due to length but elements do not wrap") {
		const String code =
				GDSCRIPT(R"(
enum MyNamedEnumWithALongName {ENUM_A_WITH_A_NAME, ENUM_A_WITH_B_NAME,	ENUM_A_WITH_C_NAME, ENUM_A_WITH_D_NAME}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
enum MyNamedEnumWithALongName {
	ENUM_A_WITH_A_NAME, ENUM_A_WITH_B_NAME, ENUM_A_WITH_C_NAME, ENUM_A_WITH_D_NAME
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple property enum") {
		const String code =
				GDSCRIPT(R"(
enum { A, B, C }
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Sequential signals should follow one another, then line break for the next element type") {
		const String code =
				GDSCRIPT(R"(
signal my_signal_a
signal my_signal_b
var my_variable = 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
signal my_signal_a
signal my_signal_b

var my_variable = 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should only have one extra line after an inner class") {
		const String code =
				GDSCRIPT(R"(
class InnerClass:
	var my_variable
<*noop*>
<*noop*>
)");
		const String pre_formatted =
				GDSCRIPT(R"(
class InnerClass:
	var my_variable
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a null value correctly") {
		const String code =
				GDSCRIPT(R"(
var my_value = null
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a typed null value correctly") {
		const String code =
				GDSCRIPT(R"(
var my_value: Object = null
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][ClassMembers]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][ClassSignatures]") {
	TEST_CASE("Should output a simple class") {
		const String code =
				GDSCRIPT(R"(
extends Node
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class with a name") {
		const String code =
				GDSCRIPT(R"(
extends Sprite2D
class_name MySpriteExtension
)");
		const String pre_formatted =
				GDSCRIPT(R"(
class_name MySpriteExtension
extends Sprite2D
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class that extends a subclass") {
		const String code = GDSCRIPT(R"(
extends OuterClass.InnerClass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class that extends a script file") {
		const String code =
				GDSCRIPT(R"(
extends "res://script.gd"
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class that extends a script file's subclass") {
		const String code =
				GDSCRIPT(R"(
extends "res://script.gd".SubClass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class that has the tool annotation") {
		const String code =
				GDSCRIPT(R"(
@tool
extends Node
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class with an icon set") {
		const String code =
				GDSCRIPT(R"(
@icon("res://CustomTypes/icon.svg")
class_name MyClass
extends Node
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][ClassSignatures]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][ClassFunctions]") {
	TEST_CASE("Should output a simple class method") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method with a parameter") {
		const String code =
				GDSCRIPT(R"(
func _process(delta):
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method with multiple parameters") {
		const String code =
				GDSCRIPT(R"(
func operate_lever(lever_id, operator_id):
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method with multiple parameters with specified types") {
		const String code =
				GDSCRIPT(R"(
func operate_lever(lever_id:int, operator_id:int):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func operate_lever(lever_id: int, operator_id: int):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method with multiple parameters with specified types and default values") {
		const String code =
				GDSCRIPT(R"(
func operate_lever(lever_id:int=0, operator_id:int=1):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func operate_lever(lever_id: int = 0, operator_id: int = 1):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method with multiple parameters with inferred types and default values") {
		const String code =
				GDSCRIPT(R"(
func operate_lever(lever_id:=0, operator_id:=1):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func operate_lever(lever_id := 0, operator_id := 1):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method with multiple parameters that wrap due to length") {
		const String code =
				GDSCRIPT(R"(
func a_long_function_name_with_a_lot_of_params(such_as_this_one, and_this_one, and_also_this_one, not_to_mention_this_one_over_here, but_not_this_one):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func a_long_function_name_with_a_lot_of_params(
	such_as_this_one,
	and_this_one,
	and_also_this_one,
	not_to_mention_this_one_over_here,
	but_not_this_one
):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple class method that wraps due to length, with multiple parameters that do not wrap") {
		const String code =
				GDSCRIPT(R"(
func a_medium_function_name(a_middling_length_of_params, with_a_couple_identifiers, but_not_too_many):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func a_medium_function_name(
	a_middling_length_of_params, with_a_couple_identifiers, but_not_too_many
):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with a return type") {
		const String code =
				GDSCRIPT(R"(
func _ready()->void:
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with a variable being assigned") {
		const String code =
				GDSCRIPT(R"(
func _ready()->void:
	var my_variable := 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
	var my_variable := 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with a return statement") {
		const String code =
				GDSCRIPT(R"(
func _ready()->void:
	return
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
	return
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with a non-void return statement") {
		const String code =
				GDSCRIPT(R"(
func build()->void:
	return 5
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func build() -> void:
	return 5
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with a wrapping return statement due to length") {
		const String code =
				GDSCRIPT(R"(
func build()->void:
	return another_function_with_a_long_name_and_thus("lots", "of", "parameters", "that", "take up", "space")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func build() -> void:
	return another_function_with_a_long_name_and_thus(
		"lots", "of", "parameters", "that", "take up", "space"
	)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with a breakpoint") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	breakpoint
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with an assignment") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	my_var = 50
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class method with an await statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	await get_tree().idle_frame
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked type as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	MyNakedType
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked binary op as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	2+2
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	2 + 2
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked array as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	[0,1,2]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	[0, 1, 2]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked Dictionary as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	{0:5}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	{0: 5}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked get node as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	$Node
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked literal as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	5
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked preload as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	preload("Node.tscn")
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked self as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	self
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a naked subscript as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	the_array[0]
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a ternary block as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	5 if true else 0
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with a unary block as a statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	-x
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method marked as static") {
		const String code =
				GDSCRIPT(R"(
static func build():
	return 5
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a method with an RPC annotation") {
		const String code =
				GDSCRIPT(R"(
@rpc func build():
	return 5
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@rpc
func build():
	return 5
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that does a function call from a callee") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	the_callee.the_call()
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that does a function call for super") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	super.the_call()
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should wrap a long function call argument array properly") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var arr = []
	arr.append_array(["long string 1", "long string 2", "long string 3", "long string 4", "long string 5", "long string 6"])
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var arr = []
	arr.append_array([
		"long string 1", "long string 2", "long string 3", "long string 4", "long string 5", "long string 6"
	])
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function that has an assert condition") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	assert(some_condition(), "Should have called a condition")
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function that has an assert condition that wraps due to length with a condition that breaks due to length") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	assert(some_condition("with", "a bunch of parameters", "to cause a wrap", "on multiple lines that wrap and go on a bit too long"),"Should have called a condition that wraps due to length, especially with a long message")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	assert(some_condition(
		"with",
		"a bunch of parameters",
		"to cause a wrap",
		"on multiple lines that wrap and go on a bit too long"
	), "Should have called a condition that wraps due to length, especially with a long message")
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Sequential functions should be separated by newlines") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	pass
func _process(delta):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	pass
<*noop*>
<*noop*>
func _process(delta):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should have a call with an array as an argument keep its brackets near the parentheses") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var arr = []
	arr.append_array([
		"test with a long string 1",
		"test with a long string 2",
		"test with a long string 3",
		"test with a long string 4",
	])
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should wrap long lines but not be too aggressive about parentheses") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var this_is_a_very_long_boolean_for_test_purposes: bool = false
	if this_is_a_very_long_boolean_for_test_purposes or this_is_a_very_long_boolean_for_test_purposes or this_is_a_very_long_boolean_for_test_purposes:
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var this_is_a_very_long_boolean_for_test_purposes: bool = false
	if (
		this_is_a_very_long_boolean_for_test_purposes
		or this_is_a_very_long_boolean_for_test_purposes
		or this_is_a_very_long_boolean_for_test_purposes
	):
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][ClassFunctions]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][InnerClasses]") {
	TEST_CASE("Inner class should handle variables") {
		const String code =
				GDSCRIPT(R"(
class MyInnerClass:
	var hi

	func my_inner_class_function() -> void:
		pass

	func another_one() -> void:
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
class MyInnerClass:
	var hi


	func my_inner_class_function() -> void:
		pass


	func another_one() -> void:
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][InnerClasses]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][NestedSuites]") {
	TEST_CASE("Should output a simple if true/else statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass
	else:
		pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple if/elif/else statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if 0:
		pass
	elif 1:
		pass
	else:
		pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a if statement that wraps due to length") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if some_conditional_function_with_a_true_false_return_type("and a chunk", "of long", "parameters", "with strings"):
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	if (
		some_conditional_function_with_a_true_false_return_type(
			"and a chunk", "of long", "parameters", "with strings"
		)
	):
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Nested if blocks should not stack newlines at end of suite") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if true:
		if true:
			if true:
				pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an infinite while loop") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	while true:
		pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a while loop with a condition call") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	while some_conditional_function_with_a_true_false_return_type("and a chunk", "of long", "parameters", "with strings"):
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	while (
		some_conditional_function_with_a_true_false_return_type(
			"and a chunk", "of long", "parameters", "with strings"
		)
	):
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a while loop with a break statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	while true:
		break
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	while true:
		break
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a while loop with a continue statement") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	while true:
		continue
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	while true:
		continue
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple match block") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := true
	match test:
		true:
			pass
		false:
			pass
		
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := true
	match test:
		true:
			pass
		false:
			pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with multiple patterns") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := 50
	match test:
		50, 75, 100:
			pass
		60, 85, 105:
			pass
		
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := 50
	match test:
		50, 75, 100:
			pass
		60, 85, 105:
			pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with a wildcard entry") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := 50
	match test:
		50:
			print(50)
		_:
			print("Not 50")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := 50
	match test:
		50:
			print(50)
		_:
			print("Not 50")
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with a variable entry") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := 50
	match test:
		MY_CONST:
			print(50)
		MY_OTHER_CONST:
			print("Not 50")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := 50
	match test:
		MY_CONST:
			print(50)
		MY_OTHER_CONST:
			print("Not 50")
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with a binding entry") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := 30
	match test:
		50:
			print(50)
		25:
			print(25)
		var result:
			print("Not %s" % [result])
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := 30
	match test:
		50:
			print(50)
		25:
			print(25)
		var result:
			print("Not %s" % [result])
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with a array entry") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := [0, 1, 2]
	match test:
		[0, 1, 2]:
			print(50)
		[3, 4, 5]:
			print(25)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := [0, 1, 2]
	match test:
		[0, 1, 2]:
			print(50)
		[3, 4, 5]:
			print(25)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with an open-ended array entry") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := [0, 1, 2]
	match test:
		[0, 1, ..]:
			print(50)
		[3, 4, 5]:
			print(25)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := [0, 1, 2]
	match test:
		[0, 1, ..]:
			print(50)
		[3, 4, 5]:
			print(25)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a match block with a dictionary entry") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var test := {"friend": "Me", "best": true}
	match test:
		{"friend": "Me", "best": true}:
			print("happy")
		{"friend": "Me", "best": false}:
			print("sad")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var test := {"friend": "Me", "best": true}
	match test:
		{"friend": "Me", "best": true}:
			print("happy")
		{"friend": "Me", "best": false}:
			print("sad")
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a simple for loop") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	for i in 10:
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	for i in 10:
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a for loop with a wrapped condition due to length") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	for i in ["A long string here","A long string there","A long string, everywhere","Hither and thither","The long strings go","Forcing us to wrap conditional statements"]:
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
	for i in [
		"A long string here",
		"A long string there",
		"A long string, everywhere",
		"Hither and thither",
		"The long strings go",
		"Forcing us to wrap conditional statements",
	]:
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable that contains a lambda") {
		const String code =
				GDSCRIPT(R"(
var my_lambda = func():
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_lambda = func():
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not crash with malformed match") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	var x = 0
	match x:
		0
)");
		GDScriptFormat formatter;
		String output;
		Error err = formatter.format(code, output);
		CHECK(err != OK);
	}

	TEST_CASE("Should not error out on a lambda wrapped in a multiline") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var the_lambda = (
		func():
			return true
	)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var the_lambda = func():
		return true
)");

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][NestedSuites]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Usability]") {
	TEST_CASE("Should format differently based on desired wrapping length") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	if my_condition_is_long_enough("it should wrap", "due to length", "on multiple lines"):
		print("Told you")
)");

		const String pre_formatted80 =
				GDSCRIPT(R"(
func _ready() -> void:
	if (
		my_condition_is_long_enough(
			"it should wrap", "due to length", "on multiple lines"
		)
	):
		print("Told you")
)");

		const String pre_formatted100 =
				GDSCRIPT(R"(
func _ready() -> void:
	if my_condition_is_long_enough("it should wrap", "due to length", "on multiple lines"):
		print("Told you")
)");

		GDScriptFormat formatter;
		String output80;
		String output100;
		Error err;
		formatter.indent_in_multiline_block = 1;
		formatter.line_length_maximum = 80;
		err = formatter.format(code, output80);
		CHECK(err == OK);
		formatter.line_length_maximum = 100;
		err = formatter.format(code, output100);
		CHECK(err == OK);

		CHECK_EQ(output80, pre_formatted80);
		CHECK_EQ(output100, pre_formatted100);
	}

	TEST_CASE("Should output user-inputted incidental new lines") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var my_variable

	var my_other_variable
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var my_variable

	var my_other_variable
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output user-inputted incidental new lines but compress multiple lines into one") {
		const String code = "func _ready():\n\tvar my_variable\n\n\n\n\tvar my_other_variable";
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var my_variable

	var my_other_variable
)");

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Usability]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Comments]") {
	TEST_CASE("Should output a class header with all related comments") {
		const String code =
				GDSCRIPT(R"(
# Tool header
@tool # Tool inline
# Icon header
@icon("res://icon.png") # Icon inline
# Class name header
class_name MyClass # Class name inline
# Extends header
extends Node # Extends inline
)");
		const String pre_formatted =
				GDSCRIPT(R"(
# Tool header
@tool # Tool inline
# Icon header
@icon("res://icon.png") # Icon inline
# Class name header
class_name MyClass # Class name inline
# Extends header
extends Node # Extends inline
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment above it") {
		const String code =
				GDSCRIPT(R"(
# A Docstring for the variable
var my_variable = 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
# A Docstring for the variable
var my_variable = 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to it") {
		const String code =
				GDSCRIPT(R"(
var my_variable = 0 # With an explanatory text
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable = 0 # With an explanatory text
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a wrapped variable with a comment next to it") {
		const String code =
				GDSCRIPT(R"(
var my_variable = some_conditional_value() + "A fairly long string, to cause a wrap" # With an explanatory text
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable = (
	some_conditional_value()
	+ "A fairly long string, to cause a wrap" # With an explanatory text
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment above it after a variable with a comment next to it") {
		const String code =
				GDSCRIPT(R"(
var my_variable = 0 # My first variable
# My Second variable
var my_other_variable = 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable = 0 # My first variable
# My Second variable
var my_other_variable = 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a wrapped variable with a comment on a binary operator element inside of it") {
		const String code =
				GDSCRIPT(R"(
var my_variable = (
	0 # My first variable
	+ 1
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable = (
	0 # My first variable
	+ 1
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function calculating the triangle area from 3 Vector3's") {
		const String code =
				GDSCRIPT(R"(
func    triangle_area(p1    :Vector3  , p2:Vector3,p3   :  Vector3):

	return (p2 - p1).cross( p3 - p1 ).length() / 2.0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func triangle_area(p1: Vector3, p2: Vector3, p3: Vector3):
	return (p2 - p1).cross(p3 - p1).length() / 2.0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function call with a comment on one of its parameters") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := some_function_call(
	0 #with that parameter
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready
var my_variable := some_function_call(
	0 # with that parameter
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function call with a comment on one of its non-literal parameters") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := some_function_call(
	SOME_CONST #with that parameter
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready
var my_variable := some_function_call(
	SOME_CONST # with that parameter
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function call with a comment on a nested call's parameters") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := some_function_call(
	some_nested_call(
		0 #with that parameter
	)
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready
var my_variable := some_function_call(
	some_nested_call(
		0 # with that parameter
	)
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a constant with a comment above it") {
		const String code =
				GDSCRIPT(R"(
# A Docstring for the constant
const MY_VARIABLE := 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
# A Docstring for the constant
const MY_VARIABLE := 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a constant with a comment next to it") {
		const String code =
				GDSCRIPT(R"(
const MY_VARIABLE := 0 # A comment for the constant
)");
		const String pre_formatted =
				GDSCRIPT(R"(
const MY_VARIABLE := 0 # A comment for the constant
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a constant with a comment next to one of its binary operator elements") {
		const String code =
				GDSCRIPT(R"(
const MY_VARIABLE := (
	0 # A comment for the literal
	+ 3
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
const MY_VARIABLE := (
	0 # A comment for the literal
	+ 3
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a constant with a comment next to one of its nested binary operator elements") {
		const String code =
				GDSCRIPT(R"(
const MY_VARIABLE := (
	0
	+ 4 # A comment for the literal
	+ 3
	+ 8
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
const MY_VARIABLE := (
	0
	+ 4 # A comment for the literal
	+ 3
	+ 8
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to a function call parameter") {
		const String code =
				GDSCRIPT(R"(
var my_variable := my_call(
	0,
	1 # The comment is here
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := my_call(
	0,
	1 # The comment is here
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to a nested function call parameter") {
		const String code =
				GDSCRIPT(R"(
var my_variable := my_call(
	0,
	my_other_call(
		0,
		1 # The comment is here
	)
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := my_call(
	0,
	my_other_call(
		0,
		1 # The comment is here
	)
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element in an array") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [
	0,
	1 # The comment is here
]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	0,
	1, # The comment is here
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element in a dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	"job": "Investigator" # The comment is here
}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	"job": "Investigator", # The comment is here
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element value in a dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	"job": (
		"Investigator"
	) # The comment is here
}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	"job": "Investigator", # The comment is here
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment above an element key in a dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	# The comment is here
	"job": "Investigator"
}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	# The comment is here
	"job": "Investigator",
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment above an element value in a dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	"job": (
		# The comment is here
		"Investigator"
	)
}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	# The comment is here
	"job": "Investigator",
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element key and value in a dictionary") {
		const String code =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	"job": ( # There is a comment here
		"Investigator"
	) # And a comment here
}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := {
	"name": "Elizabeth",
	# There is a comment here
	"job": "Investigator", # And a comment here
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element in an array") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [
	0, 1, 2, 3,
	4, # This is the special one
	5, 6, 7, 8
]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	0,
	1,
	2,
	3,
	4, # This is the special one
	5,
	6,
	7,
	8,
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment above an element in an array") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [
	0, 1, 2, 3,
	# This is the special one
	4,
	5, 6, 7, 8
]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	0,
	1,
	2,
	3,
	# This is the special one
	4,
	5,
	6,
	7,
	8,
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element in a nested array") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [
	0, 1, 2, [
		0, 1, 2, # The comment can go here
		3
	], 4, 5, 6, 7, 8, 9
]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	0,
	1,
	2,
	[
		0,
		1,
		2, # The comment can go here
		3,
	],
	4,
	5,
	6,
	7,
	8,
	9,
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an element in a double nested array") {
		const String code =
				GDSCRIPT(R"(
var my_variable := [0,1,2,[0,1,[
	0,1, # The comment can go here
	2
],3],4,5,6,7,8,9]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_variable := [
	0,
	1,
	2,
	[
		0,
		1,
		[
			0,
			1, # The comment can go here
			2,
		],
		3,
	],
	4,
	5,
	6,
	7,
	8,
	9,
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to a cast statement") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := (
	$Player as CharacterBody2D # The comment
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready var my_variable := $Player as CharacterBody2D # The comment
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to an identifier") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := (
	MY_CONST # The comment
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready var my_variable := MY_CONST # The comment
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to a get node") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := (
	$Player/Sprite2D # The comment
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready var my_variable := $Player/Sprite2D # The comment
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to a preload value") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := preload(
	"res://Player/PlayerSprite.png" # The comment
)
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment next to a subscript index") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := MY_CONST_ARRAY[
	0 # The comment
]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready
var my_variable := MY_CONST_ARRAY[
	0 # The comment
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment as part of a ternary operator") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := (
	50 if SOME_DEVELOPMENT_CONST # is enabled
	else 75
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready
var my_variable := (
	50 if SOME_DEVELOPMENT_CONST # is enabled
	else 75
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a variable with a comment as part of a ternary operator on the false") {
		const String code =
				GDSCRIPT(R"(
@onready var my_variable := (
	50 if SOME_DEVELOPMENT_CONST
	else 75 # is 75
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready
var my_variable := (
	50 if SOME_DEVELOPMENT_CONST
	else 75 # is 75
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an assignment with a comment above it") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	# A comment!
	some_value = 30
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an assignment with a comment next to it") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	some_value = 30 # A comment!
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an assignment with a comment next to its literal value") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	some_value = (
		30 # A comment!
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
	some_value = 30 # A comment!
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an assignment with a comment next to its identifier value") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	some_value = (
		some_other # A comment!
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
	some_value = some_other # A comment!
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an await with a comment") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	await get_tree().process_frame # A comment goes here
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a get node statement with a comment") {
		const String code =
				GDSCRIPT(R"(
@onready var some_var := (
	$Path/To/Node # Comment here
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
@onready var some_var := $Path/To/Node # Comment here
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with a comment above it") {
		const String code =
				GDSCRIPT(R"(
# Comment above
func _ready() -> void:
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with two comments above it") {
		const String code =
				GDSCRIPT(R"(
# Comment 1
# Comment 2
func _ready() -> void:
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with a comment next to it") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void: # Comment next
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with a comment next to a parameter") {
		const String code =
				GDSCRIPT(R"(
func a_custom_function(
	a_parameter # with a comment
) -> void:
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with a comment next to one parameter") {
		const String code =
				GDSCRIPT(R"(
func a_custom_function(
	a_parameter, # with a comment
	another_parameter
) -> void:
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class with a comment above and next to it") {
		const String code =
				GDSCRIPT(R"(
# The comment above
class MyClass: # The comment next
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a signal with a comment inside its parameters") {
		const String code =
				GDSCRIPT(R"(
# The comment above
signal some_signal(with,
	some, # And a comment here
	params
)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
# The comment above
signal some_signal(
	with,
	some, # And a comment here
	params
)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function return with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	# return header
	return # end early
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an assert with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
	assert(some_complex_condition(
		"that", # comment!
		"breaks"
	), "And a message") # And a comment
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output an unnamed enum with comments") {
		const String code =
				GDSCRIPT(R"(
# Enum header
enum { VALUE_1, VALUE_2 = 3, VALUE_3,
# Value header
VALUE_4, # value inline
} # enum inline
)");
		const String pre_formatted =
				GDSCRIPT(R"(
# Enum header
enum {
	VALUE_1,
	VALUE_2 = 3,
	VALUE_3,
	# Value header
	VALUE_4, # value inline
} # enum inline
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should out a match statement with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# Match header
	match some_value: # Match inline
		# Value header
		0: # Value inline
			pass
		1:
			pass
		2:
			pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output if blocks with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# If header
	if my_condition: # if inline
		pass
	# elif header
	elif my_other_condition: # elif inline
		pass
	# else header
	else: # else inline
		pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a for loop with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# For header
	for i in my_condition: # For inline
		pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a while loop with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# While header
	while my_condition: # While inline
		pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a break statement with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	for i in my_condition:
		# Break header
		break # break inline
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a continue statement with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	for i in my_condition:
		# Continue header
		continue # Continue inline
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a pass statement with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# Pass header
	pass # Pass inline
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a breakpoint statement with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# Breakpoint header
	breakpoint # Breakpoint inline
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a prop with comments") {
		const String code =
				GDSCRIPT(R"(
# prop header
var my_property: # prop inline
	# setter header
	set(value): # setter inline
		my_property = value
	# getter header
	get: # getter inline
		return my_property
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a function with a footer comment") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	pass

	# Comment at the bottom

func _other_function():
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	pass

	# Comment at the bottom
<*noop*>
<*noop*>
func _other_function():
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output a class with a footer comment") {
		const String code =
				GDSCRIPT(R"(
extends Node

class SubClass:
	extends Resource

	# Footer comment

# Footer comment
)");
		const String pre_formatted =
				GDSCRIPT(R"(
extends Node
<*noop*>
<*noop*>
class SubClass:
	extends Resource

	# Footer comment

# Footer comment
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output docstrings on classes and functions") {
		const String code =
				GDSCRIPT(R"(
## A class that has a specific documented job
class_name MyClass
extends RefCounted

## Returns a value
func some_public_api_func() -> int:
	return 0
)");
		const String pre_formatted =
				GDSCRIPT(R"(
## A class that has a specific documented job
class_name MyClass
extends RefCounted
<*noop*>
<*noop*>
## Returns a value
func some_public_api_func() -> int:
	return 0
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output indexed calls with comments") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	# Header
	sd.call_one() # Inline
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	# Header
	sd.call_one() # Inline
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not cause errors with disabled lines") {
		const String code =
				GDSCRIPT(R"(
func _ready() -> void:
#	print("disabled code")
	print("enabled code")
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready() -> void:
#	print("disabled code")
	print("enabled code")
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not cause errors with disabled lines in class") {
		const String code =
				GDSCRIPT(R"(
class SomeClass:
#	var a_disabled_var
	var an_enabled_var
)");
		const String pre_formatted =
				GDSCRIPT(R"(
class SomeClass:
#	var a_disabled_var
	var an_enabled_var
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not cause errors with disabled lines in property") {
		const String code =
				GDSCRIPT(R"(
var some_var:
#	get: disabled line
	get:
		return some_var
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var some_var:
#	get: disabled line
	get:
		return some_var
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not cause errors with disabled lines in both properties") {
		const String code =
				GDSCRIPT(R"(
var some_var:
#	get: disabled line
	get:
		return some_var
#	set(v):
	set(v):
		some_var = v
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var some_var:
#	set(v):
	set(v):
		some_var = v
#	get: disabled line
	get:
		return some_var
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not crash on empty array with inline comment over no element") {
		const String code =
				GDSCRIPT(R"(
var array = [ # Comment
]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var array = [
	# Comment
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not crash on empty dict with inline comment over no element") {
		const String code =
				GDSCRIPT(R"(
var dict = { # Comment
}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var dict = {
	# Comment
}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not crash on empty parameter with inline comment over no element") {
		const String code =
				GDSCRIPT(R"(
func some_func( # Comment
):
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func some_func(
	# Comment
):
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not crash on empty call with inline comment over no element") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	some_func( # Comment
	)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	some_func(
		# Comment
	)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a disabled statement in a if block") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if true:
#		comment
		print("hi")
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a disabled statement in a parameter block") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	print(
#		"50"
		"30"
	)
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a footer in an array with only one element") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var array = ["string"
		# Comment
	]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var array = [
		"string",
		# Comment
	]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a footer in a dictionary with only one element") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var dictionary = {"string": "string"
		# Comment
	}
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	var dictionary = {
		"string": "string",
		# Comment
	}
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a footer in a call with only one parameter") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	call("string"
		# Comment
	)
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	call(
		"string"
		# Comment
	)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should continue parsing a suite after a dedented comment") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass
#	comment
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass

#	comment
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should output comments above an onready variable") {
		const String code =
				GDSCRIPT(R"(
## I am a comment describing var hi
@onready var hi
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("A full commented function should output correctly") {
		// Code by clayjohn at https://github.com/godotengine/godot-docs/issues/4834
		const String code =
				GDSCRIPT(R"(
func _ready():
	# We will be using our own RenderingDevice to handle the compute commands
	var rd = RenderingServer.create_local_rendering_device()

	# Create shader and pipeline
	var shader_file = load("res://compute_example.glsl")
	var shader_bytecode = shader_file.get_bytecode()
	var shader = rd.shader_create(shader_bytecode)
	var pipeline = rd.compute_pipeline_create(shader)

	# Data for compute shaders has to come as an array of bytes
	var pba = PackedByteArray()
	pba.resize(64)
	for i in range(16):
		pba.encode_float(i * 4, 2.0)

	# Create storage buffer
	# Data not needed, can just create with length
	var storage_buffer = rd.storage_buffer_create(64, pba)

	# Create uniform set using the storage buffer
	var u = RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = 0
	u.add_id(storage_buffer)
	var uniform_set = rd.uniform_set_create([u], shader, 0)

	# Start compute list to start recording our compute commands
	var compute_list = rd.compute_list_begin()
	# Bind the pipeline, this tells the GPU what shader to use
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	# Binds the uniform set with the data we want to give our shader
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	# Dispatch 1x1x1 (XxYxZ) work groups
	rd.compute_list_dispatch(compute_list, 2, 1, 1)
	# rd.compute_list_add_barrier(compute_list)
	# Tell the GPU we are done with this compute task
	rd.compute_list_end()
	# Force the GPU to start our commands
	rd.submit()
	# Force the CPU to wait for the GPU to finish with the recorded commands
	rd.sync()

	# Now we can grab our data from the storage buffer
	var byte_data = rd.buffer_get_data(storage_buffer)
	for i in range(16):
		print(byte_data.decode_float(i * 4))
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	# We will be using our own RenderingDevice to handle the compute commands
	var rd = RenderingServer.create_local_rendering_device()

	# Create shader and pipeline
	var shader_file = load("res://compute_example.glsl")
	var shader_bytecode = shader_file.get_bytecode()
	var shader = rd.shader_create(shader_bytecode)
	var pipeline = rd.compute_pipeline_create(shader)

	# Data for compute shaders has to come as an array of bytes
	var pba = PackedByteArray()
	pba.resize(64)
	for i in range(16):
		pba.encode_float(i * 4, 2.0)

	# Create storage buffer
	# Data not needed, can just create with length
	var storage_buffer = rd.storage_buffer_create(64, pba)

	# Create uniform set using the storage buffer
	var u = RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = 0
	u.add_id(storage_buffer)
	var uniform_set = rd.uniform_set_create([u], shader, 0)

	# Start compute list to start recording our compute commands
	var compute_list = rd.compute_list_begin()
	# Bind the pipeline, this tells the GPU what shader to use
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	# Binds the uniform set with the data we want to give our shader
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	# Dispatch 1x1x1 (XxYxZ) work groups
	rd.compute_list_dispatch(compute_list, 2, 1, 1)
	# rd.compute_list_add_barrier(compute_list)
	# Tell the GPU we are done with this compute task
	rd.compute_list_end()
	# Force the GPU to start our commands
	rd.submit()
	# Force the CPU to wait for the GPU to finish with the recorded commands
	rd.sync()

	# Now we can grab our data from the storage buffer
	var byte_data = rd.buffer_get_data(storage_buffer)
	for i in range(16):
		print(byte_data.decode_float(i * 4))
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a disabled line between an if and its else") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass
#		Comment
	else:
		pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass
#		Comment
	else:
		pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not error out on a disabled line in the middle of a suite") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass
#		Comment
		pass
	pass
)");
		const String pre_formatted =
				GDSCRIPT(R"(
func _ready():
	if true:
		pass
#		Comment
		pass
	pass
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should not have an extra line after an array footer with a long first member") {
		const String code =
				GDSCRIPT(R"(
var my_array = [
	"there is a bug with an extra newline at the end of arrays but only when the contents have long lines",
	# Comment

]
)");
		const String pre_formatted =
				GDSCRIPT(R"(
var my_array = [
	"there is a bug with an extra newline at the end of arrays but only when the contents have long lines",
	# Comment
]
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Should accept a comment in the last statement of a function when accessing an array") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var v = [0]
	var a = v[0] # A


func another():
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Comments on top of static functions should stay") {
		const String code =
				GDSCRIPT(R"(
## This comment should stay after format
static func test():
	pass
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Comment at the end of a return value should stay") {
		const String code =
				GDSCRIPT(R"(
func test_func() -> bool:
	if true:
		return true # In this case, stop.
	return false
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Comments at the end of `get_node` should not multiply") {
		const String code =
				GDSCRIPT(R"(
@onready var a_node = $ANode
@onready var b_node = a_node.get_node(^"BNode") # Test
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Comments should not be considered as dictionary keys") {
		const String code =
				GDSCRIPT(R"(
func _ready():
	var dict = {
		"key1": "value1", # Comment
		# Comment
		"key2": "value2",
	}
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Comments should not count as expressions") {
		SUBCASE("Function body") {
			const String code =
					GDSCRIPT(R"(
func _run() -> void:
	# This should not count as an expression
)");
			GDScriptFormat formatter;
			formatter.indent_in_multiline_block = 1;
			String output;
			Error err = formatter.format(code, output);
			CHECK(err != OK);
		}

		SUBCASE("Condition body") {
			const String code =
					GDSCRIPT(R"(
func _run() -> void:
	if true:
		# This should not count as an expression
)");
			GDScriptFormat formatter;
			formatter.indent_in_multiline_block = 1;
			String output;
			Error err = formatter.format(code, output);
			CHECK(err != OK);
		}
	}

	TEST_CASE("Long header comments should not make a small expression split") {
		const String code = GDSCRIPT(R"(
func _ready():
	var some_node
	# This bug only happens when the comment on the previous line exceeds 80 chars.
	some_node.visible = 5 > 3
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Comments]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Syntax]") {
	TEST_CASE("Syntactic sugar should be preserved") {
		SUBCASE("get_node ($)") {
			// Should let as-is
			String code =
					GDSCRIPT(R"(
@onready var child_node = $ChildNode
)");
			String pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);

			// Should remove useless quotes
			code =
					GDSCRIPT(R"(
@onready var child_node = $"ChildNode"
)");
			pre_formatted =
					GDSCRIPT(R"(
@onready var child_node = $ChildNode
)");

			CHECK_FORMAT(code, pre_formatted);

			// Should keep quotes
			code =
					GDSCRIPT(R"(
@onready var sub_child_node = $"ChildNode/Node$/Path"
)");
			pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("Unique names (%)") {
			// Should let as-is
			String code =
					GDSCRIPT(R"(
@onready var unique_node = %UniqueNode
)");
			String pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);

			// Should remove useless quotes
			code =
					GDSCRIPT(R"(
@onready var unique_node = %UniqueNodeIndeed
)");
			pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);

			// Should keep quotes
			code =
					GDSCRIPT(R"(
@onready var unique_node = %"UniqueNodeIndeedWith$"
)");
			pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);
		}
	}

	TEST_CASE("Parentheses should be kept") {
		SUBCASE("Using format%") {
			const String code =
					GDSCRIPT(R"(
func _ready():
	var number = 1234.567
	var string = "%1.1f k" % (number * 0.001)
)");
			const String pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);
		}
	}

	TEST_CASE("Literals should be kept as-is") {
		SUBCASE("Number below the float precision") {
			const String code =
					GDSCRIPT(R"(
func _ready():
	var number = 0.0000000000000000000000000000000000000000000000000000000000000001
)");
			const String pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);
		}
	}

	TEST_CASE("Annotations should not be split if the line is not too long") {
		SUBCASE("Under the line limit") {
			const String code =
					GDSCRIPT(R"(
@export_exp_easing var very_small_var: float = 0.0
)");
			const String pre_formatted = code;

			CHECK_FORMAT(code, pre_formatted);
		}

		SUBCASE("Above the line limit") {
			const String code =
					GDSCRIPT(R"(
@export_exp_easing var very_long_long_long_long_long_long_long_long_long_long_long_long_long_long_var: float = 0.0
)");
			const String pre_formatted =
					GDSCRIPT(R"(
@export_exp_easing
var very_long_long_long_long_long_long_long_long_long_long_long_long_long_long_var: float = 0.0
)");

			CHECK_FORMAT(code, pre_formatted);
		}
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Syntax]")

TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Misc]") {
	TEST_CASE("The custom newlines should be remembered") {
		// `vformat` because "misc/scripts/header_guards.sh"
		// don't like two consecutive new lines.
		const String code = GDSCRIPT(R"(
extends Node2D
<*noop*>
<*noop*>
class Light:
	var energy

var energy_slider := Range.new()
var height_slider := Range.new()
var light := Light.new()

func _ready() -> void:
	energy_slider.value_changed.connect(func(value):
		light.energy = energy_slider.value)

	height_slider.value_changed.connect(func(value):
		light.height = height_slider.value)

	height_slider.value_changed.connect(func(value):
		light.height = height_slider.value)
)");
		const String pre_formatted = GDSCRIPT(R"(
extends Node2D
<*noop*>
<*noop*>
class Light:
	var energy

var energy_slider := Range.new()
var height_slider := Range.new()
var light := Light.new()
<*noop*>
<*noop*>
func _ready() -> void:
	energy_slider.value_changed.connect(func(value):
		light.energy = energy_slider.value)

	height_slider.value_changed.connect(func(value):
		light.height = height_slider.value)

	height_slider.value_changed.connect(func(value):
		light.height = height_slider.value)
)");

		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Function default values should have spaces") {
		String code;
		String pre_formatted;

		// Without typing
		code =
				GDSCRIPT(R"(
func my_func(optional_param= null):
	pass
)");
		pre_formatted =
				GDSCRIPT(R"(
func my_func(optional_param = null):
	pass
)");
		CHECK_FORMAT(code, pre_formatted);

		// With explicit typing
		code =
				GDSCRIPT(R"(
func my_func(optional_param: Node =null):
	pass
)");
		pre_formatted =
				GDSCRIPT(R"(
func my_func(optional_param: Node = null):
	pass
)");
		CHECK_FORMAT(code, pre_formatted);

		// With inferred typing
		code =
				GDSCRIPT(R"(
func my_func(optional_param:=""):
	pass
)");
		pre_formatted =
				GDSCRIPT(R"(
func my_func(optional_param := ""):
	pass
)");
		CHECK_FORMAT(code, pre_formatted);
	}

	TEST_CASE("Comments after a bitshift should not create a multiline expression") {
		const String code = GDSCRIPT(R"(
var _MAX_FILE_SIZE = 1 << 20 # 1 MiB
)");
		const String pre_formatted = code;

		CHECK_FORMAT(code, pre_formatted);
	}
} // TEST_SUITE("[Modules][GDScript][GDScriptFormatter][Misc]")

} //namespace GDScriptTests

#endif // TEST_GDSCRIPT_FORMATTER_H
