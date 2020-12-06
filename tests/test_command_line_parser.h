/*************************************************************************/
/*  test_command_line_parser.h                                           */
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

#ifndef TEST_COMMAND_LINE_PARSER_H
#define TEST_COMMAND_LINE_PARSER_H

#include "core/command_line_parser.h"

#include "thirdparty/doctest/doctest.h"

namespace TestCommandLineParser {

TEST_CASE("[CommandLineParser] Built-in options") {
	CommandLineParser parser;
	Ref<CommandLineOption> option;
	CAPTURE(parser.get_error());

	SUBCASE("[CommandLineParser] Help option") {
		option = parser.add_help_option();
	}
	SUBCASE("[CommandLineParser] Version option") {
		option = parser.add_version_option();
	}

	CHECK_MESSAGE(
			parser.get_option_count() == 1,
			"options count should match the expected");
	CHECK_MESSAGE(
			parser.get_option(0) == option,
			"The added built-in option should be available in the parser.");
}

template <int arg_count>
class ParserWithOptionFixture {
protected:
	CommandLineParser parser;
	Ref<CommandLineOption> option = memnew(CommandLineOption(sarray("i", "input"), arg_count));

	String short_option() {
		return parser.get_short_prefixes().get(0) + option->get_names().get(0);
	}
	String long_option() {
		return parser.get_long_prefixes().get(0) + option->get_names().get(1);
	}

	ParserWithOptionFixture() {
		parser.add_option(option);
	}
};

TEST_CASE_FIXTURE(ParserWithOptionFixture<0>, "[CommandLineParser] Option functions") {
	const int options_count = parser.get_option_count();
	parser.remove_option(0);
	CHECK_MESSAGE(
			parser.get_option_count() == options_count - 1,
			"The number of options should decrease by 1 after the option is removed.");
	parser.add_option(option);
	CHECK_MESSAGE(
			parser.get_option_count() == options_count,
			"The number of options should increase by 1 after adding a option.");
	CHECK_MESSAGE(
			parser.find_option(option->get_names().get(0)) == option,
			"Searching for a option by name should find the correct option.");
	CHECK_MESSAGE(
			parser.find_option(option->get_names().get(1)) == option,
			"Searching for a option by name should find the correct option.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<1>, "[CommandLineParser] Options validation") {
	option->set_default_args(sarray("a"));

	REQUIRE_MESSAGE(
			parser.parse_args(PackedStringArray()) == OK,
			"Parsing with a valid option should succeed.");

	ERR_PRINT_OFF;

	const int arg_count = option->get_arg_count();
	option->set_arg_count(arg_count + 1);
	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) != OK,
			"Parsing with a positional option that takes no arguments should fail.");
	option->set_arg_count(arg_count);

	const PackedStringArray names = option->get_names();
	option->set_names(PackedStringArray());
	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) != OK,
			"Parsing with a option that does not have any names should fail.");
	option->set_names(names);

	const PackedStringArray default_args = option->get_default_args();
	option->set_default_args(sarray("a", "b"));
	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) != OK,
			"Parsing with a option that have different number of arguments and default arguments should fail.");
	option->set_default_args(default_args);

	const bool required = option->is_required();
	option->set_required(!required);
	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) != OK,
			"Parsing with a option that required required and have default arguments should fail.");
	option->set_required(required);

	option->set_static_checker([](const String &) { return false; }, String());
	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) != OK,
			"Parsing with a option that have default arguments that do not pass checker should fail.");
	option->remove_checker();

	parser.add_option(option);
	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) != OK,
			"Parsing with multiple options that have the same name should fail.");
	parser.remove_option(1);

	ERR_PRINT_ON;

	CHECK_MESSAGE(
			parser.parse_args(PackedStringArray()) == OK,
			"Parsing with a valid option (after reverting all changes) should be succesfull.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<0>, "[CommandLineParser] Arguments forwarding") {
	const PackedStringArray expected_values = sarray("arg1", "arg2");

	CHECK_MESSAGE(
			parser.parse_args(sarray("--", expected_values[0], expected_values[1])) != OK,
			"Parsing forwarded arguments should fail if disabled.");

	parser.set_allow_forwarding_args(true);
	REQUIRE_MESSAGE(
			parser.parse_args(sarray("--")) == OK,
			"Forwarding zero arguments should succeed.");
	CHECK_MESSAGE(
			parser.get_forwarded_args().is_empty(),
			"Forwarded arguments should be empty if no arguments forwarded.");

	REQUIRE_MESSAGE(
			parser.parse_args(sarray("--", expected_values[0], expected_values[1])) == OK,
			"Forwarding two arguments should succeed.");
	CHECK_MESSAGE(
			parser.get_forwarded_args() == expected_values,
			"Forwarded arguments should match with expected.");

	REQUIRE_MESSAGE(
			parser.parse_args(sarray(short_option(), "--", expected_values[0], expected_values[1])) == OK,
			"Forwarding two arguments after a option should succeed.");
	CHECK_MESSAGE(
			parser.get_forwarded_args() == expected_values,
			"Forwarded arguments after a option should match with expected.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<0>, "[CommandLineParser] Option without arguments") {
	String option_name;
	CAPTURE(option_name);
	SUBCASE("[CommandLineParser] Short option") {
		option_name = short_option();

		CHECK_MESSAGE(
				parser.parse_args(sarray("-u")) != OK,
				"Parsing with an unknown option should fail.");
		CHECK_MESSAGE(
				parser.parse_args(sarray(option_name, option_name)) != OK,
				"Parsing with two same options should fail.");

		const Ref<CommandLineOption> another_option = memnew(CommandLineOption(sarray("a"), 0));
		parser.add_option(another_option);
		const String another_option_name = parser.get_short_prefixes().get(0) + another_option->get_names().get(0);
		CAPTURE(another_option_name);

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(option_name + another_option->get_names().get(0))) == OK,
				"Parsing with two compound options should succeed.");
		CHECK_MESSAGE(
				parser.is_set(option),
				"The parsed compound option should be set.");
		CHECK_MESSAGE(
				parser.is_set(another_option),
				"The parsed compound option should be set.");
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0))) == OK,
				"Parsing with two compound options should succeed.");
		CHECK_MESSAGE(
				parser.is_set(option),
				"The parsed compound option should be set.");
		CHECK_MESSAGE(
				parser.is_set(another_option),
				"The parsed compound option should be set.");

		parser.set_allow_compound(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(option_name + another_option->get_names().get(0))) != OK,
				"Parsing with compound options should fail if not allowed.");
	}
	SUBCASE("[CommandLineParser] Long option") {
		CHECK_MESSAGE(
				parser.parse_args(sarray("--test")) != OK,
				"Parsing with an unknown option should fail.");
		CHECK_MESSAGE(
				parser.parse_args(sarray(long_option(), short_option())) != OK,
				"Parsing with two same options should fail.");

		option_name = long_option();
	}

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, "value")) != OK,
			"Parsing with an argument should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name + "=value")) != OK,
			"Parsing with an adjucent argument should fail.");

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, option_name)) != OK,
			"Parsing with two same options should fail.");
	option->set_multitoken(true);
	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, option_name)) == OK,
			"Parsing with two same multitoken options should succeed.");
	CHECK_MESSAGE(
			parser.get_occurence_count(option) == 2,
			"The parsed multitoken option occurence count should match the expected.");

	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name)) == OK,
			"Parsing without arguments should succeed.");
	CHECK_MESSAGE(
			parser.is_set(option),
			"The parsed option should be set.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<1>, "[CommandLineParser] Option that takes 1 argument") {
	const String expected_value = "arg";
	String option_name;
	CAPTURE(option_name);
	CAPTURE(expected_value);
	SUBCASE("[CommandLineParser] Short option") {
		option_name = short_option();

		const Ref<CommandLineOption> another_option = memnew(CommandLineOption(sarray("a"), 0));
		parser.add_option(another_option);
		const String another_option_name = parser.get_short_prefixes().get(0) + another_option->get_names().get(0);
		CAPTURE(another_option_name);

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0) + expected_value)) == OK,
				"Parsing with two compound options should succeed.");
		CHECK_MESSAGE(
				parser.get_value(option) == expected_value,
				"The parsed compound option value should match the expected.");
		CHECK_MESSAGE(
				parser.is_set(another_option),
				"The parsed compound option should be set.");

		parser.set_allow_compound(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0) + expected_value)) != OK,
				"Parsing with compound options should fail if not allowed.");

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(option_name + expected_value)) == OK,
				"Parsing with sticky argument should succeed.");
		CHECK_MESSAGE(
				parser.get_value(option) == expected_value,
				"The parsed sticky argument value should match the expected.");

		parser.set_allow_sticky(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(option_name + expected_value)) != OK,
				"Parsing with sticky argument should fail if not allowed.");
	}
	SUBCASE("[CommandLineParser] Long option") {
		option_name = long_option();
	}

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name)) != OK,
			"Parsing without arguments should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name + "=")) != OK,
			"Parsing with missing adjacent argument should fail.");
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_value)) == OK,
			"Parsing with an argument should succeed.");
	CHECK_MESSAGE(
			parser.get_value(option) == expected_value,
			"The parsed argument value should match the expected.");
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name + "=" + expected_value)) == OK,
			"Parsing with an adjacent argument should succeed.");
	CHECK_MESSAGE(
			parser.get_value(option) == expected_value,
			"The parsed adjacent argument value should match the expected.");

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, expected_value, option_name, expected_value)) != OK,
			"Parsing with two same options should fail.");
	option->set_multitoken(true);
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_value, option_name, expected_value)) == OK,
			"Parsing with two same multitoken options should succeed.");
	CHECK_MESSAGE(
			parser.get_occurence_count(option) == 2,
			"The parsed multitoken option occurence count should match the expected.");
	CHECK_MESSAGE(
			parser.get_values(option) == sarray(expected_value, expected_value),
			"The parsed multitoken argument values should match the expected.");

	CHECK_MESSAGE(
			parser.parse_args(sarray(expected_value)) != OK,
			"Parsing with a positional argument without any option marked as positional should fail.");
	option->set_positional(true);
	CHECK_MESSAGE(
			parser.parse_args(sarray(expected_value)) == OK,
			"Parsing with a positional argument with the option marked as positional should succeed.");
	CHECK_MESSAGE(
			parser.get_value(option) == expected_value,
			"The parsed positional argument value should match the expected.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<2>, "[CommandLineParser] Option that takes 2 arguments") {
	const PackedStringArray expected_values = sarray("arg1", "arg2");
	String option_name;
	CAPTURE(option_name);
	SUBCASE("[CommandLineParser] Short option") {
		option_name = short_option();

		const Ref<CommandLineOption> another_option = memnew(CommandLineOption(sarray("a"), 0));
		parser.add_option(another_option);
		const String another_option_name = parser.get_short_prefixes().get(0) + another_option->get_names().get(0);
		CAPTURE(another_option_name);

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0) + expected_values[0], expected_values[1])) == OK,
				"Parsing with two compound options should succeed.");
		CHECK_MESSAGE(
				parser.get_values(option) == expected_values,
				"The parsed compound option values should match the expected.");
		CHECK_MESSAGE(
				parser.is_set(another_option),
				"The parsed compound option should be set.");

		parser.set_allow_compound(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0) + expected_values[0], expected_values[1])) != OK,
				"Parsing with compound option should fail if not allowed.");

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(option_name + expected_values[0], expected_values[1])) == OK,
				"Parsing with sticky arguments should succeed.");
		CHECK_MESSAGE(
				parser.get_values(option) == expected_values,
				"The parsed sticky argument values should match the expected.");

		parser.set_allow_sticky(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(option_name + expected_values[0], expected_values[1])) != OK,
				"Parsing with sticky arguments should fail if not allowed.");
	}
	SUBCASE("[CommandLineParser] Long option") {
		option_name = long_option();
	}

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name)) != OK,
			"Parsing without arguments should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0])) != OK,
			"Parsing with less number of arguments should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[0], expected_values[1])) != OK,
			"Parsing with more number of arguments should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name + "=" + expected_values[0], expected_values[1])) != OK,
			"Parsing with an adjacent argument should fail.");
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[1])) == OK,
			"Parsing with two arguments should succeed.");
	CHECK_MESSAGE(
			parser.get_values(option) == expected_values,
			"The parsed argument value should match the expected.");

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[1], option_name, expected_values[0], expected_values[1])) != OK,
			"Parsing with two same options should fail.");
	option->set_multitoken(true);
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[1], option_name, expected_values[0], expected_values[1])) == OK,
			"Parsing with two same multitoken options should succeed.");
	CHECK_MESSAGE(
			parser.get_occurence_count(option) == 2,
			"The parsed multitoken option occurence count should match the expected.");
	CHECK_MESSAGE(
			parser.get_values(option) == sarray(expected_values[0], expected_values[1], expected_values[0], expected_values[1]),
			"The parsed multitoken argument values should match the expected.");

	CHECK_MESSAGE(
			parser.parse_args(expected_values) != OK,
			"Parsing with positional arguments without any option marked as positional should fail.");
	option->set_positional(true);
	REQUIRE_MESSAGE(
			parser.parse_args(expected_values) == OK,
			"Parsing with positional arguments should succeed.");
	CHECK_MESSAGE(
			parser.get_values(option) == expected_values,
			"The parsed positional argument values should match the expected.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<-1>, "[CommandLineParser] Option that takes all arguments left") {
	const PackedStringArray expected_values = sarray("arg1", "arg2");
	String option_name;
	CAPTURE(option_name);
	SUBCASE("[CommandLineParser] Short option") {
		option_name = short_option();

		const Ref<CommandLineOption> another_option = memnew(CommandLineOption(sarray("a"), 0));
		parser.add_option(another_option);
		const String another_option_name = parser.get_short_prefixes().get(0) + another_option->get_names().get(0);
		CAPTURE(another_option_name);

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0) + expected_values[0], expected_values[1])) == OK,
				"Parsing with two compound options should succeed.");
		CHECK_MESSAGE(
				parser.get_values(option) == expected_values,
				"The parsed compound option values should match the expected.");
		CHECK_MESSAGE(
				parser.is_set(another_option),
				"The parsed compound option should be set.");

		parser.set_allow_compound(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(another_option_name + option->get_names().get(0) + expected_values[0], expected_values[1])) != OK,
				"Parsing with compound option should fail if not allowed.");

		REQUIRE_MESSAGE(
				parser.parse_args(sarray(option_name + expected_values[0], expected_values[1])) == OK,
				"Parsing with sticky arguments should succeed.");
		CHECK_MESSAGE(
				parser.get_values(option) == expected_values,
				"The parsed sticky argument values should match the expected.");

		parser.set_allow_sticky(false);
		CHECK_MESSAGE(
				parser.parse_args(sarray(option_name + expected_values[0], expected_values[1])) != OK,
				"Parsing with sticky arguments should fail if not allowed.");
	}
	SUBCASE("[CommandLineParser] Long option") {
		option_name = long_option();
	}

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name)) != OK,
			"Parsing without arguments should fail.");
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0])) == OK,
			"Parsing with an argument should succeed.");
	CHECK_MESSAGE(
			parser.get_value(option) == expected_values[0],
			"The parsed argument value should match the expected.");
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name + "=" + expected_values[0])) == OK,
			"Parsing with an adjacent argument should succeed.");
	CHECK_MESSAGE(
			parser.get_value(option) == expected_values[0],
			"The parsed adjacent argument value should math the expected.");
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[1])) == OK,
			"Parsing with two arguments should succeed.");
	CHECK_MESSAGE(
			parser.get_values(option) == expected_values,
			"The parsed argument values should match the expected.");

	CHECK_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[1], option_name, expected_values[0], expected_values[1])) != OK,
			"Parsing with two same options should fail.");
	option->set_multitoken(true);
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(option_name, expected_values[0], expected_values[1], option_name, expected_values[0], expected_values[1])) == OK,
			"Parsing with two same multitoken options should succeed.");
	CHECK_MESSAGE(
			parser.get_occurence_count(option) == 2,
			"The parsed multitoken option occurence count should match the expected.");
	CHECK_MESSAGE(
			parser.get_values(option) == sarray(expected_values[0], expected_values[1], expected_values[0], expected_values[1]),
			"The parsed multitoken argument values should match the expected.");

	option->set_positional(true);
	REQUIRE_MESSAGE(
			parser.parse_args(sarray(expected_values[0])) == OK,
			"Parsing with a positional argument should succeed.");
	CHECK_MESSAGE(
			parser.get_value(option) == expected_values[0],
			"The parsed positional argument value should math the expected.");
	REQUIRE_MESSAGE(
			parser.parse_args(expected_values) == OK,
			"Parsing with positional arguments should succeed.");
	CHECK_MESSAGE(
			parser.get_values(option) == expected_values,
			"Parsed positional argument values should math the expected.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<1>, "[CommandLineParser] Argument checks") {
	class CheckerObject : public Object {
	public:
		bool check(const String &arg) {
			return arg == "true";
		}
	};

	option->set_static_checker([](const String &arg) { return arg == "true"; }, String());
	CHECK_MESSAGE(
			parser.parse_args(sarray(short_option(), "false")) != OK,
			"Parsing with argument that does not satisfy the checker should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(short_option(), "true")) == OK,
			"Parsing with argument that satisfies the checker should succeed.");

	CheckerObject checker_object;
	option->set_checker(callable_mp(&checker_object, &CheckerObject::check), "Specified argument is not equal to 'true'.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(short_option(), "false")) != OK,
			"Parsing with argument that does not satisfy the checker should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(short_option(), "true")) == OK,
			"Parsing with argument that satisfies the checker should succeed.");

	option->set_allowed_args(sarray("true", "false"));
	CHECK_MESSAGE(
			parser.parse_args(sarray(short_option(), "fail")) != OK,
			"Parsing with argument that does not satisfy the allowed values should fail.");
	CHECK_MESSAGE(
			parser.parse_args(sarray(short_option(), "true")) == OK,
			"Parsing with argument that satisfies the allowed values should succeed.");
}

TEST_CASE("[CommandLineParser] Multiply options") {
	CommandLineParser parser;
	Ref<CommandLineOption> option1 = memnew(CommandLineOption(sarray("o", "one")));
	parser.add_option(option1);
	Ref<CommandLineOption> option2 = memnew(CommandLineOption(sarray("t", "two"), 2));
	parser.add_option(option2);
	Ref<CommandLineOption> option3 = memnew(CommandLineOption(sarray("a", "all"), -1));
	parser.add_option(option3);
	const String expected_arg = "arg1";
	const PackedStringArray expected_args2 = sarray("arg3", "arg4");
	const PackedStringArray expected_args3 = sarray("arg5", "arg6");
	CAPTURE(expected_arg);

	SUBCASE("[CommandLineParser] Short options") {
		const String short_option1 = parser.get_short_prefixes().get(0) + option1->get_names().get(0);
		const String short_option2 = parser.get_short_prefixes().get(0) + option2->get_names().get(0);
		const String short_option3 = parser.get_short_prefixes().get(0) + option3->get_names().get(0);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(short_option1, expected_arg, short_option2, expected_args2[0], expected_args2[1], short_option3, expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Long options") {
		const String long_option1 = parser.get_long_prefixes().get(0) + option1->get_names().get(1);
		const String long_option2 = parser.get_long_prefixes().get(0) + option2->get_names().get(1);
		const String long_option3 = parser.get_long_prefixes().get(0) + option3->get_names().get(1);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(long_option1, expected_arg, long_option2, expected_args2[0], expected_args2[1], long_option3, expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Short options with positional") {
		option2->set_positional(true);
		const String short_option1 = parser.get_short_prefixes().get(0) + option1->get_names().get(0);
		const String short_option3 = parser.get_short_prefixes().get(0) + option3->get_names().get(0);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(short_option1, expected_arg, expected_args2[0], expected_args2[1], short_option3, expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Long options with positional") {
		option2->set_positional(true);
		const String long_option1 = parser.get_long_prefixes().get(0) + option1->get_names().get(1);
		const String long_option3 = parser.get_long_prefixes().get(0) + option3->get_names().get(1);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(long_option1, expected_arg, expected_args2[0], expected_args2[1], long_option3, expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Short options with option first positional") {
		option1->set_positional(true);
		const String short_option2 = parser.get_short_prefixes().get(0) + option2->get_names().get(0);
		const String short_option3 = parser.get_short_prefixes().get(0) + option3->get_names().get(0);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(expected_arg, short_option2, expected_args2[0], expected_args2[1], short_option3, expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Long options with option first positional") {
		option1->set_positional(true);
		const String long_option2 = parser.get_long_prefixes().get(0) + option2->get_names().get(1);
		const String long_option3 = parser.get_long_prefixes().get(0) + option3->get_names().get(1);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(expected_arg, long_option2, expected_args2[0], expected_args2[1], long_option3, expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Short options with option last positional") {
		option3->set_positional(true);
		const String short_option1 = parser.get_short_prefixes().get(0) + option1->get_names().get(0);
		const String short_option2 = parser.get_short_prefixes().get(0) + option2->get_names().get(0);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(short_option1, expected_arg, short_option2, expected_args2[0], expected_args2[1], expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] Long options with option last positional") {
		option3->set_positional(true);
		const String long_option1 = parser.get_long_prefixes().get(0) + option1->get_names().get(1);
		const String long_option2 = parser.get_long_prefixes().get(0) + option2->get_names().get(1);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(long_option1, expected_arg, long_option2, expected_args2[0], expected_args2[1], expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}
	SUBCASE("[CommandLineParser] All positional") {
		option1->set_positional(true);
		option2->set_positional(true);
		option3->set_positional(true);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(expected_arg, expected_args2[0], expected_args2[1], expected_args3[0], expected_args3[1])) == OK,
				"Parsing should succeed.");
	}

	CHECK_MESSAGE(
			parser.get_value(option1) == expected_arg,
			"The parsed value should match the expected.");
	CHECK_MESSAGE(
			parser.get_values(option2) == expected_args2,
			"The parsed value should match the expected.");
	CHECK_MESSAGE(
			parser.get_values(option3) == expected_args3,
			"The parsed value should match the expected.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<0>, "[CommandLineParser] Required arguments") {
	option->set_required(true);
	const Ref<CommandLineOption> another_option = memnew(CommandLineOption(sarray("a"), 0));
	parser.add_option(another_option);
	const String another_option_name = parser.get_short_prefixes().get(0) + another_option->get_names().get(0);
	CAPTURE(another_option_name);

	REQUIRE_MESSAGE(
			parser.parse_args(sarray(another_option_name)) == OK,
			"Parsing empty arguments should succeed.");
	CHECK_MESSAGE(
			parser.validate() != OK,
			"Calling validate() should fail.");
	CHECK_MESSAGE(
			parser.is_set(another_option),
			"Another option should be set");
	CHECK_FALSE_MESSAGE(
			parser.is_set(option),
			"The required option shouldn't be set");

	REQUIRE_MESSAGE(
			parser.parse_args(sarray(short_option())) == OK,
			"Parsing with the required option should succeed.");
	CHECK_MESSAGE(
			parser.validate() == OK,
			"Calling validate() should succeed.");
	CHECK_FALSE_MESSAGE(
			parser.is_set(another_option),
			"Another option shouldn't be set");
	CHECK_MESSAGE(
			parser.is_set(option),
			"The required option should be set");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<0>, "[CommandLineParser] Prefixes") {
	String prefix;

	SUBCASE("[CommandLineParser] Short option with default prefix") {
		prefix = parser.get_short_prefixes().get(0);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(short_option())) == OK,
				"Parsing short option should succeed.");
	}
	SUBCASE("[CommandLineParser] Long option with default prefix") {
		prefix = parser.get_long_prefixes().get(0);
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(long_option())) == OK,
				"Parsing long option should succeed.");
	}
	SUBCASE("[CommandLineParser] Short option with Windows-like prefix") {
		ERR_PRINT_OFF;
		parser.set_short_prefixes(PackedStringArray());
		CHECK_MESSAGE(
				parser.parse_args(PackedStringArray()) != OK,
				"Parsing without short prefixes specified should fail.");

		ERR_PRINT_ON;
		prefix = "/";
		parser.set_short_prefixes(sarray(prefix));
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(short_option())) == OK,
				"Parsing short option should succeed.");
	}
	SUBCASE("[CommandLineParser] Long option with Windows-like prefix") {
		ERR_PRINT_OFF;
		parser.set_long_prefixes(PackedStringArray());
		CHECK_MESSAGE(
				parser.parse_args(PackedStringArray()) != OK,
				"Parsing without long prefixes specified should fail.");

		ERR_PRINT_ON;
		prefix = "/";
		parser.set_long_prefixes(sarray(prefix));
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(long_option())) == OK,
				"Parsing long option should succeed.");
	}

	CHECK_MESSAGE(
			parser.is_set(option),
			"The parsed option should be set.");
	CHECK_MESSAGE(
			parser.get_prefix(option) == prefix,
			"The parsed option prefix should match the expected.");
}

TEST_CASE_FIXTURE(ParserWithOptionFixture<1>, "[CommandLineParser] Validated signal") {
	struct SignalWatcher : public Object {
		PackedStringArray expected_values;
		bool func_called = false;

		void check_validated_value(const PackedStringArray &values) {
			func_called = true;
			CHECK_MESSAGE(
					values == expected_values,
					"The validated value from signal should match the expected.");
		}
	};

	SignalWatcher watcher;
	option->connect("validated", callable_mp(&watcher, &SignalWatcher::check_validated_value));
	const String expected_value = "arg";
	watcher.expected_values = sarray(expected_value);
	CAPTURE(expected_value);

	SUBCASE("[CommandLineParser] Short option") {
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(short_option(), expected_value)) == OK,
				"Parsing short option should succeed.");
	}
	SUBCASE("[CommandLineParser] Long option") {
		REQUIRE_MESSAGE(
				parser.parse_args(sarray(long_option(), expected_value)) == OK,
				"Parsing long option should succeed.");
	}

	REQUIRE_MESSAGE(
			parser.validate() == OK,
			"Calling validate() should succeed.");

	CHECK_MESSAGE(
			watcher.func_called,
			"Function check_validated_value() should be called.");
}

TEST_CASE("[CommandLineParser] Help message") {
	CommandLineParser parser;
	Ref<CommandLineHelpFormat> format = memnew(CommandLineHelpFormat);
	format->set_header("Test header");
	format->set_footer("Test footer");
	format->set_usage_title("godot");

	Ref<CommandLineOption> option1 = memnew(CommandLineOption(sarray("o", "one")));
	option1->set_description("Not a long description.");
	option1->set_arg_text("<argument>");
	option1->set_required(true);
	option1->set_positional(true);
	parser.add_option(option1);

	Ref<CommandLineOption> option2 = memnew(CommandLineOption(sarray("v", "very-very-very-very-very-vey-long-option"), 0));
	option2->set_description("Very very very very very very very long description that should be splitted into a several lines.");
	parser.add_option(option2);

	Ref<CommandLineOption> option3 = memnew(CommandLineOption(sarray("d", "description")));
	option3->set_description("Veryveryveryveryveryveryverylongdescriptionwordthatshouldbesplittedtoo.");
	option3->set_required(true);
	parser.add_option(option3);

	Ref<CommandLineOption> hidden_option = memnew(CommandLineOption(sarray("h", "hidden")));
	hidden_option->set_description("Should not be visible.");
	hidden_option->set_hidden(true);
	parser.add_option(hidden_option);

	// Use translatable strings as in class to avoid test failures in different locales.
	const String expected_usage = vformat(RTR("Usage: %s"), format->get_usage_title()) + " " + RTR("[options]") + " [--one] <argument> --description <arg>\n";
	const String expected_options_description = R"(
  -o, --one <argument>                  Not a long description.
  -v, --very-very-very-very-very-vey-long-option
                                        Very very very very very very very long
                                        description that should be splitted into
                                        a several lines.
  -d, --description <arg>               Veryveryveryveryveryveryverylongdescript
                                        ionwordthatshouldbesplittedtoo.
)";

	CHECK_MESSAGE(
			parser.get_help_text(format) == format->get_header() + "\n" + expected_usage + expected_options_description + format->get_footer(),
			"Help message shold match the expected.");

	format->set_autogenerate_usage(false);
	CHECK_MESSAGE(
			parser.get_help_text(format) == format->get_header() + "\n" + expected_options_description + format->get_footer(),
			"Help message without usage shold match the expected.");
}
} // namespace TestCommandLineParser

#endif // TEST_COMMAND_LINE_PARSER_H
