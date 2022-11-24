/*************************************************************************/
/*  test_string.cpp                                                      */
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

#include "test_string.h"

#include "core/io/ip_address.h"
#include "core/os/os.h"
#include "core/ustring.h"

#include "modules/modules_enabled.gen.h" // For regex.
#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif

#include <stdio.h>
#include <wchar.h>

namespace TestString {

bool test_1() {
	OS::get_singleton()->print("\n\nTest 1: Assign from cstr\n");

	String s = "Hello";

	OS::get_singleton()->print("\tExpected: Hello\n");
	OS::get_singleton()->print("\tResulted: %ls\n", s.c_str());

	return (wcscmp(s.c_str(), L"Hello") == 0);
}

bool test_2() {
	OS::get_singleton()->print("\n\nTest 2: Assign from string (operator=)\n");

	String s = "Dolly";
	const String &t = s;

	OS::get_singleton()->print("\tExpected: Dolly\n");
	OS::get_singleton()->print("\tResulted: %ls\n", t.c_str());

	return (wcscmp(t.c_str(), L"Dolly") == 0);
}

bool test_3() {
	OS::get_singleton()->print("\n\nTest 3: Assign from c-string (copycon)\n");

	String s("Sheep");
	const String &t(s);

	OS::get_singleton()->print("\tExpected: Sheep\n");
	OS::get_singleton()->print("\tResulted: %ls\n", t.c_str());

	return (wcscmp(t.c_str(), L"Sheep") == 0);
}

bool test_4() {
	OS::get_singleton()->print("\n\nTest 4: Assign from c-widechar (operator=)\n");

	String s(L"Give me");

	OS::get_singleton()->print("\tExpected: Give me\n");
	OS::get_singleton()->print("\tResulted: %ls\n", s.c_str());

	return (wcscmp(s.c_str(), L"Give me") == 0);
}

bool test_5() {
	OS::get_singleton()->print("\n\nTest 5: Assign from c-widechar (copycon)\n");

	String s(L"Wool");

	OS::get_singleton()->print("\tExpected: Wool\n");
	OS::get_singleton()->print("\tResulted: %ls\n", s.c_str());

	return (wcscmp(s.c_str(), L"Wool") == 0);
}

bool test_6() {
	OS::get_singleton()->print("\n\nTest 6: comparisons (equal)\n");

	String s = "Test Compare";

	OS::get_singleton()->print("\tComparing to \"Test Compare\"\n");

	if (!(s == "Test Compare")) {
		return false;
	}

	if (!(s == L"Test Compare")) {
		return false;
	}

	if (!(s == String("Test Compare"))) {
		return false;
	}

	return true;
}

bool test_7() {
	OS::get_singleton()->print("\n\nTest 7: comparisons (unequal)\n");

	String s = "Test Compare";

	OS::get_singleton()->print("\tComparing to \"Test Compare\"\n");

	if (!(s != "Peanut")) {
		return false;
	}

	if (!(s != L"Coconut")) {
		return false;
	}

	if (!(s != String("Butter"))) {
		return false;
	}

	return true;
}

bool test_8() {
	OS::get_singleton()->print("\n\nTest 8: comparisons (operator<)\n");

	String s = "Bees";

	OS::get_singleton()->print("\tComparing to \"Bees\"\n");

	if (!(s < "Elephant")) {
		return false;
	}

	if (s < L"Amber") {
		return false;
	}

	if (s < String("Beatrix")) {
		return false;
	}

	return true;
}

bool test_9() {
	OS::get_singleton()->print("\n\nTest 9: Concatenation\n");

	String s;

	s += "Have";
	s += ' ';
	s += 'a';
	s += String(" ");
	s = s + L"Nice";
	s = s + " ";
	s = s + String("Day");

	OS::get_singleton()->print("\tComparing to \"Have a Nice Day\"\n");

	return (s == "Have a Nice Day");
}

bool test_10() {
	OS::get_singleton()->print("\n\nTest 10: Misc funcs (size/length/empty/etc)\n");

	if (!String("").empty()) {
		return false;
	}

	if (String("Mellon").size() != 7) {
		return false;
	}

	if (String("Oranges").length() != 7) {
		return false;
	}

	return true;
}

bool test_11() {
	OS::get_singleton()->print("\n\nTest 11: Operator[]\n");

	String a = "Kugar Sane";

	a[0] = 'S';
	a[6] = 'C';

	if (a != "Sugar Cane") {
		return false;
	}

	if (a[1] != 'u') {
		return false;
	}

	return true;
}

bool test_12() {
	OS::get_singleton()->print("\n\nTest 12: case functions\n");

	String a = "MoMoNgA";

	if (a.to_upper() != "MOMONGA") {
		return false;
	}

	if (a.nocasecmp_to("momonga") != 0) {
		return false;
	}

	return true;
}

bool test_13() {
	OS::get_singleton()->print("\n\nTest 13: UTF8\n");

	/* how can i embed UTF in here? */

	static const CharType ustr[] = { 0x304A, 0x360F, 0x3088, 0x3046, 0 };
	//static const wchar_t ustr[] = { 'P', 0xCE, 'p',0xD3, 0 };
	String s = ustr;

	OS::get_singleton()->print("\tUnicode: %ls\n", ustr);
	s.parse_utf8(s.utf8().get_data());
	OS::get_singleton()->print("\tConvert/Parse UTF8: %ls\n", s.c_str());

	return (s == ustr);
}

bool test_14() {
	OS::get_singleton()->print("\n\nTest 14: ASCII\n");

	String s = L"Primero Leche";
	OS::get_singleton()->print("\tAscii: %s\n", s.ascii().get_data());

	String t = s.ascii().get_data();
	return (s == t);
}

bool test_15() {
	OS::get_singleton()->print("\n\nTest 15: substr\n");

	String s = "Killer Baby";
	OS::get_singleton()->print("\tsubstr(3,4) of \"%ls\" is \"%ls\"\n", s.c_str(), s.substr(3, 4).c_str());

	return (s.substr(3, 4) == "ler ");
}

bool test_16() {
	OS::get_singleton()->print("\n\nTest 16: find\n");

	String s = "Pretty Woman";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\t\"tty\" is at %i pos.\n", s.find("tty"));
	OS::get_singleton()->print("\t\"Revenge of the Monster Truck\" is at %i pos.\n", s.find("Revenge of the Monster Truck"));

	if (s.find("tty") != 3) {
		return false;
	}

	if (s.find("Revenge of the Monster Truck") != -1) {
		return false;
	}

	return true;
}

bool test_17() {
	OS::get_singleton()->print("\n\nTest 17: find no case\n");

	String s = "Pretty Whale";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\t\"WHA\" is at %i pos.\n", s.findn("WHA"));
	OS::get_singleton()->print("\t\"Revenge of the Monster SawFish\" is at %i pos.\n", s.findn("Revenge of the Monster Truck"));

	if (s.findn("WHA") != 7) {
		return false;
	}

	if (s.findn("Revenge of the Monster SawFish") != -1) {
		return false;
	}

	return true;
}

bool test_18() {
	OS::get_singleton()->print("\n\nTest 18: find no case\n");

	String s = "Pretty Whale";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\t\"WHA\" is at %i pos.\n", s.findn("WHA"));
	OS::get_singleton()->print("\t\"Revenge of the Monster SawFish\" is at %i pos.\n", s.findn("Revenge of the Monster Truck"));

	if (s.findn("WHA") != 7) {
		return false;
	}

	if (s.findn("Revenge of the Monster SawFish") != -1) {
		return false;
	}

	return true;
}

bool test_19() {
	OS::get_singleton()->print("\n\nTest 19: Search & replace\n");

	String s = "Happy Birthday, Anna!";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());

	s = s.replace("Birthday", "Halloween");
	OS::get_singleton()->print("\tReplaced Birthday/Halloween: %ls.\n", s.c_str());

	return (s == "Happy Halloween, Anna!");
}

bool test_20() {
	OS::get_singleton()->print("\n\nTest 20: Insertion\n");

	String s = "Who is Frederic?";

	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	s = s.insert(s.find("?"), " Chopin");
	OS::get_singleton()->print("\tInserted Chopin: %ls.\n", s.c_str());

	return (s == "Who is Frederic Chopin?");
}

bool test_21() {
	OS::get_singleton()->print("\n\nTest 21: Number -> String\n");

	OS::get_singleton()->print("\tPi is %f\n", 33.141593);
	OS::get_singleton()->print("\tPi String is %ls\n", String::num(3.141593).c_str());

	return String::num(3.141593) == "3.141593";
}

bool test_22() {
	OS::get_singleton()->print("\n\nTest 22: String -> Int\n");

	static const char *nums[4] = { "1237461283", "- 22", "0", " - 1123412" };
	static const int num[4] = { 1237461283, -22, 0, -1123412 };

	for (int i = 0; i < 4; i++) {
		OS::get_singleton()->print("\tString: \"%s\" as Int is %i\n", nums[i], String(nums[i]).to_int());

		if (String(nums[i]).to_int() != num[i]) {
			return false;
		}
	}

	return true;
}

bool test_23() {
	OS::get_singleton()->print("\n\nTest 23: String -> Float\n");

	static const char *nums[4] = { "-12348298412.2", "0.05", "2.0002", " -0.0001" };
	static const double num[4] = { -12348298412.2, 0.05, 2.0002, -0.0001 };

	for (int i = 0; i < 4; i++) {
		OS::get_singleton()->print("\tString: \"%s\" as Float is %f\n", nums[i], String(nums[i]).to_double());

		if (ABS(String(nums[i]).to_double() - num[i]) > 0.00001) {
			return false;
		}
	}

	return true;
}

bool test_24() {
	OS::get_singleton()->print("\n\nTest 24: Slicing\n");

	String s = "Mars,Jupiter,Saturn,Uranus";

	const char *slices[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };

	OS::get_singleton()->print("\tSlicing \"%ls\" by \"%s\"..\n", s.c_str(), ",");

	for (int i = 0; i < s.get_slice_count(","); i++) {
		OS::get_singleton()->print("\t\t%i- %ls\n", i + 1, s.get_slice(",", i).c_str());

		if (s.get_slice(",", i) != slices[i]) {
			return false;
		}
	}

	return true;
}

bool test_25() {
	OS::get_singleton()->print("\n\nTest 25: Erasing\n");

	String s = "Josephine is such a cute girl!";

	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\tRemoving \"cute\"\n");

	s.erase(s.find("cute "), String("cute ").length());
	OS::get_singleton()->print("\tResult: %ls\n", s.c_str());

	return (s == "Josephine is such a girl!");
}

bool test_26() {
	OS::get_singleton()->print("\n\nTest 26: RegEx substitution\n");

#ifndef MODULE_REGEX_ENABLED
	OS::get_singleton()->print("\tRegEx module disabled, can't run test.");
	return false;
#else
	String s = "Double all the vowels.";

	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\tRepeating instances of 'aeiou' once\n");

	RegEx re("(?<vowel>[aeiou])");
	s = re.sub(s, "$0$vowel", true);

	OS::get_singleton()->print("\tResult: %ls\n", s.c_str());

	return (s == "Doouublee aall thee vooweels.");
#endif
}

struct test_27_data {
	char const *data;
	char const *begin;
	bool expected;
};

bool test_27() {
	OS::get_singleton()->print("\n\nTest 27: begins_with\n");
	test_27_data tc[] = {
		{ "res://foobar", "res://", true },
		{ "res", "res://", false },
		{ "abc", "abc", true }
	};
	size_t count = sizeof(tc) / sizeof(tc[0]);
	bool state = true;
	for (size_t i = 0; state && i < count; ++i) {
		String s = tc[i].data;
		state = s.begins_with(tc[i].begin) == tc[i].expected;
		if (state) {
			String sb = tc[i].begin;
			state = s.begins_with(sb) == tc[i].expected;
		}
		if (!state) {
			OS::get_singleton()->print("\n\t Failure on:\n\t\tstring: %s\n\t\tbegin: %s\n\t\texpected: %s\n", tc[i].data, tc[i].begin, tc[i].expected ? "true" : "false");
			break;
		}
	};
	return state;
};

bool test_28() {
	OS::get_singleton()->print("\n\nTest 28: sprintf\n");

	bool success, state = true;
	char output_format[] = "\tTest:\t%ls => %ls (%s)\n";
	String format, output;
	Array args;
	bool error;

	// %%
	format = "fish %% frog";
	args.clear();
	output = format.sprintf(args, &error);
	success = (output == String("fish % frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	//////// INTS

	// Int
	format = "fish %d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish 5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int left padded with zeroes.
	format = "fish %05d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish 00005 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int left padded with spaces.
	format = "fish %5d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish     5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int right padded with spaces.
	format = "fish %-5d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish 5     frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int with sign (positive).
	format = "fish %+d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish +5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative int.
	format = "fish %d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	success = (output == String("fish -5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative int left padded with spaces.
	format = "fish %5d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	success = (output == String("fish    -5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative int left padded with zeros.
	format = "fish %05d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	success = (output == String("fish -0005 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative int right padded with spaces.
	format = "fish %-5d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	success = (output == String("fish -5    frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative int right padded with zeros. (0 ignored)
	format = "fish %-05d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	success = (output == String("fish -5    frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Hex (lower)
	format = "fish %x frog";
	args.clear();
	args.push_back(45);
	output = format.sprintf(args, &error);
	success = (output == String("fish 2d frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Hex (upper)
	format = "fish %X frog";
	args.clear();
	args.push_back(45);
	output = format.sprintf(args, &error);
	success = (output == String("fish 2D frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Octal
	format = "fish %o frog";
	args.clear();
	args.push_back(99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 143 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	////// REALS

	// Real
	format = "fish %f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.990000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real left-padded
	format = "fish %11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish   99.990000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real (infinity) left-padded
	format = "fish %11f frog";
	args.clear();
	args.push_back(INFINITY);
	output = format.sprintf(args, &error);
	success = (output == String("fish         inf frog") && !error);
	state = state && success;

	// Real right-padded
	format = "fish %-11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.990000   frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real given int.
	format = "fish %f frog";
	args.clear();
	args.push_back(99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.000000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with sign (positive).
	format = "fish %+f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish +99.990000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with 1 decimals.
	format = "fish %.1f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 100.0 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with 12 decimals.
	format = "fish %.12f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.990000000000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with no decimals.
	format = "fish %.f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 100 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative real right padded with zeros. (0 ignored)
	format = "fish %-011f frog";
	args.clear();
	args.push_back(-99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish -99.990000  frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	/////// Strings.

	// String
	format = "fish %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish cheese frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// String left-padded
	format = "fish %10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish     cheese frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// String right-padded
	format = "fish %-10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish cheese     frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	///// Characters

	// Character as string.
	format = "fish %c frog";
	args.clear();
	args.push_back("A");
	output = format.sprintf(args, &error);
	success = (output == String("fish A frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Character as int.
	format = "fish %c frog";
	args.clear();
	args.push_back(65);
	output = format.sprintf(args, &error);
	success = (output == String("fish A frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	///// Dynamic width

	// String dynamic width
	format = "fish %*s frog";
	args.clear();
	args.push_back(10);
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish     cheese frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int dynamic width
	format = "fish %*d frog";
	args.clear();
	args.push_back(10);
	args.push_back(99);
	output = format.sprintf(args, &error);
	success = (output == String("fish         99 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Float dynamic width
	format = "fish %*.*f frog";
	args.clear();
	args.push_back(10);
	args.push_back(3);
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish     99.990 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	///// Errors

	// More formats than arguments.
	format = "fish %s %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "not enough arguments for format string" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// More arguments than formats.
	format = "fish %s frog";
	args.clear();
	args.push_back("hello");
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "not all arguments converted during string formatting" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Incomplete format.
	format = "fish %10";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "incomplete format" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Bad character in format string
	format = "fish %&f frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "unsupported format character" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Too many decimals.
	format = "fish %2.2.2f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == "too many decimal points in format" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// * not a number
	format = "fish %*f frog";
	args.clear();
	args.push_back("cheese");
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == "* wants number" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Character too long.
	format = "fish %c frog";
	args.clear();
	args.push_back("sc");
	output = format.sprintf(args, &error);
	success = (output == "%c requires number or single-character string" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Character bad type.
	format = "fish %c frog";
	args.clear();
	args.push_back(Array());
	output = format.sprintf(args, &error);
	success = (output == "%c requires number or single-character string" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	return state;
}

bool test_29() {
	bool state = true;

	IP_Address ip0("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
	OS::get_singleton()->print("ip0 is %ls\n", String(ip0).c_str());

	IP_Address ip(0x0123, 0x4567, 0x89ab, 0xcdef, true);
	OS::get_singleton()->print("ip6 is %ls\n", String(ip).c_str());

	IP_Address ip2("fe80::52e5:49ff:fe93:1baf");
	OS::get_singleton()->print("ip6 is %ls\n", String(ip2).c_str());

	IP_Address ip3("::ffff:192.168.0.1");
	OS::get_singleton()->print("ip6 is %ls\n", String(ip3).c_str());

	String ip4 = "192.168.0.1";
	bool success = ip4.is_valid_ip_address();
	OS::get_singleton()->print("Is valid ipv4: %ls, %s\n", ip4.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip4 = "192.368.0.1";
	success = (!ip4.is_valid_ip_address());
	OS::get_singleton()->print("Is invalid ipv4: %ls, %s\n", ip4.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	String ip6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334";
	success = ip6.is_valid_ip_address();
	OS::get_singleton()->print("Is valid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "2001:0db8:85j3:0000:0000:8a2e:0370:7334";
	success = (!ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is invalid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "2001:0db8:85f345:0000:0000:8a2e:0370:7334";
	success = (!ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is invalid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "2001:0db8::0:8a2e:370:7334";
	success = (ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is valid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "::ffff:192.168.0.1";
	success = (ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is valid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	return state;
};

bool test_30() {
	bool state = true;
	bool success = true;
	String input = "bytes2var";
	String output = "Bytes 2 Var";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "linear2db";
	output = "Linear 2 Db";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "vector3";
	output = "Vector 3";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "sha256";
	output = "Sha 256";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "2db";
	output = "2 Db";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "PascalCase";
	output = "Pascal Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "PascalPascalCase";
	output = "Pascal Pascal Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_case";
	output = "Snake Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_snake_case";
	output = "Snake Snake Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "sha256sum";
	output = "Sha 256 Sum";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "cat2dog";
	output = "Cat 2 Dog";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "function(name)";
	output = "Function(name)";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls (existing incorrect behavior): %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_case_function(snake_case_arg)";
	output = "Snake Case Function(snake Case Arg)";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls (existing incorrect behavior): %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_case_function( snake_case_arg )";
	output = "Snake Case Function( Snake Case Arg )";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	return state;
}

bool test_31() {
	bool state = true;
	bool success;

	String a = "";
	success = a[0] == 0;
	OS::get_singleton()->print("Is 0 String[0]:, %s\n", success ? "OK" : "FAIL");
	if (!success) {
		state = false;
	}

	String b = "Godot";
	success = b[b.size()] == 0;
	OS::get_singleton()->print("Is 0 String[size()]:, %s\n", success ? "OK" : "FAIL");
	if (!success) {
		state = false;
	}

	const String c = "";
	success = c[0] == 0;
	OS::get_singleton()->print("Is 0 const String[0]:, %s\n", success ? "OK" : "FAIL");
	if (!success) {
		state = false;
	}

	const String d = "Godot";
	success = d[d.size()] == 0;
	OS::get_singleton()->print("Is 0 const String[size()]:, %s\n", success ? "OK" : "FAIL");
	if (!success) {
		state = false;
	}

	return state;
};

bool test_32() {
#define STRIP_TEST(x)                                            \
	{                                                            \
		bool success = x;                                        \
		state = state && success;                                \
		if (!success) {                                          \
			OS::get_singleton()->print("\tfailed at: %s\n", #x); \
		}                                                        \
	}

	OS::get_singleton()->print("\n\nTest 32: lstrip and rstrip\n");
	bool state = true;

	// strip none
	STRIP_TEST(String("abc").lstrip("") == "abc");
	STRIP_TEST(String("abc").rstrip("") == "abc");
	// strip one
	STRIP_TEST(String("abc").lstrip("a") == "bc");
	STRIP_TEST(String("abc").rstrip("c") == "ab");
	// strip lots
	STRIP_TEST(String("bababbababccc").lstrip("ab") == "ccc");
	STRIP_TEST(String("aaabcbcbcbbcbbc").rstrip("cb") == "aaa");
	// strip empty string
	STRIP_TEST(String("").lstrip("") == "");
	STRIP_TEST(String("").rstrip("") == "");
	// strip to empty string
	STRIP_TEST(String("abcabcabc").lstrip("bca") == "");
	STRIP_TEST(String("abcabcabc").rstrip("bca") == "");
	// don't strip wrong end
	STRIP_TEST(String("abc").lstrip("c") == "abc");
	STRIP_TEST(String("abca").lstrip("a") == "bca");
	STRIP_TEST(String("abc").rstrip("a") == "abc");
	STRIP_TEST(String("abca").rstrip("a") == "abc");
	// in utf-8 "¿" (\u00bf) has the same first byte as "µ" (\u00b5)
	// and the same second as "ÿ" (\u00ff)
	STRIP_TEST(String::utf8("¿").lstrip(String::utf8("µÿ")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("¿").rstrip(String::utf8("µÿ")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("µ¿ÿ").lstrip(String::utf8("µÿ")) == String::utf8("¿ÿ"));
	STRIP_TEST(String::utf8("µ¿ÿ").rstrip(String::utf8("µÿ")) == String::utf8("µ¿"));

	// the above tests repeated with additional superfluous strip chars

	// strip none
	STRIP_TEST(String("abc").lstrip("qwjkl") == "abc");
	STRIP_TEST(String("abc").rstrip("qwjkl") == "abc");
	// strip one
	STRIP_TEST(String("abc").lstrip("qwajkl") == "bc");
	STRIP_TEST(String("abc").rstrip("qwcjkl") == "ab");
	// strip lots
	STRIP_TEST(String("bababbababccc").lstrip("qwabjkl") == "ccc");
	STRIP_TEST(String("aaabcbcbcbbcbbc").rstrip("qwcbjkl") == "aaa");
	// strip empty string
	STRIP_TEST(String("").lstrip("qwjkl") == "");
	STRIP_TEST(String("").rstrip("qwjkl") == "");
	// strip to empty string
	STRIP_TEST(String("abcabcabc").lstrip("qwbcajkl") == "");
	STRIP_TEST(String("abcabcabc").rstrip("qwbcajkl") == "");
	// don't strip wrong end
	STRIP_TEST(String("abc").lstrip("qwcjkl") == "abc");
	STRIP_TEST(String("abca").lstrip("qwajkl") == "bca");
	STRIP_TEST(String("abc").rstrip("qwajkl") == "abc");
	STRIP_TEST(String("abca").rstrip("qwajkl") == "abc");
	// in utf-8 "¿" (\u00bf) has the same first byte as "µ" (\u00b5)
	// and the same second as "ÿ" (\u00ff)
	STRIP_TEST(String::utf8("¿").lstrip(String::utf8("qwaµÿjkl")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("¿").rstrip(String::utf8("qwaµÿjkl")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("µ¿ÿ").lstrip(String::utf8("qwaµÿjkl")) == String::utf8("¿ÿ"));
	STRIP_TEST(String::utf8("µ¿ÿ").rstrip(String::utf8("qwaµÿjkl")) == String::utf8("µ¿"));

	return state;

#undef STRIP_TEST
}

bool test_33() {
	OS::get_singleton()->print("\n\nTest 33: parse_utf8(null, -1)\n");

	String empty;
	return empty.parse_utf8(nullptr, -1);
}

bool test_34() {
	OS::get_singleton()->print("\n\nTest 34: Cyrillic to_lower()\n");

	String upper = String::utf8("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ");
	String lower = String::utf8("абвгдеёжзийклмнопрстуфхцчшщъыьэюя");

	String test = upper.to_lower();

	bool state = test == lower;

	return state;
}

bool test_35() {
#define COUNT_TEST(x)                                            \
	{                                                            \
		bool success = x;                                        \
		state = state && success;                                \
		if (!success) {                                          \
			OS::get_singleton()->print("\tfailed at: %s\n", #x); \
		}                                                        \
	}

	OS::get_singleton()->print("\n\nTest 35: count and countn function\n");
	bool state = true;

	COUNT_TEST(String("").count("Test") == 0);
	COUNT_TEST(String("Test").count("") == 0);
	COUNT_TEST(String("Test").count("test") == 0);
	COUNT_TEST(String("Test").count("TEST") == 0);
	COUNT_TEST(String("TEST").count("TEST") == 1);
	COUNT_TEST(String("Test").count("Test") == 1);
	COUNT_TEST(String("aTest").count("Test") == 1);
	COUNT_TEST(String("Testa").count("Test") == 1);
	COUNT_TEST(String("TestTestTest").count("Test") == 3);
	COUNT_TEST(String("TestTestTest").count("TestTest") == 1);
	COUNT_TEST(String("TestGodotTestGodotTestGodot").count("Test") == 3);

	COUNT_TEST(String("TestTestTestTest").count("Test", 4, 8) == 1);
	COUNT_TEST(String("TestTestTestTest").count("Test", 4, 12) == 2);
	COUNT_TEST(String("TestTestTestTest").count("Test", 4, 16) == 3);
	COUNT_TEST(String("TestTestTestTest").count("Test", 4) == 3);

	COUNT_TEST(String("Test").countn("test") == 1);
	COUNT_TEST(String("Test").countn("TEST") == 1);
	COUNT_TEST(String("testTest-Testatest").countn("tEst") == 4);
	COUNT_TEST(String("testTest-TeStatest").countn("tEsT", 4, 16) == 2);

	return state;
}

bool test_36() {
#define CHECK(X)                                          \
	if (!(X)) {                                           \
		OS::get_singleton()->print("\tFAIL at %s\n", #X); \
		return false;                                     \
	} else {                                              \
		OS::get_singleton()->print("\tPASS\n");           \
	}
	OS::get_singleton()->print("\n\nTest 36: xml unescape\n");
	// Named entities
	String input = "&quot;&amp;&apos;&lt;&gt;";
	CHECK(input.xml_unescape() == "\"&\'<>");

	// Numeric entities
	input = "&#x41;&#66;";
	CHECK(input.xml_unescape() == "AB");

	input = "&#0;&x#0;More text";
	String result = input.xml_unescape();
	// Didn't put in a leading NUL and terminate the string
	CHECK(input.length() > 0);
	CHECK(input[0] != '\0');
	// Entity should be left as-is if invalid
	CHECK(input.xml_unescape() == input);

	// Shouldn't consume without ending in a ';'
	input = "&#66";
	CHECK(input.xml_unescape() == input);
	input = "&#x41";
	CHECK(input.xml_unescape() == input);

	// Invalid characters should make the entity ignored
	input = "&#x41SomeIrrelevantText;";
	CHECK(input.xml_unescape() == input);
	input = "&#66SomeIrrelevantText;";
	CHECK(input.xml_unescape() == input);
	return true;
}

bool test_37() {
#define CHECK_EQ(X, Y)                                            \
	if ((X) != (Y)) {                                             \
		OS::get_singleton()->print("\tFAIL: %s != %s\n", #X, #Y); \
		return false;                                             \
	} else {                                                      \
		OS::get_singleton()->print("\tPASS\n");                   \
	}
	OS::get_singleton()->print("\n\nTest 37: Word wrap\n");

	// Long words.
	CHECK_EQ(String("12345678").word_wrap(8), "12345678");
	CHECK_EQ(String("1234567812345678").word_wrap(8), "12345678\n12345678");
	CHECK_EQ(String("123456781234567812345678").word_wrap(8), "12345678\n12345678\n12345678");

	// Long line.
	CHECK_EQ(String("123 567 123456 123").word_wrap(8), "123 567\n123456\n123");

	// Force newline at line length should not create another newline.
	CHECK_EQ(String("12345678 123").word_wrap(8), "12345678\n123");
	CHECK_EQ(String("12345678\n123").word_wrap(8), "12345678\n123");

	// Wrapping removes spaces.
	CHECK_EQ(String("1234567   123").word_wrap(8), "1234567\n123");
	CHECK_EQ(String("12345678  123").word_wrap(8), "12345678\n123");

	// Wrapping does not remove leading space.
	CHECK_EQ(String("  123456   123   12").word_wrap(8), "  123456\n123   12");
	CHECK_EQ(String("  123456\n   456   12").word_wrap(8), "  123456\n   456\n12");
	CHECK_EQ(String("  123456\n   4  12345678").word_wrap(8), "  123456\n   4\n12345678");
	CHECK_EQ(String("  123456\n   4  12345678123").word_wrap(8), "  123456\n   4\n12345678\n123");

	return true;
}

typedef bool (*TestFunc)();

TestFunc test_funcs[] = {

	test_1,
	test_2,
	test_3,
	test_4,
	test_5,
	test_6,
	test_7,
	test_8,
	test_9,
	test_10,
	test_11,
	test_12,
	test_13,
	test_14,
	test_15,
	test_16,
	test_17,
	test_18,
	test_19,
	test_20,
	test_21,
	test_22,
	test_23,
	test_24,
	test_25,
	test_26,
	test_27,
	test_28,
	test_29,
	test_30,
	test_31,
	test_32,
	test_33,
	test_34,
	test_35,
	test_36,
	test_37,
	nullptr

};

MainLoop *test() {
	/** A character length != wchar_t may be forced, so the tests won't work */

	ERR_FAIL_COND_V(sizeof(CharType) != sizeof(wchar_t), nullptr);

	int count = 0;
	int passed = 0;

	while (true) {
		if (!test_funcs[count]) {
			break;
		}
		bool pass = test_funcs[count]();
		if (pass) {
			passed++;
		}
		OS::get_singleton()->print("\t%s\n", pass ? "PASS" : "FAILED");

		count++;
	}

	OS::get_singleton()->print("\n\n\n");
	OS::get_singleton()->print("*************\n");
	OS::get_singleton()->print("***TOTALS!***\n");
	OS::get_singleton()->print("*************\n");

	OS::get_singleton()->print("Passed %i of %i tests\n", passed, count);

	return nullptr;
}
} // namespace TestString
