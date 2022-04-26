/*************************************************************************/
/*  test_theme.cpp                                                       */
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

#include "test_theme.h"

#include "core/os/os.h"
#include "main/tests/test_tools.h"
#include "scene/resources/theme.h"

#define CHECK(X)                                                             \
	if (!(X)) {                                                              \
		OS::get_singleton()->print("\tFAIL at line %d: %s\n", __LINE__, #X); \
		return false;                                                        \
	} else {                                                                 \
		OS::get_singleton()->print("\tPASS\n");                              \
	}

namespace TestTheme {

class Fixture {
public:
	struct DataEntry {
		Theme::DataType type;
		Variant value;
	} const valid_data[Theme::DATA_TYPE_MAX] = {
		{ Theme::DATA_TYPE_COLOR, Color() },
		{ Theme::DATA_TYPE_CONSTANT, 42 },
		{ Theme::DATA_TYPE_FONT, Ref<Font>(memnew(BitmapFont)) },
		{ Theme::DATA_TYPE_ICON, Ref<Texture>(memnew(ImageTexture)) },
		{ Theme::DATA_TYPE_STYLEBOX, Ref<StyleBox>(memnew(StyleBoxFlat)) },
	};

	const StringName valid_item_name = "valid_item_name";
	const StringName valid_type_name = "ValidTypeName";
};

bool test_good_theme_type_names() {
	Fixture fixture;
	StringName names[] = {
		"", // Empty name.
		"CapitalizedName",
		"snake_cased_name",
		"42",
		"_Underscore_",
	};

	// add_type
	for (const StringName &name : names) {
		Ref<Theme> theme = memnew(Theme);

		ErrorDetector ed;
		theme->add_type(name);
		CHECK(!ed.has_error);
	}

	// set_theme_item
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);

			ErrorDetector ed;
			theme->set_theme_item(entry.type, fixture.valid_item_name, name, entry.value);
			CHECK(!ed.has_error);
		}
	}

	// add_theme_item_type
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);

			ErrorDetector ed;
			theme->add_theme_item_type(entry.type, name);
			CHECK(!ed.has_error);
		}
	}

	// set_type_variation
	for (const StringName &name : names) {
		Ref<Theme> theme = memnew(Theme);

		ErrorDetector ed;
		theme->set_type_variation(fixture.valid_type_name, name);
		CHECK(ed.has_error == (name == StringName()));
	}
	for (const StringName &name : names) {
		Ref<Theme> theme = memnew(Theme);

		ErrorDetector ed;
		theme->set_type_variation(name, fixture.valid_type_name);
		CHECK(ed.has_error == (name == StringName()));
	}

	return true;
}

bool test_bad_theme_type_names() {
	Fixture fixture;
	StringName names[] = {
		"With/Slash",
		"With Space",
		"With@various$symbols!",
		String::utf8("contains_汉字"),
	};

	// add_type
	for (const StringName &name : names) {
		Ref<Theme> theme = memnew(Theme);

		ErrorDetector ed;
		theme->add_type(name);
		CHECK(ed.has_error);
	}

	// set_theme_item
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);

			ErrorDetector ed;
			theme->set_theme_item(entry.type, fixture.valid_item_name, name, entry.value);
			CHECK(ed.has_error);
		}
	}

	// add_theme_item_type
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);

			ErrorDetector ed;
			theme->add_theme_item_type(entry.type, name);
			CHECK(ed.has_error);
		}
	}

	// set_type_variation
	for (const StringName &name : names) {
		Ref<Theme> theme = memnew(Theme);

		ErrorDetector ed;
		theme->set_type_variation(fixture.valid_type_name, name);
		CHECK(ed.has_error);
	}
	for (const StringName &name : names) {
		Ref<Theme> theme = memnew(Theme);

		ErrorDetector ed;
		theme->set_type_variation(name, fixture.valid_type_name);
		CHECK(ed.has_error);
	}

	return true;
}

bool test_good_theme_item_names() {
	Fixture fixture;
	StringName names[] = {
		"CapitalizedName",
		"snake_cased_name",
		"42",
		"_Underscore_",
	};

	// set_theme_item
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);

			ErrorDetector ed;
			theme->set_theme_item(entry.type, name, fixture.valid_type_name, entry.value);
			CHECK(!ed.has_error);
			CHECK(theme->has_theme_item(entry.type, name, fixture.valid_type_name));
		}
	}

	// rename_theme_item
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);
			theme->set_theme_item(entry.type, fixture.valid_item_name, fixture.valid_type_name, entry.value);

			ErrorDetector ed;
			theme->rename_theme_item(entry.type, fixture.valid_item_name, name, fixture.valid_type_name);
			CHECK(!ed.has_error);
			CHECK(!theme->has_theme_item(entry.type, fixture.valid_item_name, fixture.valid_type_name));
			CHECK(theme->has_theme_item(entry.type, name, fixture.valid_type_name));
		}
	}

	return true;
}

bool test_bad_theme_item_names() {
	Fixture fixture;
	StringName names[] = {
		"", // Empty name.
		"With/Slash",
		"With Space",
		"With@various$symbols!",
		String::utf8("contains_汉字"),
	};

	// set_theme_item
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);

			ErrorDetector ed;
			theme->set_theme_item(entry.type, name, fixture.valid_type_name, entry.value);
			CHECK(ed.has_error);
			CHECK(!theme->has_theme_item(entry.type, name, fixture.valid_type_name));
		}
	}

	// rename_theme_item
	for (const StringName &name : names) {
		for (const Fixture::DataEntry &entry : fixture.valid_data) {
			Ref<Theme> theme = memnew(Theme);
			theme->set_theme_item(entry.type, fixture.valid_item_name, fixture.valid_type_name, entry.value);

			ErrorDetector ed;
			theme->rename_theme_item(entry.type, fixture.valid_item_name, name, fixture.valid_type_name);
			CHECK(ed.has_error);
			CHECK(theme->has_theme_item(entry.type, fixture.valid_item_name, fixture.valid_type_name));
			CHECK(!theme->has_theme_item(entry.type, name, fixture.valid_type_name));
		}
	}

	return true;
}

typedef bool (*TestFunc)();
TestFunc test_funcs[] = {
	test_good_theme_type_names,
	test_bad_theme_type_names,
	test_good_theme_item_names,
	test_bad_theme_item_names,
	nullptr
};

MainLoop *test() {
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
} // namespace TestTheme
