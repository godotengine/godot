/*************************************************************************/
/*  test_loader.cpp                                                      */
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

#include "test_loader.h"

#include "core/script_language.h"
#include "core/io/resource_loader.h"

bool TestLoader::from_path(Ref<TestSuite> test_suite, const String &path) {
	Error error;
	DirAccessRef directory = DirAccess::open(path, &error);
	ERR_FAIL_COND_V(Error::OK != error, false);
	return from_directory(test_suite, directory);
}

bool TestLoader::from_directory(Ref<TestSuite> test_suite, DirAccessRef &directory) {
	ERR_FAIL_COND_V(Error::OK != directory->list_dir_begin(), false);
	String filename = directory->get_next();
	while (filename != "") {
		if (filename == "." || filename == "..") {
		} else if (directory->current_is_dir()) {
			from_path(test_suite, directory->get_current_dir() + "/" + filename);
		} else {
			if (filename.ends_with("_test.gd")) {
				Ref<Script> script = ResourceLoader::load(directory->get_current_dir() + "/" + filename);
				if (!script.is_null()) {
					if (script->can_instance()) {
						TestCase *test_case = memnew(TestCase);
						ScriptInstance *instance = script->instance_create(test_case);
						test_suite->add_test(test_case);
					}
				}
			}
		}
		filename = directory->get_next();
	}
	directory->list_dir_end();
	return true;
}

Ref<TestSuite> TestLoader::_from_path(const String &path) {
	Ref<TestSuite> test_suite(memnew(TestSuite));
	ERR_FAIL_COND_V(from_path(test_suite, path), NULL);
	return test_suite;
}

Ref<TestSuite> TestLoader::_from_directory(Ref<_Directory> directory) {
	return _from_path(directory->get_current_dir());
}

void TestLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("from_path", "path"), &TestLoader::_from_path);
	ClassDB::bind_method(D_METHOD("from_directory", "directory"), &TestLoader::_from_directory);
}
