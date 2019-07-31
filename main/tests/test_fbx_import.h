/*************************************************************************/
/*  test_string.h                                                        */
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

#ifndef TEST_FBX_IMPORT_H
#define TEST_FBX_IMPORT_H
#include "main/main.h"
#include "core/io/ip_address.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include <thirdparty/doctest/doctest.h>

#include "modules/regex/regex.h"

#include <wchar.h>
//#include "core/math/math_funcs.h"
#include <stdio.h>

namespace TestFbxImport {

class GodotEngineTestFixture {
private:
	OS* os;
public:
	GodotEngineTestFixture() {
		//Main::cleanup(); // we must first shit this down
		os = OS::get_singleton();
		//print
		// todo
		//os->print("command line args %s\n", os->get_cmdline_args)
		os->clear_cmdline_args();
		Main::start(); // startup engine
	}

	virtual ~GodotEngineTestFixture()
	{
		//Main::cleanup(); // stop engine
	}
};

TEST_CASE_FIXTURE(GodotEngineTestFixture, "[Model import] Godot initialisation test") {
	// Do some stuff
	OS::get_singleton()->print("hello world test fixture works\n");
}

} // namespace TestFbxImport

#endif // TEST_FBX_IMPORT_H
