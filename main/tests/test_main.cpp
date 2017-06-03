/*************************************************************************/
/*  test_main.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "list.h"
#include "os/main_loop.h"

#ifdef DEBUG_ENABLED

#include "test_containers.h"
#include "test_gui.h"
#include "test_math.h"
#include "test_physics.h"
#include "test_physics_2d.h"
#include "test_render.h"
#include "test_sound.h"
#include "test_string.h"

#include "test_gdscript.h"
#include "test_image.h"
#include "test_io.h"
#include "test_shader_lang.h"

const char **tests_get_names() {

	static const char *test_names[] = {
		"string",
		"containers",
		"math",
		"render",
		"multimesh",
		"gui",
		"io",
		"shaderlang",
		"physics",
		NULL
	};

	return test_names;
}

MainLoop *test_main(String p_test, const List<String> &p_args) {

	if (p_test == "string") {

		return TestString::test();
	}

	if (p_test == "containers") {

		return TestContainers::test();
	}

	if (p_test == "math") {

		return TestMath::test();
	}

	if (p_test == "physics") {

		return TestPhysics::test();
	}

	if (p_test == "physics_2d") {

		return TestPhysics2D::test();
	}

	if (p_test == "render") {

		return TestRender::test();
	}

#ifndef _3D_DISABLED
	if (p_test == "gui") {

		return TestGUI::test();
	}
#endif

	//if (p_test=="sound") {

	//	return TestSound::test();
	//}

	if (p_test == "io") {

		return TestIO::test();
	}

	if (p_test == "shaderlang") {

		return TestShaderLang::test();
	}

	if (p_test == "gd_tokenizer") {

		return TestGDScript::test(TestGDScript::TEST_TOKENIZER);
	}

	if (p_test == "gd_parser") {

		return TestGDScript::test(TestGDScript::TEST_PARSER);
	}

	if (p_test == "gd_compiler") {

		return TestGDScript::test(TestGDScript::TEST_COMPILER);
	}

	if (p_test == "gd_bytecode") {

		return TestGDScript::test(TestGDScript::TEST_BYTECODE);
	}

	if (p_test == "image") {

		return TestImage::test();
	}

	return NULL;
}

#else

const char **tests_get_names() {

	static const char *test_names[] = {
		NULL
	};

	return test_names;
}

MainLoop *test_main(String p_test, const List<String> &p_args) {

	return NULL;
}

#endif
