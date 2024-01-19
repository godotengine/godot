/**************************************************************************/
/*  test_gdscript_uid.h                                                   */
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

#ifndef TEST_GDSCRIPT_UID_H
#define TEST_GDSCRIPT_UID_H

#ifdef TOOLS_ENABLED

#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "gdscript_test_runner.h"

#include "../gdscript.h"
#include "tests/test_macros.h"

namespace GDScriptTests {

static HashMap<String, ResourceUID::ID> id_cache;

ResourceUID::ID _resource_saver_get_resource_id_for_path(const String &p_path, bool p_generate) {
	return ResourceUID::get_singleton()->text_to_id("uid://baba");
}

static void test_script(const String &p_source, const String &p_target_source) {
	const String script_path = OS::get_singleton()->get_cache_path().path_join("script.gd");

	Ref<GDScript> script;
	script.instantiate();
	script->set_source_code(p_source);
	ResourceSaver::save(script, script_path);

	Ref<FileAccess> fa = FileAccess::open(script_path, FileAccess::READ);
	CHECK_EQ(fa->get_as_text(), p_target_source);
}

TEST_SUITE("[Modules][GDScript][UID]") {
	TEST_CASE("[ResourceSaver] Adding UID line to script") {
		init_language("modules/gdscript/tests/scripts");
		ResourceSaver::set_get_resource_id_for_path(_resource_saver_get_resource_id_for_path);

		const String source = R"(extends Node
class_name TestClass
)";
		const String final_source = R"(@uid("uid://baba") # Generated automatically, do not modify.
extends Node
class_name TestClass
)";

		// Script has no UID, add it.
		test_script(source, final_source);
	}

	TEST_CASE("[ResourceSaver] Updating UID line in script") {
		init_language("modules/gdscript/tests/scripts");
		ResourceSaver::set_get_resource_id_for_path(_resource_saver_get_resource_id_for_path);

		const String wrong_id_source = R"(

@uid(
	"uid://dead"
	) # G
extends Node
class_name TestClass
)";
		const String corrected_id_source = R"(

@uid("uid://baba") # Generated automatically, do not modify.
extends Node
class_name TestClass
)";
		const String correct_id_source = R"(@uid("uid://baba") # G
extends Node
class_name TestClass
)";

		// Script has wrong UID saved. Remove it and add a correct one.
		// Inserts in the same line, but multiline annotations are flattened.
		test_script(wrong_id_source, corrected_id_source);
		// The stored UID is correct, so do not modify it.
		test_script(correct_id_source, correct_id_source);
	}
}

} // namespace GDScriptTests

#endif

#endif // TEST_GDSCRIPT_UID_H
