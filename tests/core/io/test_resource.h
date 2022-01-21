/*************************************************************************/
/*  test_resource.h                                                      */
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

#ifndef TEST_RESOURCE
#define TEST_RESOURCE

#include "core/io/resource.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"

#include "thirdparty/doctest/doctest.h"

namespace TestResource {

TEST_CASE("[Resource] Duplication") {
	Ref<Resource> resource = memnew(Resource);
	resource->set_name("Hello world");
	Ref<Resource> child_resource = memnew(Resource);
	child_resource->set_name("I'm a child resource");
	resource->set_meta("other_resource", child_resource);

	Ref<Resource> resource_dupe = resource->duplicate();
	const Ref<Resource> &resource_dupe_reference = resource_dupe;
	resource_dupe->set_name("Changed name");
	child_resource->set_name("My name was changed too");

	CHECK_MESSAGE(
			resource_dupe->get_name() == "Changed name",
			"Duplicated resource should have the new name.");
	CHECK_MESSAGE(
			resource_dupe_reference->get_name() == "Changed name",
			"Reference to the duplicated resource should have the new name.");
	CHECK_MESSAGE(
			resource->get_name() == "Hello world",
			"Original resource name should not be affected after editing the duplicate's name.");
	CHECK_MESSAGE(
			Ref<Resource>(resource_dupe->get_meta("other_resource"))->get_name() == "My name was changed too",
			"Duplicated resource should share its child resource with the original.");
}

TEST_CASE("[Resource] Saving and loading") {
	Ref<Resource> resource = memnew(Resource);
	resource->set_name("Hello world");
	resource->set_meta("    ExampleMetadata    ", Vector2i(40, 80));
	resource->set_meta("string", "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks");
	Ref<Resource> child_resource = memnew(Resource);
	child_resource->set_name("I'm a child resource");
	resource->set_meta("other_resource", child_resource);
	const String save_path_binary = OS::get_singleton()->get_cache_path().plus_file("resource.res");
	const String save_path_text = OS::get_singleton()->get_cache_path().plus_file("resource.tres");
	ResourceSaver::save(save_path_binary, resource);
	ResourceSaver::save(save_path_text, resource);

	const Ref<Resource> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
	CHECK_MESSAGE(
			loaded_resource_binary->get_name() == "Hello world",
			"The loaded resource name should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_resource_binary->get_meta("    ExampleMetadata    ") == Vector2i(40, 80),
			"The loaded resource metadata should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_resource_binary->get_meta("string") == "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks",
			"The loaded resource metadata should be equal to the expected value.");
	const Ref<Resource> &loaded_child_resource_binary = loaded_resource_binary->get_meta("other_resource");
	CHECK_MESSAGE(
			loaded_child_resource_binary->get_name() == "I'm a child resource",
			"The loaded child resource name should be equal to the expected value.");

	const Ref<Resource> &loaded_resource_text = ResourceLoader::load(save_path_text);
	CHECK_MESSAGE(
			loaded_resource_text->get_name() == "Hello world",
			"The loaded resource name should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_resource_text->get_meta("    ExampleMetadata    ") == Vector2i(40, 80),
			"The loaded resource metadata should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_resource_text->get_meta("string") == "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks",
			"The loaded resource metadata should be equal to the expected value.");
	const Ref<Resource> &loaded_child_resource_text = loaded_resource_text->get_meta("other_resource");
	CHECK_MESSAGE(
			loaded_child_resource_text->get_name() == "I'm a child resource",
			"The loaded child resource name should be equal to the expected value.");
}
} // namespace TestResource

#endif // TEST_RESOURCE
