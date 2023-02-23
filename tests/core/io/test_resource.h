/**************************************************************************/
/*  test_resource.h                                                       */
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

#ifndef TEST_RESOURCE_H
#define TEST_RESOURCE_H

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
	resource->set_meta("ExampleMetadata", Vector2i(40, 80));
	resource->set_meta("string", "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks");
	Ref<Resource> child_resource = memnew(Resource);
	child_resource->set_name("I'm a child resource");
	resource->set_meta("other_resource", child_resource);
	const String save_path_binary = OS::get_singleton()->get_cache_path().path_join("resource.res");
	const String save_path_text = OS::get_singleton()->get_cache_path().path_join("resource.tres");
	ResourceSaver::save(resource, save_path_binary);
	ResourceSaver::save(resource, save_path_text);

	const Ref<Resource> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
	CHECK_MESSAGE(
			loaded_resource_binary->get_name() == "Hello world",
			"The loaded resource name should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_resource_binary->get_meta("ExampleMetadata") == Vector2i(40, 80),
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
			loaded_resource_text->get_meta("ExampleMetadata") == Vector2i(40, 80),
			"The loaded resource metadata should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_resource_text->get_meta("string") == "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks",
			"The loaded resource metadata should be equal to the expected value.");
	const Ref<Resource> &loaded_child_resource_text = loaded_resource_text->get_meta("other_resource");
	CHECK_MESSAGE(
			loaded_child_resource_text->get_name() == "I'm a child resource",
			"The loaded child resource name should be equal to the expected value.");
}

TEST_CASE("[Resource] Breaking circular references on save") {
	Ref<Resource> resource_a = memnew(Resource);
	resource_a->set_name("A");
	Ref<Resource> resource_b = memnew(Resource);
	resource_b->set_name("B");
	Ref<Resource> resource_c = memnew(Resource);
	resource_c->set_name("C");
	resource_a->set_meta("next", resource_b);
	resource_b->set_meta("next", resource_c);
	resource_c->set_meta("next", resource_b);

	const String save_path_binary = OS::get_singleton()->get_cache_path().path_join("resource.res");
	const String save_path_text = OS::get_singleton()->get_cache_path().path_join("resource.tres");
	ResourceSaver::save(resource_a, save_path_binary);
	ResourceSaver::save(resource_a, save_path_text);

	const Ref<Resource> &loaded_resource_a_binary = ResourceLoader::load(save_path_binary);
	CHECK_MESSAGE(
			loaded_resource_a_binary->get_name() == "A",
			"The loaded resource name should be equal to the expected value.");
	const Ref<Resource> &loaded_resource_b_binary = loaded_resource_a_binary->get_meta("next");
	CHECK_MESSAGE(
			loaded_resource_b_binary->get_name() == "B",
			"The loaded child resource name should be equal to the expected value.");
	const Ref<Resource> &loaded_resource_c_binary = loaded_resource_b_binary->get_meta("next");
	CHECK_MESSAGE(
			loaded_resource_c_binary->get_name() == "C",
			"The loaded child resource name should be equal to the expected value.");
	CHECK_MESSAGE(
			!loaded_resource_c_binary->get_meta("next"),
			"The loaded child resource circular reference should be NULL.");

	const Ref<Resource> &loaded_resource_a_text = ResourceLoader::load(save_path_text);
	CHECK_MESSAGE(
			loaded_resource_a_text->get_name() == "A",
			"The loaded resource name should be equal to the expected value.");
	const Ref<Resource> &loaded_resource_b_text = loaded_resource_a_text->get_meta("next");
	CHECK_MESSAGE(
			loaded_resource_b_text->get_name() == "B",
			"The loaded child resource name should be equal to the expected value.");
	const Ref<Resource> &loaded_resource_c_text = loaded_resource_b_text->get_meta("next");
	CHECK_MESSAGE(
			loaded_resource_c_text->get_name() == "C",
			"The loaded child resource name should be equal to the expected value.");
	CHECK_MESSAGE(
			!loaded_resource_c_text->get_meta("next"),
			"The loaded child resource circular reference should be NULL.");

	// Break circular reference to avoid memory leak
	resource_c->remove_meta("next");
}

static void resource_fuzz_test(bool test_simultaneous_load, bool test_cyclic_dependency) {
	const String test_dir = OS::get_singleton()->get_cache_path().path_join("godot_thread_test");
	DirAccess::make_dir_absolute(test_dir);
	for (const String &f : DirAccess::get_files_at(test_dir)) {
		// Clean up files at the beginning in case there's a previously failed run
		DirAccess::remove_absolute(test_dir.path_join(f));
	}

	const int count = 25;
	const int cycle_length = 5;
	int sum = 0;

	// Create circular dependencies to test load failures.
	// Also skip some so that we test failure to load nonexistent files.
	for (int i = 2; i < count; i++) {
		if (!test_cyclic_dependency) {
			break;
		}

		const String resource_name = test_dir.path_join(itos(i) + "-cyc.tres");
		Ref<Resource> resource = memnew(Resource);
		resource->set_name("Cyclic Resource");
		resource->set_meta("Addend", 0);

		// Create groups of smaller cycles [0-4], [5-9], [10-14], ...
		int link = ((i / cycle_length) * cycle_length) + ((i + 1) % cycle_length);
		Ref<Resource> child_resource = memnew(Resource);
		child_resource->set_path(itos(link) + "-cyc.tres");
		resource->set_meta("other_resource", child_resource);

		ResourceSaver::save(resource, resource_name);
	}

	// Create sequence of resources that each reference the previous one.
	// 0-ext.tres  <-  1-ext.tres  <-  2-ext.tres...
	for (int i = 0; i < count; i++) {
		const int addend = Math::rand() % 100;
		const String resource_name = test_dir.path_join(itos(i) + "-ext.tres");
		Ref<Resource> resource = memnew(Resource);
		resource->set_name("External Resource");
		resource->set_meta("Addend", addend);
		sum += addend;

		if (i != 0) {
			Ref<Resource> child_resource = memnew(Resource);
			child_resource->set_path(itos(i - 1) + "-ext.tres");
			resource->set_meta("other_resource", child_resource);
		}
		ResourceSaver::save(resource, resource_name);
	}
	// Create resources that reference the above chain.
	for (int i = 0; i < count; i++) {
		const String resource_name = test_dir.path_join(itos(i) + ".tres");
		Ref<Resource> resource = memnew(Resource);
		resource->set_name("Top Level Resource");
		resource->set_meta("Addend", 0);
		resource->set_meta("ID", i);

		Ref<Resource> child_resource = memnew(Resource);
		child_resource->set_path(itos(count - 1) + "-ext.tres");
		resource->set_meta("other_resource", child_resource);

		ResourceSaver::save(resource, resource_name);

		CHECK(ResourceLoader::load_threaded_get_status(resource_name) == ResourceLoader::THREAD_LOAD_INVALID_RESOURCE);
	}

	// Since we're testing threaded loading, and the cyclic dependencies are designed to fail,
	// there's no way to disable error messages at a finer granularity than this.
	ERR_PRINT_OFF;

	// Test threaded loading of above resources
	for (int i = 0; i < 500; i++) {
		const int id = Math::rand() % count;
		const bool is_cycle = Math::rand() % 2;
		const String resource_name = test_dir.path_join(itos(id) + (is_cycle ? "-cyc.tres" : ".tres"));

		// Spawn a new thread at random
		if (Math::rand() % 2) {
			// Randomly decide sub-thread and cache settings
			ResourceLoader::load_threaded_request(resource_name, "", Math::rand() % 2, ResourceFormatLoader::CacheMode(Math::rand() % 3));
			CHECK(ResourceLoader::load_threaded_get_status(resource_name) != ResourceLoader::THREAD_LOAD_INVALID_RESOURCE);

			if (test_simultaneous_load && (Math::rand() % 2)) {
				continue;
			}
		}

		const bool load_requested = ResourceLoader::load_threaded_get_status(resource_name) != ResourceLoader::THREAD_LOAD_INVALID_RESOURCE;

		// Randomly wait for a previously spawned thread
		if (load_requested && (Math::rand() % 2)) {
			while (ResourceLoader::load_threaded_get_status(resource_name) == ResourceLoader::THREAD_LOAD_IN_PROGRESS) {
				OS::get_singleton()->delay_usec(1);
			}
			if (is_cycle) {
				CHECK(ResourceLoader::load_threaded_get_status(resource_name) == ResourceLoader::THREAD_LOAD_FAILED);
			} else {
				CHECK(ResourceLoader::load_threaded_get_status(resource_name) == ResourceLoader::THREAD_LOAD_LOADED);
			}
		}

		// Join thread using a random method
		const Ref<Resource> &resource = (load_requested && (Math::rand() % 2))
				? ResourceLoader::load_threaded_get(resource_name)
				: ResourceLoader::load(resource_name, "", ResourceFormatLoader::CacheMode(Math::rand() % 3));

		if (is_cycle) {
			continue;
		}

		REQUIRE(resource.is_valid());

		// Check if resource data loaded correctly
		int meta_id = resource->get_meta("ID");
		CHECK(meta_id == id);

		int loaded_sum = 0;
		const Resource *r = resource.ptr();
		while (r) {
			int addend = r->get_meta("Addend");
			loaded_sum += addend;

			const Ref<Resource> &new_r = r->get_meta("other_resource");
			r = new_r.ptr();
		}

		// Check if all external resources were loaded correctly
		CHECK(loaded_sum == sum);
	}

	// Join threads.
	for (int is_cycle = 0; is_cycle < 2; is_cycle++) {
		for (int i = 0; i < count; i++) {
			const String resource_name = test_dir.path_join(itos(i) + (is_cycle ? "-cyc.tres" : ".tres"));
			while (ResourceLoader::load_threaded_get_status(resource_name) != ResourceLoader::THREAD_LOAD_INVALID_RESOURCE) {
				ResourceLoader::load_threaded_get(resource_name);
			}
		}
	}

	ERR_PRINT_ON;
}

TEST_CASE("[Resource] Simultaneous loading fuzz test") {
	resource_fuzz_test(true, false);
}
TEST_CASE_PENDING("[Resource] Cyclic dependencies fuzz test") {
	resource_fuzz_test(true, true);
}

} // namespace TestResource

#endif // TEST_RESOURCE_H
