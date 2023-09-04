/**************************************************************************/
/*  test_pck_packer.h                                                     */
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

#ifndef TEST_PCK_PACKER_H
#define TEST_PCK_PACKER_H

#include "core/io/file_access_pack.h"
#include "core/io/pck_packer.h"
#include "core/io/pck_reader.h"
#include "core/os/os.h"

#include "tests/test_utils.h"
#include "thirdparty/doctest/doctest.h"

namespace TestPCKPacker {

TEST_CASE("[PCKPacker] Pack an empty PCK file") {
	PCKPacker pck_packer;
	const String output_pck_path = OS::get_singleton()->get_cache_path().path_join("output_empty.pck");
	CHECK_MESSAGE(
			pck_packer.pck_start(output_pck_path) == OK,
			"Starting a PCK file should return an OK error code.");

	CHECK_MESSAGE(
			pck_packer.flush() == OK,
			"Flushing the PCK should return an OK error code.");

	Error err;
	Ref<FileAccess> f = FileAccess::open(output_pck_path, FileAccess::READ, &err);
	CHECK_MESSAGE(
			err == OK,
			"The generated empty PCK file should be opened successfully.");
	CHECK_MESSAGE(
			f->get_length() >= 100,
			"The generated empty PCK file shouldn't be too small (it should have the PCK header).");
	CHECK_MESSAGE(
			f->get_length() <= 500,
			"The generated empty PCK file shouldn't be too large.");
}

TEST_CASE("[PCKPacker] Pack empty with zero alignment invalid") {
	PCKPacker pck_packer;
	const String output_pck_path = OS::get_singleton()->get_cache_path().path_join("output_empty.pck");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(pck_packer.pck_start(output_pck_path, 0) != OK, "PCK with zero alignment should fail.");
	ERR_PRINT_ON;
}

TEST_CASE("[PCKPacker] Pack empty with invalid key") {
	PCKPacker pck_packer;
	const String output_pck_path = OS::get_singleton()->get_cache_path().path_join("output_empty.pck");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(pck_packer.pck_start(output_pck_path, 32, "") != OK, "PCK with invalid key should fail.");
	ERR_PRINT_ON;
}

static bool check_file(PCKReader &p_pck_reader, const String &p_source_path, const String &p_pack_path) {
	const PackedByteArray data = p_pck_reader.read_file(p_pack_path, true);

	Error err = OK;
	Ref<FileAccess> source_file = FileAccess::open(p_source_path, FileAccess::READ, &err);
	CHECK_MESSAGE(
			err == OK,
			"[BUG] Cannot open source file.");

	PackedByteArray srcdata;
	srcdata.resize(source_file->get_length());
	source_file->get_buffer((uint8_t *)srcdata.ptr(), srcdata.size());

	return data == srcdata;
}

TEST_CASE("[PCKPacker] Pack a PCK file with some files and directories") {
	PCKPacker pck_packer;
	const String output_pck_path = OS::get_singleton()->get_cache_path().path_join("output_with_files.pck");
	CHECK_MESSAGE(
			pck_packer.pck_start(output_pck_path) == OK,
			"Starting a PCK file should return an OK error code.");

	const String base_dir = OS::get_singleton()->get_executable_path().get_base_dir();

	CHECK_MESSAGE(
			pck_packer.add_file("version.py", base_dir.path_join("../version.py"), "version.py") == OK,
			"Adding a file to the PCK should return an OK error code.");
	CHECK_MESSAGE(
			pck_packer.add_file("some/directories with spaces/to/create/icon.png", base_dir.path_join("../icon.png")) == OK,
			"Adding a file to a new subdirectory in the PCK should return an OK error code.");
	CHECK_MESSAGE(
			pck_packer.add_file("some/directories with spaces/to/create/icon.svg", base_dir.path_join("../icon.svg")) == OK,
			"Adding a file to an existing subdirectory in the PCK should return an OK error code.");
	CHECK_MESSAGE(
			pck_packer.add_file("some/directories with spaces/to/create/icon.png", base_dir.path_join("../logo.png")) == OK,
			"Overriding a non-flushed file to an existing subdirectory in the PCK should return an OK error code.");
	CHECK_MESSAGE(
			pck_packer.flush() == OK,
			"Flushing the PCK should return an OK error code.");

	Error err;
	Ref<FileAccess> f = FileAccess::open(output_pck_path, FileAccess::READ, &err);
	CHECK_MESSAGE(
			err == OK,
			"The generated non-empty PCK file should be opened successfully.");
	CHECK_MESSAGE(
			f->get_length() >= 18000,
			"The generated non-empty PCK file should be large enough to actually hold the contents specified above.");
	CHECK_MESSAGE(
			f->get_length() <= 27000,
			"The generated non-empty PCK file shouldn't be too large.");

	// Now check the contents of the generated file with PCKReader.
	PCKReader pck_reader;
	CHECK_MESSAGE(
			pck_reader.open(output_pck_path, 0) == OK,
			"Opening a PCK file should return an OK error code.");

	PackedStringArray expected_files = { "version.py", "some/directories with spaces/to/create/icon.png", "some/directories with spaces/to/create/icon.svg" };
	CHECK_MESSAGE(pck_reader.get_files() == expected_files, "PCK file has unexpected file list.");

	CHECK_MESSAGE(
			check_file(pck_reader, base_dir.path_join("../version.py"), "version.py"),
			"File \"version.py\" in the pack does not match source.");
	CHECK_MESSAGE(
			check_file(pck_reader, base_dir.path_join("../icon.svg"), "some/directories with spaces/to/create/icon.svg"),
			"File \"some/directories with spaces/to/create/icon.svg\" in the pack does not match source.");
	CHECK_MESSAGE(
			check_file(pck_reader, base_dir.path_join("../logo.png"), "some/directories with spaces/to/create/icon.png"),
			"File \"some/directories with spaces/to/create/logo.png\" in the pack does not match source.");
}
} // namespace TestPCKPacker

#endif // TEST_PCK_PACKER_H
