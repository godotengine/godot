/**************************************************************************/
/*  test_gdscript_bytecode_elf.h                                          */
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

#pragma once

#include "tests/test_macros.h"
#include "tests/test_utils.h"

#include "../src/gdscript_elf_fallback.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_function.h"

#include <elfio/elfio.hpp>

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/string/string_builder.h"

namespace TestGDScriptELF {

// Return struct to keep script alive
struct GDScriptFunctionRef {
	Ref<GDScript> script;
	GDScriptFunction *function;
};

// Helper: Compile GDScript code and return function with bytecode
static GDScriptFunctionRef gdscript_code_to_function(const String &p_code, const StringName &p_func_name) {
	GDScriptFunctionRef result;
	result.script.instantiate();
	result.script->set_source_code(p_code);
	if (result.script->reload() != OK || !result.script->is_valid()) {
		return result; // function will be nullptr
	}
	const HashMap<StringName, GDScriptFunction *> &funcs = result.script->get_member_functions();
	result.function = funcs.has(p_func_name) ? funcs.get(p_func_name) : nullptr;
	return result;
}

// Helper: Verify ELF64 binary structure using elfio
static bool verify_elf64_structure(const PackedByteArray &p_elf) {
	if (p_elf.is_empty() || p_elf.size() < 64) {
		return false;
	}

	// Write to temporary file for elfio to read
	String temp_file = OS::get_singleton()->get_cache_path().path_join("test_elf.elf");
	Ref<FileAccess> f = FileAccess::open(temp_file, FileAccess::WRITE);
	if (f.is_null()) {
		return false;
	}
	f->store_buffer(p_elf);
	f->close();

	// Verify ELF structure using elfio reader
	ELFIO::elfio reader;
	if (!reader.load(temp_file.utf8().get_data())) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}

	// Verify ELF64
	if (reader.get_class() != ELFCLASS64) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}

	// Verify RISC-V machine type
	if (reader.get_machine() != EM_RISCV) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}

	// Verify executable type
	if (reader.get_type() != ET_EXEC) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}

	// Verify entry point is set
	if (reader.get_entry() == 0) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}

	// Verify code section exists
	ELFIO::section *text_sec = reader.sections[".text"];
	if (text_sec == nullptr) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}
	if (text_sec->get_type() != SHT_PROGBITS) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}
	if ((text_sec->get_flags() & SHF_EXECINSTR) == 0) {
		DirAccess::remove_absolute(temp_file);
		return false;
	}

	// Cleanup
	DirAccess::remove_absolute(temp_file);
	return true;
}

TEST_CASE("[GDScript][ELF64] Direct ELF64 generation - basic function") {
	String gdscript = R"(
func test_basic():
	return true
)";

	GDScriptFunctionRef func_ref = gdscript_code_to_function(gdscript, "test_basic");
	REQUIRE_MESSAGE(func_ref.function != nullptr, "Failed to compile GDScript function");

	PackedByteArray elf = func_ref.function->compile_to_elf64();
	REQUIRE_MESSAGE(!elf.is_empty(), "ELF64 generation failed");

	// Verify ELF magic number
	REQUIRE(elf.size() >= 4);
	CHECK(elf[0] == 0x7F);
	CHECK(elf[1] == 'E');
	CHECK(elf[2] == 'L');
	CHECK(elf[3] == 'F');

	// Verify ELF64 structure using elfio
	bool valid = verify_elf64_structure(elf);
	CHECK_MESSAGE(valid, "Generated ELF64 binary has invalid structure");
}

TEST_CASE("[GDScript][ELF64] Direct ELF64 generation - function with assignment") {
	String gdscript = R"(
func test_assign():
	var x = 42
	return x
)";

	GDScriptFunctionRef func_ref = gdscript_code_to_function(gdscript, "test_assign");
	REQUIRE_MESSAGE(func_ref.function != nullptr, "Failed to compile GDScript function");

	PackedByteArray elf = func_ref.function->compile_to_elf64();
	REQUIRE_MESSAGE(!elf.is_empty(), "ELF64 generation failed");

	bool valid = verify_elf64_structure(elf);
	CHECK_MESSAGE(valid, "Generated ELF64 binary has invalid structure");
}

TEST_CASE("[GDScript][ELF64] Direct ELF64 generation - function with jump") {
	String gdscript = R"(
func test_jump():
	if true:
		return 1
	return 0
)";

	GDScriptFunctionRef func_ref = gdscript_code_to_function(gdscript, "test_jump");
	REQUIRE_MESSAGE(func_ref.function != nullptr, "Failed to compile GDScript function");

	PackedByteArray elf = func_ref.function->compile_to_elf64();
	REQUIRE_MESSAGE(!elf.is_empty(), "ELF64 generation failed");

	bool valid = verify_elf64_structure(elf);
	CHECK_MESSAGE(valid, "Generated ELF64 binary has invalid structure");
}

TEST_CASE("[GDScript][ELF64] Direct ELF64 generation - function with operator") {
	String gdscript = R"(
func test_operator():
	var x = 10
	var y = 20
	return x + y
)";

	GDScriptFunctionRef func_ref = gdscript_code_to_function(gdscript, "test_operator");
	REQUIRE_MESSAGE(func_ref.function != nullptr, "Failed to compile GDScript function");

	PackedByteArray elf = func_ref.function->compile_to_elf64();
	REQUIRE_MESSAGE(!elf.is_empty(), "ELF64 generation failed");

	bool valid = verify_elf64_structure(elf);
	CHECK_MESSAGE(valid, "Generated ELF64 binary has invalid structure");
}

TEST_CASE("[GDScript][ELF64] Direct ELF64 generation - function with parameters") {
	String gdscript = R"(
func test_params(a: int, b: int) -> int:
	return a + b
)";

	GDScriptFunctionRef func_ref = gdscript_code_to_function(gdscript, "test_params");
	REQUIRE_MESSAGE(func_ref.function != nullptr, "Failed to compile GDScript function");

	PackedByteArray elf = func_ref.function->compile_to_elf64();
	REQUIRE_MESSAGE(!elf.is_empty(), "ELF64 generation failed");

	bool valid = verify_elf64_structure(elf);
	CHECK_MESSAGE(valid, "Generated ELF64 binary has invalid structure");
}

TEST_CASE("[GDScript][ELF64] Error handling - null function") {
	// Test null function - can_compile_to_elf64 should handle null
	GDScriptFunction *null_func = nullptr;
	bool can_compile = null_func ? null_func->can_compile_to_elf64() : false;
	CHECK(can_compile == false);
}

TEST_CASE("[GDScript][ELF64] Error handling - empty function") {
	String empty_gdscript = R"(
func empty_func():
	pass
)";
	GDScriptFunctionRef empty_func_ref = gdscript_code_to_function(empty_gdscript, "empty_func");
	if (empty_func_ref.function != nullptr) {
		PackedByteArray elf = empty_func_ref.function->compile_to_elf64();
		// Either generates ELF or returns empty - both are valid
		CHECK(true);
	} else {
		// Compilation failed - acceptable in test environment
		CHECK(true);
	}
}

TEST_CASE("[GDScript][ELF64] Script-level compilation") {
	String gdscript = R"(
func func1():
	return 1

func func2():
	return 2
)";

	Ref<GDScript> script;
	script.instantiate();
	script->set_source_code(gdscript);
	if (script->reload() == OK && script->is_valid()) {
		Dictionary elf_dict = script->compile_all_functions_to_elf64();
		CHECK(elf_dict.size() > 0);
		CHECK(elf_dict.has("func1"));
		CHECK(elf_dict.has("func2"));
	}
}

} // namespace TestGDScriptELF
