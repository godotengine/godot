/**************************************************************************/
/*  gdscript_elf64_writer.h                                              */
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

#include "core/variant/variant.h"
#include "gdscript_elf64_mode.h"

// Forward declarations
class GDScriptFunction;
namespace ELFIO {
	class elfio;
}

// Write ELF64 binary format from GDScript bytecode using elfio library
class GDScriptELF64Writer {
public:
	// Write ELF64 binary from bytecode using elfio library
	static PackedByteArray write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode = ELF64CompilationMode::GODOT_SYSCALL);

	// Check if function can be written to ELF64
	static bool can_write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode = ELF64CompilationMode::GODOT_SYSCALL);

private:
	// Convert elfio binary to PackedByteArray
	static PackedByteArray elfio_to_packed_byte_array(ELFIO::elfio &p_writer);
};
