/**************************************************************************/
/*  gdscript_elf64_writer.cpp                                             */
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

#include "gdscript_elf64_writer.h"

#include "gdscript_riscv_encoder.h"
#include "modules/gdscript/gdscript_function.h"

#include <elfio/elfio.hpp>
#include <sstream>

PackedByteArray GDScriptELF64Writer::write_elf64(GDScriptFunction *p_function) {
	if (!p_function || p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		return PackedByteArray();
	}

	// 1. Encode RISC-V instructions from bytecode
	PackedByteArray code = GDScriptRISCVEncoder::encode_function(p_function);
	if (code.is_empty()) {
		return PackedByteArray();
	}

	// 2. Create ELF64 file using elfio
	ELFIO::elfio writer;
	writer.create(ELFIO::ELFCLASS64, ELFIO::ELFDATA2LSB);
	writer.set_os_abi(ELFIO::ELFOSABI_NONE); // ELFOSABI_NONE = SYSV
	writer.set_type(ELFIO::ET_EXEC);
	writer.set_machine(ELFIO::EM_RISCV); // From elfio/elf_types.hpp

	// 3. Create .text section with code
	ELFIO::section *text_sec = writer.sections.add(".text");
	text_sec->set_type(ELFIO::SHT_PROGBITS);
	text_sec->set_flags(ELFIO::SHF_ALLOC | ELFIO::SHF_EXECINSTR);
	text_sec->set_addr_align(0x10);

	// Convert PackedByteArray to char* for elfio
	const uint8_t *code_data = code.ptr();
	text_sec->set_data(reinterpret_cast<const char *>(code_data), code.size());

	// 4. Create loadable segment
	const ELFIO::Elf64_Addr ENTRY_POINT = 0x10000;
	const ELFIO::Elf_Xword PAGE_SIZE = 0x1000;

	ELFIO::segment *text_seg = writer.segments.add();
	text_seg->set_type(ELFIO::PT_LOAD);
	text_seg->set_virtual_address(ENTRY_POINT);
	text_seg->set_physical_address(ENTRY_POINT);
	text_seg->set_flags(ELFIO::PF_X | ELFIO::PF_R); // Executable + Readable
	text_seg->set_align(PAGE_SIZE);
	text_seg->add_section(text_sec, text_sec->get_addr_align());

	// 5. Set entry point
	writer.set_entry(ENTRY_POINT);

	// 6. Save to string stream and convert to PackedByteArray
	return elfio_to_packed_byte_array(writer);
}

bool GDScriptELF64Writer::can_write_elf64(GDScriptFunction *p_function) {
	if (!p_function) {
		return false;
	}
	// Check if function has bytecode
	return p_function->_code_ptr != nullptr && p_function->_code_size > 0;
}

PackedByteArray GDScriptELF64Writer::elfio_to_packed_byte_array(ELFIO::elfio &p_writer) {
	std::ostringstream oss;
	if (!p_writer.save(oss)) {
		return PackedByteArray();
	}

	std::string elf_data = oss.str();
	PackedByteArray result;
	result.resize(elf_data.size());
	memcpy(result.ptrw(), elf_data.data(), elf_data.size());
	return result;
}
