/**************************************************************************/
/*  crash_handler.h                                                       */
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

#if defined(DEBUG_ENABLED)

#include "core/crypto/crypto_core.h"
#include "core/io/marshalls.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"

namespace CrashHandlerBase {
struct ModData {
	String mod_name;
	uint64_t offset = 0;
};

struct SymInfo {
	uint64_t offset = 0;
	int8_t mod_idx = 0;
};

static void print_encoded_trace(const LocalVector<ModData> &p_mod_info, const LocalVector<SymInfo> &p_sym_info, const String &p_name) {
	PackedByteArray encoded_trace;

	{
		CharString cs = p_name.utf8();
		encoded_trace.resize(1 + cs.length());
		encoded_trace.write[0] = cs.length();
		memcpy(encoded_trace.ptrw() + 1, cs.get_data(), cs.length());
	}
	{
		CharString cs;

#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
		cs = "x86_64";
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
		cs = "x86_32";
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
		cs = "arm64";
#elif defined(__arm__) || defined(_M_ARM)
		cs = "arm32";
#elif defined(__riscv)
		cs = "rv64";
#elif defined(__powerpc64__)
		cs = "ppc64";
#elif defined(__loongarch64)
		cs = "loongarch64";
#elif defined(__wasm64__)
		cs = "wasm64";
#elif defined(__wasm32__)
		cs = "wasm32";
#else
		cs = "?";
#endif
		int64_t off = encoded_trace.size();
		encoded_trace.resize(off + 1 + cs.length());
		encoded_trace.write[off] = cs.length();
		memcpy(encoded_trace.ptrw() + off + 1, cs.get_data(), cs.length());
	}

	encoded_trace.push_back(p_mod_info.size());
	for (unsigned int i = 0; i < p_mod_info.size(); i++) {
		CharString cs = p_mod_info[i].mod_name.utf8();
		int64_t off = encoded_trace.size();
		encoded_trace.resize(off + 2 + 8 + cs.length());
		encode_uint64(p_mod_info[i].offset, encoded_trace.ptrw() + off);
		encode_uint16(cs.length(), encoded_trace.ptrw() + off + 8);
		memcpy(encoded_trace.ptrw() + off + 8 + 2, cs.get_data(), cs.length());
	}
	encoded_trace.push_back(p_sym_info.size());
	for (unsigned int i = 0; i < p_sym_info.size(); i++) {
		int64_t off = encoded_trace.size();
		encoded_trace.resize(off + 8 + 1);
		encode_uint64(p_sym_info[i].offset, encoded_trace.ptrw() + off);
		encoded_trace.write[off + 8] = p_sym_info[i].mod_idx;
	}
	print_error("Base64-encoded stack trace to symbolicate with 'misc/scripts/retrace.py':\n");
	print_error(vformat("\x1b[22;90m%s\x1b[0m", CryptoCore::b64_encode_str(encoded_trace.ptr(), encoded_trace.size())));
}
}; //namespace CrashHandlerBase

#endif
