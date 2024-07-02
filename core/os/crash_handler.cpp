/**************************************************************************/
/*  crash_handler.cpp                                                     */
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

#include "crash_handler.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

void CrashHandlerBase::print_header(int p_signal) const {
	String msg;
	const ProjectSettings *proj_settings = ProjectSettings::get_singleton();
	if (proj_settings) {
		msg = proj_settings->get("debug/settings/crash_handler/message");
	}

	// Dump the backtrace to stderr with a message to the user.
	print_error("\n================================================================");
	if (p_signal != 0) {
		print_error(vformat("Program crashed with signal %d", p_signal));
	} else {
		print_error("Program crashed");
	}

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	if (String(VERSION_HASH).is_empty()) {
		print_error(vformat("Engine version: %s", VERSION_FULL_NAME));
	} else {
		print_error(vformat("Engine version: %s (%s)", VERSION_FULL_NAME, VERSION_HASH));
	}
	print_error(msg);
	print_error("");
	print_error("Dumping the backtrace...");
}

void CrashHandlerBase::print_trace(const CrashHandlerBase::TraceData &p_data) const {
	print_error("================================================================");
	print_error("");
	int longest_name = 0;
	int longest_addr = 0;
	for (int i = 0; i < (int)p_data.trace.size(); i++) {
		const AddressData &addr = p_data.trace[i];
		longest_name = MAX(longest_name, p_data.modules[addr.module_idx].fname.get_file().length());
		longest_addr = MAX(longest_addr, vformat("%ux", addr.address - addr.base).length());
	}
	for (int i = 0; i < (int)p_data.trace.size(); i++) {
		const AddressData &addr = p_data.trace[i];
		print_error(vformat("%3d: %" + itos(longest_addr) + "ux %" + itos(longest_name) + "s - %s", i + 1, addr.address - addr.base, p_data.modules[addr.module_idx].fname.get_file(), addr.fname));
	}
	print_error("");
	print_error("-- END OF BACKTRACE --");
	print_error("================================================================");
}

String CrashHandlerBase::encode_trace(const CrashHandlerBase::TraceData &p_data) const {
	String ret;

	Dictionary td;
	td["o"] = OS::get_singleton()->get_identifier();
	td["a"] = Engine::get_singleton()->get_architecture_name();
	if (String(VERSION_HASH).is_empty()) {
		td["v"] = String(VERSION_FULL_BUILD);
	} else {
		td["v"] = String(VERSION_HASH);
	}
	td["s"] = (int8_t)p_data.signal;
	PackedStringArray modules;
	for (int i = 0; i < (int)p_data.modules.size(); i++) {
		modules.push_back(p_data.modules[i].fname);
	}
	td["m"] = modules;
	PackedInt32Array trace_addr;
	PackedByteArray trace_mod;
	for (int i = 0; i < (int)p_data.trace.size(); i++) {
		const AddressData &addr = p_data.trace[i];
		trace_addr.push_back(addr.address - addr.base);
		trace_mod.push_back(addr.module_idx);
	}
	td["d"] = trace_addr;
	td["t"] = trace_mod;

	int len;
	Error err = encode_variant(td, nullptr, len, false);
	if (err == OK) {
		Vector<uint8_t> buff;
		buff.resize(len);
		err = encode_variant(td, buff.ptrw(), len, false);
		if (err == OK) {
			ret = CryptoCore::b64_encode_str(buff.ptr(), buff.size());
		}
	}
	return ret;
}

CrashHandlerBase::TraceData CrashHandlerBase::decode_trace(const String &p_trace_b64) const {
	TraceData out;

	CharString cs = p_trace_b64.ascii();
	Vector<uint8_t> buff;
	buff.resize(cs.length() / 4 * 3 + 1);

	size_t dict_len = 0;
	Error err = CryptoCore::b64_decode(buff.ptrw(), buff.size(), &dict_len, (unsigned char *)cs.get_data(), cs.length());
	if (err) {
		ERR_FAIL_V_MSG(out, "Invalid trace data, base64 decode error.");
	}
	Variant variant;
	err = decode_variant(variant, buff.ptr(), dict_len, nullptr, false);
	if (err) {
		ERR_FAIL_V_MSG(out, "Invalid trace data, variant decede error.");
	}
	Dictionary td = variant;
	if (!td.has("o") || !td.has("a") || !td.has("v") || !td.has("s") || !td.has("m") || !td.has("d") || !td.has("t")) {
		ERR_FAIL_V_MSG(out, "Invalid trace data, missing dictionary keys.");
	}
	String os_name = td["o"];
	if (os_name != OS::get_singleton()->get_identifier()) {
		ERR_FAIL_V_MSG(out, vformat("Mismatching trace OS, trace was recorded on %s.", os_name));
	}
	String arch_name = td["a"];
	if (arch_name != Engine::get_singleton()->get_architecture_name()) {
		ERR_FAIL_V_MSG(out, vformat("Mismatching trace architecture, trace was recorded on %s.", arch_name));
	}
	String version = td["v"];
	if (version != String(VERSION_FULL_BUILD) && version != String(VERSION_HASH)) {
		ERR_FAIL_V_MSG(out, vformat("Mismatching trace engine version, trace was recorded on %s.", version));
	}

	PackedStringArray modules = td["m"];
	for (int i = 0; i < (int)modules.size(); i++) {
		ModuleData md;
		md.fname = modules[i];
		out.modules.push_back(md);
	}
	PackedInt32Array trace_addr = td["d"];
	PackedByteArray trace_mod = td["t"];
	for (int i = 0; i < (int)trace_addr.size(); i++) {
		AddressData addr;
		addr.address = trace_addr[i];
		addr.module_idx = trace_mod[i];
		out.trace.push_back(addr);
		decode_address(out, out.trace.size() - 1, true);
	}
	out.signal = td["s"];

	return out;
}
