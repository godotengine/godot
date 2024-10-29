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

#ifndef CRASH_HANDLER_H
#define CRASH_HANDLER_H

#include "os.h"

class CrashHandlerBase {
public:
	struct ModuleData {
		String fname; // Store.
		uint64_t load_address = 0;
	};
	struct AddressData {
		bool system = false;
		int module_idx = -1; // Store.
		uint64_t faddress = 0;
		uint64_t address = 0; // Store.
		uint64_t base = 0;
		String fname;
	};
	struct TraceData {
		LocalVector<ModuleData> modules;
		LocalVector<AddressData> trace;
		int signal = 0;
	};

protected:
	bool disabled;

public:
	virtual void initialize() = 0;

	virtual TraceData collect_trace(int p_signal) const = 0;
	virtual void decode_address(TraceData &p_data, int p_address_idx, bool p_remap = false) const = 0;

	void print_header(int p_signal) const;
	void print_trace(const TraceData &p_data) const;
	String encode_trace(const TraceData &p_data) const;
	TraceData decode_trace(const String &p_trace_b64) const;

	virtual void disable() = 0;
	bool is_disabled() const { return disabled; }

	virtual ~CrashHandlerBase() {}
};

#endif // CRASH_HANDLER_H
