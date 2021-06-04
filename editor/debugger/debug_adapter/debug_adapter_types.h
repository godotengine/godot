/*************************************************************************/
/*  debug_adapter_types.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DEBUG_ADAPTER_TYPES_H
#define DEBUG_ADAPTER_TYPES_H

#include "core/io/json.h"
#include "core/variant/dictionary.h"

namespace DAP {

enum ErrorType {
	UNKNOWN
};

enum StopReason {
	STEP,
	BREAKPOINT,
	EXCEPTION,
	PAUSE
};

struct Source {
	String name;
	String path;
	Array checksums;

	_FORCE_INLINE_ void from_json(const Dictionary &p_params) {
		name = p_params["name"];
		path = p_params["path"];
		checksums = p_params["checksums"];
	}

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["name"] = name;
		dict["path"] = path;
		dict["checksums"] = checksums;

		return dict;
	}
};

struct Breakpoint {
	bool verified;
	Source source;
	int line;

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["verified"] = verified;
		dict["source"] = source.to_json();
		dict["line"] = line;

		return dict;
	}
};

struct BreakpointLocation {
	int line;

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["line"] = line;

		return dict;
	}
};

struct Capabilities {
	bool supportsConfigurationDoneRequest = true;
	bool supportsEvaluateForHovers = true;
	bool supportsSetVariable = true;
	String supportedChecksumAlgorithms[2] = { "MD5", "SHA256" };
	bool supportsRestartRequest = true;
	bool supportsValueFormattingOptions = true;
	bool supportTerminateDebuggee = true;
	bool supportSuspendDebuggee = true;
	bool supportsTerminateRequest = true;
	bool supportsBreakpointLocationsRequest = true;

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["supportsConfigurationDoneRequest"] = supportsConfigurationDoneRequest;
		dict["supportsEvaluateForHovers"] = supportsEvaluateForHovers;
		dict["supportsSetVariable"] = supportsSetVariable;
		dict["supportsRestartRequest"] = supportsRestartRequest;
		dict["supportsValueFormattingOptions"] = supportsValueFormattingOptions;
		dict["supportTerminateDebuggee"] = supportTerminateDebuggee;
		dict["supportSuspendDebuggee"] = supportSuspendDebuggee;
		dict["supportsTerminateRequest"] = supportsTerminateRequest;
		dict["supportsBreakpointLocationsRequest"] = supportsBreakpointLocationsRequest;

		Array arr;
		arr.push_back(supportedChecksumAlgorithms[0]);
		arr.push_back(supportedChecksumAlgorithms[1]);
		dict["supportedChecksumAlgorithms"] = arr;

		return dict;
	}
};

struct Checksum {
	String algorithm;
	String checksum;

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["algorithm"] = algorithm;
		dict["checksum"] = checksum;

		return dict;
	}
};

struct Message {
	int id;
	String format;
	bool sendTelemetry = false; // Just in case :)
	bool showUser;

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["id"] = id;
		dict["format"] = format;
		dict["sendTelemetry"] = sendTelemetry;
		dict["showUser"] = showUser;

		return dict;
	}
};

struct SourceBreakpoint {
	int line;

	_FORCE_INLINE_ void from_json(const Dictionary &p_params) {
		line = p_params["line"];
	}
};

struct StackFrame {
	int id;
	String name;
	Source source;
	int line;
	int column;

	_FORCE_INLINE_ void from_json(const Dictionary &p_params) {
		id = p_params["id"];
		name = p_params["name"];
		source.from_json(p_params["source"]);
		line = p_params["line"];
		column = p_params["column"];
	}
};

struct Thread {
	int id;
	String name;

	_FORCE_INLINE_ Dictionary to_json() const {
		Dictionary dict;
		dict["id"] = id;
		dict["name"] = name;

		return dict;
	}
};

} // namespace DAP

#endif
