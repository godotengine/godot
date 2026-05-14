/**************************************************************************/
/*  tests_caller.cpp                                                      */
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

#include "headers/tests.h"

#include <cstdio>

#ifdef WEB_ENABLED

// Handler now receives the control pointers directly
typedef void (*cmd_handler_t)(uint32_t cmd, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);

<<<<<<< HEAD
#define STATUS_PENDING 1
#define STATUS_DONE 2

// Handler now receives the control pointers directly
typedef void (*cmd_handler_t)(uint32_t cmd, volatile uint8_t *payload,
		volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);

static void handle_run_variant_tests(uint32_t cmd, volatile uint8_t *payload,
		volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);
=======
static void handle_run_variant_tests(uint32_t cmd, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);
>>>>>>> e952f3b (Fixed Quaternion in bridge_helpers.h, helpers.cs and CoreTypes.cs to)

static cmd_handler_t command_handlers[] = {
	nullptr, // 0 = CMD_NONE
	handle_run_variant_tests, // 1 = CMD_RUN_VARIANT_TESTS
};
static const uint32_t handler_count = sizeof(command_handlers) / sizeof(command_handlers[0]);

void process_api_commands() {
<<<<<<< HEAD
	volatile uint32_t *cmd_ptr = reinterpret_cast<volatile uint32_t *>(CMD_OFFSET);
	volatile uint8_t *status_ptr = reinterpret_cast<volatile uint8_t *>(STATUS_OFFSET);
	volatile uint8_t *payload = reinterpret_cast<volatile uint8_t *>(CMD_DATA);
=======
	volatile uint32_t *cmd_ptr = CMD_OFFSET;
	volatile uint8_t *status_ptr = STATUS_OFFSET;
>>>>>>> e952f3b (Fixed Quaternion in bridge_helpers.h, helpers.cs and CoreTypes.cs to)

	if (*status_ptr != STATUS_PENDING || *cmd_ptr == CMD_NONE) {
		return;
	}

	uint32_t cmd = *cmd_ptr;
	if (cmd >= handler_count || command_handlers[cmd] == nullptr) {
		*status_ptr = STATUS_DONE;
		*cmd_ptr = CMD_NONE;
		printf("[C++] Unknown command, cleared.\n");
		return;
	}

<<<<<<< HEAD
	command_handlers[cmd](cmd, payload, cmd_ptr, status_ptr);
}

static void handle_run_variant_tests(uint32_t cmd, volatile uint8_t *payload,
=======
	command_handlers[cmd](cmd, cmd_ptr, status_ptr);
}

static void handle_run_variant_tests(uint32_t cmd,
>>>>>>> e952f3b (Fixed Quaternion in bridge_helpers.h, helpers.cs and CoreTypes.cs to)
		volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr) {
	printf("[C++] Running variant tests...\n");

	// The test data is stored at absolute addresses, so we pass a base of 0.
<<<<<<< HEAD
	volatile uint8_t *memory = nullptr;
=======
>>>>>>> e952f3b (Fixed Quaternion in bridge_helpers.h, helpers.cs and CoreTypes.cs to)
	String error;
	bool ok = VariantBridgeTests::run_all_tests(error);
	printf("[C++] Test result: %s\n", ok ? "PASS" : "FAIL");
	if (!ok && !error.is_empty()) {
		printf("[C++] Error: %s\n", error.utf8().get_data());
	}

<<<<<<< HEAD
	// Write result directly via absolute offsets, using the provided status/cmd pointers
	volatile uint8_t *result_ptr = reinterpret_cast<volatile uint8_t *>(RESULT_OFFSET);
	*result_ptr = (uint8_t)(ok ? 1 : 0);
=======
	// Write result to RESULT_OFFSET
	volatile uint8_t *result_ptr = RESULT_OFFSET;
	*result_ptr = static_cast<uint8_t>(ok);
>>>>>>> e952f3b (Fixed Quaternion in bridge_helpers.h, helpers.cs and CoreTypes.cs to)
	*status_ptr = STATUS_DONE;
	*cmd_ptr = CMD_NONE;
}

// 2. Handle non-web platforms
#else

void process_api_commands() {
	// Stub: Does nothing on Windows/Linux
}

#endif
