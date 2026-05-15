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

#ifdef WEB_ENABLED

#include "headers/tests.h"

#include <cstdio>

// Handler receives the control pointers
typedef void (*cmd_handler_t)(uint32_t cmd, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);

static void handle_run_variant_tests(uint32_t cmd, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);

static cmd_handler_t command_handlers[] = {
	nullptr, // 0 = CMD_NONE
	handle_run_variant_tests, // 1 = CMD_RUN_VARIANT_TESTS
};
static const uint32_t handler_count = sizeof(command_handlers) / sizeof(command_handlers[0]);

void process_api_commands() {
	volatile uint32_t *cmd_ptr = CMD_OFFSET;
	volatile uint8_t *status_ptr = STATUS_OFFSET;

	if (*status_ptr != STATUS_PENDING || *cmd_ptr == CMD_NONE) {
		return;
	}

	uint32_t cmd = *cmd_ptr;
	if (cmd >= handler_count || command_handlers[cmd] == nullptr) {
		*status_ptr = STATUS_DONE;
		*cmd_ptr = CMD_NONE;
		return;
	}

	command_handlers[cmd](cmd, cmd_ptr, status_ptr);
}

static void handle_run_variant_tests(uint32_t cmd,
		volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr) {
	printf("[C++] Running variant read‑write tests...\n");

	String error;
	bool ok = VariantBridgeTests::run_read_tests(error);
	printf("[C++] Test result: %s\n", ok ? "PASS" : "FAIL");
	if (!ok && !error.is_empty()) {
		printf("[C++] Error: %s\n", error.utf8().get_data());
	}

	// After completing the Read tests, the writes run
	VariantBridgeTests::run_write_tests();

	// Write result to RESULT_OFFSET
	volatile uint8_t *result_ptr = RESULT_OFFSET;
	*result_ptr = static_cast<uint8_t>(ok);
	*status_ptr = STATUS_DONE;
	*cmd_ptr = CMD_NONE;
}

// 2. Handle non-web platforms
#else

void process_api_commands() {
	// Stub: Does nothing on Windows/Linux
}

#endif
