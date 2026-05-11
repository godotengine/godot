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

#include <cstdio> // for printf

#define CMD_OFFSET 1000000
#define STATUS_OFFSET 1000004
#define RESULT_OFFSET 1000008
#define CMD_DATA 1000016

#define CMD_NONE 0
#define CMD_RUN_VARIANT_TESTS 1

#define STATUS_IDLE 0
#define STATUS_PENDING 1
#define STATUS_DONE 2

typedef void (*cmd_handler_t)(uint32_t cmd, volatile uint8_t *memory);

static void handle_run_variant_tests(uint32_t cmd, volatile uint8_t *memory);

static cmd_handler_t command_handlers[] = {
	nullptr, // 0 = CMD_NONE
	handle_run_variant_tests, // 1 = CMD_RUN_VARIANT_TESTS
};
static const uint32_t handler_count = sizeof(command_handlers) / sizeof(command_handlers[0]);

void process_api_commands() {
	volatile uint8_t *memory = (volatile uint8_t *)0; // absolute addresses

	int status = read_int32(memory, STATUS_OFFSET);
	int cmd = read_int32(memory, CMD_OFFSET);

	// Print only when there is a potential command or status changed
	printf("[C++] Frame: status=%d, cmd=%d\n", status, cmd);

	if (status != STATUS_PENDING || cmd == CMD_NONE) {
		return;
	}

	printf("[C++] Handling command %u\n", (uint32_t)cmd);

	if (cmd < 0 || (uint32_t)cmd >= handler_count || command_handlers[cmd] == nullptr) {
		writer<int>(memory, CMD_OFFSET, CMD_NONE);
		writer<int>(memory, STATUS_OFFSET, STATUS_DONE);
		printf("[C++] Unknown command, cleared.\n");
		return;
	}

	command_handlers[cmd]((uint32_t)cmd, memory);
}

static void handle_run_variant_tests(uint32_t cmd, volatile uint8_t *memory) {
	printf("[C++] Running variant tests...\n");
	String error;
	bool ok = VariantBridgeTests::run_all_tests(memory, error);
	printf("[C++] Test result: %s\n", ok ? "PASS" : "FAIL");
	if (!ok && !error.is_empty()) {
		// Print the error string on failure
		CharString utf8 = error.utf8();
		printf("[C++] Error: %s\n", utf8.get_data());
	}

	writer<uint8_t>(memory, RESULT_OFFSET, ok ? 1 : 0);
	writer<int>(memory, STATUS_OFFSET, STATUS_DONE);
	writer<int>(memory, CMD_OFFSET, CMD_NONE);
}
