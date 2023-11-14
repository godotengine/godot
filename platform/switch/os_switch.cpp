/**************************************************************************/
/*  os_switch.cpp                                                         */
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

#include "switch/kernel/random.h"

#include "os_switch.h"

void OS_Switch::initialize() {
	print("initialize\n");
	initialize_core();
	_random_generator.init();
}

void OS_Switch::finalize() {
	print("finalize\n");
	finalize_core();
}

void OS_Switch::initialize_core() {
	OS_Unix::initialize_core();
}

void OS_Switch::finalize_core() {
	OS_Unix::finalize_core();
}

Error OS_Switch::get_entropy(uint8_t *r_buffer, int p_bytes) {
	ERR_FAIL_COND_V(_random_generator.get_random_bytes(r_buffer, p_bytes), FAILED);
	return OK;
}

void OS_Switch::initialize_joypads() {
	print("initialize_joypads\n");
}

void OS_Switch::delete_main_loop() {
}

bool OS_Switch::_check_internal_feature_support(const String &p_feature) {
	return false;
}

void OS_Switch::run() {
	print("run\n");
}

OS_Switch::OS_Switch(const std::vector<std::string> &args) :
		_args(args) {
	socketInitializeDefault();
	nxlinkStdio();
	romfsInit();
	print("OS_Switch\n");
}

OS_Switch::~OS_Switch() {
	print("~OS_Switch\n");
	romfsExit();
	socketExit();
}
