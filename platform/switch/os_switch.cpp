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

#include "core/config/project_settings.h"
#include "main/main.h"

#include "display_server_switch.h"
#include "os_switch.h"

void OS_Switch::initialize() {
	print("initialize\n");
	initialize_core();
	_random_generator.init();
}

void OS_Switch::finalize() {
	print("finalize\n");
	finalize_core();
	delete_main_loop();
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
	_joypads.initialize(Input::get_singleton());
}

void OS_Switch::delete_main_loop() {
	if (_main_loop) {
		memdelete(_main_loop);
	}
	_main_loop = nullptr;
}

bool OS_Switch::_check_internal_feature_support(const String &p_feature) {
	return false;
}

String OS_Switch::get_data_path() const {
	return "sdmc:/switch";
}

String OS_Switch::get_config_path() const {
	return "sdmc:/switch";
}

String OS_Switch::get_cache_path() const {
	return "sdmc:/switch";
}

String OS_Switch::get_user_data_dir() const {
	String appname = get_safe_dir_name(GLOBAL_GET("application/config/name"));
	if (!appname.is_empty()) {
		return get_data_path().path_join(get_godot_dir_name()).path_join("app_userdata").path_join(appname);
	}
	return get_data_path().path_join(get_godot_dir_name()).path_join("app_userdata").path_join("unnamed");
}

void OS_Switch::run() {
	if (!_main_loop) {
		return;
	}

	_main_loop->initialize();

	while (appletMainLoop()) {
		DisplayServer::get_singleton()->process_events(); // get rid of pending events

		_joypads.process();

		u32 kDown = padGetButtonsDown(&_joypads.get_pad());
		if (kDown & HidNpadButton_Plus) {
			break;
		}

		if (Main::iteration()) {
			break;
		}
	}

	_main_loop->finalize();
}

OS_Switch::OS_Switch(const std::vector<std::string> &args) :
		_args(args) {
	socketInitializeDefault();
	nxlinkStdio();
	romfsInit();

	//this will provide the create_function to the Main to instanciate the DisplayServer
	DisplayServerSwitch::register_NVN_driver();

	print("OS_Switch\n");
}

OS_Switch::~OS_Switch() {
	print("~OS_Switch\n");
	romfsExit();
	socketExit();
}
