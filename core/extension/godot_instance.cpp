/**************************************************************************/
/*  godot_instance.cpp                                                    */
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

#include "godot_instance.h"

#include "core/extension/gdextension_manager.h"
#include "core/os/main_loop.h"
#include "main/main.h"
#include "servers/display/display_server.h"

void GodotInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start"), &GodotInstance::start);
	ClassDB::bind_method(D_METHOD("is_started"), &GodotInstance::is_started);
	ClassDB::bind_method(D_METHOD("iteration"), &GodotInstance::iteration);
	ClassDB::bind_method(D_METHOD("focus_in"), &GodotInstance::focus_in);
	ClassDB::bind_method(D_METHOD("focus_out"), &GodotInstance::focus_out);
	ClassDB::bind_method(D_METHOD("pause"), &GodotInstance::pause);
	ClassDB::bind_method(D_METHOD("resume"), &GodotInstance::resume);
}

GodotInstance::GodotInstance() {
}

GodotInstance::~GodotInstance() {
}

bool GodotInstance::initialize(GDExtensionInitializationFunction p_init_func) {
	print_verbose("Godot Instance initialization");
	GDExtensionManager *gdextension_manager = GDExtensionManager::get_singleton();
	GDExtensionConstPtr<const GDExtensionInitializationFunction> ptr((const GDExtensionInitializationFunction *)&p_init_func);
	GDExtensionManager::LoadStatus status = gdextension_manager->load_extension_from_function("libgodot://main", ptr);
	return status == GDExtensionManager::LoadStatus::LOAD_STATUS_OK;
}

bool GodotInstance::start() {
	print_verbose("GodotInstance::start()");
	Error err = Main::setup2();
	if (err != OK) {
		return false;
	}
	started = Main::start() == EXIT_SUCCESS;
	if (started) {
		OS::get_singleton()->get_main_loop()->initialize();
	}
	return started;
}

bool GodotInstance::is_started() {
	return started;
}

bool GodotInstance::iteration() {
	DisplayServer::get_singleton()->process_events();
	return Main::iteration();
}

void GodotInstance::stop() {
	print_verbose("GodotInstance::stop()");
	if (started) {
		OS::get_singleton()->get_main_loop()->finalize();
	}
	started = false;
}

void GodotInstance::focus_out() {
	print_verbose("GodotInstance::focus_out()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
		}
	}
}

void GodotInstance::focus_in() {
	print_verbose("GodotInstance::focus_in()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
		}
	}
}

void GodotInstance::pause() {
	print_verbose("GodotInstance::pause()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_PAUSED);
		}
	}
}

void GodotInstance::resume() {
	print_verbose("GodotInstance::resume()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_RESUMED);
		}
	}
}
