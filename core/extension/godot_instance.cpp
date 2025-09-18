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
#include "main/main.h"
#include "servers/display_server.h"

#define GODOT_INSTANCE_LOG(...) print_line(__VA_ARGS__)

TaskExecutor::TaskExecutor(InvokeCallbackFunction p_async_func, ExecutorData p_async_data, InvokeCallbackFunction p_sync_func, ExecutorData p_sync_data) {
	async_func = p_async_func;
	async_data = p_async_data;
	sync_func = p_sync_func;
	sync_data = p_sync_data;
}

void TaskExecutor::sync(std::function<void()> p_callback) {
	sync_func(&TaskExecutor::invokeCallback, new std::function<void()>(p_callback), sync_data);
}

void TaskExecutor::async(std::function<void()> p_callback) {
	async_func(&TaskExecutor::invokeCallback, new std::function<void()>(p_callback), async_data);
}

void TaskExecutor::invokeCallback(void *p_callback) {
	std::function<void()> *callback = (std::function<void()> *)p_callback;
	(*callback)();
	delete callback;
}

void GodotInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start"), &GodotInstance::start);
	ClassDB::bind_method(D_METHOD("is_started"), &GodotInstance::is_started);
	ClassDB::bind_method(D_METHOD("iteration"), &GodotInstance::iteration);
	ClassDB::bind_method(D_METHOD("focus_in"), &GodotInstance::focus_in);
	ClassDB::bind_method(D_METHOD("focus_out"), &GodotInstance::focus_out);
	ClassDB::bind_method(D_METHOD("pause"), &GodotInstance::pause);
	ClassDB::bind_method(D_METHOD("resume"), &GodotInstance::resume);
	ClassDB::bind_method(D_METHOD("execute", "callback", "async"), &GodotInstance::execute);
}

GodotInstance::GodotInstance() {
}

GodotInstance::~GodotInstance() {
}

bool GodotInstance::initialize(GDExtensionInitializationFunction p_init_func, GodotInstanceCallbacks *p_callbacks) {
	GODOT_INSTANCE_LOG("Godot Instance initialization");
	callbacks = p_callbacks;
	GDExtensionManager *gdextension_manager = GDExtensionManager::get_singleton();
	GDExtensionConstPtr<const GDExtensionInitializationFunction> ptr((const GDExtensionInitializationFunction *)&p_init_func);
	GDExtensionManager::LoadStatus status = gdextension_manager->load_function_extension("libgodot://main", ptr);
	return status == GDExtensionManager::LoadStatus::LOAD_STATUS_OK;
}

#define CALL_CB(cb)          \
	if (callbacks) {         \
		callbacks->cb(this); \
	}

bool GodotInstance::start() {
	GODOT_INSTANCE_LOG("GodotInstance::start()");
	CALL_CB(before_setup2);
	Error err = Main::setup2();
	if (err != OK) {
		return false;
	}
	CALL_CB(before_start);
	started = Main::start() == EXIT_SUCCESS;
	if (started) {
		OS::get_singleton()->get_main_loop()->initialize();
		CALL_CB(after_start);
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
	GODOT_INSTANCE_LOG("GodotInstance::stop()");
	if (started) {
		OS::get_singleton()->get_main_loop()->finalize();
	}
	started = false;
}

void GodotInstance::focus_out() {
	GODOT_INSTANCE_LOG("GodotInstance::focus_out()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
		}

		callbacks->focus_out(this);
	}
}

void GodotInstance::focus_in() {
	GODOT_INSTANCE_LOG("GodotInstance::focus_in()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
		}
		callbacks->focus_in(this);
	}
}

void GodotInstance::pause() {
	GODOT_INSTANCE_LOG("GodotInstance::pause()");
	if (started) {
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_PAUSED);
		}
		callbacks->pause(this);
	}
}

void GodotInstance::resume() {
	GODOT_INSTANCE_LOG("GodotInstance::resume()");
	if (started) {
		callbacks->resume(this);
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_RESUMED);
		}
	}
}

void GodotInstance::set_executor(TaskExecutor *p_executor) {
	executor = p_executor;
}

TaskExecutor *GodotInstance::get_executor() {
	return executor;
}

void GodotInstance::execute(Callable p_callback, bool p_async) {
	if (executor == nullptr) {
		p_callback.call();
		return;
	}
	if (p_async) {
		executor->async([p_callback]() {
			p_callback.call();
		});
	} else {
		executor->sync([p_callback]() {
			p_callback.call();
		});
	}
}
