/**************************************************************************/
/*  libgodot_ios.mm                                                       */
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

#include "core/extension/godot_instance.h"
#include "core/extension/libgodot.h"
#include "core/io/libgodot_logger.h"
#include "main/main.h"

#import "os_ios.h"

static GodotInstance *instance = nullptr;

class GodotInstanceCallbacksIOS : public GodotInstanceCallbacks {
public:
	void focus_out(GodotInstance *p_instance) override {
		OS_IOS::get_singleton()->on_focus_out();
	}
	void focus_in(GodotInstance *p_instance) override {
		OS_IOS::get_singleton()->on_focus_in();
	}
	void pause(GodotInstance *p_instance) override {
		p_instance->focus_out();
	}
	void resume(GodotInstance *p_instance) override {
		p_instance->focus_in();
	}
};

static GodotInstanceCallbacksIOS callbacks;

extern LIBGODOT_API GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func, InvokeCallbackFunction p_async_func, ExecutorData p_async_data, InvokeCallbackFunction p_sync_func, ExecutorData p_sync_data, LogCallbackFunction p_log_func, LogCallbackData p_log_data) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot Instance may be created.");

	TaskExecutor *executor = nullptr;
	if (p_async_func != nullptr && p_sync_func != nullptr) {
		executor = new TaskExecutor(p_async_func, p_async_data, p_sync_func, p_sync_data);
	}

	std::function<void()> init = [executor, p_argv, p_argc, p_init_func, p_log_func, p_log_data]() {
		OS_IOS *os = new OS_IOS();
		if (p_log_func != nullptr && p_log_data != nullptr) {
			LibGodotLogger *logger = memnew(LibGodotLogger);
			logger->set_callback_function(p_log_func, p_log_data);
			os->add_logger(logger);
		}

		Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
		if (err != OK) {
			return;
		}

		instance = memnew(GodotInstance);
		if (!instance->initialize(p_init_func, &callbacks)) {
			memdelete(instance);
			instance = nullptr;
			os->print("GodotInstance initialization error occurred");
			return;
		}

		os->initialize_modules();

		instance->set_executor(executor);
	};
	if (executor != nullptr) {
		executor->sync(init);
	} else {
		init();
	}

	return (GDExtensionObjectPtr)instance;
}

void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = (GodotInstance *)p_godot_instance;
	if (instance == godot_instance) {
		const std::function<void()> deinit = [godot_instance]() {
			godot_instance->stop();
			memdelete(godot_instance);
			instance = nullptr;
			Main::cleanup(true);
			delete OS_IOS::get_singleton();
		};

		TaskExecutor *executor = instance->get_executor();
		if (executor != nullptr) {
			executor->sync(deinit);
			delete executor;
		} else {
			deinit();
		}
	}
}
