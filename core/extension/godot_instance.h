/**************************************************************************/
/*  godot_instance.h                                                      */
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

#pragma once

#include "core/extension/gdextension_interface.h"
#include "core/object/class_db.h"
#include "core/object/object.h"

#include <functional>

typedef void *CallbackData;
typedef void *ExecutorData;
typedef void (*InvokeCallback)(CallbackData p_data);
typedef void (*InvokeCallbackFunction)(InvokeCallback p_callback, CallbackData p_callback_data, ExecutorData p_executor_data);

class GodotInstance;
class GodotInstanceCallbacks {
public:
	virtual void before_setup2(GodotInstance *p_instance) {}
	virtual void before_start(GodotInstance *p_instance) {}
	virtual void after_start(GodotInstance *p_instance) {}
	virtual void focus_out(GodotInstance *p_instance) {}
	virtual void focus_in(GodotInstance *p_instance) {}
	virtual void pause(GodotInstance *p_instance) {}
	virtual void resume(GodotInstance *p_instance) {}
};

class TaskExecutor {
	InvokeCallbackFunction async_func;
	ExecutorData async_data;
	InvokeCallbackFunction sync_func;
	ExecutorData sync_data;

public:
	TaskExecutor(InvokeCallbackFunction p_async_func, ExecutorData p_async_data, InvokeCallbackFunction p_sync_func, ExecutorData p_sync_data);
	void sync(std::function<void()> p_callback);
	void async(std::function<void()> p_callback);

	static void invokeCallback(void *p_callback);
};

class GodotInstance : public Object {
	GDCLASS(GodotInstance, Object);

	static void _bind_methods();

	bool started = false;

	GodotInstanceCallbacks *callbacks = nullptr;
	TaskExecutor *executor = nullptr;

public:
	GodotInstance();
	~GodotInstance();

	bool initialize(GDExtensionInitializationFunction p_init_func, GodotInstanceCallbacks *p_callbacks = nullptr);

	bool start();
	bool is_started();
	bool iteration();
	void stop();

	void focus_out();
	void focus_in();
	void pause();
	void resume();

	void set_executor(TaskExecutor *p_executor);
	TaskExecutor *get_executor();
	void execute(Callable p_callback, bool p_async);
};
