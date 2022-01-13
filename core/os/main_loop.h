/*************************************************************************/
/*  main_loop.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MAIN_LOOP_H
#define MAIN_LOOP_H

#include "core/os/input_event.h"
#include "core/reference.h"
#include "core/script_language.h"

class MainLoop : public Object {
	GDCLASS(MainLoop, Object);
	OBJ_CATEGORY("Main Loop");

	Ref<Script> init_script;

protected:
	static void _bind_methods();

public:
	enum {
		//make sure these are replicated in Node
		NOTIFICATION_WM_MOUSE_ENTER = 1002,
		NOTIFICATION_WM_MOUSE_EXIT = 1003,
		NOTIFICATION_WM_FOCUS_IN = 1004,
		NOTIFICATION_WM_FOCUS_OUT = 1005,
		NOTIFICATION_WM_QUIT_REQUEST = 1006,
		NOTIFICATION_WM_GO_BACK_REQUEST = 1007,
		NOTIFICATION_WM_UNFOCUS_REQUEST = 1008,
		NOTIFICATION_OS_MEMORY_WARNING = 1009,
		NOTIFICATION_TRANSLATION_CHANGED = 1010,
		NOTIFICATION_WM_ABOUT = 1011,
		NOTIFICATION_CRASH = 1012,
		NOTIFICATION_OS_IME_UPDATE = 1013,
		NOTIFICATION_APP_RESUMED = 1014,
		NOTIFICATION_APP_PAUSED = 1015,
	};

	virtual void input_event(const Ref<InputEvent> &p_event);
	virtual void input_text(const String &p_text);

	virtual void init();
	virtual bool iteration(float p_time);
	virtual bool idle(float p_time);
	virtual void finish();

	virtual void drop_files(const Vector<String> &p_files, int p_from_screen = 0);
	virtual void global_menu_action(const Variant &p_id, const Variant &p_meta);

	void set_init_script(const Ref<Script> &p_init_script);

	MainLoop();
	virtual ~MainLoop();
};

#endif
