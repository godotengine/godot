/*************************************************************************/
/*  script_debugger_local.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SCRIPT_DEBUGGER_LOCAL_H
#define SCRIPT_DEBUGGER_LOCAL_H

#include "script_language.h"

class ScriptDebuggerLocal : public ScriptDebugger {

	bool profiling;
	float frame_time, idle_time, physics_time, physics_frame_time;
	uint64_t idle_accum;

	Vector<ScriptLanguage::ProfilingInfo> pinfo;

public:
	void debug(ScriptLanguage *p_script, bool p_can_continue);
	virtual void send_message(const String &p_message, const Array &p_args);

	virtual bool is_profiling() const { return profiling; }
	virtual void add_profiling_frame_data(const StringName &p_name, const Array &p_data) {}

	virtual void idle_poll();

	virtual void profiling_start();
	virtual void profiling_end();
	virtual void profiling_set_frame_times(float p_frame_time, float p_idle_time, float p_physics_time, float p_physics_frame_time);

	ScriptDebuggerLocal();
};

#endif // SCRIPT_DEBUGGER_LOCAL_H
