/*************************************************************************/
/*  script_debugger_local.cpp                                            */
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
#include "script_debugger_local.h"

#include "os/os.h"

void ScriptDebuggerLocal::debug(ScriptLanguage *p_script, bool p_can_continue) {

	print_line("Debugger Break, Reason: '" + p_script->debug_get_error() + "'");
	print_line("*Frame " + itos(0) + " - " + p_script->debug_get_stack_level_source(0) + ":" + itos(p_script->debug_get_stack_level_line(0)) + " in function '" + p_script->debug_get_stack_level_function(0) + "'");
	print_line("Enter \"help\" for assistance.");
	int current_frame = 0;
	int total_frames = p_script->debug_get_stack_level_count();
	while (true) {

		OS::get_singleton()->print("debug> ");
		String line = OS::get_singleton()->get_stdin_string().strip_edges();

		if (line == "") {
			print_line("Debugger Break, Reason: '" + p_script->debug_get_error() + "'");
			print_line("*Frame " + itos(current_frame) + " - " + p_script->debug_get_stack_level_source(current_frame) + ":" + itos(p_script->debug_get_stack_level_line(current_frame)) + " in function '" + p_script->debug_get_stack_level_function(current_frame) + "'");
			print_line("Enter \"help\" for assistance.");
		} else if (line == "c" || line == "continue")
			break;
		else if (line == "bt" || line == "breakpoint") {

			for (int i = 0; i < total_frames; i++) {

				String cfi = (current_frame == i) ? "*" : " "; //current frame indicator
				print_line(cfi + "Frame " + itos(i) + " - " + p_script->debug_get_stack_level_source(i) + ":" + itos(p_script->debug_get_stack_level_line(i)) + " in function '" + p_script->debug_get_stack_level_function(i) + "'");
			}

		} else if (line.begins_with("fr") || line.begins_with("frame")) {

			if (line.get_slice_count(" ") == 1) {
				print_line("*Frame " + itos(current_frame) + " - " + p_script->debug_get_stack_level_source(current_frame) + ":" + itos(p_script->debug_get_stack_level_line(current_frame)) + " in function '" + p_script->debug_get_stack_level_function(current_frame) + "'");
			} else {
				int frame = line.get_slicec(' ', 1).to_int();
				if (frame < 0 || frame >= total_frames) {
					print_line("Error: Invalid frame.");
				} else {
					current_frame = frame;
					print_line("*Frame " + itos(frame) + " - " + p_script->debug_get_stack_level_source(frame) + ":" + itos(p_script->debug_get_stack_level_line(frame)) + " in function '" + p_script->debug_get_stack_level_function(frame) + "'");
				}
			}

		} else if (line == "lv" || line == "locals") {

			List<String> locals;
			List<Variant> values;
			p_script->debug_get_stack_level_locals(current_frame, &locals, &values);
			List<Variant>::Element *V = values.front();
			for (List<String>::Element *E = locals.front(); E; E = E->next()) {
				print_line(E->get() + ": " + String(V->get()));
				V = V->next();
			}

		} else if (line == "gv" || line == "globals") {

			List<String> locals;
			List<Variant> values;
			p_script->debug_get_globals(&locals, &values);
			List<Variant>::Element *V = values.front();
			for (List<String>::Element *E = locals.front(); E; E = E->next()) {
				print_line(E->get() + ": " + String(V->get()));
				V = V->next();
			}

		} else if (line == "mv" || line == "members") {

			List<String> locals;
			List<Variant> values;
			p_script->debug_get_stack_level_members(current_frame, &locals, &values);
			List<Variant>::Element *V = values.front();
			for (List<String>::Element *E = locals.front(); E; E = E->next()) {
				print_line(E->get() + ": " + String(V->get()));
				V = V->next();
			}

		} else if (line.begins_with("p") || line.begins_with("print")) {

			if (line.get_slice_count(" ") <= 1) {
				print_line("Usage: print <expre>");
			} else {

				String expr = line.get_slicec(' ', 2);
				String res = p_script->debug_parse_stack_level_expression(current_frame, expr);
				print_line(res);
			}

		} else if (line == "s" || line == "step") {

			set_depth(-1);
			set_lines_left(1);
			break;
		} else if (line.begins_with("n") || line.begins_with("next")) {

			set_depth(0);
			set_lines_left(1);
			break;
		} else if (line.begins_with("br") || line.begins_with("break")) {

			if (line.get_slice_count(" ") <= 1) {
				//show breakpoints
			} else {

				String bppos = line.get_slicec(' ', 1);
				String source = bppos.get_slicec(':', 0).strip_edges();
				int line = bppos.get_slicec(':', 1).strip_edges().to_int();

				source = breakpoint_find_source(source);

				insert_breakpoint(line, source);

				print_line("BreakPoint at " + source + ":" + itos(line));
			}

		} else if (line.begins_with("delete")) {

			if (line.get_slice_count(" ") <= 1) {
				clear_breakpoints();
			} else {

				String bppos = line.get_slicec(' ', 1);
				String source = bppos.get_slicec(':', 0).strip_edges();
				int line = bppos.get_slicec(':', 1).strip_edges().to_int();

				source = breakpoint_find_source(source);

				remove_breakpoint(line, source);

				print_line("Removed BreakPoint at " + source + ":" + itos(line));
			}

		} else if (line == "h" || line == "help") {

			print_line("Built-In Debugger command list:\n");
			print_line("\tc,continue :\t\t Continue execution.");
			print_line("\tbt,backtrace :\t\t Show stack trace (frames).");
			print_line("\tfr,frame <frame>:\t Change current frame.");
			print_line("\tlv,locals :\t\t Show local variables for current frame.");
			print_line("\tmv,members :\t\t Show member variables for \"this\" in frame.");
			print_line("\tgv,globals :\t\t Show global variables.");
			print_line("\tp,print <expr> :\t Execute and print variable in expression.");
			print_line("\ts,step :\t\t Step to next line.");
			print_line("\tn,next :\t\t Next line.");
			print_line("\tbr,break source:line :\t Place a breakpoint.");
			print_line("\tdelete [source:line]:\t\t Delete one/all breakpoints.");
		} else {
			print_line("Error: Invalid command, enter \"help\" for assistance.");
		}
	}
}

struct _ScriptDebuggerLocalProfileInfoSort {

	bool operator()(const ScriptLanguage::ProfilingInfo &A, const ScriptLanguage::ProfilingInfo &B) const {
		return A.total_time > B.total_time;
	}
};

void ScriptDebuggerLocal::profiling_set_frame_times(float p_frame_time, float p_idle_time, float p_physics_time, float p_physics_frame_time) {

	frame_time = p_frame_time;
	idle_time = p_idle_time;
	physics_time = p_physics_time;
	physics_frame_time = p_physics_frame_time;
}

void ScriptDebuggerLocal::idle_poll() {

	if (!profiling)
		return;

	uint64_t diff = OS::get_singleton()->get_ticks_usec() - idle_accum;

	if (diff < 1000000) //show every one second
		return;

	idle_accum = OS::get_singleton()->get_ticks_usec();

	int ofs = 0;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ofs += ScriptServer::get_language(i)->profiling_get_frame_data(&pinfo[ofs], pinfo.size() - ofs);
	}

	SortArray<ScriptLanguage::ProfilingInfo, _ScriptDebuggerLocalProfileInfoSort> sort;
	sort.sort(pinfo.ptr(), ofs);

	//falta el frame time

	uint64_t script_time_us = 0;

	for (int i = 0; i < ofs; i++) {

		script_time_us += pinfo[i].self_time;
	}

	float script_time = USEC_TO_SEC(script_time_us);

	float total_time = frame_time;

	//print script total

	print_line("FRAME: total: " + rtos(frame_time) + " script: " + rtos(script_time) + "/" + itos(script_time * 100 / total_time) + " %");

	for (int i = 0; i < ofs; i++) {

		print_line(itos(i) + ":" + pinfo[i].signature);
		float tt = USEC_TO_SEC(pinfo[i].total_time);
		float st = USEC_TO_SEC(pinfo[i].self_time);
		print_line("\ttotal: " + rtos(tt) + "/" + itos(tt * 100 / total_time) + " % \tself: " + rtos(st) + "/" + itos(st * 100 / total_time) + " % tcalls: " + itos(pinfo[i].call_count));
	}
}

void ScriptDebuggerLocal::profiling_start() {

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->profiling_start();
	}

	print_line("BEGIN PROFILING");
	profiling = true;
	pinfo.resize(32768);
	frame_time = 0;
	physics_time = 0;
	idle_time = 0;
	physics_frame_time = 0;
}

void ScriptDebuggerLocal::profiling_end() {

	int ofs = 0;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ofs += ScriptServer::get_language(i)->profiling_get_accumulated_data(&pinfo[ofs], pinfo.size() - ofs);
	}

	SortArray<ScriptLanguage::ProfilingInfo, _ScriptDebuggerLocalProfileInfoSort> sort;
	sort.sort(pinfo.ptr(), ofs);

	uint64_t total_us = 0;
	for (int i = 0; i < ofs; i++) {
		total_us += pinfo[i].self_time;
	}

	float total_time = total_us / 1000000.0;

	for (int i = 0; i < ofs; i++) {

		print_line(itos(i) + ":" + pinfo[i].signature);
		float tt = USEC_TO_SEC(pinfo[i].total_time);
		float st = USEC_TO_SEC(pinfo[i].self_time);
		print_line("\ttotal_ms: " + rtos(tt) + "\tself_ms: " + rtos(st) + "total%: " + itos(tt * 100 / total_time) + "\tself%: " + itos(st * 100 / total_time) + "\tcalls: " + itos(pinfo[i].call_count));
	}

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->profiling_stop();
	}

	profiling = false;
}

void ScriptDebuggerLocal::send_message(const String &p_message, const Array &p_args) {

	print_line("MESSAGE: '" + p_message + "' - " + String(Variant(p_args)));
}

ScriptDebuggerLocal::ScriptDebuggerLocal() {

	profiling = false;
	idle_accum = OS::get_singleton()->get_ticks_usec();
}
