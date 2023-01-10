/**************************************************************************/
/*  script_debugger_local.cpp                                             */
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

#include "script_debugger_local.h"

#include "core/os/os.h"
#include "scene/main/scene_tree.h"

void ScriptDebuggerLocal::debug(ScriptLanguage *p_script, bool p_can_continue, bool p_is_error_breakpoint) {
	if (!target_function.empty()) {
		String current_function = p_script->debug_get_stack_level_function(0);
		if (current_function != target_function) {
			set_depth(0);
			set_lines_left(1);
			return;
		}
		target_function = "";
	}

	print_line("\nDebugger Break, Reason: '" + p_script->debug_get_error() + "'");
	print_line("*Frame " + itos(0) + " - " + p_script->debug_get_stack_level_source(0) + ":" + itos(p_script->debug_get_stack_level_line(0)) + " in function '" + p_script->debug_get_stack_level_function(0) + "'");
	print_line("Enter \"help\" for assistance.");
	int current_frame = 0;
	int total_frames = p_script->debug_get_stack_level_count();
	while (true) {
		OS::get_singleton()->print("debug> ");
		String line = OS::get_singleton()->get_stdin_string().strip_edges();

		// Cache options
		String variable_prefix = options["variable_prefix"];

		if (line == "") {
			print_line("\nDebugger Break, Reason: '" + p_script->debug_get_error() + "'");
			print_line("*Frame " + itos(current_frame) + " - " + p_script->debug_get_stack_level_source(current_frame) + ":" + itos(p_script->debug_get_stack_level_line(current_frame)) + " in function '" + p_script->debug_get_stack_level_function(current_frame) + "'");
			print_line("Enter \"help\" for assistance.");
		} else if (line == "c" || line == "continue") {
			break;
		} else if (line == "bt" || line == "breakpoint") {
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

		} else if (line.begins_with("set")) {
			if (line.get_slice_count(" ") == 1) {
				for (Map<String, String>::Element *E = options.front(); E; E = E->next()) {
					print_line("\t" + E->key() + "=" + E->value());
				}

			} else {
				String key_value = line.get_slicec(' ', 1);
				int value_pos = key_value.find("=");

				if (value_pos < 0) {
					print_line("Error: Invalid set format. Use: set key=value");
				} else {
					String key = key_value.left(value_pos);

					if (!options.has(key)) {
						print_line("Error: Unknown option " + key);
					} else {
						// Allow explicit tab character
						String value = key_value.right(value_pos + 1).replace("\\t", "\t");

						options[key] = value;
					}
				}
			}

		} else if (line == "lv" || line == "locals") {
			List<String> locals;
			List<Variant> values;
			p_script->debug_get_stack_level_locals(current_frame, &locals, &values);
			print_variables(locals, values, variable_prefix);

		} else if (line == "gv" || line == "globals") {
			List<String> globals;
			List<Variant> values;
			p_script->debug_get_globals(&globals, &values);
			print_variables(globals, values, variable_prefix);

		} else if (line == "mv" || line == "members") {
			List<String> members;
			List<Variant> values;
			p_script->debug_get_stack_level_members(current_frame, &members, &values);
			print_variables(members, values, variable_prefix);

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
		} else if (line == "n" || line == "next") {
			set_depth(0);
			set_lines_left(1);
			break;
		} else if (line == "fin" || line == "finish") {
			String current_function = p_script->debug_get_stack_level_function(0);

			for (int i = 0; i < total_frames; i++) {
				target_function = p_script->debug_get_stack_level_function(i);
				if (target_function != current_function) {
					set_depth(0);
					set_lines_left(1);
					return;
				}
			}

			print_line("Error: Reached last frame.");
			target_function = "";

		} else if (line.begins_with("br") || line.begins_with("break")) {
			if (line.get_slice_count(" ") <= 1) {
				const Map<int, Set<StringName>> &breakpoints = get_breakpoints();
				if (breakpoints.size() == 0) {
					print_line("No Breakpoints.");
					continue;
				}

				print_line("Breakpoint(s): " + itos(breakpoints.size()));
				for (Map<int, Set<StringName>>::Element *E = breakpoints.front(); E; E = E->next()) {
					print_line("\t" + String(E->value().front()->get()) + ":" + itos(E->key()));
				}

			} else {
				Pair<String, int> breakpoint = to_breakpoint(line);

				String source = breakpoint.first;
				int linenr = breakpoint.second;

				if (source.empty()) {
					continue;
				}

				insert_breakpoint(linenr, source);

				print_line("Added breakpoint at " + source + ":" + itos(linenr));
			}

		} else if (line == "q" || line == "quit") {
			// Do not stop again on quit
			clear_breakpoints();
			ScriptDebugger::get_singleton()->set_depth(-1);
			ScriptDebugger::get_singleton()->set_lines_left(-1);

			SceneTree::get_singleton()->quit();
			break;
		} else if (line.begins_with("delete")) {
			if (line.get_slice_count(" ") <= 1) {
				clear_breakpoints();
			} else {
				Pair<String, int> breakpoint = to_breakpoint(line);

				String source = breakpoint.first;
				int linenr = breakpoint.second;

				if (source.empty()) {
					continue;
				}

				remove_breakpoint(linenr, source);

				print_line("Removed breakpoint at " + source + ":" + itos(linenr));
			}

		} else if (line == "h" || line == "help") {
			print_line("Built-In Debugger command list:\n");
			print_line("\tc,continue\t\t Continue execution.");
			print_line("\tbt,backtrace\t\t Show stack trace (frames).");
			print_line("\tfr,frame <frame>:\t Change current frame.");
			print_line("\tlv,locals\t\t Show local variables for current frame.");
			print_line("\tmv,members\t\t Show member variables for \"this\" in frame.");
			print_line("\tgv,globals\t\t Show global variables.");
			print_line("\tp,print <expr>\t\t Execute and print variable in expression.");
			print_line("\ts,step\t\t\t Step to next line.");
			print_line("\tn,next\t\t\t Next line.");
			print_line("\tfin,finish\t\t Step out of current frame.");
			print_line("\tbr,break [source:line]\t List all breakpoints or place a breakpoint.");
			print_line("\tdelete [source:line]:\t Delete one/all breakpoints.");
			print_line("\tset [key=value]:\t List all options, or set one.");
			print_line("\tq,quit\t\t\t Quit application.");
		} else {
			print_line("Error: Invalid command, enter \"help\" for assistance.");
		}
	}
}

void ScriptDebuggerLocal::print_variables(const List<String> &names, const List<Variant> &values, const String &variable_prefix) {
	String value;
	Vector<String> value_lines;
	const List<Variant>::Element *V = values.front();
	for (const List<String>::Element *E = names.front(); E; E = E->next()) {
		value = String(V->get());

		if (variable_prefix.empty()) {
			print_line(E->get() + ": " + String(V->get()));
		} else {
			print_line(E->get() + ":");
			value_lines = value.split("\n");
			for (int i = 0; i < value_lines.size(); ++i) {
				print_line(variable_prefix + value_lines[i]);
			}
		}

		V = V->next();
	}
}

Pair<String, int> ScriptDebuggerLocal::to_breakpoint(const String &p_line) {
	String breakpoint_part = p_line.get_slicec(' ', 1);
	Pair<String, int> breakpoint;

	int last_colon = breakpoint_part.rfind(":");
	if (last_colon < 0) {
		print_line("Error: Invalid breakpoint format. Expected [source:line]");
		return breakpoint;
	}

	breakpoint.first = breakpoint_find_source(breakpoint_part.left(last_colon).strip_edges());
	breakpoint.second = breakpoint_part.right(last_colon).strip_edges().to_int();

	return breakpoint;
}

struct _ScriptDebuggerLocalProfileInfoSort {
	bool operator()(const ScriptLanguage::ProfilingInfo &A, const ScriptLanguage::ProfilingInfo &B) const {
		return A.total_time > B.total_time;
	}
};

void ScriptDebuggerLocal::profiling_set_frame_times(float p_frame_time, float p_process_time, float p_physics_time, float p_physics_frame_time) {
	frame_time = p_frame_time;
	process_time = p_process_time;
	physics_time = p_physics_time;
	physics_frame_time = p_physics_frame_time;
}

void ScriptDebuggerLocal::idle_poll() {
	if (!profiling) {
		return;
	}

	uint64_t diff = OS::get_singleton()->get_ticks_usec() - idle_accum;

	if (diff < 1000000) { //show every one second
		return;
	}

	idle_accum = OS::get_singleton()->get_ticks_usec();

	int ofs = 0;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ofs += ScriptServer::get_language(i)->profiling_get_frame_data(&pinfo.write[ofs], pinfo.size() - ofs);
	}

	SortArray<ScriptLanguage::ProfilingInfo, _ScriptDebuggerLocalProfileInfoSort> sort;
	sort.sort(pinfo.ptrw(), ofs);

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
	process_time = 0;
	physics_frame_time = 0;
}

void ScriptDebuggerLocal::profiling_end() {
	int ofs = 0;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ofs += ScriptServer::get_language(i)->profiling_get_accumulated_data(&pinfo.write[ofs], pinfo.size() - ofs);
	}

	SortArray<ScriptLanguage::ProfilingInfo, _ScriptDebuggerLocalProfileInfoSort> sort;
	sort.sort(pinfo.ptrw(), ofs);

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
	// This needs to be cleaned up entirely.
	// print_line("MESSAGE: '" + p_message + "' - " + String(Variant(p_args)));
}

void ScriptDebuggerLocal::send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, ErrorHandlerType p_type, const Vector<ScriptLanguage::StackInfo> &p_stack_info) {
	print_line("ERROR: '" + (p_descr.empty() ? p_err : p_descr) + "'");
}

ScriptDebuggerLocal::ScriptDebuggerLocal() {
	profiling = false;
	idle_accum = OS::get_singleton()->get_ticks_usec();
	options["variable_prefix"] = "";
}
