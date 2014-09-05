/*************************************************************************/
/*  script_debugger_local.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "os/file_access.h"

void ScriptDebuggerLocal::print_source(ScriptLanguage *p_script,int p_frame,bool p_print_source,bool p_list_source) {
	String source=p_script->debug_get_stack_level_source(p_frame);
	int line=p_script->debug_get_stack_level_line(p_frame);

	print_line("*Frame "+itos(p_frame)+" - "+source+":"+itos(line)+" in function '"+p_script->debug_get_stack_level_function(p_frame)+"'");

	if(!p_print_source)
		return;

	Error err;
	FileAccess *f=FileAccess::open(source,FileAccess::READ,&err);
	if(f!=NULL) {

		// Read source lines
		Vector<String> lines;
		while(!f->eof_reached())
			lines.push_back(f->get_line());
		f->close();

		int line_from=line;
		int line_to=line;

		if(p_list_source) {
			line_from = MAX(1, line - 5);
			line_to = MIN(line + 5, lines.size());
		}

		for(int n = line_from; n <= line_to; n++) {
			print_line(((n==line)?"-> ":"   ")+String::num(n)+":"+lines[n - 1]);
		}
	}
}

void ScriptDebuggerLocal::debug(ScriptLanguage *p_script,bool p_can_continue) {

	print_line("Debugger Break, Reason: '"+p_script->debug_get_error()+"'");
	print_source(p_script,0,true);
	//print_line("Enter \"help\" for assistance.");
	int current_frame=0;
	int total_frames=p_script->debug_get_stack_level_count();
	while(true) {

		OS::get_singleton()->print("debug> ");
		String line = OS::get_singleton()->get_stdin_string().strip_edges();

		if (line=="") {
			print_line("Debugger Break, Reason: '"+p_script->debug_get_error()+"'");
			print_source(p_script,current_frame,true);
			//print_line("Enter \"help\" for assistance.");
		} else if (line=="c" || line=="continue")
			break;
		else if (line=="l" || line=="list") {
			print_source(p_script,current_frame,true,true);
		}
		else if (line=="bt" || line=="breakpoint") {

			for(int i=0;i<total_frames;i++) {

				String cfi=(current_frame==i)?"*":" "; //current frame indicator
				print_source(p_script,i,false);
			}

		} else if (line.begins_with("fr") || line.begins_with("frame")) {

			if (line.get_slice_count(" ")==1) {
				print_source(p_script,current_frame,true);
			} else {
				int frame = line.get_slice(" ",1).to_int();
				if (frame<0 || frame >=total_frames) {
					print_line("Error: Invalid frame.");
				} else {
					current_frame=frame;
					print_source(p_script,frame,true);
				}
			}

		} else if (line=="lv" || line=="locals") {

			List<String> locals;
			List<Variant> values;
			p_script->debug_get_stack_level_locals(current_frame,&locals, &values);
			List<Variant>::Element* V = values.front();
			for (List<String>::Element *E=locals.front();E;E=E->next()) {
				print_line(E->get() + ": " + String(V->get()));
				V = V->next();
			}

		} else if (line=="gv" || line=="globals") {

			List<String> locals;
			List<Variant> values;
			p_script->debug_get_globals(&locals, &values);
			List<Variant>::Element* V = values.front();
			for (List<String>::Element *E=locals.front();E;E=E->next()) {
				print_line(E->get() + ": " + String(V->get()));
				V = V->next();
			}

		} else if (line=="mv" || line=="members") {

			List<String> locals;
			List<Variant> values;
			p_script->debug_get_stack_level_members(current_frame,&locals, &values);
			List<Variant>::Element* V = values.front();
			for (List<String>::Element *E=locals.front();E;E=E->next()) {
				print_line(E->get() + ": " + String(V->get()));
				V = V->next();
			}

		} else if (line.begins_with("p") || line.begins_with("print")) {

			if (line.get_slice_count(" ")<1) {
				print_line("Usage: print <expre>");
			} else {

				String expr = line.get_slice(" ",1);
				String res = p_script->debug_parse_stack_level_expression(current_frame,expr);
				print_line(res);
			}

		} else if (line=="s" || line=="step") {

			set_depth(-1);
			set_lines_left(1);
			break;
		} else if (line.begins_with("n") || line.begins_with("next")) {

			set_depth(0);
			set_lines_left(1);
			break;
		} else if (line.begins_with("br") || line.begins_with("break")) {

			if (line.get_slice_count(" ")<=1) {
				//show breakpoints
			} else {


				String bppos=line.get_slice(" ",1);
				String source=bppos.get_slice(":",0).strip_edges();
				int line=bppos.get_slice(":",1).strip_edges().to_int();

				source = breakpoint_find_source(source);

				insert_breakpoint(line,source);

				print_line("BreakPoint at "+source+":"+itos(line));
			}

		} else if (line.begins_with("delete")) {

			if (line.get_slice_count(" ")<=1) {
				clear_breakpoints();
			} else {

				String bppos=line.get_slice(" ",1);
				String source=bppos.get_slice(":",0).strip_edges();
				int line=bppos.get_slice(":",1).strip_edges().to_int();

				source = breakpoint_find_source(source);

				remove_breakpoint(line,source);

				print_line("Removed BreakPoint at "+source+":"+itos(line));

			}

		} else if (line=="h" || line=="help") {

			print_line("Built-In Debugger command list:\n");
			print_line("\tc,continue :\t\t Continue execution.");
			print_line("\tbt,backtrace :\t\t Show stack trace (frames).");
			print_line("\tfr,frame <frame>:\t Change current frame.");
			print_line("\tl,list:\t List current source codes.");
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

void ScriptDebuggerLocal::send_message(const String& p_message, const Array &p_args) {

	print_line("MESSAGE: '"+p_message+"' - "+String(Variant(p_args)));
}

ScriptDebuggerLocal::ScriptDebuggerLocal() {

}
