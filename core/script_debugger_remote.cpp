/*************************************************************************/
/*  script_debugger_remote.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#include "script_debugger_remote.h"
#include "os/os.h"
#include "io/ip.h"
#include "globals.h"

void ScriptDebuggerRemote::_send_video_memory() {

	List<ResourceUsage> usage;
	if (resource_usage_func)
		resource_usage_func(&usage);

	usage.sort();

	packet_peer_stream->put_var("message:video_mem");
	packet_peer_stream->put_var(usage.size()*4);


	for(List<ResourceUsage>::Element *E=usage.front();E;E=E->next()) {

		packet_peer_stream->put_var(E->get().path);
		packet_peer_stream->put_var(E->get().type);
		packet_peer_stream->put_var(E->get().format);
		packet_peer_stream->put_var(E->get().vram);
	}

}

Error ScriptDebuggerRemote::connect_to_host(const String& p_host,uint16_t p_port) {


    IP_Address ip;
    if (p_host.is_valid_ip_address())
	    ip=p_host;
    else
	    ip = IP::get_singleton()->resolve_hostname(p_host);


    int port = p_port;

    int tries = 3;
    tcp_client->connect(ip, port);

    while (tries--) {

        if (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
            break;
        } else {

            OS::get_singleton()->delay_usec(1000000);
            print_line("Remote Debugger: Connection failed with status: " + String::num(tcp_client->get_status())+"'', retrying in 1 sec.");
        };
    };

    if (tcp_client->get_status() != StreamPeerTCP::STATUS_CONNECTED) {

        print_line("Remote Debugger: Unable to connect");
        return FAILED;
    };

//    print_line("Remote Debugger: Connection OK!");
    packet_peer_stream->set_stream_peer(tcp_client);

    return OK;
}

static int _ScriptDebuggerRemote_found_id=0;
static Object* _ScriptDebuggerRemote_find=NULL;
static void _ScriptDebuggerRemote_debug_func(Object *p_obj) {

	if (_ScriptDebuggerRemote_find==p_obj) {
		_ScriptDebuggerRemote_found_id=p_obj->get_instance_ID();
	}
}

static ObjectID safe_get_instance_id(const Variant& p_v) {

	Object *o = p_v;
	if (o==NULL)
		return 0;
	else {

		REF r = p_v;
		if (r.is_valid()) {

			return r->get_instance_ID();
		} else {


			_ScriptDebuggerRemote_found_id=0;
			_ScriptDebuggerRemote_find=NULL;
			ObjectDB::debug_objects(_ScriptDebuggerRemote_debug_func);
			return _ScriptDebuggerRemote_found_id;

		}
	}
}

void ScriptDebuggerRemote::debug(ScriptLanguage *p_script,bool p_can_continue) {

	if (!tcp_client->is_connected()) {
		ERR_EXPLAIN("Script Debugger failed to connect, but being used anyway.");
		ERR_FAIL();
	}

	packet_peer_stream->put_var("debug_enter");
	packet_peer_stream->put_var(2);
	packet_peer_stream->put_var(p_can_continue);
	packet_peer_stream->put_var(p_script->debug_get_error());


	while(true) {

		_get_output();

		if (packet_peer_stream->get_available_packet_count()>0) {

			Variant var;
			Error err = packet_peer_stream->get_var(var);
			ERR_CONTINUE( err != OK);
			ERR_CONTINUE( var.get_type()!=Variant::ARRAY );

			Array cmd = var;

			ERR_CONTINUE( cmd.size()==0);
			ERR_CONTINUE( cmd[0].get_type()!=Variant::STRING );

			String command = cmd[0];



			if (command=="get_stack_dump") {

				packet_peer_stream->put_var("stack_dump");
				int slc = p_script->debug_get_stack_level_count();
				packet_peer_stream->put_var( slc );

				for(int i=0;i<slc;i++) {

					Dictionary d;
					d["file"]=p_script->debug_get_stack_level_source(i);
					d["line"]=p_script->debug_get_stack_level_line(i);
					d["function"]=p_script->debug_get_stack_level_function(i);
					//d["id"]=p_script->debug_get_stack_level_
					d["id"]=0;

					packet_peer_stream->put_var( d );
				}

			} else if (command=="get_stack_frame_vars") {

				cmd.remove(0);
				ERR_CONTINUE( cmd.size()!=1 );
				int lv = cmd[0];

				List<String> members;
				List<Variant> member_vals;

				p_script->debug_get_stack_level_members(lv,&members,&member_vals);



				ERR_CONTINUE( members.size() !=member_vals.size() );

				List<String> locals;
				List<Variant> local_vals;

				p_script->debug_get_stack_level_locals(lv,&locals,&local_vals);

				ERR_CONTINUE( locals.size() !=local_vals.size() );

				packet_peer_stream->put_var("stack_frame_vars");
				packet_peer_stream->put_var(2+locals.size()*2+members.size()*2);

				{ //members
					packet_peer_stream->put_var(members.size());

					List<String>::Element *E=members.front();
					List<Variant>::Element *F=member_vals.front();

					while(E) {

						if (F->get().get_type()==Variant::OBJECT) {
							packet_peer_stream->put_var("*"+E->get());
							packet_peer_stream->put_var(safe_get_instance_id(F->get()));
						} else {
							packet_peer_stream->put_var(E->get());
							packet_peer_stream->put_var(F->get());
						}

						E=E->next();
						F=F->next();
					}

				}


				{ //locals
					packet_peer_stream->put_var(locals.size());

					List<String>::Element *E=locals.front();
					List<Variant>::Element *F=local_vals.front();

					while(E) {

						if (F->get().get_type()==Variant::OBJECT) {
							packet_peer_stream->put_var("*"+E->get());
							packet_peer_stream->put_var(safe_get_instance_id(F->get()));
						} else {
							packet_peer_stream->put_var(E->get());
							packet_peer_stream->put_var(F->get());
						}

						E=E->next();
						F=F->next();
					}

				}



			} else if (command=="step") {

				set_depth(-1);
				set_lines_left(1);
				break;
			} else if (command=="next") {

				set_depth(0);
				set_lines_left(1);
				break;

			} else if (command=="continue") {

				set_depth(-1);
				set_lines_left(-1);
				break;
			} else if (command=="break") {
				ERR_PRINT("Got break when already broke!");
				break;
			} else if (command=="request_scene_tree") {

				if (request_scene_tree)
					request_scene_tree(request_scene_tree_ud);

			} else if (command=="request_video_mem") {

				_send_video_memory();
			} else if (command=="breakpoint") {

				bool set = cmd[3];
				if (set)
					insert_breakpoint(cmd[2],cmd[1]);
				else
					remove_breakpoint(cmd[2],cmd[1]);

			} else {
				_parse_live_edit(cmd);
			}



		} else {
			OS::get_singleton()->delay_usec(10000);
		}

	}

	packet_peer_stream->put_var("debug_exit");
	packet_peer_stream->put_var(0);

}


void ScriptDebuggerRemote::_get_output() {

	mutex->lock();
	if (output_strings.size()) {

		locking=true;
		packet_peer_stream->put_var("output");
		packet_peer_stream->put_var(output_strings .size());

		while(output_strings.size()) {

			packet_peer_stream->put_var(output_strings.front()->get());
			output_strings.pop_front();
		}
		locking=false;

	}

	while (messages.size()) {
		locking=true;
		packet_peer_stream->put_var("message:"+messages.front()->get().message);
		packet_peer_stream->put_var(messages.front()->get().data.size());
		for(int i=0;i<messages.front()->get().data.size();i++) {
			packet_peer_stream->put_var(messages.front()->get().data[i]);
		}
		messages.pop_front();
		locking=false;
	}

	while (errors.size()) {
		locking=true;
		packet_peer_stream->put_var("error");
		OutputError oe = errors.front()->get();

		packet_peer_stream->put_var(oe.callstack.size()+2);

		Array error_data;

		error_data.push_back(oe.hr);
		error_data.push_back(oe.min);
		error_data.push_back(oe.sec);
		error_data.push_back(oe.msec);
		error_data.push_back(oe.source_func);
		error_data.push_back(oe.source_file);
		error_data.push_back(oe.source_line);
		error_data.push_back(oe.error);
		error_data.push_back(oe.error_descr);
		error_data.push_back(oe.warning);
		packet_peer_stream->put_var(error_data);
		packet_peer_stream->put_var(oe.callstack.size());
		for(int i=0;i<oe.callstack.size();i++) {
			packet_peer_stream->put_var(oe.callstack[i]);

		}

		errors.pop_front();
		locking=false;

	}
	mutex->unlock();
}

void ScriptDebuggerRemote::line_poll() {

	//the purpose of this is just processing events every now and then when the script might get too busy
	//otherwise bugs like infinite loops cant be catched
	if (poll_every%512==0)
		_poll_events();
	poll_every++;

}


void ScriptDebuggerRemote::_err_handler(void* ud,const char* p_func,const char*p_file,int p_line,const char *p_err, const char * p_descr,ErrorHandlerType p_type) {

	if (p_type==ERR_HANDLER_SCRIPT)
		return; //ignore script errors, those go through debugger

	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote*)ud;

	OutputError oe;
	oe.error=p_err;
	oe.error_descr=p_descr;
	oe.source_file=p_file;
	oe.source_line=p_line;
	oe.source_func=p_func;
	oe.warning=p_type==ERR_HANDLER_WARNING;
	uint64_t time = OS::get_singleton()->get_ticks_msec();
	oe.hr=time/3600000;
	oe.min=(time/60000)%60;
	oe.sec=(time/1000)%60;
	oe.msec=time%1000;
	Array cstack;

	Vector<ScriptLanguage::StackInfo> si;

	for(int i=0;i<ScriptServer::get_language_count();i++) {
		si=ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size())
			break;
	}

	cstack.resize(si.size()*2);
	for(int i=0;i<si.size();i++) {
		String path;
		int line=0;
		if (si[i].script.is_valid()) {
			path=si[i].script->get_path();
			line=si[i].line;
		}
		cstack[i*2+0]=path;
		cstack[i*2+1]=line;
	}

	oe.callstack=cstack;


	sdr->mutex->lock();

	if (!sdr->locking && sdr->tcp_client->is_connected()) {

		sdr->errors.push_back(oe);
	}

	sdr->mutex->unlock();
}


bool ScriptDebuggerRemote::_parse_live_edit(const Array& cmd) {

	String cmdstr = cmd[0];
	if (!live_edit_funcs || !cmdstr.begins_with("live_"))
		return false;


	//print_line(Variant(cmd).get_construct_string());
	if (cmdstr=="live_set_root") {

		if (!live_edit_funcs->root_func)
			return true;
		//print_line("root: "+Variant(cmd).get_construct_string());
		live_edit_funcs->root_func(live_edit_funcs->udata,cmd[1],cmd[2]);

	} else if (cmdstr=="live_node_path") {

		if (!live_edit_funcs->node_path_func)
			return true;
		//print_line("path: "+Variant(cmd).get_construct_string());

		live_edit_funcs->node_path_func(live_edit_funcs->udata,cmd[1],cmd[2]);

	} else if (cmdstr=="live_res_path") {

		if (!live_edit_funcs->res_path_func)
			return true;
		live_edit_funcs->res_path_func(live_edit_funcs->udata,cmd[1],cmd[2]);

	} else if (cmdstr=="live_node_prop_res") {
		if (!live_edit_funcs->node_set_res_func)
			return true;

		live_edit_funcs->node_set_res_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_node_prop") {

		if (!live_edit_funcs->node_set_func)
			return true;
		live_edit_funcs->node_set_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_res_prop_res") {

		if (!live_edit_funcs->res_set_res_func)
			return true;
		live_edit_funcs->res_set_res_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_res_prop") {

		if (!live_edit_funcs->res_set_func)
			return true;
		live_edit_funcs->res_set_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_node_call") {

		if (!live_edit_funcs->node_call_func)
			return true;
		live_edit_funcs->node_call_func(live_edit_funcs->udata,cmd[1],cmd[2],  cmd[3],cmd[4],cmd[5],cmd[6],cmd[7]);

	} else if (cmdstr=="live_res_call") {

		if (!live_edit_funcs->res_call_func)
			return true;
		live_edit_funcs->res_call_func(live_edit_funcs->udata,cmd[1],cmd[2],  cmd[3],cmd[4],cmd[5],cmd[6],cmd[7]);

	} else if (cmdstr=="live_create_node") {

		live_edit_funcs->tree_create_node_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_instance_node") {

		live_edit_funcs->tree_instance_node_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_remove_node") {

		live_edit_funcs->tree_remove_node_func(live_edit_funcs->udata,cmd[1]);

	} else if (cmdstr=="live_remove_and_keep_node") {

		live_edit_funcs->tree_remove_and_keep_node_func(live_edit_funcs->udata,cmd[1],cmd[2]);
	} else if (cmdstr=="live_restore_node") {

		live_edit_funcs->tree_restore_node_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3]);

	} else if (cmdstr=="live_duplicate_node") {

		live_edit_funcs->tree_duplicate_node_func(live_edit_funcs->udata,cmd[1],cmd[2]);
	} else if (cmdstr=="live_reparent_node") {

		live_edit_funcs->tree_reparent_node_func(live_edit_funcs->udata,cmd[1],cmd[2],cmd[3],cmd[4]);

	} else {

		return false;
	}

	return true;
}

void ScriptDebuggerRemote::_poll_events() {

	while(packet_peer_stream->get_available_packet_count()>0) {

		_get_output();

		//send over output_strings

		Variant var;
		Error err = packet_peer_stream->get_var(var);

		ERR_CONTINUE( err != OK);
		ERR_CONTINUE( var.get_type()!=Variant::ARRAY );

		Array cmd = var;

		ERR_CONTINUE( cmd.size()==0);
		ERR_CONTINUE( cmd[0].get_type()!=Variant::STRING );

		String command = cmd[0];
		//cmd.remove(0);

		if (command=="break") {

			if (get_break_language())
				debug(get_break_language());
		} else if (command=="request_scene_tree") {

			if (request_scene_tree)
				request_scene_tree(request_scene_tree_ud);
		} else if (command=="request_video_mem") {

			_send_video_memory();
		} else if (command=="breakpoint") {

			bool set = cmd[3];
			if (set)
				insert_breakpoint(cmd[2],cmd[1]);
			else
				remove_breakpoint(cmd[2],cmd[1]);
		} else {
			_parse_live_edit(cmd);
		}

	}

}


void ScriptDebuggerRemote::idle_poll() {

	    _get_output();


	    if (requested_quit) {

		    packet_peer_stream->put_var("kill_me");
		    packet_peer_stream->put_var(0);
		    requested_quit=false;

	    }


	    if (performance) {

		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt-last_perf_time > 1000) {

			last_perf_time=pt;
			int max = performance->get("MONITOR_MAX");
			Array arr;
			arr.resize(max);
			for(int i=0;i<max;i++) {
				arr[i]=performance->call("get_monitor",i);
			}
			packet_peer_stream->put_var("performance");
			packet_peer_stream->put_var(1);
			packet_peer_stream->put_var(arr);

		}
	    }

	    _poll_events();

}


void ScriptDebuggerRemote::send_message(const String& p_message, const Array &p_args) {

	mutex->lock();
	if (!locking && tcp_client->is_connected()) {

		Message msg;
		msg.message=p_message;
		msg.data=p_args;
		messages.push_back(msg);
	}
	mutex->unlock();
}


void ScriptDebuggerRemote::_print_handler(void *p_this,const String& p_string) {

	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote*)p_this;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec()/1000;
	sdr->msec_count+=ticks-sdr->last_msec;
	sdr->last_msec=ticks;

	if (sdr->msec_count>1000) {
		sdr->char_count=0;
		sdr->msec_count=0;
	}

	String s = p_string;
	int allowed_chars = MIN(MAX(sdr->max_cps - sdr->char_count,0), s.length());

	if (allowed_chars==0)
		return;

	if (allowed_chars<s.length()) {
		s=s.substr(0,allowed_chars);
	}

	sdr->char_count+=allowed_chars;

	if (sdr->char_count>=sdr->max_cps) {
		s+="\n[output overflow, print less text!]\n";
	}

	sdr->mutex->lock();
	if (!sdr->locking && sdr->tcp_client->is_connected()) {

		sdr->output_strings.push_back(s);
	}
	sdr->mutex->unlock();
}

void ScriptDebuggerRemote::request_quit() {

	requested_quit=true;
}

void ScriptDebuggerRemote::set_request_scene_tree_message_func(RequestSceneTreeMessageFunc p_func, void *p_udata) {

	request_scene_tree=p_func;
	request_scene_tree_ud=p_udata;
}

void ScriptDebuggerRemote::set_live_edit_funcs(LiveEditFuncs *p_funcs) {

	live_edit_funcs=p_funcs;
}

ScriptDebuggerRemote::ResourceUsageFunc ScriptDebuggerRemote::resource_usage_func=NULL;

ScriptDebuggerRemote::ScriptDebuggerRemote() {

	tcp_client  = StreamPeerTCP::create_ref();
	packet_peer_stream = Ref<PacketPeerStream>( memnew(PacketPeerStream) );
	packet_peer_stream->set_stream_peer(tcp_client);
	mutex = Mutex::create();
	locking=false;

	phl.printfunc=_print_handler;
	phl.userdata=this;
	add_print_handler(&phl);
	requested_quit=false;
	performance = Globals::get_singleton()->get_singleton_object("Performance");
	last_perf_time=0;
	poll_every=0;
	request_scene_tree=NULL;
	live_edit_funcs=NULL;
	max_cps = GLOBAL_DEF("debug/max_remote_stdout_chars_per_second",2048);
	char_count=0;
	msec_count=0;
	last_msec=0;

	eh.errfunc=_err_handler;
	eh.userdata=this;
	add_error_handler(&eh);

}

ScriptDebuggerRemote::~ScriptDebuggerRemote() {

	remove_print_handler(&phl);
	remove_error_handler(&eh);
	memdelete(mutex);

}
