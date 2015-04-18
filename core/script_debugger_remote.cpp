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

Error ScriptDebuggerRemote::connect_to_host(const String& p_host,uint16_t p_port) {


    IP_Address ip = IP::get_singleton()->resolve_hostname(p_host);


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
			cmd.remove(0);


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
	mutex->unlock();
}

void ScriptDebuggerRemote::line_poll() {

	//the purpose of this is just processing events every now and then when the script might get too busy
	//otherwise bugs like infinite loops cant be catched
	if (poll_every%512==0)
		_poll_events();
	poll_every++;

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
		cmd.remove(0);

		if (command=="break") {

			if (get_break_language())
				debug(get_break_language());
		} else if (command=="request_scene_tree") {

			if (request_scene_tree)
				request_scene_tree(request_scene_tree_ud);
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

	sdr->mutex->lock();
	if (!sdr->locking && sdr->tcp_client->is_connected()) {

		sdr->output_strings .push_back(p_string);
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

}

ScriptDebuggerRemote::~ScriptDebuggerRemote() {

	remove_print_handler(&phl);
	memdelete(mutex);


}
