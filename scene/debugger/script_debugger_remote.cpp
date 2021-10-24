/*************************************************************************/
/*  script_debugger_remote.cpp                                           */
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

#include "script_debugger_remote.h"

#include "core/engine.h"
#include "core/io/ip.h"
#include "core/io/marshalls.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"
#include "servers/visual_server.h"

void ScriptDebuggerRemote::_send_video_memory() {
	List<ResourceUsage> usage;
	if (resource_usage_func) {
		resource_usage_func(&usage);
	}

	usage.sort();

	packet_peer_stream->put_var("message:video_mem");
	packet_peer_stream->put_var(usage.size() * 4);

	for (List<ResourceUsage>::Element *E = usage.front(); E; E = E->next()) {
		packet_peer_stream->put_var(E->get().path);
		packet_peer_stream->put_var(E->get().type);
		packet_peer_stream->put_var(E->get().format);
		packet_peer_stream->put_var(E->get().vram);
	}
}

Error ScriptDebuggerRemote::connect_to_host(const String &p_host, uint16_t p_port) {
	IP_Address ip;
	if (p_host.is_valid_ip_address()) {
		ip = p_host;
	} else {
		ip = IP::get_singleton()->resolve_hostname(p_host);
	}

	int port = p_port;

	const int tries = 6;
	int waits[tries] = { 1, 10, 100, 1000, 1000, 1000 };

	tcp_client->connect_to_host(ip, port);

	for (int i = 0; i < tries; i++) {
		if (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
			print_verbose("Remote Debugger: Connected!");
			break;
		} else {
			const int ms = waits[i];
			OS::get_singleton()->delay_usec(ms * 1000);
			print_verbose("Remote Debugger: Connection failed with status: '" + String::num(tcp_client->get_status()) + "', retrying in " + String::num(ms) + " msec.");
		};
	};

	if (tcp_client->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		ERR_PRINT("Remote Debugger: Unable to connect. Status: " + String::num(tcp_client->get_status()) + ".");
		return FAILED;
	};

	packet_peer_stream->set_stream_peer(tcp_client);

	return OK;
}

void ScriptDebuggerRemote::_put_variable(const String &p_name, const Variant &p_variable) {
	packet_peer_stream->put_var(p_name);

	Variant var = p_variable;
	if (p_variable.get_type() == Variant::OBJECT && p_variable.operator Object *() == nullptr) {
		var = Variant();
	}

	int len = 0;
	Error err = encode_variant(var, nullptr, len, true);
	if (err != OK)
		ERR_PRINT("Failed to encode variant.");

	if (len > packet_peer_stream->get_output_buffer_max_size()) { //limit to max size
		packet_peer_stream->put_var(Variant());
	} else {
		packet_peer_stream->put_var(var);
	}
}

void ScriptDebuggerRemote::_save_node(ObjectID id, const String &p_path) {
	Node *node = Object::cast_to<Node>(ObjectDB::get_instance(id));
	ERR_FAIL_COND(!node);

	Ref<PackedScene> ps = memnew(PackedScene);
	ps->pack(node);
	ResourceSaver::save(p_path, ps);
}

void ScriptDebuggerRemote::debug(ScriptLanguage *p_script, bool p_can_continue, bool p_is_error_breakpoint) {
	//this function is called when there is a debugger break (bug on script)
	//or when execution is paused from editor

	if (skip_breakpoints && !p_is_error_breakpoint) {
		return;
	}

	ERR_FAIL_COND_MSG(!tcp_client->is_connected_to_host(), "Script Debugger failed to connect, but being used anyway.");

	if (allow_focus_steal_pid) {
		OS::get_singleton()->enable_for_stealing_focus(allow_focus_steal_pid);
	}

	packet_peer_stream->put_var("debug_enter");
	packet_peer_stream->put_var(2);
	packet_peer_stream->put_var(p_can_continue);
	packet_peer_stream->put_var(p_script->debug_get_error());

	skip_profile_frame = true; // to avoid super long frame time for the frame

	Input::MouseMode mouse_mode = Input::get_singleton()->get_mouse_mode();
	if (mouse_mode != Input::MOUSE_MODE_VISIBLE) {
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	}

	uint64_t loop_begin_usec = 0;
	uint64_t loop_time_sec = 0;
	while (true) {
		loop_begin_usec = OS::get_singleton()->get_ticks_usec();

		_get_output();

		if (packet_peer_stream->get_available_packet_count() > 0) {
			Variant var;
			Error err = packet_peer_stream->get_var(var);

			ERR_CONTINUE(err != OK);
			ERR_CONTINUE(var.get_type() != Variant::ARRAY);

			Array cmd = var;

			ERR_CONTINUE(cmd.size() == 0);
			ERR_CONTINUE(cmd[0].get_type() != Variant::STRING);

			String command = cmd[0];

			if (command == "get_stack_dump") {
				packet_peer_stream->put_var("stack_dump");
				int slc = p_script->debug_get_stack_level_count();
				packet_peer_stream->put_var(slc);

				for (int i = 0; i < slc; i++) {
					Dictionary d;
					d["file"] = p_script->debug_get_stack_level_source(i);
					d["line"] = p_script->debug_get_stack_level_line(i);
					d["function"] = p_script->debug_get_stack_level_function(i);
					//d["id"]=p_script->debug_get_stack_level_
					d["id"] = 0;

					packet_peer_stream->put_var(d);
				}

			} else if (command == "get_stack_frame_vars") {
				cmd.remove(0);
				ERR_CONTINUE(cmd.size() != 1);
				int lv = cmd[0];

				List<String> members;
				List<Variant> member_vals;
				if (ScriptInstance *inst = p_script->debug_get_stack_level_instance(lv)) {
					members.push_back("self");
					member_vals.push_back(inst->get_owner());
				}
				p_script->debug_get_stack_level_members(lv, &members, &member_vals);
				ERR_CONTINUE(members.size() != member_vals.size());

				List<String> locals;
				List<Variant> local_vals;
				p_script->debug_get_stack_level_locals(lv, &locals, &local_vals);
				ERR_CONTINUE(locals.size() != local_vals.size());

				List<String> globals;
				List<Variant> globals_vals;
				p_script->debug_get_globals(&globals, &globals_vals);
				ERR_CONTINUE(globals.size() != globals_vals.size());

				packet_peer_stream->put_var("stack_frame_vars");
				packet_peer_stream->put_var(3 + (locals.size() + members.size() + globals.size()) * 2);

				{ //locals
					packet_peer_stream->put_var(locals.size());

					List<String>::Element *E = locals.front();
					List<Variant>::Element *F = local_vals.front();

					while (E) {
						_put_variable(E->get(), F->get());

						E = E->next();
						F = F->next();
					}
				}

				{ //members
					packet_peer_stream->put_var(members.size());

					List<String>::Element *E = members.front();
					List<Variant>::Element *F = member_vals.front();

					while (E) {
						_put_variable(E->get(), F->get());

						E = E->next();
						F = F->next();
					}
				}

				{ //globals
					packet_peer_stream->put_var(globals.size());

					List<String>::Element *E = globals.front();
					List<Variant>::Element *F = globals_vals.front();

					while (E) {
						_put_variable(E->get(), F->get());

						E = E->next();
						F = F->next();
					}
				}

			} else if (command == "step") {
				set_depth(-1);
				set_lines_left(1);
				break;
			} else if (command == "next") {
				set_depth(0);
				set_lines_left(1);
				break;

			} else if (command == "continue") {
				set_depth(-1);
				set_lines_left(-1);
				OS::get_singleton()->move_window_to_foreground();
				break;
			} else if (command == "break") {
				ERR_PRINT("Got break when already broke!");
				break;
			} else if (command == "request_scene_tree") {
#ifdef DEBUG_ENABLED
				if (scene_tree) {
					scene_tree->_debugger_request_tree();
				}
#endif
			} else if (command == "request_video_mem") {
				_send_video_memory();
			} else if (command == "inspect_object") {
				ObjectID id = cmd[1];
				_send_object_id(id);
			} else if (command == "set_object_property") {
				_set_object_property(cmd[1], cmd[2], cmd[3]);

			} else if (command == "override_camera_2D:set") {
				bool enforce = cmd[1];

				if (scene_tree) {
					scene_tree->get_root()->enable_canvas_transform_override(enforce);
				}
			} else if (command == "override_camera_2D:transform") {
				Transform2D transform = cmd[1];

				if (scene_tree) {
					scene_tree->get_root()->set_canvas_transform_override(transform);
				}
			} else if (command == "override_camera_3D:set") {
				bool enable = cmd[1];

				if (scene_tree) {
					scene_tree->get_root()->enable_camera_override(enable);
				}
			} else if (command == "override_camera_3D:transform") {
				Transform transform = cmd[1];
				bool is_perspective = cmd[2];
				float size_or_fov = cmd[3];
				float near = cmd[4];
				float far = cmd[5];

				if (scene_tree) {
					if (is_perspective) {
						scene_tree->get_root()->set_camera_override_perspective(size_or_fov, near, far);
					} else {
						scene_tree->get_root()->set_camera_override_orthogonal(size_or_fov, near, far);
					}
					scene_tree->get_root()->set_camera_override_transform(transform);
				}

			} else if (command == "reload_scripts") {
				reload_all_scripts = true;
			} else if (command == "breakpoint") {
				bool set = cmd[3];
				if (set) {
					insert_breakpoint(cmd[2], cmd[1]);
				} else {
					remove_breakpoint(cmd[2], cmd[1]);
				}

			} else if (command == "save_node") {
				_save_node(cmd[1], cmd[2]);
			} else if (command == "set_skip_breakpoints") {
				skip_breakpoints = cmd[1];
			} else {
				_parse_live_edit(cmd);
			}

		} else {
			OS::get_singleton()->delay_usec(10000);
			OS::get_singleton()->process_and_drop_events();
		}

		// This is for the camera override to stay live even when the game is paused from the editor
		loop_time_sec = (OS::get_singleton()->get_ticks_usec() - loop_begin_usec) / 1000000.0f;
		VisualServer::get_singleton()->sync();
		if (VisualServer::get_singleton()->has_changed()) {
			VisualServer::get_singleton()->draw(true, loop_time_sec * Engine::get_singleton()->get_time_scale());
		}
	}

	packet_peer_stream->put_var("debug_exit");
	packet_peer_stream->put_var(0);

	if (mouse_mode != Input::MOUSE_MODE_VISIBLE) {
		Input::get_singleton()->set_mouse_mode(mouse_mode);
	}
}

void ScriptDebuggerRemote::_get_output() {
	mutex.lock();
	if (output_strings.size()) {
		locking = true;
		packet_peer_stream->put_var("output");
		packet_peer_stream->put_var(output_strings.size());

		while (output_strings.size()) {
			const OutputString &output_string = output_strings.front()->get();

			Array msg_data;
			msg_data.push_back(output_string.message);
			msg_data.push_back(output_string.type);

			packet_peer_stream->put_var(msg_data);

			output_strings.pop_front();
		}
		locking = false;
	}

	if (n_messages_dropped > 0) {
		Message msg;
		msg.message = "Too many messages! " + String::num_int64(n_messages_dropped) + " messages were dropped.";
		messages.push_back(msg);
		n_messages_dropped = 0;
	}

	while (messages.size()) {
		locking = true;
		packet_peer_stream->put_var("message:" + messages.front()->get().message);
		packet_peer_stream->put_var(messages.front()->get().data.size());
		for (int i = 0; i < messages.front()->get().data.size(); i++) {
			packet_peer_stream->put_var(messages.front()->get().data[i]);
		}
		messages.pop_front();
		locking = false;
	}

	if (n_errors_dropped == 1) {
		// Only print one message about dropping per second
		OutputError oe;
		oe.error = "TOO_MANY_ERRORS";
		oe.error_descr = "Too many errors! Ignoring errors for up to 1 second.";
		oe.warning = false;
		uint64_t time = OS::get_singleton()->get_ticks_msec();
		oe.hr = time / 3600000;
		oe.min = (time / 60000) % 60;
		oe.sec = (time / 1000) % 60;
		oe.msec = time % 1000;
		errors.push_back(oe);
	}

	if (n_warnings_dropped == 1) {
		// Only print one message about dropping per second
		OutputError oe;
		oe.error = "TOO_MANY_WARNINGS";
		oe.error_descr = "Too many warnings! Ignoring warnings for up to 1 second.";
		oe.warning = true;
		uint64_t time = OS::get_singleton()->get_ticks_msec();
		oe.hr = time / 3600000;
		oe.min = (time / 60000) % 60;
		oe.sec = (time / 1000) % 60;
		oe.msec = time % 1000;
		errors.push_back(oe);
	}

	while (errors.size()) {
		locking = true;
		packet_peer_stream->put_var("error");
		OutputError oe = errors.front()->get();

		packet_peer_stream->put_var(oe.callstack.size() + 2);

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
		for (int i = 0; i < oe.callstack.size(); i++) {
			packet_peer_stream->put_var(oe.callstack[i]);
		}

		errors.pop_front();
		locking = false;
	}
	mutex.unlock();
}

void ScriptDebuggerRemote::line_poll() {
	//the purpose of this is just processing events every now and then when the script might get too busy
	//otherwise bugs like infinite loops can't be caught
	if (poll_every % 2048 == 0) {
		_poll_events();
	}
	poll_every++;
}

void ScriptDebuggerRemote::_err_handler(void *ud, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, ErrorHandlerType p_type) {
	if (p_type == ERR_HANDLER_SCRIPT) {
		return; //ignore script errors, those go through debugger
	}

	Vector<ScriptLanguage::StackInfo> si;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		si = ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size()) {
			break;
		}
	}

	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote *)ud;
	sdr->send_error(p_func, p_file, p_line, p_err, p_descr, p_type, si);
}

bool ScriptDebuggerRemote::_parse_live_edit(const Array &p_command) {
#ifdef DEBUG_ENABLED

	String cmdstr = p_command[0];
	if (!scene_tree || !cmdstr.begins_with("live_")) {
		return false;
	}

	if (cmdstr == "live_set_root") {
		scene_tree->_live_edit_root_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_node_path") {
		scene_tree->_live_edit_node_path_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_res_path") {
		scene_tree->_live_edit_res_path_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_node_prop_res") {
		scene_tree->_live_edit_node_set_res_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_node_prop") {
		scene_tree->_live_edit_node_set_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_res_prop_res") {
		scene_tree->_live_edit_res_set_res_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_res_prop") {
		scene_tree->_live_edit_res_set_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_node_call") {
		scene_tree->_live_edit_node_call_func(p_command[1], p_command[2], p_command[3], p_command[4], p_command[5], p_command[6], p_command[7], p_command[8], p_command[9], p_command[10]);

	} else if (cmdstr == "live_res_call") {
		scene_tree->_live_edit_res_call_func(p_command[1], p_command[2], p_command[3], p_command[4], p_command[5], p_command[6], p_command[7], p_command[8], p_command[9], p_command[10]);

	} else if (cmdstr == "live_create_node") {
		scene_tree->_live_edit_create_node_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_instance_node") {
		scene_tree->_live_edit_instance_node_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_remove_node") {
		scene_tree->_live_edit_remove_node_func(p_command[1]);

	} else if (cmdstr == "live_remove_and_keep_node") {
		scene_tree->_live_edit_remove_and_keep_node_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_restore_node") {
		scene_tree->_live_edit_restore_node_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_duplicate_node") {
		scene_tree->_live_edit_duplicate_node_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_reparent_node") {
		scene_tree->_live_edit_reparent_node_func(p_command[1], p_command[2], p_command[3], p_command[4]);

	} else {
		return false;
	}

	return true;
#else

	return false;
#endif
}

void ScriptDebuggerRemote::_send_object_id(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return;
	}

	typedef Pair<PropertyInfo, Variant> PropertyDesc;
	List<PropertyDesc> properties;

	if (ScriptInstance *si = obj->get_script_instance()) {
		if (!si->get_script().is_null()) {
			typedef Map<const Script *, Set<StringName>> ScriptMemberMap;
			typedef Map<const Script *, Map<StringName, Variant>> ScriptConstantsMap;

			ScriptMemberMap members;
			members[si->get_script().ptr()] = Set<StringName>();
			si->get_script()->get_members(&(members[si->get_script().ptr()]));

			ScriptConstantsMap constants;
			constants[si->get_script().ptr()] = Map<StringName, Variant>();
			si->get_script()->get_constants(&(constants[si->get_script().ptr()]));

			Ref<Script> base = si->get_script()->get_base_script();
			while (base.is_valid()) {
				members[base.ptr()] = Set<StringName>();
				base->get_members(&(members[base.ptr()]));

				constants[base.ptr()] = Map<StringName, Variant>();
				base->get_constants(&(constants[base.ptr()]));

				base = base->get_base_script();
			}

			for (ScriptMemberMap::Element *sm = members.front(); sm; sm = sm->next()) {
				for (Set<StringName>::Element *E = sm->get().front(); E; E = E->next()) {
					Variant m;
					if (si->get(E->get(), m)) {
						String script_path = sm->key() == si->get_script().ptr() ? "" : sm->key()->get_path().get_file() + "/";
						PropertyInfo pi(m.get_type(), "Members/" + script_path + E->get());
						properties.push_back(PropertyDesc(pi, m));
					}
				}
			}

			for (ScriptConstantsMap::Element *sc = constants.front(); sc; sc = sc->next()) {
				for (Map<StringName, Variant>::Element *E = sc->get().front(); E; E = E->next()) {
					String script_path = sc->key() == si->get_script().ptr() ? "" : sc->key()->get_path().get_file() + "/";
					if (E->value().get_type() == Variant::OBJECT) {
						Variant id = ((Object *)E->value())->get_instance_id();
						PropertyInfo pi(id.get_type(), "Constants/" + E->key(), PROPERTY_HINT_OBJECT_ID, "Object");
						properties.push_back(PropertyDesc(pi, id));
					} else {
						PropertyInfo pi(E->value().get_type(), "Constants/" + script_path + E->key());
						properties.push_back(PropertyDesc(pi, E->value()));
					}
				}
			}
		}
	}

	if (Node *node = Object::cast_to<Node>(obj)) {
		// in some cases node will not be in tree here
		// for instance where it created as variable and not yet added to tree
		// in such cases we can't ask for it's path
		if (node->is_inside_tree()) {
			PropertyInfo pi(Variant::NODE_PATH, String("Node/path"));
			properties.push_front(PropertyDesc(pi, node->get_path()));
		} else {
			PropertyInfo pi(Variant::STRING, String("Node/path"));
			properties.push_front(PropertyDesc(pi, "[Orphan]"));
		}

	} else if (Resource *res = Object::cast_to<Resource>(obj)) {
		if (Script *s = Object::cast_to<Script>(res)) {
			Map<StringName, Variant> constants;
			s->get_constants(&constants);
			for (Map<StringName, Variant>::Element *E = constants.front(); E; E = E->next()) {
				if (E->value().get_type() == Variant::OBJECT) {
					Variant id = ((Object *)E->value())->get_instance_id();
					PropertyInfo pi(id.get_type(), "Constants/" + E->key(), PROPERTY_HINT_OBJECT_ID, "Object");
					properties.push_front(PropertyDesc(pi, E->value()));
				} else {
					PropertyInfo pi(E->value().get_type(), String("Constants/") + E->key());
					properties.push_front(PropertyDesc(pi, E->value()));
				}
			}
		}
	}

	List<PropertyInfo> pinfo;
	obj->get_property_list(&pinfo, true);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		if (E->get().usage & (PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CATEGORY)) {
			properties.push_back(PropertyDesc(E->get(), obj->get(E->get().name)));
		}
	}

	Array send_props;
	for (int i = 0; i < properties.size(); i++) {
		const PropertyInfo &pi = properties[i].first;
		Variant &var = properties[i].second;

		RES res = var;

		Array prop;
		prop.push_back(pi.name);
		prop.push_back(pi.type);

		//only send information that can be sent..
		int len = 0; //test how big is this to encode
		encode_variant(var, nullptr, len);
		if (len > packet_peer_stream->get_output_buffer_max_size()) { //limit to max size
			prop.push_back(PROPERTY_HINT_OBJECT_TOO_BIG);
			prop.push_back("");
			prop.push_back(pi.usage);
			prop.push_back(Variant());
		} else {
			prop.push_back(pi.hint);
			prop.push_back(pi.hint_string);
			prop.push_back(pi.usage);

			if (!res.is_null() && !res->get_path().empty()) {
				var = res->get_path();
			}

			prop.push_back(var);
		}
		send_props.push_back(prop);
	}

	packet_peer_stream->put_var("message:inspect_object");
	packet_peer_stream->put_var(3);
	packet_peer_stream->put_var(p_id);
	packet_peer_stream->put_var(obj->get_class());
	packet_peer_stream->put_var(send_props);
}

void ScriptDebuggerRemote::_set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value) {
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return;
	}

	String prop_name = p_property;
	if (p_property.begins_with("Members/")) {
		Vector<String> ss = p_property.split("/");
		prop_name = ss[ss.size() - 1];
	}

	obj->set(prop_name, p_value);
}

void ScriptDebuggerRemote::_poll_events() {
	//this si called from ::idle_poll, happens only when running the game,
	//does not get called while on debug break

	while (packet_peer_stream->get_available_packet_count() > 0) {
		_get_output();

		//send over output_strings

		Variant var;
		Error err = packet_peer_stream->get_var(var);

		ERR_CONTINUE(err != OK);
		ERR_CONTINUE(var.get_type() != Variant::ARRAY);

		Array cmd = var;

		ERR_CONTINUE(cmd.size() == 0);
		ERR_CONTINUE(cmd[0].get_type() != Variant::STRING);

		String command = cmd[0];
		//cmd.remove(0);

		if (command == "break") {
			if (get_break_language()) {
				debug(get_break_language());
			}
		} else if (command == "request_scene_tree") {
#ifdef DEBUG_ENABLED
			if (scene_tree) {
				scene_tree->_debugger_request_tree();
			}
#endif
		} else if (command == "request_video_mem") {
			_send_video_memory();
		} else if (command == "inspect_object") {
			ObjectID id = cmd[1];
			_send_object_id(id);
		} else if (command == "set_object_property") {
			_set_object_property(cmd[1], cmd[2], cmd[3]);

		} else if (command == "start_profiling") {
			for (int i = 0; i < ScriptServer::get_language_count(); i++) {
				ScriptServer::get_language(i)->profiling_start();
			}

			max_frame_functions = cmd[1];
			profiler_function_signature_map.clear();
			profiling = true;
			frame_time = 0;
			idle_time = 0;
			physics_time = 0;
			physics_frame_time = 0;

			print_line("PROFILING ALRIGHT!");

		} else if (command == "stop_profiling") {
			for (int i = 0; i < ScriptServer::get_language_count(); i++) {
				ScriptServer::get_language(i)->profiling_stop();
			}
			profiling = false;
			_send_profiling_data(false);
			print_line("PROFILING END!");
		} else if (command == "start_network_profiling") {
			multiplayer->profiling_start();
			profiling_network = true;
		} else if (command == "stop_network_profiling") {
			multiplayer->profiling_end();
			profiling_network = false;
		} else if (command == "override_camera_2D:set") {
			bool enforce = cmd[1];

			if (scene_tree) {
				scene_tree->get_root()->enable_canvas_transform_override(enforce);
			}
		} else if (command == "override_camera_2D:transform") {
			Transform2D transform = cmd[1];

			if (scene_tree) {
				scene_tree->get_root()->set_canvas_transform_override(transform);
			}
		} else if (command == "override_camera_3D:set") {
			bool enable = cmd[1];

			if (scene_tree) {
				scene_tree->get_root()->enable_camera_override(enable);
			}
		} else if (command == "override_camera_3D:transform") {
			Transform transform = cmd[1];
			bool is_perspective = cmd[2];
			float size_or_fov = cmd[3];
			float near = cmd[4];
			float far = cmd[5];

			if (scene_tree) {
				if (is_perspective) {
					scene_tree->get_root()->set_camera_override_perspective(size_or_fov, near, far);
				} else {
					scene_tree->get_root()->set_camera_override_orthogonal(size_or_fov, near, far);
				}
				scene_tree->get_root()->set_camera_override_transform(transform);
			}

		} else if (command == "reload_scripts") {
			reload_all_scripts = true;
		} else if (command == "breakpoint") {
			bool set = cmd[3];
			if (set) {
				insert_breakpoint(cmd[2], cmd[1]);
			} else {
				remove_breakpoint(cmd[2], cmd[1]);
			}
		} else if (command == "set_skip_breakpoints") {
			skip_breakpoints = cmd[1];
		} else {
			_parse_live_edit(cmd);
		}
	}
}

void ScriptDebuggerRemote::_send_profiling_data(bool p_for_frame) {
	int ofs = 0;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		if (p_for_frame) {
			ofs += ScriptServer::get_language(i)->profiling_get_frame_data(&profile_info.write[ofs], profile_info.size() - ofs);
		} else {
			ofs += ScriptServer::get_language(i)->profiling_get_accumulated_data(&profile_info.write[ofs], profile_info.size() - ofs);
		}
	}

	for (int i = 0; i < ofs; i++) {
		profile_info_ptrs.write[i] = &profile_info.write[i];
	}

	SortArray<ScriptLanguage::ProfilingInfo *, ProfileInfoSort> sa;
	sa.sort(profile_info_ptrs.ptrw(), ofs);

	int to_send = MIN(ofs, max_frame_functions);

	//check signatures first
	uint64_t total_script_time = 0;

	for (int i = 0; i < to_send; i++) {
		if (!profiler_function_signature_map.has(profile_info_ptrs[i]->signature)) {
			int idx = profiler_function_signature_map.size();
			packet_peer_stream->put_var("profile_sig");
			packet_peer_stream->put_var(2);
			packet_peer_stream->put_var(profile_info_ptrs[i]->signature);
			packet_peer_stream->put_var(idx);

			profiler_function_signature_map[profile_info_ptrs[i]->signature] = idx;
		}

		total_script_time += profile_info_ptrs[i]->self_time;
	}

	//send frames then

	if (p_for_frame) {
		packet_peer_stream->put_var("profile_frame");
		packet_peer_stream->put_var(8 + profile_frame_data.size() * 2 + to_send * 4);
	} else {
		packet_peer_stream->put_var("profile_total");
		packet_peer_stream->put_var(8 + to_send * 4);
	}

	packet_peer_stream->put_var(Engine::get_singleton()->get_idle_frames()); //total frame time
	packet_peer_stream->put_var(frame_time); //total frame time
	packet_peer_stream->put_var(idle_time); //idle frame time
	packet_peer_stream->put_var(physics_time); //fixed frame time
	packet_peer_stream->put_var(physics_frame_time); //fixed frame time

	packet_peer_stream->put_var(USEC_TO_SEC(total_script_time)); //total script execution time

	if (p_for_frame) {
		packet_peer_stream->put_var(profile_frame_data.size()); //how many profile framedatas to send
		packet_peer_stream->put_var(to_send); //how many script functions to send
		for (int i = 0; i < profile_frame_data.size(); i++) {
			packet_peer_stream->put_var(profile_frame_data[i].name);
			packet_peer_stream->put_var(profile_frame_data[i].data);
		}
	} else {
		packet_peer_stream->put_var(0); //how many script functions to send
		packet_peer_stream->put_var(to_send); //how many script functions to send
	}

	for (int i = 0; i < to_send; i++) {
		int sig_id = -1;

		if (profiler_function_signature_map.has(profile_info_ptrs[i]->signature)) {
			sig_id = profiler_function_signature_map[profile_info_ptrs[i]->signature];
		}

		packet_peer_stream->put_var(sig_id);
		packet_peer_stream->put_var(profile_info_ptrs[i]->call_count);
		packet_peer_stream->put_var(profile_info_ptrs[i]->total_time / 1000000.0);
		packet_peer_stream->put_var(profile_info_ptrs[i]->self_time / 1000000.0);
	}

	if (p_for_frame) {
		profile_frame_data.clear();
	}
}

void ScriptDebuggerRemote::idle_poll() {
	// this function is called every frame, except when there is a debugger break (::debug() in this class)
	// execution stops and remains in the ::debug function

	_get_output();

	if (requested_quit) {
		packet_peer_stream->put_var("kill_me");
		packet_peer_stream->put_var(0);
		requested_quit = false;
	}

	if (performance) {
		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt - last_perf_time > 1000) {
			last_perf_time = pt;
			int max = performance->get("MONITOR_MAX");
			Array arr;
			arr.resize(max);
			for (int i = 0; i < max; i++) {
				arr[i] = performance->call("get_monitor", i);
			}
			packet_peer_stream->put_var("performance");
			packet_peer_stream->put_var(1);
			packet_peer_stream->put_var(arr);
		}
	}

	if (profiling) {
		if (skip_profile_frame) {
			skip_profile_frame = false;
		} else {
			//send profiling info normally
			_send_profiling_data(true);
		}
	}

	if (profiling_network) {
		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt - last_net_bandwidth_time > 200) {
			last_net_bandwidth_time = pt;
			_send_network_bandwidth_usage();
		}
		if (pt - last_net_prof_time > 100) {
			last_net_prof_time = pt;
			_send_network_profiling_data();
		}
	}

	if (reload_all_scripts) {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->reload_all_scripts();
		}
		reload_all_scripts = false;
	}

	_poll_events();
}

void ScriptDebuggerRemote::_send_network_profiling_data() {
	ERR_FAIL_COND(multiplayer.is_null());

	int n_nodes = multiplayer->get_profiling_frame(&network_profile_info.write[0]);

	packet_peer_stream->put_var("network_profile");
	packet_peer_stream->put_var(n_nodes * 6);
	for (int i = 0; i < n_nodes; ++i) {
		packet_peer_stream->put_var(network_profile_info[i].node);
		packet_peer_stream->put_var(network_profile_info[i].node_path);
		packet_peer_stream->put_var(network_profile_info[i].incoming_rpc);
		packet_peer_stream->put_var(network_profile_info[i].incoming_rset);
		packet_peer_stream->put_var(network_profile_info[i].outgoing_rpc);
		packet_peer_stream->put_var(network_profile_info[i].outgoing_rset);
	}
}

void ScriptDebuggerRemote::_send_network_bandwidth_usage() {
	ERR_FAIL_COND(multiplayer.is_null());

	int incoming_bandwidth = multiplayer->get_incoming_bandwidth_usage();
	int outgoing_bandwidth = multiplayer->get_outgoing_bandwidth_usage();

	packet_peer_stream->put_var("network_bandwidth");
	packet_peer_stream->put_var(2);
	packet_peer_stream->put_var(incoming_bandwidth);
	packet_peer_stream->put_var(outgoing_bandwidth);
}

void ScriptDebuggerRemote::send_message(const String &p_message, const Array &p_args) {
	mutex.lock();
	if (!locking && tcp_client->is_connected_to_host()) {
		if (messages.size() >= max_messages_per_frame) {
			n_messages_dropped++;
		} else {
			Message msg;
			msg.message = p_message;
			msg.data = p_args;
			messages.push_back(msg);
		}
	}
	mutex.unlock();
}

void ScriptDebuggerRemote::send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, ErrorHandlerType p_type, const Vector<ScriptLanguage::StackInfo> &p_stack_info) {
	OutputError oe;
	oe.error = p_err;
	oe.error_descr = p_descr;
	oe.source_file = p_file;
	oe.source_line = p_line;
	oe.source_func = p_func;
	oe.warning = p_type == ERR_HANDLER_WARNING;
	uint64_t time = OS::get_singleton()->get_ticks_msec();
	oe.hr = time / 3600000;
	oe.min = (time / 60000) % 60;
	oe.sec = (time / 1000) % 60;
	oe.msec = time % 1000;
	Array cstack;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec() / 1000;
	msec_count += ticks - last_msec;
	last_msec = ticks;

	if (msec_count > 1000) {
		msec_count = 0;

		err_count = 0;
		n_errors_dropped = 0;
		warn_count = 0;
		n_warnings_dropped = 0;
	}

	cstack.resize(p_stack_info.size() * 3);
	for (int i = 0; i < p_stack_info.size(); i++) {
		cstack[i * 3 + 0] = p_stack_info[i].file;
		cstack[i * 3 + 1] = p_stack_info[i].func;
		cstack[i * 3 + 2] = p_stack_info[i].line;
	}

	oe.callstack = cstack;
	if (oe.warning) {
		warn_count++;
	} else {
		err_count++;
	}

	mutex.lock();

	if (!locking && tcp_client->is_connected_to_host()) {
		if (oe.warning) {
			if (warn_count > max_warnings_per_second) {
				n_warnings_dropped++;
			} else {
				errors.push_back(oe);
			}
		} else {
			if (err_count > max_errors_per_second) {
				n_errors_dropped++;
			} else {
				errors.push_back(oe);
			}
		}
	}

	mutex.unlock();
}

void ScriptDebuggerRemote::_print_handler(void *p_this, const String &p_string, bool p_error) {
	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote *)p_this;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec() / 1000;
	sdr->msec_count += ticks - sdr->last_msec;
	sdr->last_msec = ticks;

	if (sdr->msec_count > 1000) {
		sdr->char_count = 0;
		sdr->msec_count = 0;
	}

	String s = p_string;
	int allowed_chars = MIN(MAX(sdr->max_cps - sdr->char_count, 0), s.length());

	if (allowed_chars == 0 && s.length() > 0) {
		return;
	}

	if (allowed_chars < s.length()) {
		s = s.substr(0, allowed_chars);
	}

	sdr->char_count += allowed_chars;
	bool overflowed = sdr->char_count >= sdr->max_cps;

	sdr->mutex.lock();
	if (!sdr->locking && sdr->tcp_client->is_connected_to_host()) {
		if (overflowed) {
			s += "[...]";
		}

		OutputString output_string;
		output_string.message = s;
		output_string.type = p_error ? MESSAGE_TYPE_ERROR : MESSAGE_TYPE_LOG;
		sdr->output_strings.push_back(output_string);

		if (overflowed) {
			output_string.message = "[output overflow, print less text!]";
			output_string.type = MESSAGE_TYPE_ERROR;
			sdr->output_strings.push_back(output_string);
		}
	}
	sdr->mutex.unlock();
}

void ScriptDebuggerRemote::request_quit() {
	requested_quit = true;
}

void ScriptDebuggerRemote::set_multiplayer(Ref<MultiplayerAPI> p_multiplayer) {
	multiplayer = p_multiplayer;
}

bool ScriptDebuggerRemote::is_profiling() const {
	return profiling;
}
void ScriptDebuggerRemote::add_profiling_frame_data(const StringName &p_name, const Array &p_data) {
	int idx = -1;
	for (int i = 0; i < profile_frame_data.size(); i++) {
		if (profile_frame_data[i].name == p_name) {
			idx = i;
			break;
		}
	}

	FrameData fd;
	fd.name = p_name;
	fd.data = p_data;

	if (idx == -1) {
		profile_frame_data.push_back(fd);
	} else {
		profile_frame_data.write[idx] = fd;
	}
}

void ScriptDebuggerRemote::profiling_start() {
	//ignores this, uses it via connection
}

void ScriptDebuggerRemote::profiling_end() {
	//ignores this, uses it via connection
}

void ScriptDebuggerRemote::profiling_set_frame_times(float p_frame_time, float p_idle_time, float p_physics_time, float p_physics_frame_time) {
	frame_time = p_frame_time;
	idle_time = p_idle_time;
	physics_time = p_physics_time;
	physics_frame_time = p_physics_frame_time;
}

void ScriptDebuggerRemote::set_skip_breakpoints(bool p_skip_breakpoints) {
	skip_breakpoints = p_skip_breakpoints;
}

void ScriptDebuggerRemote::set_allow_focus_steal_pid(OS::ProcessID p_pid) {
	allow_focus_steal_pid = p_pid;
}

ScriptDebuggerRemote::ResourceUsageFunc ScriptDebuggerRemote::resource_usage_func = nullptr;

ScriptDebuggerRemote::ScriptDebuggerRemote() :
		profiling(false),
		profiling_network(false),
		max_frame_functions(16),
		skip_profile_frame(false),
		reload_all_scripts(false),
		tcp_client(Ref<StreamPeerTCP>(memnew(StreamPeerTCP))),
		packet_peer_stream(Ref<PacketPeerStream>(memnew(PacketPeerStream))),
		last_perf_time(0),
		last_net_prof_time(0),
		last_net_bandwidth_time(0),
		performance(Engine::get_singleton()->get_singleton_object("Performance")),
		requested_quit(false),
		max_messages_per_frame(GLOBAL_GET("network/limits/debugger_stdout/max_messages_per_frame")),
		n_messages_dropped(0),
		max_errors_per_second(GLOBAL_GET("network/limits/debugger_stdout/max_errors_per_second")),
		max_warnings_per_second(GLOBAL_GET("network/limits/debugger_stdout/max_warnings_per_second")),
		n_errors_dropped(0),
		n_warnings_dropped(0),
		max_cps(GLOBAL_GET("network/limits/debugger_stdout/max_chars_per_second")),
		char_count(0),
		err_count(0),
		warn_count(0),
		last_msec(0),
		msec_count(0),
		allow_focus_steal_pid(0),
		locking(false),
		poll_every(0),
		scene_tree(nullptr) {
	packet_peer_stream->set_stream_peer(tcp_client);
	packet_peer_stream->set_output_buffer_max_size((1024 * 1024 * 8) - 4); // 8 MiB should be way more than enough, minus 4 bytes for separator.

	phl.printfunc = _print_handler;
	phl.userdata = this;
	add_print_handler(&phl);

	eh.errfunc = _err_handler;
	eh.userdata = this;
	add_error_handler(&eh);

	profile_info.resize(GLOBAL_GET("debug/settings/profiler/max_functions"));
	network_profile_info.resize(GLOBAL_GET("debug/settings/profiler/max_functions"));
	profile_info_ptrs.resize(profile_info.size());
}

ScriptDebuggerRemote::~ScriptDebuggerRemote() {
	remove_print_handler(&phl);
	remove_error_handler(&eh);
}
