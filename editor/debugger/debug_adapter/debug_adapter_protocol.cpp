/**************************************************************************/
/*  debug_adapter_protocol.cpp                                            */
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

#include "debug_adapter_protocol.h"

#include "core/config/project_settings.h"
#include "core/debugger/debugger_marshalls.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "editor/debugger/debug_adapter/debug_adapter_parser.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/run/editor_run_bar.h"
#include "editor/settings/editor_settings.h"

DebugAdapterProtocol *DebugAdapterProtocol::singleton = nullptr;

Error DAPeer::handle_data() {
	int read = 0;
	// Read headers
	if (!has_header) {
		if (!connection->get_available_bytes()) {
			return OK;
		}
		while (true) {
			if (req_pos >= DAP_MAX_BUFFER_SIZE) {
				req_pos = 0;
				ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Response header too big");
			}
			Error err = connection->get_partial_data(&req_buf[req_pos], 1, read);
			if (err != OK) {
				return FAILED;
			} else if (read != 1) { // Busy, wait until next poll
				return ERR_BUSY;
			}
			char *r = (char *)req_buf;
			int l = req_pos;

			// End of headers
			if (l > 3 && r[l] == '\n' && r[l - 1] == '\r' && r[l - 2] == '\n' && r[l - 3] == '\r') {
				r[l - 3] = '\0'; // Null terminate to read string
				String header = String::utf8(r);
				content_length = header.substr(16).to_int();
				has_header = true;
				req_pos = 0;
				break;
			}
			req_pos++;
		}
	}
	if (has_header) {
		while (req_pos < content_length) {
			if (content_length >= DAP_MAX_BUFFER_SIZE) {
				req_pos = 0;
				has_header = false;
				ERR_FAIL_COND_V_MSG(req_pos >= DAP_MAX_BUFFER_SIZE, ERR_OUT_OF_MEMORY, "Response content too big");
			}
			Error err = connection->get_partial_data(&req_buf[req_pos], content_length - req_pos, read);
			if (err != OK) {
				return FAILED;
			} else if (read < content_length - req_pos) {
				return ERR_BUSY;
			}
			req_pos += read;
		}

		// Parse data
		String msg = String::utf8((const char *)req_buf, req_pos);

		// Apply a timestamp if it there's none yet
		if (!timestamp) {
			timestamp = OS::get_singleton()->get_ticks_msec();
		}

		// Response
		if (DebugAdapterProtocol::get_singleton()->process_message(msg)) {
			// Reset to read again
			req_pos = 0;
			has_header = false;
			timestamp = 0;
		}
	}
	return OK;
}

Error DAPeer::send_data() {
	while (res_queue.size()) {
		Dictionary data = res_queue.front()->get();
		if (!data.has("seq")) {
			data["seq"] = ++seq;
		}
		const Vector<uint8_t> &formatted_data = format_output(data);

		int data_sent = 0;
		while (data_sent < formatted_data.size()) {
			int curr_sent = 0;
			Error err = connection->put_partial_data(formatted_data.ptr() + data_sent, formatted_data.size() - data_sent, curr_sent);
			if (err != OK) {
				return err;
			}
			data_sent += curr_sent;
		}
		res_queue.pop_front();
	}
	return OK;
}

Vector<uint8_t> DAPeer::format_output(const Dictionary &p_params) const {
	const Vector<uint8_t> &content = Variant(p_params).to_json_string().to_utf8_buffer();
	Vector<uint8_t> response = vformat("Content-Length: %d\r\n\r\n", content.size()).to_utf8_buffer();

	response.append_array(content);
	return response;
}

Error DebugAdapterProtocol::on_client_connected() {
	ERR_FAIL_COND_V_MSG(clients.size() >= DAP_MAX_CLIENTS, FAILED, "Max client limits reached");

	Ref<StreamPeerTCP> tcp_peer = server->take_connection();
	ERR_FAIL_COND_V_MSG(tcp_peer.is_null(), FAILED, "Failed to take incoming DAP connection.");
	tcp_peer->set_no_delay(true);
	Ref<DAPeer> peer = memnew(DAPeer);
	peer->connection = tcp_peer;
	clients.push_back(peer);

	EditorDebuggerNode::get_singleton()->get_default_debugger()->set_external_debugger(true);
	EditorNode::get_log()->add_message("[DAP] Connection Taken", EditorLog::MSG_TYPE_EDITOR);

	// Main thread always exists, but when DAP requests it, Godot might not have such data
	// created already. But since this is necessary for some requests (e.g. pausing), we
	// create it here manually instead.
	if (!thread_data_list.has(Thread::MAIN_ID)) {
		Ref<ThreadData> main_thread_data;
		main_thread_data.instantiate();
		main_thread_data->name = TTR("Main Thread");
		thread_data_list.insert(Thread::MAIN_ID, main_thread_data);
	}

	return OK;
}

void DebugAdapterProtocol::on_client_disconnected(const Ref<DAPeer> &p_peer) {
	clients.erase(p_peer);
	if (!clients.size()) {
		reset_ids();
		EditorDebuggerNode::get_singleton()->get_default_debugger()->set_external_debugger(false);
	}
	EditorNode::get_log()->add_message("[DAP] Disconnected", EditorLog::MSG_TYPE_EDITOR);
}

DebugAdapterProtocol::DAPRemoteID DebugAdapterProtocol::generate_remote_id(const Ref<ThreadData> &p_thread) {
	DAPRemoteID id = object_id++;
	thread_remote_data_lookup.insert(id, p_thread);
	return id;
}

void DebugAdapterProtocol::reset_current_info() {
	_current_request = "";
	_current_peer.unref();
}

void DebugAdapterProtocol::reset_ids() {
	breakpoint_id = 0;
	object_id = 0;

	thread_data_list.clear();
	thread_remote_data_lookup.clear();
	breakpoint_list.clear();
	breakpoint_source_list.clear();
	scope_list.clear();
	variable_list.clear();
	object_list.clear();
	object_pending_list.clear();
	eval_list.clear();
	eval_pending_list.clear();
}

int DebugAdapterProtocol::parse_variant(const Variant &p_var, const Ref<ThreadData> &p_thread) {
	switch (p_var.get_type()) {
		case Variant::VECTOR2:
		case Variant::VECTOR2I: {
			int id = generate_remote_id(p_thread);
			Vector2 vec = p_var;
			const String type_scalar = Variant::get_type_name(p_var.get_type() == Variant::VECTOR2 ? Variant::FLOAT : Variant::INT);
			DAP::Variable x, y;
			x.name = "x";
			y.name = "y";
			x.type = type_scalar;
			y.type = type_scalar;
			x.value = rtos(vec.x);
			y.value = rtos(vec.y);

			Array arr = { x.to_json(), y.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::RECT2:
		case Variant::RECT2I: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Rect2 rect = p_var;
			const String type_scalar = Variant::get_type_name(p_var.get_type() == Variant::RECT2 ? Variant::FLOAT : Variant::INT);
			DAP::Variable x, y, w, h;
			x.name = "x";
			y.name = "y";
			w.name = "w";
			h.name = "h";
			x.type = type_scalar;
			y.type = type_scalar;
			w.type = type_scalar;
			h.type = type_scalar;
			x.value = rtos(rect.position.x);
			y.value = rtos(rect.position.y);
			w.value = rtos(rect.size.x);
			h.value = rtos(rect.size.y);

			Array arr = { x.to_json(), y.to_json(), w.to_json(), h.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::VECTOR3:
		case Variant::VECTOR3I: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Vector3 vec = p_var;
			const String type_scalar = Variant::get_type_name(p_var.get_type() == Variant::VECTOR3 ? Variant::FLOAT : Variant::INT);
			DAP::Variable x, y, z;
			x.name = "x";
			y.name = "y";
			z.name = "z";
			x.type = type_scalar;
			y.type = type_scalar;
			z.type = type_scalar;
			x.value = rtos(vec.x);
			y.value = rtos(vec.y);
			z.value = rtos(vec.z);

			Array arr = { x.to_json(), y.to_json(), z.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::TRANSFORM2D: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Transform2D transform = p_var;
			const String type_vec2 = Variant::get_type_name(Variant::VECTOR2);
			DAP::Variable x, y, origin;
			x.name = "x";
			y.name = "y";
			origin.name = "origin";
			x.type = type_vec2;
			y.type = type_vec2;
			origin.type = type_vec2;
			x.value = String(transform.columns[0]);
			y.value = String(transform.columns[1]);
			origin.value = String(transform.columns[2]);
			x.variablesReference = parse_variant(transform.columns[0], p_thread);
			y.variablesReference = parse_variant(transform.columns[1], p_thread);
			origin.variablesReference = parse_variant(transform.columns[2], p_thread);

			Array arr = { x.to_json(), y.to_json(), origin.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PLANE: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Plane plane = p_var;
			DAP::Variable d, normal;
			d.name = "d";
			normal.name = "normal";
			d.type = Variant::get_type_name(Variant::FLOAT);
			normal.type = Variant::get_type_name(Variant::VECTOR3);
			d.value = rtos(plane.d);
			normal.value = String(plane.normal);
			normal.variablesReference = parse_variant(plane.normal, p_thread);

			Array arr = { d.to_json(), normal.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::QUATERNION: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Quaternion quat = p_var;
			const String type_float = Variant::get_type_name(Variant::FLOAT);
			DAP::Variable x, y, z, w;
			x.name = "x";
			y.name = "y";
			z.name = "z";
			w.name = "w";
			x.type = type_float;
			y.type = type_float;
			z.type = type_float;
			w.type = type_float;
			x.value = rtos(quat.x);
			y.value = rtos(quat.y);
			z.value = rtos(quat.z);
			w.value = rtos(quat.w);

			Array arr = { x.to_json(), y.to_json(), z.to_json(), w.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::AABB: {
			DAPRemoteID id = generate_remote_id(p_thread);
			AABB aabb = p_var;
			const String type_vec3 = Variant::get_type_name(Variant::VECTOR3);
			DAP::Variable position, size;
			position.name = "position";
			size.name = "size";
			position.type = type_vec3;
			size.type = type_vec3;
			position.value = String(aabb.position);
			size.value = String(aabb.size);
			position.variablesReference = parse_variant(aabb.position, p_thread);
			size.variablesReference = parse_variant(aabb.size, p_thread);

			Array arr = { position.to_json(), size.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::BASIS: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Basis basis = p_var;
			const String type_vec3 = Variant::get_type_name(Variant::VECTOR3);
			DAP::Variable x, y, z;
			x.name = "x";
			y.name = "y";
			z.name = "z";
			x.type = type_vec3;
			y.type = type_vec3;
			z.type = type_vec3;
			x.value = String(basis.rows[0]);
			y.value = String(basis.rows[1]);
			z.value = String(basis.rows[2]);
			x.variablesReference = parse_variant(basis.rows[0], p_thread);
			y.variablesReference = parse_variant(basis.rows[1], p_thread);
			z.variablesReference = parse_variant(basis.rows[2], p_thread);

			Array arr = { x.to_json(), y.to_json(), z.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::TRANSFORM3D: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Transform3D transform = p_var;
			DAP::Variable basis, origin;
			basis.name = "basis";
			origin.name = "origin";
			basis.type = Variant::get_type_name(Variant::BASIS);
			origin.type = Variant::get_type_name(Variant::VECTOR3);
			basis.value = String(transform.basis);
			origin.value = String(transform.origin);
			basis.variablesReference = parse_variant(transform.basis, p_thread);
			origin.variablesReference = parse_variant(transform.origin, p_thread);

			Array arr = { basis.to_json(), origin.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::COLOR: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Color color = p_var;
			const String type_float = Variant::get_type_name(Variant::FLOAT);
			DAP::Variable r, g, b, a;
			r.name = "r";
			g.name = "g";
			b.name = "b";
			a.name = "a";
			r.type = type_float;
			g.type = type_float;
			b.type = type_float;
			a.type = type_float;
			r.value = rtos(color.r);
			g.value = rtos(color.g);
			b.value = rtos(color.b);
			a.value = rtos(color.a);

			Array arr = { r.to_json(), g.to_json(), b.to_json(), a.to_json() };
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = Variant::get_type_name(array[i].get_type());
				var.value = array[i];
				var.variablesReference = parse_variant(array[i], p_thread);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::DICTIONARY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			Dictionary dictionary = p_var;
			Array arr;

			for (const KeyValue<Variant, Variant> &kv : dictionary) {
				DAP::Variable var;
				var.name = kv.key;
				Variant value = kv.value;
				var.type = Variant::get_type_name(value.get_type());
				var.value = value;
				var.variablesReference = parse_variant(value, p_thread);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_BYTE_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedByteArray array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = "byte";
				var.value = itos(array[i]);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_INT32_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedInt32Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = "int";
				var.value = itos(array[i]);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_INT64_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedInt64Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = "long";
				var.value = itos(array[i]);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedFloat32Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = "float";
				var.value = rtos(array[i]);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedFloat64Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = "double";
				var.value = rtos(array[i]);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_STRING_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedStringArray array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = Variant::get_type_name(Variant::STRING);
				var.value = array[i];
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedVector2Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = Variant::get_type_name(Variant::VECTOR2);
				var.value = String(array[i]);
				var.variablesReference = parse_variant(array[i], p_thread);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedVector3Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = Variant::get_type_name(Variant::VECTOR3);
				var.value = String(array[i]);
				var.variablesReference = parse_variant(array[i], p_thread);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_COLOR_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedColorArray array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr = { size.to_json() };

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = Variant::get_type_name(Variant::COLOR);
				var.value = String(array[i]);
				var.variablesReference = parse_variant(array[i], p_thread);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::PACKED_VECTOR4_ARRAY: {
			DAPRemoteID id = generate_remote_id(p_thread);
			PackedVector4Array array = p_var;
			DAP::Variable size;
			size.name = "size";
			size.type = Variant::get_type_name(Variant::INT);
			size.value = itos(array.size());

			Array arr;
			arr.push_back(size.to_json());

			for (int i = 0; i < array.size(); i++) {
				DAP::Variable var;
				var.name = itos(i);
				var.type = Variant::get_type_name(Variant::VECTOR4);
				var.value = String(array[i]);
				var.variablesReference = parse_variant(array[i], p_thread);
				arr.push_back(var.to_json());
			}
			variable_list.insert(id, arr);
			return id;
		}
		case Variant::OBJECT: {
			// Objects have to be requested from the debuggee. This has do be done
			// in a lazy way, as retrieving object properties takes time.
			EncodedObjectAsID *encoded_obj = Object::cast_to<EncodedObjectAsID>(p_var);

			// Object may be null; in that case, return early.
			if (!encoded_obj) {
				return 0;
			}

			// Object may have been already requested.
			ObjectID remote_object_id = encoded_obj->get_object_id();
			if (object_list.has(remote_object_id)) {
				return object_list[remote_object_id];
			}

			// Queue requesting the object.
			DAPRemoteID id = generate_remote_id(p_thread);
			object_list.insert(remote_object_id, id);
			return id;
		}
		default:
			// Simple atomic stuff, or too complex to be manipulated
			return 0;
	}
}

void DebugAdapterProtocol::parse_object(SceneDebuggerObject &p_obj) {
	// If the object is not on the pending list, we weren't expecting it. Ignore it.
	ObjectID remote_object_id = p_obj.id;
	if (!object_pending_list.has(remote_object_id)) {
		return;
	}

	ERR_FAIL_COND(!object_list.has(remote_object_id));
	const Ref<ThreadData> thread = object_pending_list[remote_object_id];
	object_pending_list.erase(remote_object_id);

	// Populate DAP::Variable's with the object's properties. These properties will be divided by categories.
	Array properties;
	Array script_members;
	Array script_constants;
	Array script_node;
	DAP::Variable node_type;
	Array node_properties;

	for (SceneDebuggerObject::SceneDebuggerProperty &property : p_obj.properties) {
		PropertyInfo &info = property.first;

		// Script members ("Members/" prefix)
		if (info.name.begins_with("Members/")) {
			info.name = info.name.trim_prefix("Members/");
			script_members.push_back(parse_object_variable(property, thread));
		}

		// Script constants ("Constants/" prefix)
		else if (info.name.begins_with("Constants/")) {
			info.name = info.name.trim_prefix("Constants/");
			script_constants.push_back(parse_object_variable(property, thread));
		}

		// Script node ("Node/" prefix)
		else if (info.name.begins_with("Node/")) {
			info.name = info.name.trim_prefix("Node/");
			script_node.push_back(parse_object_variable(property, thread));
		}

		// Regular categories (with type Variant::NIL)
		else if (info.type == Variant::NIL) {
			if (!node_properties.is_empty()) {
				node_type.value = itos(node_properties.size());
				variable_list.insert(node_type.variablesReference, node_properties.duplicate());
				properties.push_back(node_type.to_json());
			}

			node_type.name = info.name;
			node_type.type = "Category";
			node_type.variablesReference = generate_remote_id(thread);
			node_properties.clear();
		}

		// Regular properties.
		else {
			node_properties.push_back(parse_object_variable(property, thread));
		}
	}

	// Add the last category.
	if (!node_properties.is_empty()) {
		node_type.value = itos(node_properties.size());
		variable_list.insert(node_type.variablesReference, node_properties.duplicate());
		properties.push_back(node_type.to_json());
	}

	// Add the script categories, in reverse order to be at the front of the array:
	// ( [members; constants; node; category1; category2; ...] )
	if (!script_node.is_empty()) {
		DAP::Variable node;
		node.name = "Node";
		node.type = "Category";
		node.value = itos(script_node.size());
		node.variablesReference = generate_remote_id(thread);
		variable_list.insert(node.variablesReference, script_node);
		properties.push_front(node.to_json());
	}

	if (!script_constants.is_empty()) {
		DAP::Variable constants;
		constants.name = "Constants";
		constants.type = "Category";
		constants.value = itos(script_constants.size());
		constants.variablesReference = generate_remote_id(thread);
		variable_list.insert(constants.variablesReference, script_constants);
		properties.push_front(constants.to_json());
	}

	if (!script_members.is_empty()) {
		DAP::Variable members;
		members.name = "Members";
		members.type = "Category";
		members.value = itos(script_members.size());
		members.variablesReference = generate_remote_id(thread);
		variable_list.insert(members.variablesReference, script_members);
		properties.push_front(members.to_json());
	}

	variable_list.insert(object_list[remote_object_id], properties);
}

void DebugAdapterProtocol::parse_evaluation(DebuggerMarshalls::ScriptStackVariable &p_var) {
	// If the eval is not on the pending list, we weren't expecting it. Ignore it.
	String eval = p_var.name;
	if (!eval_pending_list.erase(eval)) {
		return;
	}

	DAP::Variable variable;
	variable.name = p_var.name;
	variable.value = p_var.value;
	variable.type = Variant::get_type_name(p_var.value.get_type());
	// Godot currently only performs evaluations on the main thread.
	variable.variablesReference = parse_variant(p_var.value, thread_data_list[Thread::MAIN_ID]);

	eval_list.insert(variable.name, variable);
}

const Variant DebugAdapterProtocol::parse_object_variable(const SceneDebuggerObject::SceneDebuggerProperty &p_property, const Ref<ThreadData> &p_thread) {
	const PropertyInfo &info = p_property.first;
	const Variant &value = p_property.second;

	DAP::Variable var;
	var.name = info.name;
	var.type = Variant::get_type_name(info.type);
	var.value = value;
	var.variablesReference = parse_variant(value, p_thread);

	return var.to_json();
}

ObjectID DebugAdapterProtocol::search_object_id(DAPRemoteID p_object_id) {
	for (const KeyValue<ObjectID, DAPRemoteID> &E : object_list) {
		if (E.value == p_object_id) {
			return E.key;
		}
	}
	return ObjectID();
}

bool DebugAdapterProtocol::request_remote_object(const ObjectID &p_object_id, const Ref<ThreadData> &p_thread) {
	// If the object is already on the pending list, we don't need to request it again.
	if (object_pending_list.has(p_object_id)) {
		return false;
	}

	TypedArray<uint64_t> arr;
	arr.append(p_object_id);
	EditorDebuggerNode::get_singleton()->get_default_debugger()->request_remote_objects(arr);
	object_pending_list.insert(p_object_id, p_thread);

	return true;
}

bool DebugAdapterProtocol::request_remote_evaluate(const String &p_eval, int p_stack_frame) {
	// If the eval is already on the pending list, we don't need to request it again
	if (eval_pending_list.has(p_eval)) {
		return false;
	}

	EditorDebuggerNode::get_singleton()->get_default_debugger()->request_remote_evaluate(p_eval, p_stack_frame);
	eval_pending_list.insert(p_eval);

	return true;
}

const DAP::Source &DebugAdapterProtocol::fetch_source(const String &p_path) {
	const String &global_path = ProjectSettings::get_singleton()->globalize_path(p_path);

	HashMap<String, DAP::Source>::Iterator E = breakpoint_source_list.find(global_path);
	if (E != breakpoint_source_list.end()) {
		return E->value;
	}
	DAP::Source &added_source = breakpoint_source_list.insert(global_path, DAP::Source())->value;
	added_source.name = global_path.get_file();
	added_source.path = global_path;
	added_source.compute_checksums();

	return added_source;
}

void DebugAdapterProtocol::update_source(const String &p_path) {
	const String &global_path = ProjectSettings::get_singleton()->globalize_path(p_path);

	HashMap<String, DAP::Source>::Iterator E = breakpoint_source_list.find(global_path);
	if (E != breakpoint_source_list.end()) {
		E->value.compute_checksums();
	}
}

bool DebugAdapterProtocol::process_message(const String &p_text) {
	JSON json;
	ERR_FAIL_COND_V_MSG(json.parse(p_text) != OK, true, "Malformed message!");
	Dictionary params = json.get_data();
	bool completed = true;

	// While JSON does not distinguish floats and ints, "seq" is an integer by specification. See https://github.com/godotengine/godot/issues/108288
	if (params.has("seq")) {
		params["seq"] = (int)params["seq"];
	}

	if (OS::get_singleton()->get_ticks_msec() - _current_peer->timestamp > _request_timeout) {
		Dictionary response = parser->prepare_error_response(params, DAP::ErrorType::TIMEOUT);
		_current_peer->res_queue.push_front(response);
		return true;
	}

	// Append "req_" to any command received; prevents name clash with existing functions, and possibly exploiting
	String command = "req_" + (String)params["command"];
	if (parser->has_method(command)) {
		_current_request = params["command"];

		Array args = { params };
		Dictionary response = parser->callv(command, args);
		if (!response.is_empty()) {
			_current_peer->res_queue.push_front(response);
		} else {
			// Launch request needs to be deferred until we receive a configurationDone request.
			if (command != "req_launch") {
				completed = false;
			}
		}
	}

	reset_current_info();
	return completed;
}

void DebugAdapterProtocol::notify_initialized() {
	Dictionary event = parser->ev_initialized();
	_current_peer->res_queue.push_back(event);
}

void DebugAdapterProtocol::notify_process() {
	String launch_mode = _current_peer->attached ? "attach" : "launch";

	Dictionary event = parser->ev_process(launch_mode);
	for (const Ref<DAPeer> &peer : clients) {
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_terminated() {
	Dictionary event = parser->ev_terminated();
	for (const Ref<DAPeer> &peer : clients) {
		if ((_current_request == "launch" || _current_request == "restart") && _current_peer == peer) {
			continue;
		}
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_exited(int p_exitcode) {
	Dictionary event = parser->ev_exited(p_exitcode);
	for (const Ref<DAPeer> &peer : clients) {
		if ((_current_request == "launch" || _current_request == "restart") && _current_peer == peer) {
			continue;
		}
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_stopped_paused(Thread::ID p_thread_id) {
	Dictionary event = parser->ev_stopped_paused(p_thread_id);
	for (const Ref<DAPeer> &peer : clients) {
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_stopped_exception(const String &p_error, Thread::ID p_thread_id) {
	Dictionary event = parser->ev_stopped_exception(p_error, p_thread_id);
	for (const Ref<DAPeer> &peer : clients) {
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_stopped_breakpoint(int p_id, Thread::ID p_thread_id) {
	Dictionary event = parser->ev_stopped_breakpoint(p_id, p_thread_id);
	for (const Ref<DAPeer> &peer : clients) {
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_stopped_step(Thread::ID p_thread_id) {
	Dictionary event = parser->ev_stopped_step(p_thread_id);
	for (const Ref<DAPeer> &peer : clients) {
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_continued(Thread::ID p_thread_id) {
	Dictionary event = parser->ev_continued(p_thread_id);
	for (const Ref<DAPeer> &peer : clients) {
		if (_current_request == "continue" && peer == _current_peer) {
			continue;
		}
		peer->res_queue.push_back(event);
	}

	reset_ids();
}

void DebugAdapterProtocol::notify_output(const String &p_message, RemoteDebugger::MessageType p_type) {
	Dictionary event = parser->ev_output(p_message, p_type);
	for (const Ref<DAPeer> &peer : clients) {
		peer->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_custom_data(const String &p_msg, const Array &p_data) {
	Dictionary event = parser->ev_custom_data(p_msg, p_data);
	for (const Ref<DAPeer> &peer : clients) {
		if (peer->supportsCustomData) {
			peer->res_queue.push_back(event);
		}
	}
}

void DebugAdapterProtocol::notify_breakpoint(const DAP::Breakpoint &p_breakpoint, bool p_enabled) {
	Dictionary event = parser->ev_breakpoint(p_breakpoint, p_enabled);
	for (const Ref<DAPeer> &peer : clients) {
		if (_current_request == "setBreakpoints" && peer == _current_peer) {
			continue;
		}
		peer->res_queue.push_back(event);
	}
}

Array DebugAdapterProtocol::update_breakpoints(const String &p_path, const Array &p_lines) {
	Array updated_breakpoints;

	// Add breakpoints
	for (int i = 0; i < p_lines.size(); i++) {
		DAP::Breakpoint breakpoint(fetch_source(p_path));
		breakpoint.line = p_lines[i];

		// Avoid duplicated entries.
		List<DAP::Breakpoint>::Element *E = breakpoint_list.find(breakpoint);
		if (E) {
			updated_breakpoints.push_back(E->get().to_json());
			continue;
		}

		EditorDebuggerNode::get_singleton()->get_default_debugger()->_set_breakpoint(p_path, p_lines[i], true);

		// Breakpoints are inserted at the end of the breakpoint list.
		List<DAP::Breakpoint>::Element *added_breakpoint = breakpoint_list.back();
		ERR_FAIL_NULL_V(added_breakpoint, Array());
		ERR_FAIL_COND_V(!(added_breakpoint->get() == breakpoint), Array());
		updated_breakpoints.push_back(added_breakpoint->get().to_json());
	}

	// Remove breakpoints
	// Must be deferred because we are iterating the breakpoint list.
	Vector<int> to_remove;

	for (const DAP::Breakpoint &b : breakpoint_list) {
		if (b.source->path == p_path && !p_lines.has(b.line)) {
			to_remove.push_back(b.line);
		}
	}

	// Safe to remove queued data now.
	for (const int &line : to_remove) {
		EditorDebuggerNode::get_singleton()->get_default_debugger()->_set_breakpoint(p_path, line, false);
	}

	return updated_breakpoints;
}

void DebugAdapterProtocol::on_debug_paused() {
	// Godot currently only supports pausing/resuming the main thread.
	if (EditorRunBar::get_singleton()->get_pause_button()->is_pressed()) {
		notify_stopped_paused(Thread::MAIN_ID);
	} else {
		notify_continued(Thread::MAIN_ID);
	}
}

void DebugAdapterProtocol::on_debug_stopped() {
	notify_exited();
	notify_terminated();
	reset_ids();
}

void DebugAdapterProtocol::on_debug_output(const String &p_message, int p_type) {
	notify_output(p_message, RemoteDebugger::MessageType(p_type));
}

void DebugAdapterProtocol::on_debug_breaked(bool p_reallydid, bool p_can_debug, const String &p_reason, bool p_has_stackdump, Thread::ID p_thread_id) {
	if (p_thread_id == Thread::UNASSIGNED_ID) {
		return;
	}

	if (!p_reallydid) {
		notify_continued(p_thread_id);
		return;
	}

	if (p_reason.is_empty()) {
		notify_stopped_paused(p_thread_id);
		return;
	}

	ERR_FAIL_COND(!thread_data_list.has(p_thread_id));
	const Ref<ThreadData> &thread = thread_data_list[p_thread_id];

	if (p_reason == "Breakpoint") {
		if (thread->stepping) {
			notify_stopped_step(p_thread_id);
			thread->stepping = false;
		} else {
			thread->processing_breakpoint = true; // Wait for stack_dump to find where the breakpoint happened
		}
	} else {
		notify_stopped_exception(p_reason, p_thread_id);
	}

	thread->processing_stackdump = p_has_stackdump;
}

void DebugAdapterProtocol::on_debug_breakpoint_toggled(const String &p_path, int p_line, bool p_enabled) {
	DAP::Breakpoint breakpoint(fetch_source(p_path));
	breakpoint.verified = true;
	breakpoint.line = p_line;

	if (p_enabled) {
		// Add the breakpoint
		breakpoint.id = breakpoint_id++;
		breakpoint_list.push_back(breakpoint);
	} else {
		// Remove the breakpoint
		List<DAP::Breakpoint>::Element *E = breakpoint_list.find(breakpoint);
		if (E) {
			breakpoint.id = E->get().id;
			breakpoint_list.erase(E);
		}
	}

	notify_breakpoint(breakpoint, p_enabled);
}

void DebugAdapterProtocol::on_debug_stack_dump(const Array &p_stack_dump, Thread::ID p_thread_id) {
	if (p_stack_dump.is_empty()) {
		return;
	}

	ERR_FAIL_COND(!thread_data_list.has(p_thread_id));
	const Ref<ThreadData> &thread = thread_data_list[p_thread_id];

	if (thread->processing_breakpoint) {
		// Find existing breakpoint
		Dictionary d = p_stack_dump.front();
		DAP::Breakpoint breakpoint(fetch_source(d["file"]));
		breakpoint.line = d["line"];

		List<DAP::Breakpoint>::Element *E = breakpoint_list.find(breakpoint);
		if (E) {
			notify_stopped_breakpoint(E->get().id, p_thread_id);
		}

		thread->processing_breakpoint = false;
	}

	// Assign the current object ID as the current frame. This ID will be assigned
	// to a DAP::StackFrame object below, since we know the stack frame list isn't empty.
	thread->current_frame = object_id;
	thread->processing_stackdump = false;
	for (const DAP::StackFrame &stackframe : thread->stackframe_list) {
		scope_list.erase(stackframe.id);
	}
	thread->stackframe_list.clear();
	thread->godot_stackframe_ids.clear();

	// Fill in stacktrace information
	for (int i = 0; i < p_stack_dump.size(); i++) {
		Dictionary stack_info = p_stack_dump[i];

		DAP::StackFrame stackframe(fetch_source(stack_info["file"]));
		stackframe.id = generate_remote_id(thread);
		stackframe.name = stack_info["function"];
		stackframe.line = stack_info["line"];
		stackframe.column = 0;

		// Information for "Locals", "Members" and "Globals" variables respectively
		Vector<int> scope_ids;
		for (int j = 0; j < 3; j++) {
			scope_ids.push_back(generate_remote_id(thread));
		}

		thread->stackframe_list.push_back(stackframe);
		thread->godot_stackframe_ids.insert(stackframe.id, i);
		scope_list.insert(stackframe.id, scope_ids);
	}
}

void DebugAdapterProtocol::on_debug_stack_frame_vars(int p_size, Thread::ID p_thread_id) {
	ERR_FAIL_COND(!thread_data_list.has(p_thread_id));
	const Ref<ThreadData> &thread = thread_data_list[p_thread_id];

	thread->remaining_vars = p_size;
	ERR_FAIL_COND(!scope_list.has(thread->current_frame));
	for (const int &var_id : scope_list[thread->current_frame]) {
		if (variable_list.has(var_id)) {
			variable_list.find(var_id)->value.clear();
		} else {
			variable_list.insert(var_id, Array());
		}
	}
}

void DebugAdapterProtocol::on_debug_stack_frame_var(const Array &p_data, Thread::ID p_thread_id) {
	ERR_FAIL_COND(!thread_data_list.has(p_thread_id));
	Ref<ThreadData> &thread = thread_data_list[p_thread_id];

	DebuggerMarshalls::ScriptStackVariable stack_var;
	stack_var.deserialize(p_data);

	ERR_FAIL_COND(!scope_list.has(thread->current_frame));
	const Vector<DAPRemoteID> &scope_ids = scope_list[thread->current_frame];

	ERR_FAIL_COND(scope_ids.size() != 3);
	ERR_FAIL_INDEX(stack_var.type, 4);
	DAPRemoteID var_id = scope_ids.get(stack_var.type);

	DAP::Variable variable;

	variable.name = stack_var.name;
	variable.value = stack_var.value;
	variable.type = Variant::get_type_name(stack_var.value.get_type());
	variable.variablesReference = parse_variant(stack_var.value, thread);

	variable_list.find(var_id)->value.push_back(variable.to_json());
	thread->remaining_vars--;
}

void DebugAdapterProtocol::on_debug_data(const String &p_msg, const Array &p_data, Thread::ID p_thread_id) {
	// Ignore data that is already handled by DAP
	if (p_msg == "debug_exit" || p_msg == "stack_dump" || p_msg == "stack_frame_vars" || p_msg == "stack_frame_var" || p_msg == "output" || p_msg == "request_quit") {
		return;
	}

	if (p_msg == "debug_enter") {
		if (!thread_data_list.has(p_thread_id)) {
			Ref<ThreadData> data;
			data.instantiate();
			data->name = (p_thread_id == Thread::get_main_id()) ? TTR("Main Thread") : itos(p_thread_id);
			thread_data_list.insert(p_thread_id, data);
		}
	} else if (p_msg == "scene:inspect_objects") {
		if (!p_data.is_empty()) {
			// An object was requested from the debuggee; parse it.
			SceneDebuggerObject remote_obj;
			remote_obj.deserialize(p_data[0]);

			parse_object(remote_obj);
		}
#ifndef DISABLE_DEPRECATED
	} else if (p_msg == "scene:inspect_object") {
		if (!p_data.is_empty()) {
			// Legacy single object response format.
			SceneDebuggerObject remote_obj;
			remote_obj.deserialize(p_data);

			parse_object(remote_obj);
		}
#endif // DISABLE_DEPRECATED
	} else if (p_msg == "evaluation_return") {
		// An evaluation was requested from the debuggee; parse it.
		DebuggerMarshalls::ScriptStackVariable remote_evaluation;
		remote_evaluation.deserialize(p_data);

		parse_evaluation(remote_evaluation);
	}

	notify_custom_data(p_msg, p_data);
}

void DebugAdapterProtocol::poll() {
	if (server->is_connection_available()) {
		on_client_connected();
	}
	List<Ref<DAPeer>> to_delete;
	for (const Ref<DAPeer> &peer : clients) {
		peer->connection->poll();
		StreamPeerTCP::Status status = peer->connection->get_status();
		if (status == StreamPeerTCP::STATUS_NONE || status == StreamPeerTCP::STATUS_ERROR) {
			to_delete.push_back(peer);
		} else {
			_current_peer = peer;
			Error err = peer->handle_data();
			if (err != OK && err != ERR_BUSY) {
				to_delete.push_back(peer);
			}
			err = peer->send_data();
			if (err != OK && err != ERR_BUSY) {
				to_delete.push_back(peer);
			}
		}
	}

	for (const Ref<DAPeer> &peer : to_delete) {
		on_client_disconnected(peer);
	}
	to_delete.clear();
}

Error DebugAdapterProtocol::start(int p_port, const IPAddress &p_bind_ip) {
	_request_timeout = (uint64_t)_EDITOR_GET("network/debug_adapter/request_timeout");
	_sync_breakpoints = (bool)_EDITOR_GET("network/debug_adapter/sync_breakpoints");
	_initialized = true;
	return server->listen(p_port, p_bind_ip);
}

void DebugAdapterProtocol::stop() {
	for (const Ref<DAPeer> &peer : clients) {
		peer->connection->disconnect_from_host();
	}

	clients.clear();
	server->stop();
	_initialized = false;
}

DebugAdapterProtocol::DebugAdapterProtocol() {
	server.instantiate();
	singleton = this;
	parser = memnew(DebugAdapterParser);

	reset_ids();

	EditorRunBar::get_singleton()->get_pause_button()->connect(SceneStringName(pressed), callable_mp(this, &DebugAdapterProtocol::on_debug_paused));

	EditorDebuggerNode *debugger_node = EditorDebuggerNode::get_singleton();
	debugger_node->connect("breakpoint_toggled", callable_mp(this, &DebugAdapterProtocol::on_debug_breakpoint_toggled));

	debugger_node->get_default_debugger()->connect("stopped", callable_mp(this, &DebugAdapterProtocol::on_debug_stopped));
	debugger_node->get_default_debugger()->connect(SceneStringName(output), callable_mp(this, &DebugAdapterProtocol::on_debug_output));
	debugger_node->get_default_debugger()->connect("breaked", callable_mp(this, &DebugAdapterProtocol::on_debug_breaked));
	debugger_node->get_default_debugger()->connect("stack_dump", callable_mp(this, &DebugAdapterProtocol::on_debug_stack_dump));
	debugger_node->get_default_debugger()->connect("stack_frame_vars", callable_mp(this, &DebugAdapterProtocol::on_debug_stack_frame_vars));
	debugger_node->get_default_debugger()->connect("stack_frame_var", callable_mp(this, &DebugAdapterProtocol::on_debug_stack_frame_var));
	debugger_node->get_default_debugger()->connect("debug_data", callable_mp(this, &DebugAdapterProtocol::on_debug_data));
}

DebugAdapterProtocol::~DebugAdapterProtocol() {
	memdelete(parser);
}
