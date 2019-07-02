/*************************************************************************/
/*  stream_peer_ssl.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "stream_peer_ssl.h"

#include "core/io/certs_compressed.gen.h"
#include "core/io/compression.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"

StreamPeerSSL *(*StreamPeerSSL::_create)() = NULL;

StreamPeerSSL *StreamPeerSSL::create() {

	if (_create)
		return _create();
	return NULL;
}

StreamPeerSSL::LoadCertsFromMemory StreamPeerSSL::load_certs_func = NULL;
bool StreamPeerSSL::available = false;

void StreamPeerSSL::load_certs_from_memory(const PoolByteArray &p_memory) {
	if (load_certs_func)
		load_certs_func(p_memory);
}

void StreamPeerSSL::load_certs_from_file(String p_path) {
	if (p_path != "") {
		PoolByteArray certs = get_cert_file_as_array(p_path);
		if (certs.size() > 0)
			load_certs_func(certs);
	}
}

bool StreamPeerSSL::is_available() {
	return available;
}

void StreamPeerSSL::set_blocking_handshake_enabled(bool p_enabled) {
	blocking_handshake = p_enabled;
}

bool StreamPeerSSL::is_blocking_handshake_enabled() const {
	return blocking_handshake;
}

PoolByteArray StreamPeerSSL::get_cert_file_as_array(String p_path) {

	PoolByteArray out;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (f) {
		int flen = f->get_len();
		out.resize(flen + 1);
		PoolByteArray::Write w = out.write();
		f->get_buffer(w.ptr(), flen);
		w[flen] = 0; // Make sure it ends with string terminator
		memdelete(f);
#ifdef DEBUG_ENABLED
		print_verbose(vformat("Loaded certs from '%s'.", p_path));
#endif
	}

	return out;
}

PoolByteArray StreamPeerSSL::get_project_cert_array() {

	PoolByteArray out;
	String certs_path = GLOBAL_DEF("network/ssl/certificates", "");
	ProjectSettings::get_singleton()->set_custom_property_info("network/ssl/certificates", PropertyInfo(Variant::STRING, "network/ssl/certificates", PROPERTY_HINT_FILE, "*.crt"));

	if (certs_path != "") {
		// Use certs defined in project settings.
		return get_cert_file_as_array(certs_path);
	}
#ifdef BUILTIN_CERTS_ENABLED
	else {
		// Use builtin certs only if user did not override it in project settings.
		out.resize(_certs_uncompressed_size + 1);
		PoolByteArray::Write w = out.write();
		Compression::decompress(w.ptr(), _certs_uncompressed_size, _certs_compressed, _certs_compressed_size, Compression::MODE_DEFLATE);
		w[_certs_uncompressed_size] = 0; // Make sure it ends with string terminator
#ifdef DEBUG_ENABLED
		print_verbose("Loaded builtin certs");
#endif
	}
#endif

	return out;
}

void StreamPeerSSL::_bind_methods() {

	ClassDB::bind_method(D_METHOD("poll"), &StreamPeerSSL::poll);
	ClassDB::bind_method(D_METHOD("accept_stream", "base"), &StreamPeerSSL::accept_stream);
	ClassDB::bind_method(D_METHOD("connect_to_stream", "stream", "validate_certs", "for_hostname"), &StreamPeerSSL::connect_to_stream, DEFVAL(false), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("get_status"), &StreamPeerSSL::get_status);
	ClassDB::bind_method(D_METHOD("disconnect_from_stream"), &StreamPeerSSL::disconnect_from_stream);
	ClassDB::bind_method(D_METHOD("set_blocking_handshake_enabled", "enabled"), &StreamPeerSSL::set_blocking_handshake_enabled);
	ClassDB::bind_method(D_METHOD("is_blocking_handshake_enabled"), &StreamPeerSSL::is_blocking_handshake_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "blocking_handshake"), "set_blocking_handshake_enabled", "is_blocking_handshake_enabled");

	BIND_ENUM_CONSTANT(STATUS_DISCONNECTED);
	BIND_ENUM_CONSTANT(STATUS_HANDSHAKING);
	BIND_ENUM_CONSTANT(STATUS_CONNECTED);
	BIND_ENUM_CONSTANT(STATUS_ERROR);
	BIND_ENUM_CONSTANT(STATUS_ERROR_HOSTNAME_MISMATCH);
}

StreamPeerSSL::StreamPeerSSL() {
	blocking_handshake = true;
}
