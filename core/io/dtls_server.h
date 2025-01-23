/**************************************************************************/
/*  dtls_server.h                                                         */
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

#ifndef DTLS_SERVER_H
#define DTLS_SERVER_H

#include "core/io/net_socket.h"
#include "core/io/packet_peer_dtls.h"

class DTLSServer : public RefCounted {
	GDCLASS(DTLSServer, RefCounted);

protected:
	static DTLSServer *(*_create)(bool p_notify_postinitialize);
	static void _bind_methods();

	static bool available;

public:
	static bool is_available();
	static DTLSServer *create(bool p_notify_postinitialize = true);

	virtual Error setup(Ref<TLSOptions> p_options) = 0;
	virtual void stop() = 0;
	virtual Ref<PacketPeerDTLS> take_connection(Ref<PacketPeerUDP> p_peer) = 0;

	DTLSServer() {}
};

#endif // DTLS_SERVER_H
