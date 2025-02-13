/**************************************************************************/
/*  dtls_server_mbedtls.h                                                 */
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

#ifndef DTLS_SERVER_MBEDTLS_H
#define DTLS_SERVER_MBEDTLS_H

#include "tls_context_mbedtls.h"

#include "core/io/dtls_server.h"

class DTLSServerMbedTLS : public DTLSServer {
private:
	static DTLSServer *_create_func(bool p_notify_postinitialize);
	Ref<TLSOptions> tls_options;
	Ref<CookieContextMbedTLS> cookies;

public:
	static void initialize();
	static void finalize();

	virtual Error setup(Ref<TLSOptions> p_options);
	virtual void stop();
	virtual Ref<PacketPeerDTLS> take_connection(Ref<PacketPeerUDP> p_peer);

	DTLSServerMbedTLS();
	~DTLSServerMbedTLS();
};

#endif // DTLS_SERVER_MBEDTLS_H
