/**************************************************************************/
/*  uds_socket_unix.h                                                     */
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

#pragma once

#if defined(UNIX_ENABLED) && !defined(UNIX_SOCKET_UNAVAILABLE)

#include "core/io/uds_socket.h"

#include <sys/socket.h>
#include <sys/un.h>

class UDSSocketUnix : public UDSSocket {
private:
	int _sock = -1;
	CharString path;
	bool unlink_on_close = false;

	enum SocketError {
		ERR_SOCKET_WOULD_BLOCK,
		ERR_SOCKET_IS_CONNECTED,
		ERR_SOCKET_IN_PROGRESS,
		ERR_SOCKET_ADDRESS_INVALID_OR_UNAVAILABLE,
		ERR_SOCKET_UNAUTHORIZED,
		ERR_SOCKET_BUFFER_TOO_SMALL,
		ERR_SOCKET_OTHER,
	};

	SocketError _get_socket_error() const;
	_FORCE_INLINE_ void _set_close_exec_enabled(bool p_enabled);
	static socklen_t _set_sockaddr(struct sockaddr_un *p_addr, const CharString &p_path);

protected:
	static UDSSocket *_create_func();

public:
	static void make_default();
	static void cleanup();

	virtual Error open() override;
	virtual void close() override;
	virtual Error bind(const String &p_path) override;
	virtual Error listen(int p_max_pending) override;
	virtual Error connect_to_host(const String &p_path) override;
	virtual Error poll(PollType p_type, int timeout) const override;
	virtual Error recv(uint8_t *p_buffer, int p_len, int &r_read) override;
	virtual Error send(const uint8_t *p_buffer, int p_len, int &r_sent) override;
	virtual Ref<UDSSocket> accept() override;

	virtual bool is_open() const override;
	virtual int get_available_bytes() const override;

	virtual void set_blocking_enabled(bool p_enabled) override;

	UDSSocketUnix();
	~UDSSocketUnix() override;
};

#endif // UNIX_ENABLED && !UNIX_SOCKET_UNAVAILABLE
