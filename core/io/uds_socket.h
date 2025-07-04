/**************************************************************************/
/*  uds_socket.h                                                          */
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

#include "core/object/ref_counted.h"

class UDSSocket : public RefCounted {
	GDCLASS(UDSSocket, RefCounted);

protected:
	static UDSSocket *(*_create)();

public:
	static UDSSocket *create();

	enum PollType : int32_t {
		POLL_TYPE_IN,
		POLL_TYPE_OUT,
		POLL_TYPE_IN_OUT
	};

	virtual Error open() = 0;
	virtual void close() = 0;
	virtual Error bind(const String &p_path) = 0;
	virtual Error listen(int p_max_pending) = 0;
	virtual Error connect_to_host(const String &p_path) = 0;
	virtual Error poll(PollType p_type, int timeout) const = 0;
	virtual Error recv(uint8_t *p_buffer, int p_len, int &r_read) = 0;
	virtual Error send(const uint8_t *p_buffer, int p_len, int &r_sent) = 0;
	virtual Ref<UDSSocket> accept() = 0;

	virtual bool is_open() const = 0;
	virtual int get_available_bytes() const = 0;

	virtual void set_blocking_enabled(bool p_enabled) = 0;

	virtual ~UDSSocket() {}
};
