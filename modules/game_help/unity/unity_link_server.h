/**************************************************************************/
/*  editor_file_server.h                                                  */
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

#ifndef EDITOR_FILE_SERVER_H
#define EDITOR_FILE_SERVER_H

#include "core/io/packet_peer.h"
#include "core/io/tcp_server.h"
#include "core/object/class_db.h"
#include "core/os/thread.h"

class UnityLinkServer : public Object {
	GDCLASS(UnityLinkServer, Object);
	static void _bind_methods()
	{
		ClassDB::bind_method(D_METHOD("set_animation_load_callback", "callback"), &UnityLinkServer::set_animation_load_callback);
	}

	private:

	Ref<TCPServer> server;
	String password;
	Callable on_load_animation;
	int port = 0;
	bool active = false;


	void set_animation_load_callback(const Callable &p_callback)
	{
		on_load_animation = p_callback;
	}
public:
	void poll();

	void start();
	void stop();

	bool is_active() const;

	UnityLinkServer();
	~UnityLinkServer();
};

#endif // EDITOR_FILE_SERVER_H
