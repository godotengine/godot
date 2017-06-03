/*************************************************************************/
/*  tcp_server.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "tcp_server.h"

TCP_Server *(*TCP_Server::_create)() = NULL;

Ref<TCP_Server> TCP_Server::create_ref() {

	if (!_create)
		return NULL;
	return Ref<TCP_Server>(_create());
}

TCP_Server *TCP_Server::create() {

	if (!_create)
		return NULL;
	return _create();
}

void TCP_Server::_bind_methods() {

	ClassDB::bind_method(D_METHOD("listen", "port", "bind_address"), &TCP_Server::listen, DEFVAL("*"));
	ClassDB::bind_method(D_METHOD("is_connection_available"), &TCP_Server::is_connection_available);
	ClassDB::bind_method(D_METHOD("take_connection"), &TCP_Server::take_connection);
	ClassDB::bind_method(D_METHOD("stop"), &TCP_Server::stop);
}

TCP_Server::TCP_Server() {
}
