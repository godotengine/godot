/*************************************************************************/
/*  tcp_server.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef TCP_SERVER_H
#define TCP_SERVER_H

#include "io/stream_peer.h"
#include "io/ip.h"
#include "stream_peer_tcp.h"

class TCP_Server : public Reference {

	GDCLASS( TCP_Server, Reference );
protected:

	IP::Type ip_type;

	static TCP_Server* (*_create)();

	//bind helper
	Error _listen(uint16_t p_port, DVector<String> p_accepted_hosts=DVector<String>());
	static void _bind_methods();
public:

	virtual void set_ip_type(IP::Type p_type);
	virtual Error listen(uint16_t p_port, const List<String> *p_accepted_hosts=NULL)=0;
	virtual bool is_connection_available() const=0;
	virtual Ref<StreamPeerTCP> take_connection()=0;

	virtual void stop()=0; //stop listening

	static Ref<TCP_Server> create_ref();
	static TCP_Server* create();

	TCP_Server();
};

#endif // TCP_SERVER_H
