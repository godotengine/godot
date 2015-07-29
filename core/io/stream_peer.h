/*************************************************************************/
/*  stream_peer.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef STREAM_PEER_H
#define STREAM_PEER_H

#include "reference.h"

class StreamPeer : public Reference {
	OBJ_TYPE( StreamPeer, Reference );
	OBJ_CATEGORY("Networking");
protected:
	static void _bind_methods();

	//bind helpers
	Error _put_data(const DVector<uint8_t>& p_data);
	Array _put_partial_data(const DVector<uint8_t>& p_data);

	Array _get_data(int p_bytes);
	Array _get_partial_data(int p_bytes);

public:

	virtual Error put_data(const uint8_t* p_data,int p_bytes)=0; ///< put a whole chunk of data, blocking until it sent
	virtual Error put_partial_data(const uint8_t* p_data,int p_bytes, int &r_sent)=0; ///< put as much data as possible, without blocking.

	virtual Error get_data(uint8_t* p_buffer, int p_bytes)=0; ///< read p_bytes of data, if p_bytes > available, it will block
	virtual Error get_partial_data(uint8_t* p_buffer, int p_bytes,int &r_received)=0; ///< read as much data as p_bytes into buffer, if less was read, return in r_received

	StreamPeer() {}
};

#endif // STREAM_PEER_H
