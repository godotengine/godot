/*************************************************************************/
/*  stream_peer.cpp                                                      */
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
#include "stream_peer.h"


Error StreamPeer::_put_data(const DVector<uint8_t>& p_data) {

	int len = p_data.size();
	if (len==0)
		return OK;
	DVector<uint8_t>::Read r = p_data.read();
	return put_data(&r[0],len);
}

Array StreamPeer::_put_partial_data(const DVector<uint8_t>& p_data) {

	Array ret;

	int len = p_data.size();
	if (len==0) {
		ret.push_back(OK);
		ret.push_back(0);
		return ret;
	}

	DVector<uint8_t>::Read r = p_data.read();
	int sent;
	Error err = put_partial_data(&r[0],len,sent);

	if (err!=OK) {
		sent=0;
	}
	ret.push_back(err);
	ret.push_back(sent);
	return ret;
}


Array StreamPeer::_get_data(int p_bytes) {

	Array ret;

	DVector<uint8_t> data;
	data.resize(p_bytes);
	if (data.size()!=p_bytes) {

		ret.push_back(ERR_OUT_OF_MEMORY);
		ret.push_back(DVector<uint8_t>());
		return ret;
	}

	DVector<uint8_t>::Write w = data.write();
	Error err = get_data(&w[0],p_bytes);
	w = DVector<uint8_t>::Write();
	ret.push_back(err);
	ret.push_back(data);
	return ret;

}

Array StreamPeer::_get_partial_data(int p_bytes) {

	Array ret;

	DVector<uint8_t> data;
	data.resize(p_bytes);
	if (data.size()!=p_bytes) {

		ret.push_back(ERR_OUT_OF_MEMORY);
		ret.push_back(DVector<uint8_t>());
		return ret;
	}

	DVector<uint8_t>::Write w = data.write();
	int received;
	Error err = get_partial_data(&w[0],p_bytes,received);
	w = DVector<uint8_t>::Write();

	if (err!=OK) {
		data.resize(0);
	} else 	if (received!=data.size()) {

		data.resize(received);
	}

	ret.push_back(err);
	ret.push_back(data);
	return ret;

}


void StreamPeer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("put_data","data"),&StreamPeer::_put_data);
	ObjectTypeDB::bind_method(_MD("put_partial_data","data"),&StreamPeer::_put_partial_data);

	ObjectTypeDB::bind_method(_MD("get_data","bytes"),&StreamPeer::_get_data);
	ObjectTypeDB::bind_method(_MD("get_partial_data","bytes"),&StreamPeer::_get_partial_data);
}
