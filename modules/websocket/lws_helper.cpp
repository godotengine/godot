/*************************************************************************/
/*  lws_helper.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#if !defined(JAVASCRIPT_ENABLED)

#include "lws_helper.h"

_LWSRef *_lws_create_ref(void *obj) {

	_LWSRef *out = (_LWSRef *)memalloc(sizeof(_LWSRef));
	out->is_destroying = false;
	out->free_context = false;
	out->is_polling = false;
	out->obj = obj;
	out->is_valid = true;
	out->lws_structs = NULL;
	out->lws_names = NULL;
	return out;
}

void _lws_free_ref(_LWSRef *ref) {
	// Free strings and structs
	memfree(ref->lws_structs);
	memfree(ref->lws_names);
	// Free ref
	memfree(ref);
}

bool _lws_destroy(struct lws_context *context, _LWSRef *ref) {
	if (context == NULL || ref->is_destroying)
		return false;

	if (ref->is_polling) {
		ref->free_context = true;
		return false;
	}

	ref->is_destroying = true;
	lws_context_destroy(context);
	_lws_free_ref(ref);
	return true;
}

bool _lws_poll(struct lws_context *context, _LWSRef *ref) {

	ERR_FAIL_COND_V(context == NULL, false);
	ERR_FAIL_COND_V(ref == NULL, false);

	ref->is_polling = true;
	lws_service(context, 0);
	ref->is_polling = false;

	if (!ref->free_context)
		return false; // Nothing to do

	bool is_valid = ref->is_valid; // Might have been destroyed by poll

	_lws_destroy(context, ref); // Will destroy context and ref

	return is_valid; // If the object should NULL its context and ref
}

/*
 * Prepare the protocol_structs to be fed to context.
 * Also prepare the protocol string used by the client.
 */
void _lws_make_protocols(void *p_obj, lws_callback_function *p_callback, PoolVector<String> p_names, _LWSRef **r_lws_ref) {
	// The input strings might go away after this call, we need to copy them.
	// We will clear them when destroying the context.
	int i;
	int len = p_names.size();
	size_t data_size = sizeof(struct LWSPeer::PeerData);
	PoolVector<String>::Read pnr = p_names.read();

	// This is a reference connecting the object with lws keep track of status, mallocs, etc.
	// Must survive as long the context.
	// Must be freed manually when context creation fails.
	_LWSRef *ref = _lws_create_ref(p_obj);

	// LWS protocol structs.
	ref->lws_structs = (struct lws_protocols *)memalloc(sizeof(struct lws_protocols) * (len + 2));
	memset(ref->lws_structs, 0, sizeof(struct lws_protocols) * (len + 2));

	CharString strings = p_names.join(",").ascii();
	int str_len = strings.length();

	// Joined string of protocols, double the size: comma separated first, NULL separated last
	ref->lws_names = (char *)memalloc((str_len + 1) * 2); // Plus the terminator

	char *names_ptr = ref->lws_names;
	struct lws_protocols *structs_ptr = ref->lws_structs;

	// Comma separated protocols string to be used in client Sec-WebSocket-Protocol header
	if (str_len > 0)
		copymem(names_ptr, strings.get_data(), str_len);
	names_ptr[str_len] = '\0'; // NULL terminator

	// NULL terminated protocol strings to be used in protocol structs
	if (str_len > 0)
		copymem(&names_ptr[str_len + 1], strings.get_data(), str_len);
	names_ptr[(str_len * 2) + 1] = '\0'; // NULL terminator
	int pos = str_len + 1;

	// The first protocol is the default for any http request (before upgrade).
	// It is also used as the websocket protocol when no subprotocol is specified.
	structs_ptr[0].name = "default";
	structs_ptr[0].callback = p_callback;
	structs_ptr[0].per_session_data_size = data_size;
	structs_ptr[0].rx_buffer_size = LWS_BUF_SIZE;
	structs_ptr[0].tx_packet_size = LWS_PACKET_SIZE;
	// Add user defined protocols
	for (i = 0; i < len; i++) {
		structs_ptr[i + 1].name = (const char *)&names_ptr[pos];
		structs_ptr[i + 1].callback = p_callback;
		structs_ptr[i + 1].per_session_data_size = data_size;
		structs_ptr[i + 1].rx_buffer_size = LWS_BUF_SIZE;
		structs_ptr[i + 1].tx_packet_size = LWS_PACKET_SIZE;
		pos += pnr[i].ascii().length() + 1;
		names_ptr[pos - 1] = '\0';
	}
	// Add protocols terminator
	structs_ptr[len + 1].name = NULL;
	structs_ptr[len + 1].callback = NULL;
	structs_ptr[len + 1].per_session_data_size = 0;
	structs_ptr[len + 1].rx_buffer_size = 0;

	*r_lws_ref = ref;
}

#endif
