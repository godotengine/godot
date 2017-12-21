/*************************************************************************/
/*  lws_helper.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef LWS_HELPER_H
#define LWS_HELPER_H

#include "core/io/stream_peer.h"
#include "core/os/os.h"
#include "core/reference.h"
#include "core/ring_buffer.h"
#include "lws_peer.h"

struct _LWSRef {
	bool free_context;
	bool is_polling;
	bool is_valid;
	bool is_destroying;
	void *obj;
	struct lws_protocols *lws_structs;
	char *lws_names;
};

static _LWSRef *_lws_create_ref(void *obj) {

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

static void _lws_free_ref(_LWSRef *ref) {
	// Free strings and structs
	memfree(ref->lws_structs);
	memfree(ref->lws_names);
	// Free ref
	memfree(ref);
}

static bool _lws_destroy(struct lws_context *context, _LWSRef *ref) {
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

static bool _lws_poll(struct lws_context *context, _LWSRef *ref) {

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
 * prepare the protocol_structs to be fed to context
 * also prepare the protocol string used by the client
 */
static void _lws_make_protocols(void *p_obj, lws_callback_function *p_callback, PoolVector<String> p_names, _LWSRef **r_lws_ref) {
	/* the input strings might go away after this call,
	 * we need to copy them. Will clear them when
	 * detroying the context */
	int i;
	int len = p_names.size();
	size_t data_size = sizeof(struct LWSPeer::PeerData);
	PoolVector<String>::Read pnr = p_names.read();

	/*
	 * This is a reference connecting the object with lws
	 * keep track of status, mallocs, etc.
	 * Must survive as long the context
	 * Must be freed manually when context creation fails.
	 */
	_LWSRef *ref = _lws_create_ref(p_obj);

	/* LWS protocol structs */
	ref->lws_structs = (struct lws_protocols *)memalloc(sizeof(struct lws_protocols) * (len + 2));

	CharString strings = p_names.join(",").ascii();
	int str_len = strings.length();

	/* Joined string of protocols, double the size: comma separated first, NULL separated last */
	ref->lws_names = (char *)memalloc((str_len + 1) * 2); /* plus the terminator */

	char *names_ptr = ref->lws_names;
	struct lws_protocols *structs_ptr = ref->lws_structs;

	copymem(names_ptr, strings.get_data(), str_len);
	names_ptr[str_len] = '\0'; /* NULL terminator */
	/* NULL terminated strings to be used in protocol structs */
	copymem(&names_ptr[str_len + 1], strings.get_data(), str_len);
	names_ptr[(str_len * 2) + 1] = '\0'; /* NULL terminator */
	int pos = str_len + 1;

	/* the first protocol is always http-only */
	structs_ptr[0].name = "http-only";
	structs_ptr[0].callback = p_callback;
	structs_ptr[0].per_session_data_size = data_size;
	structs_ptr[0].rx_buffer_size = 0;
	/* add user defined protocols */
	for (i = 0; i < len; i++) {
		structs_ptr[i + 1].name = (const char *)&names_ptr[pos];
		structs_ptr[i + 1].callback = p_callback;
		structs_ptr[i + 1].per_session_data_size = data_size;
		structs_ptr[i + 1].rx_buffer_size = 0;
		pos += pnr[i].ascii().length() + 1;
		names_ptr[pos - 1] = '\0';
	}
	/* add protocols terminator */
	structs_ptr[len + 1].name = NULL;
	structs_ptr[len + 1].callback = NULL;
	structs_ptr[len + 1].per_session_data_size = 0;
	structs_ptr[len + 1].rx_buffer_size = 0;

	*r_lws_ref = ref;
}

/* clang-format off */
#define LWS_HELPER(CNAME) \
protected:															\
	struct _LWSRef *_lws_ref;												\
	struct lws_context *context;												\
																\
	static int _lws_gd_callback(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len) {	\
																\
		if (wsi == NULL) {												\
			return 0;												\
		}														\
																\
		struct _LWSRef *ref = (struct _LWSRef *)lws_context_user(lws_get_context(wsi));					\
		if (!ref->is_valid)												\
			return 0;												\
		CNAME *helper = (CNAME *)ref->obj;										\
		return helper->_handle_cb(wsi, reason, user, in, len);								\
	}															\
																\
	void invalidate_lws_ref() {												\
		if (_lws_ref != NULL)												\
			_lws_ref->is_valid = false;										\
	}															\
																\
	void destroy_context() {												\
		if (_lws_destroy(context, _lws_ref)) {										\
			context = NULL;												\
			_lws_ref = NULL;											\
		}														\
	}															\
																\
public:																\
	virtual int _handle_cb(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len);		\
																\
	void _lws_poll() {													\
		ERR_FAIL_COND(context == NULL);											\
																\
		if (::_lws_poll(context, _lws_ref)) {										\
			context = NULL;												\
			_lws_ref = NULL;											\
		}														\
	}															\
																\
protected:

	/* clang-format on */

#endif // LWS_HELPER_H
