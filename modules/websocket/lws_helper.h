/*************************************************************************/
/*  lws_helper.h                                                         */
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
#ifndef LWS_HELPER_H
#define LWS_HELPER_H

#define LWS_BUF_SIZE 65536
#define LWS_PACKET_SIZE LWS_BUF_SIZE

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

_LWSRef *_lws_create_ref(void *obj);
void _lws_free_ref(_LWSRef *ref);
bool _lws_destroy(struct lws_context *context, _LWSRef *ref);
bool _lws_poll(struct lws_context *context, _LWSRef *ref);
void _lws_make_protocols(void *p_obj, lws_callback_function *p_callback, PoolVector<String> p_names, _LWSRef **r_lws_ref);

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
