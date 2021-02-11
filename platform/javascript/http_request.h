/*************************************************************************/
/*  http_request.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef HTTP_REQUEST_H
#define HTTP_REQUEST_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"

typedef enum {
	XHR_READY_STATE_UNSENT = 0,
	XHR_READY_STATE_OPENED = 1,
	XHR_READY_STATE_HEADERS_RECEIVED = 2,
	XHR_READY_STATE_LOADING = 3,
	XHR_READY_STATE_DONE = 4,
} godot_xhr_ready_state_t;

extern int godot_xhr_new();
extern void godot_xhr_reset(int p_xhr_id);
extern void godot_xhr_free(int p_xhr_id);

extern int godot_xhr_open(int p_xhr_id, const char *p_method, const char *p_url, const char *p_user = NULL, const char *p_password = NULL);

extern void godot_xhr_set_request_header(int p_xhr_id, const char *p_header, const char *p_value);

extern void godot_xhr_send(int p_xhr_id, const void *p_data, int p_len);
extern void godot_xhr_abort(int p_xhr_id);

/* this is an HTTPClient::ResponseCode, not ::Status */
extern int godot_xhr_get_status(int p_xhr_id);
extern godot_xhr_ready_state_t godot_xhr_get_ready_state(int p_xhr_id);

extern int godot_xhr_get_response_headers_length(int p_xhr_id);
extern void godot_xhr_get_response_headers(int p_xhr_id, char *r_dst, int p_len);

extern int godot_xhr_get_response_length(int p_xhr_id);
extern void godot_xhr_get_response(int p_xhr_id, void *r_dst, int p_len);

#ifdef __cplusplus
}
#endif

#endif /* HTTP_REQUEST_H */
