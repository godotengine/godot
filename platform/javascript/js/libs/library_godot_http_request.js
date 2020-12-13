/*************************************************************************/
/*  http_request.js                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

const GodotHTTPRequest = {
	$GodotHTTPRequest__deps: ['$GodotRuntime'],
	$GodotHTTPRequest: {
		requests: [],

		getUnusedRequestId: function () {
			const idMax = GodotHTTPRequest.requests.length;
			for (let potentialId = 0; potentialId < idMax; ++potentialId) {
				if (GodotHTTPRequest.requests[potentialId] instanceof XMLHttpRequest) {
					continue;
				}
				return potentialId;
			}
			GodotHTTPRequest.requests.push(null);
			return idMax;
		},

		setupRequest: function (xhr) {
			xhr.responseType = 'arraybuffer';
		},
	},

	godot_xhr_new__sig: 'i',
	godot_xhr_new: function () {
		const newId = GodotHTTPRequest.getUnusedRequestId();
		GodotHTTPRequest.requests[newId] = new XMLHttpRequest();
		GodotHTTPRequest.setupRequest(GodotHTTPRequest.requests[newId]);
		return newId;
	},

	godot_xhr_reset__sig: 'vi',
	godot_xhr_reset: function (xhrId) {
		GodotHTTPRequest.requests[xhrId] = new XMLHttpRequest();
		GodotHTTPRequest.setupRequest(GodotHTTPRequest.requests[xhrId]);
	},

	godot_xhr_free__sig: 'vi',
	godot_xhr_free: function (xhrId) {
		GodotHTTPRequest.requests[xhrId].abort();
		GodotHTTPRequest.requests[xhrId] = null;
	},

	godot_xhr_open__sig: 'viiiii',
	godot_xhr_open: function (xhrId, method, url, p_user, p_password) {
		const user = p_user > 0 ? GodotRuntime.parseString(p_user) : null;
		const password = p_password > 0 ? GodotRuntime.parseString(p_password) : null;
		GodotHTTPRequest.requests[xhrId].open(GodotRuntime.parseString(method), GodotRuntime.parseString(url), true, user, password);
	},

	godot_xhr_set_request_header__sig: 'viii',
	godot_xhr_set_request_header: function (xhrId, header, value) {
		GodotHTTPRequest.requests[xhrId].setRequestHeader(GodotRuntime.parseString(header), GodotRuntime.parseString(value));
	},

	godot_xhr_send_null__sig: 'vi',
	godot_xhr_send_null: function (xhrId) {
		GodotHTTPRequest.requests[xhrId].send();
	},

	godot_xhr_send_string__sig: 'vii',
	godot_xhr_send_string: function (xhrId, strPtr) {
		if (!strPtr) {
			GodotRuntime.error('Failed to send string per XHR: null pointer');
			return;
		}
		GodotHTTPRequest.requests[xhrId].send(GodotRuntime.parseString(strPtr));
	},

	godot_xhr_send_data__sig: 'viii',
	godot_xhr_send_data: function (xhrId, ptr, len) {
		if (!ptr) {
			GodotRuntime.error('Failed to send data per XHR: null pointer');
			return;
		}
		if (len < 0) {
			GodotRuntime.error('Failed to send data per XHR: buffer length less than 0');
			return;
		}
		GodotHTTPRequest.requests[xhrId].send(HEAPU8.subarray(ptr, ptr + len));
	},

	godot_xhr_abort__sig: 'vi',
	godot_xhr_abort: function (xhrId) {
		GodotHTTPRequest.requests[xhrId].abort();
	},

	godot_xhr_get_status__sig: 'ii',
	godot_xhr_get_status: function (xhrId) {
		return GodotHTTPRequest.requests[xhrId].status;
	},

	godot_xhr_get_ready_state__sig: 'ii',
	godot_xhr_get_ready_state: function (xhrId) {
		return GodotHTTPRequest.requests[xhrId].readyState;
	},

	godot_xhr_get_response_headers_length__sig: 'ii',
	godot_xhr_get_response_headers_length: function (xhrId) {
		const headers = GodotHTTPRequest.requests[xhrId].getAllResponseHeaders();
		return headers === null ? 0 : GodotRuntime.strlen(headers);
	},

	godot_xhr_get_response_headers__sig: 'viii',
	godot_xhr_get_response_headers: function (xhrId, dst, len) {
		const str = GodotHTTPRequest.requests[xhrId].getAllResponseHeaders();
		if (str === null) {
			return;
		}
		GodotRuntime.stringToHeap(str, dst, len);
	},

	godot_xhr_get_response_length__sig: 'ii',
	godot_xhr_get_response_length: function (xhrId) {
		const body = GodotHTTPRequest.requests[xhrId].response;
		return body === null ? 0 : body.byteLength;
	},

	godot_xhr_get_response__sig: 'viii',
	godot_xhr_get_response: function (xhrId, dst, len) {
		let buf = GodotHTTPRequest.requests[xhrId].response;
		if (buf === null) {
			return;
		}
		buf = new Uint8Array(buf).subarray(0, len);
		HEAPU8.set(buf, dst);
	},
};

autoAddDeps(GodotHTTPRequest, '$GodotHTTPRequest');
mergeInto(LibraryManager.library, GodotHTTPRequest);
