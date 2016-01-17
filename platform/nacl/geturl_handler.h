/*************************************************************************/
/*  geturl_handler.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef EXAMPLES_GETURL_GETURL_HANDLER_H_
#define EXAMPLES_GETURL_GETURL_HANDLER_H_

#include "core/ustring.h"
#include "core/vector.h"
#include "ppapi/cpp/completion_callback.h"
#include "ppapi/cpp/url_loader.h"
#include "ppapi/cpp/url_request_info.h"
#include "ppapi/cpp/instance.h"
#include "ppapi/utility/completion_callback_factory.h"

#define READ_BUFFER_SIZE 32768

// GetURLHandler is used to download data from |url|. When download is
// finished or when an error occurs, it posts a message back to the browser
// with the results encoded in the message as a string and self-destroys.
//
// EXAMPLE USAGE:
// GetURLHandler* handler* = GetURLHandler::Create(instance,url);
// handler->Start();
//
class GetURLHandler {

public:

	enum Status {

		STATUS_NONE,
		STATUS_IN_PROGRESS,
		STATUS_COMPLETED,
		STATUS_ERROR,
	};

private:

	Status status;

	// Callback fo the pp::URLLoader::Open().
	// Called by pp::URLLoader when response headers are received or when an
	// error occurs (in response to the call of pp::URLLoader::Open()).
	// Look at <ppapi/c/ppb_url_loader.h> and
	// <ppapi/cpp/url_loader.h> for more information about pp::URLLoader.
	void OnOpen(int32_t result);

	// Callback fo the pp::URLLoader::ReadResponseBody().
	// |result| contains the number of bytes read or an error code.
	// Appends data from this->buffer_ to this->url_response_body_.
	void OnRead(int32_t result);

	// Reads the response body (asynchronously) into this->buffer_.
	// OnRead() will be called when bytes are received or when an error occurs.
	void ReadBody();

	// Append data bytes read from the URL onto the internal buffer.  Does
	// nothing if |num_bytes| is 0.
	void AppendDataBytes(const char* buffer, int32_t num_bytes);

	pp::Instance* instance_;  // Weak pointer.
	String url_;  // URL to be downloaded.
	pp::URLRequestInfo url_request_;
	pp::URLLoader url_loader_;  // URLLoader provides an API to download URLs.
	char buffer_[READ_BUFFER_SIZE];  // Temporary buffer for reads.
	Vector<uint8_t> data;  // Contains accumulated downloaded data.
	pp::CompletionCallbackFactory<GetURLHandler> cc_factory_;
	bool complete;

	GetURLHandler(const GetURLHandler&);
	void operator=(const GetURLHandler&);

public:
	// Creates instance of GetURLHandler on the heap.
	// GetURLHandler objects shall be created only on the heap (they
	// self-destroy when all data is in).
	// Initiates page (URL) download.
	void Start();

	Status get_status() const;
	Vector<uint8_t> get_data() const;

	int get_bytes_read() const;

	GetURLHandler(pp::Instance* instance_, const String& url);
	~GetURLHandler();
};

#endif  // EXAMPLES_GETURL_GETURL_HANDLER_H_

