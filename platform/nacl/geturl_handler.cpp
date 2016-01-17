/*************************************************************************/
/*  geturl_handler.cpp                                                   */
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
#include "geturl_handler.h"

#include "core/os/copymem.h"

#include <stdio.h>
#include <stdlib.h>
#include "ppapi/c/pp_errors.h"
#include "ppapi/c/ppb_instance.h"
#include "ppapi/cpp/module.h"
#include "ppapi/cpp/var.h"

void GetURLHandler::Start() {
  pp::CompletionCallback cc =
     cc_factory_.NewCallback(&GetURLHandler::OnOpen);
  url_loader_.Open(url_request_, cc);
}

void GetURLHandler::OnOpen(int32_t result) {
  if (result != PP_OK) {
	status = STATUS_ERROR;
    return;
  }
  // Here you would process the headers. A real program would want to at least
  // check the HTTP code and potentially cancel the request.
  // pp::URLResponseInfo response = loader_.GetResponseInfo();

  // Start streaming.
  ReadBody();
}

void GetURLHandler::AppendDataBytes(const char* buffer, int32_t num_bytes) {
  if (num_bytes <= 0)
    return;
  // Make sure we don't get a buffer overrun.
  num_bytes = std::min(READ_BUFFER_SIZE, num_bytes);
  int ofs = data.size();
  data.resize(ofs + num_bytes);
  copymem(&data[ofs], buffer, num_bytes);
}

void GetURLHandler::OnRead(int32_t result) {
  if (result == PP_OK) {
    // Streaming the file is complete.
	status = STATUS_COMPLETED;
	instance_->HandleMessage("package_finished");
	instance_->HandleMessage(0);
	printf("completed!\n");
  } else if (result > 0) {
    // The URLLoader just filled "result" number of bytes into our buffer.
    // Save them and perform another read.
    AppendDataBytes(buffer_, result);
    ReadBody();
  } else {
    // A read error occurred.
	status = STATUS_ERROR;
	ERR_FAIL_COND(result < 0);
  }
}

void GetURLHandler::ReadBody() {
  // Note that you specifically want an "optional" callback here. This will
  // allow ReadBody() to return synchronously, ignoring your completion
  // callback, if data is available. For fast connections and large files,
  // reading as fast as we can will make a large performance difference
  // However, in the case of a synchronous return, we need to be sure to run
  // the callback we created since the loader won't do anything with it.
  pp::CompletionCallback cc =
      cc_factory_.NewOptionalCallback(&GetURLHandler::OnRead);
  int32_t result = PP_OK;
  do {
    result = url_loader_.ReadResponseBody(buffer_, sizeof(buffer_), cc);
    // Handle streaming data directly. Note that we *don't* want to call
    // OnRead here, since in the case of result > 0 it will schedule
    // another call to this function. If the network is very fast, we could
    // end up with a deeply recursive stack.
    if (result > 0) {
      AppendDataBytes(buffer_, result);
    }
  } while (result > 0);

  if (result != PP_OK_COMPLETIONPENDING) {
    // Either we reached the end of the stream (result == PP_OK) or there was
    // an error. We want OnRead to get called no matter what to handle
    // that case, whether the error is synchronous or asynchronous. If the
    // result code *is* COMPLETIONPENDING, our callback will be called
    // asynchronously.
    cc.Run(result);
  }
}

GetURLHandler::Status GetURLHandler::get_status() const {

	return status;
};

Vector<uint8_t> GetURLHandler::get_data() const {

	return data;
};

int GetURLHandler::get_bytes_read() const {

	return data.size();
};

GetURLHandler::GetURLHandler(pp::Instance* instance,
							 const String& url)
	: instance_(instance),
	  url_(url),
	  url_request_(instance),
	  url_loader_(instance),
	  cc_factory_(this) {
  url_request_.SetURL(std::string(url.utf8().get_data()));
  url_request_.SetMethod("GET");
  status = STATUS_NONE;
  printf("url handler for url %ls!\n", url.c_str());
}

GetURLHandler::~GetURLHandler() {
}


