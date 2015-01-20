/*************************************************************************/
/*  platform.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifdef MODULE_SDK_ENABLED

#include "platform.h"

Platform *Platform::instance = NULL;


void Platform::_bind_methods() {

 	ObjectTypeDB::bind_method(_MD("request"),&Platform::request);
 	ObjectTypeDB::bind_method(_MD("get_pending_event_count"),&Platform::get_pending_event_count);
 	ObjectTypeDB::bind_method(_MD("pop_pending_event"),&Platform::pop_pending_event);
}

String Platform::request(Variant p_params) {

	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("type"), "invalid_param");

	String type = params["type"];

	return on_request(type, params);
}

String Platform::on_request(const String& p_type, const Dictionary& p_params) {

	return "ok";
}

void Platform::post_event(Variant p_event) {

 	pending_events.push_back(p_event);
}

int Platform::get_pending_event_count() {

	return pending_events.size();
}

Variant Platform::pop_pending_event() {

	Variant front = pending_events.front()->get();
	pending_events.pop_front();
	return front;
}

Platform *Platform::get_singleton() {

	return instance;
}

Platform::Platform() {

	instance = this;
}

Platform::~Platform() {

	instance = NULL;
}


#endif // MODULE_SDK_ENABLED
