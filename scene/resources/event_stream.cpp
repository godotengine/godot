/*************************************************************************/
/*  event_stream.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "event_stream.h"


Error EventStreamPlayback::play() {
	if (stream.is_valid())
		stop();

	Error err = _play();
	if (err)
		return err;


	playing=true;
	AudioServer::get_singleton()->stream_set_active(stream,true);

	return OK;
}

void EventStreamPlayback::stop(){

	if (!playing)
		return;

	AudioServer::get_singleton()->stream_set_active(stream,false);
	_stop();
	playing=false;


}
bool EventStreamPlayback::is_playing() const{

	return playing;
}


EventStreamPlayback::EventStreamPlayback() {

	playing=false;
	estream.playback=this;
	stream=AudioServer::get_singleton()->event_stream_create(&estream);

}

EventStreamPlayback::~EventStreamPlayback() {

	AudioServer::get_singleton()->free(stream);

}



EventStream::EventStream()
{


}

