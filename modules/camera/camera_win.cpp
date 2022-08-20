/*************************************************************************/
/*  camera_win.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "camera_win.h"

///@TODO sorry guys, I got about 80% through implementing this using DirectShow only
// to find out Microsoft deprecated half the API and its replacement is as confusing
// as they could make it. Joey suggested looking into libuvc which offers a more direct
// route to webcams over USB and this is very promising but it wouldn't compile on
// windows for me...I've gutted the classes I implemented DirectShow in just to have
// a skeleton for someone to work on, mail me for more details or if you want a copy....

//////////////////////////////////////////////////////////////////////////
// CameraFeedWindows - Subclass for our camera feed on windows

/// @TODO need to implement this

class CameraFeedWindows : public CameraFeed {
private:
protected:
public:
	CameraFeedWindows();
	virtual ~CameraFeedWindows();

	bool activate_feed();
	void deactivate_feed();
};

CameraFeedWindows::CameraFeedWindows() {
	///@TODO implement this, should store information about our available camera
}

CameraFeedWindows::~CameraFeedWindows() {
	// make sure we stop recording if we are!
	if (is_active()) {
		deactivate_feed();
	};

	///@TODO free up anything used by this
};

bool CameraFeedWindows::activate_feed() {
	///@TODO this should activate our camera and start the process of capturing frames

	return true;
};

///@TODO we should probably have a callback method here that is being called by the
// camera API which provides frames and call back into the CameraServer to update our texture

void CameraFeedWindows::deactivate_feed() {
	///@TODO this should deactivate our camera and stop the process of capturing frames
}

//////////////////////////////////////////////////////////////////////////
// CameraWindows - Subclass for our camera server on windows

void CameraWindows::add_active_cameras() {
	///@TODO scan through any active cameras and create CameraFeedWindows objects for them
}

CameraWindows::CameraWindows() {
	// Find cameras active right now
	add_active_cameras();

	// need to add something that will react to devices being connected/removed...
};
