/**************************************************************************/
/*  camera_win.cpp                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "camera_external_feed.h"

//////////////////////////////////////////////////////////////////////////
// CameraAndroidFeed - Subclass for our camera feed on windows

/// @TODO need to implement this

class CameraAndroidFeed : public CameraFeed {
private:
protected:
public:
	CameraAndroidFeed();
	virtual ~CameraAndroidFeed();

	bool activate_feed();
	void deactivate_feed();
};

CameraAndroidFeed::CameraAndroidFeed() {
	///@TODO implement this, should store information about our available camera
}

CameraAndroidFeed::~CameraAndroidFeed() {
	// make sure we stop recording if we are!
	if (is_active()) {
		deactivate_feed();
	};

	///@TODO free up anything used by this
};

bool CameraAndroidFeed::activate_feed() {
	///@TODO this should activate our camera and start the process of capturing frames

	return true;
};

///@TODO we should probably have a callback method here that is being called by the
// camera API which provides frames and call back into the CameraServer to update our texture

void CameraAndroidFeed::deactivate_feed() {
	///@TODO this should deactivate our camera and stop the process of capturing frames
}

//////////////////////////////////////////////////////////////////////////
// CameraAndroid - Subclass for our camera server on windows

void CameraAndroid::add_active_cameras() {
	///@TODO scan through any active cameras and create CameraAndroidFeed objects for them
}

CameraAndroid::CameraAndroid() {
	// Find cameras active right now
	add_active_cameras();

	// need to add something that will react to devices being connected/removed...
};
