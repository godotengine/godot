/*************************************************************************/
/*  openxr_htc_vive_tracker_extension.h                                  */
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

#ifndef OPENXR_HTC_VIVE_TRACKER_EXTENSION_H
#define OPENXR_HTC_VIVE_TRACKER_EXTENSION_H

#include "openxr_extension_wrapper.h"

class OpenXRHTCViveTrackerExtension : public OpenXRExtensionWrapper {
public:
	static OpenXRHTCViveTrackerExtension *get_singleton();

	OpenXRHTCViveTrackerExtension(OpenXRAPI *p_openxr_api);
	virtual ~OpenXRHTCViveTrackerExtension() override;

	bool is_available();
	virtual bool on_event_polled(const XrEventDataBuffer &event) override;

private:
	static OpenXRHTCViveTrackerExtension *singleton;

	bool available = false;
};

#endif // !OPENXR_HTC_VIVE_TRACKER_EXTENSION_H
