/**************************************************************************/
/*  openxr_htc_controller_extension.h                                     */
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

#ifndef OPENXR_HTC_CONTROLLER_EXTENSION_H
#define OPENXR_HTC_CONTROLLER_EXTENSION_H

#include "openxr_extension_wrapper.h"

class OpenXRHTCControllerExtension : public OpenXRExtensionWrapper {
public:
	enum HTCControllers {
		// Note, HTC Vive Wand controllers are part of the core spec and not part of our extension.
		HTC_VIVE_COSMOS,
		HTC_VIVE_FOCUS3,
		HTC_HAND_INTERACTION,
		HTC_MAX_CONTROLLERS
	};

	virtual HashMap<String, bool *> get_requested_extensions() override;

	PackedStringArray get_suggested_tracker_names() override;

	bool is_available(HTCControllers p_type);

	virtual void on_register_metadata() override;

private:
	bool available[HTC_MAX_CONTROLLERS] = { false, false };
};

#endif // OPENXR_HTC_CONTROLLER_EXTENSION_H
