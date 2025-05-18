/**************************************************************************/
/*  openxr_overlay_extension.h                                            */
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

#ifndef OPENXR_OVERLAY_EXTENSION_H
#define OPENXR_OVERLAY_EXTENSION_H

#include "openxr_extension_wrapper.h"
class OpenXROverlayExtension : public OpenXRExtensionWrapper {
public:
	static OpenXROverlayExtension *get_singleton();

	OpenXROverlayExtension();
	~OpenXROverlayExtension();

	virtual HashMap<String, bool *> get_requested_extensions() override;

	virtual void *set_session_create_and_get_next_pointer(void *p_next_pointer) override;

	bool is_available();
	bool is_enabled();

	uint32_t get_session_layers_placement() const;
	void set_session_layers_placement(uint32_t p_session_layers_placement);

private:
	static OpenXROverlayExtension *singleton;
	uint32_t session_layers_placement = 0;
	bool available = false;
	bool enabled = false;
};

#endif // OPENXR_OVERLAY_EXTENSION_H
