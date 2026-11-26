/**************************************************************************/
/*  openxr_stationary_reference_space_extension.h                         */
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

#pragma once

#include "../util.h"
#include "openxr_extension_wrapper.h"

class OpenXRStationaryReferenceSpaceExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRStationaryReferenceSpaceExtension, OpenXRExtensionWrapper);

protected:
	static void _bind_methods();

public:
	static OpenXRStationaryReferenceSpaceExtension *get_singleton();

	OpenXRStationaryReferenceSpaceExtension();
	virtual ~OpenXRStationaryReferenceSpaceExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;

	bool is_available() const;

	String get_generation_id() const;

private:
	static OpenXRStationaryReferenceSpaceExtension *singleton;

	bool available = false;

	EXT_PROTO_XRRESULT_FUNC3(xrGetStationaryReferenceSpaceGenerationIdEXT, (XrSession), session, (const XrStationaryReferenceSpaceGenerationIdGetInfoEXT *), get_info, (XrStationaryReferenceSpaceGenerationIdResultEXT *), generation_id_result);
};
