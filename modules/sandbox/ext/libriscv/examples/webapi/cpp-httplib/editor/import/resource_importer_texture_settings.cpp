/**************************************************************************/
/*  resource_importer_texture_settings.cpp                                */
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

#include "resource_importer_texture_settings.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"

// ResourceImporterTextureSettings contains code used by
// multiple texture importers and the export dialog.
bool ResourceImporterTextureSettings::should_import_s3tc_bptc() {
	if (GLOBAL_GET("rendering/textures/vram_compression/import_s3tc_bptc")) {
		return true;
	}
	// If the project settings override is not enabled, import
	// S3TC/BPTC only when the host operating system needs it.
	return OS::get_singleton()->get_preferred_texture_format() == OS::PREFERRED_TEXTURE_FORMAT_S3TC_BPTC;
}

bool ResourceImporterTextureSettings::should_import_etc2_astc() {
	if (GLOBAL_GET("rendering/textures/vram_compression/import_etc2_astc")) {
		return true;
	}
	// If the project settings override is not enabled, import
	// ETC2/ASTC only when the host operating system needs it.
	return OS::get_singleton()->get_preferred_texture_format() == OS::PREFERRED_TEXTURE_FORMAT_ETC2_ASTC;
}
