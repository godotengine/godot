/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "export.h"
#include "editor/editor_export.h"
#include "platform/x11/logo.h"
#include "scene/resources/texture.h"

void register_x11_exporter() {

	Ref<EditorExportPlatformPC> platform;
	platform.instance();

	Image img(_x11_logo);
	Ref<ImageTexture> logo;
	logo.instance();
	logo->create_from_image(img);
	platform->set_logo(logo);
	platform->set_name("Linux/X11");
	platform->set_extension("bin");
	platform->set_release_32("linux_x11_32_release");
	platform->set_debug_32("linux_x11_32_debug");
	platform->set_release_64("linux_x11_64_release");
	platform->set_debug_64("linux_x11_64_debug");

	EditorExport::get_singleton()->add_export_platform(platform);

#if 0
	Image img(_x11_logo);
	Ref<ImageTexture> logo = memnew( ImageTexture );
	logo->create_from_image(img);

	{
		Ref<EditorExportPlatformPC> exporter = Ref<EditorExportPlatformPC>( memnew(EditorExportPlatformPC) );
		exporter->set_binary_extension("");
		exporter->set_release_binary32("linux_x11_32_release");
		exporter->set_debug_binary32("linux_x11_32_debug");
		exporter->set_release_binary64("linux_x11_64_release");
		exporter->set_debug_binary64("linux_x11_64_debug");
		exporter->set_name("Linux X11");
		exporter->set_logo(logo);
		EditorImportExport::get_singleton()->add_export_platform(exporter);
	}

#endif
}
