/*************************************************************************/
/*  export.cpp                                                           */
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

#include "export.h"

#include "export_plugin.h"

void register_javascript_exporter() {
	EDITOR_DEF("export/web/http_host", "localhost");
	EDITOR_DEF("export/web/http_port", 8060);
	EDITOR_DEF("export/web/use_ssl", false);
	EDITOR_DEF("export/web/ssl_key", "");
	EDITOR_DEF("export/web/ssl_certificate", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "export/web/http_port", PROPERTY_HINT_RANGE, "1,65535,1"));
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/web/ssl_key", PROPERTY_HINT_GLOBAL_FILE, "*.key"));
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/web/ssl_certificate", PROPERTY_HINT_GLOBAL_FILE, "*.crt,*.pem"));

	Ref<EditorExportPlatformJavaScript> platform;
	platform.instantiate();
	EditorExport::get_singleton()->add_export_platform(platform);
}
