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

void register_uwp_exporter() {
#ifdef WINDOWS_ENABLED
	EDITOR_DEF("export/uwp/signtool", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/uwp/signtool", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
	EDITOR_DEF("export/uwp/debug_certificate", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/uwp/debug_certificate", PROPERTY_HINT_GLOBAL_FILE, "*.pfx"));
	EDITOR_DEF("export/uwp/debug_password", "");
	EDITOR_DEF("export/uwp/debug_algorithm", 2); // SHA256 is the default
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "export/uwp/debug_algorithm", PROPERTY_HINT_ENUM, "MD5,SHA1,SHA256"));
#endif // WINDOWS_ENABLED

	Ref<EditorExportPlatformUWP> exporter;
	exporter.instantiate();
	EditorExport::get_singleton()->add_export_platform(exporter);
}
