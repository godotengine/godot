/**************************************************************************/
/*  gdextension_interface_header_generator.h                              */
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

#ifdef TOOLS_ENABLED

#include "core/io/file_access.h"

class GDExtensionInterfaceHeaderGenerator {
public:
	static void generate_gdextension_interface_header(const String &p_path);

private:
	static void write_doc(const Ref<FileAccess> &p_fa, const Array &p_doc, const String &p_indent = "");
	static void write_simple_type(const Ref<FileAccess> &p_fa, const Dictionary &p_type);
	static void write_enum_type(const Ref<FileAccess> &p_fa, const Dictionary &p_enum);
	static void write_function_type(const Ref<FileAccess> &p_fa, const Dictionary &p_func);
	static void write_struct_type(const Ref<FileAccess> &p_fa, const Dictionary &p_struct);

	static String format_type_and_name(const String &p_type, const String &p_name);
	static String make_deprecated_message(const Dictionary &p_data);
	static String make_deprecated_comment_for_type(const Dictionary &p_type);
	static String make_args_text(const Array &p_args);

	static void write_interface(const Ref<FileAccess> &p_fa, const Dictionary &p_interface);
};

#endif // TOOLS_ENABLED
