/*************************************************************************/
/*  editor_property_name_processor.h                                     */
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

#ifndef EDITOR_PROPERTY_NAME_PROCESSOR_H
#define EDITOR_PROPERTY_NAME_PROCESSOR_H

#include "scene/main/node.h"

class EditorPropertyNameProcessor : public Node {
	GDCLASS(EditorPropertyNameProcessor, Node);

	static EditorPropertyNameProcessor *singleton;

	Map<String, String> capitalize_string_remaps;

	String _capitalize_name(const String &p_name) const;

public:
	static EditorPropertyNameProcessor *get_singleton() { return singleton; }

	// Capitalize & localize property path segments.
	String process_name(const String &p_name) const;

	// Make tooltip string for names processed by process_name().
	String make_tooltip_for_name(const String &p_name) const;

	EditorPropertyNameProcessor();
	~EditorPropertyNameProcessor();
};

#endif // EDITOR_PROPERTY_NAME_PROCESSOR_H
