/*************************************************************************/
/*  output_strings.h                                                     */
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
#ifndef OUTPUT_STRINGS_H
#define OUTPUT_STRINGS_H

#include "map.h"
#include "scene/gui/control.h"
#include "scene/gui/scroll_bar.h"

class OutputStrings : public Control {

	GDCLASS(OutputStrings, Control);

public:
	enum LineType {

		LINE_NORMAL,
		LINE_WARNING,
		LINE_ERROR,
		LINE_LINK
	};

private:
	struct Line {

		LineType type;
		Variant meta;
		String text;
	};

	int font_height;
	int size_height;

	Size2 margin;
	typedef Map<int, Line> LineMap;
	Map<int, Line> line_map;

	VScrollBar *v_scroll;
	HScrollBar *h_scroll;

	bool following;
	int line_max_count;
	bool updating;

	void _vscroll_changed(float p_value);
	void _hscroll_changed(float p_value);
	void update_scrollbars();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void add_line(const String &p_text, const Variant &p_meta = Variant(), const LineType p_type = LINE_NORMAL);

	OutputStrings();
};

#endif // OUTPUT_STRINGS_H
