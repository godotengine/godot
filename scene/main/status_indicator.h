/**************************************************************************/
/*  status_indicator.h                                                    */
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

#ifndef STATUS_INDICATOR_H
#define STATUS_INDICATOR_H

#include "scene/main/node.h"
#include "servers/display_server.h"

class StatusIndicator : public Node {
	GDCLASS(StatusIndicator, Node);

	Ref<Image> icon;
	String tooltip;
	bool visible = true;
	DisplayServer::IndicatorID iid = DisplayServer::INVALID_INDICATOR_ID;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	void _callback(MouseButton p_index, const Point2i &p_pos);

public:
	void set_icon(const Ref<Image> &p_icon);
	Ref<Image> get_icon() const;

	void set_tooltip(const String &p_tooltip);
	String get_tooltip() const;

	void set_visible(bool p_visible);
	bool is_visible() const;
};

#endif // STATUS_INDICATOR_H
