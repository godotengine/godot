/**************************************************************************/
/*  caption_button_overlay.h                                              */
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

#include "scene/gui/control.h"

// Engine-drawn close/minimize/maximize button cluster rendered over the
// title-bar area when WINDOW_FLAG_EXTEND_TO_TITLE is active on platforms that
// do not provide native caption buttons (e.g. Windows).
class CaptionButtonOverlay : public Control {
	GDCLASS(CaptionButtonOverlay, Control);

public:
	enum ButtonZone {
		ZONE_NONE = -1,
		ZONE_MINIMIZE = 0,
		ZONE_MAXIMIZE = 1,
		ZONE_CLOSE = 2,
	};

private:
	// DPI scale factor cached from the last NOTIFICATION_WM_DPI_CHANGE (1.0 = 96 DPI).
	// Icon geometry is authored at 96 DPI (10 × 10 px) and multiplied by this value.
	float _dpi_scale = 1.0f;

	bool window_focused = true;
	bool window_maximized = false;
	bool minimize_disabled = false;
	bool maximize_disabled = false;

	ButtonZone hover_zone = ZONE_NONE;
	bool pressing = false;

	// Returns the local-space rect for a given zone, taking RTL into account.
	Rect2 _zone_rect(ButtonZone p_zone) const;
	ButtonZone _zone_at(const Vector2 &p_pos) const;

	void _draw_icon_minimize(const Rect2 &p_rect, const Color &p_color);
	void _draw_icon_maximize(const Rect2 &p_rect, const Color &p_color);
	void _draw_icon_restore(const Rect2 &p_rect, const Color &p_color);
	void _draw_icon_close(const Rect2 &p_rect, const Color &p_color);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

public:
	void set_window_focused(bool p_focused);
	void set_window_maximized(bool p_maximized);
	bool get_window_maximized() const { return window_maximized; }

	void set_minimize_disabled(bool p_disabled);
	void set_maximize_disabled(bool p_disabled);

	CaptionButtonOverlay();
};
