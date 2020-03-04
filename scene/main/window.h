/*************************************************************************/
/*  window.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef WINDOW_H
#define WINDOW_H

#include "scene/main/viewport.h"
#include "servers/display_server.h"

class Window : public Viewport {
	GDCLASS(Window, Viewport)
public:
	enum Mode {
		MODE_WINDOWED = DisplayServer::WINDOW_MODE_WINDOWED,
		MODE_MINIMIZED = DisplayServer::WINDOW_MODE_MINIMIZED,
		MODE_MAXIMIZED = DisplayServer::WINDOW_MODE_MAXIMIZED,
		MODE_FULLSCREEN = DisplayServer::WINDOW_MODE_FULLSCREEN
	};

	enum Flags {
		FLAG_RESIZE_DISABLED = DisplayServer::WINDOW_FLAG_RESIZE_DISABLED,
		FLAG_BORDERLESS = DisplayServer::WINDOW_FLAG_BORDERLESS,
		FLAG_ALWAYS_ON_TOP = DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP,
		FLAG_TRANSPARENT = DisplayServer::WINDOW_FLAG_TRANSPARENT,
		FLAG_MAX = DisplayServer::WINDOW_FLAG_MAX,
	};

	enum ContentScaleMode {
		CONTENT_SCALE_MODE_DISABLED,
		CONTENT_SCALE_MODE_OBJECTS,
		CONTENT_SCALE_MODE_PIXELS,

	};

	enum ContentScaleAspect {
		CONTENT_SCALE_ASPECT_IGNORE,
		CONTENT_SCALE_ASPECT_KEEP,
		CONTENT_SCALE_ASPECT_KEEP_WIDTH,
		CONTENT_SCALE_ASPECT_KEEP_HEIGHT,
		CONTENT_SCALE_ASPECT_EXPAND,

	};
	enum {
		DEFAULT_WINDOW_SIZE = 100
	};

private:
	DisplayServer::WindowID window_id = DisplayServer::INVALID_WINDOW_ID;

	String title;
	mutable int current_screen = 0;
	mutable Vector2i position;
	mutable Size2i size = Size2i(DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE);
	mutable Size2i min_size;
	mutable Size2i max_size;
	mutable Mode mode = MODE_WINDOWED;
	mutable bool flags[FLAG_MAX];
	bool visible = true;

	bool use_font_oversampling = false;

	Size2i content_scale_size;
	ContentScaleMode content_scale_mode;
	ContentScaleAspect content_scale_aspect;

	void _make_window();
	void _clear_window();
	void _update_from_window();

	void _resize_callback(const Size2i &p_callback);

	void _update_size();

	virtual DisplayServer::WindowID get_window_id() const;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_title(const String &p_title);
	String get_title() const;

	void set_current_screen(int p_screen);
	int get_current_screen() const;

	void set_position(const Point2i &p_position);
	Point2i get_position() const;

	void set_size(const Size2i &p_size);
	Size2i get_size() const;

	Size2i get_real_size() const;

	void set_max_size(const Size2i &p_max_size);
	Size2i get_max_size() const;

	void set_min_size(const Size2i &p_min_size);
	Size2i get_min_size() const;

	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_flag(Flags p_flag, bool p_enabled);
	bool get_flag(Flags p_flag) const;

	bool is_maximize_allowed() const;

	void request_attention();
	void move_to_foreground();

	void set_visible(bool p_visible);
	bool is_visible() const;

	bool can_draw() const;

	void set_ime_active(bool p_active);
	void set_ime_position(const Point2i &p_pos);

	bool is_embedded() const;

	void set_content_scale_size(const Size2i &p_size);
	Size2i get_content_scale_size() const;

	void set_content_scale_mode(const ContentScaleMode &p_mode);
	ContentScaleMode get_content_scale_mode() const;

	void set_content_scale_aspect(const ContentScaleAspect &p_aspect);
	ContentScaleAspect get_content_scale_aspect() const;

	void set_use_font_oversampling(bool p_oversampling);
	bool is_using_font_oversampling() const;
	Window();
	~Window();
};

VARIANT_ENUM_CAST(Window::Window::Mode);
VARIANT_ENUM_CAST(Window::Window::Flags);
VARIANT_ENUM_CAST(Window::ContentScaleMode);
VARIANT_ENUM_CAST(Window::ContentScaleAspect);

#endif // WINDOW_H
