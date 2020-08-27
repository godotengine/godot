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
#include "scene/resources/theme.h"
#include "servers/display_server.h"

class Control;
class Window : public Viewport {
	GDCLASS(Window, Viewport)
public:
	enum Mode {
		MODE_WINDOWED = DisplayServer::WINDOW_MODE_WINDOWED,
		MODE_MINIMIZED = DisplayServer::WINDOW_MODE_MINIMIZED,
		MODE_MAXIMIZED = DisplayServer::WINDOW_MODE_MAXIMIZED,
		MODE_FULLSCREEN = DisplayServer::WINDOW_MODE_FULLSCREEN,
	};

	enum Flags {
		FLAG_RESIZE_DISABLED = DisplayServer::WINDOW_FLAG_RESIZE_DISABLED,
		FLAG_BORDERLESS = DisplayServer::WINDOW_FLAG_BORDERLESS,
		FLAG_ALWAYS_ON_TOP = DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP,
		FLAG_TRANSPARENT = DisplayServer::WINDOW_FLAG_TRANSPARENT,
		FLAG_NO_FOCUS = DisplayServer::WINDOW_FLAG_NO_FOCUS,
		FLAG_MAX = DisplayServer::WINDOW_FLAG_MAX,
	};

	enum ContentScaleMode {
		CONTENT_SCALE_MODE_DISABLED,
		CONTENT_SCALE_MODE_CANVAS_ITEMS,
		CONTENT_SCALE_MODE_VIEWPORT,
	};

	enum ContentScaleAspect {
		CONTENT_SCALE_ASPECT_IGNORE,
		CONTENT_SCALE_ASPECT_KEEP,
		CONTENT_SCALE_ASPECT_KEEP_WIDTH,
		CONTENT_SCALE_ASPECT_KEEP_HEIGHT,
		CONTENT_SCALE_ASPECT_EXPAND,
	};

	enum {
		DEFAULT_WINDOW_SIZE = 100,
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
	bool focused = false;

	bool use_font_oversampling = false;
	bool transient = false;
	bool exclusive = false;
	bool wrap_controls = false;
	bool updating_child_controls = false;
	bool clamp_to_embedder = false;

	void _update_child_controls();

	Size2i content_scale_size;
	ContentScaleMode content_scale_mode;
	ContentScaleAspect content_scale_aspect;

	void _make_window();
	void _clear_window();
	void _update_from_window();

	void _update_viewport_size();
	void _update_window_size();

	void _propagate_window_notification(Node *p_node, int p_notification);

	void _update_window_callbacks();

	void _clear_transient();
	void _make_transient();
	Window *transient_parent = nullptr;
	Window *exclusive_child = nullptr;
	Set<Window *> transient_children;

	friend class Control;
	Ref<Theme> theme;
	Control *theme_owner = nullptr;
	Window *theme_owner_window = nullptr;

	Viewport *embedder = nullptr;

	friend class Viewport; //friend back, can call the methods below

	void _window_input(const Ref<InputEvent> &p_ev);
	void _window_input_text(const String &p_text);
	void _window_drop_files(const Vector<String> &p_files);
	void _rect_changed_callback(const Rect2i &p_callback);
	void _event_callback(DisplayServer::WindowEvent p_event);
	virtual bool _can_consume_input_events() const override;

protected:
	Viewport *_get_embedder() const;
	virtual Rect2i _popup_adjust_rect() const { return Rect2i(); }

	virtual void _post_popup() {}
	virtual Size2 _get_contents_minimum_size() const;
	static void _bind_methods();
	void _notification(int p_what);

	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

public:
	enum {

		NOTIFICATION_VISIBILITY_CHANGED = 30,
		NOTIFICATION_POST_POPUP = 31,
		NOTIFICATION_THEME_CHANGED = 32,
	};

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

	void show();
	void hide();

	void set_transient(bool p_transient);
	bool is_transient() const;

	void set_exclusive(bool p_exclusive);
	bool is_exclusive() const;

	void set_clamp_to_embedder(bool p_enable);
	bool is_clamped_to_embedder() const;

	bool can_draw() const;

	void set_ime_active(bool p_active);
	void set_ime_position(const Point2i &p_pos);

	bool is_embedded() const;

	void set_content_scale_size(const Size2i &p_size);
	Size2i get_content_scale_size() const;

	void set_content_scale_mode(ContentScaleMode p_mode);
	ContentScaleMode get_content_scale_mode() const;

	void set_content_scale_aspect(ContentScaleAspect p_aspect);
	ContentScaleAspect get_content_scale_aspect() const;

	void set_use_font_oversampling(bool p_oversampling);
	bool is_using_font_oversampling() const;

	void set_wrap_controls(bool p_enable);
	bool is_wrapping_controls() const;
	void child_controls_changed();

	Window *get_parent_visible_window() const;
	Viewport *get_parent_viewport() const;
	void popup(const Rect2i &p_rect = Rect2i());
	void popup_on_parent(const Rect2i &p_parent_rect);
	void popup_centered_ratio(float p_ratio = 0.8);
	void popup_centered(const Size2i &p_minsize = Size2i());
	void popup_centered_clamped(const Size2i &p_size = Size2i(), float p_fallback_ratio = 0.75);

	void set_theme(const Ref<Theme> &p_theme);
	Ref<Theme> get_theme() const;

	Size2 get_contents_minimum_size() const;

	void grab_focus();
	bool has_focus() const;

	Rect2i get_usable_parent_rect() const;

	Ref<Texture2D> get_theme_icon(const StringName &p_name, const StringName &p_type = StringName()) const;
	Ref<Shader> get_theme_shader(const StringName &p_name, const StringName &p_type = StringName()) const;
	Ref<StyleBox> get_theme_stylebox(const StringName &p_name, const StringName &p_type = StringName()) const;
	Ref<Font> get_theme_font(const StringName &p_name, const StringName &p_type = StringName()) const;
	Color get_theme_color(const StringName &p_name, const StringName &p_type = StringName()) const;
	int get_theme_constant(const StringName &p_name, const StringName &p_type = StringName()) const;

	bool has_theme_icon(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_theme_shader(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_theme_stylebox(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_theme_font(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_theme_color(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_theme_constant(const StringName &p_name, const StringName &p_type = StringName()) const;

	Rect2i get_parent_rect() const;
	virtual DisplayServer::WindowID get_window_id() const override;

	Window();
	~Window();
};

VARIANT_ENUM_CAST(Window::Mode);
VARIANT_ENUM_CAST(Window::Flags);
VARIANT_ENUM_CAST(Window::ContentScaleMode);
VARIANT_ENUM_CAST(Window::ContentScaleAspect);

#endif // WINDOW_H
