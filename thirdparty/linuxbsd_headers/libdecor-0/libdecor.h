/*
 * Copyright © 2017-2018 Red Hat Inc.
 * Copyright © 2018 Jonas Ådahl
 * Copyright © 2019 Christian Rauch
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef LIBDECOR_H
#define LIBDECOR_H

#include <stdbool.h>
#include <wayland-client.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && __GNUC__ >= 4
#define LIBDECOR_EXPORT __attribute__ ((visibility("default")))
#else
#define LIBDECOR_EXPORT
#endif

struct xdg_toplevel;

/** \class libdecor
 *
 * \brief A libdecor context instance.
 */
struct libdecor;

/** \class libdecor_frame
 *
 * \brief A frame used for decorating a Wayland surface.
 */
struct libdecor_frame;

/** \class libdecor_configuration
 *
 * \brief An object representing a toplevel window configuration.
 */
struct libdecor_configuration;

/** \class libdecor_state
 *
 * \brief An object corresponding to a configured content state.
 */
struct libdecor_state;

enum libdecor_error {
	LIBDECOR_ERROR_COMPOSITOR_INCOMPATIBLE,
	LIBDECOR_ERROR_INVALID_FRAME_CONFIGURATION,
};

enum libdecor_window_state {
	LIBDECOR_WINDOW_STATE_NONE = 0,
	LIBDECOR_WINDOW_STATE_ACTIVE = 1 << 0,
	LIBDECOR_WINDOW_STATE_MAXIMIZED = 1 << 1,
	LIBDECOR_WINDOW_STATE_FULLSCREEN = 1 << 2,
	LIBDECOR_WINDOW_STATE_TILED_LEFT = 1 << 3,
	LIBDECOR_WINDOW_STATE_TILED_RIGHT = 1 << 4,
	LIBDECOR_WINDOW_STATE_TILED_TOP = 1 << 5,
	LIBDECOR_WINDOW_STATE_TILED_BOTTOM = 1 << 6,
	LIBDECOR_WINDOW_STATE_SUSPENDED = 1 << 7,
};

enum libdecor_resize_edge {
	LIBDECOR_RESIZE_EDGE_NONE,
	LIBDECOR_RESIZE_EDGE_TOP,
	LIBDECOR_RESIZE_EDGE_BOTTOM,
	LIBDECOR_RESIZE_EDGE_LEFT,
	LIBDECOR_RESIZE_EDGE_TOP_LEFT,
	LIBDECOR_RESIZE_EDGE_BOTTOM_LEFT,
	LIBDECOR_RESIZE_EDGE_RIGHT,
	LIBDECOR_RESIZE_EDGE_TOP_RIGHT,
	LIBDECOR_RESIZE_EDGE_BOTTOM_RIGHT,
};

enum libdecor_capabilities {
	LIBDECOR_ACTION_MOVE = 1 << 0,
	LIBDECOR_ACTION_RESIZE = 1 << 1,
	LIBDECOR_ACTION_MINIMIZE = 1 << 2,
	LIBDECOR_ACTION_FULLSCREEN = 1 << 3,
	LIBDECOR_ACTION_CLOSE = 1 << 4,
};

struct libdecor_interface {
	/**
	 * An error event
	 */
	void (* error)(struct libdecor *context,
		       enum libdecor_error error,
		       const char *message);

	/* Reserved */
	void (* reserved0)(void);
	void (* reserved1)(void);
	void (* reserved2)(void);
	void (* reserved3)(void);
	void (* reserved4)(void);
	void (* reserved5)(void);
	void (* reserved6)(void);
	void (* reserved7)(void);
	void (* reserved8)(void);
	void (* reserved9)(void);
};

/**
 * Interface for integrating a Wayland surface with libdecor.
 */
struct libdecor_frame_interface {
	/**
	 * A new configuration was received. An application should respond to
	 * this by creating a suitable libdecor_state, and apply it using
	 * libdecor_frame_commit.
	 */
	void (* configure)(struct libdecor_frame *frame,
			   struct libdecor_configuration *configuration,
			   void *user_data);

	/**
	 * The window was requested to be closed by the compositor.
	 */
	void (* close)(struct libdecor_frame *frame,
		       void *user_data);

	/**
	 * The window decoration asked to have the main surface to be
	 * committed. This is required when the decoration is implemented using
	 * synchronous subsurfaces.
	 */
	void (* commit)(struct libdecor_frame *frame,
			void *user_data);

	/**
	 * Any mapped popup that has a grab on the given seat should be
	 * dismissed.
	 */
	void (* dismiss_popup)(struct libdecor_frame *frame,
			       const char *seat_name,
			       void *user_data);

	/* Reserved */
	void (* reserved0)(void);
	void (* reserved1)(void);
	void (* reserved2)(void);
	void (* reserved3)(void);
	void (* reserved4)(void);
	void (* reserved5)(void);
	void (* reserved6)(void);
	void (* reserved7)(void);
	void (* reserved8)(void);
	void (* reserved9)(void);
};

/**
 * Remove a reference to the libdecor instance. When the reference count
 * reaches zero, it is freed.
 */
void
libdecor_unref(struct libdecor *context);

/**
 * Create a new libdecor context for the given wl_display.
 */
struct libdecor *
libdecor_new(struct wl_display *display,
	     struct libdecor_interface *iface);

/**
 * Get the file descriptor used by libdecor. This is similar to
 * wl_display_get_fd(), thus should be polled, and when data is available,
 * libdecor_dispatch() should be called.
 */
int
libdecor_get_fd(struct libdecor *context);

/**
 * Dispatch events. This function should be called when data is available on
 * the file descriptor returned by libdecor_get_fd(). If timeout is zero, this
 * function will never block.
 */
int
libdecor_dispatch(struct libdecor *context,
		  int timeout);

/**
 * Decorate the given content wl_surface.
 *
 * This will create an xdg_surface and an xdg_toplevel, and integrate it
 * properly with the windowing system, including creating appropriate
 * decorations when needed, as well as handle windowing integration events such
 * as resizing, moving, maximizing, etc.
 *
 * The passed wl_surface should only contain actual application content,
 * without any window decoration.
 */
struct libdecor_frame *
libdecor_decorate(struct libdecor *context,
		  struct wl_surface *surface,
		  struct libdecor_frame_interface *iface,
		  void *user_data);

/**
 * Add a reference to the frame object.
 */
void
libdecor_frame_ref(struct libdecor_frame *frame);

/**
 * Remove a reference to the frame object. When the reference count reaches
 * zero, the frame object is destroyed.
 */
void
libdecor_frame_unref(struct libdecor_frame *frame);

/**
 * Set the visibility of the frame.
 *
 * If an application wants to be borderless, it can set the frame visibility to
 * false.
 */
void
libdecor_frame_set_visibility(struct libdecor_frame *frame,
			      bool visible);

/**
 * Get the visibility of the frame.
 */
bool
libdecor_frame_is_visible(struct libdecor_frame *frame);


/**
 * Set the parent of the window.
 *
 * This can be used to stack multiple toplevel windows above or under each
 * other.
 */
void
libdecor_frame_set_parent(struct libdecor_frame *frame,
			  struct libdecor_frame *parent);

/**
 * Set the title of the window.
 */
void
libdecor_frame_set_title(struct libdecor_frame *frame,
			 const char *title);

/**
 * Get the title of the window.
 */
const char *
libdecor_frame_get_title(struct libdecor_frame *frame);

/**
 * Set the application ID of the window.
 */
void
libdecor_frame_set_app_id(struct libdecor_frame *frame,
			  const char *app_id);

/**
 * Set new capabilities of the window.
 *
 * This determines whether e.g. a window decoration should show a maximize
 * button, etc.
 *
 * Setting a capability does not implicitly unset any other.
 */
void
libdecor_frame_set_capabilities(struct libdecor_frame *frame,
				enum libdecor_capabilities capabilities);

/**
 * Unset capabilities of the window.
 *
 * The opposite of libdecor_frame_set_capabilities.
 */
void
libdecor_frame_unset_capabilities(struct libdecor_frame *frame,
				  enum libdecor_capabilities capabilities);

/**
 * Check whether the window has any of the given capabilities.
 */
bool
libdecor_frame_has_capability(struct libdecor_frame *frame,
			      enum libdecor_capabilities capability);

/**
 * Show the window menu.
 */
void
libdecor_frame_show_window_menu(struct libdecor_frame *frame,
				struct wl_seat *wl_seat,
				uint32_t serial,
				int x,
				int y);

/**
 * Issue a popup grab on the window. Call this when a xdg_popup is mapped, so
 * that it can be properly dismissed by the decorations.
 */
void
libdecor_frame_popup_grab(struct libdecor_frame *frame,
			  const char *seat_name);

/**
 * Release the popup grab. Call this when you unmap a popup.
 */
void
libdecor_frame_popup_ungrab(struct libdecor_frame *frame,
			    const char *seat_name);

/**
 * Translate content surface local coordinates to toplevel window local
 * coordinates.
 *
 * This can be used to translate surface coordinates to coordinates useful for
 * e.g. showing the window menu, or positioning a popup.
 */
void
libdecor_frame_translate_coordinate(struct libdecor_frame *frame,
				    int surface_x,
				    int surface_y,
				    int *frame_x,
				    int *frame_y);

/**
 * Set the min content size.
 *
 * This translates roughly to xdg_toplevel_set_min_size().
 */
void
libdecor_frame_set_min_content_size(struct libdecor_frame *frame,
				    int content_width,
				    int content_height);

/**
 * Set the max content size.
 *
 * This translates roughly to xdg_toplevel_set_max_size().
 */
void
libdecor_frame_set_max_content_size(struct libdecor_frame *frame,
				    int content_width,
				    int content_height);

/**
 * Get the min content size.
 */
void
libdecor_frame_get_min_content_size(const struct libdecor_frame *frame,
				    int *content_width,
				    int *content_height);

/**
 * Get the max content size.
 */
void
libdecor_frame_get_max_content_size(const struct libdecor_frame *frame,
				    int *content_width,
				    int *content_height);

/**
 * Initiate an interactive resize.
 *
 * This roughly translates to xdg_toplevel_resize().
 */
void
libdecor_frame_resize(struct libdecor_frame *frame,
		      struct wl_seat *wl_seat,
		      uint32_t serial,
		      enum libdecor_resize_edge edge);

/**
 * Initiate an interactive move.
 *
 * This roughly translates to xdg_toplevel_move().
 */
void
libdecor_frame_move(struct libdecor_frame *frame,
		    struct wl_seat *wl_seat,
		    uint32_t serial);

/**
 * Commit a new window state. This can be called on application driven resizes
 * when the window is floating, or in response to received configurations, i.e.
 * from e.g. interactive resizes or state changes.
 */
void
libdecor_frame_commit(struct libdecor_frame *frame,
		      struct libdecor_state *state,
		      struct libdecor_configuration *configuration);

/**
 * Minimize the window.
 *
 * Roughly translates to xdg_toplevel_set_minimized().
 */
void
libdecor_frame_set_minimized(struct libdecor_frame *frame);

/**
 * Maximize the window.
 *
 * Roughly translates to xdg_toplevel_set_maximized().
 */
void
libdecor_frame_set_maximized(struct libdecor_frame *frame);

/**
 * Unmaximize the window.
 *
 * Roughly translates to xdg_toplevel_unset_maximized().
 */
void
libdecor_frame_unset_maximized(struct libdecor_frame *frame);

/**
 * Fullscreen the window.
 *
 * Roughly translates to xdg_toplevel_set_fullscreen().
 */
void
libdecor_frame_set_fullscreen(struct libdecor_frame *frame,
			      struct wl_output *output);

/**
 * Unfullscreen the window.
 *
 * Roughly translates to xdg_toplevel_unset_unfullscreen().
 */
void
libdecor_frame_unset_fullscreen(struct libdecor_frame *frame);

/**
 * Return true if the window is floating.
 *
 * A window is floating when it's not maximized, tiled, fullscreen, or in any
 * similar way with a fixed size and state.
 * Note that this function uses the "applied" configuration. If this function
 * is used in the 'configure' callback, the provided configuration has to be
 * applied via 'libdecor_frame_commit' first, before it will reflect the current
 * window state from the provided configuration.
 */
bool
libdecor_frame_is_floating(struct libdecor_frame *frame);

/**
 * Close the window.
 *
 * Roughly translates to xdg_toplevel_close().
 */
void
libdecor_frame_close(struct libdecor_frame *frame);

/**
 * Map the window.
 *
 * This will eventually result in the initial configure event.
 */
void
libdecor_frame_map(struct libdecor_frame *frame);

/**
 * Get the associated xdg_surface for content wl_surface.
 */
struct xdg_surface *
libdecor_frame_get_xdg_surface(struct libdecor_frame *frame);

/**
 * Get the associated xdg_toplevel for the content wl_surface.
 */
struct xdg_toplevel *
libdecor_frame_get_xdg_toplevel(struct libdecor_frame *frame);

/**
 * Create a new content surface state.
 */
struct libdecor_state *
libdecor_state_new(int width,
		   int height);

/**
 * Free a content surface state.
 */
void
libdecor_state_free(struct libdecor_state *state);

/**
 * Get the expected size of the content for this configuration.
 *
 * If the configuration doesn't contain a size, false is returned.
 */
bool
libdecor_configuration_get_content_size(struct libdecor_configuration *configuration,
					struct libdecor_frame *frame,
					int *width,
					int *height);

/**
 * Get the window state for this configuration.
 *
 * If the configuration doesn't contain any associated window state, false is
 * returned, and the application should assume the window state remains
 * unchanged.
 */
bool
libdecor_configuration_get_window_state(struct libdecor_configuration *configuration,
					enum libdecor_window_state *window_state);

#ifdef __cplusplus
}
#endif

#endif /* LIBDECOR_H */
