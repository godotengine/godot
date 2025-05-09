/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_X11

#include "SDL_x11video.h"
#include "SDL_x11dyn.h"
#include "SDL_x11messagebox.h"

#include <X11/keysym.h>
#include <locale.h>

#define SDL_FORK_MESSAGEBOX 1
#define SDL_SET_LOCALE      1

#if SDL_FORK_MESSAGEBOX
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#endif

#define MAX_BUTTONS       8   // Maximum number of buttons supported
#define MIN_BUTTON_WIDTH  64  // Minimum button width
#define MIN_DIALOG_WIDTH  200 // Minimum dialog width
#define MIN_DIALOG_HEIGHT 100 // Minimum dialog height

static const char g_MessageBoxFontLatin1[] =
    "-*-*-medium-r-normal--0-120-*-*-p-0-iso8859-1";

static const char* g_MessageBoxFont[] = {
    "-*-*-medium-r-normal--*-120-*-*-*-*-iso10646-1",  // explicitly unicode (iso10646-1)
    "-*-*-medium-r-*--*-120-*-*-*-*-iso10646-1",  // explicitly unicode (iso10646-1)
    "-misc-*-*-*-*--*-*-*-*-*-*-iso10646-1",  // misc unicode (fix for some systems)
    "-*-*-*-*-*--*-*-*-*-*-*-iso10646-1",  // just give me anything Unicode.
    "-*-*-medium-r-normal--*-120-*-*-*-*-iso8859-1",  // explicitly latin1, in case low-ASCII works out.
    "-*-*-medium-r-*--*-120-*-*-*-*-iso8859-1",  // explicitly latin1, in case low-ASCII works out.
    "-misc-*-*-*-*--*-*-*-*-*-*-iso8859-1",  // misc latin1 (fix for some systems)
    "-*-*-*-*-*--*-*-*-*-*-*-iso8859-1",  // just give me anything latin1.
    NULL
};

static const SDL_MessageBoxColor g_default_colors[SDL_MESSAGEBOX_COLOR_COUNT] = {
    { 56, 54, 53 },    // SDL_MESSAGEBOX_COLOR_BACKGROUND,
    { 209, 207, 205 }, // SDL_MESSAGEBOX_COLOR_TEXT,
    { 140, 135, 129 }, // SDL_MESSAGEBOX_COLOR_BUTTON_BORDER,
    { 105, 102, 99 },  // SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND,
    { 205, 202, 53 },  // SDL_MESSAGEBOX_COLOR_BUTTON_SELECTED,
};

#define SDL_MAKE_RGB(_r, _g, _b) (((Uint32)(_r) << 16) | \
                                  ((Uint32)(_g) << 8) |  \
                                  ((Uint32)(_b)))

typedef struct SDL_MessageBoxButtonDataX11
{
    int x, y;       // Text position
    int length;     // Text length
    int text_width; // Text width

    SDL_Rect rect; // Rectangle for entire button

    const SDL_MessageBoxButtonData *buttondata; // Button data from caller
} SDL_MessageBoxButtonDataX11;

typedef struct TextLineData
{
    int width;        // Width of this text line
    int length;       // String length of this text line
    const char *text; // Text for this line
} TextLineData;

typedef struct SDL_MessageBoxDataX11
{
    Display *display;
    int screen;
    Window window;
#ifdef SDL_VIDEO_DRIVER_X11_XDBE
    XdbeBackBuffer buf;
    bool xdbe; // Whether Xdbe is present or not
#endif
    long event_mask;
    Atom wm_protocols;
    Atom wm_delete_message;

    int dialog_width;  // Dialog box width.
    int dialog_height; // Dialog box height.

    XFontSet font_set;        // for UTF-8 systems
    XFontStruct *font_struct; // Latin1 (ASCII) fallback.
    int xtext, ytext;         // Text position to start drawing at.
    int numlines;             // Count of Text lines.
    int text_height;          // Height for text lines.
    TextLineData *linedata;

    int *pbuttonid; // Pointer to user return buttonID value.

    int button_press_index; // Index into buttondata/buttonpos for button which is pressed (or -1).
    int mouse_over_index;   // Index into buttondata/buttonpos for button mouse is over (or -1).

    int numbuttons; // Count of buttons.
    const SDL_MessageBoxButtonData *buttondata;
    SDL_MessageBoxButtonDataX11 buttonpos[MAX_BUTTONS];

    Uint32 color[SDL_MESSAGEBOX_COLOR_COUNT];

    const SDL_MessageBoxData *messageboxdata;
} SDL_MessageBoxDataX11;

// Maximum helper for ints.
static SDL_INLINE int IntMax(int a, int b)
{
    return (a > b) ? a : b;
}

// Return width and height for a string.
static void GetTextWidthHeight(SDL_MessageBoxDataX11 *data, const char *str, int nbytes, int *pwidth, int *pheight)
{
#ifdef X_HAVE_UTF8_STRING
    if (SDL_X11_HAVE_UTF8) {
        XRectangle overall_ink, overall_logical;
        X11_Xutf8TextExtents(data->font_set, str, nbytes, &overall_ink, &overall_logical);
        *pwidth = overall_logical.width;
        *pheight = overall_logical.height;
    } else
#endif
    {
        XCharStruct text_structure;
        int font_direction, font_ascent, font_descent;
        X11_XTextExtents(data->font_struct, str, nbytes,
                         &font_direction, &font_ascent, &font_descent,
                         &text_structure);
        *pwidth = text_structure.width;
        *pheight = text_structure.ascent + text_structure.descent;
    }
}

// Return index of button if position x,y is contained therein.
static int GetHitButtonIndex(SDL_MessageBoxDataX11 *data, int x, int y)
{
    int i;
    int numbuttons = data->numbuttons;
    SDL_MessageBoxButtonDataX11 *buttonpos = data->buttonpos;

    for (i = 0; i < numbuttons; i++) {
        SDL_Rect *rect = &buttonpos[i].rect;

        if ((x >= rect->x) &&
            (x <= (rect->x + rect->w)) &&
            (y >= rect->y) &&
            (y <= (rect->y + rect->h))) {
            return i;
        }
    }

    return -1;
}

// Initialize SDL_MessageBoxData structure and Display, etc.
static bool X11_MessageBoxInit(SDL_MessageBoxDataX11 *data, const SDL_MessageBoxData *messageboxdata, int *pbuttonid)
{
    int i;
    int numbuttons = messageboxdata->numbuttons;
    const SDL_MessageBoxButtonData *buttondata = messageboxdata->buttons;
    const SDL_MessageBoxColor *colorhints;

    if (numbuttons > MAX_BUTTONS) {
        return SDL_SetError("Too many buttons (%d max allowed)", MAX_BUTTONS);
    }

    data->dialog_width = MIN_DIALOG_WIDTH;
    data->dialog_height = MIN_DIALOG_HEIGHT;
    data->messageboxdata = messageboxdata;
    data->buttondata = buttondata;
    data->numbuttons = numbuttons;
    data->pbuttonid = pbuttonid;

    data->display = X11_XOpenDisplay(NULL);
    if (!data->display) {
        return SDL_SetError("Couldn't open X11 display");
    }

#ifdef X_HAVE_UTF8_STRING
    if (SDL_X11_HAVE_UTF8) {
        char **missing = NULL;
        int num_missing = 0;
        int i_font;
        for (i_font = 0; g_MessageBoxFont[i_font]; ++i_font) {
            data->font_set = X11_XCreateFontSet(data->display, g_MessageBoxFont[i_font],
                                                &missing, &num_missing, NULL);
            if (missing) {
                X11_XFreeStringList(missing);
            }
            if (data->font_set) {
                break;
            }
        }
        if (!data->font_set) {
            return SDL_SetError("Couldn't load x11 message box font");
        }
    } else
#endif
    {
        data->font_struct = X11_XLoadQueryFont(data->display, g_MessageBoxFontLatin1);
        if (!data->font_struct) {
            return SDL_SetError("Couldn't load font %s", g_MessageBoxFontLatin1);
        }
    }

    if (messageboxdata->colorScheme) {
        colorhints = messageboxdata->colorScheme->colors;
    } else {
        colorhints = g_default_colors;
    }

    // Convert our SDL_MessageBoxColor r,g,b values to packed RGB format.
    for (i = 0; i < SDL_MESSAGEBOX_COLOR_COUNT; i++) {
        data->color[i] = SDL_MAKE_RGB(colorhints[i].r, colorhints[i].g, colorhints[i].b);
    }

    return true;
}

static int CountLinesOfText(const char *text)
{
    int result = 0;
    while (text && *text) {
        const char *lf = SDL_strchr(text, '\n');
        result++; // even without an endline, this counts as a line.
        text = lf ? lf + 1 : NULL;
    }
    return result;
}

// Calculate and initialize text and button locations.
static bool X11_MessageBoxInitPositions(SDL_MessageBoxDataX11 *data)
{
    int i;
    int ybuttons;
    int text_width_max = 0;
    int button_text_height = 0;
    int button_width = MIN_BUTTON_WIDTH;
    const SDL_MessageBoxData *messageboxdata = data->messageboxdata;

    // Go over text and break linefeeds into separate lines.
    if (messageboxdata && messageboxdata->message[0]) {
        const char *text = messageboxdata->message;
        const int linecount = CountLinesOfText(text);
        TextLineData *plinedata = (TextLineData *)SDL_malloc(sizeof(TextLineData) * linecount);

        if (!plinedata) {
            return false;
        }

        data->linedata = plinedata;
        data->numlines = linecount;

        for (i = 0; i < linecount; i++, plinedata++) {
            const char *lf = SDL_strchr(text, '\n');
            const int length = lf ? (lf - text) : SDL_strlen(text);
            int height;

            plinedata->text = text;

            GetTextWidthHeight(data, text, length, &plinedata->width, &height);

            // Text and widths are the largest we've ever seen.
            data->text_height = IntMax(data->text_height, height);
            text_width_max = IntMax(text_width_max, plinedata->width);

            plinedata->length = length;
            if (lf && (lf > text) && (lf[-1] == '\r')) {
                plinedata->length--;
            }

            text += length + 1;

            // Break if there are no more linefeeds.
            if (!lf) {
                break;
            }
        }

        // Bump up the text height slightly.
        data->text_height += 2;
    }

    // Loop through all buttons and calculate the button widths and height.
    for (i = 0; i < data->numbuttons; i++) {
        int height;

        data->buttonpos[i].buttondata = &data->buttondata[i];
        data->buttonpos[i].length = SDL_strlen(data->buttondata[i].text);

        GetTextWidthHeight(data, data->buttondata[i].text, SDL_strlen(data->buttondata[i].text),
                           &data->buttonpos[i].text_width, &height);

        button_width = IntMax(button_width, data->buttonpos[i].text_width);
        button_text_height = IntMax(button_text_height, height);
    }

    if (data->numlines) {
        // x,y for this line of text.
        data->xtext = data->text_height;
        data->ytext = data->text_height + data->text_height;

        // Bump button y down to bottom of text.
        ybuttons = 3 * data->ytext / 2 + (data->numlines - 1) * data->text_height;

        // Bump the dialog box width and height up if needed.
        data->dialog_width = IntMax(data->dialog_width, 2 * data->xtext + text_width_max);
        data->dialog_height = IntMax(data->dialog_height, ybuttons);
    } else {
        // Button y starts at height of button text.
        ybuttons = button_text_height;
    }

    if (data->numbuttons) {
        int x, y;
        int width_of_buttons;
        int button_spacing = button_text_height;
        int button_height = 2 * button_text_height;

        // Bump button width up a bit.
        button_width += button_text_height;

        // Get width of all buttons lined up.
        width_of_buttons = data->numbuttons * button_width + (data->numbuttons - 1) * button_spacing;

        // Bump up dialog width and height if buttons are wider than text.
        data->dialog_width = IntMax(data->dialog_width, width_of_buttons + 2 * button_spacing);
        data->dialog_height = IntMax(data->dialog_height, ybuttons + 2 * button_height);

        // Location for first button.
        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
            x = data->dialog_width - (data->dialog_width - width_of_buttons) / 2 - (button_width + button_spacing);
        } else {
            x = (data->dialog_width - width_of_buttons) / 2;
        }
        y = ybuttons + (data->dialog_height - ybuttons - button_height) / 2;

        for (i = 0; i < data->numbuttons; i++) {
            // Button coordinates.
            data->buttonpos[i].rect.x = x;
            data->buttonpos[i].rect.y = y;
            data->buttonpos[i].rect.w = button_width;
            data->buttonpos[i].rect.h = button_height;

            // Button text coordinates.
            data->buttonpos[i].x = x + (button_width - data->buttonpos[i].text_width) / 2;
            data->buttonpos[i].y = y + (button_height - button_text_height - 1) / 2 + button_text_height;

            // Scoot over for next button.
            if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
                x -= button_width + button_spacing;
            } else {
                x += button_width + button_spacing;
            }
        }
    }

    return true;
}

// Free SDL_MessageBoxData data.
static void X11_MessageBoxShutdown(SDL_MessageBoxDataX11 *data)
{
    if (data->font_set) {
        X11_XFreeFontSet(data->display, data->font_set);
        data->font_set = NULL;
    }

    if (data->font_struct) {
        X11_XFreeFont(data->display, data->font_struct);
        data->font_struct = NULL;
    }

#ifdef SDL_VIDEO_DRIVER_X11_XDBE
    if (SDL_X11_HAVE_XDBE && data->xdbe) {
        X11_XdbeDeallocateBackBufferName(data->display, data->buf);
    }
#endif

    if (data->display) {
        if (data->window != None) {
            X11_XWithdrawWindow(data->display, data->window, data->screen);
            X11_XDestroyWindow(data->display, data->window);
            data->window = None;
        }

        X11_XCloseDisplay(data->display);
        data->display = NULL;
    }

    SDL_free(data->linedata);
}

// Create and set up our X11 dialog box indow.
static bool X11_MessageBoxCreateWindow(SDL_MessageBoxDataX11 *data)
{
    int x, y;
    XSizeHints *sizehints;
    XSetWindowAttributes wnd_attr;
    Atom _NET_WM_WINDOW_TYPE, _NET_WM_WINDOW_TYPE_DIALOG;
    Display *display = data->display;
    SDL_WindowData *windowdata = NULL;
    const SDL_MessageBoxData *messageboxdata = data->messageboxdata;

    if (messageboxdata->window) {
        SDL_DisplayData *displaydata = SDL_GetDisplayDriverDataForWindow(messageboxdata->window);
        windowdata = messageboxdata->window->internal;
        data->screen = displaydata->screen;
    } else {
        data->screen = DefaultScreen(display);
    }

    data->event_mask = ExposureMask |
                       ButtonPressMask | ButtonReleaseMask | KeyPressMask | KeyReleaseMask |
                       StructureNotifyMask | FocusChangeMask | PointerMotionMask;
    wnd_attr.event_mask = data->event_mask;

    data->window = X11_XCreateWindow(
        display, RootWindow(display, data->screen),
        0, 0,
        data->dialog_width, data->dialog_height,
        0, CopyFromParent, InputOutput, CopyFromParent,
        CWEventMask, &wnd_attr);
    if (data->window == None) {
        return SDL_SetError("Couldn't create X window");
    }

    if (windowdata) {
        Atom _NET_WM_STATE = X11_XInternAtom(display, "_NET_WM_STATE", False);
        Atom stateatoms[16];
        size_t statecount = 0;
        // Set some message-boxy window states when attached to a parent window...
        // we skip the taskbar since this will pop to the front when the parent window is clicked in the taskbar, etc
        stateatoms[statecount++] = X11_XInternAtom(display, "_NET_WM_STATE_SKIP_TASKBAR", False);
        stateatoms[statecount++] = X11_XInternAtom(display, "_NET_WM_STATE_SKIP_PAGER", False);
        stateatoms[statecount++] = X11_XInternAtom(display, "_NET_WM_STATE_FOCUSED", False);
        stateatoms[statecount++] = X11_XInternAtom(display, "_NET_WM_STATE_MODAL", False);
        SDL_assert(statecount <= SDL_arraysize(stateatoms));
        X11_XChangeProperty(display, data->window, _NET_WM_STATE, XA_ATOM, 32,
                            PropModeReplace, (unsigned char *)stateatoms, statecount);

        // http://tronche.com/gui/x/icccm/sec-4.html#WM_TRANSIENT_FOR
        X11_XSetTransientForHint(display, data->window, windowdata->xwindow);
    }

    SDL_X11_SetWindowTitle(display, data->window, (char *)messageboxdata->title);

    // Let the window manager know this is a dialog box
    _NET_WM_WINDOW_TYPE = X11_XInternAtom(display, "_NET_WM_WINDOW_TYPE", False);
    _NET_WM_WINDOW_TYPE_DIALOG = X11_XInternAtom(display, "_NET_WM_WINDOW_TYPE_DIALOG", False);
    X11_XChangeProperty(display, data->window, _NET_WM_WINDOW_TYPE, XA_ATOM, 32,
                        PropModeReplace,
                        (unsigned char *)&_NET_WM_WINDOW_TYPE_DIALOG, 1);

    // Allow the window to be deleted by the window manager
    data->wm_delete_message = X11_XInternAtom(display, "WM_DELETE_WINDOW", False);
    X11_XSetWMProtocols(display, data->window, &data->wm_delete_message, 1);

    data->wm_protocols = X11_XInternAtom(display, "WM_PROTOCOLS", False);

    if (windowdata) {
        XWindowAttributes attrib;
        Window dummy;

        X11_XGetWindowAttributes(display, windowdata->xwindow, &attrib);
        x = attrib.x + (attrib.width - data->dialog_width) / 2;
        y = attrib.y + (attrib.height - data->dialog_height) / 3;
        X11_XTranslateCoordinates(display, windowdata->xwindow, RootWindow(display, data->screen), x, y, &x, &y, &dummy);
    } else {
        const SDL_VideoDevice *dev = SDL_GetVideoDevice();
        if (dev && dev->displays && dev->num_displays > 0) {
            const SDL_VideoDisplay *dpy = dev->displays[0];
            const SDL_DisplayData *dpydata = dpy->internal;
            x = dpydata->x + ((dpy->current_mode->w - data->dialog_width) / 2);
            y = dpydata->y + ((dpy->current_mode->h - data->dialog_height) / 3);
        } else { // oh well. This will misposition on a multi-head setup. Init first next time.
            x = (DisplayWidth(display, data->screen) - data->dialog_width) / 2;
            y = (DisplayHeight(display, data->screen) - data->dialog_height) / 3;
        }
    }
    X11_XMoveWindow(display, data->window, x, y);

    sizehints = X11_XAllocSizeHints();
    if (sizehints) {
        sizehints->flags = USPosition | USSize | PMaxSize | PMinSize;
        sizehints->x = x;
        sizehints->y = y;
        sizehints->width = data->dialog_width;
        sizehints->height = data->dialog_height;

        sizehints->min_width = sizehints->max_width = data->dialog_width;
        sizehints->min_height = sizehints->max_height = data->dialog_height;

        X11_XSetWMNormalHints(display, data->window, sizehints);

        X11_XFree(sizehints);
    }

    X11_XMapRaised(display, data->window);

#ifdef SDL_VIDEO_DRIVER_X11_XDBE
    // Initialise a back buffer for double buffering
    if (SDL_X11_HAVE_XDBE) {
        int xdbe_major, xdbe_minor;
        if (X11_XdbeQueryExtension(display, &xdbe_major, &xdbe_minor) != 0) {
            data->xdbe = true;
            data->buf = X11_XdbeAllocateBackBufferName(display, data->window, XdbeUndefined);
        } else {
            data->xdbe = false;
        }
    }
#endif

    return true;
}

// Draw our message box.
static void X11_MessageBoxDraw(SDL_MessageBoxDataX11 *data, GC ctx)
{
    int i;
    Drawable window = data->window;
    Display *display = data->display;

#ifdef SDL_VIDEO_DRIVER_X11_XDBE
    if (SDL_X11_HAVE_XDBE && data->xdbe) {
        window = data->buf;
        X11_XdbeBeginIdiom(data->display);
    }
#endif

    X11_XSetForeground(display, ctx, data->color[SDL_MESSAGEBOX_COLOR_BACKGROUND]);
    X11_XFillRectangle(display, window, ctx, 0, 0, data->dialog_width, data->dialog_height);

    X11_XSetForeground(display, ctx, data->color[SDL_MESSAGEBOX_COLOR_TEXT]);
    for (i = 0; i < data->numlines; i++) {
        TextLineData *plinedata = &data->linedata[i];

#ifdef X_HAVE_UTF8_STRING
        if (SDL_X11_HAVE_UTF8) {
            X11_Xutf8DrawString(display, window, data->font_set, ctx,
                                data->xtext, data->ytext + i * data->text_height,
                                plinedata->text, plinedata->length);
        } else
#endif
        {
            X11_XDrawString(display, window, ctx,
                            data->xtext, data->ytext + i * data->text_height,
                            plinedata->text, plinedata->length);
        }
    }

    for (i = 0; i < data->numbuttons; i++) {
        SDL_MessageBoxButtonDataX11 *buttondatax11 = &data->buttonpos[i];
        const SDL_MessageBoxButtonData *buttondata = buttondatax11->buttondata;
        int border = (buttondata->flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) ? 2 : 0;
        int offset = ((data->mouse_over_index == i) && (data->button_press_index == data->mouse_over_index)) ? 1 : 0;

        X11_XSetForeground(display, ctx, data->color[SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND]);
        X11_XFillRectangle(display, window, ctx,
                           buttondatax11->rect.x - border, buttondatax11->rect.y - border,
                           buttondatax11->rect.w + 2 * border, buttondatax11->rect.h + 2 * border);

        X11_XSetForeground(display, ctx, data->color[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER]);
        X11_XDrawRectangle(display, window, ctx,
                           buttondatax11->rect.x, buttondatax11->rect.y,
                           buttondatax11->rect.w, buttondatax11->rect.h);

        X11_XSetForeground(display, ctx, (data->mouse_over_index == i) ? data->color[SDL_MESSAGEBOX_COLOR_BUTTON_SELECTED] : data->color[SDL_MESSAGEBOX_COLOR_TEXT]);

#ifdef X_HAVE_UTF8_STRING
        if (SDL_X11_HAVE_UTF8) {
            X11_Xutf8DrawString(display, window, data->font_set, ctx,
                                buttondatax11->x + offset,
                                buttondatax11->y + offset,
                                buttondata->text, buttondatax11->length);
        } else
#endif
        {
            X11_XDrawString(display, window, ctx,
                            buttondatax11->x + offset, buttondatax11->y + offset,
                            buttondata->text, buttondatax11->length);
        }
    }

#ifdef SDL_VIDEO_DRIVER_X11_XDBE
    if (SDL_X11_HAVE_XDBE && data->xdbe) {
        XdbeSwapInfo swap_info;
        swap_info.swap_window = data->window;
        swap_info.swap_action = XdbeUndefined;
        X11_XdbeSwapBuffers(data->display, &swap_info, 1);
        X11_XdbeEndIdiom(data->display);
    }
#endif
}

// NOLINTNEXTLINE(readability-non-const-parameter): cannot make XPointer a const pointer due to typedef
static Bool X11_MessageBoxEventTest(Display *display, XEvent *event, XPointer arg)
{
    const SDL_MessageBoxDataX11 *data = (const SDL_MessageBoxDataX11 *)arg;
    return ((event->xany.display == data->display) && (event->xany.window == data->window)) ? True : False;
}

// Loop and handle message box event messages until something kills it.
static bool X11_MessageBoxLoop(SDL_MessageBoxDataX11 *data)
{
    GC ctx;
    XGCValues ctx_vals;
    bool close_dialog = false;
    bool has_focus = true;
    KeySym last_key_pressed = XK_VoidSymbol;
    unsigned long gcflags = GCForeground | GCBackground;
#ifdef X_HAVE_UTF8_STRING
    const int have_utf8 = SDL_X11_HAVE_UTF8;
#else
    const int have_utf8 = 0;
#endif

    SDL_zero(ctx_vals);
    ctx_vals.foreground = data->color[SDL_MESSAGEBOX_COLOR_BACKGROUND];
    ctx_vals.background = data->color[SDL_MESSAGEBOX_COLOR_BACKGROUND];

    if (!have_utf8) {
        gcflags |= GCFont;
        ctx_vals.font = data->font_struct->fid;
    }

    ctx = X11_XCreateGC(data->display, data->window, gcflags, &ctx_vals);
    if (ctx == None) {
        return SDL_SetError("Couldn't create graphics context");
    }

    data->button_press_index = -1; // Reset what button is currently depressed.
    data->mouse_over_index = -1;   // Reset what button the mouse is over.

    while (!close_dialog) {
        XEvent e;
        bool draw = true;

        // can't use XWindowEvent() because it can't handle ClientMessage events.
        // can't use XNextEvent() because we only want events for this window.
        X11_XIfEvent(data->display, &e, X11_MessageBoxEventTest, (XPointer)data);

        /* If X11_XFilterEvent returns True, then some input method has filtered the
           event, and the client should discard the event. */
        if ((e.type != Expose) && X11_XFilterEvent(&e, None)) {
            continue;
        }

        switch (e.type) {
        case Expose:
            if (e.xexpose.count > 0) {
                draw = false;
            }
            break;

        case FocusIn:
            // Got focus.
            has_focus = true;
            break;

        case FocusOut:
            // lost focus. Reset button and mouse info.
            has_focus = false;
            data->button_press_index = -1;
            data->mouse_over_index = -1;
            break;

        case MotionNotify:
            if (has_focus) {
                // Mouse moved...
                const int previndex = data->mouse_over_index;
                data->mouse_over_index = GetHitButtonIndex(data, e.xbutton.x, e.xbutton.y);
                if (data->mouse_over_index == previndex) {
                    draw = false;
                }
            }
            break;

        case ClientMessage:
            if (e.xclient.message_type == data->wm_protocols &&
                e.xclient.format == 32 &&
                e.xclient.data.l[0] == data->wm_delete_message) {
                close_dialog = true;
            }
            break;

        case KeyPress:
            // Store key press - we make sure in key release that we got both.
            last_key_pressed = X11_XLookupKeysym(&e.xkey, 0);
            break;

        case KeyRelease:
        {
            Uint32 mask = 0;
            KeySym key = X11_XLookupKeysym(&e.xkey, 0);

            // If this is a key release for something we didn't get the key down for, then bail.
            if (key != last_key_pressed) {
                break;
            }

            if (key == XK_Escape) {
                mask = SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT;
            } else if ((key == XK_Return) || (key == XK_KP_Enter)) {
                mask = SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT;
            }

            if (mask) {
                int i;

                // Look for first button with this mask set, and return it if found.
                for (i = 0; i < data->numbuttons; i++) {
                    SDL_MessageBoxButtonDataX11 *buttondatax11 = &data->buttonpos[i];

                    if (buttondatax11->buttondata->flags & mask) {
                        *data->pbuttonid = buttondatax11->buttondata->buttonID;
                        close_dialog = true;
                        break;
                    }
                }
            }
            break;
        }

        case ButtonPress:
            data->button_press_index = -1;
            if (e.xbutton.button == Button1) {
                // Find index of button they clicked on.
                data->button_press_index = GetHitButtonIndex(data, e.xbutton.x, e.xbutton.y);
            }
            break;

        case ButtonRelease:
            // If button is released over the same button that was clicked down on, then return it.
            if ((e.xbutton.button == Button1) && (data->button_press_index >= 0)) {
                int button = GetHitButtonIndex(data, e.xbutton.x, e.xbutton.y);

                if (data->button_press_index == button) {
                    SDL_MessageBoxButtonDataX11 *buttondatax11 = &data->buttonpos[button];

                    *data->pbuttonid = buttondatax11->buttondata->buttonID;
                    close_dialog = true;
                }
            }
            data->button_press_index = -1;
            break;
        }

        if (draw) {
            // Draw our dialog box.
            X11_MessageBoxDraw(data, ctx);
        }
    }

    X11_XFreeGC(data->display, ctx);
    return true;
}

static bool X11_ShowMessageBoxImpl(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    bool result = false;
    SDL_MessageBoxDataX11 data;
#if SDL_SET_LOCALE
    char *origlocale;
#endif

    SDL_zero(data);

    if (!SDL_X11_LoadSymbols()) {
        return false;
    }

#if SDL_SET_LOCALE
    origlocale = setlocale(LC_ALL, NULL);
    if (origlocale) {
        origlocale = SDL_strdup(origlocale);
        if (!origlocale) {
            return false;
        }
        (void)setlocale(LC_ALL, "");
    }
#endif

    // This code could get called from multiple threads maybe?
    X11_XInitThreads();

    // Initialize the return buttonID value to -1 (for error or dialogbox closed).
    *buttonID = -1;

    // Init and display the message box.
    if (X11_MessageBoxInit(&data, messageboxdata, buttonID) &&
        X11_MessageBoxInitPositions(&data) &&
        X11_MessageBoxCreateWindow(&data)) {
        result = X11_MessageBoxLoop(&data);
    }

    X11_MessageBoxShutdown(&data);

#if SDL_SET_LOCALE
    if (origlocale) {
        (void)setlocale(LC_ALL, origlocale);
        SDL_free(origlocale);
    }
#endif

    return result;
}

// Display an x11 message box.
bool X11_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
#if SDL_FORK_MESSAGEBOX
    // Use a child process to protect against setlocale(). Annoying.
    pid_t pid;
    int fds[2];
    int status = 0;
    bool result = true;

    if (pipe(fds) == -1) {
        return X11_ShowMessageBoxImpl(messageboxdata, buttonID); // oh well.
    }

    pid = fork();
    if (pid == -1) { // failed
        close(fds[0]);
        close(fds[1]);
        return X11_ShowMessageBoxImpl(messageboxdata, buttonID); // oh well.
    } else if (pid == 0) {                                       // we're the child
        int exitcode = 0;
        close(fds[0]);
        result = X11_ShowMessageBoxImpl(messageboxdata, buttonID);
        if (write(fds[1], &result, sizeof(result)) != sizeof(result)) {
            exitcode = 1;
        } else if (write(fds[1], buttonID, sizeof(*buttonID)) != sizeof(*buttonID)) {
            exitcode = 1;
        }
        close(fds[1]);
        _exit(exitcode); // don't run atexit() stuff, static destructors, etc.
    } else {             // we're the parent
        pid_t rc;
        close(fds[1]);
        do {
            rc = waitpid(pid, &status, 0);
        } while ((rc == -1) && (errno == EINTR));

        SDL_assert(rc == pid); // not sure what to do if this fails.

        if ((rc == -1) || (!WIFEXITED(status)) || (WEXITSTATUS(status) != 0)) {
            result = SDL_SetError("msgbox child process failed");
        } else if ((read(fds[0], &result, sizeof(result)) != sizeof(result)) ||
                   (read(fds[0], buttonID, sizeof(*buttonID)) != sizeof(*buttonID))) {
            result = SDL_SetError("read from msgbox child process failed");
            *buttonID = 0;
        }
        close(fds[0]);

        return result;
    }
#else
    return X11_ShowMessageBoxImpl(messageboxdata, buttonID);
#endif
}
#endif // SDL_VIDEO_DRIVER_X11
