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
#include "SDL_x11settings.h"
#include "edid.h"
#include "../../events/SDL_displayevents_c.h"

// #define X11MODES_DEBUG

/* Timeout and revert mode switches if the timespan has elapsed without the window becoming fullscreen.
 * 5 seconds seems good from testing.
 */
#define MODE_SWITCH_TIMEOUT_NS SDL_NS_PER_SECOND * 5

/* I'm becoming more and more convinced that the application should never
 * use XRandR, and it's the window manager's responsibility to track and
 * manage display modes for fullscreen windows.  Right now XRandR is completely
 * broken with respect to window manager behavior on every window manager that
 * I can find.  For example, on Unity 3D if you show a fullscreen window while
 * the resolution is changing (within ~250 ms) your window will retain the
 * fullscreen state hint but be decorated and windowed.
 *
 * However, many people swear by it, so let them swear at it. :)
 */
// #define XRANDR_DISABLED_BY_DEFAULT

static float GetGlobalContentScale(SDL_VideoDevice *_this)
{
    static double scale_factor = 0.0;

    if (scale_factor <= 0.0) {

        // First use the forced scaling factor specified by the app/user
        const char *hint = SDL_GetHint(SDL_HINT_VIDEO_X11_SCALING_FACTOR);
        if (hint && *hint) {
            double value = SDL_atof(hint);
            if (value >= 1.0f && value <= 10.0f) {
                scale_factor = value;
            }
        }

        // If that failed, try "Xft.dpi" from the XResourcesDatabase...
        if (scale_factor <= 0.0)
        {
            SDL_VideoData *data = _this->internal;
            Display *display = data->display;
            char *resource_manager;
            XrmDatabase db;
            XrmValue value;
            char *type;

            X11_XrmInitialize();

            resource_manager = X11_XResourceManagerString(display);
            if (resource_manager) {
                db = X11_XrmGetStringDatabase(resource_manager);

                // Get the value of Xft.dpi from the Database
                if (X11_XrmGetResource(db, "Xft.dpi", "String", &type, &value)) {
                    if (value.addr && type && SDL_strcmp(type, "String") == 0) {
                        int dpi = SDL_atoi(value.addr);
                        scale_factor  = dpi / 96.0;
                    }
                }
                X11_XrmDestroyDatabase(db);
            }
        }

        // If that failed, try the XSETTINGS keys...
        if (scale_factor <= 0.0) {
            scale_factor = X11_GetXsettingsIntKey(_this, "Gdk/WindowScalingFactor", -1);

            // The Xft/DPI key is stored in increments of 1024th
            if (scale_factor <= 0.0) {
                int dpi = X11_GetXsettingsIntKey(_this, "Xft/DPI", -1);
                if (dpi > 0) {
                    scale_factor = (double) dpi / 1024.0;
                    scale_factor /= 96.0;
                }
            }
        }

        // If that failed, try the GDK_SCALE envvar...
        if (scale_factor <= 0.0) {
            const char *scale_str = SDL_getenv("GDK_SCALE");
            if (scale_str) {
                scale_factor = SDL_atoi(scale_str);
            }
        }

        // Nothing or a bad value, just fall back to 1.0
        if (scale_factor <= 0.0) {
            scale_factor = 1.0;
        }
    }

    return (float)scale_factor;
}

static bool get_visualinfo(Display *display, int screen, XVisualInfo *vinfo)
{
    const char *visual_id = SDL_GetHint(SDL_HINT_VIDEO_X11_VISUALID);
    int depth;

    // Look for an exact visual, if requested
    if (visual_id && *visual_id) {
        XVisualInfo *vi, template;
        int nvis;

        SDL_zero(template);
        template.visualid = SDL_strtol(visual_id, NULL, 0);
        vi = X11_XGetVisualInfo(display, VisualIDMask, &template, &nvis);
        if (vi) {
            *vinfo = *vi;
            X11_XFree(vi);
            return true;
        }
    }

    depth = DefaultDepth(display, screen);
    if ((X11_UseDirectColorVisuals() &&
         X11_XMatchVisualInfo(display, screen, depth, DirectColor, vinfo)) ||
        X11_XMatchVisualInfo(display, screen, depth, TrueColor, vinfo) ||
        X11_XMatchVisualInfo(display, screen, depth, PseudoColor, vinfo) ||
        X11_XMatchVisualInfo(display, screen, depth, StaticColor, vinfo)) {
        return true;
    }
    return false;
}

bool X11_GetVisualInfoFromVisual(Display *display, Visual *visual, XVisualInfo *vinfo)
{
    XVisualInfo *vi;
    int nvis;

    vinfo->visualid = X11_XVisualIDFromVisual(visual);
    vi = X11_XGetVisualInfo(display, VisualIDMask, vinfo, &nvis);
    if (vi) {
        *vinfo = *vi;
        X11_XFree(vi);
        return true;
    }
    return false;
}

SDL_PixelFormat X11_GetPixelFormatFromVisualInfo(Display *display, XVisualInfo *vinfo)
{
    if (vinfo->class == DirectColor || vinfo->class == TrueColor) {
        int bpp;
        Uint32 Rmask, Gmask, Bmask, Amask;

        Rmask = vinfo->visual->red_mask;
        Gmask = vinfo->visual->green_mask;
        Bmask = vinfo->visual->blue_mask;
        if (vinfo->depth == 32) {
            Amask = (0xFFFFFFFF & ~(Rmask | Gmask | Bmask));
        } else {
            Amask = 0;
        }

        bpp = vinfo->depth;
        if (bpp == 24) {
            int i, n;
            XPixmapFormatValues *p = X11_XListPixmapFormats(display, &n);
            if (p) {
                for (i = 0; i < n; ++i) {
                    if (p[i].depth == 24) {
                        bpp = p[i].bits_per_pixel;
                        break;
                    }
                }
                X11_XFree(p);
            }
        }

        return SDL_GetPixelFormatForMasks(bpp, Rmask, Gmask, Bmask, Amask);
    }

    if (vinfo->class == PseudoColor || vinfo->class == StaticColor) {
        switch (vinfo->depth) {
        case 8:
            return SDL_PIXELFORMAT_INDEX8;
        case 4:
            if (BitmapBitOrder(display) == LSBFirst) {
                return SDL_PIXELFORMAT_INDEX4LSB;
            } else {
                return SDL_PIXELFORMAT_INDEX4MSB;
            }
            // break; -Wunreachable-code-break
        case 1:
            if (BitmapBitOrder(display) == LSBFirst) {
                return SDL_PIXELFORMAT_INDEX1LSB;
            } else {
                return SDL_PIXELFORMAT_INDEX1MSB;
            }
            // break; -Wunreachable-code-break
        }
    }

    return SDL_PIXELFORMAT_UNKNOWN;
}

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
static bool CheckXRandR(Display *display, int *major, int *minor)
{
    // Default the extension not available
    *major = *minor = 0;

    // Allow environment override
#ifdef XRANDR_DISABLED_BY_DEFAULT
    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_X11_XRANDR, false)) {
#ifdef X11MODES_DEBUG
        printf("XRandR disabled by default due to window manager issues\n");
#endif
        return false;
    }
#else
    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_X11_XRANDR, true)) {
#ifdef X11MODES_DEBUG
        printf("XRandR disabled due to hint\n");
#endif
        return false;
    }
#endif // XRANDR_DISABLED_BY_DEFAULT

    if (!SDL_X11_HAVE_XRANDR) {
#ifdef X11MODES_DEBUG
        printf("XRandR support not available\n");
#endif
        return false;
    }

    // Query the extension version
    *major = 1;
    *minor = 3; // we want 1.3
    if (!X11_XRRQueryVersion(display, major, minor)) {
#ifdef X11MODES_DEBUG
        printf("XRandR not active on the display\n");
#endif
        *major = *minor = 0;
        return false;
    }
#ifdef X11MODES_DEBUG
    printf("XRandR available at version %d.%d!\n", *major, *minor);
#endif
    return true;
}

#define XRANDR_ROTATION_LEFT  (1 << 1)
#define XRANDR_ROTATION_RIGHT (1 << 3)

static void CalculateXRandRRefreshRate(const XRRModeInfo *info, int *numerator, int *denominator)
{
    unsigned int vTotal = info->vTotal;

    if (info->modeFlags & RR_DoubleScan) {
        // doublescan doubles the number of lines
        vTotal *= 2;
    }

    if (info->modeFlags & RR_Interlace) {
        // interlace splits the frame into two fields
        // the field rate is what is typically reported by monitors
        vTotal /= 2;
    }

    if (info->hTotal && vTotal) {
        *numerator = info->dotClock;
        *denominator = (info->hTotal * vTotal);
    } else {
        *numerator = 0;
        *denominator = 0;
    }
}

static bool SetXRandRModeInfo(Display *display, XRRScreenResources *res, RRCrtc crtc,
                                  RRMode modeID, SDL_DisplayMode *mode)
{
    int i;
    for (i = 0; i < res->nmode; ++i) {
        const XRRModeInfo *info = &res->modes[i];
        if (info->id == modeID) {
            XRRCrtcInfo *crtcinfo;
            Rotation rotation = 0;
            XFixed scale_w = 0x10000, scale_h = 0x10000;
            XRRCrtcTransformAttributes *attr;

            crtcinfo = X11_XRRGetCrtcInfo(display, res, crtc);
            if (crtcinfo) {
                rotation = crtcinfo->rotation;
                X11_XRRFreeCrtcInfo(crtcinfo);
            }
            if (X11_XRRGetCrtcTransform(display, crtc, &attr) && attr) {
                scale_w = attr->currentTransform.matrix[0][0];
                scale_h = attr->currentTransform.matrix[1][1];
                X11_XFree(attr);
            }

            if (rotation & (XRANDR_ROTATION_LEFT | XRANDR_ROTATION_RIGHT)) {
                mode->w = (info->height * scale_w + 0xffff) >> 16;
                mode->h = (info->width * scale_h + 0xffff) >> 16;
            } else {
                mode->w = (info->width * scale_w + 0xffff) >> 16;
                mode->h = (info->height * scale_h + 0xffff) >> 16;
            }
            CalculateXRandRRefreshRate(info, &mode->refresh_rate_numerator, &mode->refresh_rate_denominator);
            mode->internal->xrandr_mode = modeID;
#ifdef X11MODES_DEBUG
            printf("XRandR mode %d: %dx%d@%d/%dHz\n", (int)modeID,
                   mode->screen_w, mode->screen_h, mode->refresh_rate_numerator, mode->refresh_rate_denominator);
#endif
            return true;
        }
    }
    return false;
}

static void SetXRandRDisplayName(Display *dpy, Atom EDID, char *name, const size_t namelen, RROutput output, const unsigned long widthmm, const unsigned long heightmm)
{
    // See if we can get the EDID data for the real monitor name
    int inches;
    int nprop;
    Atom *props = X11_XRRListOutputProperties(dpy, output, &nprop);
    int i;

    for (i = 0; i < nprop; ++i) {
        unsigned char *prop;
        int actual_format;
        unsigned long nitems, bytes_after;
        Atom actual_type;

        if (props[i] == EDID) {
            if (X11_XRRGetOutputProperty(dpy, output, props[i], 0, 100, False,
                                         False, AnyPropertyType, &actual_type,
                                         &actual_format, &nitems, &bytes_after,
                                         &prop) == Success) {
                MonitorInfo *info = decode_edid(prop);
                if (info) {
#ifdef X11MODES_DEBUG
                    printf("Found EDID data for %s\n", name);
                    dump_monitor_info(info);
#endif
                    SDL_strlcpy(name, info->dsc_product_name, namelen);
                    SDL_free(info);
                }
                X11_XFree(prop);
            }
            break;
        }
    }

    if (props) {
        X11_XFree(props);
    }

    inches = (int)((SDL_sqrtf(widthmm * widthmm + heightmm * heightmm) / 25.4f) + 0.5f);
    if (*name && inches) {
        const size_t len = SDL_strlen(name);
        (void)SDL_snprintf(&name[len], namelen - len, " %d\"", inches);
    }

#ifdef X11MODES_DEBUG
    printf("Display name: %s\n", name);
#endif
}

static bool X11_FillXRandRDisplayInfo(SDL_VideoDevice *_this, Display *dpy, int screen, RROutput outputid, XRRScreenResources *res, SDL_VideoDisplay *display, char *display_name, size_t display_name_size)
{
    Atom EDID = X11_XInternAtom(dpy, "EDID", False);
    XRROutputInfo *output_info;
    int display_x, display_y;
    unsigned long display_mm_width, display_mm_height;
    SDL_DisplayData *displaydata;
    SDL_DisplayMode mode;
    SDL_DisplayModeData *modedata;
    RRMode modeID;
    RRCrtc output_crtc;
    XRRCrtcInfo *crtc;
    XVisualInfo vinfo;
    Uint32 pixelformat;
    XPixmapFormatValues *pixmapformats;
    int scanline_pad;
    int i, n;

    if (!display || !display_name) {
        return false; // invalid parameters
    }

    if (!get_visualinfo(dpy, screen, &vinfo)) {
        return false; // uh, skip this screen?
    }

    pixelformat = X11_GetPixelFormatFromVisualInfo(dpy, &vinfo);
    if (SDL_ISPIXELFORMAT_INDEXED(pixelformat)) {
        return false; // Palettized video modes are no longer supported, ignore this one.
    }

    scanline_pad = SDL_BYTESPERPIXEL(pixelformat) * 8;
    pixmapformats = X11_XListPixmapFormats(dpy, &n);
    if (pixmapformats) {
        for (i = 0; i < n; i++) {
            if (pixmapformats[i].depth == vinfo.depth) {
                scanline_pad = pixmapformats[i].scanline_pad;
                break;
            }
        }
        X11_XFree(pixmapformats);
    }

    output_info = X11_XRRGetOutputInfo(dpy, res, outputid);
    if (!output_info || !output_info->crtc || output_info->connection == RR_Disconnected) {
        X11_XRRFreeOutputInfo(output_info);
        return false; // ignore this one.
    }

    SDL_strlcpy(display_name, output_info->name, display_name_size);
    display_mm_width = output_info->mm_width;
    display_mm_height = output_info->mm_height;
    output_crtc = output_info->crtc;
    X11_XRRFreeOutputInfo(output_info);

    crtc = X11_XRRGetCrtcInfo(dpy, res, output_crtc);
    if (!crtc) {
        return false; // oh well, ignore it.
    }

    SDL_zero(mode);
    modeID = crtc->mode;
    mode.w = crtc->width;
    mode.h = crtc->height;
    mode.format = pixelformat;

    display_x = crtc->x;
    display_y = crtc->y;

    X11_XRRFreeCrtcInfo(crtc);

    displaydata = (SDL_DisplayData *)SDL_calloc(1, sizeof(*displaydata));
    if (!displaydata) {
        return false;
    }

    modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
    if (!modedata) {
        SDL_free(displaydata);
        return false;
    }

    modedata->xrandr_mode = modeID;
    mode.internal = modedata;

    displaydata->screen = screen;
    displaydata->visual = vinfo.visual;
    displaydata->depth = vinfo.depth;
    displaydata->scanline_pad = scanline_pad;
    displaydata->x = display_x;
    displaydata->y = display_y;
    displaydata->use_xrandr = true;
    displaydata->xrandr_output = outputid;
    SDL_strlcpy(displaydata->connector_name, display_name, sizeof(displaydata->connector_name));

    SetXRandRModeInfo(dpy, res, output_crtc, modeID, &mode);
    SetXRandRDisplayName(dpy, EDID, display_name, display_name_size, outputid, display_mm_width, display_mm_height);

    SDL_zero(*display);
    if (*display_name) {
        display->name = display_name;
    }
    display->desktop_mode = mode;
    display->content_scale = GetGlobalContentScale(_this);
    display->internal = displaydata;

    return true;
}

static bool X11_AddXRandRDisplay(SDL_VideoDevice *_this, Display *dpy, int screen, RROutput outputid, XRRScreenResources *res, bool send_event)
{
    SDL_VideoDisplay display;
    char display_name[128];

    if (!X11_FillXRandRDisplayInfo(_this, dpy, screen, outputid, res, &display, display_name, sizeof(display_name))) {
        return true; // failed to query data, skip this display
    }

    if (SDL_AddVideoDisplay(&display, send_event) == 0) {
        return false;
    }

    return true;
}


static bool X11_UpdateXRandRDisplay(SDL_VideoDevice *_this, Display *dpy, int screen, RROutput outputid, XRRScreenResources *res, SDL_VideoDisplay *existing_display)
{
    SDL_VideoDisplay display;
    char display_name[128];

    if (!X11_FillXRandRDisplayInfo(_this, dpy, screen, outputid, res, &display, display_name, sizeof(display_name))) {
        return false; // failed to query current display state
    }

    // update mode - this call takes ownership of display.desktop_mode.internal
    SDL_SetDesktopDisplayMode(existing_display, &display.desktop_mode);

    // update bounds
    if (existing_display->internal->x != display.internal->x ||
        existing_display->internal->y != display.internal->y) {
        existing_display->internal->x = display.internal->x;
        existing_display->internal->y = display.internal->y;
        SDL_SendDisplayEvent(existing_display, SDL_EVENT_DISPLAY_MOVED, 0, 0);
    }

    // update scale
    SDL_SetDisplayContentScale(existing_display, display.content_scale);

    // SDL_DisplayData is updated piece-meal above, free our local copy of this data
    SDL_free( display.internal );

    return true;
}

static XRRScreenResources *X11_GetScreenResources(Display *dpy, int screen)
{
    XRRScreenResources *res = X11_XRRGetScreenResourcesCurrent(dpy, RootWindow(dpy, screen));
    if (!res || res->noutput == 0) {
        if (res) {
            X11_XRRFreeScreenResources(res);
        }
        res = X11_XRRGetScreenResources(dpy, RootWindow(dpy, screen));
    }
    return res;
}

static void X11_CheckDisplaysMoved(SDL_VideoDevice *_this, Display *dpy)
{
    const int screencount = ScreenCount(dpy);

    SDL_DisplayID *displays = SDL_GetDisplays(NULL);
    if (!displays) {
        return;
    }

    for (int screen = 0; screen < screencount; ++screen) {
        XRRScreenResources *res = X11_GetScreenResources(dpy, screen);
        if (!res) {
            continue;
        }

        for (int i = 0; displays[i]; ++i) {
            SDL_VideoDisplay *display = SDL_GetVideoDisplay(displays[i]);
            const SDL_DisplayData *displaydata = display->internal;
            if (displaydata->screen == screen) {
                X11_UpdateXRandRDisplay(_this, dpy, screen, displaydata->xrandr_output, res, display);
            }
        }
        X11_XRRFreeScreenResources(res);
    }
    SDL_free(displays);
}

static void X11_CheckDisplaysRemoved(SDL_VideoDevice *_this, Display *dpy)
{
    const int screencount = ScreenCount(dpy);
    int num_displays = 0;

    SDL_DisplayID *displays = SDL_GetDisplays(&num_displays);
    if (!displays) {
        return;
    }

    for (int screen = 0; screen < screencount; ++screen) {
        XRRScreenResources *res = X11_GetScreenResources(dpy, screen);
        if (!res) {
            continue;
        }

        for (int output = 0; output < res->noutput; output++) {
            for (int i = 0; i < num_displays; ++i) {
                if (!displays[i]) {
                    // We already removed this display from the list
                    continue;
                }

                SDL_VideoDisplay *display = SDL_GetVideoDisplay(displays[i]);
                const SDL_DisplayData *displaydata = display->internal;
                if (displaydata->xrandr_output == res->outputs[output]) {
                    // This display is active, remove it from the list
                    displays[i] = 0;
                    break;
                }
            }
        }
        X11_XRRFreeScreenResources(res);
    }

    for (int i = 0; i < num_displays; ++i) {
        if (displays[i]) {
            // This display wasn't in the XRandR list
            SDL_DelVideoDisplay(displays[i], true);
        }
    }
    SDL_free(displays);
}

static void X11_HandleXRandROutputChange(SDL_VideoDevice *_this, const XRROutputChangeNotifyEvent *ev)
{
    SDL_DisplayID *displays;
    SDL_VideoDisplay *display = NULL;
    int i;

#if 0
    printf("XRROutputChangeNotifyEvent! [output=%u, crtc=%u, mode=%u, rotation=%u, connection=%u]\n", (unsigned int) ev->output, (unsigned int) ev->crtc, (unsigned int) ev->mode, (unsigned int) ev->rotation, (unsigned int) ev->connection);
#endif

    // XWayland doesn't always send output disconnected events
    X11_CheckDisplaysRemoved(_this, ev->display);

    displays = SDL_GetDisplays(NULL);
    if (displays) {
        for (i = 0; displays[i]; ++i) {
            SDL_VideoDisplay *thisdisplay = SDL_GetVideoDisplay(displays[i]);
            const SDL_DisplayData *displaydata = thisdisplay->internal;
            if (displaydata->xrandr_output == ev->output) {
                display = thisdisplay;
                break;
            }
        }
        SDL_free(displays);
    }

    if (ev->connection == RR_Disconnected) { // output is going away
        if (display) {
            SDL_DelVideoDisplay(display->id, true);
        }
        X11_CheckDisplaysMoved(_this, ev->display);

    } else if (ev->connection == RR_Connected) { // output is coming online
        if (!display) {
            Display *dpy = ev->display;
            const int screen = DefaultScreen(dpy);
            XRRScreenResources *res = X11_GetScreenResources(dpy, screen);
            if (res) {
                X11_AddXRandRDisplay(_this, dpy, screen, ev->output, res, true);
                X11_XRRFreeScreenResources(res);
            }
        }
        X11_CheckDisplaysMoved(_this, ev->display);
    }
}

void X11_HandleXRandREvent(SDL_VideoDevice *_this, const XEvent *xevent)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_assert(xevent->type == (videodata->xrandr_event_base + RRNotify));

    switch (((const XRRNotifyEvent *)xevent)->subtype) {
    case RRNotify_OutputChange:
        X11_HandleXRandROutputChange(_this, (const XRROutputChangeNotifyEvent *)xevent);
        break;
    default:
        break;
    }
}

static void X11_SortOutputsByPriorityHint(SDL_VideoDevice *_this)
{
    const char *name_hint = SDL_GetHint(SDL_HINT_VIDEO_DISPLAY_PRIORITY);

    if (name_hint) {
        char *saveptr;
        char *str = SDL_strdup(name_hint);
        SDL_VideoDisplay **sorted_list = SDL_malloc(sizeof(SDL_VideoDisplay *) * _this->num_displays);

        if (str && sorted_list) {
            int sorted_index = 0;

            // Sort the requested displays to the front of the list.
            const char *token = SDL_strtok_r(str, ",", &saveptr);
            while (token) {
                for (int i = 0; i < _this->num_displays; ++i) {
                    SDL_VideoDisplay *d = _this->displays[i];
                    if (d) {
                        SDL_DisplayData *data = d->internal;
                        if (SDL_strcmp(token, data->connector_name) == 0) {
                            sorted_list[sorted_index++] = d;
                            _this->displays[i] = NULL;
                            break;
                        }
                    }
                }

                token = SDL_strtok_r(NULL, ",", &saveptr);
            }

            // Append the remaining displays to the end of the list.
            for (int i = 0; i < _this->num_displays; ++i) {
                if (_this->displays[i]) {
                    sorted_list[sorted_index++] = _this->displays[i];
                }
            }

            // Copy the sorted list back to the display list.
            SDL_memcpy(_this->displays, sorted_list, sizeof(SDL_VideoDisplay *) * _this->num_displays);
        }

        SDL_free(str);
        SDL_free(sorted_list);
    }
}

static bool X11_InitModes_XRandR(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    Display *dpy = data->display;
    const int screencount = ScreenCount(dpy);
    const int default_screen = DefaultScreen(dpy);
    RROutput primary = X11_XRRGetOutputPrimary(dpy, RootWindow(dpy, default_screen));
    int xrandr_error_base = 0;
    int looking_for_primary;
    int output;
    int screen;

    if (!X11_XRRQueryExtension(dpy, &data->xrandr_event_base, &xrandr_error_base)) {
        return SDL_SetError("XRRQueryExtension failed");
    }

    for (looking_for_primary = 1; looking_for_primary >= 0; looking_for_primary--) {
        for (screen = 0; screen < screencount; screen++) {

            // we want the primary output first, and then skipped later.
            if (looking_for_primary && (screen != default_screen)) {
                continue;
            }

            XRRScreenResources *res = X11_GetScreenResources(dpy, screen);
            if (!res) {
                continue;
            }

            for (output = 0; output < res->noutput; output++) {
                // The primary output _should_ always be sorted first, but just in case...
                if ((looking_for_primary && (res->outputs[output] != primary)) ||
                    (!looking_for_primary && (screen == default_screen) && (res->outputs[output] == primary))) {
                    continue;
                }
                if (!X11_AddXRandRDisplay(_this, dpy, screen, res->outputs[output], res, false)) {
                    break;
                }
            }

            X11_XRRFreeScreenResources(res);

            // This will generate events for displays that come and go at runtime.
            X11_XRRSelectInput(dpy, RootWindow(dpy, screen), RROutputChangeNotifyMask);
        }
    }

    if (_this->num_displays == 0) {
        return SDL_SetError("No available displays");
    }

    X11_SortOutputsByPriorityHint(_this);

    return true;
}
#endif // SDL_VIDEO_DRIVER_X11_XRANDR

/* This is used if there's no better functionality--like XRandR--to use.
   It won't attempt to supply different display modes at all, but it can
   enumerate the current displays and their current sizes. */
static bool X11_InitModes_StdXlib(SDL_VideoDevice *_this)
{
    // !!! FIXME: a lot of copy/paste from X11_InitModes_XRandR in this function.
    SDL_VideoData *data = _this->internal;
    Display *dpy = data->display;
    const int default_screen = DefaultScreen(dpy);
    Screen *screen = ScreenOfDisplay(dpy, default_screen);
    int scanline_pad, n, i;
    SDL_DisplayModeData *modedata;
    SDL_DisplayData *displaydata;
    SDL_DisplayMode mode;
    XPixmapFormatValues *pixmapformats;
    Uint32 pixelformat;
    XVisualInfo vinfo;
    SDL_VideoDisplay display;

    // note that generally even if you have a multiple physical monitors, ScreenCount(dpy) still only reports ONE screen.

    if (!get_visualinfo(dpy, default_screen, &vinfo)) {
        return SDL_SetError("Failed to find an X11 visual for the primary display");
    }

    pixelformat = X11_GetPixelFormatFromVisualInfo(dpy, &vinfo);
    if (SDL_ISPIXELFORMAT_INDEXED(pixelformat)) {
        return SDL_SetError("Palettized video modes are no longer supported");
    }

    SDL_zero(mode);
    mode.w = WidthOfScreen(screen);
    mode.h = HeightOfScreen(screen);
    mode.format = pixelformat;

    displaydata = (SDL_DisplayData *)SDL_calloc(1, sizeof(*displaydata));
    if (!displaydata) {
        return false;
    }

    modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
    if (!modedata) {
        SDL_free(displaydata);
        return false;
    }
    mode.internal = modedata;

    displaydata->screen = default_screen;
    displaydata->visual = vinfo.visual;
    displaydata->depth = vinfo.depth;

    scanline_pad = SDL_BYTESPERPIXEL(pixelformat) * 8;
    pixmapformats = X11_XListPixmapFormats(dpy, &n);
    if (pixmapformats) {
        for (i = 0; i < n; ++i) {
            if (pixmapformats[i].depth == vinfo.depth) {
                scanline_pad = pixmapformats[i].scanline_pad;
                break;
            }
        }
        X11_XFree(pixmapformats);
    }

    displaydata->scanline_pad = scanline_pad;
    displaydata->x = 0;
    displaydata->y = 0;
    displaydata->use_xrandr = false;

    SDL_zero(display);
    display.name = (char *)"Generic X11 Display"; /* this is just copied and thrown away, it's safe to cast to char* here. */
    display.desktop_mode = mode;
    display.internal = displaydata;
    display.content_scale = GetGlobalContentScale(_this);
    if (SDL_AddVideoDisplay(&display, true) == 0) {
        return false;
    }
    return true;
}

bool X11_InitModes(SDL_VideoDevice *_this)
{
    /* XRandR is the One True Modern Way to do this on X11. If this
       fails, we just won't report any display modes except the current
       desktop size. */
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    {
        SDL_VideoData *data = _this->internal;
        int xrandr_major, xrandr_minor;
        // require at least XRandR v1.3
        if (CheckXRandR(data->display, &xrandr_major, &xrandr_minor) &&
            (xrandr_major >= 2 || (xrandr_major == 1 && xrandr_minor >= 3)) &&
            X11_InitModes_XRandR(_this)) {
            return true;
        }
    }
#endif // SDL_VIDEO_DRIVER_X11_XRANDR

    // still here? Just set up an extremely basic display.
    return X11_InitModes_StdXlib(_this);
}

bool X11_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *sdl_display)
{
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    SDL_DisplayData *data = sdl_display->internal;
    SDL_DisplayMode mode;

    /* Unfortunately X11 requires the window to be created with the correct
     * visual and depth ahead of time, but the SDL API allows you to create
     * a window before setting the fullscreen display mode.  This means that
     * we have to use the same format for all windows and all display modes.
     * (or support recreating the window with a new visual behind the scenes)
     */
    SDL_zero(mode);
    mode.format = sdl_display->desktop_mode.format;

    if (data->use_xrandr) {
        Display *display = _this->internal->display;
        XRRScreenResources *res;

        res = X11_XRRGetScreenResources(display, RootWindow(display, data->screen));
        if (res) {
            SDL_DisplayModeData *modedata;
            XRROutputInfo *output_info;
            int i;

            output_info = X11_XRRGetOutputInfo(display, res, data->xrandr_output);
            if (output_info && output_info->connection != RR_Disconnected) {
                for (i = 0; i < output_info->nmode; ++i) {
                    modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
                    if (!modedata) {
                        continue;
                    }
                    mode.internal = modedata;

                    if (!SetXRandRModeInfo(display, res, output_info->crtc, output_info->modes[i], &mode) ||
                        !SDL_AddFullscreenDisplayMode(sdl_display, &mode)) {
                        SDL_free(modedata);
                    }
                }
            }
            X11_XRRFreeOutputInfo(output_info);
            X11_XRRFreeScreenResources(res);
        }
    }
#endif // SDL_VIDEO_DRIVER_X11_XRANDR
    return true;
}

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
// This catches an error from XRRSetScreenSize, as a workaround for now.
// !!! FIXME: remove this later when we have a better solution.
static int (*PreXRRSetScreenSizeErrorHandler)(Display *, XErrorEvent *) = NULL;
static int SDL_XRRSetScreenSizeErrHandler(Display *d, XErrorEvent *e)
{
    // BadMatch: https://github.com/libsdl-org/SDL/issues/4561
    // BadValue: https://github.com/libsdl-org/SDL/issues/4840
    if ((e->error_code == BadMatch) || (e->error_code == BadValue)) {
        return 0;
    }

    return PreXRRSetScreenSizeErrorHandler(d, e);
}
#endif

bool X11_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *sdl_display, SDL_DisplayMode *mode)
{
    SDL_VideoData *viddata = _this->internal;
    SDL_DisplayData *data = sdl_display->internal;

    viddata->last_mode_change_deadline = SDL_GetTicks() + (PENDING_FOCUS_TIME * 2);

    // XWayland mode switches are emulated with viewports and thus instantaneous.
    if (!viddata->is_xwayland) {
        if (sdl_display->current_mode != mode) {
            data->mode_switch_deadline_ns = SDL_GetTicksNS() + MODE_SWITCH_TIMEOUT_NS;
        } else {
            data->mode_switch_deadline_ns = 0;
        }
    }

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    if (data->use_xrandr) {
        Display *display = viddata->display;
        SDL_DisplayModeData *modedata = mode->internal;
        int mm_width, mm_height;
        XRRScreenResources *res;
        XRROutputInfo *output_info;
        XRRCrtcInfo *crtc;
        Status status;

        res = X11_XRRGetScreenResources(display, RootWindow(display, data->screen));
        if (!res) {
            return SDL_SetError("Couldn't get XRandR screen resources");
        }

        output_info = X11_XRRGetOutputInfo(display, res, data->xrandr_output);
        if (!output_info || output_info->connection == RR_Disconnected) {
            X11_XRRFreeScreenResources(res);
            return SDL_SetError("Couldn't get XRandR output info");
        }

        crtc = X11_XRRGetCrtcInfo(display, res, output_info->crtc);
        if (!crtc) {
            X11_XRRFreeOutputInfo(output_info);
            X11_XRRFreeScreenResources(res);
            return SDL_SetError("Couldn't get XRandR crtc info");
        }

        if (crtc->mode == modedata->xrandr_mode) {
#ifdef X11MODES_DEBUG
            printf("already in desired mode 0x%lx (%ux%u), nothing to do\n",
                   crtc->mode, crtc->width, crtc->height);
#endif
            status = Success;
            goto freeInfo;
        }

        X11_XGrabServer(display);
        status = X11_XRRSetCrtcConfig(display, res, output_info->crtc, CurrentTime,
                                      0, 0, None, crtc->rotation, NULL, 0);
        if (status != Success) {
            goto ungrabServer;
        }

        mm_width = mode->w * DisplayWidthMM(display, data->screen) / DisplayWidth(display, data->screen);
        mm_height = mode->h * DisplayHeightMM(display, data->screen) / DisplayHeight(display, data->screen);

        /* !!! FIXME: this can get into a problem scenario when a window is
           bigger than a physical monitor in a configuration where one screen
           spans multiple physical monitors. A detailed reproduction case is
           discussed at https://github.com/libsdl-org/SDL/issues/4561 ...
           for now we cheat and just catch the X11 error and carry on, which
           is likely to cause subtle issues but is better than outright
           crashing */
        X11_XSync(display, False);
        PreXRRSetScreenSizeErrorHandler = X11_XSetErrorHandler(SDL_XRRSetScreenSizeErrHandler);
        X11_XRRSetScreenSize(display, RootWindow(display, data->screen),
                             mode->w, mode->h, mm_width, mm_height);
        X11_XSync(display, False);
        X11_XSetErrorHandler(PreXRRSetScreenSizeErrorHandler);

        status = X11_XRRSetCrtcConfig(display, res, output_info->crtc, CurrentTime,
                                      crtc->x, crtc->y, modedata->xrandr_mode, crtc->rotation,
                                      &data->xrandr_output, 1);

    ungrabServer:
        X11_XUngrabServer(display);
    freeInfo:
        X11_XRRFreeCrtcInfo(crtc);
        X11_XRRFreeOutputInfo(output_info);
        X11_XRRFreeScreenResources(res);

        if (status != Success) {
            return SDL_SetError("X11_XRRSetCrtcConfig failed");
        }
    }
#else
    (void)data;
#endif // SDL_VIDEO_DRIVER_X11_XRANDR

    return true;
}

void X11_QuitModes(SDL_VideoDevice *_this)
{
}

bool X11_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *sdl_display, SDL_Rect *rect)
{
    SDL_DisplayData *data = sdl_display->internal;

    rect->x = data->x;
    rect->y = data->y;
    rect->w = sdl_display->current_mode->w;
    rect->h = sdl_display->current_mode->h;
    return true;
}

bool X11_GetDisplayUsableBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *sdl_display, SDL_Rect *rect)
{
    SDL_VideoData *data = _this->internal;
    Display *display = data->display;
    Atom _NET_WORKAREA;
    int real_format;
    Atom real_type;
    unsigned long items_read = 0, items_left = 0;
    unsigned char *propdata = NULL;
    bool result = false;

    if (!X11_GetDisplayBounds(_this, sdl_display, rect)) {
        return false;
    }

    _NET_WORKAREA = X11_XInternAtom(display, "_NET_WORKAREA", False);
    int status = X11_XGetWindowProperty(display, DefaultRootWindow(display),
                                    _NET_WORKAREA, 0L, 4L, False, XA_CARDINAL,
                                    &real_type, &real_format, &items_read,
                                    &items_left, &propdata);
    if ((status == Success) && (items_read >= 4)) {
        const long *p = (long *)propdata;
        const SDL_Rect usable = { (int)p[0], (int)p[1], (int)p[2], (int)p[3] };
        result = true;
        if (!SDL_GetRectIntersection(rect, &usable, rect)) {
            SDL_zerop(rect);
        }
    }

    if (propdata) {
        X11_XFree(propdata);
    }

    return result;
}

#endif // SDL_VIDEO_DRIVER_X11
