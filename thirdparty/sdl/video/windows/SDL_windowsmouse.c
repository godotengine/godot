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

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

#include "SDL_windowsvideo.h"
#include "SDL_windowsevents.h"
#include "SDL_windowsrawinput.h"

#include "../SDL_video_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../joystick/usb_ids.h"
#include "../../core/windows/SDL_windows.h" // for checking windows version


typedef struct CachedCursor
{
    float scale;
    HCURSOR cursor;
    struct CachedCursor *next;
} CachedCursor;

struct SDL_CursorData
{
    SDL_Surface *surface;
    int hot_x;
    int hot_y;
    CachedCursor *cache;
    HCURSOR cursor;
};

typedef struct
{
    Uint64 xs[5];
    Uint64 ys[5];
    Sint64 residual[2];
    Uint32 dpiscale;
    Uint32 dpidenom;
    int last_node;
    bool enhanced;
    bool dpiaware;
} WIN_MouseData;

DWORD SDL_last_warp_time = 0;
HCURSOR SDL_cursor = NULL;
static SDL_Cursor *SDL_blank_cursor = NULL;
static WIN_MouseData WIN_system_scale_data;

static SDL_Cursor *WIN_CreateCursorAndData(HCURSOR hcursor)
{
    if (!hcursor) {
        return NULL;
    }

    SDL_Cursor *cursor = (SDL_Cursor *)SDL_calloc(1, sizeof(*cursor));
    if (!cursor) {
        return NULL;
    }

    SDL_CursorData *data = (SDL_CursorData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        SDL_free(cursor);
        return NULL;
    }

    data->cursor = hcursor;
    cursor->internal = data;
    return cursor;
}


static bool IsMonochromeSurface(SDL_Surface *surface)
{
    int x, y;
    Uint8 r, g, b, a;

    SDL_assert(surface->format == SDL_PIXELFORMAT_ARGB8888);

    for (y = 0; y < surface->h; y++) {
        for (x = 0; x < surface->w; x++) {
            SDL_ReadSurfacePixel(surface, x, y, &r, &g, &b, &a);

            // Black or white pixel.
            if (!((r == 0x00 && g == 0x00 && b == 0x00) || (r == 0xff && g == 0xff && b == 0xff))) {
                return false;
            }

            // Transparent or opaque pixel.
            if (!(a == 0x00 || a == 0xff)) {
                return false;
            }
        }
    }

    return true;
}

static HBITMAP CreateColorBitmap(SDL_Surface *surface)
{
    HBITMAP bitmap;
    BITMAPINFO bi;
    void *pixels;

    SDL_assert(surface->format == SDL_PIXELFORMAT_ARGB8888);

    SDL_zero(bi);
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = surface->w;
    bi.bmiHeader.biHeight = -surface->h; // Invert height to make the top-down DIB.
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    bitmap = CreateDIBSection(NULL, &bi, DIB_RGB_COLORS, &pixels, NULL, 0);
    if (!bitmap || !pixels) {
        WIN_SetError("CreateDIBSection()");
        if (bitmap) {
            DeleteObject(bitmap);
        }
        return NULL;
    }

    SDL_memcpy(pixels, surface->pixels, surface->pitch * surface->h);

    return bitmap;
}

/* Generate bitmap with a mask and optional monochrome image data.
 *
 * For info on the expected mask format see:
 * https://devblogs.microsoft.com/oldnewthing/20101018-00/?p=12513
 */
static HBITMAP CreateMaskBitmap(SDL_Surface *surface, bool is_monochrome)
{
    HBITMAP bitmap;
    bool isstack;
    void *pixels;
    int x, y;
    Uint8 r, g, b, a;
    Uint8 *dst;
    const int pitch = ((surface->w + 15) & ~15) / 8;
    const int size = pitch * surface->h;
    static const unsigned char masks[] = { 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1 };

    SDL_assert(surface->format == SDL_PIXELFORMAT_ARGB8888);

    pixels = SDL_small_alloc(Uint8, size * (is_monochrome ? 2 : 1), &isstack);
    if (!pixels) {
        SDL_OutOfMemory();
        return NULL;
    }

    dst = (Uint8 *)pixels;

    // Make the mask completely transparent.
    SDL_memset(dst, 0xff, size);
    if (is_monochrome) {
        SDL_memset(dst + size, 0x00, size);
    }

    for (y = 0; y < surface->h; y++, dst += pitch) {
        for (x = 0; x < surface->w; x++) {
            SDL_ReadSurfacePixel(surface, x, y, &r, &g, &b, &a);

            if (a != 0) {
                // Reset bit of an opaque pixel.
                dst[x >> 3] &= ~masks[x & 7];
            }

            if (is_monochrome && !(r == 0x00 && g == 0x00 && b == 0x00)) {
                // Set bit of white or inverted pixel.
                dst[size + (x >> 3)] |= masks[x & 7];
            }
        }
    }

    bitmap = CreateBitmap(surface->w, surface->h * (is_monochrome ? 2 : 1), 1, 1, pixels);
    SDL_small_free(pixels, isstack);
    if (!bitmap) {
        WIN_SetError("CreateBitmap()");
        return NULL;
    }

    return bitmap;
}

static HCURSOR WIN_CreateHCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    HCURSOR hcursor;
    ICONINFO ii;
    bool is_monochrome = IsMonochromeSurface(surface);

    SDL_zero(ii);
    ii.fIcon = FALSE;
    ii.xHotspot = (DWORD)hot_x;
    ii.yHotspot = (DWORD)hot_y;
    ii.hbmMask = CreateMaskBitmap(surface, is_monochrome);
    ii.hbmColor = is_monochrome ? NULL : CreateColorBitmap(surface);

    if (!ii.hbmMask || (!is_monochrome && !ii.hbmColor)) {
        SDL_SetError("Couldn't create cursor bitmaps");
        if (ii.hbmMask) {
            DeleteObject(ii.hbmMask);
        }
        if (ii.hbmColor) {
            DeleteObject(ii.hbmColor);
        }
        return NULL;
    }

    hcursor = CreateIconIndirect(&ii);
    if (!hcursor) {
        WIN_SetError("CreateIconIndirect()");
        DeleteObject(ii.hbmMask);
        if (ii.hbmColor) {
            DeleteObject(ii.hbmColor);
        }
        return NULL;
    }

    DeleteObject(ii.hbmMask);
    if (ii.hbmColor) {
        DeleteObject(ii.hbmColor);
    }

    return hcursor;
}

static SDL_Cursor *WIN_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    if (!SDL_SurfaceHasAlternateImages(surface)) {
        HCURSOR hcursor = WIN_CreateHCursor(surface, hot_x, hot_y);
        if (!hcursor) {
            return NULL;
        }
        return WIN_CreateCursorAndData(hcursor);
    }

    // Dynamically generate cursors at the appropriate DPI
    SDL_Cursor *cursor = (SDL_Cursor *)SDL_calloc(1, sizeof(*cursor));
    if (cursor) {
        SDL_CursorData *data = (SDL_CursorData *)SDL_calloc(1, sizeof(*data));
        if (!data) {
            SDL_free(cursor);
            return NULL;
        }
        data->hot_x = hot_x;
        data->hot_y = hot_y;
        data->surface = surface;
        ++surface->refcount;
        cursor->internal = data;
    }
    return cursor;
}

static SDL_Cursor *WIN_CreateBlankCursor(void)
{
    SDL_Cursor *cursor = NULL;
    SDL_Surface *surface = SDL_CreateSurface(32, 32, SDL_PIXELFORMAT_ARGB8888);
    if (surface) {
        cursor = WIN_CreateCursor(surface, 0, 0);
        SDL_DestroySurface(surface);
    }
    return cursor;
}

static SDL_Cursor *WIN_CreateSystemCursor(SDL_SystemCursor id)
{
    LPCTSTR name;

    switch (id) {
    default:
        SDL_assert(!"Unknown system cursor ID");
        return NULL;
    case SDL_SYSTEM_CURSOR_DEFAULT:
        name = IDC_ARROW;
        break;
    case SDL_SYSTEM_CURSOR_TEXT:
        name = IDC_IBEAM;
        break;
    case SDL_SYSTEM_CURSOR_WAIT:
        name = IDC_WAIT;
        break;
    case SDL_SYSTEM_CURSOR_CROSSHAIR:
        name = IDC_CROSS;
        break;
    case SDL_SYSTEM_CURSOR_PROGRESS:
        name = IDC_APPSTARTING;
        break;
    case SDL_SYSTEM_CURSOR_NWSE_RESIZE:
        name = IDC_SIZENWSE;
        break;
    case SDL_SYSTEM_CURSOR_NESW_RESIZE:
        name = IDC_SIZENESW;
        break;
    case SDL_SYSTEM_CURSOR_EW_RESIZE:
        name = IDC_SIZEWE;
        break;
    case SDL_SYSTEM_CURSOR_NS_RESIZE:
        name = IDC_SIZENS;
        break;
    case SDL_SYSTEM_CURSOR_MOVE:
        name = IDC_SIZEALL;
        break;
    case SDL_SYSTEM_CURSOR_NOT_ALLOWED:
        name = IDC_NO;
        break;
    case SDL_SYSTEM_CURSOR_POINTER:
        name = IDC_HAND;
        break;
    case SDL_SYSTEM_CURSOR_NW_RESIZE:
        name = IDC_SIZENWSE;
        break;
    case SDL_SYSTEM_CURSOR_N_RESIZE:
        name = IDC_SIZENS;
        break;
    case SDL_SYSTEM_CURSOR_NE_RESIZE:
        name = IDC_SIZENESW;
        break;
    case SDL_SYSTEM_CURSOR_E_RESIZE:
        name = IDC_SIZEWE;
        break;
    case SDL_SYSTEM_CURSOR_SE_RESIZE:
        name = IDC_SIZENWSE;
        break;
    case SDL_SYSTEM_CURSOR_S_RESIZE:
        name = IDC_SIZENS;
        break;
    case SDL_SYSTEM_CURSOR_SW_RESIZE:
        name = IDC_SIZENESW;
        break;
    case SDL_SYSTEM_CURSOR_W_RESIZE:
        name = IDC_SIZEWE;
        break;
    }
    return WIN_CreateCursorAndData(LoadCursor(NULL, name));
}

static SDL_Cursor *WIN_CreateDefaultCursor(void)
{
    SDL_SystemCursor id = SDL_GetDefaultSystemCursor();
    return WIN_CreateSystemCursor(id);
}

static void WIN_FreeCursor(SDL_Cursor *cursor)
{
    SDL_CursorData *data = cursor->internal;

    if (data->surface) {
        SDL_DestroySurface(data->surface);
    }
    while (data->cache) {
        CachedCursor *entry = data->cache;
        data->cache = entry->next;
        if (entry->cursor) {
            DestroyCursor(entry->cursor);
        }
        SDL_free(entry);
    }
    if (data->cursor) {
        DestroyCursor(data->cursor);
    }
    SDL_free(data);
    SDL_free(cursor);
}

static HCURSOR GetCachedCursor(SDL_Cursor *cursor)
{
    SDL_CursorData *data = cursor->internal;

    SDL_Window *focus = SDL_GetMouseFocus();
    if (!focus) {
        return NULL;
    }

    float scale = SDL_GetDisplayContentScale(SDL_GetDisplayForWindow(focus));
    for (CachedCursor *entry = data->cache; entry; entry = entry->next) {
        if (scale == entry->scale) {
            return entry->cursor;
        }
    }

    // Need to create a cursor for this content scale
    SDL_Surface *surface = NULL;
    HCURSOR hcursor = NULL;
    CachedCursor *entry = NULL;

    surface = SDL_GetSurfaceImage(data->surface, scale);
    if (!surface) {
        goto error;
    }

    int hot_x = (int)SDL_round(data->hot_x * scale);
    int hot_y = (int)SDL_round(data->hot_y * scale);
    hcursor = WIN_CreateHCursor(surface, hot_x, hot_y);
    if (!hcursor) {
        goto error;
    }

    entry = (CachedCursor *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        goto error;
    }
    entry->cursor = hcursor;
    entry->scale = scale;
    entry->next = data->cache;
    data->cache = entry;

    SDL_DestroySurface(surface);

    return hcursor;

error:
    if (surface) {
        SDL_DestroySurface(surface);
    }
    if (hcursor) {
        DestroyCursor(hcursor);
    }
    SDL_free(entry);
    return NULL;
}

static bool WIN_ShowCursor(SDL_Cursor *cursor)
{
    if (!cursor) {
        cursor = SDL_blank_cursor;
    }
    if (cursor) {
        if (cursor->internal->surface) {
            SDL_cursor = GetCachedCursor(cursor);
        } else {
            SDL_cursor = cursor->internal->cursor;
        }
    } else {
        SDL_cursor = NULL;
    }
    if (SDL_GetMouseFocus() != NULL) {
        SetCursor(SDL_cursor);
    }
    return true;
}

void WIN_SetCursorPos(int x, int y)
{
    // We need to jitter the value because otherwise Windows will occasionally inexplicably ignore the SetCursorPos() or SendInput()
    SetCursorPos(x, y);
    SetCursorPos(x + 1, y);
    SetCursorPos(x, y);

    // Flush any mouse motion prior to or associated with this warp
#ifdef _MSC_VER // We explicitly want to use GetTickCount(), not GetTickCount64()
#pragma warning(push)
#pragma warning(disable : 28159)
#endif
    SDL_last_warp_time = GetTickCount();
    if (!SDL_last_warp_time) {
        SDL_last_warp_time = 1;
    }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

static bool WIN_WarpMouse(SDL_Window *window, float x, float y)
{
    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    POINT pt;

    // Don't warp the mouse while we're doing a modal interaction
    if (data->in_title_click || data->focus_click_pending) {
        return true;
    }

    pt.x = (int)SDL_roundf(x);
    pt.y = (int)SDL_roundf(y);
    ClientToScreen(hwnd, &pt);
    WIN_SetCursorPos(pt.x, pt.y);

    // Send the exact mouse motion associated with this warp
    SDL_SendMouseMotion(0, window, SDL_GLOBAL_MOUSE_ID, false, x, y);
    return true;
}

static bool WIN_WarpMouseGlobal(float x, float y)
{
    POINT pt;

    pt.x = (int)SDL_roundf(x);
    pt.y = (int)SDL_roundf(y);
    SetCursorPos(pt.x, pt.y);
    return true;
}

static bool WIN_SetRelativeMouseMode(bool enabled)
{
    return WIN_SetRawMouseEnabled(SDL_GetVideoDevice(), enabled);
}

static bool WIN_CaptureMouse(SDL_Window *window)
{
    if (window) {
        SDL_WindowData *data = window->internal;
        SetCapture(data->hwnd);
    } else {
        SDL_Window *focus_window = SDL_GetMouseFocus();

        if (focus_window) {
            SDL_WindowData *data = focus_window->internal;
            if (!data->mouse_tracked) {
                SDL_SetMouseFocus(NULL);
            }
        }
        ReleaseCapture();
    }

    return true;
}

static SDL_MouseButtonFlags WIN_GetGlobalMouseState(float *x, float *y)
{
    SDL_MouseButtonFlags result = 0;
    POINT pt = { 0, 0 };
    bool swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;

    GetCursorPos(&pt);
    *x = (float)pt.x;
    *y = (float)pt.y;

    result |= GetAsyncKeyState(!swapButtons ? VK_LBUTTON : VK_RBUTTON) & 0x8000 ? SDL_BUTTON_LMASK : 0;
    result |= GetAsyncKeyState(!swapButtons ? VK_RBUTTON : VK_LBUTTON) & 0x8000 ? SDL_BUTTON_RMASK : 0;
    result |= GetAsyncKeyState(VK_MBUTTON) & 0x8000 ? SDL_BUTTON_MMASK : 0;
    result |= GetAsyncKeyState(VK_XBUTTON1) & 0x8000 ? SDL_BUTTON_X1MASK : 0;
    result |= GetAsyncKeyState(VK_XBUTTON2) & 0x8000 ? SDL_BUTTON_X2MASK : 0;

    return result;
}

static void WIN_ApplySystemScale(void *internal, Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, float *x, float *y)
{
    if (!internal) {
        return;
    }
    WIN_MouseData *data = (WIN_MouseData *)internal;

    SDL_VideoDisplay *display = window ? SDL_GetVideoDisplayForWindow(window) : SDL_GetVideoDisplay(SDL_GetPrimaryDisplay());

    Sint64 ix = (Sint64)*x * 65536;
    Sint64 iy = (Sint64)*y * 65536;
    Uint32 dpi = display ? (Uint32)(display->content_scale * USER_DEFAULT_SCREEN_DPI) : USER_DEFAULT_SCREEN_DPI;

    if (!data->enhanced) { // early return if flat scale
        dpi = data->dpiscale * (data->dpiaware ? dpi : USER_DEFAULT_SCREEN_DPI);
        ix *= dpi;
        iy *= dpi;
        ix /= USER_DEFAULT_SCREEN_DPI;
        iy /= USER_DEFAULT_SCREEN_DPI;
        ix /= 32;
        iy /= 32;
        // data->residual[0] += ix;
        // data->residual[1] += iy;
        // ix = 65536 * (data->residual[0] / 65536);
        // iy = 65536 * (data->residual[1] / 65536);
        // data->residual[0] -= ix;
        // data->residual[1] -= iy;
        *x = (float)ix / 65536.0f;
        *y = (float)iy / 65536.0f;
        return;
    }

    Uint64 *xs = data->xs;
    Uint64 *ys = data->ys;
    Uint64 absx = SDL_abs(ix);
    Uint64 absy = SDL_abs(iy);
    Uint64 speed = SDL_min(absx, absy) + (SDL_max(absx, absy) << 1); // super cursed approximation used by Windows
    if (speed == 0) {
        return;
    }

    int i, j, k;
    for (i = 1; i < 5; i++) {
        j = i;
        if (speed < xs[j]) {
            break;
        }
    }
    i -= 1;
    j -= 1;
    k = data->last_node;
    data->last_node = j;

    Uint32 denom = data->dpidenom;
    Sint64 scale = 0;
    Sint64 xdiff = xs[j+1] - xs[j];
    Sint64 ydiff = ys[j+1] - ys[j];
    if (xdiff != 0) {
        Sint64 slope = ydiff / xdiff;
        Sint64 inter = slope * xs[i] - ys[i];
        scale += slope - inter / speed;
    }

    if (j > k) {
        denom <<= 1;
        xdiff = xs[k+1] - xs[k];
        ydiff = ys[k+1] - ys[k];
        if (xdiff != 0) {
            Sint64 slope = ydiff / xdiff;
            Sint64 inter = slope * xs[k] - ys[k];
            scale += slope - inter / speed;
        }
    }

    scale *= dpi;
    ix *= scale;
    iy *= scale;
    ix /= denom;
    iy /= denom;
    // data->residual[0] += ix;
    // data->residual[1] += iy;
    // ix = 65536 * (data->residual[0] / 65536);
    // iy = 65536 * (data->residual[1] / 65536);
    // data->residual[0] -= ix;
    // data->residual[1] -= iy;
    *x = (float)ix / 65536.0f;
    *y = (float)iy / 65536.0f;
}

void WIN_InitMouse(SDL_VideoDevice *_this)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    mouse->CreateCursor = WIN_CreateCursor;
    mouse->CreateSystemCursor = WIN_CreateSystemCursor;
    mouse->ShowCursor = WIN_ShowCursor;
    mouse->FreeCursor = WIN_FreeCursor;
    mouse->WarpMouse = WIN_WarpMouse;
    mouse->WarpMouseGlobal = WIN_WarpMouseGlobal;
    mouse->SetRelativeMouseMode = WIN_SetRelativeMouseMode;
    mouse->CaptureMouse = WIN_CaptureMouse;
    mouse->GetGlobalMouseState = WIN_GetGlobalMouseState;
    mouse->ApplySystemScale = WIN_ApplySystemScale;
    mouse->system_scale_data = &WIN_system_scale_data;

    SDL_SetDefaultCursor(WIN_CreateDefaultCursor());

    SDL_blank_cursor = WIN_CreateBlankCursor();

    WIN_UpdateMouseSystemScale();
}

void WIN_QuitMouse(SDL_VideoDevice *_this)
{
    if (SDL_blank_cursor) {
        WIN_FreeCursor(SDL_blank_cursor);
        SDL_blank_cursor = NULL;
    }
}

static void ReadMouseCurve(int v, Uint64 xs[5], Uint64 ys[5])
{
    bool win8 = WIN_IsWindows8OrGreater();
    DWORD xbuff[10] = {
        0x00000000, 0,
        0x00006e15, 0,
        0x00014000, 0,
        0x0003dc29, 0,
        0x00280000, 0
    };
    DWORD ybuff[10] = {
        0x00000000, 0,
        win8 ? 0x000111fd : 0x00015eb8, 0,
        win8 ? 0x00042400 : 0x00054ccd, 0,
        win8 ? 0x0012fc00 : 0x00184ccd, 0,
        win8 ? 0x01bbc000 : 0x02380000, 0
    };
    DWORD xsize = sizeof(xbuff);
    DWORD ysize = sizeof(ybuff);
    HKEY open_handle;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", 0, KEY_READ, &open_handle) == ERROR_SUCCESS) {
        RegQueryValueExW(open_handle, L"SmoothMouseXCurve", NULL, NULL, (BYTE*)xbuff, &xsize);
        RegQueryValueExW(open_handle, L"SmoothMouseYCurve", NULL, NULL, (BYTE*)ybuff, &ysize);
        RegCloseKey(open_handle);
    }
    xs[0] = 0; // first node must always be origin
    ys[0] = 0; // first node must always be origin
    int i;
    for (i = 1; i < 5; i++) {
        xs[i] = (7 * (Uint64)xbuff[i*2]);
        ys[i] = (v * (Uint64)ybuff[i*2]) << 17;
    }
}

void WIN_UpdateMouseSystemScale(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->ApplySystemScale == WIN_ApplySystemScale) {
        mouse->system_scale_data = &WIN_system_scale_data;
    }

    // always reinitialize to valid defaults, whether fetch was successful or not.
    WIN_MouseData *data = &WIN_system_scale_data;
    data->residual[0] = 0;
    data->residual[1] = 0;
    data->dpiscale = 32;
    data->dpidenom = (10 * (WIN_IsWindows8OrGreater() ? 120 : 150)) << 16;
    data->dpiaware = WIN_IsPerMonitorV2DPIAware(SDL_GetVideoDevice());
    data->enhanced = false;

    int v = 10;
    if (SystemParametersInfo(SPI_GETMOUSESPEED, 0, &v, 0)) {
        v = SDL_max(1, SDL_min(v, 20));
        data->dpiscale = SDL_max(SDL_max(v, (v - 2) * 4), (v - 6) * 8);
    }

    int params[3];
    if (SystemParametersInfo(SPI_GETMOUSE, 0, &params, 0)) {
        data->enhanced = params[2] ? true : false;
        if (params[2]) {
            ReadMouseCurve(v, data->xs, data->ys);
        }
    }
}

#endif // SDL_VIDEO_DRIVER_WINDOWS
