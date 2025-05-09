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

#include "SDL_surface_utils.h"

#include "../SDL_surface_c.h"

#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
HICON CreateIconFromSurface(SDL_Surface *surface)
{
    SDL_Surface *s = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ARGB8888);
    if (!s) {
        return NULL;
    }

    /* The dimensions will be needed after s is freed */
    const int width = s->w;
    const int height = s->h;

    BITMAPINFO bmpInfo;
    SDL_zero(bmpInfo);
    bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmpInfo.bmiHeader.biWidth = width;
    bmpInfo.bmiHeader.biHeight = -height; /* Top-down bitmap */
    bmpInfo.bmiHeader.biPlanes = 1;
    bmpInfo.bmiHeader.biBitCount = 32;
    bmpInfo.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(NULL);
    void* pBits = NULL;
    HBITMAP hBitmap = CreateDIBSection(hdc, &bmpInfo, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (!hBitmap) {
        ReleaseDC(NULL, hdc);
        SDL_DestroySurface(s);
        return NULL;
    }

    SDL_memcpy(pBits, s->pixels, width * height * 4);

    SDL_DestroySurface(s);

    HBITMAP hMask = CreateBitmap(width, height, 1, 1, NULL);
    if (!hMask) {
        DeleteObject(hBitmap);
        ReleaseDC(NULL, hdc);
        return NULL;
    }

    HDC hdcMem = CreateCompatibleDC(hdc);
    HGDIOBJ oldBitmap = SelectObject(hdcMem, hMask);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            BYTE* pixel = (BYTE*)pBits + (y * width + x) * 4;
            BYTE alpha = pixel[3];
            COLORREF maskColor = (alpha == 0) ? RGB(0, 0, 0) : RGB(255, 255, 255);
            SetPixel(hdcMem, x, y, maskColor);
        }
    }

    ICONINFO iconInfo;
    iconInfo.fIcon = TRUE;
    iconInfo.xHotspot = 0;
    iconInfo.yHotspot = 0;
    iconInfo.hbmMask = hMask;
    iconInfo.hbmColor = hBitmap;

    HICON hIcon = CreateIconIndirect(&iconInfo);

    SelectObject(hdcMem, oldBitmap);
    DeleteDC(hdcMem);
    DeleteObject(hBitmap);
    DeleteObject(hMask);
    ReleaseDC(NULL, hdc);

    return hIcon;
}
#endif
