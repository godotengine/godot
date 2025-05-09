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

#ifndef SDL_waylandclipboard_h_
#define SDL_waylandclipboard_h_

extern const char **Wayland_GetTextMimeTypes(SDL_VideoDevice *_this, size_t *num_mime_types);
extern bool Wayland_SetClipboardData(SDL_VideoDevice *_this);
extern void *Wayland_GetClipboardData(SDL_VideoDevice *_this, const char *mime_type, size_t *length);
extern bool Wayland_HasClipboardData(SDL_VideoDevice *_this, const char *mime_type);
extern bool Wayland_SetPrimarySelectionText(SDL_VideoDevice *_this, const char *text);
extern char *Wayland_GetPrimarySelectionText(SDL_VideoDevice *_this);
extern bool Wayland_HasPrimarySelectionText(SDL_VideoDevice *_this);

#endif // SDL_waylandclipboard_h_
