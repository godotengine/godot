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

#ifdef __cplusplus
extern "C" {
#endif
// Handle the BeApp specific portions of the application

// Initialize the Be Application, if it's not already started
extern bool SDL_InitBeApp(void);

// Quit the Be Application, if there's nothing left to do
extern void SDL_QuitBeApp(void);

// Be Application Signature
extern const char *SDL_signature;


#ifdef __cplusplus
}
#endif
