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

#ifndef SDL_syscond_generic_h_
#define SDL_syscond_generic_h_

#ifdef SDL_THREAD_GENERIC_COND_SUFFIX

SDL_Condition *SDL_CreateCondition_generic(void);
void SDL_DestroyCondition_generic(SDL_Condition *cond);
void SDL_SignalCondition_generic(SDL_Condition *cond);
void SDL_BroadcastCondition_generic(SDL_Condition *cond);
bool SDL_WaitConditionTimeoutNS_generic(SDL_Condition *cond, SDL_Mutex *mutex, Sint64 timeoutNS);

#endif // SDL_THREAD_GENERIC_COND_SUFFIX

#endif // SDL_syscond_generic_h_
