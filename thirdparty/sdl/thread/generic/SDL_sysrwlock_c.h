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

#ifndef SDL_sysrwlock_c_h_
#define SDL_sysrwlock_c_h_

#ifdef SDL_THREAD_GENERIC_RWLOCK_SUFFIX

SDL_RWLock *SDL_CreateRWLock_generic(void);
void SDL_DestroyRWLock_generic(SDL_RWLock *rwlock);
void SDL_LockRWLockForReading_generic(SDL_RWLock *rwlock);
void  SDL_LockRWLockForWriting_generic(SDL_RWLock *rwlock);
bool SDL_TryLockRWLockForReading_generic(SDL_RWLock *rwlock);
bool SDL_TryLockRWLockForWriting_generic(SDL_RWLock *rwlock);
void SDL_UnlockRWLock_generic(SDL_RWLock *rwlock);

#endif // SDL_THREAD_GENERIC_RWLOCK_SUFFIX

#endif // SDL_sysrwlock_c_h_
