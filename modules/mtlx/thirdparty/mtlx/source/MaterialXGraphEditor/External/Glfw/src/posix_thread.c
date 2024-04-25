//========================================================================
// GLFW 3.4 POSIX - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2002-2006 Marcus Geelnard
// Copyright (c) 2006-2017 Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// It is fine to use C99 in this file because it will not be built with VS
//========================================================================

#include "internal.h"

#include <assert.h>
#include <string.h>


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

GLFWbool _glfwPlatformCreateTls(_GLFWtls* tls)
{
    assert(tls->posix.allocated == GLFW_FALSE);

    if (pthread_key_create(&tls->posix.key, NULL) != 0)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "POSIX: Failed to create context TLS");
        return GLFW_FALSE;
    }

    tls->posix.allocated = GLFW_TRUE;
    return GLFW_TRUE;
}

void _glfwPlatformDestroyTls(_GLFWtls* tls)
{
    if (tls->posix.allocated)
        pthread_key_delete(tls->posix.key);
    memset(tls, 0, sizeof(_GLFWtls));
}

void* _glfwPlatformGetTls(_GLFWtls* tls)
{
    assert(tls->posix.allocated == GLFW_TRUE);
    return pthread_getspecific(tls->posix.key);
}

void _glfwPlatformSetTls(_GLFWtls* tls, void* value)
{
    assert(tls->posix.allocated == GLFW_TRUE);
    pthread_setspecific(tls->posix.key, value);
}

GLFWbool _glfwPlatformCreateMutex(_GLFWmutex* mutex)
{
    assert(mutex->posix.allocated == GLFW_FALSE);

    if (pthread_mutex_init(&mutex->posix.handle, NULL) != 0)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR, "POSIX: Failed to create mutex");
        return GLFW_FALSE;
    }

    return mutex->posix.allocated = GLFW_TRUE;
}

void _glfwPlatformDestroyMutex(_GLFWmutex* mutex)
{
    if (mutex->posix.allocated)
        pthread_mutex_destroy(&mutex->posix.handle);
    memset(mutex, 0, sizeof(_GLFWmutex));
}

void _glfwPlatformLockMutex(_GLFWmutex* mutex)
{
    assert(mutex->posix.allocated == GLFW_TRUE);
    pthread_mutex_lock(&mutex->posix.handle);
}

void _glfwPlatformUnlockMutex(_GLFWmutex* mutex)
{
    assert(mutex->posix.allocated == GLFW_TRUE);
    pthread_mutex_unlock(&mutex->posix.handle);
}

