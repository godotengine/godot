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

#include "SDL_getenv_c.h"

#if defined(SDL_PLATFORM_WINDOWS)
#include "../core/windows/SDL_windows.h"
#endif

#ifdef SDL_PLATFORM_ANDROID
#include "../core/android/SDL_android.h"
#endif

#if defined(SDL_PLATFORM_WINDOWS)
#define HAVE_WIN32_ENVIRONMENT
#elif defined(HAVE_GETENV) && \
      (defined(HAVE_SETENV) || defined(HAVE_PUTENV)) && \
      (defined(HAVE_UNSETENV) || defined(HAVE_PUTENV))
#define HAVE_LIBC_ENVIRONMENT
#if defined(SDL_PLATFORM_MACOS)
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#elif defined(SDL_PLATFORM_FREEBSD)
#include <dlfcn.h>
static char **get_environ_rtld(void)
{
    char ***environ_rtld = (char ***)dlsym(RTLD_DEFAULT, "environ");
    return environ_rtld ? *environ_rtld : NULL;
}
#define environ (get_environ_rtld())
#else
extern char **environ;
#endif
#else
#define HAVE_LOCAL_ENVIRONMENT
static char **environ;
#endif

#if defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_FREEBSD)
#include <stdlib.h>
#endif

struct SDL_Environment
{
    SDL_Mutex *lock;   // !!! FIXME: reuse SDL_HashTable's lock.
    SDL_HashTable *strings;
};
static SDL_Environment *SDL_environment;

SDL_Environment *SDL_GetEnvironment(void)
{
    if (!SDL_environment) {
        SDL_environment = SDL_CreateEnvironment(true);
    }
    return SDL_environment;
}

bool SDL_InitEnvironment(void)
{
    return (SDL_GetEnvironment() != NULL);
}

void SDL_QuitEnvironment(void)
{
    SDL_Environment *env = SDL_environment;

    if (env) {
        SDL_environment = NULL;
        SDL_DestroyEnvironment(env);
    }
}

SDL_Environment *SDL_CreateEnvironment(bool populated)
{
    SDL_Environment *env = SDL_calloc(1, sizeof(*env));
    if (!env) {
        return NULL;
    }

    env->strings = SDL_CreateHashTable(0, false, SDL_HashString, SDL_KeyMatchString, SDL_DestroyHashKey, NULL);
    if (!env->strings) {
        SDL_free(env);
        return NULL;
    }

    // Don't fail if we can't create a mutex (e.g. on a single-thread environment)  // !!! FIXME: single-threaded environments should still return a non-NULL, do-nothing object here. Check for failure!
    env->lock = SDL_CreateMutex();

    if (populated) {
#ifdef SDL_PLATFORM_WINDOWS
        LPWCH strings = GetEnvironmentStringsW();
        if (strings) {
            for (LPWCH string = strings; *string; string += SDL_wcslen(string) + 1) {
                char *variable = WIN_StringToUTF8W(string);
                if (!variable) {
                    continue;
                }

                char *value = SDL_strchr(variable, '=');
                if (!value || value == variable) {
                    SDL_free(variable);
                    continue;
                }
                *value++ = '\0';

                SDL_InsertIntoHashTable(env->strings, variable, value, true);
            }
            FreeEnvironmentStringsW(strings);
        }
#else
#ifdef SDL_PLATFORM_ANDROID
        // Make sure variables from the application manifest are available
        Android_JNI_GetManifestEnvironmentVariables();
#endif
        char **strings = environ;
        if (strings) {
            for (int i = 0; strings[i]; ++i) {
                char *variable = SDL_strdup(strings[i]);
                if (!variable) {
                    continue;
                }

                char *value = SDL_strchr(variable, '=');
                if (!value || value == variable) {
                    SDL_free(variable);
                    continue;
                }
                *value++ = '\0';

                SDL_InsertIntoHashTable(env->strings, variable, value, true);
            }
        }
#endif // SDL_PLATFORM_WINDOWS
    }

    return env;
}

const char *SDL_GetEnvironmentVariable(SDL_Environment *env, const char *name)
{
#if defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_FREEBSD)
    return getenv(name);
#else
    const char *result = NULL;

    if (!env) {
        return NULL;
    } else if (!name || *name == '\0') {
        return NULL;
    }

    SDL_LockMutex(env->lock);
    {
        const char *value;

        if (SDL_FindInHashTable(env->strings, name, (const void **)&value)) {
            result = SDL_GetPersistentString(value);
        }
    }
    SDL_UnlockMutex(env->lock);

    return result;
#endif
}

typedef struct CountEnvStringsData
{
    size_t count;
    size_t length;
} CountEnvStringsData;

static bool SDLCALL CountEnvStrings(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    CountEnvStringsData *data = (CountEnvStringsData *) userdata;
    data->length += SDL_strlen((const char *) key) + 1 + SDL_strlen((const char *) value) + 1;
    data->count++;
    return true;  // keep iterating.
}

typedef struct CopyEnvStringsData
{
    char **result;
    char *string;
    size_t count;
} CopyEnvStringsData;

static bool SDLCALL CopyEnvStrings(void *userdata, const SDL_HashTable *table, const void *vkey, const void *vvalue)
{
    CopyEnvStringsData *data = (CopyEnvStringsData *) userdata;
    const char *key = (const char *) vkey;
    const char *value = (const char *) vvalue;
    size_t len;

    len = SDL_strlen(key);
    data->result[data->count] = data->string;
    SDL_memcpy(data->string, key, len);
    data->string += len;
    *(data->string++) = '=';

    len = SDL_strlen(value);
    SDL_memcpy(data->string, value, len);
    data->string += len;
    *(data->string++) = '\0';
    data->count++;

    return true;  // keep iterating.
}

char **SDL_GetEnvironmentVariables(SDL_Environment *env)
{
    char **result = NULL;

    if (!env) {
        SDL_InvalidParamError("env");
        return NULL;
    }

    SDL_LockMutex(env->lock);
    {
        // First pass, get the size we need for all the strings
        CountEnvStringsData countdata = { 0, 0 };
        SDL_IterateHashTable(env->strings, CountEnvStrings, &countdata);

        // Allocate memory for the strings
        result = (char **)SDL_malloc((countdata.count + 1) * sizeof(*result) + countdata.length);
        if (result) {
            // Second pass, copy the strings
            char *string = (char *)(result + countdata.count + 1);
            CopyEnvStringsData cpydata = { result, string, 0 };
            SDL_IterateHashTable(env->strings, CopyEnvStrings, &cpydata);
            SDL_assert(countdata.count == cpydata.count);
            result[cpydata.count] = NULL;
        }
    }
    SDL_UnlockMutex(env->lock);

    return result;
}

bool SDL_SetEnvironmentVariable(SDL_Environment *env, const char *name, const char *value, bool overwrite)
{
#if defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_FREEBSD)
    return setenv(name, value, overwrite);
#else
    bool result = false;

    if (!env) {
        return SDL_InvalidParamError("env");
    } else if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL) {
        return SDL_InvalidParamError("name");
    } else if (!value) {
        return SDL_InvalidParamError("value");
    }

    SDL_LockMutex(env->lock);
    {
        char *string = NULL;
        if (SDL_asprintf(&string, "%s=%s", name, value) > 0) {
            const size_t len = SDL_strlen(name);
            string[len] = '\0';
            const char *origname = name;
            name = string;
            value = string + len + 1;
            result = SDL_InsertIntoHashTable(env->strings, name, value, overwrite);
            if (!result) {
                SDL_free(string);
                if (!overwrite) {
                    const void *existing_value = NULL;
                    // !!! FIXME: InsertIntoHashTable does this lookup too, maybe we should have a means to report that, to avoid duplicate work?
                    if (SDL_FindInHashTable(env->strings, origname, &existing_value)) {
                        result = true;  // it already existed, and we refused to overwrite it. Call it success.
                    }
                }
            }
        }
    }
    SDL_UnlockMutex(env->lock);

    return result;
#endif
}

bool SDL_UnsetEnvironmentVariable(SDL_Environment *env, const char *name)
{
#if defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_FREEBSD)
    return unsetenv(name);
#else
    bool result = false;

    if (!env) {
        return SDL_InvalidParamError("env");
    } else if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL) {
        return SDL_InvalidParamError("name");
    }

    SDL_LockMutex(env->lock);
    {
        const void *value;
        if (SDL_FindInHashTable(env->strings, name, &value)) {
            result = SDL_RemoveFromHashTable(env->strings, name);
        } else {
            result = true;
        }
    }
    SDL_UnlockMutex(env->lock);

    return result;
#endif
}

void SDL_DestroyEnvironment(SDL_Environment *env)
{
    if (!env || env == SDL_environment) {
        return;
    }

    SDL_DestroyMutex(env->lock);
    SDL_DestroyHashTable(env->strings);
    SDL_free(env);
}

// Put a variable into the environment
// Note: Name may not contain a '=' character. (Reference: http://www.unix.com/man-page/Linux/3/setenv/)
#ifdef HAVE_LIBC_ENVIRONMENT
#if defined(HAVE_SETENV)
int SDL_setenv_unsafe(const char *name, const char *value, int overwrite)
{
    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL || !value) {
        return -1;
    }

    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), name, value, (overwrite != 0));

    return setenv(name, value, overwrite);
}
// We have a real environment table, but no real setenv? Fake it w/ putenv.
#else
int SDL_setenv_unsafe(const char *name, const char *value, int overwrite)
{
    char *new_variable;

    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL || !value) {
        return -1;
    }

    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), name, value, (overwrite != 0));

    if (getenv(name) != NULL) {
        if (!overwrite) {
            return 0; // leave the existing one there.
        }
    }

    // This leaks. Sorry. Get a better OS so we don't have to do this.
    SDL_asprintf(&new_variable, "%s=%s", name, value);
    if (!new_variable) {
        return -1;
    }
    return putenv(new_variable);
}
#endif
#elif defined(HAVE_WIN32_ENVIRONMENT)
int SDL_setenv_unsafe(const char *name, const char *value, int overwrite)
{
    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL || !value) {
        return -1;
    }

    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), name, value, (overwrite != 0));

    if (!overwrite) {
        if (GetEnvironmentVariableA(name, NULL, 0) > 0) {
            return 0; // asked not to overwrite existing value.
        }
    }
    if (!SetEnvironmentVariableA(name, value)) {
        return -1;
    }
    return 0;
}
#else // roll our own

int SDL_setenv_unsafe(const char *name, const char *value, int overwrite)
{
    int added;
    size_t len, i;
    char **new_env;
    char *new_variable;

    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL || !value) {
        return -1;
    }

    // See if it already exists
    if (!overwrite && SDL_getenv_unsafe(name)) {
        return 0;
    }

    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), name, value, (overwrite != 0));

    // Allocate memory for the variable
    len = SDL_strlen(name) + SDL_strlen(value) + 2;
    new_variable = (char *)SDL_malloc(len);
    if (!new_variable) {
        return -1;
    }

    SDL_snprintf(new_variable, len, "%s=%s", name, value);
    value = new_variable + SDL_strlen(name) + 1;
    name = new_variable;

    // Actually put it into the environment
    added = 0;
    i = 0;
    if (environ) {
        // Check to see if it's already there...
        len = (value - name);
        for (; environ[i]; ++i) {
            if (SDL_strncmp(environ[i], name, len) == 0) {
                // If we found it, just replace the entry
                SDL_free(environ[i]);
                environ[i] = new_variable;
                added = 1;
                break;
            }
        }
    }

    // Didn't find it in the environment, expand and add
    if (!added) {
        new_env = SDL_realloc(environ, (i + 2) * sizeof(char *));
        if (new_env) {
            environ = new_env;
            environ[i++] = new_variable;
            environ[i++] = (char *)0;
            added = 1;
        } else {
            SDL_free(new_variable);
        }
    }
    return added ? 0 : -1;
}
#endif // HAVE_LIBC_ENVIRONMENT

#ifdef HAVE_LIBC_ENVIRONMENT
#if defined(HAVE_UNSETENV)
int SDL_unsetenv_unsafe(const char *name)
{
    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL) {
        return -1;
    }

    SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), name);

    return unsetenv(name);
}
// We have a real environment table, but no unsetenv? Fake it w/ putenv.
#else
int SDL_unsetenv_unsafe(const char *name)
{
    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL) {
        return -1;
    }

    SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), name);

    // Hope this environment uses the non-standard extension of removing the environment variable if it has no '='
    return putenv(name);
}
#endif
#elif defined(HAVE_WIN32_ENVIRONMENT)
int SDL_unsetenv_unsafe(const char *name)
{
    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL) {
        return -1;
    }

    SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), name);

    if (!SetEnvironmentVariableA(name, NULL)) {
        return -1;
    }
    return 0;
}
#else
int SDL_unsetenv_unsafe(const char *name)
{
    size_t len, i;

    // Input validation
    if (!name || *name == '\0' || SDL_strchr(name, '=') != NULL) {
        return -1;
    }

    SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), name);

    if (environ) {
        len = SDL_strlen(name);
        for (i = 0; environ[i]; ++i) {
            if ((SDL_strncmp(environ[i], name, len) == 0) &&
                (environ[i][len] == '=')) {
                // Just clear out this entry for now
                *environ[i] = '\0';
                break;
            }
        }
    }
    return 0;
}
#endif // HAVE_LIBC_ENVIRONMENT

// Retrieve a variable named "name" from the environment
#ifdef HAVE_LIBC_ENVIRONMENT
const char *SDL_getenv_unsafe(const char *name)
{
#ifdef SDL_PLATFORM_ANDROID
    // Make sure variables from the application manifest are available
    Android_JNI_GetManifestEnvironmentVariables();
#endif

    // Input validation
    if (!name || *name == '\0') {
        return NULL;
    }

    return getenv(name);
}
#elif defined(HAVE_WIN32_ENVIRONMENT)
const char *SDL_getenv_unsafe(const char *name)
{
    DWORD length, maxlen = 0;
    char *string = NULL;
    const char *result = NULL;

    // Input validation
    if (!name || *name == '\0') {
        return NULL;
    }

    for ( ; ; ) {
        SetLastError(ERROR_SUCCESS);
        length = GetEnvironmentVariableA(name, string, maxlen);

        if (length > maxlen) {
            char *temp = (char *)SDL_realloc(string, length);
            if (!temp)  {
                return NULL;
            }
            string = temp;
            maxlen = length;
        } else {
            if (GetLastError() != ERROR_SUCCESS) {
                if (string) {
                    SDL_free(string);
                }
                return NULL;
            }
            break;
        }
    }
    if (string) {
        result = SDL_GetPersistentString(string);
        SDL_free(string);
    }
    return result;
}
#else
const char *SDL_getenv_unsafe(const char *name)
{
    size_t len, i;
    const char *value = NULL;

    // Input validation
    if (!name || *name == '\0') {
        return NULL;
    }

    if (environ) {
        len = SDL_strlen(name);
        for (i = 0; environ[i]; ++i) {
            if ((SDL_strncmp(environ[i], name, len) == 0) &&
                (environ[i][len] == '=')) {
                value = &environ[i][len + 1];
                break;
            }
        }
    }
    return value;
}
#endif // HAVE_LIBC_ENVIRONMENT

const char *SDL_getenv(const char *name)
{
    return SDL_GetEnvironmentVariable(SDL_GetEnvironment(), name);
}
