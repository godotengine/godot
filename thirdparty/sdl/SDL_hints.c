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

#include "SDL_hints_c.h"

#ifdef SDL_PLATFORM_ANDROID
#include "core/android/SDL_android.h"
#endif

typedef struct SDL_HintWatch
{
    SDL_HintCallback callback;
    void *userdata;
    struct SDL_HintWatch *next;
} SDL_HintWatch;

typedef struct SDL_Hint
{
    char *value;
    SDL_HintPriority priority;
    SDL_HintWatch *callbacks;
} SDL_Hint;

static SDL_AtomicU32 SDL_hint_props;


void SDL_InitHints(void)
{
}

void SDL_QuitHints(void)
{
    SDL_PropertiesID props;
    do {
        props = SDL_GetAtomicU32(&SDL_hint_props);
    } while (!SDL_CompareAndSwapAtomicU32(&SDL_hint_props, props, 0));

    if (props) {
        SDL_DestroyProperties(props);
    }
}

static SDL_PropertiesID GetHintProperties(bool create)
{
    SDL_PropertiesID props = SDL_GetAtomicU32(&SDL_hint_props);
    if (!props && create) {
        props = SDL_CreateProperties();
        if (!SDL_CompareAndSwapAtomicU32(&SDL_hint_props, 0, props)) {
            // Somebody else created hint properties before us, just use those
            SDL_DestroyProperties(props);
            props = SDL_GetAtomicU32(&SDL_hint_props);
        }
    }
    return props;
}

static void SDLCALL CleanupHintProperty(void *userdata, void *value)
{
    SDL_Hint *hint = (SDL_Hint *) value;
    SDL_free(hint->value);

    SDL_HintWatch *entry = hint->callbacks;
    while (entry) {
        SDL_HintWatch *freeable = entry;
        entry = entry->next;
        SDL_free(freeable);
    }
    SDL_free(hint);
}

static const char* GetHintEnvironmentVariable(const char *name)
{
    const char *result = SDL_getenv(name);
    if (!result && name && *name) {
        // fall back to old (SDL2) names of environment variables that
        // are important to users (e.g. many use SDL_VIDEODRIVER=wayland)
        if (SDL_strcmp(name, SDL_HINT_VIDEO_DRIVER) == 0) {
            result = SDL_getenv("SDL_VIDEODRIVER");
        } else if (SDL_strcmp(name, SDL_HINT_AUDIO_DRIVER) == 0) {
            result = SDL_getenv("SDL_AUDIODRIVER");
        }
    }
    return result;
}

bool SDL_SetHintWithPriority(const char *name, const char *value, SDL_HintPriority priority)
{
    if (!name || !*name) {
        return SDL_InvalidParamError("name");
    }

    const char *env = GetHintEnvironmentVariable(name);
    if (env && (priority < SDL_HINT_OVERRIDE)) {
        return SDL_SetError("An environment variable is taking priority");
    }

    const SDL_PropertiesID hints = GetHintProperties(true);
    if (!hints) {
        return false;
    }

    bool result = false;

    SDL_LockProperties(hints);

    SDL_Hint *hint = (SDL_Hint *)SDL_GetPointerProperty(hints, name, NULL);
    if (hint) {
        if (priority >= hint->priority) {
            if (hint->value != value && (!value || !hint->value || SDL_strcmp(hint->value, value) != 0)) {
                char *old_value = hint->value;

                hint->value = value ? SDL_strdup(value) : NULL;
                SDL_HintWatch *entry = hint->callbacks;
                while (entry) {
                    // Save the next entry in case this one is deleted
                    SDL_HintWatch *next = entry->next;
                    entry->callback(entry->userdata, name, old_value, value);
                    entry = next;
                }
                SDL_free(old_value);
            }
            hint->priority = priority;
            result = true;
        }
    } else {  // Couldn't find the hint? Add a new one.
        hint = (SDL_Hint *)SDL_malloc(sizeof(*hint));
        if (hint) {
            hint->value = value ? SDL_strdup(value) : NULL;
            hint->priority = priority;
            hint->callbacks = NULL;
            result = SDL_SetPointerPropertyWithCleanup(hints, name, hint, CleanupHintProperty, NULL);
        }
    }

#ifdef SDL_PLATFORM_ANDROID
    if (SDL_strcmp(name, SDL_HINT_ANDROID_ALLOW_RECREATE_ACTIVITY) == 0) {
        // Special handling for this hint, which needs to persist outside the normal application flow
        Android_SetAllowRecreateActivity(SDL_GetStringBoolean(value, false));
    }
#endif // SDL_PLATFORM_ANDROID

    SDL_UnlockProperties(hints);

    return result;
}

bool SDL_ResetHint(const char *name)
{
    if (!name || !*name) {
        return SDL_InvalidParamError("name");
    }

    const char *env = GetHintEnvironmentVariable(name);

    const SDL_PropertiesID hints = GetHintProperties(false);
    if (!hints) {
        return false;
    }

    bool result = false;

    SDL_LockProperties(hints);

    SDL_Hint *hint = (SDL_Hint *)SDL_GetPointerProperty(hints, name, NULL);
    if (hint) {
        if ((!env && hint->value) || (env && !hint->value) || (env && SDL_strcmp(env, hint->value) != 0)) {
            for (SDL_HintWatch *entry = hint->callbacks; entry;) {
                // Save the next entry in case this one is deleted
                SDL_HintWatch *next = entry->next;
                entry->callback(entry->userdata, name, hint->value, env);
                entry = next;
            }
        }
        SDL_free(hint->value);
        hint->value = NULL;
        hint->priority = SDL_HINT_DEFAULT;
        result = true;
    }

#ifdef SDL_PLATFORM_ANDROID
    if (SDL_strcmp(name, SDL_HINT_ANDROID_ALLOW_RECREATE_ACTIVITY) == 0) {
        // Special handling for this hint, which needs to persist outside the normal application flow
        if (env) {
            Android_SetAllowRecreateActivity(SDL_GetStringBoolean(env, false));
        } else {
            Android_SetAllowRecreateActivity(false);
        }
    }
#endif // SDL_PLATFORM_ANDROID

    SDL_UnlockProperties(hints);

    return result;
}

static void SDLCALL ResetHintsCallback(void *userdata, SDL_PropertiesID hints, const char *name)
{
    SDL_Hint *hint = (SDL_Hint *)SDL_GetPointerProperty(hints, name, NULL);
    if (!hint) {
        return;  // uh...okay.
    }

    const char *env = GetHintEnvironmentVariable(name);
    if ((!env && hint->value) || (env && !hint->value) || (env && SDL_strcmp(env, hint->value) != 0)) {
        SDL_HintWatch *entry = hint->callbacks;
        while (entry) {
            // Save the next entry in case this one is deleted
            SDL_HintWatch *next = entry->next;
            entry->callback(entry->userdata, name, hint->value, env);
            entry = next;
        }
    }
    SDL_free(hint->value);
    hint->value = NULL;
    hint->priority = SDL_HINT_DEFAULT;

#ifdef SDL_PLATFORM_ANDROID
    if (SDL_strcmp(name, SDL_HINT_ANDROID_ALLOW_RECREATE_ACTIVITY) == 0) {
        // Special handling for this hint, which needs to persist outside the normal application flow
        if (env) {
            Android_SetAllowRecreateActivity(SDL_GetStringBoolean(env, false));
        } else {
            Android_SetAllowRecreateActivity(false);
        }
    }
#endif // SDL_PLATFORM_ANDROID
}

void SDL_ResetHints(void)
{
    SDL_EnumerateProperties(GetHintProperties(false), ResetHintsCallback, NULL);
}

bool SDL_SetHint(const char *name, const char *value)
{
    return SDL_SetHintWithPriority(name, value, SDL_HINT_NORMAL);
}

const char *SDL_GetHint(const char *name)
{
    if (!name) {
        return NULL;
    }

    const char *result = GetHintEnvironmentVariable(name);

    const SDL_PropertiesID hints = GetHintProperties(false);
    if (hints) {
        SDL_LockProperties(hints);

        SDL_Hint *hint = (SDL_Hint *)SDL_GetPointerProperty(hints, name, NULL);
        if (hint) {
            if (!result || hint->priority == SDL_HINT_OVERRIDE) {
                result = SDL_GetPersistentString(hint->value);
            }
        }

        SDL_UnlockProperties(hints);
    }

    return result;
}

int SDL_GetStringInteger(const char *value, int default_value)
{
    if (!value || !*value) {
        return default_value;
    }
    if (SDL_strcasecmp(value, "false") == 0) {
        return 0;
    }
    if (SDL_strcasecmp(value, "true") == 0) {
        return 1;
    }
    if (*value == '-' || SDL_isdigit(*value)) {
        return SDL_atoi(value);
    }
    return default_value;
}

bool SDL_GetStringBoolean(const char *value, bool default_value)
{
    if (!value || !*value) {
        return default_value;
    }
    if (*value == '0' || SDL_strcasecmp(value, "false") == 0) {
        return false;
    }
    return true;
}

bool SDL_GetHintBoolean(const char *name, bool default_value)
{
    const char *hint = SDL_GetHint(name);
    return SDL_GetStringBoolean(hint, default_value);
}

bool SDL_AddHintCallback(const char *name, SDL_HintCallback callback, void *userdata)
{
    if (!name || !*name) {
        return SDL_InvalidParamError("name");
    } else if (!callback) {
        return SDL_InvalidParamError("callback");
    }

    const SDL_PropertiesID hints = GetHintProperties(true);
    if (!hints) {
        return false;
    }

    SDL_HintWatch *entry = (SDL_HintWatch *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        return false;
    }
    entry->callback = callback;
    entry->userdata = userdata;

    bool result = false;

    SDL_LockProperties(hints);

    SDL_RemoveHintCallback(name, callback, userdata);

    SDL_Hint *hint = (SDL_Hint *)SDL_GetPointerProperty(hints, name, NULL);
    if (hint) {
        result = true;
    } else {  // Need to add a hint entry for this watcher
        hint = (SDL_Hint *)SDL_malloc(sizeof(*hint));
        if (!hint) {
            SDL_free(entry);
            SDL_UnlockProperties(hints);
            return false;
        } else {
            hint->value = NULL;
            hint->priority = SDL_HINT_DEFAULT;
            hint->callbacks = NULL;
            result = SDL_SetPointerPropertyWithCleanup(hints, name, hint, CleanupHintProperty, NULL);
        }
    }

    // Add it to the callbacks for this hint
    entry->next = hint->callbacks;
    hint->callbacks = entry;

    // Now call it with the current value
    const char *value = SDL_GetHint(name);
    callback(userdata, name, value, value);

    SDL_UnlockProperties(hints);

    return result;
}

void SDL_RemoveHintCallback(const char *name, SDL_HintCallback callback, void *userdata)
{
    if (!name || !*name) {
        return;
    }

    const SDL_PropertiesID hints = GetHintProperties(false);
    if (!hints) {
        return;
    }

    SDL_LockProperties(hints);
    SDL_Hint *hint = (SDL_Hint *)SDL_GetPointerProperty(hints, name, NULL);
    if (hint) {
        SDL_HintWatch *prev = NULL;
        for (SDL_HintWatch *entry = hint->callbacks; entry; entry = entry->next) {
            if ((callback == entry->callback) && (userdata == entry->userdata)) {
                if (prev) {
                    prev->next = entry->next;
                } else {
                    hint->callbacks = entry->next;
                }
                SDL_free(entry);
                break;
            }
            prev = entry;
        }
    }
    SDL_UnlockProperties(hints);
}

