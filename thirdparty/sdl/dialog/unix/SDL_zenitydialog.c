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

#include "../SDL_dialog_utils.h"

#define X11_HANDLE_MAX_WIDTH 28
typedef struct
{
    SDL_DialogFileCallback callback;
    void *userdata;
    void *argv;

    /* Zenity only works with X11 handles apparently */
    char x11_window_handle[X11_HANDLE_MAX_WIDTH];
    /* These are part of argv, but are tracked separately for deallocation purposes */
    int nfilters;
    char **filters_slice;
    char *filename;
    char *title;
    char *accept;
    char *cancel;
} zenityArgs;

static char *zenity_clean_name(const char *name)
{
    char *newname = SDL_strdup(name);

    /* Filter out "|", which Zenity considers a special character. Let's hope
       there aren't others. TODO: find something better. */
    for (char *c = newname; *c; c++) {
        if (*c == '|') {
            // Zenity doesn't support escaping with '\'
            *c = '/';
        }
    }

    return newname;
}

static bool get_x11_window_handle(SDL_PropertiesID props, char *out)
{
    SDL_Window *window = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, NULL);
    if (!window) {
        return false;
    }
    SDL_PropertiesID window_props = SDL_GetWindowProperties(window);
    if (!window_props) {
        return false;
    }
    Uint64 handle = (Uint64)SDL_GetNumberProperty(window_props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
    if (!handle) {
        return false;
    }
    if (SDL_snprintf(out, X11_HANDLE_MAX_WIDTH, "0x%" SDL_PRIx64, handle) >= X11_HANDLE_MAX_WIDTH) {
        return false;
    };
    return true;
}

/* Exec call format:
 *
 *     zenity --file-selection --separator=\n [--multiple]
 *                         [--directory] [--save --confirm-overwrite]
 *                         [--filename FILENAME] [--modal --attach 0x11w1nd0w]
 *                         [--title TITLE] [--ok-label ACCEPT]
 *                         [--cancel-label CANCEL]
 *                         [--file-filter=Filter Name | *.filt *.fn ...]...
 */
static zenityArgs *create_zenity_args(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    zenityArgs *args = SDL_calloc(1, sizeof(*args));
    if (!args) {
        return NULL;
    }
    args->callback = callback;
    args->userdata = userdata;
    args->nfilters = SDL_GetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, 0);

    const char **argv = SDL_malloc(
        sizeof(*argv) * (3   /* zenity --file-selection --separator=\n */
                         + 1 /* --multiple */
                         + 2 /* --directory | --save --confirm-overwrite */
                         + 2 /* --filename [file] */
                         + 3 /* --modal --attach [handle] */
                         + 2 /* --title [title] */
                         + 2 /* --ok-label [label] */
                         + 2 /* --cancel-label [label] */
                         + args->nfilters + 1 /* NULL */));
    if (!argv) {
        goto cleanup;
    }
    args->argv = argv;

    /* Properties can be destroyed as soon as the function returns; copy over what we need. */
#define COPY_STRING_PROPERTY(dst, prop)                             \
    {                                                               \
        const char *str = SDL_GetStringProperty(props, prop, NULL); \
        if (str) {                                                  \
            dst = SDL_strdup(str);                                  \
            if (!dst) {                                             \
                goto cleanup;                                       \
            }                                                       \
        }                                                           \
    }

    COPY_STRING_PROPERTY(args->filename, SDL_PROP_FILE_DIALOG_LOCATION_STRING);
    COPY_STRING_PROPERTY(args->title, SDL_PROP_FILE_DIALOG_TITLE_STRING);
    COPY_STRING_PROPERTY(args->accept, SDL_PROP_FILE_DIALOG_ACCEPT_STRING);
    COPY_STRING_PROPERTY(args->cancel, SDL_PROP_FILE_DIALOG_CANCEL_STRING);
#undef COPY_STRING_PROPERTY

    // ARGV PASS
    int argc = 0;
    argv[argc++] = "zenity";
    argv[argc++] = "--file-selection";
    argv[argc++] = "--separator=\n";

    if (SDL_GetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, false)) {
        argv[argc++] = "--multiple";
    }

    switch (type) {
    case SDL_FILEDIALOG_OPENFILE:
        break;

    case SDL_FILEDIALOG_SAVEFILE:
        argv[argc++] = "--save";
        /* Asking before overwriting while saving seems like a sane default */
        argv[argc++] = "--confirm-overwrite";
        break;

    case SDL_FILEDIALOG_OPENFOLDER:
        argv[argc++] = "--directory";
        break;
    };

    if (args->filename) {
        argv[argc++] = "--filename";
        argv[argc++] = args->filename;
    }

    if (get_x11_window_handle(props, args->x11_window_handle)) {
        argv[argc++] = "--modal";
        argv[argc++] = "--attach";
        argv[argc++] = args->x11_window_handle;
    }

    if (args->title) {
        argv[argc++] = "--title";
        argv[argc++] = args->title;
    }

    if (args->accept) {
        argv[argc++] = "--ok-label";
        argv[argc++] = args->accept;
    }

    if (args->cancel) {
        argv[argc++] = "--cancel-label";
        argv[argc++] = args->cancel;
    }

    const SDL_DialogFileFilter *filters = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, NULL);
    if (filters) {
        args->filters_slice = (char **)&argv[argc];
        for (int i = 0; i < args->nfilters; i++) {
            char *filter_str = convert_filter(filters[i],
                                              zenity_clean_name,
                                              "--file-filter=", " | ", "",
                                              "*.", " *.", "");

            if (!filter_str) {
                while (i--) {
                    SDL_free(args->filters_slice[i]);
                }
                goto cleanup;
            }

            args->filters_slice[i] = filter_str;
        }
        argc += args->nfilters;
    }

    argv[argc] = NULL;
    return args;

cleanup:
    SDL_free(args->filename);
    SDL_free(args->title);
    SDL_free(args->accept);
    SDL_free(args->cancel);
    SDL_free(argv);
    SDL_free(args);
    return NULL;
}

// TODO: Zenity survives termination of the parent

static void run_zenity(SDL_DialogFileCallback callback, void *userdata, void *argv)
{
    SDL_Process *process = NULL;
    SDL_Environment *env = NULL;
    int status = -1;
    size_t bytes_read = 0;
    char *container = NULL;
    size_t narray = 1;
    char **array = NULL;
    bool result = false;

    env = SDL_CreateEnvironment(true);
    if (!env) {
        goto done;
    }

    /* Recent versions of Zenity have different exit codes, but picks up
      different codes from the environment */
    SDL_SetEnvironmentVariable(env, "ZENITY_OK", "0", true);
    SDL_SetEnvironmentVariable(env, "ZENITY_CANCEL", "1", true);
    SDL_SetEnvironmentVariable(env, "ZENITY_ESC", "1", true);
    SDL_SetEnvironmentVariable(env, "ZENITY_EXTRA", "2", true);
    SDL_SetEnvironmentVariable(env, "ZENITY_ERROR", "2", true);
    SDL_SetEnvironmentVariable(env, "ZENITY_TIMEOUT", "2", true);

    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, argv);
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ENVIRONMENT_POINTER, env);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_NULL);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER, SDL_PROCESS_STDIO_NULL);
    process = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    if (!process) {
        goto done;
    }

    container = SDL_ReadProcess(process, &bytes_read, &status);
    if (!container) {
        goto done;
    }

    array = (char **)SDL_malloc((narray + 1) * sizeof(char *));
    if (!array) {
        goto done;
    }
    array[0] = container;
    array[1] = NULL;

    for (int i = 0; i < bytes_read; i++) {
        if (container[i] == '\n') {
            container[i] = '\0';
            // Reading from a process often leaves a trailing \n, so ignore the last one
            if (i < bytes_read - 1) {
                array[narray] = container + i + 1;
                narray++;
                char **new_array = (char **)SDL_realloc(array, (narray + 1) * sizeof(char *));
                if (!new_array) {
                    goto done;
                }
                array = new_array;
                array[narray] = NULL;
            }
        }
    }

    // 0 = the user chose one or more files, 1 = the user canceled the dialog
    if (status == 0 || status == 1) {
        callback(userdata, (const char *const *)array, -1);
    } else {
        SDL_SetError("Could not run zenity: exit code %d", status);
        callback(userdata, NULL, -1);
    }

    result = true;

done:
    SDL_free(array);
    SDL_free(container);
    SDL_DestroyEnvironment(env);
    SDL_DestroyProcess(process);

    if (!result) {
        callback(userdata, NULL, -1);
    }
}

static void free_zenity_args(zenityArgs *args)
{
    if (args->filters_slice) {
        for (int i = 0; i < args->nfilters; i++) {
            SDL_free(args->filters_slice[i]);
        }
    }
    SDL_free(args->filename);
    SDL_free(args->title);
    SDL_free(args->accept);
    SDL_free(args->cancel);
    SDL_free(args->argv);
    SDL_free(args);
}

static int run_zenity_thread(void *ptr)
{
    zenityArgs *args = ptr;
    run_zenity(args->callback, args->userdata, args->argv);
    free_zenity_args(args);
    return 0;
}

void SDL_Zenity_ShowFileDialogWithProperties(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    zenityArgs *args = create_zenity_args(type, callback, userdata, props);
    if (!args) {
        callback(userdata, NULL, -1);
        return;
    }

    SDL_Thread *thread = SDL_CreateThread(run_zenity_thread, "SDL_ZenityFileDialog", (void *)args);

    if (!thread) {
        free_zenity_args(args);
        callback(userdata, NULL, -1);
        return;
    }

    SDL_DetachThread(thread);
}

bool SDL_Zenity_detect(void)
{
    const char *args[] = {
        "zenity", "--version", NULL
    };
    int status = -1;

    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, args);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_NULL);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_NULL);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER, SDL_PROCESS_STDIO_NULL);
    SDL_Process *process = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    if (process) {
        SDL_WaitProcess(process, true, &status);
        SDL_DestroyProcess(process);
    }
    return (status == 0);
}
