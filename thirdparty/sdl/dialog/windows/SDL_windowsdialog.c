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
#include "../SDL_dialog.h"
#include "../SDL_dialog_utils.h"

#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#include "../../core/windows/SDL_windows.h"
#include "../../thread/SDL_systhread.h"

// If this number is too small, selecting too many files will give an error
#define SELECTLIST_SIZE 65536

typedef struct
{
    bool is_save;
    wchar_t *filters_str;
    char *default_file;
    SDL_Window *parent;
    DWORD flags;
    SDL_DialogFileCallback callback;
    void *userdata;
    char *title;
    char *accept;
    char *cancel;
} winArgs;

typedef struct
{
    SDL_Window *parent;
    SDL_DialogFileCallback callback;
    char *default_folder;
    void *userdata;
    char *title;
    char *accept;
    char *cancel;
} winFArgs;

void freeWinArgs(winArgs *args)
{
    SDL_free(args->default_file);
    SDL_free(args->filters_str);
    SDL_free(args->title);
    SDL_free(args->accept);
    SDL_free(args->cancel);

    SDL_free(args);
}

void freeWinFArgs(winFArgs *args)
{
    SDL_free(args->default_folder);
    SDL_free(args->title);
    SDL_free(args->accept);
    SDL_free(args->cancel);

    SDL_free(args);
}

/** Converts dialog.nFilterIndex to SDL-compatible value */
int getFilterIndex(int as_reported_by_windows)
{
    return as_reported_by_windows - 1;
}

char *clear_filt_names(const char *filt)
{
    char *cleared = SDL_strdup(filt);

    for (char *c = cleared; *c; c++) {
        /* 0x01 bytes are used as temporary replacement for the various 0x00
           bytes required by Win32 (one null byte between each filter, two at
           the end of the filters). Filter out these bytes from the filter names
           to avoid early-ending the filters if someone puts two consecutive
           0x01 bytes in their filter names. */
        if (*c == '\x01') {
            *c = ' ';
        }
    }

    return cleared;
}

// TODO: The new version of file dialogs
void windows_ShowFileDialog(void *ptr)
{
    winArgs *args = (winArgs *) ptr;
    bool is_save = args->is_save;
    const char *default_file = args->default_file;
    SDL_Window *parent = args->parent;
    DWORD flags = args->flags;
    SDL_DialogFileCallback callback = args->callback;
    void *userdata = args->userdata;
    const char *title = args->title;
    wchar_t *filter_wchar = args->filters_str;

    /* GetOpenFileName and GetSaveFileName have the same signature
       (yes, LPOPENFILENAMEW even for the save dialog) */
    typedef BOOL (WINAPI *pfnGetAnyFileNameW)(LPOPENFILENAMEW);
    typedef DWORD (WINAPI *pfnCommDlgExtendedError)(void);
    HMODULE lib = LoadLibraryW(L"Comdlg32.dll");
    pfnGetAnyFileNameW pGetAnyFileName = NULL;
    pfnCommDlgExtendedError pCommDlgExtendedError = NULL;

    if (lib) {
        pGetAnyFileName = (pfnGetAnyFileNameW) GetProcAddress(lib, is_save ? "GetSaveFileNameW" : "GetOpenFileNameW");
        pCommDlgExtendedError = (pfnCommDlgExtendedError) GetProcAddress(lib, "CommDlgExtendedError");
    } else {
        SDL_SetError("Couldn't load Comdlg32.dll");
        callback(userdata, NULL, -1);
        return;
    }

    if (!pGetAnyFileName) {
        SDL_SetError("Couldn't load GetOpenFileName/GetSaveFileName from library");
        callback(userdata, NULL, -1);
        return;
    }

    if (!pCommDlgExtendedError) {
        SDL_SetError("Couldn't load CommDlgExtendedError from library");
        callback(userdata, NULL, -1);
        return;
    }

    HWND window = NULL;

    if (parent) {
        window = (HWND) SDL_GetPointerProperty(SDL_GetWindowProperties(parent), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    }

    wchar_t *filebuffer; // lpstrFile
    wchar_t initfolder[MAX_PATH] = L""; // lpstrInitialDir

    /* If SELECTLIST_SIZE is too large, putting filebuffer on the stack might
       cause an overflow */
    filebuffer = (wchar_t *) SDL_malloc(SELECTLIST_SIZE * sizeof(wchar_t));

    // Necessary for the return code below
    SDL_memset(filebuffer, 0, SELECTLIST_SIZE * sizeof(wchar_t));

    if (default_file) {
        /* On Windows 10, 11 and possibly others, lpstrFile can be initialized
           with a path and the dialog will start at that location, but *only if
           the path contains a filename*. If it ends with a folder (directory
           separator), it fails with 0x3002 (12290) FNERR_INVALIDFILENAME. For
           that specific case, lpstrInitialDir must be used instead, but just
           for that case, because lpstrInitialDir doesn't support file names.

           On top of that, lpstrInitialDir hides a special algorithm that
           decides which folder to actually use as starting point, which may or
           may not be the one provided, or some other unrelated folder. Also,
           the algorithm changes between platforms. Assuming the documentation
           is correct, the algorithm is there under 'lpstrInitialDir':

           https://learn.microsoft.com/en-us/windows/win32/api/commdlg/ns-commdlg-openfilenamew

           Finally, lpstrFile does not support forward slashes. lpstrInitialDir
           does, though. */

        char last_c = default_file[SDL_strlen(default_file) - 1];

        if (last_c == '\\' || last_c == '/') {
            MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, default_file, -1, initfolder, MAX_PATH);
        } else {
            MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, default_file, -1, filebuffer, MAX_PATH);

            for (int i = 0; i < SELECTLIST_SIZE; i++) {
                if (filebuffer[i] == L'/') {
                    filebuffer[i] = L'\\';
                }
            }
        }
    }

    wchar_t *title_w = NULL;

    if (title) {
        title_w = WIN_UTF8ToStringW(title);
        if (!title_w) {
            SDL_free(filebuffer);
            callback(userdata, NULL, -1);
            return;
        }
    }

    OPENFILENAMEW dialog;
    dialog.lStructSize = sizeof(OPENFILENAME);
    dialog.hwndOwner = window;
    dialog.hInstance = 0;
    dialog.lpstrFilter = filter_wchar;
    dialog.lpstrCustomFilter = NULL;
    dialog.nMaxCustFilter = 0;
    dialog.nFilterIndex = 0;
    dialog.lpstrFile = filebuffer;
    dialog.nMaxFile = SELECTLIST_SIZE;
    dialog.lpstrFileTitle = NULL;
    dialog.lpstrInitialDir = *initfolder ? initfolder : NULL;
    dialog.lpstrTitle = title_w;
    dialog.Flags = flags | OFN_EXPLORER | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
    dialog.nFileOffset = 0;
    dialog.nFileExtension = 0;
    dialog.lpstrDefExt = NULL;
    dialog.lCustData = 0;
    dialog.lpfnHook = NULL;
    dialog.lpTemplateName = NULL;
    // Skipped many mac-exclusive and reserved members
    dialog.FlagsEx = 0;

    BOOL result = pGetAnyFileName(&dialog);

    SDL_free(title_w);

    if (result) {
        if (!(flags & OFN_ALLOWMULTISELECT)) {
            // File is a C string stored in dialog.lpstrFile
            char *chosen_file = WIN_StringToUTF8W(dialog.lpstrFile);
            const char *opts[2] = { chosen_file, NULL };
            callback(userdata, opts, getFilterIndex(dialog.nFilterIndex));
            SDL_free(chosen_file);
        } else {
            /* File is either a C string if the user chose a single file, else
               it's a series of strings formatted like:

                   "C:\\path\\to\\folder\0filename1.ext\0filename2.ext\0\0"

               The code below will only stop on a double NULL in all cases, so
               it is important that the rest of the buffer has been zeroed. */
            char chosen_folder[MAX_PATH];
            char chosen_file[MAX_PATH];
            wchar_t *file_ptr = dialog.lpstrFile;
            size_t nfiles = 0;
            size_t chosen_folder_size;
            char **chosen_files_list = (char **) SDL_malloc(sizeof(char *) * (nfiles + 1));

            if (!chosen_files_list) {
                callback(userdata, NULL, -1);
                SDL_free(filebuffer);
                return;
            }

            chosen_files_list[nfiles] = NULL;

            if (WideCharToMultiByte(CP_UTF8, 0, file_ptr, -1, chosen_folder, MAX_PATH, NULL, NULL) >= MAX_PATH) {
                SDL_SetError("Path too long or invalid character in path");
                SDL_free(chosen_files_list);
                callback(userdata, NULL, -1);
                SDL_free(filebuffer);
                return;
            }

            chosen_folder_size = SDL_strlen(chosen_folder);
            SDL_strlcpy(chosen_file, chosen_folder, MAX_PATH);
            chosen_file[chosen_folder_size] = '\\';

            file_ptr += SDL_strlen(chosen_folder) + 1;

            while (*file_ptr) {
                nfiles++;
                char **new_cfl = (char **) SDL_realloc(chosen_files_list, sizeof(char*) * (nfiles + 1));

                if (!new_cfl) {
                    for (size_t i = 0; i < nfiles - 1; i++) {
                        SDL_free(chosen_files_list[i]);
                    }

                    SDL_free(chosen_files_list);
                    callback(userdata, NULL, -1);
                    SDL_free(filebuffer);
                    return;
                }

                chosen_files_list = new_cfl;
                chosen_files_list[nfiles] = NULL;

                int diff = ((int) chosen_folder_size) + 1;

                if (WideCharToMultiByte(CP_UTF8, 0, file_ptr, -1, chosen_file + diff, MAX_PATH - diff, NULL, NULL) >= MAX_PATH - diff) {
                    SDL_SetError("Path too long or invalid character in path");

                    for (size_t i = 0; i < nfiles - 1; i++) {
                        SDL_free(chosen_files_list[i]);
                    }

                    SDL_free(chosen_files_list);
                    callback(userdata, NULL, -1);
                    SDL_free(filebuffer);
                    return;
                }

                file_ptr += SDL_strlen(chosen_file) + 1 - diff;

                chosen_files_list[nfiles - 1] = SDL_strdup(chosen_file);

                if (!chosen_files_list[nfiles - 1]) {
                    for (size_t i = 0; i < nfiles - 1; i++) {
                        SDL_free(chosen_files_list[i]);
                    }

                    SDL_free(chosen_files_list);
                    callback(userdata, NULL, -1);
                    SDL_free(filebuffer);
                    return;
                }
            }

            // If the user chose only one file, it's all just one string
            if (nfiles == 0) {
                nfiles++;
                char **new_cfl = (char **) SDL_realloc(chosen_files_list, sizeof(char*) * (nfiles + 1));

                if (!new_cfl) {
                    SDL_free(chosen_files_list);
                    callback(userdata, NULL, -1);
                    SDL_free(filebuffer);
                    return;
                }

                chosen_files_list = new_cfl;
                chosen_files_list[nfiles] = NULL;
                chosen_files_list[nfiles - 1] = SDL_strdup(chosen_folder);

                if (!chosen_files_list[nfiles - 1]) {
                    SDL_free(chosen_files_list);
                    callback(userdata, NULL, -1);
                    SDL_free(filebuffer);
                    return;
                }
            }

            callback(userdata, (const char * const*) chosen_files_list, getFilterIndex(dialog.nFilterIndex));

            for (size_t i = 0; i < nfiles; i++) {
                SDL_free(chosen_files_list[i]);
            }

            SDL_free(chosen_files_list);
        }
    } else {
        DWORD error = pCommDlgExtendedError();
        // Error code 0 means the user clicked the cancel button.
        if (error == 0) {
            /* Unlike SDL's handling of errors, Windows does reset the error
               code to 0 after calling GetOpenFileName if another Windows
               function before set a different error code, so it's safe to
               check for success. */
            const char *opts[1] = { NULL };
            callback(userdata, opts, getFilterIndex(dialog.nFilterIndex));
        } else {
            SDL_SetError("Windows error, CommDlgExtendedError: %ld", pCommDlgExtendedError());
            callback(userdata, NULL, -1);
        }
    }

    SDL_free(filebuffer);
}

int windows_file_dialog_thread(void *ptr)
{
    windows_ShowFileDialog(ptr);
    freeWinArgs(ptr);
    return 0;
}

int CALLBACK browse_callback_proc(HWND hwnd, UINT uMsg, LPARAM lParam, LPARAM lpData)
{
    switch (uMsg) {
    case BFFM_INITIALIZED:
        if (lpData) {
            SendMessage(hwnd, BFFM_SETSELECTION, TRUE, lpData);
        }
        break;
    case BFFM_SELCHANGED:
        break;
    case BFFM_VALIDATEFAILED:
        break;
    default:
        break;
    }
    return 0;
}

void windows_ShowFolderDialog(void *ptr)
{
    winFArgs *args = (winFArgs *) ptr;
    SDL_Window *window = args->parent;
    SDL_DialogFileCallback callback = args->callback;
    void *userdata = args->userdata;
    HWND parent = NULL;
    const char *title = args->title;

    if (window) {
        parent = (HWND) SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    }

    wchar_t *title_w = NULL;

    if (title) {
        title_w = WIN_UTF8ToStringW(title);
        if (!title_w) {
            callback(userdata, NULL, -1);
            return;
        }
    }

    wchar_t buffer[MAX_PATH];

    BROWSEINFOW dialog;
    dialog.hwndOwner = parent;
    dialog.pidlRoot = NULL;
    dialog.pszDisplayName = buffer;
    dialog.lpszTitle = title_w;
    dialog.ulFlags = BIF_USENEWUI;
    dialog.lpfn = browse_callback_proc;
    dialog.lParam = (LPARAM)args->default_folder;
    dialog.iImage = 0;

    LPITEMIDLIST lpItem = SHBrowseForFolderW(&dialog);

    SDL_free(title_w);

    if (lpItem != NULL) {
        SHGetPathFromIDListW(lpItem, buffer);
        char *chosen_file = WIN_StringToUTF8W(buffer);
        const char *files[2] = { chosen_file, NULL };
        callback(userdata, (const char * const*) files, -1);
        SDL_free(chosen_file);
    } else {
        const char *files[1] = { NULL };
        callback(userdata, (const char * const*) files, -1);
    }
}

int windows_folder_dialog_thread(void *ptr)
{
    windows_ShowFolderDialog(ptr);
    freeWinFArgs((winFArgs *)ptr);
    return 0;
}

wchar_t *win_get_filters(const SDL_DialogFileFilter *filters, int nfilters)
{
    wchar_t *filter_wchar = NULL;

    if (filters) {
        // '\x01' is used in place of a null byte
        // suffix needs two null bytes in case the filter list is empty
        char *filterlist = convert_filters(filters, nfilters, clear_filt_names,
                                           "", "", "\x01\x01", "", "\x01",
                                           "\x01", "*.", ";*.", "");

        if (!filterlist) {
            return NULL;
        }

        int filter_len = (int)SDL_strlen(filterlist);

        for (char *c = filterlist; *c; c++) {
            if (*c == '\x01') {
                *c = '\0';
            }
        }

        int filter_wlen = MultiByteToWideChar(CP_UTF8, 0, filterlist, filter_len, NULL, 0);
        filter_wchar = (wchar_t *)SDL_malloc(filter_wlen * sizeof(wchar_t));
        if (!filter_wchar) {
            SDL_free(filterlist);
            return NULL;
        }

        MultiByteToWideChar(CP_UTF8, 0, filterlist, filter_len, filter_wchar, filter_wlen);

        SDL_free(filterlist);
    }

    return filter_wchar;
}

static void ShowFileDialog(SDL_DialogFileCallback callback, void *userdata, SDL_Window *window, const SDL_DialogFileFilter *filters, int nfilters, const char *default_location, bool allow_many, bool is_save, const char *title, const char *accept, const char *cancel)
{
    winArgs *args;
    SDL_Thread *thread;
    wchar_t *filters_str;

    if (SDL_GetHint(SDL_HINT_FILE_DIALOG_DRIVER) != NULL) {
        SDL_SetError("File dialog driver unsupported");
        callback(userdata, NULL, -1);
        return;
    }

    args = (winArgs *)SDL_malloc(sizeof(*args));
    if (args == NULL) {
        callback(userdata, NULL, -1);
        return;
    }

    filters_str = win_get_filters(filters, nfilters);

    DWORD flags = 0;
    if (allow_many) {
        flags |= OFN_ALLOWMULTISELECT;
    }
    if (is_save) {
        flags |= OFN_OVERWRITEPROMPT;
    }

    if (!filters_str && filters) {
        callback(userdata, NULL, -1);
        SDL_free(args);
        return;
    }

    args->is_save = is_save;
    args->filters_str = filters_str;
    args->default_file = default_location ? SDL_strdup(default_location) : NULL;
    args->parent = window;
    args->flags = flags;
    args->callback = callback;
    args->userdata = userdata;
    args->title = title ? SDL_strdup(title) : NULL;
    args->accept = accept ? SDL_strdup(accept) : NULL;
    args->cancel = cancel ? SDL_strdup(cancel) : NULL;

    thread = SDL_CreateThread(windows_file_dialog_thread, "SDL_Windows_ShowFileDialog", (void *) args);

    if (thread == NULL) {
        callback(userdata, NULL, -1);
        // The thread won't have run, therefore the data won't have been freed
        freeWinArgs(args);
        return;
    }

    SDL_DetachThread(thread);
}

void ShowFolderDialog(SDL_DialogFileCallback callback, void *userdata, SDL_Window *window, const char *default_location, bool allow_many, const char *title, const char *accept, const char *cancel)
{
    winFArgs *args;
    SDL_Thread *thread;

    if (SDL_GetHint(SDL_HINT_FILE_DIALOG_DRIVER) != NULL) {
        SDL_SetError("File dialog driver unsupported");
        callback(userdata, NULL, -1);
        return;
    }

    args = (winFArgs *)SDL_malloc(sizeof(*args));
    if (args == NULL) {
        callback(userdata, NULL, -1);
        return;
    }

    args->parent = window;
    args->callback = callback;
    args->default_folder = default_location ? SDL_strdup(default_location) : NULL;
    args->userdata = userdata;
    args->title = title ? SDL_strdup(title) : NULL;
    args->accept = accept ? SDL_strdup(accept) : NULL;
    args->cancel = cancel ? SDL_strdup(cancel) : NULL;

    thread = SDL_CreateThread(windows_folder_dialog_thread, "SDL_Windows_ShowFolderDialog", (void *) args);

    if (thread == NULL) {
        callback(userdata, NULL, -1);
        // The thread won't have run, therefore the data won't have been freed
        freeWinFArgs(args);
        return;
    }

    SDL_DetachThread(thread);
}

void SDL_SYS_ShowFileDialogWithProperties(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    /* The internal functions will start threads, and the properties may be freed as soon as this function returns.
       Save a copy of what we need before invoking the functions and starting the threads. */
    SDL_Window *window = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, NULL);
    SDL_DialogFileFilter *filters = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, NULL);
    int nfilters = (int) SDL_GetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, 0);
    bool allow_many = SDL_GetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, false);
    const char *default_location = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, NULL);
    const char *title = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_TITLE_STRING, NULL);
    const char *accept = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_ACCEPT_STRING, NULL);
    const char *cancel = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_CANCEL_STRING, NULL);
    bool is_save = false;

    switch (type) {
    case SDL_FILEDIALOG_SAVEFILE:
        is_save = true;
        SDL_FALLTHROUGH;
    case SDL_FILEDIALOG_OPENFILE:
        ShowFileDialog(callback, userdata, window, filters, nfilters, default_location, allow_many, is_save, title, accept, cancel);
        break;

    case SDL_FILEDIALOG_OPENFOLDER:
        ShowFolderDialog(callback, userdata, window, default_location, allow_many, title, accept, cancel);
        break;
    };
}
