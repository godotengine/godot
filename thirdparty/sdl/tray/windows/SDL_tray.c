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

#include "../SDL_tray_utils.h"
#include "../../core/windows/SDL_windows.h"
#include "../../video/windows/SDL_windowswindow.h"

#include <windowsx.h>
#include <shellapi.h>

#include "../../video/windows/SDL_surface_utils.h"

#ifndef NOTIFYICON_VERSION_4
#define NOTIFYICON_VERSION_4 4
#endif
#ifndef NIF_SHOWTIP
#define NIF_SHOWTIP 0x00000080
#endif

#define WM_TRAYICON (WM_USER + 1)

struct SDL_TrayMenu {
    HMENU hMenu;

    int nEntries;
    SDL_TrayEntry **entries;

    SDL_Tray *parent_tray;
    SDL_TrayEntry *parent_entry;
};

struct SDL_TrayEntry {
    SDL_TrayMenu *parent;
    UINT_PTR id;

    char label_cache[4096];
    SDL_TrayEntryFlags flags;
    SDL_TrayCallback callback;
    void *userdata;
    SDL_TrayMenu *submenu;
};

struct SDL_Tray {
    NOTIFYICONDATAW nid;
    HWND hwnd;
    HICON icon;
    SDL_TrayMenu *menu;
};

static UINT_PTR get_next_id(void)
{
    static UINT_PTR next_id = 0;
    return ++next_id;
}

static SDL_TrayEntry *find_entry_in_menu(SDL_TrayMenu *menu, UINT_PTR id)
{
    for (int i = 0; i < menu->nEntries; i++) {
        SDL_TrayEntry *entry = menu->entries[i];

        if (entry->id == id) {
            return entry;
        }

        if (entry->submenu) {
            SDL_TrayEntry *e = find_entry_in_menu(entry->submenu, id);

            if (e) {
                return e;
            }
        }
    }

    return NULL;
}

static SDL_TrayEntry *find_entry_with_id(SDL_Tray *tray, UINT_PTR id)
{
    if (!tray->menu) {
        return NULL;
    }

    return find_entry_in_menu(tray->menu, id);
}

LRESULT CALLBACK TrayWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    SDL_Tray *tray = (SDL_Tray *) GetWindowLongPtr(hwnd, GWLP_USERDATA);
    SDL_TrayEntry *entry = NULL;

    if (!tray) {
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    switch (uMsg) {
        case WM_TRAYICON:
            if (LOWORD(lParam) == WM_CONTEXTMENU || LOWORD(lParam) == WM_LBUTTONUP) {
                SetForegroundWindow(hwnd);
                
                if (tray->menu) {
                    TrackPopupMenu(tray->menu->hMenu, TPM_BOTTOMALIGN | TPM_RIGHTALIGN, GET_X_LPARAM(wParam), GET_Y_LPARAM(wParam), 0, hwnd, NULL);
                }
            }
            break;

        case WM_COMMAND:
            entry = find_entry_with_id(tray, LOWORD(wParam));

            if (entry && (entry->flags & SDL_TRAYENTRY_CHECKBOX)) {
                SDL_SetTrayEntryChecked(entry, !SDL_GetTrayEntryChecked(entry));
            }

            if (entry && entry->callback) {
                entry->callback(entry->userdata, entry);
            }
            break;

        case WM_SETTINGCHANGE:
            if (wParam == 0 && lParam != 0 && SDL_wcscmp((wchar_t *)lParam, L"ImmersiveColorSet") == 0) {
                WIN_UpdateDarkModeForHWND(hwnd);
            }
            break;

        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

static void DestroySDLMenu(SDL_TrayMenu *menu)
{
    for (int i = 0; i < menu->nEntries; i++) {
        if (menu->entries[i] && menu->entries[i]->submenu) {
            DestroySDLMenu(menu->entries[i]->submenu);
        }
        SDL_free(menu->entries[i]);
    }
    SDL_free(menu->entries);
    DestroyMenu(menu->hMenu);
    SDL_free(menu);
}

static wchar_t *escape_label(const char *in)
{
    const char *c;
    char *c2;
    int len = 0;

    for (c = in; *c; c++) {
        len += (*c == '&') ? 2 : 1;
    }

    char *escaped = (char *)SDL_malloc(SDL_strlen(in) + len + 1);
    if (!escaped) {
        return NULL;
    }

    for (c = in, c2 = escaped; *c;) {
        if (*c == '&') {
            *c2++ = *c;
        }

        *c2++ = *c++;
    }

    *c2 = '\0';

    wchar_t *out = WIN_UTF8ToStringW(escaped);
    SDL_free(escaped);

    return out;
}

static HICON load_default_icon()
{
    HINSTANCE hInstance = GetModuleHandle(NULL);
    if (!hInstance) {
        return LoadIcon(NULL, IDI_APPLICATION);
    }

    const char *hint = SDL_GetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON_SMALL);
    if (hint && *hint) {
        HICON icon = LoadIcon(hInstance, MAKEINTRESOURCE(SDL_atoi(hint)));
        return icon ? icon : LoadIcon(NULL, IDI_APPLICATION);
    }

    hint = SDL_GetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON);
    if (hint && *hint) {
        HICON icon = LoadIcon(hInstance, MAKEINTRESOURCE(SDL_atoi(hint)));
        return icon ? icon : LoadIcon(NULL, IDI_APPLICATION);
    }

    return LoadIcon(NULL, IDI_APPLICATION);
}

void SDL_UpdateTrays(void)
{
}

SDL_Tray *SDL_CreateTray(SDL_Surface *icon, const char *tooltip)
{
    if (!SDL_IsMainThread()) {
        SDL_SetError("This function should be called on the main thread");
        return NULL;
    }

    SDL_Tray *tray = (SDL_Tray *)SDL_calloc(1, sizeof(*tray));

    if (!tray) {
        return NULL;
    }

    tray->menu = NULL;
    tray->hwnd = CreateWindowEx(0, TEXT("Message"), NULL, 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);
    SetWindowLongPtr(tray->hwnd, GWLP_WNDPROC, (LONG_PTR) TrayWindowProc);

    WIN_UpdateDarkModeForHWND(tray->hwnd);

    SDL_zero(tray->nid);
    tray->nid.cbSize = sizeof(NOTIFYICONDATAW);
    tray->nid.hWnd = tray->hwnd;
    tray->nid.uID = (UINT) get_next_id();
    tray->nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP | NIF_SHOWTIP;
    tray->nid.uCallbackMessage = WM_TRAYICON;
    tray->nid.uVersion = NOTIFYICON_VERSION_4;
    wchar_t *tooltipw = WIN_UTF8ToStringW(tooltip);
    SDL_wcslcpy(tray->nid.szTip, tooltipw, sizeof(tray->nid.szTip) / sizeof(*tray->nid.szTip));
    SDL_free(tooltipw);

    if (icon) {
        tray->nid.hIcon = CreateIconFromSurface(icon);

        if (!tray->nid.hIcon) {
            tray->nid.hIcon = load_default_icon();
        }

        tray->icon = tray->nid.hIcon;
    } else {
        tray->nid.hIcon = load_default_icon();
        tray->icon = tray->nid.hIcon;
    }

    Shell_NotifyIconW(NIM_ADD, &tray->nid);
    Shell_NotifyIconW(NIM_SETVERSION, &tray->nid);

    SetWindowLongPtr(tray->hwnd, GWLP_USERDATA, (LONG_PTR) tray);

    SDL_RegisterTray(tray);

    return tray;
}

void SDL_SetTrayIcon(SDL_Tray *tray, SDL_Surface *icon)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        return;
    }

    if (tray->icon) {
        DestroyIcon(tray->icon);
    }

    if (icon) {
        tray->nid.hIcon = CreateIconFromSurface(icon);

        if (!tray->nid.hIcon) {
            tray->nid.hIcon = load_default_icon();
        }

        tray->icon = tray->nid.hIcon;
    } else {
        tray->nid.hIcon = load_default_icon();
        tray->icon = tray->nid.hIcon;
    }

    Shell_NotifyIconW(NIM_MODIFY, &tray->nid);
}

void SDL_SetTrayTooltip(SDL_Tray *tray, const char *tooltip)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        return;
    }

    if (tooltip) {
        wchar_t *tooltipw = WIN_UTF8ToStringW(tooltip);
        SDL_wcslcpy(tray->nid.szTip, tooltipw, sizeof(tray->nid.szTip) / sizeof(*tray->nid.szTip));
        SDL_free(tooltipw);
    } else {
        tray->nid.szTip[0] = '\0';
    }

    Shell_NotifyIconW(NIM_MODIFY, &tray->nid);
}

SDL_TrayMenu *SDL_CreateTrayMenu(SDL_Tray *tray)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        SDL_InvalidParamError("tray");
        return NULL;
    }

    tray->menu = (SDL_TrayMenu *)SDL_calloc(1, sizeof(*tray->menu));

    if (!tray->menu) {
        return NULL;
    }

    tray->menu->hMenu = CreatePopupMenu();
    tray->menu->parent_tray = tray;
    tray->menu->parent_entry = NULL;

    return tray->menu;
}

SDL_TrayMenu *SDL_GetTrayMenu(SDL_Tray *tray)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        SDL_InvalidParamError("tray");
        return NULL;
    }

    return tray->menu;
}

SDL_TrayMenu *SDL_CreateTraySubmenu(SDL_TrayEntry *entry)
{
    if (!entry) {
        SDL_InvalidParamError("entry");
        return NULL;
    }

    if (!entry->submenu) {
        SDL_SetError("Cannot create submenu for entry not created with SDL_TRAYENTRY_SUBMENU");
        return NULL;
    }

    return entry->submenu;
}

SDL_TrayMenu *SDL_GetTraySubmenu(SDL_TrayEntry *entry)
{
    if (!entry) {
        SDL_InvalidParamError("entry");
        return NULL;
    }

    return entry->submenu;
}

const SDL_TrayEntry **SDL_GetTrayEntries(SDL_TrayMenu *menu, int *count)
{
    if (!menu) {
        SDL_InvalidParamError("menu");
        return NULL;
    }

    if (count) {
        *count = menu->nEntries;
    }
    return (const SDL_TrayEntry **)menu->entries;
}

void SDL_RemoveTrayEntry(SDL_TrayEntry *entry)
{
    if (!entry) {
        return;
    }

    SDL_TrayMenu *menu = entry->parent;

    bool found = false;
    for (int i = 0; i < menu->nEntries - 1; i++) {
        if (menu->entries[i] == entry) {
            found = true;
        }

        if (found) {
            menu->entries[i] = menu->entries[i + 1];
        }
    }

    if (entry->submenu) {
        DestroySDLMenu(entry->submenu);
    }

    menu->nEntries--;
    SDL_TrayEntry **new_entries = (SDL_TrayEntry **)SDL_realloc(menu->entries, (menu->nEntries + 1) * sizeof(*new_entries));

    /* Not sure why shrinking would fail, but even if it does, we can live with a "too big" array */
    if (new_entries) {
        menu->entries = new_entries;
        menu->entries[menu->nEntries] = NULL;
    }

    if (!DeleteMenu(menu->hMenu, (UINT) entry->id, MF_BYCOMMAND)) {
        /* This is somewhat useless since we don't return anything, but might help with eventual bugs */
        SDL_SetError("Couldn't destroy tray entry");
    }

    SDL_free(entry);
}

SDL_TrayEntry *SDL_InsertTrayEntryAt(SDL_TrayMenu *menu, int pos, const char *label, SDL_TrayEntryFlags flags)
{
    if (!menu) {
        SDL_InvalidParamError("menu");
        return NULL;
    }

    if (pos < -1 || pos > menu->nEntries) {
        SDL_InvalidParamError("pos");
        return NULL;
    }

    int windows_compatible_pos = pos;

    if (pos == -1) {
        pos = menu->nEntries;
    } else if (pos == menu->nEntries) {
        windows_compatible_pos = -1;
    }

    SDL_TrayEntry *entry = (SDL_TrayEntry *)SDL_calloc(1, sizeof(*entry));
    if (!entry) {
        return NULL;
    }

    wchar_t *label_w = NULL;

    if (label && (label_w = escape_label(label)) == NULL) {
        SDL_free(entry);
        return NULL;
    }

    entry->parent = menu;
    entry->flags = flags;
    entry->callback = NULL;
    entry->userdata = NULL;
    entry->submenu = NULL;
    SDL_snprintf(entry->label_cache, sizeof(entry->label_cache), "%s", label ? label : "");

    if (label != NULL && flags & SDL_TRAYENTRY_SUBMENU) {
        entry->submenu = (SDL_TrayMenu *)SDL_calloc(1, sizeof(*entry->submenu));
        if (!entry->submenu) {
            SDL_free(entry);
            SDL_free(label_w);
            return NULL;
        }

        entry->submenu->hMenu = CreatePopupMenu();
        entry->submenu->nEntries = 0;
        entry->submenu->entries = NULL;
        entry->submenu->parent_entry = entry;
        entry->submenu->parent_tray = NULL;

        entry->id = (UINT_PTR) entry->submenu->hMenu;
    } else {
        entry->id = get_next_id();
    }

    SDL_TrayEntry **new_entries = (SDL_TrayEntry **)SDL_realloc(menu->entries, (menu->nEntries + 2) * sizeof(*new_entries));
    if (!new_entries) {
        SDL_free(entry);
        SDL_free(label_w);
        if (entry->submenu) {
            DestroyMenu(entry->submenu->hMenu);
            SDL_free(entry->submenu);
        }
        return NULL;
    }

    menu->entries = new_entries;
    menu->nEntries++;

    for (int i = menu->nEntries - 1; i > pos; i--) {
        menu->entries[i] = menu->entries[i - 1];
    }

    new_entries[pos] = entry;
    new_entries[menu->nEntries] = NULL;

    if (label == NULL) {
        InsertMenuW(menu->hMenu, windows_compatible_pos, MF_SEPARATOR | MF_BYPOSITION, entry->id, NULL);
    } else {
        UINT mf = MF_STRING | MF_BYPOSITION;
        if (flags & SDL_TRAYENTRY_SUBMENU) {
            mf = MF_POPUP;
        }

        if (flags & SDL_TRAYENTRY_DISABLED) {
            mf |= MF_DISABLED | MF_GRAYED;
        }

        if (flags & SDL_TRAYENTRY_CHECKED) {
            mf |= MF_CHECKED;
        }

        InsertMenuW(menu->hMenu, windows_compatible_pos, mf, entry->id, label_w);

        SDL_free(label_w);
    }

    return entry;
}

void SDL_SetTrayEntryLabel(SDL_TrayEntry *entry, const char *label)
{
    if (!entry) {
        return;
    }

    SDL_snprintf(entry->label_cache, sizeof(entry->label_cache), "%s", label);

    wchar_t *label_w = escape_label(label);

    if (!label_w) {
        return;
    }

    MENUITEMINFOW mii;
    mii.cbSize = sizeof(MENUITEMINFOW);
    mii.fMask = MIIM_STRING;

    mii.dwTypeData = label_w;
    mii.cch = (UINT) SDL_wcslen(label_w);

    if (!SetMenuItemInfoW(entry->parent->hMenu, (UINT) entry->id, TRUE, &mii)) {
        SDL_SetError("Couldn't update tray entry label");
    }

    SDL_free(label_w);
}

const char *SDL_GetTrayEntryLabel(SDL_TrayEntry *entry)
{
    if (!entry) {
        SDL_InvalidParamError("entry");
        return NULL;
    }

    return entry->label_cache;
}

void SDL_SetTrayEntryChecked(SDL_TrayEntry *entry, bool checked)
{
    if (!entry || !(entry->flags & SDL_TRAYENTRY_CHECKBOX)) {
        return;
    }

    CheckMenuItem(entry->parent->hMenu, (UINT) entry->id, checked ? MF_CHECKED : MF_UNCHECKED);
}

bool SDL_GetTrayEntryChecked(SDL_TrayEntry *entry)
{
    if (!entry || !(entry->flags & SDL_TRAYENTRY_CHECKBOX)) {
        return false;
    }

    MENUITEMINFOW mii;
    mii.cbSize = sizeof(MENUITEMINFOW);
    mii.fMask = MIIM_STATE;

    GetMenuItemInfoW(entry->parent->hMenu, (UINT) entry->id, FALSE, &mii);

    return ((mii.fState & MFS_CHECKED) != 0);
}

void SDL_SetTrayEntryEnabled(SDL_TrayEntry *entry, bool enabled)
{
    if (!entry) {
        return;
    }

    EnableMenuItem(entry->parent->hMenu, (UINT) entry->id, MF_BYCOMMAND | (enabled ? MF_ENABLED : (MF_DISABLED | MF_GRAYED)));
}

bool SDL_GetTrayEntryEnabled(SDL_TrayEntry *entry)
{
    if (!entry) {
        return false;
    }

    MENUITEMINFOW mii;
    mii.cbSize = sizeof(MENUITEMINFOW);
    mii.fMask = MIIM_STATE;

    GetMenuItemInfoW(entry->parent->hMenu, (UINT) entry->id, FALSE, &mii);

    return ((mii.fState & MFS_ENABLED) != 0);
}

void SDL_SetTrayEntryCallback(SDL_TrayEntry *entry, SDL_TrayCallback callback, void *userdata)
{
    if (!entry) {
        return;
    }

    entry->callback = callback;
    entry->userdata = userdata;
}

void SDL_ClickTrayEntry(SDL_TrayEntry *entry)
{
	if (!entry) {
		return;
	}

	if (entry->flags & SDL_TRAYENTRY_CHECKBOX) {
		SDL_SetTrayEntryChecked(entry, !SDL_GetTrayEntryChecked(entry));
	}

	if (entry->callback) {
		entry->callback(entry->userdata, entry);
	}
}

SDL_TrayMenu *SDL_GetTrayEntryParent(SDL_TrayEntry *entry)
{
    if (!entry) {
        SDL_InvalidParamError("entry");
        return NULL;
    }

    return entry->parent;
}

SDL_TrayEntry *SDL_GetTrayMenuParentEntry(SDL_TrayMenu *menu)
{
    if (!menu) {
        SDL_InvalidParamError("menu");
        return NULL;
    }

    return menu->parent_entry;
}

SDL_Tray *SDL_GetTrayMenuParentTray(SDL_TrayMenu *menu)
{
    if (!menu) {
        SDL_InvalidParamError("menu");
        return NULL;
    }

    return menu->parent_tray;
}

void SDL_DestroyTray(SDL_Tray *tray)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        return;
    }

    SDL_UnregisterTray(tray);

    Shell_NotifyIconW(NIM_DELETE, &tray->nid);

    if (tray->menu) {
        DestroySDLMenu(tray->menu);
    }

    if (tray->icon) {
        DestroyIcon(tray->icon);
    }

    if (tray->hwnd) {
        DestroyWindow(tray->hwnd);
    }

    SDL_free(tray);
}
