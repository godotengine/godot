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

#ifdef SDL_PLATFORM_MACOS

#include <Cocoa/Cocoa.h>

#include "../SDL_tray_utils.h"
#include "../../video/SDL_surface_c.h"

/* applicationDockMenu */

struct SDL_TrayMenu {
    NSMenu *nsmenu;

    int nEntries;
    SDL_TrayEntry **entries;

    SDL_Tray *parent_tray;
    SDL_TrayEntry *parent_entry;
};

struct SDL_TrayEntry {
    NSMenuItem *nsitem;

    SDL_TrayEntryFlags flags;
    SDL_TrayCallback callback;
    void *userdata;
    SDL_TrayMenu *submenu;

    SDL_TrayMenu *parent;
};

struct SDL_Tray {
    NSStatusBar *statusBar;
    NSStatusItem *statusItem;

    SDL_TrayMenu *menu;
};

static void DestroySDLMenu(SDL_TrayMenu *menu)
{
    for (int i = 0; i < menu->nEntries; i++) {
        if (menu->entries[i] && menu->entries[i]->submenu) {
            DestroySDLMenu(menu->entries[i]->submenu);
        }
        SDL_free(menu->entries[i]);
    }

    SDL_free(menu->entries);

    if (menu->parent_entry) {
        [menu->parent_entry->parent->nsmenu setSubmenu:nil forItem:menu->parent_entry->nsitem];
    } else if (menu->parent_tray) {
        [menu->parent_tray->statusItem setMenu:nil];
    }

    SDL_free(menu);
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

    if (icon) {
        icon = SDL_ConvertSurface(icon, SDL_PIXELFORMAT_RGBA32);
        if (!icon) {
            return NULL;
        }
    }

    SDL_Tray *tray = (SDL_Tray *)SDL_calloc(1, sizeof(*tray));
    if (!tray) {
        SDL_DestroySurface(icon);
        return NULL;
    }

    tray->statusItem = nil;
    tray->statusBar = [NSStatusBar systemStatusBar];
    tray->statusItem = [tray->statusBar statusItemWithLength:NSVariableStatusItemLength];
    [[NSApplication sharedApplication] activateIgnoringOtherApps:TRUE];

    if (tooltip) {
        tray->statusItem.button.toolTip = [NSString stringWithUTF8String:tooltip];
    } else {
        tray->statusItem.button.toolTip = nil;
    }

    if (icon) {
        NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:(unsigned char **)&icon->pixels
                                                                           pixelsWide:icon->w
                                                                           pixelsHigh:icon->h
                                                                        bitsPerSample:8
                                                                      samplesPerPixel:4
                                                                             hasAlpha:YES
                                                                             isPlanar:NO
                                                                       colorSpaceName:NSCalibratedRGBColorSpace
                                                                          bytesPerRow:icon->pitch
                                                                         bitsPerPixel:32];
        NSImage *iconimg = [[NSImage alloc] initWithSize:NSMakeSize(icon->w, icon->h)];
        [iconimg addRepresentation:bitmap];

        /* A typical icon size is 22x22 on macOS. Failing to resize the icon
           may give oversized status bar buttons. */
        NSImage *iconimg22 = [[NSImage alloc] initWithSize:NSMakeSize(22, 22)];
        [iconimg22 lockFocus];
        [iconimg setSize:NSMakeSize(22, 22)];
        [iconimg drawInRect:NSMakeRect(0, 0, 22, 22)];
        [iconimg22 unlockFocus];

        tray->statusItem.button.image = iconimg22;

        SDL_DestroySurface(icon);
    }

    SDL_RegisterTray(tray);

    return tray;
}

void SDL_SetTrayIcon(SDL_Tray *tray, SDL_Surface *icon)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        return;
    }

    if (!icon) {
        tray->statusItem.button.image = nil;
        return;
    }

    icon = SDL_ConvertSurface(icon, SDL_PIXELFORMAT_RGBA32);
    if (!icon) {
        tray->statusItem.button.image = nil;
        return;
    }

    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:(unsigned char **)&icon->pixels
                                                                       pixelsWide:icon->w
                                                                       pixelsHigh:icon->h
                                                                    bitsPerSample:8
                                                                  samplesPerPixel:4
                                                                         hasAlpha:YES
                                                                         isPlanar:NO
                                                                   colorSpaceName:NSCalibratedRGBColorSpace
                                                                      bytesPerRow:icon->pitch
                                                                     bitsPerPixel:32];
    NSImage *iconimg = [[NSImage alloc] initWithSize:NSMakeSize(icon->w, icon->h)];
    [iconimg addRepresentation:bitmap];

    /* A typical icon size is 22x22 on macOS. Failing to resize the icon
       may give oversized status bar buttons. */
    NSImage *iconimg22 = [[NSImage alloc] initWithSize:NSMakeSize(22, 22)];
    [iconimg22 lockFocus];
    [iconimg setSize:NSMakeSize(22, 22)];
    [iconimg drawInRect:NSMakeRect(0, 0, 22, 22)];
    [iconimg22 unlockFocus];

    tray->statusItem.button.image = iconimg22;

    SDL_DestroySurface(icon);
}

void SDL_SetTrayTooltip(SDL_Tray *tray, const char *tooltip)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        return;
    }

    if (tooltip) {
        tray->statusItem.button.toolTip = [NSString stringWithUTF8String:tooltip];
    } else {
        tray->statusItem.button.toolTip = nil;
    }
}

SDL_TrayMenu *SDL_CreateTrayMenu(SDL_Tray *tray)
{
    if (!SDL_ObjectValid(tray, SDL_OBJECT_TYPE_TRAY)) {
        SDL_InvalidParamError("tray");
        return NULL;
    }

    SDL_TrayMenu *menu = (SDL_TrayMenu *)SDL_calloc(1, sizeof(*menu));
    if (!menu) {
        return NULL;
    }

    NSMenu *nsmenu = [[NSMenu alloc] init];
    [nsmenu setAutoenablesItems:FALSE];

    [tray->statusItem setMenu:nsmenu];

    tray->menu = menu;
    menu->nsmenu = nsmenu;
    menu->nEntries = 0;
    menu->entries = NULL;
    menu->parent_tray = tray;
    menu->parent_entry = NULL;

    return menu;
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

    if (entry->submenu) {
        SDL_SetError("Tray entry submenu already exists");
        return NULL;
    }

    if (!(entry->flags & SDL_TRAYENTRY_SUBMENU)) {
        SDL_SetError("Cannot create submenu for entry not created with SDL_TRAYENTRY_SUBMENU");
        return NULL;
    }

    SDL_TrayMenu *menu = (SDL_TrayMenu *)SDL_calloc(1, sizeof(*menu));
    if (!menu) {
        return NULL;
    }

    NSMenu *nsmenu = [[NSMenu alloc] init];
    [nsmenu setAutoenablesItems:FALSE];

    entry->submenu = menu;
    menu->nsmenu = nsmenu;
    menu->nEntries = 0;
    menu->entries = NULL;
    menu->parent_tray = NULL;
    menu->parent_entry = entry;

    [entry->parent->nsmenu setSubmenu:nsmenu forItem:entry->nsitem];

    return menu;
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

    [menu->nsmenu removeItem:entry->nsitem];

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

    if (pos == -1) {
        pos = menu->nEntries;
    }

    SDL_TrayEntry *entry = (SDL_TrayEntry *)SDL_calloc(1, sizeof(*entry));
    if (!entry) {
        return NULL;
    }

    SDL_TrayEntry **new_entries = (SDL_TrayEntry **)SDL_realloc(menu->entries, (menu->nEntries + 2) * sizeof(*new_entries));
    if (!new_entries) {
        SDL_free(entry);
        return NULL;
    }

    menu->entries = new_entries;
    menu->nEntries++;

    for (int i = menu->nEntries - 1; i > pos; i--) {
        menu->entries[i] = menu->entries[i - 1];
    }

    new_entries[pos] = entry;
    new_entries[menu->nEntries] = NULL;

    NSMenuItem *nsitem;
    if (label == NULL) {
        nsitem = [NSMenuItem separatorItem];
    } else {
        nsitem = [[NSMenuItem alloc] initWithTitle:[NSString stringWithUTF8String:label] action:@selector(menu:) keyEquivalent:@""];
        [nsitem setEnabled:((flags & SDL_TRAYENTRY_DISABLED) ? FALSE : TRUE)];
        [nsitem setState:((flags & SDL_TRAYENTRY_CHECKED) ? NSControlStateValueOn : NSControlStateValueOff)];
        [nsitem setRepresentedObject:[NSValue valueWithPointer:entry]];
    }

    [menu->nsmenu insertItem:nsitem atIndex:pos];

    entry->nsitem = nsitem;
    entry->flags = flags;
    entry->callback = NULL;
    entry->userdata = NULL;
    entry->submenu = NULL;
    entry->parent = menu;

    return entry;
}

void SDL_SetTrayEntryLabel(SDL_TrayEntry *entry, const char *label)
{
    if (!entry) {
        return;
    }

    [entry->nsitem setTitle:[NSString stringWithUTF8String:label]];
}

const char *SDL_GetTrayEntryLabel(SDL_TrayEntry *entry)
{
    if (!entry) {
        SDL_InvalidParamError("entry");
        return NULL;
    }

    return [[entry->nsitem title] UTF8String];
}

void SDL_SetTrayEntryChecked(SDL_TrayEntry *entry, bool checked)
{
    if (!entry) {
        return;
    }

    [entry->nsitem setState:(checked ? NSControlStateValueOn : NSControlStateValueOff)];
}

bool SDL_GetTrayEntryChecked(SDL_TrayEntry *entry)
{
    if (!entry) {
        return false;
    }

    return entry->nsitem.state == NSControlStateValueOn;
}

void SDL_SetTrayEntryEnabled(SDL_TrayEntry *entry, bool enabled)
{
    if (!entry || !(entry->flags & SDL_TRAYENTRY_CHECKBOX)) {
        return;
    }

    [entry->nsitem setEnabled:(enabled ? YES : NO)];
}

bool SDL_GetTrayEntryEnabled(SDL_TrayEntry *entry)
{
    if (!entry || !(entry->flags & SDL_TRAYENTRY_CHECKBOX)) {
        return false;
    }

    return entry->nsitem.enabled;
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

    [[NSStatusBar systemStatusBar] removeStatusItem:tray->statusItem];

    if (tray->menu) {
        DestroySDLMenu(tray->menu);
    }

    SDL_free(tray);
}

#endif // SDL_PLATFORM_MACOS
