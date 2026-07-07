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

/**
 * # CategoryTray
 *
 * SDL offers a way to add items to the "system tray" (more correctly called
 * the "notification area" on Windows). On platforms that offer this concept,
 * an SDL app can add a tray icon, submenus, checkboxes, and clickable
 * entries, and register a callback that is fired when the user clicks on
 * these pieces.
 */

#ifndef SDL_tray_h_
#define SDL_tray_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_video.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * An opaque handle representing a toplevel system tray object.
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_Tray SDL_Tray;

/**
 * An opaque handle representing a menu/submenu on a system tray object.
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_TrayMenu SDL_TrayMenu;

/**
 * An opaque handle representing an entry on a system tray object.
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_TrayEntry SDL_TrayEntry;

/**
 * Flags that control the creation of system tray entries.
 *
 * Some of these flags are required; exactly one of them must be specified at
 * the time a tray entry is created. Other flags are optional; zero or more of
 * those can be OR'ed together with the required flag.
 *
 * \since This datatype is available since SDL 3.2.0.
 *
 * \sa SDL_InsertTrayEntryAt
 */
typedef Uint32 SDL_TrayEntryFlags;

#define SDL_TRAYENTRY_BUTTON      0x00000001u /**< Make the entry a simple button. Required. */
#define SDL_TRAYENTRY_CHECKBOX    0x00000002u /**< Make the entry a checkbox. Required. */
#define SDL_TRAYENTRY_SUBMENU     0x00000004u /**< Prepare the entry to have a submenu. Required */
#define SDL_TRAYENTRY_DISABLED    0x80000000u /**< Make the entry disabled. Optional. */
#define SDL_TRAYENTRY_CHECKED     0x40000000u /**< Make the entry checked. This is valid only for checkboxes. Optional. */

/**
 * A callback that is invoked when a tray entry is selected.
 *
 * \param userdata an optional pointer to pass extra data to the callback when
 *                 it will be invoked.
 * \param entry the tray entry that was selected.
 *
 * \since This datatype is available since SDL 3.2.0.
 *
 * \sa SDL_SetTrayEntryCallback
 */
typedef void (SDLCALL *SDL_TrayCallback)(void *userdata, SDL_TrayEntry *entry);

/**
 * Create an icon to be placed in the operating system's tray, or equivalent.
 *
 * Many platforms advise not using a system tray unless persistence is a
 * necessary feature. Avoid needlessly creating a tray icon, as the user may
 * feel like it clutters their interface.
 *
 * Using tray icons require the video subsystem.
 *
 * \param icon a surface to be used as icon. May be NULL.
 * \param tooltip a tooltip to be displayed when the mouse hovers the icon in
 *                UTF-8 encoding. Not supported on all platforms. May be NULL.
 * \returns The newly created system tray icon.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTrayMenu
 * \sa SDL_GetTrayMenu
 * \sa SDL_DestroyTray
 */
extern SDL_DECLSPEC SDL_Tray * SDLCALL SDL_CreateTray(SDL_Surface *icon, const char *tooltip);

/**
 * Updates the system tray icon's icon.
 *
 * \param tray the tray icon to be updated.
 * \param icon the new icon. May be NULL.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTray
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetTrayIcon(SDL_Tray *tray, SDL_Surface *icon);

/**
 * Updates the system tray icon's tooltip.
 *
 * \param tray the tray icon to be updated.
 * \param tooltip the new tooltip in UTF-8 encoding. May be NULL.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTray
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetTrayTooltip(SDL_Tray *tray, const char *tooltip);

/**
 * Create a menu for a system tray.
 *
 * This should be called at most once per tray icon.
 *
 * This function does the same thing as SDL_CreateTraySubmenu(), except that
 * it takes a SDL_Tray instead of a SDL_TrayEntry.
 *
 * A menu does not need to be destroyed; it will be destroyed with the tray.
 *
 * \param tray the tray to bind the menu to.
 * \returns the newly created menu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTray
 * \sa SDL_GetTrayMenu
 * \sa SDL_GetTrayMenuParentTray
 */
extern SDL_DECLSPEC SDL_TrayMenu * SDLCALL SDL_CreateTrayMenu(SDL_Tray *tray);

/**
 * Create a submenu for a system tray entry.
 *
 * This should be called at most once per tray entry.
 *
 * This function does the same thing as SDL_CreateTrayMenu, except that it
 * takes a SDL_TrayEntry instead of a SDL_Tray.
 *
 * A menu does not need to be destroyed; it will be destroyed with the tray.
 *
 * \param entry the tray entry to bind the menu to.
 * \returns the newly created menu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_GetTraySubmenu
 * \sa SDL_GetTrayMenuParentEntry
 */
extern SDL_DECLSPEC SDL_TrayMenu * SDLCALL SDL_CreateTraySubmenu(SDL_TrayEntry *entry);

/**
 * Gets a previously created tray menu.
 *
 * You should have called SDL_CreateTrayMenu() on the tray object. This
 * function allows you to fetch it again later.
 *
 * This function does the same thing as SDL_GetTraySubmenu(), except that it
 * takes a SDL_Tray instead of a SDL_TrayEntry.
 *
 * A menu does not need to be destroyed; it will be destroyed with the tray.
 *
 * \param tray the tray entry to bind the menu to.
 * \returns the newly created menu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTray
 * \sa SDL_CreateTrayMenu
 */
extern SDL_DECLSPEC SDL_TrayMenu * SDLCALL SDL_GetTrayMenu(SDL_Tray *tray);

/**
 * Gets a previously created tray entry submenu.
 *
 * You should have called SDL_CreateTraySubmenu() on the entry object. This
 * function allows you to fetch it again later.
 *
 * This function does the same thing as SDL_GetTrayMenu(), except that it
 * takes a SDL_TrayEntry instead of a SDL_Tray.
 *
 * A menu does not need to be destroyed; it will be destroyed with the tray.
 *
 * \param entry the tray entry to bind the menu to.
 * \returns the newly created menu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_CreateTraySubmenu
 */
extern SDL_DECLSPEC SDL_TrayMenu * SDLCALL SDL_GetTraySubmenu(SDL_TrayEntry *entry);

/**
 * Returns a list of entries in the menu, in order.
 *
 * \param menu The menu to get entries from.
 * \param count An optional pointer to obtain the number of entries in the
 *              menu.
 * \returns a NULL-terminated list of entries within the given menu. The
 *          pointer becomes invalid when any function that inserts or deletes
 *          entries in the menu is called.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_RemoveTrayEntry
 * \sa SDL_InsertTrayEntryAt
 */
extern SDL_DECLSPEC const SDL_TrayEntry ** SDLCALL SDL_GetTrayEntries(SDL_TrayMenu *menu, int *count);

/**
 * Removes a tray entry.
 *
 * \param entry The entry to be deleted.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 */
extern SDL_DECLSPEC void SDLCALL SDL_RemoveTrayEntry(SDL_TrayEntry *entry);

/**
 * Insert a tray entry at a given position.
 *
 * If label is NULL, the entry will be a separator. Many functions won't work
 * for an entry that is a separator.
 *
 * An entry does not need to be destroyed; it will be destroyed with the tray.
 *
 * \param menu the menu to append the entry to.
 * \param pos the desired position for the new entry. Entries at or following
 *            this place will be moved. If pos is -1, the entry is appended.
 * \param label the text to be displayed on the entry, in UTF-8 encoding, or
 *              NULL for a separator.
 * \param flags a combination of flags, some of which are mandatory.
 * \returns the newly created entry, or NULL if pos is out of bounds.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_TrayEntryFlags
 * \sa SDL_GetTrayEntries
 * \sa SDL_RemoveTrayEntry
 * \sa SDL_GetTrayEntryParent
 */
extern SDL_DECLSPEC SDL_TrayEntry * SDLCALL SDL_InsertTrayEntryAt(SDL_TrayMenu *menu, int pos, const char *label, SDL_TrayEntryFlags flags);

/**
 * Sets the label of an entry.
 *
 * An entry cannot change between a separator and an ordinary entry; that is,
 * it is not possible to set a non-NULL label on an entry that has a NULL
 * label (separators), or to set a NULL label to an entry that has a non-NULL
 * label. The function will silently fail if that happens.
 *
 * \param entry the entry to be updated.
 * \param label the new label for the entry in UTF-8 encoding.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_GetTrayEntryLabel
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetTrayEntryLabel(SDL_TrayEntry *entry, const char *label);

/**
 * Gets the label of an entry.
 *
 * If the returned value is NULL, the entry is a separator.
 *
 * \param entry the entry to be read.
 * \returns the label of the entry in UTF-8 encoding.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_SetTrayEntryLabel
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetTrayEntryLabel(SDL_TrayEntry *entry);

/**
 * Sets whether or not an entry is checked.
 *
 * The entry must have been created with the SDL_TRAYENTRY_CHECKBOX flag.
 *
 * \param entry the entry to be updated.
 * \param checked true if the entry should be checked; false otherwise.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_GetTrayEntryChecked
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetTrayEntryChecked(SDL_TrayEntry *entry, bool checked);

/**
 * Gets whether or not an entry is checked.
 *
 * The entry must have been created with the SDL_TRAYENTRY_CHECKBOX flag.
 *
 * \param entry the entry to be read.
 * \returns true if the entry is checked; false otherwise.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_SetTrayEntryChecked
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetTrayEntryChecked(SDL_TrayEntry *entry);

/**
 * Sets whether or not an entry is enabled.
 *
 * \param entry the entry to be updated.
 * \param enabled true if the entry should be enabled; false otherwise.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_GetTrayEntryEnabled
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetTrayEntryEnabled(SDL_TrayEntry *entry, bool enabled);

/**
 * Gets whether or not an entry is enabled.
 *
 * \param entry the entry to be read.
 * \returns true if the entry is enabled; false otherwise.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 * \sa SDL_SetTrayEntryEnabled
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetTrayEntryEnabled(SDL_TrayEntry *entry);

/**
 * Sets a callback to be invoked when the entry is selected.
 *
 * \param entry the entry to be updated.
 * \param callback a callback to be invoked when the entry is selected.
 * \param userdata an optional pointer to pass extra data to the callback when
 *                 it will be invoked.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTrayEntries
 * \sa SDL_InsertTrayEntryAt
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetTrayEntryCallback(SDL_TrayEntry *entry, SDL_TrayCallback callback, void *userdata);

/**
 * Simulate a click on a tray entry.
 *
 * \param entry The entry to activate.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC void SDLCALL SDL_ClickTrayEntry(SDL_TrayEntry *entry);

/**
 * Destroys a tray object.
 *
 * This also destroys all associated menus and entries.
 *
 * \param tray the tray icon to be destroyed.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTray
 */
extern SDL_DECLSPEC void SDLCALL SDL_DestroyTray(SDL_Tray *tray);

/**
 * Gets the menu containing a certain tray entry.
 *
 * \param entry the entry for which to get the parent menu.
 * \returns the parent menu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_InsertTrayEntryAt
 */
extern SDL_DECLSPEC SDL_TrayMenu * SDLCALL SDL_GetTrayEntryParent(SDL_TrayEntry *entry);

/**
 * Gets the entry for which the menu is a submenu, if the current menu is a
 * submenu.
 *
 * Either this function or SDL_GetTrayMenuParentTray() will return non-NULL
 * for any given menu.
 *
 * \param menu the menu for which to get the parent entry.
 * \returns the parent entry, or NULL if this menu is not a submenu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTraySubmenu
 * \sa SDL_GetTrayMenuParentTray
 */
extern SDL_DECLSPEC SDL_TrayEntry * SDLCALL SDL_GetTrayMenuParentEntry(SDL_TrayMenu *menu);

/**
 * Gets the tray for which this menu is the first-level menu, if the current
 * menu isn't a submenu.
 *
 * Either this function or SDL_GetTrayMenuParentEntry() will return non-NULL
 * for any given menu.
 *
 * \param menu the menu for which to get the parent enttrayry.
 * \returns the parent tray, or NULL if this menu is a submenu.
 *
 * \threadsafety This function should be called on the thread that created the
 *               tray.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateTrayMenu
 * \sa SDL_GetTrayMenuParentEntry
 */
extern SDL_DECLSPEC SDL_Tray * SDLCALL SDL_GetTrayMenuParentTray(SDL_TrayMenu *menu);

/**
 * Update the trays.
 *
 * This is called automatically by the event loop and is only needed if you're
 * using trays but aren't handling SDL events.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC void SDLCALL SDL_UpdateTrays(void);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_tray_h_ */
