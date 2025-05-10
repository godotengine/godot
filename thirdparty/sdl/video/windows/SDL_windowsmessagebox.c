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

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#include "../../core/windows/SDL_windows.h"

#include "SDL_windowsvideo.h"

#ifndef SS_EDITCONTROL
#define SS_EDITCONTROL 0x2000
#endif

#ifndef IDOK
#define IDOK 1
#endif

#ifndef IDCANCEL
#define IDCANCEL 2
#endif

// Custom dialog return codes
#define IDCLOSED           20
#define IDINVALPTRINIT     50
#define IDINVALPTRCOMMAND  51
#define IDINVALPTRSETFOCUS 52
#define IDINVALPTRDLGITEM  53
// First button ID
#define IDBUTTONINDEX0 100

#define DLGITEMTYPEBUTTON 0x0080
#define DLGITEMTYPESTATIC 0x0082

/* Windows only sends the lower 16 bits of the control ID when a button
 * gets clicked. There are also some predefined and custom IDs that lower
 * the available number further. 2^16 - 101 buttons should be enough for
 * everyone, no need to make the code more complex.
 */
#define MAX_BUTTONS (0xffff - 100)

// Display a Windows message box

typedef HRESULT(CALLBACK *PFTASKDIALOGCALLBACK)(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam, LONG_PTR lpRefData);

enum _TASKDIALOG_FLAGS
{
    TDF_ENABLE_HYPERLINKS = 0x0001,
    TDF_USE_HICON_MAIN = 0x0002,
    TDF_USE_HICON_FOOTER = 0x0004,
    TDF_ALLOW_DIALOG_CANCELLATION = 0x0008,
    TDF_USE_COMMAND_LINKS = 0x0010,
    TDF_USE_COMMAND_LINKS_NO_ICON = 0x0020,
    TDF_EXPAND_FOOTER_AREA = 0x0040,
    TDF_EXPANDED_BY_DEFAULT = 0x0080,
    TDF_VERIFICATION_FLAG_CHECKED = 0x0100,
    TDF_SHOW_PROGRESS_BAR = 0x0200,
    TDF_SHOW_MARQUEE_PROGRESS_BAR = 0x0400,
    TDF_CALLBACK_TIMER = 0x0800,
    TDF_POSITION_RELATIVE_TO_WINDOW = 0x1000,
    TDF_RTL_LAYOUT = 0x2000,
    TDF_NO_DEFAULT_RADIO_BUTTON = 0x4000,
    TDF_CAN_BE_MINIMIZED = 0x8000,
    // #if (NTDDI_VERSION >= NTDDI_WIN8)
    TDF_NO_SET_FOREGROUND = 0x00010000, // Don't call SetForegroundWindow() when activating the dialog
                                        // #endif // (NTDDI_VERSION >= NTDDI_WIN8)
    TDF_SIZE_TO_CONTENT = 0x01000000    // used by ShellMessageBox to emulate MessageBox sizing behavior
};
typedef int TASKDIALOG_FLAGS; // Note: _TASKDIALOG_FLAGS is an int

typedef enum _TASKDIALOG_MESSAGES
{
    TDM_NAVIGATE_PAGE = WM_USER + 101,
    TDM_CLICK_BUTTON = WM_USER + 102,                        // wParam = Button ID
    TDM_SET_MARQUEE_PROGRESS_BAR = WM_USER + 103,            // wParam = 0 (nonMarque) wParam != 0 (Marquee)
    TDM_SET_PROGRESS_BAR_STATE = WM_USER + 104,              // wParam = new progress state
    TDM_SET_PROGRESS_BAR_RANGE = WM_USER + 105,              // lParam = MAKELPARAM(nMinRange, nMaxRange)
    TDM_SET_PROGRESS_BAR_POS = WM_USER + 106,                // wParam = new position
    TDM_SET_PROGRESS_BAR_MARQUEE = WM_USER + 107,            // wParam = 0 (stop marquee), wParam != 0 (start marquee), lparam = speed (milliseconds between repaints)
    TDM_SET_ELEMENT_TEXT = WM_USER + 108,                    // wParam = element (TASKDIALOG_ELEMENTS), lParam = new element text (LPCWSTR)
    TDM_CLICK_RADIO_BUTTON = WM_USER + 110,                  // wParam = Radio Button ID
    TDM_ENABLE_BUTTON = WM_USER + 111,                       // lParam = 0 (disable), lParam != 0 (enable), wParam = Button ID
    TDM_ENABLE_RADIO_BUTTON = WM_USER + 112,                 // lParam = 0 (disable), lParam != 0 (enable), wParam = Radio Button ID
    TDM_CLICK_VERIFICATION = WM_USER + 113,                  // wParam = 0 (unchecked), 1 (checked), lParam = 1 (set key focus)
    TDM_UPDATE_ELEMENT_TEXT = WM_USER + 114,                 // wParam = element (TASKDIALOG_ELEMENTS), lParam = new element text (LPCWSTR)
    TDM_SET_BUTTON_ELEVATION_REQUIRED_STATE = WM_USER + 115, // wParam = Button ID, lParam = 0 (elevation not required), lParam != 0 (elevation required)
    TDM_UPDATE_ICON = WM_USER + 116                          // wParam = icon element (TASKDIALOG_ICON_ELEMENTS), lParam = new icon (hIcon if TDF_USE_HICON_* was set, PCWSTR otherwise)
} TASKDIALOG_MESSAGES;

typedef enum _TASKDIALOG_NOTIFICATIONS
{
    TDN_CREATED = 0,
    TDN_NAVIGATED = 1,
    TDN_BUTTON_CLICKED = 2,    // wParam = Button ID
    TDN_HYPERLINK_CLICKED = 3, // lParam = (LPCWSTR)pszHREF
    TDN_TIMER = 4,             // wParam = Milliseconds since dialog created or timer reset
    TDN_DESTROYED = 5,
    TDN_RADIO_BUTTON_CLICKED = 6, // wParam = Radio Button ID
    TDN_DIALOG_CONSTRUCTED = 7,
    TDN_VERIFICATION_CLICKED = 8, // wParam = 1 if checkbox checked, 0 if not, lParam is unused and always 0
    TDN_HELP = 9,
    TDN_EXPANDO_BUTTON_CLICKED = 10 // wParam = 0 (dialog is now collapsed), wParam != 0 (dialog is now expanded)
} TASKDIALOG_NOTIFICATIONS;

typedef enum _TASKDIALOG_ELEMENTS
{
    TDE_CONTENT,
    TDE_EXPANDED_INFORMATION,
    TDE_FOOTER,
    TDE_MAIN_INSTRUCTION
} TASKDIALOG_ELEMENTS;

typedef enum _TASKDIALOG_ICON_ELEMENTS
{
    TDIE_ICON_MAIN,
    TDIE_ICON_FOOTER
} TASKDIALOG_ICON_ELEMENTS;

#define TD_WARNING_ICON     MAKEINTRESOURCEW(-1)
#define TD_ERROR_ICON       MAKEINTRESOURCEW(-2)
#define TD_INFORMATION_ICON MAKEINTRESOURCEW(-3)
#define TD_SHIELD_ICON      MAKEINTRESOURCEW(-4)

enum _TASKDIALOG_COMMON_BUTTON_FLAGS
{
    TDCBF_OK_BUTTON = 0x0001,     // selected control return value IDOK
    TDCBF_YES_BUTTON = 0x0002,    // selected control return value IDYES
    TDCBF_NO_BUTTON = 0x0004,     // selected control return value IDNO
    TDCBF_CANCEL_BUTTON = 0x0008, // selected control return value IDCANCEL
    TDCBF_RETRY_BUTTON = 0x0010,  // selected control return value IDRETRY
    TDCBF_CLOSE_BUTTON = 0x0020   // selected control return value IDCLOSE
};
typedef int TASKDIALOG_COMMON_BUTTON_FLAGS; // Note: _TASKDIALOG_COMMON_BUTTON_FLAGS is an int

#pragma pack(push, 1)

typedef struct _TASKDIALOG_BUTTON
{
    int nButtonID;
    PCWSTR pszButtonText;
} TASKDIALOG_BUTTON;

typedef struct _TASKDIALOGCONFIG
{
    UINT cbSize;
    HWND hwndParent;                                // incorrectly named, this is the owner window, not a parent.
    HINSTANCE hInstance;                            // used for MAKEINTRESOURCE() strings
    TASKDIALOG_FLAGS dwFlags;                       // TASKDIALOG_FLAGS (TDF_XXX) flags
    TASKDIALOG_COMMON_BUTTON_FLAGS dwCommonButtons; // TASKDIALOG_COMMON_BUTTON (TDCBF_XXX) flags
    PCWSTR pszWindowTitle;                          // string or MAKEINTRESOURCE()
    union
    {
        HICON hMainIcon;
        PCWSTR pszMainIcon;
    } /*DUMMYUNIONNAME*/;
    PCWSTR pszMainInstruction;
    PCWSTR pszContent;
    UINT cButtons;
    const TASKDIALOG_BUTTON *pButtons;
    int nDefaultButton;
    UINT cRadioButtons;
    const TASKDIALOG_BUTTON *pRadioButtons;
    int nDefaultRadioButton;
    PCWSTR pszVerificationText;
    PCWSTR pszExpandedInformation;
    PCWSTR pszExpandedControlText;
    PCWSTR pszCollapsedControlText;
    union
    {
        HICON hFooterIcon;
        PCWSTR pszFooterIcon;
    } /*DUMMYUNIONNAME2*/;
    PCWSTR pszFooter;
    PFTASKDIALOGCALLBACK pfCallback;
    LONG_PTR lpCallbackData;
    UINT cxWidth; // width of the Task Dialog's client area in DLU's. If 0, Task Dialog will calculate the ideal width.
} TASKDIALOGCONFIG;

typedef struct
{
    WORD dlgVer;
    WORD signature;
    DWORD helpID;
    DWORD exStyle;
    DWORD style;
    WORD cDlgItems;
    short x;
    short y;
    short cx;
    short cy;
} DLGTEMPLATEEX;

typedef struct
{
    DWORD helpID;
    DWORD exStyle;
    DWORD style;
    short x;
    short y;
    short cx;
    short cy;
    DWORD id;
} DLGITEMTEMPLATEEX;

#pragma pack(pop)

typedef struct
{
    DLGTEMPLATEEX *lpDialog;
    void *data;
    size_t size;
    size_t used;
    WORD numbuttons;
} WIN_DialogData;

static bool GetButtonIndex(const SDL_MessageBoxData *messageboxdata, SDL_MessageBoxButtonFlags flags, size_t *i)
{
    for (*i = 0; *i < (size_t)messageboxdata->numbuttons; ++*i) {
        if (messageboxdata->buttons[*i].flags & flags) {
            return true;
        }
    }
    return false;
}

static INT_PTR CALLBACK MessageBoxDialogProc(HWND hDlg, UINT iMessage, WPARAM wParam, LPARAM lParam)
{
    const SDL_MessageBoxData *messageboxdata;
    size_t buttonindex;

    switch (iMessage) {
    case WM_INITDIALOG:
        if (lParam == 0) {
            EndDialog(hDlg, IDINVALPTRINIT);
            return TRUE;
        }
        messageboxdata = (const SDL_MessageBoxData *)lParam;
        SetWindowLongPtr(hDlg, GWLP_USERDATA, lParam);

        if (GetButtonIndex(messageboxdata, SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, &buttonindex)) {
            // Focus on the first default return-key button
            HWND buttonctl = GetDlgItem(hDlg, (int)(IDBUTTONINDEX0 + buttonindex));
            if (!buttonctl) {
                EndDialog(hDlg, IDINVALPTRDLGITEM);
            }
            PostMessage(hDlg, WM_NEXTDLGCTL, (WPARAM)buttonctl, TRUE);
        } else {
            // Give the focus to the dialog window instead
            SetFocus(hDlg);
        }
        return FALSE;
    case WM_SETFOCUS:
        messageboxdata = (const SDL_MessageBoxData *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
        if (!messageboxdata) {
            EndDialog(hDlg, IDINVALPTRSETFOCUS);
            return TRUE;
        }

        // Let the default button be focused if there is one. Otherwise, prevent any initial focus.
        if (GetButtonIndex(messageboxdata, SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, &buttonindex)) {
            return FALSE;
        }
        return TRUE;
    case WM_COMMAND:
        messageboxdata = (const SDL_MessageBoxData *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
        if (!messageboxdata) {
            EndDialog(hDlg, IDINVALPTRCOMMAND);
            return TRUE;
        }

        // Return the ID of the button that was pushed
        if (wParam == IDOK) {
            if (GetButtonIndex(messageboxdata, SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, &buttonindex)) {
                EndDialog(hDlg, IDBUTTONINDEX0 + buttonindex);
            }
        } else if (wParam == IDCANCEL) {
            if (GetButtonIndex(messageboxdata, SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT, &buttonindex)) {
                EndDialog(hDlg, IDBUTTONINDEX0 + buttonindex);
            } else {
                // Closing of window was requested by user or system. It would be rude not to comply.
                EndDialog(hDlg, IDCLOSED);
            }
        } else if (wParam >= IDBUTTONINDEX0 && (int)wParam - IDBUTTONINDEX0 < messageboxdata->numbuttons) {
            EndDialog(hDlg, wParam);
        }
        return TRUE;

    default:
        break;
    }
    return FALSE;
}

static bool ExpandDialogSpace(WIN_DialogData *dialog, size_t space)
{
    // Growing memory in 64 KiB steps.
    const size_t sizestep = 0x10000;
    size_t size = dialog->size;

    if (size == 0) {
        // Start with 4 KiB or a multiple of 64 KiB to fit the data.
        size = 0x1000;
        if (SIZE_MAX - sizestep < space) {
            size = space;
        } else if (space > size) {
            size = (space + sizestep) & ~(sizestep - 1);
        }
    } else if (SIZE_MAX - dialog->used < space) {
        SDL_OutOfMemory();
        return false;
    } else if (SIZE_MAX - (dialog->used + space) < sizestep) {
        // Close to the maximum.
        size = dialog->used + space;
    } else if (size < dialog->used + space) {
        // Round up to the next 64 KiB block.
        size = dialog->used + space;
        size += sizestep - size % sizestep;
    }

    if (size > dialog->size) {
        void *data = SDL_realloc(dialog->data, size);
        if (!data) {
            return false;
        }
        dialog->data = data;
        dialog->size = size;
        dialog->lpDialog = (DLGTEMPLATEEX *)dialog->data;
    }
    return true;
}

static bool AlignDialogData(WIN_DialogData *dialog, size_t size)
{
    size_t padding = (dialog->used % size);

    if (!ExpandDialogSpace(dialog, padding)) {
        return false;
    }

    dialog->used += padding;

    return true;
}

static bool AddDialogData(WIN_DialogData *dialog, const void *data, size_t size)
{
    if (!ExpandDialogSpace(dialog, size)) {
        return false;
    }

    SDL_memcpy((Uint8 *)dialog->data + dialog->used, data, size);
    dialog->used += size;

    return true;
}

static bool AddDialogString(WIN_DialogData *dialog, const char *string)
{
    WCHAR *wstring;
    WCHAR *p;
    size_t count;
    bool status;

    if (!string) {
        string = "";
    }

    wstring = WIN_UTF8ToStringW(string);
    if (!wstring) {
        return false;
    }

    // Find out how many characters we have, including null terminator
    count = 0;
    for (p = wstring; *p; ++p) {
        ++count;
    }
    ++count;

    status = AddDialogData(dialog, wstring, count * sizeof(WCHAR));
    SDL_free(wstring);
    return status;
}

static int s_BaseUnitsX;
static int s_BaseUnitsY;
static void Vec2ToDLU(short *x, short *y)
{
    SDL_assert(s_BaseUnitsX != 0); // we init in WIN_ShowMessageBox(), which is the only public function...

    *x = (short)MulDiv(*x, 4, s_BaseUnitsX);
    *y = (short)MulDiv(*y, 8, s_BaseUnitsY);
}

static bool AddDialogControl(WIN_DialogData *dialog, WORD type, DWORD style, DWORD exStyle, int x, int y, int w, int h, int id, const char *caption, WORD ordinal)
{
    DLGITEMTEMPLATEEX item;
    WORD marker = 0xFFFF;
    WORD extraData = 0;

    SDL_zero(item);
    item.style = style;
    item.exStyle = exStyle;
    item.x = (short)x;
    item.y = (short)y;
    item.cx = (short)w;
    item.cy = (short)h;
    item.id = id;

    Vec2ToDLU(&item.x, &item.y);
    Vec2ToDLU(&item.cx, &item.cy);

    if (!AlignDialogData(dialog, sizeof(DWORD))) {
        return false;
    }
    if (!AddDialogData(dialog, &item, sizeof(item))) {
        return false;
    }
    if (!AddDialogData(dialog, &marker, sizeof(marker))) {
        return false;
    }
    if (!AddDialogData(dialog, &type, sizeof(type))) {
        return false;
    }
    if (type == DLGITEMTYPEBUTTON || (type == DLGITEMTYPESTATIC && caption)) {
        if (!AddDialogString(dialog, caption)) {
            return false;
        }
    } else {
        if (!AddDialogData(dialog, &marker, sizeof(marker))) {
            return false;
        }
        if (!AddDialogData(dialog, &ordinal, sizeof(ordinal))) {
            return false;
        }
    }
    if (!AddDialogData(dialog, &extraData, sizeof(extraData))) {
        return false;
    }
    if (type == DLGITEMTYPEBUTTON) {
        dialog->numbuttons++;
    }
    ++dialog->lpDialog->cDlgItems;

    return true;
}

static bool AddDialogStaticText(WIN_DialogData *dialog, int x, int y, int w, int h, const char *text)
{
    DWORD style = WS_VISIBLE | WS_CHILD | SS_LEFT | SS_NOPREFIX | SS_EDITCONTROL | WS_GROUP;
    return AddDialogControl(dialog, DLGITEMTYPESTATIC, style, 0, x, y, w, h, -1, text, 0);
}

static bool AddDialogStaticIcon(WIN_DialogData *dialog, int x, int y, int w, int h, Uint16 ordinal)
{
    DWORD style = WS_VISIBLE | WS_CHILD | SS_ICON | WS_GROUP;
    return AddDialogControl(dialog, DLGITEMTYPESTATIC, style, 0, x, y, w, h, -2, NULL, ordinal);
}

static bool AddDialogButton(WIN_DialogData *dialog, int x, int y, int w, int h, const char *text, int id, bool isDefault)
{
    DWORD style = WS_VISIBLE | WS_CHILD | WS_TABSTOP;
    if (isDefault) {
        style |= BS_DEFPUSHBUTTON;
    } else {
        style |= BS_PUSHBUTTON;
    }
    // The first button marks the start of the group.
    if (dialog->numbuttons == 0) {
        style |= WS_GROUP;
    }
    return AddDialogControl(dialog, DLGITEMTYPEBUTTON, style, 0, x, y, w, h, id, text, 0);
}

static void FreeDialogData(WIN_DialogData *dialog)
{
    SDL_free(dialog->data);
    SDL_free(dialog);
}

static WIN_DialogData *CreateDialogData(int w, int h, const char *caption)
{
    WIN_DialogData *dialog;
    DLGTEMPLATEEX dialogTemplate;
    WORD WordToPass;

    SDL_zero(dialogTemplate);
    dialogTemplate.dlgVer = 1;
    dialogTemplate.signature = 0xffff;
    dialogTemplate.style = (WS_CAPTION | DS_CENTER | DS_SHELLFONT);
    dialogTemplate.x = 0;
    dialogTemplate.y = 0;
    dialogTemplate.cx = (short)w;
    dialogTemplate.cy = (short)h;
    Vec2ToDLU(&dialogTemplate.cx, &dialogTemplate.cy);

    dialog = (WIN_DialogData *)SDL_calloc(1, sizeof(*dialog));
    if (!dialog) {
        return NULL;
    }

    if (!AddDialogData(dialog, &dialogTemplate, sizeof(dialogTemplate))) {
        FreeDialogData(dialog);
        return NULL;
    }

    // No menu
    WordToPass = 0;
    if (!AddDialogData(dialog, &WordToPass, 2)) {
        FreeDialogData(dialog);
        return NULL;
    }

    // No custom class
    if (!AddDialogData(dialog, &WordToPass, 2)) {
        FreeDialogData(dialog);
        return NULL;
    }

    // title
    if (!AddDialogString(dialog, caption)) {
        FreeDialogData(dialog);
        return NULL;
    }

    // Font stuff
    {
        /*
         * We want to use the system messagebox font.
         */
        BYTE ToPass;

        NONCLIENTMETRICSA NCM;
        NCM.cbSize = sizeof(NCM);
        SystemParametersInfoA(SPI_GETNONCLIENTMETRICS, 0, &NCM, 0);

        // Font size - convert to logical font size for dialog parameter.
        {
            HDC ScreenDC = GetDC(NULL);
            int LogicalPixelsY = GetDeviceCaps(ScreenDC, LOGPIXELSY);
            if (!LogicalPixelsY) {
                LogicalPixelsY = 72; // This can happen if the application runs out of GDI handles
            }

            WordToPass = (WORD)(-72 * NCM.lfMessageFont.lfHeight / LogicalPixelsY);
            ReleaseDC(NULL, ScreenDC);
        }

        if (!AddDialogData(dialog, &WordToPass, 2)) {
            FreeDialogData(dialog);
            return NULL;
        }

        // Font weight
        WordToPass = (WORD)NCM.lfMessageFont.lfWeight;
        if (!AddDialogData(dialog, &WordToPass, 2)) {
            FreeDialogData(dialog);
            return NULL;
        }

        // italic?
        ToPass = NCM.lfMessageFont.lfItalic;
        if (!AddDialogData(dialog, &ToPass, 1)) {
            FreeDialogData(dialog);
            return NULL;
        }

        // charset?
        ToPass = NCM.lfMessageFont.lfCharSet;
        if (!AddDialogData(dialog, &ToPass, 1)) {
            FreeDialogData(dialog);
            return NULL;
        }

        // font typeface.
        if (!AddDialogString(dialog, NCM.lfMessageFont.lfFaceName)) {
            FreeDialogData(dialog);
            return NULL;
        }
    }

    return dialog;
}

/* Escaping ampersands is necessary to disable mnemonics in dialog controls.
 * The caller provides a char** for dst and a size_t* for dstlen where the
 * address of the work buffer and its size will be stored. Their values must be
 * NULL and 0 on the first call. src is the string to be escaped. On error, the
 * function returns NULL and, on success, returns a pointer to the escaped
 * sequence as a read-only string that is valid until the next call or until the
 * work buffer is freed. Once all strings have been processed, it's the caller's
 * responsibility to free the work buffer with SDL_free, even on errors.
 */
static const char *EscapeAmpersands(char **dst, size_t *dstlen, const char *src)
{
    char *newdst;
    size_t ampcount = 0;
    size_t srclen = 0;

    if (!src) {
        return NULL;
    }

    while (src[srclen]) {
        if (src[srclen] == '&') {
            ampcount++;
        }
        srclen++;
    }
    srclen++;

    if (ampcount == 0) {
        // Nothing to do.
        return src;
    }
    if (SIZE_MAX - srclen < ampcount) {
        return NULL;
    }
    if (!*dst || *dstlen < srclen + ampcount) {
        // Allocating extra space in case the next strings are a bit longer.
        size_t extraspace = SIZE_MAX - (srclen + ampcount);
        if (extraspace > 512) {
            extraspace = 512;
        }
        *dstlen = srclen + ampcount + extraspace;
        SDL_free(*dst);
        *dst = NULL;
        newdst = (char *)SDL_malloc(*dstlen);
        if (!newdst) {
            return NULL;
        }
        *dst = newdst;
    } else {
        newdst = *dst;
    }

    // The escape character is the ampersand itself.
    while (srclen--) {
        if (*src == '&') {
            *newdst++ = '&';
        }
        *newdst++ = *src++;
    }

    return *dst;
}

static float WIN_GetContentScale(void)
{
    int dpi = 0;

#if 0 // We don't know what monitor the dialog will be shown on
    UINT hdpi_uint, vdpi_uint;
    if (GetDpiForMonitor(hMonitor, MDT_EFFECTIVE_DPI, &hdpi_uint, &vdpi_uint) == S_OK) {
        dpi = (int)hdpi_uint;
    }
#endif
    if (dpi == 0) {
        // Window 8.0 and below: same DPI for all monitors
        HDC hdc = GetDC(NULL);
        if (hdc) {
            dpi = GetDeviceCaps(hdc, LOGPIXELSX);
            ReleaseDC(NULL, hdc);
        }
    }
    if (dpi == 0) {
        // Safe default
        dpi = USER_DEFAULT_SCREEN_DPI;
    }
    return dpi / (float)USER_DEFAULT_SCREEN_DPI;
}

// This function is called if a Task Dialog is unsupported.
static bool WIN_ShowOldMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    WIN_DialogData *dialog;
    int i, x, y;
    HFONT DialogFont;
    SIZE Size;
    RECT TextSize;
    wchar_t *wmessage;
    TEXTMETRIC TM;
    HDC FontDC;
    INT_PTR rc;
    char *ampescape = NULL;
    size_t ampescapesize = 0;
    Uint16 defbuttoncount = 0;
    Uint16 icon = 0;
    bool result;

    HWND ParentWindow = NULL;

    const float scale = WIN_GetContentScale();
    const int ButtonWidth = (int)SDL_roundf(88 * scale);
    const int ButtonHeight = (int)SDL_roundf(26 * scale);
    const int TextMargin = (int)SDL_roundf(16 * scale);
    const int ButtonMargin = (int)SDL_roundf(12 * scale);
    const int IconWidth = GetSystemMetrics(SM_CXICON);
    const int IconHeight = GetSystemMetrics(SM_CYICON);
    const int IconMargin = (int)SDL_roundf(20 * scale);

    if (messageboxdata->numbuttons > MAX_BUTTONS) {
        return SDL_SetError("Number of buttons exceeds limit of %d", MAX_BUTTONS);
    }

    switch (messageboxdata->flags & (SDL_MESSAGEBOX_ERROR | SDL_MESSAGEBOX_WARNING | SDL_MESSAGEBOX_INFORMATION)) {
    case SDL_MESSAGEBOX_ERROR:
        icon = (Uint16)(size_t)IDI_ERROR;
        break;
    case SDL_MESSAGEBOX_WARNING:
        icon = (Uint16)(size_t)IDI_WARNING;
        break;
    case SDL_MESSAGEBOX_INFORMATION:
        icon = (Uint16)(size_t)IDI_INFORMATION;
        break;
    }

    /* Jan 25th, 2013 - dant@fleetsa.com
     *
     * I've tried to make this more reasonable, but I've run in to a lot
     * of nonsense.
     *
     * The original issue is the code was written in pixels and not
     * dialog units (DLUs). All DialogBox functions use DLUs, which
     * vary based on the selected font (yay).
     *
     * According to MSDN, the most reliable way to convert is via
     * MapDialogUnits, which requires an HWND, which we don't have
     * at time of template creation.
     *
     * We do however have:
     *  The system font (DLU width 8 for me)
     *  The font we select for the dialog (DLU width 6 for me)
     *
     * Based on experimentation, *neither* of these return the value
     * actually used. Stepping in to MapDialogUnits(), the conversion
     * is fairly clear, and uses 7 for me.
     *
     * As a result, some of this is hacky to ensure the sizing is
     * somewhat correct.
     *
     * Honestly, a long term solution is to use CreateWindow, not CreateDialog.
     *
     * In order to get text dimensions we need to have a DC with the desired font.
     * I'm assuming a dialog box in SDL is rare enough we can to the create.
     */
    FontDC = CreateCompatibleDC(0);

    {
        // Create a duplicate of the font used in system message boxes.
        LOGFONT lf;
        NONCLIENTMETRICS NCM;
        NCM.cbSize = sizeof(NCM);
        SystemParametersInfo(SPI_GETNONCLIENTMETRICS, 0, &NCM, 0);
        lf = NCM.lfMessageFont;
        DialogFont = CreateFontIndirect(&lf);
    }

    // Select the font in to our DC
    SelectObject(FontDC, DialogFont);

    {
        // Get the metrics to try and figure our DLU conversion.
        GetTextMetrics(FontDC, &TM);

        /* Calculation from the following documentation:
         * https://support.microsoft.com/en-gb/help/125681/how-to-calculate-dialog-base-units-with-non-system-based-font
         * This fixes bug 2137, dialog box calculation with a fixed-width system font
         */
        {
            SIZE extent;
            GetTextExtentPoint32A(FontDC, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", 52, &extent);
            s_BaseUnitsX = (extent.cx / 26 + 1) / 2;
        }
        // s_BaseUnitsX = TM.tmAveCharWidth + 1;
        s_BaseUnitsY = TM.tmHeight;
    }

    /* Measure the *pixel* size of the string. */
    wmessage = WIN_UTF8ToStringW(messageboxdata->message);
    SDL_zero(TextSize);
    DrawTextW(FontDC, wmessage, -1, &TextSize, DT_CALCRECT | DT_LEFT | DT_NOPREFIX | DT_EDITCONTROL);

    // Add margins and some padding for hangs, etc.
    TextSize.left += TextMargin;
    TextSize.right += TextMargin + 2;
    TextSize.top += TextMargin;
    TextSize.bottom += TextMargin + 2;

    // Done with the DC, and the string
    DeleteDC(FontDC);
    SDL_free(wmessage);

    // Increase the size of the dialog by some border spacing around the text.
    Size.cx = TextSize.right - TextSize.left;
    Size.cy = TextSize.bottom - TextSize.top;
    Size.cx += TextMargin * 2;
    Size.cy += TextMargin * 2;

    // Make dialog wider and shift text over for the icon.
    if (icon) {
        Size.cx += IconMargin + IconWidth;
        TextSize.left += IconMargin + IconWidth;
        TextSize.right += IconMargin + IconWidth;
    }

    // Ensure the size is wide enough for all of the buttons.
    if (Size.cx < (LONG)messageboxdata->numbuttons * (ButtonWidth + ButtonMargin) + ButtonMargin) {
        Size.cx = (LONG)messageboxdata->numbuttons * (ButtonWidth + ButtonMargin) + ButtonMargin;
    }

    // Reset the height to the icon size if it is actually bigger than the text.
    if (icon && Size.cy < (LONG)IconMargin * 2 + IconHeight) {
        Size.cy = (LONG)IconMargin * 2 + IconHeight;
    }

    // Add vertical space for the buttons and border.
    Size.cy += ButtonHeight + TextMargin;

    dialog = CreateDialogData(Size.cx, Size.cy, messageboxdata->title);
    if (!dialog) {
        return false;
    }

    if (icon && !AddDialogStaticIcon(dialog, IconMargin, IconMargin, IconWidth, IconHeight, icon)) {
        FreeDialogData(dialog);
        return false;
    }

    if (!AddDialogStaticText(dialog, TextSize.left, TextSize.top, TextSize.right - TextSize.left, TextSize.bottom - TextSize.top, messageboxdata->message)) {
        FreeDialogData(dialog);
        return false;
    }

    // Align the buttons to the right/bottom.
    x = Size.cx - (ButtonWidth + ButtonMargin) * messageboxdata->numbuttons;
    y = Size.cy - ButtonHeight - ButtonMargin;
    for (i = 0; i < messageboxdata->numbuttons; i++) {
        bool isdefault = false;
        const char *buttontext;
        const SDL_MessageBoxButtonData *sdlButton;

        /* We always have to create the dialog buttons from left to right
         * so that the tab order is correct.  Select the info to use
         * depending on which order was requested. */
        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT) {
            sdlButton = &messageboxdata->buttons[i];
        } else {
            sdlButton = &messageboxdata->buttons[messageboxdata->numbuttons - 1 - i];
        }

        if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) {
            defbuttoncount++;
            if (defbuttoncount == 1) {
                isdefault = true;
            }
        }

        buttontext = EscapeAmpersands(&ampescape, &ampescapesize, sdlButton->text);
        /* Make sure to provide the correct ID to keep buttons indexed in the
         * same order as how they are in messageboxdata. */
        if (!buttontext || !AddDialogButton(dialog, x, y, ButtonWidth, ButtonHeight, buttontext, IDBUTTONINDEX0 + (int)(sdlButton - messageboxdata->buttons), isdefault)) {
            FreeDialogData(dialog);
            SDL_free(ampescape);
            return false;
        }

        x += ButtonWidth + ButtonMargin;
    }
    SDL_free(ampescape);

    /* If we have a parent window, get the Instance and HWND for them
     * so that our little dialog gets exclusive focus at all times. */
    if (messageboxdata->window) {
        ParentWindow = messageboxdata->window->internal->hwnd;
    }

    rc = DialogBoxIndirectParam(NULL, (DLGTEMPLATE *)dialog->lpDialog, ParentWindow, MessageBoxDialogProc, (LPARAM)messageboxdata);
    if (rc >= IDBUTTONINDEX0 && rc - IDBUTTONINDEX0 < messageboxdata->numbuttons) {
        *buttonID = messageboxdata->buttons[rc - IDBUTTONINDEX0].buttonID;
        result = true;
    } else if (rc == IDCLOSED) {
        // Dialog window closed by user or system.
        // This could use a special return code.
        result = true;
        *buttonID = -1;
    } else {
        if (rc == 0) {
            SDL_SetError("Invalid parent window handle");
        } else if (rc == -1) {
            SDL_SetError("The message box encountered an error.");
        } else if (rc == IDINVALPTRINIT || rc == IDINVALPTRSETFOCUS || rc == IDINVALPTRCOMMAND) {
            SDL_SetError("Invalid message box pointer in dialog procedure");
        } else if (rc == IDINVALPTRDLGITEM) {
            SDL_SetError("Couldn't find dialog control of the default enter-key button");
        } else {
            SDL_SetError("An unknown error occurred");
        }
        result = false;
    }

    FreeDialogData(dialog);
    return result;
}

/* TaskDialogIndirect procedure
 * This is because SDL targets Windows XP (0x501), so this is not defined in the platform SDK.
 */
/* *INDENT-OFF* */ // clang-format off
typedef HRESULT (FAR WINAPI *TASKDIALOGINDIRECTPROC)(const TASKDIALOGCONFIG *pTaskConfig, int *pnButton, int *pnRadioButton, BOOL *pfVerificationFlagChecked);
/* *INDENT-ON* */ // clang-format on

bool WIN_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    HWND ParentWindow = NULL;
    wchar_t *wmessage;
    wchar_t *wtitle;
    TASKDIALOGCONFIG TaskConfig;
    TASKDIALOG_BUTTON *pButtons;
    TASKDIALOG_BUTTON *pButton;
    HMODULE hComctl32;
    TASKDIALOGINDIRECTPROC pfnTaskDialogIndirect;
    HRESULT hr;
    char *ampescape = NULL;
    size_t ampescapesize = 0;
    int nButton;
    int nCancelButton;
    int i;
    bool result = false;

    if (SIZE_MAX / sizeof(TASKDIALOG_BUTTON) < messageboxdata->numbuttons) {
        return SDL_OutOfMemory();
    }

    HMODULE hUser32 = GetModuleHandle(TEXT("user32.dll"));
    typedef DPI_AWARENESS_CONTEXT (WINAPI * SetThreadDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);
    SetThreadDpiAwarenessContext_t SetThreadDpiAwarenessContextFunc = (SetThreadDpiAwarenessContext_t)GetProcAddress(hUser32, "SetThreadDpiAwarenessContext");
    DPI_AWARENESS_CONTEXT previous_context = DPI_AWARENESS_CONTEXT_UNAWARE;
    if (SetThreadDpiAwarenessContextFunc) {
        previous_context = SetThreadDpiAwarenessContextFunc(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }

    // If we cannot load comctl32.dll use the old messagebox!
    hComctl32 = LoadLibrary(TEXT("comctl32.dll"));
    if (!hComctl32) {
        result = WIN_ShowOldMessageBox(messageboxdata, buttonID);
        goto done;
    }

    /* If TaskDialogIndirect doesn't exist use the old messagebox!
       This will fail prior to Windows Vista.
       The manifest file in the application may require targeting version 6 of comctl32.dll, even
       when we use LoadLibrary here!
       If you don't want to bother with manifests, put this #pragma in your app's source code somewhere:
       #pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0'  processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
     */
    pfnTaskDialogIndirect = (TASKDIALOGINDIRECTPROC)GetProcAddress(hComctl32, "TaskDialogIndirect");
    if (!pfnTaskDialogIndirect) {
        FreeLibrary(hComctl32);
        result = WIN_ShowOldMessageBox(messageboxdata, buttonID);
        goto done;
    }

    /* If we have a parent window, get the Instance and HWND for them
       so that our little dialog gets exclusive focus at all times. */
    if (messageboxdata->window) {
        ParentWindow = messageboxdata->window->internal->hwnd;
    }

    wmessage = WIN_UTF8ToStringW(messageboxdata->message);
    wtitle = WIN_UTF8ToStringW(messageboxdata->title);

    SDL_zero(TaskConfig);
    TaskConfig.cbSize = sizeof(TASKDIALOGCONFIG);
    TaskConfig.hwndParent = ParentWindow;
    TaskConfig.dwFlags = TDF_SIZE_TO_CONTENT;
    TaskConfig.pszWindowTitle = wtitle;
    if (messageboxdata->flags & SDL_MESSAGEBOX_ERROR) {
        TaskConfig.pszMainIcon = TD_ERROR_ICON;
    } else if (messageboxdata->flags & SDL_MESSAGEBOX_WARNING) {
        TaskConfig.pszMainIcon = TD_WARNING_ICON;
    } else if (messageboxdata->flags & SDL_MESSAGEBOX_INFORMATION) {
        TaskConfig.pszMainIcon = TD_INFORMATION_ICON;
    } else {
        TaskConfig.pszMainIcon = NULL;
    }

    TaskConfig.pszContent = wmessage;
    TaskConfig.cButtons = messageboxdata->numbuttons;
    pButtons = (TASKDIALOG_BUTTON *)SDL_malloc(sizeof(TASKDIALOG_BUTTON) * messageboxdata->numbuttons);
    TaskConfig.nDefaultButton = 0;
    nCancelButton = 0;
    for (i = 0; i < messageboxdata->numbuttons; i++) {
        const char *buttontext;
        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT) {
            pButton = &pButtons[i];
        } else {
            pButton = &pButtons[messageboxdata->numbuttons - 1 - i];
        }
        if (messageboxdata->buttons[i].flags & SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT) {
            nCancelButton = messageboxdata->buttons[i].buttonID;
            pButton->nButtonID = IDCANCEL;
        } else {
            pButton->nButtonID = IDBUTTONINDEX0 + i;
        }
        buttontext = EscapeAmpersands(&ampescape, &ampescapesize, messageboxdata->buttons[i].text);
        if (!buttontext) {
            int j;
            FreeLibrary(hComctl32);
            SDL_free(ampescape);
            SDL_free(wmessage);
            SDL_free(wtitle);
            for (j = 0; j < i; j++) {
                SDL_free((wchar_t *)pButtons[j].pszButtonText);
            }
            SDL_free(pButtons);
            return false;
        }
        pButton->pszButtonText = WIN_UTF8ToStringW(buttontext);
        if (messageboxdata->buttons[i].flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) {
            TaskConfig.nDefaultButton = pButton->nButtonID;
        }
    }
    TaskConfig.pButtons = pButtons;

    // Show the Task Dialog
    hr = pfnTaskDialogIndirect(&TaskConfig, &nButton, NULL, NULL);

    // Free everything
    FreeLibrary(hComctl32);
    SDL_free(ampescape);
    SDL_free(wmessage);
    SDL_free(wtitle);
    for (i = 0; i < messageboxdata->numbuttons; i++) {
        SDL_free((wchar_t *)pButtons[i].pszButtonText);
    }
    SDL_free(pButtons);

    // Check the Task Dialog was successful and give the result
    if (SUCCEEDED(hr)) {
        if (nButton == IDCANCEL) {
            *buttonID = nCancelButton;
        } else if (nButton >= IDBUTTONINDEX0 && nButton < IDBUTTONINDEX0 + messageboxdata->numbuttons) {
            *buttonID = messageboxdata->buttons[nButton - IDBUTTONINDEX0].buttonID;
        } else {
            *buttonID = -1;
        }
        result = true;
    } else {
        // We failed showing the Task Dialog, use the old message box!
        result = WIN_ShowOldMessageBox(messageboxdata, buttonID);
    }

done:
    if (SetThreadDpiAwarenessContextFunc) {
        SetThreadDpiAwarenessContextFunc(previous_context);
    }
    return result;
}

#endif // SDL_VIDEO_DRIVER_WINDOWS
