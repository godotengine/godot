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

#include "SDL_windowsvideo.h"

#include "../../events/SDL_keyboard_c.h"
#include "../../events/scancodes_windows.h"

#include <imm.h>
#include <oleauto.h>

#ifndef SDL_DISABLE_WINDOWS_IME
#if 0
#define SDL_DebugIMELog SDL_Log
#else
#define SDL_DebugIMELog(...)
#endif
static bool IME_Init(SDL_VideoData *videodata, SDL_Window *window);
static void IME_Enable(SDL_VideoData *videodata, HWND hwnd);
static void IME_Disable(SDL_VideoData *videodata, HWND hwnd);
static void IME_SetTextInputArea(SDL_VideoData *videodata, HWND hwnd, const SDL_Rect *rect, int cursor);
static void IME_ClearComposition(SDL_VideoData *videodata);
static void IME_GetCandidateList(SDL_VideoData *videodata, HWND hwnd);
static void IME_Quit(SDL_VideoData *videodata);
#else
static void IME_SetTextInputArea(SDL_VideoData *videodata, HWND hwnd, const SDL_Rect *rect, int cursor);
#endif // !SDL_DISABLE_WINDOWS_IME

#ifndef MAPVK_VK_TO_VSC
#define MAPVK_VK_TO_VSC 0
#endif
#ifndef MAPVK_VSC_TO_VK
#define MAPVK_VSC_TO_VK 1
#endif

// Alphabetic scancodes for PC keyboards
void WIN_InitKeyboard(SDL_VideoDevice *_this)
{
#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_VideoData *data = _this->internal;

    data->ime_candlistindexbase = 1;
    data->ime_composition_length = 32 * sizeof(WCHAR);
    data->ime_composition = (WCHAR *)SDL_calloc(data->ime_composition_length, sizeof(WCHAR));
#endif // !SDL_DISABLE_WINDOWS_IME

    WIN_UpdateKeymap(false);

    SDL_SetScancodeName(SDL_SCANCODE_APPLICATION, "Menu");
    SDL_SetScancodeName(SDL_SCANCODE_LGUI, "Left Windows");
    SDL_SetScancodeName(SDL_SCANCODE_RGUI, "Right Windows");

    // Are system caps/num/scroll lock active? Set our state to match.
    SDL_ToggleModState(SDL_KMOD_CAPS, (GetKeyState(VK_CAPITAL) & 0x0001) ? true : false);
    SDL_ToggleModState(SDL_KMOD_NUM, (GetKeyState(VK_NUMLOCK) & 0x0001) ? true : false);
    SDL_ToggleModState(SDL_KMOD_SCROLL, (GetKeyState(VK_SCROLL) & 0x0001) ? true : false);
}

void WIN_UpdateKeymap(bool send_event)
{
    SDL_Scancode scancode;
    SDL_Keymap *keymap;
    BYTE keyboardState[256] = { 0 };
    WCHAR buffer[16];
    SDL_Keymod mods[] = {
        SDL_KMOD_NONE,
        SDL_KMOD_SHIFT,
        SDL_KMOD_CAPS,
        (SDL_KMOD_SHIFT | SDL_KMOD_CAPS),
        SDL_KMOD_MODE,
        (SDL_KMOD_MODE | SDL_KMOD_SHIFT),
        (SDL_KMOD_MODE | SDL_KMOD_CAPS),
        (SDL_KMOD_MODE | SDL_KMOD_SHIFT | SDL_KMOD_CAPS)
    };

    WIN_ResetDeadKeys();

    keymap = SDL_CreateKeymap(true);

    for (int m = 0; m < SDL_arraysize(mods); ++m) {
        for (int i = 0; i < SDL_arraysize(windows_scancode_table); i++) {
            int vk, sc, result;
            Uint32 *ch = 0;

            // Make sure this scancode is a valid character scancode
            scancode = windows_scancode_table[i];
            if (scancode == SDL_SCANCODE_UNKNOWN ||
                scancode == SDL_SCANCODE_DELETE ||
                (SDL_GetKeymapKeycode(NULL, scancode, SDL_KMOD_NONE) & SDLK_SCANCODE_MASK)) {

                // The Colemak mapping swaps Backspace and CapsLock
                if (mods[m] == SDL_KMOD_NONE &&
                    (scancode == SDL_SCANCODE_CAPSLOCK ||
                     scancode == SDL_SCANCODE_BACKSPACE)) {
                    vk = LOBYTE(MapVirtualKey(i, MAPVK_VSC_TO_VK));
                    if (vk == VK_CAPITAL) {
                        SDL_SetKeymapEntry(keymap, scancode, mods[m], SDLK_CAPSLOCK);
                    } else if (vk == VK_BACK) {
                        SDL_SetKeymapEntry(keymap, scancode, mods[m], SDLK_BACKSPACE);
                    }
                }
                continue;
            }

            // Unpack the single byte index to make the scan code.
            sc = MAKEWORD(i & 0x7f, (i & 0x80) ? 0xe0 : 0x00);
            vk = LOBYTE(MapVirtualKey(sc, MAPVK_VSC_TO_VK));
            if (!vk) {
                continue;
            }

            // Update the keyboard state for the modifiers
            keyboardState[VK_SHIFT] = (mods[m] & SDL_KMOD_SHIFT) ? 0x80 : 0x00;
            keyboardState[VK_CAPITAL] = (mods[m] & SDL_KMOD_CAPS) ? 0x01 : 0x00;
            keyboardState[VK_CONTROL] = (mods[m] & SDL_KMOD_MODE) ? 0x80 : 0x00;
            keyboardState[VK_MENU] = (mods[m] & SDL_KMOD_MODE) ? 0x80 : 0x00;

            result = ToUnicode(vk, sc, keyboardState, buffer, 16, 0);
            buffer[SDL_abs(result)] = 0;

            // Convert UTF-16 to UTF-32 code points
            ch = (Uint32 *)SDL_iconv_string("UTF-32LE", "UTF-16LE", (const char *)buffer, (SDL_abs(result) + 1) * sizeof(WCHAR));
            if (ch) {
                /* Windows keyboard layouts can emit several UTF-32 code points on a single key press.
                 * Use <U+FFFD REPLACEMENT CHARACTER> since we cannot fit into single SDL_Keycode value in SDL keymap.
                 * See https://kbdlayout.info/features/ligatures for a list of such keys. */
                SDL_SetKeymapEntry(keymap, scancode, mods[m], ch[1] == 0 ? ch[0] : 0xfffd);
                SDL_free(ch);
            } else {
                // The default keymap doesn't have any SDL_KMOD_MODE entries, so we don't need to override them
                if (!(mods[m] & SDL_KMOD_MODE)) {
                    SDL_SetKeymapEntry(keymap, scancode, mods[m], SDLK_UNKNOWN);
                }
            }

            if (result < 0) {
                WIN_ResetDeadKeys();
            }
        }
    }

    SDL_SetKeymap(keymap, send_event);
}

void WIN_QuitKeyboard(SDL_VideoDevice *_this)
{
#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_VideoData *data = _this->internal;

    IME_Quit(data);

    if (data->ime_composition) {
        SDL_free(data->ime_composition);
        data->ime_composition = NULL;
    }
#endif // !SDL_DISABLE_WINDOWS_IME
}

void WIN_ResetDeadKeys(void)
{
    /*
    if a deadkey has been typed, but not the next character (which the deadkey might modify),
    this tries to undo the effect pressing the deadkey.
    see: http://archives.miloush.net/michkap/archive/2006/09/10/748775.html
    */
    BYTE keyboardState[256];
    WCHAR buffer[16];
    int vk, sc, result, i;

    if (!GetKeyboardState(keyboardState)) {
        return;
    }

    vk = VK_SPACE;
    sc = MapVirtualKey(vk, MAPVK_VK_TO_VSC);
    if (sc == 0) {
        // the keyboard doesn't have this key
        return;
    }

    for (i = 0; i < 5; i++) {
        result = ToUnicode(vk, sc, keyboardState, buffer, 16, 0);
        if (result > 0) {
            // success
            return;
        }
    }
}

bool WIN_StartTextInput(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    WIN_ResetDeadKeys();

#ifndef SDL_DISABLE_WINDOWS_IME
    HWND hwnd = window->internal->hwnd;
    SDL_VideoData *videodata = _this->internal;
    IME_Init(videodata, window);
    IME_Enable(videodata, hwnd);

    WIN_UpdateTextInputArea(_this, window);
#endif // !SDL_DISABLE_WINDOWS_IME

    return true;
}

bool WIN_StopTextInput(SDL_VideoDevice *_this, SDL_Window *window)
{
    WIN_ResetDeadKeys();

#ifndef SDL_DISABLE_WINDOWS_IME
    HWND hwnd = window->internal->hwnd;
    SDL_VideoData *videodata = _this->internal;
    IME_Init(videodata, window);
    IME_Disable(videodata, hwnd);
#endif // !SDL_DISABLE_WINDOWS_IME

    return true;
}

bool WIN_UpdateTextInputArea(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_WindowData *data = window->internal;

    IME_SetTextInputArea(videodata, data->hwnd, &window->text_input_rect, window->text_input_cursor);
    return true;
}

bool WIN_ClearComposition(SDL_VideoDevice *_this, SDL_Window *window)
{
#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_VideoData *videodata = _this->internal;

    IME_ClearComposition(videodata);
#endif
    return true;
}

#ifdef SDL_DISABLE_WINDOWS_IME

bool WIN_HandleIMEMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM *lParam, SDL_VideoData *videodata)
{
    return false;
}

void WIN_UpdateIMECandidates(SDL_VideoDevice *_this)
{
    return;
}

#else

#define LANG_CHT MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL)
#define LANG_CHS MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED)

#define MAKEIMEVERSION(major, minor) ((DWORD)(((BYTE)(major) << 24) | ((BYTE)(minor) << 16)))
#define IMEID_VER(id)                ((id)&0xffff0000)
#define IMEID_LANG(id)               ((id)&0x0000ffff)

#define CHT_HKL_DAYI          ((HKL)(UINT_PTR)0xE0060404)
#define CHT_HKL_NEW_PHONETIC  ((HKL)(UINT_PTR)0xE0080404)
#define CHT_HKL_NEW_CHANG_JIE ((HKL)(UINT_PTR)0xE0090404)
#define CHT_HKL_NEW_QUICK     ((HKL)(UINT_PTR)0xE00A0404)
#define CHT_HKL_HK_CANTONESE  ((HKL)(UINT_PTR)0xE00B0404)
#define CHT_IMEFILENAME1      "TINTLGNT.IME"
#define CHT_IMEFILENAME2      "CINTLGNT.IME"
#define CHT_IMEFILENAME3      "MSTCIPHA.IME"
#define IMEID_CHT_VER42       (LANG_CHT | MAKEIMEVERSION(4, 2))
#define IMEID_CHT_VER43       (LANG_CHT | MAKEIMEVERSION(4, 3))
#define IMEID_CHT_VER44       (LANG_CHT | MAKEIMEVERSION(4, 4))
#define IMEID_CHT_VER50       (LANG_CHT | MAKEIMEVERSION(5, 0))
#define IMEID_CHT_VER51       (LANG_CHT | MAKEIMEVERSION(5, 1))
#define IMEID_CHT_VER52       (LANG_CHT | MAKEIMEVERSION(5, 2))
#define IMEID_CHT_VER60       (LANG_CHT | MAKEIMEVERSION(6, 0))
#define IMEID_CHT_VER_VISTA   (LANG_CHT | MAKEIMEVERSION(7, 0))

#define CHS_HKL          ((HKL)(UINT_PTR)0xE00E0804)
#define CHS_IMEFILENAME1 "PINTLGNT.IME"
#define CHS_IMEFILENAME2 "MSSCIPYA.IME"
#define IMEID_CHS_VER41  (LANG_CHS | MAKEIMEVERSION(4, 1))
#define IMEID_CHS_VER42  (LANG_CHS | MAKEIMEVERSION(4, 2))
#define IMEID_CHS_VER53  (LANG_CHS | MAKEIMEVERSION(5, 3))

#define LANG()         LOWORD((videodata->ime_hkl))
#define PRIMLANG()     ((WORD)PRIMARYLANGID(LANG()))
#define SUBLANG()      SUBLANGID(LANG())

static void IME_UpdateInputLocale(SDL_VideoData *videodata);
static void IME_SetWindow(SDL_VideoData *videodata, SDL_Window *window);
static void IME_SetupAPI(SDL_VideoData *videodata);
static DWORD IME_GetId(SDL_VideoData *videodata, UINT uIndex);
static void IME_SendEditingEvent(SDL_VideoData *videodata);
static void IME_SendClearComposition(SDL_VideoData *videodata);

static bool IME_Init(SDL_VideoData *videodata, SDL_Window *window)
{
    HWND hwnd = window->internal->hwnd;

    if (videodata->ime_initialized) {
        return true;
    }

    const char *hint = SDL_GetHint(SDL_HINT_IME_IMPLEMENTED_UI);
    if (hint && SDL_strstr(hint, "composition")) {
        videodata->ime_internal_composition = true;
    }
    if (hint && SDL_strstr(hint, "candidates")) {
        videodata->ime_internal_candidates = true;
    }

    videodata->ime_hwnd_main = hwnd;
    videodata->ime_initialized = true;
    videodata->ime_himm32 = SDL_LoadObject("imm32.dll");
    if (!videodata->ime_himm32) {
        videodata->ime_available = false;
        SDL_ClearError();
        return true;
    }
    /* *INDENT-OFF* */ // clang-format off
    videodata->ImmLockIMC = (LPINPUTCONTEXT2 (WINAPI *)(HIMC))SDL_LoadFunction(videodata->ime_himm32, "ImmLockIMC");
    videodata->ImmUnlockIMC = (BOOL (WINAPI *)(HIMC))SDL_LoadFunction(videodata->ime_himm32, "ImmUnlockIMC");
    videodata->ImmLockIMCC = (LPVOID (WINAPI *)(HIMCC))SDL_LoadFunction(videodata->ime_himm32, "ImmLockIMCC");
    videodata->ImmUnlockIMCC = (BOOL (WINAPI *)(HIMCC))SDL_LoadFunction(videodata->ime_himm32, "ImmUnlockIMCC");
    /* *INDENT-ON* */ // clang-format on

    IME_SetWindow(videodata, window);
    videodata->ime_himc = ImmGetContext(hwnd);
    ImmReleaseContext(hwnd, videodata->ime_himc);
    if (!videodata->ime_himc) {
        videodata->ime_available = false;
        IME_Disable(videodata, hwnd);
        return true;
    }
    videodata->ime_available = true;
    IME_UpdateInputLocale(videodata);
    IME_SetupAPI(videodata);
    IME_UpdateInputLocale(videodata);
    IME_Disable(videodata, hwnd);
    return true;
}

static void IME_Enable(SDL_VideoData *videodata, HWND hwnd)
{
    if (!videodata->ime_initialized || !videodata->ime_hwnd_current) {
        return;
    }

    if (!videodata->ime_available) {
        IME_Disable(videodata, hwnd);
        return;
    }
    if (videodata->ime_hwnd_current == videodata->ime_hwnd_main) {
        ImmAssociateContext(videodata->ime_hwnd_current, videodata->ime_himc);
    }

    videodata->ime_enabled = true;
    IME_UpdateInputLocale(videodata);
}

static void IME_Disable(SDL_VideoData *videodata, HWND hwnd)
{
    if (!videodata->ime_initialized || !videodata->ime_hwnd_current) {
        return;
    }

    IME_ClearComposition(videodata);
    if (videodata->ime_hwnd_current == videodata->ime_hwnd_main) {
        ImmAssociateContext(videodata->ime_hwnd_current, (HIMC)0);
    }

    videodata->ime_enabled = false;
}

static void IME_Quit(SDL_VideoData *videodata)
{
    if (!videodata->ime_initialized) {
        return;
    }

    if (videodata->ime_hwnd_main) {
        ImmAssociateContext(videodata->ime_hwnd_main, videodata->ime_himc);
    }

    videodata->ime_hwnd_main = 0;
    videodata->ime_himc = 0;
    if (videodata->ime_himm32) {
        SDL_UnloadObject(videodata->ime_himm32);
        videodata->ime_himm32 = 0;
    }
    for (int i = 0; i < videodata->ime_candcount; ++i) {
        SDL_free(videodata->ime_candidates[i]);
        videodata->ime_candidates[i] = NULL;
    }
    videodata->ime_initialized = false;
}

static void IME_GetReadingString(SDL_VideoData *videodata, HWND hwnd)
{
    DWORD id = 0;
    HIMC himc = 0;
    WCHAR buffer[16];
    WCHAR *s = buffer;
    DWORD len = 0;
    INT err = 0;
    BOOL vertical = FALSE;
    UINT maxuilen = 0;

    videodata->ime_readingstring[0] = 0;

    id = IME_GetId(videodata, 0);
    if (!id) {
        return;
    }

    himc = ImmGetContext(hwnd);
    if (!himc) {
        return;
    }

    if (videodata->GetReadingString) {
        len = videodata->GetReadingString(himc, 0, 0, &err, &vertical, &maxuilen);
        if (len) {
            if (len > SDL_arraysize(buffer)) {
                len = SDL_arraysize(buffer);
            }

            len = videodata->GetReadingString(himc, len, s, &err, &vertical, &maxuilen);
        }
        SDL_wcslcpy(videodata->ime_readingstring, s, len);
    } else {
        LPINPUTCONTEXT2 lpimc = videodata->ImmLockIMC(himc);
        LPBYTE p = 0;
        s = 0;
        switch (id) {
        case IMEID_CHT_VER42:
        case IMEID_CHT_VER43:
        case IMEID_CHT_VER44:
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + 24);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 7 * 4 + 32 * 4);
            s = (WCHAR *)(p + 56);
            break;
        case IMEID_CHT_VER51:
        case IMEID_CHT_VER52:
        case IMEID_CHS_VER53:
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + 4);
            if (!p) {
                break;
            }

            p = *(LPBYTE *)(p + 1 * 4 + 5 * 4);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4 + 16 * 2);
            s = (WCHAR *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4);
            break;
        case IMEID_CHS_VER41:
        {
            int offset = (IME_GetId(videodata, 1) >= 0x00000002) ? 8 : 7;
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + offset * 4);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 7 * 4 + 16 * 2 * 4);
            s = (WCHAR *)(p + 6 * 4 + 16 * 2 * 1);
        } break;
        case IMEID_CHS_VER42:
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + 1 * 4 + 1 * 4 + 6 * 4);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4 + 16 * 2);
            s = (WCHAR *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4);
            break;
        }
        if (s) {
            size_t size = SDL_min((size_t)(len + 1), SDL_arraysize(videodata->ime_readingstring));
            SDL_wcslcpy(videodata->ime_readingstring, s, size);
        }

        videodata->ImmUnlockIMCC(lpimc->hPrivate);
        videodata->ImmUnlockIMC(himc);
    }
    ImmReleaseContext(hwnd, himc);
    IME_SendEditingEvent(videodata);
}

static void IME_InputLangChanged(SDL_VideoData *videodata)
{
    UINT lang = PRIMLANG();
    IME_UpdateInputLocale(videodata);

    IME_SetupAPI(videodata);
    if (lang != PRIMLANG()) {
        IME_ClearComposition(videodata);
    }
}

static DWORD IME_GetId(SDL_VideoData *videodata, UINT uIndex)
{
    static HKL hklprev = 0;
    static DWORD dwRet[2] = { 0 };
    DWORD dwVerSize = 0;
    DWORD dwVerHandle = 0;
    LPVOID lpVerBuffer = 0;
    LPVOID lpVerData = 0;
    UINT cbVerData = 0;
    char szTemp[256];
    HKL hkl = 0;
    DWORD dwLang = 0;
    SDL_assert(uIndex < sizeof(dwRet) / sizeof(dwRet[0]));

    hkl = videodata->ime_hkl;
    if (hklprev == hkl) {
        return dwRet[uIndex];
    }
    hklprev = hkl;

    SDL_assert(uIndex == 0);
    dwLang = ((DWORD_PTR)hkl & 0xffff);
    // FIXME: What does this do?
    if (videodata->ime_internal_candidates && dwLang == LANG_CHT) {
        dwRet[0] = IMEID_CHT_VER_VISTA;
        dwRet[1] = 0;
        return dwRet[0];
    }
    if (hkl != CHT_HKL_NEW_PHONETIC && hkl != CHT_HKL_NEW_CHANG_JIE && hkl != CHT_HKL_NEW_QUICK && hkl != CHT_HKL_HK_CANTONESE && hkl != CHS_HKL) {
        dwRet[0] = dwRet[1] = 0;
        return dwRet[0];
    }
    if (!ImmGetIMEFileNameA(hkl, szTemp, sizeof(szTemp) - 1)) {
        dwRet[0] = dwRet[1] = 0;
        return dwRet[0];
    }
    if (!videodata->GetReadingString) {
#define LCID_INVARIANT MAKELCID(MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), SORT_DEFAULT)
        if (CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHT_IMEFILENAME1, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHT_IMEFILENAME2, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHT_IMEFILENAME3, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHS_IMEFILENAME1, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHS_IMEFILENAME2, -1) != 2) {
            dwRet[0] = dwRet[1] = 0;
            return dwRet[0];
        }
#undef LCID_INVARIANT
        dwVerSize = GetFileVersionInfoSizeA(szTemp, &dwVerHandle);
        if (dwVerSize) {
            lpVerBuffer = SDL_malloc(dwVerSize);
            if (lpVerBuffer) {
                if (GetFileVersionInfoA(szTemp, dwVerHandle, dwVerSize, lpVerBuffer)) {
                    if (VerQueryValueA(lpVerBuffer, "\\", &lpVerData, &cbVerData)) {
#define pVerFixedInfo ((VS_FIXEDFILEINFO FAR *)lpVerData)
                        DWORD dwVer = pVerFixedInfo->dwFileVersionMS;
                        dwVer = (dwVer & 0x00ff0000) << 8 | (dwVer & 0x000000ff) << 16;
                        if ((videodata->GetReadingString) ||
                            ((dwLang == LANG_CHT) && (dwVer == MAKEIMEVERSION(4, 2) ||
                                                      dwVer == MAKEIMEVERSION(4, 3) ||
                                                      dwVer == MAKEIMEVERSION(4, 4) ||
                                                      dwVer == MAKEIMEVERSION(5, 0) ||
                                                      dwVer == MAKEIMEVERSION(5, 1) ||
                                                      dwVer == MAKEIMEVERSION(5, 2) ||
                                                      dwVer == MAKEIMEVERSION(6, 0))) ||
                            ((dwLang == LANG_CHS) && (dwVer == MAKEIMEVERSION(4, 1) ||
                                                      dwVer == MAKEIMEVERSION(4, 2) ||
                                                      dwVer == MAKEIMEVERSION(5, 3)))) {
                            dwRet[0] = dwVer | dwLang;
                            dwRet[1] = pVerFixedInfo->dwFileVersionLS;
                            SDL_free(lpVerBuffer);
                            return dwRet[0];
                        }
#undef pVerFixedInfo
                    }
                }
            }
            SDL_free(lpVerBuffer);
        }
    }
    dwRet[0] = dwRet[1] = 0;
    return dwRet[0];
}

static void IME_SetupAPI(SDL_VideoData *videodata)
{
    char ime_file[MAX_PATH + 1];
    SDL_SharedObject *hime = 0;
    HKL hkl = 0;
    videodata->GetReadingString = NULL;
    videodata->ShowReadingWindow = NULL;

    hkl = videodata->ime_hkl;
    if (!ImmGetIMEFileNameA(hkl, ime_file, sizeof(ime_file) - 1)) {
        return;
    }

    hime = SDL_LoadObject(ime_file);
    if (!hime) {
        return;
    }

    /* *INDENT-OFF* */ // clang-format off
    videodata->GetReadingString = (UINT (WINAPI *)(HIMC, UINT, LPWSTR, PINT, BOOL*, PUINT))
        SDL_LoadFunction(hime, "GetReadingString");
    videodata->ShowReadingWindow = (BOOL (WINAPI *)(HIMC, BOOL))
        SDL_LoadFunction(hime, "ShowReadingWindow");
    /* *INDENT-ON* */ // clang-format on

    if (videodata->ShowReadingWindow) {
        HIMC himc = ImmGetContext(videodata->ime_hwnd_current);
        if (himc) {
            videodata->ShowReadingWindow(himc, FALSE);
            ImmReleaseContext(videodata->ime_hwnd_current, himc);
        }
    }
}

static void IME_SetWindow(SDL_VideoData *videodata, SDL_Window *window)
{
    HWND hwnd = window->internal->hwnd;

    if (hwnd != videodata->ime_hwnd_current) {
        videodata->ime_hwnd_current = hwnd;
        SDL_zero(videodata->ime_composition_area);
        SDL_zero(videodata->ime_candidate_area);
    }

    IME_SetTextInputArea(videodata, hwnd, &window->text_input_rect, window->text_input_cursor);
}

#endif

static void IME_SetTextInputArea(SDL_VideoData *videodata, HWND hwnd, const SDL_Rect *rect, int cursor)
{
    HIMC himc;

    himc = ImmGetContext(hwnd);
    if (himc) {
        COMPOSITIONFORM cof;
        CANDIDATEFORM caf;
        int font_height = rect->h;

        LOGFONTW font;
        if (ImmGetCompositionFontW(himc, &font)) {
            font_height = font.lfHeight;
        }

        SDL_zero(cof);
        cof.dwStyle = CFS_RECT;
        cof.ptCurrentPos.x = rect->x + cursor;
        cof.ptCurrentPos.y = rect->y + (rect->h - font_height) / 2;
        cof.rcArea.left = rect->x;
        cof.rcArea.right = (LONG)rect->x + rect->w;
        cof.rcArea.top = rect->y;
        cof.rcArea.bottom = (LONG)rect->y + rect->h;
        if (SDL_memcmp(&cof, &videodata->ime_composition_area, sizeof(cof)) != 0) {
            SDL_copyp(&videodata->ime_composition_area, &cof);
            ImmSetCompositionWindow(himc, &cof);
        }

        SDL_zero(caf);
        caf.dwIndex = 0;
        caf.dwStyle = CFS_EXCLUDE;
        caf.ptCurrentPos.x = rect->x + cursor;
        caf.ptCurrentPos.y = rect->y;
        caf.rcArea.left = rect->x;
        caf.rcArea.right = (LONG)rect->x + rect->w;
        caf.rcArea.top = rect->y;
        caf.rcArea.bottom = (LONG)rect->y + rect->h;
        if (SDL_memcmp(&caf, &videodata->ime_candidate_area, sizeof(caf)) != 0) {
            SDL_copyp(&videodata->ime_candidate_area, &caf);
            ImmSetCandidateWindow(himc, &caf);
        }

        ImmReleaseContext(hwnd, himc);
    }
}

#ifndef SDL_DISABLE_WINDOWS_IME

static void IME_UpdateInputLocale(SDL_VideoData *videodata)
{
    HKL hklnext = GetKeyboardLayout(0);

    if (hklnext == videodata->ime_hkl) {
        return;
    }

    videodata->ime_hkl = hklnext;
    videodata->ime_horizontal_candidates = (PRIMLANG() == LANG_KOREAN || LANG() == LANG_CHS);
    videodata->ime_candlistindexbase = (videodata->ime_hkl == CHT_HKL_DAYI) ? 0 : 1;
}

static void IME_ClearComposition(SDL_VideoData *videodata)
{
    HIMC himc = 0;
    if (!videodata->ime_initialized) {
        return;
    }

    himc = ImmGetContext(videodata->ime_hwnd_current);
    if (!himc) {
        return;
    }

    ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_CANCEL, 0);
    ImmSetCompositionString(himc, SCS_SETSTR, TEXT(""), sizeof(TCHAR), TEXT(""), sizeof(TCHAR));

    ImmNotifyIME(himc, NI_CLOSECANDIDATE, 0, 0);
    ImmReleaseContext(videodata->ime_hwnd_current, himc);
    IME_SendClearComposition(videodata);
}

static void IME_GetCompositionString(SDL_VideoData *videodata, HIMC himc, DWORD string)
{
    LONG length;
    DWORD dwLang = ((DWORD_PTR)videodata->ime_hkl & 0xffff);

    videodata->ime_cursor = LOWORD(ImmGetCompositionStringW(himc, GCS_CURSORPOS, 0, 0));
    videodata->ime_selected_start = 0;
    videodata->ime_selected_length = 0;
    SDL_DebugIMELog("Cursor = %d", videodata->ime_cursor);

    length = ImmGetCompositionStringW(himc, string, NULL, 0);
    if (length > 0 && videodata->ime_composition_length < length) {
        if (videodata->ime_composition) {
            SDL_free(videodata->ime_composition);
        }

        videodata->ime_composition = (WCHAR *)SDL_malloc(length + sizeof(WCHAR));
        videodata->ime_composition_length = length;
    }

    length = ImmGetCompositionStringW(himc, string, videodata->ime_composition, videodata->ime_composition_length);
    if (length < 0) {
        length = 0;
    }
    length /= sizeof(WCHAR);

    if ((dwLang == LANG_CHT || dwLang == LANG_CHS) &&
        videodata->ime_cursor > 0 &&
        videodata->ime_cursor < (int)(videodata->ime_composition_length / sizeof(WCHAR)) &&
        (videodata->ime_composition[0] == 0x3000 || videodata->ime_composition[0] == 0x0020)) {
        // Traditional Chinese IMEs add a placeholder U+3000
        // Simplified Chinese IMEs seem to add a placeholder U+0020 sometimes
        for (int i = videodata->ime_cursor + 1; i < length; ++i) {
            videodata->ime_composition[i - 1] = videodata->ime_composition[i];
        }
        --length;
    }

    videodata->ime_composition[length] = 0;

    length = ImmGetCompositionStringW(himc, GCS_COMPATTR, NULL, 0);
    if (length > 0) {
        Uint8 *attributes = (Uint8 *)SDL_malloc(length);
        if (attributes) {
            int start = 0;
            int end = 0;

            length = ImmGetCompositionString(himc, GCS_COMPATTR, attributes, length);
            if (length < 0) {
                length = 0;
            }

            for (LONG i = 0; i < length; ++i) {
                SDL_DebugIMELog("attrib[%d] = %d", i, attributes[i]);
            }

            for (start = 0; start < length; ++start) {
                if (attributes[start] == ATTR_TARGET_CONVERTED || attributes[start] == ATTR_TARGET_NOTCONVERTED) {
                    break;
                }
            }

            for (end = start; end < length; ++end) {
                if (attributes[end] != ATTR_TARGET_CONVERTED && attributes[end] != ATTR_TARGET_NOTCONVERTED) {
                    break;
                }
            }

            if (end > start) {
                videodata->ime_selected_start = start;
                videodata->ime_selected_length = end - start;
            }

            SDL_free(attributes);
        }
    }
}

static void IME_SendInputEvent(SDL_VideoData *videodata)
{
    char *s = 0;
    s = WIN_StringToUTF8W(videodata->ime_composition);
    SDL_SendKeyboardText(s);
    SDL_free(s);

    videodata->ime_composition[0] = 0;
    videodata->ime_readingstring[0] = 0;
    videodata->ime_cursor = 0;
}

static void IME_SendEditingEvent(SDL_VideoData *videodata)
{
    char *s = NULL;
    WCHAR *buffer = NULL;
    size_t size = videodata->ime_composition_length;
    if (videodata->ime_readingstring[0]) {
        size_t len = SDL_min(SDL_wcslen(videodata->ime_composition), (size_t)videodata->ime_cursor);

        size += sizeof(videodata->ime_readingstring);
        buffer = (WCHAR *)SDL_malloc(size + sizeof(WCHAR));
        if (!buffer) {
            return;
        }
        buffer[0] = 0;

        SDL_wcslcpy(buffer, videodata->ime_composition, len + 1);
        SDL_wcslcat(buffer, videodata->ime_readingstring, size);
        SDL_wcslcat(buffer, &videodata->ime_composition[len], size);
    } else {
        buffer = (WCHAR *)SDL_malloc(size + sizeof(WCHAR));
        if (!buffer) {
            return;
        }
        buffer[0] = 0;
        SDL_wcslcpy(buffer, videodata->ime_composition, size);
    }

    s = WIN_StringToUTF8W(buffer);
    if (s) {
        if (videodata->ime_readingstring[0]) {
            SDL_SendEditingText(s, videodata->ime_cursor, (int)SDL_wcslen(videodata->ime_readingstring));
        } else if (videodata->ime_cursor == videodata->ime_selected_start) {
            SDL_SendEditingText(s, videodata->ime_selected_start, videodata->ime_selected_length);
        } else {
            SDL_SendEditingText(s, videodata->ime_cursor, 0);
        }
        if (*s) {
            videodata->ime_needs_clear_composition = true;
        }
        SDL_free(s);
    }
    SDL_free(buffer);
}

static void IME_SendClearComposition(SDL_VideoData *videodata)
{
    if (videodata->ime_needs_clear_composition) {
        SDL_SendEditingText("", 0, 0);
        videodata->ime_needs_clear_composition = false;
    }
}

static bool IME_OpenCandidateList(SDL_VideoData *videodata)
{
    videodata->ime_candidates_open = true;
    videodata->ime_candcount = 0;
    return true;
}

static void IME_AddCandidate(SDL_VideoData *videodata, UINT i, LPCWSTR candidate)
{
    if (videodata->ime_candidates[i]) {
        SDL_free(videodata->ime_candidates[i]);
        videodata->ime_candidates[i] = NULL;
    }

    SDL_COMPILE_TIME_ASSERT(IME_CANDIDATE_INDEXING_REQUIRES, MAX_CANDLIST == 10);
    char *candidate_utf8 = WIN_StringToUTF8W(candidate);
    SDL_asprintf(&videodata->ime_candidates[i], "%d %s", ((i + videodata->ime_candlistindexbase) % 10), candidate_utf8);
    SDL_free(candidate_utf8);

    videodata->ime_candcount = (i + 1);
}

static void IME_SendCandidateList(SDL_VideoData *videodata)
{
    SDL_SendEditingTextCandidates(videodata->ime_candidates, videodata->ime_candcount, videodata->ime_candsel, videodata->ime_horizontal_candidates);
}

static void IME_CloseCandidateList(SDL_VideoData *videodata)
{
    videodata->ime_candidates_open = false;

    if (videodata->ime_candcount > 0) {
        for (int i = 0; i < videodata->ime_candcount; ++i) {
            SDL_free(videodata->ime_candidates[i]);
            videodata->ime_candidates[i] = NULL;
        }
        videodata->ime_candcount = 0;

        SDL_SendEditingTextCandidates(NULL, 0, -1, false);
    }
}

static void IME_GetCandidateList(SDL_VideoData *videodata, HWND hwnd)
{
    HIMC himc;
    DWORD size;
    LPCANDIDATELIST cand_list;
    bool has_candidates = false;

    himc = ImmGetContext(hwnd);
    if (himc) {
        size = ImmGetCandidateListW(himc, 0, NULL, 0);
        if (size != 0) {
            cand_list = (LPCANDIDATELIST)SDL_malloc(size);
            if (cand_list != NULL) {
                size = ImmGetCandidateListW(himc, 0, cand_list, size);
                if (size != 0) {
                    if (IME_OpenCandidateList(videodata)) {
                        UINT i, j;
                        UINT page_start = 0;
                        UINT page_size = 0;

                        videodata->ime_candsel = cand_list->dwSelection;

                        if (LANG() == LANG_CHS && IME_GetId(videodata, 0)) {
                            const UINT maxcandchar = 18;
                            size_t cchars = 0;

                            for (i = 0; i < cand_list->dwCount; ++i) {
                                size_t len = SDL_wcslen((LPWSTR)((DWORD_PTR)cand_list + cand_list->dwOffset[i])) + 1;
                                if (len + cchars > maxcandchar) {
                                    if (i > cand_list->dwSelection) {
                                        break;
                                    }

                                    page_start = i;
                                    cchars = len;
                                } else {
                                    cchars += len;
                                }
                            }
                            page_size = i - page_start;
                        } else {
                            page_size = SDL_min(cand_list->dwPageSize == 0 ? MAX_CANDLIST : cand_list->dwPageSize, MAX_CANDLIST);
                            page_start = (cand_list->dwSelection / page_size) * page_size;
                        }
                        for (i = page_start, j = 0; (DWORD)i < cand_list->dwCount && j < page_size; i++, j++) {
                            LPCWSTR candidate = (LPCWSTR)((DWORD_PTR)cand_list + cand_list->dwOffset[i]);
                            IME_AddCandidate(videodata, j, candidate);
                        }

                        has_candidates = true;
                        IME_SendCandidateList(videodata);
                    }
                }
                SDL_free(cand_list);
            }
        }
        ImmReleaseContext(hwnd, himc);
    }

    if (!has_candidates) {
        IME_CloseCandidateList(videodata);
    }
}

bool WIN_HandleIMEMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM *lParam, SDL_VideoData *videodata)
{
    bool trap = false;
    HIMC himc = 0;

    if (msg == WM_IME_SETCONTEXT) {
        SDL_DebugIMELog("WM_IME_SETCONTEXT");

        LPARAM element_mask;
        if (videodata->ime_internal_composition && videodata->ime_internal_candidates) {
            element_mask = 0;
        } else {
            element_mask = ISC_SHOWUIALL;
            if (videodata->ime_internal_composition) {
                element_mask &= ~ISC_SHOWUICOMPOSITIONWINDOW;
            }
            if (videodata->ime_internal_candidates) {
                element_mask &= ~ISC_SHOWUIALLCANDIDATEWINDOW;
            }
        }
        *lParam &= element_mask;

        return false;
    }

    if (!videodata->ime_initialized || !videodata->ime_available || !videodata->ime_enabled) {
        return false;
    }

    switch (msg) {
    case WM_KEYDOWN:
        if (wParam == VK_PROCESSKEY) {
            SDL_DebugIMELog("WM_KEYDOWN VK_PROCESSKEY");
            trap = true;
        } else {
            SDL_DebugIMELog("WM_KEYDOWN normal");
        }
        break;
    case WM_INPUTLANGCHANGE:
        SDL_DebugIMELog("WM_INPUTLANGCHANGE");
        IME_InputLangChanged(videodata);
        break;
    case WM_IME_STARTCOMPOSITION:
        SDL_DebugIMELog("WM_IME_STARTCOMPOSITION");
        if (videodata->ime_internal_composition) {
            trap = true;
        }
        break;
    case WM_IME_COMPOSITION:
        SDL_DebugIMELog("WM_IME_COMPOSITION %x", lParam);
        if (videodata->ime_internal_composition) {
            trap = true;
            himc = ImmGetContext(hwnd);
            if (*lParam & GCS_RESULTSTR) {
                SDL_DebugIMELog("GCS_RESULTSTR");
                IME_GetCompositionString(videodata, himc, GCS_RESULTSTR);
                IME_SendClearComposition(videodata);
                IME_SendInputEvent(videodata);
            }
            if (*lParam & GCS_COMPSTR) {
                SDL_DebugIMELog("GCS_COMPSTR");
                videodata->ime_readingstring[0] = 0;
                IME_GetCompositionString(videodata, himc, GCS_COMPSTR);
                IME_SendEditingEvent(videodata);
            }
            ImmReleaseContext(hwnd, himc);
        }
        break;
    case WM_IME_ENDCOMPOSITION:
        SDL_DebugIMELog("WM_IME_ENDCOMPOSITION");
        if (videodata->ime_internal_composition) {
            trap = true;
            videodata->ime_composition[0] = 0;
            videodata->ime_readingstring[0] = 0;
            videodata->ime_cursor = 0;
            videodata->ime_selected_start = 0;
            videodata->ime_selected_length = 0;
            IME_SendClearComposition(videodata);
        }
        break;
    case WM_IME_NOTIFY:
        SDL_DebugIMELog("WM_IME_NOTIFY %x", wParam);
        switch (wParam) {
        case IMN_SETCOMPOSITIONWINDOW:
            SDL_DebugIMELog("IMN_SETCOMPOSITIONWINDOW");
            break;
        case IMN_SETCOMPOSITIONFONT:
            SDL_DebugIMELog("IMN_SETCOMPOSITIONFONT");
            break;
        case IMN_SETCANDIDATEPOS:
            SDL_DebugIMELog("IMN_SETCANDIDATEPOS");
            break;
        case IMN_SETCONVERSIONMODE:
        case IMN_SETOPENSTATUS:
            SDL_DebugIMELog("%s", wParam == IMN_SETCONVERSIONMODE ? "IMN_SETCONVERSIONMODE" : "IMN_SETOPENSTATUS");
            IME_UpdateInputLocale(videodata);
            break;
        case IMN_OPENCANDIDATE:
        case IMN_CHANGECANDIDATE:
            SDL_DebugIMELog("%s", wParam == IMN_OPENCANDIDATE ? "IMN_OPENCANDIDATE" : "IMN_CHANGECANDIDATE");
            if (videodata->ime_internal_candidates) {
                trap = true;
                videodata->ime_update_candidates = true;
            }
            break;
        case IMN_CLOSECANDIDATE:
            SDL_DebugIMELog("IMN_CLOSECANDIDATE");
            if (videodata->ime_internal_candidates) {
                trap = true;
                videodata->ime_update_candidates = false;
                IME_CloseCandidateList(videodata);
            }
            break;
        case IMN_PRIVATE:
        {
            DWORD dwId = IME_GetId(videodata, 0);
            SDL_DebugIMELog("IMN_PRIVATE %u", dwId);
            IME_GetReadingString(videodata, hwnd);
            switch (dwId) {
            case IMEID_CHT_VER42:
            case IMEID_CHT_VER43:
            case IMEID_CHT_VER44:
            case IMEID_CHS_VER41:
            case IMEID_CHS_VER42:
                if (*lParam == 1 || *lParam == 2) {
                    trap = true;
                }

                break;
            case IMEID_CHT_VER50:
            case IMEID_CHT_VER51:
            case IMEID_CHT_VER52:
            case IMEID_CHT_VER60:
            case IMEID_CHS_VER53:
                if (*lParam == 16 || *lParam == 17 || *lParam == 26 || *lParam == 27 || *lParam == 28) {
                    trap = true;
                }
                break;
            }
        } break;
        default:
            trap = true;
            break;
        }
        break;
    }
    return trap;
}

void WIN_UpdateIMECandidates(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;

    if (videodata->ime_update_candidates) {
        IME_GetCandidateList(videodata, videodata->ime_hwnd_current);
        videodata->ime_update_candidates = false;
    }
}

#endif // SDL_DISABLE_WINDOWS_IME

#endif // SDL_VIDEO_DRIVER_WINDOWS
