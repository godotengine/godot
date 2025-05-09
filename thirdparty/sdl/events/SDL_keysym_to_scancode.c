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

#if defined(SDL_VIDEO_DRIVER_WAYLAND) || defined(SDL_VIDEO_DRIVER_X11)

#include "SDL_keyboard_c.h"
#include "SDL_scancode_tables_c.h"
#include "SDL_keysym_to_scancode_c.h"

/* *INDENT-OFF* */ // clang-format off
static const struct {
    Uint32       keysym;
    SDL_Scancode scancode;
} KeySymToSDLScancode[] = {
    { 0xFF9C, SDL_SCANCODE_KP_1 },  // XK_KP_End
    { 0xFF99, SDL_SCANCODE_KP_2 },  // XK_KP_Down
    { 0xFF9B, SDL_SCANCODE_KP_3 },  // XK_KP_Next
    { 0xFF96, SDL_SCANCODE_KP_4 },  // XK_KP_Left
    { 0xFF9D, SDL_SCANCODE_KP_5 },  // XK_KP_Begin
    { 0xFF98, SDL_SCANCODE_KP_6 },  // XK_KP_Right
    { 0xFF95, SDL_SCANCODE_KP_7 },  // XK_KP_Home
    { 0xFF97, SDL_SCANCODE_KP_8 },  // XK_KP_Up
    { 0xFF9A, SDL_SCANCODE_KP_9 },  // XK_KP_Prior
    { 0xFF9E, SDL_SCANCODE_KP_0 },  // XK_KP_Insert
    { 0xFF9F, SDL_SCANCODE_KP_PERIOD },  // XK_KP_Delete
    { 0xFF62, SDL_SCANCODE_EXECUTE },  // XK_Execute
    { 0xFFEE, SDL_SCANCODE_APPLICATION },  // XK_Hyper_R
    { 0xFE03, SDL_SCANCODE_RALT },  // XK_ISO_Level3_Shift
    { 0xFE20, SDL_SCANCODE_TAB },  // XK_ISO_Left_Tab
    { 0xFFEB, SDL_SCANCODE_LGUI },  // XK_Super_L
    { 0xFFEC, SDL_SCANCODE_RGUI },  // XK_Super_R
    { 0xFF7E, SDL_SCANCODE_MODE },  // XK_Mode_switch
    { 0x1008FF65, SDL_SCANCODE_MENU },  // XF86MenuKB
    { 0x1008FF81, SDL_SCANCODE_F13 },   // XF86Tools
    { 0x1008FF45, SDL_SCANCODE_F14 },   // XF86Launch5
    { 0x1008FF46, SDL_SCANCODE_F15 },   // XF86Launch6
    { 0x1008FF47, SDL_SCANCODE_F16 },   // XF86Launch7
    { 0x1008FF48, SDL_SCANCODE_F17 },   // XF86Launch8
    { 0x1008FF49, SDL_SCANCODE_F18 },   // XF86Launch9
};

// This is a mapping from X keysym to Linux keycode
static const Uint32 LinuxKeycodeKeysyms[] = {
    /*   0, 0x000 */    0x0, // NoSymbol
    /*   1, 0x001 */    0xFF1B, // Escape
    /*   2, 0x002 */    0x31, // 1
    /*   3, 0x003 */    0x32, // 2
    /*   4, 0x004 */    0x33, // 3
    /*   5, 0x005 */    0x34, // 4
    /*   6, 0x006 */    0x35, // 5
    /*   7, 0x007 */    0x36, // 6
    /*   8, 0x008 */    0x37, // 7
    /*   9, 0x009 */    0x38, // 8
    /*  10, 0x00a */    0x39, // 9
    /*  11, 0x00b */    0x30, // 0
    /*  12, 0x00c */    0x2D, // minus
    /*  13, 0x00d */    0x3D, // equal
    /*  14, 0x00e */    0xFF08, // BackSpace
    /*  15, 0x00f */    0xFF09, // Tab
    /*  16, 0x010 */    0x71, // q
    /*  17, 0x011 */    0x77, // w
    /*  18, 0x012 */    0x65, // e
    /*  19, 0x013 */    0x72, // r
    /*  20, 0x014 */    0x74, // t
    /*  21, 0x015 */    0x79, // y
    /*  22, 0x016 */    0x75, // u
    /*  23, 0x017 */    0x69, // i
    /*  24, 0x018 */    0x6F, // o
    /*  25, 0x019 */    0x70, // p
    /*  26, 0x01a */    0x5B, // bracketleft
    /*  27, 0x01b */    0x5D, // bracketright
    /*  28, 0x01c */    0xFF0D, // Return
    /*  29, 0x01d */    0xFFE3, // Control_L
    /*  30, 0x01e */    0x61, // a
    /*  31, 0x01f */    0x73, // s
    /*  32, 0x020 */    0x64, // d
    /*  33, 0x021 */    0x66, // f
    /*  34, 0x022 */    0x67, // g
    /*  35, 0x023 */    0x68, // h
    /*  36, 0x024 */    0x6A, // j
    /*  37, 0x025 */    0x6B, // k
    /*  38, 0x026 */    0x6C, // l
    /*  39, 0x027 */    0x3B, // semicolon
    /*  40, 0x028 */    0x27, // apostrophe
    /*  41, 0x029 */    0x60, // grave
    /*  42, 0x02a */    0xFFE1, // Shift_L
    /*  43, 0x02b */    0x5C, // backslash
    /*  44, 0x02c */    0x7A, // z
    /*  45, 0x02d */    0x78, // x
    /*  46, 0x02e */    0x63, // c
    /*  47, 0x02f */    0x76, // v
    /*  48, 0x030 */    0x62, // b
    /*  49, 0x031 */    0x6E, // n
    /*  50, 0x032 */    0x6D, // m
    /*  51, 0x033 */    0x2C, // comma
    /*  52, 0x034 */    0x2E, // period
    /*  53, 0x035 */    0x2F, // slash
    /*  54, 0x036 */    0xFFE2, // Shift_R
    /*  55, 0x037 */    0xFFAA, // KP_Multiply
    /*  56, 0x038 */    0xFFE9, // Alt_L
    /*  57, 0x039 */    0x20, // space
    /*  58, 0x03a */    0xFFE5, // Caps_Lock
    /*  59, 0x03b */    0xFFBE, // F1
    /*  60, 0x03c */    0xFFBF, // F2
    /*  61, 0x03d */    0xFFC0, // F3
    /*  62, 0x03e */    0xFFC1, // F4
    /*  63, 0x03f */    0xFFC2, // F5
    /*  64, 0x040 */    0xFFC3, // F6
    /*  65, 0x041 */    0xFFC4, // F7
    /*  66, 0x042 */    0xFFC5, // F8
    /*  67, 0x043 */    0xFFC6, // F9
    /*  68, 0x044 */    0xFFC7, // F10
    /*  69, 0x045 */    0xFF7F, // Num_Lock
    /*  70, 0x046 */    0xFF14, // Scroll_Lock
    /*  71, 0x047 */    0xFFB7, // KP_7
    /*  72, 0x048 */    0XFFB8, // KP_8
    /*  73, 0x049 */    0XFFB9, // KP_9
    /*  74, 0x04a */    0xFFAD, // KP_Subtract
    /*  75, 0x04b */    0xFFB4, // KP_4
    /*  76, 0x04c */    0xFFB5, // KP_5
    /*  77, 0x04d */    0xFFB6, // KP_6
    /*  78, 0x04e */    0xFFAB, // KP_Add
    /*  79, 0x04f */    0xFFB1, // KP_1
    /*  80, 0x050 */    0xFFB2, // KP_2
    /*  81, 0x051 */    0xFFB3, // KP_3
    /*  82, 0x052 */    0xFFB0, // KP_0
    /*  83, 0x053 */    0xFFAE, // KP_Decimal
    /*  84, 0x054 */    0x0, // NoSymbol
    /*  85, 0x055 */    0x0, // NoSymbol
    /*  86, 0x056 */    0x3C, // less
    /*  87, 0x057 */    0xFFC8, // F11
    /*  88, 0x058 */    0xFFC9, // F12
    /*  89, 0x059 */    0x0, // NoSymbol
    /*  90, 0x05a */    0xFF26, // Katakana
    /*  91, 0x05b */    0xFF25, // Hiragana
    /*  92, 0x05c */    0xFF23, // Henkan_Mode
    /*  93, 0x05d */    0xFF27, // Hiragana_Katakana
    /*  94, 0x05e */    0xFF22, // Muhenkan
    /*  95, 0x05f */    0x0, // NoSymbol
    /*  96, 0x060 */    0xFF8D, // KP_Enter
    /*  97, 0x061 */    0xFFE4, // Control_R
    /*  98, 0x062 */    0xFFAF, // KP_Divide
    /*  99, 0x063 */    0xFF15, // Sys_Req
    /* 100, 0x064 */    0xFFEA, // Alt_R
    /* 101, 0x065 */    0xFF0A, // Linefeed
    /* 102, 0x066 */    0xFF50, // Home
    /* 103, 0x067 */    0xFF52, // Up
    /* 104, 0x068 */    0xFF55, // Prior
    /* 105, 0x069 */    0xFF51, // Left
    /* 106, 0x06a */    0xFF53, // Right
    /* 107, 0x06b */    0xFF57, // End
    /* 108, 0x06c */    0xFF54, // Down
    /* 109, 0x06d */    0xFF56, // Next
    /* 110, 0x06e */    0xFF63, // Insert
    /* 111, 0x06f */    0xFFFF, // Delete
    /* 112, 0x070 */    0x0, // NoSymbol
    /* 113, 0x071 */    0x1008FF12, // XF86AudioMute
    /* 114, 0x072 */    0x1008FF11, // XF86AudioLowerVolume
    /* 115, 0x073 */    0x1008FF13, // XF86AudioRaiseVolume
    /* 116, 0x074 */    0x1008FF2A, // XF86PowerOff
    /* 117, 0x075 */    0xFFBD, // KP_Equal
    /* 118, 0x076 */    0xB1, // plusminus
    /* 119, 0x077 */    0xFF13, // Pause
    /* 120, 0x078 */    0x1008FF4A, // XF86LaunchA
    /* 121, 0x079 */    0xFFAC, // KP_Separator
    /* 122, 0x07a */    0xFF31, // Hangul
    /* 123, 0x07b */    0xFF34, // Hangul_Hanja
    /* 124, 0x07c */    0x0, // NoSymbol
    /* 125, 0x07d */    0xFFE7, // Meta_L
    /* 126, 0x07e */    0xFFE8, // Meta_R
    /* 127, 0x07f */    0xFF67, // Menu
    /* 128, 0x080 */    0x00, // NoSymbol
    /* 129, 0x081 */    0xFF66, // Redo
    /* 130, 0x082 */    0x1005FF70, // SunProps
    /* 131, 0x083 */    0xFF65, // Undo
    /* 132, 0x084 */    0x1005FF71, // SunFront
    /* 133, 0x085 */    0x1008FF57, // XF86Copy
    /* 134, 0x086 */    0x1008FF6B, // XF86Open
    /* 135, 0x087 */    0x1008FF6D, // XF86Paste
    /* 136, 0x088 */    0xFF68, // Find
    /* 137, 0x089 */    0x1008FF58, // XF86Cut
    /* 138, 0x08a */    0xFF6A, // Help
    /* 139, 0x08b */    0xFF67, // Menu
    /* 140, 0x08c */    0x1008FF1D, // XF86Calculator
    /* 141, 0x08d */    0x0, // NoSymbol
    /* 142, 0x08e */    0x1008FF2F, // XF86Sleep
    /* 143, 0x08f */    0x1008FF2B, // XF86WakeUp
    /* 144, 0x090 */    0x1008FF5D, // XF86Explorer
    /* 145, 0x091 */    0x1008FF7B, // XF86Send
    /* 146, 0x092 */    0x0, // NoSymbol
    /* 147, 0x093 */    0x1008FF8A, // XF86Xfer
    /* 148, 0x094 */    0x1008FF41, // XF86Launch1
    /* 149, 0x095 */    0x1008FF42, // XF86Launch2
    /* 150, 0x096 */    0x1008FF2E, // XF86WWW
    /* 151, 0x097 */    0x1008FF5A, // XF86DOS
    /* 152, 0x098 */    0x1008FF2D, // XF86ScreenSaver
    /* 153, 0x099 */    0x1008FF74, // XF86RotateWindows
    /* 154, 0x09a */    0x1008FF7F, // XF86TaskPane
    /* 155, 0x09b */    0x1008FF19, // XF86Mail
    /* 156, 0x09c */    0x1008FF30, // XF86Favorites
    /* 157, 0x09d */    0x1008FF33, // XF86MyComputer
    /* 158, 0x09e */    0x1008FF26, // XF86Back
    /* 159, 0x09f */    0x1008FF27, // XF86Forward
    /* 160, 0x0a0 */    0x0, // NoSymbol
    /* 161, 0x0a1 */    0x1008FF2C, // XF86Eject
    /* 162, 0x0a2 */    0x1008FF2C, // XF86Eject
    /* 163, 0x0a3 */    0x1008FF17, // XF86AudioNext
    /* 164, 0x0a4 */    0x1008FF14, // XF86AudioPlay
    /* 165, 0x0a5 */    0x1008FF16, // XF86AudioPrev
    /* 166, 0x0a6 */    0x1008FF15, // XF86AudioStop
    /* 167, 0x0a7 */    0x1008FF1C, // XF86AudioRecord
    /* 168, 0x0a8 */    0x1008FF3E, // XF86AudioRewind
    /* 169, 0x0a9 */    0x1008FF6E, // XF86Phone
    /* 170, 0x0aa */    0x0, // NoSymbol
    /* 171, 0x0ab */    0x1008FF81, // XF86Tools
    /* 172, 0x0ac */    0x1008FF18, // XF86HomePage
    /* 173, 0x0ad */    0x1008FF73, // XF86Reload
    /* 174, 0x0ae */    0x1008FF56, // XF86Close
    /* 175, 0x0af */    0x0, // NoSymbol
    /* 176, 0x0b0 */    0x0, // NoSymbol
    /* 177, 0x0b1 */    0x1008FF78, // XF86ScrollUp
    /* 178, 0x0b2 */    0x1008FF79, // XF86ScrollDown
    /* 179, 0x0b3 */    0x0, // NoSymbol
    /* 180, 0x0b4 */    0x0, // NoSymbol
    /* 181, 0x0b5 */    0x1008FF68, // XF86New
    /* 182, 0x0b6 */    0xFF66, // Redo
    /* 183, 0x0b7 */    0xFFCA, // F13
    /* 184, 0x0b8 */    0xFFCB, // F14
    /* 185, 0x0b9 */    0xFFCC, // F15
    /* 186, 0x0ba */    0xFFCD, // F16
    /* 187, 0x0bb */    0xFFCE, // F17
    /* 188, 0x0bc */    0xFFCF, // F18
    /* 189, 0x0bd */    0xFFD0, // F19
    /* 190, 0x0be */    0xFFD1, // F20
    /* 191, 0x0bf */    0xFFD2, // F21
    /* 192, 0x0c0 */    0xFFD3, // F22
    /* 193, 0x0c1 */    0xFFD4, // F23
    /* 194, 0x0c2 */    0xFFD5, // F24
    /* 195, 0x0c3 */    0x0, // NoSymbol
    /* 196, 0x0c4 */    0x0, // NoSymbol
    /* 197, 0x0c5 */    0x0, // NoSymbol
    /* 198, 0x0c6 */    0x0, // NoSymbol
    /* 199, 0x0c7 */    0x0, // NoSymbol
    /* 200, 0x0c8 */    0x1008FF14, // XF86AudioPlay
    /* 201, 0x0c9 */    0x1008FF31, // XF86AudioPause
    /* 202, 0x0ca */    0x1008FF43, // XF86Launch3
    /* 203, 0x0cb */    0x1008FF44, // XF86Launch4
    /* 204, 0x0cc */    0x1008FF4B, // XF86LaunchB
    /* 205, 0x0cd */    0x1008FFA7, // XF86Suspend
    /* 206, 0x0ce */    0x1008FF56, // XF86Close
    /* 207, 0x0cf */    0x1008FF14, // XF86AudioPlay
    /* 208, 0x0d0 */    0x1008FF97, // XF86AudioForward
    /* 209, 0x0d1 */    0x0, // NoSymbol
    /* 210, 0x0d2 */    0xFF61, // Print
    /* 211, 0x0d3 */    0x0, // NoSymbol
    /* 212, 0x0d4 */    0x1008FF8F, // XF86WebCam
    /* 213, 0x0d5 */    0x1008FFB6, // XF86AudioPreset
    /* 214, 0x0d6 */    0x0, // NoSymbol
    /* 215, 0x0d7 */    0x1008FF19, // XF86Mail
    /* 216, 0x0d8 */    0x1008FF8E, // XF86Messenger
    /* 217, 0x0d9 */    0x1008FF1B, // XF86Search
    /* 218, 0x0da */    0x1008FF5F, // XF86Go
    /* 219, 0x0db */    0x1008FF3C, // XF86Finance
    /* 220, 0x0dc */    0x1008FF5E, // XF86Game
    /* 221, 0x0dd */    0x1008FF36, // XF86Shop
    /* 222, 0x0de */    0x0, // NoSymbol
    /* 223, 0x0df */    0xFF69, // Cancel
    /* 224, 0x0e0 */    0x1008FF03, // XF86MonBrightnessDown
    /* 225, 0x0e1 */    0x1008FF02, // XF86MonBrightnessUp
    /* 226, 0x0e2 */    0x1008FF32, // XF86AudioMedia
    /* 227, 0x0e3 */    0x1008FF59, // XF86Display
    /* 228, 0x0e4 */    0x1008FF04, // XF86KbdLightOnOff
    /* 229, 0x0e5 */    0x1008FF06, // XF86KbdBrightnessDown
    /* 230, 0x0e6 */    0x1008FF05, // XF86KbdBrightnessUp
    /* 231, 0x0e7 */    0x1008FF7B, // XF86Send
    /* 232, 0x0e8 */    0x1008FF72, // XF86Reply
    /* 233, 0x0e9 */    0x1008FF90, // XF86MailForward
    /* 234, 0x0ea */    0x1008FF77, // XF86Save
    /* 235, 0x0eb */    0x1008FF5B, // XF86Documents
    /* 236, 0x0ec */    0x1008FF93, // XF86Battery
    /* 237, 0x0ed */    0x1008FF94, // XF86Bluetooth
    /* 238, 0x0ee */    0x1008FF95, // XF86WLAN
    /* 239, 0x0ef */    0x1008FF96, // XF86UWB
    /* 240, 0x0f0 */    0x0, // NoSymbol
    /* 241, 0x0f1 */    0x1008FE22, // XF86Next_VMode
    /* 242, 0x0f2 */    0x1008FE23, // XF86Prev_VMode
    /* 243, 0x0f3 */    0x1008FF07, // XF86MonBrightnessCycle
    /* 244, 0x0f4 */    0x100810F4, // XF86BrightnessAuto
    /* 245, 0x0f5 */    0x100810F5, // XF86DisplayOff
    /* 246, 0x0f6 */    0x1008FFB4, // XF86WWAN
    /* 247, 0x0f7 */    0x1008FFB5, // XF86RFKill
};

#if 0 // Here is a script to generate the ExtendedLinuxKeycodeKeysyms table
#!/bin/bash

function process_line
{
    sym=$(echo "$1" | awk '{print $3}')
    code=$(echo "$1" | sed 's,.*_EVDEVK(\(0x[0-9A-Fa-f]*\)).*,\1,')
    value=$(grep -E "#define ${sym}\s" -R /usr/include/X11 | awk '{print $3}')
    printf "    { 0x%.8X, 0x%.3x },    /* $sym */\n" $value $code
}

grep -F "/* Use: " /usr/include/xkbcommon/xkbcommon-keysyms.h | grep -F _EVDEVK | while read line; do
    process_line "$line"
done
#endif

static const struct {
    Uint32 keysym;
    int linux_keycode;
} ExtendedLinuxKeycodeKeysyms[] = {
    { 0x1008FF2C, 0x0a2 },    // XF86XK_Eject
    { 0x1008FF68, 0x0b5 },    // XF86XK_New
    { 0x0000FF66, 0x0b6 },    // XK_Redo
    { 0x1008FF4B, 0x0cc },    // XF86XK_LaunchB
    { 0x1008FF59, 0x0e3 },    // XF86XK_Display
    { 0x1008FF04, 0x0e4 },    // XF86XK_KbdLightOnOff
    { 0x1008FF06, 0x0e5 },    // XF86XK_KbdBrightnessDown
    { 0x1008FF05, 0x0e6 },    // XF86XK_KbdBrightnessUp
    { 0x1008FF7B, 0x0e7 },    // XF86XK_Send
    { 0x1008FF72, 0x0e8 },    // XF86XK_Reply
    { 0x1008FF90, 0x0e9 },    // XF86XK_MailForward
    { 0x1008FF77, 0x0ea },    // XF86XK_Save
    { 0x1008FF5B, 0x0eb },    // XF86XK_Documents
    { 0x1008FF93, 0x0ec },    // XF86XK_Battery
    { 0x1008FF94, 0x0ed },    // XF86XK_Bluetooth
    { 0x1008FF95, 0x0ee },    // XF86XK_WLAN
    { 0x1008FF96, 0x0ef },    // XF86XK_UWB
    { 0x1008FE22, 0x0f1 },    // XF86XK_Next_VMode
    { 0x1008FE23, 0x0f2 },    // XF86XK_Prev_VMode
    { 0x1008FF07, 0x0f3 },    // XF86XK_MonBrightnessCycle
    { 0x1008FFB4, 0x0f6 },    // XF86XK_WWAN
    { 0x1008FFB5, 0x0f7 },    // XF86XK_RFKill
    { 0x1008FFB2, 0x0f8 },    // XF86XK_AudioMicMute
    { 0x1008FF9C, 0x173 },    // XF86XK_CycleAngle
    { 0x1008FFB8, 0x174 },    // XF86XK_FullScreen
    { 0x1008FF87, 0x189 },    // XF86XK_Video
    { 0x1008FF20, 0x18d },    // XF86XK_Calendar
    { 0x1008FF99, 0x19a },    // XF86XK_AudioRandomPlay
    { 0x1008FF5E, 0x1a1 },    // XF86XK_Game
    { 0x1008FF8B, 0x1a2 },    // XF86XK_ZoomIn
    { 0x1008FF8C, 0x1a3 },    // XF86XK_ZoomOut
    { 0x1008FF89, 0x1a5 },    // XF86XK_Word
    { 0x1008FF5C, 0x1a7 },    // XF86XK_Excel
    { 0x1008FF69, 0x1ab },    // XF86XK_News
    { 0x1008FF8E, 0x1ae },    // XF86XK_Messenger
    { 0x1008FF61, 0x1b1 },    // XF86XK_LogOff
    { 0x00000024, 0x1b2 },    // XK_dollar
    { 0x000020AC, 0x1b3 },    // XK_EuroSign
    { 0x1008FF9D, 0x1b4 },    // XF86XK_FrameBack
    { 0x1008FF9E, 0x1b5 },    // XF86XK_FrameForward
    { 0x0000FFF1, 0x1f1 },    // XK_braille_dot_1
    { 0x0000FFF2, 0x1f2 },    // XK_braille_dot_2
    { 0x0000FFF3, 0x1f3 },    // XK_braille_dot_3
    { 0x0000FFF4, 0x1f4 },    // XK_braille_dot_4
    { 0x0000FFF5, 0x1f5 },    // XK_braille_dot_5
    { 0x0000FFF6, 0x1f6 },    // XK_braille_dot_6
    { 0x0000FFF7, 0x1f7 },    // XK_braille_dot_7
    { 0x0000FFF8, 0x1f8 },    // XK_braille_dot_8
    { 0x0000FFF9, 0x1f9 },    // XK_braille_dot_9
    { 0x0000FFF1, 0x1fa },    // XK_braille_dot_1
    { 0x1008FFA9, 0x212 },    // XF86XK_TouchpadToggle
    { 0x1008FFB0, 0x213 },    // XF86XK_TouchpadOn
    { 0x1008FFB1, 0x214 },    // XF86XK_TouchpadOff
    { 0x1008FFB7, 0x231 },    // XF86XK_RotationLockToggle
    { 0x0000FE08, 0x248 },    // XK_ISO_Next_Group
};
/* *INDENT-ON* */ // clang-format on

SDL_Scancode SDL_GetScancodeFromKeySym(Uint32 keysym, Uint32 keycode)
{
    int i;
    Uint32 linux_keycode = 0;

    // First check our custom list
    for (i = 0; i < SDL_arraysize(KeySymToSDLScancode); ++i) {
        if (keysym == KeySymToSDLScancode[i].keysym) {
            return KeySymToSDLScancode[i].scancode;
        }
    }

    if (keysym >= 0x41 && keysym <= 0x5a) {
        // Normalize alphabetic keysyms to the lowercase form
        keysym += 0x20;
    } else if (keysym >= 0x10081000 && keysym <= 0x10081FFF) {
        /* The rest of the keysyms map to Linux keycodes, so use that mapping
         * Per xkbcommon-keysyms.h, this is actually a linux keycode.
         */
        linux_keycode = (keysym - 0x10081000);
    }
    if (!linux_keycode) {
        // See if this keysym is an exact match in our table
        i = (keycode - 8);
        if (i >= 0 && i < SDL_arraysize(LinuxKeycodeKeysyms) && keysym == LinuxKeycodeKeysyms[i]) {
            linux_keycode = i;
        } else {
            // Scan the table for this keysym
            for (i = 0; i < SDL_arraysize(LinuxKeycodeKeysyms); ++i) {
                if (keysym == LinuxKeycodeKeysyms[i]) {
                    linux_keycode = i;
                    break;
                }
            }
        }
    }
    if (!linux_keycode) {
        // Scan the extended table for this keysym
        for (i = 0; i < SDL_arraysize(ExtendedLinuxKeycodeKeysyms); ++i) {
            if (keysym == ExtendedLinuxKeycodeKeysyms[i].keysym) {
                linux_keycode = ExtendedLinuxKeycodeKeysyms[i].linux_keycode;
                break;
            }
        }
    }
    return SDL_GetScancodeFromTable(SDL_SCANCODE_TABLE_LINUX, linux_keycode);
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
