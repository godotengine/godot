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

#ifndef scancodes_xfree86_h_
#define scancodes_xfree86_h_

/* XFree86 key code to SDL scancode mapping table
   Sources:
   - atKeyNames.h from XFree86 source code
*/
/* *INDENT-OFF* */ // clang-format off
static const SDL_Scancode xfree86_scancode_table[] = {
    /*  0 */    SDL_SCANCODE_UNKNOWN,
    /*  1 */    SDL_SCANCODE_ESCAPE,
    /*  2 */    SDL_SCANCODE_1,
    /*  3 */    SDL_SCANCODE_2,
    /*  4 */    SDL_SCANCODE_3,
    /*  5 */    SDL_SCANCODE_4,
    /*  6 */    SDL_SCANCODE_5,
    /*  7 */    SDL_SCANCODE_6,
    /*  8 */    SDL_SCANCODE_7,
    /*  9 */    SDL_SCANCODE_8,
    /*  10 */   SDL_SCANCODE_9,
    /*  11 */   SDL_SCANCODE_0,
    /*  12 */   SDL_SCANCODE_MINUS,
    /*  13 */   SDL_SCANCODE_EQUALS,
    /*  14 */   SDL_SCANCODE_BACKSPACE,
    /*  15 */   SDL_SCANCODE_TAB,
    /*  16 */   SDL_SCANCODE_Q,
    /*  17 */   SDL_SCANCODE_W,
    /*  18 */   SDL_SCANCODE_E,
    /*  19 */   SDL_SCANCODE_R,
    /*  20 */   SDL_SCANCODE_T,
    /*  21 */   SDL_SCANCODE_Y,
    /*  22 */   SDL_SCANCODE_U,
    /*  23 */   SDL_SCANCODE_I,
    /*  24 */   SDL_SCANCODE_O,
    /*  25 */   SDL_SCANCODE_P,
    /*  26 */   SDL_SCANCODE_LEFTBRACKET,
    /*  27 */   SDL_SCANCODE_RIGHTBRACKET,
    /*  28 */   SDL_SCANCODE_RETURN,
    /*  29 */   SDL_SCANCODE_LCTRL,
    /*  30 */   SDL_SCANCODE_A,
    /*  31 */   SDL_SCANCODE_S,
    /*  32 */   SDL_SCANCODE_D,
    /*  33 */   SDL_SCANCODE_F,
    /*  34 */   SDL_SCANCODE_G,
    /*  35 */   SDL_SCANCODE_H,
    /*  36 */   SDL_SCANCODE_J,
    /*  37 */   SDL_SCANCODE_K,
    /*  38 */   SDL_SCANCODE_L,
    /*  39 */   SDL_SCANCODE_SEMICOLON,
    /*  40 */   SDL_SCANCODE_APOSTROPHE,
    /*  41 */   SDL_SCANCODE_GRAVE,
    /*  42 */   SDL_SCANCODE_LSHIFT,
    /*  43 */   SDL_SCANCODE_BACKSLASH,
    /*  44 */   SDL_SCANCODE_Z,
    /*  45 */   SDL_SCANCODE_X,
    /*  46 */   SDL_SCANCODE_C,
    /*  47 */   SDL_SCANCODE_V,
    /*  48 */   SDL_SCANCODE_B,
    /*  49 */   SDL_SCANCODE_N,
    /*  50 */   SDL_SCANCODE_M,
    /*  51 */   SDL_SCANCODE_COMMA,
    /*  52 */   SDL_SCANCODE_PERIOD,
    /*  53 */   SDL_SCANCODE_SLASH,
    /*  54 */   SDL_SCANCODE_RSHIFT,
    /*  55 */   SDL_SCANCODE_KP_MULTIPLY,
    /*  56 */   SDL_SCANCODE_LALT,
    /*  57 */   SDL_SCANCODE_SPACE,
    /*  58 */   SDL_SCANCODE_CAPSLOCK,
    /*  59 */   SDL_SCANCODE_F1,
    /*  60 */   SDL_SCANCODE_F2,
    /*  61 */   SDL_SCANCODE_F3,
    /*  62 */   SDL_SCANCODE_F4,
    /*  63 */   SDL_SCANCODE_F5,
    /*  64 */   SDL_SCANCODE_F6,
    /*  65 */   SDL_SCANCODE_F7,
    /*  66 */   SDL_SCANCODE_F8,
    /*  67 */   SDL_SCANCODE_F9,
    /*  68 */   SDL_SCANCODE_F10,
    /*  69 */   SDL_SCANCODE_NUMLOCKCLEAR,
    /*  70 */   SDL_SCANCODE_SCROLLLOCK,
    /*  71 */   SDL_SCANCODE_KP_7,
    /*  72 */   SDL_SCANCODE_KP_8,
    /*  73 */   SDL_SCANCODE_KP_9,
    /*  74 */   SDL_SCANCODE_KP_MINUS,
    /*  75 */   SDL_SCANCODE_KP_4,
    /*  76 */   SDL_SCANCODE_KP_5,
    /*  77 */   SDL_SCANCODE_KP_6,
    /*  78 */   SDL_SCANCODE_KP_PLUS,
    /*  79 */   SDL_SCANCODE_KP_1,
    /*  80 */   SDL_SCANCODE_KP_2,
    /*  81 */   SDL_SCANCODE_KP_3,
    /*  82 */   SDL_SCANCODE_KP_0,
    /*  83 */   SDL_SCANCODE_KP_PERIOD,
    /*  84 */   SDL_SCANCODE_SYSREQ,
    /*  85 */   SDL_SCANCODE_MODE,
    /*  86 */   SDL_SCANCODE_NONUSBACKSLASH,
    /*  87 */   SDL_SCANCODE_F11,
    /*  88 */   SDL_SCANCODE_F12,
    /*  89 */   SDL_SCANCODE_HOME,
    /*  90 */   SDL_SCANCODE_UP,
    /*  91 */   SDL_SCANCODE_PAGEUP,
    /*  92 */   SDL_SCANCODE_LEFT,
    /*  93 */   SDL_SCANCODE_UNKNOWN, // on PowerBook G4 / KEY_Begin
    /*  94 */   SDL_SCANCODE_RIGHT,
    /*  95 */   SDL_SCANCODE_END,
    /*  96 */   SDL_SCANCODE_DOWN,
    /*  97 */   SDL_SCANCODE_PAGEDOWN,
    /*  98 */   SDL_SCANCODE_INSERT,
    /*  99 */   SDL_SCANCODE_DELETE,
    /*  100 */  SDL_SCANCODE_KP_ENTER,
    /*  101 */  SDL_SCANCODE_RCTRL,
    /*  102 */  SDL_SCANCODE_PAUSE,
    /*  103 */  SDL_SCANCODE_PRINTSCREEN,
    /*  104 */  SDL_SCANCODE_KP_DIVIDE,
    /*  105 */  SDL_SCANCODE_RALT,
    /*  106 */  SDL_SCANCODE_UNKNOWN, // BREAK
    /*  107 */  SDL_SCANCODE_LGUI,
    /*  108 */  SDL_SCANCODE_RGUI,
    /*  109 */  SDL_SCANCODE_APPLICATION,
    /*  110 */  SDL_SCANCODE_F13,
    /*  111 */  SDL_SCANCODE_F14,
    /*  112 */  SDL_SCANCODE_F15,
    /*  113 */  SDL_SCANCODE_F16,
    /*  114 */  SDL_SCANCODE_F17,
    /*  115 */  SDL_SCANCODE_INTERNATIONAL1, // \_
    /*  116 */  SDL_SCANCODE_UNKNOWN, /* is translated to XK_ISO_Level3_Shift by my X server, but I have no keyboard that generates this code, so I don't know what the correct SDL_SCANCODE_* for it is */
    /*  117 */  SDL_SCANCODE_UNKNOWN,
    /*  118 */  SDL_SCANCODE_KP_EQUALS,
    /*  119 */  SDL_SCANCODE_UNKNOWN,
    /*  120 */  SDL_SCANCODE_UNKNOWN,
    /*  121 */  SDL_SCANCODE_INTERNATIONAL4, // Henkan_Mode
    /*  122 */  SDL_SCANCODE_UNKNOWN,
    /*  123 */  SDL_SCANCODE_INTERNATIONAL5, // Muhenkan
    /*  124 */  SDL_SCANCODE_UNKNOWN,
    /*  125 */  SDL_SCANCODE_INTERNATIONAL3, // Yen
    /*  126 */  SDL_SCANCODE_UNKNOWN,
    /*  127 */  SDL_SCANCODE_UNKNOWN,
    /*  128 */  SDL_SCANCODE_UNKNOWN,
    /*  129 */  SDL_SCANCODE_UNKNOWN,
    /*  130 */  SDL_SCANCODE_UNKNOWN,
    /*  131 */  SDL_SCANCODE_UNKNOWN,
    /*  132 */  SDL_SCANCODE_POWER,
    /*  133 */  SDL_SCANCODE_MUTE,
    /*  134 */  SDL_SCANCODE_VOLUMEDOWN,
    /*  135 */  SDL_SCANCODE_VOLUMEUP,
    /*  136 */  SDL_SCANCODE_HELP,
    /*  137 */  SDL_SCANCODE_STOP,
    /*  138 */  SDL_SCANCODE_AGAIN,
    /*  139 */  SDL_SCANCODE_UNKNOWN, // PROPS
    /*  140 */  SDL_SCANCODE_UNDO,
    /*  141 */  SDL_SCANCODE_UNKNOWN, // FRONT
    /*  142 */  SDL_SCANCODE_COPY,
    /*  143 */  SDL_SCANCODE_UNKNOWN, // OPEN
    /*  144 */  SDL_SCANCODE_PASTE,
    /*  145 */  SDL_SCANCODE_FIND,
    /*  146 */  SDL_SCANCODE_CUT,
};

// This is largely identical to the Linux keycode mapping
static const SDL_Scancode xfree86_scancode_table2[] = {
    /*   0, 0x000 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /*   1, 0x001 */   SDL_SCANCODE_ESCAPE,             // Escape
    /*   2, 0x002 */   SDL_SCANCODE_1,                  // 1
    /*   3, 0x003 */   SDL_SCANCODE_2,                  // 2
    /*   4, 0x004 */   SDL_SCANCODE_3,                  // 3
    /*   5, 0x005 */   SDL_SCANCODE_4,                  // 4
    /*   6, 0x006 */   SDL_SCANCODE_5,                  // 5
    /*   7, 0x007 */   SDL_SCANCODE_6,                  // 6
    /*   8, 0x008 */   SDL_SCANCODE_7,                  // 7
    /*   9, 0x009 */   SDL_SCANCODE_8,                  // 8
    /*  10, 0x00a */   SDL_SCANCODE_9,                  // 9
    /*  11, 0x00b */   SDL_SCANCODE_0,                  // 0
    /*  12, 0x00c */   SDL_SCANCODE_MINUS,              // minus
    /*  13, 0x00d */   SDL_SCANCODE_EQUALS,             // equal
    /*  14, 0x00e */   SDL_SCANCODE_BACKSPACE,          // BackSpace
    /*  15, 0x00f */   SDL_SCANCODE_TAB,                // Tab
    /*  16, 0x010 */   SDL_SCANCODE_Q,                  // q
    /*  17, 0x011 */   SDL_SCANCODE_W,                  // w
    /*  18, 0x012 */   SDL_SCANCODE_E,                  // e
    /*  19, 0x013 */   SDL_SCANCODE_R,                  // r
    /*  20, 0x014 */   SDL_SCANCODE_T,                  // t
    /*  21, 0x015 */   SDL_SCANCODE_Y,                  // y
    /*  22, 0x016 */   SDL_SCANCODE_U,                  // u
    /*  23, 0x017 */   SDL_SCANCODE_I,                  // i
    /*  24, 0x018 */   SDL_SCANCODE_O,                  // o
    /*  25, 0x019 */   SDL_SCANCODE_P,                  // p
    /*  26, 0x01a */   SDL_SCANCODE_LEFTBRACKET,        // bracketleft
    /*  27, 0x01b */   SDL_SCANCODE_RIGHTBRACKET,       // bracketright
    /*  28, 0x01c */   SDL_SCANCODE_RETURN,             // Return
    /*  29, 0x01d */   SDL_SCANCODE_LCTRL,              // Control_L
    /*  30, 0x01e */   SDL_SCANCODE_A,                  // a
    /*  31, 0x01f */   SDL_SCANCODE_S,                  // s
    /*  32, 0x020 */   SDL_SCANCODE_D,                  // d
    /*  33, 0x021 */   SDL_SCANCODE_F,                  // f
    /*  34, 0x022 */   SDL_SCANCODE_G,                  // g
    /*  35, 0x023 */   SDL_SCANCODE_H,                  // h
    /*  36, 0x024 */   SDL_SCANCODE_J,                  // j
    /*  37, 0x025 */   SDL_SCANCODE_K,                  // k
    /*  38, 0x026 */   SDL_SCANCODE_L,                  // l
    /*  39, 0x027 */   SDL_SCANCODE_SEMICOLON,          // semicolon
    /*  40, 0x028 */   SDL_SCANCODE_APOSTROPHE,         // apostrophe
    /*  41, 0x029 */   SDL_SCANCODE_GRAVE,              // grave
    /*  42, 0x02a */   SDL_SCANCODE_LSHIFT,             // Shift_L
    /*  43, 0x02b */   SDL_SCANCODE_BACKSLASH,          // backslash
    /*  44, 0x02c */   SDL_SCANCODE_Z,                  // z
    /*  45, 0x02d */   SDL_SCANCODE_X,                  // x
    /*  46, 0x02e */   SDL_SCANCODE_C,                  // c
    /*  47, 0x02f */   SDL_SCANCODE_V,                  // v
    /*  48, 0x030 */   SDL_SCANCODE_B,                  // b
    /*  49, 0x031 */   SDL_SCANCODE_N,                  // n
    /*  50, 0x032 */   SDL_SCANCODE_M,                  // m
    /*  51, 0x033 */   SDL_SCANCODE_COMMA,              // comma
    /*  52, 0x034 */   SDL_SCANCODE_PERIOD,             // period
    /*  53, 0x035 */   SDL_SCANCODE_SLASH,              // slash
    /*  54, 0x036 */   SDL_SCANCODE_RSHIFT,             // Shift_R
    /*  55, 0x037 */   SDL_SCANCODE_KP_MULTIPLY,        // KP_Multiply
    /*  56, 0x038 */   SDL_SCANCODE_LALT,               // Alt_L
    /*  57, 0x039 */   SDL_SCANCODE_SPACE,              // space
    /*  58, 0x03a */   SDL_SCANCODE_CAPSLOCK,           // Caps_Lock
    /*  59, 0x03b */   SDL_SCANCODE_F1,                 // F1
    /*  60, 0x03c */   SDL_SCANCODE_F2,                 // F2
    /*  61, 0x03d */   SDL_SCANCODE_F3,                 // F3
    /*  62, 0x03e */   SDL_SCANCODE_F4,                 // F4
    /*  63, 0x03f */   SDL_SCANCODE_F5,                 // F5
    /*  64, 0x040 */   SDL_SCANCODE_F6,                 // F6
    /*  65, 0x041 */   SDL_SCANCODE_F7,                 // F7
    /*  66, 0x042 */   SDL_SCANCODE_F8,                 // F8
    /*  67, 0x043 */   SDL_SCANCODE_F9,                 // F9
    /*  68, 0x044 */   SDL_SCANCODE_F10,                // F10
    /*  69, 0x045 */   SDL_SCANCODE_NUMLOCKCLEAR,       // Num_Lock
    /*  70, 0x046 */   SDL_SCANCODE_SCROLLLOCK,         // Scroll_Lock
    /*  71, 0x047 */   SDL_SCANCODE_KP_7,               // KP_Home
    /*  72, 0x048 */   SDL_SCANCODE_KP_8,               // KP_Up
    /*  73, 0x049 */   SDL_SCANCODE_KP_9,               // KP_Prior
    /*  74, 0x04a */   SDL_SCANCODE_KP_MINUS,           // KP_Subtract
    /*  75, 0x04b */   SDL_SCANCODE_KP_4,               // KP_Left
    /*  76, 0x04c */   SDL_SCANCODE_KP_5,               // KP_Begin
    /*  77, 0x04d */   SDL_SCANCODE_KP_6,               // KP_Right
    /*  78, 0x04e */   SDL_SCANCODE_KP_PLUS,            // KP_Add
    /*  79, 0x04f */   SDL_SCANCODE_KP_1,               // KP_End
    /*  80, 0x050 */   SDL_SCANCODE_KP_2,               // KP_Down
    /*  81, 0x051 */   SDL_SCANCODE_KP_3,               // KP_Next
    /*  82, 0x052 */   SDL_SCANCODE_KP_0,               // KP_Insert
    /*  83, 0x053 */   SDL_SCANCODE_KP_PERIOD,          // KP_Delete
    /*  84, 0x054 */   SDL_SCANCODE_RALT,               // ISO_Level3_Shift
    /*  85, 0x055 */   SDL_SCANCODE_MODE,               // ????
    /*  86, 0x056 */   SDL_SCANCODE_NONUSBACKSLASH,     // less
    /*  87, 0x057 */   SDL_SCANCODE_F11,                // F11
    /*  88, 0x058 */   SDL_SCANCODE_F12,                // F12
    /*  89, 0x059 */   SDL_SCANCODE_INTERNATIONAL1,     // \_
    /*  90, 0x05a */   SDL_SCANCODE_LANG3,              // Katakana
    /*  91, 0x05b */   SDL_SCANCODE_LANG4,              // Hiragana
    /*  92, 0x05c */   SDL_SCANCODE_INTERNATIONAL4,     // Henkan_Mode
    /*  93, 0x05d */   SDL_SCANCODE_INTERNATIONAL2,     // Hiragana_Katakana
    /*  94, 0x05e */   SDL_SCANCODE_INTERNATIONAL5,     // Muhenkan
    /*  95, 0x05f */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /*  96, 0x060 */   SDL_SCANCODE_KP_ENTER,           // KP_Enter
    /*  97, 0x061 */   SDL_SCANCODE_RCTRL,              // Control_R
    /*  98, 0x062 */   SDL_SCANCODE_KP_DIVIDE,          // KP_Divide
    /*  99, 0x063 */   SDL_SCANCODE_PRINTSCREEN,        // Print
    /* 100, 0x064 */   SDL_SCANCODE_RALT,               // ISO_Level3_Shift, ALTGR, RALT
    /* 101, 0x065 */   SDL_SCANCODE_UNKNOWN,            // Linefeed
    /* 102, 0x066 */   SDL_SCANCODE_HOME,               // Home
    /* 103, 0x067 */   SDL_SCANCODE_UP,                 // Up
    /* 104, 0x068 */   SDL_SCANCODE_PAGEUP,             // Prior
    /* 105, 0x069 */   SDL_SCANCODE_LEFT,               // Left
    /* 106, 0x06a */   SDL_SCANCODE_RIGHT,              // Right
    /* 107, 0x06b */   SDL_SCANCODE_END,                // End
    /* 108, 0x06c */   SDL_SCANCODE_DOWN,               // Down
    /* 109, 0x06d */   SDL_SCANCODE_PAGEDOWN,           // Next
    /* 110, 0x06e */   SDL_SCANCODE_INSERT,             // Insert
    /* 111, 0x06f */   SDL_SCANCODE_DELETE,             // Delete
    /* 112, 0x070 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 113, 0x071 */   SDL_SCANCODE_MUTE,               // XF86AudioMute
    /* 114, 0x072 */   SDL_SCANCODE_VOLUMEDOWN,         // XF86AudioLowerVolume
    /* 115, 0x073 */   SDL_SCANCODE_VOLUMEUP,           // XF86AudioRaiseVolume
    /* 116, 0x074 */   SDL_SCANCODE_POWER,              // XF86PowerOff
    /* 117, 0x075 */   SDL_SCANCODE_KP_EQUALS,          // KP_Equal
    /* 118, 0x076 */   SDL_SCANCODE_KP_PLUSMINUS,       // plusminus
    /* 119, 0x077 */   SDL_SCANCODE_PAUSE,              // Pause
    /* 120, 0x078 */   SDL_SCANCODE_UNKNOWN,            // XF86LaunchA
    /* 121, 0x079 */   SDL_SCANCODE_KP_PERIOD,          // KP_Decimal
    /* 122, 0x07a */   SDL_SCANCODE_LANG1,              // Hangul
    /* 123, 0x07b */   SDL_SCANCODE_LANG2,              // Hangul_Hanja
    /* 124, 0x07c */   SDL_SCANCODE_INTERNATIONAL3,     // Yen
    /* 125, 0x07d */   SDL_SCANCODE_LGUI,               // Super_L
    /* 126, 0x07e */   SDL_SCANCODE_RGUI,               // Super_R
    /* 127, 0x07f */   SDL_SCANCODE_APPLICATION,        // Menu
    /* 128, 0x080 */   SDL_SCANCODE_CANCEL,             // Cancel
    /* 129, 0x081 */   SDL_SCANCODE_AGAIN,              // Redo
    /* 130, 0x082 */   SDL_SCANCODE_UNKNOWN,            // SunProps
    /* 131, 0x083 */   SDL_SCANCODE_UNDO,               // Undo
    /* 132, 0x084 */   SDL_SCANCODE_UNKNOWN,            // SunFront
    /* 133, 0x085 */   SDL_SCANCODE_COPY,               // XF86Copy
    /* 134, 0x086 */   SDL_SCANCODE_UNKNOWN,            // SunOpen, XF86Open
    /* 135, 0x087 */   SDL_SCANCODE_PASTE,              // XF86Paste
    /* 136, 0x088 */   SDL_SCANCODE_FIND,               // Find
    /* 137, 0x089 */   SDL_SCANCODE_CUT,                // XF86Cut
    /* 138, 0x08a */   SDL_SCANCODE_HELP,               // Help
    /* 139, 0x08b */   SDL_SCANCODE_MENU,               // XF86MenuKB
    /* 140, 0x08c */   SDL_SCANCODE_UNKNOWN,            // XF86Calculator
    /* 141, 0x08d */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 142, 0x08e */   SDL_SCANCODE_SLEEP,              // XF86Sleep
    /* 143, 0x08f */   SDL_SCANCODE_UNKNOWN,            // XF86WakeUp
    /* 144, 0x090 */   SDL_SCANCODE_UNKNOWN,            // XF86Explorer
    /* 145, 0x091 */   SDL_SCANCODE_UNKNOWN,            // XF86Send
    /* 146, 0x092 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 147, 0x093 */   SDL_SCANCODE_UNKNOWN,            // XF86Xfer
    /* 148, 0x094 */   SDL_SCANCODE_UNKNOWN,            // XF86Launch1
    /* 149, 0x095 */   SDL_SCANCODE_UNKNOWN,            // XF86Launch2
    /* 150, 0x096 */   SDL_SCANCODE_UNKNOWN,            // XF86WWW
    /* 151, 0x097 */   SDL_SCANCODE_UNKNOWN,            // XF86DOS
    /* 152, 0x098 */   SDL_SCANCODE_UNKNOWN,            // XF86ScreenSaver
    /* 153, 0x099 */   SDL_SCANCODE_UNKNOWN,            // XF86RotateWindows
    /* 154, 0x09a */   SDL_SCANCODE_UNKNOWN,            // XF86TaskPane
    /* 155, 0x09b */   SDL_SCANCODE_UNKNOWN,            // XF86Mail
    /* 156, 0x09c */   SDL_SCANCODE_AC_BOOKMARKS,       // XF86Favorites
    /* 157, 0x09d */   SDL_SCANCODE_UNKNOWN,            // XF86MyComputer
    /* 158, 0x09e */   SDL_SCANCODE_AC_BACK,            // XF86Back
    /* 159, 0x09f */   SDL_SCANCODE_AC_FORWARD,         // XF86Forward
    /* 160, 0x0a0 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 161, 0x0a1 */   SDL_SCANCODE_MEDIA_EJECT,        // XF86Eject
    /* 162, 0x0a2 */   SDL_SCANCODE_MEDIA_EJECT,        // XF86Eject
    /* 163, 0x0a3 */   SDL_SCANCODE_MEDIA_NEXT_TRACK,   // XF86AudioNext
    /* 164, 0x0a4 */   SDL_SCANCODE_MEDIA_PLAY_PAUSE,   // XF86AudioPlay
    /* 165, 0x0a5 */   SDL_SCANCODE_MEDIA_PREVIOUS_TRACK, // XF86AudioPrev
    /* 166, 0x0a6 */   SDL_SCANCODE_MEDIA_STOP,         // XF86AudioStop
    /* 167, 0x0a7 */   SDL_SCANCODE_MEDIA_RECORD,       // XF86AudioRecord
    /* 168, 0x0a8 */   SDL_SCANCODE_MEDIA_REWIND,       // XF86AudioRewind
    /* 169, 0x0a9 */   SDL_SCANCODE_UNKNOWN,            // XF86Phone
    /* 170, 0x0aa */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 171, 0x0ab */   SDL_SCANCODE_F13,                // XF86Tools
    /* 172, 0x0ac */   SDL_SCANCODE_AC_HOME,            // XF86HomePage
    /* 173, 0x0ad */   SDL_SCANCODE_AC_REFRESH,         // XF86Reload
    /* 174, 0x0ae */   SDL_SCANCODE_UNKNOWN,            // XF86Close
    /* 175, 0x0af */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 176, 0x0b0 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 177, 0x0b1 */   SDL_SCANCODE_UNKNOWN,            // XF86ScrollUp
    /* 178, 0x0b2 */   SDL_SCANCODE_UNKNOWN,            // XF86ScrollDown
    /* 179, 0x0b3 */   SDL_SCANCODE_KP_LEFTPAREN,       // parenleft
    /* 180, 0x0b4 */   SDL_SCANCODE_KP_RIGHTPAREN,      // parenright
    /* 181, 0x0b5 */   SDL_SCANCODE_AC_NEW,             // XF86New
    /* 182, 0x0b6 */   SDL_SCANCODE_AGAIN,              // Redo
    /* 183, 0x0b7 */   SDL_SCANCODE_F13,                // XF86Tools
    /* 184, 0x0b8 */   SDL_SCANCODE_F14,                // XF86Launch5
    /* 185, 0x0b9 */   SDL_SCANCODE_F15,                // XF86Launch6
    /* 186, 0x0ba */   SDL_SCANCODE_F16,                // XF86Launch7
    /* 187, 0x0bb */   SDL_SCANCODE_F17,                // XF86Launch8
    /* 188, 0x0bc */   SDL_SCANCODE_F18,                // XF86Launch9
    /* 189, 0x0bd */   SDL_SCANCODE_F19,                // NoSymbol
    /* 190, 0x0be */   SDL_SCANCODE_F20,                // XF86AudioMicMute
    /* 191, 0x0bf */   SDL_SCANCODE_UNKNOWN,            // XF86TouchpadToggle
    /* 192, 0x0c0 */   SDL_SCANCODE_UNKNOWN,            // XF86TouchpadOn
    /* 193, 0x0c1 */   SDL_SCANCODE_UNKNOWN,            // XF86TouchpadOff
    /* 194, 0x0c2 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 195, 0x0c3 */   SDL_SCANCODE_MODE,               // Mode_switch
    /* 196, 0x0c4 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 197, 0x0c5 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 198, 0x0c6 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 199, 0x0c7 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 200, 0x0c8 */   SDL_SCANCODE_MEDIA_PLAY,         // XF86AudioPlay
    /* 201, 0x0c9 */   SDL_SCANCODE_MEDIA_PAUSE,        // XF86AudioPause
    /* 202, 0x0ca */   SDL_SCANCODE_UNKNOWN,            // XF86Launch3
    /* 203, 0x0cb */   SDL_SCANCODE_UNKNOWN,            // XF86Launch4
    /* 204, 0x0cc */   SDL_SCANCODE_UNKNOWN,            // XF86LaunchB
    /* 205, 0x0cd */   SDL_SCANCODE_UNKNOWN,            // XF86Suspend
    /* 206, 0x0ce */   SDL_SCANCODE_AC_CLOSE,           // XF86Close
    /* 207, 0x0cf */   SDL_SCANCODE_MEDIA_PLAY,         // XF86AudioPlay
    /* 208, 0x0d0 */   SDL_SCANCODE_MEDIA_FAST_FORWARD, // XF86AudioForward
    /* 209, 0x0d1 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 210, 0x0d2 */   SDL_SCANCODE_PRINTSCREEN,        // Print
    /* 211, 0x0d3 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 212, 0x0d4 */   SDL_SCANCODE_UNKNOWN,            // XF86WebCam
    /* 213, 0x0d5 */   SDL_SCANCODE_UNKNOWN,            // XF86AudioPreset
    /* 214, 0x0d6 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 215, 0x0d7 */   SDL_SCANCODE_UNKNOWN,            // XF86Mail
    /* 216, 0x0d8 */   SDL_SCANCODE_UNKNOWN,            // XF86Messenger
    /* 217, 0x0d9 */   SDL_SCANCODE_AC_SEARCH,          // XF86Search
    /* 218, 0x0da */   SDL_SCANCODE_UNKNOWN,            // XF86Go
    /* 219, 0x0db */   SDL_SCANCODE_UNKNOWN,            // XF86Finance
    /* 220, 0x0dc */   SDL_SCANCODE_UNKNOWN,            // XF86Game
    /* 221, 0x0dd */   SDL_SCANCODE_UNKNOWN,            // XF86Shop
    /* 222, 0x0de */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 223, 0x0df */   SDL_SCANCODE_CANCEL,             // Cancel
    /* 224, 0x0e0 */   SDL_SCANCODE_UNKNOWN,            // XF86MonBrightnessDown
    /* 225, 0x0e1 */   SDL_SCANCODE_UNKNOWN,            // XF86MonBrightnessUp
    /* 226, 0x0e2 */   SDL_SCANCODE_MEDIA_SELECT,       // XF86AudioMedia
    /* 227, 0x0e3 */   SDL_SCANCODE_UNKNOWN,            // XF86Display
    /* 228, 0x0e4 */   SDL_SCANCODE_UNKNOWN,            // XF86KbdLightOnOff
    /* 229, 0x0e5 */   SDL_SCANCODE_UNKNOWN,            // XF86KbdBrightnessDown
    /* 230, 0x0e6 */   SDL_SCANCODE_UNKNOWN,            // XF86KbdBrightnessUp
    /* 231, 0x0e7 */   SDL_SCANCODE_UNKNOWN,            // XF86Send
    /* 232, 0x0e8 */   SDL_SCANCODE_UNKNOWN,            // XF86Reply
    /* 233, 0x0e9 */   SDL_SCANCODE_UNKNOWN,            // XF86MailForward
    /* 234, 0x0ea */   SDL_SCANCODE_UNKNOWN,            // XF86Save
    /* 235, 0x0eb */   SDL_SCANCODE_UNKNOWN,            // XF86Documents
    /* 236, 0x0ec */   SDL_SCANCODE_UNKNOWN,            // XF86Battery
    /* 237, 0x0ed */   SDL_SCANCODE_UNKNOWN,            // XF86Bluetooth
    /* 238, 0x0ee */   SDL_SCANCODE_UNKNOWN,            // XF86WLAN
    /* 239, 0x0ef */   SDL_SCANCODE_UNKNOWN,            // XF86UWB
    /* 240, 0x0f0 */   SDL_SCANCODE_UNKNOWN,            // NoSymbol
    /* 241, 0x0f1 */   SDL_SCANCODE_UNKNOWN,            // XF86Next_VMode
    /* 242, 0x0f2 */   SDL_SCANCODE_UNKNOWN,            // XF86Prev_VMode
    /* 243, 0x0f3 */   SDL_SCANCODE_UNKNOWN,            // XF86MonBrightnessCycle
    /* 244, 0x0f4 */   SDL_SCANCODE_UNKNOWN,            // XF86BrightnessAuto
    /* 245, 0x0f5 */   SDL_SCANCODE_UNKNOWN,            // XF86DisplayOff
    /* 246, 0x0f6 */   SDL_SCANCODE_UNKNOWN,            // XF86WWAN
    /* 247, 0x0f7 */   SDL_SCANCODE_UNKNOWN,            // XF86RFKill
};

// Xvnc / Xtightvnc scancodes from xmodmap -pk
static const SDL_Scancode xvnc_scancode_table[] = {
    /*  0 */    SDL_SCANCODE_LCTRL,
    /*  1 */    SDL_SCANCODE_RCTRL,
    /*  2 */    SDL_SCANCODE_LSHIFT,
    /*  3 */    SDL_SCANCODE_RSHIFT,
    /*  4 */    SDL_SCANCODE_UNKNOWN, // Meta_L
    /*  5 */    SDL_SCANCODE_UNKNOWN, // Meta_R
    /*  6 */    SDL_SCANCODE_LALT,
    /*  7 */    SDL_SCANCODE_RALT,
    /*  8 */    SDL_SCANCODE_SPACE,
    /*  9 */    SDL_SCANCODE_0,
    /*  10 */   SDL_SCANCODE_1,
    /*  11 */   SDL_SCANCODE_2,
    /*  12 */   SDL_SCANCODE_3,
    /*  13 */   SDL_SCANCODE_4,
    /*  14 */   SDL_SCANCODE_5,
    /*  15 */   SDL_SCANCODE_6,
    /*  16 */   SDL_SCANCODE_7,
    /*  17 */   SDL_SCANCODE_8,
    /*  18 */   SDL_SCANCODE_9,
    /*  19 */   SDL_SCANCODE_MINUS,
    /*  20 */   SDL_SCANCODE_EQUALS,
    /*  21 */   SDL_SCANCODE_LEFTBRACKET,
    /*  22 */   SDL_SCANCODE_RIGHTBRACKET,
    /*  23 */   SDL_SCANCODE_SEMICOLON,
    /*  24 */   SDL_SCANCODE_APOSTROPHE,
    /*  25 */   SDL_SCANCODE_GRAVE,
    /*  26 */   SDL_SCANCODE_COMMA,
    /*  27 */   SDL_SCANCODE_PERIOD,
    /*  28 */   SDL_SCANCODE_SLASH,
    /*  29 */   SDL_SCANCODE_BACKSLASH,
    /*  30 */   SDL_SCANCODE_A,
    /*  31 */   SDL_SCANCODE_B,
    /*  32 */   SDL_SCANCODE_C,
    /*  33 */   SDL_SCANCODE_D,
    /*  34 */   SDL_SCANCODE_E,
    /*  35 */   SDL_SCANCODE_F,
    /*  36 */   SDL_SCANCODE_G,
    /*  37 */   SDL_SCANCODE_H,
    /*  38 */   SDL_SCANCODE_I,
    /*  39 */   SDL_SCANCODE_J,
    /*  40 */   SDL_SCANCODE_K,
    /*  41 */   SDL_SCANCODE_L,
    /*  42 */   SDL_SCANCODE_M,
    /*  43 */   SDL_SCANCODE_N,
    /*  44 */   SDL_SCANCODE_O,
    /*  45 */   SDL_SCANCODE_P,
    /*  46 */   SDL_SCANCODE_Q,
    /*  47 */   SDL_SCANCODE_R,
    /*  48 */   SDL_SCANCODE_S,
    /*  49 */   SDL_SCANCODE_T,
    /*  50 */   SDL_SCANCODE_U,
    /*  51 */   SDL_SCANCODE_V,
    /*  52 */   SDL_SCANCODE_W,
    /*  53 */   SDL_SCANCODE_X,
    /*  54 */   SDL_SCANCODE_Y,
    /*  55 */   SDL_SCANCODE_Z,
    /*  56 */   SDL_SCANCODE_BACKSPACE,
    /*  57 */   SDL_SCANCODE_RETURN,
    /*  58 */   SDL_SCANCODE_TAB,
    /*  59 */   SDL_SCANCODE_ESCAPE,
    /*  60 */   SDL_SCANCODE_DELETE,
    /*  61 */   SDL_SCANCODE_HOME,
    /*  62 */   SDL_SCANCODE_END,
    /*  63 */   SDL_SCANCODE_PAGEUP,
    /*  64 */   SDL_SCANCODE_PAGEDOWN,
    /*  65 */   SDL_SCANCODE_UP,
    /*  66 */   SDL_SCANCODE_DOWN,
    /*  67 */   SDL_SCANCODE_LEFT,
    /*  68 */   SDL_SCANCODE_RIGHT,
    /*  69 */   SDL_SCANCODE_F1,
    /*  70 */   SDL_SCANCODE_F2,
    /*  71 */   SDL_SCANCODE_F3,
    /*  72 */   SDL_SCANCODE_F4,
    /*  73 */   SDL_SCANCODE_F5,
    /*  74 */   SDL_SCANCODE_F6,
    /*  75 */   SDL_SCANCODE_F7,
    /*  76 */   SDL_SCANCODE_F8,
    /*  77 */   SDL_SCANCODE_F9,
    /*  78 */   SDL_SCANCODE_F10,
    /*  79 */   SDL_SCANCODE_F11,
    /*  80 */   SDL_SCANCODE_F12,
};

#endif // scancodes_xfree86_h_

/* *INDENT-ON* */ // clang-format on
