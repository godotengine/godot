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

/* Linux virtual key code to SDL_Keycode mapping table
   Sources:
   - Linux kernel source input.h
*/
/* *INDENT-OFF* */ // clang-format off
static SDL_Scancode const linux_scancode_table[] = {
    /*   0, 0x000 */    SDL_SCANCODE_UNKNOWN,           // KEY_RESERVED
    /*   1, 0x001 */    SDL_SCANCODE_ESCAPE,            // KEY_ESC
    /*   2, 0x002 */    SDL_SCANCODE_1,                 // KEY_1
    /*   3, 0x003 */    SDL_SCANCODE_2,                 // KEY_2
    /*   4, 0x004 */    SDL_SCANCODE_3,                 // KEY_3
    /*   5, 0x005 */    SDL_SCANCODE_4,                 // KEY_4
    /*   6, 0x006 */    SDL_SCANCODE_5,                 // KEY_5
    /*   7, 0x007 */    SDL_SCANCODE_6,                 // KEY_6
    /*   8, 0x008 */    SDL_SCANCODE_7,                 // KEY_7
    /*   9, 0x009 */    SDL_SCANCODE_8,                 // KEY_8
    /*  10, 0x00a */    SDL_SCANCODE_9,                 // KEY_9
    /*  11, 0x00b */    SDL_SCANCODE_0,                 // KEY_0
    /*  12, 0x00c */    SDL_SCANCODE_MINUS,             // KEY_MINUS
    /*  13, 0x00d */    SDL_SCANCODE_EQUALS,            // KEY_EQUAL
    /*  14, 0x00e */    SDL_SCANCODE_BACKSPACE,         // KEY_BACKSPACE
    /*  15, 0x00f */    SDL_SCANCODE_TAB,               // KEY_TAB
    /*  16, 0x010 */    SDL_SCANCODE_Q,                 // KEY_Q
    /*  17, 0x011 */    SDL_SCANCODE_W,                 // KEY_W
    /*  18, 0x012 */    SDL_SCANCODE_E,                 // KEY_E
    /*  19, 0x013 */    SDL_SCANCODE_R,                 // KEY_R
    /*  20, 0x014 */    SDL_SCANCODE_T,                 // KEY_T
    /*  21, 0x015 */    SDL_SCANCODE_Y,                 // KEY_Y
    /*  22, 0x016 */    SDL_SCANCODE_U,                 // KEY_U
    /*  23, 0x017 */    SDL_SCANCODE_I,                 // KEY_I
    /*  24, 0x018 */    SDL_SCANCODE_O,                 // KEY_O
    /*  25, 0x019 */    SDL_SCANCODE_P,                 // KEY_P
    /*  26, 0x01a */    SDL_SCANCODE_LEFTBRACKET,       // KEY_LEFTBRACE
    /*  27, 0x01b */    SDL_SCANCODE_RIGHTBRACKET,      // KEY_RIGHTBRACE
    /*  28, 0x01c */    SDL_SCANCODE_RETURN,            // KEY_ENTER
    /*  29, 0x01d */    SDL_SCANCODE_LCTRL,             // KEY_LEFTCTRL
    /*  30, 0x01e */    SDL_SCANCODE_A,                 // KEY_A
    /*  31, 0x01f */    SDL_SCANCODE_S,                 // KEY_S
    /*  32, 0x020 */    SDL_SCANCODE_D,                 // KEY_D
    /*  33, 0x021 */    SDL_SCANCODE_F,                 // KEY_F
    /*  34, 0x022 */    SDL_SCANCODE_G,                 // KEY_G
    /*  35, 0x023 */    SDL_SCANCODE_H,                 // KEY_H
    /*  36, 0x024 */    SDL_SCANCODE_J,                 // KEY_J
    /*  37, 0x025 */    SDL_SCANCODE_K,                 // KEY_K
    /*  38, 0x026 */    SDL_SCANCODE_L,                 // KEY_L
    /*  39, 0x027 */    SDL_SCANCODE_SEMICOLON,         // KEY_SEMICOLON
    /*  40, 0x028 */    SDL_SCANCODE_APOSTROPHE,        // KEY_APOSTROPHE
    /*  41, 0x029 */    SDL_SCANCODE_GRAVE,             // KEY_GRAVE
    /*  42, 0x02a */    SDL_SCANCODE_LSHIFT,            // KEY_LEFTSHIFT
    /*  43, 0x02b */    SDL_SCANCODE_BACKSLASH,         // KEY_BACKSLASH
    /*  44, 0x02c */    SDL_SCANCODE_Z,                 // KEY_Z
    /*  45, 0x02d */    SDL_SCANCODE_X,                 // KEY_X
    /*  46, 0x02e */    SDL_SCANCODE_C,                 // KEY_C
    /*  47, 0x02f */    SDL_SCANCODE_V,                 // KEY_V
    /*  48, 0x030 */    SDL_SCANCODE_B,                 // KEY_B
    /*  49, 0x031 */    SDL_SCANCODE_N,                 // KEY_N
    /*  50, 0x032 */    SDL_SCANCODE_M,                 // KEY_M
    /*  51, 0x033 */    SDL_SCANCODE_COMMA,             // KEY_COMMA
    /*  52, 0x034 */    SDL_SCANCODE_PERIOD,            // KEY_DOT
    /*  53, 0x035 */    SDL_SCANCODE_SLASH,             // KEY_SLASH
    /*  54, 0x036 */    SDL_SCANCODE_RSHIFT,            // KEY_RIGHTSHIFT
    /*  55, 0x037 */    SDL_SCANCODE_KP_MULTIPLY,       // KEY_KPASTERISK
    /*  56, 0x038 */    SDL_SCANCODE_LALT,              // KEY_LEFTALT
    /*  57, 0x039 */    SDL_SCANCODE_SPACE,             // KEY_SPACE
    /*  58, 0x03a */    SDL_SCANCODE_CAPSLOCK,          // KEY_CAPSLOCK
    /*  59, 0x03b */    SDL_SCANCODE_F1,                // KEY_F1
    /*  60, 0x03c */    SDL_SCANCODE_F2,                // KEY_F2
    /*  61, 0x03d */    SDL_SCANCODE_F3,                // KEY_F3
    /*  62, 0x03e */    SDL_SCANCODE_F4,                // KEY_F4
    /*  63, 0x03f */    SDL_SCANCODE_F5,                // KEY_F5
    /*  64, 0x040 */    SDL_SCANCODE_F6,                // KEY_F6
    /*  65, 0x041 */    SDL_SCANCODE_F7,                // KEY_F7
    /*  66, 0x042 */    SDL_SCANCODE_F8,                // KEY_F8
    /*  67, 0x043 */    SDL_SCANCODE_F9,                // KEY_F9
    /*  68, 0x044 */    SDL_SCANCODE_F10,               // KEY_F10
    /*  69, 0x045 */    SDL_SCANCODE_NUMLOCKCLEAR,      // KEY_NUMLOCK
    /*  70, 0x046 */    SDL_SCANCODE_SCROLLLOCK,        // KEY_SCROLLLOCK
    /*  71, 0x047 */    SDL_SCANCODE_KP_7,              // KEY_KP7
    /*  72, 0x048 */    SDL_SCANCODE_KP_8,              // KEY_KP8
    /*  73, 0x049 */    SDL_SCANCODE_KP_9,              // KEY_KP9
    /*  74, 0x04a */    SDL_SCANCODE_KP_MINUS,          // KEY_KPMINUS
    /*  75, 0x04b */    SDL_SCANCODE_KP_4,              // KEY_KP4
    /*  76, 0x04c */    SDL_SCANCODE_KP_5,              // KEY_KP5
    /*  77, 0x04d */    SDL_SCANCODE_KP_6,              // KEY_KP6
    /*  78, 0x04e */    SDL_SCANCODE_KP_PLUS,           // KEY_KPPLUS
    /*  79, 0x04f */    SDL_SCANCODE_KP_1,              // KEY_KP1
    /*  80, 0x050 */    SDL_SCANCODE_KP_2,              // KEY_KP2
    /*  81, 0x051 */    SDL_SCANCODE_KP_3,              // KEY_KP3
    /*  82, 0x052 */    SDL_SCANCODE_KP_0,              // KEY_KP0
    /*  83, 0x053 */    SDL_SCANCODE_KP_PERIOD,         // KEY_KPDOT
    /*  84, 0x054 */    SDL_SCANCODE_UNKNOWN,
    /*  85, 0x055 */    SDL_SCANCODE_LANG5,             // KEY_ZENKAKUHANKAKU
    /*  86, 0x056 */    SDL_SCANCODE_NONUSBACKSLASH,    // KEY_102ND
    /*  87, 0x057 */    SDL_SCANCODE_F11,               // KEY_F11
    /*  88, 0x058 */    SDL_SCANCODE_F12,               // KEY_F12
    /*  89, 0x059 */    SDL_SCANCODE_INTERNATIONAL1,    // KEY_RO
    /*  90, 0x05a */    SDL_SCANCODE_LANG3,             // KEY_KATAKANA
    /*  91, 0x05b */    SDL_SCANCODE_LANG4,             // KEY_HIRAGANA
    /*  92, 0x05c */    SDL_SCANCODE_INTERNATIONAL4,    // KEY_HENKAN
    /*  93, 0x05d */    SDL_SCANCODE_INTERNATIONAL2,    // KEY_KATAKANAHIRAGANA
    /*  94, 0x05e */    SDL_SCANCODE_INTERNATIONAL5,    // KEY_MUHENKAN
    /*  95, 0x05f */    SDL_SCANCODE_INTERNATIONAL5,    // KEY_KPJPCOMMA
    /*  96, 0x060 */    SDL_SCANCODE_KP_ENTER,          // KEY_KPENTER
    /*  97, 0x061 */    SDL_SCANCODE_RCTRL,             // KEY_RIGHTCTRL
    /*  98, 0x062 */    SDL_SCANCODE_KP_DIVIDE,         // KEY_KPSLASH
    /*  99, 0x063 */    SDL_SCANCODE_SYSREQ,            // KEY_SYSRQ
    /* 100, 0x064 */    SDL_SCANCODE_RALT,              // KEY_RIGHTALT
    /* 101, 0x065 */    SDL_SCANCODE_UNKNOWN,           // KEY_LINEFEED
    /* 102, 0x066 */    SDL_SCANCODE_HOME,              // KEY_HOME
    /* 103, 0x067 */    SDL_SCANCODE_UP,                // KEY_UP
    /* 104, 0x068 */    SDL_SCANCODE_PAGEUP,            // KEY_PAGEUP
    /* 105, 0x069 */    SDL_SCANCODE_LEFT,              // KEY_LEFT
    /* 106, 0x06a */    SDL_SCANCODE_RIGHT,             // KEY_RIGHT
    /* 107, 0x06b */    SDL_SCANCODE_END,               // KEY_END
    /* 108, 0x06c */    SDL_SCANCODE_DOWN,              // KEY_DOWN
    /* 109, 0x06d */    SDL_SCANCODE_PAGEDOWN,          // KEY_PAGEDOWN
    /* 110, 0x06e */    SDL_SCANCODE_INSERT,            // KEY_INSERT
    /* 111, 0x06f */    SDL_SCANCODE_DELETE,            // KEY_DELETE
    /* 112, 0x070 */    SDL_SCANCODE_UNKNOWN,           // KEY_MACRO
    /* 113, 0x071 */    SDL_SCANCODE_MUTE,              // KEY_MUTE
    /* 114, 0x072 */    SDL_SCANCODE_VOLUMEDOWN,        // KEY_VOLUMEDOWN
    /* 115, 0x073 */    SDL_SCANCODE_VOLUMEUP,          // KEY_VOLUMEUP
    /* 116, 0x074 */    SDL_SCANCODE_POWER,             // KEY_POWER
    /* 117, 0x075 */    SDL_SCANCODE_KP_EQUALS,         // KEY_KPEQUAL
    /* 118, 0x076 */    SDL_SCANCODE_KP_PLUSMINUS,      // KEY_KPPLUSMINUS
    /* 119, 0x077 */    SDL_SCANCODE_PAUSE,             // KEY_PAUSE
    /* 120, 0x078 */    SDL_SCANCODE_UNKNOWN,           // KEY_SCALE
    /* 121, 0x079 */    SDL_SCANCODE_KP_COMMA,          // KEY_KPCOMMA
    /* 122, 0x07a */    SDL_SCANCODE_LANG1,             // KEY_HANGEUL
    /* 123, 0x07b */    SDL_SCANCODE_LANG2,             // KEY_HANJA
    /* 124, 0x07c */    SDL_SCANCODE_INTERNATIONAL3,    // KEY_YEN
    /* 125, 0x07d */    SDL_SCANCODE_LGUI,              // KEY_LEFTMETA
    /* 126, 0x07e */    SDL_SCANCODE_RGUI,              // KEY_RIGHTMETA
    /* 127, 0x07f */    SDL_SCANCODE_APPLICATION,       // KEY_COMPOSE
    /* 128, 0x080 */    SDL_SCANCODE_STOP,              // KEY_STOP
    /* 129, 0x081 */    SDL_SCANCODE_AGAIN,             // KEY_AGAIN
    /* 130, 0x082 */    SDL_SCANCODE_AC_PROPERTIES,     // KEY_PROPS
    /* 131, 0x083 */    SDL_SCANCODE_UNDO,              // KEY_UNDO
    /* 132, 0x084 */    SDL_SCANCODE_UNKNOWN,           // KEY_FRONT
    /* 133, 0x085 */    SDL_SCANCODE_COPY,              // KEY_COPY
    /* 134, 0x086 */    SDL_SCANCODE_AC_OPEN,           // KEY_OPEN
    /* 135, 0x087 */    SDL_SCANCODE_PASTE,             // KEY_PASTE
    /* 136, 0x088 */    SDL_SCANCODE_FIND,              // KEY_FIND
    /* 137, 0x089 */    SDL_SCANCODE_CUT,               // KEY_CUT
    /* 138, 0x08a */    SDL_SCANCODE_HELP,              // KEY_HELP
    /* 139, 0x08b */    SDL_SCANCODE_MENU,              // KEY_MENU
    /* 140, 0x08c */    SDL_SCANCODE_UNKNOWN,           // KEY_CALC
    /* 141, 0x08d */    SDL_SCANCODE_UNKNOWN,           // KEY_SETUP
    /* 142, 0x08e */    SDL_SCANCODE_SLEEP,             // KEY_SLEEP
    /* 143, 0x08f */    SDL_SCANCODE_WAKE,              // KEY_WAKEUP
    /* 144, 0x090 */    SDL_SCANCODE_UNKNOWN,           // KEY_FILE
    /* 145, 0x091 */    SDL_SCANCODE_UNKNOWN,           // KEY_SENDFILE
    /* 146, 0x092 */    SDL_SCANCODE_UNKNOWN,           // KEY_DELETEFILE
    /* 147, 0x093 */    SDL_SCANCODE_UNKNOWN,           // KEY_XFER
    /* 148, 0x094 */    SDL_SCANCODE_UNKNOWN,           // KEY_PROG1
    /* 149, 0x095 */    SDL_SCANCODE_UNKNOWN,           // KEY_PROG2
    /* 150, 0x096 */    SDL_SCANCODE_UNKNOWN,           // KEY_WWW
    /* 151, 0x097 */    SDL_SCANCODE_UNKNOWN,           // KEY_MSDOS
    /* 152, 0x098 */    SDL_SCANCODE_UNKNOWN,           // KEY_COFFEE
    /* 153, 0x099 */    SDL_SCANCODE_UNKNOWN,           // KEY_ROTATE_DISPLAY
    /* 154, 0x09a */    SDL_SCANCODE_UNKNOWN,           // KEY_CYCLEWINDOWS
    /* 155, 0x09b */    SDL_SCANCODE_UNKNOWN,           // KEY_MAIL
    /* 156, 0x09c */    SDL_SCANCODE_AC_BOOKMARKS,      // KEY_BOOKMARKS
    /* 157, 0x09d */    SDL_SCANCODE_UNKNOWN,           // KEY_COMPUTER
    /* 158, 0x09e */    SDL_SCANCODE_AC_BACK,           // KEY_BACK
    /* 159, 0x09f */    SDL_SCANCODE_AC_FORWARD,        // KEY_FORWARD
    /* 160, 0x0a0 */    SDL_SCANCODE_UNKNOWN,           // KEY_CLOSECD
    /* 161, 0x0a1 */    SDL_SCANCODE_MEDIA_EJECT,       // KEY_EJECTCD
    /* 162, 0x0a2 */    SDL_SCANCODE_MEDIA_EJECT,       // KEY_EJECTCLOSECD
    /* 163, 0x0a3 */    SDL_SCANCODE_MEDIA_NEXT_TRACK,  // KEY_NEXTSONG
    /* 164, 0x0a4 */    SDL_SCANCODE_MEDIA_PLAY_PAUSE,   // KEY_PLAYPAUSE
    /* 165, 0x0a5 */    SDL_SCANCODE_MEDIA_PREVIOUS_TRACK, // KEY_PREVIOUSSONG
    /* 166, 0x0a6 */    SDL_SCANCODE_MEDIA_STOP,        // KEY_STOPCD
    /* 167, 0x0a7 */    SDL_SCANCODE_MEDIA_RECORD,      // KEY_RECORD
    /* 168, 0x0a8 */    SDL_SCANCODE_MEDIA_REWIND,      // KEY_REWIND
    /* 169, 0x0a9 */    SDL_SCANCODE_UNKNOWN,           // KEY_PHONE
    /* 170, 0x0aa */    SDL_SCANCODE_UNKNOWN,           // KEY_ISO
    /* 171, 0x0ab */    SDL_SCANCODE_UNKNOWN,           // KEY_CONFIG
    /* 172, 0x0ac */    SDL_SCANCODE_AC_HOME,           // KEY_HOMEPAGE
    /* 173, 0x0ad */    SDL_SCANCODE_AC_REFRESH,        // KEY_REFRESH
    /* 174, 0x0ae */    SDL_SCANCODE_AC_EXIT,           // KEY_EXIT
    /* 175, 0x0af */    SDL_SCANCODE_UNKNOWN,           // KEY_MOVE
    /* 176, 0x0b0 */    SDL_SCANCODE_UNKNOWN,           // KEY_EDIT
    /* 177, 0x0b1 */    SDL_SCANCODE_UNKNOWN,           // KEY_SCROLLUP
    /* 178, 0x0b2 */    SDL_SCANCODE_UNKNOWN,           // KEY_SCROLLDOWN
    /* 179, 0x0b3 */    SDL_SCANCODE_KP_LEFTPAREN,      // KEY_KPLEFTPAREN
    /* 180, 0x0b4 */    SDL_SCANCODE_KP_RIGHTPAREN,     // KEY_KPRIGHTPAREN
    /* 181, 0x0b5 */    SDL_SCANCODE_AC_NEW,            // KEY_NEW
    /* 182, 0x0b6 */    SDL_SCANCODE_AGAIN,             // KEY_REDO
    /* 183, 0x0b7 */    SDL_SCANCODE_F13,               // KEY_F13
    /* 184, 0x0b8 */    SDL_SCANCODE_F14,               // KEY_F14
    /* 185, 0x0b9 */    SDL_SCANCODE_F15,               // KEY_F15
    /* 186, 0x0ba */    SDL_SCANCODE_F16,               // KEY_F16
    /* 187, 0x0bb */    SDL_SCANCODE_F17,               // KEY_F17
    /* 188, 0x0bc */    SDL_SCANCODE_F18,               // KEY_F18
    /* 189, 0x0bd */    SDL_SCANCODE_F19,               // KEY_F19
    /* 190, 0x0be */    SDL_SCANCODE_F20,               // KEY_F20
    /* 191, 0x0bf */    SDL_SCANCODE_F21,               // KEY_F21
    /* 192, 0x0c0 */    SDL_SCANCODE_F22,               // KEY_F22
    /* 193, 0x0c1 */    SDL_SCANCODE_F23,               // KEY_F23
    /* 194, 0x0c2 */    SDL_SCANCODE_F24,               // KEY_F24
    /* 195, 0x0c3 */    SDL_SCANCODE_UNKNOWN,
    /* 196, 0x0c4 */    SDL_SCANCODE_UNKNOWN,
    /* 197, 0x0c5 */    SDL_SCANCODE_UNKNOWN,
    /* 198, 0x0c6 */    SDL_SCANCODE_UNKNOWN,
    /* 199, 0x0c7 */    SDL_SCANCODE_UNKNOWN,
    /* 200, 0x0c8 */    SDL_SCANCODE_MEDIA_PLAY,        // KEY_PLAYCD
    /* 201, 0x0c9 */    SDL_SCANCODE_MEDIA_PAUSE,       // KEY_PAUSECD
    /* 202, 0x0ca */    SDL_SCANCODE_UNKNOWN,           // KEY_PROG3
    /* 203, 0x0cb */    SDL_SCANCODE_UNKNOWN,           // KEY_PROG4
    /* 204, 0x0cc */    SDL_SCANCODE_UNKNOWN,           // KEY_ALL_APPLICATIONS
    /* 205, 0x0cd */    SDL_SCANCODE_UNKNOWN,           // KEY_SUSPEND
    /* 206, 0x0ce */    SDL_SCANCODE_AC_CLOSE,          // KEY_CLOSE
    /* 207, 0x0cf */    SDL_SCANCODE_MEDIA_PLAY,        // KEY_PLAY
    /* 208, 0x0d0 */    SDL_SCANCODE_MEDIA_FAST_FORWARD, // KEY_FASTFORWARD
    /* 209, 0x0d1 */    SDL_SCANCODE_UNKNOWN,           // KEY_BASSBOOST
    /* 210, 0x0d2 */    SDL_SCANCODE_PRINTSCREEN,       // KEY_PRINT
    /* 211, 0x0d3 */    SDL_SCANCODE_UNKNOWN,           // KEY_HP
    /* 212, 0x0d4 */    SDL_SCANCODE_UNKNOWN,           // KEY_CAMERA
    /* 213, 0x0d5 */    SDL_SCANCODE_UNKNOWN,           // KEY_SOUND
    /* 214, 0x0d6 */    SDL_SCANCODE_UNKNOWN,           // KEY_QUESTION
    /* 215, 0x0d7 */    SDL_SCANCODE_UNKNOWN,           // KEY_EMAIL
    /* 216, 0x0d8 */    SDL_SCANCODE_UNKNOWN,           // KEY_CHAT
    /* 217, 0x0d9 */    SDL_SCANCODE_AC_SEARCH,         // KEY_SEARCH
    /* 218, 0x0da */    SDL_SCANCODE_UNKNOWN,           // KEY_CONNECT
    /* 219, 0x0db */    SDL_SCANCODE_UNKNOWN,           // KEY_FINANCE
    /* 220, 0x0dc */    SDL_SCANCODE_UNKNOWN,           // KEY_SPORT
    /* 221, 0x0dd */    SDL_SCANCODE_UNKNOWN,           // KEY_SHOP
    /* 222, 0x0de */    SDL_SCANCODE_ALTERASE,          // KEY_ALTERASE
    /* 223, 0x0df */    SDL_SCANCODE_CANCEL,            // KEY_CANCEL
    /* 224, 0x0e0 */    SDL_SCANCODE_UNKNOWN,           // KEY_BRIGHTNESSDOWN
    /* 225, 0x0e1 */    SDL_SCANCODE_UNKNOWN,           // KEY_BRIGHTNESSUP
    /* 226, 0x0e2 */    SDL_SCANCODE_MEDIA_SELECT,      // KEY_MEDIA
    /* 227, 0x0e3 */    SDL_SCANCODE_UNKNOWN,           // KEY_SWITCHVIDEOMODE
    /* 228, 0x0e4 */    SDL_SCANCODE_UNKNOWN,           // KEY_KBDILLUMTOGGLE
    /* 229, 0x0e5 */    SDL_SCANCODE_UNKNOWN,           // KEY_KBDILLUMDOWN
    /* 230, 0x0e6 */    SDL_SCANCODE_UNKNOWN,           // KEY_KBDILLUMUP
    /* 231, 0x0e7 */    SDL_SCANCODE_UNKNOWN,           // KEY_SEND
    /* 232, 0x0e8 */    SDL_SCANCODE_UNKNOWN,           // KEY_REPLY
    /* 233, 0x0e9 */    SDL_SCANCODE_UNKNOWN,           // KEY_FORWARDMAIL
    /* 234, 0x0ea */    SDL_SCANCODE_AC_SAVE,           // KEY_SAVE
    /* 235, 0x0eb */    SDL_SCANCODE_UNKNOWN,           // KEY_DOCUMENTS
    /* 236, 0x0ec */    SDL_SCANCODE_UNKNOWN,           // KEY_BATTERY
    /* 237, 0x0ed */    SDL_SCANCODE_UNKNOWN,           // KEY_BLUETOOTH
    /* 238, 0x0ee */    SDL_SCANCODE_UNKNOWN,           // KEY_WLAN
    /* 239, 0x0ef */    SDL_SCANCODE_UNKNOWN,           // KEY_UWB
    /* 240, 0x0f0 */    SDL_SCANCODE_UNKNOWN,           // KEY_UNKNOWN
    /* 241, 0x0f1 */    SDL_SCANCODE_UNKNOWN,           // KEY_VIDEO_NEXT
    /* 242, 0x0f2 */    SDL_SCANCODE_UNKNOWN,           // KEY_VIDEO_PREV
    /* 243, 0x0f3 */    SDL_SCANCODE_UNKNOWN,           // KEY_BRIGHTNESS_CYCLE
    /* 244, 0x0f4 */    SDL_SCANCODE_UNKNOWN,           // KEY_BRIGHTNESS_AUTO
    /* 245, 0x0f5 */    SDL_SCANCODE_UNKNOWN,           // KEY_DISPLAY_OFF
    /* 246, 0x0f6 */    SDL_SCANCODE_UNKNOWN,           // KEY_WWAN
    /* 247, 0x0f7 */    SDL_SCANCODE_UNKNOWN,           // KEY_RFKILL
    /* 248, 0x0f8 */    SDL_SCANCODE_UNKNOWN,           // KEY_MICMUTE
    /* 249, 0x0f9 */    SDL_SCANCODE_UNKNOWN,
    /* 250, 0x0fa */    SDL_SCANCODE_UNKNOWN,
    /* 251, 0x0fb */    SDL_SCANCODE_UNKNOWN,
    /* 252, 0x0fc */    SDL_SCANCODE_UNKNOWN,
    /* 253, 0x0fd */    SDL_SCANCODE_UNKNOWN,
    /* 254, 0x0fe */    SDL_SCANCODE_UNKNOWN,
    /* 255, 0x0ff */    SDL_SCANCODE_UNKNOWN,
    /* 256, 0x100 */    SDL_SCANCODE_UNKNOWN,
    /* 257, 0x101 */    SDL_SCANCODE_UNKNOWN,
    /* 258, 0x102 */    SDL_SCANCODE_UNKNOWN,
    /* 259, 0x103 */    SDL_SCANCODE_UNKNOWN,
    /* 260, 0x104 */    SDL_SCANCODE_UNKNOWN,
    /* 261, 0x105 */    SDL_SCANCODE_UNKNOWN,
    /* 262, 0x106 */    SDL_SCANCODE_UNKNOWN,
    /* 263, 0x107 */    SDL_SCANCODE_UNKNOWN,
    /* 264, 0x108 */    SDL_SCANCODE_UNKNOWN,
    /* 265, 0x109 */    SDL_SCANCODE_UNKNOWN,
    /* 266, 0x10a */    SDL_SCANCODE_UNKNOWN,
    /* 267, 0x10b */    SDL_SCANCODE_UNKNOWN,
    /* 268, 0x10c */    SDL_SCANCODE_UNKNOWN,
    /* 269, 0x10d */    SDL_SCANCODE_UNKNOWN,
    /* 270, 0x10e */    SDL_SCANCODE_UNKNOWN,
    /* 271, 0x10f */    SDL_SCANCODE_UNKNOWN,
    /* 272, 0x110 */    SDL_SCANCODE_UNKNOWN,
    /* 273, 0x111 */    SDL_SCANCODE_UNKNOWN,
    /* 274, 0x112 */    SDL_SCANCODE_UNKNOWN,
    /* 275, 0x113 */    SDL_SCANCODE_UNKNOWN,
    /* 276, 0x114 */    SDL_SCANCODE_UNKNOWN,
    /* 277, 0x115 */    SDL_SCANCODE_UNKNOWN,
    /* 278, 0x116 */    SDL_SCANCODE_UNKNOWN,
    /* 279, 0x117 */    SDL_SCANCODE_UNKNOWN,
    /* 280, 0x118 */    SDL_SCANCODE_UNKNOWN,
    /* 281, 0x119 */    SDL_SCANCODE_UNKNOWN,
    /* 282, 0x11a */    SDL_SCANCODE_UNKNOWN,
    /* 283, 0x11b */    SDL_SCANCODE_UNKNOWN,
    /* 284, 0x11c */    SDL_SCANCODE_UNKNOWN,
    /* 285, 0x11d */    SDL_SCANCODE_UNKNOWN,
    /* 286, 0x11e */    SDL_SCANCODE_UNKNOWN,
    /* 287, 0x11f */    SDL_SCANCODE_UNKNOWN,
    /* 288, 0x120 */    SDL_SCANCODE_UNKNOWN,
    /* 289, 0x121 */    SDL_SCANCODE_UNKNOWN,
    /* 290, 0x122 */    SDL_SCANCODE_UNKNOWN,
    /* 291, 0x123 */    SDL_SCANCODE_UNKNOWN,
    /* 292, 0x124 */    SDL_SCANCODE_UNKNOWN,
    /* 293, 0x125 */    SDL_SCANCODE_UNKNOWN,
    /* 294, 0x126 */    SDL_SCANCODE_UNKNOWN,
    /* 295, 0x127 */    SDL_SCANCODE_UNKNOWN,
    /* 296, 0x128 */    SDL_SCANCODE_UNKNOWN,
    /* 297, 0x129 */    SDL_SCANCODE_UNKNOWN,
    /* 298, 0x12a */    SDL_SCANCODE_UNKNOWN,
    /* 299, 0x12b */    SDL_SCANCODE_UNKNOWN,
    /* 300, 0x12c */    SDL_SCANCODE_UNKNOWN,
    /* 301, 0x12d */    SDL_SCANCODE_UNKNOWN,
    /* 302, 0x12e */    SDL_SCANCODE_UNKNOWN,
    /* 303, 0x12f */    SDL_SCANCODE_UNKNOWN,
    /* 304, 0x130 */    SDL_SCANCODE_UNKNOWN,
    /* 305, 0x131 */    SDL_SCANCODE_UNKNOWN,
    /* 306, 0x132 */    SDL_SCANCODE_UNKNOWN,
    /* 307, 0x133 */    SDL_SCANCODE_UNKNOWN,
    /* 308, 0x134 */    SDL_SCANCODE_UNKNOWN,
    /* 309, 0x135 */    SDL_SCANCODE_UNKNOWN,
    /* 310, 0x136 */    SDL_SCANCODE_UNKNOWN,
    /* 311, 0x137 */    SDL_SCANCODE_UNKNOWN,
    /* 312, 0x138 */    SDL_SCANCODE_UNKNOWN,
    /* 313, 0x139 */    SDL_SCANCODE_UNKNOWN,
    /* 314, 0x13a */    SDL_SCANCODE_UNKNOWN,
    /* 315, 0x13b */    SDL_SCANCODE_UNKNOWN,
    /* 316, 0x13c */    SDL_SCANCODE_UNKNOWN,
    /* 317, 0x13d */    SDL_SCANCODE_UNKNOWN,
    /* 318, 0x13e */    SDL_SCANCODE_UNKNOWN,
    /* 319, 0x13f */    SDL_SCANCODE_UNKNOWN,
    /* 320, 0x140 */    SDL_SCANCODE_UNKNOWN,
    /* 321, 0x141 */    SDL_SCANCODE_UNKNOWN,
    /* 322, 0x142 */    SDL_SCANCODE_UNKNOWN,
    /* 323, 0x143 */    SDL_SCANCODE_UNKNOWN,
    /* 324, 0x144 */    SDL_SCANCODE_UNKNOWN,
    /* 325, 0x145 */    SDL_SCANCODE_UNKNOWN,
    /* 326, 0x146 */    SDL_SCANCODE_UNKNOWN,
    /* 327, 0x147 */    SDL_SCANCODE_UNKNOWN,
    /* 328, 0x148 */    SDL_SCANCODE_UNKNOWN,
    /* 329, 0x149 */    SDL_SCANCODE_UNKNOWN,
    /* 330, 0x14a */    SDL_SCANCODE_UNKNOWN,
    /* 331, 0x14b */    SDL_SCANCODE_UNKNOWN,
    /* 332, 0x14c */    SDL_SCANCODE_UNKNOWN,
    /* 333, 0x14d */    SDL_SCANCODE_UNKNOWN,
    /* 334, 0x14e */    SDL_SCANCODE_UNKNOWN,
    /* 335, 0x14f */    SDL_SCANCODE_UNKNOWN,
    /* 336, 0x150 */    SDL_SCANCODE_UNKNOWN,
    /* 337, 0x151 */    SDL_SCANCODE_UNKNOWN,
    /* 338, 0x152 */    SDL_SCANCODE_UNKNOWN,
    /* 339, 0x153 */    SDL_SCANCODE_UNKNOWN,
    /* 340, 0x154 */    SDL_SCANCODE_UNKNOWN,
    /* 341, 0x155 */    SDL_SCANCODE_UNKNOWN,
    /* 342, 0x156 */    SDL_SCANCODE_UNKNOWN,
    /* 343, 0x157 */    SDL_SCANCODE_UNKNOWN,
    /* 344, 0x158 */    SDL_SCANCODE_UNKNOWN,
    /* 345, 0x159 */    SDL_SCANCODE_UNKNOWN,
    /* 346, 0x15a */    SDL_SCANCODE_UNKNOWN,
    /* 347, 0x15b */    SDL_SCANCODE_UNKNOWN,
    /* 348, 0x15c */    SDL_SCANCODE_UNKNOWN,
    /* 349, 0x15d */    SDL_SCANCODE_UNKNOWN,
    /* 350, 0x15e */    SDL_SCANCODE_UNKNOWN,
    /* 351, 0x15f */    SDL_SCANCODE_UNKNOWN,
    /* 352, 0x160 */    SDL_SCANCODE_UNKNOWN,            // KEY_OK
    /* 353, 0x161 */    SDL_SCANCODE_SELECT,             // KEY_SELECT
    /* 354, 0x162 */    SDL_SCANCODE_UNKNOWN,            // KEY_GOTO
    /* 355, 0x163 */    SDL_SCANCODE_CLEAR,              // KEY_CLEAR
    /* 356, 0x164 */    SDL_SCANCODE_UNKNOWN,            // KEY_POWER2
    /* 357, 0x165 */    SDL_SCANCODE_UNKNOWN,            // KEY_OPTION
    /* 358, 0x166 */    SDL_SCANCODE_UNKNOWN,            // KEY_INFO
    /* 359, 0x167 */    SDL_SCANCODE_UNKNOWN,            // KEY_TIME
    /* 360, 0x168 */    SDL_SCANCODE_UNKNOWN,            // KEY_VENDOR
    /* 361, 0x169 */    SDL_SCANCODE_UNKNOWN,            // KEY_ARCHIVE
    /* 362, 0x16a */    SDL_SCANCODE_UNKNOWN,            // KEY_PROGRAM
    /* 363, 0x16b */    SDL_SCANCODE_UNKNOWN,            // KEY_CHANNEL
    /* 364, 0x16c */    SDL_SCANCODE_UNKNOWN,            // KEY_FAVORITES
    /* 365, 0x16d */    SDL_SCANCODE_UNKNOWN,            // KEY_EPG
    /* 366, 0x16e */    SDL_SCANCODE_UNKNOWN,            // KEY_PVR
    /* 367, 0x16f */    SDL_SCANCODE_UNKNOWN,            // KEY_MHP
    /* 368, 0x170 */    SDL_SCANCODE_UNKNOWN,            // KEY_LANGUAGE
    /* 369, 0x171 */    SDL_SCANCODE_UNKNOWN,            // KEY_TITLE
    /* 370, 0x172 */    SDL_SCANCODE_UNKNOWN,            // KEY_SUBTITLE
    /* 371, 0x173 */    SDL_SCANCODE_UNKNOWN,            // KEY_ANGLE
    /* 372, 0x174 */    SDL_SCANCODE_UNKNOWN,            // KEY_FULL_SCREEN
    /* 373, 0x175 */    SDL_SCANCODE_MODE,               // KEY_MODE
    /* 374, 0x176 */    SDL_SCANCODE_UNKNOWN,            // KEY_KEYBOARD
    /* 375, 0x177 */    SDL_SCANCODE_UNKNOWN,            // KEY_ASPECT_RATIO
    /* 376, 0x178 */    SDL_SCANCODE_UNKNOWN,            // KEY_PC
    /* 377, 0x179 */    SDL_SCANCODE_UNKNOWN,            // KEY_TV
    /* 378, 0x17a */    SDL_SCANCODE_UNKNOWN,            // KEY_TV2
    /* 379, 0x17b */    SDL_SCANCODE_UNKNOWN,            // KEY_VCR
    /* 380, 0x17c */    SDL_SCANCODE_UNKNOWN,            // KEY_VCR2
    /* 381, 0x17d */    SDL_SCANCODE_UNKNOWN,            // KEY_SAT
    /* 382, 0x17e */    SDL_SCANCODE_UNKNOWN,            // KEY_SAT2
    /* 383, 0x17f */    SDL_SCANCODE_UNKNOWN,            // KEY_CD
    /* 384, 0x180 */    SDL_SCANCODE_UNKNOWN,            // KEY_TAPE
    /* 385, 0x181 */    SDL_SCANCODE_UNKNOWN,            // KEY_RADIO
    /* 386, 0x182 */    SDL_SCANCODE_UNKNOWN,            // KEY_TUNER
    /* 387, 0x183 */    SDL_SCANCODE_UNKNOWN,            // KEY_PLAYER
    /* 388, 0x184 */    SDL_SCANCODE_UNKNOWN,            // KEY_TEXT
    /* 389, 0x185 */    SDL_SCANCODE_UNKNOWN,            // KEY_DVD
    /* 390, 0x186 */    SDL_SCANCODE_UNKNOWN,            // KEY_AUX
    /* 391, 0x187 */    SDL_SCANCODE_UNKNOWN,            // KEY_MP3
    /* 392, 0x188 */    SDL_SCANCODE_UNKNOWN,            // KEY_AUDIO
    /* 393, 0x189 */    SDL_SCANCODE_UNKNOWN,            // KEY_VIDEO
    /* 394, 0x18a */    SDL_SCANCODE_UNKNOWN,            // KEY_DIRECTORY
    /* 395, 0x18b */    SDL_SCANCODE_UNKNOWN,            // KEY_LIST
    /* 396, 0x18c */    SDL_SCANCODE_UNKNOWN,            // KEY_MEMO
    /* 397, 0x18d */    SDL_SCANCODE_UNKNOWN,            // KEY_CALENDAR
    /* 398, 0x18e */    SDL_SCANCODE_UNKNOWN,            // KEY_RED
    /* 399, 0x18f */    SDL_SCANCODE_UNKNOWN,            // KEY_GREEN
    /* 400, 0x190 */    SDL_SCANCODE_UNKNOWN,            // KEY_YELLOW
    /* 401, 0x191 */    SDL_SCANCODE_UNKNOWN,            // KEY_BLUE
    /* 402, 0x192 */    SDL_SCANCODE_CHANNEL_INCREMENT,  // KEY_CHANNELUP
    /* 403, 0x193 */    SDL_SCANCODE_CHANNEL_DECREMENT,  // KEY_CHANNELDOWN
#if 0 // We don't have any mapped scancodes after this point (yet)
    /* 404, 0x194 */    SDL_SCANCODE_UNKNOWN,            // KEY_FIRST
    /* 405, 0x195 */    SDL_SCANCODE_UNKNOWN,            // KEY_LAST
    /* 406, 0x196 */    SDL_SCANCODE_UNKNOWN,            // KEY_AB
    /* 407, 0x197 */    SDL_SCANCODE_UNKNOWN,            // KEY_NEXT
    /* 408, 0x198 */    SDL_SCANCODE_UNKNOWN,            // KEY_RESTART
    /* 409, 0x199 */    SDL_SCANCODE_UNKNOWN,            // KEY_SLOW
    /* 410, 0x19a */    SDL_SCANCODE_UNKNOWN,            // KEY_SHUFFLE
    /* 411, 0x19b */    SDL_SCANCODE_UNKNOWN,            // KEY_BREAK
    /* 412, 0x19c */    SDL_SCANCODE_UNKNOWN,            // KEY_PREVIOUS
    /* 413, 0x19d */    SDL_SCANCODE_UNKNOWN,            // KEY_DIGITS
    /* 414, 0x19e */    SDL_SCANCODE_UNKNOWN,            // KEY_TEEN
    /* 415, 0x19f */    SDL_SCANCODE_UNKNOWN,            // KEY_TWEN
    /* 416, 0x1a0 */    SDL_SCANCODE_UNKNOWN,            // KEY_VIDEOPHONE
    /* 417, 0x1a1 */    SDL_SCANCODE_UNKNOWN,            // KEY_GAMES
    /* 418, 0x1a2 */    SDL_SCANCODE_UNKNOWN,            // KEY_ZOOMIN
    /* 419, 0x1a3 */    SDL_SCANCODE_UNKNOWN,            // KEY_ZOOMOUT
    /* 420, 0x1a4 */    SDL_SCANCODE_UNKNOWN,            // KEY_ZOOMRESET
    /* 421, 0x1a5 */    SDL_SCANCODE_UNKNOWN,            // KEY_WORDPROCESSOR
    /* 422, 0x1a6 */    SDL_SCANCODE_UNKNOWN,            // KEY_EDITOR
    /* 423, 0x1a7 */    SDL_SCANCODE_UNKNOWN,            // KEY_SPREADSHEET
    /* 424, 0x1a8 */    SDL_SCANCODE_UNKNOWN,            // KEY_GRAPHICSEDITOR
    /* 425, 0x1a9 */    SDL_SCANCODE_UNKNOWN,            // KEY_PRESENTATION
    /* 426, 0x1aa */    SDL_SCANCODE_UNKNOWN,            // KEY_DATABASE
    /* 427, 0x1ab */    SDL_SCANCODE_UNKNOWN,            // KEY_NEWS
    /* 428, 0x1ac */    SDL_SCANCODE_UNKNOWN,            // KEY_VOICEMAIL
    /* 429, 0x1ad */    SDL_SCANCODE_UNKNOWN,            // KEY_ADDRESSBOOK
    /* 430, 0x1ae */    SDL_SCANCODE_UNKNOWN,            // KEY_MESSENGER
    /* 431, 0x1af */    SDL_SCANCODE_UNKNOWN,            // KEY_DISPLAYTOGGLE
    /* 432, 0x1b0 */    SDL_SCANCODE_UNKNOWN,            // KEY_SPELLCHECK
    /* 433, 0x1b1 */    SDL_SCANCODE_UNKNOWN,            // KEY_LOGOFF
    /* 434, 0x1b2 */    SDL_SCANCODE_UNKNOWN,            // KEY_DOLLAR
    /* 435, 0x1b3 */    SDL_SCANCODE_UNKNOWN,            // KEY_EURO
    /* 436, 0x1b4 */    SDL_SCANCODE_UNKNOWN,            // KEY_FRAMEBACK
    /* 437, 0x1b5 */    SDL_SCANCODE_UNKNOWN,            // KEY_FRAMEFORWARD
    /* 438, 0x1b6 */    SDL_SCANCODE_UNKNOWN,            // KEY_CONTEXT_MENU
    /* 439, 0x1b7 */    SDL_SCANCODE_UNKNOWN,            // KEY_MEDIA_REPEAT
    /* 440, 0x1b8 */    SDL_SCANCODE_UNKNOWN,            // KEY_10CHANNELSUP
    /* 441, 0x1b9 */    SDL_SCANCODE_UNKNOWN,            // KEY_10CHANNELSDOWN
    /* 442, 0x1ba */    SDL_SCANCODE_UNKNOWN,            // KEY_IMAGES
    /* 443, 0x1bb */    SDL_SCANCODE_UNKNOWN,
    /* 444, 0x1bc */    SDL_SCANCODE_UNKNOWN,            // KEY_NOTIFICATION_CENTER
    /* 445, 0x1bd */    SDL_SCANCODE_UNKNOWN,            // KEY_PICKUP_PHONE
    /* 446, 0x1be */    SDL_SCANCODE_UNKNOWN,            // KEY_HANGUP_PHONE
    /* 447, 0x1bf */    SDL_SCANCODE_UNKNOWN,
    /* 448, 0x1c0 */    SDL_SCANCODE_UNKNOWN,            // KEY_DEL_EOL
    /* 449, 0x1c1 */    SDL_SCANCODE_UNKNOWN,            // KEY_DEL_EOS
    /* 450, 0x1c2 */    SDL_SCANCODE_UNKNOWN,            // KEY_INS_LINE
    /* 451, 0x1c3 */    SDL_SCANCODE_UNKNOWN,            // KEY_DEL_LINE
    /* 452, 0x1c4 */    SDL_SCANCODE_UNKNOWN,
    /* 453, 0x1c5 */    SDL_SCANCODE_UNKNOWN,
    /* 454, 0x1c6 */    SDL_SCANCODE_UNKNOWN,
    /* 455, 0x1c7 */    SDL_SCANCODE_UNKNOWN,
    /* 456, 0x1c8 */    SDL_SCANCODE_UNKNOWN,
    /* 457, 0x1c9 */    SDL_SCANCODE_UNKNOWN,
    /* 458, 0x1ca */    SDL_SCANCODE_UNKNOWN,
    /* 459, 0x1cb */    SDL_SCANCODE_UNKNOWN,
    /* 460, 0x1cc */    SDL_SCANCODE_UNKNOWN,
    /* 461, 0x1cd */    SDL_SCANCODE_UNKNOWN,
    /* 462, 0x1ce */    SDL_SCANCODE_UNKNOWN,
    /* 463, 0x1cf */    SDL_SCANCODE_UNKNOWN,
    /* 464, 0x1d0 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN
    /* 465, 0x1d1 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_ESC
    /* 466, 0x1d2 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F1
    /* 467, 0x1d3 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F2
    /* 468, 0x1d4 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F3
    /* 469, 0x1d5 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F4
    /* 470, 0x1d6 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F5
    /* 471, 0x1d7 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F6
    /* 472, 0x1d8 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F7
    /* 473, 0x1d9 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F8
    /* 474, 0x1da */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F9
    /* 475, 0x1db */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F10
    /* 476, 0x1dc */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F11
    /* 477, 0x1dd */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F12
    /* 478, 0x1de */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_1
    /* 479, 0x1df */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_2
    /* 480, 0x1e0 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_D
    /* 481, 0x1e1 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_E
    /* 482, 0x1e2 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_F
    /* 483, 0x1e3 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_S
    /* 484, 0x1e4 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_B
    /* 485, 0x1e5 */    SDL_SCANCODE_UNKNOWN,            // KEY_FN_RIGHT_SHIFT
    /* 486, 0x1e6 */    SDL_SCANCODE_UNKNOWN,
    /* 487, 0x1e7 */    SDL_SCANCODE_UNKNOWN,
    /* 488, 0x1e8 */    SDL_SCANCODE_UNKNOWN,
    /* 489, 0x1e9 */    SDL_SCANCODE_UNKNOWN,
    /* 490, 0x1ea */    SDL_SCANCODE_UNKNOWN,
    /* 491, 0x1eb */    SDL_SCANCODE_UNKNOWN,
    /* 492, 0x1ec */    SDL_SCANCODE_UNKNOWN,
    /* 493, 0x1ed */    SDL_SCANCODE_UNKNOWN,
    /* 494, 0x1ee */    SDL_SCANCODE_UNKNOWN,
    /* 495, 0x1ef */    SDL_SCANCODE_UNKNOWN,
    /* 496, 0x1f0 */    SDL_SCANCODE_UNKNOWN,
    /* 497, 0x1f1 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT1
    /* 498, 0x1f2 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT2
    /* 499, 0x1f3 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT3
    /* 500, 0x1f4 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT4
    /* 501, 0x1f5 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT5
    /* 502, 0x1f6 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT6
    /* 503, 0x1f7 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT7
    /* 504, 0x1f8 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT8
    /* 505, 0x1f9 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT9
    /* 506, 0x1fa */    SDL_SCANCODE_UNKNOWN,            // KEY_BRL_DOT10
    /* 507, 0x1fb */    SDL_SCANCODE_UNKNOWN,
    /* 508, 0x1fc */    SDL_SCANCODE_UNKNOWN,
    /* 509, 0x1fd */    SDL_SCANCODE_UNKNOWN,
    /* 510, 0x1fe */    SDL_SCANCODE_UNKNOWN,
    /* 511, 0x1ff */    SDL_SCANCODE_UNKNOWN,
    /* 512, 0x200 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_0
    /* 513, 0x201 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_1
    /* 514, 0x202 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_2
    /* 515, 0x203 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_3
    /* 516, 0x204 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_4
    /* 517, 0x205 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_5
    /* 518, 0x206 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_6
    /* 519, 0x207 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_7
    /* 520, 0x208 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_8
    /* 521, 0x209 */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_9
    /* 522, 0x20a */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_STAR
    /* 523, 0x20b */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_POUND
    /* 524, 0x20c */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_A
    /* 525, 0x20d */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_B
    /* 526, 0x20e */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_C
    /* 527, 0x20f */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_D
    /* 528, 0x210 */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_FOCUS
    /* 529, 0x211 */    SDL_SCANCODE_UNKNOWN,            // KEY_WPS_BUTTON
    /* 530, 0x212 */    SDL_SCANCODE_UNKNOWN,            // KEY_TOUCHPAD_TOGGLE
    /* 531, 0x213 */    SDL_SCANCODE_UNKNOWN,            // KEY_TOUCHPAD_ON
    /* 532, 0x214 */    SDL_SCANCODE_UNKNOWN,            // KEY_TOUCHPAD_OFF
    /* 533, 0x215 */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_ZOOMIN
    /* 534, 0x216 */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_ZOOMOUT
    /* 535, 0x217 */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_UP
    /* 536, 0x218 */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_DOWN
    /* 537, 0x219 */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_LEFT
    /* 538, 0x21a */    SDL_SCANCODE_UNKNOWN,            // KEY_CAMERA_RIGHT
    /* 539, 0x21b */    SDL_SCANCODE_UNKNOWN,            // KEY_ATTENDANT_ON
    /* 540, 0x21c */    SDL_SCANCODE_UNKNOWN,            // KEY_ATTENDANT_OFF
    /* 541, 0x21d */    SDL_SCANCODE_UNKNOWN,            // KEY_ATTENDANT_TOGGLE
    /* 542, 0x21e */    SDL_SCANCODE_UNKNOWN,            // KEY_LIGHTS_TOGGLE
    /* 543, 0x21f */    SDL_SCANCODE_UNKNOWN,
    /* 544, 0x220 */    SDL_SCANCODE_UNKNOWN,
    /* 545, 0x221 */    SDL_SCANCODE_UNKNOWN,
    /* 546, 0x222 */    SDL_SCANCODE_UNKNOWN,
    /* 547, 0x223 */    SDL_SCANCODE_UNKNOWN,
    /* 548, 0x224 */    SDL_SCANCODE_UNKNOWN,
    /* 549, 0x225 */    SDL_SCANCODE_UNKNOWN,
    /* 550, 0x226 */    SDL_SCANCODE_UNKNOWN,
    /* 551, 0x227 */    SDL_SCANCODE_UNKNOWN,
    /* 552, 0x228 */    SDL_SCANCODE_UNKNOWN,
    /* 553, 0x229 */    SDL_SCANCODE_UNKNOWN,
    /* 554, 0x22a */    SDL_SCANCODE_UNKNOWN,
    /* 555, 0x22b */    SDL_SCANCODE_UNKNOWN,
    /* 556, 0x22c */    SDL_SCANCODE_UNKNOWN,
    /* 557, 0x22d */    SDL_SCANCODE_UNKNOWN,
    /* 558, 0x22e */    SDL_SCANCODE_UNKNOWN,
    /* 559, 0x22f */    SDL_SCANCODE_UNKNOWN,
    /* 560, 0x230 */    SDL_SCANCODE_UNKNOWN,            // KEY_ALS_TOGGLE
    /* 561, 0x231 */    SDL_SCANCODE_UNKNOWN,            // KEY_ROTATE_LOCK_TOGGLE
    /* 562, 0x232 */    SDL_SCANCODE_UNKNOWN,
    /* 563, 0x233 */    SDL_SCANCODE_UNKNOWN,
    /* 564, 0x234 */    SDL_SCANCODE_UNKNOWN,
    /* 565, 0x235 */    SDL_SCANCODE_UNKNOWN,
    /* 566, 0x236 */    SDL_SCANCODE_UNKNOWN,
    /* 567, 0x237 */    SDL_SCANCODE_UNKNOWN,
    /* 568, 0x238 */    SDL_SCANCODE_UNKNOWN,
    /* 569, 0x239 */    SDL_SCANCODE_UNKNOWN,
    /* 570, 0x23a */    SDL_SCANCODE_UNKNOWN,
    /* 571, 0x23b */    SDL_SCANCODE_UNKNOWN,
    /* 572, 0x23c */    SDL_SCANCODE_UNKNOWN,
    /* 573, 0x23d */    SDL_SCANCODE_UNKNOWN,
    /* 574, 0x23e */    SDL_SCANCODE_UNKNOWN,
    /* 575, 0x23f */    SDL_SCANCODE_UNKNOWN,
    /* 576, 0x240 */    SDL_SCANCODE_UNKNOWN,            // KEY_BUTTONCONFIG
    /* 577, 0x241 */    SDL_SCANCODE_UNKNOWN,            // KEY_TASKMANAGER
    /* 578, 0x242 */    SDL_SCANCODE_UNKNOWN,            // KEY_JOURNAL
    /* 579, 0x243 */    SDL_SCANCODE_UNKNOWN,            // KEY_CONTROLPANEL
    /* 580, 0x244 */    SDL_SCANCODE_UNKNOWN,            // KEY_APPSELECT
    /* 581, 0x245 */    SDL_SCANCODE_UNKNOWN,            // KEY_SCREENSAVER
    /* 582, 0x246 */    SDL_SCANCODE_UNKNOWN,            // KEY_VOICECOMMAND
    /* 583, 0x247 */    SDL_SCANCODE_UNKNOWN,            // KEY_ASSISTANT
    /* 584, 0x248 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBD_LAYOUT_NEXT
    /* 585, 0x249 */    SDL_SCANCODE_UNKNOWN,            // KEY_EMOJI_PICKER
    /* 586, 0x24a */    SDL_SCANCODE_UNKNOWN,            // KEY_DICTATE
    /* 587, 0x24b */    SDL_SCANCODE_UNKNOWN,
    /* 588, 0x24c */    SDL_SCANCODE_UNKNOWN,
    /* 589, 0x24d */    SDL_SCANCODE_UNKNOWN,
    /* 590, 0x24e */    SDL_SCANCODE_UNKNOWN,
    /* 591, 0x24f */    SDL_SCANCODE_UNKNOWN,
    /* 592, 0x250 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRIGHTNESS_MIN
    /* 593, 0x251 */    SDL_SCANCODE_UNKNOWN,            // KEY_BRIGHTNESS_MAX
    /* 594, 0x252 */    SDL_SCANCODE_UNKNOWN,
    /* 595, 0x253 */    SDL_SCANCODE_UNKNOWN,
    /* 596, 0x254 */    SDL_SCANCODE_UNKNOWN,
    /* 597, 0x255 */    SDL_SCANCODE_UNKNOWN,
    /* 598, 0x256 */    SDL_SCANCODE_UNKNOWN,
    /* 599, 0x257 */    SDL_SCANCODE_UNKNOWN,
    /* 600, 0x258 */    SDL_SCANCODE_UNKNOWN,
    /* 601, 0x259 */    SDL_SCANCODE_UNKNOWN,
    /* 602, 0x25a */    SDL_SCANCODE_UNKNOWN,
    /* 603, 0x25b */    SDL_SCANCODE_UNKNOWN,
    /* 604, 0x25c */    SDL_SCANCODE_UNKNOWN,
    /* 605, 0x25d */    SDL_SCANCODE_UNKNOWN,
    /* 606, 0x25e */    SDL_SCANCODE_UNKNOWN,
    /* 607, 0x25f */    SDL_SCANCODE_UNKNOWN,
    /* 608, 0x260 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBDINPUTASSIST_PREV
    /* 609, 0x261 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBDINPUTASSIST_NEXT
    /* 610, 0x262 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBDINPUTASSIST_PREVGROUP
    /* 611, 0x263 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBDINPUTASSIST_NEXTGROUP
    /* 612, 0x264 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBDINPUTASSIST_ACCEPT
    /* 613, 0x265 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBDINPUTASSIST_CANCEL
    /* 614, 0x266 */    SDL_SCANCODE_UNKNOWN,            // KEY_RIGHT_UP
    /* 615, 0x267 */    SDL_SCANCODE_UNKNOWN,            // KEY_RIGHT_DOWN
    /* 616, 0x268 */    SDL_SCANCODE_UNKNOWN,            // KEY_LEFT_UP
    /* 617, 0x269 */    SDL_SCANCODE_UNKNOWN,            // KEY_LEFT_DOWN
    /* 618, 0x26a */    SDL_SCANCODE_UNKNOWN,            // KEY_ROOT_MENU
    /* 619, 0x26b */    SDL_SCANCODE_UNKNOWN,            // KEY_MEDIA_TOP_MENU
    /* 620, 0x26c */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_11
    /* 621, 0x26d */    SDL_SCANCODE_UNKNOWN,            // KEY_NUMERIC_12
    /* 622, 0x26e */    SDL_SCANCODE_UNKNOWN,            // KEY_AUDIO_DESC
    /* 623, 0x26f */    SDL_SCANCODE_UNKNOWN,            // KEY_3D_MODE
    /* 624, 0x270 */    SDL_SCANCODE_UNKNOWN,            // KEY_NEXT_FAVORITE
    /* 625, 0x271 */    SDL_SCANCODE_UNKNOWN,            // KEY_STOP_RECORD
    /* 626, 0x272 */    SDL_SCANCODE_UNKNOWN,            // KEY_PAUSE_RECORD
    /* 627, 0x273 */    SDL_SCANCODE_UNKNOWN,            // KEY_VOD
    /* 628, 0x274 */    SDL_SCANCODE_UNKNOWN,            // KEY_UNMUTE
    /* 629, 0x275 */    SDL_SCANCODE_UNKNOWN,            // KEY_FASTREVERSE
    /* 630, 0x276 */    SDL_SCANCODE_UNKNOWN,            // KEY_SLOWREVERSE
    /* 631, 0x277 */    SDL_SCANCODE_UNKNOWN,            // KEY_DATA
    /* 632, 0x278 */    SDL_SCANCODE_UNKNOWN,            // KEY_ONSCREEN_KEYBOARD
    /* 633, 0x279 */    SDL_SCANCODE_UNKNOWN,            // KEY_PRIVACY_SCREEN_TOGGLE
    /* 634, 0x27a */    SDL_SCANCODE_UNKNOWN,            // KEY_SELECTIVE_SCREENSHOT
    /* 635, 0x27b */    SDL_SCANCODE_UNKNOWN,
    /* 636, 0x27c */    SDL_SCANCODE_UNKNOWN,
    /* 637, 0x27d */    SDL_SCANCODE_UNKNOWN,
    /* 638, 0x27e */    SDL_SCANCODE_UNKNOWN,
    /* 639, 0x27f */    SDL_SCANCODE_UNKNOWN,
    /* 640, 0x280 */    SDL_SCANCODE_UNKNOWN,
    /* 641, 0x281 */    SDL_SCANCODE_UNKNOWN,
    /* 642, 0x282 */    SDL_SCANCODE_UNKNOWN,
    /* 643, 0x283 */    SDL_SCANCODE_UNKNOWN,
    /* 644, 0x284 */    SDL_SCANCODE_UNKNOWN,
    /* 645, 0x285 */    SDL_SCANCODE_UNKNOWN,
    /* 646, 0x286 */    SDL_SCANCODE_UNKNOWN,
    /* 647, 0x287 */    SDL_SCANCODE_UNKNOWN,
    /* 648, 0x288 */    SDL_SCANCODE_UNKNOWN,
    /* 649, 0x289 */    SDL_SCANCODE_UNKNOWN,
    /* 650, 0x28a */    SDL_SCANCODE_UNKNOWN,
    /* 651, 0x28b */    SDL_SCANCODE_UNKNOWN,
    /* 652, 0x28c */    SDL_SCANCODE_UNKNOWN,
    /* 653, 0x28d */    SDL_SCANCODE_UNKNOWN,
    /* 654, 0x28e */    SDL_SCANCODE_UNKNOWN,
    /* 655, 0x28f */    SDL_SCANCODE_UNKNOWN,
    /* 656, 0x290 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO1
    /* 657, 0x291 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO2
    /* 658, 0x292 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO3
    /* 659, 0x293 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO4
    /* 660, 0x294 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO5
    /* 661, 0x295 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO6
    /* 662, 0x296 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO7
    /* 663, 0x297 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO8
    /* 664, 0x298 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO9
    /* 665, 0x299 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO10
    /* 666, 0x29a */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO11
    /* 667, 0x29b */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO12
    /* 668, 0x29c */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO13
    /* 669, 0x29d */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO14
    /* 670, 0x29e */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO15
    /* 671, 0x29f */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO16
    /* 672, 0x2a0 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO17
    /* 673, 0x2a1 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO18
    /* 674, 0x2a2 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO19
    /* 675, 0x2a3 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO20
    /* 676, 0x2a4 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO21
    /* 677, 0x2a5 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO22
    /* 678, 0x2a6 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO23
    /* 679, 0x2a7 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO24
    /* 680, 0x2a8 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO25
    /* 681, 0x2a9 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO26
    /* 682, 0x2aa */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO27
    /* 683, 0x2ab */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO28
    /* 684, 0x2ac */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO29
    /* 685, 0x2ad */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO30
    /* 686, 0x2ae */    SDL_SCANCODE_UNKNOWN,
    /* 687, 0x2af */    SDL_SCANCODE_UNKNOWN,
    /* 688, 0x2b0 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO_RECORD_START
    /* 689, 0x2b1 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO_RECORD_STOP
    /* 690, 0x2b2 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO_PRESET_CYCLE
    /* 691, 0x2b3 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO_PRESET1
    /* 692, 0x2b4 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO_PRESET2
    /* 693, 0x2b5 */    SDL_SCANCODE_UNKNOWN,            // KEY_MACRO_PRESET3
    /* 694, 0x2b6 */    SDL_SCANCODE_UNKNOWN,
    /* 695, 0x2b7 */    SDL_SCANCODE_UNKNOWN,
    /* 696, 0x2b8 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBD_LCD_MENU1
    /* 697, 0x2b9 */    SDL_SCANCODE_UNKNOWN,            // KEY_KBD_LCD_MENU2
    /* 698, 0x2ba */    SDL_SCANCODE_UNKNOWN,            // KEY_KBD_LCD_MENU3
    /* 699, 0x2bb */    SDL_SCANCODE_UNKNOWN,            // KEY_KBD_LCD_MENU4
    /* 700, 0x2bc */    SDL_SCANCODE_UNKNOWN,            // KEY_KBD_LCD_MENU5
    /* 701, 0x2bd */    SDL_SCANCODE_UNKNOWN,
    /* 702, 0x2be */    SDL_SCANCODE_UNKNOWN,
    /* 703, 0x2bf */    SDL_SCANCODE_UNKNOWN,
    /* 704, 0x2c0 */    SDL_SCANCODE_UNKNOWN,
    /* 705, 0x2c1 */    SDL_SCANCODE_UNKNOWN,
    /* 706, 0x2c2 */    SDL_SCANCODE_UNKNOWN,
    /* 707, 0x2c3 */    SDL_SCANCODE_UNKNOWN,
    /* 708, 0x2c4 */    SDL_SCANCODE_UNKNOWN,
    /* 709, 0x2c5 */    SDL_SCANCODE_UNKNOWN,
    /* 710, 0x2c6 */    SDL_SCANCODE_UNKNOWN,
    /* 711, 0x2c7 */    SDL_SCANCODE_UNKNOWN,
    /* 712, 0x2c8 */    SDL_SCANCODE_UNKNOWN,
    /* 713, 0x2c9 */    SDL_SCANCODE_UNKNOWN,
    /* 714, 0x2ca */    SDL_SCANCODE_UNKNOWN,
    /* 715, 0x2cb */    SDL_SCANCODE_UNKNOWN,
    /* 716, 0x2cc */    SDL_SCANCODE_UNKNOWN,
    /* 717, 0x2cd */    SDL_SCANCODE_UNKNOWN,
    /* 718, 0x2ce */    SDL_SCANCODE_UNKNOWN,
    /* 719, 0x2cf */    SDL_SCANCODE_UNKNOWN,
    /* 720, 0x2d0 */    SDL_SCANCODE_UNKNOWN,
    /* 721, 0x2d1 */    SDL_SCANCODE_UNKNOWN,
    /* 722, 0x2d2 */    SDL_SCANCODE_UNKNOWN,
    /* 723, 0x2d3 */    SDL_SCANCODE_UNKNOWN,
    /* 724, 0x2d4 */    SDL_SCANCODE_UNKNOWN,
    /* 725, 0x2d5 */    SDL_SCANCODE_UNKNOWN,
    /* 726, 0x2d6 */    SDL_SCANCODE_UNKNOWN,
    /* 727, 0x2d7 */    SDL_SCANCODE_UNKNOWN,
    /* 728, 0x2d8 */    SDL_SCANCODE_UNKNOWN,
    /* 729, 0x2d9 */    SDL_SCANCODE_UNKNOWN,
    /* 730, 0x2da */    SDL_SCANCODE_UNKNOWN,
    /* 731, 0x2db */    SDL_SCANCODE_UNKNOWN,
    /* 732, 0x2dc */    SDL_SCANCODE_UNKNOWN,
    /* 733, 0x2dd */    SDL_SCANCODE_UNKNOWN,
    /* 734, 0x2de */    SDL_SCANCODE_UNKNOWN,
    /* 735, 0x2df */    SDL_SCANCODE_UNKNOWN,
    /* 736, 0x2e0 */    SDL_SCANCODE_UNKNOWN,
    /* 737, 0x2e1 */    SDL_SCANCODE_UNKNOWN,
    /* 738, 0x2e2 */    SDL_SCANCODE_UNKNOWN,
    /* 739, 0x2e3 */    SDL_SCANCODE_UNKNOWN,
    /* 740, 0x2e4 */    SDL_SCANCODE_UNKNOWN,
    /* 741, 0x2e5 */    SDL_SCANCODE_UNKNOWN,
    /* 742, 0x2e6 */    SDL_SCANCODE_UNKNOWN,
    /* 743, 0x2e7 */    SDL_SCANCODE_UNKNOWN,
    /* 744, 0x2e8 */    SDL_SCANCODE_UNKNOWN,
    /* 745, 0x2e9 */    SDL_SCANCODE_UNKNOWN,
    /* 746, 0x2ea */    SDL_SCANCODE_UNKNOWN,
    /* 747, 0x2eb */    SDL_SCANCODE_UNKNOWN,
    /* 748, 0x2ec */    SDL_SCANCODE_UNKNOWN,
    /* 749, 0x2ed */    SDL_SCANCODE_UNKNOWN,
    /* 750, 0x2ee */    SDL_SCANCODE_UNKNOWN,
    /* 751, 0x2ef */    SDL_SCANCODE_UNKNOWN,
    /* 752, 0x2f0 */    SDL_SCANCODE_UNKNOWN,
    /* 753, 0x2f1 */    SDL_SCANCODE_UNKNOWN,
    /* 754, 0x2f2 */    SDL_SCANCODE_UNKNOWN,
    /* 755, 0x2f3 */    SDL_SCANCODE_UNKNOWN,
    /* 756, 0x2f4 */    SDL_SCANCODE_UNKNOWN,
    /* 757, 0x2f5 */    SDL_SCANCODE_UNKNOWN,
    /* 758, 0x2f6 */    SDL_SCANCODE_UNKNOWN,
    /* 759, 0x2f7 */    SDL_SCANCODE_UNKNOWN,
    /* 760, 0x2f8 */    SDL_SCANCODE_UNKNOWN,
    /* 761, 0x2f9 */    SDL_SCANCODE_UNKNOWN,
    /* 762, 0x2fa */    SDL_SCANCODE_UNKNOWN,
    /* 763, 0x2fb */    SDL_SCANCODE_UNKNOWN,
    /* 764, 0x2fc */    SDL_SCANCODE_UNKNOWN,
    /* 765, 0x2fd */    SDL_SCANCODE_UNKNOWN,
    /* 766, 0x2fe */    SDL_SCANCODE_UNKNOWN,
    /* 767, 0x2ff */    SDL_SCANCODE_UNKNOWN,            // KEY_MAX
#endif // 0
};

#if 0 // A shell script to update the Linux key names in this file
#!/bin/bash

function get_keyname
{
    value=$(echo "$1" | awk '{print $3}')
    grep -F KEY_ /usr/include/linux/input-event-codes.h | while read line; do
        read -ra fields <<<"$line"
        if [ "${fields[2]}" = "$value" ]; then
            echo "${fields[1]}"
            return
        fi
    done
}

grep -F SDL_SCANCODE scancodes_linux.h | while read line; do
    if [ $(echo "$line" | awk '{print NF}') -eq 5 ]; then
        name=$(get_keyname "$line")
        if [ "$name" != "" ]; then
            echo "    $line            /* $name */"
            continue
        fi
    fi
    echo "    $line"
done
#endif // end script

#if 0 // A shell script to get comments from the Linux header for these keys
#!/bin/bash

function get_comment
{
    name=$(echo "$1" | awk '{print $7}')
    if [ "$name" != "" ]; then
        grep -E "$name\s" /usr/include/linux/input-event-codes.h | grep -F "/*" | sed 's,[^/]*/,/,'
    fi
}

grep -F SDL_SCANCODE scancodes_linux.h | while read line; do
    comment=$(get_comment "$line")
    if [ "$comment" != "" ]; then
        echo "    $line $comment"
    fi
done
#endif // end script


/* *INDENT-ON* */ // clang-format on
