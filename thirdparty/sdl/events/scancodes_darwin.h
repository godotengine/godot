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

/* Mac virtual key code to SDL scancode mapping table
   Sources:
   - Inside Macintosh: Text <http://developer.apple.com/documentation/mac/Text/Text-571.html>
   - Apple USB keyboard driver source <http://darwinsource.opendarwin.org/10.4.6.ppc/IOHIDFamily-172.8/IOHIDFamily/Cosmo_USB2ADB.c>
   - experimentation on various ADB and USB ISO keyboards and one ADB ANSI keyboard
*/
/* *INDENT-OFF* */ // clang-format off
static const SDL_Scancode darwin_scancode_table[] = {
    /*   0 */   SDL_SCANCODE_A,
    /*   1 */   SDL_SCANCODE_S,
    /*   2 */   SDL_SCANCODE_D,
    /*   3 */   SDL_SCANCODE_F,
    /*   4 */   SDL_SCANCODE_H,
    /*   5 */   SDL_SCANCODE_G,
    /*   6 */   SDL_SCANCODE_Z,
    /*   7 */   SDL_SCANCODE_X,
    /*   8 */   SDL_SCANCODE_C,
    /*   9 */   SDL_SCANCODE_V,
    /*  10 */   SDL_SCANCODE_NONUSBACKSLASH, // SDL_SCANCODE_NONUSBACKSLASH on ANSI and JIS keyboards (if this key would exist there), SDL_SCANCODE_GRAVE on ISO. (The USB keyboard driver actually translates these usage codes to different virtual key codes depending on whether the keyboard is ISO/ANSI/JIS. That's why you have to help it identify the keyboard type when you plug in a PC USB keyboard. It's a historical thing - ADB keyboards are wired this way.)
    /*  11 */   SDL_SCANCODE_B,
    /*  12 */   SDL_SCANCODE_Q,
    /*  13 */   SDL_SCANCODE_W,
    /*  14 */   SDL_SCANCODE_E,
    /*  15 */   SDL_SCANCODE_R,
    /*  16 */   SDL_SCANCODE_Y,
    /*  17 */   SDL_SCANCODE_T,
    /*  18 */   SDL_SCANCODE_1,
    /*  19 */   SDL_SCANCODE_2,
    /*  20 */   SDL_SCANCODE_3,
    /*  21 */   SDL_SCANCODE_4,
    /*  22 */   SDL_SCANCODE_6,
    /*  23 */   SDL_SCANCODE_5,
    /*  24 */   SDL_SCANCODE_EQUALS,
    /*  25 */   SDL_SCANCODE_9,
    /*  26 */   SDL_SCANCODE_7,
    /*  27 */   SDL_SCANCODE_MINUS,
    /*  28 */   SDL_SCANCODE_8,
    /*  29 */   SDL_SCANCODE_0,
    /*  30 */   SDL_SCANCODE_RIGHTBRACKET,
    /*  31 */   SDL_SCANCODE_O,
    /*  32 */   SDL_SCANCODE_U,
    /*  33 */   SDL_SCANCODE_LEFTBRACKET,
    /*  34 */   SDL_SCANCODE_I,
    /*  35 */   SDL_SCANCODE_P,
    /*  36 */   SDL_SCANCODE_RETURN,
    /*  37 */   SDL_SCANCODE_L,
    /*  38 */   SDL_SCANCODE_J,
    /*  39 */   SDL_SCANCODE_APOSTROPHE,
    /*  40 */   SDL_SCANCODE_K,
    /*  41 */   SDL_SCANCODE_SEMICOLON,
    /*  42 */   SDL_SCANCODE_BACKSLASH,
    /*  43 */   SDL_SCANCODE_COMMA,
    /*  44 */   SDL_SCANCODE_SLASH,
    /*  45 */   SDL_SCANCODE_N,
    /*  46 */   SDL_SCANCODE_M,
    /*  47 */   SDL_SCANCODE_PERIOD,
    /*  48 */   SDL_SCANCODE_TAB,
    /*  49 */   SDL_SCANCODE_SPACE,
    /*  50 */   SDL_SCANCODE_GRAVE, // SDL_SCANCODE_GRAVE on ANSI and JIS keyboards, SDL_SCANCODE_NONUSBACKSLASH on ISO (see comment about virtual key code 10 above)
    /*  51 */   SDL_SCANCODE_BACKSPACE,
    /*  52 */   SDL_SCANCODE_KP_ENTER, // keyboard enter on portables
    /*  53 */   SDL_SCANCODE_ESCAPE,
    /*  54 */   SDL_SCANCODE_RGUI,
    /*  55 */   SDL_SCANCODE_LGUI,
    /*  56 */   SDL_SCANCODE_LSHIFT,
    /*  57 */   SDL_SCANCODE_CAPSLOCK,
    /*  58 */   SDL_SCANCODE_LALT,
    /*  59 */   SDL_SCANCODE_LCTRL,
    /*  60 */   SDL_SCANCODE_RSHIFT,
    /*  61 */   SDL_SCANCODE_RALT,
    /*  62 */   SDL_SCANCODE_RCTRL,
    /*  63 */   SDL_SCANCODE_RGUI, // fn on portables, acts as a hardware-level modifier already, so we don't generate events for it, also XK_Meta_R
    /*  64 */   SDL_SCANCODE_F17,
    /*  65 */   SDL_SCANCODE_KP_PERIOD,
    /*  66 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /*  67 */   SDL_SCANCODE_KP_MULTIPLY,
    /*  68 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /*  69 */   SDL_SCANCODE_KP_PLUS,
    /*  70 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /*  71 */   SDL_SCANCODE_NUMLOCKCLEAR,
    /*  72 */   SDL_SCANCODE_VOLUMEUP,
    /*  73 */   SDL_SCANCODE_VOLUMEDOWN,
    /*  74 */   SDL_SCANCODE_MUTE,
    /*  75 */   SDL_SCANCODE_KP_DIVIDE,
    /*  76 */   SDL_SCANCODE_KP_ENTER, // keypad enter on external keyboards, fn-return on portables
    /*  77 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /*  78 */   SDL_SCANCODE_KP_MINUS,
    /*  79 */   SDL_SCANCODE_F18,
    /*  80 */   SDL_SCANCODE_F19,
    /*  81 */   SDL_SCANCODE_KP_EQUALS,
    /*  82 */   SDL_SCANCODE_KP_0,
    /*  83 */   SDL_SCANCODE_KP_1,
    /*  84 */   SDL_SCANCODE_KP_2,
    /*  85 */   SDL_SCANCODE_KP_3,
    /*  86 */   SDL_SCANCODE_KP_4,
    /*  87 */   SDL_SCANCODE_KP_5,
    /*  88 */   SDL_SCANCODE_KP_6,
    /*  89 */   SDL_SCANCODE_KP_7,
    /*  90 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /*  91 */   SDL_SCANCODE_KP_8,
    /*  92 */   SDL_SCANCODE_KP_9,
    /*  93 */   SDL_SCANCODE_INTERNATIONAL3, // Cosmo_USB2ADB.c says "Yen (JIS)"
    /*  94 */   SDL_SCANCODE_INTERNATIONAL1, // Cosmo_USB2ADB.c says "Ro (JIS)"
    /*  95 */   SDL_SCANCODE_KP_COMMA, // Cosmo_USB2ADB.c says ", JIS only"
    /*  96 */   SDL_SCANCODE_F5,
    /*  97 */   SDL_SCANCODE_F6,
    /*  98 */   SDL_SCANCODE_F7,
    /*  99 */   SDL_SCANCODE_F3,
    /* 100 */   SDL_SCANCODE_F8,
    /* 101 */   SDL_SCANCODE_F9,
    /* 102 */   SDL_SCANCODE_LANG2, // Cosmo_USB2ADB.c says "Eisu"
    /* 103 */   SDL_SCANCODE_F11,
    /* 104 */   SDL_SCANCODE_LANG1, // Cosmo_USB2ADB.c says "Kana"
    /* 105 */   SDL_SCANCODE_PRINTSCREEN, // On ADB keyboards, this key is labeled "F13/print screen". Problem: USB has different usage codes for these two functions. On Apple USB keyboards, the key is labeled "F13" and sends the F13 usage code (SDL_SCANCODE_F13). I decided to use SDL_SCANCODE_PRINTSCREEN here nevertheless since SDL applications are more likely to assume the presence of a print screen key than an F13 key.
    /* 106 */   SDL_SCANCODE_F16,
    /* 107 */   SDL_SCANCODE_SCROLLLOCK, // F14/scroll lock, see comment about F13/print screen above
    /* 108 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /* 109 */   SDL_SCANCODE_F10,
    /* 110 */   SDL_SCANCODE_APPLICATION, // windows contextual menu key, fn-enter on portables
    /* 111 */   SDL_SCANCODE_F12,
    /* 112 */   SDL_SCANCODE_UNKNOWN, // unknown (unused?)
    /* 113 */   SDL_SCANCODE_PAUSE, // F15/pause, see comment about F13/print screen above
    /* 114 */   SDL_SCANCODE_INSERT, // the key is actually labeled "help" on Apple keyboards, and works as such in Mac OS, but it sends the "insert" usage code even on Apple USB keyboards
    /* 115 */   SDL_SCANCODE_HOME,
    /* 116 */   SDL_SCANCODE_PAGEUP,
    /* 117 */   SDL_SCANCODE_DELETE,
    /* 118 */   SDL_SCANCODE_F4,
    /* 119 */   SDL_SCANCODE_END,
    /* 120 */   SDL_SCANCODE_F2,
    /* 121 */   SDL_SCANCODE_PAGEDOWN,
    /* 122 */   SDL_SCANCODE_F1,
    /* 123 */   SDL_SCANCODE_LEFT,
    /* 124 */   SDL_SCANCODE_RIGHT,
    /* 125 */   SDL_SCANCODE_DOWN,
    /* 126 */   SDL_SCANCODE_UP,
    /* 127 */   SDL_SCANCODE_POWER
};
/* *INDENT-ON* */ // clang-format on
