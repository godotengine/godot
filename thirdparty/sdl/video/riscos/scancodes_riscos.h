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

/* RISC OS key code to SDL_Keycode mapping table
   Sources:
   - https://www.riscosopen.org/wiki/documentation/show/Keyboard Scan Codes
*/
/* *INDENT-OFF* */ // clang-format off
static SDL_Scancode const riscos_scancode_table[] = {
     /*   0 */   SDL_SCANCODE_UNKNOWN,   // Shift
     /*   1 */   SDL_SCANCODE_UNKNOWN,   // Ctrl
     /*   2 */   SDL_SCANCODE_UNKNOWN,   // Alt
     /*   3 */   SDL_SCANCODE_LSHIFT,
     /*   4 */   SDL_SCANCODE_LCTRL,
     /*   5 */   SDL_SCANCODE_LALT,
     /*   6 */   SDL_SCANCODE_RSHIFT,
     /*   7 */   SDL_SCANCODE_RCTRL,
     /*   8 */   SDL_SCANCODE_RALT,
     /*   9 */   SDL_SCANCODE_UNKNOWN,   // Left mouse
     /*  10 */   SDL_SCANCODE_UNKNOWN,   // Center mouse
     /*  11 */   SDL_SCANCODE_UNKNOWN,   // Right mouse
     /*  12 */   SDL_SCANCODE_UNKNOWN,
     /*  13 */   SDL_SCANCODE_UNKNOWN,
     /*  14 */   SDL_SCANCODE_UNKNOWN,
     /*  15 */   SDL_SCANCODE_UNKNOWN,
     /*  16 */   SDL_SCANCODE_Q,
     /*  17 */   SDL_SCANCODE_3,
     /*  18 */   SDL_SCANCODE_4,
     /*  19 */   SDL_SCANCODE_5,
     /*  20 */   SDL_SCANCODE_F4,
     /*  21 */   SDL_SCANCODE_8,
     /*  22 */   SDL_SCANCODE_F7,
     /*  23 */   SDL_SCANCODE_MINUS,
     /*  24 */   SDL_SCANCODE_6,         // Duplicate of 52
     /*  25 */   SDL_SCANCODE_LEFT,
     /*  26 */   SDL_SCANCODE_KP_6,
     /*  27 */   SDL_SCANCODE_KP_7,
     /*  28 */   SDL_SCANCODE_F11,
     /*  29 */   SDL_SCANCODE_F12,
     /*  30 */   SDL_SCANCODE_F10,
     /*  31 */   SDL_SCANCODE_SCROLLLOCK,
     /*  32 */   SDL_SCANCODE_PRINTSCREEN,
     /*  33 */   SDL_SCANCODE_W,
     /*  34 */   SDL_SCANCODE_E,
     /*  35 */   SDL_SCANCODE_T,
     /*  36 */   SDL_SCANCODE_7,
     /*  37 */   SDL_SCANCODE_I,
     /*  38 */   SDL_SCANCODE_9,
     /*  39 */   SDL_SCANCODE_0,
     /*  40 */   SDL_SCANCODE_MINUS,     // Duplicate of 23
     /*  41 */   SDL_SCANCODE_DOWN,
     /*  42 */   SDL_SCANCODE_KP_8,
     /*  43 */   SDL_SCANCODE_KP_9,
     /*  44 */   SDL_SCANCODE_PAUSE,
     /*  45 */   SDL_SCANCODE_GRAVE,
     /*  46 */   SDL_SCANCODE_CURRENCYUNIT,
     /*  47 */   SDL_SCANCODE_BACKSPACE,
     /*  48 */   SDL_SCANCODE_1,
     /*  49 */   SDL_SCANCODE_2,
     /*  50 */   SDL_SCANCODE_D,
     /*  51 */   SDL_SCANCODE_R,
     /*  52 */   SDL_SCANCODE_6,
     /*  53 */   SDL_SCANCODE_U,
     /*  54 */   SDL_SCANCODE_O,
     /*  55 */   SDL_SCANCODE_P,
     /*  56 */   SDL_SCANCODE_LEFTBRACKET,
     /*  57 */   SDL_SCANCODE_UP,
     /*  58 */   SDL_SCANCODE_KP_PLUS,
     /*  59 */   SDL_SCANCODE_KP_MINUS,
     /*  60 */   SDL_SCANCODE_KP_ENTER,
     /*  61 */   SDL_SCANCODE_INSERT,
     /*  62 */   SDL_SCANCODE_HOME,
     /*  63 */   SDL_SCANCODE_PAGEUP,
     /*  64 */   SDL_SCANCODE_CAPSLOCK,
     /*  65 */   SDL_SCANCODE_A,
     /*  66 */   SDL_SCANCODE_X,
     /*  67 */   SDL_SCANCODE_F,
     /*  68 */   SDL_SCANCODE_Y,
     /*  69 */   SDL_SCANCODE_J,
     /*  70 */   SDL_SCANCODE_K,
     /*  71 */   SDL_SCANCODE_2,         // Duplicate of 49
     /*  72 */   SDL_SCANCODE_SEMICOLON, // Duplicate of 87
     /*  73 */   SDL_SCANCODE_RETURN,
     /*  74 */   SDL_SCANCODE_KP_DIVIDE,
     /*  75 */   SDL_SCANCODE_UNKNOWN,
     /*  76 */   SDL_SCANCODE_KP_PERIOD,
     /*  77 */   SDL_SCANCODE_NUMLOCKCLEAR,
     /*  78 */   SDL_SCANCODE_PAGEDOWN,
     /*  79 */   SDL_SCANCODE_APOSTROPHE,
     /*  80 */   SDL_SCANCODE_UNKNOWN,
     /*  81 */   SDL_SCANCODE_S,
     /*  82 */   SDL_SCANCODE_C,
     /*  83 */   SDL_SCANCODE_G,
     /*  84 */   SDL_SCANCODE_H,
     /*  85 */   SDL_SCANCODE_N,
     /*  86 */   SDL_SCANCODE_L,
     /*  87 */   SDL_SCANCODE_SEMICOLON,
     /*  88 */   SDL_SCANCODE_RIGHTBRACKET,
     /*  89 */   SDL_SCANCODE_DELETE,
     /*  90 */   SDL_SCANCODE_KP_HASH,
     /*  91 */   SDL_SCANCODE_KP_MULTIPLY,
     /*  92 */   SDL_SCANCODE_UNKNOWN,
     /*  93 */   SDL_SCANCODE_EQUALS,
     /*  94 */   SDL_SCANCODE_NONUSBACKSLASH,
     /*  95 */   SDL_SCANCODE_UNKNOWN,
     /*  96 */   SDL_SCANCODE_TAB,
     /*  97 */   SDL_SCANCODE_Z,
     /*  98 */   SDL_SCANCODE_SPACE,
     /*  99 */   SDL_SCANCODE_V,
     /* 100 */   SDL_SCANCODE_B,
     /* 101 */   SDL_SCANCODE_M,
     /* 102 */   SDL_SCANCODE_COMMA,
     /* 103 */   SDL_SCANCODE_PERIOD,
     /* 104 */   SDL_SCANCODE_SLASH,
     /* 105 */   SDL_SCANCODE_END,
     /* 106 */   SDL_SCANCODE_KP_0,
     /* 107 */   SDL_SCANCODE_KP_1,
     /* 108 */   SDL_SCANCODE_KP_3,
     /* 109 */   SDL_SCANCODE_UNKNOWN,
     /* 110 */   SDL_SCANCODE_UNKNOWN,
     /* 111 */   SDL_SCANCODE_UNKNOWN,
     /* 112 */   SDL_SCANCODE_ESCAPE,
     /* 113 */   SDL_SCANCODE_F1,
     /* 114 */   SDL_SCANCODE_F2,
     /* 115 */   SDL_SCANCODE_F3,
     /* 116 */   SDL_SCANCODE_F5,
     /* 117 */   SDL_SCANCODE_F6,
     /* 118 */   SDL_SCANCODE_F8,
     /* 119 */   SDL_SCANCODE_F9,
     /* 120 */   SDL_SCANCODE_BACKSLASH,
     /* 121 */   SDL_SCANCODE_RIGHT,
     /* 122 */   SDL_SCANCODE_KP_4,
     /* 123 */   SDL_SCANCODE_KP_5,
     /* 124 */   SDL_SCANCODE_KP_2,
     /* 125 */   SDL_SCANCODE_LGUI,
     /* 126 */   SDL_SCANCODE_RGUI,
     /* 127 */   SDL_SCANCODE_MENU
};
/* *INDENT-ON* */ // clang-format on
