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
#include <dev/wscons/wsksymvar.h>
#include <dev/wscons/wsksymdef.h>
#include "SDL_wscons.h"
#include <sys/time.h>
#include <dev/wscons/wsconsio.h>
#include <dev/wscons/wsdisplay_usl_io.h>
#include <termios.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/param.h>
#include <unistd.h>

#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"

#ifdef SDL_PLATFORM_NETBSD
#define KS_GROUP_Ascii    KS_GROUP_Plain
#define KS_Cmd_ScrollBack KS_Cmd_ScrollFastUp
#define KS_Cmd_ScrollFwd  KS_Cmd_ScrollFastDown
#endif

#define RETIFIOCTLERR(x) \
    if ((x) == -1) {     \
        SDL_free(input); \
        input = NULL;    \
        return NULL;     \
    }

typedef struct SDL_WSCONS_mouse_input_data SDL_WSCONS_mouse_input_data;
extern SDL_WSCONS_mouse_input_data *SDL_WSCONS_Init_Mouse(void);
extern void updateMouse(SDL_WSCONS_mouse_input_data *input);
extern void SDL_WSCONS_Quit_Mouse(SDL_WSCONS_mouse_input_data *input);

// Conversion table courtesy of /usr/src/sys/dev/wscons/wskbdutil.c
static const unsigned char latin1_to_upper[256] = {
    // 0  8  1  9  2  a  3  b  4  c  5  d  6  e  7  f
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 3
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 3
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 4
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 4
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 5
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 5
    0x00, 'A', 'B', 'C', 'D', 'E', 'F', 'G',        // 6
    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',         // 6
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',         // 7
    'X', 'Y', 'Z', 0x00, 0x00, 0x00, 0x00, 0x00,    // 7
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 8
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 8
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 9
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 9
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // a
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // a
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // b
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // b
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // c
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // c
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // d
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // d
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, // e
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, // e
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0x00, // f
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0x00, // f
};

// Compose table courtesy of /usr/src/sys/dev/wscons/wskbdutil.c
static struct SDL_wscons_compose_tab_s
{
    keysym_t elem[2];
    keysym_t result;
} compose_tab[] = {
    { { KS_plus, KS_plus }, KS_numbersign },
    { { KS_a, KS_a }, KS_at },
    { { KS_parenleft, KS_parenleft }, KS_bracketleft },
    { { KS_slash, KS_slash }, KS_backslash },
    { { KS_parenright, KS_parenright }, KS_bracketright },
    { { KS_parenleft, KS_minus }, KS_braceleft },
    { { KS_slash, KS_minus }, KS_bar },
    { { KS_parenright, KS_minus }, KS_braceright },
    { { KS_exclam, KS_exclam }, KS_exclamdown },
    { { KS_c, KS_slash }, KS_cent },
    { { KS_l, KS_minus }, KS_sterling },
    { { KS_y, KS_minus }, KS_yen },
    { { KS_s, KS_o }, KS_section },
    { { KS_x, KS_o }, KS_currency },
    { { KS_c, KS_o }, KS_copyright },
    { { KS_less, KS_less }, KS_guillemotleft },
    { { KS_greater, KS_greater }, KS_guillemotright },
    { { KS_question, KS_question }, KS_questiondown },
    { { KS_dead_acute, KS_space }, KS_apostrophe },
    { { KS_dead_grave, KS_space }, KS_grave },
    { { KS_dead_tilde, KS_space }, KS_asciitilde },
    { { KS_dead_circumflex, KS_space }, KS_asciicircum },
    { { KS_dead_diaeresis, KS_space }, KS_quotedbl },
    { { KS_dead_cedilla, KS_space }, KS_comma },
    { { KS_dead_circumflex, KS_A }, KS_Acircumflex },
    { { KS_dead_diaeresis, KS_A }, KS_Adiaeresis },
    { { KS_dead_grave, KS_A }, KS_Agrave },
    { { KS_dead_abovering, KS_A }, KS_Aring },
    { { KS_dead_tilde, KS_A }, KS_Atilde },
    { { KS_dead_cedilla, KS_C }, KS_Ccedilla },
    { { KS_dead_acute, KS_E }, KS_Eacute },
    { { KS_dead_circumflex, KS_E }, KS_Ecircumflex },
    { { KS_dead_diaeresis, KS_E }, KS_Ediaeresis },
    { { KS_dead_grave, KS_E }, KS_Egrave },
    { { KS_dead_acute, KS_I }, KS_Iacute },
    { { KS_dead_circumflex, KS_I }, KS_Icircumflex },
    { { KS_dead_diaeresis, KS_I }, KS_Idiaeresis },
    { { KS_dead_grave, KS_I }, KS_Igrave },
    { { KS_dead_tilde, KS_N }, KS_Ntilde },
    { { KS_dead_acute, KS_O }, KS_Oacute },
    { { KS_dead_circumflex, KS_O }, KS_Ocircumflex },
    { { KS_dead_diaeresis, KS_O }, KS_Odiaeresis },
    { { KS_dead_grave, KS_O }, KS_Ograve },
    { { KS_dead_tilde, KS_O }, KS_Otilde },
    { { KS_dead_acute, KS_U }, KS_Uacute },
    { { KS_dead_circumflex, KS_U }, KS_Ucircumflex },
    { { KS_dead_diaeresis, KS_U }, KS_Udiaeresis },
    { { KS_dead_grave, KS_U }, KS_Ugrave },
    { { KS_dead_acute, KS_Y }, KS_Yacute },
    { { KS_dead_acute, KS_a }, KS_aacute },
    { { KS_dead_circumflex, KS_a }, KS_acircumflex },
    { { KS_dead_diaeresis, KS_a }, KS_adiaeresis },
    { { KS_dead_grave, KS_a }, KS_agrave },
    { { KS_dead_abovering, KS_a }, KS_aring },
    { { KS_dead_tilde, KS_a }, KS_atilde },
    { { KS_dead_cedilla, KS_c }, KS_ccedilla },
    { { KS_dead_acute, KS_e }, KS_eacute },
    { { KS_dead_circumflex, KS_e }, KS_ecircumflex },
    { { KS_dead_diaeresis, KS_e }, KS_ediaeresis },
    { { KS_dead_grave, KS_e }, KS_egrave },
    { { KS_dead_acute, KS_i }, KS_iacute },
    { { KS_dead_circumflex, KS_i }, KS_icircumflex },
    { { KS_dead_diaeresis, KS_i }, KS_idiaeresis },
    { { KS_dead_grave, KS_i }, KS_igrave },
    { { KS_dead_tilde, KS_n }, KS_ntilde },
    { { KS_dead_acute, KS_o }, KS_oacute },
    { { KS_dead_circumflex, KS_o }, KS_ocircumflex },
    { { KS_dead_diaeresis, KS_o }, KS_odiaeresis },
    { { KS_dead_grave, KS_o }, KS_ograve },
    { { KS_dead_tilde, KS_o }, KS_otilde },
    { { KS_dead_acute, KS_u }, KS_uacute },
    { { KS_dead_circumflex, KS_u }, KS_ucircumflex },
    { { KS_dead_diaeresis, KS_u }, KS_udiaeresis },
    { { KS_dead_grave, KS_u }, KS_ugrave },
    { { KS_dead_acute, KS_y }, KS_yacute },
    { { KS_dead_diaeresis, KS_y }, KS_ydiaeresis },
    { { KS_quotedbl, KS_A }, KS_Adiaeresis },
    { { KS_quotedbl, KS_E }, KS_Ediaeresis },
    { { KS_quotedbl, KS_I }, KS_Idiaeresis },
    { { KS_quotedbl, KS_O }, KS_Odiaeresis },
    { { KS_quotedbl, KS_U }, KS_Udiaeresis },
    { { KS_quotedbl, KS_a }, KS_adiaeresis },
    { { KS_quotedbl, KS_e }, KS_ediaeresis },
    { { KS_quotedbl, KS_i }, KS_idiaeresis },
    { { KS_quotedbl, KS_o }, KS_odiaeresis },
    { { KS_quotedbl, KS_u }, KS_udiaeresis },
    { { KS_quotedbl, KS_y }, KS_ydiaeresis },
    { { KS_acute, KS_A }, KS_Aacute },
    { { KS_asciicircum, KS_A }, KS_Acircumflex },
    { { KS_grave, KS_A }, KS_Agrave },
    { { KS_asterisk, KS_A }, KS_Aring },
    { { KS_asciitilde, KS_A }, KS_Atilde },
    { { KS_cedilla, KS_C }, KS_Ccedilla },
    { { KS_acute, KS_E }, KS_Eacute },
    { { KS_asciicircum, KS_E }, KS_Ecircumflex },
    { { KS_grave, KS_E }, KS_Egrave },
    { { KS_acute, KS_I }, KS_Iacute },
    { { KS_asciicircum, KS_I }, KS_Icircumflex },
    { { KS_grave, KS_I }, KS_Igrave },
    { { KS_asciitilde, KS_N }, KS_Ntilde },
    { { KS_acute, KS_O }, KS_Oacute },
    { { KS_asciicircum, KS_O }, KS_Ocircumflex },
    { { KS_grave, KS_O }, KS_Ograve },
    { { KS_asciitilde, KS_O }, KS_Otilde },
    { { KS_acute, KS_U }, KS_Uacute },
    { { KS_asciicircum, KS_U }, KS_Ucircumflex },
    { { KS_grave, KS_U }, KS_Ugrave },
    { { KS_acute, KS_Y }, KS_Yacute },
    { { KS_acute, KS_a }, KS_aacute },
    { { KS_asciicircum, KS_a }, KS_acircumflex },
    { { KS_grave, KS_a }, KS_agrave },
    { { KS_asterisk, KS_a }, KS_aring },
    { { KS_asciitilde, KS_a }, KS_atilde },
    { { KS_cedilla, KS_c }, KS_ccedilla },
    { { KS_acute, KS_e }, KS_eacute },
    { { KS_asciicircum, KS_e }, KS_ecircumflex },
    { { KS_grave, KS_e }, KS_egrave },
    { { KS_acute, KS_i }, KS_iacute },
    { { KS_asciicircum, KS_i }, KS_icircumflex },
    { { KS_grave, KS_i }, KS_igrave },
    { { KS_asciitilde, KS_n }, KS_ntilde },
    { { KS_acute, KS_o }, KS_oacute },
    { { KS_asciicircum, KS_o }, KS_ocircumflex },
    { { KS_grave, KS_o }, KS_ograve },
    { { KS_asciitilde, KS_o }, KS_otilde },
    { { KS_acute, KS_u }, KS_uacute },
    { { KS_asciicircum, KS_u }, KS_ucircumflex },
    { { KS_grave, KS_u }, KS_ugrave },
    { { KS_acute, KS_y }, KS_yacute },
#ifndef SDL_PLATFORM_NETBSD
    { { KS_dead_caron, KS_space }, KS_L2_caron },
    { { KS_dead_caron, KS_S }, KS_L2_Scaron },
    { { KS_dead_caron, KS_Z }, KS_L2_Zcaron },
    { { KS_dead_caron, KS_s }, KS_L2_scaron },
    { { KS_dead_caron, KS_z }, KS_L2_zcaron }
#endif
};

static keysym_t ksym_upcase(keysym_t ksym)
{
    if (ksym >= KS_f1 && ksym <= KS_f20) {
        return KS_F1 - KS_f1 + ksym;
    }

    if (KS_GROUP(ksym) == KS_GROUP_Ascii && ksym <= 0xff && latin1_to_upper[ksym] != 0x00) {
        return latin1_to_upper[ksym];
    }

    return ksym;
}
static struct wscons_keycode_to_SDL
{
    keysym_t sourcekey;
    SDL_Scancode targetKey;
} conversion_table[] = {
    { KS_Menu, SDL_SCANCODE_APPLICATION },
    { KS_Up, SDL_SCANCODE_UP },
    { KS_Down, SDL_SCANCODE_DOWN },
    { KS_Left, SDL_SCANCODE_LEFT },
    { KS_Right, SDL_SCANCODE_RIGHT },
    { KS_Hold_Screen, SDL_SCANCODE_SCROLLLOCK },
    { KS_Num_Lock, SDL_SCANCODE_NUMLOCKCLEAR },
    { KS_Caps_Lock, SDL_SCANCODE_CAPSLOCK },
    { KS_BackSpace, SDL_SCANCODE_BACKSPACE },
    { KS_space, SDL_SCANCODE_SPACE },
    { KS_Delete, SDL_SCANCODE_BACKSPACE },
    { KS_Home, SDL_SCANCODE_HOME },
    { KS_End, SDL_SCANCODE_END },
    { KS_Pause, SDL_SCANCODE_PAUSE },
    { KS_Print_Screen, SDL_SCANCODE_PRINTSCREEN },
    { KS_Insert, SDL_SCANCODE_INSERT },
    { KS_Escape, SDL_SCANCODE_ESCAPE },
    { KS_Return, SDL_SCANCODE_RETURN },
    { KS_Linefeed, SDL_SCANCODE_RETURN },
    { KS_KP_Delete, SDL_SCANCODE_DELETE },
    { KS_KP_Insert, SDL_SCANCODE_INSERT },
    { KS_Control_L, SDL_SCANCODE_LCTRL },
    { KS_Control_R, SDL_SCANCODE_RCTRL },
    { KS_Shift_L, SDL_SCANCODE_LSHIFT },
    { KS_Shift_R, SDL_SCANCODE_RSHIFT },
    { KS_Alt_L, SDL_SCANCODE_LALT },
    { KS_Alt_R, SDL_SCANCODE_RALT },
    { KS_grave, SDL_SCANCODE_GRAVE },

    { KS_KP_0, SDL_SCANCODE_KP_0 },
    { KS_KP_1, SDL_SCANCODE_KP_1 },
    { KS_KP_2, SDL_SCANCODE_KP_2 },
    { KS_KP_3, SDL_SCANCODE_KP_3 },
    { KS_KP_4, SDL_SCANCODE_KP_4 },
    { KS_KP_5, SDL_SCANCODE_KP_5 },
    { KS_KP_6, SDL_SCANCODE_KP_6 },
    { KS_KP_7, SDL_SCANCODE_KP_7 },
    { KS_KP_8, SDL_SCANCODE_KP_8 },
    { KS_KP_9, SDL_SCANCODE_KP_9 },
    { KS_KP_Enter, SDL_SCANCODE_KP_ENTER },
    { KS_KP_Multiply, SDL_SCANCODE_KP_MULTIPLY },
    { KS_KP_Add, SDL_SCANCODE_KP_PLUS },
    { KS_KP_Subtract, SDL_SCANCODE_KP_MINUS },
    { KS_KP_Divide, SDL_SCANCODE_KP_DIVIDE },
    { KS_KP_Up, SDL_SCANCODE_UP },
    { KS_KP_Down, SDL_SCANCODE_DOWN },
    { KS_KP_Left, SDL_SCANCODE_LEFT },
    { KS_KP_Right, SDL_SCANCODE_RIGHT },
    { KS_KP_Equal, SDL_SCANCODE_KP_EQUALS },
    { KS_f1, SDL_SCANCODE_F1 },
    { KS_f2, SDL_SCANCODE_F2 },
    { KS_f3, SDL_SCANCODE_F3 },
    { KS_f4, SDL_SCANCODE_F4 },
    { KS_f5, SDL_SCANCODE_F5 },
    { KS_f6, SDL_SCANCODE_F6 },
    { KS_f7, SDL_SCANCODE_F7 },
    { KS_f8, SDL_SCANCODE_F8 },
    { KS_f9, SDL_SCANCODE_F9 },
    { KS_f10, SDL_SCANCODE_F10 },
    { KS_f11, SDL_SCANCODE_F11 },
    { KS_f12, SDL_SCANCODE_F12 },
    { KS_f13, SDL_SCANCODE_F13 },
    { KS_f14, SDL_SCANCODE_F14 },
    { KS_f15, SDL_SCANCODE_F15 },
    { KS_f16, SDL_SCANCODE_F16 },
    { KS_f17, SDL_SCANCODE_F17 },
    { KS_f18, SDL_SCANCODE_F18 },
    { KS_f19, SDL_SCANCODE_F19 },
    { KS_f20, SDL_SCANCODE_F20 },
#ifndef SDL_PLATFORM_NETBSD
    { KS_f21, SDL_SCANCODE_F21 },
    { KS_f22, SDL_SCANCODE_F22 },
    { KS_f23, SDL_SCANCODE_F23 },
    { KS_f24, SDL_SCANCODE_F24 },
#endif
    { KS_Meta_L, SDL_SCANCODE_LGUI },
    { KS_Meta_R, SDL_SCANCODE_RGUI },
    { KS_Zenkaku_Hankaku, SDL_SCANCODE_LANG5 },
    { KS_Hiragana_Katakana, SDL_SCANCODE_INTERNATIONAL2 },
    { KS_yen, SDL_SCANCODE_INTERNATIONAL3 },
    { KS_Henkan, SDL_SCANCODE_INTERNATIONAL4 },
    { KS_Muhenkan, SDL_SCANCODE_INTERNATIONAL5 },
    { KS_KP_Prior, SDL_SCANCODE_PRIOR },

    { KS_a, SDL_SCANCODE_A },
    { KS_b, SDL_SCANCODE_B },
    { KS_c, SDL_SCANCODE_C },
    { KS_d, SDL_SCANCODE_D },
    { KS_e, SDL_SCANCODE_E },
    { KS_f, SDL_SCANCODE_F },
    { KS_g, SDL_SCANCODE_G },
    { KS_h, SDL_SCANCODE_H },
    { KS_i, SDL_SCANCODE_I },
    { KS_j, SDL_SCANCODE_J },
    { KS_k, SDL_SCANCODE_K },
    { KS_l, SDL_SCANCODE_L },
    { KS_m, SDL_SCANCODE_M },
    { KS_n, SDL_SCANCODE_N },
    { KS_o, SDL_SCANCODE_O },
    { KS_p, SDL_SCANCODE_P },
    { KS_q, SDL_SCANCODE_Q },
    { KS_r, SDL_SCANCODE_R },
    { KS_s, SDL_SCANCODE_S },
    { KS_t, SDL_SCANCODE_T },
    { KS_u, SDL_SCANCODE_U },
    { KS_v, SDL_SCANCODE_V },
    { KS_w, SDL_SCANCODE_W },
    { KS_x, SDL_SCANCODE_X },
    { KS_y, SDL_SCANCODE_Y },
    { KS_z, SDL_SCANCODE_Z },

    { KS_0, SDL_SCANCODE_0 },
    { KS_1, SDL_SCANCODE_1 },
    { KS_2, SDL_SCANCODE_2 },
    { KS_3, SDL_SCANCODE_3 },
    { KS_4, SDL_SCANCODE_4 },
    { KS_5, SDL_SCANCODE_5 },
    { KS_6, SDL_SCANCODE_6 },
    { KS_7, SDL_SCANCODE_7 },
    { KS_8, SDL_SCANCODE_8 },
    { KS_9, SDL_SCANCODE_9 },
    { KS_minus, SDL_SCANCODE_MINUS },
    { KS_equal, SDL_SCANCODE_EQUALS },
    { KS_Tab, SDL_SCANCODE_TAB },
    { KS_KP_Tab, SDL_SCANCODE_KP_TAB },
    { KS_apostrophe, SDL_SCANCODE_APOSTROPHE },
    { KS_bracketleft, SDL_SCANCODE_LEFTBRACKET },
    { KS_bracketright, SDL_SCANCODE_RIGHTBRACKET },
    { KS_semicolon, SDL_SCANCODE_SEMICOLON },
    { KS_comma, SDL_SCANCODE_COMMA },
    { KS_period, SDL_SCANCODE_PERIOD },
    { KS_slash, SDL_SCANCODE_SLASH },
    { KS_backslash, SDL_SCANCODE_BACKSLASH }
};

typedef struct
{
    int fd;
    SDL_KeyboardID keyboardID;
    struct wskbd_map_data keymap;
    int ledstate;
    int origledstate;
    int shiftstate[4];
    int shiftheldstate[8];
    int lockheldstate[5];
    kbd_t encoding;
    char text[128];
    unsigned int text_len;
    keysym_t composebuffer[2];
    unsigned char composelen;
    int type;
} SDL_WSCONS_input_data;

static SDL_WSCONS_input_data *inputs[4] = { NULL, NULL, NULL, NULL };
static SDL_WSCONS_mouse_input_data *mouseInputData = NULL;
#define IS_CONTROL_HELD (input->shiftstate[2] > 0)
#define IS_ALT_HELD     (input->shiftstate[1] > 0)
#define IS_SHIFT_HELD   ((input->shiftstate[0] > 0) || (input->ledstate & (1 << 5)))

#define IS_ALTGR_MODE    ((input->ledstate & (1 << 4)) || (input->shiftstate[3] > 0))
#define IS_NUMLOCK_ON    (input->ledstate & LED_NUM)
#define IS_SCROLLLOCK_ON (input->ledstate & LED_SCR)
#define IS_CAPSLOCK_ON   (input->ledstate & LED_CAP)
static SDL_WSCONS_input_data *SDL_WSCONS_Init_Keyboard(const char *dev)
{
#ifdef WSKBDIO_SETVERSION
    int version = WSKBDIO_EVENT_VERSION;
#endif
    SDL_WSCONS_input_data *input = (SDL_WSCONS_input_data *)SDL_calloc(1, sizeof(SDL_WSCONS_input_data));

    if (!input) {
        return NULL;
    }

    input->fd = open(dev, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (input->fd == -1) {
        SDL_free(input);
        input = NULL;
        return NULL;
    }

    input->keyboardID = SDL_GetNextObjectID();
    SDL_AddKeyboard(input->keyboardID, NULL, false);

    input->keymap.map = SDL_calloc(KS_NUMKEYCODES, sizeof(struct wscons_keymap));
    if (!input->keymap.map) {
        SDL_free(input);
        return NULL;
    }
    input->keymap.maplen = KS_NUMKEYCODES;
    RETIFIOCTLERR(ioctl(input->fd, WSKBDIO_GETMAP, &input->keymap));
    RETIFIOCTLERR(ioctl(input->fd, WSKBDIO_GETLEDS, &input->ledstate));
    input->origledstate = input->ledstate;
    RETIFIOCTLERR(ioctl(input->fd, WSKBDIO_GETENCODING, &input->encoding));
    RETIFIOCTLERR(ioctl(input->fd, WSKBDIO_GTYPE, &input->type));
#ifdef WSKBDIO_SETVERSION
    RETIFIOCTLERR(ioctl(input->fd, WSKBDIO_SETVERSION, &version));
#endif
    return input;
}

void SDL_WSCONS_Init(void)
{
    inputs[0] = SDL_WSCONS_Init_Keyboard("/dev/wskbd0");
    inputs[1] = SDL_WSCONS_Init_Keyboard("/dev/wskbd1");
    inputs[2] = SDL_WSCONS_Init_Keyboard("/dev/wskbd2");
    inputs[3] = SDL_WSCONS_Init_Keyboard("/dev/wskbd3");

    mouseInputData = SDL_WSCONS_Init_Mouse();
    return;
}

void SDL_WSCONS_Quit(void)
{
    int i = 0;
    SDL_WSCONS_input_data *input = NULL;

    SDL_WSCONS_Quit_Mouse(mouseInputData);
    mouseInputData = NULL;
    for (i = 0; i < 4; i++) {
        input = inputs[i];
        if (input) {
            if (input->fd != -1 && input->fd != 0) {
                ioctl(input->fd, WSKBDIO_SETLEDS, &input->origledstate);
                close(input->fd);
                input->fd = -1;
            }
            SDL_free(input);
            input = NULL;
        }
        inputs[i] = NULL;
    }
}

static void put_queue(SDL_WSCONS_input_data *kbd, uint c)
{
    // c is already part of a UTF-8 sequence and safe to add as a character
    if (kbd->text_len < (sizeof(kbd->text) - 1)) {
        kbd->text[kbd->text_len++] = (char)(c);
    }
}

static void put_utf8(SDL_WSCONS_input_data *input, uint c)
{
    if (c < 0x80)
        /*  0******* */
        put_queue(input, c);
    else if (c < 0x800) {
        /* 110***** 10****** */
        put_queue(input, 0xc0 | (c >> 6));
        put_queue(input, 0x80 | (c & 0x3f));
    } else if (c < 0x10000) {
        if (c >= 0xD800 && c <= 0xF500) {
            return;
        }
        if (c == 0xFFFF) {
            return;
        }
        /* 1110**** 10****** 10****** */
        put_queue(input, 0xe0 | (c >> 12));
        put_queue(input, 0x80 | ((c >> 6) & 0x3f));
        put_queue(input, 0x80 | (c & 0x3f));
    } else if (c < 0x110000) {
        /* 11110*** 10****** 10****** 10****** */
        put_queue(input, 0xf0 | (c >> 18));
        put_queue(input, 0x80 | ((c >> 12) & 0x3f));
        put_queue(input, 0x80 | ((c >> 6) & 0x3f));
        put_queue(input, 0x80 | (c & 0x3f));
    }
}

static void Translate_to_text(SDL_WSCONS_input_data *input, keysym_t ksym)
{
    if (KS_GROUP(ksym) == KS_GROUP_Keypad) {
        if (SDL_isprint(ksym & 0xFF)) {
            ksym &= 0xFF;
        }
    }
    switch (ksym) {
    case KS_Escape:
    case KS_Delete:
    case KS_BackSpace:
    case KS_Return:
    case KS_Linefeed:
        // All of these are unprintable characters. Ignore them
        break;
    default:
        put_utf8(input, ksym);
        break;
    }
    if (input->text_len > 0) {
        input->text[input->text_len] = '\0';
        SDL_SendKeyboardText(input->text);
        // SDL_memset(input->text, 0, sizeof(input->text));
        input->text_len = 0;
        input->text[0] = 0;
    }
}

static void Translate_to_keycode(SDL_WSCONS_input_data *input, int type, keysym_t ksym, Uint64 timestamp)
{
    struct wscons_keymap keyDesc = input->keymap.map[ksym];
    keysym_t *group = &keyDesc.group1[KS_GROUP(keyDesc.group1[0]) == KS_GROUP_Keypad && IS_NUMLOCK_ON ? !IS_SHIFT_HELD : 0];
    int i = 0;

    // Check command first, then group[0]
    switch (keyDesc.command) {
    case KS_Cmd_ScrollBack:
    {
        SDL_SendKeyboardKey(timestamp, input->keyboardID, 0, SDL_SCANCODE_PAGEUP, (type == WSCONS_EVENT_KEY_DOWN));
        return;
    }
    case KS_Cmd_ScrollFwd:
    {
        SDL_SendKeyboardKey(timestamp, input->keyboardID, 0, SDL_SCANCODE_PAGEDOWN, (type == WSCONS_EVENT_KEY_DOWN));
        return;
    }
    default:
        break;
    }
    for (i = 0; i < SDL_arraysize(conversion_table); i++) {
        if (conversion_table[i].sourcekey == group[0]) {
            SDL_SendKeyboardKey(timestamp, input->keyboardID, group[0], conversion_table[i].targetKey, (type == WSCONS_EVENT_KEY_DOWN));
            return;
        }
    }
    SDL_SendKeyboardKey(timestamp, input->keyboardID, group[0], SDL_SCANCODE_UNKNOWN, (type == WSCONS_EVENT_KEY_DOWN));
}

static Uint64 GetEventTimestamp(struct timespec *time)
{
    // FIXME: Get the event time in the SDL tick time base
    return SDL_GetTicksNS();
}

static void updateKeyboard(SDL_WSCONS_input_data *input)
{
    struct wscons_event events[64];
    int type;
    int n, i, gindex, acc_i;
    keysym_t *group;
    keysym_t ksym, result;

    if (!input) {
        return;
    }
    if ((n = read(input->fd, events, sizeof(events))) > 0) {
        n /= sizeof(struct wscons_event);
        for (i = 0; i < n; i++) {
            Uint64 timestamp = GetEventTimestamp(&events[i].time);
            type = events[i].type;
            switch (type) {
            case WSCONS_EVENT_KEY_DOWN:
            {
                switch (input->keymap.map[events[i].value].group1[0]) {
                case KS_Hold_Screen:
                {
                    if (input->lockheldstate[0] >= 1) {
                        break;
                    }
                    input->ledstate ^= LED_SCR;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->lockheldstate[0] = 1;
                    break;
                }
                case KS_Num_Lock:
                {
                    if (input->lockheldstate[1] >= 1) {
                        break;
                    }
                    input->ledstate ^= LED_NUM;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->lockheldstate[1] = 1;
                    break;
                }
                case KS_Caps_Lock:
                {
                    if (input->lockheldstate[2] >= 1) {
                        break;
                    }
                    input->ledstate ^= LED_CAP;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->lockheldstate[2] = 1;
                    break;
                }
#ifndef SDL_PLATFORM_NETBSD
                case KS_Mode_Lock:
                {
                    if (input->lockheldstate[3] >= 1) {
                        break;
                    }
                    input->ledstate ^= 1 << 4;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->lockheldstate[3] = 1;
                    break;
                }
#endif
                case KS_Shift_Lock:
                {
                    if (input->lockheldstate[4] >= 1) {
                        break;
                    }
                    input->ledstate ^= 1 << 5;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->lockheldstate[4] = 1;
                    break;
                }
                case KS_Shift_L:
                {
                    if (input->shiftheldstate[0]) {
                        break;
                    }
                    input->shiftstate[0]++;
                    input->shiftheldstate[0] = 1;
                    break;
                }
                case KS_Shift_R:
                {
                    if (input->shiftheldstate[1]) {
                        break;
                    }
                    input->shiftstate[0]++;
                    input->shiftheldstate[1] = 1;
                    break;
                }
                case KS_Alt_L:
                {
                    if (input->shiftheldstate[2]) {
                        break;
                    }
                    input->shiftstate[1]++;
                    input->shiftheldstate[2] = 1;
                    break;
                }
                case KS_Alt_R:
                {
                    if (input->shiftheldstate[3]) {
                        break;
                    }
                    input->shiftstate[1]++;
                    input->shiftheldstate[3] = 1;
                    break;
                }
                case KS_Control_L:
                {
                    if (input->shiftheldstate[4]) {
                        break;
                    }
                    input->shiftstate[2]++;
                    input->shiftheldstate[4] = 1;
                    break;
                }
                case KS_Control_R:
                {
                    if (input->shiftheldstate[5]) {
                        break;
                    }
                    input->shiftstate[2]++;
                    input->shiftheldstate[5] = 1;
                    break;
                }
                case KS_Mode_switch:
                {
                    if (input->shiftheldstate[6]) {
                        break;
                    }
                    input->shiftstate[3]++;
                    input->shiftheldstate[6] = 1;
                    break;
                }
                }
            } break;
            case WSCONS_EVENT_KEY_UP:
            {
                switch (input->keymap.map[events[i].value].group1[0]) {
                case KS_Hold_Screen:
                {
                    if (input->lockheldstate[0]) {
                        input->lockheldstate[0] = 0;
                    }
                } break;
                case KS_Num_Lock:
                {
                    if (input->lockheldstate[1]) {
                        input->lockheldstate[1] = 0;
                    }
                } break;
                case KS_Caps_Lock:
                {
                    if (input->lockheldstate[2]) {
                        input->lockheldstate[2] = 0;
                    }
                } break;
#ifndef SDL_PLATFORM_NETBSD
                case KS_Mode_Lock:
                {
                    if (input->lockheldstate[3]) {
                        input->lockheldstate[3] = 0;
                    }
                } break;
#endif
                case KS_Shift_Lock:
                {
                    if (input->lockheldstate[4]) {
                        input->lockheldstate[4] = 0;
                    }
                } break;
                case KS_Shift_L:
                {
                    input->shiftheldstate[0] = 0;
                    if (input->shiftstate[0]) {
                        input->shiftstate[0]--;
                    }
                    break;
                }
                case KS_Shift_R:
                {
                    input->shiftheldstate[1] = 0;
                    if (input->shiftstate[0]) {
                        input->shiftstate[0]--;
                    }
                    break;
                }
                case KS_Alt_L:
                {
                    input->shiftheldstate[2] = 0;
                    if (input->shiftstate[1]) {
                        input->shiftstate[1]--;
                    }
                    break;
                }
                case KS_Alt_R:
                {
                    input->shiftheldstate[3] = 0;
                    if (input->shiftstate[1]) {
                        input->shiftstate[1]--;
                    }
                    break;
                }
                case KS_Control_L:
                {
                    input->shiftheldstate[4] = 0;
                    if (input->shiftstate[2]) {
                        input->shiftstate[2]--;
                    }
                    break;
                }
                case KS_Control_R:
                {
                    input->shiftheldstate[5] = 0;
                    if (input->shiftstate[2]) {
                        input->shiftstate[2]--;
                    }
                    break;
                }
                case KS_Mode_switch:
                {
                    input->shiftheldstate[6] = 0;
                    if (input->shiftstate[3]) {
                        input->shiftstate[3]--;
                    }
                    break;
                }
                }
            } break;
            case WSCONS_EVENT_ALL_KEYS_UP:
                for (i = 0; i < SDL_SCANCODE_COUNT; i++) {
                    SDL_SendKeyboardKey(timestamp, input->keyboardID, 0, (SDL_Scancode)i, false);
                }
                break;
            default:
                break;
            }

            if (input->type == WSKBD_TYPE_USB && events[i].value <= 0xE7) {
                SDL_SendKeyboardKey(timestamp, input->keyboardID, 0, (SDL_Scancode)events[i].value, (type == WSCONS_EVENT_KEY_DOWN));
            } else {
                Translate_to_keycode(input, type, events[i].value, timestamp);
            }

            if (type == WSCONS_EVENT_KEY_UP) {
                continue;
            }

            if (IS_ALTGR_MODE && !IS_CONTROL_HELD)
                group = &input->keymap.map[events[i].value].group2[0];
            else
                group = &input->keymap.map[events[i].value].group1[0];

            if (IS_NUMLOCK_ON && KS_GROUP(group[1]) == KS_GROUP_Keypad) {
                gindex = !IS_SHIFT_HELD;
                ksym = group[gindex];
            } else {
                if (IS_CAPSLOCK_ON && !IS_SHIFT_HELD) {
                    gindex = 0;
                    ksym = ksym_upcase(group[0]);
                } else {
                    gindex = IS_SHIFT_HELD;
                    ksym = group[gindex];
                }
            }
            result = KS_voidSymbol;

            switch (KS_GROUP(ksym)) {
            case KS_GROUP_Ascii:
            case KS_GROUP_Keypad:
            case KS_GROUP_Function:
                result = ksym;
                break;
            case KS_GROUP_Mod:
                if (ksym == KS_Multi_key) {
                    input->ledstate |= WSKBD_LED_COMPOSE;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->composelen = 2;
                    input->composebuffer[0] = input->composebuffer[1] = 0;
                }
                break;
            case KS_GROUP_Dead:
                if (input->composelen == 0) {
                    input->ledstate |= WSKBD_LED_COMPOSE;
                    ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                    input->composelen = 1;
                    input->composebuffer[0] = ksym;
                    input->composebuffer[1] = 0;
                } else
                    result = ksym;
                break;
            }
            if (result == KS_voidSymbol) {
                continue;
            }

            if (input->composelen > 0) {
                if (input->composelen == 2 && group == &input->keymap.map[events[i].value].group2[0]) {
                    if (input->keymap.map[events[i].value].group2[gindex] == input->keymap.map[events[i].value].group1[gindex]) {
                        input->composelen = 0;
                        input->composebuffer[0] = input->composebuffer[1] = 0;
                    }
                }

                if (input->composelen != 0) {
                    input->composebuffer[2 - input->composelen] = result;
                    if (--input->composelen == 0) {
                        result = KS_voidSymbol;
                        input->ledstate &= ~WSKBD_LED_COMPOSE;
                        ioctl(input->fd, WSKBDIO_SETLEDS, &input->ledstate);
                        for (acc_i = 0; acc_i < SDL_arraysize(compose_tab); acc_i++) {
                            if ((compose_tab[acc_i].elem[0] == input->composebuffer[0] && compose_tab[acc_i].elem[1] == input->composebuffer[1]) || (compose_tab[acc_i].elem[0] == input->composebuffer[1] && compose_tab[acc_i].elem[1] == input->composebuffer[0])) {
                                result = compose_tab[acc_i].result;
                                break;
                            }
                        }
                    } else
                        continue;
                }
            }

            if (KS_GROUP(result) == KS_GROUP_Ascii) {
                if (IS_CONTROL_HELD) {
                    if ((result >= KS_at && result <= KS_z) || result == KS_space) {
                        result = result & 0x1f;
                    } else if (result == KS_2) {
                        result = 0x00;
                    } else if (result >= KS_3 && result <= KS_7) {
                        result = KS_Escape + (result - KS_3);
                    } else if (result == KS_8) {
                        result = KS_Delete;
                    }
                }
                if (IS_ALT_HELD) {
                    if (input->encoding & KB_METAESC) {
                        Translate_to_keycode(input, WSCONS_EVENT_KEY_DOWN, KS_Escape, 0);
                        Translate_to_text(input, result);
                        continue;
                    } else {
                        result |= 0x80;
                    }
                }
            }
            Translate_to_text(input, result);
            continue;
        }
    }
}

void SDL_WSCONS_PumpEvents(void)
{
    int i = 0;
    for (i = 0; i < 4; i++) {
        updateKeyboard(inputs[i]);
    }
    if (mouseInputData) {
        updateMouse(mouseInputData);
    }
}
