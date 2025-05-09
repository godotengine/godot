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

/*
 *  Font related functions of SDL test framework.
 *
 *  This code is a part of the SDL test library, not the main SDL library.
 */

#ifndef SDL_test_font_h_
#define SDL_test_font_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* Function prototypes */

extern int FONT_CHARACTER_SIZE;

#define FONT_LINE_HEIGHT    (FONT_CHARACTER_SIZE + 2)

/*
 *  Draw a string in the currently set font.
 *
 *  \param renderer The renderer to draw on.
 *  \param x The X coordinate of the upper left corner of the character.
 *  \param y The Y coordinate of the upper left corner of the character.
 *  \param c The character to draw.
 *
 *  \returns true on success, false on failure.
 */
bool SDLCALL SDLTest_DrawCharacter(SDL_Renderer *renderer, float x, float y, Uint32 c);

/*
 *  Draw a UTF-8 string in the currently set font.
 *
 *  The font currently only supports characters in the Basic Latin and Latin-1 Supplement sets.
 *
 *  \param renderer The renderer to draw on.
 *  \param x The X coordinate of the upper left corner of the string.
 *  \param y The Y coordinate of the upper left corner of the string.
 *  \param s The string to draw.
 *
 *  \returns true on success, false on failure.
 */
bool SDLCALL SDLTest_DrawString(SDL_Renderer *renderer, float x, float y, const char *s);

/*
 *  Data used for multi-line text output
 */
typedef struct SDLTest_TextWindow
{
    SDL_FRect rect;
    int current;
    int numlines;
    char **lines;
} SDLTest_TextWindow;

/*
 *  Create a multi-line text output window
 *
 *  \param x The X coordinate of the upper left corner of the window.
 *  \param y The Y coordinate of the upper left corner of the window.
 *  \param w The width of the window (currently ignored)
 *  \param h The height of the window (currently ignored)
 *
 *  \returns the new window, or NULL on failure.
 *
 *  \since This function is available since SDL 3.2.0.
 */
SDLTest_TextWindow * SDLCALL SDLTest_TextWindowCreate(float x, float y, float w, float h);

/*
 *  Display a multi-line text output window
 *
 *  This function should be called every frame to display the text
 *
 *  \param textwin The text output window
 *  \param renderer The renderer to use for display
 *
 *  \since This function is available since SDL 3.2.0.
 */
void SDLCALL SDLTest_TextWindowDisplay(SDLTest_TextWindow *textwin, SDL_Renderer *renderer);

/*
 *  Add text to a multi-line text output window
 *
 *  Adds UTF-8 text to the end of the current text. The newline character starts a
 *  new line of text. The backspace character deletes the last character or, if the
 *  line is empty, deletes the line and goes to the end of the previous line.
 *
 *  \param textwin The text output window
 *  \param fmt A printf() style format string
 *  \param ...  additional parameters matching % tokens in the `fmt` string, if any
 *
 *  \since This function is available since SDL 3.2.0.
 */
void SDLCALL SDLTest_TextWindowAddText(SDLTest_TextWindow *textwin, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) SDL_PRINTF_VARARG_FUNC(2);

/*
 *  Add text to a multi-line text output window
 *
 *  Adds UTF-8 text to the end of the current text. The newline character starts a
 *  new line of text. The backspace character deletes the last character or, if the
 *  line is empty, deletes the line and goes to the end of the previous line.
 *
 *  \param textwin The text output window
 *  \param text The text to add to the window
 *  \param len The length, in bytes, of the text to add to the window
 *
 *  \since This function is available since SDL 3.2.0.
 */
void SDLCALL SDLTest_TextWindowAddTextWithLength(SDLTest_TextWindow *textwin, const char *text, size_t len);

/*
 *  Clear the text in a multi-line text output window
 *
 *  \param textwin The text output window
 *
 *  \since This function is available since SDL 3.2.0.
 */
void SDLCALL SDLTest_TextWindowClear(SDLTest_TextWindow *textwin);

/*
 *  Free the storage associated with a multi-line text output window
 *
 *  \param textwin The text output window
 *
 *  \since This function is available since SDL 3.2.0.
 */
void SDLCALL SDLTest_TextWindowDestroy(SDLTest_TextWindow *textwin);

/*
 *  Cleanup textures used by font drawing functions.
 */
void SDLCALL SDLTest_CleanupTextDrawing(void);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_test_font_h_ */
