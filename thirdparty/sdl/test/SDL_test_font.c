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
#include <SDL3/SDL_test.h>

#define UTF8_IsTrailingByte(c) ((c) >= 0x80 && (c) <= 0xBF)

int FONT_CHARACTER_SIZE = SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE;

bool SDLTest_DrawCharacter(SDL_Renderer *renderer, float x, float y, Uint32 c)
{
    char str[5];
    char *ptr = SDL_UCS4ToUTF8(c, str);
    *ptr = '\0';
    return SDL_RenderDebugText(renderer, x, y, str);
}

bool SDLTest_DrawString(SDL_Renderer *renderer, float x, float y, const char *s)
{
    return SDL_RenderDebugText(renderer, x, y, s);
}

SDLTest_TextWindow *SDLTest_TextWindowCreate(float x, float y, float w, float h)
{
    SDLTest_TextWindow *textwin = (SDLTest_TextWindow *)SDL_malloc(sizeof(*textwin));

    if (!textwin) {
        return NULL;
    }

    textwin->rect.x = x;
    textwin->rect.y = y;
    textwin->rect.w = w;
    textwin->rect.h = h;
    textwin->current = 0;
    textwin->numlines = (int)SDL_ceilf(h / FONT_LINE_HEIGHT);
    textwin->lines = (char **)SDL_calloc(textwin->numlines, sizeof(*textwin->lines));
    if (!textwin->lines) {
        SDL_free(textwin);
        return NULL;
    }
    return textwin;
}

void SDLTest_TextWindowDisplay(SDLTest_TextWindow *textwin, SDL_Renderer *renderer)
{
    int i;
    float y;

    for (y = textwin->rect.y, i = 0; i < textwin->numlines; ++i, y += FONT_LINE_HEIGHT) {
        if (textwin->lines[i]) {
            SDLTest_DrawString(renderer, textwin->rect.x, y, textwin->lines[i]);
        }
    }
}

void SDLTest_TextWindowAddText(SDLTest_TextWindow *textwin, const char *fmt, ...)
{
    char text[1024];
    va_list ap;

    va_start(ap, fmt);
    (void)SDL_vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);

    SDLTest_TextWindowAddTextWithLength(textwin, text, SDL_strlen(text));
}

void SDLTest_TextWindowAddTextWithLength(SDLTest_TextWindow *textwin, const char *text, size_t len)
{
    size_t existing;
    bool newline = false;
    char *line;

    if (len > 0 && text[len - 1] == '\n') {
        --len;
        newline = true;
    }

    if (textwin->lines[textwin->current]) {
        existing = SDL_strlen(textwin->lines[textwin->current]);
    } else {
        existing = 0;
    }

    if (*text == '\b') {
        if (existing) {
            while (existing > 1 && UTF8_IsTrailingByte((Uint8)textwin->lines[textwin->current][existing - 1])) {
                --existing;
            }
            --existing;
            textwin->lines[textwin->current][existing] = '\0';
        } else if (textwin->current > 0) {
            SDL_free(textwin->lines[textwin->current]);
            textwin->lines[textwin->current] = NULL;
            --textwin->current;
        }
        return;
    }

    line = (char *)SDL_realloc(textwin->lines[textwin->current], existing + len + 1);
    if (line) {
        SDL_memcpy(&line[existing], text, len);
        line[existing + len] = '\0';
        textwin->lines[textwin->current] = line;
        if (newline) {
            if (textwin->current == textwin->numlines - 1) {
                SDL_free(textwin->lines[0]);
                SDL_memmove(&textwin->lines[0], &textwin->lines[1], (textwin->numlines - 1) * sizeof(textwin->lines[1]));
                textwin->lines[textwin->current] = NULL;
            } else {
                ++textwin->current;
            }
        }
    }
}

void SDLTest_TextWindowClear(SDLTest_TextWindow *textwin)
{
    int i;

    for (i = 0; i < textwin->numlines; ++i) {
        if (textwin->lines[i]) {
            SDL_free(textwin->lines[i]);
            textwin->lines[i] = NULL;
        }
    }
    textwin->current = 0;
}

void SDLTest_TextWindowDestroy(SDLTest_TextWindow *textwin)
{
    if (textwin) {
        SDLTest_TextWindowClear(textwin);
        SDL_free(textwin->lines);
        SDL_free(textwin);
    }
}

void SDLTest_CleanupTextDrawing(void)
{
}
