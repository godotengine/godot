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

// This file is #included twice to support int and float versions with the same code.

static bool SDL_RECT_CAN_OVERFLOW(const RECTTYPE *rect)
{
    if (rect->x <= (SCALARTYPE)(SDL_MIN_SINT32 / 2) ||
        rect->x >= (SCALARTYPE)(SDL_MAX_SINT32 / 2) ||
        rect->y <= (SCALARTYPE)(SDL_MIN_SINT32 / 2) ||
        rect->y >= (SCALARTYPE)(SDL_MAX_SINT32 / 2) ||
        rect->w >= (SCALARTYPE)(SDL_MAX_SINT32 / 2) ||
        rect->h >= (SCALARTYPE)(SDL_MAX_SINT32 / 2)) {
        return true;
    }
    return false;
}

bool SDL_HASINTERSECTION(const RECTTYPE *A, const RECTTYPE *B)
{
    SCALARTYPE Amin, Amax, Bmin, Bmax;

    if (!A) {
        SDL_InvalidParamError("A");
        return false;
    } else if (!B) {
        SDL_InvalidParamError("B");
        return false;
    } else if (SDL_RECT_CAN_OVERFLOW(A) ||
               SDL_RECT_CAN_OVERFLOW(B)) {
        SDL_SetError("Potential rect math overflow");
        return false;
    } else if (SDL_RECTEMPTY(A) || SDL_RECTEMPTY(B)) {
        return false; // Special cases for empty rects
    }

    // Horizontal intersection
    Amin = A->x;
    Amax = Amin + A->w;
    Bmin = B->x;
    Bmax = Bmin + B->w;
    if (Bmin > Amin) {
        Amin = Bmin;
    }
    if (Bmax < Amax) {
        Amax = Bmax;
    }
    if ((Amax - ENCLOSEPOINTS_EPSILON) < Amin) {
        return false;
    }
    // Vertical intersection
    Amin = A->y;
    Amax = Amin + A->h;
    Bmin = B->y;
    Bmax = Bmin + B->h;
    if (Bmin > Amin) {
        Amin = Bmin;
    }
    if (Bmax < Amax) {
        Amax = Bmax;
    }
    if ((Amax - ENCLOSEPOINTS_EPSILON) < Amin) {
        return false;
    }
    return true;
}

bool SDL_INTERSECTRECT(const RECTTYPE *A, const RECTTYPE *B, RECTTYPE *result)
{
    SCALARTYPE Amin, Amax, Bmin, Bmax;

    if (!A) {
        SDL_InvalidParamError("A");
        return false;
    } else if (!B) {
        SDL_InvalidParamError("B");
        return false;
    } else if (SDL_RECT_CAN_OVERFLOW(A) ||
               SDL_RECT_CAN_OVERFLOW(B)) {
        SDL_SetError("Potential rect math overflow");
        return false;
    } else if (!result) {
        SDL_InvalidParamError("result");
        return false;
    } else if (SDL_RECTEMPTY(A) || SDL_RECTEMPTY(B)) { // Special cases for empty rects
        result->w = 0;
        result->h = 0;
        return false;
    }

    // Horizontal intersection
    Amin = A->x;
    Amax = Amin + A->w;
    Bmin = B->x;
    Bmax = Bmin + B->w;
    if (Bmin > Amin) {
        Amin = Bmin;
    }
    result->x = Amin;
    if (Bmax < Amax) {
        Amax = Bmax;
    }
    result->w = Amax - Amin;

    // Vertical intersection
    Amin = A->y;
    Amax = Amin + A->h;
    Bmin = B->y;
    Bmax = Bmin + B->h;
    if (Bmin > Amin) {
        Amin = Bmin;
    }
    result->y = Amin;
    if (Bmax < Amax) {
        Amax = Bmax;
    }
    result->h = Amax - Amin;

    return !SDL_RECTEMPTY(result);
}

bool SDL_UNIONRECT(const RECTTYPE *A, const RECTTYPE *B, RECTTYPE *result)
{
    SCALARTYPE Amin, Amax, Bmin, Bmax;

    if (!A) {
        return SDL_InvalidParamError("A");
    } else if (!B) {
        return SDL_InvalidParamError("B");
    } else if (SDL_RECT_CAN_OVERFLOW(A) ||
               SDL_RECT_CAN_OVERFLOW(B)) {
        return SDL_SetError("Potential rect math overflow");
    } else if (!result) {
        return SDL_InvalidParamError("result");
    } else if (SDL_RECTEMPTY(A)) { // Special cases for empty Rects
        if (SDL_RECTEMPTY(B)) {    // A and B empty
            SDL_zerop(result);
        } else { // A empty, B not empty
            *result = *B;
        }
        return true;
    } else if (SDL_RECTEMPTY(B)) { // A not empty, B empty
        *result = *A;
        return true;
    }

    // Horizontal union
    Amin = A->x;
    Amax = Amin + A->w;
    Bmin = B->x;
    Bmax = Bmin + B->w;
    if (Bmin < Amin) {
        Amin = Bmin;
    }
    result->x = Amin;
    if (Bmax > Amax) {
        Amax = Bmax;
    }
    result->w = Amax - Amin;

    // Vertical union
    Amin = A->y;
    Amax = Amin + A->h;
    Bmin = B->y;
    Bmax = Bmin + B->h;
    if (Bmin < Amin) {
        Amin = Bmin;
    }
    result->y = Amin;
    if (Bmax > Amax) {
        Amax = Bmax;
    }
    result->h = Amax - Amin;
    return true;
}

bool SDL_ENCLOSEPOINTS(const POINTTYPE *points, int count, const RECTTYPE *clip, RECTTYPE *result)
{
    SCALARTYPE minx = 0;
    SCALARTYPE miny = 0;
    SCALARTYPE maxx = 0;
    SCALARTYPE maxy = 0;
    SCALARTYPE x, y;
    int i;

    if (!points) {
        SDL_InvalidParamError("points");
        return false;
    } else if (count < 1) {
        SDL_InvalidParamError("count");
        return false;
    }

    if (clip) {
        bool added = false;
        const SCALARTYPE clip_minx = clip->x;
        const SCALARTYPE clip_miny = clip->y;
        const SCALARTYPE clip_maxx = clip->x + clip->w - ENCLOSEPOINTS_EPSILON;
        const SCALARTYPE clip_maxy = clip->y + clip->h - ENCLOSEPOINTS_EPSILON;

        // Special case for empty rectangle
        if (SDL_RECTEMPTY(clip)) {
            return false;
        }

        for (i = 0; i < count; ++i) {
            x = points[i].x;
            y = points[i].y;

            if (x < clip_minx || x > clip_maxx ||
                y < clip_miny || y > clip_maxy) {
                continue;
            }
            if (!added) {
                // Special case: if no result was requested, we are done
                if (!result) {
                    return true;
                }

                // First point added
                minx = maxx = x;
                miny = maxy = y;
                added = true;
                continue;
            }
            if (x < minx) {
                minx = x;
            } else if (x > maxx) {
                maxx = x;
            }
            if (y < miny) {
                miny = y;
            } else if (y > maxy) {
                maxy = y;
            }
        }
        if (!added) {
            return false;
        }
    } else {
        // Special case: if no result was requested, we are done
        if (!result) {
            return true;
        }

        // No clipping, always add the first point
        minx = maxx = points[0].x;
        miny = maxy = points[0].y;

        for (i = 1; i < count; ++i) {
            x = points[i].x;
            y = points[i].y;

            if (x < minx) {
                minx = x;
            } else if (x > maxx) {
                maxx = x;
            }
            if (y < miny) {
                miny = y;
            } else if (y > maxy) {
                maxy = y;
            }
        }
    }

    if (result) {
        result->x = minx;
        result->y = miny;
        result->w = (maxx - minx) + ENCLOSEPOINTS_EPSILON;
        result->h = (maxy - miny) + ENCLOSEPOINTS_EPSILON;
    }
    return true;
}

// Use the Cohen-Sutherland algorithm for line clipping
static int COMPUTEOUTCODE(const RECTTYPE *rect, SCALARTYPE x, SCALARTYPE y)
{
    int code = 0;
    if (y < rect->y) {
        code |= CODE_TOP;
    } else if (y > (rect->y + rect->h - ENCLOSEPOINTS_EPSILON)) {
        code |= CODE_BOTTOM;
    }
    if (x < rect->x) {
        code |= CODE_LEFT;
    } else if (x > (rect->x + rect->w - ENCLOSEPOINTS_EPSILON)) {
        code |= CODE_RIGHT;
    }
    return code;
}

bool SDL_INTERSECTRECTANDLINE(const RECTTYPE *rect, SCALARTYPE *X1, SCALARTYPE *Y1, SCALARTYPE *X2, SCALARTYPE *Y2)
{
    SCALARTYPE x = 0;
    SCALARTYPE y = 0;
    SCALARTYPE x1, y1;
    SCALARTYPE x2, y2;
    SCALARTYPE rectx1;
    SCALARTYPE recty1;
    SCALARTYPE rectx2;
    SCALARTYPE recty2;
    int outcode1, outcode2;

    if (!rect) {
        SDL_InvalidParamError("rect");
        return false;
    } else if (SDL_RECT_CAN_OVERFLOW(rect)) {
        SDL_SetError("Potential rect math overflow");
        return false;
    } else if (!X1) {
        SDL_InvalidParamError("X1");
        return false;
    } else if (!Y1) {
        SDL_InvalidParamError("Y1");
        return false;
    } else if (!X2) {
        SDL_InvalidParamError("X2");
        return false;
    } else if (!Y2) {
        SDL_InvalidParamError("Y2");
        return false;
    } else if (SDL_RECTEMPTY(rect)) {
        return false; // Special case for empty rect
    }

    x1 = *X1;
    y1 = *Y1;
    x2 = *X2;
    y2 = *Y2;
    rectx1 = rect->x;
    recty1 = rect->y;
    rectx2 = rect->x + rect->w - ENCLOSEPOINTS_EPSILON;
    recty2 = rect->y + rect->h - ENCLOSEPOINTS_EPSILON;

    // Check to see if entire line is inside rect
    if (x1 >= rectx1 && x1 <= rectx2 && x2 >= rectx1 && x2 <= rectx2 &&
        y1 >= recty1 && y1 <= recty2 && y2 >= recty1 && y2 <= recty2) {
        return true;
    }

    // Check to see if entire line is to one side of rect
    if ((x1 < rectx1 && x2 < rectx1) || (x1 > rectx2 && x2 > rectx2) ||
        (y1 < recty1 && y2 < recty1) || (y1 > recty2 && y2 > recty2)) {
        return false;
    }

    if (y1 == y2) { // Horizontal line, easy to clip
        if (x1 < rectx1) {
            *X1 = rectx1;
        } else if (x1 > rectx2) {
            *X1 = rectx2;
        }
        if (x2 < rectx1) {
            *X2 = rectx1;
        } else if (x2 > rectx2) {
            *X2 = rectx2;
        }
        return true;
    }

    if (x1 == x2) { // Vertical line, easy to clip
        if (y1 < recty1) {
            *Y1 = recty1;
        } else if (y1 > recty2) {
            *Y1 = recty2;
        }
        if (y2 < recty1) {
            *Y2 = recty1;
        } else if (y2 > recty2) {
            *Y2 = recty2;
        }
        return true;
    }

    // More complicated Cohen-Sutherland algorithm
    outcode1 = COMPUTEOUTCODE(rect, x1, y1);
    outcode2 = COMPUTEOUTCODE(rect, x2, y2);
    while (outcode1 || outcode2) {
        if (outcode1 & outcode2) {
            return false;
        }

        if (outcode1) {
            if (outcode1 & CODE_TOP) {
                y = recty1;
                x = (SCALARTYPE) (x1 + ((BIGSCALARTYPE)(x2 - x1) * (y - y1)) / (y2 - y1));
            } else if (outcode1 & CODE_BOTTOM) {
                y = recty2;
                x = (SCALARTYPE) (x1 + ((BIGSCALARTYPE)(x2 - x1) * (y - y1)) / (y2 - y1));
            } else if (outcode1 & CODE_LEFT) {
                x = rectx1;
                y = (SCALARTYPE) (y1 + ((BIGSCALARTYPE)(y2 - y1) * (x - x1)) / (x2 - x1));
            } else if (outcode1 & CODE_RIGHT) {
                x = rectx2;
                y = (SCALARTYPE) (y1 + ((BIGSCALARTYPE)(y2 - y1) * (x - x1)) / (x2 - x1));
            }
            x1 = x;
            y1 = y;
            outcode1 = COMPUTEOUTCODE(rect, x, y);
        } else {
            if (outcode2 & CODE_TOP) {
                SDL_assert(y2 != y1); // if equal: division by zero.
                y = recty1;
                x = (SCALARTYPE) (x1 + ((BIGSCALARTYPE)(x2 - x1) * (y - y1)) / (y2 - y1));
            } else if (outcode2 & CODE_BOTTOM) {
                SDL_assert(y2 != y1); // if equal: division by zero.
                y = recty2;
                x = (SCALARTYPE) (x1 + ((BIGSCALARTYPE)(x2 - x1) * (y - y1)) / (y2 - y1));
            } else if (outcode2 & CODE_LEFT) {
                /* If this assertion ever fires, here's the static analysis that warned about it:
                   http://buildbot.libsdl.org/sdl-static-analysis/sdl-macosx-static-analysis/sdl-macosx-static-analysis-1101/report-b0d01a.html#EndPath */
                SDL_assert(x2 != x1); // if equal: division by zero.
                x = rectx1;
                y = (SCALARTYPE) (y1 + ((BIGSCALARTYPE)(y2 - y1) * (x - x1)) / (x2 - x1));
            } else if (outcode2 & CODE_RIGHT) {
                /* If this assertion ever fires, here's the static analysis that warned about it:
                   http://buildbot.libsdl.org/sdl-static-analysis/sdl-macosx-static-analysis/sdl-macosx-static-analysis-1101/report-39b114.html#EndPath */
                SDL_assert(x2 != x1); // if equal: division by zero.
                x = rectx2;
                y = (SCALARTYPE) (y1 + ((BIGSCALARTYPE)(y2 - y1) * (x - x1)) / (x2 - x1));
            }
            x2 = x;
            y2 = y;
            outcode2 = COMPUTEOUTCODE(rect, x, y);
        }
    }
    *X1 = x1;
    *Y1 = y1;
    *X2 = x2;
    *Y2 = y2;
    return true;
}

#undef RECTTYPE
#undef POINTTYPE
#undef SCALARTYPE
#undef BIGSCALARTYPE
#undef COMPUTEOUTCODE
#undef ENCLOSEPOINTS_EPSILON
#undef SDL_RECT_CAN_OVERFLOW
#undef SDL_HASINTERSECTION
#undef SDL_INTERSECTRECT
#undef SDL_RECTEMPTY
#undef SDL_UNIONRECT
#undef SDL_ENCLOSEPOINTS
#undef SDL_INTERSECTRECTANDLINE
