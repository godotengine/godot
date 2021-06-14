
/***************************************************************************/
/*                                                                         */
/*  ftstroke.c                                                             */
/*                                                                         */
/*    FreeType path stroker (body).                                        */
/*                                                                         */
/*  Copyright 2002-2006, 2008-2011, 2013 by                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#include "sw_ft_stroker.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "sw_ft_math.h"

/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                      BEZIER COMPUTATIONS                      *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/

#define SW_FT_SMALL_CONIC_THRESHOLD (SW_FT_ANGLE_PI / 6)
#define SW_FT_SMALL_CUBIC_THRESHOLD (SW_FT_ANGLE_PI / 8)

#define SW_FT_EPSILON 2

#define SW_FT_IS_SMALL(x) ((x) > -SW_FT_EPSILON && (x) < SW_FT_EPSILON)

static SW_FT_Pos ft_pos_abs(SW_FT_Pos x)
{
    return x >= 0 ? x : -x;
}

static void ft_conic_split(SW_FT_Vector* base)
{
    SW_FT_Pos a, b;

    base[4].x = base[2].x;
    a = base[0].x + base[1].x;
    b = base[1].x + base[2].x;
    base[3].x = b >> 1;
    base[2].x = ( a + b ) >> 2;
    base[1].x = a >> 1;

    base[4].y = base[2].y;
    a = base[0].y + base[1].y;
    b = base[1].y + base[2].y;
    base[3].y = b >> 1;
    base[2].y = ( a + b ) >> 2;
    base[1].y = a >> 1;
}

static SW_FT_Bool ft_conic_is_small_enough(SW_FT_Vector* base,
                                           SW_FT_Angle*  angle_in,
                                           SW_FT_Angle*  angle_out)
{
    SW_FT_Vector d1, d2;
    SW_FT_Angle  theta;
    SW_FT_Int    close1, close2;

    d1.x = base[1].x - base[2].x;
    d1.y = base[1].y - base[2].y;
    d2.x = base[0].x - base[1].x;
    d2.y = base[0].y - base[1].y;

    close1 = SW_FT_IS_SMALL(d1.x) && SW_FT_IS_SMALL(d1.y);
    close2 = SW_FT_IS_SMALL(d2.x) && SW_FT_IS_SMALL(d2.y);

    if (close1) {
        if (close2) {
            /* basically a point;                      */
            /* do nothing to retain original direction */
        } else {
            *angle_in = *angle_out = SW_FT_Atan2(d2.x, d2.y);
        }
    } else /* !close1 */
    {
        if (close2) {
            *angle_in = *angle_out = SW_FT_Atan2(d1.x, d1.y);
        } else {
            *angle_in = SW_FT_Atan2(d1.x, d1.y);
            *angle_out = SW_FT_Atan2(d2.x, d2.y);
        }
    }

    theta = ft_pos_abs(SW_FT_Angle_Diff(*angle_in, *angle_out));

    return SW_FT_BOOL(theta < SW_FT_SMALL_CONIC_THRESHOLD);
}

static void ft_cubic_split(SW_FT_Vector* base)
{
    SW_FT_Pos a, b, c;

    base[6].x = base[3].x;
    a = base[0].x + base[1].x;
    b = base[1].x + base[2].x;
    c = base[2].x + base[3].x;
    base[5].x = c >> 1;
    c += b;
    base[4].x = c >> 2;
    base[1].x = a >> 1;
    a += b;
    base[2].x = a >> 2;
    base[3].x = ( a + c ) >> 3;

    base[6].y = base[3].y;
    a = base[0].y + base[1].y;
    b = base[1].y + base[2].y;
    c = base[2].y + base[3].y;
    base[5].y = c >> 1;
    c += b;
    base[4].y = c >> 2;
    base[1].y = a >> 1;
    a += b;
    base[2].y = a >> 2;
    base[3].y = ( a + c ) >> 3;
}

/* Return the average of `angle1' and `angle2'.            */
/* This gives correct result even if `angle1' and `angle2' */
/* have opposite signs.                                    */
static SW_FT_Angle ft_angle_mean(SW_FT_Angle angle1, SW_FT_Angle angle2)
{
    return angle1 + SW_FT_Angle_Diff(angle1, angle2) / 2;
}

static SW_FT_Bool ft_cubic_is_small_enough(SW_FT_Vector* base,
                                           SW_FT_Angle*  angle_in,
                                           SW_FT_Angle*  angle_mid,
                                           SW_FT_Angle*  angle_out)
{
    SW_FT_Vector d1, d2, d3;
    SW_FT_Angle  theta1, theta2;
    SW_FT_Int    close1, close2, close3;

    d1.x = base[2].x - base[3].x;
    d1.y = base[2].y - base[3].y;
    d2.x = base[1].x - base[2].x;
    d2.y = base[1].y - base[2].y;
    d3.x = base[0].x - base[1].x;
    d3.y = base[0].y - base[1].y;

    close1 = SW_FT_IS_SMALL(d1.x) && SW_FT_IS_SMALL(d1.y);
    close2 = SW_FT_IS_SMALL(d2.x) && SW_FT_IS_SMALL(d2.y);
    close3 = SW_FT_IS_SMALL(d3.x) && SW_FT_IS_SMALL(d3.y);

    if (close1) {
        if (close2) {
            if (close3) {
                /* basically a point;                      */
                /* do nothing to retain original direction */
            } else /* !close3 */
            {
                *angle_in = *angle_mid = *angle_out = SW_FT_Atan2(d3.x, d3.y);
            }
        } else /* !close2 */
        {
            if (close3) {
                *angle_in = *angle_mid = *angle_out = SW_FT_Atan2(d2.x, d2.y);
            } else /* !close3 */
            {
                *angle_in = *angle_mid = SW_FT_Atan2(d2.x, d2.y);
                *angle_out = SW_FT_Atan2(d3.x, d3.y);
            }
        }
    } else /* !close1 */
    {
        if (close2) {
            if (close3) {
                *angle_in = *angle_mid = *angle_out = SW_FT_Atan2(d1.x, d1.y);
            } else /* !close3 */
            {
                *angle_in = SW_FT_Atan2(d1.x, d1.y);
                *angle_out = SW_FT_Atan2(d3.x, d3.y);
                *angle_mid = ft_angle_mean(*angle_in, *angle_out);
            }
        } else /* !close2 */
        {
            if (close3) {
                *angle_in = SW_FT_Atan2(d1.x, d1.y);
                *angle_mid = *angle_out = SW_FT_Atan2(d2.x, d2.y);
            } else /* !close3 */
            {
                *angle_in = SW_FT_Atan2(d1.x, d1.y);
                *angle_mid = SW_FT_Atan2(d2.x, d2.y);
                *angle_out = SW_FT_Atan2(d3.x, d3.y);
            }
        }
    }

    theta1 = ft_pos_abs(SW_FT_Angle_Diff(*angle_in, *angle_mid));
    theta2 = ft_pos_abs(SW_FT_Angle_Diff(*angle_mid, *angle_out));

    return SW_FT_BOOL(theta1 < SW_FT_SMALL_CUBIC_THRESHOLD &&
                      theta2 < SW_FT_SMALL_CUBIC_THRESHOLD);
}

/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                       STROKE BORDERS                          *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/

typedef enum SW_FT_StrokeTags_ {
    SW_FT_STROKE_TAG_ON = 1,    /* on-curve point  */
    SW_FT_STROKE_TAG_CUBIC = 2, /* cubic off-point */
    SW_FT_STROKE_TAG_BEGIN = 4, /* sub-path start  */
    SW_FT_STROKE_TAG_END = 8    /* sub-path end    */

} SW_FT_StrokeTags;

#define SW_FT_STROKE_TAG_BEGIN_END \
    (SW_FT_STROKE_TAG_BEGIN | SW_FT_STROKE_TAG_END)

typedef struct SW_FT_StrokeBorderRec_ {
    SW_FT_UInt    num_points;
    SW_FT_UInt    max_points;
    SW_FT_Vector* points;
    SW_FT_Byte*   tags;
    SW_FT_Bool    movable; /* TRUE for ends of lineto borders */
    SW_FT_Int     start;   /* index of current sub-path start point */
    SW_FT_Bool    valid;

} SW_FT_StrokeBorderRec, *SW_FT_StrokeBorder;

SW_FT_Error SW_FT_Outline_Check(SW_FT_Outline* outline)
{
    if (outline) {
        SW_FT_Int n_points = outline->n_points;
        SW_FT_Int n_contours = outline->n_contours;
        SW_FT_Int end0, end;
        SW_FT_Int n;

        /* empty glyph? */
        if (n_points == 0 && n_contours == 0) return 0;

        /* check point and contour counts */
        if (n_points <= 0 || n_contours <= 0) goto Bad;

        end0 = end = -1;
        for (n = 0; n < n_contours; n++) {
            end = outline->contours[n];

            /* note that we don't accept empty contours */
            if (end <= end0 || end >= n_points) goto Bad;

            end0 = end;
        }

        if (end != n_points - 1) goto Bad;

        /* XXX: check the tags array */
        return 0;
    }

Bad:
    return -1;  // SW_FT_THROW( Invalid_Argument );
}

void SW_FT_Outline_Get_CBox(const SW_FT_Outline* outline, SW_FT_BBox* acbox)
{
    SW_FT_Pos xMin, yMin, xMax, yMax;

    if (outline && acbox) {
        if (outline->n_points == 0) {
            xMin = 0;
            yMin = 0;
            xMax = 0;
            yMax = 0;
        } else {
            SW_FT_Vector* vec = outline->points;
            SW_FT_Vector* limit = vec + outline->n_points;

            xMin = xMax = vec->x;
            yMin = yMax = vec->y;
            vec++;

            for (; vec < limit; vec++) {
                SW_FT_Pos x, y;

                x = vec->x;
                if (x < xMin) xMin = x;
                if (x > xMax) xMax = x;

                y = vec->y;
                if (y < yMin) yMin = y;
                if (y > yMax) yMax = y;
            }
        }
        acbox->xMin = xMin;
        acbox->xMax = xMax;
        acbox->yMin = yMin;
        acbox->yMax = yMax;
    }
}

static SW_FT_Error ft_stroke_border_grow(SW_FT_StrokeBorder border,
                                         SW_FT_UInt         new_points)
{
    SW_FT_UInt  old_max = border->max_points;
    SW_FT_UInt  new_max = border->num_points + new_points;
    SW_FT_Error error = 0;

    if (new_max > old_max) {
        SW_FT_UInt cur_max = old_max;

        while (cur_max < new_max) cur_max += (cur_max >> 1) + 16;

        border->points = (SW_FT_Vector*)realloc(border->points,
                                                cur_max * sizeof(SW_FT_Vector));
        border->tags =
            (SW_FT_Byte*)realloc(border->tags, cur_max * sizeof(SW_FT_Byte));

        if (!border->points || !border->tags) goto Exit;

        border->max_points = cur_max;
    }

Exit:
    return error;
}

static void ft_stroke_border_close(SW_FT_StrokeBorder border,
                                   SW_FT_Bool         reverse)
{
    SW_FT_UInt start = border->start;
    SW_FT_UInt count = border->num_points;

    assert(border->start >= 0);

    /* don't record empty paths! */
    if (count <= start + 1U)
        border->num_points = start;
    else {
        /* copy the last point to the start of this sub-path, since */
        /* it contains the `adjusted' starting coordinates          */
        border->num_points = --count;
        border->points[start] = border->points[count];

        if (reverse) {
            /* reverse the points */
            {
                SW_FT_Vector* vec1 = border->points + start + 1;
                SW_FT_Vector* vec2 = border->points + count - 1;

                for (; vec1 < vec2; vec1++, vec2--) {
                    SW_FT_Vector tmp;

                    tmp = *vec1;
                    *vec1 = *vec2;
                    *vec2 = tmp;
                }
            }

            /* then the tags */
            {
                SW_FT_Byte* tag1 = border->tags + start + 1;
                SW_FT_Byte* tag2 = border->tags + count - 1;

                for (; tag1 < tag2; tag1++, tag2--) {
                    SW_FT_Byte tmp;

                    tmp = *tag1;
                    *tag1 = *tag2;
                    *tag2 = tmp;
                }
            }
        }

        border->tags[start] |= SW_FT_STROKE_TAG_BEGIN;
        border->tags[count - 1] |= SW_FT_STROKE_TAG_END;
    }

    border->start = -1;
    border->movable = FALSE;
}

static SW_FT_Error ft_stroke_border_lineto(SW_FT_StrokeBorder border,
                                           SW_FT_Vector* to, SW_FT_Bool movable)
{
    SW_FT_Error error = 0;

    assert(border->start >= 0);

    if (border->movable) {
        /* move last point */
        border->points[border->num_points - 1] = *to;
    } else {
        /* don't add zero-length lineto */
        if (border->num_points > 0 &&
            SW_FT_IS_SMALL(border->points[border->num_points - 1].x - to->x) &&
            SW_FT_IS_SMALL(border->points[border->num_points - 1].y - to->y))
            return error;

        /* add one point */
        error = ft_stroke_border_grow(border, 1);
        if (!error) {
            SW_FT_Vector* vec = border->points + border->num_points;
            SW_FT_Byte*   tag = border->tags + border->num_points;

            vec[0] = *to;
            tag[0] = SW_FT_STROKE_TAG_ON;

            border->num_points += 1;
        }
    }
    border->movable = movable;
    return error;
}

static SW_FT_Error ft_stroke_border_conicto(SW_FT_StrokeBorder border,
                                            SW_FT_Vector*      control,
                                            SW_FT_Vector*      to)
{
    SW_FT_Error error;

    assert(border->start >= 0);

    error = ft_stroke_border_grow(border, 2);
    if (!error) {
        SW_FT_Vector* vec = border->points + border->num_points;
        SW_FT_Byte*   tag = border->tags + border->num_points;

        vec[0] = *control;
        vec[1] = *to;

        tag[0] = 0;
        tag[1] = SW_FT_STROKE_TAG_ON;

        border->num_points += 2;
    }

    border->movable = FALSE;

    return error;
}

static SW_FT_Error ft_stroke_border_cubicto(SW_FT_StrokeBorder border,
                                            SW_FT_Vector*      control1,
                                            SW_FT_Vector*      control2,
                                            SW_FT_Vector*      to)
{
    SW_FT_Error error;

    assert(border->start >= 0);

    error = ft_stroke_border_grow(border, 3);
    if (!error) {
        SW_FT_Vector* vec = border->points + border->num_points;
        SW_FT_Byte*   tag = border->tags + border->num_points;

        vec[0] = *control1;
        vec[1] = *control2;
        vec[2] = *to;

        tag[0] = SW_FT_STROKE_TAG_CUBIC;
        tag[1] = SW_FT_STROKE_TAG_CUBIC;
        tag[2] = SW_FT_STROKE_TAG_ON;

        border->num_points += 3;
    }

    border->movable = FALSE;

    return error;
}

#define SW_FT_ARC_CUBIC_ANGLE (SW_FT_ANGLE_PI / 2)


static SW_FT_Error
ft_stroke_border_arcto( SW_FT_StrokeBorder  border,
                        SW_FT_Vector*       center,
                        SW_FT_Fixed         radius,
                        SW_FT_Angle         angle_start,
                        SW_FT_Angle         angle_diff )
{
    SW_FT_Fixed   coef;
    SW_FT_Vector  a0, a1, a2, a3;
    SW_FT_Int     i, arcs = 1;
    SW_FT_Error   error = 0;


    /* number of cubic arcs to draw */
    while (  angle_diff > SW_FT_ARC_CUBIC_ANGLE * arcs ||
            -angle_diff > SW_FT_ARC_CUBIC_ANGLE * arcs )
      arcs++;

    /* control tangents */
    coef  = SW_FT_Tan( angle_diff / ( 4 * arcs ) );
    coef += coef / 3;

    /* compute start and first control point */
    SW_FT_Vector_From_Polar( &a0, radius, angle_start );
    a1.x = SW_FT_MulFix( -a0.y, coef );
    a1.y = SW_FT_MulFix(  a0.x, coef );

    a0.x += center->x;
    a0.y += center->y;
    a1.x += a0.x;
    a1.y += a0.y;

    for ( i = 1; i <= arcs; i++ )
    {
      /* compute end and second control point */
      SW_FT_Vector_From_Polar( &a3, radius,
                            angle_start + i * angle_diff / arcs );
      a2.x = SW_FT_MulFix(  a3.y, coef );
      a2.y = SW_FT_MulFix( -a3.x, coef );

      a3.x += center->x;
      a3.y += center->y;
      a2.x += a3.x;
      a2.y += a3.y;

      /* add cubic arc */
      error = ft_stroke_border_cubicto( border, &a1, &a2, &a3 );
      if ( error )
        break;

      /* a0 = a3; */
      a1.x = a3.x - a2.x + a3.x;
      a1.y = a3.y - a2.y + a3.y;
    }

    return error;
}

static SW_FT_Error ft_stroke_border_moveto(SW_FT_StrokeBorder border,
                                           SW_FT_Vector*      to)
{
    /* close current open path if any ? */
    if (border->start >= 0) ft_stroke_border_close(border, FALSE);

    border->start = border->num_points;
    border->movable = FALSE;

    return ft_stroke_border_lineto(border, to, FALSE);
}

static void ft_stroke_border_init(SW_FT_StrokeBorder border)
{
    border->points = NULL;
    border->tags = NULL;

    border->num_points = 0;
    border->max_points = 0;
    border->start = -1;
    border->valid = FALSE;
}

static void ft_stroke_border_reset(SW_FT_StrokeBorder border)
{
    border->num_points = 0;
    border->start = -1;
    border->valid = FALSE;
}

static void ft_stroke_border_done(SW_FT_StrokeBorder border)
{
    free(border->points);
    free(border->tags);

    border->num_points = 0;
    border->max_points = 0;
    border->start = -1;
    border->valid = FALSE;
}

static SW_FT_Error ft_stroke_border_get_counts(SW_FT_StrokeBorder border,
                                               SW_FT_UInt*        anum_points,
                                               SW_FT_UInt*        anum_contours)
{
    SW_FT_Error error = 0;
    SW_FT_UInt  num_points = 0;
    SW_FT_UInt  num_contours = 0;

    SW_FT_UInt    count = border->num_points;
    SW_FT_Vector* point = border->points;
    SW_FT_Byte*   tags = border->tags;
    SW_FT_Int     in_contour = 0;

    for (; count > 0; count--, num_points++, point++, tags++) {
        if (tags[0] & SW_FT_STROKE_TAG_BEGIN) {
            if (in_contour != 0) goto Fail;

            in_contour = 1;
        } else if (in_contour == 0)
            goto Fail;

        if (tags[0] & SW_FT_STROKE_TAG_END) {
            in_contour = 0;
            num_contours++;
        }
    }

    if (in_contour != 0) goto Fail;

    border->valid = TRUE;

Exit:
    *anum_points = num_points;
    *anum_contours = num_contours;
    return error;

Fail:
    num_points = 0;
    num_contours = 0;
    goto Exit;
}

static void ft_stroke_border_export(SW_FT_StrokeBorder border,
                                    SW_FT_Outline*     outline)
{
    /* copy point locations */
    memcpy(outline->points + outline->n_points, border->points,
           border->num_points * sizeof(SW_FT_Vector));

    /* copy tags */
    {
        SW_FT_UInt  count = border->num_points;
        SW_FT_Byte* read = border->tags;
        SW_FT_Byte* write = (SW_FT_Byte*)outline->tags + outline->n_points;

        for (; count > 0; count--, read++, write++) {
            if (*read & SW_FT_STROKE_TAG_ON)
                *write = SW_FT_CURVE_TAG_ON;
            else if (*read & SW_FT_STROKE_TAG_CUBIC)
                *write = SW_FT_CURVE_TAG_CUBIC;
            else
                *write = SW_FT_CURVE_TAG_CONIC;
        }
    }

    /* copy contours */
    {
        SW_FT_UInt   count = border->num_points;
        SW_FT_Byte*  tags = border->tags;
        SW_FT_Short* write = outline->contours + outline->n_contours;
        SW_FT_Short  idx = (SW_FT_Short)outline->n_points;

        for (; count > 0; count--, tags++, idx++) {
            if (*tags & SW_FT_STROKE_TAG_END) {
                *write++ = idx;
                outline->n_contours++;
            }
        }
    }

    outline->n_points = (short)(outline->n_points + border->num_points);

    assert(SW_FT_Outline_Check(outline) == 0);
}

/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                           STROKER                             *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/

#define SW_FT_SIDE_TO_ROTATE(s) (SW_FT_ANGLE_PI2 - (s)*SW_FT_ANGLE_PI)

typedef struct SW_FT_StrokerRec_ {
    SW_FT_Angle  angle_in;            /* direction into curr join */
    SW_FT_Angle  angle_out;           /* direction out of join  */
    SW_FT_Vector center;              /* current position */
    SW_FT_Fixed  line_length;         /* length of last lineto */
    SW_FT_Bool   first_point;         /* is this the start? */
    SW_FT_Bool   subpath_open;        /* is the subpath open? */
    SW_FT_Angle  subpath_angle;       /* subpath start direction */
    SW_FT_Vector subpath_start;       /* subpath start position */
    SW_FT_Fixed  subpath_line_length; /* subpath start lineto len */
    SW_FT_Bool   handle_wide_strokes; /* use wide strokes logic? */

    SW_FT_Stroker_LineCap  line_cap;
    SW_FT_Stroker_LineJoin line_join;
    SW_FT_Stroker_LineJoin line_join_saved;
    SW_FT_Fixed            miter_limit;
    SW_FT_Fixed            radius;

    SW_FT_StrokeBorderRec borders[2];
} SW_FT_StrokerRec;

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_New(SW_FT_Stroker* astroker)
{
    SW_FT_Error   error = 0; /* assigned in SW_FT_NEW */
    SW_FT_Stroker stroker = NULL;

    stroker = (SW_FT_StrokerRec*)calloc(1, sizeof(SW_FT_StrokerRec));
    if (stroker) {
        ft_stroke_border_init(&stroker->borders[0]);
        ft_stroke_border_init(&stroker->borders[1]);
    }

    *astroker = stroker;

    return error;
}

void SW_FT_Stroker_Rewind(SW_FT_Stroker stroker)
{
    if (stroker) {
        ft_stroke_border_reset(&stroker->borders[0]);
        ft_stroke_border_reset(&stroker->borders[1]);
    }
}

/* documentation is in ftstroke.h */

void SW_FT_Stroker_Set(SW_FT_Stroker stroker, SW_FT_Fixed radius,
                       SW_FT_Stroker_LineCap  line_cap,
                       SW_FT_Stroker_LineJoin line_join,
                       SW_FT_Fixed            miter_limit)
{
    stroker->radius = radius;
    stroker->line_cap = line_cap;
    stroker->line_join = line_join;
    stroker->miter_limit = miter_limit;

    /* ensure miter limit has sensible value */
    if (stroker->miter_limit < 0x10000) stroker->miter_limit = 0x10000;

    /* save line join style:                                           */
    /* line join style can be temporarily changed when stroking curves */
    stroker->line_join_saved = line_join;

    SW_FT_Stroker_Rewind(stroker);
}

/* documentation is in ftstroke.h */

void SW_FT_Stroker_Done(SW_FT_Stroker stroker)
{
    if (stroker) {
        ft_stroke_border_done(&stroker->borders[0]);
        ft_stroke_border_done(&stroker->borders[1]);

        free(stroker);
    }
}

/* create a circular arc at a corner or cap */
static SW_FT_Error ft_stroker_arcto(SW_FT_Stroker stroker, SW_FT_Int side)
{
    SW_FT_Angle        total, rotate;
    SW_FT_Fixed        radius = stroker->radius;
    SW_FT_Error        error = 0;
    SW_FT_StrokeBorder border = stroker->borders + side;

    rotate = SW_FT_SIDE_TO_ROTATE(side);

    total = SW_FT_Angle_Diff(stroker->angle_in, stroker->angle_out);
    if (total == SW_FT_ANGLE_PI) total = -rotate * 2;

    error = ft_stroke_border_arcto(border, &stroker->center, radius,
                                   stroker->angle_in + rotate, total);
    border->movable = FALSE;
    return error;
}

/* add a cap at the end of an opened path */
static SW_FT_Error
ft_stroker_cap(SW_FT_Stroker stroker,
               SW_FT_Angle angle,
               SW_FT_Int side)
{
    SW_FT_Error error = 0;

    if (stroker->line_cap == SW_FT_STROKER_LINECAP_ROUND)
    {
        /* add a round cap */
        stroker->angle_in = angle;
        stroker->angle_out = angle + SW_FT_ANGLE_PI;

        error = ft_stroker_arcto(stroker, side);
    }
    else
    {
        /* add a square or butt cap */
        SW_FT_Vector        middle, delta;
        SW_FT_Fixed         radius = stroker->radius;
        SW_FT_StrokeBorder  border = stroker->borders + side;

        /* compute middle point and first angle point */
        SW_FT_Vector_From_Polar( &middle, radius, angle );
        delta.x = side ?  middle.y : -middle.y;
        delta.y = side ? -middle.x :  middle.x;

        if ( stroker->line_cap == SW_FT_STROKER_LINECAP_SQUARE )
        {
            middle.x += stroker->center.x;
            middle.y += stroker->center.y;
        }
        else  /* SW_FT_STROKER_LINECAP_BUTT */
        {
            middle.x  = stroker->center.x;
            middle.y  = stroker->center.y;
        }

        delta.x  += middle.x;
        delta.y  += middle.y;

        error = ft_stroke_border_lineto( border, &delta, FALSE );
        if ( error )
        goto Exit;

        /* compute second angle point */
        delta.x = middle.x - delta.x + middle.x;
        delta.y = middle.y - delta.y + middle.y;

        error = ft_stroke_border_lineto( border, &delta, FALSE );
    }

Exit:
    return error;
}

/* process an inside corner, i.e. compute intersection */
static SW_FT_Error ft_stroker_inside(SW_FT_Stroker stroker, SW_FT_Int side,
                                     SW_FT_Fixed line_length)
{
    SW_FT_StrokeBorder border = stroker->borders + side;
    SW_FT_Angle        phi, theta, rotate;
    SW_FT_Fixed        length;
    SW_FT_Vector       sigma, delta;
    SW_FT_Error        error = 0;
    SW_FT_Bool         intersect; /* use intersection of lines? */

    rotate = SW_FT_SIDE_TO_ROTATE(side);

    theta = SW_FT_Angle_Diff(stroker->angle_in, stroker->angle_out) / 2;

    /* Only intersect borders if between two lineto's and both */
    /* lines are long enough (line_length is zero for curves). */
    if (!border->movable || line_length == 0  ||
         theta > 0x59C000 || theta < -0x59C000 )
        intersect = FALSE;
    else {
      /* compute minimum required length of lines */
      SW_FT_Fixed  min_length;


      SW_FT_Vector_Unit( &sigma, theta );
      min_length =
        ft_pos_abs( SW_FT_MulDiv( stroker->radius, sigma.y, sigma.x ) );

      intersect = SW_FT_BOOL( min_length                         &&
                           stroker->line_length >= min_length &&
                           line_length          >= min_length );
    }

    if (!intersect) {
        SW_FT_Vector_From_Polar(&delta, stroker->radius,
                                stroker->angle_out + rotate);
        delta.x += stroker->center.x;
        delta.y += stroker->center.y;

        border->movable = FALSE;
    } else {
        /* compute median angle */
        phi = stroker->angle_in + theta + rotate;

      length = SW_FT_DivFix( stroker->radius, sigma.x );

      SW_FT_Vector_From_Polar( &delta, length, phi );
      delta.x += stroker->center.x;
      delta.y += stroker->center.y;
    }

    error = ft_stroke_border_lineto(border, &delta, FALSE);

    return error;
}

  /* process an outside corner, i.e. compute bevel/miter/round */
static SW_FT_Error
ft_stroker_outside( SW_FT_Stroker  stroker,
                    SW_FT_Int      side,
                    SW_FT_Fixed    line_length )
{
    SW_FT_StrokeBorder  border = stroker->borders + side;
    SW_FT_Error         error;
    SW_FT_Angle         rotate;


    if ( stroker->line_join == SW_FT_STROKER_LINEJOIN_ROUND )
      error = ft_stroker_arcto( stroker, side );
    else
    {
      /* this is a mitered (pointed) or beveled (truncated) corner */
      SW_FT_Fixed   radius = stroker->radius;
      SW_FT_Vector  sigma;
      SW_FT_Angle   theta = 0, phi = 0;
      SW_FT_Bool    bevel, fixed_bevel;


      rotate = SW_FT_SIDE_TO_ROTATE( side );

      bevel =
        SW_FT_BOOL( stroker->line_join == SW_FT_STROKER_LINEJOIN_BEVEL );

      fixed_bevel =
        SW_FT_BOOL( stroker->line_join != SW_FT_STROKER_LINEJOIN_MITER_VARIABLE );

      /* check miter limit first */
      if ( !bevel )
      {
        theta = SW_FT_Angle_Diff( stroker->angle_in, stroker->angle_out ) / 2;

        if ( theta == SW_FT_ANGLE_PI2 )
          theta = -rotate;

        phi    = stroker->angle_in + theta + rotate;

        SW_FT_Vector_From_Polar( &sigma, stroker->miter_limit, theta );

        /* is miter limit exceeded? */
        if ( sigma.x < 0x10000L )
        {
          /* don't create variable bevels for very small deviations; */
          /* FT_Sin(x) = 0 for x <= 57                               */
          if ( fixed_bevel || ft_pos_abs( theta ) > 57 )
            bevel = TRUE;
        }
      }

      if ( bevel )  /* this is a bevel (broken angle) */
      {
        if ( fixed_bevel )
        {
          /* the outer corners are simply joined together */
          SW_FT_Vector  delta;


          /* add bevel */
          SW_FT_Vector_From_Polar( &delta,
                                radius,
                                stroker->angle_out + rotate );
          delta.x += stroker->center.x;
          delta.y += stroker->center.y;

          border->movable = FALSE;
          error = ft_stroke_border_lineto( border, &delta, FALSE );
        }
        else /* variable bevel or clipped miter */
        {
          /* the miter is truncated */
          SW_FT_Vector  middle, delta;
          SW_FT_Fixed   coef;


          /* compute middle point and first angle point */
          SW_FT_Vector_From_Polar( &middle,
                                   SW_FT_MulFix( radius, stroker->miter_limit ),
                                   phi );

          coef    = SW_FT_DivFix(  0x10000L - sigma.x, sigma.y );
          delta.x = SW_FT_MulFix(  middle.y, coef );
          delta.y = SW_FT_MulFix( -middle.x, coef );

          middle.x += stroker->center.x;
          middle.y += stroker->center.y;
          delta.x  += middle.x;
          delta.y  += middle.y;

          error = ft_stroke_border_lineto( border, &delta, FALSE );
          if ( error )
            goto Exit;

          /* compute second angle point */
          delta.x = middle.x - delta.x + middle.x;
          delta.y = middle.y - delta.y + middle.y;

          error = ft_stroke_border_lineto( border, &delta, FALSE );
          if ( error )
            goto Exit;

          /* finally, add an end point; only needed if not lineto */
          /* (line_length is zero for curves)                     */
          if ( line_length == 0 )
          {
            SW_FT_Vector_From_Polar( &delta,
                                  radius,
                                  stroker->angle_out + rotate );

            delta.x += stroker->center.x;
            delta.y += stroker->center.y;

            error = ft_stroke_border_lineto( border, &delta, FALSE );
          }
        }
      }
      else /* this is a miter (intersection) */
      {
        SW_FT_Fixed   length;
        SW_FT_Vector  delta;


        length = SW_FT_MulDiv( stroker->radius, stroker->miter_limit, sigma.x );

        SW_FT_Vector_From_Polar( &delta, length, phi );
        delta.x += stroker->center.x;
        delta.y += stroker->center.y;

        error = ft_stroke_border_lineto( border, &delta, FALSE );
        if ( error )
          goto Exit;

        /* now add an end point; only needed if not lineto */
        /* (line_length is zero for curves)                */
        if ( line_length == 0 )
        {
          SW_FT_Vector_From_Polar( &delta,
                                stroker->radius,
                                stroker->angle_out + rotate );
          delta.x += stroker->center.x;
          delta.y += stroker->center.y;

          error = ft_stroke_border_lineto( border, &delta, FALSE );
        }
      }
    }

  Exit:
    return error;
}

static SW_FT_Error ft_stroker_process_corner(SW_FT_Stroker stroker,
                                             SW_FT_Fixed   line_length)
{
    SW_FT_Error error = 0;
    SW_FT_Angle turn;
    SW_FT_Int   inside_side;

    turn = SW_FT_Angle_Diff(stroker->angle_in, stroker->angle_out);

    /* no specific corner processing is required if the turn is 0 */
    if (turn == 0) goto Exit;

    /* when we turn to the right, the inside side is 0 */
    inside_side = 0;

    /* otherwise, the inside side is 1 */
    if (turn < 0) inside_side = 1;

    /* process the inside side */
    error = ft_stroker_inside(stroker, inside_side, line_length);
    if (error) goto Exit;

    /* process the outside side */
    error = ft_stroker_outside(stroker, 1 - inside_side, line_length);

Exit:
    return error;
}

/* add two points to the left and right borders corresponding to the */
/* start of the subpath                                              */
static SW_FT_Error ft_stroker_subpath_start(SW_FT_Stroker stroker,
                                            SW_FT_Angle   start_angle,
                                            SW_FT_Fixed   line_length)
{
    SW_FT_Vector       delta;
    SW_FT_Vector       point;
    SW_FT_Error        error;
    SW_FT_StrokeBorder border;

    SW_FT_Vector_From_Polar(&delta, stroker->radius,
                            start_angle + SW_FT_ANGLE_PI2);

    point.x = stroker->center.x + delta.x;
    point.y = stroker->center.y + delta.y;

    border = stroker->borders;
    error = ft_stroke_border_moveto(border, &point);
    if (error) goto Exit;

    point.x = stroker->center.x - delta.x;
    point.y = stroker->center.y - delta.y;

    border++;
    error = ft_stroke_border_moveto(border, &point);

    /* save angle, position, and line length for last join */
    /* (line_length is zero for curves)                    */
    stroker->subpath_angle = start_angle;
    stroker->first_point = FALSE;
    stroker->subpath_line_length = line_length;

Exit:
    return error;
}

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_LineTo(SW_FT_Stroker stroker, SW_FT_Vector* to)
{
    SW_FT_Error        error = 0;
    SW_FT_StrokeBorder border;
    SW_FT_Vector       delta;
    SW_FT_Angle        angle;
    SW_FT_Int          side;
    SW_FT_Fixed        line_length;

    delta.x = to->x - stroker->center.x;
    delta.y = to->y - stroker->center.y;

    /* a zero-length lineto is a no-op; avoid creating a spurious corner */
    if (delta.x == 0 && delta.y == 0) goto Exit;

    /* compute length of line */
    line_length = SW_FT_Vector_Length(&delta);

    angle = SW_FT_Atan2(delta.x, delta.y);
    SW_FT_Vector_From_Polar(&delta, stroker->radius, angle + SW_FT_ANGLE_PI2);

    /* process corner if necessary */
    if (stroker->first_point) {
        /* This is the first segment of a subpath.  We need to     */
        /* add a point to each border at their respective starting */
        /* point locations.                                        */
        error = ft_stroker_subpath_start(stroker, angle, line_length);
        if (error) goto Exit;
    } else {
        /* process the current corner */
        stroker->angle_out = angle;
        error = ft_stroker_process_corner(stroker, line_length);
        if (error) goto Exit;
    }

    /* now add a line segment to both the `inside' and `outside' paths */
    for (border = stroker->borders, side = 1; side >= 0; side--, border++) {
        SW_FT_Vector point;

        point.x = to->x + delta.x;
        point.y = to->y + delta.y;

        /* the ends of lineto borders are movable */
        error = ft_stroke_border_lineto(border, &point, TRUE);
        if (error) goto Exit;

        delta.x = -delta.x;
        delta.y = -delta.y;
    }

    stroker->angle_in = angle;
    stroker->center = *to;
    stroker->line_length = line_length;

Exit:
    return error;
}

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_ConicTo(SW_FT_Stroker stroker, SW_FT_Vector* control,
                                  SW_FT_Vector* to)
{
    SW_FT_Error   error = 0;
    SW_FT_Vector  bez_stack[34];
    SW_FT_Vector* arc;
    SW_FT_Vector* limit = bez_stack + 30;
    SW_FT_Bool    first_arc = TRUE;

    /* if all control points are coincident, this is a no-op; */
    /* avoid creating a spurious corner                       */
    if (SW_FT_IS_SMALL(stroker->center.x - control->x) &&
        SW_FT_IS_SMALL(stroker->center.y - control->y) &&
        SW_FT_IS_SMALL(control->x - to->x) &&
        SW_FT_IS_SMALL(control->y - to->y)) {
        stroker->center = *to;
        goto Exit;
    }

    arc = bez_stack;
    arc[0] = *to;
    arc[1] = *control;
    arc[2] = stroker->center;

    while (arc >= bez_stack) {
        SW_FT_Angle angle_in, angle_out;

        /* initialize with current direction */
        angle_in = angle_out = stroker->angle_in;

        if (arc < limit &&
            !ft_conic_is_small_enough(arc, &angle_in, &angle_out)) {
            if (stroker->first_point) stroker->angle_in = angle_in;

            ft_conic_split(arc);
            arc += 2;
            continue;
        }

        if (first_arc) {
            first_arc = FALSE;

            /* process corner if necessary */
            if (stroker->first_point)
                error = ft_stroker_subpath_start(stroker, angle_in, 0);
            else {
                stroker->angle_out = angle_in;
                error = ft_stroker_process_corner(stroker, 0);
            }
        } else if (ft_pos_abs(SW_FT_Angle_Diff(stroker->angle_in, angle_in)) >
                   SW_FT_SMALL_CONIC_THRESHOLD / 4) {
            /* if the deviation from one arc to the next is too great, */
            /* add a round corner                                      */
            stroker->center = arc[2];
            stroker->angle_out = angle_in;
            stroker->line_join = SW_FT_STROKER_LINEJOIN_ROUND;

            error = ft_stroker_process_corner(stroker, 0);

            /* reinstate line join style */
            stroker->line_join = stroker->line_join_saved;
        }

        if (error) goto Exit;

        /* the arc's angle is small enough; we can add it directly to each */
        /* border                                                          */
        {
            SW_FT_Vector       ctrl, end;
            SW_FT_Angle        theta, phi, rotate, alpha0 = 0;
            SW_FT_Fixed        length;
            SW_FT_StrokeBorder border;
            SW_FT_Int          side;

            theta = SW_FT_Angle_Diff(angle_in, angle_out) / 2;
            phi = angle_in + theta;
            length = SW_FT_DivFix(stroker->radius, SW_FT_Cos(theta));

            /* compute direction of original arc */
            if (stroker->handle_wide_strokes)
                alpha0 = SW_FT_Atan2(arc[0].x - arc[2].x, arc[0].y - arc[2].y);

            for (border = stroker->borders, side = 0; side <= 1;
                 side++, border++) {
                rotate = SW_FT_SIDE_TO_ROTATE(side);

                /* compute control point */
                SW_FT_Vector_From_Polar(&ctrl, length, phi + rotate);
                ctrl.x += arc[1].x;
                ctrl.y += arc[1].y;

                /* compute end point */
                SW_FT_Vector_From_Polar(&end, stroker->radius,
                                        angle_out + rotate);
                end.x += arc[0].x;
                end.y += arc[0].y;

                if (stroker->handle_wide_strokes) {
                    SW_FT_Vector start;
                    SW_FT_Angle  alpha1;

                    /* determine whether the border radius is greater than the
                     */
                    /* radius of curvature of the original arc */
                    start = border->points[border->num_points - 1];

                    alpha1 = SW_FT_Atan2(end.x - start.x, end.y - start.y);

                    /* is the direction of the border arc opposite to */
                    /* that of the original arc? */
                    if (ft_pos_abs(SW_FT_Angle_Diff(alpha0, alpha1)) >
                        SW_FT_ANGLE_PI / 2) {
                        SW_FT_Angle  beta, gamma;
                        SW_FT_Vector bvec, delta;
                        SW_FT_Fixed  blen, sinA, sinB, alen;

                        /* use the sine rule to find the intersection point */
                        beta =
                            SW_FT_Atan2(arc[2].x - start.x, arc[2].y - start.y);
                        gamma = SW_FT_Atan2(arc[0].x - end.x, arc[0].y - end.y);

                        bvec.x = end.x - start.x;
                        bvec.y = end.y - start.y;

                        blen = SW_FT_Vector_Length(&bvec);

                        sinA = ft_pos_abs(SW_FT_Sin(alpha1 - gamma));
                        sinB = ft_pos_abs(SW_FT_Sin(beta - gamma));

                        alen = SW_FT_MulDiv(blen, sinA, sinB);

                        SW_FT_Vector_From_Polar(&delta, alen, beta);
                        delta.x += start.x;
                        delta.y += start.y;

                        /* circumnavigate the negative sector backwards */
                        border->movable = FALSE;
                        error = ft_stroke_border_lineto(border, &delta, FALSE);
                        if (error) goto Exit;
                        error = ft_stroke_border_lineto(border, &end, FALSE);
                        if (error) goto Exit;
                        error = ft_stroke_border_conicto(border, &ctrl, &start);
                        if (error) goto Exit;
                        /* and then move to the endpoint */
                        error = ft_stroke_border_lineto(border, &end, FALSE);
                        if (error) goto Exit;

                        continue;
                    }

                    /* else fall through */
                }

                /* simply add an arc */
                error = ft_stroke_border_conicto(border, &ctrl, &end);
                if (error) goto Exit;
            }
        }

        arc -= 2;

        stroker->angle_in = angle_out;
    }

    stroker->center = *to;

Exit:
    return error;
}

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_CubicTo(SW_FT_Stroker stroker, SW_FT_Vector* control1,
                                  SW_FT_Vector* control2, SW_FT_Vector* to)
{
    SW_FT_Error   error = 0;
    SW_FT_Vector  bez_stack[37];
    SW_FT_Vector* arc;
    SW_FT_Vector* limit = bez_stack + 32;
    SW_FT_Bool    first_arc = TRUE;

    /* if all control points are coincident, this is a no-op; */
    /* avoid creating a spurious corner */
    if (SW_FT_IS_SMALL(stroker->center.x - control1->x) &&
        SW_FT_IS_SMALL(stroker->center.y - control1->y) &&
        SW_FT_IS_SMALL(control1->x - control2->x) &&
        SW_FT_IS_SMALL(control1->y - control2->y) &&
        SW_FT_IS_SMALL(control2->x - to->x) &&
        SW_FT_IS_SMALL(control2->y - to->y)) {
        stroker->center = *to;
        goto Exit;
    }

    arc = bez_stack;
    arc[0] = *to;
    arc[1] = *control2;
    arc[2] = *control1;
    arc[3] = stroker->center;

    while (arc >= bez_stack) {
        SW_FT_Angle angle_in, angle_mid, angle_out;

        /* initialize with current direction */
        angle_in = angle_out = angle_mid = stroker->angle_in;

        if (arc < limit &&
            !ft_cubic_is_small_enough(arc, &angle_in, &angle_mid, &angle_out)) {
            if (stroker->first_point) stroker->angle_in = angle_in;

            ft_cubic_split(arc);
            arc += 3;
            continue;
        }

        if (first_arc) {
            first_arc = FALSE;

            /* process corner if necessary */
            if (stroker->first_point)
                error = ft_stroker_subpath_start(stroker, angle_in, 0);
            else {
                stroker->angle_out = angle_in;
                error = ft_stroker_process_corner(stroker, 0);
            }
        } else if (ft_pos_abs(SW_FT_Angle_Diff(stroker->angle_in, angle_in)) >
                   SW_FT_SMALL_CUBIC_THRESHOLD / 4) {
            /* if the deviation from one arc to the next is too great, */
            /* add a round corner                                      */
            stroker->center = arc[3];
            stroker->angle_out = angle_in;
            stroker->line_join = SW_FT_STROKER_LINEJOIN_ROUND;

            error = ft_stroker_process_corner(stroker, 0);

            /* reinstate line join style */
            stroker->line_join = stroker->line_join_saved;
        }

        if (error) goto Exit;

        /* the arc's angle is small enough; we can add it directly to each */
        /* border                                                          */
        {
            SW_FT_Vector       ctrl1, ctrl2, end;
            SW_FT_Angle        theta1, phi1, theta2, phi2, rotate, alpha0 = 0;
            SW_FT_Fixed        length1, length2;
            SW_FT_StrokeBorder border;
            SW_FT_Int          side;

            theta1 = SW_FT_Angle_Diff(angle_in, angle_mid) / 2;
            theta2 = SW_FT_Angle_Diff(angle_mid, angle_out) / 2;
            phi1 = ft_angle_mean(angle_in, angle_mid);
            phi2 = ft_angle_mean(angle_mid, angle_out);
            length1 = SW_FT_DivFix(stroker->radius, SW_FT_Cos(theta1));
            length2 = SW_FT_DivFix(stroker->radius, SW_FT_Cos(theta2));

            /* compute direction of original arc */
            if (stroker->handle_wide_strokes)
                alpha0 = SW_FT_Atan2(arc[0].x - arc[3].x, arc[0].y - arc[3].y);

            for (border = stroker->borders, side = 0; side <= 1;
                 side++, border++) {
                rotate = SW_FT_SIDE_TO_ROTATE(side);

                /* compute control points */
                SW_FT_Vector_From_Polar(&ctrl1, length1, phi1 + rotate);
                ctrl1.x += arc[2].x;
                ctrl1.y += arc[2].y;

                SW_FT_Vector_From_Polar(&ctrl2, length2, phi2 + rotate);
                ctrl2.x += arc[1].x;
                ctrl2.y += arc[1].y;

                /* compute end point */
                SW_FT_Vector_From_Polar(&end, stroker->radius,
                                        angle_out + rotate);
                end.x += arc[0].x;
                end.y += arc[0].y;

                if (stroker->handle_wide_strokes) {
                    SW_FT_Vector start;
                    SW_FT_Angle  alpha1;

                    /* determine whether the border radius is greater than the
                     */
                    /* radius of curvature of the original arc */
                    start = border->points[border->num_points - 1];

                    alpha1 = SW_FT_Atan2(end.x - start.x, end.y - start.y);

                    /* is the direction of the border arc opposite to */
                    /* that of the original arc? */
                    if (ft_pos_abs(SW_FT_Angle_Diff(alpha0, alpha1)) >
                        SW_FT_ANGLE_PI / 2) {
                        SW_FT_Angle  beta, gamma;
                        SW_FT_Vector bvec, delta;
                        SW_FT_Fixed  blen, sinA, sinB, alen;

                        /* use the sine rule to find the intersection point */
                        beta =
                            SW_FT_Atan2(arc[3].x - start.x, arc[3].y - start.y);
                        gamma = SW_FT_Atan2(arc[0].x - end.x, arc[0].y - end.y);

                        bvec.x = end.x - start.x;
                        bvec.y = end.y - start.y;

                        blen = SW_FT_Vector_Length(&bvec);

                        sinA = ft_pos_abs(SW_FT_Sin(alpha1 - gamma));
                        sinB = ft_pos_abs(SW_FT_Sin(beta - gamma));

                        alen = SW_FT_MulDiv(blen, sinA, sinB);

                        SW_FT_Vector_From_Polar(&delta, alen, beta);
                        delta.x += start.x;
                        delta.y += start.y;

                        /* circumnavigate the negative sector backwards */
                        border->movable = FALSE;
                        error = ft_stroke_border_lineto(border, &delta, FALSE);
                        if (error) goto Exit;
                        error = ft_stroke_border_lineto(border, &end, FALSE);
                        if (error) goto Exit;
                        error = ft_stroke_border_cubicto(border, &ctrl2, &ctrl1,
                                                         &start);
                        if (error) goto Exit;
                        /* and then move to the endpoint */
                        error = ft_stroke_border_lineto(border, &end, FALSE);
                        if (error) goto Exit;

                        continue;
                    }

                    /* else fall through */
                }

                /* simply add an arc */
                error = ft_stroke_border_cubicto(border, &ctrl1, &ctrl2, &end);
                if (error) goto Exit;
            }
        }

        arc -= 3;

        stroker->angle_in = angle_out;
    }

    stroker->center = *to;

Exit:
    return error;
}

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_BeginSubPath(SW_FT_Stroker stroker, SW_FT_Vector* to,
                                       SW_FT_Bool open)
{
    /* We cannot process the first point, because there is not enough      */
    /* information regarding its corner/cap.  The latter will be processed */
    /* in the `SW_FT_Stroker_EndSubPath' routine.                             */
    /*                                                                     */
    stroker->first_point = TRUE;
    stroker->center = *to;
    stroker->subpath_open = open;

    /* Determine if we need to check whether the border radius is greater */
    /* than the radius of curvature of a curve, to handle this case       */
    /* specially.  This is only required if bevel joins or butt caps may  */
    /* be created, because round & miter joins and round & square caps    */
    /* cover the negative sector created with wide strokes.               */
    stroker->handle_wide_strokes =
        SW_FT_BOOL(stroker->line_join != SW_FT_STROKER_LINEJOIN_ROUND ||
                   (stroker->subpath_open &&
                    stroker->line_cap == SW_FT_STROKER_LINECAP_BUTT));

    /* record the subpath start point for each border */
    stroker->subpath_start = *to;

    stroker->angle_in = 0;

    return 0;
}

static SW_FT_Error ft_stroker_add_reverse_left(SW_FT_Stroker stroker,
                                               SW_FT_Bool    open)
{
    SW_FT_StrokeBorder right = stroker->borders + 0;
    SW_FT_StrokeBorder left = stroker->borders + 1;
    SW_FT_Int          new_points;
    SW_FT_Error        error = 0;

    assert(left->start >= 0);

    new_points = left->num_points - left->start;
    if (new_points > 0) {
        error = ft_stroke_border_grow(right, (SW_FT_UInt)new_points);
        if (error) goto Exit;

        {
            SW_FT_Vector* dst_point = right->points + right->num_points;
            SW_FT_Byte*   dst_tag = right->tags + right->num_points;
            SW_FT_Vector* src_point = left->points + left->num_points - 1;
            SW_FT_Byte*   src_tag = left->tags + left->num_points - 1;

            while (src_point >= left->points + left->start) {
                *dst_point = *src_point;
                *dst_tag = *src_tag;

                if (open)
                    dst_tag[0] &= ~SW_FT_STROKE_TAG_BEGIN_END;
                else {
                    SW_FT_Byte ttag =
                        (SW_FT_Byte)(dst_tag[0] & SW_FT_STROKE_TAG_BEGIN_END);

                    /* switch begin/end tags if necessary */
                    if (ttag == SW_FT_STROKE_TAG_BEGIN ||
                        ttag == SW_FT_STROKE_TAG_END)
                        dst_tag[0] ^= SW_FT_STROKE_TAG_BEGIN_END;
                }

                src_point--;
                src_tag--;
                dst_point++;
                dst_tag++;
            }
        }

        left->num_points = left->start;
        right->num_points += new_points;

        right->movable = FALSE;
        left->movable = FALSE;
    }

Exit:
    return error;
}

/* documentation is in ftstroke.h */

/* there's a lot of magic in this function! */
SW_FT_Error SW_FT_Stroker_EndSubPath(SW_FT_Stroker stroker)
{
    SW_FT_Error error = 0;

    if (stroker->subpath_open) {
        SW_FT_StrokeBorder right = stroker->borders;

        /* All right, this is an opened path, we need to add a cap between */
        /* right & left, add the reverse of left, then add a final cap     */
        /* between left & right.                                           */
        error = ft_stroker_cap(stroker, stroker->angle_in, 0);
        if (error) goto Exit;

        /* add reversed points from `left' to `right' */
        error = ft_stroker_add_reverse_left(stroker, TRUE);
        if (error) goto Exit;

        /* now add the final cap */
        stroker->center = stroker->subpath_start;
        error =
            ft_stroker_cap(stroker, stroker->subpath_angle + SW_FT_ANGLE_PI, 0);
        if (error) goto Exit;

        /* Now end the right subpath accordingly.  The left one is */
        /* rewind and doesn't need further processing.             */
        ft_stroke_border_close(right, FALSE);
    } else {
        SW_FT_Angle turn;
        SW_FT_Int   inside_side;

        /* close the path if needed */
        if (stroker->center.x != stroker->subpath_start.x ||
            stroker->center.y != stroker->subpath_start.y) {
            error = SW_FT_Stroker_LineTo(stroker, &stroker->subpath_start);
            if (error) goto Exit;
        }

        /* process the corner */
        stroker->angle_out = stroker->subpath_angle;
        turn = SW_FT_Angle_Diff(stroker->angle_in, stroker->angle_out);

        /* no specific corner processing is required if the turn is 0 */
        if (turn != 0) {
            /* when we turn to the right, the inside side is 0 */
            inside_side = 0;

            /* otherwise, the inside side is 1 */
            if (turn < 0) inside_side = 1;

            error = ft_stroker_inside(stroker, inside_side,
                                      stroker->subpath_line_length);
            if (error) goto Exit;

            /* process the outside side */
            error = ft_stroker_outside(stroker, 1 - inside_side,
                                       stroker->subpath_line_length);
            if (error) goto Exit;
        }

        /* then end our two subpaths */
        ft_stroke_border_close(stroker->borders + 0, FALSE);
        ft_stroke_border_close(stroker->borders + 1, TRUE);
    }

Exit:
    return error;
}

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_GetBorderCounts(SW_FT_Stroker       stroker,
                                          SW_FT_StrokerBorder border,
                                          SW_FT_UInt*         anum_points,
                                          SW_FT_UInt*         anum_contours)
{
    SW_FT_UInt  num_points = 0, num_contours = 0;
    SW_FT_Error error;

    if (!stroker || border > 1) {
        error = -1;  // SW_FT_THROW( Invalid_Argument );
        goto Exit;
    }

    error = ft_stroke_border_get_counts(stroker->borders + border, &num_points,
                                        &num_contours);
Exit:
    if (anum_points) *anum_points = num_points;

    if (anum_contours) *anum_contours = num_contours;

    return error;
}

/* documentation is in ftstroke.h */

SW_FT_Error SW_FT_Stroker_GetCounts(SW_FT_Stroker stroker,
                                    SW_FT_UInt*   anum_points,
                                    SW_FT_UInt*   anum_contours)
{
    SW_FT_UInt  count1, count2, num_points = 0;
    SW_FT_UInt  count3, count4, num_contours = 0;
    SW_FT_Error error;

    error = ft_stroke_border_get_counts(stroker->borders + 0, &count1, &count2);
    if (error) goto Exit;

    error = ft_stroke_border_get_counts(stroker->borders + 1, &count3, &count4);
    if (error) goto Exit;

    num_points = count1 + count3;
    num_contours = count2 + count4;

Exit:
    *anum_points = num_points;
    *anum_contours = num_contours;
    return error;
}

/* documentation is in ftstroke.h */

void SW_FT_Stroker_ExportBorder(SW_FT_Stroker       stroker,
                                SW_FT_StrokerBorder border,
                                SW_FT_Outline*      outline)
{
    if (border == SW_FT_STROKER_BORDER_LEFT ||
        border == SW_FT_STROKER_BORDER_RIGHT) {
        SW_FT_StrokeBorder sborder = &stroker->borders[border];

        if (sborder->valid) ft_stroke_border_export(sborder, outline);
    }
}

/* documentation is in ftstroke.h */

void SW_FT_Stroker_Export(SW_FT_Stroker stroker, SW_FT_Outline* outline)
{
    SW_FT_Stroker_ExportBorder(stroker, SW_FT_STROKER_BORDER_LEFT, outline);
    SW_FT_Stroker_ExportBorder(stroker, SW_FT_STROKER_BORDER_RIGHT, outline);
}

/* documentation is in ftstroke.h */

/*
 *  The following is very similar to SW_FT_Outline_Decompose, except
 *  that we do support opened paths, and do not scale the outline.
 */
SW_FT_Error SW_FT_Stroker_ParseOutline(SW_FT_Stroker        stroker,
                                       const SW_FT_Outline* outline)
{
    SW_FT_Vector v_last;
    SW_FT_Vector v_control;
    SW_FT_Vector v_start;

    SW_FT_Vector* point;
    SW_FT_Vector* limit;
    char*         tags;

    SW_FT_Error error;

    SW_FT_Int  n;     /* index of contour in outline     */
    SW_FT_UInt first; /* index of first point in contour */
    SW_FT_Int  tag;   /* current point's state           */

    if (!outline || !stroker) return -1;  // SW_FT_THROW( Invalid_Argument );

    SW_FT_Stroker_Rewind(stroker);

    first = 0;

    for (n = 0; n < outline->n_contours; n++) {
        SW_FT_UInt last; /* index of last point in contour */

        last = outline->contours[n];
        limit = outline->points + last;

        /* skip empty points; we don't stroke these */
        if (last <= first) {
            first = last + 1;
            continue;
        }

        v_start = outline->points[first];
        v_last = outline->points[last];

        v_control = v_start;

        point = outline->points + first;
        tags = outline->tags + first;
        tag = SW_FT_CURVE_TAG(tags[0]);

        /* A contour cannot start with a cubic control point! */
        if (tag == SW_FT_CURVE_TAG_CUBIC) goto Invalid_Outline;

        /* check first point to determine origin */
        if (tag == SW_FT_CURVE_TAG_CONIC) {
            /* First point is conic control.  Yes, this happens. */
            if (SW_FT_CURVE_TAG(outline->tags[last]) == SW_FT_CURVE_TAG_ON) {
                /* start at last point if it is on the curve */
                v_start = v_last;
                limit--;
            } else {
                /* if both first and last points are conic, */
                /* start at their middle                    */
                v_start.x = (v_start.x + v_last.x) / 2;
                v_start.y = (v_start.y + v_last.y) / 2;
            }
            point--;
            tags--;
        }

        error = SW_FT_Stroker_BeginSubPath(stroker, &v_start, outline->contours_flag[n]);
        if (error) goto Exit;

        while (point < limit) {
            point++;
            tags++;

            tag = SW_FT_CURVE_TAG(tags[0]);
            switch (tag) {
            case SW_FT_CURVE_TAG_ON: /* emit a single line_to */
            {
                SW_FT_Vector vec;

                vec.x = point->x;
                vec.y = point->y;

                error = SW_FT_Stroker_LineTo(stroker, &vec);
                if (error) goto Exit;
                continue;
            }

            case SW_FT_CURVE_TAG_CONIC: /* consume conic arcs */
                v_control.x = point->x;
                v_control.y = point->y;

            Do_Conic:
                if (point < limit) {
                    SW_FT_Vector vec;
                    SW_FT_Vector v_middle;

                    point++;
                    tags++;
                    tag = SW_FT_CURVE_TAG(tags[0]);

                    vec = point[0];

                    if (tag == SW_FT_CURVE_TAG_ON) {
                        error =
                            SW_FT_Stroker_ConicTo(stroker, &v_control, &vec);
                        if (error) goto Exit;
                        continue;
                    }

                    if (tag != SW_FT_CURVE_TAG_CONIC) goto Invalid_Outline;

                    v_middle.x = (v_control.x + vec.x) / 2;
                    v_middle.y = (v_control.y + vec.y) / 2;

                    error =
                        SW_FT_Stroker_ConicTo(stroker, &v_control, &v_middle);
                    if (error) goto Exit;

                    v_control = vec;
                    goto Do_Conic;
                }

                error = SW_FT_Stroker_ConicTo(stroker, &v_control, &v_start);
                goto Close;

            default: /* SW_FT_CURVE_TAG_CUBIC */
            {
                SW_FT_Vector vec1, vec2;

                if (point + 1 > limit ||
                    SW_FT_CURVE_TAG(tags[1]) != SW_FT_CURVE_TAG_CUBIC)
                    goto Invalid_Outline;

                point += 2;
                tags += 2;

                vec1 = point[-2];
                vec2 = point[-1];

                if (point <= limit) {
                    SW_FT_Vector vec;

                    vec = point[0];

                    error = SW_FT_Stroker_CubicTo(stroker, &vec1, &vec2, &vec);
                    if (error) goto Exit;
                    continue;
                }

                error = SW_FT_Stroker_CubicTo(stroker, &vec1, &vec2, &v_start);
                goto Close;
            }
            }
        }

    Close:
        if (error) goto Exit;

        if (stroker->first_point) {
            stroker->subpath_open = TRUE;
            error = ft_stroker_subpath_start(stroker, 0, 0);
            if (error) goto Exit;
        }

        error = SW_FT_Stroker_EndSubPath(stroker);
        if (error) goto Exit;

        first = last + 1;
    }

    return 0;

Exit:
    return error;

Invalid_Outline:
    return -2;  // SW_FT_THROW( Invalid_Outline );
}

/* END */
