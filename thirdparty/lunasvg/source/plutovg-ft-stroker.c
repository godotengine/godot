
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
/*  license, FTL.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#include "plutovg-ft-stroker.h"
#include "plutovg-ft-math.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                      BEZIER COMPUTATIONS                      *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/

#define PVG_FT_SMALL_CONIC_THRESHOLD (PVG_FT_ANGLE_PI / 6)
#define PVG_FT_SMALL_CUBIC_THRESHOLD (PVG_FT_ANGLE_PI / 8)

#define PVG_FT_EPSILON 2

#define PVG_FT_IS_SMALL(x) ((x) > -PVG_FT_EPSILON && (x) < PVG_FT_EPSILON)

static PVG_FT_Pos ft_pos_abs(PVG_FT_Pos x)
{
    return x >= 0 ? x : -x;
}

static void ft_conic_split(PVG_FT_Vector* base)
{
    PVG_FT_Pos a, b;

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

static PVG_FT_Bool ft_conic_is_small_enough(PVG_FT_Vector* base,
                                           PVG_FT_Angle*  angle_in,
                                           PVG_FT_Angle*  angle_out)
{
    PVG_FT_Vector d1, d2;
    PVG_FT_Angle  theta;
    PVG_FT_Int    close1, close2;

    d1.x = base[1].x - base[2].x;
    d1.y = base[1].y - base[2].y;
    d2.x = base[0].x - base[1].x;
    d2.y = base[0].y - base[1].y;

    close1 = PVG_FT_IS_SMALL(d1.x) && PVG_FT_IS_SMALL(d1.y);
    close2 = PVG_FT_IS_SMALL(d2.x) && PVG_FT_IS_SMALL(d2.y);

    if (close1) {
        if (close2) {
            /* basically a point;                      */
            /* do nothing to retain original direction */
        } else {
            *angle_in = *angle_out = PVG_FT_Atan2(d2.x, d2.y);
        }
    } else /* !close1 */
    {
        if (close2) {
            *angle_in = *angle_out = PVG_FT_Atan2(d1.x, d1.y);
        } else {
            *angle_in = PVG_FT_Atan2(d1.x, d1.y);
            *angle_out = PVG_FT_Atan2(d2.x, d2.y);
        }
    }

    theta = ft_pos_abs(PVG_FT_Angle_Diff(*angle_in, *angle_out));

    return PVG_FT_BOOL(theta < PVG_FT_SMALL_CONIC_THRESHOLD);
}

static void ft_cubic_split(PVG_FT_Vector* base)
{
    PVG_FT_Pos a, b, c;

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
static PVG_FT_Angle ft_angle_mean(PVG_FT_Angle angle1, PVG_FT_Angle angle2)
{
    return angle1 + PVG_FT_Angle_Diff(angle1, angle2) / 2;
}

static PVG_FT_Bool ft_cubic_is_small_enough(PVG_FT_Vector* base,
                                           PVG_FT_Angle*  angle_in,
                                           PVG_FT_Angle*  angle_mid,
                                           PVG_FT_Angle*  angle_out)
{
    PVG_FT_Vector d1, d2, d3;
    PVG_FT_Angle  theta1, theta2;
    PVG_FT_Int    close1, close2, close3;

    d1.x = base[2].x - base[3].x;
    d1.y = base[2].y - base[3].y;
    d2.x = base[1].x - base[2].x;
    d2.y = base[1].y - base[2].y;
    d3.x = base[0].x - base[1].x;
    d3.y = base[0].y - base[1].y;

    close1 = PVG_FT_IS_SMALL(d1.x) && PVG_FT_IS_SMALL(d1.y);
    close2 = PVG_FT_IS_SMALL(d2.x) && PVG_FT_IS_SMALL(d2.y);
    close3 = PVG_FT_IS_SMALL(d3.x) && PVG_FT_IS_SMALL(d3.y);

    if (close1) {
        if (close2) {
            if (close3) {
                /* basically a point;                      */
                /* do nothing to retain original direction */
            } else /* !close3 */
            {
                *angle_in = *angle_mid = *angle_out = PVG_FT_Atan2(d3.x, d3.y);
            }
        } else /* !close2 */
        {
            if (close3) {
                *angle_in = *angle_mid = *angle_out = PVG_FT_Atan2(d2.x, d2.y);
            } else /* !close3 */
            {
                *angle_in = *angle_mid = PVG_FT_Atan2(d2.x, d2.y);
                *angle_out = PVG_FT_Atan2(d3.x, d3.y);
            }
        }
    } else /* !close1 */
    {
        if (close2) {
            if (close3) {
                *angle_in = *angle_mid = *angle_out = PVG_FT_Atan2(d1.x, d1.y);
            } else /* !close3 */
            {
                *angle_in = PVG_FT_Atan2(d1.x, d1.y);
                *angle_out = PVG_FT_Atan2(d3.x, d3.y);
                *angle_mid = ft_angle_mean(*angle_in, *angle_out);
            }
        } else /* !close2 */
        {
            if (close3) {
                *angle_in = PVG_FT_Atan2(d1.x, d1.y);
                *angle_mid = *angle_out = PVG_FT_Atan2(d2.x, d2.y);
            } else /* !close3 */
            {
                *angle_in = PVG_FT_Atan2(d1.x, d1.y);
                *angle_mid = PVG_FT_Atan2(d2.x, d2.y);
                *angle_out = PVG_FT_Atan2(d3.x, d3.y);
            }
        }
    }

    theta1 = ft_pos_abs(PVG_FT_Angle_Diff(*angle_in, *angle_mid));
    theta2 = ft_pos_abs(PVG_FT_Angle_Diff(*angle_mid, *angle_out));

    return PVG_FT_BOOL(theta1 < PVG_FT_SMALL_CUBIC_THRESHOLD &&
                      theta2 < PVG_FT_SMALL_CUBIC_THRESHOLD);
}

/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                       STROKE BORDERS                          *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/

typedef enum PVG_FT_StrokeTags_ {
    PVG_FT_STROKE_TAG_ON = 1,    /* on-curve point  */
    PVG_FT_STROKE_TAG_CUBIC = 2, /* cubic off-point */
    PVG_FT_STROKE_TAG_BEGIN = 4, /* sub-path start  */
    PVG_FT_STROKE_TAG_END = 8    /* sub-path end    */

} PVG_FT_StrokeTags;

#define PVG_FT_STROKE_TAG_BEGIN_END \
    (PVG_FT_STROKE_TAG_BEGIN | PVG_FT_STROKE_TAG_END)

typedef struct PVG_FT_StrokeBorderRec_ {
    PVG_FT_UInt    num_points;
    PVG_FT_UInt    max_points;
    PVG_FT_Vector* points;
    PVG_FT_Byte*   tags;
    PVG_FT_Bool    movable; /* TRUE for ends of lineto borders */
    PVG_FT_Int     start;   /* index of current sub-path start point */
    PVG_FT_Bool    valid;

} PVG_FT_StrokeBorderRec, *PVG_FT_StrokeBorder;

static PVG_FT_Error ft_stroke_border_grow(PVG_FT_StrokeBorder border,
                                         PVG_FT_UInt         new_points)
{
    PVG_FT_UInt  old_max = border->max_points;
    PVG_FT_UInt  new_max = border->num_points + new_points;
    PVG_FT_Error error = 0;

    if (new_max > old_max) {
        PVG_FT_UInt cur_max = old_max;

        while (cur_max < new_max) cur_max += (cur_max >> 1) + 16;

        border->points = (PVG_FT_Vector*)realloc(border->points,
                                                cur_max * sizeof(PVG_FT_Vector));
        border->tags =
            (PVG_FT_Byte*)realloc(border->tags, cur_max * sizeof(PVG_FT_Byte));

        if (!border->points || !border->tags) goto Exit;

        border->max_points = cur_max;
    }

Exit:
    return error;
}

static void ft_stroke_border_close(PVG_FT_StrokeBorder border,
                                   PVG_FT_Bool         reverse)
{
    PVG_FT_UInt start = border->start;
    PVG_FT_UInt count = border->num_points;

    assert(border->start >= 0);

    /* don't record empty paths! */
    if (count <= start + 1U)
        border->num_points = start;
    else {
        /* copy the last point to the start of this sub-path, since */
        /* it contains the `adjusted' starting coordinates          */
        border->num_points = --count;
        border->points[start] = border->points[count];
        border->tags[start]   = border->tags[count];

        if (reverse) {
            /* reverse the points */
            {
                PVG_FT_Vector* vec1 = border->points + start + 1;
                PVG_FT_Vector* vec2 = border->points + count - 1;

                for (; vec1 < vec2; vec1++, vec2--) {
                    PVG_FT_Vector tmp;

                    tmp = *vec1;
                    *vec1 = *vec2;
                    *vec2 = tmp;
                }
            }

            /* then the tags */
            {
                PVG_FT_Byte* tag1 = border->tags + start + 1;
                PVG_FT_Byte* tag2 = border->tags + count - 1;

                for (; tag1 < tag2; tag1++, tag2--) {
                    PVG_FT_Byte tmp;

                    tmp = *tag1;
                    *tag1 = *tag2;
                    *tag2 = tmp;
                }
            }
        }

        border->tags[start] |= PVG_FT_STROKE_TAG_BEGIN;
        border->tags[count - 1] |= PVG_FT_STROKE_TAG_END;
    }

    border->start = -1;
    border->movable = FALSE;
}

static PVG_FT_Error ft_stroke_border_lineto(PVG_FT_StrokeBorder border,
                                           PVG_FT_Vector* to, PVG_FT_Bool movable)
{
    PVG_FT_Error error = 0;

    assert(border->start >= 0);

    if (border->movable) {
        /* move last point */
        border->points[border->num_points - 1] = *to;
    } else {
        /* don't add zero-length lineto, but always add moveto */
        if (border->num_points > border->start &&
            PVG_FT_IS_SMALL(border->points[border->num_points - 1].x - to->x) &&
            PVG_FT_IS_SMALL(border->points[border->num_points - 1].y - to->y))
            return error;

        /* add one point */
        error = ft_stroke_border_grow(border, 1);
        if (!error) {
            PVG_FT_Vector* vec = border->points + border->num_points;
            PVG_FT_Byte*   tag = border->tags + border->num_points;

            vec[0] = *to;
            tag[0] = PVG_FT_STROKE_TAG_ON;

            border->num_points += 1;
        }
    }
    border->movable = movable;
    return error;
}

static PVG_FT_Error ft_stroke_border_conicto(PVG_FT_StrokeBorder border,
                                            PVG_FT_Vector*      control,
                                            PVG_FT_Vector*      to)
{
    PVG_FT_Error error;

    assert(border->start >= 0);

    error = ft_stroke_border_grow(border, 2);
    if (!error) {
        PVG_FT_Vector* vec = border->points + border->num_points;
        PVG_FT_Byte*   tag = border->tags + border->num_points;

        vec[0] = *control;
        vec[1] = *to;

        tag[0] = 0;
        tag[1] = PVG_FT_STROKE_TAG_ON;

        border->num_points += 2;
    }

    border->movable = FALSE;

    return error;
}

static PVG_FT_Error ft_stroke_border_cubicto(PVG_FT_StrokeBorder border,
                                            PVG_FT_Vector*      control1,
                                            PVG_FT_Vector*      control2,
                                            PVG_FT_Vector*      to)
{
    PVG_FT_Error error;

    assert(border->start >= 0);

    error = ft_stroke_border_grow(border, 3);
    if (!error) {
        PVG_FT_Vector* vec = border->points + border->num_points;
        PVG_FT_Byte*   tag = border->tags + border->num_points;

        vec[0] = *control1;
        vec[1] = *control2;
        vec[2] = *to;

        tag[0] = PVG_FT_STROKE_TAG_CUBIC;
        tag[1] = PVG_FT_STROKE_TAG_CUBIC;
        tag[2] = PVG_FT_STROKE_TAG_ON;

        border->num_points += 3;
    }

    border->movable = FALSE;

    return error;
}

#define PVG_FT_ARC_CUBIC_ANGLE (PVG_FT_ANGLE_PI / 2)


static PVG_FT_Error
ft_stroke_border_arcto( PVG_FT_StrokeBorder  border,
                        PVG_FT_Vector*       center,
                        PVG_FT_Fixed         radius,
                        PVG_FT_Angle         angle_start,
                        PVG_FT_Angle         angle_diff )
{
    PVG_FT_Fixed   coef;
    PVG_FT_Vector  a0, a1, a2, a3;
    PVG_FT_Int     i, arcs = 1;
    PVG_FT_Error   error = 0;


    /* number of cubic arcs to draw */
    while (  angle_diff > PVG_FT_ARC_CUBIC_ANGLE * arcs ||
            -angle_diff > PVG_FT_ARC_CUBIC_ANGLE * arcs )
      arcs++;

    /* control tangents */
    coef  = PVG_FT_Tan( angle_diff / ( 4 * arcs ) );
    coef += coef / 3;

    /* compute start and first control point */
    PVG_FT_Vector_From_Polar( &a0, radius, angle_start );
    a1.x = PVG_FT_MulFix( -a0.y, coef );
    a1.y = PVG_FT_MulFix(  a0.x, coef );

    a0.x += center->x;
    a0.y += center->y;
    a1.x += a0.x;
    a1.y += a0.y;

    for ( i = 1; i <= arcs; i++ )
    {
      /* compute end and second control point */
      PVG_FT_Vector_From_Polar( &a3, radius,
                            angle_start + i * angle_diff / arcs );
      a2.x = PVG_FT_MulFix(  a3.y, coef );
      a2.y = PVG_FT_MulFix( -a3.x, coef );

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

static PVG_FT_Error ft_stroke_border_moveto(PVG_FT_StrokeBorder border,
                                           PVG_FT_Vector*      to)
{
    /* close current open path if any ? */
    if (border->start >= 0) ft_stroke_border_close(border, FALSE);

    border->start = border->num_points;
    border->movable = FALSE;

    return ft_stroke_border_lineto(border, to, FALSE);
}

static void ft_stroke_border_init(PVG_FT_StrokeBorder border)
{
    border->points = NULL;
    border->tags = NULL;

    border->num_points = 0;
    border->max_points = 0;
    border->start = -1;
    border->valid = FALSE;
}

static void ft_stroke_border_reset(PVG_FT_StrokeBorder border)
{
    border->num_points = 0;
    border->start = -1;
    border->valid = FALSE;
}

static void ft_stroke_border_done(PVG_FT_StrokeBorder border)
{
    free(border->points);
    free(border->tags);

    border->num_points = 0;
    border->max_points = 0;
    border->start = -1;
    border->valid = FALSE;
}

static PVG_FT_Error ft_stroke_border_get_counts(PVG_FT_StrokeBorder border,
                                               PVG_FT_UInt*        anum_points,
                                               PVG_FT_UInt*        anum_contours)
{
    PVG_FT_Error error = 0;
    PVG_FT_UInt  num_points = 0;
    PVG_FT_UInt  num_contours = 0;

    PVG_FT_UInt    count = border->num_points;
    PVG_FT_Vector* point = border->points;
    PVG_FT_Byte*   tags = border->tags;
    PVG_FT_Int     in_contour = 0;

    for (; count > 0; count--, num_points++, point++, tags++) {
        if (tags[0] & PVG_FT_STROKE_TAG_BEGIN) {
            if (in_contour != 0) goto Fail;

            in_contour = 1;
        } else if (in_contour == 0)
            goto Fail;

        if (tags[0] & PVG_FT_STROKE_TAG_END) {
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

static void ft_stroke_border_export(PVG_FT_StrokeBorder border,
                                    PVG_FT_Outline*     outline)
{
    /* copy point locations */
    memcpy(outline->points + outline->n_points, border->points,
           border->num_points * sizeof(PVG_FT_Vector));

    /* copy tags */
    {
        PVG_FT_UInt  count = border->num_points;
        PVG_FT_Byte* read = border->tags;
        PVG_FT_Byte* write = (PVG_FT_Byte*)outline->tags + outline->n_points;

        for (; count > 0; count--, read++, write++) {
            if (*read & PVG_FT_STROKE_TAG_ON)
                *write = PVG_FT_CURVE_TAG_ON;
            else if (*read & PVG_FT_STROKE_TAG_CUBIC)
                *write = PVG_FT_CURVE_TAG_CUBIC;
            else
                *write = PVG_FT_CURVE_TAG_CONIC;
        }
    }

    /* copy contours */
    {
        PVG_FT_UInt   count = border->num_points;
        PVG_FT_Byte*  tags = border->tags;
        PVG_FT_Int* write = outline->contours + outline->n_contours;
        PVG_FT_Int  idx = (PVG_FT_Int)outline->n_points;

        for (; count > 0; count--, tags++, idx++) {
            if (*tags & PVG_FT_STROKE_TAG_END) {
                *write++ = idx;
                outline->n_contours++;
            }
        }
    }

    outline->n_points = (int)(outline->n_points + border->num_points);

    assert(PVG_FT_Outline_Check(outline) == 0);
}

/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                           STROKER                             *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/

#define PVG_FT_SIDE_TO_ROTATE(s) (PVG_FT_ANGLE_PI2 - (s)*PVG_FT_ANGLE_PI)

typedef struct PVG_FT_StrokerRec_ {
    PVG_FT_Angle  angle_in;            /* direction into curr join */
    PVG_FT_Angle  angle_out;           /* direction out of join  */
    PVG_FT_Vector center;              /* current position */
    PVG_FT_Fixed  line_length;         /* length of last lineto */
    PVG_FT_Bool   first_point;         /* is this the start? */
    PVG_FT_Bool   subpath_open;        /* is the subpath open? */
    PVG_FT_Angle  subpath_angle;       /* subpath start direction */
    PVG_FT_Vector subpath_start;       /* subpath start position */
    PVG_FT_Fixed  subpath_line_length; /* subpath start lineto len */
    PVG_FT_Bool   handle_wide_strokes; /* use wide strokes logic? */

    PVG_FT_Stroker_LineCap  line_cap;
    PVG_FT_Stroker_LineJoin line_join;
    PVG_FT_Stroker_LineJoin line_join_saved;
    PVG_FT_Fixed            miter_limit;
    PVG_FT_Fixed            radius;

    PVG_FT_StrokeBorderRec borders[2];
} PVG_FT_StrokerRec;

/* documentation is in ftstroke.h */

PVG_FT_Error PVG_FT_Stroker_New(PVG_FT_Stroker* astroker)
{
    PVG_FT_Error   error = 0; /* assigned in PVG_FT_NEW */
    PVG_FT_Stroker stroker = NULL;

    stroker = (PVG_FT_StrokerRec*)calloc(1, sizeof(PVG_FT_StrokerRec));
    if (stroker) {
        ft_stroke_border_init(&stroker->borders[0]);
        ft_stroke_border_init(&stroker->borders[1]);
    }

    *astroker = stroker;

    return error;
}

void PVG_FT_Stroker_Rewind(PVG_FT_Stroker stroker)
{
    if (stroker) {
        ft_stroke_border_reset(&stroker->borders[0]);
        ft_stroke_border_reset(&stroker->borders[1]);
    }
}

/* documentation is in ftstroke.h */

void PVG_FT_Stroker_Set(PVG_FT_Stroker stroker, PVG_FT_Fixed radius,
                       PVG_FT_Stroker_LineCap  line_cap,
                       PVG_FT_Stroker_LineJoin line_join,
                       PVG_FT_Fixed            miter_limit)
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

    PVG_FT_Stroker_Rewind(stroker);
}

/* documentation is in ftstroke.h */

void PVG_FT_Stroker_Done(PVG_FT_Stroker stroker)
{
    if (stroker) {
        ft_stroke_border_done(&stroker->borders[0]);
        ft_stroke_border_done(&stroker->borders[1]);

        free(stroker);
    }
}

/* create a circular arc at a corner or cap */
static PVG_FT_Error ft_stroker_arcto(PVG_FT_Stroker stroker, PVG_FT_Int side)
{
    PVG_FT_Angle        total, rotate;
    PVG_FT_Fixed        radius = stroker->radius;
    PVG_FT_Error        error = 0;
    PVG_FT_StrokeBorder border = stroker->borders + side;

    rotate = PVG_FT_SIDE_TO_ROTATE(side);

    total = PVG_FT_Angle_Diff(stroker->angle_in, stroker->angle_out);
    if (total == PVG_FT_ANGLE_PI) total = -rotate * 2;

    error = ft_stroke_border_arcto(border, &stroker->center, radius,
                                   stroker->angle_in + rotate, total);
    border->movable = FALSE;
    return error;
}

/* add a cap at the end of an opened path */
static PVG_FT_Error
ft_stroker_cap(PVG_FT_Stroker stroker,
               PVG_FT_Angle angle,
               PVG_FT_Int side)
{
    PVG_FT_Error error = 0;

    if (stroker->line_cap == PVG_FT_STROKER_LINECAP_ROUND)
    {
        /* add a round cap */
        stroker->angle_in = angle;
        stroker->angle_out = angle + PVG_FT_ANGLE_PI;

        error = ft_stroker_arcto(stroker, side);
    }
    else
    {
        /* add a square or butt cap */
        PVG_FT_Vector        middle, delta;
        PVG_FT_Fixed         radius = stroker->radius;
        PVG_FT_StrokeBorder  border = stroker->borders + side;

        /* compute middle point and first angle point */
        PVG_FT_Vector_From_Polar( &middle, radius, angle );
        delta.x = side ?  middle.y : -middle.y;
        delta.y = side ? -middle.x :  middle.x;

        if ( stroker->line_cap == PVG_FT_STROKER_LINECAP_SQUARE )
        {
            middle.x += stroker->center.x;
            middle.y += stroker->center.y;
        }
        else  /* PVG_FT_STROKER_LINECAP_BUTT */
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
static PVG_FT_Error ft_stroker_inside(PVG_FT_Stroker stroker, PVG_FT_Int side,
                                     PVG_FT_Fixed line_length)
{
    PVG_FT_StrokeBorder border = stroker->borders + side;
    PVG_FT_Angle        phi, theta, rotate;
    PVG_FT_Fixed        length;
    PVG_FT_Vector       sigma = {0, 0};
    PVG_FT_Vector       delta;
    PVG_FT_Error        error = 0;
    PVG_FT_Bool         intersect; /* use intersection of lines? */

    rotate = PVG_FT_SIDE_TO_ROTATE(side);

    theta = PVG_FT_Angle_Diff(stroker->angle_in, stroker->angle_out) / 2;

    /* Only intersect borders if between two lineto's and both */
    /* lines are long enough (line_length is zero for curves). */
    if (!border->movable || line_length == 0  ||
         theta > 0x59C000 || theta < -0x59C000 )
        intersect = FALSE;
    else {
      /* compute minimum required length of lines */
      PVG_FT_Fixed  min_length;


      PVG_FT_Vector_Unit( &sigma, theta );
      min_length =
        ft_pos_abs( PVG_FT_MulDiv( stroker->radius, sigma.y, sigma.x ) );

      intersect = PVG_FT_BOOL( min_length                         &&
                           stroker->line_length >= min_length &&
                           line_length          >= min_length );
    }

    if (!intersect) {
        PVG_FT_Vector_From_Polar(&delta, stroker->radius,
                                stroker->angle_out + rotate);
        delta.x += stroker->center.x;
        delta.y += stroker->center.y;

        border->movable = FALSE;
    } else {
        /* compute median angle */
        phi = stroker->angle_in + theta + rotate;

      length = PVG_FT_DivFix( stroker->radius, sigma.x );

      PVG_FT_Vector_From_Polar( &delta, length, phi );
      delta.x += stroker->center.x;
      delta.y += stroker->center.y;
    }

    error = ft_stroke_border_lineto(border, &delta, FALSE);

    return error;
}

  /* process an outside corner, i.e. compute bevel/miter/round */
static PVG_FT_Error
ft_stroker_outside( PVG_FT_Stroker  stroker,
                    PVG_FT_Int      side,
                    PVG_FT_Fixed    line_length )
{
    PVG_FT_StrokeBorder  border = stroker->borders + side;
    PVG_FT_Error         error;
    PVG_FT_Angle         rotate;


    if ( stroker->line_join == PVG_FT_STROKER_LINEJOIN_ROUND )
      error = ft_stroker_arcto( stroker, side );
    else
    {
      /* this is a mitered (pointed) or beveled (truncated) corner */
      PVG_FT_Fixed   radius = stroker->radius;
      PVG_FT_Vector  sigma = {0, 0};
      PVG_FT_Angle   theta = 0, phi = 0;
      PVG_FT_Bool    bevel, fixed_bevel;


      rotate = PVG_FT_SIDE_TO_ROTATE( side );

      bevel =
        PVG_FT_BOOL( stroker->line_join == PVG_FT_STROKER_LINEJOIN_BEVEL );

      fixed_bevel =
        PVG_FT_BOOL( stroker->line_join != PVG_FT_STROKER_LINEJOIN_MITER_VARIABLE );

      /* check miter limit first */
      if ( !bevel )
      {
        theta = PVG_FT_Angle_Diff( stroker->angle_in, stroker->angle_out ) / 2;

        if ( theta == PVG_FT_ANGLE_PI2 )
          theta = -rotate;

        phi    = stroker->angle_in + theta + rotate;

        PVG_FT_Vector_From_Polar( &sigma, stroker->miter_limit, theta );

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
          PVG_FT_Vector  delta;


          /* add bevel */
          PVG_FT_Vector_From_Polar( &delta,
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
          PVG_FT_Vector  middle, delta;
          PVG_FT_Fixed   coef;


          /* compute middle point and first angle point */
          PVG_FT_Vector_From_Polar( &middle,
                                   PVG_FT_MulFix( radius, stroker->miter_limit ),
                                   phi );

          coef    = PVG_FT_DivFix(  0x10000L - sigma.x, sigma.y );
          delta.x = PVG_FT_MulFix(  middle.y, coef );
          delta.y = PVG_FT_MulFix( -middle.x, coef );

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
            PVG_FT_Vector_From_Polar( &delta,
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
        PVG_FT_Fixed   length;
        PVG_FT_Vector  delta;


        length = PVG_FT_MulDiv( stroker->radius, stroker->miter_limit, sigma.x );

        PVG_FT_Vector_From_Polar( &delta, length, phi );
        delta.x += stroker->center.x;
        delta.y += stroker->center.y;

        error = ft_stroke_border_lineto( border, &delta, FALSE );
        if ( error )
          goto Exit;

        /* now add an end point; only needed if not lineto */
        /* (line_length is zero for curves)                */
        if ( line_length == 0 )
        {
          PVG_FT_Vector_From_Polar( &delta,
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

static PVG_FT_Error ft_stroker_process_corner(PVG_FT_Stroker stroker,
                                             PVG_FT_Fixed   line_length)
{
    PVG_FT_Error error = 0;
    PVG_FT_Angle turn;
    PVG_FT_Int   inside_side;

    turn = PVG_FT_Angle_Diff(stroker->angle_in, stroker->angle_out);

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
static PVG_FT_Error ft_stroker_subpath_start(PVG_FT_Stroker stroker,
                                            PVG_FT_Angle   start_angle,
                                            PVG_FT_Fixed   line_length)
{
    PVG_FT_Vector       delta;
    PVG_FT_Vector       point;
    PVG_FT_Error        error;
    PVG_FT_StrokeBorder border;

    PVG_FT_Vector_From_Polar(&delta, stroker->radius,
                            start_angle + PVG_FT_ANGLE_PI2);

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

PVG_FT_Error PVG_FT_Stroker_LineTo(PVG_FT_Stroker stroker, PVG_FT_Vector* to)
{
    PVG_FT_Error        error = 0;
    PVG_FT_StrokeBorder border;
    PVG_FT_Vector       delta;
    PVG_FT_Angle        angle;
    PVG_FT_Int          side;
    PVG_FT_Fixed        line_length;

    delta.x = to->x - stroker->center.x;
    delta.y = to->y - stroker->center.y;

    /* a zero-length lineto is a no-op; avoid creating a spurious corner */
    if (delta.x == 0 && delta.y == 0) goto Exit;

    /* compute length of line */
    line_length = PVG_FT_Vector_Length(&delta);

    angle = PVG_FT_Atan2(delta.x, delta.y);
    PVG_FT_Vector_From_Polar(&delta, stroker->radius, angle + PVG_FT_ANGLE_PI2);

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
        PVG_FT_Vector point;

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

PVG_FT_Error PVG_FT_Stroker_ConicTo(PVG_FT_Stroker stroker, PVG_FT_Vector* control,
                                  PVG_FT_Vector* to)
{
    PVG_FT_Error   error = 0;
    PVG_FT_Vector  bez_stack[34];
    PVG_FT_Vector* arc;
    PVG_FT_Vector* limit = bez_stack + 30;
    PVG_FT_Bool    first_arc = TRUE;

    /* if all control points are coincident, this is a no-op; */
    /* avoid creating a spurious corner                       */
    if (PVG_FT_IS_SMALL(stroker->center.x - control->x) &&
        PVG_FT_IS_SMALL(stroker->center.y - control->y) &&
        PVG_FT_IS_SMALL(control->x - to->x) &&
        PVG_FT_IS_SMALL(control->y - to->y)) {
        stroker->center = *to;
        goto Exit;
    }

    arc = bez_stack;
    arc[0] = *to;
    arc[1] = *control;
    arc[2] = stroker->center;

    while (arc >= bez_stack) {
        PVG_FT_Angle angle_in, angle_out;

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
        } else if (ft_pos_abs(PVG_FT_Angle_Diff(stroker->angle_in, angle_in)) >
                   PVG_FT_SMALL_CONIC_THRESHOLD / 4) {
            /* if the deviation from one arc to the next is too great, */
            /* add a round corner                                      */
            stroker->center = arc[2];
            stroker->angle_out = angle_in;
            stroker->line_join = PVG_FT_STROKER_LINEJOIN_ROUND;

            error = ft_stroker_process_corner(stroker, 0);

            /* reinstate line join style */
            stroker->line_join = stroker->line_join_saved;
        }

        if (error) goto Exit;

        /* the arc's angle is small enough; we can add it directly to each */
        /* border                                                          */
        {
            PVG_FT_Vector       ctrl, end;
            PVG_FT_Angle        theta, phi, rotate, alpha0 = 0;
            PVG_FT_Fixed        length;
            PVG_FT_StrokeBorder border;
            PVG_FT_Int          side;

            theta = PVG_FT_Angle_Diff(angle_in, angle_out) / 2;
            phi = angle_in + theta;
            length = PVG_FT_DivFix(stroker->radius, PVG_FT_Cos(theta));

            /* compute direction of original arc */
            if (stroker->handle_wide_strokes)
                alpha0 = PVG_FT_Atan2(arc[0].x - arc[2].x, arc[0].y - arc[2].y);

            for (border = stroker->borders, side = 0; side <= 1;
                 side++, border++) {
                rotate = PVG_FT_SIDE_TO_ROTATE(side);

                /* compute control point */
                PVG_FT_Vector_From_Polar(&ctrl, length, phi + rotate);
                ctrl.x += arc[1].x;
                ctrl.y += arc[1].y;

                /* compute end point */
                PVG_FT_Vector_From_Polar(&end, stroker->radius,
                                        angle_out + rotate);
                end.x += arc[0].x;
                end.y += arc[0].y;

                if (stroker->handle_wide_strokes) {
                    PVG_FT_Vector start;
                    PVG_FT_Angle  alpha1;

                    /* determine whether the border radius is greater than the
                     */
                    /* radius of curvature of the original arc */
                    start = border->points[border->num_points - 1];

                    alpha1 = PVG_FT_Atan2(end.x - start.x, end.y - start.y);

                    /* is the direction of the border arc opposite to */
                    /* that of the original arc? */
                    if (ft_pos_abs(PVG_FT_Angle_Diff(alpha0, alpha1)) >
                        PVG_FT_ANGLE_PI / 2) {
                        PVG_FT_Angle  beta, gamma;
                        PVG_FT_Vector bvec, delta;
                        PVG_FT_Fixed  blen, sinA, sinB, alen;

                        /* use the sine rule to find the intersection point */
                        beta =
                            PVG_FT_Atan2(arc[2].x - start.x, arc[2].y - start.y);
                        gamma = PVG_FT_Atan2(arc[0].x - end.x, arc[0].y - end.y);

                        bvec.x = end.x - start.x;
                        bvec.y = end.y - start.y;

                        blen = PVG_FT_Vector_Length(&bvec);

                        sinA = ft_pos_abs(PVG_FT_Sin(alpha1 - gamma));
                        sinB = ft_pos_abs(PVG_FT_Sin(beta - gamma));

                        alen = PVG_FT_MulDiv(blen, sinA, sinB);

                        PVG_FT_Vector_From_Polar(&delta, alen, beta);
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
    stroker->line_length = 0;

Exit:
    return error;
}

/* documentation is in ftstroke.h */

PVG_FT_Error PVG_FT_Stroker_CubicTo(PVG_FT_Stroker stroker, PVG_FT_Vector* control1,
                                  PVG_FT_Vector* control2, PVG_FT_Vector* to)
{
    PVG_FT_Error   error = 0;
    PVG_FT_Vector  bez_stack[37];
    PVG_FT_Vector* arc;
    PVG_FT_Vector* limit = bez_stack + 32;
    PVG_FT_Bool    first_arc = TRUE;

    /* if all control points are coincident, this is a no-op; */
    /* avoid creating a spurious corner */
    if (PVG_FT_IS_SMALL(stroker->center.x - control1->x) &&
        PVG_FT_IS_SMALL(stroker->center.y - control1->y) &&
        PVG_FT_IS_SMALL(control1->x - control2->x) &&
        PVG_FT_IS_SMALL(control1->y - control2->y) &&
        PVG_FT_IS_SMALL(control2->x - to->x) &&
        PVG_FT_IS_SMALL(control2->y - to->y)) {
        stroker->center = *to;
        goto Exit;
    }

    arc = bez_stack;
    arc[0] = *to;
    arc[1] = *control2;
    arc[2] = *control1;
    arc[3] = stroker->center;

    while (arc >= bez_stack) {
        PVG_FT_Angle angle_in, angle_mid, angle_out;

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
        } else if (ft_pos_abs(PVG_FT_Angle_Diff(stroker->angle_in, angle_in)) >
                   PVG_FT_SMALL_CUBIC_THRESHOLD / 4) {
            /* if the deviation from one arc to the next is too great, */
            /* add a round corner                                      */
            stroker->center = arc[3];
            stroker->angle_out = angle_in;
            stroker->line_join = PVG_FT_STROKER_LINEJOIN_ROUND;

            error = ft_stroker_process_corner(stroker, 0);

            /* reinstate line join style */
            stroker->line_join = stroker->line_join_saved;
        }

        if (error) goto Exit;

        /* the arc's angle is small enough; we can add it directly to each */
        /* border                                                          */
        {
            PVG_FT_Vector       ctrl1, ctrl2, end;
            PVG_FT_Angle        theta1, phi1, theta2, phi2, rotate, alpha0 = 0;
            PVG_FT_Fixed        length1, length2;
            PVG_FT_StrokeBorder border;
            PVG_FT_Int          side;

            theta1 = PVG_FT_Angle_Diff(angle_in, angle_mid) / 2;
            theta2 = PVG_FT_Angle_Diff(angle_mid, angle_out) / 2;
            phi1 = ft_angle_mean(angle_in, angle_mid);
            phi2 = ft_angle_mean(angle_mid, angle_out);
            length1 = PVG_FT_DivFix(stroker->radius, PVG_FT_Cos(theta1));
            length2 = PVG_FT_DivFix(stroker->radius, PVG_FT_Cos(theta2));

            /* compute direction of original arc */
            if (stroker->handle_wide_strokes)
                alpha0 = PVG_FT_Atan2(arc[0].x - arc[3].x, arc[0].y - arc[3].y);

            for (border = stroker->borders, side = 0; side <= 1;
                 side++, border++) {
                rotate = PVG_FT_SIDE_TO_ROTATE(side);

                /* compute control points */
                PVG_FT_Vector_From_Polar(&ctrl1, length1, phi1 + rotate);
                ctrl1.x += arc[2].x;
                ctrl1.y += arc[2].y;

                PVG_FT_Vector_From_Polar(&ctrl2, length2, phi2 + rotate);
                ctrl2.x += arc[1].x;
                ctrl2.y += arc[1].y;

                /* compute end point */
                PVG_FT_Vector_From_Polar(&end, stroker->radius,
                                        angle_out + rotate);
                end.x += arc[0].x;
                end.y += arc[0].y;

                if (stroker->handle_wide_strokes) {
                    PVG_FT_Vector start;
                    PVG_FT_Angle  alpha1;

                    /* determine whether the border radius is greater than the
                     */
                    /* radius of curvature of the original arc */
                    start = border->points[border->num_points - 1];

                    alpha1 = PVG_FT_Atan2(end.x - start.x, end.y - start.y);

                    /* is the direction of the border arc opposite to */
                    /* that of the original arc? */
                    if (ft_pos_abs(PVG_FT_Angle_Diff(alpha0, alpha1)) >
                        PVG_FT_ANGLE_PI / 2) {
                        PVG_FT_Angle  beta, gamma;
                        PVG_FT_Vector bvec, delta;
                        PVG_FT_Fixed  blen, sinA, sinB, alen;

                        /* use the sine rule to find the intersection point */
                        beta =
                            PVG_FT_Atan2(arc[3].x - start.x, arc[3].y - start.y);
                        gamma = PVG_FT_Atan2(arc[0].x - end.x, arc[0].y - end.y);

                        bvec.x = end.x - start.x;
                        bvec.y = end.y - start.y;

                        blen = PVG_FT_Vector_Length(&bvec);

                        sinA = ft_pos_abs(PVG_FT_Sin(alpha1 - gamma));
                        sinB = ft_pos_abs(PVG_FT_Sin(beta - gamma));

                        alen = PVG_FT_MulDiv(blen, sinA, sinB);

                        PVG_FT_Vector_From_Polar(&delta, alen, beta);
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
    stroker->line_length = 0;

Exit:
    return error;
}

/* documentation is in ftstroke.h */

PVG_FT_Error PVG_FT_Stroker_BeginSubPath(PVG_FT_Stroker stroker, PVG_FT_Vector* to,
                                       PVG_FT_Bool open)
{
    /* We cannot process the first point, because there is not enough      */
    /* information regarding its corner/cap.  The latter will be processed */
    /* in the `PVG_FT_Stroker_EndSubPath' routine.                             */
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
        PVG_FT_BOOL(stroker->line_join != PVG_FT_STROKER_LINEJOIN_ROUND ||
                   (stroker->subpath_open &&
                    stroker->line_cap == PVG_FT_STROKER_LINECAP_BUTT));

    /* record the subpath start point for each border */
    stroker->subpath_start = *to;

    stroker->angle_in = 0;

    return 0;
}

static PVG_FT_Error ft_stroker_add_reverse_left(PVG_FT_Stroker stroker,
                                               PVG_FT_Bool    open)
{
    PVG_FT_StrokeBorder right = stroker->borders + 0;
    PVG_FT_StrokeBorder left = stroker->borders + 1;
    PVG_FT_Int          new_points;
    PVG_FT_Error        error = 0;

    assert(left->start >= 0);

    new_points = left->num_points - left->start;
    if (new_points > 0) {
        error = ft_stroke_border_grow(right, (PVG_FT_UInt)new_points);
        if (error) goto Exit;

        {
            PVG_FT_Vector* dst_point = right->points + right->num_points;
            PVG_FT_Byte*   dst_tag = right->tags + right->num_points;
            PVG_FT_Vector* src_point = left->points + left->num_points - 1;
            PVG_FT_Byte*   src_tag = left->tags + left->num_points - 1;

            while (src_point >= left->points + left->start) {
                *dst_point = *src_point;
                *dst_tag = *src_tag;

                if (open)
                    dst_tag[0] &= ~PVG_FT_STROKE_TAG_BEGIN_END;
                else {
                    PVG_FT_Byte ttag =
                        (PVG_FT_Byte)(dst_tag[0] & PVG_FT_STROKE_TAG_BEGIN_END);

                    /* switch begin/end tags if necessary */
                    if (ttag == PVG_FT_STROKE_TAG_BEGIN ||
                        ttag == PVG_FT_STROKE_TAG_END)
                        dst_tag[0] ^= PVG_FT_STROKE_TAG_BEGIN_END;
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
PVG_FT_Error PVG_FT_Stroker_EndSubPath(PVG_FT_Stroker stroker)
{
    PVG_FT_Error error = 0;

    if (stroker->subpath_open) {
        PVG_FT_StrokeBorder right = stroker->borders;

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
            ft_stroker_cap(stroker, stroker->subpath_angle + PVG_FT_ANGLE_PI, 0);
        if (error) goto Exit;

        /* Now end the right subpath accordingly.  The left one is */
        /* rewind and doesn't need further processing.             */
        ft_stroke_border_close(right, FALSE);
    } else {
        PVG_FT_Angle turn;
        PVG_FT_Int   inside_side;

        /* close the path if needed */
        if (stroker->center.x != stroker->subpath_start.x ||
            stroker->center.y != stroker->subpath_start.y) {
            error = PVG_FT_Stroker_LineTo(stroker, &stroker->subpath_start);
            if (error) goto Exit;
        }

        /* process the corner */
        stroker->angle_out = stroker->subpath_angle;
        turn = PVG_FT_Angle_Diff(stroker->angle_in, stroker->angle_out);

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

PVG_FT_Error PVG_FT_Stroker_GetBorderCounts(PVG_FT_Stroker       stroker,
                                          PVG_FT_StrokerBorder border,
                                          PVG_FT_UInt*         anum_points,
                                          PVG_FT_UInt*         anum_contours)
{
    PVG_FT_UInt  num_points = 0, num_contours = 0;
    PVG_FT_Error error;

    if (!stroker || border > 1) {
        error = -1;  // PVG_FT_THROW( Invalid_Argument );
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

PVG_FT_Error PVG_FT_Stroker_GetCounts(PVG_FT_Stroker stroker,
                                    PVG_FT_UInt*   anum_points,
                                    PVG_FT_UInt*   anum_contours)
{
    PVG_FT_UInt  count1, count2, num_points = 0;
    PVG_FT_UInt  count3, count4, num_contours = 0;
    PVG_FT_Error error;

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

void PVG_FT_Stroker_ExportBorder(PVG_FT_Stroker       stroker,
                                PVG_FT_StrokerBorder border,
                                PVG_FT_Outline*      outline)
{
    if (border == PVG_FT_STROKER_BORDER_LEFT ||
        border == PVG_FT_STROKER_BORDER_RIGHT) {
        PVG_FT_StrokeBorder sborder = &stroker->borders[border];

        if (sborder->valid) ft_stroke_border_export(sborder, outline);
    }
}

/* documentation is in ftstroke.h */

void PVG_FT_Stroker_Export(PVG_FT_Stroker stroker, PVG_FT_Outline* outline)
{
    PVG_FT_Stroker_ExportBorder(stroker, PVG_FT_STROKER_BORDER_LEFT, outline);
    PVG_FT_Stroker_ExportBorder(stroker, PVG_FT_STROKER_BORDER_RIGHT, outline);
}

/* documentation is in ftstroke.h */

/*
 *  The following is very similar to PVG_FT_Outline_Decompose, except
 *  that we do support opened paths, and do not scale the outline.
 */
PVG_FT_Error PVG_FT_Stroker_ParseOutline(PVG_FT_Stroker        stroker,
                                       const PVG_FT_Outline* outline)
{
    PVG_FT_Vector v_last;
    PVG_FT_Vector v_control;
    PVG_FT_Vector v_start;

    PVG_FT_Vector* point;
    PVG_FT_Vector* limit;
    char*         tags;

    PVG_FT_Error error;

    PVG_FT_Int  n;     /* index of contour in outline     */
    PVG_FT_UInt first; /* index of first point in contour */
    PVG_FT_Int  tag;   /* current point's state           */

    if (!outline || !stroker) return -1;  // PVG_FT_THROW( Invalid_Argument );

    PVG_FT_Stroker_Rewind(stroker);

    first = 0;

    for (n = 0; n < outline->n_contours; n++) {
        PVG_FT_UInt last; /* index of last point in contour */

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
        tag = PVG_FT_CURVE_TAG(tags[0]);

        /* A contour cannot start with a cubic control point! */
        if (tag == PVG_FT_CURVE_TAG_CUBIC) goto Invalid_Outline;

        /* check first point to determine origin */
        if (tag == PVG_FT_CURVE_TAG_CONIC) {
            /* First point is conic control.  Yes, this happens. */
            if (PVG_FT_CURVE_TAG(outline->tags[last]) == PVG_FT_CURVE_TAG_ON) {
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

        error = PVG_FT_Stroker_BeginSubPath(stroker, &v_start, outline->contours_flag[n]);
        if (error) goto Exit;

        while (point < limit) {
            point++;
            tags++;

            tag = PVG_FT_CURVE_TAG(tags[0]);
            switch (tag) {
            case PVG_FT_CURVE_TAG_ON: /* emit a single line_to */
            {
                PVG_FT_Vector vec;

                vec.x = point->x;
                vec.y = point->y;

                error = PVG_FT_Stroker_LineTo(stroker, &vec);
                if (error) goto Exit;
                continue;
            }

            case PVG_FT_CURVE_TAG_CONIC: /* consume conic arcs */
                v_control.x = point->x;
                v_control.y = point->y;

            Do_Conic:
                if (point < limit) {
                    PVG_FT_Vector vec;
                    PVG_FT_Vector v_middle;

                    point++;
                    tags++;
                    tag = PVG_FT_CURVE_TAG(tags[0]);

                    vec = point[0];

                    if (tag == PVG_FT_CURVE_TAG_ON) {
                        error =
                            PVG_FT_Stroker_ConicTo(stroker, &v_control, &vec);
                        if (error) goto Exit;
                        continue;
                    }

                    if (tag != PVG_FT_CURVE_TAG_CONIC) goto Invalid_Outline;

                    v_middle.x = (v_control.x + vec.x) / 2;
                    v_middle.y = (v_control.y + vec.y) / 2;

                    error =
                        PVG_FT_Stroker_ConicTo(stroker, &v_control, &v_middle);
                    if (error) goto Exit;

                    v_control = vec;
                    goto Do_Conic;
                }

                error = PVG_FT_Stroker_ConicTo(stroker, &v_control, &v_start);
                goto Close;

            default: /* PVG_FT_CURVE_TAG_CUBIC */
            {
                PVG_FT_Vector vec1, vec2;

                if (point + 1 > limit ||
                    PVG_FT_CURVE_TAG(tags[1]) != PVG_FT_CURVE_TAG_CUBIC)
                    goto Invalid_Outline;

                point += 2;
                tags += 2;

                vec1 = point[-2];
                vec2 = point[-1];

                if (point <= limit) {
                    PVG_FT_Vector vec;

                    vec = point[0];

                    error = PVG_FT_Stroker_CubicTo(stroker, &vec1, &vec2, &vec);
                    if (error) goto Exit;
                    continue;
                }

                error = PVG_FT_Stroker_CubicTo(stroker, &vec1, &vec2, &v_start);
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

        error = PVG_FT_Stroker_EndSubPath(stroker);
        if (error) goto Exit;

        first = last + 1;
    }

    return 0;

Exit:
    return error;

Invalid_Outline:
    return -2;  // PVG_FT_THROW( Invalid_Outline );
}

/* END */
