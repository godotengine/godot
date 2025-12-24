/*
 * cmyk.h
 *
 * Copyright (C) 2017-2018, 2022, 2024-2025, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains convenience functions for performing quick & dirty
 * CMYK<->RGB conversion.  This algorithm is suitable for testing purposes
 * only.  Properly converting between CMYK and RGB requires a color management
 * system.
 */

#ifndef CMYK_H
#define CMYK_H

#include "jinclude.h"
#define JPEG_INTERNALS
#include "jpeglib.h"
#include "jsamplecomp.h"


/* Fully reversible */

INLINE
LOCAL(void)
rgb_to_cmyk(int maxval, _JSAMPLE r, _JSAMPLE g, _JSAMPLE b,
            _JSAMPLE *c, _JSAMPLE *m, _JSAMPLE *y, _JSAMPLE *k)
{
  double ctmp = 1.0 - ((double)r / (double)maxval);
  double mtmp = 1.0 - ((double)g / (double)maxval);
  double ytmp = 1.0 - ((double)b / (double)maxval);
  double ktmp = MIN(MIN(ctmp, mtmp), ytmp);

  if (ktmp == 1.0) ctmp = mtmp = ytmp = 0.0;
  else {
    ctmp = (ctmp - ktmp) / (1.0 - ktmp);
    mtmp = (mtmp - ktmp) / (1.0 - ktmp);
    ytmp = (ytmp - ktmp) / (1.0 - ktmp);
  }
  *c = (_JSAMPLE)((double)maxval - ctmp * (double)maxval + 0.5);
  *m = (_JSAMPLE)((double)maxval - mtmp * (double)maxval + 0.5);
  *y = (_JSAMPLE)((double)maxval - ytmp * (double)maxval + 0.5);
  *k = (_JSAMPLE)((double)maxval - ktmp * (double)maxval + 0.5);
}


/* Fully reversible only for C/M/Y/K values generated with rgb_to_cmyk() */

INLINE
LOCAL(void)
cmyk_to_rgb(int maxval, _JSAMPLE c, _JSAMPLE m, _JSAMPLE y, _JSAMPLE k,
            _JSAMPLE *r, _JSAMPLE *g, _JSAMPLE *b)
{
  *r = (_JSAMPLE)((double)c * (double)k / (double)maxval + 0.5);
  *g = (_JSAMPLE)((double)m * (double)k / (double)maxval + 0.5);
  *b = (_JSAMPLE)((double)y * (double)k / (double)maxval + 0.5);
}


#endif /* CMYK_H */
