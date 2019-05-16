/****************************************************************************
 *
 * ftsmooth.h
 *
 *   Anti-aliasing renderer interface (specification).
 *
 * Copyright (C) 1996-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTSMOOTH_H_
#define FTSMOOTH_H_


#include <ft2build.h>
#include FT_RENDER_H


FT_BEGIN_HEADER


  FT_DECLARE_RENDERER( ft_smooth_renderer_class )

  FT_DECLARE_RENDERER( ft_smooth_lcd_renderer_class )

  FT_DECLARE_RENDERER( ft_smooth_lcdv_renderer_class )


FT_END_HEADER

#endif /* FTSMOOTH_H_ */


/* END */
