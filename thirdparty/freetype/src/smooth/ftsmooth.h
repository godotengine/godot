/***************************************************************************/
/*                                                                         */
/*  ftsmooth.h                                                             */
/*                                                                         */
/*    Anti-aliasing renderer interface (specification).                    */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTSMOOTH_H_
#define FTSMOOTH_H_


#include <ft2build.h>
#include FT_RENDER_H


FT_BEGIN_HEADER


#ifndef FT_CONFIG_OPTION_NO_STD_RASTER
  FT_DECLARE_RENDERER( ft_std_renderer_class )
#endif

#ifndef FT_CONFIG_OPTION_NO_SMOOTH_RASTER
  FT_DECLARE_RENDERER( ft_smooth_renderer_class )

  FT_DECLARE_RENDERER( ft_smooth_lcd_renderer_class )

  FT_DECLARE_RENDERER( ft_smooth_lcd_v_renderer_class )
#endif



FT_END_HEADER

#endif /* FTSMOOTH_H_ */


/* END */
