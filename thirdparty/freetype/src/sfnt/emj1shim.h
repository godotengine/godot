/****************************************************************************
 *
 * emj1shim.h
 *
 */


#ifndef EMJ1SHIM_H_
#define EMJ1SHIM_H_


#include "ttload.h"


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_USE_EMJC

  FT_LOCAL( FT_Error )
  Load_SBit_Emj1( FT_GlyphSlot     slot,
                 FT_Int           x_offset,
                 FT_Int           y_offset,
                 FT_Int           pix_bits,
                 TT_SBit_Metrics  metrics,
                 FT_Memory        memory,
                 FT_Byte*         data,
                 FT_UInt          png_len,
                 FT_Bool          populate_map_and_metrics,
                 FT_Bool          metrics_only );

#endif

FT_END_HEADER

#endif /* EMJ1SHIM_H_ */


/* END */
