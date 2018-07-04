/***************************************************************************/
/*                                                                         */
/*  t1decode.h                                                             */
/*                                                                         */
/*    PostScript Type 1 decoding routines (specification).                 */
/*                                                                         */
/*  Copyright 2000-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef T1DECODE_H_
#define T1DECODE_H_


#include <ft2build.h>
#include FT_INTERNAL_POSTSCRIPT_AUX_H
#include FT_INTERNAL_TYPE1_TYPES_H


FT_BEGIN_HEADER


  FT_CALLBACK_TABLE
  const T1_Decoder_FuncsRec  t1_decoder_funcs;

  FT_LOCAL( FT_Int )
  t1_lookup_glyph_by_stdcharcode_ps( PS_Decoder*  decoder,
                                     FT_Int       charcode );

#ifdef T1_CONFIG_OPTION_OLD_ENGINE
  FT_LOCAL( FT_Error )
  t1_decoder_parse_glyph( T1_Decoder  decoder,
                          FT_UInt     glyph_index );

  FT_LOCAL( FT_Error )
  t1_decoder_parse_charstrings( T1_Decoder  decoder,
                                FT_Byte*    base,
                                FT_UInt     len );
#else
  FT_LOCAL( FT_Error )
  t1_decoder_parse_metrics( T1_Decoder  decoder,
                            FT_Byte*    charstring_base,
                            FT_UInt     charstring_len );
#endif

  FT_LOCAL( FT_Error )
  t1_decoder_init( T1_Decoder           decoder,
                   FT_Face              face,
                   FT_Size              size,
                   FT_GlyphSlot         slot,
                   FT_Byte**            glyph_names,
                   PS_Blend             blend,
                   FT_Bool              hinting,
                   FT_Render_Mode       hint_mode,
                   T1_Decoder_Callback  parse_glyph );

  FT_LOCAL( void )
  t1_decoder_done( T1_Decoder  decoder );


FT_END_HEADER

#endif /* T1DECODE_H_ */


/* END */
