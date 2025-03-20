/****************************************************************************
 *
 * ttpost.c
 *
 *   PostScript name table processing for TrueType and OpenType fonts
 *   (body).
 *
 * Copyright (C) 1996-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

  /**************************************************************************
   *
   * The post table is not completely loaded by the core engine.  This
   * file loads the missing PS glyph names and implements an API to access
   * them.
   *
   */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>


#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES

#include "ttpost.h"

#include "sferrors.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttpost


  /* If this configuration macro is defined, we rely on the `psnames' */
  /* module to grab the glyph names.                                  */

#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES


#include <freetype/internal/services/svpscmap.h>

#define MAC_NAME( x )  (FT_String*)psnames->macintosh_name( (FT_UInt)(x) )


#else /* !FT_CONFIG_OPTION_POSTSCRIPT_NAMES */


   /* Otherwise, we ignore the `psnames' module, and provide our own  */
   /* table of Mac names.  Thus, it is possible to build a version of */
   /* FreeType without the Type 1 driver & psnames module.            */

#define MAC_NAME( x )  (FT_String*)tt_post_default_names[x]

  /* the 258 default Mac PS glyph names; see file `tools/glnames.py' */

  static const FT_String* const  tt_post_default_names[258] =
  {
    /*   0 */
    ".notdef", ".null", "nonmarkingreturn", "space", "exclam",
    "quotedbl", "numbersign", "dollar", "percent", "ampersand",
    /*  10 */
    "quotesingle", "parenleft", "parenright", "asterisk", "plus",
    "comma", "hyphen", "period", "slash", "zero",
    /*  20 */
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "colon",
    /*  30 */
    "semicolon", "less", "equal", "greater", "question",
    "at", "A", "B", "C", "D",
    /*  40 */
    "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N",
    /*  50 */
    "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X",
    /*  60 */
    "Y", "Z", "bracketleft", "backslash", "bracketright",
    "asciicircum", "underscore", "grave", "a", "b",
    /*  70 */
    "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l",
    /*  80 */
    "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v",
    /*  90 */
    "w", "x", "y", "z", "braceleft",
    "bar", "braceright", "asciitilde", "Adieresis", "Aring",
    /* 100 */
    "Ccedilla", "Eacute", "Ntilde", "Odieresis", "Udieresis",
    "aacute", "agrave", "acircumflex", "adieresis", "atilde",
    /* 110 */
    "aring", "ccedilla", "eacute", "egrave", "ecircumflex",
    "edieresis", "iacute", "igrave", "icircumflex", "idieresis",
    /* 120 */
    "ntilde", "oacute", "ograve", "ocircumflex", "odieresis",
    "otilde", "uacute", "ugrave", "ucircumflex", "udieresis",
    /* 130 */
    "dagger", "degree", "cent", "sterling", "section",
    "bullet", "paragraph", "germandbls", "registered", "copyright",
    /* 140 */
    "trademark", "acute", "dieresis", "notequal", "AE",
    "Oslash", "infinity", "plusminus", "lessequal", "greaterequal",
    /* 150 */
    "yen", "mu", "partialdiff", "summation", "product",
    "pi", "integral", "ordfeminine", "ordmasculine", "Omega",
    /* 160 */
    "ae", "oslash", "questiondown", "exclamdown", "logicalnot",
    "radical", "florin", "approxequal", "Delta", "guillemotleft",
    /* 170 */
    "guillemotright", "ellipsis", "nonbreakingspace", "Agrave", "Atilde",
    "Otilde", "OE", "oe", "endash", "emdash",
    /* 180 */
    "quotedblleft", "quotedblright", "quoteleft", "quoteright", "divide",
    "lozenge", "ydieresis", "Ydieresis", "fraction", "currency",
    /* 190 */
    "guilsinglleft", "guilsinglright", "fi", "fl", "daggerdbl",
    "periodcentered", "quotesinglbase", "quotedblbase", "perthousand", "Acircumflex",
    /* 200 */
    "Ecircumflex", "Aacute", "Edieresis", "Egrave", "Iacute",
    "Icircumflex", "Idieresis", "Igrave", "Oacute", "Ocircumflex",
    /* 210 */
    "apple", "Ograve", "Uacute", "Ucircumflex", "Ugrave",
    "dotlessi", "circumflex", "tilde", "macron", "breve",
    /* 220 */
    "dotaccent", "ring", "cedilla", "hungarumlaut", "ogonek",
    "caron", "Lslash", "lslash", "Scaron", "scaron",
    /* 230 */
    "Zcaron", "zcaron", "brokenbar", "Eth", "eth",
    "Yacute", "yacute", "Thorn", "thorn", "minus",
    /* 240 */
    "multiply", "onesuperior", "twosuperior", "threesuperior", "onehalf",
    "onequarter", "threequarters", "franc", "Gbreve", "gbreve",
    /* 250 */
    "Idotaccent", "Scedilla", "scedilla", "Cacute", "cacute",
    "Ccaron", "ccaron", "dcroat",
  };


#endif /* !FT_CONFIG_OPTION_POSTSCRIPT_NAMES */


  static FT_Error
  load_format_20( TT_Post_Names  names,
                  FT_Stream      stream,
                  FT_UShort      num_glyphs,
                  FT_ULong       post_len )
  {
    FT_Memory   memory = stream->memory;
    FT_Error    error;

    FT_UShort   n;
    FT_UShort   num_names = 0;

    FT_UShort*  glyph_indices = NULL;
    FT_Byte**   name_strings  = NULL;
    FT_Byte*    q;


    if ( (FT_ULong)num_glyphs * 2 > post_len )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* load the indices and note their maximum */
    if ( FT_QNEW_ARRAY( glyph_indices, num_glyphs ) ||
         FT_FRAME_ENTER( num_glyphs * 2 )           )
      goto Fail;

    q = (FT_Byte*)stream->cursor;

    for ( n = 0; n < num_glyphs; n++ )
    {
      FT_UShort  idx = FT_NEXT_USHORT( q );


      if ( idx > num_names )
        num_names = idx;

      glyph_indices[n] = idx;
    }

    FT_FRAME_EXIT();

    /* compute number of names stored in the table */
    num_names = num_names > 257 ? num_names - 257 : 0;

    /* now load the name strings */
    if ( num_names )
    {
      FT_Byte*   p;
      FT_Byte*   p_end;


      post_len -= (FT_ULong)num_glyphs * 2;

      if ( FT_QALLOC( name_strings, num_names * sizeof ( FT_Byte* ) +
                                    post_len + 1 ) )
        goto Fail;

      p = (FT_Byte*)( name_strings + num_names );
      if ( FT_STREAM_READ( p, post_len ) )
        goto Fail;

      p_end = p + post_len;

      /* convert from Pascal- to C-strings and set pointers */
      for ( n = 0; p < p_end && n < num_names; n++ )
      {
        FT_UInt  len = *p;


        /* names in the Adobe Glyph List are shorter than 40 characters */
        if ( len >= 40U )
          FT_TRACE4(( "load_format_20: unusual %u-char name found\n", len ));

        *p++            = 0;
        name_strings[n] = p;
        p              += len;
      }
      *p_end = 0;

      /* deal with missing or insufficient string data */
      if ( n < num_names )
      {
        FT_TRACE4(( "load_format_20: %hu PostScript names are truncated\n",
                    (FT_UShort)( num_names - n ) ));

        for ( ; n < num_names; n++ )
          name_strings[n] = p_end;
      }
    }

    /* all right, set table fields and exit successfully */
    names->num_glyphs    = num_glyphs;
    names->num_names     = num_names;
    names->glyph_indices = glyph_indices;
    names->glyph_names   = name_strings;

    return FT_Err_Ok;

  Fail:
    FT_FREE( name_strings );
    FT_FREE( glyph_indices );

  Exit:
    return error;
  }


  static FT_Error
  load_format_25( TT_Post_Names  names,
                  FT_Stream      stream,
                  FT_UShort      num_glyphs,
                  FT_ULong       post_len )
  {
    FT_Memory  memory = stream->memory;
    FT_Error   error;

    FT_UShort   n;
    FT_UShort*  glyph_indices = NULL;
    FT_Byte*    q;


    /* check the number of glyphs, including the theoretical limit */
    if ( num_glyphs > post_len  ||
         num_glyphs > 258 + 128 )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* load the indices and check their Mac range */
    if ( FT_QNEW_ARRAY( glyph_indices, num_glyphs ) ||
         FT_FRAME_ENTER( num_glyphs )               )
      goto Fail;

    q = (FT_Byte*)stream->cursor;

    for ( n = 0; n < num_glyphs; n++ )
    {
      FT_Int  idx = n + FT_NEXT_CHAR( q );


      if ( idx < 0 || idx > 257 )
        idx = 0;

      glyph_indices[n] = (FT_UShort)idx;
    }

    FT_FRAME_EXIT();

    /* OK, set table fields and exit successfully */
    names->num_glyphs    = num_glyphs;
    names->glyph_indices = glyph_indices;

    return FT_Err_Ok;

  Fail:
    FT_FREE( glyph_indices );

  Exit:
    return error;
  }


  static FT_Error
  load_post_names( TT_Face  face )
  {
    FT_Error   error = FT_Err_Ok;
    FT_Stream  stream = face->root.stream;
    FT_Fixed   format = face->postscript.FormatType;
    FT_ULong   post_len;
    FT_UShort  num_glyphs;


    /* seek to the beginning of the PS names table */
    error = face->goto_table( face, TTAG_post, stream, &post_len );
    if ( error )
      goto Exit;

    /* UNDOCUMENTED!  The number of glyphs in this table can be smaller */
    /* than the value in the maxp table (cf. cyberbit.ttf).             */
    if ( post_len < 34                            ||
         FT_STREAM_SKIP( 32 )                     ||
         FT_READ_USHORT( num_glyphs )             ||
         num_glyphs > face->max_profile.numGlyphs ||
         num_glyphs == 0 )
      goto Exit;

    /* now read postscript names data */
    if ( format == 0x00020000L )
      error = load_format_20( &face->postscript_names, stream,
                              num_glyphs, post_len - 34 );
    else if ( format == 0x00025000L )
      error = load_format_25( &face->postscript_names, stream,
                              num_glyphs, post_len - 34 );

  Exit:
    face->postscript_names.loaded = 1;  /* even if failed */
    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_ps_names( TT_Face  face )
  {
    FT_Memory      memory = face->root.memory;
    TT_Post_Names  names  = &face->postscript_names;


    if ( names->num_glyphs )
    {
      FT_FREE( names->glyph_indices );
      names->num_glyphs = 0;
    }

    if ( names->num_names )
    {
      FT_FREE( names->glyph_names );
      names->num_names = 0;
    }

    names->loaded = 0;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_get_ps_name
   *
   * @Description:
   *   Get the PostScript glyph name of a glyph.
   *
   * @Input:
   *   face ::
   *     A handle to the parent face.
   *
   *   idx ::
   *     The glyph index.
   *
   * @InOut:
   *   PSname ::
   *     The address of a string pointer.  Undefined in case of
   *     error, otherwise it is a pointer to the glyph name.
   *
   *     You must not modify the returned string!
   *
   * @Output:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_get_ps_name( TT_Face      face,
                       FT_UInt      idx,
                       FT_String**  PSname )
  {
    FT_Error       error;
    FT_Fixed       format;

#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES
    FT_Service_PsCMaps  psnames;
#endif


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( idx >= (FT_UInt)face->max_profile.numGlyphs )
      return FT_THROW( Invalid_Glyph_Index );

#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES
    psnames = (FT_Service_PsCMaps)face->psnames;
    if ( !psnames )
      return FT_THROW( Unimplemented_Feature );
#endif

    /* `.notdef' by default */
    *PSname = MAC_NAME( 0 );

    format = face->postscript.FormatType;

    if ( format == 0x00020000L ||
         format == 0x00025000L )
    {
      TT_Post_Names  names = &face->postscript_names;


      if ( !names->loaded )
      {
        error = load_post_names( face );
        if ( error )
          goto End;
      }

      if ( idx < (FT_UInt)names->num_glyphs )
      {
        FT_UShort  name_index = names->glyph_indices[idx];


        if ( name_index < 258 )
          *PSname = MAC_NAME( name_index );
        else  /* only for version 2.0 */
          *PSname = (FT_String*)names->glyph_names[name_index - 258];
      }
    }

    /* version 1.0 is only valid with 258 glyphs */
    else if ( format == 0x00010000L              &&
              face->max_profile.numGlyphs == 258 )
      *PSname = MAC_NAME( idx );

    /* nothing to do for format == 0x00030000L */

  End:
    /* post format errors ignored */
    return FT_Err_Ok;
  }

#else /* !TT_CONFIG_OPTION_POSTSCRIPT_NAMES */

  /* ANSI C doesn't like empty source files */
  typedef int  tt_post_dummy_;

#endif /* !TT_CONFIG_OPTION_POSTSCRIPT_NAMES */


/* END */
