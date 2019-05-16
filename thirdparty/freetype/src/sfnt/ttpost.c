/****************************************************************************
 *
 * ttpost.c
 *
 *   PostScript name table processing for TrueType and OpenType fonts
 *   (body).
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

  /**************************************************************************
   *
   * The post table is not completely loaded by the core engine.  This
   * file loads the missing PS glyph names and implements an API to access
   * them.
   *
   */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TAGS_H


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


#include FT_SERVICE_POSTSCRIPT_CMAPS_H

#define MAC_NAME( x )  (FT_String*)psnames->macintosh_name( (FT_UInt)(x) )


#else /* FT_CONFIG_OPTION_POSTSCRIPT_NAMES */


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


#endif /* FT_CONFIG_OPTION_POSTSCRIPT_NAMES */


  static FT_Error
  load_format_20( TT_Face    face,
                  FT_Stream  stream,
                  FT_ULong   post_limit )
  {
    FT_Memory   memory = stream->memory;
    FT_Error    error;

    FT_Int      num_glyphs;
    FT_UShort   num_names;

    FT_UShort*  glyph_indices = NULL;
    FT_Char**   name_strings  = NULL;


    if ( FT_READ_USHORT( num_glyphs ) )
      goto Exit;

    /* UNDOCUMENTED!  The number of glyphs in this table can be smaller */
    /* than the value in the maxp table (cf. cyberbit.ttf).             */

    /* There already exist fonts which have more than 32768 glyph names */
    /* in this table, so the test for this threshold has been dropped.  */

    if ( num_glyphs > face->max_profile.numGlyphs )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* load the indices */
    {
      FT_Int  n;


      if ( FT_NEW_ARRAY ( glyph_indices, num_glyphs ) ||
           FT_FRAME_ENTER( num_glyphs * 2L )          )
        goto Fail;

      for ( n = 0; n < num_glyphs; n++ )
        glyph_indices[n] = FT_GET_USHORT();

      FT_FRAME_EXIT();
    }

    /* compute number of names stored in table */
    {
      FT_Int  n;


      num_names = 0;

      for ( n = 0; n < num_glyphs; n++ )
      {
        FT_Int  idx;


        idx = glyph_indices[n];
        if ( idx >= 258 )
        {
          idx -= 257;
          if ( idx > num_names )
            num_names = (FT_UShort)idx;
        }
      }
    }

    /* now load the name strings */
    {
      FT_UShort  n;


      if ( FT_NEW_ARRAY( name_strings, num_names ) )
        goto Fail;

      for ( n = 0; n < num_names; n++ )
      {
        FT_UInt  len;


        if ( FT_STREAM_POS() >= post_limit )
          break;
        else
        {
          FT_TRACE6(( "load_format_20: %d byte left in post table\n",
                      post_limit - FT_STREAM_POS() ));

          if ( FT_READ_BYTE( len ) )
            goto Fail1;
        }

        if ( len > post_limit                   ||
             FT_STREAM_POS() > post_limit - len )
        {
          FT_Int  d = (FT_Int)post_limit - (FT_Int)FT_STREAM_POS();


          FT_ERROR(( "load_format_20:"
                     " exceeding string length (%d),"
                     " truncating at end of post table (%d byte left)\n",
                     len, d ));
          len = (FT_UInt)FT_MAX( 0, d );
        }

        if ( FT_NEW_ARRAY( name_strings[n], len + 1 ) ||
             FT_STREAM_READ( name_strings[n], len   ) )
          goto Fail1;

        name_strings[n][len] = '\0';
      }

      if ( n < num_names )
      {
        FT_ERROR(( "load_format_20:"
                   " all entries in post table are already parsed,"
                   " using NULL names for gid %d - %d\n",
                    n, num_names - 1 ));
        for ( ; n < num_names; n++ )
          if ( FT_NEW_ARRAY( name_strings[n], 1 ) )
            goto Fail1;
          else
            name_strings[n][0] = '\0';
      }
    }

    /* all right, set table fields and exit successfully */
    {
      TT_Post_20  table = &face->postscript_names.names.format_20;


      table->num_glyphs    = (FT_UShort)num_glyphs;
      table->num_names     = (FT_UShort)num_names;
      table->glyph_indices = glyph_indices;
      table->glyph_names   = name_strings;
    }
    return FT_Err_Ok;

  Fail1:
    {
      FT_UShort  n;


      for ( n = 0; n < num_names; n++ )
        FT_FREE( name_strings[n] );
    }

  Fail:
    FT_FREE( name_strings );
    FT_FREE( glyph_indices );

  Exit:
    return error;
  }


  static FT_Error
  load_format_25( TT_Face    face,
                  FT_Stream  stream,
                  FT_ULong   post_limit )
  {
    FT_Memory  memory = stream->memory;
    FT_Error   error;

    FT_Int     num_glyphs;
    FT_Char*   offset_table = NULL;

    FT_UNUSED( post_limit );


    if ( FT_READ_USHORT( num_glyphs ) )
      goto Exit;

    /* check the number of glyphs */
    if ( num_glyphs > face->max_profile.numGlyphs ||
         num_glyphs > 258                         ||
         num_glyphs < 1                           )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    if ( FT_NEW_ARRAY( offset_table, num_glyphs )   ||
         FT_STREAM_READ( offset_table, num_glyphs ) )
      goto Fail;

    /* now check the offset table */
    {
      FT_Int  n;


      for ( n = 0; n < num_glyphs; n++ )
      {
        FT_Long  idx = (FT_Long)n + offset_table[n];


        if ( idx < 0 || idx > num_glyphs )
        {
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }
      }
    }

    /* OK, set table fields and exit successfully */
    {
      TT_Post_25  table = &face->postscript_names.names.format_25;


      table->num_glyphs = (FT_UShort)num_glyphs;
      table->offsets    = offset_table;
    }

    return FT_Err_Ok;

  Fail:
    FT_FREE( offset_table );

  Exit:
    return error;
  }


  static FT_Error
  load_post_names( TT_Face  face )
  {
    FT_Stream  stream;
    FT_Error   error;
    FT_Fixed   format;
    FT_ULong   post_len;
    FT_ULong   post_limit;


    /* get a stream for the face's resource */
    stream = face->root.stream;

    /* seek to the beginning of the PS names table */
    error = face->goto_table( face, TTAG_post, stream, &post_len );
    if ( error )
      goto Exit;

    post_limit = FT_STREAM_POS() + post_len;

    format = face->postscript.FormatType;

    /* go to beginning of subtable */
    if ( FT_STREAM_SKIP( 32 ) )
      goto Exit;

    /* now read postscript table */
    if ( format == 0x00020000L )
      error = load_format_20( face, stream, post_limit );
    else if ( format == 0x00025000L )
      error = load_format_25( face, stream, post_limit );
    else
      error = FT_THROW( Invalid_File_Format );

    face->postscript_names.loaded = 1;

  Exit:
    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_ps_names( TT_Face  face )
  {
    FT_Memory      memory = face->root.memory;
    TT_Post_Names  names  = &face->postscript_names;
    FT_Fixed       format;


    if ( names->loaded )
    {
      format = face->postscript.FormatType;

      if ( format == 0x00020000L )
      {
        TT_Post_20  table = &names->names.format_20;
        FT_UShort   n;


        FT_FREE( table->glyph_indices );
        table->num_glyphs = 0;

        for ( n = 0; n < table->num_names; n++ )
          FT_FREE( table->glyph_names[n] );

        FT_FREE( table->glyph_names );
        table->num_names = 0;
      }
      else if ( format == 0x00025000L )
      {
        TT_Post_25  table = &names->names.format_25;


        FT_FREE( table->offsets );
        table->num_glyphs = 0;
      }
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
    TT_Post_Names  names;
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

    names = &face->postscript_names;

    /* `.notdef' by default */
    *PSname = MAC_NAME( 0 );

    format = face->postscript.FormatType;

    if ( format == 0x00010000L )
    {
      if ( idx < 258 )                    /* paranoid checking */
        *PSname = MAC_NAME( idx );
    }
    else if ( format == 0x00020000L )
    {
      TT_Post_20  table = &names->names.format_20;


      if ( !names->loaded )
      {
        error = load_post_names( face );
        if ( error )
          goto End;
      }

      if ( idx < (FT_UInt)table->num_glyphs )
      {
        FT_UShort  name_index = table->glyph_indices[idx];


        if ( name_index < 258 )
          *PSname = MAC_NAME( name_index );
        else
          *PSname = (FT_String*)table->glyph_names[name_index - 258];
      }
    }
    else if ( format == 0x00025000L )
    {
      TT_Post_25  table = &names->names.format_25;


      if ( !names->loaded )
      {
        error = load_post_names( face );
        if ( error )
          goto End;
      }

      if ( idx < (FT_UInt)table->num_glyphs )    /* paranoid checking */
        *PSname = MAC_NAME( (FT_Int)idx + table->offsets[idx] );
    }

    /* nothing to do for format == 0x00030000L */

  End:
    return FT_Err_Ok;
  }

#else /* !TT_CONFIG_OPTION_POSTSCRIPT_NAMES */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_post_dummy;

#endif /* !TT_CONFIG_OPTION_POSTSCRIPT_NAMES */


/* END */
