/****************************************************************************
 *
 * ttload.c
 *
 *   Load the basic TrueType tables, i.e., tables that can be either in
 *   TTF or OTF fonts (body).
 *
 * Copyright (C) 1996-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include "ttload.h"

#include "sferrors.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttload


  /**************************************************************************
   *
   * @Function:
   *   tt_face_lookup_table
   *
   * @Description:
   *   Looks for a TrueType table by name.
   *
   * @Input:
   *   face ::
   *     A face object handle.
   *
   *   tag ::
   *     The searched tag.
   *
   * @Return:
   *   A pointer to the table directory entry.  0 if not found.
   */
  FT_LOCAL_DEF( TT_Table  )
  tt_face_lookup_table( TT_Face   face,
                        FT_ULong  tag  )
  {
    TT_Table  entry;
    TT_Table  limit;
#ifdef FT_DEBUG_LEVEL_TRACE
    FT_Bool   zero_length = FALSE;
#endif


    FT_TRACE4(( "tt_face_lookup_table: %p, `%c%c%c%c' -- ",
                (void *)face,
                (FT_Char)( tag >> 24 ),
                (FT_Char)( tag >> 16 ),
                (FT_Char)( tag >> 8  ),
                (FT_Char)( tag       ) ));

    entry = face->dir_tables;
    limit = entry + face->num_tables;

    for ( ; entry < limit; entry++ )
    {
      /* For compatibility with Windows, we consider    */
      /* zero-length tables the same as missing tables. */
      if ( entry->Tag == tag )
      {
        if ( entry->Length != 0 )
        {
          FT_TRACE4(( "found table.\n" ));
          return entry;
        }
#ifdef FT_DEBUG_LEVEL_TRACE
        zero_length = TRUE;
#endif
      }
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( zero_length )
      FT_TRACE4(( "ignoring empty table\n" ));
    else
      FT_TRACE4(( "could not find table\n" ));
#endif

    return NULL;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_goto_table
   *
   * @Description:
   *   Looks for a TrueType table by name, then seek a stream to it.
   *
   * @Input:
   *   face ::
   *     A face object handle.
   *
   *   tag ::
   *     The searched tag.
   *
   *   stream ::
   *     The stream to seek when the table is found.
   *
   * @Output:
   *   length ::
   *     The length of the table if found, undefined otherwise.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_goto_table( TT_Face    face,
                      FT_ULong   tag,
                      FT_Stream  stream,
                      FT_ULong*  length )
  {
    TT_Table  table;
    FT_Error  error;


    table = tt_face_lookup_table( face, tag );
    if ( table )
    {
      if ( length )
        *length = table->Length;

      if ( FT_STREAM_SEEK( table->Offset ) )
        goto Exit;
    }
    else
      error = FT_THROW( Table_Missing );

  Exit:
    return error;
  }


  /* Here, we                                                         */
  /*                                                                  */
  /* - check that `num_tables' is valid (and adjust it if necessary); */
  /*   also return the number of valid table entries                  */
  /*                                                                  */
  /* - look for a `head' table, check its size, and parse it to check */
  /*   whether its `magic' field is correctly set                     */
  /*                                                                  */
  /* - errors (except errors returned by stream handling)             */
  /*                                                                  */
  /*     SFNT_Err_Unknown_File_Format:                                */
  /*       no table is defined in directory, it is not sfnt-wrapped   */
  /*       data                                                       */
  /*     SFNT_Err_Table_Missing:                                      */
  /*       table directory is valid, but essential tables             */
  /*       (head/bhed/SING) are missing                               */
  /*                                                                  */
  static FT_Error
  check_table_dir( SFNT_Header  sfnt,
                   FT_Stream    stream,
                   FT_UShort*   valid )
  {
    FT_Error   error;
    FT_UShort  nn, valid_entries = 0;
    FT_UInt    has_head = 0, has_sing = 0, has_meta = 0;
    FT_ULong   offset = sfnt->offset + 12;

    static const FT_Frame_Field  table_dir_entry_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_TableRec

      FT_FRAME_START( 16 ),
        FT_FRAME_ULONG( Tag ),
        FT_FRAME_ULONG( CheckSum ),
        FT_FRAME_ULONG( Offset ),
        FT_FRAME_ULONG( Length ),
      FT_FRAME_END
    };


    if ( FT_STREAM_SEEK( offset ) )
      goto Exit;

    for ( nn = 0; nn < sfnt->num_tables; nn++ )
    {
      TT_TableRec  table;


      if ( FT_STREAM_READ_FIELDS( table_dir_entry_fields, &table ) )
      {
        FT_TRACE2(( "check_table_dir:"
                    " can read only %d table%s in font (instead of %d)\n",
                    nn, nn == 1 ? "" : "s", sfnt->num_tables ));
        sfnt->num_tables = nn;
        break;
      }

      /* we ignore invalid tables */

      if ( table.Offset > stream->size )
      {
        FT_TRACE2(( "check_table_dir: table entry %d invalid\n", nn ));
        continue;
      }
      else if ( table.Length > stream->size - table.Offset )
      {
        /* Some tables have such a simple structure that clipping its     */
        /* contents is harmless.  This also makes FreeType less sensitive */
        /* to invalid table lengths (which programs like Acroread seem to */
        /* ignore in general).                                            */

        if ( table.Tag == TTAG_hmtx ||
             table.Tag == TTAG_vmtx )
          valid_entries++;
        else
        {
          FT_TRACE2(( "check_table_dir: table entry %d invalid\n", nn ));
          continue;
        }
      }
      else
        valid_entries++;

      if ( table.Tag == TTAG_head || table.Tag == TTAG_bhed )
      {
        FT_UInt32  magic;


#ifndef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
        if ( table.Tag == TTAG_head )
#endif
          has_head = 1;

        /*
         * The table length should be 0x36, but certain font tools make it
         * 0x38, so we will just check that it is greater.
         *
         * Note that according to the specification, the table must be
         * padded to 32-bit lengths, but this doesn't apply to the value of
         * its `Length' field!
         *
         */
        if ( table.Length < 0x36 )
        {
          FT_TRACE2(( "check_table_dir:"
                      " `head' or `bhed' table too small\n" ));
          error = FT_THROW( Table_Missing );
          goto Exit;
        }

        if ( FT_STREAM_SEEK( table.Offset + 12 ) ||
             FT_READ_ULONG( magic )              )
          goto Exit;

        if ( magic != 0x5F0F3CF5UL )
          FT_TRACE2(( "check_table_dir:"
                      " invalid magic number in `head' or `bhed' table\n"));

        if ( FT_STREAM_SEEK( offset + ( nn + 1 ) * 16 ) )
          goto Exit;
      }
      else if ( table.Tag == TTAG_SING )
        has_sing = 1;
      else if ( table.Tag == TTAG_META )
        has_meta = 1;
    }

    *valid = valid_entries;

    if ( !valid_entries )
    {
      FT_TRACE2(( "check_table_dir: no valid tables found\n" ));
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    /* if `sing' and `meta' tables are present, there is no `head' table */
    if ( has_head || ( has_sing && has_meta ) )
    {
      error = FT_Err_Ok;
      goto Exit;
    }
    else
    {
      FT_TRACE2(( "check_table_dir:" ));
#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
      FT_TRACE2(( " neither `head', `bhed', nor `sing' table found\n" ));
#else
      FT_TRACE2(( " neither `head' nor `sing' table found\n" ));
#endif
      error = FT_THROW( Table_Missing );
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_font_dir
   *
   * @Description:
   *   Loads the header of a SFNT font file.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     The input stream.
   *
   * @Output:
   *   sfnt ::
   *     The SFNT header.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   *
   * @Note:
   *   The stream cursor must be at the beginning of the font directory.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_font_dir( TT_Face    face,
                         FT_Stream  stream )
  {
    SFNT_HeaderRec  sfnt;
    FT_Error        error;
    FT_Memory       memory = stream->memory;
    FT_UShort       nn, valid_entries = 0;

    static const FT_Frame_Field  offset_table_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  SFNT_HeaderRec

      FT_FRAME_START( 8 ),
        FT_FRAME_USHORT( num_tables ),
        FT_FRAME_USHORT( search_range ),
        FT_FRAME_USHORT( entry_selector ),
        FT_FRAME_USHORT( range_shift ),
      FT_FRAME_END
    };


    FT_TRACE2(( "tt_face_load_font_dir: %p\n", (void *)face ));

    /* read the offset table */

    sfnt.offset = FT_STREAM_POS();

    if ( FT_READ_ULONG( sfnt.format_tag )                    ||
         FT_STREAM_READ_FIELDS( offset_table_fields, &sfnt ) )
      goto Exit;

    /* many fonts don't have these fields set correctly */
#if 0
    if ( sfnt.search_range != 1 << ( sfnt.entry_selector + 4 )        ||
         sfnt.search_range + sfnt.range_shift != sfnt.num_tables << 4 )
      return FT_THROW( Unknown_File_Format );
#endif

    /* load the table directory */

    FT_TRACE2(( "-- Number of tables: %10u\n",    sfnt.num_tables ));
    FT_TRACE2(( "-- Format version:   0x%08lx\n", sfnt.format_tag ));

    if ( sfnt.format_tag != TTAG_OTTO )
    {
      /* check first */
      error = check_table_dir( &sfnt, stream, &valid_entries );
      if ( error )
      {
        FT_TRACE2(( "tt_face_load_font_dir:"
                    " invalid table directory for TrueType\n" ));
        goto Exit;
      }
    }
    else
    {
      valid_entries = sfnt.num_tables;
      if ( !valid_entries )
      {
        FT_TRACE2(( "tt_face_load_font_dir: no valid tables found\n" ));
        error = FT_THROW( Unknown_File_Format );
        goto Exit;
      }
    }

    face->num_tables = valid_entries;
    face->format_tag = sfnt.format_tag;

    if ( FT_QNEW_ARRAY( face->dir_tables, face->num_tables ) )
      goto Exit;

    if ( FT_STREAM_SEEK( sfnt.offset + 12 )      ||
         FT_FRAME_ENTER( sfnt.num_tables * 16L ) )
      goto Exit;

    FT_TRACE2(( "\n" ));
    FT_TRACE2(( "  tag    offset    length   checksum\n" ));
    FT_TRACE2(( "  ----------------------------------\n" ));

    valid_entries = 0;
    for ( nn = 0; nn < sfnt.num_tables; nn++ )
    {
      TT_TableRec  entry;
      FT_UShort    i;
      FT_Bool      duplicate;


      entry.Tag      = FT_GET_TAG4();
      entry.CheckSum = FT_GET_ULONG();
      entry.Offset   = FT_GET_ULONG();
      entry.Length   = FT_GET_ULONG();

      /* ignore invalid tables that can't be sanitized */

      if ( entry.Offset > stream->size )
        continue;
      else if ( entry.Length > stream->size - entry.Offset )
      {
        if ( entry.Tag == TTAG_hmtx ||
             entry.Tag == TTAG_vmtx )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          FT_ULong  old_length = entry.Length;
#endif


          /* make metrics table length a multiple of 4 */
          entry.Length = ( stream->size - entry.Offset ) & ~3U;

          FT_TRACE2(( "  %c%c%c%c  %08lx  %08lx  %08lx"
                      " (sanitized; original length %08lx)",
                      (FT_Char)( entry.Tag >> 24 ),
                      (FT_Char)( entry.Tag >> 16 ),
                      (FT_Char)( entry.Tag >> 8  ),
                      (FT_Char)( entry.Tag       ),
                      entry.Offset,
                      entry.Length,
                      entry.CheckSum,
                      old_length ));
        }
        else
          continue;
      }
#ifdef FT_DEBUG_LEVEL_TRACE
      else
        FT_TRACE2(( "  %c%c%c%c  %08lx  %08lx  %08lx",
                    (FT_Char)( entry.Tag >> 24 ),
                    (FT_Char)( entry.Tag >> 16 ),
                    (FT_Char)( entry.Tag >> 8  ),
                    (FT_Char)( entry.Tag       ),
                    entry.Offset,
                    entry.Length,
                    entry.CheckSum ));
#endif

      /* ignore duplicate tables – the first one wins */
      duplicate = 0;
      for ( i = 0; i < valid_entries; i++ )
      {
        if ( face->dir_tables[i].Tag == entry.Tag )
        {
          duplicate = 1;
          break;
        }
      }
      if ( duplicate )
      {
        FT_TRACE2(( "  (duplicate, ignored)\n" ));
        continue;
      }
      else
      {
        FT_TRACE2(( "\n" ));

        /* we finally have a valid entry */
        face->dir_tables[valid_entries++] = entry;
      }
    }

    /* final adjustment to number of tables */
    face->num_tables = valid_entries;

    FT_FRAME_EXIT();

    FT_TRACE2(( "table directory loaded\n" ));
    FT_TRACE2(( "\n" ));

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_any
   *
   * @Description:
   *   Loads any font table into client memory.
   *
   * @Input:
   *   face ::
   *     The face object to look for.
   *
   *   tag ::
   *     The tag of table to load.  Use the value 0 if you want
   *     to access the whole font file, else set this parameter
   *     to a valid TrueType table tag that you can forge with
   *     the MAKE_TT_TAG macro.
   *
   *   offset ::
   *     The starting offset in the table (or the file if
   *     tag == 0).
   *
   *   length ::
   *     The address of the decision variable:
   *
   *     If length == NULL:
   *       Loads the whole table.  Returns an error if
   *       `offset' == 0!
   *
   *     If *length == 0:
   *       Exits immediately; returning the length of the given
   *       table or of the font file, depending on the value of
   *       `tag'.
   *
   *     If *length != 0:
   *       Loads the next `length' bytes of table or font,
   *       starting at offset `offset' (in table or font too).
   *
   * @Output:
   *   buffer ::
   *     The address of target buffer.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_any( TT_Face    face,
                    FT_ULong   tag,
                    FT_Long    offset,
                    FT_Byte*   buffer,
                    FT_ULong*  length )
  {
    FT_Error   error;
    FT_Stream  stream;
    TT_Table   table;
    FT_ULong   size;


    if ( tag != 0 )
    {
      /* look for tag in font directory */
      table = tt_face_lookup_table( face, tag );
      if ( !table )
      {
        error = FT_THROW( Table_Missing );
        goto Exit;
      }

      offset += table->Offset;
      size    = table->Length;
    }
    else
      /* tag == 0 -- the user wants to access the font file directly */
      size = face->root.stream->size;

    if ( length && *length == 0 )
    {
      *length = size;

      return FT_Err_Ok;
    }

    if ( length )
      size = *length;

    stream = face->root.stream;
    /* the `if' is syntactic sugar for picky compilers */
    if ( FT_STREAM_READ_AT( offset, buffer, size ) )
      goto Exit;

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_generic_header
   *
   * @Description:
   *   Loads the TrueType table `head' or `bhed'.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     The input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_face_load_generic_header( TT_Face    face,
                               FT_Stream  stream,
                               FT_ULong   tag )
  {
    FT_Error    error;
    TT_Header*  header;

    static const FT_Frame_Field  header_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_Header

      FT_FRAME_START( 54 ),
        FT_FRAME_ULONG ( Table_Version ),
        FT_FRAME_ULONG ( Font_Revision ),
        FT_FRAME_LONG  ( CheckSum_Adjust ),
        FT_FRAME_LONG  ( Magic_Number ),
        FT_FRAME_USHORT( Flags ),
        FT_FRAME_USHORT( Units_Per_EM ),
        FT_FRAME_ULONG ( Created[0] ),
        FT_FRAME_ULONG ( Created[1] ),
        FT_FRAME_ULONG ( Modified[0] ),
        FT_FRAME_ULONG ( Modified[1] ),
        FT_FRAME_SHORT ( xMin ),
        FT_FRAME_SHORT ( yMin ),
        FT_FRAME_SHORT ( xMax ),
        FT_FRAME_SHORT ( yMax ),
        FT_FRAME_USHORT( Mac_Style ),
        FT_FRAME_USHORT( Lowest_Rec_PPEM ),
        FT_FRAME_SHORT ( Font_Direction ),
        FT_FRAME_SHORT ( Index_To_Loc_Format ),
        FT_FRAME_SHORT ( Glyph_Data_Format ),
      FT_FRAME_END
    };


    error = face->goto_table( face, tag, stream, 0 );
    if ( error )
      goto Exit;

    header = &face->header;

    if ( FT_STREAM_READ_FIELDS( header_fields, header ) )
      goto Exit;

    FT_TRACE3(( "Units per EM: %4u\n", header->Units_Per_EM ));
    FT_TRACE3(( "IndexToLoc:   %4d\n", header->Index_To_Loc_Format ));

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_head( TT_Face    face,
                     FT_Stream  stream )
  {
    return tt_face_load_generic_header( face, stream, TTAG_head );
  }


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  FT_LOCAL_DEF( FT_Error )
  tt_face_load_bhed( TT_Face    face,
                     FT_Stream  stream )
  {
    return tt_face_load_generic_header( face, stream, TTAG_bhed );
  }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_maxp
   *
   * @Description:
   *   Loads the maximum profile into a face object.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     The input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_maxp( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error        error;
    TT_MaxProfile*  maxProfile = &face->max_profile;

    static const FT_Frame_Field  maxp_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_MaxProfile

      FT_FRAME_START( 6 ),
        FT_FRAME_LONG  ( version ),
        FT_FRAME_USHORT( numGlyphs ),
      FT_FRAME_END
    };

    static const FT_Frame_Field  maxp_fields_extra[] =
    {
      FT_FRAME_START( 26 ),
        FT_FRAME_USHORT( maxPoints ),
        FT_FRAME_USHORT( maxContours ),
        FT_FRAME_USHORT( maxCompositePoints ),
        FT_FRAME_USHORT( maxCompositeContours ),
        FT_FRAME_USHORT( maxZones ),
        FT_FRAME_USHORT( maxTwilightPoints ),
        FT_FRAME_USHORT( maxStorage ),
        FT_FRAME_USHORT( maxFunctionDefs ),
        FT_FRAME_USHORT( maxInstructionDefs ),
        FT_FRAME_USHORT( maxStackElements ),
        FT_FRAME_USHORT( maxSizeOfInstructions ),
        FT_FRAME_USHORT( maxComponentElements ),
        FT_FRAME_USHORT( maxComponentDepth ),
      FT_FRAME_END
    };


    error = face->goto_table( face, TTAG_maxp, stream, 0 );
    if ( error )
      goto Exit;

    if ( FT_STREAM_READ_FIELDS( maxp_fields, maxProfile ) )
      goto Exit;

    maxProfile->maxPoints             = 0;
    maxProfile->maxContours           = 0;
    maxProfile->maxCompositePoints    = 0;
    maxProfile->maxCompositeContours  = 0;
    maxProfile->maxZones              = 0;
    maxProfile->maxTwilightPoints     = 0;
    maxProfile->maxStorage            = 0;
    maxProfile->maxFunctionDefs       = 0;
    maxProfile->maxInstructionDefs    = 0;
    maxProfile->maxStackElements      = 0;
    maxProfile->maxSizeOfInstructions = 0;
    maxProfile->maxComponentElements  = 0;
    maxProfile->maxComponentDepth     = 0;

    if ( maxProfile->version >= 0x10000L )
    {
      if ( FT_STREAM_READ_FIELDS( maxp_fields_extra, maxProfile ) )
        goto Exit;

      /* XXX: an adjustment that is necessary to load certain */
      /*      broken fonts like `Keystrokes MT' :-(           */
      /*                                                      */
      /*   We allocate 64 function entries by default when    */
      /*   the maxFunctionDefs value is smaller.              */

      if ( maxProfile->maxFunctionDefs < 64 )
        maxProfile->maxFunctionDefs = 64;

      /* we add 4 phantom points later */
      if ( maxProfile->maxTwilightPoints > ( 0xFFFFU - 4 ) )
      {
        FT_TRACE0(( "tt_face_load_maxp:"
                    " too much twilight points in `maxp' table;\n" ));
        FT_TRACE0(( "                  "
                    " some glyphs might be rendered incorrectly\n" ));

        maxProfile->maxTwilightPoints = 0xFFFFU - 4;
      }
    }

    FT_TRACE3(( "numGlyphs: %u\n", maxProfile->numGlyphs ));

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_name
   *
   * @Description:
   *   Loads the name records.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     The input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_name( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error      error;
    FT_Memory     memory = stream->memory;
    FT_ULong      table_pos, table_len;
    FT_ULong      storage_start, storage_limit;
    TT_NameTable  table;
    TT_Name       names    = NULL;
    TT_LangTag    langTags = NULL;

    static const FT_Frame_Field  name_table_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_NameTableRec

      FT_FRAME_START( 6 ),
        FT_FRAME_USHORT( format ),
        FT_FRAME_USHORT( numNameRecords ),
        FT_FRAME_USHORT( storageOffset ),
      FT_FRAME_END
    };

    static const FT_Frame_Field  name_record_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_NameRec

      /* no FT_FRAME_START */
        FT_FRAME_USHORT( platformID ),
        FT_FRAME_USHORT( encodingID ),
        FT_FRAME_USHORT( languageID ),
        FT_FRAME_USHORT( nameID ),
        FT_FRAME_USHORT( stringLength ),
        FT_FRAME_USHORT( stringOffset ),
      FT_FRAME_END
    };

    static const FT_Frame_Field  langTag_record_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_LangTagRec

      /* no FT_FRAME_START */
        FT_FRAME_USHORT( stringLength ),
        FT_FRAME_USHORT( stringOffset ),
      FT_FRAME_END
    };


    table         = &face->name_table;
    table->stream = stream;

    error = face->goto_table( face, TTAG_name, stream, &table_len );
    if ( error )
      goto Exit;

    table_pos = FT_STREAM_POS();

    if ( FT_STREAM_READ_FIELDS( name_table_fields, table ) )
      goto Exit;

    /* Some popular Asian fonts have an invalid `storageOffset' value (it */
    /* should be at least `6 + 12*numNameRecords').  However, the string  */
    /* offsets, computed as `storageOffset + entry->stringOffset', are    */
    /* valid pointers within the name table...                            */
    /*                                                                    */
    /* We thus can't check `storageOffset' right now.                     */
    /*                                                                    */
    storage_start = table_pos + 6 + 12 * table->numNameRecords;
    storage_limit = table_pos + table_len;

    if ( storage_start > storage_limit )
    {
      FT_ERROR(( "tt_face_load_name: invalid `name' table\n" ));
      error = FT_THROW( Name_Table_Missing );
      goto Exit;
    }

    /* `name' format 1 contains additional language tag records, */
    /* which we load first                                       */
    if ( table->format == 1 )
    {
      if ( FT_STREAM_SEEK( storage_start )            ||
           FT_READ_USHORT( table->numLangTagRecords ) )
        goto Exit;

      storage_start += 2 + 4 * table->numLangTagRecords;

      /* allocate language tag records array */
      if ( FT_QNEW_ARRAY( langTags, table->numLangTagRecords ) ||
           FT_FRAME_ENTER( table->numLangTagRecords * 4 )      )
        goto Exit;

      /* load language tags */
      {
        TT_LangTag  entry = langTags;
        TT_LangTag  limit = FT_OFFSET( entry, table->numLangTagRecords );


        for ( ; entry < limit; entry++ )
        {
          (void)FT_STREAM_READ_FIELDS( langTag_record_fields, entry );

          /* check that the langTag string is within the table */
          entry->stringOffset += table_pos + table->storageOffset;
          if ( entry->stringOffset                       < storage_start ||
               entry->stringOffset + entry->stringLength > storage_limit )
          {
            /* invalid entry; ignore it */
            entry->stringLength = 0;
          }

          /* mark the string as not yet loaded */
          entry->string = NULL;
        }

        table->langTags = langTags;
        langTags        = NULL;
      }

      FT_FRAME_EXIT();

      (void)FT_STREAM_SEEK( table_pos + 6 );
    }

    /* allocate name records array */
    if ( FT_QNEW_ARRAY( names, table->numNameRecords ) ||
         FT_FRAME_ENTER( table->numNameRecords * 12 )  )
      goto Exit;

    /* load name records */
    {
      TT_Name  entry = names;
      FT_UInt  count = table->numNameRecords;
      FT_UInt  valid = 0;


      for ( ; count > 0; count-- )
      {
        if ( FT_STREAM_READ_FIELDS( name_record_fields, entry ) )
          continue;

        /* check that the name is not empty */
        if ( entry->stringLength == 0 )
          continue;

        /* check that the name string is within the table */
        entry->stringOffset += table_pos + table->storageOffset;
        if ( entry->stringOffset                       < storage_start ||
             entry->stringOffset + entry->stringLength > storage_limit )
        {
          /* invalid entry; ignore it */
          continue;
        }

        /* assure that we have a valid language tag ID, and   */
        /* that the corresponding langTag entry is valid, too */
        if ( table->format == 1 && entry->languageID >= 0x8000U )
        {
          if ( entry->languageID - 0x8000U >= table->numLangTagRecords    ||
               !table->langTags[entry->languageID - 0x8000U].stringLength )
          {
            /* invalid entry; ignore it */
            continue;
          }
        }

        /* mark the string as not yet converted */
        entry->string = NULL;

        valid++;
        entry++;
      }

      /* reduce array size to the actually used elements */
      FT_MEM_QRENEW_ARRAY( names,
                           table->numNameRecords,
                           valid );
      table->names          = names;
      names                 = NULL;
      table->numNameRecords = valid;
    }

    FT_FRAME_EXIT();

    /* everything went well, update face->num_names */
    face->num_names = (FT_UShort)table->numNameRecords;

  Exit:
    FT_FREE( names );
    FT_FREE( langTags );
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_free_name
   *
   * @Description:
   *   Frees the name records.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   */
  FT_LOCAL_DEF( void )
  tt_face_free_name( TT_Face  face )
  {
    FT_Memory     memory = face->root.driver->root.memory;
    TT_NameTable  table  = &face->name_table;


    if ( table->names )
    {
      TT_Name  entry = table->names;
      TT_Name  limit = entry + table->numNameRecords;


      for ( ; entry < limit; entry++ )
        FT_FREE( entry->string );

      FT_FREE( table->names );
    }

    if ( table->langTags )
    {
      TT_LangTag  entry = table->langTags;
      TT_LangTag  limit = entry + table->numLangTagRecords;


      for ( ; entry < limit; entry++ )
        FT_FREE( entry->string );

      FT_FREE( table->langTags );
    }

    table->numNameRecords    = 0;
    table->numLangTagRecords = 0;
    table->format            = 0;
    table->storageOffset     = 0;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_cmap
   *
   * @Description:
   *   Loads the cmap directory in a face object.  The cmaps themselves
   *   are loaded on demand in the `ttcmap.c' module.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */

  FT_LOCAL_DEF( FT_Error )
  tt_face_load_cmap( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error  error;


    error = face->goto_table( face, TTAG_cmap, stream, &face->cmap_size );
    if ( error )
      goto Exit;

    if ( FT_FRAME_EXTRACT( face->cmap_size, face->cmap_table ) )
      face->cmap_size = 0;

  Exit:
    return error;
  }



  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_os2
   *
   * @Description:
   *   Loads the OS2 table.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_os2( TT_Face    face,
                    FT_Stream  stream )
  {
    FT_Error  error;
    TT_OS2*   os2;

    static const FT_Frame_Field  os2_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_OS2

      FT_FRAME_START( 78 ),
        FT_FRAME_USHORT( version ),
        FT_FRAME_SHORT ( xAvgCharWidth ),
        FT_FRAME_USHORT( usWeightClass ),
        FT_FRAME_USHORT( usWidthClass ),
        FT_FRAME_SHORT ( fsType ),
        FT_FRAME_SHORT ( ySubscriptXSize ),
        FT_FRAME_SHORT ( ySubscriptYSize ),
        FT_FRAME_SHORT ( ySubscriptXOffset ),
        FT_FRAME_SHORT ( ySubscriptYOffset ),
        FT_FRAME_SHORT ( ySuperscriptXSize ),
        FT_FRAME_SHORT ( ySuperscriptYSize ),
        FT_FRAME_SHORT ( ySuperscriptXOffset ),
        FT_FRAME_SHORT ( ySuperscriptYOffset ),
        FT_FRAME_SHORT ( yStrikeoutSize ),
        FT_FRAME_SHORT ( yStrikeoutPosition ),
        FT_FRAME_SHORT ( sFamilyClass ),
        FT_FRAME_BYTE  ( panose[0] ),
        FT_FRAME_BYTE  ( panose[1] ),
        FT_FRAME_BYTE  ( panose[2] ),
        FT_FRAME_BYTE  ( panose[3] ),
        FT_FRAME_BYTE  ( panose[4] ),
        FT_FRAME_BYTE  ( panose[5] ),
        FT_FRAME_BYTE  ( panose[6] ),
        FT_FRAME_BYTE  ( panose[7] ),
        FT_FRAME_BYTE  ( panose[8] ),
        FT_FRAME_BYTE  ( panose[9] ),
        FT_FRAME_ULONG ( ulUnicodeRange1 ),
        FT_FRAME_ULONG ( ulUnicodeRange2 ),
        FT_FRAME_ULONG ( ulUnicodeRange3 ),
        FT_FRAME_ULONG ( ulUnicodeRange4 ),
        FT_FRAME_BYTE  ( achVendID[0] ),
        FT_FRAME_BYTE  ( achVendID[1] ),
        FT_FRAME_BYTE  ( achVendID[2] ),
        FT_FRAME_BYTE  ( achVendID[3] ),

        FT_FRAME_USHORT( fsSelection ),
        FT_FRAME_USHORT( usFirstCharIndex ),
        FT_FRAME_USHORT( usLastCharIndex ),
        FT_FRAME_SHORT ( sTypoAscender ),
        FT_FRAME_SHORT ( sTypoDescender ),
        FT_FRAME_SHORT ( sTypoLineGap ),
        FT_FRAME_USHORT( usWinAscent ),
        FT_FRAME_USHORT( usWinDescent ),
      FT_FRAME_END
    };

    /* `OS/2' version 1 and newer */
    static const FT_Frame_Field  os2_fields_extra1[] =
    {
      FT_FRAME_START( 8 ),
        FT_FRAME_ULONG( ulCodePageRange1 ),
        FT_FRAME_ULONG( ulCodePageRange2 ),
      FT_FRAME_END
    };

    /* `OS/2' version 2 and newer */
    static const FT_Frame_Field  os2_fields_extra2[] =
    {
      FT_FRAME_START( 10 ),
        FT_FRAME_SHORT ( sxHeight ),
        FT_FRAME_SHORT ( sCapHeight ),
        FT_FRAME_USHORT( usDefaultChar ),
        FT_FRAME_USHORT( usBreakChar ),
        FT_FRAME_USHORT( usMaxContext ),
      FT_FRAME_END
    };

    /* `OS/2' version 5 and newer */
    static const FT_Frame_Field  os2_fields_extra5[] =
    {
      FT_FRAME_START( 4 ),
        FT_FRAME_USHORT( usLowerOpticalPointSize ),
        FT_FRAME_USHORT( usUpperOpticalPointSize ),
      FT_FRAME_END
    };


    /* We now support old Mac fonts where the OS/2 table doesn't  */
    /* exist.  Simply put, we set the `version' field to 0xFFFF   */
    /* and test this value each time we need to access the table. */
    error = face->goto_table( face, TTAG_OS2, stream, 0 );
    if ( error )
      goto Exit;

    os2 = &face->os2;

    if ( FT_STREAM_READ_FIELDS( os2_fields, os2 ) )
      goto Exit;

    os2->ulCodePageRange1        = 0;
    os2->ulCodePageRange2        = 0;
    os2->sxHeight                = 0;
    os2->sCapHeight              = 0;
    os2->usDefaultChar           = 0;
    os2->usBreakChar             = 0;
    os2->usMaxContext            = 0;
    os2->usLowerOpticalPointSize = 0;
    os2->usUpperOpticalPointSize = 0xFFFF;

    if ( os2->version >= 0x0001 )
    {
      /* only version 1 tables */
      if ( FT_STREAM_READ_FIELDS( os2_fields_extra1, os2 ) )
        goto Exit;

      if ( os2->version >= 0x0002 )
      {
        /* only version 2 tables */
        if ( FT_STREAM_READ_FIELDS( os2_fields_extra2, os2 ) )
          goto Exit;

        if ( os2->version >= 0x0005 )
        {
          /* only version 5 tables */
          if ( FT_STREAM_READ_FIELDS( os2_fields_extra5, os2 ) )
            goto Exit;
        }
      }
    }

    FT_TRACE3(( "sTypoAscender:  %4d\n",   os2->sTypoAscender ));
    FT_TRACE3(( "sTypoDescender: %4d\n",   os2->sTypoDescender ));
    FT_TRACE3(( "usWinAscent:    %4u\n",   os2->usWinAscent ));
    FT_TRACE3(( "usWinDescent:   %4u\n",   os2->usWinDescent ));
    FT_TRACE3(( "fsSelection:    0x%2x\n", os2->fsSelection ));

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_postscript
   *
   * @Description:
   *   Loads the Postscript table.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_post( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error        error;
    TT_Postscript*  post = &face->postscript;

    static const FT_Frame_Field  post_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_Postscript

      FT_FRAME_START( 32 ),
        FT_FRAME_LONG ( FormatType ),
        FT_FRAME_LONG ( italicAngle ),
        FT_FRAME_SHORT( underlinePosition ),
        FT_FRAME_SHORT( underlineThickness ),
        FT_FRAME_ULONG( isFixedPitch ),
        FT_FRAME_ULONG( minMemType42 ),
        FT_FRAME_ULONG( maxMemType42 ),
        FT_FRAME_ULONG( minMemType1 ),
        FT_FRAME_ULONG( maxMemType1 ),
      FT_FRAME_END
    };


    error = face->goto_table( face, TTAG_post, stream, 0 );
    if ( error )
      return error;

    if ( FT_STREAM_READ_FIELDS( post_fields, post ) )
      return error;

    if ( post->FormatType != 0x00030000L &&
         post->FormatType != 0x00025000L &&
         post->FormatType != 0x00020000L &&
         post->FormatType != 0x00010000L )
      return FT_THROW( Invalid_Post_Table_Format );

    /* we don't load the glyph names, we do that in another */
    /* module (ttpost).                                     */

    FT_TRACE3(( "FormatType:   0x%lx\n", post->FormatType ));
    FT_TRACE3(( "isFixedPitch:   %s\n", post->isFixedPitch
                                        ? "  yes" : "   no" ));

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_pclt
   *
   * @Description:
   *   Loads the PCL 5 Table.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_pclt( TT_Face    face,
                     FT_Stream  stream )
  {
    static const FT_Frame_Field  pclt_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TT_PCLT

      FT_FRAME_START( 54 ),
        FT_FRAME_ULONG ( Version ),
        FT_FRAME_ULONG ( FontNumber ),
        FT_FRAME_USHORT( Pitch ),
        FT_FRAME_USHORT( xHeight ),
        FT_FRAME_USHORT( Style ),
        FT_FRAME_USHORT( TypeFamily ),
        FT_FRAME_USHORT( CapHeight ),
        FT_FRAME_USHORT( SymbolSet ),
        FT_FRAME_BYTES ( TypeFace, 16 ),
        FT_FRAME_BYTES ( CharacterComplement, 8 ),
        FT_FRAME_BYTES ( FileName, 6 ),
        FT_FRAME_CHAR  ( StrokeWeight ),
        FT_FRAME_CHAR  ( WidthType ),
        FT_FRAME_BYTE  ( SerifStyle ),
        FT_FRAME_BYTE  ( Reserved ),
      FT_FRAME_END
    };

    FT_Error  error;
    TT_PCLT*  pclt = &face->pclt;


    /* optional table */
    error = face->goto_table( face, TTAG_PCLT, stream, 0 );
    if ( error )
      goto Exit;

    if ( FT_STREAM_READ_FIELDS( pclt_fields, pclt ) )
      goto Exit;

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_gasp
   *
   * @Description:
   *   Loads the `gasp' table into a face object.
   *
   * @Input:
   *   face ::
   *     A handle to the target face object.
   *
   *   stream ::
   *     The input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_gasp( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = stream->memory;

    FT_UShort      j, num_ranges;
    TT_GaspRange   gasp_ranges = NULL;


    /* the gasp table is optional */
    error = face->goto_table( face, TTAG_gasp, stream, 0 );
    if ( error )
      goto Exit;

    if ( FT_FRAME_ENTER( 4L ) )
      goto Exit;

    face->gasp.version = FT_GET_USHORT();
    num_ranges         = FT_GET_USHORT();

    FT_FRAME_EXIT();

    /* only support versions 0 and 1 of the table */
    if ( face->gasp.version >= 2 )
    {
      face->gasp.numRanges = 0;
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    FT_TRACE3(( "numRanges: %hu\n", num_ranges ));

    if ( FT_QNEW_ARRAY( gasp_ranges, num_ranges ) ||
         FT_FRAME_ENTER( num_ranges * 4L )        )
      goto Exit;

    for ( j = 0; j < num_ranges; j++ )
    {
      gasp_ranges[j].maxPPEM  = FT_GET_USHORT();
      gasp_ranges[j].gaspFlag = FT_GET_USHORT();

      FT_TRACE3(( "gaspRange %d: rangeMaxPPEM %5d, rangeGaspBehavior 0x%x\n",
                  j,
                  gasp_ranges[j].maxPPEM,
                  gasp_ranges[j].gaspFlag ));
    }

    face->gasp.gaspRanges = gasp_ranges;
    gasp_ranges           = NULL;
    face->gasp.numRanges  = num_ranges;

    FT_FRAME_EXIT();

  Exit:
    FT_FREE( gasp_ranges );
    return error;
  }


/* END */
