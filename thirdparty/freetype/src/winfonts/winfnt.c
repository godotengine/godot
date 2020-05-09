/****************************************************************************
 *
 * winfnt.c
 *
 *   FreeType font driver for Windows FNT/FON files
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 * Copyright 2003 Huw D M Davies for Codeweavers
 * Copyright 2007 Dmitry Timoshkov for Codeweavers
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_WINFONTS_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_OBJECTS_H
#include FT_TRUETYPE_IDS_H

#include "winfnt.h"
#include "fnterrs.h"
#include FT_SERVICE_WINFNT_H
#include FT_SERVICE_FONT_FORMAT_H

  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  winfnt


  static const FT_Frame_Field  winmz_header_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinMZ_HeaderRec

    FT_FRAME_START( 64 ),
      FT_FRAME_USHORT_LE ( magic ),
      FT_FRAME_SKIP_BYTES( 29 * 2 ),
      FT_FRAME_ULONG_LE  ( lfanew ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winne_header_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinNE_HeaderRec

    FT_FRAME_START( 40 ),
      FT_FRAME_USHORT_LE ( magic ),
      FT_FRAME_SKIP_BYTES( 34 ),
      FT_FRAME_USHORT_LE ( resource_tab_offset ),
      FT_FRAME_USHORT_LE ( rname_tab_offset ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winpe32_header_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinPE32_HeaderRec

    FT_FRAME_START( 248 ),
      FT_FRAME_ULONG_LE  ( magic ),   /* PE00 */
      FT_FRAME_USHORT_LE ( machine ), /* 0x014C - i386 */
      FT_FRAME_USHORT_LE ( number_of_sections ),
      FT_FRAME_SKIP_BYTES( 12 ),
      FT_FRAME_USHORT_LE ( size_of_optional_header ),
      FT_FRAME_SKIP_BYTES( 2 ),
      FT_FRAME_USHORT_LE ( magic32 ), /* 0x10B */
      FT_FRAME_SKIP_BYTES( 110 ),
      FT_FRAME_ULONG_LE  ( rsrc_virtual_address ),
      FT_FRAME_ULONG_LE  ( rsrc_size ),
      FT_FRAME_SKIP_BYTES( 104 ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winpe32_section_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinPE32_SectionRec

    FT_FRAME_START( 40 ),
      FT_FRAME_BYTES     ( name, 8 ),
      FT_FRAME_SKIP_BYTES( 4 ),
      FT_FRAME_ULONG_LE  ( virtual_address ),
      FT_FRAME_ULONG_LE  ( size_of_raw_data ),
      FT_FRAME_ULONG_LE  ( pointer_to_raw_data ),
      FT_FRAME_SKIP_BYTES( 16 ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winpe_rsrc_dir_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinPE_RsrcDirRec

    FT_FRAME_START( 16 ),
      FT_FRAME_ULONG_LE ( characteristics ),
      FT_FRAME_ULONG_LE ( time_date_stamp ),
      FT_FRAME_USHORT_LE( major_version ),
      FT_FRAME_USHORT_LE( minor_version ),
      FT_FRAME_USHORT_LE( number_of_named_entries ),
      FT_FRAME_USHORT_LE( number_of_id_entries ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winpe_rsrc_dir_entry_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinPE_RsrcDirEntryRec

    FT_FRAME_START( 8 ),
      FT_FRAME_ULONG_LE( name ),
      FT_FRAME_ULONG_LE( offset ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winpe_rsrc_data_entry_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WinPE_RsrcDataEntryRec

    FT_FRAME_START( 16 ),
      FT_FRAME_ULONG_LE( offset_to_data ),
      FT_FRAME_ULONG_LE( size ),
      FT_FRAME_ULONG_LE( code_page ),
      FT_FRAME_ULONG_LE( reserved ),
    FT_FRAME_END
  };

  static const FT_Frame_Field  winfnt_header_fields[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  FT_WinFNT_HeaderRec

    FT_FRAME_START( 148 ),
      FT_FRAME_USHORT_LE( version ),
      FT_FRAME_ULONG_LE ( file_size ),
      FT_FRAME_BYTES    ( copyright, 60 ),
      FT_FRAME_USHORT_LE( file_type ),
      FT_FRAME_USHORT_LE( nominal_point_size ),
      FT_FRAME_USHORT_LE( vertical_resolution ),
      FT_FRAME_USHORT_LE( horizontal_resolution ),
      FT_FRAME_USHORT_LE( ascent ),
      FT_FRAME_USHORT_LE( internal_leading ),
      FT_FRAME_USHORT_LE( external_leading ),
      FT_FRAME_BYTE     ( italic ),
      FT_FRAME_BYTE     ( underline ),
      FT_FRAME_BYTE     ( strike_out ),
      FT_FRAME_USHORT_LE( weight ),
      FT_FRAME_BYTE     ( charset ),
      FT_FRAME_USHORT_LE( pixel_width ),
      FT_FRAME_USHORT_LE( pixel_height ),
      FT_FRAME_BYTE     ( pitch_and_family ),
      FT_FRAME_USHORT_LE( avg_width ),
      FT_FRAME_USHORT_LE( max_width ),
      FT_FRAME_BYTE     ( first_char ),
      FT_FRAME_BYTE     ( last_char ),
      FT_FRAME_BYTE     ( default_char ),
      FT_FRAME_BYTE     ( break_char ),
      FT_FRAME_USHORT_LE( bytes_per_row ),
      FT_FRAME_ULONG_LE ( device_offset ),
      FT_FRAME_ULONG_LE ( face_name_offset ),
      FT_FRAME_ULONG_LE ( bits_pointer ),
      FT_FRAME_ULONG_LE ( bits_offset ),
      FT_FRAME_BYTE     ( reserved ),
      FT_FRAME_ULONG_LE ( flags ),
      FT_FRAME_USHORT_LE( A_space ),
      FT_FRAME_USHORT_LE( B_space ),
      FT_FRAME_USHORT_LE( C_space ),
      FT_FRAME_ULONG_LE ( color_table_offset ),
      FT_FRAME_BYTES    ( reserved1, 16 ),
    FT_FRAME_END
  };


  static void
  fnt_font_done( FNT_Face face )
  {
    FT_Memory  memory = FT_FACE( face )->memory;
    FT_Stream  stream = FT_FACE( face )->stream;
    FNT_Font   font   = face->font;


    if ( !font )
      return;

    if ( font->fnt_frame )
      FT_FRAME_RELEASE( font->fnt_frame );
    FT_FREE( font->family_name );

    FT_FREE( font );
    face->font = NULL;
  }


  static FT_Error
  fnt_font_load( FNT_Font   font,
                 FT_Stream  stream )
  {
    FT_Error          error;
    FT_WinFNT_Header  header = &font->header;
    FT_Bool           new_format;
    FT_UInt           size;


    /* first of all, read the FNT header */
    if ( FT_STREAM_SEEK( font->offset )                        ||
         FT_STREAM_READ_FIELDS( winfnt_header_fields, header ) )
      goto Exit;

    /* check header */
    if ( header->version != 0x200 &&
         header->version != 0x300 )
    {
      FT_TRACE2(( "  not a Windows FNT file\n" ));
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    new_format = FT_BOOL( font->header.version == 0x300 );
    size       = new_format ? 148 : 118;

    if ( header->file_size < size )
    {
      FT_TRACE2(( "  not a Windows FNT file\n" ));
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    /* Version 2 doesn't have these fields */
    if ( header->version == 0x200 )
    {
      header->flags   = 0;
      header->A_space = 0;
      header->B_space = 0;
      header->C_space = 0;

      header->color_table_offset = 0;
    }

    if ( header->file_type & 1 )
    {
      FT_TRACE2(( "[can't handle vector FNT fonts]\n" ));
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    /* this is a FNT file/table; extract its frame */
    if ( FT_STREAM_SEEK( font->offset )                         ||
         FT_FRAME_EXTRACT( header->file_size, font->fnt_frame ) )
      goto Exit;

  Exit:
    return error;
  }


  static FT_Error
  fnt_face_get_dll_font( FNT_Face  face,
                         FT_Int    face_instance_index )
  {
    FT_Error         error;
    FT_Stream        stream = FT_FACE( face )->stream;
    FT_Memory        memory = FT_FACE( face )->memory;
    WinMZ_HeaderRec  mz_header;
    FT_Long          face_index;


    face->font = NULL;

    face_index = FT_ABS( face_instance_index ) & 0xFFFF;

    /* does it begin with an MZ header? */
    if ( FT_STREAM_SEEK( 0 )                                      ||
         FT_STREAM_READ_FIELDS( winmz_header_fields, &mz_header ) )
      goto Exit;

    error = FT_ERR( Unknown_File_Format );
    if ( mz_header.magic == WINFNT_MZ_MAGIC )
    {
      /* yes, now look for an NE header in the file */
      WinNE_HeaderRec  ne_header;


      FT_TRACE2(( "MZ signature found\n" ));

      if ( FT_STREAM_SEEK( mz_header.lfanew )                       ||
           FT_STREAM_READ_FIELDS( winne_header_fields, &ne_header ) )
        goto Exit;

      error = FT_ERR( Unknown_File_Format );
      if ( ne_header.magic == WINFNT_NE_MAGIC )
      {
        /* good, now look into the resource table for each FNT resource */
        FT_ULong   res_offset  = mz_header.lfanew +
                                   ne_header.resource_tab_offset;
        FT_UShort  size_shift;
        FT_UShort  font_count  = 0;
        FT_ULong   font_offset = 0;


        FT_TRACE2(( "NE signature found\n" ));

        if ( FT_STREAM_SEEK( res_offset )                    ||
             FT_FRAME_ENTER( ne_header.rname_tab_offset -
                             ne_header.resource_tab_offset ) )
          goto Exit;

        size_shift = FT_GET_USHORT_LE();

        /* Microsoft's specification of the executable-file header format */
        /* for `New Executable' (NE) doesn't give a limit for the         */
        /* alignment shift count; however, in 1985, the year of the       */
        /* specification release, only 32bit values were supported, thus  */
        /* anything larger than 16 doesn't make sense in general, given   */
        /* that file offsets are 16bit values, shifted by the alignment   */
        /* shift count                                                    */
        if ( size_shift > 16 )
        {
          FT_TRACE2(( "invalid alignment shift count for resource data\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit1;
        }


        for (;;)
        {
          FT_UShort  type_id, count;


          type_id = FT_GET_USHORT_LE();
          if ( !type_id )
            break;

          count = FT_GET_USHORT_LE();

          if ( type_id == 0x8008U )
          {
            font_count  = count;
            font_offset = FT_STREAM_POS() + 4 +
                          (FT_ULong)( stream->cursor - stream->limit );
            break;
          }

          stream->cursor += 4 + count * 12;
        }

        FT_FRAME_EXIT();

        if ( !font_count || !font_offset )
        {
          FT_TRACE2(( "this file doesn't contain any FNT resources\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        /* loading `winfnt_header_fields' needs at least 118 bytes;    */
        /* use this as a rough measure to check the expected font size */
        if ( font_count * 118UL > stream->size )
        {
          FT_TRACE2(( "invalid number of faces\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        face->root.num_faces = font_count;

        if ( face_instance_index < 0 )
          goto Exit;

        if ( face_index >= font_count )
        {
          error = FT_THROW( Invalid_Argument );
          goto Exit;
        }

        if ( FT_NEW( face->font ) )
          goto Exit;

        if ( FT_STREAM_SEEK( font_offset + (FT_ULong)face_index * 12 ) ||
             FT_FRAME_ENTER( 12 )                                      )
          goto Fail;

        face->font->offset   = (FT_ULong)FT_GET_USHORT_LE() << size_shift;
        face->font->fnt_size = (FT_ULong)FT_GET_USHORT_LE() << size_shift;

        stream->cursor += 8;

        FT_FRAME_EXIT();

        error = fnt_font_load( face->font, stream );
      }
      else if ( ne_header.magic == WINFNT_PE_MAGIC )
      {
        WinPE32_HeaderRec       pe32_header;
        WinPE32_SectionRec      pe32_section;
        WinPE_RsrcDirRec        root_dir, name_dir, lang_dir;
        WinPE_RsrcDirEntryRec   dir_entry1, dir_entry2, dir_entry3;
        WinPE_RsrcDataEntryRec  data_entry;

        FT_ULong   root_dir_offset, name_dir_offset, lang_dir_offset;
        FT_UShort  i, j, k;


        FT_TRACE2(( "PE signature found\n" ));

        if ( FT_STREAM_SEEK( mz_header.lfanew )                           ||
             FT_STREAM_READ_FIELDS( winpe32_header_fields, &pe32_header ) )
          goto Exit;

        FT_TRACE2(( "magic %04lx, machine %02x, number_of_sections %u, "
                    "size_of_optional_header %02x\n"
                    "magic32 %02x, rsrc_virtual_address %04lx, "
                    "rsrc_size %04lx\n",
                    pe32_header.magic, pe32_header.machine,
                    pe32_header.number_of_sections,
                    pe32_header.size_of_optional_header,
                    pe32_header.magic32, pe32_header.rsrc_virtual_address,
                    pe32_header.rsrc_size ));

        if ( pe32_header.magic != WINFNT_PE_MAGIC /* check full signature */ ||
             pe32_header.machine != 0x014C /* i386 */                        ||
             pe32_header.size_of_optional_header != 0xE0 /* FIXME */         ||
             pe32_header.magic32 != 0x10B                                    )
        {
          FT_TRACE2(( "this file has an invalid PE header\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        face->root.num_faces = 0;

        for ( i = 0; i < pe32_header.number_of_sections; i++ )
        {
          if ( FT_STREAM_READ_FIELDS( winpe32_section_fields,
                                      &pe32_section ) )
            goto Exit;

          FT_TRACE2(( "name %.8s, va %04lx, size %04lx, offset %04lx\n",
                      pe32_section.name, pe32_section.virtual_address,
                      pe32_section.size_of_raw_data,
                      pe32_section.pointer_to_raw_data ));

          if ( pe32_header.rsrc_virtual_address ==
                 pe32_section.virtual_address )
            goto Found_rsrc_section;
        }

        FT_TRACE2(( "this file doesn't contain any resources\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;

      Found_rsrc_section:
        FT_TRACE2(( "found resources section %.8s\n", pe32_section.name ));

        if ( FT_STREAM_SEEK( pe32_section.pointer_to_raw_data )        ||
             FT_STREAM_READ_FIELDS( winpe_rsrc_dir_fields, &root_dir ) )
          goto Exit;

        root_dir_offset = pe32_section.pointer_to_raw_data;

        for ( i = 0; i < root_dir.number_of_named_entries +
                           root_dir.number_of_id_entries; i++ )
        {
          if ( FT_STREAM_SEEK( root_dir_offset + 16 + i * 8 )      ||
               FT_STREAM_READ_FIELDS( winpe_rsrc_dir_entry_fields,
                                      &dir_entry1 )                )
            goto Exit;

          if ( !(dir_entry1.offset & 0x80000000UL ) /* DataIsDirectory */ )
          {
            error = FT_THROW( Invalid_File_Format );
            goto Exit;
          }

          dir_entry1.offset &= ~0x80000000UL;

          name_dir_offset = pe32_section.pointer_to_raw_data +
                            dir_entry1.offset;

          if ( FT_STREAM_SEEK( pe32_section.pointer_to_raw_data +
                               dir_entry1.offset )                       ||
               FT_STREAM_READ_FIELDS( winpe_rsrc_dir_fields, &name_dir ) )
            goto Exit;

          for ( j = 0; j < name_dir.number_of_named_entries +
                             name_dir.number_of_id_entries; j++ )
          {
            if ( FT_STREAM_SEEK( name_dir_offset + 16 + j * 8 )      ||
                 FT_STREAM_READ_FIELDS( winpe_rsrc_dir_entry_fields,
                                        &dir_entry2 )                )
              goto Exit;

            if ( !(dir_entry2.offset & 0x80000000UL ) /* DataIsDirectory */ )
            {
              error = FT_THROW( Invalid_File_Format );
              goto Exit;
            }

            dir_entry2.offset &= ~0x80000000UL;

            lang_dir_offset = pe32_section.pointer_to_raw_data +
                                dir_entry2.offset;

            if ( FT_STREAM_SEEK( pe32_section.pointer_to_raw_data +
                                   dir_entry2.offset )                     ||
                 FT_STREAM_READ_FIELDS( winpe_rsrc_dir_fields, &lang_dir ) )
              goto Exit;

            for ( k = 0; k < lang_dir.number_of_named_entries +
                               lang_dir.number_of_id_entries; k++ )
            {
              if ( FT_STREAM_SEEK( lang_dir_offset + 16 + k * 8 )      ||
                   FT_STREAM_READ_FIELDS( winpe_rsrc_dir_entry_fields,
                                          &dir_entry3 )                )
                goto Exit;

              if ( dir_entry2.offset & 0x80000000UL /* DataIsDirectory */ )
              {
                error = FT_THROW( Invalid_File_Format );
                goto Exit;
              }

              if ( dir_entry1.name == 8 /* RT_FONT */ )
              {
                if ( FT_STREAM_SEEK( root_dir_offset + dir_entry3.offset ) ||
                     FT_STREAM_READ_FIELDS( winpe_rsrc_data_entry_fields,
                                            &data_entry )                  )
                  goto Exit;

                FT_TRACE2(( "found font #%lu, offset %04lx, "
                            "size %04lx, cp %lu\n",
                            dir_entry2.name,
                            pe32_section.pointer_to_raw_data +
                              data_entry.offset_to_data -
                              pe32_section.virtual_address,
                            data_entry.size, data_entry.code_page ));

                if ( face_index == face->root.num_faces )
                {
                  if ( FT_NEW( face->font ) )
                    goto Exit;

                  face->font->offset   = pe32_section.pointer_to_raw_data +
                                           data_entry.offset_to_data -
                                           pe32_section.virtual_address;
                  face->font->fnt_size = data_entry.size;

                  error = fnt_font_load( face->font, stream );
                  if ( error )
                  {
                    FT_TRACE2(( "font #%lu load error 0x%x\n",
                                dir_entry2.name, error ));
                    goto Fail;
                  }
                  else
                    FT_TRACE2(( "font #%lu successfully loaded\n",
                                dir_entry2.name ));
                }

                face->root.num_faces++;
              }
            }
          }
        }
      }

      if ( !face->root.num_faces )
      {
        FT_TRACE2(( "this file doesn't contain any RT_FONT resources\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      if ( face_index >= face->root.num_faces )
      {
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }
    }

  Fail:
    if ( error )
      fnt_font_done( face );

  Exit:
    return error;

  Exit1:
    FT_FRAME_EXIT();
    goto Exit;
  }


  typedef struct  FNT_CMapRec_
  {
    FT_CMapRec  cmap;
    FT_UInt32   first;
    FT_UInt32   count;

  } FNT_CMapRec, *FNT_CMap;


  static FT_Error
  fnt_cmap_init( FNT_CMap    cmap,
                 FT_Pointer  pointer )
  {
    FNT_Face  face = (FNT_Face)FT_CMAP_FACE( cmap );
    FNT_Font  font = face->font;

    FT_UNUSED( pointer );


    cmap->first = (FT_UInt32)  font->header.first_char;
    cmap->count = (FT_UInt32)( font->header.last_char - cmap->first + 1 );

    return 0;
  }


  static FT_UInt
  fnt_cmap_char_index( FNT_CMap   cmap,
                       FT_UInt32  char_code )
  {
    FT_UInt  gindex = 0;


    char_code -= cmap->first;
    if ( char_code < cmap->count )
      /* we artificially increase the glyph index; */
      /* FNT_Load_Glyph reverts to the right one   */
      gindex = (FT_UInt)( char_code + 1 );
    return gindex;
  }


  static FT_UInt32
  fnt_cmap_char_next( FNT_CMap    cmap,
                      FT_UInt32  *pchar_code )
  {
    FT_UInt    gindex = 0;
    FT_UInt32  result = 0;
    FT_UInt32  char_code = *pchar_code + 1;


    if ( char_code <= cmap->first )
    {
      result = cmap->first;
      gindex = 1;
    }
    else
    {
      char_code -= cmap->first;
      if ( char_code < cmap->count )
      {
        result = cmap->first + char_code;
        gindex = (FT_UInt)( char_code + 1 );
      }
    }

    *pchar_code = result;
    return gindex;
  }


  static const FT_CMap_ClassRec  fnt_cmap_class_rec =
  {
    sizeof ( FNT_CMapRec ),

    (FT_CMap_InitFunc)     fnt_cmap_init,
    (FT_CMap_DoneFunc)     NULL,
    (FT_CMap_CharIndexFunc)fnt_cmap_char_index,
    (FT_CMap_CharNextFunc) fnt_cmap_char_next,

    NULL, NULL, NULL, NULL, NULL
  };

  static FT_CMap_Class const  fnt_cmap_class = &fnt_cmap_class_rec;


  static void
  FNT_Face_Done( FT_Face  fntface )       /* FNT_Face */
  {
    FNT_Face   face = (FNT_Face)fntface;
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );

    fnt_font_done( face );

    FT_FREE( fntface->available_sizes );
    fntface->num_fixed_sizes = 0;
  }


  static FT_Error
  FNT_Face_Init( FT_Stream      stream,
                 FT_Face        fntface,        /* FNT_Face */
                 FT_Int         face_instance_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    FNT_Face   face   = (FNT_Face)fntface;
    FT_Error   error;
    FT_Memory  memory = FT_FACE_MEMORY( face );
    FT_Int     face_index;

    FT_UNUSED( num_params );
    FT_UNUSED( params );


    FT_TRACE2(( "Windows FNT driver\n" ));

    face_index = FT_ABS( face_instance_index ) & 0xFFFF;

    /* try to load font from a DLL */
    error = fnt_face_get_dll_font( face, face_instance_index );
    if ( !error && face_instance_index < 0 )
      goto Exit;

    if ( FT_ERR_EQ( error, Unknown_File_Format ) )
    {
      /* this didn't work; try to load a single FNT font */
      FNT_Font  font;

      if ( FT_NEW( face->font ) )
        goto Exit;

      fntface->num_faces = 1;

      font           = face->font;
      font->offset   = 0;
      font->fnt_size = stream->size;

      error = fnt_font_load( font, stream );

      if ( !error )
      {
        if ( face_instance_index < 0 )
          goto Exit;

        if ( face_index > 0 )
          error = FT_THROW( Invalid_Argument );
      }
    }

    if ( error )
      goto Fail;

    /* sanity check */
    if ( !face->font->header.pixel_height )
    {
      FT_TRACE2(( "invalid pixel height\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    /* we now need to fill the root FT_Face fields */
    /* with relevant information                   */
    {
      FT_Face   root = FT_FACE( face );
      FNT_Font  font = face->font;
      FT_ULong  family_size;


      root->face_index = face_index;

      root->face_flags |= FT_FACE_FLAG_FIXED_SIZES |
                          FT_FACE_FLAG_HORIZONTAL;

      if ( font->header.avg_width == font->header.max_width )
        root->face_flags |= FT_FACE_FLAG_FIXED_WIDTH;

      if ( font->header.italic )
        root->style_flags |= FT_STYLE_FLAG_ITALIC;

      if ( font->header.weight >= 800 )
        root->style_flags |= FT_STYLE_FLAG_BOLD;

      /* set up the `fixed_sizes' array */
      if ( FT_NEW_ARRAY( root->available_sizes, 1 ) )
        goto Fail;

      root->num_fixed_sizes = 1;

      {
        FT_Bitmap_Size*  bsize = root->available_sizes;
        FT_UShort        x_res, y_res;


        bsize->width  = (FT_Short)font->header.avg_width;
        bsize->height = (FT_Short)( font->header.pixel_height +
                                    font->header.external_leading );
        bsize->size   = font->header.nominal_point_size << 6;

        x_res = font->header.horizontal_resolution;
        if ( !x_res )
          x_res = 72;

        y_res = font->header.vertical_resolution;
        if ( !y_res )
          y_res = 72;

        bsize->y_ppem = FT_MulDiv( bsize->size, y_res, 72 );
        bsize->y_ppem = FT_PIX_ROUND( bsize->y_ppem );

        /*
         * this reads:
         *
         * the nominal height is larger than the bbox's height
         *
         * => nominal_point_size contains incorrect value;
         *    use pixel_height as the nominal height
         */
        if ( bsize->y_ppem > ( font->header.pixel_height << 6 ) )
        {
          FT_TRACE2(( "use pixel_height as the nominal height\n" ));

          bsize->y_ppem = font->header.pixel_height << 6;
          bsize->size   = FT_MulDiv( bsize->y_ppem, 72, y_res );
        }

        bsize->x_ppem = FT_MulDiv( bsize->size, x_res, 72 );
        bsize->x_ppem = FT_PIX_ROUND( bsize->x_ppem );
      }

      {
        FT_CharMapRec  charmap;


        charmap.encoding    = FT_ENCODING_NONE;
        /* initial platform/encoding should indicate unset status? */
        charmap.platform_id = TT_PLATFORM_APPLE_UNICODE;
        charmap.encoding_id = TT_APPLE_ID_DEFAULT;
        charmap.face        = root;

        if ( font->header.charset == FT_WinFNT_ID_MAC )
        {
          charmap.encoding    = FT_ENCODING_APPLE_ROMAN;
          charmap.platform_id = TT_PLATFORM_MACINTOSH;
/*        charmap.encoding_id = TT_MAC_ID_ROMAN; */
        }

        error = FT_CMap_New( fnt_cmap_class,
                             NULL,
                             &charmap,
                             NULL );
        if ( error )
          goto Fail;
      }

      /* set up remaining flags */

      if ( font->header.last_char < font->header.first_char )
      {
        FT_TRACE2(( "invalid number of glyphs\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }

      /* reserve one slot for the .notdef glyph at index 0 */
      root->num_glyphs = font->header.last_char -
                         font->header.first_char + 1 + 1;

      if ( font->header.face_name_offset >= font->header.file_size )
      {
        FT_TRACE2(( "invalid family name offset\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }
      family_size = font->header.file_size - font->header.face_name_offset;
      /* Some broken fonts don't delimit the face name with a final */
      /* NULL byte -- the frame is erroneously one byte too small.  */
      /* We thus allocate one more byte, setting it explicitly to   */
      /* zero.                                                      */
      if ( FT_ALLOC( font->family_name, family_size + 1 ) )
        goto Fail;

      FT_MEM_COPY( font->family_name,
                   font->fnt_frame + font->header.face_name_offset,
                   family_size );

      font->family_name[family_size] = '\0';

      if ( FT_REALLOC( font->family_name,
                       family_size,
                       ft_strlen( font->family_name ) + 1 ) )
        goto Fail;

      root->family_name = font->family_name;
      root->style_name  = (char *)"Regular";

      if ( root->style_flags & FT_STYLE_FLAG_BOLD )
      {
        if ( root->style_flags & FT_STYLE_FLAG_ITALIC )
          root->style_name = (char *)"Bold Italic";
        else
          root->style_name = (char *)"Bold";
      }
      else if ( root->style_flags & FT_STYLE_FLAG_ITALIC )
        root->style_name = (char *)"Italic";
    }
    goto Exit;

  Fail:
    FNT_Face_Done( fntface );

  Exit:
    return error;
  }


  static FT_Error
  FNT_Size_Select( FT_Size   size,
                   FT_ULong  strike_index )
  {
    FNT_Face          face   = (FNT_Face)size->face;
    FT_WinFNT_Header  header = &face->font->header;

    FT_UNUSED( strike_index );


    FT_Select_Metrics( size->face, 0 );

    size->metrics.ascender    = header->ascent * 64;
    size->metrics.descender   = -( header->pixel_height -
                                   header->ascent ) * 64;
    size->metrics.max_advance = header->max_width * 64;

    return FT_Err_Ok;
  }


  static FT_Error
  FNT_Size_Request( FT_Size          size,
                    FT_Size_Request  req )
  {
    FNT_Face          face    = (FNT_Face)size->face;
    FT_WinFNT_Header  header  = &face->font->header;
    FT_Bitmap_Size*   bsize   = size->face->available_sizes;
    FT_Error          error   = FT_ERR( Invalid_Pixel_Size );
    FT_Long           height;


    height = FT_REQUEST_HEIGHT( req );
    height = ( height + 32 ) >> 6;

    switch ( req->type )
    {
    case FT_SIZE_REQUEST_TYPE_NOMINAL:
      if ( height == ( ( bsize->y_ppem + 32 ) >> 6 ) )
        error = FT_Err_Ok;
      break;

    case FT_SIZE_REQUEST_TYPE_REAL_DIM:
      if ( height == header->pixel_height )
        error = FT_Err_Ok;
      break;

    default:
      error = FT_THROW( Unimplemented_Feature );
      break;
    }

    if ( error )
      return error;
    else
      return FNT_Size_Select( size, 0 );
  }


  static FT_Error
  FNT_Load_Glyph( FT_GlyphSlot  slot,
                  FT_Size       size,
                  FT_UInt       glyph_index,
                  FT_Int32      load_flags )
  {
    FNT_Face    face   = (FNT_Face)FT_SIZE_FACE( size );
    FNT_Font    font;
    FT_Error    error  = FT_Err_Ok;
    FT_Byte*    p;
    FT_UInt     len;
    FT_Bitmap*  bitmap = &slot->bitmap;
    FT_ULong    offset;
    FT_Bool     new_format;


    if ( !face )
    {
      error = FT_THROW( Invalid_Face_Handle );
      goto Exit;
    }

    font = face->font;

    if ( !font                                                   ||
         glyph_index >= (FT_UInt)( FT_FACE( face )->num_glyphs ) )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_TRACE1(( "FNT_Load_Glyph: glyph index %d\n", glyph_index ));

    if ( glyph_index > 0 )
      glyph_index--;                           /* revert to real index */
    else
      glyph_index = font->header.default_char; /* the `.notdef' glyph  */

    new_format = FT_BOOL( font->header.version == 0x300 );
    len        = new_format ? 6 : 4;

    /* get glyph width and offset */
    offset = ( new_format ? 148 : 118 ) + len * glyph_index;

    if ( offset >= font->header.file_size - 2 - ( new_format ? 4 : 2 ) )
    {
      FT_TRACE2(( "invalid FNT offset\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    p = font->fnt_frame + offset;

    bitmap->width = FT_NEXT_USHORT_LE( p );

    /* jump to glyph entry */
    if ( new_format )
      offset = FT_NEXT_ULONG_LE( p );
    else
      offset = FT_NEXT_USHORT_LE( p );

    if ( offset >= font->header.file_size )
    {
      FT_TRACE2(( "invalid FNT offset\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    bitmap->rows       = font->header.pixel_height;
    bitmap->pixel_mode = FT_PIXEL_MODE_MONO;

    slot->bitmap_left     = 0;
    slot->bitmap_top      = font->header.ascent;
    slot->format          = FT_GLYPH_FORMAT_BITMAP;

    /* now set up metrics */
    slot->metrics.width        = (FT_Pos)( bitmap->width << 6 );
    slot->metrics.height       = (FT_Pos)( bitmap->rows << 6 );
    slot->metrics.horiAdvance  = (FT_Pos)( bitmap->width << 6 );
    slot->metrics.horiBearingX = 0;
    slot->metrics.horiBearingY = slot->bitmap_top << 6;

    ft_synthesize_vertical_metrics( &slot->metrics,
                                    (FT_Pos)( bitmap->rows << 6 ) );

    if ( load_flags & FT_LOAD_BITMAP_METRICS_ONLY )
      goto Exit;

    /* jump to glyph data */
    p = font->fnt_frame + /* font->header.bits_offset */ + offset;

    /* allocate and build bitmap */
    {
      FT_Memory  memory = FT_FACE_MEMORY( slot->face );
      FT_UInt    pitch  = ( bitmap->width + 7 ) >> 3;
      FT_Byte*   column;
      FT_Byte*   write;


      bitmap->pitch = (int)pitch;
      if ( !pitch                                                 ||
           offset + pitch * bitmap->rows > font->header.file_size )
      {
        FT_TRACE2(( "invalid bitmap width\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* note: since glyphs are stored in columns and not in rows we */
      /*       can't use ft_glyphslot_set_bitmap                     */
      if ( FT_ALLOC_MULT( bitmap->buffer, bitmap->rows, pitch ) )
        goto Exit;

      column = (FT_Byte*)bitmap->buffer;

      for ( ; pitch > 0; pitch--, column++ )
      {
        FT_Byte*  limit = p + bitmap->rows;


        for ( write = column; p < limit; p++, write += bitmap->pitch )
          *write = *p;
      }

      slot->internal->flags = FT_GLYPH_OWN_BITMAP;
    }

  Exit:
    return error;
  }


  static FT_Error
  winfnt_get_header( FT_Face               face,
                     FT_WinFNT_HeaderRec  *aheader )
  {
    FNT_Font  font = ((FNT_Face)face)->font;


    *aheader = font->header;

    return 0;
  }


  static const FT_Service_WinFntRec  winfnt_service_rec =
  {
    winfnt_get_header       /* get_header */
  };

  /*
   * SERVICE LIST
   *
   */

  static const FT_ServiceDescRec  winfnt_services[] =
  {
    { FT_SERVICE_ID_FONT_FORMAT, FT_FONT_FORMAT_WINFNT },
    { FT_SERVICE_ID_WINFNT,      &winfnt_service_rec },
    { NULL, NULL }
  };


  static FT_Module_Interface
  winfnt_get_service( FT_Module         module,
                      const FT_String*  service_id )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( winfnt_services, service_id );
  }




  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  winfnt_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER        |
      FT_MODULE_DRIVER_NO_OUTLINES,
      sizeof ( FT_DriverRec ),

      "winfonts",
      0x10000L,
      0x20000L,

      NULL, /* module-specific interface */

      NULL,                     /* FT_Module_Constructor  module_init   */
      NULL,                     /* FT_Module_Destructor   module_done   */
      winfnt_get_service        /* FT_Module_Requester    get_interface */
    },

    sizeof ( FNT_FaceRec ),
    sizeof ( FT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    FNT_Face_Init,              /* FT_Face_InitFunc  init_face */
    FNT_Face_Done,              /* FT_Face_DoneFunc  done_face */
    NULL,                       /* FT_Size_InitFunc  init_size */
    NULL,                       /* FT_Size_DoneFunc  done_size */
    NULL,                       /* FT_Slot_InitFunc  init_slot */
    NULL,                       /* FT_Slot_DoneFunc  done_slot */

    FNT_Load_Glyph,             /* FT_Slot_LoadFunc  load_glyph */

    NULL,                       /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
    NULL,                       /* FT_Face_GetAdvancesFunc  get_advances */

    FNT_Size_Request,           /* FT_Size_RequestFunc  request_size */
    FNT_Size_Select             /* FT_Size_SelectFunc   select_size  */
  };


/* END */
