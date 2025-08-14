/****************************************************************************
 *
 * winfnt.h
 *
 *   FreeType font driver for Windows FNT/FON files
 *
 * Copyright (C) 1996-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 * Copyright 2007 Dmitry Timoshkov for Codeweavers
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef WINFNT_H_
#define WINFNT_H_


#include <freetype/ftwinfnt.h>
#include <freetype/internal/ftdrv.h>


FT_BEGIN_HEADER


  typedef struct  WinMZ_HeaderRec_
  {
    FT_UShort  magic;
    /* skipped content */
    FT_UShort  lfanew;

  } WinMZ_HeaderRec;


  typedef struct  WinNE_HeaderRec_
  {
    FT_UShort  magic;
    /* skipped content */
    FT_UShort  resource_tab_offset;
    FT_UShort  rname_tab_offset;

  } WinNE_HeaderRec;


  typedef struct  WinPE32_HeaderRec_
  {
    FT_ULong   magic;
    FT_UShort  machine;
    FT_UShort  number_of_sections;
    /* skipped content */
    FT_UShort  size_of_optional_header;
    /* skipped content */
    FT_UShort  magic32;
    /* skipped content */
    FT_ULong   rsrc_virtual_address;
    FT_ULong   rsrc_size;
    /* skipped content */

  } WinPE32_HeaderRec;


  typedef struct  WinPE32_SectionRec_
  {
    FT_Byte   name[8];
    /* skipped content */
    FT_ULong  virtual_address;
    FT_ULong  size_of_raw_data;
    FT_ULong  pointer_to_raw_data;
    /* skipped content */

  } WinPE32_SectionRec;


  typedef struct  WinPE_RsrcDirRec_
  {
    FT_ULong   characteristics;
    FT_ULong   time_date_stamp;
    FT_UShort  major_version;
    FT_UShort  minor_version;
    FT_UShort  number_of_named_entries;
    FT_UShort  number_of_id_entries;

  } WinPE_RsrcDirRec;


  typedef struct  WinPE_RsrcDirEntryRec_
  {
    FT_ULong  name;
    FT_ULong  offset;

  } WinPE_RsrcDirEntryRec;


  typedef struct  WinPE_RsrcDataEntryRec_
  {
    FT_ULong  offset_to_data;
    FT_ULong  size;
    FT_ULong  code_page;
    FT_ULong  reserved;

  } WinPE_RsrcDataEntryRec;


  typedef struct  WinNameInfoRec_
  {
    FT_UShort  offset;
    FT_UShort  length;
    FT_UShort  flags;
    FT_UShort  id;
    FT_UShort  handle;
    FT_UShort  usage;

  } WinNameInfoRec;


  typedef struct  WinResourceInfoRec_
  {
    FT_UShort  type_id;
    FT_UShort  count;

  } WinResourceInfoRec;


#define WINFNT_MZ_MAGIC  0x5A4D
#define WINFNT_NE_MAGIC  0x454E
#define WINFNT_PE_MAGIC  0x4550


  typedef struct  FNT_FontRec_
  {
    FT_ULong             offset;

    FT_WinFNT_HeaderRec  header;

    FT_Byte*             fnt_frame;
    FT_ULong             fnt_size;
    FT_String*           family_name;

  } FNT_FontRec, *FNT_Font;


  typedef struct  FNT_FaceRec_
  {
    FT_FaceRec     root;
    FNT_Font       font;

  } FNT_FaceRec, *FNT_Face;


  FT_EXPORT_VAR( const FT_Driver_ClassRec )  winfnt_driver_class;


FT_END_HEADER


#endif /* WINFNT_H_ */


/* END */
