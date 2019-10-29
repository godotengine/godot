/****************************************************************************
 *
 * wofftypes.h
 *
 *   Basic WOFF/WOFF2 type definitions and interface (specification
 *   only).
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


#ifndef WOFFTYPES_H_
#define WOFFTYPES_H_


#include <ft2build.h>
#include FT_TRUETYPE_TABLES_H
#include FT_INTERNAL_OBJECTS_H


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @struct:
   *   WOFF_HeaderRec
   *
   * @description:
   *   WOFF file format header.
   *
   * @fields:
   *   See
   *
   *     https://www.w3.org/TR/WOFF/#WOFFHeader
   */
  typedef struct  WOFF_HeaderRec_
  {
    FT_ULong   signature;
    FT_ULong   flavor;
    FT_ULong   length;
    FT_UShort  num_tables;
    FT_UShort  reserved;
    FT_ULong   totalSfntSize;
    FT_UShort  majorVersion;
    FT_UShort  minorVersion;
    FT_ULong   metaOffset;
    FT_ULong   metaLength;
    FT_ULong   metaOrigLength;
    FT_ULong   privOffset;
    FT_ULong   privLength;

  } WOFF_HeaderRec, *WOFF_Header;


  /**************************************************************************
   *
   * @struct:
   *   WOFF_TableRec
   *
   * @description:
   *   This structure describes a given table of a WOFF font.
   *
   * @fields:
   *   Tag ::
   *     A four-bytes tag describing the table.
   *
   *   Offset ::
   *     The offset of the table from the start of the WOFF font in its
   *     resource.
   *
   *   CompLength ::
   *     Compressed table length (in bytes).
   *
   *   OrigLength ::
   *     Uncompressed table length (in bytes).
   *
   *   CheckSum ::
   *     The table checksum.  This value can be ignored.
   *
   *   OrigOffset ::
   *     The uncompressed table file offset.  This value gets computed while
   *     constructing the (uncompressed) SFNT header.  It is not contained in
   *     the WOFF file.
   */
  typedef struct  WOFF_TableRec_
  {
    FT_ULong  Tag;           /* table ID                  */
    FT_ULong  Offset;        /* table file offset         */
    FT_ULong  CompLength;    /* compressed table length   */
    FT_ULong  OrigLength;    /* uncompressed table length */
    FT_ULong  CheckSum;      /* uncompressed checksum     */

    FT_ULong  OrigOffset;    /* uncompressed table file offset */
                             /* (not in the WOFF file)         */
  } WOFF_TableRec, *WOFF_Table;


FT_END_HEADER

#endif /* WOFFTYPES_H_ */


/* END */
