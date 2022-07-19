/****************************************************************************
 *
 * ftmac.h
 *
 *   Additional Mac-specific API.
 *
 * Copyright (C) 1996-2020 by
 * Just van Rossum, David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


/****************************************************************************
 *
 * NOTE: Include this file after `FT_FREETYPE_H` and after any
 *       Mac-specific headers (because this header uses Mac types such as
 *       'Handle', 'FSSpec', 'FSRef', etc.)
 *
 */


#ifndef FTMAC_H_
#define FTMAC_H_




FT_BEGIN_HEADER


  /* gcc-3.1 and later can warn about functions tagged as deprecated */
#ifndef FT_DEPRECATED_ATTRIBUTE
#if defined( __GNUC__ )                                     && \
    ( ( __GNUC__ >= 4 )                                  ||    \
      ( ( __GNUC__ == 3 ) && ( __GNUC_MINOR__ >= 1 ) ) )
#define FT_DEPRECATED_ATTRIBUTE  __attribute__(( deprecated ))
#else
#define FT_DEPRECATED_ATTRIBUTE
#endif
#endif


  /**************************************************************************
   *
   * @section:
   *   mac_specific
   *
   * @title:
   *   Mac Specific Interface
   *
   * @abstract:
   *   Only available on the Macintosh.
   *
   * @description:
   *   The following definitions are only available if FreeType is compiled
   *   on a Macintosh.
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_New_Face_From_FOND
   *
   * @description:
   *   Create a new face object from a FOND resource.
   *
   * @inout:
   *   library ::
   *     A handle to the library resource.
   *
   * @input:
   *   fond ::
   *     A FOND resource.
   *
   *   face_index ::
   *     Only supported for the -1 'sanity check' special case.
   *
   * @output:
   *   aface ::
   *     A handle to a new face object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @example:
   *   This function can be used to create @FT_Face objects from fonts that
   *   are installed in the system as follows.
   *
   *   ```
   *     fond  = GetResource( 'FOND', fontName );
   *     error = FT_New_Face_From_FOND( library, fond, 0, &face );
   *   ```
   */
  FT_EXPORT( FT_Error )
  FT_New_Face_From_FOND( FT_Library  library,
                         Handle      fond,
                         FT_Long     face_index,
                         FT_Face    *aface )
                       FT_DEPRECATED_ATTRIBUTE;


  /**************************************************************************
   *
   * @function:
   *   FT_GetFile_From_Mac_Name
   *
   * @description:
   *   Return an FSSpec for the disk file containing the named font.
   *
   * @input:
   *   fontName ::
   *     Mac OS name of the font (e.g., Times New Roman Bold).
   *
   * @output:
   *   pathSpec ::
   *     FSSpec to the file.  For passing to @FT_New_Face_From_FSSpec.
   *
   *   face_index ::
   *     Index of the face.  For passing to @FT_New_Face_From_FSSpec.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_GetFile_From_Mac_Name( const char*  fontName,
                            FSSpec*      pathSpec,
                            FT_Long*     face_index )
                          FT_DEPRECATED_ATTRIBUTE;


  /**************************************************************************
   *
   * @function:
   *   FT_GetFile_From_Mac_ATS_Name
   *
   * @description:
   *   Return an FSSpec for the disk file containing the named font.
   *
   * @input:
   *   fontName ::
   *     Mac OS name of the font in ATS framework.
   *
   * @output:
   *   pathSpec ::
   *     FSSpec to the file. For passing to @FT_New_Face_From_FSSpec.
   *
   *   face_index ::
   *     Index of the face. For passing to @FT_New_Face_From_FSSpec.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_GetFile_From_Mac_ATS_Name( const char*  fontName,
                                FSSpec*      pathSpec,
                                FT_Long*     face_index )
                              FT_DEPRECATED_ATTRIBUTE;


  /**************************************************************************
   *
   * @function:
   *   FT_GetFilePath_From_Mac_ATS_Name
   *
   * @description:
   *   Return a pathname of the disk file and face index for given font name
   *   that is handled by ATS framework.
   *
   * @input:
   *   fontName ::
   *     Mac OS name of the font in ATS framework.
   *
   * @output:
   *   path ::
   *     Buffer to store pathname of the file.  For passing to @FT_New_Face.
   *     The client must allocate this buffer before calling this function.
   *
   *   maxPathSize ::
   *     Lengths of the buffer `path` that client allocated.
   *
   *   face_index ::
   *     Index of the face.  For passing to @FT_New_Face.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  FT_EXPORT( FT_Error )
  FT_GetFilePath_From_Mac_ATS_Name( const char*  fontName,
                                    UInt8*       path,
                                    UInt32       maxPathSize,
                                    FT_Long*     face_index )
                                  FT_DEPRECATED_ATTRIBUTE;


  /**************************************************************************
   *
   * @function:
   *   FT_New_Face_From_FSSpec
   *
   * @description:
   *   Create a new face object from a given resource and typeface index
   *   using an FSSpec to the font file.
   *
   * @inout:
   *   library ::
   *     A handle to the library resource.
   *
   * @input:
   *   spec ::
   *     FSSpec to the font file.
   *
   *   face_index ::
   *     The index of the face within the resource.  The first face has
   *     index~0.
   * @output:
   *   aface ::
   *     A handle to a new face object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   @FT_New_Face_From_FSSpec is identical to @FT_New_Face except it
   *   accepts an FSSpec instead of a path.
   */
  FT_EXPORT( FT_Error )
  FT_New_Face_From_FSSpec( FT_Library     library,
                           const FSSpec  *spec,
                           FT_Long        face_index,
                           FT_Face       *aface )
                         FT_DEPRECATED_ATTRIBUTE;


  /**************************************************************************
   *
   * @function:
   *   FT_New_Face_From_FSRef
   *
   * @description:
   *   Create a new face object from a given resource and typeface index
   *   using an FSRef to the font file.
   *
   * @inout:
   *   library ::
   *     A handle to the library resource.
   *
   * @input:
   *   spec ::
   *     FSRef to the font file.
   *
   *   face_index ::
   *     The index of the face within the resource.  The first face has
   *     index~0.
   * @output:
   *   aface ::
   *     A handle to a new face object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   @FT_New_Face_From_FSRef is identical to @FT_New_Face except it accepts
   *   an FSRef instead of a path.
   */
  FT_EXPORT( FT_Error )
  FT_New_Face_From_FSRef( FT_Library    library,
                          const FSRef  *ref,
                          FT_Long       face_index,
                          FT_Face      *aface )
                        FT_DEPRECATED_ATTRIBUTE;

  /* */


FT_END_HEADER


#endif /* FTMAC_H_ */


/* END */
