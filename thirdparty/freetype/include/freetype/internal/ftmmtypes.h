/****************************************************************************
 *
 * ftmmtypes.h
 *
 *   OpenType Variations type definitions for internal use
 *   with the multi-masters service (specification).
 *
 * Copyright (C) 2022-2023 by
 * David Turner, Robert Wilhelm, Werner Lemberg, George Williams, and
 * Dominik Röttsches.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTMMTYPES_H_
#define FTMMTYPES_H_

FT_BEGIN_HEADER


  typedef FT_Int32  FT_ItemVarDelta;

  typedef struct  GX_ItemVarDataRec_
  {
    FT_UInt            itemCount;       /* number of delta sets per item    */
    FT_UInt            regionIdxCount;  /* number of region indices         */
    FT_UInt*           regionIndices;   /* array of `regionCount' indices;  */
                                        /* these index `varRegionList'      */
    FT_ItemVarDelta*   deltaSet;        /* array of `itemCount' deltas      */
                                        /* use `innerIndex' for this array  */

  } GX_ItemVarDataRec, *GX_ItemVarData;


  /* contribution of one axis to a region */
  typedef struct  GX_AxisCoordsRec_
  {
    FT_Fixed  startCoord;
    FT_Fixed  peakCoord;      /* zero means no effect (factor = 1) */
    FT_Fixed  endCoord;

  } GX_AxisCoordsRec, *GX_AxisCoords;


  typedef struct  GX_VarRegionRec_
  {
    GX_AxisCoords  axisList;               /* array of axisCount records */

  } GX_VarRegionRec, *GX_VarRegion;


  /* item variation store */
  typedef struct  GX_ItemVarStoreRec_
  {
    FT_UInt         dataCount;
    GX_ItemVarData  varData;            /* array of dataCount records;     */
                                        /* use `outerIndex' for this array */
    FT_UShort     axisCount;
    FT_UInt       regionCount;          /* total number of regions defined */
    GX_VarRegion  varRegionList;

  } GX_ItemVarStoreRec, *GX_ItemVarStore;


  typedef struct  GX_DeltaSetIdxMapRec_
  {
    FT_ULong  mapCount;
    FT_UInt*  outerIndex;               /* indices to item var data */
    FT_UInt*  innerIndex;               /* indices to delta set     */

  } GX_DeltaSetIdxMapRec, *GX_DeltaSetIdxMap;


FT_END_HEADER

#endif /* FTMMTYPES_H_ */


/* END */
