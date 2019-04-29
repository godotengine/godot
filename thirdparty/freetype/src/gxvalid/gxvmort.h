/****************************************************************************
 *
 * gxvmort.h
 *
 *   TrueTypeGX/AAT common definition for mort table (specification).
 *
 * Copyright (C) 2004-2019 by
 * suzuki toshiya, Masatake YAMATO, Red Hat K.K.,
 * David Turner, Robert Wilhelm, and Werner Lemberg.
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
 * gxvalid is derived from both gxlayout module and otvalid module.
 * Development of gxlayout is supported by the Information-technology
 * Promotion Agency(IPA), Japan.
 *
 */


#ifndef GXVMORT_H_
#define GXVMORT_H_

#include "gxvalid.h"
#include "gxvcommn.h"

#include FT_SFNT_NAMES_H


  typedef struct  GXV_mort_featureRec_
  {
    FT_UShort  featureType;
    FT_UShort  featureSetting;
    FT_ULong   enableFlags;
    FT_ULong   disableFlags;

  } GXV_mort_featureRec, *GXV_mort_feature;

#define GXV_MORT_FEATURE_OFF  {0, 1, 0x00000000UL, 0x00000000UL}

#define IS_GXV_MORT_FEATURE_OFF( f )              \
          ( (f).featureType    == 0            || \
            (f).featureSetting == 1            || \
            (f).enableFlags    == 0x00000000UL || \
            (f).disableFlags   == 0x00000000UL )


  FT_LOCAL( void )
  gxv_mort_featurearray_validate( FT_Bytes       table,
                                  FT_Bytes       limit,
                                  FT_ULong       nFeatureFlags,
                                  GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_coverage_validate( FT_UShort      coverage,
                              GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type0_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type1_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type2_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type4_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type5_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );


#endif /* GXVMORT_H_ */


/* END */
