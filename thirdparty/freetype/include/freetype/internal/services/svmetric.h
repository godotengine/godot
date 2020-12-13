/****************************************************************************
 *
 * svmetric.h
 *
 *   The FreeType services for metrics variations (specification).
 *
 * Copyright (C) 2016-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVMETRIC_H_
#define SVMETRIC_H_

#include <freetype/internal/ftserv.h>


FT_BEGIN_HEADER


  /*
   * A service to manage the `HVAR, `MVAR', and `VVAR' OpenType tables.
   *
   */

#define FT_SERVICE_ID_METRICS_VARIATIONS  "metrics-variations"


  /* HVAR */

  typedef FT_Error
  (*FT_HAdvance_Adjust_Func)( FT_Face  face,
                              FT_UInt  gindex,
                              FT_Int  *avalue );

  typedef FT_Error
  (*FT_LSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  typedef FT_Error
  (*FT_RSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  /* VVAR */

  typedef FT_Error
  (*FT_VAdvance_Adjust_Func)( FT_Face  face,
                              FT_UInt  gindex,
                              FT_Int  *avalue );

  typedef FT_Error
  (*FT_TSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  typedef FT_Error
  (*FT_BSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  typedef FT_Error
  (*FT_VOrg_Adjust_Func)( FT_Face  face,
                          FT_UInt  gindex,
                          FT_Int  *avalue );

  /* MVAR */

  typedef void
  (*FT_Metrics_Adjust_Func)( FT_Face  face );


  FT_DEFINE_SERVICE( MetricsVariations )
  {
    FT_HAdvance_Adjust_Func  hadvance_adjust;
    FT_LSB_Adjust_Func       lsb_adjust;
    FT_RSB_Adjust_Func       rsb_adjust;

    FT_VAdvance_Adjust_Func  vadvance_adjust;
    FT_TSB_Adjust_Func       tsb_adjust;
    FT_BSB_Adjust_Func       bsb_adjust;
    FT_VOrg_Adjust_Func      vorg_adjust;

    FT_Metrics_Adjust_Func   metrics_adjust;
  };


#define FT_DEFINE_SERVICE_METRICSVARIATIONSREC( class_,            \
                                                hadvance_adjust_,  \
                                                lsb_adjust_,       \
                                                rsb_adjust_,       \
                                                vadvance_adjust_,  \
                                                tsb_adjust_,       \
                                                bsb_adjust_,       \
                                                vorg_adjust_,      \
                                                metrics_adjust_  ) \
  static const FT_Service_MetricsVariationsRec  class_ =           \
  {                                                                \
    hadvance_adjust_,                                              \
    lsb_adjust_,                                                   \
    rsb_adjust_,                                                   \
    vadvance_adjust_,                                              \
    tsb_adjust_,                                                   \
    bsb_adjust_,                                                   \
    vorg_adjust_,                                                  \
    metrics_adjust_                                                \
  };

  /* */


FT_END_HEADER

#endif /* SVMETRIC_H_ */


/* END */
