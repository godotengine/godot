/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ogg/ogg.h>
#include "quant.h"
#include "decint.h"

/*The maximum output of the DCT with +/- 255 inputs is +/- 8157.
  These minimum quantizers ensure the result after quantization (and after
   prediction for DC) will be no more than +/- 510.
  The tokenization system can handle values up to +/- 580, so there is no need
   to do any coefficient clamping.
  I would rather have allowed smaller quantizers and had to clamp, but these
   minimums were required when constructing the original VP3 matrices and have
   been formalized in the spec.*/
static const unsigned OC_DC_QUANT_MIN[2]={4<<2,8<<2};
static const unsigned OC_AC_QUANT_MIN[2]={2<<2,4<<2};

/*Initializes the dequantization tables from a set of quantizer info.
  Currently the dequantizer (and elsewhere enquantizer) tables are expected to
   be initialized as pointing to the storage reserved for them in the
   oc_theora_state (resp. oc_enc_ctx) structure.
  If some tables are duplicates of others, the pointers will be adjusted to
   point to a single copy of the tables, but the storage for them will not be
   freed.
  If you're concerned about the memory footprint, the obvious thing to do is
   to move the storage out of its fixed place in the structures and allocate
   it on demand.
  However, a much, much better option is to only store the quantization
   matrices being used for the current frame, and to recalculate these as the
   qi values change between frames (this is what VP3 did).*/
void oc_dequant_tables_init(ogg_uint16_t *_dequant[64][3][2],
 int _pp_dc_scale[64],const th_quant_info *_qinfo){
  /*Coding mode: intra or inter.*/
  int          qti;
  /*Y', C_b, C_r*/
  int          pli;
  for(qti=0;qti<2;qti++)for(pli=0;pli<3;pli++){
    /*Quality index.*/
    int qi;
    /*Range iterator.*/
    int qri;
    for(qi=0,qri=0;qri<=_qinfo->qi_ranges[qti][pli].nranges;qri++){
      th_quant_base base;
      ogg_uint32_t  q;
      int           qi_start;
      int           qi_end;
      memcpy(base,_qinfo->qi_ranges[qti][pli].base_matrices[qri],
       sizeof(base));
      qi_start=qi;
      if(qri==_qinfo->qi_ranges[qti][pli].nranges)qi_end=qi+1;
      else qi_end=qi+_qinfo->qi_ranges[qti][pli].sizes[qri];
      /*Iterate over quality indices in this range.*/
      for(;;){
        ogg_uint32_t qfac;
        int          zzi;
        int          ci;
        /*In the original VP3.2 code, the rounding offset and the size of the
           dead zone around 0 were controlled by a "sharpness" parameter.
          The size of our dead zone is now controlled by the per-coefficient
           quality thresholds returned by our HVS module.
          We round down from a more accurate value when the quality of the
           reconstruction does not fall below our threshold and it saves bits.
          Hence, all of that VP3.2 code is gone from here, and the remaining
           floating point code has been implemented as equivalent integer code
           with exact precision.*/
        qfac=(ogg_uint32_t)_qinfo->dc_scale[qi]*base[0];
        /*For postprocessing, not dequantization.*/
        if(_pp_dc_scale!=NULL)_pp_dc_scale[qi]=(int)(qfac/160);
        /*Scale DC the coefficient from the proper table.*/
        q=(qfac/100)<<2;
        q=OC_CLAMPI(OC_DC_QUANT_MIN[qti],q,OC_QUANT_MAX);
        _dequant[qi][pli][qti][0]=(ogg_uint16_t)q;
        /*Now scale AC coefficients from the proper table.*/
        for(zzi=1;zzi<64;zzi++){
          q=((ogg_uint32_t)_qinfo->ac_scale[qi]*base[OC_FZIG_ZAG[zzi]]/100)<<2;
          q=OC_CLAMPI(OC_AC_QUANT_MIN[qti],q,OC_QUANT_MAX);
          _dequant[qi][pli][qti][zzi]=(ogg_uint16_t)q;
        }
        /*If this is a duplicate of a previous matrix, use that instead.
          This simple check helps us improve cache coherency later.*/
        {
          int dupe;
          int qtj;
          int plj;
          dupe=0;
          for(qtj=0;qtj<=qti;qtj++){
            for(plj=0;plj<(qtj<qti?3:pli);plj++){
              if(!memcmp(_dequant[qi][pli][qti],_dequant[qi][plj][qtj],
               sizeof(oc_quant_table))){
                dupe=1;
                break;
              }
            }
            if(dupe)break;
          }
          if(dupe)_dequant[qi][pli][qti]=_dequant[qi][plj][qtj];
        }
        if(++qi>=qi_end)break;
        /*Interpolate the next base matrix.*/
        for(ci=0;ci<64;ci++){
          base[ci]=(unsigned char)(
           (2*((qi_end-qi)*_qinfo->qi_ranges[qti][pli].base_matrices[qri][ci]+
           (qi-qi_start)*_qinfo->qi_ranges[qti][pli].base_matrices[qri+1][ci])
           +_qinfo->qi_ranges[qti][pli].sizes[qri])/
           (2*_qinfo->qi_ranges[qti][pli].sizes[qri]));
        }
      }
    }
  }
}
