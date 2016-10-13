/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function:
  last mod: $Id: enquant.c 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/
#include <stdlib.h>
#include <string.h>
#include "encint.h"



void oc_quant_params_pack(oggpack_buffer *_opb,const th_quant_info *_qinfo){
  const th_quant_ranges *qranges;
  const th_quant_base   *base_mats[2*3*64];
  int                    indices[2][3][64];
  int                    nbase_mats;
  int                    nbits;
  int                    ci;
  int                    qi;
  int                    qri;
  int                    qti;
  int                    pli;
  int                    qtj;
  int                    plj;
  int                    bmi;
  int                    i;
  i=_qinfo->loop_filter_limits[0];
  for(qi=1;qi<64;qi++)i=OC_MAXI(i,_qinfo->loop_filter_limits[qi]);
  nbits=OC_ILOG_32(i);
  oggpackB_write(_opb,nbits,3);
  for(qi=0;qi<64;qi++){
    oggpackB_write(_opb,_qinfo->loop_filter_limits[qi],nbits);
  }
  /*580 bits for VP3.*/
  i=1;
  for(qi=0;qi<64;qi++)i=OC_MAXI(_qinfo->ac_scale[qi],i);
  nbits=OC_ILOGNZ_32(i);
  oggpackB_write(_opb,nbits-1,4);
  for(qi=0;qi<64;qi++)oggpackB_write(_opb,_qinfo->ac_scale[qi],nbits);
  /*516 bits for VP3.*/
  i=1;
  for(qi=0;qi<64;qi++)i=OC_MAXI(_qinfo->dc_scale[qi],i);
  nbits=OC_ILOGNZ_32(i);
  oggpackB_write(_opb,nbits-1,4);
  for(qi=0;qi<64;qi++)oggpackB_write(_opb,_qinfo->dc_scale[qi],nbits);
  /*Consolidate any duplicate base matrices.*/
  nbase_mats=0;
  for(qti=0;qti<2;qti++)for(pli=0;pli<3;pli++){
    qranges=_qinfo->qi_ranges[qti]+pli;
    for(qri=0;qri<=qranges->nranges;qri++){
      for(bmi=0;;bmi++){
        if(bmi>=nbase_mats){
          base_mats[bmi]=qranges->base_matrices+qri;
          indices[qti][pli][qri]=nbase_mats++;
          break;
        }
        else if(memcmp(base_mats[bmi][0],qranges->base_matrices[qri],
         sizeof(base_mats[bmi][0]))==0){
          indices[qti][pli][qri]=bmi;
          break;
        }
      }
    }
  }
  /*Write out the list of unique base matrices.
    1545 bits for VP3 matrices.*/
  oggpackB_write(_opb,nbase_mats-1,9);
  for(bmi=0;bmi<nbase_mats;bmi++){
    for(ci=0;ci<64;ci++)oggpackB_write(_opb,base_mats[bmi][0][ci],8);
  }
  /*Now store quant ranges and their associated indices into the base matrix
     list.
    46 bits for VP3 matrices.*/
  nbits=OC_ILOG_32(nbase_mats-1);
  for(i=0;i<6;i++){
    qti=i/3;
    pli=i%3;
    qranges=_qinfo->qi_ranges[qti]+pli;
    if(i>0){
      if(qti>0){
        if(qranges->nranges==_qinfo->qi_ranges[qti-1][pli].nranges&&
         memcmp(qranges->sizes,_qinfo->qi_ranges[qti-1][pli].sizes,
         qranges->nranges*sizeof(qranges->sizes[0]))==0&&
         memcmp(indices[qti][pli],indices[qti-1][pli],
         (qranges->nranges+1)*sizeof(indices[qti][pli][0]))==0){
          oggpackB_write(_opb,1,2);
          continue;
        }
      }
      qtj=(i-1)/3;
      plj=(i-1)%3;
      if(qranges->nranges==_qinfo->qi_ranges[qtj][plj].nranges&&
       memcmp(qranges->sizes,_qinfo->qi_ranges[qtj][plj].sizes,
       qranges->nranges*sizeof(qranges->sizes[0]))==0&&
       memcmp(indices[qti][pli],indices[qtj][plj],
       (qranges->nranges+1)*sizeof(indices[qti][pli][0]))==0){
        oggpackB_write(_opb,0,1+(qti>0));
        continue;
      }
      oggpackB_write(_opb,1,1);
    }
    oggpackB_write(_opb,indices[qti][pli][0],nbits);
    for(qi=qri=0;qi<63;qri++){
      oggpackB_write(_opb,qranges->sizes[qri]-1,OC_ILOG_32(62-qi));
      qi+=qranges->sizes[qri];
      oggpackB_write(_opb,indices[qti][pli][qri+1],nbits);
    }
  }
}

static void oc_iquant_init(oc_iquant *_this,ogg_uint16_t _d){
  ogg_uint32_t t;
  int          l;
  _d<<=1;
  l=OC_ILOGNZ_32(_d)-1;
  t=1+((ogg_uint32_t)1<<16+l)/_d;
  _this->m=(ogg_int16_t)(t-0x10000);
  _this->l=l;
}

/*See comments at oc_dequant_tables_init() for how the quantization tables'
   storage should be initialized.*/
void oc_enquant_tables_init(ogg_uint16_t *_dequant[64][3][2],
 oc_iquant *_enquant[64][3][2],const th_quant_info *_qinfo){
  int qi;
  int pli;
  int qti;
  /*Initialize the dequantization tables first.*/
  oc_dequant_tables_init(_dequant,NULL,_qinfo);
  /*Derive the quantization tables directly from the dequantization tables.*/
  for(qi=0;qi<64;qi++)for(qti=0;qti<2;qti++)for(pli=0;pli<3;pli++){
    int zzi;
    int plj;
    int qtj;
    int dupe;
    dupe=0;
    for(qtj=0;qtj<=qti;qtj++){
      for(plj=0;plj<(qtj<qti?3:pli);plj++){
        if(_dequant[qi][pli][qti]==_dequant[qi][plj][qtj]){
          dupe=1;
          break;
        }
      }
      if(dupe)break;
    }
    if(dupe){
      _enquant[qi][pli][qti]=_enquant[qi][plj][qtj];
      continue;
    }
    /*In the original VP3.2 code, the rounding offset and the size of the
       dead zone around 0 were controlled by a "sharpness" parameter.
      We now R-D optimize the tokens for each block after quantization,
       so the rounding offset should always be 1/2, and an explicit dead
       zone is unnecessary.
      Hence, all of that VP3.2 code is gone from here, and the remaining
       floating point code has been implemented as equivalent integer
       code with exact precision.*/
    for(zzi=0;zzi<64;zzi++){
      oc_iquant_init(_enquant[qi][pli][qti]+zzi,
       _dequant[qi][pli][qti][zzi]);
    }
  }
}



/*This table gives the square root of the fraction of the squared magnitude of
   each DCT coefficient relative to the total, scaled by 2**16, for both INTRA
   and INTER modes.
  These values were measured after motion-compensated prediction, before
   quantization, over a large set of test video (from QCIF to 1080p) encoded at
   all possible rates.
  The DC coefficient takes into account the DPCM prediction (using the
   quantized values from neighboring blocks, as the encoder does, but still
   before quantization of the coefficient in the current block).
  The results differ significantly from the expected variance (e.g., using an
   AR(1) model of the signal with rho=0.95, as is frequently done to compute
   the coding gain of the DCT).
  We use them to estimate an "average" quantizer for a given quantizer matrix,
   as this is used to parameterize a number of the rate control decisions.
  These values are themselves probably quantizer-matrix dependent, since the
   shape of the matrix affects the noise distribution in the reference frames,
   but they should at least give us _some_ amount of adaptivity to different
   matrices, as opposed to hard-coding a table of average Q values for the
   current set.
  The main features they capture are that a) only a few of the quantizers in
   the upper-left corner contribute anything significant at all (though INTER
   mode is significantly flatter) and b) the DPCM prediction of the DC
   coefficient gives a very minor improvement in the INTRA case and a quite
   significant one in the INTER case (over the expected variance).*/
static const ogg_uint16_t OC_RPSD[2][64]={
  {
    52725,17370,10399, 6867, 5115, 3798, 2942, 2076,
    17370, 9900, 6948, 4994, 3836, 2869, 2229, 1619,
    10399, 6948, 5516, 4202, 3376, 2573, 2015, 1461,
     6867, 4994, 4202, 3377, 2800, 2164, 1718, 1243,
     5115, 3836, 3376, 2800, 2391, 1884, 1530, 1091,
     3798, 2869, 2573, 2164, 1884, 1495, 1212,  873,
     2942, 2229, 2015, 1718, 1530, 1212, 1001,  704,
     2076, 1619, 1461, 1243, 1091,  873,  704,  474
  },
  {
    23411,15604,13529,11601,10683, 8958, 7840, 6142,
    15604,11901,10718, 9108, 8290, 6961, 6023, 4487,
    13529,10718, 9961, 8527, 7945, 6689, 5742, 4333,
    11601, 9108, 8527, 7414, 7084, 5923, 5175, 3743,
    10683, 8290, 7945, 7084, 6771, 5754, 4793, 3504,
     8958, 6961, 6689, 5923, 5754, 4679, 3936, 2989,
     7840, 6023, 5742, 5175, 4793, 3936, 3522, 2558,
     6142, 4487, 4333, 3743, 3504, 2989, 2558, 1829
  }
};

/*The fraction of the squared magnitude of the residuals in each color channel
   relative to the total, scaled by 2**16, for each pixel format.
  These values were measured after motion-compensated prediction, before
   quantization, over a large set of test video encoded at all possible rates.
  TODO: These values are only from INTER frames; it should be re-measured for
   INTRA frames.*/
static const ogg_uint16_t OC_PCD[4][3]={
  {59926, 3038, 2572},
  {55201, 5597, 4738},
  {55201, 5597, 4738},
  {47682, 9669, 8185}
};


/*Compute an "average" quantizer for each qi level.
  We do one for INTER and one for INTRA, since their behavior is very
   different, but average across chroma channels.
  The basic approach is to compute a harmonic average of the squared quantizer,
   weighted by the expected squared magnitude of the DCT coefficients.
  Under the (not quite true) assumption that DCT coefficients are
   Laplacian-distributed, this preserves the product Q*lambda, where
   lambda=sqrt(2/sigma**2) is the Laplacian distribution parameter (not to be
   confused with the lambda used in R-D optimization throughout most of the
   rest of the code).
  The value Q*lambda completely determines the entropy of the coefficients.*/
void oc_enquant_qavg_init(ogg_int64_t _log_qavg[2][64],
 ogg_uint16_t *_dequant[64][3][2],int _pixel_fmt){
  int qi;
  int pli;
  int qti;
  int ci;
  for(qti=0;qti<2;qti++)for(qi=0;qi<64;qi++){
    ogg_int64_t q2;
    q2=0;
    for(pli=0;pli<3;pli++){
      ogg_uint32_t qp;
      qp=0;
      for(ci=0;ci<64;ci++){
        unsigned rq;
        unsigned qd;
        qd=_dequant[qi][pli][qti][OC_IZIG_ZAG[ci]];
        rq=(OC_RPSD[qti][ci]+(qd>>1))/qd;
        qp+=rq*(ogg_uint32_t)rq;
      }
      q2+=OC_PCD[_pixel_fmt][pli]*(ogg_int64_t)qp;
    }
    /*qavg=1.0/sqrt(q2).*/
    _log_qavg[qti][qi]=OC_Q57(48)-oc_blog64(q2)>>1;
  }
}
