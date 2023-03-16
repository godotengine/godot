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
  last mod: $Id$

 ********************************************************************/
#include <stdlib.h>
#include <string.h>
#include "encint.h"



int oc_quant_params_clone(th_quant_info *_dst,const th_quant_info *_src){
  int i;
  memcpy(_dst,_src,sizeof(*_dst));
  memset(_dst->qi_ranges,0,sizeof(_dst->qi_ranges));
  for(i=0;i<6;i++){
    int nranges;
    int qti;
    int pli;
    int qtj;
    int plj;
    int pdup;
    int qdup;
    qti=i/3;
    pli=i%3;
    qtj=(i-1)/3;
    plj=(i-1)%3;
    nranges=_src->qi_ranges[qti][pli].nranges;
    /*Check for those duplicates that can be cleanly handled by
       oc_quant_params_clear().*/
    pdup=i>0&&nranges<=_src->qi_ranges[qtj][plj].nranges;
    qdup=qti>0&&nranges<=_src->qi_ranges[0][pli].nranges;
    _dst->qi_ranges[qti][pli].nranges=nranges;
    if(pdup&&_src->qi_ranges[qti][pli].sizes==_src->qi_ranges[qtj][plj].sizes){
      _dst->qi_ranges[qti][pli].sizes=_dst->qi_ranges[qtj][plj].sizes;
    }
    else if(qdup&&_src->qi_ranges[1][pli].sizes==_src->qi_ranges[0][pli].sizes){
      _dst->qi_ranges[1][pli].sizes=_dst->qi_ranges[0][pli].sizes;
    }
    else{
      int *sizes;
      sizes=(int *)_ogg_malloc(nranges*sizeof(*sizes));
      /*Note: The caller is responsible for cleaning up any partially
         constructed qinfo.*/
      if(sizes==NULL)return TH_EFAULT;
      memcpy(sizes,_src->qi_ranges[qti][pli].sizes,nranges*sizeof(*sizes));
      _dst->qi_ranges[qti][pli].sizes=sizes;
    }
    if(pdup&&_src->qi_ranges[qti][pli].base_matrices==
     _src->qi_ranges[qtj][plj].base_matrices){
      _dst->qi_ranges[qti][pli].base_matrices=
       _dst->qi_ranges[qtj][plj].base_matrices;
    }
    else if(qdup&&_src->qi_ranges[1][pli].base_matrices==
     _src->qi_ranges[0][pli].base_matrices){
      _dst->qi_ranges[1][pli].base_matrices=
       _dst->qi_ranges[0][pli].base_matrices;
    }
    else{
      th_quant_base *base_matrices;
      base_matrices=(th_quant_base *)_ogg_malloc(
       (nranges+1)*sizeof(*base_matrices));
      /*Note: The caller is responsible for cleaning up any partially
         constructed qinfo.*/
      if(base_matrices==NULL)return TH_EFAULT;
      memcpy(base_matrices,_src->qi_ranges[qti][pli].base_matrices,
       (nranges+1)*sizeof(*base_matrices));
      _dst->qi_ranges[qti][pli].base_matrices=
       (const th_quant_base *)base_matrices;
    }
  }
  return 0;
}

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

void oc_iquant_init(oc_iquant *_this,ogg_uint16_t _d){
  ogg_uint32_t t;
  int          l;
  _d<<=1;
  l=OC_ILOGNZ_32(_d)-1;
  t=1+((ogg_uint32_t)1<<16+l)/_d;
  _this->m=(ogg_int16_t)(t-0x10000);
  _this->l=l;
}

void oc_enc_enquant_table_init_c(void *_enquant,
 const ogg_uint16_t _dequant[64]){
  oc_iquant *enquant;
  int        zzi;
  /*In the original VP3.2 code, the rounding offset and the size of the
     dead zone around 0 were controlled by a "sharpness" parameter.
    We now R-D optimize the tokens for each block after quantization,
     so the rounding offset should always be 1/2, and an explicit dead
     zone is unnecessary.
    Hence, all of that VP3.2 code is gone from here, and the remaining
     floating point code has been implemented as equivalent integer
     code with exact precision.*/
  enquant=(oc_iquant *)_enquant;
  for(zzi=0;zzi<64;zzi++)oc_iquant_init(enquant+zzi,_dequant[zzi]);
}

void oc_enc_enquant_table_fixup_c(void *_enquant[3][3][2],int _nqis){
  int pli;
  int qii;
  int qti;
  for(pli=0;pli<3;pli++)for(qii=1;qii<_nqis;qii++)for(qti=0;qti<2;qti++){
    *((oc_iquant *)_enquant[pli][qii][qti])=
     *((oc_iquant *)_enquant[pli][0][qti]);
  }
}

int oc_enc_quantize_c(ogg_int16_t _qdct[64],const ogg_int16_t _dct[64],
 const ogg_uint16_t _dequant[64],const void *_enquant){
  const oc_iquant *enquant;
  int              nonzero;
  int              zzi;
  int              val;
  int              d;
  int              s;
  enquant=(const oc_iquant *)_enquant;
  nonzero=0;
  for(zzi=0;zzi<64;zzi++){
    val=_dct[zzi];
    d=_dequant[zzi];
    val=val<<1;
    if(abs(val)>=d){
      s=OC_SIGNMASK(val);
      /*The bias added here rounds ties away from zero, since token
         optimization can only decrease the magnitude of the quantized
         value.*/
      val+=d+s^s;
      /*Note the arithmetic right shift is not guaranteed by ANSI C.
        Hopefully no one still uses ones-complement architectures.*/
      val=((enquant[zzi].m*(ogg_int32_t)val>>16)+val>>enquant[zzi].l)-s;
      _qdct[zzi]=(ogg_int16_t)val;
      nonzero=zzi;
    }
    else _qdct[zzi]=0;
  }
  return nonzero;
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
  TODO: These values are only from INTER frames; they should be re-measured for
   INTRA frames.*/
static const ogg_uint16_t OC_PCD[4][3]={
  {59926, 3038, 2572},
  {55201, 5597, 4738},
  {55201, 5597, 4738},
  {47682, 9669, 8185}
};


/*Compute "average" quantizers for each qi level to use for rate control.
  We do one for each color channel, as well as an average across color
   channels, separately for INTER and INTRA, since their behavior is very
   different.
  The basic approach is to compute a harmonic average of the squared quantizer,
   weighted by the expected squared magnitude of the DCT coefficients.
  Under the (not quite true) assumption that DCT coefficients are
   Laplacian-distributed, this preserves the product Q*lambda, where
   lambda=sqrt(2/sigma**2) is the Laplacian distribution parameter (not to be
   confused with the lambda used in R-D optimization throughout most of the
   rest of the code), when the distributions from multiple coefficients are
   pooled.
  The value Q*lambda completely determines the entropy of coefficients drawn
   from a Laplacian distribution, and thus the expected bitrate.*/
void oc_enquant_qavg_init(ogg_int64_t _log_qavg[2][64],
 ogg_int16_t _log_plq[64][3][2],ogg_uint16_t _chroma_rd_scale[2][64][2],
 ogg_uint16_t *_dequant[64][3][2],int _pixel_fmt){
  int qi;
  int pli;
  int qti;
  int ci;
  for(qti=0;qti<2;qti++)for(qi=0;qi<64;qi++){
    ogg_int64_t  q2;
    ogg_uint32_t qp[3];
    ogg_uint32_t cqp;
    ogg_uint32_t d;
    q2=0;
    for(pli=0;pli<3;pli++){
      qp[pli]=0;
      for(ci=0;ci<64;ci++){
        unsigned rq;
        unsigned qd;
        qd=_dequant[qi][pli][qti][OC_IZIG_ZAG[ci]];
        rq=(OC_RPSD[qti][ci]+(qd>>1))/qd;
        qp[pli]+=rq*(ogg_uint32_t)rq;
      }
      q2+=OC_PCD[_pixel_fmt][pli]*(ogg_int64_t)qp[pli];
      /*plq=1.0/sqrt(qp)*/
      _log_plq[qi][pli][qti]=
       (ogg_int16_t)(OC_Q10(32)-oc_blog32_q10(qp[pli])>>1);
    }
    d=OC_PCD[_pixel_fmt][1]+OC_PCD[_pixel_fmt][2];
    cqp=(ogg_uint32_t)((OC_PCD[_pixel_fmt][1]*(ogg_int64_t)qp[1]+
     OC_PCD[_pixel_fmt][2]*(ogg_int64_t)qp[2]+(d>>1))/d);
    /*chroma_rd_scale=clamp(0.25,cqp/qp[0],4)*/
    d=OC_MAXI(qp[0]+(1<<OC_RD_SCALE_BITS-1)>>OC_RD_SCALE_BITS,1);
    d=OC_CLAMPI(1<<OC_RD_SCALE_BITS-2,(cqp+(d>>1))/d,4<<OC_RD_SCALE_BITS);
    _chroma_rd_scale[qti][qi][0]=(ogg_int16_t)d;
    /*chroma_rd_iscale=clamp(0.25,qp[0]/cqp,4)*/
    d=OC_MAXI(OC_RD_ISCALE(cqp,1),1);
    d=OC_CLAMPI(1<<OC_RD_ISCALE_BITS-2,(qp[0]+(d>>1))/d,4<<OC_RD_ISCALE_BITS);
    _chroma_rd_scale[qti][qi][1]=(ogg_int16_t)d;
    /*qavg=1.0/sqrt(q2).*/
    _log_qavg[qti][qi]=OC_Q57(48)-oc_blog64(q2)>>1;
  }
}
