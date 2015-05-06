/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2009             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: floor backend 0 implementation
 last mod: $Id: floor0.c 17558 2010-10-22 00:24:41Z tterribe $

 ********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ogg/ogg.h>
#include "vorbis/codec.h"
#include "codec_internal.h"
#include "registry.h"
#include "lpc.h"
#include "lsp.h"
#include "codebook.h"
#include "scales.h"
#include "misc.h"
#include "os.h"

#include "misc.h"
#include <stdio.h>

typedef struct {
  int ln;
  int  m;
  int **linearmap;
  int  n[2];

  vorbis_info_floor0 *vi;

  long bits;
  long frames;
} vorbis_look_floor0;


/***********************************************/

static void floor0_free_info(vorbis_info_floor *i){
  vorbis_info_floor0 *info=(vorbis_info_floor0 *)i;
  if(info){
    memset(info,0,sizeof(*info));
    _ogg_free(info);
  }
}

static void floor0_free_look(vorbis_look_floor *i){
  vorbis_look_floor0 *look=(vorbis_look_floor0 *)i;
  if(look){

    if(look->linearmap){

      if(look->linearmap[0])_ogg_free(look->linearmap[0]);
      if(look->linearmap[1])_ogg_free(look->linearmap[1]);

      _ogg_free(look->linearmap);
    }
    memset(look,0,sizeof(*look));
    _ogg_free(look);
  }
}

static vorbis_info_floor *floor0_unpack (vorbis_info *vi,oggpack_buffer *opb){
  codec_setup_info     *ci=vi->codec_setup;
  int j;

  vorbis_info_floor0 *info=_ogg_malloc(sizeof(*info));
  info->order=oggpack_read(opb,8);
  info->rate=oggpack_read(opb,16);
  info->barkmap=oggpack_read(opb,16);
  info->ampbits=oggpack_read(opb,6);
  info->ampdB=oggpack_read(opb,8);
  info->numbooks=oggpack_read(opb,4)+1;

  if(info->order<1)goto err_out;
  if(info->rate<1)goto err_out;
  if(info->barkmap<1)goto err_out;
  if(info->numbooks<1)goto err_out;

  for(j=0;j<info->numbooks;j++){
    info->books[j]=oggpack_read(opb,8);
    if(info->books[j]<0 || info->books[j]>=ci->books)goto err_out;
    if(ci->book_param[info->books[j]]->maptype==0)goto err_out;
    if(ci->book_param[info->books[j]]->dim<1)goto err_out;
  }
  return(info);

 err_out:
  floor0_free_info(info);
  return(NULL);
}

/* initialize Bark scale and normalization lookups.  We could do this
   with static tables, but Vorbis allows a number of possible
   combinations, so it's best to do it computationally.

   The below is authoritative in terms of defining scale mapping.
   Note that the scale depends on the sampling rate as well as the
   linear block and mapping sizes */

static void floor0_map_lazy_init(vorbis_block      *vb,
                                 vorbis_info_floor *infoX,
                                 vorbis_look_floor0 *look){
  if(!look->linearmap[vb->W]){
    vorbis_dsp_state   *vd=vb->vd;
    vorbis_info        *vi=vd->vi;
    codec_setup_info   *ci=vi->codec_setup;
    vorbis_info_floor0 *info=(vorbis_info_floor0 *)infoX;
    int W=vb->W;
    int n=ci->blocksizes[W]/2,j;

    /* we choose a scaling constant so that:
       floor(bark(rate/2-1)*C)=mapped-1
     floor(bark(rate/2)*C)=mapped */
    float scale=look->ln/toBARK(info->rate/2.f);

    /* the mapping from a linear scale to a smaller bark scale is
       straightforward.  We do *not* make sure that the linear mapping
       does not skip bark-scale bins; the decoder simply skips them and
       the encoder may do what it wishes in filling them.  They're
       necessary in some mapping combinations to keep the scale spacing
       accurate */
    look->linearmap[W]=_ogg_malloc((n+1)*sizeof(**look->linearmap));
    for(j=0;j<n;j++){
      int val=floor( toBARK((info->rate/2.f)/n*j)
                     *scale); /* bark numbers represent band edges */
      if(val>=look->ln)val=look->ln-1; /* guard against the approximation */
      look->linearmap[W][j]=val;
    }
    look->linearmap[W][j]=-1;
    look->n[W]=n;
  }
}

static vorbis_look_floor *floor0_look(vorbis_dsp_state *vd,
                                      vorbis_info_floor *i){
  vorbis_info_floor0 *info=(vorbis_info_floor0 *)i;
  vorbis_look_floor0 *look=_ogg_calloc(1,sizeof(*look));
  look->m=info->order;
  look->ln=info->barkmap;
  look->vi=info;

  look->linearmap=_ogg_calloc(2,sizeof(*look->linearmap));

  return look;
}

static void *floor0_inverse1(vorbis_block *vb,vorbis_look_floor *i){
  vorbis_look_floor0 *look=(vorbis_look_floor0 *)i;
  vorbis_info_floor0 *info=look->vi;
  int j,k;

  int ampraw=oggpack_read(&vb->opb,info->ampbits);
  if(ampraw>0){ /* also handles the -1 out of data case */
    long maxval=(1<<info->ampbits)-1;
    float amp=(float)ampraw/maxval*info->ampdB;
    int booknum=oggpack_read(&vb->opb,_ilog(info->numbooks));

    if(booknum!=-1 && booknum<info->numbooks){ /* be paranoid */
      codec_setup_info  *ci=vb->vd->vi->codec_setup;
      codebook *b=ci->fullbooks+info->books[booknum];
      float last=0.f;

      /* the additional b->dim is a guard against any possible stack
         smash; b->dim is provably more than we can overflow the
         vector */
      float *lsp=_vorbis_block_alloc(vb,sizeof(*lsp)*(look->m+b->dim+1));

      for(j=0;j<look->m;j+=b->dim)
        if(vorbis_book_decodev_set(b,lsp+j,&vb->opb,b->dim)==-1)goto eop;
      for(j=0;j<look->m;){
        for(k=0;k<b->dim;k++,j++)lsp[j]+=last;
        last=lsp[j-1];
      }

      lsp[look->m]=amp;
      return(lsp);
    }
  }
 eop:
  return(NULL);
}

static int floor0_inverse2(vorbis_block *vb,vorbis_look_floor *i,
                           void *memo,float *out){
  vorbis_look_floor0 *look=(vorbis_look_floor0 *)i;
  vorbis_info_floor0 *info=look->vi;

  floor0_map_lazy_init(vb,info,look);

  if(memo){
    float *lsp=(float *)memo;
    float amp=lsp[look->m];

    /* take the coefficients back to a spectral envelope curve */
    vorbis_lsp_to_curve(out,
                        look->linearmap[vb->W],
                        look->n[vb->W],
                        look->ln,
                        lsp,look->m,amp,(float)info->ampdB);
    return(1);
  }
  memset(out,0,sizeof(*out)*look->n[vb->W]);
  return(0);
}

/* export hooks */
const vorbis_func_floor floor0_exportbundle={
  NULL,&floor0_unpack,&floor0_look,&floor0_free_info,
  &floor0_free_look,&floor0_inverse1,&floor0_inverse2
};
