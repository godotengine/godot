/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2007             *
 * by the Xiph.Org Foundation https://xiph.org/                     *
 *                                                                  *
 ********************************************************************

 function: simple utility that runs audio through the psychoacoustics
           without encoding

 ********************************************************************/

/* NB: this is dead code, retained purely for doc and reference value
       don't try to compile it */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vorbis/codec.h"
#include "codec_internal.h"
#include "os.h"
#include "misc.h"
#include "psy.h"
#include "mdct.h"
#include "smallft.h"
#include "window.h"
#include "scales.h"
#include "lpc.h"
#include "lsp.h"
#include "masking.h"
#include "registry.h"

static vorbis_info_psy_global _psy_set0G={
  0,   /* decaydBpms */
  8,   /* lines per eighth octave */

  /* thresh sample period, preecho clamp trigger threshhold, range, minenergy */
  256, {26.f,26.f,26.f,30.f}, {-90.f,-90.f,-90.f,-90.f}, -90.f,
  -6.f,

  0,

  0.,
  0.,
};

static vp_part _vp_part0[]={
  {    1,9e10f, 9e10f,       1.f,9999.f},
  { 9999,  .75f, 9e10f,       .5f,9999.f},
/*{ 9999, 1.5f, 9e10f,       .5f,9999.f},*/
  {   18,9e10f, 9e10f,       .5f,  30.f},
  { 9999,9e10f, 9e10f,       .5f,  30.f}
};

static vp_couple _vp_couple0[]={
  {    1,  {9e10f,9e10f,0}, {   0.f,   0.f,0}, {   0.f, 0.f,0}, {0.f,0.f,0}},
  {   18,  {9e10f,9e10f,0}, {   0.f,   0.f,0}, {   0.f, 0.f,0}, {0.f,0.f,0}},
  { 9999,  {9e10f,9e10f,0}, {   0.f, 9e10f,0}, {   0.f,22.f,1}, {0.f,0.f,0}}
};

static vorbis_info_psy _psy_set0={
  ATH_Bark_dB_lineaggressive,

  -100.f,
  -140.f,
  6.f, /* floor master att */

  /*     0  1  2   3   4   5   6   7   8   9  10  11  12  13  14  15   16   */
  /* x: 63 88 125 175 250 350 500 700 1k 1.4k 2k 2.8k 4k 5.6k 8k 11.5k 16k Hz */
  /* y: 0 10 20 30 40 50 60 70 80 90 100 dB */
   1,  /* tonemaskp */
  0.f, /* tone master att */
  /*  0   10   20   30   40   50   60   70   80   90   100 */
  {
   {-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f}, /*63*/
   {-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f}, /*88*/
   {-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f,-999.f}, /*125*/

   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*175*/
   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*250*/
   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*350*/
   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*500*/
   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*700*/
   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*1000*/
   {-30.f,-30.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*1400*/
   {-40.f,-40.f,-40.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*2000*/
   {-40.f,-40.f,-40.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*2800*/
   {-40.f,-40.f,-40.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*4000*/

   {-30.f,-35.f,-35.f,-40.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*5600*/

   {-30.f,-30.f,-33.f,-35.f,-40.f,-50.f,-60.f,-70.f,-80.f,-90.f,-100.f}, /*8000*/
   {-30.f,-30.f,-33.f,-35.f,-40.f,-45.f,-50.f,-60.f,-70.f,-85.f,-100.f}, /*11500*/
   {-24.f,-24.f,-26.f,-32.f,-32.f,-42.f,-50.f,-60.f,-70.f,-85.f,-100.f}, /*16000*/

  },

  1,/* peakattp */
  {{-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*63*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*88*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*125*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*175*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*250*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*350*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*500*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*700*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*1000*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*1400*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*2000*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*2800*/
   {-14.f,-20.f,-20.f,-20.f,-26.f,-32.f,-40.f,-40.f,-40.f,-40.f,-40.f},/*4000*/
   {-10.f,-12.f,-14.f,-16.f,-16.f,-20.f,-24.f,-30.f,-32.f,-40.f,-40.f},/*5600*/
   {-10.f,-12.f,-14.f,-16.f,-16.f,-20.f,-24.f,-30.f,-32.f,-40.f,-40.f},/*8000*/
   {-10.f,-10.f,-10.f,-12.f,-14.f,-18.f,-22.f,-28.f,-32.f,-40.f,-40.f},/*11500*/
   {-10.f,-10.f,-10.f,-12.f,-14.f,-18.f,-22.f,-28.f,-32.f,-40.f,-40.f},/*16000*/
  },

  1,/*noisemaskp */
  -10.f,  /* suppress any noise curve over maxspec+n */
  .5f,   /* low window */
  .5f,   /* high window */
  10,
  10,
  25,
  {.000f, 0.f, /*63*/
   .000f, 0.f, /*88*/
   .000f, 0.f, /*125*/
   .000f, 0.f, /*175*/
   .000f, 0.f, /*250*/
   .000f, 0.f, /*350*/
   .000f, 0.f, /*500*/
   .000f, 0.f, /*700*/
   .000f, 0.f, /*1000*/
   .300f, 0.f, /*1400*/
   .300f, 0.f, /*2000*/
   .300f, 0.f, /*2800*/
   .500f, 0.f, /*4000*/
   .700f, 0.f, /*5600*/
   .850f, 0.f, /*8000*/
   .900f, 0.f, /*11500*/
   .900f, 1.f, /*16000*/
  },

  95.f,  /* even decade + 5 is important; saves an rint() later in a
            tight loop) */
  -44.,

  32,
  _vp_part0,_vp_couple0
};

static vorbis_info_floor1 _floor_set0={1,
                                        {0},

                                        {32},
                                        {0},
                                        {0},
                                        {{-1}},

                                        2,
                                        {0,1024,

                                         88,31,243,

                                         14,54,143,460,

                                         6,3,10, 22,18,26, 41,36,47,
                                         69,61,78, 112,99,126, 185,162,211,
                                         329,282,387, 672,553,825
                                         },

                                        60,30,400,
                                        20,8,1,18.,
                                        20,600,
                                        960};


static vorbis_info_mapping0 mapping_info={1,{0,1},{0},{0},{0},0, 1, {0},{1}};
static codec_setup_info codec_setup0={ {0,0},
                                       1,1,1,1,1,0,1,
                                       {NULL},
                                       {0},{&mapping_info},
                                       {0},{NULL},
                                       {1},{&_floor_set0},
                                       {2},{NULL},
                                       {NULL},
                                       {&_psy_set0},
                                       &_psy_set0G};

static int noisy=0;
void analysis(char *base,int i,float *v,int n,int bark,int dB){
  if(noisy){
    int j;
    FILE *of;
    char buffer[80];
    sprintf(buffer,"%s_%d.m",base,i);
    of=fopen(buffer,"w");

    for(j=0;j<n;j++){
      if(dB && v[j]==0)
          fprintf(of,"\n\n");
      else{
        if(bark)
          fprintf(of,"%g ",toBARK(22050.f*j/n));
        else
          fprintf(of,"%g ",(float)j);

        if(dB){
          fprintf(of,"%g\n",todB(v+j));
        }else{
          fprintf(of,"%g\n",v[j]);
        }
      }
    }
    fclose(of);
  }
}

long frameno=0;

/****************************************************************/

int main(int argc,char *argv[]){
  int eos=0;
  float nonz=0.f;
  float acc=0.f;
  float tot=0.f;
  float ampmax=-9999,newmax;
  float local_ampmax[2];

  int framesize=2048;
  float ampmax_att_per_sec=-6.;

  float *pcm[2],*out[2],*window,*flr[2],*mask[2],*work[2];
  signed char *buffer,*buffer2;
  mdct_lookup m_look;
  drft_lookup f_look;
  vorbis_look_psy p_look;
  vorbis_look_psy_global *pg_look;
  vorbis_look_floor *floor_look;
  vorbis_info vi;
  long i,j,k;

  int ath=0;
  int decayp=0;

  argv++;
  while(*argv){
    if(*argv[0]=='-'){
      /* option */
      if(argv[0][1]=='v'){
        noisy=0;
      }
    }else
      if(*argv[0]=='+'){
        /* option */
        if(argv[0][1]=='v'){
          noisy=1;
        }
      }else
        framesize=atoi(argv[0]);
    argv++;
  }

  vi.channels=2;
  vi.codec_setup=&codec_setup0;

  pcm[0]=_ogg_malloc(framesize*sizeof(float));
  pcm[1]=_ogg_malloc(framesize*sizeof(float));
  out[0]=_ogg_calloc(framesize/2,sizeof(float));
  out[1]=_ogg_calloc(framesize/2,sizeof(float));
  work[0]=_ogg_calloc(framesize,sizeof(float));
  work[1]=_ogg_calloc(framesize,sizeof(float));
  flr[0]=_ogg_calloc(framesize/2,sizeof(float));
  flr[1]=_ogg_calloc(framesize/2,sizeof(float));
  buffer=_ogg_malloc(framesize*4);
  buffer2=buffer+framesize*2;
  window=_vorbis_window_create(0,framesize,framesize/2,framesize/2);
  mdct_init(&m_look,framesize);
  drft_init(&f_look,framesize);
  _vp_psy_init(&p_look,&_psy_set0,&_psy_set0G,framesize/2,44100);
  pg_look=_vp_global_look(&vi);
  floor_look=_floor_P[1]->look(NULL,NULL,&_floor_set0);

  /* we cheat on the WAV header; we just bypass 44 bytes and never
     verify that it matches 16bit/stereo/44.1kHz. */

  fread(buffer,1,44,stdin);
  fwrite(buffer,1,44,stdout);
  memset(buffer,0,framesize*2);

  analysis("window",0,window,framesize,0,0);

  fprintf(stderr,"Processing for frame size %d...\n",framesize);

  while(!eos){
    long bytes=fread(buffer2,1,framesize*2,stdin);
    if(bytes<framesize*2)
      memset(buffer2+bytes,0,framesize*2-bytes);

    if(bytes!=0){
      int nonzero[2];

      /* uninterleave samples */
      for(i=0;i<framesize;i++){
        pcm[0][i]=((buffer[i*4+1]<<8)|
                      (0x00ff&(int)buffer[i*4]))/32768.f;
        pcm[1][i]=((buffer[i*4+3]<<8)|
                   (0x00ff&(int)buffer[i*4+2]))/32768.f;
      }

      {
        float secs=framesize/44100.;

        ampmax+=secs*ampmax_att_per_sec;
        if(ampmax<-9999)ampmax=-9999;
      }

      for(i=0;i<2;i++){
        float scale=4.f/framesize;
        float *fft=work[i];
        float *mdct=pcm[i];
        float *logmdct=mdct+framesize/2;

        analysis("pre",frameno+i,pcm[i],framesize,0,0);

        /* fft and mdct transforms  */
        for(j=0;j<framesize;j++)
          fft[j]=pcm[i][j]*=window[j];

        drft_forward(&f_look,fft);

        local_ampmax[i]=-9999.f;
        fft[0]*=scale;
        fft[0]=todB(fft);
        for(j=1;j<framesize-1;j+=2){
          float temp=scale*FAST_HYPOT(fft[j],fft[j+1]);
          temp=fft[(j+1)>>1]=todB(&temp);
          if(temp>local_ampmax[i])local_ampmax[i]=temp;
        }
        if(local_ampmax[i]>ampmax)ampmax=local_ampmax[i];

        mdct_forward(&m_look,pcm[i],mdct);
        for(j=0;j<framesize/2;j++)
          logmdct[j]=todB(mdct+j);

        analysis("mdct",frameno+i,logmdct,framesize/2,1,0);
        analysis("fft",frameno+i,fft,framesize/2,1,0);
      }

      for(i=0;i<2;i++){
        float amp;
        float *fft=work[i];
        float *logmax=fft;
        float *mdct=pcm[i];
        float *logmdct=mdct+framesize/2;
        float *mask=fft+framesize/2;

        /* floor psychoacoustics */
        _vp_compute_mask(&p_look,
                         pg_look,
                         i,
                         fft,
                         logmdct,
                         mask,
                         ampmax,
                         local_ampmax[i],
                         framesize/2);

        analysis("mask",frameno+i,mask,framesize/2,1,0);

        {
          vorbis_block vb;
          vorbis_dsp_state vd;
          memset(&vd,0,sizeof(vd));
          vd.vi=&vi;
          vb.vd=&vd;
          vb.pcmend=framesize;

          /* floor quantization/application */
          nonzero[i]=_floor_P[1]->forward(&vb,floor_look,
                                          mdct,
                                          logmdct,
                                          mask,
                                          logmax,

                                          flr[i]);
        }

        _vp_remove_floor(&p_look,
                         pg_look,
                         logmdct,
                         mdct,
                         flr[i],
                         pcm[i],
                         local_ampmax[i]);

        for(j=0;j<framesize/2;j++)
          if(fabs(pcm[i][j])>1500)
            fprintf(stderr,"%ld ",frameno+i);

        analysis("res",frameno+i,pcm[i],framesize/2,1,0);
        analysis("codedflr",frameno+i,flr[i],framesize/2,1,1);
      }

      /* residue prequantization */
      _vp_partition_prequant(&p_look,
                             &vi,
                             pcm,
                             nonzero);

      for(i=0;i<2;i++)
        analysis("quant",frameno+i,pcm[i],framesize/2,1,0);

      /* channel coupling / stereo quantization */

      _vp_couple(&p_look,
                 &mapping_info,
                 pcm,
                 nonzero);

      for(i=0;i<2;i++)
        analysis("coupled",frameno+i,pcm[i],framesize/2,1,0);

      /* decoupling */
      for(i=mapping_info.coupling_steps-1;i>=0;i--){
        float *pcmM=pcm[mapping_info.coupling_mag[i]];
        float *pcmA=pcm[mapping_info.coupling_ang[i]];

        for(j=0;j<framesize/2;j++){
          float mag=pcmM[j];
          float ang=pcmA[j];

          if(mag>0)
            if(ang>0){
              pcmM[j]=mag;
              pcmA[j]=mag-ang;
            }else{
              pcmA[j]=mag;
              pcmM[j]=mag+ang;
            }
          else
            if(ang>0){
              pcmM[j]=mag;
              pcmA[j]=mag+ang;
            }else{
              pcmA[j]=mag;
              pcmM[j]=mag-ang;
            }
        }
      }

      for(i=0;i<2;i++)
        analysis("decoupled",frameno+i,pcm[i],framesize/2,1,0);

      for(i=0;i<2;i++){
        float amp;

        for(j=0;j<framesize/2;j++)
          pcm[i][j]*=flr[i][j];

        analysis("final",frameno+i,pcm[i],framesize/2,1,1);

        /* take it back to time */
        mdct_backward(&m_look,pcm[i],pcm[i]);

        for(j=0;j<framesize/2;j++)
          out[i][j]+=pcm[i][j]*window[j];

        analysis("out",frameno+i,out[i],framesize/2,0,0);


      }

      /* write data.  Use the part of buffer we're about to shift out */
      for(i=0;i<2;i++){
        char  *ptr=buffer+i*2;
        float *mono=out[i];
        int flag=0;
        for(j=0;j<framesize/2;j++){
          int val=mono[j]*32767.;
          /* might as well guard against clipping */
          if(val>32767){
            if(!flag)fprintf(stderr,"clipping in frame %ld ",frameno+i);
            flag=1;
            val=32767;
          }
          if(val<-32768){
            if(!flag)fprintf(stderr,"clipping in frame %ld ",frameno+i);
            flag=1;
            val=-32768;
          }
          ptr[0]=val&0xff;
          ptr[1]=(val>>8)&0xff;
          ptr+=4;
        }
      }

      fprintf(stderr,"*");
      fwrite(buffer,1,framesize*2,stdout);
      memmove(buffer,buffer2,framesize*2);

      for(i=0;i<2;i++){
        for(j=0,k=framesize/2;j<framesize/2;j++,k++)
          out[i][j]=pcm[i][k]*window[k];
      }
      frameno+=2;
    }else
      eos=1;
  }
  fprintf(stderr,"average raw bits of entropy: %.03g/sample\n",acc/tot);
  fprintf(stderr,"average nonzero samples: %.03g/%d\n",nonz/tot*framesize/2,
          framesize/2);
  fprintf(stderr,"Done\n\n");
  return 0;
}
