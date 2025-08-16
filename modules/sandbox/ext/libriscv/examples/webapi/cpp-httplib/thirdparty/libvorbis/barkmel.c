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

 function: bark scale utility

 ********************************************************************/

#include <stdio.h>
#include "scales.h"
int main(){
  int i;
  double rate;
  for(i=64;i<32000;i*=2){
    rate=48000.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));

    rate=44100.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));

    rate=32000.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));

    rate=22050.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));

    rate=16000.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));

    rate=11025.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));

    rate=8000.f;
    fprintf(stderr,"rate=%gHz, block=%d, f(1)=%.2gHz bark(1)=%.2g (of %.2g)\n\n",
            rate,i,rate/2 / (i/2),toBARK(rate/2 /(i/2)),toBARK(rate/2));


  }
  {
    float i;
    int j;
    for(i=0.,j=0;i<28;i+=1,j++){
      fprintf(stderr,"(%d) bark=%f %gHz (%d of 128)\n",
              j,i,fromBARK(i),(int)(fromBARK(i)/22050.*128.));
    }
  }
  return(0);
}

