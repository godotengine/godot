/* Copyright (c) 2008-2011 Octasic Inc.
   Written by Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "opus_types.h"
#include "opus_defines.h"

#include <math.h>
#include "mlp.h"
#include "arch.h"
#include "tansig_table.h"
#define MAX_NEURONS 100

#if 0
static OPUS_INLINE opus_val16 tansig_approx(opus_val32 _x) /* Q19 */
{
    int i;
    opus_val16 xx; /* Q11 */
    /*double x, y;*/
    opus_val16 dy, yy; /* Q14 */
    /*x = 1.9073e-06*_x;*/
    if (_x>=QCONST32(8,19))
        return QCONST32(1.,14);
    if (_x<=-QCONST32(8,19))
        return -QCONST32(1.,14);
    xx = EXTRACT16(SHR32(_x, 8));
    /*i = lrint(25*x);*/
    i = SHR32(ADD32(1024,MULT16_16(25, xx)),11);
    /*x -= .04*i;*/
    xx -= EXTRACT16(SHR32(MULT16_16(20972,i),8));
    /*x = xx*(1./2048);*/
    /*y = tansig_table[250+i];*/
    yy = tansig_table[250+i];
    /*y = yy*(1./16384);*/
    dy = 16384-MULT16_16_Q14(yy,yy);
    yy = yy + MULT16_16_Q14(MULT16_16_Q11(xx,dy),(16384 - MULT16_16_Q11(yy,xx)));
    return yy;
}
#else
/*extern const float tansig_table[501];*/
static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}
#endif

#if 0
void mlp_process(const MLP *m, const opus_val16 *in, opus_val16 *out)
{
    int j;
    opus_val16 hidden[MAX_NEURONS];
    const opus_val16 *W = m->weights;
    /* Copy to tmp_in */
    for (j=0;j<m->topo[1];j++)
    {
        int k;
        opus_val32 sum = SHL32(EXTEND32(*W++),8);
        for (k=0;k<m->topo[0];k++)
            sum = MAC16_16(sum, in[k],*W++);
        hidden[j] = tansig_approx(sum);
    }
    for (j=0;j<m->topo[2];j++)
    {
        int k;
        opus_val32 sum = SHL32(EXTEND32(*W++),14);
        for (k=0;k<m->topo[1];k++)
            sum = MAC16_16(sum, hidden[k], *W++);
        out[j] = tansig_approx(EXTRACT16(PSHR32(sum,17)));
    }
}
#else
void mlp_process(const MLP *m, const float *in, float *out)
{
    int j;
    float hidden[MAX_NEURONS];
    const float *W = m->weights;
    /* Copy to tmp_in */
    for (j=0;j<m->topo[1];j++)
    {
        int k;
        float sum = *W++;
        for (k=0;k<m->topo[0];k++)
            sum = sum + in[k]**W++;
        hidden[j] = tansig_approx(sum);
    }
    for (j=0;j<m->topo[2];j++)
    {
        int k;
        float sum = *W++;
        for (k=0;k<m->topo[1];k++)
            sum = sum + hidden[k]**W++;
        out[j] = tansig_approx(sum);
    }
}
#endif
