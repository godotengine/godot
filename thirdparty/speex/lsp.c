/*---------------------------------------------------------------------------*\
Original copyright
	FILE........: lsp.c
	AUTHOR......: David Rowe
	DATE CREATED: 24/2/93

Heavily modified by Jean-Marc Valin (c) 2002-2006 (fixed-point, 
                       optimizations, additional functions, ...)

   This file contains functions for converting Linear Prediction
   Coefficients (LPC) to Line Spectral Pair (LSP) and back. Note that the
   LSP coefficients are not in radians format but in the x domain of the
   unit circle.

   Speex License:

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
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

/*---------------------------------------------------------------------------*\

  Introduction to Line Spectrum Pairs (LSPs)
  ------------------------------------------

  LSPs are used to encode the LPC filter coefficients {ak} for
  transmission over the channel.  LSPs have several properties (like
  less sensitivity to quantisation noise) that make them superior to
  direct quantisation of {ak}.

  A(z) is a polynomial of order lpcrdr with {ak} as the coefficients.

  A(z) is transformed to P(z) and Q(z) (using a substitution and some
  algebra), to obtain something like:

    A(z) = 0.5[P(z)(z+z^-1) + Q(z)(z-z^-1)]  (1)

  As you can imagine A(z) has complex zeros all over the z-plane. P(z)
  and Q(z) have the very neat property of only having zeros _on_ the
  unit circle.  So to find them we take a test point z=exp(jw) and
  evaluate P (exp(jw)) and Q(exp(jw)) using a grid of points between 0
  and pi.

  The zeros (roots) of P(z) also happen to alternate, which is why we
  swap coefficients as we find roots.  So the process of finding the
  LSP frequencies is basically finding the roots of 5th order
  polynomials.

  The root so P(z) and Q(z) occur in symmetrical pairs at +/-w, hence
  the name Line Spectrum Pairs (LSPs).

  To convert back to ak we just evaluate (1), "clocking" an impulse
  thru it lpcrdr times gives us the impulse response of A(z) which is
  {ak}.

\*---------------------------------------------------------------------------*/


#include "config.h"


#include <math.h>
#include "lsp.h"
#include "stack_alloc.h"
#include "math_approx.h"

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#ifndef NULL
#define NULL 0
#endif

#ifdef FIXED_POINT

#define FREQ_SCALE 16384

/*#define ANGLE2X(a) (32768*cos(((a)/8192.)))*/
#define ANGLE2X(a) (SHL16(spx_cos(a),2))

/*#define X2ANGLE(x) (acos(.00006103515625*(x))*LSP_SCALING)*/
#define X2ANGLE(x) (spx_acos(x))

#ifdef BFIN_ASM
#include "lsp_bfin.h"
#endif

#else

/*#define C1 0.99940307
#define C2 -0.49558072
#define C3 0.03679168*/

#define FREQ_SCALE 1.
#define ANGLE2X(a) (spx_cos(a))
#define X2ANGLE(x) (acos(x))

#endif


/*---------------------------------------------------------------------------*\

   FUNCTION....: cheb_poly_eva()

   AUTHOR......: David Rowe
   DATE CREATED: 24/2/93

   This function evaluates a series of Chebyshev polynomials

\*---------------------------------------------------------------------------*/

#ifdef FIXED_POINT

#ifndef OVERRIDE_CHEB_POLY_EVA
static inline spx_word32_t cheb_poly_eva(
  spx_word16_t *coef, /* P or Q coefs in Q13 format               */
  spx_word16_t     x, /* cos of freq (-1.0 to 1.0) in Q14 format  */
  int              m, /* LPC order/2                              */
  char         *stack
)
{
    int i;
    spx_word16_t b0, b1;
    spx_word32_t sum;

    /*Prevents overflows*/
    if (x>16383)
       x = 16383;
    if (x<-16383)
       x = -16383;

    /* Initialise values */
    b1=16384;
    b0=x;

    /* Evaluate Chebyshev series formulation usin g iterative approach  */
    sum = ADD32(EXTEND32(coef[m]), EXTEND32(MULT16_16_P14(coef[m-1],x)));
    for(i=2;i<=m;i++)
    {
       spx_word16_t tmp=b0;
       b0 = SUB16(MULT16_16_Q13(x,b0), b1);
       b1 = tmp;
       sum = ADD32(sum, EXTEND32(MULT16_16_P14(coef[m-i],b0)));
    }
    
    return sum;
}
#endif

#else

static float cheb_poly_eva(spx_word32_t *coef, spx_word16_t x, int m, char *stack)
{
   int k;
   float b0, b1, tmp;

   /* Initial conditions */
   b0=0; /* b_(m+1) */
   b1=0; /* b_(m+2) */

   x*=2;

   /* Calculate the b_(k) */
   for(k=m;k>0;k--)
   {
      tmp=b0;                           /* tmp holds the previous value of b0 */
      b0=x*b0-b1+coef[m-k];    /* b0 holds its new value based on b0 and b1 */
      b1=tmp;                           /* b1 holds the previous value of b0 */
   }

   return(-b1+.5*x*b0+coef[m]);
}
#endif

/*---------------------------------------------------------------------------*\

    FUNCTION....: lpc_to_lsp()

    AUTHOR......: David Rowe
    DATE CREATED: 24/2/93

    This function converts LPC coefficients to LSP
    coefficients.

\*---------------------------------------------------------------------------*/

#ifdef FIXED_POINT
#define SIGN_CHANGE(a,b) (((a)&0x70000000)^((b)&0x70000000)||(b==0))
#else
#define SIGN_CHANGE(a,b) (((a)*(b))<0.0)
#endif


int lpc_to_lsp (spx_coef_t *a,int lpcrdr,spx_lsp_t *freq,int nb,spx_word16_t delta, char *stack)
/*  float *a 		     	lpc coefficients			*/
/*  int lpcrdr			order of LPC coefficients (10) 		*/
/*  float *freq 	      	LSP frequencies in the x domain       	*/
/*  int nb			number of sub-intervals (4) 		*/
/*  float delta			grid spacing interval (0.02) 		*/


{
    spx_word16_t temp_xr,xl,xr,xm=0;
    spx_word32_t psuml,psumr,psumm,temp_psumr/*,temp_qsumr*/;
    int i,j,m,flag,k;
    VARDECL(spx_word32_t *Q);                 	/* ptrs for memory allocation 		*/
    VARDECL(spx_word32_t *P);
    VARDECL(spx_word16_t *Q16);         /* ptrs for memory allocation 		*/
    VARDECL(spx_word16_t *P16);
    spx_word32_t *px;                	/* ptrs of respective P'(z) & Q'(z)	*/
    spx_word32_t *qx;
    spx_word32_t *p;
    spx_word32_t *q;
    spx_word16_t *pt;                	/* ptr used for cheb_poly_eval()
				whether P' or Q' 			*/
    int roots=0;              	/* DR 8/2/94: number of roots found 	*/
    flag = 1;                	/*  program is searching for a root when,
				1 else has found one 			*/
    m = lpcrdr/2;            	/* order of P'(z) & Q'(z) polynomials 	*/

    /* Allocate memory space for polynomials */
    ALLOC(Q, (m+1), spx_word32_t);
    ALLOC(P, (m+1), spx_word32_t);

    /* determine P'(z)'s and Q'(z)'s coefficients where
      P'(z) = P(z)/(1 + z^(-1)) and Q'(z) = Q(z)/(1-z^(-1)) */

    px = P;                      /* initialise ptrs 			*/
    qx = Q;
    p = px;
    q = qx;

#ifdef FIXED_POINT
    *px++ = LPC_SCALING;
    *qx++ = LPC_SCALING;
    for(i=0;i<m;i++){
       *px++ = SUB32(ADD32(EXTEND32(a[i]),EXTEND32(a[lpcrdr-i-1])), *p++);
       *qx++ = ADD32(SUB32(EXTEND32(a[i]),EXTEND32(a[lpcrdr-i-1])), *q++);
    }
    px = P;
    qx = Q;
    for(i=0;i<m;i++)
    {
       /*if (fabs(*px)>=32768)
          speex_warning_int("px", *px);
       if (fabs(*qx)>=32768)
       speex_warning_int("qx", *qx);*/
       *px = PSHR32(*px,2);
       *qx = PSHR32(*qx,2);
       px++;
       qx++;
    }
    /* The reason for this lies in the way cheb_poly_eva() is implemented for fixed-point */
    P[m] = PSHR32(P[m],3);
    Q[m] = PSHR32(Q[m],3);
#else
    *px++ = LPC_SCALING;
    *qx++ = LPC_SCALING;
    for(i=0;i<m;i++){
       *px++ = (a[i]+a[lpcrdr-1-i]) - *p++;
       *qx++ = (a[i]-a[lpcrdr-1-i]) + *q++;
    }
    px = P;
    qx = Q;
    for(i=0;i<m;i++){
       *px = 2**px;
       *qx = 2**qx;
       px++;
       qx++;
    }
#endif

    px = P;             	/* re-initialise ptrs 			*/
    qx = Q;

    /* now that we have computed P and Q convert to 16 bits to
       speed up cheb_poly_eval */

    ALLOC(P16, m+1, spx_word16_t);
    ALLOC(Q16, m+1, spx_word16_t);

    for (i=0;i<m+1;i++)
    {
       P16[i] = P[i];
       Q16[i] = Q[i];
    }

    /* Search for a zero in P'(z) polynomial first and then alternate to Q'(z).
    Keep alternating between the two polynomials as each zero is found 	*/

    xr = 0;             	/* initialise xr to zero 		*/
    xl = FREQ_SCALE;               	/* start at point xl = 1 		*/

    for(j=0;j<lpcrdr;j++){
	if(j&1)            	/* determines whether P' or Q' is eval. */
	    pt = Q16;
	else
	    pt = P16;

	psuml = cheb_poly_eva(pt,xl,m,stack);	/* evals poly. at xl 	*/
	flag = 1;
	while(flag && (xr >= -FREQ_SCALE)){
           spx_word16_t dd;
           /* Modified by JMV to provide smaller steps around x=+-1 */
#ifdef FIXED_POINT
           dd = MULT16_16_Q15(delta,SUB16(FREQ_SCALE, MULT16_16_Q14(MULT16_16_Q14(xl,xl),14000)));
           if (psuml<512 && psuml>-512)
              dd = PSHR16(dd,1);
#else
           dd=delta*(1-.9*xl*xl);
           if (fabs(psuml)<.2)
              dd *= .5;
#endif
           xr = SUB16(xl, dd);                        	/* interval spacing 	*/
	    psumr = cheb_poly_eva(pt,xr,m,stack);/* poly(xl-delta_x) 	*/
	    temp_psumr = psumr;
	    temp_xr = xr;

    /* if no sign change increment xr and re-evaluate poly(xr). Repeat til
    sign change.
    if a sign change has occurred the interval is bisected and then
    checked again for a sign change which determines in which
    interval the zero lies in.
    If there is no sign change between poly(xm) and poly(xl) set interval
    between xm and xr else set interval between xl and xr and repeat till
    root is located within the specified limits 			*/

	    if(SIGN_CHANGE(psumr,psuml))
            {
		roots++;

		psumm=psuml;
		for(k=0;k<=nb;k++){
#ifdef FIXED_POINT
		    xm = ADD16(PSHR16(xl,1),PSHR16(xr,1));        	/* bisect the interval 	*/
#else
                    xm = .5*(xl+xr);        	/* bisect the interval 	*/
#endif
		    psumm=cheb_poly_eva(pt,xm,m,stack);
		    /*if(psumm*psuml>0.)*/
		    if(!SIGN_CHANGE(psumm,psuml))
                    {
			psuml=psumm;
			xl=xm;
		    } else {
			psumr=psumm;
			xr=xm;
		    }
		}

	       /* once zero is found, reset initial interval to xr 	*/
	       freq[j] = X2ANGLE(xm);
	       xl = xm;
	       flag = 0;       		/* reset flag for next search 	*/
	    }
	    else{
		psuml=temp_psumr;
		xl=temp_xr;
	    }
	}
    }
    return(roots);
}

/*---------------------------------------------------------------------------*\

	FUNCTION....: lsp_to_lpc()

	AUTHOR......: David Rowe
	DATE CREATED: 24/2/93

        Converts LSP coefficients to LPC coefficients.

\*---------------------------------------------------------------------------*/

#ifdef FIXED_POINT

void lsp_to_lpc(spx_lsp_t *freq,spx_coef_t *ak,int lpcrdr, char *stack)
/*  float *freq 	array of LSP frequencies in the x domain	*/
/*  float *ak 		array of LPC coefficients 			*/
/*  int lpcrdr  	order of LPC coefficients 			*/
{
    int i,j;
    spx_word32_t xout1,xout2,xin;
    spx_word32_t mult, a;
    VARDECL(spx_word16_t *freqn);
    VARDECL(spx_word32_t **xp);
    VARDECL(spx_word32_t *xpmem);
    VARDECL(spx_word32_t **xq);
    VARDECL(spx_word32_t *xqmem);
    int m = lpcrdr>>1;

    /* 
    
       Reconstruct P(z) and Q(z) by cascading second order polynomials
       in form 1 - 2cos(w)z(-1) + z(-2), where w is the LSP frequency.
       In the time domain this is:

       y(n) = x(n) - 2cos(w)x(n-1) + x(n-2)
    
       This is what the ALLOCS below are trying to do:

         int xp[m+1][lpcrdr+1+2]; // P matrix in QIMP
         int xq[m+1][lpcrdr+1+2]; // Q matrix in QIMP

       These matrices store the output of each stage on each row.  The
       final (m-th) row has the output of the final (m-th) cascaded
       2nd order filter.  The first row is the impulse input to the
       system (not written as it is known).

       The version below takes advantage of the fact that a lot of the
       outputs are zero or known, for example if we put an inpulse
       into the first section the "clock" it 10 times only the first 3
       outputs samples are non-zero (it's an FIR filter).
    */

    ALLOC(xp, (m+1), spx_word32_t*);
    ALLOC(xpmem, (m+1)*(lpcrdr+1+2), spx_word32_t);

    ALLOC(xq, (m+1), spx_word32_t*);
    ALLOC(xqmem, (m+1)*(lpcrdr+1+2), spx_word32_t);
    
    for(i=0; i<=m; i++) {
      xp[i] = xpmem + i*(lpcrdr+1+2);
      xq[i] = xqmem + i*(lpcrdr+1+2);
    }

    /* work out 2cos terms in Q14 */

    ALLOC(freqn, lpcrdr, spx_word16_t);
    for (i=0;i<lpcrdr;i++) 
       freqn[i] = ANGLE2X(freq[i]);

    #define QIMP  21   /* scaling for impulse */

    xin = SHL32(EXTEND32(1), (QIMP-1)); /* 0.5 in QIMP format */
   
    /* first col and last non-zero values of each row are trivial */
    
    for(i=0;i<=m;i++) {
     xp[i][1] = 0;
     xp[i][2] = xin;
     xp[i][2+2*i] = xin;
     xq[i][1] = 0;
     xq[i][2] = xin;
     xq[i][2+2*i] = xin;
    }

    /* 2nd row (first output row) is trivial */

    xp[1][3] = -MULT16_32_Q14(freqn[0],xp[0][2]);
    xq[1][3] = -MULT16_32_Q14(freqn[1],xq[0][2]);

    xout1 = xout2 = 0;

    /* now generate remaining rows */

    for(i=1;i<m;i++) {

      for(j=1;j<2*(i+1)-1;j++) {
	mult = MULT16_32_Q14(freqn[2*i],xp[i][j+1]);
	xp[i+1][j+2] = ADD32(SUB32(xp[i][j+2], mult), xp[i][j]);
	mult = MULT16_32_Q14(freqn[2*i+1],xq[i][j+1]);
	xq[i+1][j+2] = ADD32(SUB32(xq[i][j+2], mult), xq[i][j]);
      }

      /* for last col xp[i][j+2] = xq[i][j+2] = 0 */

      mult = MULT16_32_Q14(freqn[2*i],xp[i][j+1]);
      xp[i+1][j+2] = SUB32(xp[i][j], mult);
      mult = MULT16_32_Q14(freqn[2*i+1],xq[i][j+1]);
      xq[i+1][j+2] = SUB32(xq[i][j], mult);
    }

    /* process last row to extra a{k} */

    for(j=1;j<=lpcrdr;j++) {
      int shift = QIMP-13;

      /* final filter sections */
      a = PSHR32(xp[m][j+2] + xout1 + xq[m][j+2] - xout2, shift); 
      xout1 = xp[m][j+2];
      xout2 = xq[m][j+2];
      
      /* hard limit ak's to +/- 32767 */

      if (a < -32767) a = -32767;
      if (a > 32767) a = 32767;
      ak[j-1] = (short)a;
     
    }

}

#else

void lsp_to_lpc(spx_lsp_t *freq,spx_coef_t *ak,int lpcrdr, char *stack)
/*  float *freq 	array of LSP frequencies in the x domain	*/
/*  float *ak 		array of LPC coefficients 			*/
/*  int lpcrdr  	order of LPC coefficients 			*/


{
    int i,j;
    float xout1,xout2,xin1,xin2;
    VARDECL(float *Wp);
    float *pw,*n1,*n2,*n3,*n4=NULL;
    VARDECL(float *x_freq);
    int m = lpcrdr>>1;

    ALLOC(Wp, 4*m+2, float);
    pw = Wp;

    /* initialise contents of array */

    for(i=0;i<=4*m+1;i++){       	/* set contents of buffer to 0 */
	*pw++ = 0.0;
    }

    /* Set pointers up */

    pw = Wp;
    xin1 = 1.0;
    xin2 = 1.0;

    ALLOC(x_freq, lpcrdr, float);
    for (i=0;i<lpcrdr;i++)
       x_freq[i] = ANGLE2X(freq[i]);

    /* reconstruct P(z) and Q(z) by  cascading second order
      polynomials in form 1 - 2xz(-1) +z(-2), where x is the
      LSP coefficient */

    for(j=0;j<=lpcrdr;j++){
       int i2=0;
	for(i=0;i<m;i++,i2+=2){
	    n1 = pw+(i*4);
	    n2 = n1 + 1;
	    n3 = n2 + 1;
	    n4 = n3 + 1;
	    xout1 = xin1 - 2.f*x_freq[i2] * *n1 + *n2;
	    xout2 = xin2 - 2.f*x_freq[i2+1] * *n3 + *n4;
	    *n2 = *n1;
	    *n4 = *n3;
	    *n1 = xin1;
	    *n3 = xin2;
	    xin1 = xout1;
	    xin2 = xout2;
	}
	xout1 = xin1 + *(n4+1);
	xout2 = xin2 - *(n4+2);
	if (j>0)
	   ak[j-1] = (xout1 + xout2)*0.5f;
	*(n4+1) = xin1;
	*(n4+2) = xin2;

	xin1 = 0.0;
	xin2 = 0.0;
    }

}
#endif


#ifdef FIXED_POINT

/*Makes sure the LSPs are stable*/
void lsp_enforce_margin(spx_lsp_t *lsp, int len, spx_word16_t margin)
{
   int i;
   spx_word16_t m = margin;
   spx_word16_t m2 = 25736-margin;
  
   if (lsp[0]<m)
      lsp[0]=m;
   if (lsp[len-1]>m2)
      lsp[len-1]=m2;
   for (i=1;i<len-1;i++)
   {
      if (lsp[i]<lsp[i-1]+m)
         lsp[i]=lsp[i-1]+m;

      if (lsp[i]>lsp[i+1]-m)
         lsp[i]= SHR16(lsp[i],1) + SHR16(lsp[i+1]-m,1);
   }
}


void lsp_interpolate(spx_lsp_t *old_lsp, spx_lsp_t *new_lsp, spx_lsp_t *interp_lsp, int len, int subframe, int nb_subframes)
{
   int i;
   spx_word16_t tmp = DIV32_16(SHL32(EXTEND32(1 + subframe),14),nb_subframes);
   spx_word16_t tmp2 = 16384-tmp;
   for (i=0;i<len;i++)
   {
      interp_lsp[i] = MULT16_16_P14(tmp2,old_lsp[i]) + MULT16_16_P14(tmp,new_lsp[i]);
   }
}

#else

/*Makes sure the LSPs are stable*/
void lsp_enforce_margin(spx_lsp_t *lsp, int len, spx_word16_t margin)
{
   int i;
   if (lsp[0]<LSP_SCALING*margin)
      lsp[0]=LSP_SCALING*margin;
   if (lsp[len-1]>LSP_SCALING*(M_PI-margin))
      lsp[len-1]=LSP_SCALING*(M_PI-margin);
   for (i=1;i<len-1;i++)
   {
      if (lsp[i]<lsp[i-1]+LSP_SCALING*margin)
         lsp[i]=lsp[i-1]+LSP_SCALING*margin;

      if (lsp[i]>lsp[i+1]-LSP_SCALING*margin)
         lsp[i]= .5f* (lsp[i] + lsp[i+1]-LSP_SCALING*margin);
   }
}


void lsp_interpolate(spx_lsp_t *old_lsp, spx_lsp_t *new_lsp, spx_lsp_t *interp_lsp, int len, int subframe, int nb_subframes)
{
   int i;
   float tmp = (1.0f + subframe)/nb_subframes;
   for (i=0;i<len;i++)
   {
      interp_lsp[i] = (1-tmp)*old_lsp[i] + tmp*new_lsp[i];
   }
}

#endif
