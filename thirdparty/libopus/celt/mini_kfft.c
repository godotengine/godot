/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */
/* This is a minimalist, concatenated version of kiss-fft just to compute real scalar FFTs. */


#include <stdlib.h>
#include <math.h>
#include <assert.h>


#define mini_kiss_fft_scalar float

typedef struct {
    mini_kiss_fft_scalar r;
    mini_kiss_fft_scalar i;
}mini_kiss_fft_cpx;

typedef struct mini_kiss_fft_state* mini_kiss_fft_cfg;


#define MINI_MAXFACTORS 32
/* e.g. an fft of length 128 has 4 factors
 as far as kissfft is concerned
 4*4*4*2
 */

typedef struct mini_kiss_fft_state{
    int nfft;
    int inverse;
    int factors[2*MINI_MAXFACTORS];
    mini_kiss_fft_cpx twiddles[1];
} mini_kiss_fft_state;

/*
  Explanation of macros dealing with complex math:

   C_MUL(m,a,b)         : m = a*b
   C_FIXDIV( c , div )  : if a fixed point impl., c /= div. noop otherwise
   C_SUB( res, a,b)     : res = a - b
   C_SUBFROM( res , a)  : res -= a
   C_ADDTO( res , a)    : res += a
 * */

#   define S_MUL(a,b) ( (a)*(b) )
#define C_MUL(m,a,b) \
    do{ (m).r = (a).r*(b).r - (a).i*(b).i;\
        (m).i = (a).r*(b).i + (a).i*(b).r; }while(0)
#   define C_FIXDIV(c,div) /* NOOP */
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r *= (s);\
        (c).i *= (s); }while(0)


#define CHECK_OVERFLOW_OP(a,op,b) /* noop */

#define  C_ADD( res, a,b)\
    do { \
        CHECK_OVERFLOW_OP((a).r,+,(b).r)\
        CHECK_OVERFLOW_OP((a).i,+,(b).i)\
        (res).r=(a).r+(b).r;  (res).i=(a).i+(b).i; \
    }while(0)
#define  C_SUB( res, a,b)\
    do { \
        CHECK_OVERFLOW_OP((a).r,-,(b).r)\
        CHECK_OVERFLOW_OP((a).i,-,(b).i)\
        (res).r=(a).r-(b).r;  (res).i=(a).i-(b).i; \
    }while(0)
#define C_ADDTO( res , a)\
    do { \
        CHECK_OVERFLOW_OP((res).r,+,(a).r)\
        CHECK_OVERFLOW_OP((res).i,+,(a).i)\
        (res).r += (a).r;  (res).i += (a).i;\
    }while(0)

#define C_SUBFROM( res , a)\
    do {\
        CHECK_OVERFLOW_OP((res).r,-,(a).r)\
        CHECK_OVERFLOW_OP((res).i,-,(a).i)\
        (res).r -= (a).r;  (res).i -= (a).i; \
    }while(0)


#  define MINI_KISS_FFT_COS(phase) (mini_kiss_fft_scalar) cos(phase)
#  define MINI_KISS_FFT_SIN(phase) (mini_kiss_fft_scalar) sin(phase)
#  define MINI_HALF_OF(x) ((x)*((mini_kiss_fft_scalar).5))

#define  mini_kf_cexp(x,phase) \
    do{ \
        (x)->r = MINI_KISS_FFT_COS(phase);\
        (x)->i = MINI_KISS_FFT_SIN(phase);\
    }while(0)




static void kf_bfly2(
        mini_kiss_fft_cpx * Fout,
        const size_t fstride,
        const mini_kiss_fft_cfg st,
        int m
        )
{
    mini_kiss_fft_cpx * Fout2;
    mini_kiss_fft_cpx * tw1 = st->twiddles;
    mini_kiss_fft_cpx t;
    Fout2 = Fout + m;
    do{
        C_FIXDIV(*Fout,2); C_FIXDIV(*Fout2,2);

        C_MUL (t,  *Fout2 , *tw1);
        tw1 += fstride;
        C_SUB( *Fout2 ,  *Fout , t );
        C_ADDTO( *Fout ,  t );
        ++Fout2;
        ++Fout;
    }while (--m);
}

static void kf_bfly4(
        mini_kiss_fft_cpx * Fout,
        const size_t fstride,
        const mini_kiss_fft_cfg st,
        const size_t m
        )
{
    mini_kiss_fft_cpx *tw1,*tw2,*tw3;
    mini_kiss_fft_cpx scratch[6];
    size_t k=m;
    const size_t m2=2*m;
    const size_t m3=3*m;


    tw3 = tw2 = tw1 = st->twiddles;

    do {
        C_FIXDIV(*Fout,4); C_FIXDIV(Fout[m],4); C_FIXDIV(Fout[m2],4); C_FIXDIV(Fout[m3],4);

        C_MUL(scratch[0],Fout[m] , *tw1 );
        C_MUL(scratch[1],Fout[m2] , *tw2 );
        C_MUL(scratch[2],Fout[m3] , *tw3 );

        C_SUB( scratch[5] , *Fout, scratch[1] );
        C_ADDTO(*Fout, scratch[1]);
        C_ADD( scratch[3] , scratch[0] , scratch[2] );
        C_SUB( scratch[4] , scratch[0] , scratch[2] );
        C_SUB( Fout[m2], *Fout, scratch[3] );
        tw1 += fstride;
        tw2 += fstride*2;
        tw3 += fstride*3;
        C_ADDTO( *Fout , scratch[3] );

        if(st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        }else{
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    }while(--k);
}

static void kf_bfly3(
         mini_kiss_fft_cpx * Fout,
         const size_t fstride,
         const mini_kiss_fft_cfg st,
         size_t m
         )
{
     size_t k=m;
     const size_t m2 = 2*m;
     mini_kiss_fft_cpx *tw1,*tw2;
     mini_kiss_fft_cpx scratch[5];
     mini_kiss_fft_cpx epi3;
     epi3 = st->twiddles[fstride*m];

     tw1=tw2=st->twiddles;

     do{
         C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

         C_MUL(scratch[1],Fout[m] , *tw1);
         C_MUL(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - MINI_HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - MINI_HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
     }while(--k);
}

static void kf_bfly5(
        mini_kiss_fft_cpx * Fout,
        const size_t fstride,
        const mini_kiss_fft_cfg st,
        int m
        )
{
    mini_kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
    int u;
    mini_kiss_fft_cpx scratch[13];
    mini_kiss_fft_cpx * twiddles = st->twiddles;
    mini_kiss_fft_cpx *tw;
    mini_kiss_fft_cpx ya,yb;
    ya = twiddles[fstride*m];
    yb = twiddles[fstride*2*m];

    Fout0=Fout;
    Fout1=Fout0+m;
    Fout2=Fout0+2*m;
    Fout3=Fout0+3*m;
    Fout4=Fout0+4*m;

    tw=st->twiddles;
    for ( u=0; u<m; ++u ) {
        C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
        scratch[0] = *Fout0;

        C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
        C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
        C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
        C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

        C_ADD( scratch[7],scratch[1],scratch[4]);
        C_SUB( scratch[10],scratch[1],scratch[4]);
        C_ADD( scratch[8],scratch[2],scratch[3]);
        C_SUB( scratch[9],scratch[2],scratch[3]);

        Fout0->r += scratch[7].r + scratch[8].r;
        Fout0->i += scratch[7].i + scratch[8].i;

        scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
        scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

        scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
        scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

        C_SUB(*Fout1,scratch[5],scratch[6]);
        C_ADD(*Fout4,scratch[5],scratch[6]);

        scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
        scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
        scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
        scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

        C_ADD(*Fout2,scratch[11],scratch[12]);
        C_SUB(*Fout3,scratch[11],scratch[12]);

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
    }
}


static
void kf_work(
        mini_kiss_fft_cpx * Fout,
        const mini_kiss_fft_cpx * f,
        const size_t fstride,
        int in_stride,
        int * factors,
        const mini_kiss_fft_cfg st
        )
{
    mini_kiss_fft_cpx * Fout_beg=Fout;
    const int p=*factors++; /* the radix  */
    const int m=*factors++; /* stage's fft length/p */
    const mini_kiss_fft_cpx * Fout_end = Fout + p*m;

    if (m==1) {
        do{
            *Fout = *f;
            f += fstride*in_stride;
        }while(++Fout != Fout_end );
    }else{
        do{
            /* recursive call:
               DFT of size m*p performed by doing
               p instances of smaller DFTs of size m,
               each one takes a decimated version of the input */
            kf_work( Fout , f, fstride*p, in_stride, factors,st);
            f += fstride*in_stride;
        }while( (Fout += m) != Fout_end );
    }

    Fout=Fout_beg;

    /* recombine the p smaller DFTs*/
    switch (p) {
        case 2: kf_bfly2(Fout,fstride,st,m); break;
        case 3: kf_bfly3(Fout,fstride,st,m); break;
        case 4: kf_bfly4(Fout,fstride,st,m); break;
        case 5: kf_bfly5(Fout,fstride,st,m); break;
        default: assert(0);
    }
}

/*  facbuf is populated by p1,m1,p2,m2, ...
    where
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static
void kf_factor(int n,int * facbuf)
{
    int p=4;
    double floor_sqrt;
    floor_sqrt = floor( sqrt((double)n) );

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}

/*
 *
 * User-callable function to allocate all necessary storage space for the fft.
 *
 * The return value is a contiguous block of memory, allocated with malloc.  As such,
 * It can be freed with free(), rather than a kiss_fft-specific function.
 * */
mini_kiss_fft_cfg mini_kiss_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem )
{

    mini_kiss_fft_cfg st=NULL;
    size_t memneeded = (sizeof(struct mini_kiss_fft_state)
        + sizeof(mini_kiss_fft_cpx)*(nfft-1)); /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( mini_kiss_fft_cfg)malloc( memneeded );
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (mini_kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        int i;
        st->nfft=nfft;
        st->inverse = inverse_fft;

        for (i=0;i<nfft;++i) {
            const double pi=3.141592653589793238462643383279502884197169399375105820974944;
            double phase = -2*pi*i / nfft;
            if (st->inverse)
                phase *= -1;
            mini_kf_cexp(st->twiddles+i, phase );
        }

        kf_factor(nfft,st->factors);
    }
    return st;
}


void mini_kiss_fft_stride(mini_kiss_fft_cfg st,const mini_kiss_fft_cpx *fin,mini_kiss_fft_cpx *fout,int in_stride)
{
    assert(fin != fout);
    kf_work( fout, fin, 1,in_stride, st->factors,st );
}

void mini_kiss_fft(mini_kiss_fft_cfg cfg,const mini_kiss_fft_cpx *fin,mini_kiss_fft_cpx *fout)
{
    mini_kiss_fft_stride(cfg,fin,fout,1);
}


typedef struct mini_kiss_fftr_state *mini_kiss_fftr_cfg;

typedef struct mini_kiss_fftr_state{
    mini_kiss_fft_cfg substate;
    mini_kiss_fft_cpx * tmpbuf;
    mini_kiss_fft_cpx * super_twiddles;
} mini_kiss_fftr_state;

mini_kiss_fftr_cfg mini_kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{

    int i;
    mini_kiss_fftr_cfg st = NULL;
    size_t subsize = 0, memneeded;

    assert ((nfft & 1) == 0);
    nfft >>= 1;

    mini_kiss_fft_alloc (nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct mini_kiss_fftr_state) + subsize + sizeof(mini_kiss_fft_cpx) * ( nfft * 3 / 2);

    if (lenmem == NULL) {
        st = (mini_kiss_fftr_cfg) malloc(memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (mini_kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (mini_kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (mini_kiss_fft_cpx *)(void *)(((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    mini_kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

    for (i = 0; i < nfft/2; ++i) {
        double phase =
            -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
        if (inverse_fft)
            phase *= -1;
        mini_kf_cexp (st->super_twiddles+i,phase);
    }
    return st;
}

void mini_kiss_fftr(mini_kiss_fftr_cfg st,const mini_kiss_fft_scalar *timedata,mini_kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    mini_kiss_fft_cpx fpnk,fpk,f1k,f2k,tw,tdc;

    assert ( !st->substate->inverse);

    ncfft = st->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    mini_kiss_fft( st->substate , (const mini_kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
    freqdata[ncfft].i = freqdata[0].i = 0;

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k-1]);

        freqdata[k].r = MINI_HALF_OF(f1k.r + tw.r);
        freqdata[k].i = MINI_HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = MINI_HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = MINI_HALF_OF(tw.i - f1k.i);
    }
}
