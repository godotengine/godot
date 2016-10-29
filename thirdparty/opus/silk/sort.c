/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/* Insertion sort (fast for already almost sorted arrays):   */
/* Best case:  O(n)   for an already sorted array            */
/* Worst case: O(n^2) for an inversely sorted array          */
/*                                                           */
/* Shell short:    http://en.wikipedia.org/wiki/Shell_sort   */

#include "SigProc_FIX.h"

void silk_insertion_sort_increasing(
    opus_int32           *a,             /* I/O   Unsorted / Sorted vector               */
    opus_int             *idx,           /* O     Index vector for the sorted elements   */
    const opus_int       L,              /* I     Vector length                          */
    const opus_int       K               /* I     Number of correctly sorted positions   */
)
{
    opus_int32    value;
    opus_int        i, j;

    /* Safety checks */
    silk_assert( K >  0 );
    silk_assert( L >  0 );
    silk_assert( L >= K );

    /* Write start indices in index vector */
    for( i = 0; i < K; i++ ) {
        idx[ i ] = i;
    }

    /* Sort vector elements by value, increasing order */
    for( i = 1; i < K; i++ ) {
        value = a[ i ];
        for( j = i - 1; ( j >= 0 ) && ( value < a[ j ] ); j-- ) {
            a[ j + 1 ]   = a[ j ];       /* Shift value */
            idx[ j + 1 ] = idx[ j ];     /* Shift index */
        }
        a[ j + 1 ]   = value;   /* Write value */
        idx[ j + 1 ] = i;       /* Write index */
    }

    /* If less than L values are asked for, check the remaining values, */
    /* but only spend CPU to ensure that the K first values are correct */
    for( i = K; i < L; i++ ) {
        value = a[ i ];
        if( value < a[ K - 1 ] ) {
            for( j = K - 2; ( j >= 0 ) && ( value < a[ j ] ); j-- ) {
                a[ j + 1 ]   = a[ j ];       /* Shift value */
                idx[ j + 1 ] = idx[ j ];     /* Shift index */
            }
            a[ j + 1 ]   = value;   /* Write value */
            idx[ j + 1 ] = i;       /* Write index */
        }
    }
}

#ifdef FIXED_POINT
/* This function is only used by the fixed-point build */
void silk_insertion_sort_decreasing_int16(
    opus_int16                  *a,                 /* I/O   Unsorted / Sorted vector                                   */
    opus_int                    *idx,               /* O     Index vector for the sorted elements                       */
    const opus_int              L,                  /* I     Vector length                                              */
    const opus_int              K                   /* I     Number of correctly sorted positions                       */
)
{
    opus_int i, j;
    opus_int value;

    /* Safety checks */
    silk_assert( K >  0 );
    silk_assert( L >  0 );
    silk_assert( L >= K );

    /* Write start indices in index vector */
    for( i = 0; i < K; i++ ) {
        idx[ i ] = i;
    }

    /* Sort vector elements by value, decreasing order */
    for( i = 1; i < K; i++ ) {
        value = a[ i ];
        for( j = i - 1; ( j >= 0 ) && ( value > a[ j ] ); j-- ) {
            a[ j + 1 ]   = a[ j ];     /* Shift value */
            idx[ j + 1 ] = idx[ j ];   /* Shift index */
        }
        a[ j + 1 ]   = value;   /* Write value */
        idx[ j + 1 ] = i;       /* Write index */
    }

    /* If less than L values are asked for, check the remaining values, */
    /* but only spend CPU to ensure that the K first values are correct */
    for( i = K; i < L; i++ ) {
        value = a[ i ];
        if( value > a[ K - 1 ] ) {
            for( j = K - 2; ( j >= 0 ) && ( value > a[ j ] ); j-- ) {
                a[ j + 1 ]   = a[ j ];     /* Shift value */
                idx[ j + 1 ] = idx[ j ];   /* Shift index */
            }
            a[ j + 1 ]   = value;   /* Write value */
            idx[ j + 1 ] = i;       /* Write index */
        }
    }
}
#endif

void silk_insertion_sort_increasing_all_values_int16(
     opus_int16                 *a,                 /* I/O   Unsorted / Sorted vector                                   */
     const opus_int             L                   /* I     Vector length                                              */
)
{
    opus_int    value;
    opus_int    i, j;

    /* Safety checks */
    silk_assert( L >  0 );

    /* Sort vector elements by value, increasing order */
    for( i = 1; i < L; i++ ) {
        value = a[ i ];
        for( j = i - 1; ( j >= 0 ) && ( value < a[ j ] ); j-- ) {
            a[ j + 1 ] = a[ j ]; /* Shift value */
        }
        a[ j + 1 ] = value; /* Write value */
    }
}
