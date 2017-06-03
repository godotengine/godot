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

#ifndef SILK_DEBUG_H
#define SILK_DEBUG_H

#include "typedef.h"
#include <stdio.h>      /* file writing */
#include <string.h>     /* strcpy, strcmp */

#ifdef  __cplusplus
extern "C"
{
#endif

unsigned long GetHighResolutionTime(void); /* O  time in usec*/

/* make SILK_DEBUG dependent on compiler's _DEBUG */
#if defined _WIN32
    #ifdef _DEBUG
        #define SILK_DEBUG  1
    #else
        #define SILK_DEBUG  0
    #endif

    /* overrule the above */
    #if 0
    /*  #define NO_ASSERTS*/
    #undef  SILK_DEBUG
    #define SILK_DEBUG  1
    #endif
#else
    #define SILK_DEBUG  0
#endif

/* Flag for using timers */
#define SILK_TIC_TOC    0


#if SILK_TIC_TOC

#if (defined(_WIN32) || defined(_WINCE))
#include <windows.h>    /* timer */
#else   /* Linux or Mac*/
#include <sys/time.h>
#endif

/*********************************/
/* timer functions for profiling */
/*********************************/
/* example:                                                         */
/*                                                                  */
/* TIC(LPC)                                                         */
/* do_LPC(in_vec, order, acoef);    // do LPC analysis              */
/* TOC(LPC)                                                         */
/*                                                                  */
/* and call the following just before exiting (from main)           */
/*                                                                  */
/* silk_TimerSave("silk_TimingData.txt");                           */
/*                                                                  */
/* results are now in silk_TimingData.txt                           */

void silk_TimerSave(char *file_name);

/* max number of timers (in different locations) */
#define silk_NUM_TIMERS_MAX                  50
/* max length of name tags in TIC(..), TOC(..) */
#define silk_NUM_TIMERS_MAX_TAG_LEN          30

extern int           silk_Timer_nTimers;
extern int           silk_Timer_depth_ctr;
extern char          silk_Timer_tags[silk_NUM_TIMERS_MAX][silk_NUM_TIMERS_MAX_TAG_LEN];
#ifdef _WIN32
extern LARGE_INTEGER silk_Timer_start[silk_NUM_TIMERS_MAX];
#else
extern unsigned long silk_Timer_start[silk_NUM_TIMERS_MAX];
#endif
extern unsigned int  silk_Timer_cnt[silk_NUM_TIMERS_MAX];
extern opus_int64    silk_Timer_sum[silk_NUM_TIMERS_MAX];
extern opus_int64    silk_Timer_max[silk_NUM_TIMERS_MAX];
extern opus_int64    silk_Timer_min[silk_NUM_TIMERS_MAX];
extern opus_int64    silk_Timer_depth[silk_NUM_TIMERS_MAX];

/* WARNING: TIC()/TOC can measure only up to 0.1 seconds at a time */
#ifdef _WIN32
#define TIC(TAG_NAME) {                                     \
    static int init = 0;                                    \
    static int ID = -1;                                     \
    if( init == 0 )                                         \
    {                                                       \
        int k;                                              \
        init = 1;                                           \
        for( k = 0; k < silk_Timer_nTimers; k++ ) {         \
            if( strcmp(silk_Timer_tags[k], #TAG_NAME) == 0 ) { \
                ID = k;                                     \
                break;                                      \
            }                                               \
        }                                                   \
        if (ID == -1) {                                     \
            ID = silk_Timer_nTimers;                        \
            silk_Timer_nTimers++;                           \
            silk_Timer_depth[ID] = silk_Timer_depth_ctr;    \
            strcpy(silk_Timer_tags[ID], #TAG_NAME);         \
            silk_Timer_cnt[ID] = 0;                         \
            silk_Timer_sum[ID] = 0;                         \
            silk_Timer_min[ID] = 0xFFFFFFFF;                \
            silk_Timer_max[ID] = 0;                         \
        }                                                   \
    }                                                       \
    silk_Timer_depth_ctr++;                                 \
    QueryPerformanceCounter(&silk_Timer_start[ID]);         \
}
#else
#define TIC(TAG_NAME) {                                     \
    static int init = 0;                                    \
    static int ID = -1;                                     \
    if( init == 0 )                                         \
    {                                                       \
        int k;                                              \
        init = 1;                                           \
        for( k = 0; k < silk_Timer_nTimers; k++ ) {         \
        if( strcmp(silk_Timer_tags[k], #TAG_NAME) == 0 ) {  \
                ID = k;                                     \
                break;                                      \
            }                                               \
        }                                                   \
        if (ID == -1) {                                     \
            ID = silk_Timer_nTimers;                        \
            silk_Timer_nTimers++;                           \
            silk_Timer_depth[ID] = silk_Timer_depth_ctr;    \
            strcpy(silk_Timer_tags[ID], #TAG_NAME);         \
            silk_Timer_cnt[ID] = 0;                         \
            silk_Timer_sum[ID] = 0;                         \
            silk_Timer_min[ID] = 0xFFFFFFFF;                \
            silk_Timer_max[ID] = 0;                         \
        }                                                   \
    }                                                       \
    silk_Timer_depth_ctr++;                                 \
    silk_Timer_start[ID] = GetHighResolutionTime();         \
}
#endif

#ifdef _WIN32
#define TOC(TAG_NAME) {                                             \
    LARGE_INTEGER lpPerformanceCount;                               \
    static int init = 0;                                            \
    static int ID = 0;                                              \
    if( init == 0 )                                                 \
    {                                                               \
        int k;                                                      \
        init = 1;                                                   \
        for( k = 0; k < silk_Timer_nTimers; k++ ) {                 \
            if( strcmp(silk_Timer_tags[k], #TAG_NAME) == 0 ) {      \
                ID = k;                                             \
                break;                                              \
            }                                                       \
        }                                                           \
    }                                                               \
    QueryPerformanceCounter(&lpPerformanceCount);                   \
    lpPerformanceCount.QuadPart -= silk_Timer_start[ID].QuadPart;   \
    if((lpPerformanceCount.QuadPart < 100000000) &&                 \
        (lpPerformanceCount.QuadPart >= 0)) {                       \
        silk_Timer_cnt[ID]++;                                       \
        silk_Timer_sum[ID] += lpPerformanceCount.QuadPart;          \
        if( lpPerformanceCount.QuadPart > silk_Timer_max[ID] )      \
            silk_Timer_max[ID] = lpPerformanceCount.QuadPart;       \
        if( lpPerformanceCount.QuadPart < silk_Timer_min[ID] )      \
            silk_Timer_min[ID] = lpPerformanceCount.QuadPart;       \
    }                                                               \
    silk_Timer_depth_ctr--;                                         \
}
#else
#define TOC(TAG_NAME) {                                             \
    unsigned long endTime;                                          \
    static int init = 0;                                            \
    static int ID = 0;                                              \
    if( init == 0 )                                                 \
    {                                                               \
        int k;                                                      \
        init = 1;                                                   \
        for( k = 0; k < silk_Timer_nTimers; k++ ) {                 \
            if( strcmp(silk_Timer_tags[k], #TAG_NAME) == 0 ) {      \
                ID = k;                                             \
                break;                                              \
            }                                                       \
        }                                                           \
    }                                                               \
    endTime = GetHighResolutionTime();                              \
    endTime -= silk_Timer_start[ID];                                \
    if((endTime < 100000000) &&                                     \
        (endTime >= 0)) {                                           \
        silk_Timer_cnt[ID]++;                                       \
        silk_Timer_sum[ID] += endTime;                              \
        if( endTime > silk_Timer_max[ID] )                          \
            silk_Timer_max[ID] = endTime;                           \
        if( endTime < silk_Timer_min[ID] )                          \
            silk_Timer_min[ID] = endTime;                           \
    }                                                               \
        silk_Timer_depth_ctr--;                                     \
}
#endif

#else /* SILK_TIC_TOC */

/* define macros as empty strings */
#define TIC(TAG_NAME)
#define TOC(TAG_NAME)
#define silk_TimerSave(FILE_NAME)

#endif /* SILK_TIC_TOC */


#if SILK_DEBUG
/************************************/
/* write data to file for debugging */
/************************************/
/* Example: DEBUG_STORE_DATA(testfile.pcm, &RIN[0], 160*sizeof(opus_int16)); */

#define silk_NUM_STORES_MAX                                  100
extern FILE *silk_debug_store_fp[ silk_NUM_STORES_MAX ];
extern int silk_debug_store_count;

/* Faster way of storing the data */
#define DEBUG_STORE_DATA( FILE_NAME, DATA_PTR, N_BYTES ) {          \
    static opus_int init = 0, cnt = 0;                              \
    static FILE **fp;                                               \
    if (init == 0) {                                                \
        init = 1;                                                   \
        cnt = silk_debug_store_count++;                             \
        silk_debug_store_fp[ cnt ] = fopen(#FILE_NAME, "wb");       \
    }                                                               \
    fwrite((DATA_PTR), (N_BYTES), 1, silk_debug_store_fp[ cnt ]);   \
}

/* Call this at the end of main() */
#define SILK_DEBUG_STORE_CLOSE_FILES {                              \
    opus_int i;                                                     \
    for( i = 0; i < silk_debug_store_count; i++ ) {                 \
        fclose( silk_debug_store_fp[ i ] );                         \
    }                                                               \
}

#else /* SILK_DEBUG */

/* define macros as empty strings */
#define DEBUG_STORE_DATA(FILE_NAME, DATA_PTR, N_BYTES)
#define SILK_DEBUG_STORE_CLOSE_FILES

#endif /* SILK_DEBUG */

#ifdef  __cplusplus
}
#endif

#endif /* SILK_DEBUG_H */
