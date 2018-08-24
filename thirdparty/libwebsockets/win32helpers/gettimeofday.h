#ifndef _GET_TIME_OF_DAY_H
#define _GET_TIME_OF_DAY_H

#include <time.h>

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

#ifdef LWS_MINGW_SUPPORT
  #include <winsock2.h>
#endif

#ifndef _TIMEZONE_DEFINED 
struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

#endif

int gettimeofday(struct timeval *tv, struct timezone *tz);

#endif
