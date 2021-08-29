/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/
/*
  A brief summary of the date string formats this parser groks:

  RFC 2616 3.3.1

  Sun, 06 Nov 1994 08:49:37 GMT  ; RFC 822, updated by RFC 1123
  Sunday, 06-Nov-94 08:49:37 GMT ; RFC 850, obsoleted by RFC 1036
  Sun Nov  6 08:49:37 1994       ; ANSI C's asctime() format

  we support dates without week day name:

  06 Nov 1994 08:49:37 GMT
  06-Nov-94 08:49:37 GMT
  Nov  6 08:49:37 1994

  without the time zone:

  06 Nov 1994 08:49:37
  06-Nov-94 08:49:37

  weird order:

  1994 Nov 6 08:49:37  (GNU date fails)
  GMT 08:49:37 06-Nov-94 Sunday
  94 6 Nov 08:49:37    (GNU date fails)

  time left out:

  1994 Nov 6
  06-Nov-94
  Sun Nov 6 94

  unusual separators:

  1994.Nov.6
  Sun/Nov/6/94/GMT

  commonly used time zone names:

  Sun, 06 Nov 1994 08:49:37 CET
  06 Nov 1994 08:49:37 EST

  time zones specified using RFC822 style:

  Sun, 12 Sep 2004 15:05:58 -0700
  Sat, 11 Sep 2004 21:32:11 +0200

  compact numerical date strings:

  20040912 15:05:58 -0700
  20040911 +0200

*/

#include "curl_setup.h"

#include <limits.h>

#include <curl/curl.h>
#include "strcase.h"
#include "warnless.h"
#include "parsedate.h"

/*
 * parsedate()
 *
 * Returns:
 *
 * PARSEDATE_OK     - a fine conversion
 * PARSEDATE_FAIL   - failed to convert
 * PARSEDATE_LATER  - time overflow at the far end of time_t
 * PARSEDATE_SOONER - time underflow at the low end of time_t
 */

static int parsedate(const char *date, time_t *output);

#define PARSEDATE_OK     0
#define PARSEDATE_FAIL   -1
#define PARSEDATE_LATER  1
#define PARSEDATE_SOONER 2

#if !defined(CURL_DISABLE_PARSEDATE) || !defined(CURL_DISABLE_FTP) || \
  !defined(CURL_DISABLE_FILE)
/* These names are also used by FTP and FILE code */
const char * const Curl_wkday[] =
{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
const char * const Curl_month[]=
{ "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
#endif

#ifndef CURL_DISABLE_PARSEDATE
static const char * const weekday[] =
{ "Monday", "Tuesday", "Wednesday", "Thursday",
  "Friday", "Saturday", "Sunday" };

struct tzinfo {
  char name[5];
  int offset; /* +/- in minutes */
};

/* Here's a bunch of frequently used time zone names. These were supported
   by the old getdate parser. */
#define tDAYZONE -60       /* offset for daylight savings time */
static const struct tzinfo tz[]= {
  {"GMT", 0},              /* Greenwich Mean */
  {"UT",  0},              /* Universal Time */
  {"UTC", 0},              /* Universal (Coordinated) */
  {"WET", 0},              /* Western European */
  {"BST", 0 tDAYZONE},     /* British Summer */
  {"WAT", 60},             /* West Africa */
  {"AST", 240},            /* Atlantic Standard */
  {"ADT", 240 tDAYZONE},   /* Atlantic Daylight */
  {"EST", 300},            /* Eastern Standard */
  {"EDT", 300 tDAYZONE},   /* Eastern Daylight */
  {"CST", 360},            /* Central Standard */
  {"CDT", 360 tDAYZONE},   /* Central Daylight */
  {"MST", 420},            /* Mountain Standard */
  {"MDT", 420 tDAYZONE},   /* Mountain Daylight */
  {"PST", 480},            /* Pacific Standard */
  {"PDT", 480 tDAYZONE},   /* Pacific Daylight */
  {"YST", 540},            /* Yukon Standard */
  {"YDT", 540 tDAYZONE},   /* Yukon Daylight */
  {"HST", 600},            /* Hawaii Standard */
  {"HDT", 600 tDAYZONE},   /* Hawaii Daylight */
  {"CAT", 600},            /* Central Alaska */
  {"AHST", 600},           /* Alaska-Hawaii Standard */
  {"NT",  660},            /* Nome */
  {"IDLW", 720},           /* International Date Line West */
  {"CET", -60},            /* Central European */
  {"MET", -60},            /* Middle European */
  {"MEWT", -60},           /* Middle European Winter */
  {"MEST", -60 tDAYZONE},  /* Middle European Summer */
  {"CEST", -60 tDAYZONE},  /* Central European Summer */
  {"MESZ", -60 tDAYZONE},  /* Middle European Summer */
  {"FWT", -60},            /* French Winter */
  {"FST", -60 tDAYZONE},   /* French Summer */
  {"EET", -120},           /* Eastern Europe, USSR Zone 1 */
  {"WAST", -420},          /* West Australian Standard */
  {"WADT", -420 tDAYZONE}, /* West Australian Daylight */
  {"CCT", -480},           /* China Coast, USSR Zone 7 */
  {"JST", -540},           /* Japan Standard, USSR Zone 8 */
  {"EAST", -600},          /* Eastern Australian Standard */
  {"EADT", -600 tDAYZONE}, /* Eastern Australian Daylight */
  {"GST", -600},           /* Guam Standard, USSR Zone 9 */
  {"NZT", -720},           /* New Zealand */
  {"NZST", -720},          /* New Zealand Standard */
  {"NZDT", -720 tDAYZONE}, /* New Zealand Daylight */
  {"IDLE", -720},          /* International Date Line East */
  /* Next up: Military timezone names. RFC822 allowed these, but (as noted in
     RFC 1123) had their signs wrong. Here we use the correct signs to match
     actual military usage.
   */
  {"A",  1 * 60},         /* Alpha */
  {"B",  2 * 60},         /* Bravo */
  {"C",  3 * 60},         /* Charlie */
  {"D",  4 * 60},         /* Delta */
  {"E",  5 * 60},         /* Echo */
  {"F",  6 * 60},         /* Foxtrot */
  {"G",  7 * 60},         /* Golf */
  {"H",  8 * 60},         /* Hotel */
  {"I",  9 * 60},         /* India */
  /* "J", Juliet is not used as a timezone, to indicate the observer's local
     time */
  {"K", 10 * 60},         /* Kilo */
  {"L", 11 * 60},         /* Lima */
  {"M", 12 * 60},         /* Mike */
  {"N",  -1 * 60},         /* November */
  {"O",  -2 * 60},         /* Oscar */
  {"P",  -3 * 60},         /* Papa */
  {"Q",  -4 * 60},         /* Quebec */
  {"R",  -5 * 60},         /* Romeo */
  {"S",  -6 * 60},         /* Sierra */
  {"T",  -7 * 60},         /* Tango */
  {"U",  -8 * 60},         /* Uniform */
  {"V",  -9 * 60},         /* Victor */
  {"W", -10 * 60},         /* Whiskey */
  {"X", -11 * 60},         /* X-ray */
  {"Y", -12 * 60},         /* Yankee */
  {"Z", 0},                /* Zulu, zero meridian, a.k.a. UTC */
};

/* returns:
   -1 no day
   0 monday - 6 sunday
*/

static int checkday(const char *check, size_t len)
{
  int i;
  const char * const *what;
  bool found = FALSE;
  if(len > 3)
    what = &weekday[0];
  else
    what = &Curl_wkday[0];
  for(i = 0; i<7; i++) {
    if(strcasecompare(check, what[0])) {
      found = TRUE;
      break;
    }
    what++;
  }
  return found?i:-1;
}

static int checkmonth(const char *check)
{
  int i;
  const char * const *what;
  bool found = FALSE;

  what = &Curl_month[0];
  for(i = 0; i<12; i++) {
    if(strcasecompare(check, what[0])) {
      found = TRUE;
      break;
    }
    what++;
  }
  return found?i:-1; /* return the offset or -1, no real offset is -1 */
}

/* return the time zone offset between GMT and the input one, in number
   of seconds or -1 if the timezone wasn't found/legal */

static int checktz(const char *check)
{
  unsigned int i;
  const struct tzinfo *what;
  bool found = FALSE;

  what = tz;
  for(i = 0; i< sizeof(tz)/sizeof(tz[0]); i++) {
    if(strcasecompare(check, what->name)) {
      found = TRUE;
      break;
    }
    what++;
  }
  return found?what->offset*60:-1;
}

static void skip(const char **date)
{
  /* skip everything that aren't letters or digits */
  while(**date && !ISALNUM(**date))
    (*date)++;
}

enum assume {
  DATE_MDAY,
  DATE_YEAR,
  DATE_TIME
};

/*
 * time2epoch: time stamp to seconds since epoch in GMT time zone.  Similar to
 * mktime but for GMT only.
 */
static time_t time2epoch(int sec, int min, int hour,
                         int mday, int mon, int year)
{
  static const int month_days_cumulative [12] =
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
  int leap_days = year - (mon <= 1);
  leap_days = ((leap_days / 4) - (leap_days / 100) + (leap_days / 400)
               - (1969 / 4) + (1969 / 100) - (1969 / 400));
  return ((((time_t) (year - 1970) * 365
            + leap_days + month_days_cumulative[mon] + mday - 1) * 24
           + hour) * 60 + min) * 60 + sec;
}

/*
 * parsedate()
 *
 * Returns:
 *
 * PARSEDATE_OK     - a fine conversion
 * PARSEDATE_FAIL   - failed to convert
 * PARSEDATE_LATER  - time overflow at the far end of time_t
 * PARSEDATE_SOONER - time underflow at the low end of time_t
 */

static int parsedate(const char *date, time_t *output)
{
  time_t t = 0;
  int wdaynum = -1;  /* day of the week number, 0-6 (mon-sun) */
  int monnum = -1;   /* month of the year number, 0-11 */
  int mdaynum = -1; /* day of month, 1 - 31 */
  int hournum = -1;
  int minnum = -1;
  int secnum = -1;
  int yearnum = -1;
  int tzoff = -1;
  enum assume dignext = DATE_MDAY;
  const char *indate = date; /* save the original pointer */
  int part = 0; /* max 6 parts */

  while(*date && (part < 6)) {
    bool found = FALSE;

    skip(&date);

    if(ISALPHA(*date)) {
      /* a name coming up */
      char buf[32]="";
      size_t len;
      if(sscanf(date, "%31[ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          "abcdefghijklmnopqrstuvwxyz]", buf))
        len = strlen(buf);
      else
        len = 0;

      if(wdaynum == -1) {
        wdaynum = checkday(buf, len);
        if(wdaynum != -1)
          found = TRUE;
      }
      if(!found && (monnum == -1)) {
        monnum = checkmonth(buf);
        if(monnum != -1)
          found = TRUE;
      }

      if(!found && (tzoff == -1)) {
        /* this just must be a time zone string */
        tzoff = checktz(buf);
        if(tzoff != -1)
          found = TRUE;
      }

      if(!found)
        return PARSEDATE_FAIL; /* bad string */

      date += len;
    }
    else if(ISDIGIT(*date)) {
      /* a digit */
      int val;
      char *end;
      int len = 0;
      if((secnum == -1) &&
         (3 == sscanf(date, "%02d:%02d:%02d%n",
                      &hournum, &minnum, &secnum, &len))) {
        /* time stamp! */
        date += len;
      }
      else if((secnum == -1) &&
              (2 == sscanf(date, "%02d:%02d%n", &hournum, &minnum, &len))) {
        /* time stamp without seconds */
        date += len;
        secnum = 0;
      }
      else {
        long lval;
        int error;
        int old_errno;

        old_errno = errno;
        errno = 0;
        lval = strtol(date, &end, 10);
        error = errno;
        if(errno != old_errno)
          errno = old_errno;

        if(error)
          return PARSEDATE_FAIL;

#if LONG_MAX != INT_MAX
        if((lval > (long)INT_MAX) || (lval < (long)INT_MIN))
          return PARSEDATE_FAIL;
#endif

        val = curlx_sltosi(lval);

        if((tzoff == -1) &&
           ((end - date) == 4) &&
           (val <= 1400) &&
           (indate< date) &&
           ((date[-1] == '+' || date[-1] == '-'))) {
          /* four digits and a value less than or equal to 1400 (to take into
             account all sorts of funny time zone diffs) and it is preceded
             with a plus or minus. This is a time zone indication.  1400 is
             picked since +1300 is frequently used and +1400 is mentioned as
             an edge number in the document "ISO C 200X Proposal: Timezone
             Functions" at http://david.tribble.com/text/c0xtimezone.html If
             anyone has a more authoritative source for the exact maximum time
             zone offsets, please speak up! */
          found = TRUE;
          tzoff = (val/100 * 60 + val%100)*60;

          /* the + and - prefix indicates the local time compared to GMT,
             this we need their reversed math to get what we want */
          tzoff = date[-1]=='+'?-tzoff:tzoff;
        }

        if(((end - date) == 8) &&
           (yearnum == -1) &&
           (monnum == -1) &&
           (mdaynum == -1)) {
          /* 8 digits, no year, month or day yet. This is YYYYMMDD */
          found = TRUE;
          yearnum = val/10000;
          monnum = (val%10000)/100-1; /* month is 0 - 11 */
          mdaynum = val%100;
        }

        if(!found && (dignext == DATE_MDAY) && (mdaynum == -1)) {
          if((val > 0) && (val<32)) {
            mdaynum = val;
            found = TRUE;
          }
          dignext = DATE_YEAR;
        }

        if(!found && (dignext == DATE_YEAR) && (yearnum == -1)) {
          yearnum = val;
          found = TRUE;
          if(yearnum < 100) {
            if(yearnum > 70)
              yearnum += 1900;
            else
              yearnum += 2000;
          }
          if(mdaynum == -1)
            dignext = DATE_MDAY;
        }

        if(!found)
          return PARSEDATE_FAIL;

        date = end;
      }
    }

    part++;
  }

  if(-1 == secnum)
    secnum = minnum = hournum = 0; /* no time, make it zero */

  if((-1 == mdaynum) ||
     (-1 == monnum) ||
     (-1 == yearnum))
    /* lacks vital info, fail */
    return PARSEDATE_FAIL;

#ifdef HAVE_TIME_T_UNSIGNED
  if(yearnum < 1970) {
    /* only positive numbers cannot return earlier */
    *output = TIME_T_MIN;
    return PARSEDATE_SOONER;
  }
#endif

#if (SIZEOF_TIME_T < 5)

#ifdef HAVE_TIME_T_UNSIGNED
  /* an unsigned 32 bit time_t can only hold dates to 2106 */
  if(yearnum > 2105) {
    *output = TIME_T_MAX;
    return PARSEDATE_LATER;
  }
#else
  /* a signed 32 bit time_t can only hold dates to the beginning of 2038 */
  if(yearnum > 2037) {
    *output = TIME_T_MAX;
    return PARSEDATE_LATER;
  }
  if(yearnum < 1903) {
    *output = TIME_T_MIN;
    return PARSEDATE_SOONER;
  }
#endif

#else
  /* The Gregorian calendar was introduced 1582 */
  if(yearnum < 1583)
    return PARSEDATE_FAIL;
#endif

  if((mdaynum > 31) || (monnum > 11) ||
     (hournum > 23) || (minnum > 59) || (secnum > 60))
    return PARSEDATE_FAIL; /* clearly an illegal date */

  /* time2epoch() returns a time_t. time_t is often 32 bits, sometimes even on
     architectures that feature 64 bit 'long' but ultimately time_t is the
     correct data type to use.
  */
  t = time2epoch(secnum, minnum, hournum, mdaynum, monnum, yearnum);

  /* Add the time zone diff between local time zone and GMT. */
  if(tzoff == -1)
    tzoff = 0;

  if((tzoff > 0) && (t > TIME_T_MAX - tzoff)) {
    *output = TIME_T_MAX;
    return PARSEDATE_LATER; /* time_t overflow */
  }

  t += tzoff;

  *output = t;

  return PARSEDATE_OK;
}
#else
/* disabled */
static int parsedate(const char *date, time_t *output)
{
  (void)date;
  *output = 0;
  return PARSEDATE_OK; /* a lie */
}
#endif

time_t curl_getdate(const char *p, const time_t *now)
{
  time_t parsed = -1;
  int rc = parsedate(p, &parsed);
  (void)now; /* legacy argument from the past that we ignore */

  if(rc == PARSEDATE_OK) {
    if(parsed == -1)
      /* avoid returning -1 for a working scenario */
      parsed++;
    return parsed;
  }
  /* everything else is fail */
  return -1;
}

/* Curl_getdate_capped() differs from curl_getdate() in that this will return
   TIME_T_MAX in case the parsed time value was too big, instead of an
   error. */

time_t Curl_getdate_capped(const char *p)
{
  time_t parsed = -1;
  int rc = parsedate(p, &parsed);

  switch(rc) {
  case PARSEDATE_OK:
    if(parsed == -1)
      /* avoid returning -1 for a working scenario */
      parsed++;
    return parsed;
  case PARSEDATE_LATER:
    /* this returns the maximum time value */
    return parsed;
  default:
    return -1; /* everything else is fail */
  }
  /* UNREACHABLE */
}

/*
 * Curl_gmtime() is a gmtime() replacement for portability. Do not use the
 * gmtime_r() or gmtime() functions anywhere else but here.
 *
 */

CURLcode Curl_gmtime(time_t intime, struct tm *store)
{
  const struct tm *tm;
#ifdef HAVE_GMTIME_R
  /* thread-safe version */
  tm = (struct tm *)gmtime_r(&intime, store);
#else
  /* !checksrc! disable BANNEDFUNC 1 */
  tm = gmtime(&intime);
  if(tm)
    *store = *tm; /* copy the pointed struct to the local copy */
#endif

  if(!tm)
    return CURLE_BAD_FUNCTION_ARGUMENT;
  return CURLE_OK;
}
