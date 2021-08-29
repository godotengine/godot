/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#include "curl_setup.h"

#include "urldata.h"
#include "sendf.h"
#include "multiif.h"
#include "progress.h"
#include "timeval.h"
#include "curl_printf.h"

/* check rate limits within this many recent milliseconds, at minimum. */
#define MIN_RATE_LIMIT_PERIOD 3000

#ifndef CURL_DISABLE_PROGRESS_METER
/* Provide a string that is 2 + 1 + 2 + 1 + 2 = 8 letters long (plus the zero
   byte) */
static void time2str(char *r, curl_off_t seconds)
{
  curl_off_t h;
  if(seconds <= 0) {
    strcpy(r, "--:--:--");
    return;
  }
  h = seconds / CURL_OFF_T_C(3600);
  if(h <= CURL_OFF_T_C(99)) {
    curl_off_t m = (seconds - (h*CURL_OFF_T_C(3600))) / CURL_OFF_T_C(60);
    curl_off_t s = (seconds - (h*CURL_OFF_T_C(3600))) - (m*CURL_OFF_T_C(60));
    msnprintf(r, 9, "%2" CURL_FORMAT_CURL_OFF_T ":%02" CURL_FORMAT_CURL_OFF_T
              ":%02" CURL_FORMAT_CURL_OFF_T, h, m, s);
  }
  else {
    /* this equals to more than 99 hours, switch to a more suitable output
       format to fit within the limits. */
    curl_off_t d = seconds / CURL_OFF_T_C(86400);
    h = (seconds - (d*CURL_OFF_T_C(86400))) / CURL_OFF_T_C(3600);
    if(d <= CURL_OFF_T_C(999))
      msnprintf(r, 9, "%3" CURL_FORMAT_CURL_OFF_T
                "d %02" CURL_FORMAT_CURL_OFF_T "h", d, h);
    else
      msnprintf(r, 9, "%7" CURL_FORMAT_CURL_OFF_T "d", d);
  }
}

/* The point of this function would be to return a string of the input data,
   but never longer than 5 columns (+ one zero byte).
   Add suffix k, M, G when suitable... */
static char *max5data(curl_off_t bytes, char *max5)
{
#define ONE_KILOBYTE  CURL_OFF_T_C(1024)
#define ONE_MEGABYTE (CURL_OFF_T_C(1024) * ONE_KILOBYTE)
#define ONE_GIGABYTE (CURL_OFF_T_C(1024) * ONE_MEGABYTE)
#define ONE_TERABYTE (CURL_OFF_T_C(1024) * ONE_GIGABYTE)
#define ONE_PETABYTE (CURL_OFF_T_C(1024) * ONE_TERABYTE)

  if(bytes < CURL_OFF_T_C(100000))
    msnprintf(max5, 6, "%5" CURL_FORMAT_CURL_OFF_T, bytes);

  else if(bytes < CURL_OFF_T_C(10000) * ONE_KILOBYTE)
    msnprintf(max5, 6, "%4" CURL_FORMAT_CURL_OFF_T "k", bytes/ONE_KILOBYTE);

  else if(bytes < CURL_OFF_T_C(100) * ONE_MEGABYTE)
    /* 'XX.XM' is good as long as we're less than 100 megs */
    msnprintf(max5, 6, "%2" CURL_FORMAT_CURL_OFF_T ".%0"
              CURL_FORMAT_CURL_OFF_T "M", bytes/ONE_MEGABYTE,
              (bytes%ONE_MEGABYTE) / (ONE_MEGABYTE/CURL_OFF_T_C(10)) );

#if (SIZEOF_CURL_OFF_T > 4)

  else if(bytes < CURL_OFF_T_C(10000) * ONE_MEGABYTE)
    /* 'XXXXM' is good until we're at 10000MB or above */
    msnprintf(max5, 6, "%4" CURL_FORMAT_CURL_OFF_T "M", bytes/ONE_MEGABYTE);

  else if(bytes < CURL_OFF_T_C(100) * ONE_GIGABYTE)
    /* 10000 MB - 100 GB, we show it as XX.XG */
    msnprintf(max5, 6, "%2" CURL_FORMAT_CURL_OFF_T ".%0"
              CURL_FORMAT_CURL_OFF_T "G", bytes/ONE_GIGABYTE,
              (bytes%ONE_GIGABYTE) / (ONE_GIGABYTE/CURL_OFF_T_C(10)) );

  else if(bytes < CURL_OFF_T_C(10000) * ONE_GIGABYTE)
    /* up to 10000GB, display without decimal: XXXXG */
    msnprintf(max5, 6, "%4" CURL_FORMAT_CURL_OFF_T "G", bytes/ONE_GIGABYTE);

  else if(bytes < CURL_OFF_T_C(10000) * ONE_TERABYTE)
    /* up to 10000TB, display without decimal: XXXXT */
    msnprintf(max5, 6, "%4" CURL_FORMAT_CURL_OFF_T "T", bytes/ONE_TERABYTE);

  else
    /* up to 10000PB, display without decimal: XXXXP */
    msnprintf(max5, 6, "%4" CURL_FORMAT_CURL_OFF_T "P", bytes/ONE_PETABYTE);

    /* 16384 petabytes (16 exabytes) is the maximum a 64 bit unsigned number
       can hold, but our data type is signed so 8192PB will be the maximum. */

#else

  else
    msnprintf(max5, 6, "%4" CURL_FORMAT_CURL_OFF_T "M", bytes/ONE_MEGABYTE);

#endif

  return max5;
}
#endif

/*

   New proposed interface, 9th of February 2000:

   pgrsStartNow() - sets start time
   pgrsSetDownloadSize(x) - known expected download size
   pgrsSetUploadSize(x) - known expected upload size
   pgrsSetDownloadCounter() - amount of data currently downloaded
   pgrsSetUploadCounter() - amount of data currently uploaded
   pgrsUpdate() - show progress
   pgrsDone() - transfer complete

*/

int Curl_pgrsDone(struct Curl_easy *data)
{
  int rc;
  data->progress.lastshow = 0;
  rc = Curl_pgrsUpdate(data); /* the final (forced) update */
  if(rc)
    return rc;

  if(!(data->progress.flags & PGRS_HIDE) &&
     !data->progress.callback)
    /* only output if we don't use a progress callback and we're not
     * hidden */
    fprintf(data->set.err, "\n");

  data->progress.speeder_c = 0; /* reset the progress meter display */
  return 0;
}

/* reset the known transfer sizes */
void Curl_pgrsResetTransferSizes(struct Curl_easy *data)
{
  Curl_pgrsSetDownloadSize(data, -1);
  Curl_pgrsSetUploadSize(data, -1);
}

/*
 *
 * Curl_pgrsTime(). Store the current time at the given label. This fetches a
 * fresh "now" and returns it.
 *
 * @unittest: 1399
 */
struct curltime Curl_pgrsTime(struct Curl_easy *data, timerid timer)
{
  struct curltime now = Curl_now();
  timediff_t *delta = NULL;

  switch(timer) {
  default:
  case TIMER_NONE:
    /* mistake filter */
    break;
  case TIMER_STARTOP:
    /* This is set at the start of a transfer */
    data->progress.t_startop = now;
    break;
  case TIMER_STARTSINGLE:
    /* This is set at the start of each single fetch */
    data->progress.t_startsingle = now;
    data->progress.is_t_startransfer_set = false;
    break;
  case TIMER_STARTACCEPT:
    data->progress.t_acceptdata = now;
    break;
  case TIMER_NAMELOOKUP:
    delta = &data->progress.t_nslookup;
    break;
  case TIMER_CONNECT:
    delta = &data->progress.t_connect;
    break;
  case TIMER_APPCONNECT:
    delta = &data->progress.t_appconnect;
    break;
  case TIMER_PRETRANSFER:
    delta = &data->progress.t_pretransfer;
    break;
  case TIMER_STARTTRANSFER:
    delta = &data->progress.t_starttransfer;
    /* prevent updating t_starttransfer unless:
     *   1) this is the first time we're setting t_starttransfer
     *   2) a redirect has occurred since the last time t_starttransfer was set
     * This prevents repeated invocations of the function from incorrectly
     * changing the t_starttransfer time.
     */
    if(data->progress.is_t_startransfer_set) {
      return now;
    }
    else {
      data->progress.is_t_startransfer_set = true;
      break;
    }
  case TIMER_POSTRANSFER:
    /* this is the normal end-of-transfer thing */
    break;
  case TIMER_REDIRECT:
    data->progress.t_redirect = Curl_timediff_us(now, data->progress.start);
    break;
  }
  if(delta) {
    timediff_t us = Curl_timediff_us(now, data->progress.t_startsingle);
    if(us < 1)
      us = 1; /* make sure at least one microsecond passed */
    *delta += us;
  }
  return now;
}

void Curl_pgrsStartNow(struct Curl_easy *data)
{
  data->progress.speeder_c = 0; /* reset the progress meter display */
  data->progress.start = Curl_now();
  data->progress.is_t_startransfer_set = false;
  data->progress.ul_limit_start = data->progress.start;
  data->progress.dl_limit_start = data->progress.start;
  data->progress.ul_limit_size = 0;
  data->progress.dl_limit_size = 0;
  data->progress.downloaded = 0;
  data->progress.uploaded = 0;
  /* clear all bits except HIDE and HEADERS_OUT */
  data->progress.flags &= PGRS_HIDE|PGRS_HEADERS_OUT;
  Curl_ratelimit(data, data->progress.start);
}

/*
 * This is used to handle speed limits, calculating how many milliseconds to
 * wait until we're back under the speed limit, if needed.
 *
 * The way it works is by having a "starting point" (time & amount of data
 * transferred by then) used in the speed computation, to be used instead of
 * the start of the transfer.  This starting point is regularly moved as
 * transfer goes on, to keep getting accurate values (instead of average over
 * the entire transfer).
 *
 * This function takes the current amount of data transferred, the amount at
 * the starting point, the limit (in bytes/s), the time of the starting point
 * and the current time.
 *
 * Returns 0 if no waiting is needed or when no waiting is needed but the
 * starting point should be reset (to current); or the number of milliseconds
 * to wait to get back under the speed limit.
 */
timediff_t Curl_pgrsLimitWaitTime(curl_off_t cursize,
                                  curl_off_t startsize,
                                  curl_off_t limit,
                                  struct curltime start,
                                  struct curltime now)
{
  curl_off_t size = cursize - startsize;
  timediff_t minimum;
  timediff_t actual;

  if(!limit || !size)
    return 0;

  /*
   * 'minimum' is the number of milliseconds 'size' should take to download to
   * stay below 'limit'.
   */
  if(size < CURL_OFF_T_MAX/1000)
    minimum = (timediff_t) (CURL_OFF_T_C(1000) * size / limit);
  else {
    minimum = (timediff_t) (size / limit);
    if(minimum < TIMEDIFF_T_MAX/1000)
      minimum *= 1000;
    else
      minimum = TIMEDIFF_T_MAX;
  }

  /*
   * 'actual' is the time in milliseconds it took to actually download the
   * last 'size' bytes.
   */
  actual = Curl_timediff(now, start);
  if(actual < minimum) {
    /* if it downloaded the data faster than the limit, make it wait the
       difference */
    return (minimum - actual);
  }

  return 0;
}

/*
 * Set the number of downloaded bytes so far.
 */
void Curl_pgrsSetDownloadCounter(struct Curl_easy *data, curl_off_t size)
{
  data->progress.downloaded = size;
}

/*
 * Update the timestamp and sizestamp to use for rate limit calculations.
 */
void Curl_ratelimit(struct Curl_easy *data, struct curltime now)
{
  /* don't set a new stamp unless the time since last update is long enough */
  if(data->set.max_recv_speed) {
    if(Curl_timediff(now, data->progress.dl_limit_start) >=
       MIN_RATE_LIMIT_PERIOD) {
      data->progress.dl_limit_start = now;
      data->progress.dl_limit_size = data->progress.downloaded;
    }
  }
  if(data->set.max_send_speed) {
    if(Curl_timediff(now, data->progress.ul_limit_start) >=
       MIN_RATE_LIMIT_PERIOD) {
      data->progress.ul_limit_start = now;
      data->progress.ul_limit_size = data->progress.uploaded;
    }
  }
}

/*
 * Set the number of uploaded bytes so far.
 */
void Curl_pgrsSetUploadCounter(struct Curl_easy *data, curl_off_t size)
{
  data->progress.uploaded = size;
}

void Curl_pgrsSetDownloadSize(struct Curl_easy *data, curl_off_t size)
{
  if(size >= 0) {
    data->progress.size_dl = size;
    data->progress.flags |= PGRS_DL_SIZE_KNOWN;
  }
  else {
    data->progress.size_dl = 0;
    data->progress.flags &= ~PGRS_DL_SIZE_KNOWN;
  }
}

void Curl_pgrsSetUploadSize(struct Curl_easy *data, curl_off_t size)
{
  if(size >= 0) {
    data->progress.size_ul = size;
    data->progress.flags |= PGRS_UL_SIZE_KNOWN;
  }
  else {
    data->progress.size_ul = 0;
    data->progress.flags &= ~PGRS_UL_SIZE_KNOWN;
  }
}

/* returns the average speed in bytes / second */
static curl_off_t trspeed(curl_off_t size, /* number of bytes */
                          curl_off_t us)   /* microseconds */
{
  if(us < 1)
    return size * 1000000;
  else if(size < CURL_OFF_T_MAX/1000000)
    return (size * 1000000) / us;
  else if(us >= 1000000)
    return size / (us / 1000000);
  else
    return CURL_OFF_T_MAX;
}

/* returns TRUE if it's time to show the progress meter */
static bool progress_calc(struct Curl_easy *data, struct curltime now)
{
  bool timetoshow = FALSE;
  struct Progress * const p = &data->progress;

  /* The time spent so far (from the start) in microseconds */
  p->timespent = Curl_timediff_us(now, p->start);
  p->dlspeed = trspeed(p->downloaded, p->timespent);
  p->ulspeed = trspeed(p->uploaded, p->timespent);

  /* Calculations done at most once a second, unless end is reached */
  if(p->lastshow != now.tv_sec) {
    int countindex; /* amount of seconds stored in the speeder array */
    int nowindex = p->speeder_c% CURR_TIME;
    p->lastshow = now.tv_sec;
    timetoshow = TRUE;

    /* Let's do the "current speed" thing, with the dl + ul speeds
       combined. Store the speed at entry 'nowindex'. */
    p->speeder[ nowindex ] = p->downloaded + p->uploaded;

    /* remember the exact time for this moment */
    p->speeder_time [ nowindex ] = now;

    /* advance our speeder_c counter, which is increased every time we get
       here and we expect it to never wrap as 2^32 is a lot of seconds! */
    p->speeder_c++;

    /* figure out how many index entries of data we have stored in our speeder
       array. With N_ENTRIES filled in, we have about N_ENTRIES-1 seconds of
       transfer. Imagine, after one second we have filled in two entries,
       after two seconds we've filled in three entries etc. */
    countindex = ((p->speeder_c >= CURR_TIME)? CURR_TIME:p->speeder_c) - 1;

    /* first of all, we don't do this if there's no counted seconds yet */
    if(countindex) {
      int checkindex;
      timediff_t span_ms;
      curl_off_t amount;

      /* Get the index position to compare with the 'nowindex' position.
         Get the oldest entry possible. While we have less than CURR_TIME
         entries, the first entry will remain the oldest. */
      checkindex = (p->speeder_c >= CURR_TIME)? p->speeder_c%CURR_TIME:0;

      /* Figure out the exact time for the time span */
      span_ms = Curl_timediff(now, p->speeder_time[checkindex]);
      if(0 == span_ms)
        span_ms = 1; /* at least one millisecond MUST have passed */

      /* Calculate the average speed the last 'span_ms' milliseconds */
      amount = p->speeder[nowindex]- p->speeder[checkindex];

      if(amount > CURL_OFF_T_C(4294967) /* 0xffffffff/1000 */)
        /* the 'amount' value is bigger than would fit in 32 bits if
           multiplied with 1000, so we use the double math for this */
        p->current_speed = (curl_off_t)
          ((double)amount/((double)span_ms/1000.0));
      else
        /* the 'amount' value is small enough to fit within 32 bits even
           when multiplied with 1000 */
        p->current_speed = amount*CURL_OFF_T_C(1000)/span_ms;
    }
    else
      /* the first second we use the average */
      p->current_speed = p->ulspeed + p->dlspeed;

  } /* Calculations end */
  return timetoshow;
}

#ifndef CURL_DISABLE_PROGRESS_METER
static void progress_meter(struct Curl_easy *data)
{
  char max5[6][10];
  curl_off_t dlpercen = 0;
  curl_off_t ulpercen = 0;
  curl_off_t total_percen = 0;
  curl_off_t total_transfer;
  curl_off_t total_expected_transfer;
  char time_left[10];
  char time_total[10];
  char time_spent[10];
  curl_off_t ulestimate = 0;
  curl_off_t dlestimate = 0;
  curl_off_t total_estimate;
  curl_off_t timespent =
    (curl_off_t)data->progress.timespent/1000000; /* seconds */

  if(!(data->progress.flags & PGRS_HEADERS_OUT)) {
    if(data->state.resume_from) {
      fprintf(data->set.err,
              "** Resuming transfer from byte position %"
              CURL_FORMAT_CURL_OFF_T "\n", data->state.resume_from);
    }
    fprintf(data->set.err,
            "  %% Total    %% Received %% Xferd  Average Speed   "
            "Time    Time     Time  Current\n"
            "                                 Dload  Upload   "
            "Total   Spent    Left  Speed\n");
    data->progress.flags |= PGRS_HEADERS_OUT; /* headers are shown */
  }

  /* Figure out the estimated time of arrival for the upload */
  if((data->progress.flags & PGRS_UL_SIZE_KNOWN) &&
     (data->progress.ulspeed > CURL_OFF_T_C(0))) {
    ulestimate = data->progress.size_ul / data->progress.ulspeed;

    if(data->progress.size_ul > CURL_OFF_T_C(10000))
      ulpercen = data->progress.uploaded /
        (data->progress.size_ul/CURL_OFF_T_C(100));
    else if(data->progress.size_ul > CURL_OFF_T_C(0))
      ulpercen = (data->progress.uploaded*100) /
        data->progress.size_ul;
  }

  /* ... and the download */
  if((data->progress.flags & PGRS_DL_SIZE_KNOWN) &&
     (data->progress.dlspeed > CURL_OFF_T_C(0))) {
    dlestimate = data->progress.size_dl / data->progress.dlspeed;

    if(data->progress.size_dl > CURL_OFF_T_C(10000))
      dlpercen = data->progress.downloaded /
        (data->progress.size_dl/CURL_OFF_T_C(100));
    else if(data->progress.size_dl > CURL_OFF_T_C(0))
      dlpercen = (data->progress.downloaded*100) /
        data->progress.size_dl;
  }

  /* Now figure out which of them is slower and use that one for the
     total estimate! */
  total_estimate = ulestimate>dlestimate?ulestimate:dlestimate;

  /* create the three time strings */
  time2str(time_left, total_estimate > 0?(total_estimate - timespent):0);
  time2str(time_total, total_estimate);
  time2str(time_spent, timespent);

  /* Get the total amount of data expected to get transferred */
  total_expected_transfer =
    ((data->progress.flags & PGRS_UL_SIZE_KNOWN)?
     data->progress.size_ul:data->progress.uploaded)+
    ((data->progress.flags & PGRS_DL_SIZE_KNOWN)?
     data->progress.size_dl:data->progress.downloaded);

  /* We have transferred this much so far */
  total_transfer = data->progress.downloaded + data->progress.uploaded;

  /* Get the percentage of data transferred so far */
  if(total_expected_transfer > CURL_OFF_T_C(10000))
    total_percen = total_transfer /
      (total_expected_transfer/CURL_OFF_T_C(100));
  else if(total_expected_transfer > CURL_OFF_T_C(0))
    total_percen = (total_transfer*100) / total_expected_transfer;

  fprintf(data->set.err,
          "\r"
          "%3" CURL_FORMAT_CURL_OFF_T " %s  "
          "%3" CURL_FORMAT_CURL_OFF_T " %s  "
          "%3" CURL_FORMAT_CURL_OFF_T " %s  %s  %s %s %s %s %s",
          total_percen,  /* 3 letters */                /* total % */
          max5data(total_expected_transfer, max5[2]),   /* total size */
          dlpercen,      /* 3 letters */                /* rcvd % */
          max5data(data->progress.downloaded, max5[0]), /* rcvd size */
          ulpercen,      /* 3 letters */                /* xfer % */
          max5data(data->progress.uploaded, max5[1]),   /* xfer size */
          max5data(data->progress.dlspeed, max5[3]),    /* avrg dl speed */
          max5data(data->progress.ulspeed, max5[4]),    /* avrg ul speed */
          time_total,    /* 8 letters */                /* total time */
          time_spent,    /* 8 letters */                /* time spent */
          time_left,     /* 8 letters */                /* time left */
          max5data(data->progress.current_speed, max5[5])
    );

  /* we flush the output stream to make it appear as soon as possible */
  fflush(data->set.err);
}
#else
 /* progress bar disabled */
#define progress_meter(x) Curl_nop_stmt
#endif


/*
 * Curl_pgrsUpdate() returns 0 for success or the value returned by the
 * progress callback!
 */
int Curl_pgrsUpdate(struct Curl_easy *data)
{
  struct curltime now = Curl_now(); /* what time is it */
  bool showprogress = progress_calc(data, now);
  if(!(data->progress.flags & PGRS_HIDE)) {
    if(data->set.fxferinfo) {
      int result;
      /* There's a callback set, call that */
      Curl_set_in_callback(data, true);
      result = data->set.fxferinfo(data->set.progress_client,
                                   data->progress.size_dl,
                                   data->progress.downloaded,
                                   data->progress.size_ul,
                                   data->progress.uploaded);
      Curl_set_in_callback(data, false);
      if(result != CURL_PROGRESSFUNC_CONTINUE) {
        if(result)
          failf(data, "Callback aborted");
        return result;
      }
    }
    else if(data->set.fprogress) {
      int result;
      /* The older deprecated callback is set, call that */
      Curl_set_in_callback(data, true);
      result = data->set.fprogress(data->set.progress_client,
                                   (double)data->progress.size_dl,
                                   (double)data->progress.downloaded,
                                   (double)data->progress.size_ul,
                                   (double)data->progress.uploaded);
      Curl_set_in_callback(data, false);
      if(result != CURL_PROGRESSFUNC_CONTINUE) {
        if(result)
          failf(data, "Callback aborted");
        return result;
      }
    }

    if(showprogress)
      progress_meter(data);
  }

  return 0;
}
