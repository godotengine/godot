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
#include "tool_setup.h"

#ifdef __VMS

#if defined(__DECC) && !defined(__VAX) && \
    defined(__CRTL_VER) && (__CRTL_VER >= 70301000)
#include <unixlib.h>
#endif

#define ENABLE_CURLX_PRINTF
#include "curlx.h"

#include "curlmsg_vms.h"
#include "tool_vms.h"

#include "memdebug.h" /* keep this as LAST include */

void decc$__posix_exit(int __status);
void decc$exit(int __status);

static int vms_shell = -1;

/* VMS has a DCL shell and also has Unix shells ported to it.
 * When curl is running under a Unix shell, we want it to be as much
 * like Unix as possible.
 */
int is_vms_shell(void)
{
  char *shell;

  /* Have we checked the shell yet? */
  if(vms_shell >= 0)
    return vms_shell;

  shell = getenv("SHELL");

  /* No shell, means DCL */
  if(!shell) {
    vms_shell = 1;
    return 1;
  }

  /* Have to make sure some one did not set shell to DCL */
  if(strcmp(shell, "DCL") == 0) {
    vms_shell = 1;
    return 1;
  }

  vms_shell = 0;
  return 0;
}

/*
 * VMS has two exit() routines.  When running under a Unix style shell, then
 * Unix style and the __posix_exit() routine is used.
 *
 * When running under the DCL shell, then the VMS encoded codes and decc$exit()
 * is used.
 *
 * We can not use exit() or return a code from main() because the actual
 * routine called depends on both the compiler version, compile options, and
 * feature macro settings, and one of the exit routines is hidden at compile
 * time.
 *
 * Since we want Curl to work properly under the VMS DCL shell and Unix
 * shells under VMS, this routine should compile correctly regardless of
 * the settings.
 */

void vms_special_exit(int code, int vms_show)
{
  int vms_code;

  /* The Posix exit mode is only available after VMS 7.0 */
#if __CRTL_VER >= 70000000
  if(is_vms_shell() == 0) {
    decc$__posix_exit(code);
  }
#endif

  if(code > CURL_LAST) {   /* If CURL_LAST exceeded then */
    vms_code = CURL_LAST;  /* curlmsg.h is out of sync.  */
  }
  else {
    vms_code = vms_cond[code] | vms_show;
  }
  decc$exit(vms_code);
}

#if defined(__DECC) && !defined(__VAX) && \
    defined(__CRTL_VER) && (__CRTL_VER >= 70301000)

/*
 * 2004-09-19 SMS.
 *
 * decc_init()
 *
 * On non-VAX systems, use LIB$INITIALIZE to set a collection of C
 * RTL features without using the DECC$* logical name method, nor
 * requiring the user to define the corresponding logical names.
 */

/* Structure to hold a DECC$* feature name and its desired value. */
struct decc_feat_t {
  char *name;
  int value;
};

/* Array of DECC$* feature names and their desired values. */
static const struct decc_feat_t decc_feat_array[] = {
  /* Preserve command-line case with SET PROCESS/PARSE_STYLE=EXTENDED */
  { "DECC$ARGV_PARSE_STYLE", 1 },
  /* Preserve case for file names on ODS5 disks. */
  { "DECC$EFS_CASE_PRESERVE", 1 },
  /* Enable multiple dots (and most characters) in ODS5 file names,
     while preserving VMS-ness of ";version". */
  { "DECC$EFS_CHARSET", 1 },
  /* List terminator. */
  { (char *)NULL, 0 }
};

/* Flag to sense if decc_init() was called. */
static int decc_init_done = -1;

/* LIB$INITIALIZE initialization function. */
static void decc_init(void)
{
  int feat_index;
  int feat_value;
  int feat_value_max;
  int feat_value_min;
  int i;
  int sts;

  /* Set the global flag to indicate that LIB$INITIALIZE worked. */
  decc_init_done = 1;

  /* Loop through all items in the decc_feat_array[]. */
  for(i = 0; decc_feat_array[i].name != NULL; i++) {

    /* Get the feature index. */
    feat_index = decc$feature_get_index(decc_feat_array[i].name);

    if(feat_index >= 0) {
      /* Valid item.  Collect its properties. */
      feat_value = decc$feature_get_value(feat_index, 1);
      feat_value_min = decc$feature_get_value(feat_index, 2);
      feat_value_max = decc$feature_get_value(feat_index, 3);

      if((decc_feat_array[i].value >= feat_value_min) &&
         (decc_feat_array[i].value <= feat_value_max)) {
        /* Valid value.  Set it if necessary. */
        if(feat_value != decc_feat_array[i].value) {
          sts = decc$feature_set_value(feat_index, 1,
                                       decc_feat_array[i].value);
        }
      }
      else {
        /* Invalid DECC feature value. */
        printf(" INVALID DECC FEATURE VALUE, %d: %d <= %s <= %d.\n",
               feat_value,
               feat_value_min, decc_feat_array[i].name, feat_value_max);
      }
    }
    else {
      /* Invalid DECC feature name. */
      printf(" UNKNOWN DECC FEATURE: %s.\n", decc_feat_array[i].name);
    }

  }
}

/* Get "decc_init()" into a valid, loaded LIB$INITIALIZE PSECT. */

#pragma nostandard

/* Establish the LIB$INITIALIZE PSECTs, with proper alignment and
   other attributes.  Note that "nopic" is significant only on VAX. */
#pragma extern_model save
#pragma extern_model strict_refdef "LIB$INITIALIZ" 2, nopic, nowrt
const int spare[8] = {0};
#pragma extern_model strict_refdef "LIB$INITIALIZE" 2, nopic, nowrt
void (*const x_decc_init)() = decc_init;
#pragma extern_model restore

/* Fake reference to ensure loading the LIB$INITIALIZE PSECT. */
#pragma extern_model save
int LIB$INITIALIZE(void);
#pragma extern_model strict_refdef
int dmy_lib$initialize = (int) LIB$INITIALIZE;
#pragma extern_model restore

#pragma standard

#endif /* __DECC && !__VAX && __CRTL_VER && __CRTL_VER >= 70301000 */

#endif /* __VMS */
