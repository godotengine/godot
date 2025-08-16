/* SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later */
/* Copyright 2010, SIL International, All rights reserved. */
#pragma once

#include <graphite2/Types.h>
#include <graphite2/Font.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** deprecated mechanism that doesn't do anything now. */
typedef enum {
    GRLOG_NONE = 0x0,
    GRLOG_FACE = 0x01,
    GRLOG_SEGMENT = 0x02,
    GRLOG_PASS = 0x04,
    GRLOG_CACHE = 0x08,

    GRLOG_OPCODE = 0x80,
    GRLOG_ALL = 0xFF
} GrLogMask;

/** Start logging all segment creation and updates on the provided face.  This
  * is logged to a JSON file, see "Segment JSON Schema.txt" for a precise
  * definition of the file
  *
  * @return true    if the file was successfully created and logging is correctly
  * 			    initialised.
  * @param face     the gr_face whose segments you want to log to the given file
  * @param log_path a utf8 encoded file name and path to log to.
  */
GR2_API bool gr_start_logging(gr_face * face, const char *log_path);


/** Stop logging on the given face.  This will close the log file created by
  * gr_start_logging.
  *
  * @param face the gr_face whose segments you want to stop logging
  */
GR2_API void gr_stop_logging(gr_face * face);

/** Start logging to a FILE object.
  * This function is deprecated as of 1.2.0, use the _face versions instead.
  *
  * @return        True on success
  * @param logfile FILE reference to output logging to
  * @param mask    What aspects of logging to report (ignored)
  */
GR2_API bool graphite_start_logging(FILE * logFile, GrLogMask mask);    //may not do anthing if disabled in the implementation of the engine.

/** Stop logging to a FILE object.
  * This function is deprecated as of 1.2.0, use the _face versions instead.
  */
GR2_API void graphite_stop_logging();

#ifdef __cplusplus
}
#endif
