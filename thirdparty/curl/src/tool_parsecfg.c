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

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_getparam.h"
#include "tool_helpers.h"
#include "tool_findfile.h"
#include "tool_msgs.h"
#include "tool_parsecfg.h"
#include "dynbuf.h"

#include "memdebug.h" /* keep this as LAST include */

/* only acknowledge colon or equals as separators if the option was not
   specified with an initial dash! */
#define ISSEP(x,dash) (!dash && (((x) == '=') || ((x) == ':')))

static const char *unslashquote(const char *line, char *param);

#define MAX_CONFIG_LINE_LENGTH (100*1024)
static bool my_get_line(FILE *fp, struct curlx_dynbuf *, bool *error);

#ifdef WIN32
static FILE *execpath(const char *filename, char **pathp)
{
  static char filebuffer[512];
  /* Get the filename of our executable. GetModuleFileName is already declared
   * via inclusions done in setup header file.  We assume that we are using
   * the ASCII version here.
   */
  unsigned long len = GetModuleFileNameA(0, filebuffer, sizeof(filebuffer));
  if(len > 0 && len < sizeof(filebuffer)) {
    /* We got a valid filename - get the directory part */
    char *lastdirchar = strrchr(filebuffer, '\\');
    if(lastdirchar) {
      size_t remaining;
      *lastdirchar = 0;
      /* If we have enough space, build the RC filename */
      remaining = sizeof(filebuffer) - strlen(filebuffer);
      if(strlen(filename) < remaining - 1) {
        FILE *f;
        msnprintf(lastdirchar, remaining, "%s%s", DIR_CHAR, filename);
        *pathp = filebuffer;
        f = fopen(filebuffer, FOPEN_READTEXT);
        return f;
      }
    }
  }

  return NULL;
}
#endif


/* return 0 on everything-is-fine, and non-zero otherwise */
int parseconfig(const char *filename, struct GlobalConfig *global)
{
  FILE *file = NULL;
  bool usedarg = FALSE;
  int rc = 0;
  struct OperationConfig *operation = global->last;
  char *pathalloc = NULL;
#ifdef WIN32
#define DOTSCORE TRUE /* look for underscore-prefixed name too */
#else
#define DOTSCORE FALSE
#endif

  if(!filename) {
    /* NULL means load .curlrc from homedir! */
    char *curlrc = findfile(".curlrc", DOTSCORE);
    if(curlrc) {
      file = fopen(curlrc, FOPEN_READTEXT);
      if(!file) {
        free(curlrc);
        return 1;
      }
      filename = pathalloc = curlrc;
    }
#ifdef WIN32 /* Windows */
    else {
      char *fullp;
      /* check for .curlrc then _curlrc in the dir of the executable */
      file = execpath(".curlrc", &fullp);
      if(!file)
        file = execpath("_curlrc", &fullp);
      if(file)
        /* this is the filename we read from */
        filename = fullp;
    }
#endif
  }
  else {
    if(strcmp(filename, "-"))
      file = fopen(filename, FOPEN_READTEXT);
    else
      file = stdin;
  }

  if(file) {
    char *line;
    char *option;
    char *param;
    int lineno = 0;
    bool dashed_option;
    struct curlx_dynbuf buf;
    bool fileerror;
    curlx_dyn_init(&buf, MAX_CONFIG_LINE_LENGTH);
    DEBUGASSERT(filename);

    while(my_get_line(file, &buf, &fileerror)) {
      int res;
      bool alloced_param = FALSE;
      lineno++;
      line = curlx_dyn_ptr(&buf);
      if(!line) {
        rc = 1; /* out of memory */
        break;
      }

      /* line with # in the first non-blank column is a comment! */
      while(*line && ISSPACE(*line))
        line++;

      switch(*line) {
      case '#':
      case '/':
      case '\r':
      case '\n':
      case '*':
      case '\0':
        curlx_dyn_reset(&buf);
        continue;
      }

      /* the option keywords starts here */
      option = line;

      /* the option starts with a dash? */
      dashed_option = option[0]=='-'?TRUE:FALSE;

      while(*line && !ISSPACE(*line) && !ISSEP(*line, dashed_option))
        line++;
      /* ... and has ended here */

      if(*line)
        *line++ = '\0'; /* null-terminate, we have a local copy of the data */

#ifdef DEBUG_CONFIG
      fprintf(stderr, "GOT: %s\n", option);
#endif

      /* pass spaces and separator(s) */
      while(*line && (ISSPACE(*line) || ISSEP(*line, dashed_option)))
        line++;

      /* the parameter starts here (unless quoted) */
      if(*line == '\"') {
        /* quoted parameter, do the quote dance */
        line++;
        param = malloc(strlen(line) + 1); /* parameter */
        if(!param) {
          /* out of memory */
          rc = 1;
          break;
        }
        alloced_param = TRUE;
        (void)unslashquote(line, param);
      }
      else {
        param = line; /* parameter starts here */
        while(*line && !ISSPACE(*line))
          line++;

        if(*line) {
          *line = '\0'; /* null-terminate */

          /* to detect mistakes better, see if there's data following */
          line++;
          /* pass all spaces */
          while(*line && ISSPACE(*line))
            line++;

          switch(*line) {
          case '\0':
          case '\r':
          case '\n':
          case '#': /* comment */
            break;
          default:
            warnf(operation->global, "%s:%d: warning: '%s' uses unquoted "
                  "whitespace in the line that may cause side-effects!\n",
                  filename, lineno, option);
          }
        }
        if(!*param)
          /* do this so getparameter can check for required parameters.
             Otherwise it always thinks there's a parameter. */
          param = NULL;
      }

#ifdef DEBUG_CONFIG
      fprintf(stderr, "PARAM: \"%s\"\n",(param ? param : "(null)"));
#endif
      res = getparameter(option, param, &usedarg, global, operation);
      operation = global->last;

      if(!res && param && *param && !usedarg)
        /* we passed in a parameter that wasn't used! */
        res = PARAM_GOT_EXTRA_PARAMETER;

      if(res == PARAM_NEXT_OPERATION) {
        if(operation->url_list && operation->url_list->url) {
          /* Allocate the next config */
          operation->next = malloc(sizeof(struct OperationConfig));
          if(operation->next) {
            /* Initialise the newly created config */
            config_init(operation->next);

            /* Set the global config pointer */
            operation->next->global = global;

            /* Update the last operation pointer */
            global->last = operation->next;

            /* Move onto the new config */
            operation->next->prev = operation;
            operation = operation->next;
          }
          else
            res = PARAM_NO_MEM;
        }
      }

      if(res != PARAM_OK && res != PARAM_NEXT_OPERATION) {
        /* the help request isn't really an error */
        if(!strcmp(filename, "-")) {
          filename = "<stdin>";
        }
        if(res != PARAM_HELP_REQUESTED &&
           res != PARAM_MANUAL_REQUESTED &&
           res != PARAM_VERSION_INFO_REQUESTED &&
           res != PARAM_ENGINES_REQUESTED) {
          const char *reason = param2text(res);
          warnf(operation->global, "%s:%d: warning: '%s' %s\n",
                filename, lineno, option, reason);
        }
      }

      if(alloced_param)
        Curl_safefree(param);

      curlx_dyn_reset(&buf);
    }
    curlx_dyn_free(&buf);
    if(file != stdin)
      fclose(file);
    if(fileerror)
      rc = 1;
  }
  else
    rc = 1; /* couldn't open the file */

  free(pathalloc);
  return rc;
}

/*
 * Copies the string from line to the buffer at param, unquoting
 * backslash-quoted characters and NUL-terminating the output string.
 * Stops at the first non-backslash-quoted double quote character or the
 * end of the input string. param must be at least as long as the input
 * string.  Returns the pointer after the last handled input character.
 */
static const char *unslashquote(const char *line, char *param)
{
  while(*line && (*line != '\"')) {
    if(*line == '\\') {
      char out;
      line++;

      /* default is to output the letter after the backslash */
      switch(out = *line) {
      case '\0':
        continue; /* this'll break out of the loop */
      case 't':
        out = '\t';
        break;
      case 'n':
        out = '\n';
        break;
      case 'r':
        out = '\r';
        break;
      case 'v':
        out = '\v';
        break;
      }
      *param++ = out;
      line++;
    }
    else
      *param++ = *line++;
  }
  *param = '\0'; /* always null-terminate */
  return line;
}

/*
 * Reads a line from the given file, ensuring is NUL terminated.
 */
static bool my_get_line(FILE *fp, struct curlx_dynbuf *db,
                        bool *error)
{
  char buf[4096];
  *error = FALSE;
  do {
    /* fgets() returns s on success, and NULL on error or when end of file
       occurs while no characters have been read. */
    if(!fgets(buf, sizeof(buf), fp))
      /* only if there's data in the line, return TRUE */
      return curlx_dyn_len(db) ? TRUE : FALSE;
    if(curlx_dyn_add(db, buf)) {
      *error = TRUE; /* error */
      return FALSE; /* stop reading */
    }
  } while(!strchr(buf, '\n'));

  return TRUE; /* continue */
}
