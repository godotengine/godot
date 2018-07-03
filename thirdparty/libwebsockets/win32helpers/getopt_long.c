
/*
 * Copyright (c) 1987, 1993, 1994, 1996
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "getopt.h"

#define lws_ptr_diff(head, tail) \
			((int)((char *)(head) - (char *)(tail)))

extern int	  opterr;	/* if error message should be printed */
extern int	  optind;	/* index into parent argv vector */
extern int	  optopt;	/* character checked for validity */
extern int	  optreset;	/* reset getopt */
extern char *optarg;	/* argument associated with option */

#define __P(x) x
#define _DIAGASSERT(x) assert(x)

static char * __progname __P((char *));
int getopt_internal __P((int, char * const *, const char *));

static char *
__progname(nargv0)
	char * nargv0;
{
	char * tmp;

	_DIAGASSERT(nargv0 != NULL);

	tmp = strrchr(nargv0, '/');
	if (tmp)
		tmp++;
	else
		tmp = nargv0;
	return(tmp);
}

#define	BADCH	(int)'?'
#define	BADARG	(int)':'
#define	EMSG	""

/*
 * getopt --
 *	Parse argc/argv argument vector.
 */
int
getopt_internal(nargc, nargv, ostr)
	int nargc;
	char * const *nargv;
	const char *ostr;
{
	static char *place = EMSG;		/* option letter processing */
	char *oli;				/* option letter list index */

	_DIAGASSERT(nargv != NULL);
	_DIAGASSERT(ostr != NULL);

	if (optreset || !*place) {		/* update scanning pointer */
		optreset = 0;
		if (optind >= nargc || *(place = nargv[optind]) != '-') {
			place = EMSG;
			return (-1);
		}
		if (place[1] && *++place == '-') {	/* found "--" */
			/* ++optind; */
			place = EMSG;
			return (-2);
		}
	}					/* option letter okay? */
	if ((optopt = (int)*place++) == (int)':' ||
	    !(oli = strchr(ostr, optopt))) {
		/*
		 * if the user didn't specify '-' as an option,
		 * assume it means -1.
		 */
		if (optopt == (int)'-')
			return (-1);
		if (!*place)
			++optind;
		if (opterr && *ostr != ':')
			(void)fprintf(stderr,
			    "%s: illegal option -- %c\n", __progname(nargv[0]), optopt);
		return (BADCH);
	}
	if (*++oli != ':') {			/* don't need argument */
		optarg = NULL;
		if (!*place)
			++optind;
	} else {				/* need an argument */
		if (*place)			/* no white space */
			optarg = place;
		else if (nargc <= ++optind) {	/* no arg */
			place = EMSG;
			if ((opterr) && (*ostr != ':'))
				(void)fprintf(stderr,
				    "%s: option requires an argument -- %c\n",
				    __progname(nargv[0]), optopt);
			return (BADARG);
		} else				/* white space */
			optarg = nargv[optind];
		place = EMSG;
		++optind;
	}
	return (optopt);			/* dump back option letter */
}

#if 0
/*
 * getopt --
 *	Parse argc/argv argument vector.
 */
int
getopt2(nargc, nargv, ostr)
	int nargc;
	char * const *nargv;
	const char *ostr;
{
	int retval;

	if ((retval = getopt_internal(nargc, nargv, ostr)) == -2) {
		retval = -1;
		++optind; 
	}
	return(retval);
}
#endif

/*
 * getopt_long --
 *	Parse argc/argv argument vector.
 */
int
getopt_long(nargc, nargv, options, long_options, index)
	int nargc;
	char ** nargv;
	char * options;
	struct option * long_options;
	int * index;
{
	int retval;

	_DIAGASSERT(nargv != NULL);
	_DIAGASSERT(options != NULL);
	_DIAGASSERT(long_options != NULL);
	/* index may be NULL */

	if ((retval = getopt_internal(nargc, nargv, options)) == -2) {
		char *current_argv = nargv[optind++] + 2, *has_equal;
		int i, current_argv_len, match = -1;

		if (*current_argv == '\0') {
			return(-1);
		}
		if ((has_equal = strchr(current_argv, '=')) != NULL) {
			current_argv_len = lws_ptr_diff(has_equal, current_argv);
			has_equal++;
		} else
			current_argv_len = (int)strlen(current_argv);

		for (i = 0; long_options[i].name; i++) { 
			if (strncmp(current_argv, long_options[i].name, current_argv_len))
				continue;

			if (strlen(long_options[i].name) == (unsigned)current_argv_len) { 
				match = i;
				break;
			}
			if (match == -1)
				match = i;
		}
		if (match != -1) {
			if (long_options[match].has_arg == required_argument ||
			    long_options[match].has_arg == optional_argument) {
				if (has_equal)
					optarg = has_equal;
				else
					optarg = nargv[optind++];
			}
			if ((long_options[match].has_arg == required_argument)
			    && (optarg == NULL)) {
				/*
				 * Missing argument, leading :
				 * indicates no error should be generated
				 */
				if ((opterr) && (*options != ':'))
					(void)fprintf(stderr,
				      "%s: option requires an argument -- %s\n",
				      __progname(nargv[0]), current_argv);
				return (BADARG);
			}
		} else { /* No matching argument */
			if ((opterr) && (*options != ':'))
				(void)fprintf(stderr,
				    "%s: illegal option -- %s\n", __progname(nargv[0]), current_argv);
			return (BADCH);
		}
		if (long_options[match].flag) {
			*long_options[match].flag = long_options[match].val;
			retval = 0;
		} else 
			retval = long_options[match].val;
		if (index)
			*index = match;
	}
	return(retval);
}
