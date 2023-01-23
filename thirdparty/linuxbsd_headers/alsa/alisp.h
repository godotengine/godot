/*
 *  ALSA lisp implementation
 *  Copyright (c) 2003 by Jaroslav Kysela <perex@perex.cz>
 *
 *
 *   This library is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as
 *   published by the Free Software Foundation; either version 2.1 of
 *   the License, or (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

struct alisp_cfg {
	int verbose: 1,
	    warning: 1,
	    debug: 1;
	snd_input_t *in;	/* program code */
	snd_output_t *out;	/* program output */
	snd_output_t *eout;	/* error output */
	snd_output_t *vout;	/* verbose output */
	snd_output_t *wout;	/* warning output */
	snd_output_t *dout;	/* debug output */
};

struct alisp_instance;
struct alisp_object;
struct alisp_seq_iterator;

struct alisp_cfg *alsa_lisp_default_cfg(snd_input_t *input);
void alsa_lisp_default_cfg_free(struct alisp_cfg *cfg);
int alsa_lisp(struct alisp_cfg *cfg, struct alisp_instance **instance);
void alsa_lisp_free(struct alisp_instance *instance);
int alsa_lisp_function(struct alisp_instance *instance, struct alisp_seq_iterator **result,
		       const char *id, const char *args, ...)
#ifndef DOC_HIDDEN
		       __attribute__ ((format (printf, 4, 5)))
#endif
		       ;
void alsa_lisp_result_free(struct alisp_instance *instance,
			   struct alisp_seq_iterator *result);
int alsa_lisp_seq_first(struct alisp_instance *instance, const char *id,
			struct alisp_seq_iterator **seq);
int alsa_lisp_seq_next(struct alisp_seq_iterator **seq);
int alsa_lisp_seq_count(struct alisp_seq_iterator *seq);
int alsa_lisp_seq_integer(struct alisp_seq_iterator *seq, long *val);
int alsa_lisp_seq_pointer(struct alisp_seq_iterator *seq, const char *ptr_id, void **ptr);
