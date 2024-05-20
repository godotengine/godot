/**
 * \file include/output.h
 * \brief Application interface library for the ALSA driver
 * \author Jaroslav Kysela <perex@perex.cz>
 * \author Abramo Bagnara <abramo@alsa-project.org>
 * \author Takashi Iwai <tiwai@suse.de>
 * \date 1998-2001
 *
 * Application interface library for the ALSA driver
 */
/*
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

#ifndef __ALSA_OUTPUT_H
#define __ALSA_OUTPUT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup Output Output Interface
 *
 *  The output functions present an interface similar to the stdio functions
 *  on top of different underlying output destinations.
 *
 *  Many PCM debugging functions (\c snd_pcm_xxx_dump_xxx) use such an output
 *  handle to be able to write not only to the screen but also to other
 *  destinations, e.g. to files or to memory buffers.
 *
 *  \{
 */

/**
 * \brief Internal structure for an output object.
 *
 * The ALSA library uses a pointer to this structure as a handle to an
 * output object. Applications don't access its contents directly.
 */
typedef struct _snd_output snd_output_t;

/** Output type. */
typedef enum _snd_output_type {
	/** Output to a stdio stream. */
	SND_OUTPUT_STDIO,
	/** Output to a memory buffer. */
	SND_OUTPUT_BUFFER
} snd_output_type_t;

int snd_output_stdio_open(snd_output_t **outputp, const char *file, const char *mode);
int snd_output_stdio_attach(snd_output_t **outputp, FILE *fp, int _close);
int snd_output_buffer_open(snd_output_t **outputp);
size_t snd_output_buffer_string(snd_output_t *output, char **buf);
int snd_output_close(snd_output_t *output);
int snd_output_printf(snd_output_t *output, const char *format, ...)
#ifndef DOC_HIDDEN
	__attribute__ ((format (printf, 2, 3)))
#endif
	;
int snd_output_vprintf(snd_output_t *output, const char *format, va_list args);
int snd_output_puts(snd_output_t *output, const char *str);
int snd_output_putc(snd_output_t *output, int c);
int snd_output_flush(snd_output_t *output);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_OUTPUT_H */

