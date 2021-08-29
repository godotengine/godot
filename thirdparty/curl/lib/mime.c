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

#include <curl/curl.h>

#include "mime.h"
#include "non-ascii.h"
#include "warnless.h"
#include "urldata.h"
#include "sendf.h"

#if (!defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_MIME)) || \
  !defined(CURL_DISABLE_SMTP) || !defined(CURL_DISABLE_IMAP)

#if defined(HAVE_LIBGEN_H) && defined(HAVE_BASENAME)
#include <libgen.h>
#endif

#include "rand.h"
#include "slist.h"
#include "strcase.h"
#include "dynbuf.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#ifdef WIN32
# ifndef R_OK
#  define R_OK 4
# endif
#endif


#define READ_ERROR                      ((size_t) -1)
#define STOP_FILLING                    ((size_t) -2)

static size_t mime_subparts_read(char *buffer, size_t size, size_t nitems,
                                 void *instream, bool *hasread);

/* Encoders. */
static size_t encoder_nop_read(char *buffer, size_t size, bool ateof,
                                curl_mimepart *part);
static curl_off_t encoder_nop_size(curl_mimepart *part);
static size_t encoder_7bit_read(char *buffer, size_t size, bool ateof,
                                curl_mimepart *part);
static size_t encoder_base64_read(char *buffer, size_t size, bool ateof,
                                curl_mimepart *part);
static curl_off_t encoder_base64_size(curl_mimepart *part);
static size_t encoder_qp_read(char *buffer, size_t size, bool ateof,
                              curl_mimepart *part);
static curl_off_t encoder_qp_size(curl_mimepart *part);

static const struct mime_encoder encoders[] = {
  {"binary", encoder_nop_read, encoder_nop_size},
  {"8bit", encoder_nop_read, encoder_nop_size},
  {"7bit", encoder_7bit_read, encoder_nop_size},
  {"base64", encoder_base64_read, encoder_base64_size},
  {"quoted-printable", encoder_qp_read, encoder_qp_size},
  {ZERO_NULL, ZERO_NULL, ZERO_NULL}
};

/* Base64 encoding table */
static const char base64[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* Quoted-printable character class table.
 *
 * We cannot rely on ctype functions since quoted-printable input data
 * is assumed to be ascii-compatible, even on non-ascii platforms. */
#define QP_OK           1       /* Can be represented by itself. */
#define QP_SP           2       /* Space or tab. */
#define QP_CR           3       /* Carriage return. */
#define QP_LF           4       /* Line-feed. */
static const unsigned char qp_class[] = {
 0,     0,     0,     0,     0,     0,     0,     0,            /* 00 - 07 */
 0,     QP_SP, QP_LF, 0,     0,     QP_CR, 0,     0,            /* 08 - 0F */
 0,     0,     0,     0,     0,     0,     0,     0,            /* 10 - 17 */
 0,     0,     0,     0,     0,     0,     0,     0,            /* 18 - 1F */
 QP_SP, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 20 - 27 */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 28 - 2F */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 30 - 37 */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, 0    , QP_OK, QP_OK,        /* 38 - 3F */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 40 - 47 */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 48 - 4F */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 50 - 57 */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 58 - 5F */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 60 - 67 */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 68 - 6F */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK,        /* 70 - 77 */
 QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, QP_OK, 0,            /* 78 - 7F */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* 80 - 8F */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* 90 - 9F */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* A0 - AF */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* B0 - BF */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* C0 - CF */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* D0 - DF */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                /* E0 - EF */
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0                 /* F0 - FF */
};


/* Binary --> hexadecimal ASCII table. */
static const char aschex[] =
  "\x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x41\x42\x43\x44\x45\x46";



#ifndef __VMS
#define filesize(name, stat_data) (stat_data.st_size)
#define fopen_read fopen

#else

#include <fabdef.h>
/*
 * get_vms_file_size does what it takes to get the real size of the file
 *
 * For fixed files, find out the size of the EOF block and adjust.
 *
 * For all others, have to read the entire file in, discarding the contents.
 * Most posted text files will be small, and binary files like zlib archives
 * and CD/DVD images should be either a STREAM_LF format or a fixed format.
 *
 */
curl_off_t VmsRealFileSize(const char *name,
                           const struct_stat *stat_buf)
{
  char buffer[8192];
  curl_off_t count;
  int ret_stat;
  FILE * file;

  file = fopen(name, FOPEN_READTEXT); /* VMS */
  if(!file)
    return 0;

  count = 0;
  ret_stat = 1;
  while(ret_stat > 0) {
    ret_stat = fread(buffer, 1, sizeof(buffer), file);
    if(ret_stat)
      count += ret_stat;
  }
  fclose(file);

  return count;
}

/*
 *
 *  VmsSpecialSize checks to see if the stat st_size can be trusted and
 *  if not to call a routine to get the correct size.
 *
 */
static curl_off_t VmsSpecialSize(const char *name,
                                 const struct_stat *stat_buf)
{
  switch(stat_buf->st_fab_rfm) {
  case FAB$C_VAR:
  case FAB$C_VFC:
    return VmsRealFileSize(name, stat_buf);
    break;
  default:
    return stat_buf->st_size;
  }
}

#define filesize(name, stat_data) VmsSpecialSize(name, &stat_data)

/*
 * vmsfopenread
 *
 * For upload to work as expected on VMS, different optional
 * parameters must be added to the fopen command based on
 * record format of the file.
 *
 */
static FILE * vmsfopenread(const char *file, const char *mode)
{
  struct_stat statbuf;
  int result;

  result = stat(file, &statbuf);

  switch(statbuf.st_fab_rfm) {
  case FAB$C_VAR:
  case FAB$C_VFC:
  case FAB$C_STMCR:
    return fopen(file, FOPEN_READTEXT); /* VMS */
    break;
  default:
    return fopen(file, FOPEN_READTEXT, "rfm=stmlf", "ctx=stm");
  }
}

#define fopen_read vmsfopenread
#endif


#ifndef HAVE_BASENAME
/*
  (Quote from The Open Group Base Specifications Issue 6 IEEE Std 1003.1, 2004
  Edition)

  The basename() function shall take the pathname pointed to by path and
  return a pointer to the final component of the pathname, deleting any
  trailing '/' characters.

  If the string pointed to by path consists entirely of the '/' character,
  basename() shall return a pointer to the string "/". If the string pointed
  to by path is exactly "//", it is implementation-defined whether '/' or "//"
  is returned.

  If path is a null pointer or points to an empty string, basename() shall
  return a pointer to the string ".".

  The basename() function may modify the string pointed to by path, and may
  return a pointer to static storage that may then be overwritten by a
  subsequent call to basename().

  The basename() function need not be reentrant. A function that is not
  required to be reentrant is not required to be thread-safe.

*/
static char *Curl_basename(char *path)
{
  /* Ignore all the details above for now and make a quick and simple
     implementation here */
  char *s1;
  char *s2;

  s1 = strrchr(path, '/');
  s2 = strrchr(path, '\\');

  if(s1 && s2) {
    path = (s1 > s2? s1 : s2) + 1;
  }
  else if(s1)
    path = s1 + 1;
  else if(s2)
    path = s2 + 1;

  return path;
}

#define basename(x)  Curl_basename((x))
#endif


/* Set readback state. */
static void mimesetstate(struct mime_state *state,
                         enum mimestate tok, void *ptr)
{
  state->state = tok;
  state->ptr = ptr;
  state->offset = 0;
}


/* Escape header string into allocated memory. */
static char *escape_string(struct Curl_easy *data,
                           const char *src, enum mimestrategy strategy)
{
  CURLcode result;
  struct dynbuf db;
  const char * const *table;
  const char * const *p;
  /* replace first character by rest of string. */
  static const char * const mimetable[] = {
    "\\\\\\",
    "\"\\\"",
    NULL
  };
  /* WHATWG HTML living standard 4.10.21.8 2 specifies:
     For field names and filenames for file fields, the result of the
     encoding in the previous bullet point must be escaped by replacing
     any 0x0A (LF) bytes with the byte sequence `%0A`, 0x0D (CR) with `%0D`
     and 0x22 (") with `%22`.
     The user agent must not perform any other escapes. */
  static const char * const formtable[] = {
    "\"%22",
    "\r%0D",
    "\n%0A",
    NULL
  };

  table = formtable;
  /* data can be NULL when this function is called indirectly from
     curl_formget(). */
  if(strategy == MIMESTRATEGY_MAIL ||
     (data && (data->set.mime_options & CURLMIMEOPT_FORMESCAPE)))
    table = mimetable;

  Curl_dyn_init(&db, CURL_MAX_INPUT_LENGTH);

  for(result = Curl_dyn_add(&db, ""); !result && *src; src++) {
    for(p = table; *p && **p != *src; p++)
      ;

    if(*p)
      result = Curl_dyn_add(&db, *p + 1);
    else
      result = Curl_dyn_addn(&db, src, 1);
  }

  return Curl_dyn_ptr(&db);
}

/* Check if header matches. */
static char *match_header(struct curl_slist *hdr, const char *lbl, size_t len)
{
  char *value = NULL;

  if(strncasecompare(hdr->data, lbl, len) && hdr->data[len] == ':')
    for(value = hdr->data + len + 1; *value == ' '; value++)
      ;
  return value;
}

/* Get a header from an slist. */
static char *search_header(struct curl_slist *hdrlist, const char *hdr)
{
  size_t len = strlen(hdr);
  char *value = NULL;

  for(; !value && hdrlist; hdrlist = hdrlist->next)
    value = match_header(hdrlist, hdr, len);

  return value;
}

static char *strippath(const char *fullfile)
{
  char *filename;
  char *base;
  filename = strdup(fullfile); /* duplicate since basename() may ruin the
                                  buffer it works on */
  if(!filename)
    return NULL;
  base = strdup(basename(filename));

  free(filename); /* free temporary buffer */

  return base; /* returns an allocated string or NULL ! */
}

/* Initialize data encoder state. */
static void cleanup_encoder_state(struct mime_encoder_state *p)
{
  p->pos = 0;
  p->bufbeg = 0;
  p->bufend = 0;
}


/* Dummy encoder. This is used for 8bit and binary content encodings. */
static size_t encoder_nop_read(char *buffer, size_t size, bool ateof,
                               struct curl_mimepart *part)
{
  struct mime_encoder_state *st = &part->encstate;
  size_t insize = st->bufend - st->bufbeg;

  (void) ateof;

  if(!size)
    return STOP_FILLING;

  if(size > insize)
    size = insize;

  if(size)
    memcpy(buffer, st->buf + st->bufbeg, size);

  st->bufbeg += size;
  return size;
}

static curl_off_t encoder_nop_size(curl_mimepart *part)
{
  return part->datasize;
}


/* 7bit encoder: the encoder is just a data validity check. */
static size_t encoder_7bit_read(char *buffer, size_t size, bool ateof,
                                curl_mimepart *part)
{
  struct mime_encoder_state *st = &part->encstate;
  size_t cursize = st->bufend - st->bufbeg;

  (void) ateof;

  if(!size)
    return STOP_FILLING;

  if(size > cursize)
    size = cursize;

  for(cursize = 0; cursize < size; cursize++) {
    *buffer = st->buf[st->bufbeg];
    if(*buffer++ & 0x80)
      return cursize? cursize: READ_ERROR;
    st->bufbeg++;
  }

  return cursize;
}


/* Base64 content encoder. */
static size_t encoder_base64_read(char *buffer, size_t size, bool ateof,
                                curl_mimepart *part)
{
  struct mime_encoder_state *st = &part->encstate;
  size_t cursize = 0;
  int i;
  char *ptr = buffer;

  while(st->bufbeg < st->bufend) {
    /* Line full ? */
    if(st->pos > MAX_ENCODED_LINE_LENGTH - 4) {
      /* Yes, we need 2 characters for CRLF. */
      if(size < 2) {
        if(!cursize)
          return STOP_FILLING;
        break;
      }
      *ptr++ = '\r';
      *ptr++ = '\n';
      st->pos = 0;
      cursize += 2;
      size -= 2;
    }

    /* Be sure there is enough space and input data for a base64 group. */
    if(size < 4) {
      if(!cursize)
        return STOP_FILLING;
      break;
    }
    if(st->bufend - st->bufbeg < 3)
      break;

    /* Encode three bytes as four characters. */
    i = st->buf[st->bufbeg++] & 0xFF;
    i = (i << 8) | (st->buf[st->bufbeg++] & 0xFF);
    i = (i << 8) | (st->buf[st->bufbeg++] & 0xFF);
    *ptr++ = base64[(i >> 18) & 0x3F];
    *ptr++ = base64[(i >> 12) & 0x3F];
    *ptr++ = base64[(i >> 6) & 0x3F];
    *ptr++ = base64[i & 0x3F];
    cursize += 4;
    st->pos += 4;
    size -= 4;
  }

  /* If at eof, we have to flush the buffered data. */
  if(ateof) {
    if(size < 4) {
      if(!cursize)
        return STOP_FILLING;
    }
    else {
      /* Buffered data size can only be 0, 1 or 2. */
      ptr[2] = ptr[3] = '=';
      i = 0;

      /* If there is buffered data */
      if(st->bufend != st->bufbeg) {

        if(st->bufend - st->bufbeg == 2)
          i = (st->buf[st->bufbeg + 1] & 0xFF) << 8;

        i |= (st->buf[st->bufbeg] & 0xFF) << 16;
        ptr[0] = base64[(i >> 18) & 0x3F];
        ptr[1] = base64[(i >> 12) & 0x3F];
        if(++st->bufbeg != st->bufend) {
          ptr[2] = base64[(i >> 6) & 0x3F];
          st->bufbeg++;
        }
        cursize += 4;
        st->pos += 4;
      }
    }
  }

#ifdef CURL_DOES_CONVERSIONS
  /* This is now textual data, Convert character codes. */
  if(part->easy && cursize) {
    CURLcode result = Curl_convert_to_network(part->easy, buffer, cursize);
    if(result)
      return READ_ERROR;
  }
#endif

  return cursize;
}

static curl_off_t encoder_base64_size(curl_mimepart *part)
{
  curl_off_t size = part->datasize;

  if(size <= 0)
    return size;    /* Unknown size or no data. */

  /* Compute base64 character count. */
  size = 4 * (1 + (size - 1) / 3);

  /* Effective character count must include CRLFs. */
  return size + 2 * ((size - 1) / MAX_ENCODED_LINE_LENGTH);
}


/* Quoted-printable lookahead.
 *
 * Check if a CRLF or end of data is in input buffer at current position + n.
 * Return -1 if more data needed, 1 if CRLF or end of data, else 0.
 */
static int qp_lookahead_eol(struct mime_encoder_state *st, int ateof, size_t n)
{
  n += st->bufbeg;
  if(n >= st->bufend && ateof)
    return 1;
  if(n + 2 > st->bufend)
    return ateof? 0: -1;
  if(qp_class[st->buf[n] & 0xFF] == QP_CR &&
     qp_class[st->buf[n + 1] & 0xFF] == QP_LF)
    return 1;
  return 0;
}

/* Quoted-printable encoder. */
static size_t encoder_qp_read(char *buffer, size_t size, bool ateof,
                              curl_mimepart *part)
{
  struct mime_encoder_state *st = &part->encstate;
  char *ptr = buffer;
  size_t cursize = 0;
  int softlinebreak;
  char buf[4];

  /* On all platforms, input is supposed to be ASCII compatible: for this
     reason, we use hexadecimal ASCII codes in this function rather than
     character constants that can be interpreted as non-ascii on some
     platforms. Preserve ASCII encoding on output too. */
  while(st->bufbeg < st->bufend) {
    size_t len = 1;
    size_t consumed = 1;
    int i = st->buf[st->bufbeg];
    buf[0] = (char) i;
    buf[1] = aschex[(i >> 4) & 0xF];
    buf[2] = aschex[i & 0xF];

    switch(qp_class[st->buf[st->bufbeg] & 0xFF]) {
    case QP_OK:          /* Not a special character. */
      break;
    case QP_SP:          /* Space or tab. */
      /* Spacing must be escaped if followed by CRLF. */
      switch(qp_lookahead_eol(st, ateof, 1)) {
      case -1:          /* More input data needed. */
        return cursize;
      case 0:           /* No encoding needed. */
        break;
      default:          /* CRLF after space or tab. */
        buf[0] = '\x3D';    /* '=' */
        len = 3;
        break;
      }
      break;
    case QP_CR:         /* Carriage return. */
      /* If followed by a line-feed, output the CRLF pair.
         Else escape it. */
      switch(qp_lookahead_eol(st, ateof, 0)) {
      case -1:          /* Need more data. */
        return cursize;
      case 1:           /* CRLF found. */
        buf[len++] = '\x0A';    /* Append '\n'. */
        consumed = 2;
        break;
      default:          /* Not followed by LF: escape. */
        buf[0] = '\x3D';    /* '=' */
        len = 3;
        break;
      }
      break;
    default:            /* Character must be escaped. */
      buf[0] = '\x3D';    /* '=' */
      len = 3;
      break;
    }

    /* Be sure the encoded character fits within maximum line length. */
    if(buf[len - 1] != '\x0A') {    /* '\n' */
      softlinebreak = st->pos + len > MAX_ENCODED_LINE_LENGTH;
      if(!softlinebreak && st->pos + len == MAX_ENCODED_LINE_LENGTH) {
        /* We may use the current line only if end of data or followed by
           a CRLF. */
        switch(qp_lookahead_eol(st, ateof, consumed)) {
        case -1:        /* Need more data. */
          return cursize;
        case 0:         /* Not followed by a CRLF. */
          softlinebreak = 1;
          break;
        }
      }
      if(softlinebreak) {
        strcpy(buf, "\x3D\x0D\x0A");    /* "=\r\n" */
        len = 3;
        consumed = 0;
      }
    }

    /* If the output buffer would overflow, do not store. */
    if(len > size) {
      if(!cursize)
        return STOP_FILLING;
      break;
    }

    /* Append to output buffer. */
    memcpy(ptr, buf, len);
    cursize += len;
    ptr += len;
    size -= len;
    st->pos += len;
    if(buf[len - 1] == '\x0A')    /* '\n' */
      st->pos = 0;
    st->bufbeg += consumed;
  }

  return cursize;
}

static curl_off_t encoder_qp_size(curl_mimepart *part)
{
  /* Determining the size can only be done by reading the data: unless the
     data size is 0, we return it as unknown (-1). */
  return part->datasize? -1: 0;
}


/* In-memory data callbacks. */
/* Argument is a pointer to the mime part. */
static size_t mime_mem_read(char *buffer, size_t size, size_t nitems,
                            void *instream)
{
  curl_mimepart *part = (curl_mimepart *) instream;
  size_t sz = curlx_sotouz(part->datasize - part->state.offset);
  (void) size;   /* Always 1.*/

  if(!nitems)
    return STOP_FILLING;

  if(sz > nitems)
    sz = nitems;

  if(sz)
    memcpy(buffer, part->data + curlx_sotouz(part->state.offset), sz);

  return sz;
}

static int mime_mem_seek(void *instream, curl_off_t offset, int whence)
{
  curl_mimepart *part = (curl_mimepart *) instream;

  switch(whence) {
  case SEEK_CUR:
    offset += part->state.offset;
    break;
  case SEEK_END:
    offset += part->datasize;
    break;
  }

  if(offset < 0 || offset > part->datasize)
    return CURL_SEEKFUNC_FAIL;

  part->state.offset = offset;
  return CURL_SEEKFUNC_OK;
}

static void mime_mem_free(void *ptr)
{
  Curl_safefree(((curl_mimepart *) ptr)->data);
}


/* Named file callbacks. */
/* Argument is a pointer to the mime part. */
static int mime_open_file(curl_mimepart *part)
{
  /* Open a MIMEKIND_FILE part. */

  if(part->fp)
    return 0;
  part->fp = fopen_read(part->data, "rb");
  return part->fp? 0: -1;
}

static size_t mime_file_read(char *buffer, size_t size, size_t nitems,
                             void *instream)
{
  curl_mimepart *part = (curl_mimepart *) instream;

  if(!nitems)
    return STOP_FILLING;

  if(mime_open_file(part))
    return READ_ERROR;

  return fread(buffer, size, nitems, part->fp);
}

static int mime_file_seek(void *instream, curl_off_t offset, int whence)
{
  curl_mimepart *part = (curl_mimepart *) instream;

  if(whence == SEEK_SET && !offset && !part->fp)
    return CURL_SEEKFUNC_OK;   /* Not open: implicitly already at BOF. */

  if(mime_open_file(part))
    return CURL_SEEKFUNC_FAIL;

  return fseek(part->fp, (long) offset, whence)?
               CURL_SEEKFUNC_CANTSEEK: CURL_SEEKFUNC_OK;
}

static void mime_file_free(void *ptr)
{
  curl_mimepart *part = (curl_mimepart *) ptr;

  if(part->fp) {
    fclose(part->fp);
    part->fp = NULL;
  }
  Curl_safefree(part->data);
  part->data = NULL;
}


/* Subparts callbacks. */
/* Argument is a pointer to the mime structure. */

/* Readback a byte string segment. */
static size_t readback_bytes(struct mime_state *state,
                             char *buffer, size_t bufsize,
                             const char *bytes, size_t numbytes,
                             const char *trail)
{
  size_t sz;
  size_t offset = curlx_sotouz(state->offset);

  if(numbytes > offset) {
    sz = numbytes - offset;
    bytes += offset;
  }
  else {
    size_t tsz = strlen(trail);

    sz = offset - numbytes;
    if(sz >= tsz)
      return 0;
    bytes = trail + sz;
    sz = tsz - sz;
  }

  if(sz > bufsize)
    sz = bufsize;

  memcpy(buffer, bytes, sz);
  state->offset += sz;
  return sz;
}

/* Read a non-encoded part content. */
static size_t read_part_content(curl_mimepart *part,
                                char *buffer, size_t bufsize, bool *hasread)
{
  size_t sz = 0;

  switch(part->lastreadstatus) {
  case 0:
  case CURL_READFUNC_ABORT:
  case CURL_READFUNC_PAUSE:
  case READ_ERROR:
    return part->lastreadstatus;
  default:
    break;
  }

  /* If we can determine we are at end of part data, spare a read. */
  if(part->datasize != (curl_off_t) -1 &&
     part->state.offset >= part->datasize) {
    /* sz is already zero. */
  }
  else {
    switch(part->kind) {
    case MIMEKIND_MULTIPART:
      /*
       * Cannot be processed as other kinds since read function requires
       * an additional parameter and is highly recursive.
       */
       sz = mime_subparts_read(buffer, 1, bufsize, part->arg, hasread);
       break;
    case MIMEKIND_FILE:
      if(part->fp && feof(part->fp))
        break;  /* At EOF. */
      /* FALLTHROUGH */
    default:
      if(part->readfunc) {
        if(!(part->flags & MIME_FAST_READ)) {
          if(*hasread)
            return STOP_FILLING;
          *hasread = TRUE;
        }
        sz = part->readfunc(buffer, 1, bufsize, part->arg);
      }
      break;
    }
  }

  switch(sz) {
  case STOP_FILLING:
    break;
  case 0:
  case CURL_READFUNC_ABORT:
  case CURL_READFUNC_PAUSE:
  case READ_ERROR:
    part->lastreadstatus = sz;
    break;
  default:
    part->state.offset += sz;
    part->lastreadstatus = sz;
    break;
  }

  return sz;
}

/* Read and encode part content. */
static size_t read_encoded_part_content(curl_mimepart *part, char *buffer,
                                        size_t bufsize, bool *hasread)
{
  struct mime_encoder_state *st = &part->encstate;
  size_t cursize = 0;
  size_t sz;
  bool ateof = FALSE;

  for(;;) {
    if(st->bufbeg < st->bufend || ateof) {
      /* Encode buffered data. */
      sz = part->encoder->encodefunc(buffer, bufsize, ateof, part);
      switch(sz) {
      case 0:
        if(ateof)
          return cursize;
        break;
      case READ_ERROR:
      case STOP_FILLING:
        return cursize? cursize: sz;
      default:
        cursize += sz;
        buffer += sz;
        bufsize -= sz;
        continue;
      }
    }

    /* We need more data in input buffer. */
    if(st->bufbeg) {
      size_t len = st->bufend - st->bufbeg;

      if(len)
        memmove(st->buf, st->buf + st->bufbeg, len);
      st->bufbeg = 0;
      st->bufend = len;
    }
    if(st->bufend >= sizeof(st->buf))
      return cursize? cursize: READ_ERROR;    /* Buffer full. */
    sz = read_part_content(part, st->buf + st->bufend,
                           sizeof(st->buf) - st->bufend, hasread);
    switch(sz) {
    case 0:
      ateof = TRUE;
      break;
    case CURL_READFUNC_ABORT:
    case CURL_READFUNC_PAUSE:
    case READ_ERROR:
    case STOP_FILLING:
      return cursize? cursize: sz;
    default:
      st->bufend += sz;
      break;
    }
  }

  /* NOTREACHED */
}

/* Readback a mime part. */
static size_t readback_part(curl_mimepart *part,
                            char *buffer, size_t bufsize, bool *hasread)
{
  size_t cursize = 0;
#ifdef CURL_DOES_CONVERSIONS
  char *convbuf = buffer;
#endif

  /* Readback from part. */

  while(bufsize) {
    size_t sz = 0;
    struct curl_slist *hdr = (struct curl_slist *) part->state.ptr;
    switch(part->state.state) {
    case MIMESTATE_BEGIN:
      mimesetstate(&part->state,
                   (part->flags & MIME_BODY_ONLY)?
                     MIMESTATE_BODY: MIMESTATE_CURLHEADERS,
                   part->curlheaders);
      break;
    case MIMESTATE_USERHEADERS:
      if(!hdr) {
        mimesetstate(&part->state, MIMESTATE_EOH, NULL);
        break;
      }
      if(match_header(hdr, "Content-Type", 12)) {
        mimesetstate(&part->state, MIMESTATE_USERHEADERS, hdr->next);
        break;
      }
      /* FALLTHROUGH */
    case MIMESTATE_CURLHEADERS:
      if(!hdr)
        mimesetstate(&part->state, MIMESTATE_USERHEADERS, part->userheaders);
      else {
        sz = readback_bytes(&part->state, buffer, bufsize,
                            hdr->data, strlen(hdr->data), "\r\n");
        if(!sz)
          mimesetstate(&part->state, part->state.state, hdr->next);
      }
      break;
    case MIMESTATE_EOH:
      sz = readback_bytes(&part->state, buffer, bufsize, "\r\n", 2, "");
      if(!sz)
        mimesetstate(&part->state, MIMESTATE_BODY, NULL);
      break;
    case MIMESTATE_BODY:
#ifdef CURL_DOES_CONVERSIONS
      if(part->easy && convbuf < buffer) {
        CURLcode result = Curl_convert_to_network(part->easy, convbuf,
                                                  buffer - convbuf);
        if(result)
          return READ_ERROR;
        convbuf = buffer;
      }
#endif
      cleanup_encoder_state(&part->encstate);
      mimesetstate(&part->state, MIMESTATE_CONTENT, NULL);
      break;
    case MIMESTATE_CONTENT:
      if(part->encoder)
        sz = read_encoded_part_content(part, buffer, bufsize, hasread);
      else
        sz = read_part_content(part, buffer, bufsize, hasread);
      switch(sz) {
      case 0:
        mimesetstate(&part->state, MIMESTATE_END, NULL);
        /* Try sparing open file descriptors. */
        if(part->kind == MIMEKIND_FILE && part->fp) {
          fclose(part->fp);
          part->fp = NULL;
        }
        /* FALLTHROUGH */
      case CURL_READFUNC_ABORT:
      case CURL_READFUNC_PAUSE:
      case READ_ERROR:
      case STOP_FILLING:
        return cursize? cursize: sz;
      }
      break;
    case MIMESTATE_END:
      return cursize;
    default:
      break;    /* Other values not in part state. */
    }

    /* Bump buffer and counters according to read size. */
    cursize += sz;
    buffer += sz;
    bufsize -= sz;
  }

#ifdef CURL_DOES_CONVERSIONS
      if(part->easy && convbuf < buffer &&
         part->state.state < MIMESTATE_BODY) {
        CURLcode result = Curl_convert_to_network(part->easy, convbuf,
                                                  buffer - convbuf);
        if(result)
          return READ_ERROR;
      }
#endif

  return cursize;
}

/* Readback from mime. Warning: not a read callback function. */
static size_t mime_subparts_read(char *buffer, size_t size, size_t nitems,
                                 void *instream, bool *hasread)
{
  curl_mime *mime = (curl_mime *) instream;
  size_t cursize = 0;
#ifdef CURL_DOES_CONVERSIONS
  char *convbuf = buffer;
#endif

  (void) size;   /* Always 1. */

  while(nitems) {
    size_t sz = 0;
    curl_mimepart *part = mime->state.ptr;
    switch(mime->state.state) {
    case MIMESTATE_BEGIN:
    case MIMESTATE_BODY:
#ifdef CURL_DOES_CONVERSIONS
      convbuf = buffer;
#endif
      mimesetstate(&mime->state, MIMESTATE_BOUNDARY1, mime->firstpart);
      /* The first boundary always follows the header termination empty line,
         so is always preceded by a CRLF. We can then spare 2 characters
         by skipping the leading CRLF in boundary. */
      mime->state.offset += 2;
      break;
    case MIMESTATE_BOUNDARY1:
      sz = readback_bytes(&mime->state, buffer, nitems, "\r\n--", 4, "");
      if(!sz)
        mimesetstate(&mime->state, MIMESTATE_BOUNDARY2, part);
      break;
    case MIMESTATE_BOUNDARY2:
      sz = readback_bytes(&mime->state, buffer, nitems, mime->boundary,
                          strlen(mime->boundary), part? "\r\n": "--\r\n");
      if(!sz) {
#ifdef CURL_DOES_CONVERSIONS
        if(mime->easy && convbuf < buffer) {
          CURLcode result = Curl_convert_to_network(mime->easy, convbuf,
                                                    buffer - convbuf);
          if(result)
            return READ_ERROR;
          convbuf = buffer;
        }
#endif
        mimesetstate(&mime->state, MIMESTATE_CONTENT, part);
      }
      break;
    case MIMESTATE_CONTENT:
      if(!part) {
        mimesetstate(&mime->state, MIMESTATE_END, NULL);
        break;
      }
      sz = readback_part(part, buffer, nitems, hasread);
      switch(sz) {
      case CURL_READFUNC_ABORT:
      case CURL_READFUNC_PAUSE:
      case READ_ERROR:
      case STOP_FILLING:
        return cursize? cursize: sz;
      case 0:
#ifdef CURL_DOES_CONVERSIONS
        convbuf = buffer;
#endif
        mimesetstate(&mime->state, MIMESTATE_BOUNDARY1, part->nextpart);
        break;
      }
      break;
    case MIMESTATE_END:
      return cursize;
    default:
      break;    /* other values not used in mime state. */
    }

    /* Bump buffer and counters according to read size. */
    cursize += sz;
    buffer += sz;
    nitems -= sz;
  }

#ifdef CURL_DOES_CONVERSIONS
      if(mime->easy && convbuf < buffer &&
         mime->state.state <= MIMESTATE_CONTENT) {
        CURLcode result = Curl_convert_to_network(mime->easy, convbuf,
                                                  buffer - convbuf);
        if(result)
          return READ_ERROR;
      }
#endif

  return cursize;
}

static int mime_part_rewind(curl_mimepart *part)
{
  int res = CURL_SEEKFUNC_OK;
  enum mimestate targetstate = MIMESTATE_BEGIN;

  if(part->flags & MIME_BODY_ONLY)
    targetstate = MIMESTATE_BODY;
  cleanup_encoder_state(&part->encstate);
  if(part->state.state > targetstate) {
    res = CURL_SEEKFUNC_CANTSEEK;
    if(part->seekfunc) {
      res = part->seekfunc(part->arg, (curl_off_t) 0, SEEK_SET);
      switch(res) {
      case CURL_SEEKFUNC_OK:
      case CURL_SEEKFUNC_FAIL:
      case CURL_SEEKFUNC_CANTSEEK:
        break;
      case -1:    /* For fseek() error. */
        res = CURL_SEEKFUNC_CANTSEEK;
        break;
      default:
        res = CURL_SEEKFUNC_FAIL;
        break;
      }
    }
  }

  if(res == CURL_SEEKFUNC_OK)
    mimesetstate(&part->state, targetstate, NULL);

  part->lastreadstatus = 1; /* Successful read status. */
  return res;
}

static int mime_subparts_seek(void *instream, curl_off_t offset, int whence)
{
  curl_mime *mime = (curl_mime *) instream;
  curl_mimepart *part;
  int result = CURL_SEEKFUNC_OK;

  if(whence != SEEK_SET || offset)
    return CURL_SEEKFUNC_CANTSEEK;    /* Only support full rewind. */

  if(mime->state.state == MIMESTATE_BEGIN)
   return CURL_SEEKFUNC_OK;           /* Already rewound. */

  for(part = mime->firstpart; part; part = part->nextpart) {
    int res = mime_part_rewind(part);
    if(res != CURL_SEEKFUNC_OK)
      result = res;
  }

  if(result == CURL_SEEKFUNC_OK)
    mimesetstate(&mime->state, MIMESTATE_BEGIN, NULL);

  return result;
}

/* Release part content. */
static void cleanup_part_content(curl_mimepart *part)
{
  if(part->freefunc)
    part->freefunc(part->arg);

  part->readfunc = NULL;
  part->seekfunc = NULL;
  part->freefunc = NULL;
  part->arg = (void *) part;          /* Defaults to part itself. */
  part->data = NULL;
  part->fp = NULL;
  part->datasize = (curl_off_t) 0;    /* No size yet. */
  cleanup_encoder_state(&part->encstate);
  part->kind = MIMEKIND_NONE;
  part->flags &= ~MIME_FAST_READ;
  part->lastreadstatus = 1; /* Successful read status. */
  part->state.state = MIMESTATE_BEGIN;
}

static void mime_subparts_free(void *ptr)
{
  curl_mime *mime = (curl_mime *) ptr;

  if(mime && mime->parent) {
    mime->parent->freefunc = NULL;  /* Be sure we won't be called again. */
    cleanup_part_content(mime->parent);  /* Avoid dangling pointer in part. */
  }
  curl_mime_free(mime);
}

/* Do not free subparts: unbind them. This is used for the top level only. */
static void mime_subparts_unbind(void *ptr)
{
  curl_mime *mime = (curl_mime *) ptr;

  if(mime && mime->parent) {
    mime->parent->freefunc = NULL;  /* Be sure we won't be called again. */
    cleanup_part_content(mime->parent);  /* Avoid dangling pointer in part. */
    mime->parent = NULL;
  }
}


void Curl_mime_cleanpart(curl_mimepart *part)
{
  cleanup_part_content(part);
  curl_slist_free_all(part->curlheaders);
  if(part->flags & MIME_USERHEADERS_OWNER)
    curl_slist_free_all(part->userheaders);
  Curl_safefree(part->mimetype);
  Curl_safefree(part->name);
  Curl_safefree(part->filename);
  Curl_mime_initpart(part, part->easy);
}

/* Recursively delete a mime handle and its parts. */
void curl_mime_free(curl_mime *mime)
{
  curl_mimepart *part;

  if(mime) {
    mime_subparts_unbind(mime);  /* Be sure it's not referenced anymore. */
    while(mime->firstpart) {
      part = mime->firstpart;
      mime->firstpart = part->nextpart;
      Curl_mime_cleanpart(part);
      free(part);
    }
    free(mime);
  }
}

CURLcode Curl_mime_duppart(curl_mimepart *dst, const curl_mimepart *src)
{
  curl_mime *mime;
  curl_mimepart *d;
  const curl_mimepart *s;
  CURLcode res = CURLE_OK;

  DEBUGASSERT(dst);

  /* Duplicate content. */
  switch(src->kind) {
  case MIMEKIND_NONE:
    break;
  case MIMEKIND_DATA:
    res = curl_mime_data(dst, src->data, (size_t) src->datasize);
    break;
  case MIMEKIND_FILE:
    res = curl_mime_filedata(dst, src->data);
    /* Do not abort duplication if file is not readable. */
    if(res == CURLE_READ_ERROR)
      res = CURLE_OK;
    break;
  case MIMEKIND_CALLBACK:
    res = curl_mime_data_cb(dst, src->datasize, src->readfunc,
                            src->seekfunc, src->freefunc, src->arg);
    break;
  case MIMEKIND_MULTIPART:
    /* No one knows about the cloned subparts, thus always attach ownership
       to the part. */
    mime = curl_mime_init(dst->easy);
    res = mime? curl_mime_subparts(dst, mime): CURLE_OUT_OF_MEMORY;

    /* Duplicate subparts. */
    for(s = ((curl_mime *) src->arg)->firstpart; !res && s; s = s->nextpart) {
      d = curl_mime_addpart(mime);
      res = d? Curl_mime_duppart(d, s): CURLE_OUT_OF_MEMORY;
    }
    break;
  default:  /* Invalid kind: should not occur. */
    res = CURLE_BAD_FUNCTION_ARGUMENT;  /* Internal error? */
    break;
  }

  /* Duplicate headers. */
  if(!res && src->userheaders) {
    struct curl_slist *hdrs = Curl_slist_duplicate(src->userheaders);

    if(!hdrs)
      res = CURLE_OUT_OF_MEMORY;
    else {
      /* No one but this procedure knows about the new header list,
         so always take ownership. */
      res = curl_mime_headers(dst, hdrs, TRUE);
      if(res)
        curl_slist_free_all(hdrs);
    }
  }

  if(!res) {
    /* Duplicate other fields. */
    dst->encoder = src->encoder;
    res = curl_mime_type(dst, src->mimetype);
  }
  if(!res)
    res = curl_mime_name(dst, src->name);
  if(!res)
    res = curl_mime_filename(dst, src->filename);

  /* If an error occurred, rollback. */
  if(res)
    Curl_mime_cleanpart(dst);

  return res;
}

/*
 * Mime build functions.
 */

/* Create a mime handle. */
curl_mime *curl_mime_init(struct Curl_easy *easy)
{
  curl_mime *mime;

  mime = (curl_mime *) malloc(sizeof(*mime));

  if(mime) {
    mime->easy = easy;
    mime->parent = NULL;
    mime->firstpart = NULL;
    mime->lastpart = NULL;

    memset(mime->boundary, '-', 24);
    if(Curl_rand_hex(easy, (unsigned char *) &mime->boundary[24],
                     MIME_RAND_BOUNDARY_CHARS + 1)) {
      /* failed to get random separator, bail out */
      free(mime);
      return NULL;
    }
    mimesetstate(&mime->state, MIMESTATE_BEGIN, NULL);
  }

  return mime;
}

/* Initialize a mime part. */
void Curl_mime_initpart(curl_mimepart *part, struct Curl_easy *easy)
{
  memset((char *) part, 0, sizeof(*part));
  part->easy = easy;
  part->lastreadstatus = 1; /* Successful read status. */
  mimesetstate(&part->state, MIMESTATE_BEGIN, NULL);
}

/* Create a mime part and append it to a mime handle's part list. */
curl_mimepart *curl_mime_addpart(curl_mime *mime)
{
  curl_mimepart *part;

  if(!mime)
    return NULL;

  part = (curl_mimepart *) malloc(sizeof(*part));

  if(part) {
    Curl_mime_initpart(part, mime->easy);
    part->parent = mime;

    if(mime->lastpart)
      mime->lastpart->nextpart = part;
    else
      mime->firstpart = part;

    mime->lastpart = part;
  }

  return part;
}

/* Set mime part name. */
CURLcode curl_mime_name(curl_mimepart *part, const char *name)
{
  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  Curl_safefree(part->name);
  part->name = NULL;

  if(name) {
    part->name = strdup(name);
    if(!part->name)
      return CURLE_OUT_OF_MEMORY;
  }

  return CURLE_OK;
}

/* Set mime part remote file name. */
CURLcode curl_mime_filename(curl_mimepart *part, const char *filename)
{
  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  Curl_safefree(part->filename);
  part->filename = NULL;

  if(filename) {
    part->filename = strdup(filename);
    if(!part->filename)
      return CURLE_OUT_OF_MEMORY;
  }

  return CURLE_OK;
}

/* Set mime part content from memory data. */
CURLcode curl_mime_data(curl_mimepart *part,
                        const char *data, size_t datasize)
{
  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  cleanup_part_content(part);

  if(data) {
    if(datasize == CURL_ZERO_TERMINATED)
      datasize = strlen(data);

    part->data = malloc(datasize + 1);
    if(!part->data)
      return CURLE_OUT_OF_MEMORY;

    part->datasize = datasize;

    if(datasize)
      memcpy(part->data, data, datasize);
    part->data[datasize] = '\0';    /* Set a null terminator as sentinel. */

    part->readfunc = mime_mem_read;
    part->seekfunc = mime_mem_seek;
    part->freefunc = mime_mem_free;
    part->flags |= MIME_FAST_READ;
    part->kind = MIMEKIND_DATA;
  }

  return CURLE_OK;
}

/* Set mime part content from named local file. */
CURLcode curl_mime_filedata(curl_mimepart *part, const char *filename)
{
  CURLcode result = CURLE_OK;

  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  cleanup_part_content(part);

  if(filename) {
    char *base;
    struct_stat sbuf;

    if(stat(filename, &sbuf) || access(filename, R_OK))
      result = CURLE_READ_ERROR;

    part->data = strdup(filename);
    if(!part->data)
      result = CURLE_OUT_OF_MEMORY;

    part->datasize = -1;
    if(!result && S_ISREG(sbuf.st_mode)) {
      part->datasize = filesize(filename, sbuf);
      part->seekfunc = mime_file_seek;
    }

    part->readfunc = mime_file_read;
    part->freefunc = mime_file_free;
    part->kind = MIMEKIND_FILE;

    /* As a side effect, set the filename to the current file's base name.
       It is possible to withdraw this by explicitly calling
       curl_mime_filename() with a NULL filename argument after the current
       call. */
    base = strippath(filename);
    if(!base)
      result = CURLE_OUT_OF_MEMORY;
    else {
      CURLcode res = curl_mime_filename(part, base);

      if(res)
        result = res;
      free(base);
    }
  }
  return result;
}

/* Set mime part type. */
CURLcode curl_mime_type(curl_mimepart *part, const char *mimetype)
{
  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  Curl_safefree(part->mimetype);
  part->mimetype = NULL;

  if(mimetype) {
    part->mimetype = strdup(mimetype);
    if(!part->mimetype)
      return CURLE_OUT_OF_MEMORY;
  }

  return CURLE_OK;
}

/* Set mime data transfer encoder. */
CURLcode curl_mime_encoder(curl_mimepart *part, const char *encoding)
{
  CURLcode result = CURLE_BAD_FUNCTION_ARGUMENT;
  const struct mime_encoder *mep;

  if(!part)
    return result;

  part->encoder = NULL;

  if(!encoding)
    return CURLE_OK;    /* Removing current encoder. */

  for(mep = encoders; mep->name; mep++)
    if(strcasecompare(encoding, mep->name)) {
      part->encoder = mep;
      result = CURLE_OK;
    }

  return result;
}

/* Set mime part headers. */
CURLcode curl_mime_headers(curl_mimepart *part,
                           struct curl_slist *headers, int take_ownership)
{
  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  if(part->flags & MIME_USERHEADERS_OWNER) {
    if(part->userheaders != headers)  /* Allow setting twice the same list. */
      curl_slist_free_all(part->userheaders);
    part->flags &= ~MIME_USERHEADERS_OWNER;
  }
  part->userheaders = headers;
  if(headers && take_ownership)
    part->flags |= MIME_USERHEADERS_OWNER;
  return CURLE_OK;
}

/* Set mime part content from callback. */
CURLcode curl_mime_data_cb(curl_mimepart *part, curl_off_t datasize,
                           curl_read_callback readfunc,
                           curl_seek_callback seekfunc,
                           curl_free_callback freefunc, void *arg)
{
  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  cleanup_part_content(part);

  if(readfunc) {
    part->readfunc = readfunc;
    part->seekfunc = seekfunc;
    part->freefunc = freefunc;
    part->arg = arg;
    part->datasize = datasize;
    part->kind = MIMEKIND_CALLBACK;
  }

  return CURLE_OK;
}

/* Set mime part content from subparts. */
CURLcode Curl_mime_set_subparts(curl_mimepart *part,
                                curl_mime *subparts, int take_ownership)
{
  curl_mime *root;

  if(!part)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  /* Accept setting twice the same subparts. */
  if(part->kind == MIMEKIND_MULTIPART && part->arg == subparts)
    return CURLE_OK;

  cleanup_part_content(part);

  if(subparts) {
    /* Must belong to the same data handle. */
    if(part->easy && subparts->easy && part->easy != subparts->easy)
      return CURLE_BAD_FUNCTION_ARGUMENT;

    /* Should not have been attached already. */
    if(subparts->parent)
      return CURLE_BAD_FUNCTION_ARGUMENT;

    /* Should not be the part's root. */
    root = part->parent;
    if(root) {
      while(root->parent && root->parent->parent)
        root = root->parent->parent;
      if(subparts == root) {
        if(part->easy)
          failf(part->easy, "Can't add itself as a subpart!");
        return CURLE_BAD_FUNCTION_ARGUMENT;
      }
    }

    subparts->parent = part;
    /* Subparts are processed internally: no read callback. */
    part->seekfunc = mime_subparts_seek;
    part->freefunc = take_ownership? mime_subparts_free: mime_subparts_unbind;
    part->arg = subparts;
    part->datasize = -1;
    part->kind = MIMEKIND_MULTIPART;
  }

  return CURLE_OK;
}

CURLcode curl_mime_subparts(curl_mimepart *part, curl_mime *subparts)
{
  return Curl_mime_set_subparts(part, subparts, TRUE);
}


/* Readback from top mime. */
/* Argument is the dummy top part. */
size_t Curl_mime_read(char *buffer, size_t size, size_t nitems, void *instream)
{
  curl_mimepart *part = (curl_mimepart *) instream;
  size_t ret;
  bool hasread;

  (void) size;   /* Always 1. */

  do {
    hasread = FALSE;
    ret = readback_part(part, buffer, nitems, &hasread);
    /*
     * If this is not possible to get some data without calling more than
     * one read callback (probably because a content encoder is not able to
     * deliver a new bunch for the few data accumulated so far), force another
     * read until we get enough data or a special exit code.
     */
  } while(ret == STOP_FILLING);

  return ret;
}

/* Rewind mime stream. */
CURLcode Curl_mime_rewind(curl_mimepart *part)
{
  return mime_part_rewind(part) == CURL_SEEKFUNC_OK?
         CURLE_OK: CURLE_SEND_FAIL_REWIND;
}

/* Compute header list size. */
static size_t slist_size(struct curl_slist *s,
                         size_t overhead, const char *skip)
{
  size_t size = 0;
  size_t skiplen = skip? strlen(skip): 0;

  for(; s; s = s->next)
    if(!skip || !match_header(s, skip, skiplen))
      size += strlen(s->data) + overhead;
  return size;
}

/* Get/compute multipart size. */
static curl_off_t multipart_size(curl_mime *mime)
{
  curl_off_t size;
  size_t boundarysize;
  curl_mimepart *part;

  if(!mime)
    return 0;           /* Not present -> empty. */

  boundarysize = 4 + strlen(mime->boundary) + 2;
  size = boundarysize;  /* Final boundary - CRLF after headers. */

  for(part = mime->firstpart; part; part = part->nextpart) {
    curl_off_t sz = Curl_mime_size(part);

    if(sz < 0)
      size = sz;

    if(size >= 0)
      size += boundarysize + sz;
  }

  return size;
}

/* Get/compute mime size. */
curl_off_t Curl_mime_size(curl_mimepart *part)
{
  curl_off_t size;

  if(part->kind == MIMEKIND_MULTIPART)
    part->datasize = multipart_size(part->arg);

  size = part->datasize;

  if(part->encoder)
    size = part->encoder->sizefunc(part);

  if(size >= 0 && !(part->flags & MIME_BODY_ONLY)) {
    /* Compute total part size. */
    size += slist_size(part->curlheaders, 2, NULL);
    size += slist_size(part->userheaders, 2, "Content-Type");
    size += 2;    /* CRLF after headers. */
  }
  return size;
}

/* Add a header. */
/* VARARGS2 */
CURLcode Curl_mime_add_header(struct curl_slist **slp, const char *fmt, ...)
{
  struct curl_slist *hdr = NULL;
  char *s = NULL;
  va_list ap;

  va_start(ap, fmt);
  s = curl_mvaprintf(fmt, ap);
  va_end(ap);

  if(s) {
    hdr = Curl_slist_append_nodup(*slp, s);
    if(hdr)
      *slp = hdr;
    else
      free(s);
  }

  return hdr? CURLE_OK: CURLE_OUT_OF_MEMORY;
}

/* Add a content type header. */
static CURLcode add_content_type(struct curl_slist **slp,
                                 const char *type, const char *boundary)
{
  return Curl_mime_add_header(slp, "Content-Type: %s%s%s", type,
                              boundary? "; boundary=": "",
                              boundary? boundary: "");
}

const char *Curl_mime_contenttype(const char *filename)
{
  /*
   * If no content type was specified, we scan through a few well-known
   * extensions and pick the first we match!
   */
  struct ContentType {
    const char *extension;
    const char *type;
  };
  static const struct ContentType ctts[] = {
    {".gif",  "image/gif"},
    {".jpg",  "image/jpeg"},
    {".jpeg", "image/jpeg"},
    {".png",  "image/png"},
    {".svg",  "image/svg+xml"},
    {".txt",  "text/plain"},
    {".htm",  "text/html"},
    {".html", "text/html"},
    {".pdf",  "application/pdf"},
    {".xml",  "application/xml"}
  };

  if(filename) {
    size_t len1 = strlen(filename);
    const char *nameend = filename + len1;
    unsigned int i;

    for(i = 0; i < sizeof(ctts) / sizeof(ctts[0]); i++) {
      size_t len2 = strlen(ctts[i].extension);

      if(len1 >= len2 && strcasecompare(nameend - len2, ctts[i].extension))
          return ctts[i].type;
    }
  }
  return NULL;
}

static bool content_type_match(const char *contenttype, const char *target)
{
  size_t len = strlen(target);

  if(contenttype && strncasecompare(contenttype, target, len))
    switch(contenttype[len]) {
    case '\0':
    case '\t':
    case '\r':
    case '\n':
    case ' ':
    case ';':
      return TRUE;
    }
  return FALSE;
}

CURLcode Curl_mime_prepare_headers(curl_mimepart *part,
                                   const char *contenttype,
                                   const char *disposition,
                                   enum mimestrategy strategy)
{
  curl_mime *mime = NULL;
  const char *boundary = NULL;
  char *customct;
  const char *cte = NULL;
  CURLcode ret = CURLE_OK;

  /* Get rid of previously prepared headers. */
  curl_slist_free_all(part->curlheaders);
  part->curlheaders = NULL;

  /* Be sure we won't access old headers later. */
  if(part->state.state == MIMESTATE_CURLHEADERS)
    mimesetstate(&part->state, MIMESTATE_CURLHEADERS, NULL);

  /* Check if content type is specified. */
  customct = part->mimetype;
  if(!customct)
    customct = search_header(part->userheaders, "Content-Type");
  if(customct)
    contenttype = customct;

  /* If content type is not specified, try to determine it. */
  if(!contenttype) {
    switch(part->kind) {
    case MIMEKIND_MULTIPART:
      contenttype = MULTIPART_CONTENTTYPE_DEFAULT;
      break;
    case MIMEKIND_FILE:
      contenttype = Curl_mime_contenttype(part->filename);
      if(!contenttype)
        contenttype = Curl_mime_contenttype(part->data);
      if(!contenttype && part->filename)
        contenttype = FILE_CONTENTTYPE_DEFAULT;
      break;
    default:
      contenttype = Curl_mime_contenttype(part->filename);
      break;
    }
  }

  if(part->kind == MIMEKIND_MULTIPART) {
    mime = (curl_mime *) part->arg;
    if(mime)
      boundary = mime->boundary;
  }
  else if(contenttype && !customct &&
          content_type_match(contenttype, "text/plain"))
    if(strategy == MIMESTRATEGY_MAIL || !part->filename)
      contenttype = NULL;

  /* Issue content-disposition header only if not already set by caller. */
  if(!search_header(part->userheaders, "Content-Disposition")) {
    if(!disposition)
      if(part->filename || part->name ||
        (contenttype && !strncasecompare(contenttype, "multipart/", 10)))
          disposition = DISPOSITION_DEFAULT;
    if(disposition && curl_strequal(disposition, "attachment") &&
     !part->name && !part->filename)
      disposition = NULL;
    if(disposition) {
      char *name = NULL;
      char *filename = NULL;

      if(part->name) {
        name = escape_string(part->easy, part->name, strategy);
        if(!name)
          ret = CURLE_OUT_OF_MEMORY;
      }
      if(!ret && part->filename) {
        filename = escape_string(part->easy, part->filename, strategy);
        if(!filename)
          ret = CURLE_OUT_OF_MEMORY;
      }
      if(!ret)
        ret = Curl_mime_add_header(&part->curlheaders,
                                   "Content-Disposition: %s%s%s%s%s%s%s",
                                   disposition,
                                   name? "; name=\"": "",
                                   name? name: "",
                                   name? "\"": "",
                                   filename? "; filename=\"": "",
                                   filename? filename: "",
                                   filename? "\"": "");
      Curl_safefree(name);
      Curl_safefree(filename);
      if(ret)
        return ret;
      }
    }

  /* Issue Content-Type header. */
  if(contenttype) {
    ret = add_content_type(&part->curlheaders, contenttype, boundary);
    if(ret)
      return ret;
  }

  /* Content-Transfer-Encoding header. */
  if(!search_header(part->userheaders, "Content-Transfer-Encoding")) {
    if(part->encoder)
      cte = part->encoder->name;
    else if(contenttype && strategy == MIMESTRATEGY_MAIL &&
     part->kind != MIMEKIND_MULTIPART)
      cte = "8bit";
    if(cte) {
      ret = Curl_mime_add_header(&part->curlheaders,
                                 "Content-Transfer-Encoding: %s", cte);
      if(ret)
        return ret;
    }
  }

  /* If we were reading curl-generated headers, restart with new ones (this
     should not occur). */
  if(part->state.state == MIMESTATE_CURLHEADERS)
    mimesetstate(&part->state, MIMESTATE_CURLHEADERS, part->curlheaders);

  /* Process subparts. */
  if(part->kind == MIMEKIND_MULTIPART && mime) {
    curl_mimepart *subpart;

    disposition = NULL;
    if(content_type_match(contenttype, "multipart/form-data"))
      disposition = "form-data";
    for(subpart = mime->firstpart; subpart; subpart = subpart->nextpart) {
      ret = Curl_mime_prepare_headers(subpart, NULL, disposition, strategy);
      if(ret)
        return ret;
    }
  }
  return ret;
}

/* Recursively reset paused status in the given part. */
void Curl_mime_unpause(curl_mimepart *part)
{
  if(part) {
    if(part->lastreadstatus == CURL_READFUNC_PAUSE)
      part->lastreadstatus = 1; /* Successful read status. */
    if(part->kind == MIMEKIND_MULTIPART) {
      curl_mime *mime = (curl_mime *) part->arg;

      if(mime) {
        curl_mimepart *subpart;

        for(subpart = mime->firstpart; subpart; subpart = subpart->nextpart)
          Curl_mime_unpause(subpart);
      }
    }
  }
}


#else /* !CURL_DISABLE_HTTP && !CURL_DISABLE_MIME ||
         !CURL_DISABLE_SMTP || !CURL_DISABLE_IMAP */

/* Mime not compiled in: define stubs for externally-referenced functions. */
curl_mime *curl_mime_init(CURL *easy)
{
  (void) easy;
  return NULL;
}

void curl_mime_free(curl_mime *mime)
{
  (void) mime;
}

curl_mimepart *curl_mime_addpart(curl_mime *mime)
{
  (void) mime;
  return NULL;
}

CURLcode curl_mime_name(curl_mimepart *part, const char *name)
{
  (void) part;
  (void) name;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_filename(curl_mimepart *part, const char *filename)
{
  (void) part;
  (void) filename;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_type(curl_mimepart *part, const char *mimetype)
{
  (void) part;
  (void) mimetype;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_encoder(curl_mimepart *part, const char *encoding)
{
  (void) part;
  (void) encoding;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_data(curl_mimepart *part,
                        const char *data, size_t datasize)
{
  (void) part;
  (void) data;
  (void) datasize;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_filedata(curl_mimepart *part, const char *filename)
{
  (void) part;
  (void) filename;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_data_cb(curl_mimepart *part,
                           curl_off_t datasize,
                           curl_read_callback readfunc,
                           curl_seek_callback seekfunc,
                           curl_free_callback freefunc,
                           void *arg)
{
  (void) part;
  (void) datasize;
  (void) readfunc;
  (void) seekfunc;
  (void) freefunc;
  (void) arg;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_subparts(curl_mimepart *part, curl_mime *subparts)
{
  (void) part;
  (void) subparts;
  return CURLE_NOT_BUILT_IN;
}

CURLcode curl_mime_headers(curl_mimepart *part,
                           struct curl_slist *headers, int take_ownership)
{
  (void) part;
  (void) headers;
  (void) take_ownership;
  return CURLE_NOT_BUILT_IN;
}

CURLcode Curl_mime_add_header(struct curl_slist **slp, const char *fmt, ...)
{
  (void)slp;
  (void)fmt;
  return CURLE_NOT_BUILT_IN;
}

#endif /* if disabled */
