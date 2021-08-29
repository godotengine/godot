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
#include "tool_setup.h"

#if defined(__AMIGA__) && !defined(__amigaos4__)
#  undef HAVE_TERMIOS_H
#endif

#ifndef HAVE_GETPASS_R
/* this file is only for systems without getpass_r() */

#ifdef HAVE_FCNTL_H
#  include <fcntl.h>
#endif

#ifdef HAVE_TERMIOS_H
#  include <termios.h>
#elif defined(HAVE_TERMIO_H)
#  include <termio.h>
#endif

#ifdef __VMS
#  include descrip
#  include starlet
#  include iodef
#endif

#ifdef WIN32
#  include <conio.h>
#endif

#ifdef NETWARE
#  ifdef __NOVELL_LIBC__
#    include <screen.h>
#  else
#    include <nwconio.h>
#  endif
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include "tool_getpass.h"

#include "memdebug.h" /* keep this as LAST include */

#ifdef __VMS
/* VMS implementation */
char *getpass_r(const char *prompt, char *buffer, size_t buflen)
{
  long sts;
  short chan;

  /* MSK, 23-JAN-2004, iosbdef.h wasn't in VAX V7.2 or CC 6.4  */
  /* distribution so I created this.  May revert back later to */
  /* struct _iosb iosb;                                        */
  struct _iosb
     {
     short int iosb$w_status; /* status     */
     short int iosb$w_bcnt;   /* byte count */
     int       unused;        /* unused     */
     } iosb;

  $DESCRIPTOR(ttdesc, "TT");

  buffer[0] = '\0';
  sts = sys$assign(&ttdesc, &chan, 0, 0);
  if(sts & 1) {
    sts = sys$qiow(0, chan,
                   IO$_READPROMPT | IO$M_NOECHO,
                   &iosb, 0, 0, buffer, buflen, 0, 0,
                   prompt, strlen(prompt));

    if((sts & 1) && (iosb.iosb$w_status & 1))
      buffer[iosb.iosb$w_bcnt] = '\0';

    sys$dassgn(chan);
  }
  return buffer; /* we always return success */
}
#define DONE
#endif /* __VMS */

#if defined(WIN32)

char *getpass_r(const char *prompt, char *buffer, size_t buflen)
{
  size_t i;
  fputs(prompt, stderr);

  for(i = 0; i < buflen; i++) {
    buffer[i] = (char)getch();
    if(buffer[i] == '\r' || buffer[i] == '\n') {
      buffer[i] = '\0';
      break;
    }
    else
      if(buffer[i] == '\b')
        /* remove this letter and if this is not the first key, remove the
           previous one as well */
        i = i - (i >= 1 ? 2 : 1);
  }
  /* since echo is disabled, print a newline */
  fputs("\n", stderr);
  /* if user didn't hit ENTER, terminate buffer */
  if(i == buflen)
    buffer[buflen-1] = '\0';

  return buffer; /* we always return success */
}
#define DONE
#endif /* WIN32 */

#ifdef NETWARE
/* NetWare implementation */
#ifdef __NOVELL_LIBC__
char *getpass_r(const char *prompt, char *buffer, size_t buflen)
{
  return getpassword(prompt, buffer, buflen);
}
#else
char *getpass_r(const char *prompt, char *buffer, size_t buflen)
{
  size_t i = 0;

  printf("%s", prompt);
  do {
    buffer[i++] = getch();
    if(buffer[i-1] == '\b') {
      /* remove this letter and if this is not the first key,
         remove the previous one as well */
      if(i > 1) {
        printf("\b \b");
        i = i - 2;
      }
      else {
        RingTheBell();
        i = i - 1;
      }
    }
    else if(buffer[i-1] != 13)
      putchar('*');

  } while((buffer[i-1] != 13) && (i < buflen));
  buffer[i-1] = '\0';
  printf("\r\n");
  return buffer;
}
#endif /* __NOVELL_LIBC__ */
#define DONE
#endif /* NETWARE */

#ifndef DONE /* not previously provided */

#ifdef HAVE_TERMIOS_H
#  define struct_term  struct termios
#elif defined(HAVE_TERMIO_H)
#  define struct_term  struct termio
#else
#  undef  struct_term
#endif

static bool ttyecho(bool enable, int fd)
{
#ifdef struct_term
  static struct_term withecho;
  static struct_term noecho;
#endif
  if(!enable) {
    /* disable echo by extracting the current 'withecho' mode and remove the
       ECHO bit and set back the struct */
#ifdef HAVE_TERMIOS_H
    tcgetattr(fd, &withecho);
    noecho = withecho;
    noecho.c_lflag &= ~ECHO;
    tcsetattr(fd, TCSANOW, &noecho);
#elif defined(HAVE_TERMIO_H)
    ioctl(fd, TCGETA, &withecho);
    noecho = withecho;
    noecho.c_lflag &= ~ECHO;
    ioctl(fd, TCSETA, &noecho);
#else
    /* neither HAVE_TERMIO_H nor HAVE_TERMIOS_H, we can't disable echo! */
    (void)fd;
    return FALSE; /* not disabled */
#endif
    return TRUE; /* disabled */
  }
  /* re-enable echo, assumes we disabled it before (and set the structs we
     now use to reset the terminal status) */
#ifdef HAVE_TERMIOS_H
  tcsetattr(fd, TCSAFLUSH, &withecho);
#elif defined(HAVE_TERMIO_H)
  ioctl(fd, TCSETA, &withecho);
#else
  return FALSE; /* not enabled */
#endif
  return TRUE; /* enabled */
}

char *getpass_r(const char *prompt, /* prompt to display */
                char *password,     /* buffer to store password in */
                size_t buflen)      /* size of buffer to store password in */
{
  ssize_t nread;
  bool disabled;
  int fd = open("/dev/tty", O_RDONLY);
  if(-1 == fd)
    fd = STDIN_FILENO; /* use stdin if the tty couldn't be used */

  disabled = ttyecho(FALSE, fd); /* disable terminal echo */

  fputs(prompt, stderr);
  nread = read(fd, password, buflen);
  if(nread > 0)
    password[--nread] = '\0'; /* null-terminate where enter is stored */
  else
    password[0] = '\0'; /* got nothing */

  if(disabled) {
    /* if echo actually was disabled, add a newline */
    fputs("\n", stderr);
    (void)ttyecho(TRUE, fd); /* enable echo */
  }

  if(STDIN_FILENO != fd)
    close(fd);

  return password; /* return pointer to buffer */
}

#endif /* DONE */
#endif /* HAVE_GETPASS_R */
