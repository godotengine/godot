#ifndef HEADER_CURL_ARPA_TELNET_H
#define HEADER_CURL_ARPA_TELNET_H
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
#ifndef CURL_DISABLE_TELNET
/*
 * Telnet option defines. Add more here if in need.
 */
#define CURL_TELOPT_BINARY   0  /* binary 8bit data */
#define CURL_TELOPT_ECHO     1  /* just echo! */
#define CURL_TELOPT_SGA      3  /* Suppress Go Ahead */
#define CURL_TELOPT_EXOPL  255  /* EXtended OPtions List */
#define CURL_TELOPT_TTYPE   24  /* Terminal TYPE */
#define CURL_TELOPT_NAWS    31  /* Negotiate About Window Size */
#define CURL_TELOPT_XDISPLOC 35 /* X DISPlay LOCation */

#define CURL_TELOPT_NEW_ENVIRON 39  /* NEW ENVIRONment variables */
#define CURL_NEW_ENV_VAR   0
#define CURL_NEW_ENV_VALUE 1

#ifndef CURL_DISABLE_VERBOSE_STRINGS
/*
 * The telnet options represented as strings
 */
static const char * const telnetoptions[]=
{
  "BINARY",      "ECHO",           "RCP",           "SUPPRESS GO AHEAD",
  "NAME",        "STATUS",         "TIMING MARK",   "RCTE",
  "NAOL",        "NAOP",           "NAOCRD",        "NAOHTS",
  "NAOHTD",      "NAOFFD",         "NAOVTS",        "NAOVTD",
  "NAOLFD",      "EXTEND ASCII",   "LOGOUT",        "BYTE MACRO",
  "DE TERMINAL", "SUPDUP",         "SUPDUP OUTPUT", "SEND LOCATION",
  "TERM TYPE",   "END OF RECORD",  "TACACS UID",    "OUTPUT MARKING",
  "TTYLOC",      "3270 REGIME",    "X3 PAD",        "NAWS",
  "TERM SPEED",  "LFLOW",          "LINEMODE",      "XDISPLOC",
  "OLD-ENVIRON", "AUTHENTICATION", "ENCRYPT",       "NEW-ENVIRON"
};
#endif

#define CURL_TELOPT_MAXIMUM CURL_TELOPT_NEW_ENVIRON

#define CURL_TELOPT_OK(x) ((x) <= CURL_TELOPT_MAXIMUM)
#define CURL_TELOPT(x)    telnetoptions[x]

#define CURL_NTELOPTS 40

/*
 * First some defines
 */
#define CURL_xEOF 236 /* End Of File */
#define CURL_SE   240 /* Sub negotiation End */
#define CURL_NOP  241 /* No OPeration */
#define CURL_DM   242 /* Data Mark */
#define CURL_GA   249 /* Go Ahead, reverse the line */
#define CURL_SB   250 /* SuBnegotiation */
#define CURL_WILL 251 /* Our side WILL use this option */
#define CURL_WONT 252 /* Our side WON'T use this option */
#define CURL_DO   253 /* DO use this option! */
#define CURL_DONT 254 /* DON'T use this option! */
#define CURL_IAC  255 /* Interpret As Command */

#ifndef CURL_DISABLE_VERBOSE_STRINGS
/*
 * Then those numbers represented as strings:
 */
static const char * const telnetcmds[]=
{
  "EOF",  "SUSP",  "ABORT", "EOR",  "SE",
  "NOP",  "DMARK", "BRK",   "IP",   "AO",
  "AYT",  "EC",    "EL",    "GA",   "SB",
  "WILL", "WONT",  "DO",    "DONT", "IAC"
};
#endif

#define CURL_TELCMD_MINIMUM CURL_xEOF /* the first one */
#define CURL_TELCMD_MAXIMUM CURL_IAC  /* surprise, 255 is the last one! ;-) */

#define CURL_TELQUAL_IS   0
#define CURL_TELQUAL_SEND 1
#define CURL_TELQUAL_INFO 2
#define CURL_TELQUAL_NAME 3

#define CURL_TELCMD_OK(x) ( ((unsigned int)(x) >= CURL_TELCMD_MINIMUM) && \
                       ((unsigned int)(x) <= CURL_TELCMD_MAXIMUM) )
#define CURL_TELCMD(x)    telnetcmds[(x)-CURL_TELCMD_MINIMUM]

#endif /* CURL_DISABLE_TELNET */

#endif /* HEADER_CURL_ARPA_TELNET_H */
