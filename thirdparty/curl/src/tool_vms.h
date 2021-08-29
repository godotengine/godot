#ifndef HEADER_CURL_TOOL_VMS_H
#define HEADER_CURL_TOOL_VMS_H
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

#ifdef __VMS

/*
 * Forward-declaration of global variable vms_show defined
 * in tool_main.c, used in main() as parameter for function
 * vms_special_exit() to allow proper curl tool exiting.
 */
extern int vms_show;

int is_vms_shell(void);
void vms_special_exit(int code, int vms_show);

#undef exit
#define exit(__code) vms_special_exit((__code), (0))

#define  VMS_STS(c,f,e,s) (((c&0xF)<<28)|((f&0xFFF)<<16)|((e&0x1FFF)<3)|(s&7))
#define  VMSSTS_HIDE  VMS_STS(1,0,0,0)

#endif /* __VMS */

#endif /* HEADER_CURL_TOOL_VMS_H */
