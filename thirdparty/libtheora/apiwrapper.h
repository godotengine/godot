/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: apiwrapper.h 13596 2007-08-23 20:05:38Z tterribe $

 ********************************************************************/

#if !defined(_apiwrapper_H)
# define _apiwrapper_H (1)
# include <ogg/ogg.h>
# include <theora/theora.h>
# include "theora/theoradec.h"
# include "theora/theoraenc.h"
# include "internal.h"

typedef struct th_api_wrapper th_api_wrapper;
typedef struct th_api_info    th_api_info;

/*Provide an entry point for the codec setup to clear itself in case we ever
   want to break pieces off into a common base library shared by encoder and
   decoder.
  In addition, this makes several other pieces of the API wrapper cleaner.*/
typedef void (*oc_setup_clear_func)(void *_ts);

/*Generally only one of these pointers will be non-NULL in any given instance.
  Technically we do not even really need this struct, since we should be able
   to figure out which one from "context", but doing it this way makes sure we
   don't flub it up.*/
struct th_api_wrapper{
  oc_setup_clear_func  clear;
  th_setup_info       *setup;
  th_dec_ctx          *decode;
  th_enc_ctx          *encode;
};

struct th_api_info{
  th_api_wrapper api;
  theora_info    info;
};


void oc_theora_info2th_info(th_info *_info,const theora_info *_ci);

#endif
