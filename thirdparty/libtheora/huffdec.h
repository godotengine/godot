/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

#if !defined(_huffdec_H)
# define _huffdec_H (1)
# include "huffman.h"
# include "bitpack.h"



int oc_huff_trees_unpack(oc_pack_buf *_opb,
 ogg_int16_t *_nodes[TH_NHUFFMAN_TABLES]);
int oc_huff_trees_copy(ogg_int16_t *_dst[TH_NHUFFMAN_TABLES],
 const ogg_int16_t *const _src[TH_NHUFFMAN_TABLES]);
void oc_huff_trees_clear(ogg_int16_t *_nodes[TH_NHUFFMAN_TABLES]);
int oc_huff_token_decode_c(oc_pack_buf *_opb,const ogg_int16_t *_node);

#endif
