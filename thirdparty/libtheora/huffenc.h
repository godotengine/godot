#if !defined(_huffenc_H)
# define _huffenc_H (1)
# include "huffman.h"



typedef th_huff_code                  th_huff_table[TH_NDCT_TOKENS];



extern const th_huff_code
 TH_VP31_HUFF_CODES[TH_NHUFFMAN_TABLES][TH_NDCT_TOKENS];



int oc_huff_codes_pack(oggpack_buffer *_opb,
 const th_huff_code _codes[TH_NHUFFMAN_TABLES][TH_NDCT_TOKENS]);

#endif
