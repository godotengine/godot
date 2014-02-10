#if !defined(_enquant_H)
# define _enquant_H (1)
# include "quant.h"

typedef struct oc_iquant oc_iquant;

#define OC_QUANT_MAX_LOG (OC_Q57(OC_STATIC_ILOG_32(OC_QUANT_MAX)-1))

/*Used to compute x/d via ((x*m>>16)+x>>l)+(x<0))
   (i.e., one 16x16->16 mul, 2 shifts, and 2 adds).
  This is not an approximation; for 16-bit x and d, it is exact.*/
struct oc_iquant{
  ogg_int16_t m;
  ogg_int16_t l;
};

typedef oc_iquant        oc_iquant_table[64];



void oc_quant_params_pack(oggpack_buffer *_opb,const th_quant_info *_qinfo);
void oc_enquant_tables_init(ogg_uint16_t *_dequant[64][3][2],
 oc_iquant *_enquant[64][3][2],const th_quant_info *_qinfo);
void oc_enquant_qavg_init(ogg_int64_t _log_qavg[2][64],
 ogg_uint16_t *_dequant[64][3][2],int _pixel_fmt);

#endif
