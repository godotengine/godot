/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation https://www.xiph.org/                 *
 *                                                                  *
 ********************************************************************

  function: mode selection code

 ********************************************************************/
#if !defined(_collect_H)
# define _collect_H (1)
# include "encint.h"
# if defined(OC_COLLECT_METRICS)
#  include <stdio.h>



typedef struct oc_mode_metrics oc_mode_metrics;



/**Sets the file name to load/store mode metrics from/to.
 * The file name string is stored by reference, and so must be valid for the
 *  lifetime of the encoder.
 * Mode metric collection uses global tables; do not attempt to perform
 *  multiple collections at once.
 * \param[in] _buf <tt>char[]</tt> The file name.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_METRICS_FILE (0x8000)



/*Accumulates various weighted sums of the measurements.
  w -> weight
  s -> SATD
  q -> log quantizer
  r -> rate (in bits)
  d -> RMSE
  All of the single letters correspond to direct, weighted sums, e.g.,
   w=sum(w_i), s=sum(s_i*w_i), etc.
  The others correspond to central moments (or co-moments) of the given order,
   e.g., sq=sum((s_i-s/w)*(q_i-q/w)*w_i).
  Because we need some moments up to fourth order, we use central moments to
   minimize the dynamic range and prevent rounding error from dominating the
   calculations.*/
struct oc_mode_metrics{
  double w;
  double s;
  double q;
  double r;
  double d;
  double s2;
  double sq;
  double q2;
  double sr;
  double qr;
  double r2;
  double sd;
  double qd;
  double d2;
  double s2q;
  double sq2;
  double sqr;
  double sqd;
  double s2q2;
};


# define OC_ZWEIGHT   (0.25)

/*TODO: It may be helpful (for block-level quantizers especially) to separate
   out the contributions from AC and DC into separate tables.*/

extern ogg_int16_t OC_MODE_LOGQ[OC_LOGQ_BINS][3][2];
extern oc_mode_rd  OC_MODE_RD_SATD[OC_LOGQ_BINS][3][2][OC_COMP_BINS];
extern oc_mode_rd  OC_MODE_RD_SAD[OC_LOGQ_BINS][3][2][OC_COMP_BINS];

extern int              OC_HAS_MODE_METRICS;
extern oc_mode_metrics  OC_MODE_METRICS_SATD[OC_LOGQ_BINS-1][3][2][OC_COMP_BINS];
extern oc_mode_metrics  OC_MODE_METRICS_SAD[OC_LOGQ_BINS-1][3][2][OC_COMP_BINS];
extern const char      *OC_MODE_METRICS_FILENAME;

void oc_mode_metrics_dump();
void oc_mode_metrics_print(FILE *_fout);

void oc_mode_metrics_add(oc_mode_metrics *_metrics,
 double _w,int _s,int _q,int _r,double _d);
void oc_mode_metrics_merge(oc_mode_metrics *_dst,
 const oc_mode_metrics *_src,int _n);
double oc_mode_metrics_solve(double *_r,double *_d,
 const oc_mode_metrics *_metrics,const int *_s0,const int *_s1,
 const int *_q0,const int *_q1,
 const double *_ra,const double *_rb,const double *_rc,
 const double *_da,const double *_db,const double *_dc,int _n);
void oc_mode_metrics_update(oc_mode_metrics (*_metrics)[3][2][OC_COMP_BINS],
 int _niters_min,int _reweight,oc_mode_rd (*_table)[3][2][OC_COMP_BINS],
 int shift,double (*_weight)[3][2][OC_COMP_BINS]);
void oc_enc_mode_metrics_load(oc_enc_ctx *_enc);
void oc_enc_mode_metrics_collect(oc_enc_ctx *_enc);

# endif
#endif
