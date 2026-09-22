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

  function:

 ********************************************************************/
#include <stdlib.h>
#include <string.h>
#include "encint.h"

/*A rough lookup table for tan(x), 0<=x<pi/2.
  The values are Q12 fixed-point and spaced at 5 degree intervals.
  These decisions are somewhat arbitrary, but sufficient for the 2nd order
   Bessel follower below.
  Values of x larger than 85 degrees are extrapolated from the last interval,
   which is way off, but "good enough".*/
static unsigned short OC_ROUGH_TAN_LOOKUP[18]={
      0,  358,  722, 1098, 1491, 1910,
   2365, 2868, 3437, 4096, 4881, 5850,
   7094, 8784,11254,15286,23230,46817
};

/*_alpha is Q24 in the range [0,0.5).
  The return values is 5.12.*/
static int oc_warp_alpha(int _alpha){
  int i;
  int d;
  int t0;
  int t1;
  i=_alpha*36>>24;
  if(i>=17)i=16;
  t0=OC_ROUGH_TAN_LOOKUP[i];
  t1=OC_ROUGH_TAN_LOOKUP[i+1];
  d=_alpha*36-(i<<24);
  return (int)(((ogg_int64_t)t0<<32)+(t1-t0<<8)*(ogg_int64_t)d>>32);
}

/*Re-initialize the Bessel filter coefficients with the specified delay.
  This does not alter the x/y state, but changes the reaction time of the
   filter.
  Altering the time constant of a reactive filter without alterning internal
   state is something that has to be done carefully, but our design operates at
   high enough delays and with small enough time constant changes to make it
   safe.*/
static void oc_iir_filter_reinit(oc_iir_filter *_f,int _delay){
  int         alpha;
  ogg_int64_t one48;
  ogg_int64_t warp;
  ogg_int64_t k1;
  ogg_int64_t k2;
  ogg_int64_t d;
  ogg_int64_t a;
  ogg_int64_t ik2;
  ogg_int64_t b1;
  ogg_int64_t b2;
  /*This borrows some code from an unreleased version of Postfish.
    See the recipe at http://unicorn.us.com/alex/2polefilters.html for details
     on deriving the filter coefficients.*/
  /*alpha is Q24*/
  alpha=(1<<24)/_delay;
  one48=(ogg_int64_t)1<<48;
  /*warp is 7.12*/
  warp=OC_MAXI(oc_warp_alpha(alpha),1);
  /*k1 is 9.12*/
  k1=3*warp;
  /*k2 is 16.24.*/
  k2=k1*warp;
  /*d is 16.15.*/
  d=((1<<12)+k1<<12)+k2+256>>9;
  /*a is 0.32, since d is larger than both 1.0 and k2.*/
  a=(k2<<23)/d;
  /*ik2 is 25.24.*/
  ik2=one48/k2;
  /*b1 is Q56; in practice, the integer ranges between -2 and 2.*/
  b1=2*a*(ik2-(1<<24));
  /*b2 is Q56; in practice, the integer ranges between -2 and 2.*/
  b2=(one48<<8)-(4*a<<24)-b1;
  /*All of the filter parameters are Q24.*/
  _f->c[0]=(ogg_int32_t)(b1+((ogg_int64_t)1<<31)>>32);
  _f->c[1]=(ogg_int32_t)(b2+((ogg_int64_t)1<<31)>>32);
  _f->g=(ogg_int32_t)(a+128>>8);
}

/*Initialize a 2nd order low-pass Bessel filter with the corresponding delay
   and initial value.
  _value is Q24.*/
static void oc_iir_filter_init(oc_iir_filter *_f,int _delay,ogg_int32_t _value){
  oc_iir_filter_reinit(_f,_delay);
  _f->y[1]=_f->y[0]=_f->x[1]=_f->x[0]=_value;
}

static ogg_int64_t oc_iir_filter_update(oc_iir_filter *_f,ogg_int32_t _x){
  ogg_int64_t c0;
  ogg_int64_t c1;
  ogg_int64_t g;
  ogg_int64_t x0;
  ogg_int64_t x1;
  ogg_int64_t y0;
  ogg_int64_t y1;
  ogg_int64_t ya;
  c0=_f->c[0];
  c1=_f->c[1];
  g=_f->g;
  x0=_f->x[0];
  x1=_f->x[1];
  y0=_f->y[0];
  y1=_f->y[1];
  ya=(_x+x0*2+x1)*g+y0*c0+y1*c1+(1<<23)>>24;
  _f->x[1]=(ogg_int32_t)x0;
  _f->x[0]=_x;
  _f->y[1]=(ogg_int32_t)y0;
  _f->y[0]=(ogg_int32_t)ya;
  return ya;
}



/*Search for the quantizer that matches the target most closely.
  We don't assume a linear ordering, but when there are ties we pick the
   quantizer closest to the old one.*/
static int oc_enc_find_qi_for_target(oc_enc_ctx *_enc,int _qti,int _qi_old,
 int _qi_min,ogg_int64_t _log_qtarget){
  ogg_int64_t best_qdiff;
  int         best_qi;
  int         qi;
  best_qi=_qi_min;
  best_qdiff=_enc->log_qavg[_qti][best_qi]-_log_qtarget;
  best_qdiff=best_qdiff+OC_SIGNMASK(best_qdiff)^OC_SIGNMASK(best_qdiff);
  for(qi=_qi_min+1;qi<64;qi++){
    ogg_int64_t qdiff;
    qdiff=_enc->log_qavg[_qti][qi]-_log_qtarget;
    qdiff=qdiff+OC_SIGNMASK(qdiff)^OC_SIGNMASK(qdiff);
    if(qdiff<best_qdiff||
     qdiff==best_qdiff&&abs(qi-_qi_old)<abs(best_qi-_qi_old)){
      best_qi=qi;
      best_qdiff=qdiff;
    }
  }
  return best_qi;
}

void oc_enc_calc_lambda(oc_enc_ctx *_enc,int _qti){
  ogg_int64_t lq;
  int         qi;
  int         qi1;
  int         nqis;
  /*For now, lambda is fixed depending on the qi value and frame type:
      lambda=qscale*(qavg[qti][qi]**2),
     where qscale=0.2125.
    This was derived by exhaustively searching for the optimal quantizer for
     the AC coefficients in each block from a number of test sequences for a
     number of fixed lambda values and fitting the peaks of the resulting
     histograms (on the log(qavg) scale).
    The same model applies to both inter and intra frames.
    A more adaptive scheme might perform better.*/
  qi=_enc->state.qis[0];
  /*If rate control is active, use the lambda for the _target_ quantizer.
    This allows us to scale to rates slightly lower than we'd normally be able
     to reach, and give the rate control a semblance of "fractional qi"
     precision.
    TODO: Add API for changing QI, and allow extra precision.*/
  if(_enc->state.info.target_bitrate>0)lq=_enc->rc.log_qtarget;
  else lq=_enc->log_qavg[_qti][qi];
  /*The resulting lambda value is less than 0x500000.*/
  _enc->lambda=(int)oc_bexp64(2*lq-0x4780BD468D6B62BLL);
  /*Select additional quantizers.
    The R-D optimal block AC quantizer statistics suggest that the distribution
     is roughly Gaussian-like with a slight positive skew.
    K-means clustering on log_qavg to select 3 quantizers produces cluster
     centers of {log_qavg-0.6,log_qavg,log_qavg+0.7}.
    Experiments confirm these are relatively good choices.

    Although we do greedy R-D optimization of the qii flags to avoid switching
     too frequently, this becomes ineffective at low rates, either because we
     do a poor job of predicting the actual R-D cost, or the greedy
     optimization is not sufficient.
    Therefore adaptive quantization is disabled above an (experimentally
     suggested) threshold of log_qavg=7.00 (e.g., below INTRA qi=12 or
     INTER qi=20 with current matrices).
    This may need to be revised if the R-D cost estimation or qii flag
     optimization strategies change.*/
  nqis=1;
  if(lq<(OC_Q57(56)>>3)&&!_enc->vp3_compatible&&
   _enc->sp_level<OC_SP_LEVEL_FAST_ANALYSIS){
    qi1=oc_enc_find_qi_for_target(_enc,_qti,OC_MAXI(qi-1,0),0,
     lq+(OC_Q57(7)+5)/10);
    if(qi1!=qi)_enc->state.qis[nqis++]=qi1;
    qi1=oc_enc_find_qi_for_target(_enc,_qti,OC_MINI(qi+1,63),0,
     lq-(OC_Q57(6)+5)/10);
    if(qi1!=qi&&qi1!=_enc->state.qis[nqis-1])_enc->state.qis[nqis++]=qi1;
  }
  _enc->state.nqis=nqis;
}

/*Binary exponential of _log_scale with 24-bit fractional precision and
   saturation.
  _log_scale: A binary logarithm in Q24 format.
  Return: The binary exponential in Q24 format, saturated to 2**47-1 if
   _log_scale was too large.*/
static ogg_int64_t oc_bexp_q24(ogg_int32_t _log_scale){
  if(_log_scale<(ogg_int32_t)23<<24){
    ogg_int64_t ret;
    ret=oc_bexp64(((ogg_int64_t)_log_scale<<33)+OC_Q57(24));
    return ret<0x7FFFFFFFFFFFLL?ret:0x7FFFFFFFFFFFLL;
  }
  return 0x7FFFFFFFFFFFLL;
}

/*Convenience function converts Q57 value to a clamped 32-bit Q24 value
  _in: input in Q57 format.
  Return: same number in Q24 */
static ogg_int32_t oc_q57_to_q24(ogg_int64_t _in){
  ogg_int64_t ret;
  ret=_in+((ogg_int64_t)1<<32)>>33;
  /*0x80000000 is automatically converted to unsigned on 32-bit systems.
    -0x7FFFFFFF-1 is needed to avoid "promoting" the whole expression to
    unsigned.*/
  return (ogg_int32_t)OC_CLAMPI(-0x7FFFFFFF-1,ret,0x7FFFFFFF);
}

/*Binary exponential of _log_scale with 24-bit fractional precision and
   saturation.
  _log_scale: A binary logarithm in Q57 format.
  Return: The binary exponential in Q24 format, saturated to 2**31-1 if
   _log_scale was too large.*/
static ogg_int32_t oc_bexp64_q24(ogg_int64_t _log_scale){
  if(_log_scale<OC_Q57(8)){
    ogg_int64_t ret;
    ret=oc_bexp64(_log_scale+OC_Q57(24));
    return ret<0x7FFFFFFF?(ogg_int32_t)ret:0x7FFFFFFF;
  }
  return 0x7FFFFFFF;
}


static void oc_enc_rc_reset(oc_enc_ctx *_enc){
  ogg_int64_t npixels;
  ogg_int64_t ibpp;
  int         inter_delay;
  /*TODO: These parameters should be exposed in a th_encode_ctl() API.*/
  _enc->rc.bits_per_frame=(_enc->state.info.target_bitrate*
   (ogg_int64_t)_enc->state.info.fps_denominator)/
   _enc->state.info.fps_numerator;
  /*Insane framerates or frame sizes mean insane bitrates.
    Let's not get carried away.*/
  if(_enc->rc.bits_per_frame>0x400000000000LL){
    _enc->rc.bits_per_frame=(ogg_int64_t)0x400000000000LL;
  }
  else if(_enc->rc.bits_per_frame<32)_enc->rc.bits_per_frame=32;
  _enc->rc.buf_delay=OC_MAXI(_enc->rc.buf_delay,12);
  _enc->rc.max=_enc->rc.bits_per_frame*_enc->rc.buf_delay;
  /*Start with a buffer fullness of 50% plus 25% of the amount we plan to spend
     on a single keyframe interval.
    We can require fully half the bits in an interval for a keyframe, so this
     initial level gives us maximum flexibility for over/under-shooting in
     subsequent frames.*/
  _enc->rc.target=(_enc->rc.max+1>>1)+(_enc->rc.bits_per_frame+2>>2)*
   OC_MINI(_enc->keyframe_frequency_force,_enc->rc.buf_delay);
  _enc->rc.fullness=_enc->rc.target;
  /*Pick exponents and initial scales for quantizer selection.*/
  npixels=_enc->state.info.frame_width*
   (ogg_int64_t)_enc->state.info.frame_height;
  _enc->rc.log_npixels=oc_blog64(npixels);
  ibpp=npixels/_enc->rc.bits_per_frame;
  if(ibpp<1){
    _enc->rc.exp[0]=59;
    _enc->rc.log_scale[0]=oc_blog64(1997)-OC_Q57(8);
  }
  else if(ibpp<2){
    _enc->rc.exp[0]=55;
    _enc->rc.log_scale[0]=oc_blog64(1604)-OC_Q57(8);
  }
  else{
    _enc->rc.exp[0]=48;
    _enc->rc.log_scale[0]=oc_blog64(834)-OC_Q57(8);
  }
  if(ibpp<4){
    _enc->rc.exp[1]=100;
    _enc->rc.log_scale[1]=oc_blog64(2249)-OC_Q57(8);
  }
  else if(ibpp<8){
    _enc->rc.exp[1]=95;
    _enc->rc.log_scale[1]=oc_blog64(1751)-OC_Q57(8);
  }
  else{
    _enc->rc.exp[1]=73;
    _enc->rc.log_scale[1]=oc_blog64(1260)-OC_Q57(8);
  }
  _enc->rc.prev_drop_count=0;
  _enc->rc.log_drop_scale=OC_Q57(0);
  /*Set up second order followers, initialized according to corresponding
     time constants.*/
  oc_iir_filter_init(&_enc->rc.scalefilter[0],4,
   oc_q57_to_q24(_enc->rc.log_scale[0]));
  inter_delay=(_enc->rc.twopass?
   OC_MAXI(_enc->keyframe_frequency_force,12):_enc->rc.buf_delay)>>1;
  _enc->rc.inter_count=0;
  /*We clamp the actual inter_delay to a minimum of 10 to work within the range
     of values where later incrementing the delay works as designed.
    10 is not an exact choice, but rather a good working trade-off.*/
  _enc->rc.inter_delay=10;
  _enc->rc.inter_delay_target=inter_delay;
  oc_iir_filter_init(&_enc->rc.scalefilter[1],_enc->rc.inter_delay,
   oc_q57_to_q24(_enc->rc.log_scale[1]));
  oc_iir_filter_init(&_enc->rc.vfrfilter,4,
   oc_bexp64_q24(_enc->rc.log_drop_scale));
}

void oc_rc_state_init(oc_rc_state *_rc,oc_enc_ctx *_enc){
  _rc->twopass=0;
  _rc->twopass_buffer_bytes=0;
  _rc->twopass_force_kf=0;
  _rc->frame_metrics=NULL;
  _rc->rate_bias=0;
  if(_enc->state.info.target_bitrate>0){
    /*The buffer size is set equal to the keyframe interval, clamped to the
       range [12,256] frames.
      The 12 frame minimum gives us some chance to distribute bit estimation
       errors.
      The 256 frame maximum means we'll require 8-10 seconds of pre-buffering
       at 24-30 fps, which is not unreasonable.*/
    _rc->buf_delay=_enc->keyframe_frequency_force>256?
     256:_enc->keyframe_frequency_force;
    /*By default, enforce all buffer constraints.*/
    _rc->drop_frames=1;
    _rc->cap_overflow=1;
    _rc->cap_underflow=0;
    oc_enc_rc_reset(_enc);
  }
}

void oc_rc_state_clear(oc_rc_state *_rc){
  _ogg_free(_rc->frame_metrics);
}

void oc_enc_rc_resize(oc_enc_ctx *_enc){
  /*If encoding has not yet begun, reset the buffer state.*/
  if(_enc->state.curframe_num<0)oc_enc_rc_reset(_enc);
  else{
    int idt;
    /*Otherwise, update the bounds on the buffer, but not the current
       fullness.*/
    _enc->rc.bits_per_frame=(_enc->state.info.target_bitrate*
     (ogg_int64_t)_enc->state.info.fps_denominator)/
     _enc->state.info.fps_numerator;
    /*Insane framerates or frame sizes mean insane bitrates.
      Let's not get carried away.*/
    if(_enc->rc.bits_per_frame>0x400000000000LL){
      _enc->rc.bits_per_frame=(ogg_int64_t)0x400000000000LL;
    }
    else if(_enc->rc.bits_per_frame<32)_enc->rc.bits_per_frame=32;
    _enc->rc.buf_delay=OC_MAXI(_enc->rc.buf_delay,12);
    _enc->rc.max=_enc->rc.bits_per_frame*_enc->rc.buf_delay;
    _enc->rc.target=(_enc->rc.max+1>>1)+(_enc->rc.bits_per_frame+2>>2)*
     OC_MINI(_enc->keyframe_frequency_force,_enc->rc.buf_delay);
    /*Update the INTER-frame scale filter delay.
      We jump to it immediately if we've already seen enough frames; otherwise
       it is simply set as the new target.*/
    _enc->rc.inter_delay_target=idt=OC_MAXI(_enc->rc.buf_delay>>1,10);
    if(idt<OC_MINI(_enc->rc.inter_delay,_enc->rc.inter_count)){
      oc_iir_filter_init(&_enc->rc.scalefilter[1],idt,
       _enc->rc.scalefilter[1].y[0]);
      _enc->rc.inter_delay=idt;
    }
  }
  /*If we're in pass-2 mode, make sure the frame metrics array is big enough
     to hold frame statistics for the full buffer.*/
  if(_enc->rc.twopass==2){
    int cfm;
    int buf_delay;
    int reset_window;
    buf_delay=_enc->rc.buf_delay;
    reset_window=_enc->rc.frame_metrics==NULL&&(_enc->rc.frames_total[0]==0||
     buf_delay<_enc->rc.frames_total[0]+_enc->rc.frames_total[1]
     +_enc->rc.frames_total[2]);
    cfm=_enc->rc.cframe_metrics;
    /*Only try to resize the frame metrics buffer if a) it's too small and
       b) we were using a finite buffer, or are about to start.*/
    if(cfm<buf_delay&&(_enc->rc.frame_metrics!=NULL||reset_window)){
      oc_frame_metrics *fm;
      int               nfm;
      int               fmh;
      fm=(oc_frame_metrics *)_ogg_realloc(_enc->rc.frame_metrics,
       buf_delay*sizeof(*_enc->rc.frame_metrics));
      if(fm==NULL){
        /*We failed to allocate a finite buffer.*/
        /*If we don't have a valid 2-pass header yet, just return; we'll reset
           the buffer size when we read the header.*/
        if(_enc->rc.frames_total[0]==0)return;
        /*Otherwise revert to the largest finite buffer previously set, or to
           whole-file buffering if we were still using that.*/
        _enc->rc.buf_delay=_enc->rc.frame_metrics!=NULL?
         cfm:_enc->rc.frames_total[0]+_enc->rc.frames_total[1]
         +_enc->rc.frames_total[2];
        oc_enc_rc_resize(_enc);
        return;
      }
      _enc->rc.frame_metrics=fm;
      _enc->rc.cframe_metrics=buf_delay;
      /*Re-organize the circular buffer.*/
      fmh=_enc->rc.frame_metrics_head;
      nfm=_enc->rc.nframe_metrics;
      if(fmh+nfm>cfm){
        int shift;
        shift=OC_MINI(fmh+nfm-cfm,buf_delay-cfm);
        memcpy(fm+cfm,fm,OC_MINI(fmh+nfm-cfm,buf_delay-cfm)*sizeof(*fm));
        if(fmh+nfm>buf_delay)memmove(fm,fm+shift,fmh+nfm-buf_delay);
      }
    }
    /*We were using whole-file buffering; now we're not.*/
    if(reset_window){
      _enc->rc.nframes[0]=_enc->rc.nframes[1]=_enc->rc.nframes[2]=0;
      _enc->rc.scale_sum[0]=_enc->rc.scale_sum[1]=0;
      _enc->rc.scale_window_end=_enc->rc.scale_window0=
       _enc->state.curframe_num+_enc->prev_dup_count+1;
      if(_enc->rc.twopass_buffer_bytes){
        int qti;
        /*We already read the metrics for the first frame in the window.*/
        *(_enc->rc.frame_metrics)=*&_enc->rc.cur_metrics;
        _enc->rc.nframe_metrics++;
        qti=_enc->rc.cur_metrics.frame_type;
        _enc->rc.nframes[qti]++;
        _enc->rc.nframes[2]+=_enc->rc.cur_metrics.dup_count;
        _enc->rc.scale_sum[qti]+=oc_bexp_q24(_enc->rc.cur_metrics.log_scale);
        _enc->rc.scale_window_end+=_enc->rc.cur_metrics.dup_count+1;
        if(_enc->rc.scale_window_end-_enc->rc.scale_window0<buf_delay){
          /*We need more frame data.*/
          _enc->rc.twopass_buffer_bytes=0;
        }
      }
    }
    /*Otherwise, we could shrink the size of the current window, if necessary,
       but leaving it like it is lets us adapt to the new buffer size more
       gracefully.*/
  }
}

/*Scale the number of frames by the number of expected drops/duplicates.*/
static int oc_rc_scale_drop(oc_rc_state *_rc,int _nframes){
  if(_rc->prev_drop_count>0||_rc->log_drop_scale>OC_Q57(0)){
    ogg_int64_t dup_scale;
    dup_scale=oc_bexp64((_rc->log_drop_scale
     +oc_blog64(_rc->prev_drop_count+1)>>1)+OC_Q57(8));
    if(dup_scale<_nframes<<8){
      int dup_scalei;
      dup_scalei=(int)dup_scale;
      if(dup_scalei>0)_nframes=((_nframes<<8)+dup_scalei-1)/dup_scalei;
    }
    else _nframes=!!_nframes;
  }
  return _nframes;
}

int oc_enc_select_qi(oc_enc_ctx *_enc,int _qti,int _clamp){
  ogg_int64_t  rate_total;
  ogg_int64_t  rate_bias;
  int          nframes[2];
  int          buf_delay;
  int          buf_pad;
  ogg_int64_t  log_qtarget;
  ogg_int64_t  log_scale0;
  ogg_int64_t  log_cur_scale;
  ogg_int64_t  log_qexp;
  int          exp0;
  int          old_qi;
  int          qi;
  /*Figure out how to re-distribute bits so that we hit our fullness target
     before the last keyframe in our current buffer window (after the current
     frame), or the end of the buffer window, whichever comes first.*/
  log_cur_scale=(ogg_int64_t)_enc->rc.scalefilter[_qti].y[0]<<33;
  buf_pad=0;
  switch(_enc->rc.twopass){
    default:{
      ogg_uint32_t next_key_frame;
      /*Single pass mode: assume only forced keyframes and attempt to estimate
         the drop count for VFR content.*/
      next_key_frame=_qti?_enc->keyframe_frequency_force
       -(_enc->state.curframe_num-_enc->state.keyframe_num):0;
      nframes[0]=(_enc->rc.buf_delay-OC_MINI(next_key_frame,_enc->rc.buf_delay)
       +_enc->keyframe_frequency_force-1)/_enc->keyframe_frequency_force;
      if(nframes[0]+_qti>1){
        nframes[0]--;
        buf_delay=next_key_frame+nframes[0]*_enc->keyframe_frequency_force;
      }
      else buf_delay=_enc->rc.buf_delay;
      nframes[1]=buf_delay-nframes[0];
      /*Downgrade the delta frame rate to correspond to the recent drop count
         history.*/
      nframes[1]=oc_rc_scale_drop(&_enc->rc,nframes[1]);
    }break;
    case 1:{
      /*Pass 1 mode: use a fixed qi value.*/
      qi=_enc->state.qis[0];
      _enc->rc.log_qtarget=_enc->log_qavg[_qti][qi];
      return qi;
    }break;
    case 2:{
      ogg_int64_t scale_sum[2];
      int         qti;
      /*Pass 2 mode: we know exactly how much of each frame type there is in
         the current buffer window, and have estimates for the scales.*/
      nframes[0]=_enc->rc.nframes[0];
      nframes[1]=_enc->rc.nframes[1];
      scale_sum[0]=_enc->rc.scale_sum[0];
      scale_sum[1]=_enc->rc.scale_sum[1];
      /*The window size can be slightly larger than the buffer window for VFR
         content; clamp it down, if appropriate (the excess will all be dup
         frames).*/
      buf_delay=OC_MINI(_enc->rc.scale_window_end-_enc->rc.scale_window0,
       _enc->rc.buf_delay);
      /*If we're approaching the end of the file, add some slack to keep us
         from slamming into a rail.
        Our rate accuracy goes down, but it keeps the result sensible.
        We position the target where the first forced keyframe beyond the end
         of the file would be (for consistency with 1-pass mode).*/
      buf_pad=OC_MINI(_enc->rc.buf_delay,_enc->state.keyframe_num
       +_enc->keyframe_frequency_force-_enc->rc.scale_window0);
      if(buf_delay<buf_pad)buf_pad-=buf_delay;
      else{
        /*Otherwise, search for the last keyframe in the buffer window and
           target that.*/
        buf_pad=0;
        /*TODO: Currently we only do this when using a finite buffer; we could
           save the position of the last keyframe in the summary data and do it
           with a whole-file buffer as well, but it isn't likely to make a
           difference.*/
        if(_enc->rc.frame_metrics!=NULL){
          int fmi;
          int fm_tail;
          fm_tail=_enc->rc.frame_metrics_head+_enc->rc.nframe_metrics;
          if(fm_tail>=_enc->rc.cframe_metrics)fm_tail-=_enc->rc.cframe_metrics;
          for(fmi=fm_tail;;){
            oc_frame_metrics *m;
            fmi--;
            if(fmi<0)fmi+=_enc->rc.cframe_metrics;
            /*Stop before we remove the first frame.*/
            if(fmi==_enc->rc.frame_metrics_head)break;
            m=_enc->rc.frame_metrics+fmi;
            /*If we find a keyframe, remove it and everything past it.*/
            if(m->frame_type==OC_INTRA_FRAME){
              do{
                qti=m->frame_type;
                nframes[qti]--;
                scale_sum[qti]-=oc_bexp_q24(m->log_scale);
                buf_delay-=m->dup_count+1;
                fmi++;
                if(fmi>=_enc->rc.cframe_metrics)fmi=0;
                m=_enc->rc.frame_metrics+fmi;
              }
              while(fmi!=fm_tail);
              /*And stop scanning backwards.*/
              break;
            }
          }
        }
      }
      /*If we're not using the same frame type as in pass 1 (because someone
         changed the keyframe interval), remove that scale estimate.
        We'll add in a replacement for the correct frame type below.*/
      qti=_enc->rc.cur_metrics.frame_type;
      if(qti!=_qti){
        nframes[qti]--;
        scale_sum[qti]-=oc_bexp_q24(_enc->rc.cur_metrics.log_scale);
      }
      /*Compute log_scale estimates for each frame type from the pass-1 scales
         we measured in the current window.*/
      for(qti=0;qti<2;qti++){
        _enc->rc.log_scale[qti]=nframes[qti]>0?
         oc_blog64(scale_sum[qti])-oc_blog64(nframes[qti])-OC_Q57(24):
         -_enc->rc.log_npixels;
      }
      /*If we're not using the same frame type as in pass 1, add a scale
         estimate for the corresponding frame using the current low-pass
         filter value.
        This is mostly to ensure we have a valid estimate even when pass 1 had
         no frames of this type in the buffer window.
        TODO: We could also plan ahead and figure out how many keyframes we'll
         be forced to add in the current buffer window.*/
      qti=_enc->rc.cur_metrics.frame_type;
      if(qti!=_qti){
        ogg_int64_t scale;
        scale=_enc->rc.log_scale[_qti]<OC_Q57(23)?
         oc_bexp64(_enc->rc.log_scale[_qti]+OC_Q57(24)):0x7FFFFFFFFFFFLL;
        scale*=nframes[_qti];
        nframes[_qti]++;
        scale+=oc_bexp_q24(log_cur_scale>>33);
        _enc->rc.log_scale[_qti]=oc_blog64(scale)
         -oc_blog64(nframes[qti])-OC_Q57(24);
      }
      else log_cur_scale=(ogg_int64_t)_enc->rc.cur_metrics.log_scale<<33;
      /*Add the padding from above.
        This basically reverts to 1-pass estimations in the last keyframe
         interval.*/
      if(buf_pad>0){
        ogg_int64_t scale;
        int         nextra_frames;
        /*Extend the buffer.*/
        buf_delay+=buf_pad;
        /*Add virtual delta frames according to the estimated drop count.*/
        nextra_frames=oc_rc_scale_drop(&_enc->rc,buf_pad);
        /*And blend in the low-pass filtered scale according to how many frames
           we added.*/
        scale=
         oc_bexp64(_enc->rc.log_scale[1]+OC_Q57(24))*(ogg_int64_t)nframes[1]
         +oc_bexp_q24(_enc->rc.scalefilter[1].y[0])*(ogg_int64_t)nextra_frames;
        nframes[1]+=nextra_frames;
        _enc->rc.log_scale[1]=oc_blog64(scale)-oc_blog64(nframes[1])-OC_Q57(24);
      }
    }break;
  }
  /*If we've been missing our target, add a penalty term.*/
  rate_bias=(_enc->rc.rate_bias/(_enc->state.curframe_num+1000))*
   (buf_delay-buf_pad);
  /*rate_total is the total bits available over the next buf_delay frames.*/
  rate_total=_enc->rc.fullness-_enc->rc.target+rate_bias
   +buf_delay*_enc->rc.bits_per_frame;
  log_scale0=_enc->rc.log_scale[_qti]+_enc->rc.log_npixels;
  /*If there aren't enough bits to achieve our desired fullness level, use the
     minimum quality permitted.*/
  if(rate_total<=buf_delay)log_qtarget=OC_QUANT_MAX_LOG;
  else{
    static const ogg_int64_t LOG_KEY_RATIO=0x0137222BB70747BALL;
    ogg_int64_t log_scale1;
    ogg_int64_t rlo;
    ogg_int64_t rhi;
    log_scale1=_enc->rc.log_scale[1-_qti]+_enc->rc.log_npixels;
    rlo=0;
    rhi=(rate_total+nframes[_qti]-1)/nframes[_qti];
    while(rlo<rhi){
      ogg_int64_t curr;
      ogg_int64_t rdiff;
      ogg_int64_t log_rpow;
      ogg_int64_t rscale;
      curr=rlo+rhi>>1;
      log_rpow=oc_blog64(curr)-log_scale0;
      log_rpow=(log_rpow+(_enc->rc.exp[_qti]>>1))/_enc->rc.exp[_qti];
      if(_qti)log_rpow+=LOG_KEY_RATIO>>6;
      else log_rpow-=LOG_KEY_RATIO>>6;
      log_rpow*=_enc->rc.exp[1-_qti];
      rscale=nframes[1-_qti]*oc_bexp64(log_scale1+log_rpow);
      rdiff=nframes[_qti]*curr+rscale-rate_total;
      if(rdiff<0)rlo=curr+1;
      else if(rdiff>0)rhi=curr-1;
      else break;
    }
    log_qtarget=OC_Q57(2)-((oc_blog64(rlo)-log_scale0+(_enc->rc.exp[_qti]>>1))/
     _enc->rc.exp[_qti]<<6);
    log_qtarget=OC_MINI(log_qtarget,OC_QUANT_MAX_LOG);
  }
  /*The above allocation looks only at the total rate we'll accumulate in the
     next buf_delay frames.
    However, we could overflow the buffer on the very next frame, so check for
     that here, if we're not using a soft target.*/
  exp0=_enc->rc.exp[_qti];
  if(_enc->rc.cap_overflow){
    ogg_int64_t margin;
    ogg_int64_t soft_limit;
    ogg_int64_t log_soft_limit;
    /*Allow 3% of the buffer for prediction error.
      This should be plenty, and we don't mind if we go a bit over; we only
       want to keep these bits from being completely wasted.*/
    margin=_enc->rc.max+31>>5;
    /*We want to use at least this many bits next frame.*/
    soft_limit=_enc->rc.fullness+_enc->rc.bits_per_frame-(_enc->rc.max-margin);
    log_soft_limit=oc_blog64(soft_limit);
    /*If we're predicting we won't use that many...*/
    log_qexp=(log_qtarget-OC_Q57(2)>>6)*exp0;
    if(log_scale0-log_qexp<log_soft_limit){
      /*Scale the adjustment based on how far into the margin we are.*/
      log_qexp+=(log_scale0-log_soft_limit-log_qexp>>32)*
       ((OC_MINI(margin,soft_limit)<<32)/margin);
      log_qtarget=((log_qexp+(exp0>>1))/exp0<<6)+OC_Q57(2);
    }
  }
  /*If this was not one of the initial frames, limit the change in quality.*/
  old_qi=_enc->state.qis[0];
  if(_clamp){
    ogg_int64_t log_qmin;
    ogg_int64_t log_qmax;
    /*Clamp the target quantizer to within [0.8*Q,1.2*Q], where Q is the
       current quantizer.
      TODO: With user-specified quant matrices, we need to enlarge these limits
       if they don't actually let us change qi values.*/
    log_qmin=_enc->log_qavg[_qti][old_qi]-0x00A4D3C25E68DC58LL;
    log_qmax=_enc->log_qavg[_qti][old_qi]+0x00A4D3C25E68DC58LL;
    log_qtarget=OC_CLAMPI(log_qmin,log_qtarget,log_qmax);
  }
  /*The above allocation looks only at the total rate we'll accumulate in the
     next buf_delay frames.
    However, we could bust the budget on the very next frame, so check for that
     here, if we're not using a soft target.*/
  /* Disabled when our minimum qi > 0; if we saturate log_qtarget to
     to the maximum possible size when we have a minimum qi, the
     resulting lambda will interact very strangely with SKIP.  The
     resulting artifacts look like waterfalls. */
  if(_enc->state.info.quality==0){
    ogg_int64_t log_hard_limit;
    /*Compute the maximum number of bits we can use in the next frame.
      Allow 50% of the rate for a single frame for prediction error.
      This may not be enough for keyframes or sudden changes in complexity.*/
    log_hard_limit=oc_blog64(_enc->rc.fullness+(_enc->rc.bits_per_frame>>1));
    /*If we're predicting we'll use more than this...*/
    log_qexp=(log_qtarget-OC_Q57(2)>>6)*exp0;
    if(log_scale0-log_qexp>log_hard_limit){
      /*Force the target to hit our limit exactly.*/
      log_qexp=log_scale0-log_hard_limit;
      log_qtarget=((log_qexp+(exp0>>1))/exp0<<6)+OC_Q57(2);
      /*If that target is unreasonable, oh well; we'll have to drop.*/
      log_qtarget=OC_MINI(log_qtarget,OC_QUANT_MAX_LOG);
    }
  }
  /*Compute a final estimate of the number of bits we plan to use.*/
  log_qexp=(log_qtarget-OC_Q57(2)>>6)*_enc->rc.exp[_qti];
  _enc->rc.rate_bias+=oc_bexp64(log_cur_scale+_enc->rc.log_npixels-log_qexp);
  qi=oc_enc_find_qi_for_target(_enc,_qti,old_qi,
   _enc->state.info.quality,log_qtarget);
  /*Save the quantizer target for lambda calculations.*/
  _enc->rc.log_qtarget=log_qtarget;
  return qi;
}

int oc_enc_update_rc_state(oc_enc_ctx *_enc,
 long _bits,int _qti,int _qi,int _trial,int _droppable){
  ogg_int64_t buf_delta;
  ogg_int64_t log_scale;
  int         dropped;
  dropped=0;
  /* Drop frames also disabled for now in the case of infinite-buffer
     two-pass mode */
  if(!_enc->rc.drop_frames||_enc->rc.twopass&&_enc->rc.frame_metrics==NULL){
    _droppable=0;
  }
  buf_delta=_enc->rc.bits_per_frame*(1+_enc->dup_count);
  if(_bits<=0){
    /*We didn't code any blocks in this frame.*/
    log_scale=OC_Q57(-64);
    _bits=0;
  }
  else{
    ogg_int64_t log_bits;
    ogg_int64_t log_qexp;
    /*Compute the estimated scale factor for this frame type.*/
    log_bits=oc_blog64(_bits);
    log_qexp=_enc->rc.log_qtarget-OC_Q57(2);
    log_qexp=(log_qexp>>6)*(_enc->rc.exp[_qti]);
    log_scale=OC_MINI(log_bits-_enc->rc.log_npixels+log_qexp,OC_Q57(16));
  }
  /*Special two-pass processing.*/
  switch(_enc->rc.twopass){
    case 1:{
      /*Pass 1 mode: save the metrics for this frame.*/
      _enc->rc.cur_metrics.log_scale=oc_q57_to_q24(log_scale);
      _enc->rc.cur_metrics.dup_count=_enc->dup_count;
      _enc->rc.cur_metrics.frame_type=_enc->state.frame_type;
      _enc->rc.cur_metrics.activity_avg=_enc->activity_avg;
      _enc->rc.twopass_buffer_bytes=0;
    }break;
    case 2:{
      /*Pass 2 mode:*/
      if(!_trial){
        ogg_int64_t next_frame_num;
        int         qti;
        /*Move the current metrics back one frame.*/
        *&_enc->rc.prev_metrics=*&_enc->rc.cur_metrics;
        next_frame_num=_enc->state.curframe_num+_enc->dup_count+1;
        /*Back out the last frame's statistics from the sliding window.*/
        qti=_enc->rc.prev_metrics.frame_type;
        _enc->rc.frames_left[qti]--;
        _enc->rc.frames_left[2]-=_enc->rc.prev_metrics.dup_count;
        _enc->rc.nframes[qti]--;
        _enc->rc.nframes[2]-=_enc->rc.prev_metrics.dup_count;
        _enc->rc.scale_sum[qti]-=oc_bexp_q24(_enc->rc.prev_metrics.log_scale);
        _enc->rc.scale_window0=(int)next_frame_num;
        /*Free the corresponding entry in the circular buffer.*/
        if(_enc->rc.frame_metrics!=NULL){
          _enc->rc.nframe_metrics--;
          _enc->rc.frame_metrics_head++;
          if(_enc->rc.frame_metrics_head>=_enc->rc.cframe_metrics){
            _enc->rc.frame_metrics_head=0;
          }
        }
        /*Mark us ready for the next 2-pass packet.*/
        _enc->rc.twopass_buffer_bytes=0;
        /*Update state, so the user doesn't have to keep calling 2pass_in after
           they've fed in all the data when we're using a finite buffer.*/
        _enc->prev_dup_count=_enc->dup_count;
        oc_enc_rc_2pass_in(_enc,NULL,0);
      }
    }break;
  }
  /*Common to all passes:*/
  if(_bits>0){
    if(_trial){
      oc_iir_filter *f;
      /*Use the estimated scale factor directly if this was a trial.*/
      f=_enc->rc.scalefilter+_qti;
      f->y[1]=f->y[0]=f->x[1]=f->x[0]=oc_q57_to_q24(log_scale);
      _enc->rc.log_scale[_qti]=log_scale;
    }
    else{
      /*Lengthen the time constant for the INTER filter as we collect more
         frame statistics, until we reach our target.*/
      if(_enc->rc.inter_delay<_enc->rc.inter_delay_target&&
       _enc->rc.inter_count>=_enc->rc.inter_delay&&_qti==OC_INTER_FRAME){
        oc_iir_filter_reinit(&_enc->rc.scalefilter[1],++_enc->rc.inter_delay);
      }
      /*Otherwise update the low-pass scale filter for this frame type,
         regardless of whether or not we dropped this frame.*/
      _enc->rc.log_scale[_qti]=oc_iir_filter_update(
       _enc->rc.scalefilter+_qti,oc_q57_to_q24(log_scale))<<33;
      /*If this frame busts our budget, it must be dropped.*/
      if(_droppable&&_enc->rc.fullness+buf_delta<_bits){
        _enc->rc.prev_drop_count+=1+_enc->dup_count;
        _bits=0;
        dropped=1;
      }
      else{
        ogg_uint32_t drop_count;
        /*Update a low-pass filter to estimate the "real" frame rate taking
           drops and duplicates into account.
          This is only done if the frame is coded, as it needs the final
           count of dropped frames.*/
        drop_count=_enc->rc.prev_drop_count+1;
        if(drop_count>0x7F)drop_count=0x7FFFFFFF;
        else drop_count<<=24;
        _enc->rc.log_drop_scale=oc_blog64(oc_iir_filter_update(
         &_enc->rc.vfrfilter,drop_count))-OC_Q57(24);
        /*Initialize the drop count for this frame to the user-requested dup
           count.
          It will be increased if we drop more frames.*/
        _enc->rc.prev_drop_count=_enc->dup_count;
      }
    }
    /*Increment the INTER frame count, for filter adaptation purposes.*/
    if(_enc->rc.inter_count<INT_MAX)_enc->rc.inter_count+=_qti;
  }
  /*Increase the drop count.*/
  else _enc->rc.prev_drop_count+=1+_enc->dup_count;
  /*And update the buffer fullness level.*/
  if(!_trial){
    _enc->rc.fullness+=buf_delta-_bits;
    /*If we're too quick filling the buffer and overflow is capped,
      that rate is lost forever.*/
    if(_enc->rc.cap_overflow&&_enc->rc.fullness>_enc->rc.max){
      _enc->rc.fullness=_enc->rc.max;
    }
    /*If we're too quick draining the buffer and underflow is capped,
      don't try to make up that rate later.*/
    if(_enc->rc.cap_underflow&&_enc->rc.fullness<0){
      _enc->rc.fullness=0;
    }
    /*Adjust the bias for the real bits we've used.*/
    _enc->rc.rate_bias-=_bits;
  }
  return dropped;
}

#define OC_RC_2PASS_VERSION   (2)
#define OC_RC_2PASS_HDR_SZ    (38)
#define OC_RC_2PASS_PACKET_SZ (12)

static void oc_rc_buffer_val(oc_rc_state *_rc,ogg_int64_t _val,int _bytes){
  while(_bytes-->0){
    _rc->twopass_buffer[_rc->twopass_buffer_bytes++]=(unsigned char)(_val&0xFF);
    _val>>=8;
  }
}

int oc_enc_rc_2pass_out(oc_enc_ctx *_enc,unsigned char **_buf){
  if(_enc->rc.twopass_buffer_bytes==0){
    if(_enc->rc.twopass==0){
      int qi;
      /*Pick first-pass qi for scale calculations.*/
      qi=oc_enc_select_qi(_enc,0,0);
      _enc->state.nqis=1;
      _enc->state.qis[0]=qi;
      _enc->rc.twopass=1;
      _enc->rc.frames_total[0]=_enc->rc.frames_total[1]=
       _enc->rc.frames_total[2]=0;
      _enc->rc.scale_sum[0]=_enc->rc.scale_sum[1]=0;
      /*Fill in dummy summary values.*/
      oc_rc_buffer_val(&_enc->rc,0x5032544F,4);
      oc_rc_buffer_val(&_enc->rc,OC_RC_2PASS_VERSION,4);
      oc_rc_buffer_val(&_enc->rc,0,OC_RC_2PASS_HDR_SZ-8);
    }
    else{
      int qti;
      qti=_enc->rc.cur_metrics.frame_type;
      _enc->rc.scale_sum[qti]+=oc_bexp_q24(_enc->rc.cur_metrics.log_scale);
      _enc->rc.frames_total[qti]++;
      _enc->rc.frames_total[2]+=_enc->rc.cur_metrics.dup_count;
      oc_rc_buffer_val(&_enc->rc,
       _enc->rc.cur_metrics.dup_count|_enc->rc.cur_metrics.frame_type<<31,4);
      oc_rc_buffer_val(&_enc->rc,_enc->rc.cur_metrics.log_scale,4);
      oc_rc_buffer_val(&_enc->rc,_enc->rc.cur_metrics.activity_avg,4);
    }
  }
  else if(_enc->packet_state==OC_PACKET_DONE&&
   _enc->rc.twopass_buffer_bytes!=OC_RC_2PASS_HDR_SZ){
    _enc->rc.twopass_buffer_bytes=0;
    oc_rc_buffer_val(&_enc->rc,0x5032544F,4);
    oc_rc_buffer_val(&_enc->rc,OC_RC_2PASS_VERSION,4);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.frames_total[0],4);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.frames_total[1],4);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.frames_total[2],4);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.exp[0],1);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.exp[1],1);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.scale_sum[0],8);
    oc_rc_buffer_val(&_enc->rc,_enc->rc.scale_sum[1],8);
  }
  else{
    /*The data for this frame has already been retrieved.*/
    *_buf=NULL;
    return 0;
  }
  *_buf=_enc->rc.twopass_buffer;
  return _enc->rc.twopass_buffer_bytes;
}

static size_t oc_rc_buffer_fill(oc_rc_state *_rc,
 unsigned char *_buf,size_t _bytes,size_t _consumed,size_t _goal){
  while(_rc->twopass_buffer_fill<_goal&&_consumed<_bytes){
    _rc->twopass_buffer[_rc->twopass_buffer_fill++]=_buf[_consumed++];
  }
  return _consumed;
}

static ogg_int64_t oc_rc_unbuffer_val(oc_rc_state *_rc,int _bytes){
  ogg_int64_t ret;
  int         shift;
  ret=0;
  shift=0;
  while(_bytes-->0){
    ret|=((ogg_int64_t)_rc->twopass_buffer[_rc->twopass_buffer_bytes++])<<shift;
    shift+=8;
  }
  return ret;
}

int oc_enc_rc_2pass_in(oc_enc_ctx *_enc,unsigned char *_buf,size_t _bytes){
  size_t consumed;
  consumed=0;
  /*Enable pass 2 mode if this is the first call.*/
  if(_enc->rc.twopass==0){
    _enc->rc.twopass=2;
    _enc->rc.twopass_buffer_fill=0;
    _enc->rc.frames_total[0]=0;
    _enc->rc.nframe_metrics=0;
    _enc->rc.cframe_metrics=0;
    _enc->rc.frame_metrics_head=0;
    _enc->rc.scale_window0=0;
    _enc->rc.scale_window_end=0;
  }
  /*If we haven't got a valid summary header yet, try to parse one.*/
  if(_enc->rc.frames_total[0]==0){
    if(!_buf){
      int frames_needed;
      /*If we're using a whole-file buffer, we just need the first frame.
        Otherwise, we may need as many as one per buffer slot.*/
      frames_needed=_enc->rc.frame_metrics==NULL?1:_enc->rc.buf_delay;
      return OC_RC_2PASS_HDR_SZ+frames_needed*OC_RC_2PASS_PACKET_SZ
       -_enc->rc.twopass_buffer_fill;
    }
    consumed=oc_rc_buffer_fill(&_enc->rc,
     _buf,_bytes,consumed,OC_RC_2PASS_HDR_SZ);
    if(_enc->rc.twopass_buffer_fill>=OC_RC_2PASS_HDR_SZ){
      ogg_int64_t scale_sum[2];
      int         exp[2];
      int         buf_delay;
      /*Read the summary header data.*/
      /*Check the magic value and version number.*/
      if(oc_rc_unbuffer_val(&_enc->rc,4)!=0x5032544F||
       oc_rc_unbuffer_val(&_enc->rc,4)!=OC_RC_2PASS_VERSION){
        _enc->rc.twopass_buffer_bytes=0;
        return TH_ENOTFORMAT;
      }
      _enc->rc.frames_total[0]=(ogg_uint32_t)oc_rc_unbuffer_val(&_enc->rc,4);
      _enc->rc.frames_total[1]=(ogg_uint32_t)oc_rc_unbuffer_val(&_enc->rc,4);
      _enc->rc.frames_total[2]=(ogg_uint32_t)oc_rc_unbuffer_val(&_enc->rc,4);
      exp[0]=(int)oc_rc_unbuffer_val(&_enc->rc,1);
      exp[1]=(int)oc_rc_unbuffer_val(&_enc->rc,1);
      scale_sum[0]=oc_rc_unbuffer_val(&_enc->rc,8);
      scale_sum[1]=oc_rc_unbuffer_val(&_enc->rc,8);
      /*Make sure the file claims to have at least one frame.
        Otherwise we probably got the placeholder data from an aborted pass 1.
        Also make sure the total frame count doesn't overflow an integer.*/
      buf_delay=_enc->rc.frames_total[0]+_enc->rc.frames_total[1]
       +_enc->rc.frames_total[2];
      if(_enc->rc.frames_total[0]==0||buf_delay<0||
       (ogg_uint32_t)buf_delay<_enc->rc.frames_total[0]||
       (ogg_uint32_t)buf_delay<_enc->rc.frames_total[1]){
        _enc->rc.frames_total[0]=0;
        _enc->rc.twopass_buffer_bytes=0;
        return TH_EBADHEADER;
      }
      /*Got a valid header; set up pass 2.*/
      _enc->rc.frames_left[0]=_enc->rc.frames_total[0];
      _enc->rc.frames_left[1]=_enc->rc.frames_total[1];
      _enc->rc.frames_left[2]=_enc->rc.frames_total[2];
      /*If the user hasn't specified a buffer size, use the whole file.*/
      if(_enc->rc.frame_metrics==NULL){
        _enc->rc.buf_delay=buf_delay;
        _enc->rc.nframes[0]=_enc->rc.frames_total[0];
        _enc->rc.nframes[1]=_enc->rc.frames_total[1];
        _enc->rc.nframes[2]=_enc->rc.frames_total[2];
        _enc->rc.scale_sum[0]=scale_sum[0];
        _enc->rc.scale_sum[1]=scale_sum[1];
        _enc->rc.scale_window_end=buf_delay;
        oc_enc_rc_reset(_enc);
      }
      _enc->rc.exp[0]=exp[0];
      _enc->rc.exp[1]=exp[1];
      /*Clear the header data from the buffer to make room for packet data.*/
      _enc->rc.twopass_buffer_fill=0;
      _enc->rc.twopass_buffer_bytes=0;
    }
  }
  if(_enc->rc.frames_total[0]!=0){
    ogg_int64_t curframe_num;
    int         nframes_total;
    curframe_num=_enc->state.curframe_num;
    if(curframe_num>=0){
      /*We just encoded a frame; make sure things matched.*/
      if(_enc->rc.prev_metrics.dup_count!=_enc->prev_dup_count){
        _enc->rc.twopass_buffer_bytes=0;
        return TH_EINVAL;
      }
    }
    curframe_num+=_enc->prev_dup_count+1;
    nframes_total=_enc->rc.frames_total[0]+_enc->rc.frames_total[1]
     +_enc->rc.frames_total[2];
    if(curframe_num>=nframes_total){
      /*We don't want any more data after the last frame, and we don't want to
         allow any more frames to be encoded.*/
      _enc->rc.twopass_buffer_bytes=0;
    }
    else if(_enc->rc.twopass_buffer_bytes==0){
      if(_enc->rc.frame_metrics==NULL){
        /*We're using a whole-file buffer:*/
        if(!_buf)return OC_RC_2PASS_PACKET_SZ-_enc->rc.twopass_buffer_fill;
        consumed=oc_rc_buffer_fill(&_enc->rc,
         _buf,_bytes,consumed,OC_RC_2PASS_PACKET_SZ);
        if(_enc->rc.twopass_buffer_fill>=OC_RC_2PASS_PACKET_SZ){
          ogg_uint32_t dup_count;
          ogg_int32_t  log_scale;
          unsigned     activity;
          int          qti;
          int          arg;
          /*Read the metrics for the next frame.*/
          dup_count=oc_rc_unbuffer_val(&_enc->rc,4);
          log_scale=oc_rc_unbuffer_val(&_enc->rc,4);
          activity=oc_rc_unbuffer_val(&_enc->rc,4);
          _enc->rc.cur_metrics.log_scale=log_scale;
          qti=(dup_count&0x80000000)>>31;
          _enc->rc.cur_metrics.dup_count=dup_count&0x7FFFFFFF;
          _enc->rc.cur_metrics.frame_type=qti;
          _enc->rc.twopass_force_kf=qti==OC_INTRA_FRAME;
          _enc->activity_avg=_enc->rc.cur_metrics.activity_avg=activity;
          /*"Helpfully" set the dup count back to what it was in pass 1.*/
          arg=_enc->rc.cur_metrics.dup_count;
          th_encode_ctl(_enc,TH_ENCCTL_SET_DUP_COUNT,&arg,sizeof(arg));
          /*Clear the buffer for the next frame.*/
          _enc->rc.twopass_buffer_fill=0;
        }
      }
      else{
        int frames_needed;
        /*We're using a finite buffer:*/
        frames_needed=OC_MINI(_enc->rc.buf_delay-OC_MINI(_enc->rc.buf_delay,
         _enc->rc.scale_window_end-_enc->rc.scale_window0),
         _enc->rc.frames_left[0]+_enc->rc.frames_left[1]
         -_enc->rc.nframes[0]-_enc->rc.nframes[1]);
        while(frames_needed>0){
          if(!_buf){
            return OC_RC_2PASS_PACKET_SZ*frames_needed
           -_enc->rc.twopass_buffer_fill;
          }
          consumed=oc_rc_buffer_fill(&_enc->rc,
           _buf,_bytes,consumed,OC_RC_2PASS_PACKET_SZ);
          if(_enc->rc.twopass_buffer_fill>=OC_RC_2PASS_PACKET_SZ){
            oc_frame_metrics *m;
            int               fmi;
            ogg_uint32_t      dup_count;
            ogg_int32_t       log_scale;
            int               qti;
            unsigned          activity;
            /*Read the metrics for the next frame.*/
            dup_count=oc_rc_unbuffer_val(&_enc->rc,4);
            log_scale=oc_rc_unbuffer_val(&_enc->rc,4);
            activity=oc_rc_unbuffer_val(&_enc->rc,4);
            /*Add the to the circular buffer.*/
            fmi=_enc->rc.frame_metrics_head+_enc->rc.nframe_metrics++;
            if(fmi>=_enc->rc.cframe_metrics)fmi-=_enc->rc.cframe_metrics;
            m=_enc->rc.frame_metrics+fmi;
            m->log_scale=log_scale;
            qti=(dup_count&0x80000000)>>31;
            m->dup_count=dup_count&0x7FFFFFFF;
            m->frame_type=qti;
            m->activity_avg=activity;
            /*And accumulate the statistics over the window.*/
            _enc->rc.nframes[qti]++;
            _enc->rc.nframes[2]+=m->dup_count;
            _enc->rc.scale_sum[qti]+=oc_bexp_q24(m->log_scale);
            _enc->rc.scale_window_end+=m->dup_count+1;
            /*Compute an upper bound on the number of remaining packets needed
               for the current window.*/
            frames_needed=OC_MINI(_enc->rc.buf_delay-OC_MINI(_enc->rc.buf_delay,
             _enc->rc.scale_window_end-_enc->rc.scale_window0),
             _enc->rc.frames_left[0]+_enc->rc.frames_left[1]
             -_enc->rc.nframes[0]-_enc->rc.nframes[1]);
            /*Clear the buffer for the next frame.*/
            _enc->rc.twopass_buffer_fill=0;
            _enc->rc.twopass_buffer_bytes=0;
          }
          /*Go back for more data.*/
          else break;
        }
        /*If we've got all the frames we need, fill in the current metrics.
          We're ready to go.*/
        if(frames_needed<=0){
          int arg;
          *&_enc->rc.cur_metrics=
           *(_enc->rc.frame_metrics+_enc->rc.frame_metrics_head);
          _enc->rc.twopass_force_kf=
           _enc->rc.cur_metrics.frame_type==OC_INTRA_FRAME;
          _enc->activity_avg=_enc->rc.cur_metrics.activity_avg;
          /*"Helpfully" set the dup count back to what it was in pass 1.*/
          arg=_enc->rc.cur_metrics.dup_count;
          th_encode_ctl(_enc,TH_ENCCTL_SET_DUP_COUNT,&arg,sizeof(arg));
          /*Mark us ready for the next frame.*/
          _enc->rc.twopass_buffer_bytes=1;
        }
      }
    }
  }
  return (int)consumed;
}
