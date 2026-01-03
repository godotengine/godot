/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2011                *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function: mode selection code

 ********************************************************************/
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "collect.h"

#if defined(OC_COLLECT_METRICS)

int              OC_HAS_MODE_METRICS;
double           OC_MODE_RD_WEIGHT_SATD[OC_LOGQ_BINS][3][2][OC_COMP_BINS];
double           OC_MODE_RD_WEIGHT_SAD[OC_LOGQ_BINS][3][2][OC_COMP_BINS];
oc_mode_metrics  OC_MODE_METRICS_SATD[OC_LOGQ_BINS-1][3][2][OC_COMP_BINS];
oc_mode_metrics  OC_MODE_METRICS_SAD[OC_LOGQ_BINS-1][3][2][OC_COMP_BINS];
const char      *OC_MODE_METRICS_FILENAME="modedec.stats";

void oc_mode_metrics_add(oc_mode_metrics *_metrics,
 double _w,int _s,int _q,int _r,double _d){
  if(_metrics->w>0){
    double ds;
    double dq;
    double dr;
    double dd;
    double ds2;
    double dq2;
    double s2;
    double sq;
    double q2;
    double sr;
    double qr;
    double sd;
    double qd;
    double s2q;
    double sq2;
    double w;
    double wa;
    double rwa;
    double rwa2;
    double rwb;
    double rwb2;
    double rw2;
    double rw3;
    double rw4;
    wa=_metrics->w;
    ds=_s-_metrics->s/wa;
    dq=_q-_metrics->q/wa;
    dr=_r-_metrics->r/wa;
    dd=_d-_metrics->d/wa;
    ds2=ds*ds;
    dq2=dq*dq;
    s2=_metrics->s2;
    sq=_metrics->sq;
    q2=_metrics->q2;
    sr=_metrics->sr;
    qr=_metrics->qr;
    sd=_metrics->sd;
    qd=_metrics->qd;
    s2q=_metrics->s2q;
    sq2=_metrics->sq2;
    w=wa+_w;
    rwa=wa/w;
    rwb=_w/w;
    rwa2=rwa*rwa;
    rwb2=rwb*rwb;
    rw2=wa*rwb;
    rw3=rw2*(rwa2-rwb2);
    rw4=_w*rwa2*rwa2+wa*rwb2*rwb2;
    _metrics->s2q2+=-2*(ds*sq2+dq*s2q)*rwb
     +(ds2*q2+4*ds*dq*sq+dq2*s2)*rwb2+ds2*dq2*rw4;
    _metrics->s2q+=(-2*ds*sq-dq*s2)*rwb+ds2*dq*rw3;
    _metrics->sq2+=(-ds*q2-2*dq*sq)*rwb+ds*dq2*rw3;
    _metrics->sqr+=(-ds*qr-dq*sr-dr*sq)*rwb+ds*dq*dr*rw3;
    _metrics->sqd+=(-ds*qd-dq*sd-dd*sq)*rwb+ds*dq*dd*rw3;
    _metrics->s2+=ds2*rw2;
    _metrics->sq+=ds*dq*rw2;
    _metrics->q2+=dq2*rw2;
    _metrics->sr+=ds*dr*rw2;
    _metrics->qr+=dq*dr*rw2;
    _metrics->r2+=dr*dr*rw2;
    _metrics->sd+=ds*dd*rw2;
    _metrics->qd+=dq*dd*rw2;
    _metrics->d2+=dd*dd*rw2;
  }
  _metrics->w+=_w;
  _metrics->s+=_s*_w;
  _metrics->q+=_q*_w;
  _metrics->r+=_r*_w;
  _metrics->d+=_d*_w;
}

void oc_mode_metrics_merge(oc_mode_metrics *_dst,
 const oc_mode_metrics *_src,int _n){
  int i;
  /*Find a non-empty set of metrics.*/
  for(i=0;i<_n&&_src[i].w==0;i++);
  if(i>=_n){
    memset(_dst,0,sizeof(*_dst));
    return;
  }
  memcpy(_dst,_src+i,sizeof(*_dst));
  /*And iterate over the remaining non-empty sets of metrics.*/
  for(i++;i<_n;i++)if(_src[i].w!=0){
    double ds;
    double dq;
    double dr;
    double dd;
    double ds2;
    double dq2;
    double s2a;
    double s2b;
    double sqa;
    double sqb;
    double q2a;
    double q2b;
    double sra;
    double srb;
    double qra;
    double qrb;
    double sda;
    double sdb;
    double qda;
    double qdb;
    double s2qa;
    double s2qb;
    double sq2a;
    double sq2b;
    double w;
    double wa;
    double wb;
    double rwa;
    double rwb;
    double rwa2;
    double rwb2;
    double rw2;
    double rw3;
    double rw4;
    wa=_dst->w;
    wb=_src[i].w;
    ds=_src[i].s/wb-_dst->s/wa;
    dq=_src[i].q/wb-_dst->q/wa;
    dr=_src[i].r/wb-_dst->r/wa;
    dd=_src[i].d/wb-_dst->d/wa;
    ds2=ds*ds;
    dq2=dq*dq;
    s2a=_dst->s2;
    sqa=_dst->sq;
    q2a=_dst->q2;
    sra=_dst->sr;
    qra=_dst->qr;
    sda=_dst->sd;
    qda=_dst->qd;
    s2qa=_dst->s2q;
    sq2a=_dst->sq2;
    s2b=_src[i].s2;
    sqb=_src[i].sq;
    q2b=_src[i].q2;
    srb=_src[i].sr;
    qrb=_src[i].qr;
    sdb=_src[i].sd;
    qdb=_src[i].qd;
    s2qb=_src[i].s2q;
    sq2b=_src[i].sq2;
    w=wa+wb;
    if(w==0)rwa=rwb=0;
    else{
      rwa=wa/w;
      rwb=wb/w;
    }
    rwa2=rwa*rwa;
    rwb2=rwb*rwb;
    rw2=wa*rwb;
    rw3=rw2*(rwa2-rwb2);
    rw4=wb*rwa2*rwa2+wa*rwb2*rwb2;
    /*
    (1,1,1) ->
     (0,0,0)#
     (1,0,0) C(1,1)*C(1,0)*C(1,0)->  d^{1,0,0}*(rwa*B_{0,1,1}-rwb*A_{0,1,1})
     (0,1,0) C(1,0)*C(1,1)*C(1,0)->  d^{0,1,0}*(rwa*B_{1,0,1}-rwb*A_{1,0,1})
     (0,0,1) C(1,0)*C(1,0)*C(1,1)->  d^{0,0,1}*(rwa*B_{1,1,0}-rwb*A_{1,1,0})
     (1,1,0)*
     (1,0,1)*
     (0,1,1)*
     (1,1,1) C(1,1)*C(1,1)*C(1,1)->  d^{1,1,1}*(rwa^3*wb-rwb^3*wa)
    (2,1) ->
     (0,0)#
     (1,0) C(2,1)*C(1,1)->2*d^{1,0}*(rwa*B_{1,1}-rwb*A_{1,1})
     (0,1) C(2,0)*C(1,1)->  d^{0,1}*(rwa*B_{2,0}-rwb*A_{2,0})
     (2,0)*
     (1,1)*
     (2,1) C(2,2)*C(1,1)->  d^{2,1}*(rwa^3*wb-rwb^3*wa)
    (2,2) ->
     (0,0)#
     (1,0) C(2,1)*C(2,0)->2*d^{1,0}*(rwa*B_{1,2}-rwb*A_{1,2})
     (0,1) C(2,0)*C(2,1)->2*d^{0,1}*(rwa*B_{2,1}-rwb*A_{2,1})
     (2,0) C(2,2)*C(2,0)->  d^{2,0}*(rwa^2*B_{0,2}+rwb^2*A_{0,2})
     (1,1) C(2,1)*C(2,1)->4*d^{1,1}*(rwa^2*B_{1,1}+rwb^2*A_{1,1})
     (0,2) C(2,0)*C(2,2)->  d^{0,2}*(rwa^2*B_{2,0}+rwb^2*A_{2,0})
     (1,2)*
     (2,1)*
     (2,2) C(2,2)*C(2,2)*d^{2,2}*(rwa^4*wb+rwb^4*wa)
    */
    _dst->s2q2+=_src[i].s2q2+2*(ds*(rwa*sq2b-rwb*sq2a)+dq*(rwa*s2qb-rwb*s2qa))
     +ds2*(rwa2*q2b+rwb2*q2a)+4*ds*dq*(rwa2*sqb+rwb2*sqa)
     +dq2*(rwa2*s2b+rwb2*s2a)+ds2*dq2*rw4;
    _dst->s2q+=_src[i].s2q+2*ds*(rwa*sqb-rwb*sqa)
     +dq*(rwa*s2b-rwb*s2a)+ds2*dq*rw3;
    _dst->sq2+=_src[i].sq2+ds*(rwa*q2b-rwb*q2a)
     +2*dq*(rwa*sqb-rwb*sqa)+ds*dq2*rw3;
    _dst->sqr+=_src[i].sqr+ds*(rwa*qrb-rwb*qra)+dq*(rwa*srb-rwb*sra)
     +dr*(rwa*sqb-rwb*sqa)+ds*dq*dr*rw3;
    _dst->sqd+=_src[i].sqd+ds*(rwa*qdb-rwb*qda)+dq*(rwa*sdb-rwb*sda)
     +dd*(rwa*sqb-rwb*sqa)+ds*dq*dd*rw3;
    _dst->s2+=_src[i].s2+ds2*rw2;
    _dst->sq+=_src[i].sq+ds*dq*rw2;
    _dst->q2+=_src[i].q2+dq2*rw2;
    _dst->sr+=_src[i].sr+ds*dr*rw2;
    _dst->qr+=_src[i].qr+dq*dr*rw2;
    _dst->r2+=_src[i].r2+dr*dr*rw2;
    _dst->sd+=_src[i].sd+ds*dd*rw2;
    _dst->qd+=_src[i].qd+dq*dd*rw2;
    _dst->d2+=_src[i].d2+dd*dd*rw2;
    _dst->w+=_src[i].w;
    _dst->s+=_src[i].s;
    _dst->q+=_src[i].q;
    _dst->r+=_src[i].r;
    _dst->d+=_src[i].d;
  }
}

/*Adjust a single corner of a set of metric bins to minimize the squared
   prediction error of R and D.
  Each bin is assumed to cover a quad like so:
    (s0,q0)    (s1,q0)
       A----------B
       |          |
       |          |
       |          |
       |          |
       C----------Z
    (s0,q1)    (s1,q1)
  The values A, B, and C are fixed, and Z is the free parameter.
  Then, for example, R_i is predicted via bilinear interpolation as
    x_i=(s_i-s0)/(s1-s0)
    y_i=(q_i-q0)/(q1-q0)
    dRds1_i=A+(B-A)*x_i
    dRds2_i=C+(Z-C)*x_i
    R_i=dRds1_i+(dRds2_i-dRds1_i)*y_i
  To find the Z that minimizes the squared prediction error over i, this can
   be rewritten as
    R_i-(A+(B-A)*x_i+(C-A)*y_i+(A-B-C)*x_i*y_i)=x_i*y_i*Z
  Letting X={...,x_i*y_i,...}^T and
   Y={...,R_i-(A+(B-A)*x_i+(C-A)*y_i+(A-B-C)*x_i*y_i),...}^T,
   the optimal Z is given by Z=(X^T.Y)/(X^T.X).
  Now, we need to compute these dot products without actually storing data for
   each sample.
  Starting with X^T.X, we have
   X^T.X = sum(x_i^2*y_i^2) = sum((s_i-s0)^2*(q_i-q0)^2)/((s1-s0)^2*(q1-q0)^2).
  Expanding the interior of the sum in a monomial basis of s_i and q_i gives
    s0^2*q0^2  *(1)
     -2*s0*q0^2*(s_i)
     -2*s0^2*q0*(q_i)
     +q0^2     *(s_i^2)
     +4*s0*q0  *(s_i*q_i)
     +s0^2     *(q_i^2)
     -2*q0     *(s_i^2*q_i)
     -2*s0     *(s_i*q_i^2)
     +1        *(s_i^2*q_i^2).
  However, computing things directly in this basis leads to gross numerical
   errors, as most of the terms will have similar size and destructive
   cancellation results.
  A much better basis is the central (co-)moment basis:
    {1,s_i-sbar,q_i-qbar,(s_i-sbar)^2,(s_i-sbar)*(q_i-qbar),(q_i-qbar)^2,
     (s_i-sbar)^2*(q_i-qbar),(s_i-sbar)*(q_i-qbar)^2,(s_i-sbar)^2*(q_i-qbar)^2},
   where sbar and qbar are the average s and q values over the bin,
   respectively.
  In that basis, letting ds=sbar-s0 and dq=qbar-q0, (s_i-s0)^2*(q_i-q0)^2 is
    ds^2*dq^2*(1)
     +dq^2   *((s_i-sbar)^2)
     +4*ds*dq*((s_i-sbar)*(q_i-qbar))
     +ds^2   *((q_i-qbar)^2)
     +2*dq   *((s_i-sbar)^2*(q_i-qbar))
     +2*ds   *((s_i-sbar)*(q_i-qbar)^2)
     +1      *((s_i-sbar)^2*(q_i-qbar)^2).
  With these expressions in the central (co-)moment bases, all we need to do
   is compute sums over the (co-)moment terms, which can be done
   incrementally (see oc_mode_metrics_add() and oc_mode_metrics_merge()),
   with no need to store the individual samples.
  Now, for X^T.Y, we have
    X^T.Y = sum((R_i-A-((B-A)/(s1-s0))*(s_i-s0)-((C-A)/(q1-q0))*(q_i-q0)
     -((A-B-C)/((s1-s0)*(q1-q0)))*(s_i-s0)*(q_i-q0))*(s_i-s0)*(q_i-q0))/
     ((s1-s0)*(q1-q0)),
   or, rewriting the constants to simplify notation,
    X^T.Y = sum((C0+C1*(s_i-s0)+C2*(q_i-q0)
     +C3*(s_i-s0)*(q_i-q0)+R_i)*(s_i-s0)*(q_i-q0))/((s1-s0)*(q1-q0)).
  Again, converting to the central (co-)moment basis, the interior of the
   above sum is
    ds*dq*(rbar+C0+C1*ds+C2*dq+C3*ds*dq)  *(1)
     +(C1*dq+C3*dq^2)                     *((s_i-sbar)^2)
     +(rbar+C0+2*C1*ds+2*C2*dq+4*C3*ds*dq)*((s_i-sbar)*(q_i-qbar))
     +(C2*ds+C3*ds^2)                     *((q_i-qbar)^2)
     +dq                                  *((s_i-sbar)*(r_i-rbar))
     +ds                                  *((q_i-qbar)*(r_i-rbar))
     +(C1+2*C3*dq)                        *((s_i-sbar)^2*(q_i-qbar))
     +(C2+2*C3*ds)                        *((s_i-sbar)*(q_i-qbar)^2)
     +1                                   *((s_i-sbar)*(q_i-qbar)*(r_i-rbar))
     +C3                                  *((s_i-sbar)^2*(q_i-qbar)^2).
  You might think it would be easier (if perhaps slightly less robust) to
   accumulate terms directly around s0 and q0.
  However, we update each corner of the bins in turn, so we would have to
   change basis to move the sums from corner to corner anyway.*/
double oc_mode_metrics_solve(double *_r,double *_d,
 const oc_mode_metrics *_metrics,const int *_s0,const int *_s1,
 const int *_q0,const int *_q1,
 const double *_ra,const double *_rb,const double *_rc,
 const double *_da,const double *_db,const double *_dc,int _n){
  double xx;
  double rxy;
  double dxy;
  double wt;
  int i;
  xx=rxy=dxy=wt=0;
  for(i=0;i<_n;i++)if(_metrics[i].w>0){
    double s10;
    double q10;
    double sq10;
    double ds;
    double dq;
    double ds2;
    double dq2;
    double r;
    double d;
    double s2;
    double sq;
    double q2;
    double sr;
    double qr;
    double sd;
    double qd;
    double s2q;
    double sq2;
    double sqr;
    double sqd;
    double s2q2;
    double c0;
    double c1;
    double c2;
    double c3;
    double w;
    w=_metrics[i].w;
    wt+=w;
    s10=_s1[i]-_s0[i];
    q10=_q1[i]-_q0[i];
    sq10=s10*q10;
    ds=_metrics[i].s/w-_s0[i];
    dq=_metrics[i].q/w-_q0[i];
    ds2=ds*ds;
    dq2=dq*dq;
    s2=_metrics[i].s2;
    sq=_metrics[i].sq;
    q2=_metrics[i].q2;
    s2q=_metrics[i].s2q;
    sq2=_metrics[i].sq2;
    s2q2=_metrics[i].s2q2;
    xx+=(dq2*(ds2*w+s2)+4*ds*dq*sq+ds2*q2+2*(dq*s2q+ds*sq2)+s2q2)/(sq10*sq10);
    r=_metrics[i].r/w;
    sr=_metrics[i].sr;
    qr=_metrics[i].qr;
    sqr=_metrics[i].sqr;
    c0=-_ra[i];
    c1=-(_rb[i]-_ra[i])/s10;
    c2=-(_rc[i]-_ra[i])/q10;
    c3=-(_ra[i]-_rb[i]-_rc[i])/sq10;
    rxy+=(ds*dq*(r+c0+c1*ds+c2*dq+c3*ds*dq)*w+(c1*dq+c3*dq2)*s2
     +(r+c0+2*(c1*ds+(c2+2*c3*ds)*dq))*sq+(c2*ds+c3*ds2)*q2+dq*sr+ds*qr
     +(c1+2*c3*dq)*s2q+(c2+2*c3*ds)*sq2+sqr+c3*s2q2)/sq10;
    d=_metrics[i].d/w;
    sd=_metrics[i].sd;
    qd=_metrics[i].qd;
    sqd=_metrics[i].sqd;
    c0=-_da[i];
    c1=-(_db[i]-_da[i])/s10;
    c2=-(_dc[i]-_da[i])/q10;
    c3=-(_da[i]-_db[i]-_dc[i])/sq10;
    dxy+=(ds*dq*(d+c0+c1*ds+c2*dq+c3*ds*dq)*w+(c1*dq+c3*dq2)*s2
     +(d+c0+2*(c1*ds+(c2+2*c3*ds)*dq))*sq+(c2*ds+c3*ds2)*q2+dq*sd+ds*qd
     +(c1+2*c3*dq)*s2q+(c2+2*c3*ds)*sq2+sqd+c3*s2q2)/sq10;
  }
  if(xx>1E-3){
    *_r=rxy/xx;
    *_d=dxy/xx;
  }
  else{
    *_r=0;
    *_d=0;
  }
  return wt;
}

/*Compile collected SATD/logq/rate/RMSE metrics into a form that's immediately
   useful for mode decision.*/
void oc_mode_metrics_update(oc_mode_metrics (*_metrics)[3][2][OC_COMP_BINS],
 int _niters_min,int _reweight,oc_mode_rd (*_table)[3][2][OC_COMP_BINS],
 int _shift,double (*_weight)[3][2][OC_COMP_BINS]){
  int niters;
  int prevdr;
  int prevdd;
  int dr;
  int dd;
  int pli;
  int qti;
  int qi;
  int si;
  dd=dr=INT_MAX;
  niters=0;
  /*The encoder interpolates rate and RMSE terms bilinearly from an
     OC_LOGQ_BINS by OC_COMP_BINS grid of sample points in _table.
    To find the sample values at the grid points that minimize the total
     squared prediction error actually requires solving a relatively sparse
     linear system with a number of variables equal to the number of grid
     points.
    Instead of writing a general sparse linear system solver, we just use
     Gauss-Seidel iteration, i.e., we update one grid point at time until
     they stop changing.*/
  do{
    prevdr=dr;
    prevdd=dd;
    dd=dr=0;
    for(pli=0;pli<3;pli++){
      for(qti=0;qti<2;qti++){
        for(qi=0;qi<OC_LOGQ_BINS;qi++){
          for(si=0;si<OC_COMP_BINS;si++){
            oc_mode_metrics m[4];
            int             s0[4];
            int             s1[4];
            int             q0[4];
            int             q1[4];
            double          ra[4];
            double          rb[4];
            double          rc[4];
            double          da[4];
            double          db[4];
            double          dc[4];
            double          r;
            double          d;
            int             rate;
            int             rmse;
            int             ds;
            int             n;
            n=0;
            /*Collect the statistics for the (up to) four bins grid point
               (si,qi) touches.*/
            if(qi>0&&si>0){
              q0[n]=OC_MODE_LOGQ[qi-1][pli][qti];
              q1[n]=OC_MODE_LOGQ[qi][pli][qti];
              s0[n]=si-1<<_shift;
              s1[n]=si<<_shift;
              ra[n]=ldexp(_table[qi-1][pli][qti][si-1].rate,-OC_BIT_SCALE);
              da[n]=ldexp(_table[qi-1][pli][qti][si-1].rmse,-OC_RMSE_SCALE);
              rb[n]=ldexp(_table[qi-1][pli][qti][si].rate,-OC_BIT_SCALE);
              db[n]=ldexp(_table[qi-1][pli][qti][si].rmse,-OC_RMSE_SCALE);
              rc[n]=ldexp(_table[qi][pli][qti][si-1].rate,-OC_BIT_SCALE);
              dc[n]=ldexp(_table[qi][pli][qti][si-1].rmse,-OC_RMSE_SCALE);
              *(m+n++)=*(_metrics[qi-1][pli][qti]+si-1);
            }
            if(qi>0){
              ds=si+1<OC_COMP_BINS?1:-1;
              q0[n]=OC_MODE_LOGQ[qi-1][pli][qti];
              q1[n]=OC_MODE_LOGQ[qi][pli][qti];
              s0[n]=si+ds<<_shift;
              s1[n]=si<<_shift;
              ra[n]=ldexp(_table[qi-1][pli][qti][si+ds].rate,-OC_BIT_SCALE);
              da[n]=
               ldexp(_table[qi-1][pli][qti][si+ds].rmse,-OC_RMSE_SCALE);
              rb[n]=ldexp(_table[qi-1][pli][qti][si].rate,-OC_BIT_SCALE);
              db[n]=ldexp(_table[qi-1][pli][qti][si].rmse,-OC_RMSE_SCALE);
              rc[n]=ldexp(_table[qi][pli][qti][si+ds].rate,-OC_BIT_SCALE);
              dc[n]=ldexp(_table[qi][pli][qti][si+ds].rmse,-OC_RMSE_SCALE);
              *(m+n++)=*(_metrics[qi-1][pli][qti]+si);
            }
            if(qi+1<OC_LOGQ_BINS&&si>0){
              q0[n]=OC_MODE_LOGQ[qi+1][pli][qti];
              q1[n]=OC_MODE_LOGQ[qi][pli][qti];
              s0[n]=si-1<<_shift;
              s1[n]=si<<_shift;
              ra[n]=ldexp(_table[qi+1][pli][qti][si-1].rate,-OC_BIT_SCALE);
              da[n]=ldexp(_table[qi+1][pli][qti][si-1].rmse,-OC_RMSE_SCALE);
              rb[n]=ldexp(_table[qi+1][pli][qti][si].rate,-OC_BIT_SCALE);
              db[n]=ldexp(_table[qi+1][pli][qti][si].rmse,-OC_RMSE_SCALE);
              rc[n]=ldexp(_table[qi][pli][qti][si-1].rate,-OC_BIT_SCALE);
              dc[n]=ldexp(_table[qi][pli][qti][si-1].rmse,-OC_RMSE_SCALE);
              *(m+n++)=*(_metrics[qi][pli][qti]+si-1);
            }
            if(qi+1<OC_LOGQ_BINS){
              ds=si+1<OC_COMP_BINS?1:-1;
              q0[n]=OC_MODE_LOGQ[qi+1][pli][qti];
              q1[n]=OC_MODE_LOGQ[qi][pli][qti];
              s0[n]=si+ds<<_shift;
              s1[n]=si<<_shift;
              ra[n]=ldexp(_table[qi+1][pli][qti][si+ds].rate,-OC_BIT_SCALE);
              da[n]=
               ldexp(_table[qi+1][pli][qti][si+ds].rmse,-OC_RMSE_SCALE);
              rb[n]=ldexp(_table[qi+1][pli][qti][si].rate,-OC_BIT_SCALE);
              db[n]=ldexp(_table[qi+1][pli][qti][si].rmse,-OC_RMSE_SCALE);
              rc[n]=ldexp(_table[qi][pli][qti][si+ds].rate,-OC_BIT_SCALE);
              dc[n]=ldexp(_table[qi][pli][qti][si+ds].rmse,-OC_RMSE_SCALE);
              *(m+n++)=*(_metrics[qi][pli][qti]+si);
            }
            /*On the first pass, initialize with a simple weighted average of
               the neighboring bins.*/
            if(!OC_HAS_MODE_METRICS&&niters==0){
              double w;
              w=r=d=0;
              while(n-->0){
                w+=m[n].w;
                r+=m[n].r;
                d+=m[n].d;
              }
              r=w>1E-3?r/w:0;
              d=w>1E-3?d/w:0;
              _weight[qi][pli][qti][si]=w;
            }
            else{
              /*Update the grid point and save the weight for later.*/
              _weight[qi][pli][qti][si]=
               oc_mode_metrics_solve(&r,&d,m,s0,s1,q0,q1,ra,rb,rc,da,db,dc,n);
            }
            rate=OC_CLAMPI(-32768,(int)(ldexp(r,OC_BIT_SCALE)+0.5),32767);
            rmse=OC_CLAMPI(-32768,(int)(ldexp(d,OC_RMSE_SCALE)+0.5),32767);
            dr+=abs(rate-_table[qi][pli][qti][si].rate);
            dd+=abs(rmse-_table[qi][pli][qti][si].rmse);
            _table[qi][pli][qti][si].rate=(ogg_int16_t)rate;
            _table[qi][pli][qti][si].rmse=(ogg_int16_t)rmse;
          }
        }
      }
    }
  }
  /*After a fixed number of initial iterations, only iterate so long as the
     total change is decreasing.
    This ensures we don't oscillate forever, which is a danger, as all of our
     results are rounded fairly coarsely.*/
  while((dr>0||dd>0)&&(niters++<_niters_min||(dr<prevdr&&dd<prevdd)));
  if(_reweight){
    /*Now, reduce the values of the optimal solution until we get enough
       samples in each bin to overcome the constant OC_ZWEIGHT factor.
      This encourages sampling under-populated bins and prevents a single large
       sample early on from discouraging coding in that bin ever again.*/
    for(pli=0;pli<3;pli++){
      for(qti=0;qti<2;qti++){
        for(qi=0;qi<OC_LOGQ_BINS;qi++){
          for(si=0;si<OC_COMP_BINS;si++){
            double wt;
            wt=_weight[qi][pli][qti][si];
            wt/=OC_ZWEIGHT+wt;
            _table[qi][pli][qti][si].rate=(ogg_int16_t)
             (_table[qi][pli][qti][si].rate*wt+0.5);
            _table[qi][pli][qti][si].rmse=(ogg_int16_t)
             (_table[qi][pli][qti][si].rmse*wt+0.5);
          }
        }
      }
    }
  }
}

/*Dump the in memory mode metrics to a file.
  Note this data format isn't portable between different platforms.*/
void oc_mode_metrics_dump(void){
  FILE *fmetrics;
  fmetrics=fopen(OC_MODE_METRICS_FILENAME,"wb");
  if(fmetrics!=NULL){
    (void)fwrite(OC_MODE_LOGQ,sizeof(OC_MODE_LOGQ),1,fmetrics);
    (void)fwrite(OC_MODE_METRICS_SATD,sizeof(OC_MODE_METRICS_SATD),1,fmetrics);
    (void)fwrite(OC_MODE_METRICS_SAD,sizeof(OC_MODE_METRICS_SAD),1,fmetrics);
    fclose(fmetrics);
  }
}

void oc_mode_metrics_print_rd(FILE *_fout,const char *_table_name,
#if !defined(OC_COLLECT_METRICS)
 const oc_mode_rd (*_mode_rd_table)[3][2][OC_COMP_BINS]){
#else
 oc_mode_rd (*_mode_rd_table)[3][2][OC_COMP_BINS]){
#endif
  int qii;
  fprintf(_fout,
   "# if !defined(OC_COLLECT_METRICS)\n"
   "static const\n"
   "# endif\n"
   "oc_mode_rd %s[OC_LOGQ_BINS][3][2][OC_COMP_BINS]={\n",_table_name);
  for(qii=0;qii<OC_LOGQ_BINS;qii++){
    int pli;
    fprintf(_fout,"  {\n");
    for(pli=0;pli<3;pli++){
      int qti;
      fprintf(_fout,"    {\n");
      for(qti=0;qti<2;qti++){
        int bin;
        int qi;
        static const char *pl_names[3]={"Y'","Cb","Cr"};
        static const char *qti_names[2]={"INTRA","INTER"};
        qi=(63*qii+(OC_LOGQ_BINS-1>>1))/(OC_LOGQ_BINS-1);
        fprintf(_fout,"      /*%s  qi=%i  %s*/\n",
         pl_names[pli],qi,qti_names[qti]);
        fprintf(_fout,"      {\n");
        fprintf(_fout,"        ");
        for(bin=0;bin<OC_COMP_BINS;bin++){
          if(bin&&!(bin&0x3))fprintf(_fout,"\n        ");
          fprintf(_fout,"{%5i,%5i}",
           _mode_rd_table[qii][pli][qti][bin].rate,
           _mode_rd_table[qii][pli][qti][bin].rmse);
          if(bin+1<OC_COMP_BINS)fprintf(_fout,",");
        }
        fprintf(_fout,"\n      }");
        if(qti<1)fprintf(_fout,",");
        fprintf(_fout,"\n");
      }
      fprintf(_fout,"    }");
      if(pli<2)fprintf(_fout,",");
      fprintf(_fout,"\n");
    }
    fprintf(_fout,"  }");
    if(qii+1<OC_LOGQ_BINS)fprintf(_fout,",");
    fprintf(_fout,"\n");
  }
  fprintf(_fout,
   "};\n"
   "\n");
}

void oc_mode_metrics_print(FILE *_fout){
  int qii;
  fprintf(_fout,
   "/*File generated by libtheora with OC_COLLECT_METRICS"
   " defined at compile time.*/\n"
   "#if !defined(_modedec_H)\n"
   "# define _modedec_H (1)\n"
   "# include \"encint.h\"\n"
   "\n"
   "\n"
   "\n"
   "/*The log of the average quantizer for each of the OC_MODE_RD table rows\n"
   "   (e.g., for the represented qi's, and each pli and qti), in Q10 format.\n"
   "  The actual statistics used by the encoder will be interpolated from\n"
   "   that table based on log_plq for the actual quantization matrix used.*/\n"
   "# if !defined(OC_COLLECT_METRICS)\n"
   "static const\n"
   "# endif\n"
   "ogg_int16_t OC_MODE_LOGQ[OC_LOGQ_BINS][3][2]={\n");
  for(qii=0;qii<OC_LOGQ_BINS;qii++){
    fprintf(_fout,"  { {0x%04X,0x%04X},{0x%04X,0x%04X},{0x%04X,0x%04X} }%s\n",
     OC_MODE_LOGQ[qii][0][0],OC_MODE_LOGQ[qii][0][1],OC_MODE_LOGQ[qii][1][0],
     OC_MODE_LOGQ[qii][1][1],OC_MODE_LOGQ[qii][2][0],OC_MODE_LOGQ[qii][2][1],
     qii+1<OC_LOGQ_BINS?",":"");
  }
  fprintf(_fout,
   "};\n"
   "\n");
  oc_mode_metrics_print_rd(_fout,"OC_MODE_RD_SATD",OC_MODE_RD_SATD);
  oc_mode_metrics_print_rd(_fout,"OC_MODE_RD_SAD",OC_MODE_RD_SAD);
  fprintf(_fout,
   "#endif\n");
}


# if !defined(OC_COLLECT_NO_ENC_FUNCS)
void oc_enc_mode_metrics_load(oc_enc_ctx *_enc){
  oc_restore_fpu(&_enc->state);
  /*Load any existing mode metrics if we haven't already.*/
  if(!OC_HAS_MODE_METRICS){
    FILE *fmetrics;
    memset(OC_MODE_METRICS_SATD,0,sizeof(OC_MODE_METRICS_SATD));
    memset(OC_MODE_METRICS_SAD,0,sizeof(OC_MODE_METRICS_SAD));
    fmetrics=fopen(OC_MODE_METRICS_FILENAME,"rb");
    if(fmetrics!=NULL){
      /*Read in the binary structures as written my oc_mode_metrics_dump().
        Note this format isn't portable between different platforms.*/
      (void)fread(OC_MODE_LOGQ,sizeof(OC_MODE_LOGQ),1,fmetrics);
      (void)fread(OC_MODE_METRICS_SATD,sizeof(OC_MODE_METRICS_SATD),1,fmetrics);
      (void)fread(OC_MODE_METRICS_SAD,sizeof(OC_MODE_METRICS_SAD),1,fmetrics);
      fclose(fmetrics);
    }
    else{
      int qii;
      int qi;
      int pli;
      int qti;
      for(qii=0;qii<OC_LOGQ_BINS;qii++){
        qi=(63*qii+(OC_LOGQ_BINS-1>>1))/(OC_LOGQ_BINS-1);
        for(pli=0;pli<3;pli++)for(qti=0;qti<2;qti++){
          OC_MODE_LOGQ[qii][pli][qti]=_enc->log_plq[qi][pli][qti];
        }
      }
    }
    oc_mode_metrics_update(OC_MODE_METRICS_SATD,100,1,
     OC_MODE_RD_SATD,OC_SATD_SHIFT,OC_MODE_RD_WEIGHT_SATD);
    oc_mode_metrics_update(OC_MODE_METRICS_SAD,100,1,
     OC_MODE_RD_SAD,OC_SAD_SHIFT,OC_MODE_RD_WEIGHT_SAD);
    OC_HAS_MODE_METRICS=1;
  }
}

/*The following token skipping code used to also be used in the decoder (and
   even at one point other places in the encoder).
  However, it was obsoleted by other optimizations, and is now only used here.
  It has been moved here to avoid generating the code when it's not needed.*/

/*Determines the number of blocks or coefficients to be skipped for a given
   token value.
  _token:      The token value to skip.
  _extra_bits: The extra bits attached to this token.
  Return: A positive value indicates that number of coefficients are to be
           skipped in the current block.
          Otherwise, the negative of the return value indicates that number of
           blocks are to be ended.*/
typedef ptrdiff_t (*oc_token_skip_func)(int _token,int _extra_bits);

/*Handles the simple end of block tokens.*/
static ptrdiff_t oc_token_skip_eob(int _token,int _extra_bits){
  int nblocks_adjust;
  nblocks_adjust=OC_UNIBBLE_TABLE32(0,1,2,3,7,15,0,0,_token)+1;
  return -_extra_bits-nblocks_adjust;
}

/*The last EOB token has a special case, where an EOB run of size zero ends all
   the remaining blocks in the frame.*/
static ptrdiff_t oc_token_skip_eob6(int _token,int _extra_bits){
  /*Note: We want to return -PTRDIFF_MAX, but that requires C99, which is not
     yet available everywhere; this should be equivalent.*/
  if(!_extra_bits)return -(~(size_t)0>>1);
  return -_extra_bits;
}

/*Handles the pure zero run tokens.*/
static ptrdiff_t oc_token_skip_zrl(int _token,int _extra_bits){
  return _extra_bits+1;
}

/*Handles a normal coefficient value token.*/
static ptrdiff_t oc_token_skip_val(void){
  return 1;
}

/*Handles a category 1A zero run/coefficient value combo token.*/
static ptrdiff_t oc_token_skip_run_cat1a(int _token){
  return _token-OC_DCT_RUN_CAT1A+2;
}

/*Handles category 1b, 1c, 2a, and 2b zero run/coefficient value combo tokens.*/
static ptrdiff_t oc_token_skip_run(int _token,int _extra_bits){
  int run_cati;
  int ncoeffs_mask;
  int ncoeffs_adjust;
  run_cati=_token-OC_DCT_RUN_CAT1B;
  ncoeffs_mask=OC_BYTE_TABLE32(3,7,0,1,run_cati);
  ncoeffs_adjust=OC_BYTE_TABLE32(7,11,2,3,run_cati);
  return (_extra_bits&ncoeffs_mask)+ncoeffs_adjust;
}

/*A jump table for computing the number of coefficients or blocks to skip for
   a given token value.
  This reduces all the conditional branches, etc., needed to parse these token
   values down to one indirect jump.*/
static const oc_token_skip_func OC_TOKEN_SKIP_TABLE[TH_NDCT_TOKENS]={
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob6,
  oc_token_skip_zrl,
  oc_token_skip_zrl,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  oc_token_skip_run,
  oc_token_skip_run,
  oc_token_skip_run,
  oc_token_skip_run
};

/*Determines the number of blocks or coefficients to be skipped for a given
   token value.
  _token:      The token value to skip.
  _extra_bits: The extra bits attached to this token.
  Return: A positive value indicates that number of coefficients are to be
           skipped in the current block.
          Otherwise, the negative of the return value indicates that number of
           blocks are to be ended.
          0 will never be returned, so that at least one coefficient in one
           block will always be decoded for every token.*/
static ptrdiff_t oc_dct_token_skip(int _token,int _extra_bits){
  return (*OC_TOKEN_SKIP_TABLE[_token])(_token,_extra_bits);
}


void oc_enc_mode_metrics_collect(oc_enc_ctx *_enc){
  static const unsigned char OC_ZZI_HUFF_OFFSET[64]={
     0,16,16,16,16,16,32,32,
    32,32,32,32,32,32,32,48,
    48,48,48,48,48,48,48,48,
    48,48,48,48,64,64,64,64,
    64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64
  };
  const oc_fragment *frags;
  const unsigned    *frag_sad;
  const unsigned    *frag_satd;
  const unsigned    *frag_ssd;
  const ptrdiff_t   *coded_fragis;
  ptrdiff_t          ncoded_fragis;
  ptrdiff_t          fragii;
  double             fragw;
  int                modelines[3][3][2];
  int                qti;
  int                qii;
  int                qi;
  int                pli;
  int                zzi;
  int                token;
  int                eb;
  oc_restore_fpu(&_enc->state);
  /*Figure out which metric bins to use for this frame's quantizers.*/
  for(qii=0;qii<_enc->state.nqis;qii++){
    for(pli=0;pli<3;pli++){
      for(qti=0;qti<2;qti++){
        int log_plq;
        int modeline;
        log_plq=_enc->log_plq[_enc->state.qis[qii]][pli][qti];
        for(modeline=0;modeline<OC_LOGQ_BINS-1&&
         OC_MODE_LOGQ[modeline+1][pli][qti]>log_plq;modeline++);
        modelines[qii][pli][qti]=modeline;
      }
    }
  }
  qti=_enc->state.frame_type;
  frags=_enc->state.frags;
  frag_sad=_enc->frag_sad;
  frag_satd=_enc->frag_satd;
  frag_ssd=_enc->frag_ssd;
  coded_fragis=_enc->state.coded_fragis;
  ncoded_fragis=fragii=0;
  /*Weight the fragments by the inverse frame size; this prevents HD content
     from dominating the statistics.*/
  fragw=1.0/_enc->state.nfrags;
  for(pli=0;pli<3;pli++){
    ptrdiff_t ti[64];
    int       eob_token[64];
    int       eob_run[64];
    /*Set up token indices and eob run counts.
      We don't bother trying to figure out the real cost of the runs that span
       coefficients; instead we use the costs that were available when R-D
       token optimization was done.*/
    for(zzi=0;zzi<64;zzi++){
      ti[zzi]=_enc->dct_token_offs[pli][zzi];
      if(ti[zzi]>0){
        token=_enc->dct_tokens[pli][zzi][0];
        eb=_enc->extra_bits[pli][zzi][0];
        eob_token[zzi]=token;
        eob_run[zzi]=-oc_dct_token_skip(token,eb);
      }
      else{
        eob_token[zzi]=OC_NDCT_EOB_TOKEN_MAX;
        eob_run[zzi]=0;
      }
    }
    /*Scan the list of coded fragments for this plane.*/
    ncoded_fragis+=_enc->state.ncoded_fragis[pli];
    for(;fragii<ncoded_fragis;fragii++){
      ptrdiff_t fragi;
      int       frag_bits;
      int       huffi;
      int       skip;
      int       mb_mode;
      unsigned  sad;
      unsigned  satd;
      double    sqrt_ssd;
      int       bin;
      int       qtj;
      fragi=coded_fragis[fragii];
      frag_bits=0;
      for(zzi=0;zzi<64;){
        if(eob_run[zzi]>0){
          /*We've reached the end of the block.*/
          eob_run[zzi]--;
          break;
        }
        huffi=_enc->huff_idxs[qti][zzi>0][pli+1>>1]
         +OC_ZZI_HUFF_OFFSET[zzi];
        if(eob_token[zzi]<OC_NDCT_EOB_TOKEN_MAX){
          /*This token caused an EOB run to be flushed.
            Therefore it gets the bits associated with it.*/
          frag_bits+=_enc->huff_codes[huffi][eob_token[zzi]].nbits
           +OC_DCT_TOKEN_EXTRA_BITS[eob_token[zzi]];
          eob_token[zzi]=OC_NDCT_EOB_TOKEN_MAX;
        }
        token=_enc->dct_tokens[pli][zzi][ti[zzi]];
        eb=_enc->extra_bits[pli][zzi][ti[zzi]];
        ti[zzi]++;
        skip=oc_dct_token_skip(token,eb);
        if(skip<0){
          eob_token[zzi]=token;
          eob_run[zzi]=-skip;
        }
        else{
          /*A regular DCT value token; accumulate the bits for it.*/
          frag_bits+=_enc->huff_codes[huffi][token].nbits
           +OC_DCT_TOKEN_EXTRA_BITS[token];
          zzi+=skip;
        }
      }
      mb_mode=frags[fragi].mb_mode;
      qii=frags[fragi].qii;
      qi=_enc->state.qis[qii];
      sad=frag_sad[fragi]<<(pli+1&2);
      satd=frag_satd[fragi]<<(pli+1&2);
      sqrt_ssd=sqrt(frag_ssd[fragi]);
      qtj=mb_mode!=OC_MODE_INTRA;
      /*Accumulate statistics.
        The rate (frag_bits) and RMSE (sqrt(frag_ssd)) are not scaled by
         OC_BIT_SCALE and OC_RMSE_SCALE; this lets us change the scale factor
         yet still use old data.*/
      bin=OC_MINI(satd>>OC_SATD_SHIFT,OC_COMP_BINS-1);
      oc_mode_metrics_add(
       OC_MODE_METRICS_SATD[modelines[qii][pli][qtj]][pli][qtj]+bin,
       fragw,satd,_enc->log_plq[qi][pli][qtj],frag_bits,sqrt_ssd);
      bin=OC_MINI(sad>>OC_SAD_SHIFT,OC_COMP_BINS-1);
      oc_mode_metrics_add(
       OC_MODE_METRICS_SAD[modelines[qii][pli][qtj]][pli][qtj]+bin,
       fragw,sad,_enc->log_plq[qi][pli][qtj],frag_bits,sqrt_ssd);
    }
  }
  /*Update global SA(T)D/logq/rate/RMSE estimation matrix.*/
  oc_mode_metrics_update(OC_MODE_METRICS_SATD,4,1,
   OC_MODE_RD_SATD,OC_SATD_SHIFT,OC_MODE_RD_WEIGHT_SATD);
  oc_mode_metrics_update(OC_MODE_METRICS_SAD,4,1,
   OC_MODE_RD_SAD,OC_SAD_SHIFT,OC_MODE_RD_WEIGHT_SAD);
}
# endif

#endif
