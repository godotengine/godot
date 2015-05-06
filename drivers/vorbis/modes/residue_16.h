/********************************************************************
 *                                                                  *
 * This FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2009             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: toplevel residue templates 16/22kHz
 last mod: $Id: residue_16.h 16962 2010-03-11 07:30:34Z xiphmont $

 ********************************************************************/

/***** residue backends *********************************************/

static const static_bookblock _resbook_16s_0={
  {
    {0},
    {0,0,&_16c0_s_p1_0},
    {0},
    {0,0,&_16c0_s_p3_0},
    {0,0,&_16c0_s_p4_0},
    {0,0,&_16c0_s_p5_0},
    {0,0,&_16c0_s_p6_0},
    {&_16c0_s_p7_0,&_16c0_s_p7_1},
    {&_16c0_s_p8_0,&_16c0_s_p8_1},
    {&_16c0_s_p9_0,&_16c0_s_p9_1,&_16c0_s_p9_2}
   }
};
static const static_bookblock _resbook_16s_1={
  {
    {0},
    {0,0,&_16c1_s_p1_0},
    {0},
    {0,0,&_16c1_s_p3_0},
    {0,0,&_16c1_s_p4_0},
    {0,0,&_16c1_s_p5_0},
    {0,0,&_16c1_s_p6_0},
    {&_16c1_s_p7_0,&_16c1_s_p7_1},
    {&_16c1_s_p8_0,&_16c1_s_p8_1},
    {&_16c1_s_p9_0,&_16c1_s_p9_1,&_16c1_s_p9_2}
   }
};
static const static_bookblock _resbook_16s_2={
  {
    {0},
    {0,0,&_16c2_s_p1_0},
    {0,0,&_16c2_s_p2_0},
    {0,0,&_16c2_s_p3_0},
    {0,0,&_16c2_s_p4_0},
    {&_16c2_s_p5_0,&_16c2_s_p5_1},
    {&_16c2_s_p6_0,&_16c2_s_p6_1},
    {&_16c2_s_p7_0,&_16c2_s_p7_1},
    {&_16c2_s_p8_0,&_16c2_s_p8_1},
    {&_16c2_s_p9_0,&_16c2_s_p9_1,&_16c2_s_p9_2}
   }
};

static const vorbis_residue_template _res_16s_0[]={
  {2,0,32,  &_residue_44_mid,
   &_huff_book__16c0_s_single,&_huff_book__16c0_s_single,
   &_resbook_16s_0,&_resbook_16s_0},
};
static const vorbis_residue_template _res_16s_1[]={
  {2,0,32,  &_residue_44_mid,
   &_huff_book__16c1_s_short,&_huff_book__16c1_s_short,
   &_resbook_16s_1,&_resbook_16s_1},

  {2,0,32,  &_residue_44_mid,
   &_huff_book__16c1_s_long,&_huff_book__16c1_s_long,
   &_resbook_16s_1,&_resbook_16s_1}
};
static const vorbis_residue_template _res_16s_2[]={
  {2,0,32,  &_residue_44_high,
   &_huff_book__16c2_s_short,&_huff_book__16c2_s_short,
   &_resbook_16s_2,&_resbook_16s_2},

  {2,0,32,  &_residue_44_high,
   &_huff_book__16c2_s_long,&_huff_book__16c2_s_long,
   &_resbook_16s_2,&_resbook_16s_2}
};

static const vorbis_mapping_template _mapres_template_16_stereo[3]={
  { _map_nominal, _res_16s_0 }, /* 0 */
  { _map_nominal, _res_16s_1 }, /* 1 */
  { _map_nominal, _res_16s_2 }, /* 2 */
};

static const static_bookblock _resbook_16u_0={
  {
    {0},
    {0,0,&_16u0__p1_0},
    {0,0,&_16u0__p2_0},
    {0,0,&_16u0__p3_0},
    {0,0,&_16u0__p4_0},
    {0,0,&_16u0__p5_0},
    {&_16u0__p6_0,&_16u0__p6_1},
    {&_16u0__p7_0,&_16u0__p7_1,&_16u0__p7_2}
   }
};
static const static_bookblock _resbook_16u_1={
  {
    {0},
    {0,0,&_16u1__p1_0},
    {0,0,&_16u1__p2_0},
    {0,0,&_16u1__p3_0},
    {0,0,&_16u1__p4_0},
    {0,0,&_16u1__p5_0},
    {0,0,&_16u1__p6_0},
    {&_16u1__p7_0,&_16u1__p7_1},
    {&_16u1__p8_0,&_16u1__p8_1},
    {&_16u1__p9_0,&_16u1__p9_1,&_16u1__p9_2}
   }
};
static const static_bookblock _resbook_16u_2={
  {
    {0},
    {0,0,&_16u2_p1_0},
    {0,0,&_16u2_p2_0},
    {0,0,&_16u2_p3_0},
    {0,0,&_16u2_p4_0},
    {&_16u2_p5_0,&_16u2_p5_1},
    {&_16u2_p6_0,&_16u2_p6_1},
    {&_16u2_p7_0,&_16u2_p7_1},
    {&_16u2_p8_0,&_16u2_p8_1},
    {&_16u2_p9_0,&_16u2_p9_1,&_16u2_p9_2}
   }
};

static const vorbis_residue_template _res_16u_0[]={
  {1,0,32,  &_residue_44_low_un,
   &_huff_book__16u0__single,&_huff_book__16u0__single,
   &_resbook_16u_0,&_resbook_16u_0},
};
static const vorbis_residue_template _res_16u_1[]={
  {1,0,32,  &_residue_44_mid_un,
   &_huff_book__16u1__short,&_huff_book__16u1__short,
   &_resbook_16u_1,&_resbook_16u_1},

  {1,0,32,  &_residue_44_mid_un,
   &_huff_book__16u1__long,&_huff_book__16u1__long,
   &_resbook_16u_1,&_resbook_16u_1}
};
static const vorbis_residue_template _res_16u_2[]={
  {1,0,32,  &_residue_44_hi_un,
   &_huff_book__16u2__short,&_huff_book__16u2__short,
   &_resbook_16u_2,&_resbook_16u_2},

  {1,0,32,  &_residue_44_hi_un,
   &_huff_book__16u2__long,&_huff_book__16u2__long,
   &_resbook_16u_2,&_resbook_16u_2}
};


static const vorbis_mapping_template _mapres_template_16_uncoupled[3]={
  { _map_nominal_u, _res_16u_0 }, /* 0 */
  { _map_nominal_u, _res_16u_1 }, /* 1 */
  { _map_nominal_u, _res_16u_2 }, /* 2 */
};
