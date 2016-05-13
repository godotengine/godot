/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2010             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: toplevel residue templates for 32/44.1/48kHz uncoupled
 last mod: $Id$

 ********************************************************************/

#include "vorbis/codec.h"
#include "backends.h"

#include "books/coupled/res_books_51.h"

/***** residue backends *********************************************/

static const vorbis_info_residue0 _residue_44p_lo={
  0,-1, -1, 7,-1,-1,
  /* 0   1   2   3   4   5   6   7   8  */
  {0},
  {-1},
  {  0,  1,  2,  7, 17, 31},
  {  0,  0, 99,  7, 17, 31},
};

static const vorbis_info_residue0 _residue_44p={
  0,-1, -1, 8,-1,-1,
  /* 0   1   2   3   4   5   6   7   8  */
  {0},
  {-1},
  {  0,  1,  1,   2,  7, 17, 31},
  {  0,  0, 99,  99,  7, 17, 31},
};

static const vorbis_info_residue0 _residue_44p_hi={
  0,-1, -1, 8,-1,-1,
  /* 0   1   2   3   4   5   6   7   8  */
  {0},
  {-1},
  {  0,  1,  2,  4,  7, 17, 31},
  {  0,  1,  2,  4,  7, 17, 31},
};

static const vorbis_info_residue0 _residue_44p_lfe={
  0,-1, -1, 2,-1,-1,
  /* 0   1   2   3   4   5   6   7   8  */
  {0},
  {-1},
  { 32},
  { -1}
};

static const static_bookblock _resbook_44p_n1={
  {
    {0},
    {0,&_44pn1_p1_0},

    {&_44pn1_p2_0,&_44pn1_p2_1,0},
    {&_44pn1_p3_0,&_44pn1_p3_1,0},
    {&_44pn1_p4_0,&_44pn1_p4_1,0},

    {&_44pn1_p5_0,&_44pn1_p5_1,&_44pn1_p4_1},
    {&_44pn1_p6_0,&_44pn1_p6_1,&_44pn1_p6_2},
   }
};

static const static_bookblock _resbook_44p_0={
  {
    {0},
    {0,&_44p0_p1_0},

    {&_44p0_p2_0,&_44p0_p2_1,0},
    {&_44p0_p3_0,&_44p0_p3_1,0},
    {&_44p0_p4_0,&_44p0_p4_1,0},

    {&_44p0_p5_0,&_44p0_p5_1,&_44p0_p4_1},
    {&_44p0_p6_0,&_44p0_p6_1,&_44p0_p6_2},
   }
};

static const static_bookblock _resbook_44p_1={
  {
    {0},
    {0,&_44p1_p1_0},

    {&_44p1_p2_0,&_44p1_p2_1,0},
    {&_44p1_p3_0,&_44p1_p3_1,0},
    {&_44p1_p4_0,&_44p1_p4_1,0},

    {&_44p1_p5_0,&_44p1_p5_1,&_44p1_p4_1},
    {&_44p1_p6_0,&_44p1_p6_1,&_44p1_p6_2},
   }
};

static const static_bookblock _resbook_44p_2={
  {
    {0},
    {0,0,&_44p2_p1_0},
    {0,&_44p2_p2_0,0},

    {&_44p2_p3_0,&_44p2_p3_1,0},
    {&_44p2_p4_0,&_44p2_p4_1,0},
    {&_44p2_p5_0,&_44p2_p5_1,0},

    {&_44p2_p6_0,&_44p2_p6_1,&_44p2_p5_1},
    {&_44p2_p7_0,&_44p2_p7_1,&_44p2_p7_2,&_44p2_p7_3}
   }
};
static const static_bookblock _resbook_44p_3={
  {
    {0},
    {0,0,&_44p3_p1_0},
    {0,&_44p3_p2_0,0},

    {&_44p3_p3_0,&_44p3_p3_1,0},
    {&_44p3_p4_0,&_44p3_p4_1,0},
    {&_44p3_p5_0,&_44p3_p5_1,0},

    {&_44p3_p6_0,&_44p3_p6_1,&_44p3_p5_1},
    {&_44p3_p7_0,&_44p3_p7_1,&_44p3_p7_2,&_44p3_p7_3}
   }
};
static const static_bookblock _resbook_44p_4={
  {
    {0},
    {0,0,&_44p4_p1_0},
    {0,&_44p4_p2_0,0},

    {&_44p4_p3_0,&_44p4_p3_1,0},
    {&_44p4_p4_0,&_44p4_p4_1,0},
    {&_44p4_p5_0,&_44p4_p5_1,0},

    {&_44p4_p6_0,&_44p4_p6_1,&_44p4_p5_1},
    {&_44p4_p7_0,&_44p4_p7_1,&_44p4_p7_2,&_44p4_p7_3}
   }
};
static const static_bookblock _resbook_44p_5={
  {
    {0},
    {0,0,&_44p5_p1_0},
    {0,&_44p5_p2_0,0},

    {&_44p5_p3_0,&_44p5_p3_1,0},
    {&_44p5_p4_0,&_44p5_p4_1,0},
    {&_44p5_p5_0,&_44p5_p5_1,0},

    {&_44p5_p6_0,&_44p5_p6_1,&_44p5_p5_1},
    {&_44p5_p7_0,&_44p5_p7_1,&_44p5_p7_2,&_44p5_p7_3}
   }
};
static const static_bookblock _resbook_44p_6={
  {
    {0},
    {0,0,&_44p6_p1_0},
    {0,&_44p6_p2_0,0},

    {&_44p6_p3_0,&_44p6_p3_1,0},
    {&_44p6_p4_0,&_44p6_p4_1,0},
    {&_44p6_p5_0,&_44p6_p5_1,0},

    {&_44p6_p6_0,&_44p6_p6_1,&_44p6_p5_1},
    {&_44p6_p7_0,&_44p6_p7_1,&_44p6_p7_2,&_44p6_p7_3}
   }
};
static const static_bookblock _resbook_44p_7={
  {
    {0},
    {0,0,&_44p7_p1_0},
    {0,&_44p7_p2_0,0},

    {&_44p7_p3_0,&_44p7_p3_1,0},
    {&_44p7_p4_0,&_44p7_p4_1,0},
    {&_44p7_p5_0,&_44p7_p5_1,0},

    {&_44p7_p6_0,&_44p7_p6_1,&_44p7_p5_1},
    {&_44p7_p7_0,&_44p7_p7_1,&_44p7_p7_2,&_44p7_p7_3}
   }
};
static const static_bookblock _resbook_44p_8={
  {
    {0},
    {0,0,&_44p8_p1_0},
    {0,&_44p8_p2_0,0},

    {&_44p8_p3_0,&_44p8_p3_1,0},
    {&_44p8_p4_0,&_44p8_p4_1,0},
    {&_44p8_p5_0,&_44p8_p5_1,0},

    {&_44p8_p6_0,&_44p8_p6_1,&_44p8_p5_1},
    {&_44p8_p7_0,&_44p8_p7_1,&_44p8_p7_2,&_44p8_p7_3}
   }
};
static const static_bookblock _resbook_44p_9={
  {
    {0},
    {0,0,&_44p9_p1_0},
    {0,&_44p9_p2_0,0},

    {&_44p9_p3_0,&_44p9_p3_1,0},
    {&_44p9_p4_0,&_44p9_p4_1,0},
    {&_44p9_p5_0,&_44p9_p5_1,0},

    {&_44p9_p6_0,&_44p9_p6_1,&_44p9_p5_1},
    {&_44p9_p7_0,&_44p9_p7_1,&_44p9_p7_2,&_44p9_p7_3}
   }
};

static const static_bookblock _resbook_44p_ln1={
  {
    {&_44pn1_l0_0,&_44pn1_l0_1,0},
    {&_44pn1_l1_0,&_44pn1_p6_1,&_44pn1_p6_2},
   }
};
static const static_bookblock _resbook_44p_l0={
  {
    {&_44p0_l0_0,&_44p0_l0_1,0},
    {&_44p0_l1_0,&_44p0_p6_1,&_44p0_p6_2},
   }
};
static const static_bookblock _resbook_44p_l1={
  {
    {&_44p1_l0_0,&_44p1_l0_1,0},
    {&_44p1_l1_0,&_44p1_p6_1,&_44p1_p6_2},
   }
};
static const static_bookblock _resbook_44p_l2={
  {
    {&_44p2_l0_0,&_44p2_l0_1,0},
    {&_44p2_l1_0,&_44p2_p7_2,&_44p2_p7_3},
   }
};
static const static_bookblock _resbook_44p_l3={
  {
    {&_44p3_l0_0,&_44p3_l0_1,0},
    {&_44p3_l1_0,&_44p3_p7_2,&_44p3_p7_3},
   }
};
static const static_bookblock _resbook_44p_l4={
  {
    {&_44p4_l0_0,&_44p4_l0_1,0},
    {&_44p4_l1_0,&_44p4_p7_2,&_44p4_p7_3},
   }
};
static const static_bookblock _resbook_44p_l5={
  {
    {&_44p5_l0_0,&_44p5_l0_1,0},
    {&_44p5_l1_0,&_44p5_p7_2,&_44p5_p7_3},
   }
};
static const static_bookblock _resbook_44p_l6={
  {
    {&_44p6_l0_0,&_44p6_l0_1,0},
    {&_44p6_l1_0,&_44p6_p7_2,&_44p6_p7_3},
   }
};
static const static_bookblock _resbook_44p_l7={
  {
    {&_44p7_l0_0,&_44p7_l0_1,0},
    {&_44p7_l1_0,&_44p7_p7_2,&_44p7_p7_3},
   }
};
static const static_bookblock _resbook_44p_l8={
  {
    {&_44p8_l0_0,&_44p8_l0_1,0},
    {&_44p8_l1_0,&_44p8_p7_2,&_44p8_p7_3},
   }
};
static const static_bookblock _resbook_44p_l9={
  {
    {&_44p9_l0_0,&_44p9_l0_1,0},
    {&_44p9_l1_0,&_44p9_p7_2,&_44p9_p7_3},
   }
};


static const vorbis_info_mapping0 _map_nominal_51[2]={
  {2, {0,0,0,0,0,1}, {0,2}, {0,2}, 4,{0,3,0,0},{2,4,1,3}},
  {2, {0,0,0,0,0,1}, {1,2}, {1,2}, 4,{0,3,0,0},{2,4,1,3}}
};
static const vorbis_info_mapping0 _map_nominal_51u[2]={
  {2, {0,0,0,0,0,1}, {0,2}, {0,2}, 0,{0},{0}},
  {2, {0,0,0,0,0,1}, {1,2}, {1,2}, 0,{0},{0}}
};

static const vorbis_residue_template _res_44p51_n1[]={
  {2,0,30,  &_residue_44p_lo,
   &_huff_book__44pn1_short,&_huff_book__44pn1_short,
   &_resbook_44p_n1,&_resbook_44p_n1},

  {2,0,30,  &_residue_44p_lo,
   &_huff_book__44pn1_long,&_huff_book__44pn1_long,
   &_resbook_44p_n1,&_resbook_44p_n1},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44pn1_lfe,&_huff_book__44pn1_lfe,
   &_resbook_44p_ln1,&_resbook_44p_ln1}
};
static const vorbis_residue_template _res_44p51_0[]={
  {2,0,15,  &_residue_44p_lo,
   &_huff_book__44p0_short,&_huff_book__44p0_short,
   &_resbook_44p_0,&_resbook_44p_0},

  {2,0,30,  &_residue_44p_lo,
   &_huff_book__44p0_long,&_huff_book__44p0_long,
   &_resbook_44p_0,&_resbook_44p_0},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p0_lfe,&_huff_book__44p0_lfe,
   &_resbook_44p_l0,&_resbook_44p_l0}
};
static const vorbis_residue_template _res_44p51_1[]={
  {2,0,15,  &_residue_44p_lo,
   &_huff_book__44p1_short,&_huff_book__44p1_short,
   &_resbook_44p_1,&_resbook_44p_1},

  {2,0,30,  &_residue_44p_lo,
   &_huff_book__44p1_long,&_huff_book__44p1_long,
   &_resbook_44p_1,&_resbook_44p_1},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p1_lfe,&_huff_book__44p1_lfe,
   &_resbook_44p_l1,&_resbook_44p_l1}
};
static const vorbis_residue_template _res_44p51_2[]={
  {2,0,15,  &_residue_44p,
   &_huff_book__44p2_short,&_huff_book__44p2_short,
   &_resbook_44p_2,&_resbook_44p_2},

  {2,0,30,  &_residue_44p,
   &_huff_book__44p2_long,&_huff_book__44p2_long,
   &_resbook_44p_2,&_resbook_44p_2},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p2_lfe,&_huff_book__44p2_lfe,
   &_resbook_44p_l2,&_resbook_44p_l2}
};
static const vorbis_residue_template _res_44p51_3[]={
  {2,0,15,  &_residue_44p,
   &_huff_book__44p3_short,&_huff_book__44p3_short,
   &_resbook_44p_3,&_resbook_44p_3},

  {2,0,30,  &_residue_44p,
   &_huff_book__44p3_long,&_huff_book__44p3_long,
   &_resbook_44p_3,&_resbook_44p_3},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p3_lfe,&_huff_book__44p3_lfe,
   &_resbook_44p_l3,&_resbook_44p_l3}
};
static const vorbis_residue_template _res_44p51_4[]={
  {2,0,15,  &_residue_44p,
   &_huff_book__44p4_short,&_huff_book__44p4_short,
   &_resbook_44p_4,&_resbook_44p_4},

  {2,0,30,  &_residue_44p,
   &_huff_book__44p4_long,&_huff_book__44p4_long,
   &_resbook_44p_4,&_resbook_44p_4},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p4_lfe,&_huff_book__44p4_lfe,
   &_resbook_44p_l4,&_resbook_44p_l4}
};
static const vorbis_residue_template _res_44p51_5[]={
  {2,0,15,  &_residue_44p_hi,
   &_huff_book__44p5_short,&_huff_book__44p5_short,
   &_resbook_44p_5,&_resbook_44p_5},

  {2,0,30,  &_residue_44p_hi,
   &_huff_book__44p5_long,&_huff_book__44p5_long,
   &_resbook_44p_5,&_resbook_44p_5},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p5_lfe,&_huff_book__44p5_lfe,
   &_resbook_44p_l5,&_resbook_44p_l5}
};
static const vorbis_residue_template _res_44p51_6[]={
  {2,0,15,  &_residue_44p_hi,
   &_huff_book__44p6_short,&_huff_book__44p6_short,
   &_resbook_44p_6,&_resbook_44p_6},

  {2,0,30,  &_residue_44p_hi,
   &_huff_book__44p6_long,&_huff_book__44p6_long,
   &_resbook_44p_6,&_resbook_44p_6},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p6_lfe,&_huff_book__44p6_lfe,
   &_resbook_44p_l6,&_resbook_44p_l6}
};


static const vorbis_residue_template _res_44p51_7[]={
  {2,0,15,  &_residue_44p_hi,
   &_huff_book__44p7_short,&_huff_book__44p7_short,
   &_resbook_44p_7,&_resbook_44p_7},

  {2,0,30,  &_residue_44p_hi,
   &_huff_book__44p7_long,&_huff_book__44p7_long,
   &_resbook_44p_7,&_resbook_44p_7},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p6_lfe,&_huff_book__44p6_lfe,
   &_resbook_44p_l6,&_resbook_44p_l6}
};
static const vorbis_residue_template _res_44p51_8[]={
  {2,0,15,  &_residue_44p_hi,
   &_huff_book__44p8_short,&_huff_book__44p8_short,
   &_resbook_44p_8,&_resbook_44p_8},

  {2,0,30,  &_residue_44p_hi,
   &_huff_book__44p8_long,&_huff_book__44p8_long,
   &_resbook_44p_8,&_resbook_44p_8},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p6_lfe,&_huff_book__44p6_lfe,
   &_resbook_44p_l6,&_resbook_44p_l6}
};
static const vorbis_residue_template _res_44p51_9[]={
  {2,0,15,  &_residue_44p_hi,
   &_huff_book__44p9_short,&_huff_book__44p9_short,
   &_resbook_44p_9,&_resbook_44p_9},

  {2,0,30,  &_residue_44p_hi,
   &_huff_book__44p9_long,&_huff_book__44p9_long,
   &_resbook_44p_9,&_resbook_44p_9},

  {1,2,6,  &_residue_44p_lfe,
   &_huff_book__44p6_lfe,&_huff_book__44p6_lfe,
   &_resbook_44p_l6,&_resbook_44p_l6}
};

static const vorbis_mapping_template _mapres_template_44_51[]={
  { _map_nominal_51, _res_44p51_n1 }, /* -1 */
  { _map_nominal_51, _res_44p51_0 }, /* 0 */
  { _map_nominal_51, _res_44p51_1 }, /* 1 */
  { _map_nominal_51, _res_44p51_2 }, /* 2 */
  { _map_nominal_51, _res_44p51_3 }, /* 3 */
  { _map_nominal_51, _res_44p51_4 }, /* 4 */
  { _map_nominal_51u, _res_44p51_5 }, /* 5 */
  { _map_nominal_51u, _res_44p51_6 }, /* 6 */
  { _map_nominal_51u, _res_44p51_7 }, /* 7 */
  { _map_nominal_51u, _res_44p51_8 }, /* 8 */
  { _map_nominal_51u, _res_44p51_9 }, /* 9 */
};
