/*
  Copyright (c) 2005-2009, The Musepack Development Team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

  * Neither the name of the The Musepack Development Team nor the
  names of its contributors may be used to endorse or promote
  products derived from this software without specific prior
  written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/// \file synth_filter.c
/// Synthesis functions.
/// \todo document me
#include <string.h>
#include "mpcdec.h"
#include "decoder.h"
#include "mpcdec_math.h"

/* C O N S T A N T S */
#define MPC_FIXED_POINT_SYNTH_FIX 2

#undef _
#ifdef MPC_FIXED_POINT
#define _(value)  MPC_MAKE_FRACT_CONST((double)value/(double)(0x40000))
#else
#define _(value)  MAKE_MPC_SAMPLE((double)value/(double)(0x10000))
#endif


static const MPC_SAMPLE_FORMAT  Di_opt [32] [16] = {
    { _(  0), _( -29), _( 213), _( -459), _( 2037), _(-5153), _(  6574), _(-37489), _(75038), _(37489), _(6574), _( 5153), _(2037), _( 459), _(213), _(29) },
    { _( -1), _( -31), _( 218), _( -519), _( 2000), _(-5517), _(  5959), _(-39336), _(74992), _(35640), _(7134), _( 4788), _(2063), _( 401), _(208), _(26) },
    { _( -1), _( -35), _( 222), _( -581), _( 1952), _(-5879), _(  5288), _(-41176), _(74856), _(33791), _(7640), _( 4425), _(2080), _( 347), _(202), _(24) },
    { _( -1), _( -38), _( 225), _( -645), _( 1893), _(-6237), _(  4561), _(-43006), _(74630), _(31947), _(8092), _( 4063), _(2087), _( 294), _(196), _(21) },
    { _( -1), _( -41), _( 227), _( -711), _( 1822), _(-6589), _(  3776), _(-44821), _(74313), _(30112), _(8492), _( 3705), _(2085), _( 244), _(190), _(19) },
    { _( -1), _( -45), _( 228), _( -779), _( 1739), _(-6935), _(  2935), _(-46617), _(73908), _(28289), _(8840), _( 3351), _(2075), _( 197), _(183), _(17) },
    { _( -1), _( -49), _( 228), _( -848), _( 1644), _(-7271), _(  2037), _(-48390), _(73415), _(26482), _(9139), _( 3004), _(2057), _( 153), _(176), _(16) },
    { _( -2), _( -53), _( 227), _( -919), _( 1535), _(-7597), _(  1082), _(-50137), _(72835), _(24694), _(9389), _( 2663), _(2032), _( 111), _(169), _(14) },
    { _( -2), _( -58), _( 224), _( -991), _( 1414), _(-7910), _(    70), _(-51853), _(72169), _(22929), _(9592), _( 2330), _(2001), _(  72), _(161), _(13) },
    { _( -2), _( -63), _( 221), _(-1064), _( 1280), _(-8209), _(  -998), _(-53534), _(71420), _(21189), _(9750), _( 2006), _(1962), _(  36), _(154), _(11) },
    { _( -2), _( -68), _( 215), _(-1137), _( 1131), _(-8491), _( -2122), _(-55178), _(70590), _(19478), _(9863), _( 1692), _(1919), _(   2), _(147), _(10) },
    { _( -3), _( -73), _( 208), _(-1210), _(  970), _(-8755), _( -3300), _(-56778), _(69679), _(17799), _(9935), _( 1388), _(1870), _( -29), _(139), _( 9) },
    { _( -3), _( -79), _( 200), _(-1283), _(  794), _(-8998), _( -4533), _(-58333), _(68692), _(16155), _(9966), _( 1095), _(1817), _( -57), _(132), _( 8) },
    { _( -4), _( -85), _( 189), _(-1356), _(  605), _(-9219), _( -5818), _(-59838), _(67629), _(14548), _(9959), _(  814), _(1759), _( -83), _(125), _( 7) },
    { _( -4), _( -91), _( 177), _(-1428), _(  402), _(-9416), _( -7154), _(-61289), _(66494), _(12980), _(9916), _(  545), _(1698), _(-106), _(117), _( 7) },
    { _( -5), _( -97), _( 163), _(-1498), _(  185), _(-9585), _( -8540), _(-62684), _(65290), _(11455), _(9838), _(  288), _(1634), _(-127), _(111), _( 6) },
    { _( -5), _(-104), _( 146), _(-1567), _(  -45), _(-9727), _( -9975), _(-64019), _(64019), _( 9975), _(9727), _(   45), _(1567), _(-146), _(104), _( 5) },
    { _( -6), _(-111), _( 127), _(-1634), _( -288), _(-9838), _(-11455), _(-65290), _(62684), _( 8540), _(9585), _( -185), _(1498), _(-163), _( 97), _( 5) },
    { _( -7), _(-117), _( 106), _(-1698), _( -545), _(-9916), _(-12980), _(-66494), _(61289), _( 7154), _(9416), _( -402), _(1428), _(-177), _( 91), _( 4) },
    { _( -7), _(-125), _(  83), _(-1759), _( -814), _(-9959), _(-14548), _(-67629), _(59838), _( 5818), _(9219), _( -605), _(1356), _(-189), _( 85), _( 4) },
    { _( -8), _(-132), _(  57), _(-1817), _(-1095), _(-9966), _(-16155), _(-68692), _(58333), _( 4533), _(8998), _( -794), _(1283), _(-200), _( 79), _( 3) },
    { _( -9), _(-139), _(  29), _(-1870), _(-1388), _(-9935), _(-17799), _(-69679), _(56778), _( 3300), _(8755), _( -970), _(1210), _(-208), _( 73), _( 3) },
    { _(-10), _(-147), _(  -2), _(-1919), _(-1692), _(-9863), _(-19478), _(-70590), _(55178), _( 2122), _(8491), _(-1131), _(1137), _(-215), _( 68), _( 2) },
    { _(-11), _(-154), _( -36), _(-1962), _(-2006), _(-9750), _(-21189), _(-71420), _(53534), _(  998), _(8209), _(-1280), _(1064), _(-221), _( 63), _( 2) },
    { _(-13), _(-161), _( -72), _(-2001), _(-2330), _(-9592), _(-22929), _(-72169), _(51853), _(  -70), _(7910), _(-1414), _( 991), _(-224), _( 58), _( 2) },
    { _(-14), _(-169), _(-111), _(-2032), _(-2663), _(-9389), _(-24694), _(-72835), _(50137), _(-1082), _(7597), _(-1535), _( 919), _(-227), _( 53), _( 2) },
    { _(-16), _(-176), _(-153), _(-2057), _(-3004), _(-9139), _(-26482), _(-73415), _(48390), _(-2037), _(7271), _(-1644), _( 848), _(-228), _( 49), _( 1) },
    { _(-17), _(-183), _(-197), _(-2075), _(-3351), _(-8840), _(-28289), _(-73908), _(46617), _(-2935), _(6935), _(-1739), _( 779), _(-228), _( 45), _( 1) },
    { _(-19), _(-190), _(-244), _(-2085), _(-3705), _(-8492), _(-30112), _(-74313), _(44821), _(-3776), _(6589), _(-1822), _( 711), _(-227), _( 41), _( 1) },
    { _(-21), _(-196), _(-294), _(-2087), _(-4063), _(-8092), _(-31947), _(-74630), _(43006), _(-4561), _(6237), _(-1893), _( 645), _(-225), _( 38), _( 1) },
    { _(-24), _(-202), _(-347), _(-2080), _(-4425), _(-7640), _(-33791), _(-74856), _(41176), _(-5288), _(5879), _(-1952), _( 581), _(-222), _( 35), _( 1) },
    { _(-26), _(-208), _(-401), _(-2063), _(-4788), _(-7134), _(-35640), _(-74992), _(39336), _(-5959), _(5517), _(-2000), _( 519), _(-218), _( 31), _( 1) }
};

#undef  _

static void
mpc_compute_new_V(const MPC_SAMPLE_FORMAT* p_sample, MPC_SAMPLE_FORMAT* pV)
{
    // Calculating new V-buffer values for left channel
    // calculate new V-values (ISO-11172-3, p. 39)
    // based upon fast-MDCT algorithm by Byeong Gi Lee
    MPC_SAMPLE_FORMAT A00, A01, A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15;
    MPC_SAMPLE_FORMAT B00, B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B13, B14, B15;
    MPC_SAMPLE_FORMAT tmp;

    A00 = p_sample[ 0] + p_sample[31];
    A01 = p_sample[ 1] + p_sample[30];
    A02 = p_sample[ 2] + p_sample[29];
    A03 = p_sample[ 3] + p_sample[28];
    A04 = p_sample[ 4] + p_sample[27];
    A05 = p_sample[ 5] + p_sample[26];
    A06 = p_sample[ 6] + p_sample[25];
    A07 = p_sample[ 7] + p_sample[24];
    A08 = p_sample[ 8] + p_sample[23];
    A09 = p_sample[ 9] + p_sample[22];
    A10 = p_sample[10] + p_sample[21];
    A11 = p_sample[11] + p_sample[20];
    A12 = p_sample[12] + p_sample[19];
    A13 = p_sample[13] + p_sample[18];
    A14 = p_sample[14] + p_sample[17];
    A15 = p_sample[15] + p_sample[16];

    B00 = A00 + A15;
    B01 = A01 + A14;
    B02 = A02 + A13;
    B03 = A03 + A12;
    B04 = A04 + A11;
    B05 = A05 + A10;
    B06 = A06 + A09;
    B07 = A07 + A08;;
    B08 = MPC_SCALE_CONST((A00 - A15) , 0.5024192929f , 31);
    B09 = MPC_SCALE_CONST((A01 - A14) , 0.5224986076f , 31);
    B10 = MPC_SCALE_CONST((A02 - A13) , 0.5669440627f , 31);
    B11 = MPC_SCALE_CONST((A03 - A12) , 0.6468217969f , 31);
    B12 = MPC_SCALE_CONST((A04 - A11) , 0.7881546021f , 31);
    B13 = MPC_SCALE_CONST((A05 - A10) , 1.0606776476f , 30);
    B14 = MPC_SCALE_CONST((A06 - A09) , 1.7224471569f , 30);
    B15 = MPC_SCALE_CONST((A07 - A08) , 5.1011486053f , 28);

    A00 =  B00 + B07;
    A01 =  B01 + B06;
    A02 =  B02 + B05;
    A03 =  B03 + B04;
    A04 = MPC_SCALE_CONST((B00 - B07) , 0.5097956061f , 31);
    A05 = MPC_SCALE_CONST((B01 - B06) , 0.6013448834f , 31);
    A06 = MPC_SCALE_CONST((B02 - B05) , 0.8999761939f , 31);
    A07 = MPC_SCALE_CONST((B03 - B04) , 2.5629155636f , 29);
    A08 =  B08 + B15;
    A09 =  B09 + B14;
    A10 =  B10 + B13;
    A11 =  B11 + B12;
    A12 = MPC_SCALE_CONST((B08 - B15) , 0.5097956061f , 31);
    A13 = MPC_SCALE_CONST((B09 - B14) , 0.6013448834f , 31);
    A14 = MPC_SCALE_CONST((B10 - B13) , 0.8999761939f , 31);
    A15 = MPC_SCALE_CONST((B11 - B12) , 2.5629155636f , 29);

    B00 =  A00 + A03;
    B01 =  A01 + A02;
    B02 = MPC_MULTIPLY_FRACT_CONST_FIX((A00 - A03) , 0.5411961079f , 1);
    B03 = MPC_MULTIPLY_FRACT_CONST_FIX((A01 - A02) , 1.3065630198f , 2);
    B04 =  A04 + A07;
    B05 =  A05 + A06;
    B06 = MPC_MULTIPLY_FRACT_CONST_FIX((A04 - A07) , 0.5411961079f , 1);
    B07 = MPC_MULTIPLY_FRACT_CONST_FIX((A05 - A06) , 1.3065630198f , 2);
    B08 =  A08 + A11;
    B09 =  A09 + A10;
    B10 = MPC_MULTIPLY_FRACT_CONST_FIX((A08 - A11) , 0.5411961079f , 1);
    B11 = MPC_MULTIPLY_FRACT_CONST_FIX((A09 - A10) , 1.3065630198f , 2);
    B12 =  A12 + A15;
    B13 =  A13 + A14;
    B14 = MPC_MULTIPLY_FRACT_CONST_FIX((A12 - A15) , 0.5411961079f , 1);
    B15 = MPC_MULTIPLY_FRACT_CONST_FIX((A13 - A14) , 1.3065630198f , 2);

    A00 =  B00 + B01;
    A01 = MPC_MULTIPLY_FRACT_CONST_FIX((B00 - B01) , 0.7071067691f , 1);
    A02 =  B02 + B03;
    A03 = MPC_MULTIPLY_FRACT_CONST_FIX((B02 - B03) , 0.7071067691f , 1);
    A04 =  B04 + B05;
    A05 = MPC_MULTIPLY_FRACT_CONST_FIX((B04 - B05) , 0.7071067691f , 1);
    A06 =  B06 + B07;
    A07 = MPC_MULTIPLY_FRACT_CONST_FIX((B06 - B07) , 0.7071067691f , 1);
    A08 =  B08 + B09;
    A09 = MPC_MULTIPLY_FRACT_CONST_FIX((B08 - B09) , 0.7071067691f , 1);
    A10 =  B10 + B11;
    A11 = MPC_MULTIPLY_FRACT_CONST_FIX((B10 - B11) , 0.7071067691f , 1);
    A12 =  B12 + B13;
    A13 = MPC_MULTIPLY_FRACT_CONST_FIX((B12 - B13) , 0.7071067691f , 1);
    A14 =  B14 + B15;
    A15 = MPC_MULTIPLY_FRACT_CONST_FIX((B14 - B15) , 0.7071067691f , 1);

    pV[48] = -A00;
    pV[ 0] =  A01;
    pV[40] = -A02 - (pV[ 8] = A03);
    pV[36] = -((pV[ 4] = A05 + (pV[12] = A07)) + A06);
    pV[44] = - A04 - A06 - A07;
    pV[ 6] = (pV[10] = A11 + (pV[14] = A15)) + A13;
    pV[38] = (pV[34] = -(pV[ 2] = A09 + A13 + A15) - A14) + A09 - A10 - A11;
    pV[46] = (tmp = -(A12 + A14 + A15)) - A08;
    pV[42] = tmp - A10 - A11;

    A00 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 0] - p_sample[31]) , 0.5006030202f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A01 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 1] - p_sample[30]) , 0.5054709315f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A02 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 2] - p_sample[29]) , 0.5154473186f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A03 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 3] - p_sample[28]) , 0.5310425758f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A04 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 4] - p_sample[27]) , 0.5531039238f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A05 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 5] - p_sample[26]) , 0.5829349756f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A06 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 6] - p_sample[25]) , 0.6225041151f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A07 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 7] - p_sample[24]) , 0.6748083234f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A08 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 8] - p_sample[23]) , 0.7445362806f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A09 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[ 9] - p_sample[22]) , 0.8393496275f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A10 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[10] - p_sample[21]) , 0.9725682139f ,     MPC_FIXED_POINT_SYNTH_FIX);
#if MPC_FIXED_POINT_SYNTH_FIX>=2
    A11 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[11] - p_sample[20]) , 1.1694399118f ,     MPC_FIXED_POINT_SYNTH_FIX);
    A12 = MPC_MULTIPLY_FRACT_CONST_SHR((p_sample[12] - p_sample[19]) , 1.4841645956f ,     MPC_FIXED_POINT_SYNTH_FIX);
#else
    A11 = MPC_SCALE_CONST_SHR         ((p_sample[11] - p_sample[20]) , 1.1694399118f , 30, MPC_FIXED_POINT_SYNTH_FIX);
    A12 = MPC_SCALE_CONST_SHR         ((p_sample[12] - p_sample[19]) , 1.4841645956f , 30, MPC_FIXED_POINT_SYNTH_FIX);
#endif
    A13 = MPC_SCALE_CONST_SHR         ((p_sample[13] - p_sample[18]) , 2.0577809811f , 29, MPC_FIXED_POINT_SYNTH_FIX);
    A14 = MPC_SCALE_CONST_SHR         ((p_sample[14] - p_sample[17]) , 3.4076085091f , 29, MPC_FIXED_POINT_SYNTH_FIX);
    A15 = MPC_SCALE_CONST_SHR         ((p_sample[15] - p_sample[16]) , 10.1900081635f, 27 ,MPC_FIXED_POINT_SYNTH_FIX);

    B00 =  A00 + A15;
    B01 =  A01 + A14;
    B02 =  A02 + A13;
    B03 =  A03 + A12;
    B04 =  A04 + A11;
    B05 =  A05 + A10;
    B06 =  A06 + A09;
    B07 =  A07 + A08;
    B08 = MPC_SCALE_CONST((A00 - A15) , 0.5024192929f , 31);
    B09 = MPC_SCALE_CONST((A01 - A14) , 0.5224986076f , 31);
    B10 = MPC_SCALE_CONST((A02 - A13) , 0.5669440627f , 31);
    B11 = MPC_SCALE_CONST((A03 - A12) , 0.6468217969f , 31);
    B12 = MPC_SCALE_CONST((A04 - A11) , 0.7881546021f , 31);
    B13 = MPC_SCALE_CONST((A05 - A10) , 1.0606776476f , 30);
    B14 = MPC_SCALE_CONST((A06 - A09) , 1.7224471569f , 30);
    B15 = MPC_SCALE_CONST((A07 - A08) , 5.1011486053f , 28);

    A00 =  B00 + B07;
    A01 =  B01 + B06;
    A02 =  B02 + B05;
    A03 =  B03 + B04;
    A04 = MPC_SCALE_CONST((B00 - B07) , 0.5097956061f , 31);
    A05 = MPC_SCALE_CONST((B01 - B06) , 0.6013448834f , 31);
    A06 = MPC_SCALE_CONST((B02 - B05) , 0.8999761939f , 31);
    A07 = MPC_SCALE_CONST((B03 - B04) , 2.5629155636f , 29);
    A08 =  B08 + B15;
    A09 =  B09 + B14;
    A10 =  B10 + B13;
    A11 =  B11 + B12;
    A12 = MPC_SCALE_CONST((B08 - B15) , 0.5097956061f , 31);
    A13 = MPC_SCALE_CONST((B09 - B14) , 0.6013448834f , 31);
    A14 = MPC_SCALE_CONST((B10 - B13) , 0.8999761939f , 31);
    A15 = MPC_SCALE_CONST((B11 - B12) , 2.5629155636f , 29);

    B00 =  A00 + A03;
    B01 =  A01 + A02;
    B02 = MPC_SCALE_CONST((A00 - A03) , 0.5411961079f , 31);
    B03 = MPC_SCALE_CONST((A01 - A02) , 1.3065630198f , 30);
    B04 =  A04 + A07;
    B05 =  A05 + A06;
    B06 = MPC_SCALE_CONST((A04 - A07) , 0.5411961079f , 31);
    B07 = MPC_SCALE_CONST((A05 - A06) , 1.3065630198f , 30);
    B08 =  A08 + A11;
    B09 =  A09 + A10;
    B10 = MPC_SCALE_CONST((A08 - A11) , 0.5411961079f , 31);
    B11 = MPC_SCALE_CONST((A09 - A10) , 1.3065630198f , 30);
    B12 =  A12 + A15;
    B13 =  A13 + A14;
    B14 = MPC_SCALE_CONST((A12 - A15) , 0.5411961079f , 31);
    B15 = MPC_SCALE_CONST((A13 - A14) , 1.3065630198f , 30);

    A00 = MPC_SHL(B00 + B01, MPC_FIXED_POINT_SYNTH_FIX);
    A01 = MPC_SCALE_CONST_SHL((B00 - B01) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A02 = MPC_SHL(B02 + B03, MPC_FIXED_POINT_SYNTH_FIX);
    A03 = MPC_SCALE_CONST_SHL((B02 - B03) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A04 = MPC_SHL(B04 + B05, MPC_FIXED_POINT_SYNTH_FIX);
    A05 = MPC_SCALE_CONST_SHL((B04 - B05) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A06 = MPC_SHL(B06 + B07, MPC_FIXED_POINT_SYNTH_FIX);
    A07 = MPC_SCALE_CONST_SHL((B06 - B07) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A08 = MPC_SHL(B08 + B09, MPC_FIXED_POINT_SYNTH_FIX);
    A09 = MPC_SCALE_CONST_SHL((B08 - B09) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A10 = MPC_SHL(B10 + B11, MPC_FIXED_POINT_SYNTH_FIX);
    A11 = MPC_SCALE_CONST_SHL((B10 - B11) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A12 = MPC_SHL(B12 + B13, MPC_FIXED_POINT_SYNTH_FIX);
    A13 = MPC_SCALE_CONST_SHL((B12 - B13) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);
    A14 = MPC_SHL(B14 + B15, MPC_FIXED_POINT_SYNTH_FIX);
    A15 = MPC_SCALE_CONST_SHL((B14 - B15) , 0.7071067691f , 31, MPC_FIXED_POINT_SYNTH_FIX);

    // mehrfach verwendete Ausdrcke: A04+A06+A07, A09+A13+A15
    pV[ 5] = (pV[11] = (pV[13] = A07 + (pV[15] = A15)) + A11) + A05 + A13;
    pV[ 7] = (pV[ 9] = A03 + A11 + A15) + A13;
    pV[33] = -(pV[ 1] = A01 + A09 + A13 + A15) - A14;
    pV[35] = -(pV[ 3] = A05 + A07 + A09 + A13 + A15) - A06 - A14;
    pV[37] = (tmp = -(A10 + A11 + A13 + A14 + A15)) - A05 - A06 - A07;
    pV[39] = tmp - A02 - A03;                      // abh�gig vom Befehl drber
    pV[41] = (tmp += A13 - A12) - A02 - A03;       // abh�gig vom Befehl 2 drber
    pV[43] = tmp - A04 - A06 - A07;                // abh�gig von Befehlen 1 und 3 drber
    pV[47] = (tmp = -(A08 + A12 + A14 + A15)) - A00;
    pV[45] = tmp - A04 - A06 - A07;                // abh�gig vom Befehl drber

    pV[32] = -pV[ 0];
    pV[31] = -pV[ 1];
    pV[30] = -pV[ 2];
    pV[29] = -pV[ 3];
    pV[28] = -pV[ 4];
    pV[27] = -pV[ 5];
    pV[26] = -pV[ 6];
    pV[25] = -pV[ 7];
    pV[24] = -pV[ 8];
    pV[23] = -pV[ 9];
    pV[22] = -pV[10];
    pV[21] = -pV[11];
    pV[20] = -pV[12];
    pV[19] = -pV[13];
    pV[18] = -pV[14];
    pV[17] = -pV[15];

    pV[63] =  pV[33];
    pV[62] =  pV[34];
    pV[61] =  pV[35];
    pV[60] =  pV[36];
    pV[59] =  pV[37];
    pV[58] =  pV[38];
    pV[57] =  pV[39];
    pV[56] =  pV[40];
    pV[55] =  pV[41];
    pV[54] =  pV[42];
    pV[53] =  pV[43];
    pV[52] =  pV[44];
    pV[51] =  pV[45];
    pV[50] =  pV[46];
    pV[49] =  pV[47];
}

static void
mpc_synthese_filter_float_internal(MPC_SAMPLE_FORMAT* p_out, MPC_SAMPLE_FORMAT* pV, const MPC_SAMPLE_FORMAT* pY, mpc_int_t channels)
{
    mpc_uint32_t n;
    for ( n = 0; n < 36; n++, pY += 32 )
    {
        MPC_SAMPLE_FORMAT*       pData = p_out;
        const MPC_SAMPLE_FORMAT* pD    = (const MPC_SAMPLE_FORMAT*) &Di_opt;
        mpc_int32_t k;
        pV -= 64;
        mpc_compute_new_V( pY, pV );
        for ( k = 0; k < 32; k++, pD += 16, pV++ )
        {
            *pData = MPC_SHL(
                     MPC_MULTIPLY_FRACT(pV[  0], pD[ 0]) + MPC_MULTIPLY_FRACT(pV[ 96], pD[ 1]) + MPC_MULTIPLY_FRACT(pV[128], pD[ 2]) + MPC_MULTIPLY_FRACT(pV[224], pD[ 3])
                   + MPC_MULTIPLY_FRACT(pV[256], pD[ 4]) + MPC_MULTIPLY_FRACT(pV[352], pD[ 5]) + MPC_MULTIPLY_FRACT(pV[384], pD[ 6]) + MPC_MULTIPLY_FRACT(pV[480], pD[ 7])
                   + MPC_MULTIPLY_FRACT(pV[512], pD[ 8]) + MPC_MULTIPLY_FRACT(pV[608], pD[ 9]) + MPC_MULTIPLY_FRACT(pV[640], pD[10]) + MPC_MULTIPLY_FRACT(pV[736], pD[11])
                   + MPC_MULTIPLY_FRACT(pV[768], pD[12]) + MPC_MULTIPLY_FRACT(pV[864], pD[13]) + MPC_MULTIPLY_FRACT(pV[896], pD[14]) + MPC_MULTIPLY_FRACT(pV[992], pD[15])
                   , 2);
            pData += channels;
        }
        pV    -= 32; //bleh
        p_out += 32 * channels;
    }
}

void
mpc_decoder_synthese_filter_float(mpc_decoder* p_dec, MPC_SAMPLE_FORMAT* p_out, mpc_int_t channels)
{
    /********* left channel ********/
    memmove(&p_dec->V_L[MPC_V_MEM], p_dec->V_L, 960 * sizeof *p_dec->V_L);
	mpc_synthese_filter_float_internal(p_out, &p_dec->V_L[MPC_V_MEM], p_dec->Y_L[0], channels);

    /******** right channel ********/
	if (channels > 1) {
		memmove(&p_dec->V_R[MPC_V_MEM], p_dec->V_R, 960 * sizeof *p_dec->V_R);
		mpc_synthese_filter_float_internal(p_out + 1, &p_dec->V_R[MPC_V_MEM], p_dec->Y_R[0], channels);
	}
}

/*******************************************/
/*                                         */
/*            dithered synthesis           */
/*                                         */
/*******************************************/

static const mpc_uint8_t Parity [256] = {  // parity
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0
};

/*
 *  This is a simple random number generator with good quality for audio purposes.
 *  It consists of two polycounters with opposite rotation direction and different
 *  periods. The periods are coprime, so the total period is the product of both.
 *
 *     -------------------------------------------------------------------------------------------------
 * +-> |31:30:29:28:27:26:25:24:23:22:21:20:19:18:17:16:15:14:13:12:11:10: 9: 8: 7: 6: 5: 4: 3: 2: 1: 0|
 * |   -------------------------------------------------------------------------------------------------
 * |                                                                          |  |  |  |     |        |
 * |                                                                          +--+--+--+-XOR-+--------+
 * |                                                                                      |
 * +--------------------------------------------------------------------------------------+
 *
 *     -------------------------------------------------------------------------------------------------
 *     |31:30:29:28:27:26:25:24:23:22:21:20:19:18:17:16:15:14:13:12:11:10: 9: 8: 7: 6: 5: 4: 3: 2: 1: 0| <-+
 *     -------------------------------------------------------------------------------------------------   |
 *       |  |           |  |                                                                               |
 *       +--+----XOR----+--+                                                                               |
 *                |                                                                                        |
 *                +----------------------------------------------------------------------------------------+
 *
 *
 *  The first has an period of 3*5*17*257*65537, the second of 7*47*73*178481,
 *  which gives a period of 18.410.713.077.675.721.215. The result is the
 *  XORed values of both generators.
 */
mpc_uint32_t
mpc_random_int(mpc_decoder* p_dec)
{
#if 1
    mpc_uint32_t  t1, t2, t3, t4;

    t3   = t1 = p_dec->__r1; t4   = t2 = p_dec->__r2; // Parity calculation is done via table lookup, this is also available
    t1  &= 0xF5;             t2 >>= 25;               // on CPUs without parity, can be implemented in C and avoid unpredictable
    t1   = Parity[t1];       t2  &= 0x63;             // jumps and slow rotate through the carry flag operations.
    t1 <<= 31;               t2   = Parity[t2];

    return (p_dec->__r1 = (t3 >> 1) | t1 ) ^ (p_dec->__r2 = (t4 + t4) | t2 );
#else
    return (p_dec->__r1 = (p_dec->__r1 >> 1) | ((mpc_uint32_t) Parity[ p_dec->__r1 & 0xF5] << 31))
         ^ (p_dec->__r2 = (p_dec->__r2 << 1) |  (mpc_uint32_t) Parity[(p_dec->__r2 >> 25) & 0x63]);
#endif
}
