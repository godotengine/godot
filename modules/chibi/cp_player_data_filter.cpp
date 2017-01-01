/*************************************************************************/
/*  cp_player_data_filter.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/


#include "cp_player_data.h"

static float filter_cutoff[256] = {
     130,  132,  134,  136,  138,  140,  142,  144,
     146,  148,  151,  153,  155,  157,  160,  162,
     164,  167,  169,  172,  174,  177,  179,  182,
     184,  187,  190,  193,  195,  198,  201,  204,
     207,  210,  213,  216,  220,  223,  226,  229,
     233,  236,  239,  243,  246,  250,  254,  257,
     261,  265,  269,  273,  277,  281,  285,  289,
     293,  297,  302,  306,  311,  315,  320,  324,
     329,  334,  339,  344,  349,  354,  359,  364,
     369,  375,  380,  386,  391,  397,  403,  409,
     415,  421,  427,  433,  440,  446,  452,  459,
     466,  472,  479,  486,  493,  501,  508,  515,
     523,  530,  538,  546,  554,  562,  570,  578,
     587,  595,  604,  613,  622,  631,  640,  649,
     659,  668,  678,  688,  698,  708,  718,  729,
     739,  750,  761,  772,  783,  795,  806,  818,
     830,  842,  854,  867,  880,  892,  905,  918,
     932,  945,  959,  973,  987, 1002, 1016, 1031,
    1046, 1061, 1077, 1092, 1108, 1124, 1141, 1157,
    1174, 1191, 1209, 1226, 1244, 1262, 1280, 1299,
    1318, 1337, 1357, 1376, 1396, 1417, 1437, 1458,
    1479, 1501, 1523, 1545, 1567, 1590, 1613, 1637,
    1661, 1685, 1709, 1734, 1760, 1785, 1811, 1837,
    1864, 1891, 1919, 1947, 1975, 2004, 2033, 2062,
    2093, 2123, 2154, 2185, 2217, 2249, 2282, 2315,
    2349, 2383, 2418, 2453, 2489, 2525, 2561, 2599,
    2637, 2675, 2714, 2753, 2793, 2834, 2875, 2917,
    2959, 3003, 3046, 3091, 3135, 3181, 3227, 3274,
    3322, 3370, 3419, 3469, 3520, 3571, 3623, 3675,
    3729, 3783, 3838, 3894, 3951, 4008, 4066, 4125,
    4186, 4246, 4308, 4371, 4434, 4499, 4564, 4631,
    4698, 4766, 4836, 4906, 4978, 5050, 5123, 5198
};


void CPPlayer::Filter_Control::process() {
	
	
	final_cutoff=it_cutoff;
	if (envelope_cutoff>=0) {
		
		envelope_cutoff=envelope_cutoff*255/64;
		final_cutoff=final_cutoff*envelope_cutoff/255;
		if (final_cutoff>=0xFF) final_cutoff=0xFE;
		
	}
	
}

void CPPlayer::Filter_Control::set_filter_parameters(int *p_cutoff,uint8_t *p_reso) {

	

	*p_cutoff=filter_cutoff[final_cutoff];
	*p_reso=it_reso;
}
