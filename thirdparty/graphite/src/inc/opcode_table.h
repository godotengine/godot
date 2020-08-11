/*  GRAPHITE2 LICENSING

    Copyright 2010, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/
// This file will be pulled into and integrated into a machine implmentation
// DO NOT build directly
#pragma once

#define do2(n)  do_(n) ,do_(n)
#define NILOP   0U

// types or parameters are: (.. is inclusive)
//      number - any byte
//      output_class - 0 .. silf.m_nClass
//      input_class - 0 .. silf.m_nClass
//      sattrnum - 0 .. 29 (gr_slatJWidth) , 55 (gr_slatUserDefn)
//      attrid - 0 .. silf.numUser() where sattrnum == 55; 0..silf.m_iMaxComp where sattrnum == 15 otherwise 0
//      gattrnum - 0 .. face->getGlyphFaceCache->numAttrs()
//      gmetric - 0 .. 11 (kgmetDescent)
//      featidx - 0 .. face.numFeatures()
//      level - any byte
static const opcode_t opcode_table[] =
{
    {{do2(nop)},                                    0, "NOP"},

    {{do2(push_byte)},                              1, "PUSH_BYTE"},                // number
    {{do2(push_byte_u)},                            1, "PUSH_BYTE_U"},              // number
    {{do2(push_short)},                             2, "PUSH_SHORT"},               // number number
    {{do2(push_short_u)},                           2, "PUSH_SHORT_U"},             // number number
    {{do2(push_long)},                              4, "PUSH_LONG"},                // number number number number

    {{do2(add)},                                    0, "ADD"},
    {{do2(sub)},                                    0, "SUB"},
    {{do2(mul)},                                    0, "MUL"},
    {{do2(div_)},                                   0, "DIV"},
    {{do2(min_)},                                   0, "MIN"},
    {{do2(max_)},                                   0, "MAX"},
    {{do2(neg)},                                    0, "NEG"},
    {{do2(trunc8)},                                 0, "TRUNC8"},
    {{do2(trunc16)},                                0, "TRUNC16"},

    {{do2(cond)},                                   0, "COND"},
    {{do2(and_)},                                   0, "AND"},      // 0x10
    {{do2(or_)},                                    0, "OR"},
    {{do2(not_)},                                   0, "NOT"},
    {{do2(equal)},                                  0, "EQUAL"},
    {{do2(not_eq_)},                                0, "NOT_EQ"},
    {{do2(less)},                                   0, "LESS"},
    {{do2(gtr)},                                    0, "GTR"},
    {{do2(less_eq)},                                0, "LESS_EQ"},
    {{do2(gtr_eq)},                                 0, "GTR_EQ"},   // 0x18

    {{do_(next), NILOP},                            0, "NEXT"},
    {{NILOP, NILOP},                                1, "NEXT_N"},                   // number <= smap.end - map
    {{do_(next), NILOP},                            0, "COPY_NEXT"},
    {{do_(put_glyph_8bit_obs), NILOP},              1, "PUT_GLYPH_8BIT_OBS"},       // output_class
    {{do_(put_subs_8bit_obs), NILOP},               3, "PUT_SUBS_8BIT_OBS"},        // slot input_class output_class
    {{do_(put_copy), NILOP},                        1, "PUT_COPY"},                 // slot
    {{do_(insert), NILOP},                          0, "INSERT"},
    {{do_(delete_), NILOP},                         0, "DELETE"},   // 0x20
    {{do_(assoc), NILOP},                     VARARGS, "ASSOC"},
    {{NILOP ,do_(cntxt_item)},                      2, "CNTXT_ITEM"},               // slot offset

    {{do_(attr_set), NILOP},                        1, "ATTR_SET"},                 // sattrnum
    {{do_(attr_add), NILOP},                        1, "ATTR_ADD"},                 // sattrnum
    {{do_(attr_sub), NILOP},                        1, "ATTR_SUB"},                 // sattrnum
    {{do_(attr_set_slot), NILOP},                   1, "ATTR_SET_SLOT"},            // sattrnum
    {{do_(iattr_set_slot), NILOP},                  2, "IATTR_SET_SLOT"},           // sattrnum attrid
    {{do2(push_slot_attr)},                         2, "PUSH_SLOT_ATTR"},           // sattrnum slot
    {{do2(push_glyph_attr_obs)},                    2, "PUSH_GLYPH_ATTR_OBS"},      // gattrnum slot
    {{do2(push_glyph_metric)},                      3, "PUSH_GLYPH_METRIC"},        // gmetric slot level
    {{do2(push_feat)},                              2, "PUSH_FEAT"},                // featidx slot

    {{do2(push_att_to_gattr_obs)},                  2, "PUSH_ATT_TO_GATTR_OBS"},    // gattrnum slot
    {{do2(push_att_to_glyph_metric)},               3, "PUSH_ATT_TO_GLYPH_METRIC"}, // gmetric slot level
    {{do2(push_islot_attr)},                        3, "PUSH_ISLOT_ATTR"},          // sattrnum slot attrid

    {{NILOP,NILOP},                                 3, "PUSH_IGLYPH_ATTR"},

    {{do2(pop_ret)},                                0, "POP_RET"},  // 0x30
    {{do2(ret_zero)},                               0, "RET_ZERO"},
    {{do2(ret_true)},                               0, "RET_TRUE"},

    {{do_(iattr_set), NILOP},                       2, "IATTR_SET"},                // sattrnum attrid
    {{do_(iattr_add), NILOP},                       2, "IATTR_ADD"},                // sattrnum attrid
    {{do_(iattr_sub), NILOP},                       2, "IATTR_SUB"},                // sattrnum attrid
    {{do2(push_proc_state)},                        1, "PUSH_PROC_STATE"},          // dummy
    {{do2(push_version)},                           0, "PUSH_VERSION"},
    {{do_(put_subs), NILOP},                        5, "PUT_SUBS"},                 // slot input_class input_class output_class output_class
    {{NILOP,NILOP},                                 0, "PUT_SUBS2"},
    {{NILOP,NILOP},                                 0, "PUT_SUBS3"},
    {{do_(put_glyph), NILOP},                       2, "PUT_GLYPH"},                // output_class output_class
    {{do2(push_glyph_attr)},                        3, "PUSH_GLYPH_ATTR"},          // gattrnum gattrnum slot
    {{do2(push_att_to_glyph_attr)},                 3, "PUSH_ATT_TO_GLYPH_ATTR"},   // gattrnum gattrnum slot
    {{do2(bor)},                                    0, "BITOR"},
    {{do2(band)},                                   0, "BITAND"},
    {{do2(bnot)},                                   0, "BITNOT"},   // 0x40
    {{do2(setbits)},                                4, "BITSET"},
    {{do_(set_feat), NILOP},                        2, "SET_FEAT"},                 // featidx slot
    // private opcodes for internal use only, comes after all other on disk opcodes.
    {{do_(temp_copy), NILOP},                       0, "TEMP_COPY"}
};
