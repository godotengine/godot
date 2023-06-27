/*
 * Copyright © 2018  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 */

#include "hb-ot-face.hh"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-cff2-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-kern-table.hh"
#include "hb-ot-meta-table.hh"
#include "hb-ot-name-table.hh"
#include "hb-ot-post-table.hh"
#include "OT/Color/CBDT/CBDT.hh"
#include "OT/Color/sbix/sbix.hh"
#include "OT/Color/svg/svg.hh"
#include "hb-ot-layout-gdef-table.hh"
#include "hb-ot-layout-gsub-table.hh"
#include "hb-ot-layout-gpos-table.hh"


void hb_ot_face_t::init0 (hb_face_t *face)
{
  this->face = face;
#define HB_OT_TABLE(Namespace, Type) Type.init0 ();
#include "hb-ot-face-table-list.hh"
#undef HB_OT_TABLE
}
void hb_ot_face_t::fini ()
{
#define HB_OT_TABLE(Namespace, Type) Type.fini ();
#include "hb-ot-face-table-list.hh"
#undef HB_OT_TABLE
}
