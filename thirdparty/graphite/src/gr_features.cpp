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
#include "graphite2/Font.h"
#include "inc/Face.h"
#include "inc/FeatureMap.h"
#include "inc/FeatureVal.h"
#include "inc/NameTable.h"

using namespace graphite2;

extern "C" {


gr_uint16 gr_fref_feature_value(const gr_feature_ref* pfeatureref, const gr_feature_val* feats)    //returns 0 if either pointer is NULL
{
    if (!pfeatureref || !feats) return 0;

    return pfeatureref->getFeatureVal(*feats);
}


int gr_fref_set_feature_value(const gr_feature_ref* pfeatureref, gr_uint16 val, gr_feature_val* pDest)
{
    if (!pfeatureref || !pDest) return 0;

    return pfeatureref->applyValToFeature(val, *pDest);
}


gr_uint32 gr_fref_id(const gr_feature_ref* pfeatureref)    //returns 0 if pointer is NULL
{
  if (!pfeatureref)
    return 0;

  return pfeatureref->getId();
}


gr_uint16 gr_fref_n_values(const gr_feature_ref* pfeatureref)
{
    if(!pfeatureref)
        return 0;
    return pfeatureref->getNumSettings();
}


gr_int16 gr_fref_value(const gr_feature_ref* pfeatureref, gr_uint16 settingno)
{
    if(!pfeatureref || (settingno >= pfeatureref->getNumSettings()))
    {
        return 0;
    }
    return pfeatureref->getSettingValue(settingno);
}


void* gr_fref_label(const gr_feature_ref* pfeatureref, gr_uint16 *langId, gr_encform utf, gr_uint32 *length)
{
    if(!pfeatureref)
    {
        langId = 0;
        length = 0;
        return NULL;
    }
    uint16 label = pfeatureref->getNameId();
    NameTable * names = pfeatureref->getFace().nameTable();
    if (!names)
    {
        langId = 0;
        length = 0;
        return NULL;
    }
    return names->getName(*langId, label, utf, *length);
}


void* gr_fref_value_label(const gr_feature_ref*pfeatureref, gr_uint16 setting,
    gr_uint16 *langId, gr_encform utf, gr_uint32 *length)
{
    if(!pfeatureref || (setting >= pfeatureref->getNumSettings()))
    {
        langId = 0;
        length = 0;
        return NULL;
    }
    uint16 label = pfeatureref->getSettingName(setting);
    NameTable * names = pfeatureref->getFace().nameTable();
    if (!names)
    {
        langId = 0;
        length = 0;
        return NULL;
    }
    return names->getName(*langId, label, utf, *length);
}


void gr_label_destroy(void * label)
{
    free(label);
}

gr_feature_val* gr_featureval_clone(const gr_feature_val* pfeatures/*may be NULL*/)
{                      //When finished with the Features, call features_destroy
    return static_cast<gr_feature_val*>(pfeatures ? new Features(*pfeatures) : new Features);
}

void gr_featureval_destroy(gr_feature_val *p)
{
    delete static_cast<Features*>(p);
}


} // extern "C"
