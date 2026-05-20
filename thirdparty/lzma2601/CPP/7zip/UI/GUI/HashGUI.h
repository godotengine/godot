// HashGUI.h

#ifndef ZIP7_INC_HASH_GUI_H
#define ZIP7_INC_HASH_GUI_H

#include "../Common/HashCalc.h"
#include "../Common/Property.h"

HRESULT HashCalcGUI(
    DECL_EXTERNAL_CODECS_LOC_VARS
    const NWildcard::CCensor &censor,
    const CHashOptions &options,
    bool &messageWasDisplayed);

typedef CObjectVector<CProperty> CPropNameValPairs;

void AddValuePair(CPropNameValPairs &pairs, UINT resourceID, UInt64 value);
void AddSizeValue(UString &s, UInt64 value);
void AddSizeValuePair(CPropNameValPairs &pairs, UINT resourceID, UInt64 value);

void AddHashBundleRes(CPropNameValPairs &s, const CHashBundle &hb);
void AddHashBundleRes(UString &s, const CHashBundle &hb);

void ShowHashResults(const CPropNameValPairs &propPairs, HWND hwnd);
void ShowHashResults(const CHashBundle &hb, HWND hwnd);

#endif
