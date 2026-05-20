// GUI/ExtractGUI.h

#ifndef ZIP7_INC_EXTRACT_GUI_H
#define ZIP7_INC_EXTRACT_GUI_H

#include "../Common/Extract.h"

#include "../FileManager/ExtractCallback.h"

/*
  RESULT can be S_OK, even if there are errors!!!
  if RESULT == S_OK, check extractCallback->IsOK() after ExtractGUI().

  RESULT = E_ABORT - user break.
  RESULT != E_ABORT:
  {
   messageWasDisplayed = true  - message was displayed already.
   messageWasDisplayed = false - there was some internal error, so you must show error message.
  }
*/

HRESULT ExtractGUI(
    // DECL_EXTERNAL_CODECS_LOC_VARS
    CCodecs *codecs,
    const CObjectVector<COpenType> &formatIndices,
    const CIntVector &excludedFormatIndices,
    UStringVector &archivePaths,
    UStringVector &archivePathsFull,
    const NWildcard::CCensorNode &wildcardCensor,
    CExtractOptions &options,
    #ifndef Z7_SFX
    CHashBundle *hb,
    #endif
    bool showDialog,
    bool &messageWasDisplayed,
    CExtractCallbackImp *extractCallback,
    HWND hwndParent = NULL);

#endif
