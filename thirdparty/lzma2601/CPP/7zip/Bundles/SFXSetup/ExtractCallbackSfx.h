// ExtractCallbackSfx.h

#ifndef ZIP7_INC_EXTRACT_CALLBACK_SFX_H
#define ZIP7_INC_EXTRACT_CALLBACK_SFX_H

#include "resource.h"

#include "../../../Windows/ResourceString.h"

#include "../../Archive/IArchive.h"

#include "../../Common/FileStreams.h"
#include "../../ICoder.h"

#include "../../UI/FileManager/LangUtils.h"

#ifndef _NO_PROGRESS
#include "../../UI/FileManager/ProgressDialog.h"
#endif
#include "../../UI/Common/ArchiveOpenCallback.h"

class CExtractCallbackImp Z7_final:
  public IArchiveExtractCallback,
  public IOpenCallbackUI,
  public CMyUnknownImp
{
  Z7_COM_UNKNOWN_IMP_0
  Z7_IFACE_COM7_IMP(IProgress)
  Z7_IFACE_COM7_IMP(IArchiveExtractCallback)
  Z7_IFACE_IMP(IOpenCallbackUI)

  CMyComPtr<IInArchive> _archiveHandler;
  FString _directoryPath;
  UString _filePath;
  FString _diskFilePath;

  bool _extractMode;
  struct CProcessedFileInfo
  {
    FILETIME MTime;
    bool IsDir;
    UInt32 Attributes;
  } _processedFileInfo;

  COutFileStream *_outFileStreamSpec;
  CMyComPtr<ISequentialOutStream> _outFileStream;

  UString _itemDefaultName;
  FILETIME _defaultMTime;
  UInt32 _defaultAttributes;

  void CreateComplexDirectory(const UStringVector &dirPathParts);
public:
  #ifndef _NO_PROGRESS
  CProgressDialog ProgressDialog;
  #endif

  bool _isCorrupt;
  UString _message;

  void Init(IInArchive *archiveHandler,
    const FString &directoryPath,
    const UString &itemDefaultName,
    const FILETIME &defaultMTime,
    UInt32 defaultAttributes);

  #ifndef _NO_PROGRESS
  HRESULT StartProgressDialog(const UString &title, NWindows::CThread &thread)
  {
    ProgressDialog.Create(title, thread, NULL);
    {
      ProgressDialog.SetText(LangString(IDS_PROGRESS_EXTRACTING));
    }

    ProgressDialog.Show(SW_SHOWNORMAL);
    return S_OK;
  }
  ~CExtractCallbackImp() { ProgressDialog.Destroy(); }
  #endif

};

#endif
