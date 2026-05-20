// ExtractEngine.cpp

#include "StdAfx.h"

#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileName.h"
#include "../../../Windows/Thread.h"

#include "../../UI/Common/OpenArchive.h"

#include "../../UI/FileManager/FormatUtils.h"
#include "../../UI/FileManager/LangUtils.h"

#include "ExtractCallbackSfx.h"
#include "ExtractEngine.h"

using namespace NWindows;
using namespace NFile;
using namespace NDir;

static LPCSTR const kCantFindArchive = "Cannot find archive file";
static LPCSTR const kCantOpenArchive = "Cannot open the file as archive";

struct CThreadExtracting
{
  CCodecs *Codecs;
  FString FileName;
  FString DestFolder;

  CExtractCallbackImp *ExtractCallbackSpec;
  CMyComPtr<IArchiveExtractCallback> ExtractCallback;

  CArchiveLink ArchiveLink;
  HRESULT Result;
  UString ErrorMessage;

  void Process2()
  {
    NFind::CFileInfo fi;
    if (!fi.Find(FileName))
    {
      ErrorMessage = kCantFindArchive;
      Result = E_FAIL;
      return;
    }

    CObjectVector<COpenType> incl;
    CIntVector excl;
    COpenOptions options;
    options.codecs = Codecs;
    options.types = &incl;
    options.excludedFormats = &excl;
    options.filePath = fs2us(FileName);
    
    Result = ArchiveLink.Open2(options, ExtractCallbackSpec);
    if (Result != S_OK)
    {
      ErrorMessage = kCantOpenArchive;
      return;
    }

    FString dirPath = DestFolder;
    NName::NormalizeDirPathPrefix(dirPath);
    
    if (!CreateComplexDir(dirPath))
    {
      ErrorMessage = MyFormatNew(IDS_CANNOT_CREATE_FOLDER, fs2us(dirPath));
      Result = E_FAIL;
      return;
    }

    ExtractCallbackSpec->Init(ArchiveLink.GetArchive(), dirPath, (UString)"Default", fi.MTime, 0);

    Result = ArchiveLink.GetArchive()->Extract(NULL, (UInt32)(Int32)-1 , BoolToInt(false), ExtractCallback);
  }

  void Process()
  {
    try
    {
      #ifndef _NO_PROGRESS
      CProgressCloser closer(ExtractCallbackSpec->ProgressDialog);
      #endif
      Process2();
    }
    catch(...) { Result = E_FAIL; }
  }
  
  static THREAD_FUNC_DECL MyThreadFunction(void *param)
  {
    ((CThreadExtracting *)param)->Process();
    return 0;
  }
};

HRESULT ExtractArchive(CCodecs *codecs, const FString &fileName, const FString &destFolder,
    bool showProgress, bool &isCorrupt, UString &errorMessage)
{
  isCorrupt = false;
  CThreadExtracting t;

  t.Codecs = codecs;
  t.FileName = fileName;
  t.DestFolder = destFolder;

  t.ExtractCallbackSpec = new CExtractCallbackImp;
  t.ExtractCallback = t.ExtractCallbackSpec;
  
  #ifndef _NO_PROGRESS

  if (showProgress)
  {
    t.ExtractCallbackSpec->ProgressDialog.IconID = IDI_ICON;
    NWindows::CThread thread;
    const WRes wres = thread.Create(CThreadExtracting::MyThreadFunction, &t);
    if (wres != 0)
      return HRESULT_FROM_WIN32(wres);
    
    UString title;
    LangString(IDS_PROGRESS_EXTRACTING, title);
    t.ExtractCallbackSpec->StartProgressDialog(title, thread);
  }
  else

  #endif
  {
    t.Process2();
  }

  errorMessage = t.ErrorMessage;
  if (errorMessage.IsEmpty())
    errorMessage = t.ExtractCallbackSpec->_message;
  isCorrupt = t.ExtractCallbackSpec->_isCorrupt;
  return t.Result;
}
