// ArchiveOpenCallback.h

#ifndef ZIP7_INC_ARCHIVE_OPEN_CALLBACK_H
#define ZIP7_INC_ARCHIVE_OPEN_CALLBACK_H

#include "../../../Common/MyCom.h"

#include "../../../Windows/FileFind.h"

#include "../../Common/FileStreams.h"

#ifndef Z7_NO_CRYPTO
#include "../../IPassword.h"
#endif
#include "../../Archive/IArchive.h"

Z7_PURE_INTERFACES_BEGIN

#ifdef Z7_NO_CRYPTO

#define Z7_IFACEM_IOpenCallbackUI_Crypto(x)

#else

#define Z7_IFACEM_IOpenCallbackUI_Crypto(x) \
  virtual HRESULT Open_CryptoGetTextPassword(BSTR *password) x \
  /* virtual HRESULT Open_GetPasswordIfAny(bool &passwordIsDefined, UString &password) x */ \
  /* virtual bool Open_WasPasswordAsked() x */ \
  /* virtual void Open_Clear_PasswordWasAsked_Flag() x */  \
  
#endif

#define Z7_IFACEN_IOpenCallbackUI(x) \
  virtual HRESULT Open_CheckBreak() x \
  virtual HRESULT Open_SetTotal(const UInt64 *files, const UInt64 *bytes) x \
  virtual HRESULT Open_SetCompleted(const UInt64 *files, const UInt64 *bytes) x \
  virtual HRESULT Open_Finished() x \
  Z7_IFACEM_IOpenCallbackUI_Crypto(x)

Z7_IFACE_DECL_PURE(IOpenCallbackUI)

Z7_PURE_INTERFACES_END


class CMultiStreams Z7_final
{
public:
  struct CSubStream
  {
    CMyComPtr<IInStream> Stream;
    CInFileStream *FileSpec;
    FString Path;
    // UInt64 Size;
    UInt64 LocalPos;
    int Next; // next older
    int Prev; // prev newer
    // bool IsOpen;

    CSubStream():
        FileSpec(NULL),
        // Size(0),
        LocalPos(0),
        Next(-1),
        Prev(-1)
        // IsOpen(false)
        {}
  };

  CObjectVector<CSubStream> Streams;
private:
  // we must use critical section here, if we want to access from different volumnes simultaneously
  int Head; // newest
  int Tail; // oldest
  unsigned NumListItems;
  unsigned NumOpenFiles_AllowedMax;
public:

  CMultiStreams();
  void Init();
  HRESULT PrepareToOpenNew();
  void InsertToList(unsigned index);
  void RemoveFromList(CSubStream &s);
  void CloseFile(unsigned index);
  HRESULT EnsureOpen(unsigned index);
};


/*
  We need COpenCallbackImp class for multivolume processing.
  Also we use it as proxy from COM interfaces (IArchiveOpenCallback) to internal (IOpenCallbackUI) interfaces.
  If archive is multivolume:
    COpenCallbackImp object will exist after Open stage.
    COpenCallbackImp object will be deleted when last reference
      from each volume object (CInFileStreamVol) will be closed (when archive will be closed).
*/

class COpenCallbackImp Z7_final:
  public IArchiveOpenCallback,
  public IArchiveOpenVolumeCallback,
  public IArchiveOpenSetSubArchiveName,
 #ifndef Z7_NO_CRYPTO
  public ICryptoGetTextPassword,
 #endif
  public IProgress, // IProgress is used for 7zFM
  public CMyUnknownImp
{
  Z7_COM_QI_BEGIN2(IArchiveOpenCallback)
  Z7_COM_QI_ENTRY(IArchiveOpenVolumeCallback)
  Z7_COM_QI_ENTRY(IArchiveOpenSetSubArchiveName)
 #ifndef Z7_NO_CRYPTO
  Z7_COM_QI_ENTRY(ICryptoGetTextPassword)
 #endif
  // Z7_COM_QI_ENTRY(IProgress) // the code doesn't require it
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE
  
  Z7_IFACE_COM7_IMP(IArchiveOpenCallback)
  Z7_IFACE_COM7_IMP(IArchiveOpenVolumeCallback)
  Z7_IFACE_COM7_IMP(IProgress)
public:
  Z7_IFACE_COM7_IMP(IArchiveOpenSetSubArchiveName)
private:
 #ifndef Z7_NO_CRYPTO
  Z7_IFACE_COM7_IMP(ICryptoGetTextPassword)
 #endif

  bool _subArchiveMode;

public:
  bool PasswordWasAsked;
  UStringVector FileNames;
  CBoolVector FileNames_WasUsed;
  CRecordVector<UInt64> FileSizes;

  void AtCloseFile(unsigned fileIndex)
  {
    FileNames_WasUsed[fileIndex] = false;
    Volumes.CloseFile(fileIndex);
  }

  /* we have two ways to Callback from this object
      1) IArchiveOpenCallback * ReOpenCallback - for ReOpen function, when IOpenCallbackUI is not available
      2) IOpenCallbackUI *Callback - for usual callback
     we can't transfer IOpenCallbackUI pointer via internal interface,
     so we use ReOpenCallback to callback without IOpenCallbackUI.
  */

  /* we use Callback/ReOpenCallback only at Open stage.
     So the CMyComPtr reference counter is not required,
     and we don't want additional reference to unused object,
     if COpenCallbackImp is not closed
  */
  IArchiveOpenCallback *ReOpenCallback;
  // CMyComPtr<IArchiveOpenCallback> ReOpenCallback;
  IOpenCallbackUI *Callback;
  // CMyComPtr<IUnknown> Callback_Ref;

private:
  FString _folderPrefix;
  UString _subArchiveName;
  NWindows::NFile::NFind::CFileInfo _fileInfo;

public:
  CMultiStreams Volumes;
  
  // UInt64 TotalSize;

  COpenCallbackImp():
      _subArchiveMode(false),
      PasswordWasAsked(false),
      ReOpenCallback(NULL),
      Callback(NULL)  {}
  
  HRESULT Init2(const FString &folderPrefix, const FString &fileName);

  bool SetSecondFileInfo(CFSTR newName)
  {
    return _fileInfo.Find_FollowLink(newName) && !_fileInfo.IsDir();
  }
};

#endif
