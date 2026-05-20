// ExtractCallback.h

#ifndef ZIP7_INC_EXTRACT_CALLBACK_H
#define ZIP7_INC_EXTRACT_CALLBACK_H

#include "../../../../C/Alloc.h"

#include "../../../Common/MyCom.h"
#include "../../../Common/StringConvert.h"

#ifndef Z7_SFX
#include "../Agent/IFolderArchive.h"
#endif

#include "../Common/ArchiveExtractCallback.h"
#include "../Common/ArchiveOpenCallback.h"

#ifndef Z7_NO_CRYPTO
#include "../../IPassword.h"
#endif

#ifndef Z7_SFX
#include "IFolder.h"
#endif

#include "ProgressDialog2.h"

#ifndef Z7_SFX

class CGrowBuf
{
  Byte *_items;
  size_t _size;

  Z7_CLASS_NO_COPY(CGrowBuf)

public:
  void Free()
  {
    MyFree(_items);
    _items = NULL;
    _size = 0;
  }

  // newSize >= keepSize
  bool ReAlloc_KeepData(size_t newSize, size_t keepSize)
  {
    void *buf = NULL;
    if (newSize)
    {
      buf = MyAlloc(newSize);
      if (!buf)
        return false;
    }
    if (keepSize)
      memcpy(buf, _items, keepSize);
    MyFree(_items);
    _items = (Byte *)buf;
    _size = newSize;
    return true;
  }

  CGrowBuf(): _items(NULL), _size(0) {}
  ~CGrowBuf() { MyFree(_items); }

  operator Byte *() { return _items; }
  operator const Byte *() const { return _items; }
  size_t Size() const { return _size; }
};


struct CVirtFile
{
  CGrowBuf Data;
  
  UInt64 ExpectedSize; // size from props request. 0 if unknown
  size_t WrittenSize;  // size of written data in (Data) buffer
                       //   use (WrittenSize) only if (CVirtFileSystem::_newVirtFileStream_IsReadyToWrite == false)
  UString BaseName;    // original name of file inside archive,
                       // It's not path. So any path separators
                       // should be treated as part of name (or as incorrect chars)
  UString AltStreamName;

  bool CTime_Defined;
  bool ATime_Defined;
  bool MTime_Defined;
  bool Attrib_Defined;
  
  // bool IsDir;
  bool IsAltStream;
  bool ColonWasUsed;
  DWORD Attrib;

  FILETIME CTime;
  FILETIME ATime;
  FILETIME MTime;

  CVirtFile():
    CTime_Defined(false),
    ATime_Defined(false),
    MTime_Defined(false),
    Attrib_Defined(false),
    // IsDir(false),
    IsAltStream(false),
    ColonWasUsed(false)
    {}
};


/*
  We use CVirtFileSystem only for single file extraction:
  It supports the following cases and names:
     - "fileName" : single file
     - "fileName" item (main base file) and additional "fileName:altStream" items
     - "altStream" : single item without "fileName:" prefix.
  If file is flushed to disk, it uses Get_Correct_FsFile_Name(name).
*/
 
Z7_CLASS_IMP_NOQIB_1(
  CVirtFileSystem,
  ISequentialOutStream
)
  unsigned _numFlushed;
public:
  bool IsAltStreamFile; // in:
      // = true,  if extracting file is alt stream without "fileName:" prefix.
      // = false, if extracting file is normal file, but additional
      //          alt streams "fileName:altStream" items are possible.
private:
  bool _newVirtFileStream_IsReadyToWrite;    // it can non real file (if can't open alt stream)
  bool _needWriteToRealFile;  // we need real writing to open file.
  bool _wasSwitchedToFsMode;
  bool _altStream_NeedRestore_Attrib_bool;
  DWORD _altStream_NeedRestore_AttribVal;

  CMyComPtr2<ISequentialOutStream, COutFileStream> _outFileStream;
public:
  CObjectVector<CVirtFile> Files;
  size_t MaxTotalAllocSize; // remain size, including Files.Back()
  FString DirPrefix; // files will be flushed to this FS directory.
  UString FileName; // name of file that will be extracted.
                    // it can be name of alt stream without "fileName:" prefix, if (IsAltStreamFile == trye).
                    // we use that name to detect altStream part in "FileName:altStream".
  CByteBuffer ZoneBuf;
  int Index_of_MainExtractedFile_in_Files; // out: index in Files. == -1, if expected file was not extracted
  int Index_of_ZoneBuf_AltStream_in_Files; // out: index in Files. == -1, if no zonbuf alt stream
  

  CVirtFileSystem()
  {
    _numFlushed = 0;
    IsAltStreamFile = false;
    _newVirtFileStream_IsReadyToWrite = false;
    _needWriteToRealFile = false;
    _wasSwitchedToFsMode = false;
    _altStream_NeedRestore_Attrib_bool = false;
    MaxTotalAllocSize = (size_t)0 - 1;
    Index_of_MainExtractedFile_in_Files = -1;
    Index_of_ZoneBuf_AltStream_in_Files = -1;
  }

  bool WasStreamFlushedToFS() const { return _wasSwitchedToFsMode; }

  HRESULT CloseMemFile()
  {
    if (_wasSwitchedToFsMode)
      return FlushToDisk(true); // closeLast
    CVirtFile &file = Files.Back();
    if (file.Data.Size() != file.WrittenSize)
      file.Data.ReAlloc_KeepData(file.WrittenSize, file.WrittenSize);
    return S_OK;
  }

  HRESULT FlushToDisk(bool closeLast);
};

#endif
  


class CExtractCallbackImp Z7_final:
  public IFolderArchiveExtractCallback,
  /* IExtractCallbackUI:
       before v23.00 : it         included IFolderArchiveExtractCallback
       since  v23.00 : it doesn't include  IFolderArchiveExtractCallback
  */
  public IExtractCallbackUI, // NON-COM interface since 23.00
  public IOpenCallbackUI,    // NON-COM interface
  public IFolderArchiveExtractCallback2,
 #ifndef Z7_SFX
  public IFolderOperationsExtractCallback,
  public IFolderExtractToStreamCallback,
  public ICompressProgressInfo,
  public IArchiveRequestMemoryUseCallback,
 #endif
 #ifndef Z7_NO_CRYPTO
  public ICryptoGetTextPassword,
 #endif
  public CMyUnknownImp
{
  Z7_COM_QI_BEGIN2(IFolderArchiveExtractCallback)
  Z7_COM_QI_ENTRY(IFolderArchiveExtractCallback2)
 #ifndef Z7_SFX
  Z7_COM_QI_ENTRY(IFolderOperationsExtractCallback)
  Z7_COM_QI_ENTRY(IFolderExtractToStreamCallback)
  Z7_COM_QI_ENTRY(ICompressProgressInfo)
  Z7_COM_QI_ENTRY(IArchiveRequestMemoryUseCallback)
 #endif
 #ifndef Z7_NO_CRYPTO
  Z7_COM_QI_ENTRY(ICryptoGetTextPassword)
 #endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_IMP(IExtractCallbackUI)
  Z7_IFACE_IMP(IOpenCallbackUI)
  Z7_IFACE_COM7_IMP(IProgress)
  Z7_IFACE_COM7_IMP(IFolderArchiveExtractCallback)
  Z7_IFACE_COM7_IMP(IFolderArchiveExtractCallback2)
 #ifndef Z7_SFX
  Z7_IFACE_COM7_IMP(IFolderOperationsExtractCallback)
  Z7_IFACE_COM7_IMP(IFolderExtractToStreamCallback)
  Z7_IFACE_COM7_IMP(ICompressProgressInfo)
  Z7_IFACE_COM7_IMP(IArchiveRequestMemoryUseCallback)
 #endif
 #ifndef Z7_NO_CRYPTO
  Z7_IFACE_COM7_IMP(ICryptoGetTextPassword)
 #endif

  bool _needWriteArchivePath;
  bool _isFolder;
  bool _totalFiles_Defined;
  bool _totalBytes_Defined;
public:
  bool MultiArcMode;
  bool ProcessAltStreams;
  bool StreamMode; // set to true, if you want the callee to call GetStream7()
  bool ThereAreMessageErrors;
  bool Src_Is_IO_FS_Folder;

#ifndef Z7_NO_CRYPTO
  bool PasswordIsDefined;
  bool PasswordWasAsked;
#endif

private:
#ifndef Z7_SFX
  bool _needUpdateStat;
  bool _newVirtFileWasAdded;
  bool _isAltStream;
  // bool _extractMode;
  // bool _testMode;
  bool _hashStream_WasUsed;
  bool _curSize_Defined;
  bool NeedAddFile;

  bool _remember;
  bool _skipArc;
#endif

public:
  bool YesToAll;
  bool TestMode;

  UInt32 NumArchiveErrors;
  NExtract::NOverwriteMode::EEnum OverwriteMode;

private:
  UString _currentArchivePath;
  UString _currentFilePath;
  UString _filePath;  // virtual path than will be sent via IFolderExtractToStreamCallback

#ifndef Z7_SFX
  UInt64 _curSize;
  CMyComPtr2<ISequentialOutStream, COutStreamWithHash> _hashStream;
  IHashCalc *_hashCalc; // it's for stat in Test operation
#endif

public:
  CProgressDialog *ProgressDialog;

#ifndef Z7_SFX
  CVirtFileSystem *VirtFileSystemSpec;
  CMyComPtr<ISequentialOutStream> VirtFileSystem;
  UInt64 NumFolders;
  UInt64 NumFiles;
#endif

#ifndef Z7_NO_CRYPTO
  UString Password;
#endif

  UString _lang_Extracting;
  UString _lang_Testing;
  UString _lang_Skipping;
  UString _lang_Reading;
  UString _lang_Empty;

  CExtractCallbackImp():
      _totalFiles_Defined(false)
    , _totalBytes_Defined(false)
    , MultiArcMode(false)
    , ProcessAltStreams(true)
    , StreamMode(false)
    , ThereAreMessageErrors(false)
    , Src_Is_IO_FS_Folder(false)
#ifndef Z7_NO_CRYPTO
    , PasswordIsDefined(false)
    , PasswordWasAsked(false)
#endif
#ifndef Z7_SFX
    , _remember(false)
    , _skipArc(false)
#endif
    , YesToAll(false)
    , TestMode(false)
    , OverwriteMode(NExtract::NOverwriteMode::kAsk)
#ifndef Z7_SFX
    , _hashCalc(NULL)
#endif
    {}
   
  ~CExtractCallbackImp();
  void Init();

  HRESULT SetCurrentFilePath2(const wchar_t *filePath);
  void AddError_Message(LPCWSTR message);
  void AddError_Message_ShowArcPath(LPCWSTR message);
  HRESULT MessageError(const char *message, const FString &path);
  void Add_ArchiveName_Error();

  #ifndef Z7_SFX
  void SetHashCalc(IHashCalc *hashCalc) { _hashCalc = hashCalc; }

  void SetHashMethods(IHashCalc *hash)
  {
    if (!hash)
      return;
    _hashStream.Create_if_Empty();
    _hashStream->_hash = hash;
  }
  #endif

  bool IsOK() const { return NumArchiveErrors == 0 && !ThereAreMessageErrors; }
};

#endif
