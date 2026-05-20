// Client7z.cpp

#include "StdAfx.h"

#include <stdio.h>

#include "../../../Common/MyWindows.h"
#include "../../../Common/MyInitGuid.h"

#include "../../../Common/Defs.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"

#include "../../../Windows/DLL.h"
#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileFind.h"
#include "../../../Windows/FileName.h"
#include "../../../Windows/NtCheck.h"
#include "../../../Windows/PropVariant.h"
#include "../../../Windows/PropVariantConv.h"

#include "../../Common/FileStreams.h"

#include "../../Archive/IArchive.h"

#if 0
// for password request functions:
#include "../../UI/Console/UserInputUtils.h"
#endif

#include "../../IPassword.h"
#include "../../../../C/7zVersion.h"

#ifdef _WIN32
extern
HINSTANCE g_hInstance;
HINSTANCE g_hInstance = NULL;
#endif

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

// You can find full list of all GUIDs supported by 7-Zip in Guid.txt file.
// 7z format GUID: {23170F69-40C1-278A-1000-000110070000}

#define DEFINE_GUID_ARC(name, id) Z7_DEFINE_GUID(name, \
  0x23170F69, 0x40C1, 0x278A, 0x10, 0x00, 0x00, 0x01, 0x10, id, 0x00, 0x00);

enum
{
  kId_Zip = 1,
  kId_BZip2 = 2,
  kId_7z = 7,
  kId_Xz = 0xC,
  kId_Tar = 0xEE,
  kId_GZip = 0xEF
};

// use another id, if you want to support other formats (zip, Xz, ...).
// DEFINE_GUID_ARC (CLSID_Format, kId_Zip)
// DEFINE_GUID_ARC (CLSID_Format, kId_BZip2)
// DEFINE_GUID_ARC (CLSID_Format, kId_Xz)
// DEFINE_GUID_ARC (CLSID_Format, kId_Tar)
// DEFINE_GUID_ARC (CLSID_Format, kId_GZip)
DEFINE_GUID_ARC (CLSID_Format, kId_7z)

using namespace NWindows;
using namespace NFile;
using namespace NDir;

#ifdef _WIN32
#define kDllName "7z.dll"
#else
#define kDllName "7z.so"
#endif

static const char * const kCopyrightString =
  "\n"
  "7-Zip"
  " (" kDllName " client)"
  " " MY_VERSION
  " : " MY_COPYRIGHT_DATE
  "\n";

static const char * const kHelpString =
"Usage: 7zcl.exe [a | l | x] archive.7z [fileName ...]\n"
"Examples:\n"
"  7zcl.exe a archive.7z f1.txt f2.txt  : compress two files to archive.7z\n"
"  7zcl.exe l archive.7z   : List contents of archive.7z\n"
"  7zcl.exe x archive.7z   : eXtract files from archive.7z\n";


static void Convert_UString_to_AString(const UString &s, AString &temp)
{
  int codePage = CP_OEMCP;
  /*
  int g_CodePage = -1;
  int codePage = g_CodePage;
  if (codePage == -1)
    codePage = CP_OEMCP;
  if (codePage == CP_UTF8)
    ConvertUnicodeToUTF8(s, temp);
  else
  */
    UnicodeStringToMultiByte2(temp, s, (UINT)codePage);
}

static FString CmdStringToFString(const char *s)
{
  return us2fs(GetUnicodeString(s));
}

static void Print(const char *s)
{
  fputs(s, stdout);
}

static void Print(const AString &s)
{
  Print(s.Ptr());
}

static void Print(const UString &s)
{
  AString as;
  Convert_UString_to_AString(s, as);
  Print(as);
}

static void Print(const wchar_t *s)
{
  Print(UString(s));
}

static void PrintNewLine()
{
  Print("\n");
}

static void PrintStringLn(const char *s)
{
  Print(s);
  PrintNewLine();
}

static void PrintError(const char *message)
{
  Print("Error: ");
  PrintNewLine();
  Print(message);
  PrintNewLine();
}

static void PrintError(const char *message, const FString &name)
{
  PrintError(message);
  Print(name);
}


static HRESULT IsArchiveItemProp(IInArchive *archive, UInt32 index, PROPID propID, bool &result)
{
  NCOM::CPropVariant prop;
  RINOK(archive->GetProperty(index, propID, &prop))
  if (prop.vt == VT_BOOL)
    result = VARIANT_BOOLToBool(prop.boolVal);
  else if (prop.vt == VT_EMPTY)
    result = false;
  else
    return E_FAIL;
  return S_OK;
}

static HRESULT IsArchiveItemFolder(IInArchive *archive, UInt32 index, bool &result)
{
  return IsArchiveItemProp(archive, index, kpidIsDir, result);
}


static const wchar_t * const kEmptyFileAlias = L"[Content]";


//////////////////////////////////////////////////////////////
// Archive Open callback class


class CArchiveOpenCallback Z7_final:
  public IArchiveOpenCallback,
  public ICryptoGetTextPassword,
  public CMyUnknownImp
{
  Z7_IFACES_IMP_UNK_2(IArchiveOpenCallback, ICryptoGetTextPassword)
public:

  bool PasswordIsDefined;
  UString Password;

  CArchiveOpenCallback() : PasswordIsDefined(false) {}
};

Z7_COM7F_IMF(CArchiveOpenCallback::SetTotal(const UInt64 * /* files */, const UInt64 * /* bytes */))
{
  return S_OK;
}

Z7_COM7F_IMF(CArchiveOpenCallback::SetCompleted(const UInt64 * /* files */, const UInt64 * /* bytes */))
{
  return S_OK;
}
  
Z7_COM7F_IMF(CArchiveOpenCallback::CryptoGetTextPassword(BSTR *password))
{
  if (!PasswordIsDefined)
  {
    // You can ask real password here from user
#if 0
    RINOK(GetPassword_HRESULT(&g_StdOut, Password))
    PasswordIsDefined = true;
#else
    PrintError("Password is not defined");
    return E_ABORT;
#endif
  }
  return StringToBstr(Password, password);
}



static const char * const kIncorrectCommand = "incorrect command";

//////////////////////////////////////////////////////////////
// Archive Extracting callback class

static const char * const kTestingString    =  "Testing     ";
static const char * const kExtractingString =  "Extracting  ";
static const char * const kSkippingString   =  "Skipping    ";
static const char * const kReadingString    =  "Reading     ";

static const char * const kUnsupportedMethod = "Unsupported Method";
static const char * const kCRCFailed = "CRC Failed";
static const char * const kDataError = "Data Error";
static const char * const kUnavailableData = "Unavailable data";
static const char * const kUnexpectedEnd = "Unexpected end of data";
static const char * const kDataAfterEnd = "There are some data after the end of the payload data";
static const char * const kIsNotArc = "Is not archive";
static const char * const kHeadersError = "Headers Error";


struct CArcTime
{
  FILETIME FT;
  UInt16 Prec;
  Byte Ns100;
  bool Def;

  CArcTime()
  {
    Clear();
  }

  void Clear()
  {
    FT.dwHighDateTime = FT.dwLowDateTime = 0;
    Prec = 0;
    Ns100 = 0;
    Def = false;
  }

  bool IsZero() const
  {
    return FT.dwLowDateTime == 0 && FT.dwHighDateTime == 0 && Ns100 == 0;
  }

  int GetNumDigits() const
  {
    if (Prec == k_PropVar_TimePrec_Unix ||
        Prec == k_PropVar_TimePrec_DOS)
      return 0;
    if (Prec == k_PropVar_TimePrec_HighPrec)
      return 9;
    if (Prec == k_PropVar_TimePrec_0)
      return 7;
    int digits = (int)Prec - (int)k_PropVar_TimePrec_Base;
    if (digits < 0)
      digits = 0;
    return digits;
  }

  void Write_To_FiTime(CFiTime &dest) const
  {
   #ifdef _WIN32
    dest = FT;
   #else
    if (FILETIME_To_timespec(FT, dest))
    if ((Prec == k_PropVar_TimePrec_Base + 8 ||
         Prec == k_PropVar_TimePrec_Base + 9)
        && Ns100 != 0)
    {
      dest.tv_nsec += Ns100;
    }
   #endif
  }

  void Set_From_Prop(const PROPVARIANT &prop)
  {
    FT = prop.filetime;
    unsigned prec = 0;
    unsigned ns100 = 0;
    const unsigned prec_Temp = prop.wReserved1;
    if (prec_Temp != 0
        && prec_Temp <= k_PropVar_TimePrec_1ns
        && prop.wReserved3 == 0)
    {
      const unsigned ns100_Temp = prop.wReserved2;
      if (ns100_Temp < 100)
      {
        ns100 = ns100_Temp;
        prec = prec_Temp;
      }
    }
    Prec = (UInt16)prec;
    Ns100 = (Byte)ns100;
    Def = true;
  }
};



class CArchiveExtractCallback Z7_final:
  public IArchiveExtractCallback,
  public ICryptoGetTextPassword,
  public CMyUnknownImp
{
  Z7_IFACES_IMP_UNK_2(IArchiveExtractCallback, ICryptoGetTextPassword)
  Z7_IFACE_COM7_IMP(IProgress)

  CMyComPtr<IInArchive> _archiveHandler;
  FString _directoryPath;  // Output directory
  UString _filePath;       // name inside arcvhive
  FString _diskFilePath;   // full path to file on disk
  bool _extractMode;
  struct CProcessedFileInfo
  {
    CArcTime MTime;
    UInt32 Attrib;
    bool isDir;
    bool Attrib_Defined;
  } _processedFileInfo;

  COutFileStream *_outFileStreamSpec;
  CMyComPtr<ISequentialOutStream> _outFileStream;

public:
  void Init(IInArchive *archiveHandler, const FString &directoryPath);

  UInt64 NumErrors;
  bool PasswordIsDefined;
  UString Password;

  CArchiveExtractCallback() : PasswordIsDefined(false) {}
};

void CArchiveExtractCallback::Init(IInArchive *archiveHandler, const FString &directoryPath)
{
  NumErrors = 0;
  _archiveHandler = archiveHandler;
  _directoryPath = directoryPath;
  NName::NormalizeDirPathPrefix(_directoryPath);
}

Z7_COM7F_IMF(CArchiveExtractCallback::SetTotal(UInt64 /* size */))
{
  return S_OK;
}

Z7_COM7F_IMF(CArchiveExtractCallback::SetCompleted(const UInt64 * /* completeValue */))
{
  return S_OK;
}

Z7_COM7F_IMF(CArchiveExtractCallback::GetStream(UInt32 index,
    ISequentialOutStream **outStream, Int32 askExtractMode))
{
  *outStream = NULL;
  _outFileStream.Release();

  {
    // Get Name
    NCOM::CPropVariant prop;
    RINOK(_archiveHandler->GetProperty(index, kpidPath, &prop))
    
    UString fullPath;
    if (prop.vt == VT_EMPTY)
      fullPath = kEmptyFileAlias;
    else
    {
      if (prop.vt != VT_BSTR)
        return E_FAIL;
      fullPath = prop.bstrVal;
    }
    _filePath = fullPath;
  }

  if (askExtractMode != NArchive::NExtract::NAskMode::kExtract)
    return S_OK;

  {
    // Get Attrib
    NCOM::CPropVariant prop;
    RINOK(_archiveHandler->GetProperty(index, kpidAttrib, &prop))
    if (prop.vt == VT_EMPTY)
    {
      _processedFileInfo.Attrib = 0;
      _processedFileInfo.Attrib_Defined = false;
    }
    else
    {
      if (prop.vt != VT_UI4)
        return E_FAIL;
      _processedFileInfo.Attrib = prop.ulVal;
      _processedFileInfo.Attrib_Defined = true;
    }
  }

  RINOK(IsArchiveItemFolder(_archiveHandler, index, _processedFileInfo.isDir))

  {
    _processedFileInfo.MTime.Clear();
    // Get Modified Time
    NCOM::CPropVariant prop;
    RINOK(_archiveHandler->GetProperty(index, kpidMTime, &prop))
    switch (prop.vt)
    {
      case VT_EMPTY:
        // _processedFileInfo.MTime = _utcMTimeDefault;
        break;
      case VT_FILETIME:
        _processedFileInfo.MTime.Set_From_Prop(prop);
        break;
      default:
        return E_FAIL;
    }

  }
  {
    // Get Size
    NCOM::CPropVariant prop;
    RINOK(_archiveHandler->GetProperty(index, kpidSize, &prop))
    UInt64 newFileSize;
    /* bool newFileSizeDefined = */ ConvertPropVariantToUInt64(prop, newFileSize);
  }

  
  {
    // Create folders for file
    int slashPos = _filePath.ReverseFind_PathSepar();
    if (slashPos >= 0)
      CreateComplexDir(_directoryPath + us2fs(_filePath.Left(slashPos)));
  }

  FString fullProcessedPath = _directoryPath + us2fs(_filePath);
  _diskFilePath = fullProcessedPath;

  if (_processedFileInfo.isDir)
  {
    CreateComplexDir(fullProcessedPath);
  }
  else
  {
    NFind::CFileInfo fi;
    if (fi.Find(fullProcessedPath))
    {
      if (!DeleteFileAlways(fullProcessedPath))
      {
        PrintError("Cannot delete output file", fullProcessedPath);
        return E_ABORT;
      }
    }
    
    _outFileStreamSpec = new COutFileStream;
    CMyComPtr<ISequentialOutStream> outStreamLoc(_outFileStreamSpec);
    if (!_outFileStreamSpec->Create_ALWAYS(fullProcessedPath))
    {
      PrintError("Cannot open output file", fullProcessedPath);
      return E_ABORT;
    }
    _outFileStream = outStreamLoc;
    *outStream = outStreamLoc.Detach();
  }
  return S_OK;
}

Z7_COM7F_IMF(CArchiveExtractCallback::PrepareOperation(Int32 askExtractMode))
{
  _extractMode = false;
  switch (askExtractMode)
  {
    case NArchive::NExtract::NAskMode::kExtract:  _extractMode = true; break;
  }
  switch (askExtractMode)
  {
    case NArchive::NExtract::NAskMode::kExtract:  Print(kExtractingString); break;
    case NArchive::NExtract::NAskMode::kTest:  Print(kTestingString); break;
    case NArchive::NExtract::NAskMode::kSkip:  Print(kSkippingString); break;
    case NArchive::NExtract::NAskMode::kReadExternal: Print(kReadingString); break;
    default:
      Print("??? "); break;
  }
  Print(_filePath);
  return S_OK;
}

Z7_COM7F_IMF(CArchiveExtractCallback::SetOperationResult(Int32 operationResult))
{
  switch (operationResult)
  {
    case NArchive::NExtract::NOperationResult::kOK:
      break;
    default:
    {
      NumErrors++;
      Print("  :  ");
      const char *s = NULL;
      switch (operationResult)
      {
        case NArchive::NExtract::NOperationResult::kUnsupportedMethod:
          s = kUnsupportedMethod;
          break;
        case NArchive::NExtract::NOperationResult::kCRCError:
          s = kCRCFailed;
          break;
        case NArchive::NExtract::NOperationResult::kDataError:
          s = kDataError;
          break;
        case NArchive::NExtract::NOperationResult::kUnavailable:
          s = kUnavailableData;
          break;
        case NArchive::NExtract::NOperationResult::kUnexpectedEnd:
          s = kUnexpectedEnd;
          break;
        case NArchive::NExtract::NOperationResult::kDataAfterEnd:
          s = kDataAfterEnd;
          break;
        case NArchive::NExtract::NOperationResult::kIsNotArc:
          s = kIsNotArc;
          break;
        case NArchive::NExtract::NOperationResult::kHeadersError:
          s = kHeadersError;
          break;
      }
      if (s)
      {
        Print("Error : ");
        Print(s);
      }
      else
      {
        char temp[16];
        ConvertUInt32ToString((UInt32)operationResult, temp);
        Print("Error #");
        Print(temp);
      }
    }
  }

  if (_outFileStream)
  {
    if (_processedFileInfo.MTime.Def)
    {
      CFiTime ft;
      _processedFileInfo.MTime.Write_To_FiTime(ft);
      _outFileStreamSpec->SetMTime(&ft);
    }
    RINOK(_outFileStreamSpec->Close())
  }
  _outFileStream.Release();
  if (_extractMode && _processedFileInfo.Attrib_Defined)
    SetFileAttrib_PosixHighDetect(_diskFilePath, _processedFileInfo.Attrib);
  PrintNewLine();
  return S_OK;
}


Z7_COM7F_IMF(CArchiveExtractCallback::CryptoGetTextPassword(BSTR *password))
{
  if (!PasswordIsDefined)
  {
#if 0
    // You can ask real password here from user
    RINOK(GetPassword_HRESULT(&g_StdOut, Password))
    PasswordIsDefined = true;
#else
    PrintError("Password is not defined");
    return E_ABORT;
#endif
  }
  return StringToBstr(Password, password);
}



//////////////////////////////////////////////////////////////
// Archive Creating callback class

struct CDirItem: public NWindows::NFile::NFind::CFileInfoBase
{
  UString Path_For_Handler;
  FString FullPath; // for filesystem

  CDirItem(const NWindows::NFile::NFind::CFileInfo &fi):
      CFileInfoBase(fi)
    {}
};

class CArchiveUpdateCallback Z7_final:
  public IArchiveUpdateCallback2,
  public ICryptoGetTextPassword2,
  public CMyUnknownImp
{
  Z7_IFACES_IMP_UNK_2(IArchiveUpdateCallback2, ICryptoGetTextPassword2)
  Z7_IFACE_COM7_IMP(IProgress)
  Z7_IFACE_COM7_IMP(IArchiveUpdateCallback)

public:
  CRecordVector<UInt64> VolumesSizes;
  UString VolName;
  UString VolExt;

  FString DirPrefix;
  const CObjectVector<CDirItem> *DirItems;

  bool PasswordIsDefined;
  UString Password;
  bool AskPassword;

  bool m_NeedBeClosed;

  FStringVector FailedFiles;
  CRecordVector<HRESULT> FailedCodes;

  CArchiveUpdateCallback():
      DirItems(NULL),
      PasswordIsDefined(false),
      AskPassword(false)
      {}

  ~CArchiveUpdateCallback() { Finilize(); }
  HRESULT Finilize();

  void Init(const CObjectVector<CDirItem> *dirItems)
  {
    DirItems = dirItems;
    m_NeedBeClosed = false;
    FailedFiles.Clear();
    FailedCodes.Clear();
  }
};

Z7_COM7F_IMF(CArchiveUpdateCallback::SetTotal(UInt64 /* size */))
{
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::SetCompleted(const UInt64 * /* completeValue */))
{
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetUpdateItemInfo(UInt32 /* index */,
      Int32 *newData, Int32 *newProperties, UInt32 *indexInArchive))
{
  if (newData)
    *newData = BoolToInt(true);
  if (newProperties)
    *newProperties = BoolToInt(true);
  if (indexInArchive)
    *indexInArchive = (UInt32)(Int32)-1;
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetProperty(UInt32 index, PROPID propID, PROPVARIANT *value))
{
  NCOM::CPropVariant prop;
  
  if (propID == kpidIsAnti)
  {
    prop = false;
    prop.Detach(value);
    return S_OK;
  }

  {
    const CDirItem &di = (*DirItems)[index];
    switch (propID)
    {
      case kpidPath:  prop = di.Path_For_Handler; break;
      case kpidIsDir:  prop = di.IsDir(); break;
      case kpidSize:  prop = di.Size; break;
      case kpidCTime:  PropVariant_SetFrom_FiTime(prop, di.CTime); break;
      case kpidATime:  PropVariant_SetFrom_FiTime(prop, di.ATime); break;
      case kpidMTime:  PropVariant_SetFrom_FiTime(prop, di.MTime); break;
      case kpidAttrib:  prop = (UInt32)di.GetWinAttrib(); break;
      case kpidPosixAttrib: prop = (UInt32)di.GetPosixAttrib(); break;
    }
  }
  prop.Detach(value);
  return S_OK;
}

HRESULT CArchiveUpdateCallback::Finilize()
{
  if (m_NeedBeClosed)
  {
    PrintNewLine();
    m_NeedBeClosed = false;
  }
  return S_OK;
}

static void GetStream2(const wchar_t *name)
{
  Print("Compressing  ");
  if (name[0] == 0)
    name = kEmptyFileAlias;
  Print(name);
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetStream(UInt32 index, ISequentialInStream **inStream))
{
  RINOK(Finilize())

  const CDirItem &dirItem = (*DirItems)[index];
  GetStream2(dirItem.Path_For_Handler);
 
  if (dirItem.IsDir())
    return S_OK;

  {
    CInFileStream *inStreamSpec = new CInFileStream;
    CMyComPtr<ISequentialInStream> inStreamLoc(inStreamSpec);
    FString path = DirPrefix + dirItem.FullPath;
    if (!inStreamSpec->Open(path))
    {
      const DWORD sysError = ::GetLastError();
      FailedCodes.Add(HRESULT_FROM_WIN32(sysError));
      FailedFiles.Add(path);
      // if (systemError == ERROR_SHARING_VIOLATION)
      {
        PrintNewLine();
        PrintError("WARNING: can't open file");
        // Print(NError::MyFormatMessageW(systemError));
        return S_FALSE;
      }
      // return sysError;
    }
    *inStream = inStreamLoc.Detach();
  }
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::SetOperationResult(Int32 /* operationResult */))
{
  m_NeedBeClosed = true;
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetVolumeSize(UInt32 index, UInt64 *size))
{
  if (VolumesSizes.Size() == 0)
    return S_FALSE;
  if (index >= (UInt32)VolumesSizes.Size())
    index = VolumesSizes.Size() - 1;
  *size = VolumesSizes[index];
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetVolumeStream(UInt32 index, ISequentialOutStream **volumeStream))
{
  wchar_t temp[16];
  ConvertUInt32ToString(index + 1, temp);
  UString res = temp;
  while (res.Len() < 2)
    res.InsertAtFront(L'0');
  UString fileName = VolName;
  fileName.Add_Dot();
  fileName += res;
  fileName += VolExt;
  COutFileStream *streamSpec = new COutFileStream;
  CMyComPtr<ISequentialOutStream> streamLoc(streamSpec);
  if (!streamSpec->Create_NEW(us2fs(fileName)))
    return GetLastError_noZero_HRESULT();
  *volumeStream = streamLoc.Detach();
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::CryptoGetTextPassword2(Int32 *passwordIsDefined, BSTR *password))
{
  if (!PasswordIsDefined)
  {
    if (AskPassword)
    {
#if 0
      RINOK(GetPassword_HRESULT(&g_StdOut, Password))
      PasswordIsDefined = true;
#else
      PrintError("Password is not defined");
      return E_ABORT;
#endif
    }
  }
  *passwordIsDefined = BoolToInt(PasswordIsDefined);
  return StringToBstr(Password, password);
}


// Main function

#if defined(_UNICODE) && !defined(_WIN64) && !defined(UNDER_CE)
#define NT_CHECK_FAIL_ACTION PrintError("Unsupported Windows version"); return 1;
#endif

int Z7_CDECL main(int numArgs, const char *args[])
{
  NT_CHECK

  #ifdef ENV_HAVE_LOCALE
  MY_SetLocale();
  #endif

  PrintStringLn(kCopyrightString);

  if (numArgs < 2)
  {
    PrintStringLn(kHelpString);
    return 0;
  }

  FString dllPrefix;

  #ifdef _WIN32
  dllPrefix = NDLL::GetModuleDirPrefix();
  #else
  {
    AString s (args[0]);
    int sep = s.ReverseFind_PathSepar();
    s.DeleteFrom(sep + 1);
    dllPrefix = s;
  }
  #endif

  NDLL::CLibrary lib;
  if (!lib.Load(dllPrefix + FTEXT(kDllName)))
  {
    PrintError("Cannot load 7-zip library");
    return 1;
  }

#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wc++98-compat-pedantic"
#endif

#ifdef _WIN32
Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION
#endif

  Func_CreateObject
     f_CreateObject = Z7_GET_PROC_ADDRESS(
  Func_CreateObject, lib.Get_HMODULE(),
      "CreateObject");
  if (!f_CreateObject)
  {
    PrintError("Cannot get CreateObject");
    return 1;
  }

  char c = 0;
  UString password;
  bool passwordIsDefined = false;
  CObjectVector<FString> params;

  for (int curCmd = 1; curCmd < numArgs; curCmd++)
  {
    AString a(args[curCmd]);

    if (!a.IsEmpty())
    {
      if (a[0] == '-')
      {
        if (!passwordIsDefined && a[1] == 'p')
        {
          password = GetUnicodeString(a.Ptr(2));
          passwordIsDefined = true;
          continue;
        }
      }
      else
      {
        if (c)
        {
          params.Add(CmdStringToFString(a));
          continue;
        }
        if (a.Len() == 1)
        {
          c = (char)MyCharLower_Ascii(a[0]);
          continue;
        }
      }
    }
    {
      PrintError(kIncorrectCommand);
      return 1;
    }
  }

  if (!c || params.Size() < 1)
  {
    PrintError(kIncorrectCommand);
    return 1;
  }

  const FString &archiveName = params[0];
  
  if (c == 'a')
  {
    // create archive command
    if (params.Size() < 2)
    {
      PrintError(kIncorrectCommand);
      return 1;
    }
    CObjectVector<CDirItem> dirItems;
    {
      unsigned i;
      for (i = 1; i < params.Size(); i++)
      {
        const FString &name = params[i];
        
        NFind::CFileInfo fi;
        if (!fi.Find(name))
        {
          PrintError("Can't find file", name);
          return 1;
        }

        CDirItem di(fi);
        
        di.Path_For_Handler = fs2us(name);
        di.FullPath = name;
        dirItems.Add(di);
      }
    }

    COutFileStream *outFileStreamSpec = new COutFileStream;
    CMyComPtr<IOutStream> outFileStream = outFileStreamSpec;
    if (!outFileStreamSpec->Create_NEW(archiveName))
    {
      PrintError("can't create archive file");
      return 1;
    }

    CMyComPtr<IOutArchive> outArchive;
    if (f_CreateObject(&CLSID_Format, &IID_IOutArchive, (void **)&outArchive) != S_OK)
    {
      PrintError("Cannot get class object");
      return 1;
    }

    CArchiveUpdateCallback *updateCallbackSpec = new CArchiveUpdateCallback;
    CMyComPtr<IArchiveUpdateCallback2> updateCallback(updateCallbackSpec);
    updateCallbackSpec->Init(&dirItems);
    updateCallbackSpec->PasswordIsDefined = passwordIsDefined;
    updateCallbackSpec->Password = password;

    /*
    {
      const wchar_t *names[] =
      {
        L"m",
        L"s",
        L"x"
      };
      const unsigned kNumProps = Z7_ARRAY_SIZE(names);
      NCOM::CPropVariant values[kNumProps] =
      {
        L"lzma",
        false,    // solid mode OFF
        (UInt32)9 // compression level = 9 - ultra
      };
      CMyComPtr<ISetProperties> setProperties;
      outArchive->QueryInterface(IID_ISetProperties, (void **)&setProperties);
      if (!setProperties)
      {
        PrintError("ISetProperties unsupported");
        return 1;
      }
      if (setProperties->SetProperties(names, values, kNumProps) != S_OK)
      {
        PrintError("SetProperties() error");
        return 1;
      }
    }
    */
    
    HRESULT result = outArchive->UpdateItems(outFileStream, dirItems.Size(), updateCallback);
    
    updateCallbackSpec->Finilize();
    
    if (result != S_OK)
    {
      PrintError("Update Error");
      return 1;
    }
    
    FOR_VECTOR (i, updateCallbackSpec->FailedFiles)
    {
      PrintNewLine();
      PrintError("Error for file", updateCallbackSpec->FailedFiles[i]);
    }
    
    if (updateCallbackSpec->FailedFiles.Size() != 0)
      return 1;
  }
  else
  {
    if (params.Size() != 1)
    {
      PrintError(kIncorrectCommand);
      return 1;
    }

    bool listCommand;
    
    if (c == 'l')
      listCommand = true;
    else if (c == 'x')
      listCommand = false;
    else
    {
      PrintError(kIncorrectCommand);
      return 1;
    }
  
    CMyComPtr<IInArchive> archive;
    if (f_CreateObject(&CLSID_Format, &IID_IInArchive, (void **)&archive) != S_OK)
    {
      PrintError("Cannot get class object");
      return 1;
    }
    
    CInFileStream *fileSpec = new CInFileStream;
    CMyComPtr<IInStream> file = fileSpec;
    
    if (!fileSpec->Open(archiveName))
    {
      PrintError("Cannot open archive file", archiveName);
      return 1;
    }

    {
      CArchiveOpenCallback *openCallbackSpec = new CArchiveOpenCallback;
      CMyComPtr<IArchiveOpenCallback> openCallback(openCallbackSpec);
      openCallbackSpec->PasswordIsDefined = passwordIsDefined;
      openCallbackSpec->Password = password;
      
      const UInt64 scanSize = 1 << 23;
      if (archive->Open(file, &scanSize, openCallback) != S_OK)
      {
        PrintError("Cannot open file as archive", archiveName);
        return 1;
      }
    }
    
    if (listCommand)
    {
      // List command
      UInt32 numItems = 0;
      archive->GetNumberOfItems(&numItems);
      for (UInt32 i = 0; i < numItems; i++)
      {
        {
          // Get uncompressed size of file
          NCOM::CPropVariant prop;
          archive->GetProperty(i, kpidSize, &prop);
          char s[64];
          ConvertPropVariantToShortString(prop, s);
          Print(s);
          Print("  ");
        }
        {
          // Get name of file
          NCOM::CPropVariant prop;
          archive->GetProperty(i, kpidPath, &prop);
          if (prop.vt == VT_BSTR)
            Print(prop.bstrVal);
          else if (prop.vt != VT_EMPTY)
            Print("ERROR!");
        }
        PrintNewLine();
      }
    }
    else
    {
      // Extract command
      CArchiveExtractCallback *extractCallbackSpec = new CArchiveExtractCallback;
      CMyComPtr<IArchiveExtractCallback> extractCallback(extractCallbackSpec);
      extractCallbackSpec->Init(archive, FString()); // second parameter is output folder path
      extractCallbackSpec->PasswordIsDefined = passwordIsDefined;
      extractCallbackSpec->Password = password;

      /*
      const wchar_t *names[] =
      {
        L"mt",
        L"mtf"
      };
      const unsigned kNumProps = sizeof(names) / sizeof(names[0]);
      NCOM::CPropVariant values[kNumProps] =
      {
        (UInt32)1,
        false
      };
      CMyComPtr<ISetProperties> setProperties;
      archive->QueryInterface(IID_ISetProperties, (void **)&setProperties);
      if (setProperties)
      {
        if (setProperties->SetProperties(names, values, kNumProps) != S_OK)
        {
          PrintError("SetProperties() error");
          return 1;
        }
      }
      */

      HRESULT result = archive->Extract(NULL, (UInt32)(Int32)(-1), false, extractCallback);
  
      if (result != S_OK)
      {
        PrintError("Extract Error");
        return 1;
      }
    }
  }

  return 0;
}
