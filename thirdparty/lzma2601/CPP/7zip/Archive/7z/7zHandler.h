// 7z/Handler.h

#ifndef ZIP7_7Z_HANDLER_H
#define ZIP7_7Z_HANDLER_H

#include "../../ICoder.h"
#include "../IArchive.h"

#include "../../Common/CreateCoder.h"

#ifndef Z7_7Z_SET_PROPERTIES

#ifdef Z7_EXTRACT_ONLY
  #if !defined(Z7_ST) && !defined(Z7_SFX)
    #define Z7_7Z_SET_PROPERTIES
  #endif
#else
  #define Z7_7Z_SET_PROPERTIES
#endif

#endif

// #ifdef Z7_7Z_SET_PROPERTIES
#include "../Common/HandlerOut.h"
// #endif

#include "7zCompressionMode.h"
#include "7zIn.h"

namespace NArchive {
namespace N7z {


#ifndef Z7_EXTRACT_ONLY

class COutHandler: public CMultiMethodProps
{
  HRESULT SetSolidFromString(const UString &s);
  HRESULT SetSolidFromPROPVARIANT(const PROPVARIANT &value);
public:
  UInt64 _numSolidFiles;
  UInt64 _numSolidBytes;
  bool _numSolidBytesDefined;
  bool _solidExtension;
  bool _useTypeSorting;

  bool _compressHeaders;
  bool _encryptHeadersSpecified;
  bool _encryptHeaders;
  // bool _useParents; 9.26

  CHandlerTimeOptions TimeOptions;

  CBoolPair Write_Attrib;

  bool _useMultiThreadMixer;
  bool _removeSfxBlock;
  // bool _volumeMode;

  UInt32 _decoderCompatibilityVersion;
  CUIntVector _enabledFilters;
  CUIntVector _disabledFilters;
  
  void InitSolidFiles() { _numSolidFiles = (UInt64)(Int64)(-1); }
  void InitSolidSize()  { _numSolidBytes = (UInt64)(Int64)(-1); }
  void InitSolid()
  {
    InitSolidFiles();
    InitSolidSize();
    _solidExtension = false;
    _numSolidBytesDefined = false;
  }

  void InitProps7z();
  void InitProps();

  COutHandler() { InitProps7z(); }

  HRESULT SetProperty(const wchar_t *name, const PROPVARIANT &value);
};

#endif

class CHandler Z7_final:
  public IInArchive,
  public IArchiveGetRawProps,
  
  #ifdef Z7_7Z_SET_PROPERTIES
  public ISetProperties,
  #endif
  
  #ifndef Z7_EXTRACT_ONLY
  public IOutArchive,
  #endif
  
  Z7_PUBLIC_ISetCompressCodecsInfo_IFEC
  
  public CMyUnknownImp,

  #ifndef Z7_EXTRACT_ONLY
    public COutHandler
  #else
    public CCommonMethodProps
  #endif
{
  Z7_COM_QI_BEGIN2(IInArchive)
  Z7_COM_QI_ENTRY(IArchiveGetRawProps)
 #ifdef Z7_7Z_SET_PROPERTIES
  Z7_COM_QI_ENTRY(ISetProperties)
 #endif
 #ifndef Z7_EXTRACT_ONLY
  Z7_COM_QI_ENTRY(IOutArchive)
 #endif
  Z7_COM_QI_ENTRY_ISetCompressCodecsInfo_IFEC
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_COM7_IMP(IInArchive)
  Z7_IFACE_COM7_IMP(IArchiveGetRawProps)
 #ifdef Z7_7Z_SET_PROPERTIES
  Z7_IFACE_COM7_IMP(ISetProperties)
 #endif
 #ifndef Z7_EXTRACT_ONLY
  Z7_IFACE_COM7_IMP(IOutArchive)
 #endif
  DECL_ISetCompressCodecsInfo

private:
  CMyComPtr<IInStream> _inStream;
  NArchive::N7z::CDbEx _db;
  
 #ifndef Z7_NO_CRYPTO
  bool _isEncrypted;
  bool _passwordIsDefined;
  UString _password; // _Wipe
 #endif

  #ifdef Z7_EXTRACT_ONLY
  
  #ifdef Z7_7Z_SET_PROPERTIES
  bool _useMultiThreadMixer;
  #endif

  UInt32 _crcSize;

  #else
  
  CRecordVector<CBond2> _bonds;

  HRESULT PropsMethod_To_FullMethod(CMethodFull &dest, const COneMethodInfo &m);
  HRESULT SetHeaderMethod(CCompressionMethodMode &headerMethod);
  HRESULT SetMainMethod(CCompressionMethodMode &method);

  #endif

  bool IsFolderEncrypted(CNum folderIndex) const;
  #ifndef Z7_SFX

  CRecordVector<UInt64> _fileInfoPopIDs;
  void FillPopIDs();
  void AddMethodName(AString &s, UInt64 id);
  HRESULT SetMethodToProp(CNum folderIndex, PROPVARIANT *prop) const;

  #endif

  DECL_EXTERNAL_CODECS_VARS

public:
  CHandler();
  ~CHandler()
  {
    Close();
  }
};

}}

#endif
