// IFileExtractCallback.h

#ifndef ZIP7_INC_I_FILE_EXTRACT_CALLBACK_H
#define ZIP7_INC_I_FILE_EXTRACT_CALLBACK_H

#include "../../../Common/MyString.h"

#include "../../IDecl.h"

#include "LoadCodecs.h"
#include "OpenArchive.h"

Z7_PURE_INTERFACES_BEGIN

#define Z7_IFACE_CONSTR_FOLDERARC_SUB(i, base, n) \
  Z7_DECL_IFACE_7ZIP_SUB(i, base, 1, n) \
  { Z7_IFACE_COM7_PURE(i) };

#define Z7_IFACE_CONSTR_FOLDERARC(i, n) \
        Z7_IFACE_CONSTR_FOLDERARC_SUB(i, IUnknown, n)

namespace NOverwriteAnswer
{
  enum EEnum
  {
    kYes,
    kYesToAll,
    kNo,
    kNoToAll,
    kAutoRename,
    kCancel
  };
}


/* ---------- IFolderArchiveExtractCallback ----------
is implemented by
  Console/ExtractCallbackConsole.h  CExtractCallbackConsole
  FileManager/ExtractCallback.h     CExtractCallbackImp
  FAR/ExtractEngine.cpp             CExtractCallBackImp: (QueryInterface is not supported)

IID_IFolderArchiveExtractCallback is requested by:
  - Agent/ArchiveFolder.cpp
      CAgentFolder::CopyTo(..., IFolderOperationsExtractCallback *callback)
      is sent to IArchiveFolder::Extract()

  - FileManager/PanelCopy.cpp
      CPanel::CopyTo(), if (options->testMode)
      is sent to IArchiveFolder::Extract()

 IFolderArchiveExtractCallback is used by Common/ArchiveExtractCallback.cpp
*/

#define Z7_IFACEM_IFolderArchiveExtractCallback(x) \
  x(AskOverwrite( \
      const wchar_t *existName, const FILETIME *existTime, const UInt64 *existSize, \
      const wchar_t *newName, const FILETIME *newTime, const UInt64 *newSize, \
      Int32 *answer)) \
  x(PrepareOperation(const wchar_t *name, Int32 isFolder, Int32 askExtractMode, const UInt64 *position)) \
  x(MessageError(const wchar_t *message)) \
  x(SetOperationResult(Int32 opRes, Int32 encrypted)) \

Z7_IFACE_CONSTR_FOLDERARC_SUB(IFolderArchiveExtractCallback, IProgress, 0x07)

#define Z7_IFACEM_IFolderArchiveExtractCallback2(x) \
  x(ReportExtractResult(Int32 opRes, Int32 encrypted, const wchar_t *name)) \

Z7_IFACE_CONSTR_FOLDERARC(IFolderArchiveExtractCallback2, 0x08)

/* ---------- IExtractCallbackUI ----------
is implemented by
  Console/ExtractCallbackConsole.h  CExtractCallbackConsole
  FileManager/ExtractCallback.h     CExtractCallbackImp
*/

#ifdef Z7_NO_CRYPTO
  #define Z7_IFACEM_IExtractCallbackUI_Crypto(px)
#else
  #define Z7_IFACEM_IExtractCallbackUI_Crypto(px) \
  virtual HRESULT SetPassword(const UString &password) px
#endif

#define Z7_IFACEN_IExtractCallbackUI(px) \
  virtual HRESULT BeforeOpen(const wchar_t *name, bool testMode) px \
  virtual HRESULT OpenResult(const CCodecs *codecs, const CArchiveLink &arcLink, const wchar_t *name, HRESULT result) px \
  virtual HRESULT ThereAreNoFiles() px \
  virtual HRESULT ExtractResult(HRESULT result) px \
  Z7_IFACEM_IExtractCallbackUI_Crypto(px)

// IExtractCallbackUI - is non-COM interface
// IFolderArchiveExtractCallback - is COM interface
// Z7_IFACE_DECL_PURE_(IExtractCallbackUI, IFolderArchiveExtractCallback)
Z7_IFACE_DECL_PURE(IExtractCallbackUI)



#define Z7_IFACEM_IGetProp(x) \
  x(GetProp(PROPID propID, PROPVARIANT *value)) \

Z7_IFACE_CONSTR_FOLDERARC(IGetProp, 0x20)

#define Z7_IFACEM_IFolderExtractToStreamCallback(x) \
  x(UseExtractToStream(Int32 *res)) \
  x(GetStream7(const wchar_t *name, Int32 isDir, ISequentialOutStream **outStream, Int32 askExtractMode, IGetProp *getProp)) \
  x(PrepareOperation7(Int32 askExtractMode)) \
  x(SetOperationResult8(Int32 resultEOperationResult, Int32 encrypted, UInt64 size)) \

Z7_IFACE_CONSTR_FOLDERARC(IFolderExtractToStreamCallback, 0x31)

Z7_PURE_INTERFACES_END

#endif
