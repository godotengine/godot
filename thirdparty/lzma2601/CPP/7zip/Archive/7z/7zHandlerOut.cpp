// 7zHandlerOut.cpp

#include "StdAfx.h"

#include "../../../Common/ComTry.h"
#include "../../../Common/StringToInt.h"
#include "../../../Common/Wildcard.h"

#include "../Common/ItemNameUtils.h"
#include "../Common/ParseProperties.h"

#include "7zHandler.h"
#include "7zOut.h"
#include "7zUpdate.h"

#ifndef Z7_EXTRACT_ONLY

using namespace NWindows;

namespace NArchive {
namespace N7z {

static const UInt32 k_decoderCompatibilityVersion = 2301;
// 7-Zip version 2301 supports ARM64 filter

#define k_LZMA_Name "LZMA"
#define kDefaultMethodName "LZMA2"
#define k_Copy_Name "Copy"

#define k_MatchFinder_ForHeaders "BT2"

static const UInt32 k_NumFastBytes_ForHeaders = 273;
static const UInt32 k_Level_ForHeaders = 5;
static const UInt32 k_Dictionary_ForHeaders =
  #ifdef UNDER_CE
  1 << 18;
  #else
  1 << 20;
  #endif

Z7_COM7F_IMF(CHandler::GetFileTimeType(UInt32 *type))
{
  *type = NFileTimeType::kWindows;
  return S_OK;
}

HRESULT CHandler::PropsMethod_To_FullMethod(CMethodFull &dest, const COneMethodInfo &m)
{
  bool isFilter;
  dest.CodecIndex = FindMethod_Index(
      EXTERNAL_CODECS_VARS
      m.MethodName, true,
      dest.Id, dest.NumStreams, isFilter);
  if (dest.CodecIndex < 0)
    return E_INVALIDARG;
  (CProps &)dest = (CProps &)m;
  return S_OK;
}

HRESULT CHandler::SetHeaderMethod(CCompressionMethodMode &headerMethod)
{
  if (!_compressHeaders)
    return S_OK;
  COneMethodInfo m;
  m.MethodName = k_LZMA_Name;
  m.AddProp_Ascii(NCoderPropID::kMatchFinder, k_MatchFinder_ForHeaders);
  m.AddProp_Level(k_Level_ForHeaders);
  m.AddProp32(NCoderPropID::kNumFastBytes, k_NumFastBytes_ForHeaders);
  m.AddProp32(NCoderPropID::kDictionarySize, k_Dictionary_ForHeaders);
  m.AddProp_NumThreads(1);

  CMethodFull &methodFull = headerMethod.Methods.AddNew();
  return PropsMethod_To_FullMethod(methodFull, m);
}


HRESULT CHandler::SetMainMethod(CCompressionMethodMode &methodMode)
{
  methodMode.Bonds = _bonds;

  // we create local copy of _methods. So we can modify it.
  CObjectVector<COneMethodInfo> methods = _methods;

  {
    FOR_VECTOR (i, methods)
    {
      AString &methodName = methods[i].MethodName;
      if (methodName.IsEmpty())
        methodName = kDefaultMethodName;
    }
    if (methods.IsEmpty())
    {
      COneMethodInfo &m = methods.AddNew();
      m.MethodName = (GetLevel() == 0 ? k_Copy_Name : kDefaultMethodName);
      methodMode.DefaultMethod_was_Inserted = true;
    }
  }

  if (!_filterMethod.MethodName.IsEmpty())
  {
    // if (methodMode.Bonds.IsEmpty())
    {
      FOR_VECTOR (k, methodMode.Bonds)
      {
        CBond2 &bond = methodMode.Bonds[k];
        bond.InCoder++;
        bond.OutCoder++;
      }
      methods.Insert(0, _filterMethod);
      methodMode.Filter_was_Inserted = true;
    }
  }

  const UInt64 kSolidBytes_Min = 1 << 24;
  const UInt64 kSolidBytes_Max = (UInt64)1 << 32;  // for non-LZMA2 methods

  bool needSolid = false;
  
  FOR_VECTOR (i, methods)
  {
    COneMethodInfo &oneMethodInfo = methods[i];

    SetGlobalLevelTo(oneMethodInfo);

#ifndef Z7_ST
    const bool numThreads_WasSpecifiedInMethod = (oneMethodInfo.Get_NumThreads() >= 0);
    if (!numThreads_WasSpecifiedInMethod)
    {
      // here we set the (NCoderPropID::kNumThreads) property in each method, only if there is no such property already
      CMultiMethodProps::SetMethodThreadsTo_IfNotFinded(oneMethodInfo, methodMode.NumThreads);
    }
    if (methodMode.NumThreadGroups > 1)
      CMultiMethodProps::Set_Method_NumThreadGroups_IfNotFinded(oneMethodInfo, methodMode.NumThreadGroups);
#endif

    CMethodFull &methodFull = methodMode.Methods.AddNew();
    RINOK(PropsMethod_To_FullMethod(methodFull, oneMethodInfo))

#ifndef Z7_ST
    methodFull.Set_NumThreads = true;
    methodFull.NumThreads = methodMode.NumThreads;
#endif

    if (methodFull.Id != k_Copy)
      needSolid = true;

    UInt64 dicSize;
    switch (methodFull.Id)
    {
      case k_LZMA:
      case k_LZMA2: dicSize = oneMethodInfo.Get_Lzma_DicSize(); break;
      case k_PPMD: dicSize = oneMethodInfo.Get_Ppmd_MemSize(); break;
      case k_Deflate: dicSize = (UInt32)1 << 15; break;
      case k_Deflate64: dicSize = (UInt32)1 << 16; break;
      case k_BZip2: dicSize = oneMethodInfo.Get_BZip2_BlockSize(); break;
      // case k_ZSTD: dicSize = 1 << 23; break;
      default: continue;
    }

    UInt64 numSolidBytes;

    /*
    if (methodFull.Id == k_ZSTD)
    {
      // continue;
      NCompress::NZstd::CEncoderProps encoderProps;
      RINOK(oneMethodInfo.Set_PropsTo_zstd(encoderProps));
      CZstdEncProps &zstdProps = encoderProps.EncProps;
      ZstdEncProps_NormalizeFull(&zstdProps);
      UInt64 cs = (UInt64)(zstdProps.jobSize);
      UInt32 winSize = (UInt32)(1 << zstdProps.windowLog);
      if (cs < winSize)
        cs = winSize;
      numSolidBytes = cs << 6;
      const UInt64 kSolidBytes_Zstd_Max = ((UInt64)1 << 34);
      if (numSolidBytes > kSolidBytes_Zstd_Max)
        numSolidBytes = kSolidBytes_Zstd_Max;

      methodFull.Set_NumThreads = false; // we don't use ICompressSetCoderMt::SetNumberOfThreads() for LZMA2 encoder

      #ifndef Z7_ST
      if (!numThreads_WasSpecifiedInMethod
          && !methodMode.NumThreads_WasForced
          && methodMode.MemoryUsageLimit_WasSet
          )
      {
        const UInt32 numThreads_Original = methodMode.NumThreads;
        const UInt32 numThreads_New = ZstdEncProps_GetNumThreads_for_MemUsageLimit(
            &zstdProps,
            methodMode.MemoryUsageLimit,
            numThreads_Original);
        if (numThreads_Original != numThreads_New)
        {
          CMultiMethodProps::SetMethodThreadsTo_Replace(methodFull, numThreads_New);
        }
      }
      #endif
    }
    else
    */
    if (methodFull.Id == k_LZMA2)
    {
      // he we calculate default chunk Size for LZMA2 as defined in LZMA2 encoder code
      /* lzma2 code use dictionary up to fake 4 GiB to calculate ChunkSize.
         So we do same */
      UInt64 cs = (UInt64)dicSize << 2;
      const UInt32 kMinSize = (UInt32)1 << 20;
      const UInt32 kMaxSize = (UInt32)1 << 28;
      if (cs < kMinSize) cs = kMinSize;
      if (cs > kMaxSize) cs = kMaxSize;
      if (cs < dicSize) cs = dicSize;
      cs += (kMinSize - 1);
      cs &= ~(UInt64)(kMinSize - 1);
      // we want to use at least 64 chunks (threads) per one solid block.

      // here we don't use chunkSize property
      numSolidBytes = cs << 6;

      // here we get real chunkSize
      cs = oneMethodInfo.Get_Xz_BlockSize();
      if (dicSize > cs)
          dicSize = cs;

      const UInt64 kSolidBytes_Lzma2_Max = (UInt64)1 << 34;
      if (numSolidBytes > kSolidBytes_Lzma2_Max)
          numSolidBytes = kSolidBytes_Lzma2_Max;

      methodFull.Set_NumThreads = false; // we don't use ICompressSetCoderMt::SetNumberOfThreads() for LZMA2 encoder

      #ifndef Z7_ST
      if (!numThreads_WasSpecifiedInMethod
          && !methodMode.NumThreads_WasForced
          && methodMode.MemoryUsageLimit_WasSet)
      {
        const UInt32 lzmaThreads = oneMethodInfo.Get_Lzma_NumThreads();
        const UInt32 numBlockThreads_Original = methodMode.NumThreads / lzmaThreads;

        if (numBlockThreads_Original > 1)
        {
          /*
            const UInt32 kNumThreads_Max = 1024;
            if (numBlockThreads > kNumMaxThreads)
            numBlockThreads = kNumMaxThreads;
          */

          UInt32 numBlockThreads = numBlockThreads_Original;
          const UInt64 lzmaMemUsage = oneMethodInfo.Get_Lzma_MemUsage(false); // solid
          
          for (; numBlockThreads > 1; numBlockThreads--)
          {
            UInt64 size = numBlockThreads * (lzmaMemUsage + cs);
            UInt32 numPackChunks = numBlockThreads + (numBlockThreads / 8) + 1;
            if (cs < ((UInt32)1 << 26)) numPackChunks++;
            if (cs < ((UInt32)1 << 24)) numPackChunks++;
            if (cs < ((UInt32)1 << 22)) numPackChunks++;
            size += numPackChunks * cs;
            // printf("\nnumBlockThreads = %d, size = %d\n", (unsigned)(numBlockThreads), (unsigned)(size >> 20));
            if (size <= methodMode.MemoryUsageLimit)
              break;
          }

          if (numBlockThreads == 0)
            numBlockThreads = 1;
          if (numBlockThreads != numBlockThreads_Original)
          {
            const UInt32 numThreads_New = numBlockThreads * lzmaThreads;
            CMultiMethodProps::SetMethodThreadsTo_Replace(methodFull, numThreads_New);
          }
        }
      }
      #endif
    }
    else
    {
      numSolidBytes = (UInt64)dicSize << 7;
      if (numSolidBytes > kSolidBytes_Max)
          numSolidBytes = kSolidBytes_Max;
    }

    if (_numSolidBytesDefined)
      continue;

    if (numSolidBytes < kSolidBytes_Min)
        numSolidBytes = kSolidBytes_Min;
    _numSolidBytes = numSolidBytes;
    _numSolidBytesDefined = true;
  }

  if (!_numSolidBytesDefined)
  {
    if (needSolid)
      _numSolidBytes = kSolidBytes_Max;
    else
      _numSolidBytes = 0;
  }
  _numSolidBytesDefined = true;


  return S_OK;
}



static HRESULT GetTime(IArchiveUpdateCallback *updateCallback, unsigned index, PROPID propID, UInt64 &ft, bool &ftDefined)
{
  // ft = 0;
  // ftDefined = false;
  NCOM::CPropVariant prop;
  RINOK(updateCallback->GetProperty(index, propID, &prop))
  if (prop.vt == VT_FILETIME)
  {
    ft = prop.filetime.dwLowDateTime | ((UInt64)prop.filetime.dwHighDateTime << 32);
    ftDefined = true;
  }
  else if (prop.vt != VT_EMPTY)
    return E_INVALIDARG;
  else
  {
    ft = 0;
    ftDefined = false;
  }
  return S_OK;
}

/*

#ifdef _WIN32
static const wchar_t kDirDelimiter1 = L'\\';
#endif
static const wchar_t kDirDelimiter2 = L'/';

static inline bool IsCharDirLimiter(wchar_t c)
{
  return (
    #ifdef _WIN32
    c == kDirDelimiter1 ||
    #endif
    c == kDirDelimiter2);
}

static int FillSortIndex(CObjectVector<CTreeFolder> &treeFolders, int cur, int curSortIndex)
{
  CTreeFolder &tf = treeFolders[cur];
  tf.SortIndex = curSortIndex++;
  for (int i = 0; i < tf.SubFolders.Size(); i++)
    curSortIndex = FillSortIndex(treeFolders, tf.SubFolders[i], curSortIndex);
  tf.SortIndexEnd = curSortIndex;
  return curSortIndex;
}

static int FindSubFolder(const CObjectVector<CTreeFolder> &treeFolders, int cur, const UString &name, int &insertPos)
{
  const CIntVector &subFolders = treeFolders[cur].SubFolders;
  int left = 0, right = subFolders.Size();
  insertPos = -1;
  for (;;)
  {
    if (left == right)
    {
      insertPos = left;
      return -1;
    }
    int mid = (left + right) / 2;
    int midFolder = subFolders[mid];
    int compare = CompareFileNames(name, treeFolders[midFolder].Name);
    if (compare == 0)
      return midFolder;
    if (compare < 0)
      right = mid;
    else
      left = mid + 1;
  }
}

static int AddFolder(CObjectVector<CTreeFolder> &treeFolders, int cur, const UString &name)
{
  int insertPos;
  int folderIndex = FindSubFolder(treeFolders, cur, name, insertPos);
  if (folderIndex < 0)
  {
    folderIndex = treeFolders.Size();
    CTreeFolder &newFolder = treeFolders.AddNew();
    newFolder.Parent = cur;
    newFolder.Name = name;
    treeFolders[cur].SubFolders.Insert(insertPos, folderIndex);
  }
  // else if (treeFolders[folderIndex].IsAltStreamFolder != isAltStreamFolder) throw 1123234234;
  return folderIndex;
}
*/

Z7_COM7F_IMF(CHandler::UpdateItems(ISequentialOutStream *outStream, UInt32 numItems,
    IArchiveUpdateCallback *updateCallback))
{
  COM_TRY_BEGIN

  const CDbEx *db = NULL;
  #ifdef Z7_7Z_VOL
  if (_volumes.Size() > 1)
    return E_FAIL;
  const CVolume *volume = 0;
  if (_volumes.Size() == 1)
  {
    volume = &_volumes.Front();
    db = &volume->Database;
  }
  #else
  if (_inStream)
    db = &_db;
  #endif

  if (db && !db->CanUpdate())
    return E_NOTIMPL;

  /*
  Z7_DECL_CMyComPtr_QI_FROM(
      IArchiveGetRawProps,
      getRawProps, updateCallback)

  CUniqBlocks secureBlocks;
  secureBlocks.AddUniq(NULL, 0);

  CObjectVector<CTreeFolder> treeFolders;
  {
    CTreeFolder folder;
    folder.Parent = -1;
    treeFolders.Add(folder);
  }
  */

  CObjectVector<CUpdateItem> updateItems;

  bool need_CTime = (TimeOptions.Write_CTime.Def && TimeOptions.Write_CTime.Val);
  bool need_ATime = (TimeOptions.Write_ATime.Def && TimeOptions.Write_ATime.Val);
  bool need_MTime = (TimeOptions.Write_MTime.Def ? TimeOptions.Write_MTime.Val : true);
  bool need_Attrib = (Write_Attrib.Def ? Write_Attrib.Val : true);
  
  if (db && !db->Files.IsEmpty())
  {
    if (!TimeOptions.Write_CTime.Def) need_CTime = !db->CTime.Defs.IsEmpty();
    if (!TimeOptions.Write_ATime.Def) need_ATime = !db->ATime.Defs.IsEmpty();
    if (!TimeOptions.Write_MTime.Def) need_MTime = !db->MTime.Defs.IsEmpty();
    if (!Write_Attrib.Def) need_Attrib = !db->Attrib.Defs.IsEmpty();
  }

  // UString s;
  UString name;

  for (UInt32 i = 0; i < numItems; i++)
  {
    Int32 newData, newProps;
    UInt32 indexInArchive;
    if (!updateCallback)
      return E_FAIL;
    RINOK(updateCallback->GetUpdateItemInfo(i, &newData, &newProps, &indexInArchive))
    CUpdateItem ui;
    ui.NewProps = IntToBool(newProps);
    ui.NewData = IntToBool(newData);
    ui.IndexInArchive = (int)indexInArchive;
    ui.IndexInClient = i;
    ui.IsAnti = false;
    ui.Size = 0;

    name.Empty();
    // bool isAltStream = false;
    if (ui.IndexInArchive != -1)
    {
      if (!db || (unsigned)ui.IndexInArchive >= db->Files.Size())
        return E_INVALIDARG;
      const CFileItem &fi = db->Files[(unsigned)ui.IndexInArchive];
      if (!ui.NewProps)
      {
        _db.GetPath((unsigned)ui.IndexInArchive, name);
      }
      ui.IsDir = fi.IsDir;
      ui.Size = fi.Size;
      // isAltStream = fi.IsAltStream;
      ui.IsAnti = db->IsItemAnti((unsigned)ui.IndexInArchive);
      
      if (!ui.NewProps)
      {
        ui.CTimeDefined = db->CTime.GetItem((unsigned)ui.IndexInArchive, ui.CTime);
        ui.ATimeDefined = db->ATime.GetItem((unsigned)ui.IndexInArchive, ui.ATime);
        ui.MTimeDefined = db->MTime.GetItem((unsigned)ui.IndexInArchive, ui.MTime);
      }
    }

    if (ui.NewProps)
    {
      bool folderStatusIsDefined;
      if (need_Attrib)
      {
        NCOM::CPropVariant prop;
        RINOK(updateCallback->GetProperty(i, kpidAttrib, &prop))
        if (prop.vt == VT_EMPTY)
          ui.AttribDefined = false;
        else if (prop.vt != VT_UI4)
          return E_INVALIDARG;
        else
        {
          ui.Attrib = prop.ulVal;
          ui.AttribDefined = true;
        }
      }
      
      // we need MTime to sort files.
      if (need_CTime) RINOK(GetTime(updateCallback, i, kpidCTime, ui.CTime, ui.CTimeDefined))
      if (need_ATime) RINOK(GetTime(updateCallback, i, kpidATime, ui.ATime, ui.ATimeDefined))
      if (need_MTime) RINOK(GetTime(updateCallback, i, kpidMTime, ui.MTime, ui.MTimeDefined))

      /*
      if (getRawProps)
      {
        const void *data;
        UInt32 dataSize;
        UInt32 propType;

        getRawProps->GetRawProp(i, kpidNtSecure, &data, &dataSize, &propType);
        if (dataSize != 0 && propType != NPropDataType::kRaw)
          return E_FAIL;
        ui.SecureIndex = secureBlocks.AddUniq((const Byte *)data, dataSize);
      }
      */

      {
        NCOM::CPropVariant prop;
        RINOK(updateCallback->GetProperty(i, kpidPath, &prop))
        if (prop.vt == VT_EMPTY)
        {
        }
        else if (prop.vt != VT_BSTR)
          return E_INVALIDARG;
        else
        {
          name = prop.bstrVal;
          NItemName::ReplaceSlashes_OsToUnix(name);
        }
      }
      {
        NCOM::CPropVariant prop;
        RINOK(updateCallback->GetProperty(i, kpidIsDir, &prop))
        if (prop.vt == VT_EMPTY)
          folderStatusIsDefined = false;
        else if (prop.vt != VT_BOOL)
          return E_INVALIDARG;
        else
        {
          ui.IsDir = (prop.boolVal != VARIANT_FALSE);
          folderStatusIsDefined = true;
        }
      }

      {
        NCOM::CPropVariant prop;
        RINOK(updateCallback->GetProperty(i, kpidIsAnti, &prop))
        if (prop.vt == VT_EMPTY)
          ui.IsAnti = false;
        else if (prop.vt != VT_BOOL)
          return E_INVALIDARG;
        else
          ui.IsAnti = (prop.boolVal != VARIANT_FALSE);
      }

      /*
      {
        NCOM::CPropVariant prop;
        RINOK(updateCallback->GetProperty(i, kpidIsAltStream, &prop));
        if (prop.vt == VT_EMPTY)
          isAltStream = false;
        else if (prop.vt != VT_BOOL)
          return E_INVALIDARG;
        else
          isAltStream = (prop.boolVal != VARIANT_FALSE);
      }
      */

      if (ui.IsAnti)
      {
        ui.AttribDefined = false;

        ui.CTimeDefined = false;
        ui.ATimeDefined = false;
        ui.MTimeDefined = false;
        
        ui.Size = 0;
      }

      if (!folderStatusIsDefined && ui.AttribDefined)
        ui.SetDirStatusFromAttrib();
    }
    else
    {
      /*
      if (_db.SecureIDs.IsEmpty())
        ui.SecureIndex = secureBlocks.AddUniq(NULL, 0);
      else
      {
        int id = _db.SecureIDs[ui.IndexInArchive];
        size_t offs = _db.SecureOffsets[id];
        size_t size = _db.SecureOffsets[id + 1] - offs;
        ui.SecureIndex = secureBlocks.AddUniq(_db.SecureBuf + offs, size);
      }
      */
    }

    /*
    {
      int folderIndex = 0;
      if (_useParents)
      {
        int j;
        s.Empty();
        for (j = 0; j < name.Len(); j++)
        {
          wchar_t c = name[j];
          if (IsCharDirLimiter(c))
          {
            folderIndex = AddFolder(treeFolders, folderIndex, s);
            s.Empty();
            continue;
          }
          s += c;
        }
        if (isAltStream)
        {
          int colonPos = s.Find(':');
          if (colonPos < 0)
          {
            // isAltStream = false;
            return E_INVALIDARG;
          }
          UString mainName = s.Left(colonPos);
          int newFolderIndex = AddFolder(treeFolders, folderIndex, mainName);
          if (treeFolders[newFolderIndex].UpdateItemIndex < 0)
          {
            for (int j = updateItems.Size() - 1; j >= 0; j--)
            {
              CUpdateItem &ui2 = updateItems[j];
              if (ui2.ParentFolderIndex == folderIndex
                  && ui2.Name == mainName)
              {
                ui2.TreeFolderIndex = newFolderIndex;
                treeFolders[newFolderIndex].UpdateItemIndex = j;
              }
            }
          }
          folderIndex = newFolderIndex;
          s.Delete(0, colonPos + 1);
        }
        ui.Name = s;
      }
      else
        ui.Name = name;
      ui.IsAltStream = isAltStream;
      ui.ParentFolderIndex = folderIndex;
      ui.TreeFolderIndex = -1;
      if (ui.IsDir && !s.IsEmpty())
      {
        ui.TreeFolderIndex = AddFolder(treeFolders, folderIndex, s);
        treeFolders[ui.TreeFolderIndex].UpdateItemIndex = updateItems.Size();
      }
    }
    */
    ui.Name = name;

    if (ui.NewData)
    {
      ui.Size = 0;
      if (!ui.IsDir)
      {
        NCOM::CPropVariant prop;
        RINOK(updateCallback->GetProperty(i, kpidSize, &prop))
        if (prop.vt != VT_UI8)
          return E_INVALIDARG;
        ui.Size = (UInt64)prop.uhVal.QuadPart;
        if (ui.Size != 0 && ui.IsAnti)
          return E_INVALIDARG;
      }
    }
    
    updateItems.Add(ui);
  }

  /*
  FillSortIndex(treeFolders, 0, 0);
  for (i = 0; i < (UInt32)updateItems.Size(); i++)
  {
    CUpdateItem &ui = updateItems[i];
    ui.ParentSortIndex = treeFolders[ui.ParentFolderIndex].SortIndex;
    ui.ParentSortIndexEnd = treeFolders[ui.ParentFolderIndex].SortIndexEnd;
  }
  */

  CCompressionMethodMode methodMode, headerMethod;

  methodMode.MemoryUsageLimit = _memUsage_Compress;
  methodMode.MemoryUsageLimit_WasSet = _memUsage_WasSet;

  #ifndef Z7_ST
  {
    UInt32 numThreads = _numThreads;
    const UInt32 kNumThreads_Max = 1024;
    if (numThreads > kNumThreads_Max)
      numThreads = kNumThreads_Max;
    methodMode.NumThreads = numThreads;
    methodMode.NumThreads_WasForced = _numThreads_WasForced;
    methodMode.MultiThreadMixer = _useMultiThreadMixer;
#ifdef _WIN32
    methodMode.NumThreadGroups = _numThreadGroups; // _change it
#endif
    // headerMethod.NumThreads = 1;
    headerMethod.MultiThreadMixer = _useMultiThreadMixer;
  }
  #endif

  const HRESULT res = SetMainMethod(methodMode);
  RINOK(res)

  RINOK(SetHeaderMethod(headerMethod))
  
  Z7_DECL_CMyComPtr_QI_FROM(
    ICryptoGetTextPassword2,
    getPassword2, updateCallback)

  methodMode.PasswordIsDefined = false;
  methodMode.Password.Wipe_and_Empty();
  if (getPassword2)
  {
    CMyComBSTR_Wipe password;
    Int32 passwordIsDefined;
    RINOK(getPassword2->CryptoGetTextPassword2(&passwordIsDefined, &password))
    methodMode.PasswordIsDefined = IntToBool(passwordIsDefined);
    if (methodMode.PasswordIsDefined && password)
      methodMode.Password = password;
  }

  bool compressMainHeader = _compressHeaders;  // check it

  bool encryptHeaders = false;

  #ifndef Z7_NO_CRYPTO
  if (!methodMode.PasswordIsDefined && _passwordIsDefined)
  {
    // if header is compressed, we use that password for updated archive
    methodMode.PasswordIsDefined = true;
    methodMode.Password = _password;
  }
  #endif

  if (methodMode.PasswordIsDefined)
  {
    if (_encryptHeadersSpecified)
      encryptHeaders = _encryptHeaders;
    #ifndef Z7_NO_CRYPTO
    else
      encryptHeaders = _passwordIsDefined;
    #endif
    compressMainHeader = true;
    if (encryptHeaders)
    {
      headerMethod.PasswordIsDefined = methodMode.PasswordIsDefined;
      headerMethod.Password = methodMode.Password;
    }
  }

  if (numItems < 2)
    compressMainHeader = false;

  const int level = GetLevel();

  CUpdateOptions options;
  options.Need_CTime = need_CTime;
  options.Need_ATime = need_ATime;
  options.Need_MTime = need_MTime;
  options.Need_Attrib = need_Attrib;
  // options.Need_Crc = (_crcSize != 0); // for debug

  options.Method = &methodMode;
  options.HeaderMethod = (_compressHeaders || encryptHeaders) ? &headerMethod : NULL;
  options.UseFilters = (level != 0 && _autoFilter && !methodMode.Filter_was_Inserted);
  options.MaxFilter = (level >= 8);
  options.AnalysisLevel = GetAnalysisLevel();

  options.SetFilterSupporting_ver_enabled_disabled(
      _decoderCompatibilityVersion,
      _enabledFilters,
      _disabledFilters);

  options.HeaderOptions.CompressMainHeader = compressMainHeader;
  /*
  options.HeaderOptions.WriteCTime = Write_CTime;
  options.HeaderOptions.WriteATime = Write_ATime;
  options.HeaderOptions.WriteMTime = Write_MTime;
  options.HeaderOptions.WriteAttrib = Write_Attrib;
  */
  
  options.NumSolidFiles = _numSolidFiles;
  options.NumSolidBytes = _numSolidBytes;
  options.SolidExtension = _solidExtension;
  options.UseTypeSorting = _useTypeSorting;

  options.RemoveSfxBlock = _removeSfxBlock;
  // options.VolumeMode = _volumeMode;

  options.MultiThreadMixer = _useMultiThreadMixer;

  /*
  if (secureBlocks.Sorted.Size() > 1)
  {
    secureBlocks.GetReverseMap();
    for (int i = 0; i < updateItems.Size(); i++)
    {
      int &secureIndex = updateItems[i].SecureIndex;
      secureIndex = secureBlocks.BufIndexToSortedIndex[secureIndex];
    }
  }
  */

  return Update(
      EXTERNAL_CODECS_VARS
      #ifdef Z7_7Z_VOL
      volume ? volume->Stream: 0,
      volume ? db : 0,
      #else
      _inStream,
      db,
      #endif
      updateItems,
      // treeFolders,
      // secureBlocks,
      outStream, updateCallback, options);

  COM_TRY_END
}

static HRESULT ParseBond(UString &srcString, UInt32 &coder, UInt32 &stream)
{
  stream = 0;
  {
    const unsigned index = ParseStringToUInt32(srcString, coder);
    if (index == 0)
      return E_INVALIDARG;
    srcString.DeleteFrontal(index);
  }
  if (srcString[0] == 's')
  {
    srcString.Delete(0);
    const unsigned index = ParseStringToUInt32(srcString, stream);
    if (index == 0)
      return E_INVALIDARG;
    srcString.DeleteFrontal(index);
  }
  return S_OK;
}

void COutHandler::InitProps7z()
{
  _removeSfxBlock = false;
  _compressHeaders = true;
  _encryptHeadersSpecified = false;
  _encryptHeaders = false;
  // _useParents = false;
  
  TimeOptions.Init();
  Write_Attrib.Init();

  _useMultiThreadMixer = true;

  // _volumeMode = false;

  InitSolid();
  _useTypeSorting = false;

  _decoderCompatibilityVersion = k_decoderCompatibilityVersion;
  _enabledFilters.Clear();
  _disabledFilters.Clear();
}

void COutHandler::InitProps()
{
  CMultiMethodProps::Init();
  InitProps7z();
}



HRESULT COutHandler::SetSolidFromString(const UString &s)
{
  UString s2 = s;
  s2.MakeLower_Ascii();
  for (unsigned i = 0; i < s2.Len();)
  {
    const wchar_t *start = ((const wchar_t *)s2) + i;
    const wchar_t *end;
    UInt64 v = ConvertStringToUInt64(start, &end);
    if (start == end)
    {
      if (s2[i++] != 'e')
        return E_INVALIDARG;
      _solidExtension = true;
      continue;
    }
    i += (unsigned)(end - start);
    if (i == s2.Len())
      return E_INVALIDARG;
    const wchar_t c = s2[i++];
    if (c == 'f')
    {
      if (v < 1)
        v = 1;
      _numSolidFiles = v;
    }
    else
    {
      unsigned numBits;
      switch (c)
      {
        case 'b': numBits =  0; break;
        case 'k': numBits = 10; break;
        case 'm': numBits = 20; break;
        case 'g': numBits = 30; break;
        case 't': numBits = 40; break;
        default: return E_INVALIDARG;
      }
      _numSolidBytes = (v << numBits);
      _numSolidBytesDefined = true;
      /*
      if (_numSolidBytes == 0)
        _numSolidFiles = 1;
      */
    }
  }
  return S_OK;
}

HRESULT COutHandler::SetSolidFromPROPVARIANT(const PROPVARIANT &value)
{
  bool isSolid;
  switch (value.vt)
  {
    case VT_EMPTY: isSolid = true; break;
    case VT_BOOL: isSolid = (value.boolVal != VARIANT_FALSE); break;
    case VT_BSTR:
      if (StringToBool(value.bstrVal, isSolid))
        break;
      return SetSolidFromString(value.bstrVal);
    default: return E_INVALIDARG;
  }
  if (isSolid)
    InitSolid();
  else
    _numSolidFiles = 1;
  return S_OK;
}

static HRESULT PROPVARIANT_to_BoolPair(const PROPVARIANT &prop, CBoolPair &dest)
{
  RINOK(PROPVARIANT_to_bool(prop, dest.Val))
  dest.Def = true;
  return S_OK;
}

struct C_Id_Name_pair
{
  UInt32 Id;
  const char *Name;
};

static const C_Id_Name_pair g_filter_pairs[] =
{
  { k_Delta, "Delta" },
  { k_ARM64, "ARM64" },
  { k_RISCV, "RISCV" },
  { k_SWAP2, "SWAP2" },
  { k_SWAP4, "SWAP4" },
  { k_BCJ,   "BCJ" },
  { k_BCJ2 , "BCJ2" },
  { k_PPC,   "PPC" },
  { k_IA64,  "IA64" },
  { k_ARM,   "ARM" },
  { k_ARMT,  "ARMT" },
  { k_SPARC, "SPARC" }
};


HRESULT COutHandler::SetProperty(const wchar_t *nameSpec, const PROPVARIANT &value)
{
  UString name = nameSpec;
  name.MakeLower_Ascii();
  if (name.IsEmpty())
    return E_INVALIDARG;
  
  if (name[0] == L's')
  {
    name.Delete(0);
    if (name.IsEmpty())
      return SetSolidFromPROPVARIANT(value);
    if (value.vt != VT_EMPTY)
      return E_INVALIDARG;
    return SetSolidFromString(name);
  }

  UInt32 number;
  const unsigned index = ParseStringToUInt32(name, number);
  // UString realName = name.Ptr(index);
  if (index == 0)
  {
    if (name.IsEqualTo("rsfx")) return PROPVARIANT_to_bool(value, _removeSfxBlock);
    if (name.IsEqualTo("hc")) return PROPVARIANT_to_bool(value, _compressHeaders);
    // if (name.IsEqualToNoCase(L"HS")) return PROPVARIANT_to_bool(value, _useParents);
    
    if (name.IsEqualTo("hcf"))
    {
      bool compressHeadersFull = true;
      RINOK(PROPVARIANT_to_bool(value, compressHeadersFull))
      return compressHeadersFull ? S_OK: E_INVALIDARG;
    }
    
    if (name.IsEqualTo("he"))
    {
      RINOK(PROPVARIANT_to_bool(value, _encryptHeaders))
      _encryptHeadersSpecified = true;
      return S_OK;
    }
    
    {
      bool processed;
      RINOK(TimeOptions.Parse(name, value, processed))
      if (processed)
      {
        if (   TimeOptions.Prec != (UInt32)(Int32)-1
            && TimeOptions.Prec != k_PropVar_TimePrec_0
            && TimeOptions.Prec != k_PropVar_TimePrec_HighPrec
            && TimeOptions.Prec != k_PropVar_TimePrec_100ns)
          return E_INVALIDARG;
        return S_OK;
      }
    }

    if (name.IsEqualTo("tr")) return PROPVARIANT_to_BoolPair(value, Write_Attrib);
    
    if (name.IsEqualTo("mtf")) return PROPVARIANT_to_bool(value, _useMultiThreadMixer);

    if (name.IsEqualTo("qs")) return PROPVARIANT_to_bool(value, _useTypeSorting);

    if (name.IsPrefixedBy_Ascii_NoCase("yv"))
    {
      name.Delete(0, 2);
      UInt32 v = 1 << 16;  // if no number is noit specified, we use big value
      RINOK(ParsePropToUInt32(name, value, v))
      _decoderCompatibilityVersion = v;
      // if (v == 0) _decoderCompatibilityVersion = k_decoderCompatibilityVersion;
      return S_OK;
    }

    if (name.IsPrefixedBy_Ascii_NoCase("yf"))
    {
      name.Delete(0, 2);
      CUIntVector *vec;
           if (name.IsEqualTo_Ascii_NoCase("a")) vec = &_enabledFilters;
      else if (name.IsEqualTo_Ascii_NoCase("d")) vec = &_disabledFilters;
      else return E_INVALIDARG;

      if (value.vt != VT_BSTR)
        return E_INVALIDARG;
      for (unsigned k = 0;; k++)
      {
        if (k == Z7_ARRAY_SIZE(g_filter_pairs))
        {
          // maybe we can ignore unsupported filter names here?
          return E_INVALIDARG;
        }
        const C_Id_Name_pair &pair = g_filter_pairs[k];
        if (StringsAreEqualNoCase_Ascii(value.bstrVal, pair.Name))
        {
          vec->AddToUniqueSorted(pair.Id);
          break;
        }
      }
      return S_OK;
    }

    // if (name.IsEqualTo("v"))  return PROPVARIANT_to_bool(value, _volumeMode);
  }
  return CMultiMethodProps::SetProperty(name, value);
}

Z7_COM7F_IMF(CHandler::SetProperties(const wchar_t * const *names, const PROPVARIANT *values, UInt32 numProps))
{
  COM_TRY_BEGIN
  _bonds.Clear();
  InitProps();

  for (UInt32 i = 0; i < numProps; i++)
  {
    UString name = names[i];
    name.MakeLower_Ascii();
    if (name.IsEmpty())
      return E_INVALIDARG;

    const PROPVARIANT &value = values[i];

    if (name.Find(L':') >= 0) // 'b' was used as NCoderPropID::kBlockSize2 before v23
    if (name[0] == 'b')
    {
      if (value.vt != VT_EMPTY)
        return E_INVALIDARG;
      name.Delete(0);
      
      CBond2 bond;
      RINOK(ParseBond(name, bond.OutCoder, bond.OutStream))
      if (name[0] != ':')
        return E_INVALIDARG;
      name.Delete(0);
      UInt32 inStream = 0;
      RINOK(ParseBond(name, bond.InCoder, inStream))
      if (inStream != 0)
        return E_INVALIDARG;
      if (!name.IsEmpty())
        return E_INVALIDARG;
      _bonds.Add(bond);
      continue;
    }

    RINOK(SetProperty(name, value))
  }

  unsigned numEmptyMethods = GetNumEmptyMethods();
  if (numEmptyMethods > 0)
  {
    unsigned k;
    for (k = 0; k < _bonds.Size(); k++)
    {
      const CBond2 &bond = _bonds[k];
      if (bond.InCoder < (UInt32)numEmptyMethods ||
          bond.OutCoder < (UInt32)numEmptyMethods)
        return E_INVALIDARG;
    }
    for (k = 0; k < _bonds.Size(); k++)
    {
      CBond2 &bond = _bonds[k];
      bond.InCoder -= (UInt32)numEmptyMethods;
      bond.OutCoder -= (UInt32)numEmptyMethods;
    }
    _methods.DeleteFrontal(numEmptyMethods);
  }
  
  FOR_VECTOR (k, _bonds)
  {
    const CBond2 &bond = _bonds[k];
    if (bond.InCoder >= (UInt32)_methods.Size() ||
        bond.OutCoder >= (UInt32)_methods.Size())
      return E_INVALIDARG;
  }

  return S_OK;
  COM_TRY_END
}

}}

#endif
