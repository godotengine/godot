// 7zProperties.cpp

#include "StdAfx.h"

#include "7zHandler.h"
#include "7zProperties.h"

namespace NArchive {
namespace N7z {

struct CPropMap
{
  Byte FilePropID;
  // CStatProp StatProp;
  VARTYPE vt;
  UInt32 StatPropID;
};

// #define STAT_PROP(name, id, vt)  { name, id, vt }
#define STAT_PROP(name, id, vt)  vt, id

#define STAT_PROP2(id, vt) STAT_PROP(NULL, id, vt)

#define k_7z_id_Encrypted    97
#define k_7z_id_Method       98
#define k_7z_id_Block        99

static const CPropMap kPropMap[] =
{
  { NID::kName, STAT_PROP2(kpidPath, VT_BSTR) },
  { NID::kSize, STAT_PROP2(kpidSize, VT_UI8) },
  { NID::kPackInfo, STAT_PROP2(kpidPackSize, VT_UI8) },
  
  #ifdef Z7_7Z_SHOW_PACK_STREAMS_SIZES
#define k_7z_id_PackedSize0 100
  { k_7z_id_PackedSize0 + 0, STAT_PROP("Pack0", kpidPackedSize0, VT_UI8) },
  { k_7z_id_PackedSize0 + 1, STAT_PROP("Pack1", kpidPackedSize1, VT_UI8) },
  { k_7z_id_PackedSize0 + 2, STAT_PROP("Pack2", kpidPackedSize2, VT_UI8) },
  { k_7z_id_PackedSize0 + 3, STAT_PROP("Pack3", kpidPackedSize3, VT_UI8) },
  { k_7z_id_PackedSize0 + 4, STAT_PROP("Pack4", kpidPackedSize4, VT_UI8) },
  #endif

  { NID::kCTime, STAT_PROP2(kpidCTime, VT_FILETIME) },
  { NID::kMTime, STAT_PROP2(kpidMTime, VT_FILETIME) },
  { NID::kATime, STAT_PROP2(kpidATime, VT_FILETIME) },
  { NID::kWinAttrib, STAT_PROP2(kpidAttrib, VT_UI4) },
  { NID::kStartPos, STAT_PROP2(kpidPosition, VT_UI8) },

  { NID::kCRC, STAT_PROP2(kpidCRC, VT_UI4) },
  // { NID::kIsAux, STAT_PROP2(kpidIsAux, VT_BOOL) },
  { NID::kAnti, STAT_PROP2(kpidIsAnti, VT_BOOL) }

  #ifndef Z7_SFX
  , { k_7z_id_Encrypted, STAT_PROP2(kpidEncrypted, VT_BOOL) }
  , { k_7z_id_Method,    STAT_PROP2(kpidMethod, VT_BSTR) }
  , { k_7z_id_Block,     STAT_PROP2(kpidBlock, VT_UI4) }
  #endif
};

static void CopyOneItem(CRecordVector<UInt64> &src,
    CRecordVector<UInt64> &dest, const UInt32 item)
{
  FOR_VECTOR (i, src)
    if (src[i] == item)
    {
      dest.Add(item);
      src.Delete(i);
      return;
    }
}

static void RemoveOneItem(CRecordVector<UInt64> &src, const UInt32 item)
{
  FOR_VECTOR (i, src)
    if (src[i] == item)
    {
      src.Delete(i);
      return;
    }
}

static void InsertToHead(CRecordVector<UInt64> &dest, const UInt32 item)
{
  FOR_VECTOR (i, dest)
    if (dest[i] == item)
    {
      dest.Delete(i);
      break;
    }
  dest.Insert(0, item);
}

#define COPY_ONE_ITEM(id) CopyOneItem(fileInfoPopIDs, _fileInfoPopIDs, NID::id);

void CHandler::FillPopIDs()
{
  _fileInfoPopIDs.Clear();

  #ifdef Z7_7Z_VOL
  if (_volumes.Size() < 1)
    return;
  const CVolume &volume = _volumes.Front();
  const CArchiveDatabaseEx &_db = volume.Database;
  #endif

  CRecordVector<UInt64> fileInfoPopIDs = _db.ArcInfo.FileInfoPopIDs;

  RemoveOneItem(fileInfoPopIDs, NID::kEmptyStream);
  RemoveOneItem(fileInfoPopIDs, NID::kEmptyFile);
  /*
  RemoveOneItem(fileInfoPopIDs, NID::kParent);
  RemoveOneItem(fileInfoPopIDs, NID::kNtSecure);
  */

  COPY_ONE_ITEM(kName)
  COPY_ONE_ITEM(kAnti)
  COPY_ONE_ITEM(kSize)
  COPY_ONE_ITEM(kPackInfo)
  COPY_ONE_ITEM(kCTime)
  COPY_ONE_ITEM(kMTime)
  COPY_ONE_ITEM(kATime)
  COPY_ONE_ITEM(kWinAttrib)
  COPY_ONE_ITEM(kCRC)
  COPY_ONE_ITEM(kComment)

  _fileInfoPopIDs += fileInfoPopIDs;
 
  #ifndef Z7_SFX
  _fileInfoPopIDs.Add(k_7z_id_Encrypted);
  _fileInfoPopIDs.Add(k_7z_id_Method);
  _fileInfoPopIDs.Add(k_7z_id_Block);
  #endif

  #ifdef Z7_7Z_SHOW_PACK_STREAMS_SIZES
  for (unsigned i = 0; i < 5; i++)
    _fileInfoPopIDs.Add(k_7z_id_PackedSize0 + i);
  #endif

  #ifndef Z7_SFX
  InsertToHead(_fileInfoPopIDs, NID::kMTime);
  InsertToHead(_fileInfoPopIDs, NID::kPackInfo);
  InsertToHead(_fileInfoPopIDs, NID::kSize);
  InsertToHead(_fileInfoPopIDs, NID::kName);
  #endif
}

Z7_COM7F_IMF(CHandler::GetNumberOfProperties(UInt32 *numProps))
{
  *numProps = _fileInfoPopIDs.Size();
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetPropertyInfo(UInt32 index, BSTR *name, PROPID *propID, VARTYPE *varType))
{
  if (index >= _fileInfoPopIDs.Size())
    return E_INVALIDARG;
  const UInt64 id = _fileInfoPopIDs[index];
  for (unsigned i = 0; i < Z7_ARRAY_SIZE(kPropMap); i++)
  {
    const CPropMap &pr = kPropMap[i];
    if (pr.FilePropID == id)
    {
      *propID = pr.StatPropID;
      *varType = pr.vt;
      /*
      const CStatProp &st = pr.StatProp;
      *propID = st.PropID;
      *varType = st.vt;
      */
      /*
      if (st.lpwstrName)
        *name = ::SysAllocString(st.lpwstrName);
      else
      */
        *name = NULL;
      return S_OK;
    }
  }
  return E_INVALIDARG;
}

}}
