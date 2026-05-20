// UpdatePair.cpp

#include "StdAfx.h"

#include <time.h>
// #include <stdio.h>

#include "../../../Common/Wildcard.h"

#include "../../../Windows/TimeUtils.h"

#include "SortUtils.h"
#include "UpdatePair.h"

using namespace NWindows;
using namespace NTime;


/*
  a2.Prec =
  {
    0 (k_PropVar_TimePrec_0):
       if GetProperty(kpidMTime) returned 0 and
          GetProperty(kpidTimeType) did not returned VT_UI4.
       7z, wim, tar in 7-Zip before v21)
    in that case we use
      (prec) that is set by IOutArchive::GetFileTimeType()
  }
*/
       
static int MyCompareTime(unsigned prec, const CFiTime &f1, const CArcTime &a2)
{
  // except of precision, we also have limitation, when timestamp is out of range

  /* if (Prec) in archive item is defined, then use global (prec) */
  if (a2.Prec != k_PropVar_TimePrec_0)
    prec = a2.Prec;

  CArcTime a1;
  a1.Set_From_FiTime(f1);
  /* Set_From_FiTime() must set full form precision:
     k_PropVar_TimePrec_Base + numDigits
     windows: 7 digits, non-windows: 9 digits */

  if (prec == k_PropVar_TimePrec_DOS)
  {
    const UInt32 dosTime1 = a1.Get_DosTime();
    const UInt32 dosTime2 = a2.Get_DosTime();
    return MyCompare(dosTime1, dosTime2);
  }

  if (prec == k_PropVar_TimePrec_Unix)
  {
    const Int64 u2 = FileTime_To_UnixTime64(a2.FT);
    if (u2 == 0 || u2 == (UInt32)0xFFFFFFFF)
    {
      // timestamp probably was saturated in archive to 32-bit
      // so we use saturated 32-bit value for disk file too.
      UInt32 u1;
      FileTime_To_UnixTime(a1.FT, u1);
      const UInt32 u2_32 = (UInt32)u2;
      return MyCompare(u1, u2_32);
    }

    const Int64 u1 = FileTime_To_UnixTime64(a1.FT);
    return MyCompare(u1, u2);
    // prec = k_PropVar_TimePrec_Base; // for debug
  }

  if (prec == k_PropVar_TimePrec_0)
    prec = k_PropVar_TimePrec_Base + 7;
  else if (prec == k_PropVar_TimePrec_HighPrec)
    prec = k_PropVar_TimePrec_Base + 9;
  else if (prec < k_PropVar_TimePrec_Base)
    prec = k_PropVar_TimePrec_Base;
  else if (prec > k_PropVar_TimePrec_Base + 9)
    prec = k_PropVar_TimePrec_Base + 7;

  // prec now is full form: k_PropVar_TimePrec_Base + numDigits;
  if (prec > a1.Prec && a1.Prec >= k_PropVar_TimePrec_Base)
    prec = a1.Prec;

  const unsigned numDigits = prec - k_PropVar_TimePrec_Base;
  if (numDigits >= 7)
  {
    const int comp = CompareFileTime(&a1.FT, &a2.FT);
    if (comp != 0 || numDigits == 7)
      return comp;
    return MyCompare(a1.Ns100, a2.Ns100);
  }
  UInt32 d = 1;
  for (unsigned k = numDigits; k < 7; k++)
    d *= 10;
  const UInt64 v1 = a1.Get_FILETIME_as_UInt64() / d * d;
  const UInt64 v2 = a2.Get_FILETIME_as_UInt64() / d * d;
  // printf("\ndelta=%d numDigits=%d\n", (unsigned)(v1- v2), numDigits);
  return MyCompare(v1, v2);
}



static const char * const k_Duplicate_inArc_Message = "Duplicate filename in archive:";
static const char * const k_Duplicate_inDir_Message = "Duplicate filename on disk:";
static const char * const k_NotCensoredCollision_Message = "Internal file name collision (file on disk, file in archive):";

Z7_ATTR_NORETURN
static
void ThrowError(const char *message, const UString &s1, const UString &s2)
{
  UString m (message);
  m.Add_LF(); m += s1;
  m.Add_LF(); m += s2;
  throw m;
}

static int CompareArcItemsBase(const CArcItem &ai1, const CArcItem &ai2)
{
  const int res = CompareFileNames(ai1.Name, ai2.Name);
  if (res != 0)
    return res;
  if (ai1.IsDir != ai2.IsDir)
    return ai1.IsDir ? -1 : 1;
  return 0;
}

static int CompareArcItems(const unsigned *p1, const unsigned *p2, void *param)
{
  const unsigned i1 = *p1;
  const unsigned i2 = *p2;
  const CObjectVector<CArcItem> &arcItems = *(const CObjectVector<CArcItem> *)param;
  const int res = CompareArcItemsBase(arcItems[i1], arcItems[i2]);
  if (res != 0)
    return res;
  return MyCompare(i1, i2);
}

void GetUpdatePairInfoList(
    const CDirItems &dirItems,
    const CObjectVector<CArcItem> &arcItems,
    NFileTimeType::EEnum fileTimeType,
    CRecordVector<CUpdatePair> &updatePairs)
{
  CUIntVector dirIndices, arcIndices;
  
  const unsigned numDirItems = dirItems.Items.Size();
  const unsigned numArcItems = arcItems.Size();
  
  CIntArr duplicatedArcItem(numArcItems);
  {
    int *vals = &duplicatedArcItem[0];
    for (unsigned i = 0; i < numArcItems; i++)
      vals[i] = 0;
  }

  {
    arcIndices.ClearAndSetSize(numArcItems);
    if (numArcItems != 0)
    {
      unsigned *vals = &arcIndices[0];
      for (unsigned i = 0; i < numArcItems; i++)
        vals[i] = i;
    }
    arcIndices.Sort(CompareArcItems, (void *)&arcItems);
    for (unsigned i = 0; i + 1 < numArcItems; i++)
      if (CompareArcItemsBase(
          arcItems[arcIndices[i]],
          arcItems[arcIndices[i + 1]]) == 0)
      {
        duplicatedArcItem[i] = 1;
        duplicatedArcItem[i + 1] = -1;
      }
  }

  UStringVector dirNames;
  {
    dirNames.ClearAndReserve(numDirItems);
    unsigned i;
    for (i = 0; i < numDirItems; i++)
      dirNames.AddInReserved(dirItems.GetLogPath(i));
    SortFileNames(dirNames, dirIndices);
    for (i = 0; i + 1 < numDirItems; i++)
    {
      const UString &s1 = dirNames[dirIndices[i]];
      const UString &s2 = dirNames[dirIndices[i + 1]];
      if (CompareFileNames(s1, s2) == 0)
        ThrowError(k_Duplicate_inDir_Message, s1, s2);
    }
  }
  
  unsigned dirIndex = 0;
  unsigned arcIndex = 0;

  int prevHostFile = -1;
  const UString *prevHostName = NULL;
  
  while (dirIndex < numDirItems || arcIndex < numArcItems)
  {
    CUpdatePair pair;
    
    int dirIndex2 = -1;
    int arcIndex2 = -1;
    const CDirItem *di = NULL;
    const CArcItem *ai = NULL;
    
    int compareResult = -1;
    const UString *name = NULL;
    
    if (dirIndex < numDirItems)
    {
      dirIndex2 = (int)dirIndices[dirIndex];
      di = &dirItems.Items[(unsigned)dirIndex2];
    }

    if (arcIndex < numArcItems)
    {
      arcIndex2 = (int)arcIndices[arcIndex];
      ai = &arcItems[(unsigned)arcIndex2];
      compareResult = 1;
      if (dirIndex < numDirItems)
      {
        compareResult = CompareFileNames(dirNames[(unsigned)dirIndex2], ai->Name);
        if (compareResult == 0)
        {
          if (di->IsDir() != ai->IsDir)
            compareResult = (ai->IsDir ? 1 : -1);
        }
      }
    }
    
    if (compareResult < 0)
    {
      name = &dirNames[(unsigned)dirIndex2];
      pair.State = NUpdateArchive::NPairState::kOnlyOnDisk;
      pair.DirIndex = dirIndex2;
      dirIndex++;
    }
    else if (compareResult > 0)
    {
      name = &ai->Name;
      pair.State = ai->Censored ?
          NUpdateArchive::NPairState::kOnlyInArchive:
          NUpdateArchive::NPairState::kNotMasked;
      pair.ArcIndex = arcIndex2;
      arcIndex++;
    }
    else
    {
      const int dupl = duplicatedArcItem[arcIndex];
      if (dupl != 0)
        ThrowError(k_Duplicate_inArc_Message, ai->Name, arcItems[arcIndices[(unsigned)((int)arcIndex + dupl)]].Name);

      name = &dirNames[(unsigned)dirIndex2];
      if (!ai->Censored)
        ThrowError(k_NotCensoredCollision_Message, *name, ai->Name);
      
      pair.DirIndex = dirIndex2;
      pair.ArcIndex = arcIndex2;

      int compResult = 0;
      if (ai->MTime.Def)
      {
        compResult = MyCompareTime((unsigned)fileTimeType, di->MTime, ai->MTime);
      }
      switch (compResult)
      {
        case -1: pair.State = NUpdateArchive::NPairState::kNewInArchive; break;
        case  1: pair.State = NUpdateArchive::NPairState::kOldInArchive; break;
        default:
          pair.State = (ai->Size_Defined && di->Size == ai->Size) ?
              NUpdateArchive::NPairState::kSameFiles :
              NUpdateArchive::NPairState::kUnknowNewerFiles;
      }
      
      dirIndex++;
      arcIndex++;
    }
    
    if (
       #ifdef _WIN32
        (di && di->IsAltStream) ||
       #endif
        (ai && ai->IsAltStream))
    {
      if (prevHostName)
      {
        const unsigned hostLen = prevHostName->Len();
        if (name->Len() > hostLen)
          if ((*name)[hostLen] == ':' && CompareFileNames(*prevHostName, name->Left(hostLen)) == 0)
            pair.HostIndex = prevHostFile;
      }
    }
    else
    {
      prevHostFile = (int)updatePairs.Size();
      prevHostName = name;
    }
    
    updatePairs.Add(pair);
  }

  updatePairs.ReserveDown();
}
