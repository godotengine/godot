// UniqBlocks.cpp

#include "StdAfx.h"

#include <string.h>

#include "UniqBlocks.h"

unsigned CUniqBlocks::AddUniq(const Byte *data, size_t size)
{
  unsigned left = 0, right = Sorted.Size();
  while (left != right)
  {
    const unsigned mid = (unsigned)(((size_t)left + (size_t)right) / 2);
    const unsigned index = Sorted[mid];
    const CByteBuffer &buf = Bufs[index];
    const size_t sizeMid = buf.Size();
    if (size < sizeMid)
      right = mid;
    else if (size > sizeMid)
      left = mid + 1;
    else
    {
      if (size == 0)
        return index;
      const int cmp = memcmp(data, buf, size);
      if (cmp == 0)
        return index;
      if (cmp < 0)
        right = mid;
      else
        left = mid + 1;
    }
  }
  unsigned index = Bufs.Size();
  Sorted.Insert(left, index);
  Bufs.AddNew().CopyFrom(data, size);
  return index;
}

UInt64 CUniqBlocks::GetTotalSizeInBytes() const
{
  UInt64 size = 0;
  FOR_VECTOR (i, Bufs)
    size += Bufs[i].Size();
  return size;
}

void CUniqBlocks::GetReverseMap()
{
  unsigned num = Sorted.Size();
  BufIndexToSortedIndex.ClearAndSetSize(num);
  unsigned *p = &BufIndexToSortedIndex[0];
  const unsigned *sorted = &Sorted[0];
  for (unsigned i = 0; i < num; i++)
    p[sorted[i]] = i;
}
