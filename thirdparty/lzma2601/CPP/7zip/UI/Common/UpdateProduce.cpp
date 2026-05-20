// UpdateProduce.cpp

#include "StdAfx.h"

#include "UpdateProduce.h"

using namespace NUpdateArchive;

static const char * const kUpdateActionSetCollision = "Internal collision in update action set";

void UpdateProduce(
    const CRecordVector<CUpdatePair> &updatePairs,
    const CActionSet &actionSet,
    CRecordVector<CUpdatePair2> &operationChain,
    IUpdateProduceCallback *callback)
{
  FOR_VECTOR (i, updatePairs)
  {
    const CUpdatePair &pair = updatePairs[i];

    CUpdatePair2 up2;
    up2.DirIndex = pair.DirIndex;
    up2.ArcIndex = pair.ArcIndex;
    up2.NewData = up2.NewProps = true;
    up2.UseArcProps = false;
    
    switch ((int)actionSet.StateActions[(unsigned)pair.State])
    {
      case NPairAction::kIgnore:
        if (pair.ArcIndex >= 0 && callback)
          callback->ShowDeleteFile((unsigned)pair.ArcIndex);
        continue;

      case NPairAction::kCopy:
        if (pair.State == NPairState::kOnlyOnDisk)
          throw kUpdateActionSetCollision;
        if (pair.State == NPairState::kOnlyInArchive)
        {
          if (pair.HostIndex >= 0)
          {
            /*
              ignore alt stream if
                1) no such alt stream in Disk
                2) there is Host file in disk
            */
            if (updatePairs[(unsigned)pair.HostIndex].DirIndex >= 0)
              continue;
          }
        }
        up2.NewData = up2.NewProps = false;
        up2.UseArcProps = true;
        break;
      
      case NPairAction::kCompress:
        if (pair.State == NPairState::kOnlyInArchive ||
            pair.State == NPairState::kNotMasked)
          throw kUpdateActionSetCollision;
        break;
      
      case NPairAction::kCompressAsAnti:
        up2.IsAnti = true;
        up2.UseArcProps = (pair.ArcIndex >= 0);
        break;
      
      default: throw 123; // break; // is unexpected case
    }

    up2.IsSameTime = ((unsigned)pair.State == NUpdateArchive::NPairState::kSameFiles);

    operationChain.Add(up2);
  }
  
  operationChain.ReserveDown();
}
