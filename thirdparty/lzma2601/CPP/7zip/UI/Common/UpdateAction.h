// UpdateAction.h

#ifndef ZIP7_INC_UPDATE_ACTION_H
#define ZIP7_INC_UPDATE_ACTION_H

namespace NUpdateArchive {

  namespace NPairState
  {
    const unsigned kNumValues = 7;
    enum EEnum
    {
      kNotMasked = 0,
      kOnlyInArchive,
      kOnlyOnDisk,
      kNewInArchive,
      kOldInArchive,
      kSameFiles,
      kUnknowNewerFiles
    };
  }
 
  namespace NPairAction
  {
    enum EEnum
    {
      kIgnore = 0,
      kCopy,
      kCompress,
      kCompressAsAnti
    };
  }
  
  struct CActionSet
  {
    NPairAction::EEnum StateActions[NPairState::kNumValues];
    
    bool IsEqualTo(const CActionSet &a) const
    {
      for (unsigned i = 0; i < NPairState::kNumValues; i++)
        if (StateActions[i] != a.StateActions[i])
          return false;
      return true;
    }

    bool NeedScanning() const
    {
      unsigned i;
      for (i = 0; i < NPairState::kNumValues; i++)
        if (StateActions[i] == NPairAction::kCompress)
          return true;
      for (i = 1; i < NPairState::kNumValues; i++)
        if (StateActions[i] != NPairAction::kIgnore)
          return true;
      return false;
    }
  };
  
  extern const CActionSet k_ActionSet_Add;
  extern const CActionSet k_ActionSet_Update;
  extern const CActionSet k_ActionSet_Fresh;
  extern const CActionSet k_ActionSet_Sync;
  extern const CActionSet k_ActionSet_Delete;
}

#endif
