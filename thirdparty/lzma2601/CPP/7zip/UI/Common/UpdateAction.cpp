// UpdateAction.cpp

#include "StdAfx.h"

#include "UpdateAction.h"

namespace NUpdateArchive {

const CActionSet k_ActionSet_Add =
{{
  NPairAction::kCopy,
  NPairAction::kCopy,
  NPairAction::kCompress,
  NPairAction::kCompress,
  NPairAction::kCompress,
  NPairAction::kCompress,
  NPairAction::kCompress
}};

const CActionSet k_ActionSet_Update =
{{
  NPairAction::kCopy,
  NPairAction::kCopy,
  NPairAction::kCompress,
  NPairAction::kCopy,
  NPairAction::kCompress,
  NPairAction::kCopy,
  NPairAction::kCompress
}};

const CActionSet k_ActionSet_Fresh =
{{
  NPairAction::kCopy,
  NPairAction::kCopy,
  NPairAction::kIgnore,
  NPairAction::kCopy,
  NPairAction::kCompress,
  NPairAction::kCopy,
  NPairAction::kCompress
}};

const CActionSet k_ActionSet_Sync =
{{
  NPairAction::kCopy,
  NPairAction::kIgnore,
  NPairAction::kCompress,
  NPairAction::kCopy,
  NPairAction::kCompress,
  NPairAction::kCopy,
  NPairAction::kCompress,
}};

const CActionSet k_ActionSet_Delete =
{{
  NPairAction::kCopy,
  NPairAction::kIgnore,
  NPairAction::kIgnore,
  NPairAction::kIgnore,
  NPairAction::kIgnore,
  NPairAction::kIgnore,
  NPairAction::kIgnore
}};

}
