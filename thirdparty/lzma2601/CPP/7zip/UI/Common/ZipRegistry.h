// ZipRegistry.h

#ifndef ZIP7_INC_ZIP_REGISTRY_H
#define ZIP7_INC_ZIP_REGISTRY_H

#include "../../../Common/MyTypes.h"
#include "../../../Common/MyString.h"

#include "../../Common/MethodProps.h"

#include "ExtractMode.h"

/*
CBoolPair::Def in writing functions means:
  if (  CBoolPair::Def ), we write CBoolPair::Val
  if ( !CBoolPair::Def )
  {
    in NCompression functions we delete registry value
    in another functions we do nothing
  }
*/

namespace NExtract
{
  struct CInfo
  {
    NPathMode::EEnum PathMode;
    NOverwriteMode::EEnum OverwriteMode;
    bool PathMode_Force;
    bool OverwriteMode_Force;
    
    CBoolPair SplitDest;
    CBoolPair ElimDup;
    // CBoolPair AltStreams;
    CBoolPair NtSecurity;
    CBoolPair ShowPassword;

    UStringVector Paths;

    void Save() const;
    void Load();
  };
  
  void Save_ShowPassword(bool showPassword);
  bool Read_ShowPassword();

  void Save_LimitGB(UInt32 limit_GB);
  UInt32 Read_LimitGB();
}

namespace NCompression
{
  struct CMemUse
  {
    // UString Str;
    bool IsDefined;
    bool IsPercent;
    UInt64 Val;

    CMemUse():
      IsDefined(false),
      IsPercent(false),
      Val(0)
      {}

    void Clear()
    {
      // Str.Empty();
      IsDefined = false;
      IsPercent = false;
      Val = 0;
    }

    UInt64 GetBytes(UInt64 ramSize) const
    {
      if (!IsPercent)
        return Val;
      return Calc_From_Val_Percents(ramSize, Val);
    }
    void Parse(const UString &s);
  };

  struct CFormatOptions
  {
    UInt32 Level;
    UInt32 Dictionary;
    // UInt32 DictionaryChain;
    UInt32 Order;
    UInt32 BlockLogSize;
    UInt32 NumThreads;
    
    UInt32 TimePrec;
    CBoolPair MTime;
    CBoolPair ATime;
    CBoolPair CTime;
    CBoolPair SetArcMTime;

    CSysString FormatID;
    UString Method;
    UString Options;
    UString EncryptionMethod;
    UString MemUse;

    void Reset_TimePrec()
    {
      TimePrec = (UInt32)(Int32)-1;
    }

    bool IsSet_TimePrec() const
    {
      return TimePrec != (UInt32)(Int32)-1;
    }


    void Reset_BlockLogSize()
    {
      BlockLogSize = (UInt32)(Int32)-1;
    }

    void ResetForLevelChange()
    {
      BlockLogSize = NumThreads = Level = Dictionary = Order = (UInt32)(Int32)-1;
      // DictionaryChain = (UInt32)(Int32)-1;
      Method.Empty();
      // Options.Empty();
      // EncryptionMethod.Empty();
    }
    CFormatOptions()
    {
      // TimePrec = 0;
      Reset_TimePrec();
      ResetForLevelChange();
    }
  };

  struct CInfo
  {
    UInt32 Level;
    bool ShowPassword;
    bool EncryptHeaders;

    CBoolPair NtSecurity;
    CBoolPair AltStreams;
    CBoolPair HardLinks;
    CBoolPair SymLinks;

    CBoolPair PreserveATime;

    UString ArcType;
    UStringVector ArcPaths;

    CObjectVector<CFormatOptions> Formats;

    void Save() const;
    void Load();
  };
}

namespace NWorkDir
{
  namespace NMode
  {
    enum EEnum
    {
      kSystem,
      kCurrent,
      kSpecified
    };
  }
  struct CInfo
  {
    NMode::EEnum Mode;
    bool ForRemovableOnly;
    FString Path;

    void SetForRemovableOnlyDefault() { ForRemovableOnly = true; }
    void SetDefault()
    {
      Mode = NMode::kSystem;
      Path.Empty();
      SetForRemovableOnlyDefault();
    }

    void Save() const;
    void Load();
  };
}


struct CContextMenuInfo
{
  CBoolPair Cascaded;
  CBoolPair MenuIcons;
  CBoolPair ElimDup;
  
  bool Flags_Def;
  UInt32 Flags;
  UInt32 WriteZone;

  /*
  CContextMenuInfo():
      Flags_Def(0),
      WriteZone((UInt32)(Int32)-1),
      Flags((UInt32)(Int32)-1)
      {}
  */

  void Save() const;
  void Load();
};

#endif
