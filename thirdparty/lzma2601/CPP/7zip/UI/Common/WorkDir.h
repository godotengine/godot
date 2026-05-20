// WorkDir.h

#ifndef ZIP7_INC_WORK_DIR_H
#define ZIP7_INC_WORK_DIR_H

#include "../../../Windows/FileDir.h"

#include "../../Common/FileStreams.h"

#include "ZipRegistry.h"

FString GetWorkDir(const NWorkDir::CInfo &workDirInfo, const FString &path, FString &fileName);

class CWorkDirTempFile  MY_UNCOPYABLE
{
  FString _originalPath;
  NWindows::NFile::NDir::CTempFile _tempFile;
  COutFileStream *_outStreamSpec;
public:
  CMyComPtr<IOutStream> OutStream;

  const FString &Get_OriginalFilePath() const { return _originalPath; }
  const FString &Get_TempFilePath() const { return _tempFile.GetPath(); }

  HRESULT CreateTempFile(const FString &originalPath);
  HRESULT MoveToOriginal(bool deleteOriginal,
      NWindows::NFile::NDir::ICopyFileProgress *progress = NULL);
};

#endif
