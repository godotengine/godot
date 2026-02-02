// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// USDA(Ascii) writer
//

#include "usda-writer.hh"

#if !defined(TINYUSDZ_DISABLE_MODULE_USDA_WRITER)

#include <fstream>
#include <iostream>
#include <sstream>

#include "pprinter.hh"
#include "value-pprint.hh"
#include "tinyusdz.hh"
#include "io-util.hh"

namespace tinyusdz {
namespace usda {

namespace {


}  // namespace

bool SaveAsUSDA(const std::string &filename, const Stage &stage,
                std::string *warn, std::string *err) {

  (void)warn;

  // TODO: Handle warn and err on export.
  std::string s = stage.ExportToString();

  if (!io::WriteWholeFile(filename, reinterpret_cast<const unsigned char *>(s.data()), s.size(), err)) {
    return false;
  }

  std::cout << "Wrote to [" << filename << "]\n";

  return true;
}

#if defined(_WIN32)
bool SaveAsUSDA(const std::wstring &filename, const Stage &stage,
                std::string *warn, std::string *err) {

  (void)warn;

  // TODO: Handle warn and err on export.
  std::string s = stage.ExportToString();

  if (!io::WriteWholeFile(filename, reinterpret_cast<const unsigned char *>(s.data()), s.size(), err)) {
    return false;
  }

  std::wcout << "Wrote to [" << filename << "]\n";

  return true;
}
#endif

} // namespace usda
}  // namespace tinyusdz


#else

namespace tinyusdz {
namespace usda {

bool SaveAsUSDA(const std::string &filename, const Stage &stage, std::string *warn, std::string *err) {
  (void)filename;
  (void)stage;
  (void)warn;

  if (err) {
    (*err) = "USDA Writer feature is disabled in this build.\n";
  }
  return false;
}



} // namespace usda
}  // namespace tinyusdz
#endif


