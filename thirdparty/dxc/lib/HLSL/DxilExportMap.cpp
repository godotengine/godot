///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilExportMap.cpp                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// dxilutil::ExportMap for handling -exports option.                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/DxilExportMap.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include <string>
#include <vector>
#include <set>

using namespace llvm;
using namespace hlsl;

namespace hlsl {
namespace dxilutil {

void ExportMap::clear() {
  m_ExportMap.clear();
}
bool ExportMap::empty() const {
  return m_ExportMap.empty();
}

bool ExportMap::ParseExports(const std::vector<std::string> &exportOpts, llvm::raw_ostream &errors) {
  for (auto &str : exportOpts) {
    llvm::StringRef exports = StoreString(str);
    size_t start = 0;
    size_t end = llvm::StringRef::npos;
    // def1;def2;...
    while (true) {
      end = exports.find_first_of(';', start);
      llvm::StringRef exportDef = exports.slice(start, end);

      // def: export1[[,export2,...]=internal]
      llvm::StringRef internalName = exportDef;
      size_t equals = exportDef.find_first_of('=');
      if (equals != llvm::StringRef::npos) {
        internalName = exportDef.substr(equals + 1);
        size_t exportStart = 0;
        while (true) {
          size_t comma = exportDef.find_first_of(',', exportStart);
          if (comma == llvm::StringRef::npos || comma > equals)
            break;
          if (exportStart < comma)
            Add(exportDef.slice(exportStart, comma), internalName);
          exportStart = comma + 1;
        }
        if (exportStart < equals)
          Add(exportDef.slice(exportStart, equals), internalName);
      } else {
        Add(internalName);
      }

      if (equals == 0 || internalName.empty()) {
        errors << "Invalid syntax for -exports: '" << exportDef
          << "'.  Syntax is: export1[[,export2,...]=internal][;...]";
        return false;
      }
      if (end == llvm::StringRef::npos)
        break;
      start = end + 1;
    }
  }
  return true;
}

void ExportMap::Add(llvm::StringRef exportName, llvm::StringRef internalName) {
  // Incoming strings may be escaped (because they originally come from arguments)
  // Unescape them here, if necessary
  if (exportName.startswith("\\")) {
    std::string str;
    llvm::raw_string_ostream os(str);
    PrintUnescapedString(exportName, os);
    exportName = StoreString(os.str());
  }
  if (internalName.startswith("\\")) {
    std::string str;
    llvm::raw_string_ostream os(str);
    PrintUnescapedString(internalName, os);
    internalName = StoreString(os.str());
  }

  if (internalName.empty())
    internalName = exportName;
  exportName = DemangleFunctionName(exportName);
  m_ExportMap[internalName].insert(exportName);
}

ExportMap::const_iterator ExportMap::GetExportsByName(llvm::StringRef Name) const {
  ExportMap::const_iterator it = m_ExportMap.find(Name);
  StringRef unmangled = DemangleFunctionName(Name);
  if (it == end()) {
    if (Name.startswith(ManglingPrefix)) {
      it = m_ExportMap.find(unmangled);
    }
    else if (Name.startswith(EntryPrefix)) {
      it = m_ExportMap.find(Name.substr(strlen(EntryPrefix)));
    }
  }
  return it;
}

bool ExportMap::IsExported(llvm::StringRef original) const {
  if (m_ExportMap.empty())
    return true;
  return GetExportsByName(original) != end();
}

void ExportMap::BeginProcessing() {
  m_ExportNames.clear();
  m_NameCollisions.clear();
  m_UnusedExports.clear();
  for (auto &it : m_ExportMap) {
    m_UnusedExports.emplace(it.getKey());
  }
}

bool ExportMap::ProcessFunction(llvm::Function *F, bool collisionAvoidanceRenaming) {
  // Skip if already added.  This can happen due to patch constant functions.
  if (m_RenameMap.find(F) != m_RenameMap.end())
    return true;

  StringRef originalName = F->getName();
  StringRef unmangled = DemangleFunctionName(originalName);
  auto it = GetExportsByName(F->getName());

  // Early out if not exported, and do optional collision avoidance
  if (it == end()) {
    F->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    if (collisionAvoidanceRenaming) {
      std::string internalName = (Twine("internal.") + unmangled).str();
      internalName = dxilutil::ReplaceFunctionName(originalName, internalName);
      F->setName(internalName);
    }
    return false;
  }

  F->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

  // Add entry to m_RenameMap:
  auto &renames = m_RenameMap[F];
  const llvm::StringSet<> &exportRenames = it->getValue();
  llvm::StringRef internalName = it->getKey();

  // mark export used
  UseExport(internalName);

  // Add identity first
  auto itIdentity = exportRenames.find(unmangled);
  if (exportRenames.empty() || itIdentity != exportRenames.end()) {
    if (exportRenames.size() > 1)
      renames.insert(originalName);
    ExportName(originalName);
  } else if (collisionAvoidanceRenaming) {
    // do optional collision avoidance for exports being renamed
    std::string tempName = (Twine("temp.") + unmangled).str();
    tempName = dxilutil::ReplaceFunctionName(originalName, tempName);
    F->setName(tempName);
  }

  for (auto itName = exportRenames.begin(); itName != exportRenames.end(); itName++) {
    // Now add actual renames
    if (itName != itIdentity) {
      StringRef newName = StoreString(dxilutil::ReplaceFunctionName(F->getName(), itName->getKey()));
      renames.insert(newName);
      ExportName(newName);
    }
  }

  return true;
}

void ExportMap::RegisterExportedFunction(llvm::Function *F) {
  // Skip if already added
  if (m_RenameMap.find(F) != m_RenameMap.end())
    return;
  F->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  NameSet &renames = m_RenameMap[F];
  (void)(renames);  // Don't actually add anything
  ExportName(F->getName());
}

void ExportMap::UseExport(llvm::StringRef internalName) {
  auto it = m_UnusedExports.find(internalName);
  if (it != m_UnusedExports.end())
    m_UnusedExports.erase(it);
}
void ExportMap::ExportName(llvm::StringRef exportName) {
  auto result = m_ExportNames.insert(exportName);
  if (!result.second) {
    // Already present, report collision
    m_NameCollisions.insert(exportName);
  }
}

bool ExportMap::EndProcessing() const {
  return m_UnusedExports.empty() && m_NameCollisions.empty();
}

llvm::StringRef ExportMap::StoreString(llvm::StringRef str) {
  return *m_StringStorage.insert(str).first;
}

} // dxilutil
} // hlsl
