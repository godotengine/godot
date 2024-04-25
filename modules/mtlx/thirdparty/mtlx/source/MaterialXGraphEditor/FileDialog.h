//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_FILEDIALOG_H
#define MATERIALX_FILEDIALOG_H

#include <MaterialXFormat/File.h>

namespace mx = MaterialX;

// A native file browser class, based on the implementation in NanoGUI.
class FileDialog
{
  public:
    enum Flags
    {
        SelectDirectory = 1 << 0,  // select directory instead of regular file
        EnterNewFilename = 1 << 1, // allow user to enter new filename when selecting regular file
        NoTitleBar = 1 << 2,       // hide window title bar
    };

  public:
    FileDialog(int flags = 0);
    void setTitle(const std::string& title);
    void setTypeFilters(const mx::StringVec& typeFilters);
    void open();
    bool isOpened();
    void display();
    bool hasSelected();
    mx::FilePath getSelected();
    void clearSelected();

  private:
    int _flags;
    std::string _title;
    bool _openFlag = false;
    bool _isOpened = false;
    mx::FilePathVec _selectedFilenames;
    std::vector<std::pair<std::string, std::string>> _filetypes;
};

mx::StringVec launchFileDialog(const std::vector<std::pair<std::string, std::string>>& filetypes, bool save, bool multiple);

#endif
