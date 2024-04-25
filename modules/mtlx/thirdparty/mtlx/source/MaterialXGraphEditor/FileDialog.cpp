//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGraphEditor/FileDialog.h>

#include <MaterialXCore/Exception.h>

#if defined(_WIN32)
    #include <windows.h>
#endif

FileDialog::FileDialog(int flags) :
    _flags(flags)
{
}

void FileDialog::setTitle(const std::string& title)
{
    _title = title;
}

void FileDialog::setTypeFilters(const mx::StringVec& typeFilters)
{
    _filetypes.clear();

    for (auto typefilter : typeFilters)
    {
        std::string minusExt = typefilter.substr(1, typefilter.size() - 1);
        std::pair<std::string, std::string> filterPair = { minusExt, minusExt };
        _filetypes.push_back(filterPair);
    }
}

void FileDialog::open()
{
    clearSelected();
    _openFlag = true;
}

bool FileDialog::isOpened()
{
    return _isOpened;
}

bool FileDialog::hasSelected()
{
    return !_selectedFilenames.empty();
}

mx::FilePath FileDialog::getSelected()
{
    return _selectedFilenames.empty() ? mx::FilePath() : _selectedFilenames[0];
}

void FileDialog::clearSelected()
{
    _selectedFilenames.clear();
}

void FileDialog::display()
{
    // Only call the dialog if it's not already displayed
    if (!_openFlag || _isOpened)
    {
        return;
    }
    _openFlag = false;

    // Check if we want to save or open
    bool save = !(_flags & SelectDirectory) && (_flags & EnterNewFilename);

    mx::StringVec result = launchFileDialog(_filetypes, save, false);
    std::string path = result.empty() ? "" : result.front();
    if (!path.empty())
    {
        _selectedFilenames.push_back(path);
    }

    _isOpened = false;
}

#if !defined(__APPLE__)
mx::StringVec launchFileDialog(const std::vector<std::pair<std::string, std::string>>& filetypes, bool save, bool multiple)
{
    static const int FILE_DIALOG_MAX_BUFFER = 16384;
    if (save && multiple)
    {
        throw mx::Exception("save and multiple must not both be true.");
    }

    #if defined(_WIN32)
    OPENFILENAME ofn;
    ZeroMemory(&ofn, sizeof(OPENFILENAME));
    ofn.lStructSize = sizeof(OPENFILENAME);
    char tmp[FILE_DIALOG_MAX_BUFFER];
    ofn.lpstrFile = tmp;
    ZeroMemory(tmp, FILE_DIALOG_MAX_BUFFER);
    ofn.nMaxFile = FILE_DIALOG_MAX_BUFFER;
    ofn.nFilterIndex = 1;

    std::string filter;

    if (!save && filetypes.size() > 1)
    {
        filter.append("Supported file types (");
        for (size_t i = 0; i < filetypes.size(); ++i)
        {
            filter.append("*.");
            filter.append(filetypes[i].first);
            if (i + 1 < filetypes.size())
                filter.append(";");
        }
        filter.append(")");
        filter.push_back('\0');
        for (size_t i = 0; i < filetypes.size(); ++i)
        {
            filter.append("*.");
            filter.append(filetypes[i].first);
            if (i + 1 < filetypes.size())
                filter.append(";");
        }
        filter.push_back('\0');
    }
    for (auto pair : filetypes)
    {
        filter.append(pair.second);
        filter.append(" (*.");
        filter.append(pair.first);
        filter.append(")");
        filter.push_back('\0');
        filter.append("*.");
        filter.append(pair.first);
        filter.push_back('\0');
    }
    filter.push_back('\0');
    ofn.lpstrFilter = filter.data();

    if (save)
    {
        ofn.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
        if (GetSaveFileNameA(&ofn) == FALSE)
            return {};
    }
    else
    {
        ofn.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        if (multiple)
            ofn.Flags |= OFN_ALLOWMULTISELECT;
        if (GetOpenFileNameA(&ofn) == FALSE)
            return {};
    }

    size_t i = 0;
    mx::StringVec result;
    while (tmp[i] != '\0')
    {
        result.emplace_back(&tmp[i]);
        i += result.back().size() + 1;
    }

    if (result.size() > 1)
    {
        for (i = 1; i < result.size(); ++i)
        {
            result[i] = result[0] + "\\" + result[i];
        }
        result.erase(begin(result));
    }

    if (save && ofn.nFilterIndex > 0)
    {
        auto ext = filetypes[ofn.nFilterIndex - 1].first;
        if (ext != "*")
        {
            ext.insert(0, ".");

            auto& name = result.front();
            if (name.size() <= ext.size() ||
                name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
            {
                name.append(ext);
            }
        }
    }

    return result;
    #else
    char buffer[FILE_DIALOG_MAX_BUFFER];
    buffer[0] = '\0';

    std::string cmd = "zenity --file-selection ";
    // The safest separator for multiple selected paths is /, since / can never occur
    // in file names. Only where two paths are concatenated will there be two / following
    // each other.
    if (multiple)
        cmd += "--multiple --separator=\"/\" ";
    if (save)
        cmd += "--save ";
    cmd += "--file-filter=\"";
    for (auto pair : filetypes)
        cmd += "\"*." + pair.first + "\" ";
    cmd += "\"";
    FILE* output = popen(cmd.c_str(), "r");
    if (output == nullptr)
        throw mx::Exception("popen() failed -- could not launch zenity!");
    while (fgets(buffer, FILE_DIALOG_MAX_BUFFER, output) != NULL)
        ;
    pclose(output);
    std::string paths(buffer);
    paths.erase(std::remove(paths.begin(), paths.end(), '\n'), paths.end());

    mx::StringVec result;
    while (!paths.empty())
    {
        size_t end = paths.find("//");
        if (end == std::string::npos)
        {
            result.emplace_back(paths);
            paths = "";
        }
        else
        {
            result.emplace_back(paths.substr(0, end));
            paths = paths.substr(end + 1);
        }
    }

    return result;
    #endif
}
#endif
