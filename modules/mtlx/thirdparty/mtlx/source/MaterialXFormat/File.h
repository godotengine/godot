//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_FILE_H
#define MATERIALX_FILE_H

/// @file
/// Cross-platform support for file and search paths

#include <MaterialXFormat/Export.h>

#include <MaterialXCore/Util.h>

MATERIALX_NAMESPACE_BEGIN

class FilePath;
using FilePathVec = vector<FilePath>;

extern MX_FORMAT_API const string PATH_LIST_SEPARATOR;
extern MX_FORMAT_API const string MATERIALX_SEARCH_PATH_ENV_VAR;

/// @class FilePath
/// A generic file path, supporting both syntactic and file system operations.
class MX_FORMAT_API FilePath
{
  public:
    enum Type
    {
        TypeRelative = 0,
        TypeAbsolute = 1,
        TypeNetwork = 2
    };

    enum Format
    {
        FormatWindows = 0,
        FormatPosix = 1,
#if defined(_WIN32)
        FormatNative = FormatWindows
#else
        FormatNative = FormatPosix
#endif
    };

  public:
    FilePath() :
        _type(TypeRelative)
    {
    }
    ~FilePath() { }

    bool operator==(const FilePath& rhs) const
    {
        return _vec == rhs._vec &&
               _type == rhs._type;
    }
    bool operator!=(const FilePath& rhs) const
    {
        return !(*this == rhs);
    }

    /// @name Syntactic Operations
    /// @{

    /// Construct a path from a standard string.
    FilePath(const string& str)
    {
        assign(str);
    }

    /// Construct a path from a C-style string.
    FilePath(const char* str)
    {
        assign(str ? string(str) : EMPTY_STRING);
    }

    /// Convert a path to a standard string.
    operator string() const
    {
        return asString();
    }

    /// Assign a path from a standard string.
    void assign(const string& str);

    /// Return this path as a standard string with the given format.
    string asString(Format format = FormatNative) const;

    /// Return true if the given path is empty.
    bool isEmpty() const
    {
        return _vec.empty();
    }

    /// Return true if the given path is absolute.
    bool isAbsolute() const
    {
        return _type != TypeRelative;
    }

    /// Return the base name of the given path, with leading directory
    /// information removed.
    const string& getBaseName() const
    {
        if (isEmpty())
        {
            return EMPTY_STRING;
        }
        return _vec[_vec.size() - 1];
    }

    /// Return the parent directory of the given path, if any.  If no
    /// parent directory is present, then the empty path is returned.
    FilePath getParentPath() const
    {
        FilePath parent(*this);
        if (!parent.isEmpty())
        {
            parent._vec.pop_back();
        }
        return parent;
    }

    /// Return the file extension of the given path.
    string getExtension() const
    {
        const string& baseName = getBaseName();
        size_t i = baseName.rfind('.');
        return i != string::npos ? baseName.substr(i + 1) : EMPTY_STRING;
    }

    /// Add a file extension to the given path.
    void addExtension(const string& ext)
    {
        assign(asString() + "." + ext);
    }

    /// Remove the file extension, if any, from the given path.
    void removeExtension()
    {
        if (!isEmpty())
        {
            string& baseName = _vec[_vec.size() - 1];
            size_t i = baseName.rfind('.');
            if (i != string::npos)
            {
                baseName = baseName.substr(0, i);
            }
        }
    }

    /// Concatenate two paths with a directory separator, returning the
    /// combined path.
    FilePath operator/(const FilePath& rhs) const;

    /// Return the number of strings in the path.
    size_t size() const
    {
        return _vec.size();
    }

    /// Return the string at the given index.
    string operator[](size_t index)
    {
        return _vec[index];
    }

    /// Return the const string at the given index.
    const string& operator[](size_t index) const
    {
        return _vec[index];
    }

    /// Return a normalized version of the given path, collapsing current path and
    /// parent path references so that 'a/./b' and 'c/../d/../a/b' become 'a/b'.
    FilePath getNormalized() const;

    /// @}
    /// @name File System Operations
    /// @{

    /// Return true if the given path exists on the file system.
    bool exists() const;

    /// Return true if the given path is a directory on the file system.
    bool isDirectory() const;

    /// Return a vector of all files in the given directory with the given extension.
    FilePathVec getFilesInDirectory(const string& extension) const;

    /// Return a vector of all directories at or beneath the given path.
    FilePathVec getSubDirectories() const;

    /// Create a directory on the file system at the given path.
    void createDirectory() const;

    /// Set the current working directory of the file system.
    bool setCurrentPath();

    /// @}

    /// Return the current working directory of the file system.
    static FilePath getCurrentPath();

    /// Return the directory containing the executable module.
    static FilePath getModulePath();

  private:
    StringVec _vec;
    Type _type;
};

/// @class FileSearchPath
/// A sequence of file paths, which may be queried to find the first instance
/// of a given filename on the file system.
class MX_FORMAT_API FileSearchPath
{
  public:
    using Iterator = FilePathVec::iterator;
    using ConstIterator = FilePathVec::const_iterator;

  public:
    FileSearchPath() = default;

    /// Construct a search path from a string.
    /// @param searchPath A string containing a sequence of file paths joined
    ///    by separator characters.
    /// @param sep The set of separator characters used in the search path.
    ///    Defaults to the PATH_LIST_SEPARATOR character.
    FileSearchPath(const string& searchPath, const string& sep = PATH_LIST_SEPARATOR)
    {
        for (const string& path : splitString(searchPath, sep))
        {
            if (!path.empty())
            {
                append(FilePath(path));
            }
        }
    }

    /// Convert this sequence to a string using the given separator.
    string asString(const string& sep = PATH_LIST_SEPARATOR) const
    {
        string str;
        for (size_t i = 0; i < _paths.size(); i++)
        {
            str += _paths[i];
            if (i + 1 < _paths.size())
            {
                str += sep;
            }
        }
        return str;
    }

    /// Append the given path to the sequence.
    void append(const FilePath& path)
    {
        _paths.push_back(path);
    }

    /// Append the given search path to the sequence.
    void append(const FileSearchPath& searchPath)
    {
        for (const FilePath& path : searchPath)
        {
            _paths.push_back(path);
        }
    }

    /// Prepend the given path to the sequence.
    void prepend(const FilePath& path)
    {
        _paths.insert(_paths.begin(), path);
    }

    /// Clear all paths from the sequence.
    void clear()
    {
        _paths.clear();
    }

    /// Return the number of paths in the sequence.
    size_t size() const
    {
        return _paths.size();
    }

    /// Return true if the search path is empty.
    bool isEmpty() const
    {
        return _paths.empty();
    }

    /// Return the path at the given index.
    FilePath& operator[](size_t index)
    {
        return _paths[index];
    }

    /// Return the const path at the given index.
    const FilePath& operator[](size_t index) const
    {
        return _paths[index];
    }

    /// Given an input filename, iterate through each path in this sequence,
    /// returning the first combined path found on the file system.
    /// On success, the combined path is returned; otherwise the original
    /// filename is returned unmodified.
    FilePath find(const FilePath& filename) const
    {
        if (_paths.empty() || filename.isEmpty())
        {
            return filename;
        }
        if (!filename.isAbsolute())
        {
            for (const FilePath& path : _paths)
            {
                FilePath combined = path / filename;
                if (combined.exists())
                {
                    return combined;
                }
            }
        }
        return filename;
    }

    /// @name Iterators
    /// @{

    Iterator begin() { return _paths.begin(); }
    ConstIterator begin() const { return _paths.begin(); }

    Iterator end() { return _paths.end(); }
    ConstIterator end() const { return _paths.end(); }

    /// @}

  private:
    FilePathVec _paths;
};

/// Return a FileSearchPath object from search path environment variable.
MX_FORMAT_API FileSearchPath getEnvironmentPath(const string& sep = PATH_LIST_SEPARATOR);

MATERIALX_NAMESPACE_END

#endif
