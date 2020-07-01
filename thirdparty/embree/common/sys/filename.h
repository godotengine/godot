// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "platform.h"

namespace embree
{
  /*! Convenience class for handling file names and paths. */
  class FileName
  {
  public:

    /*! create an empty filename */
    FileName ();

    /*! create a valid filename from a string */
    FileName (const char* filename);

    /*! create a valid filename from a string */
    FileName (const std::string& filename);
    
    /*! returns path to home folder */
    static FileName homeFolder();

    /*! returns path to executable */
    static FileName executableFolder();

    /*! auto convert into a string */
    operator std::string() const { return filename; }

    /*! returns a string of the filename */
    const std::string str() const { return filename; }

    /*! returns a c-string of the filename */
    const char* c_str() const { return filename.c_str(); }

    /*! returns the path of a filename */
    FileName path() const;

    /*! returns the file of a filename  */
    std::string base() const;

    /*! returns the base of a filename without extension */
    std::string name() const;

    /*! returns the file extension */
    std::string ext() const;

    /*! drops the file extension */
    FileName dropExt() const;

    /*! replaces the file extension */
    FileName setExt(const std::string& ext = "") const;

    /*! adds file extension */
    FileName addExt(const std::string& ext = "") const;

    /*! concatenates two filenames to this/other */
    FileName operator +( const FileName& other ) const;

    /*! concatenates two filenames to this/other */
    FileName operator +( const std::string& other ) const;

    /*! removes the base from a filename (if possible) */
    FileName operator -( const FileName& base ) const;

    /*! == operator */
    friend bool operator==(const FileName& a, const FileName& b);

    /*! != operator */
    friend bool operator!=(const FileName& a, const FileName& b);

    /*! output operator */
    friend std::ostream& operator<<(std::ostream& cout, const FileName& filename);
   
  private:
    std::string filename;
  };
}
