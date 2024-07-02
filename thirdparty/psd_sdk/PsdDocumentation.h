// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#error Documentation only. Do not include this file.


/// \mainpage PSD Library SDK
/// 
/// Thank you for downloading the PSD Library SDK!
///
///
/// \par Getting started
///
/// In order to gain an overview over what's available in the SDK, you should start reading the "Modules" page on the left-hand side
/// of this documentation. The most interesting modules are going to be "Interfaces", "Sections", and "Parser". Furthermore, it is
/// probably best to open the provided sample code to get a first glimpse of how to use the SDK.
///
/// All the functionality in the SDK is grouped into modules, and the "Modules" page provides an
/// overview of each. Browsing individual modules will give a detailed description of its purpose, and documents all the
/// classes, namespaces, functions, and macros which are part of it.
///
/// A list of all individual namespaces and classes is also available by browsing the "Namespaces" and "Classes" list on the left-hand
/// side, respectively.
///
/// Additionally, all files which make up the SDK documentation are available by clicking on "Files".
///
///
/// \par Linking with the code
///
/// In case you want to link with pre-built binaries, the SDK ships with binary libraries for Visual Studio 2008, 2010, 2012, 2013,
/// and 2015 in several different flavors: debug & release, statically & dynamically linked CRT, and 32-bit & 64-bit.
///
/// The binary libraries are located in the <tt>bin</tt> folder.
///
///
/// \par Building the code
///
/// If you'd rather build the code from scratch to setup compiler options to your liking, all source code files which are part of 
/// this particular SDK can be found in the <tt>src</tt> directory of the SDK root path.
/// Solution and project files for Visual Studio 2008, 2010, 2012, 2013, and 2015 can be found in the <tt>build</tt> directory.
/// The code can be built by opening the <tt>Psd.sln</tt> in the directory corresponding to the Visual Studio version, and rebuilding
/// the solution.
///
///
/// \par Dropping the code into existing projects
///
/// In case you want to drop the provided source code into one of your existing projects right away, don't worry. Building the
/// library from scratch is as easy as adding all source files (.h, .c & .cpp) from the <tt>src</tt> directory to your existing
/// project. The source does not need any external #defines or specific #include directories to be set up, and compiles out of the box
/// under Visual Studio 2008 SP1, 2010 SP1, 2012 Update 4, 2013 Update 4, 2015, GCC 5.2.0, and Clang 3.6.2, so you should be good to go.
///
///
/// \par Sample program & source code
///
/// In addition to the source code, binaries, and documentation, the SDK also ships with a small sample program.
/// The sample program demonstrates the different functions of the SDK, and shows how those can be utilized together. It loads a
/// .PSD file located in the <tt>bin</tt> directory, and spits out individual .TGA files.
///
/// Note that the PSD Library source code is only available in the <b>Full SDK</b>, not the <b>Evaluation SDK</b>.
///
///
/// \par Evaluation
///
/// We would like to get in contact with you regarding future use of the SDK after an evaluation period of 14 days.
/// During that period, please feel free to ask questions, give feedback and report bugs by writing to evaluation@molecular-matters.com.


/// \defgroup ImageUtil
/// \brief Image manipulation routines needed by the different parsers.
/// \details This module contains functions for decompressing RLE-compressed images, interleaving planar RGB data, and copying layer
/// data to a canvas.


/// \defgroup Interfaces
/// \brief Contains abstract class interfaces that provide hooks for memory management and file I/O.


/// \defgroup Allocators
/// \brief Contains allocator interfaces and implementations that provide hooks for customized memory management.


/// \defgroup Files
/// \brief Contains file interfaces and implementations that provide hooks for customized file I/O.


/// \defgroup Parser
/// \brief Provides functions for parsing the different sections of a .PSD file.
/// \details The functions contained in this module deal with parsing and extracting data from the different sections of
/// a .PSD file. Each parser is able to parse only one specific section of a file, and offers one function for creating a new
/// instance of parsed data, and one function for carrying out proper deletion and cleanup.


/// \defgroup Platform
/// \brief Provides abstractions for compiler-, platform- and OS-specific features.
/// \details All the macros, functions and classes contained in this module abstract either a compiler-, platform- or
/// OS-specific feature.


/// \defgroup Sections
/// \brief Contains structures pertaining to sections found in a .PSD file.


/// \defgroup Types
/// \brief Contains structures, enumerations, and other types pertaining to pieces of information parsed from a .PSD file.


/// \defgroup Util
/// \brief Convenience classes and functions dealing with bit manipulation, endian conversion, casts, etc.
/// \details The classes and functions available in this module are first and foremost utility classes, mostly on top of concrete
/// implementations, using only what is available through the public interface.


/// \defgroup Exporter
/// \brief Provides functions for exporting sections and layers into a .PSD file.
