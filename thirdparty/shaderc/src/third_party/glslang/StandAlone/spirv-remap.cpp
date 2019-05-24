//
// Copyright (C) 2015 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>

#include "../SPIRV/SPVRemapper.h"

namespace {

    typedef unsigned int SpvWord;

    // Poor man's basename: given a complete path, return file portion.
    // E.g:
    //      Linux:  /foo/bar/test  -> test
    //      Win:   c:\foo\bar\test -> test
    // It's not very efficient, but that doesn't matter for our minimal-duty use.
    // Using boost::filesystem would be better in many ways, but want to avoid that dependency.

    // OS dependent path separator (avoiding boost::filesystem dependency)
#if defined(_WIN32)
    char path_sep_char() { return '\\'; }
#else
    char path_sep_char() { return '/';  }
#endif

    std::string basename(const std::string filename)
    {
        const size_t sepLoc = filename.find_last_of(path_sep_char());

        return (sepLoc == filename.npos) ? filename : filename.substr(sepLoc+1);
    }

    void errHandler(const std::string& str) {
        std::cout << str << std::endl;
        exit(5);
    }

    void logHandler(const std::string& str) {
        std::cout << str << std::endl;
    }

    // Read word stream from disk
    void read(std::vector<SpvWord>& spv, const std::string& inFilename, int verbosity)
    {
        std::ifstream fp;

        if (verbosity > 0)
            logHandler(std::string("  reading: ") + inFilename);

        spv.clear();
        fp.open(inFilename, std::fstream::in | std::fstream::binary);

        if (fp.fail())
            errHandler("error opening file for read: ");

        // Reserve space (for efficiency, not for correctness)
        fp.seekg(0, fp.end);
        spv.reserve(size_t(fp.tellg()) / sizeof(SpvWord));
        fp.seekg(0, fp.beg);

        while (!fp.eof()) {
            SpvWord inWord;
            fp.read((char *)&inWord, sizeof(inWord));

            if (!fp.eof()) {
                spv.push_back(inWord);
                if (fp.fail())
                    errHandler(std::string("error reading file: ") + inFilename);
            }
        }
    }

    void write(std::vector<SpvWord>& spv, const std::string& outFile, int verbosity)
    {
        if (outFile.empty())
            errHandler("missing output filename.");

        std::ofstream fp;

        if (verbosity > 0)
            logHandler(std::string("  writing: ") + outFile);

        fp.open(outFile, std::fstream::out | std::fstream::binary);

        if (fp.fail())
            errHandler(std::string("error opening file for write: ") + outFile);

        for (auto it = spv.cbegin(); it != spv.cend(); ++it) {
            SpvWord word = *it;
            fp.write((char *)&word, sizeof(word));
            if (fp.fail())
                errHandler(std::string("error writing file: ") + outFile);
        }

        // file is closed by destructor
    }

    // Print helpful usage message to stdout, and exit
    void usage(const char* const name, const char* const msg = 0)
    {
        if (msg)
            std::cout << msg << std::endl << std::endl;

        std::cout << "Usage: " << std::endl;

        std::cout << "  " << basename(name)
            << " [-v[v[...]] | --verbose [int]]"
            << " [--map (all|types|names|funcs)]"
            << " [--dce (all|types|funcs)]"
            << " [--opt (all|loadstore)]"
            << " [--strip-all | --strip all | -s]"
            << " [--do-everything]"
            << " --input | -i file1 [file2...] --output|-o DESTDIR"
            << std::endl;

        std::cout << "  " << basename(name) << " [--version | -V]" << std::endl;
        std::cout << "  " << basename(name) << " [--help | -?]" << std::endl;

        exit(5);
    }

    // grind through each SPIR in turn
    void execute(const std::vector<std::string>& inputFile, const std::string& outputDir,
        int opts, int verbosity)
    {
        for (auto it = inputFile.cbegin(); it != inputFile.cend(); ++it) {
            const std::string &filename = *it;
            std::vector<SpvWord> spv;
            read(spv, filename, verbosity);
            spv::spirvbin_t(verbosity).remap(spv, opts);

            const std::string outfile = outputDir + path_sep_char() + basename(filename);

            write(spv, outfile, verbosity);
        }

        if (verbosity > 0)
            std::cout << "Done: " << inputFile.size() << " file(s) processed" << std::endl;
    }

    // Parse command line options
    void parseCmdLine(int argc, char** argv, std::vector<std::string>& inputFile,
        std::string& outputDir,
        int& options,
        int& verbosity)
    {
        if (argc < 2)
            usage(argv[0]);

        verbosity  = 0;
        options    = spv::spirvbin_t::NONE;

        // Parse command line.
        // boost::program_options would be quite a bit nicer, but we don't want to
        // introduce a dependency on boost.
        for (int a=1; a<argc; ) {
            const std::string arg = argv[a];

            if (arg == "--output" || arg == "-o") {
                // Output directory
                if (++a >= argc)
                    usage(argv[0], "--output requires an argument");
                if (!outputDir.empty())
                    usage(argv[0], "--output can be provided only once");

                outputDir = argv[a++];

                // Remove trailing directory separator characters
                while (!outputDir.empty() && outputDir.back() == path_sep_char())
                    outputDir.pop_back();

            }
            else if (arg == "-vv")     { verbosity = 2; ++a; } // verbosity shortcuts
            else if (arg == "-vvv")    { verbosity = 3; ++a; } // ...
            else if (arg == "-vvvv")   { verbosity = 4; ++a; } // ...
            else if (arg == "-vvvvv")  { verbosity = 5; ++a; } // ...

            else if (arg == "--verbose" || arg == "-v") {
                ++a;
                verbosity = 1;

                if (a < argc) {
                    char* end_ptr = 0;
                    int verb = ::strtol(argv[a], &end_ptr, 10);
                    // If we have not read to the end of the string or
                    // the string contained no elements, then we do not want to
                    // store the value.
                    if (*end_ptr == '\0' && end_ptr != argv[a]) {
                        verbosity = verb;
                        ++a;
                    }
                }
            }
            else if (arg == "--version" || arg == "-V") {
                std::cout << basename(argv[0]) << " version 0.97 " << __DATE__ << " " << __TIME__ << std::endl;
                exit(0);
            } else if (arg == "--input" || arg == "-i") {
                // Collect input files
                for (++a; a < argc && argv[a][0] != '-'; ++a)
                    inputFile.push_back(argv[a]);
            } else if (arg == "--do-everything") {
                ++a;
                options = options | spv::spirvbin_t::DO_EVERYTHING;
            } else if (arg == "--strip-all" || arg == "-s") {
                ++a;
                options = options | spv::spirvbin_t::STRIP;
            } else if (arg == "--strip") {
                ++a;
                if (strncmp(argv[a], "all", 3) == 0) {
                    options = options | spv::spirvbin_t::STRIP;
                    ++a;
                }
            } else if (arg == "--dce") {
                // Parse comma (or colon, etc) separated list of things to dce
                ++a;
                for (const char* c = argv[a]; *c; ++c) {
                    if (strncmp(c, "all", 3) == 0) {
                        options = (options | spv::spirvbin_t::DCE_ALL);
                        c += 3;
                    } else if (strncmp(c, "*", 1) == 0) {
                        options = (options | spv::spirvbin_t::DCE_ALL);
                        c += 1;
                    } else if (strncmp(c, "funcs", 5) == 0) {
                        options = (options | spv::spirvbin_t::DCE_FUNCS);
                        c += 5;
                    } else if (strncmp(c, "types", 5) == 0) {
                        options = (options | spv::spirvbin_t::DCE_TYPES);
                        c += 5;
                    }
                }
                ++a;
            } else if (arg == "--map") {
                // Parse comma (or colon, etc) separated list of things to map
                ++a;
                for (const char* c = argv[a]; *c; ++c) {
                    if (strncmp(c, "all", 3) == 0) {
                        options = (options | spv::spirvbin_t::MAP_ALL);
                        c += 3;
                    } else if (strncmp(c, "*", 1) == 0) {
                        options = (options | spv::spirvbin_t::MAP_ALL);
                        c += 1;
                    } else if (strncmp(c, "types", 5) == 0) {
                        options = (options | spv::spirvbin_t::MAP_TYPES);
                        c += 5;
                    } else if (strncmp(c, "names", 5) == 0) {
                        options = (options | spv::spirvbin_t::MAP_NAMES);
                        c += 5;
                    } else if (strncmp(c, "funcs", 5) == 0) {
                        options = (options | spv::spirvbin_t::MAP_FUNCS);
                        c += 5;
                    }
                }
                ++a;
            } else if (arg == "--opt") {
                ++a;
                for (const char* c = argv[a]; *c; ++c) {
                    if (strncmp(c, "all", 3) == 0) {
                        options = (options | spv::spirvbin_t::OPT_ALL);
                        c += 3;
                    } else if (strncmp(c, "*", 1) == 0) {
                        options = (options | spv::spirvbin_t::OPT_ALL);
                        c += 1;
                    } else if (strncmp(c, "loadstore", 9) == 0) {
                        options = (options | spv::spirvbin_t::OPT_LOADSTORE);
                        c += 9;
                    }
                }
                ++a;
            } else if (arg == "--help" || arg == "-?") {
                usage(argv[0]);
            } else {
                usage(argv[0], "Unknown command line option");
            }
        }
    }

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> inputFile;
    std::string              outputDir;
    int                      opts;
    int                      verbosity;

#ifdef use_cpp11
    // handle errors by exiting
    spv::spirvbin_t::registerErrorHandler(errHandler);

    // Log messages to std::cout
    spv::spirvbin_t::registerLogHandler(logHandler);
#endif

    if (argc < 2)
        usage(argv[0]);

    parseCmdLine(argc, argv, inputFile, outputDir, opts, verbosity);

    if (outputDir.empty())
        usage(argv[0], "Output directory required");

    std::string errmsg;

    // Main operations: read, remap, and write.
    execute(inputFile, outputDir, opts, verbosity);

    // If we get here, everything went OK!  Nothing more to be done.
}
