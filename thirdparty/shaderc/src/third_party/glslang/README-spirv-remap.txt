
VERSION
--------------------------------------------------------------------------------
spirv-remap 0.97

INTRO:
--------------------------------------------------------------------------------
spirv-remap is a utility to improve compression of SPIR-V binary files via
entropy reduction, plus optional stripping of debug information and
load/store optimization.  It transforms SPIR-V to SPIR-V, remapping IDs.  The
resulting modules have an increased ID range (IDs are not as tightly packed
around zero), but will compress better when multiple modules are compressed
together, since compressor's dictionary can find better cross module
commonality.

Remapping is accomplished via canonicalization.  Thus, modules can be
compressed one at a time with no loss of quality relative to operating on
many modules at once.  The command line tool operates on multiple modules
only in the trivial repetition sense, for ease of use.  The remapper API
only accepts a single module at a time.

There are two modes of use: command line, and a C++11 API.  Both are
described below.

spirv-remap is currently in an alpha state.  Although there are no known
remapping defects, it has only been exercised on one real world game shader
workload.


FEEDBACK
--------------------------------------------------------------------------------
Report defects, enhancements requests, code improvements, etc to:
   spvremapper@lunarg.com


COMMAND LINE USAGE:
--------------------------------------------------------------------------------
Examples are given with a verbosity of one (-v), but more verbosity can be
had via -vv, -vvv, etc, or an integer parameter to --verbose, such as
"--verbose 4".  With no verbosity, the command is silent and returns 0 on
success, and a positive integer error on failure.

Pre-built binaries for several OSs are available.  Examples presented are
for Linux.  Command line arguments can be provided in any order.

1. Basic ID remapping

Perform ID remapping on all shaders in "*.spv", writing new files with
the same basenames to /tmp/out_dir.

  spirv-remap -v --map all --input *.spv --output /tmp/out_dir

2. Perform all possible size reductions

  spirv-remap-linux-64 -v --do-everything --input *.spv --output /tmp/out_dir

Note that --do-everything is a synonym for:

  --map all --dce all --opt all --strip all

API USAGE:
--------------------------------------------------------------------------------

The public interface to the remapper is defined in SPIRV/SPVRemapper.h as follows:

namespace spv {

class spirvbin_t
{
public:
   enum Options { ... };
   spirvbin_t(int verbose = 0);  // construct

   // remap an existing binary in memory
   void remap(std::vector<std::uint32_t>& spv, std::uint32_t opts = DO_EVERYTHING);

   // Type for error/log handler functions
   typedef std::function<void(const std::string&)> errorfn_t;
   typedef std::function<void(const std::string&)> logfn_t;

   // Register error/log handling functions (can be c/c++ fn, lambda fn, or functor)
   static void registerErrorHandler(errorfn_t handler) { errorHandler = handler; }
   static void registerLogHandler(logfn_t handler)     { logHandler   = handler; }
};

} // namespace spv

The class definition is in SPVRemapper.cpp.

remap() accepts an std::vector of SPIR-V words, modifies them per the
request given in 'opts', and leaves the 'spv' container with the result.
It is safe to instantiate one spirvbin_t per thread and process a different
SPIR-V in each.

The "opts" parameter to remap() accepts a bit mask of desired remapping
options.  See REMAPPING AND OPTIMIZATION OPTIONS.

On error, the function supplied to registerErrorHandler() will be invoked.
This can be a standard C/C++ function, a lambda function, or a functor.
The default handler simply calls exit(5); The error handler is a static
member, so need only be set up once, not once per spirvbin_t instance.

Log messages are supplied to registerLogHandler().  By default, log
messages are eaten silently.  The log handler is also a static member.

BUILD DEPENDENCIES:
--------------------------------------------------------------------------------
 1. C++11 compatible compiler
 2. cmake
 3. glslang


BUILDING
--------------------------------------------------------------------------------
The standalone remapper is built along side glslangValidator through its
normal build process.


REMAPPING AND OPTIMIZATION OPTIONS
--------------------------------------------------------------------------------
API:
   These are bits defined under spv::spirvbin_t::, and can be
   bitwise or-ed together as desired.

   MAP_TYPES      = canonicalize type IDs
   MAP_NAMES      = canonicalize named data
   MAP_FUNCS      = canonicalize function bodies
   DCE_FUNCS      = remove dead functions
   DCE_VARS       = remove dead variables
   DCE_TYPES      = remove dead types
   OPT_LOADSTORE  = optimize unneeded load/stores
   MAP_ALL        = (MAP_TYPES | MAP_NAMES | MAP_FUNCS)
   DCE_ALL        = (DCE_FUNCS | DCE_VARS | DCE_TYPES)
   OPT_ALL        = (OPT_LOADSTORE)
   ALL_BUT_STRIP  = (MAP_ALL | DCE_ALL | OPT_ALL)
   DO_EVERYTHING  = (STRIP | ALL_BUT_STRIP)

