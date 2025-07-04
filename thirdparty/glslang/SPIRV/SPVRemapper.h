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

#ifndef SPIRVREMAPPER_H
#define SPIRVREMAPPER_H

#include <string>
#include <vector>
#include <cstdlib>
#include <exception>

#ifdef GLSLANG_IS_SHARED_LIBRARY
    #ifdef _WIN32
        #ifdef GLSLANG_EXPORTING
            #define GLSLANG_EXPORT __declspec(dllexport)
        #else
            #define GLSLANG_EXPORT __declspec(dllimport)
        #endif
    #elif __GNUC__ >= 4
        #define GLSLANG_EXPORT __attribute__((visibility("default")))
    #endif
#endif // GLSLANG_IS_SHARED_LIBRARY
#ifndef GLSLANG_EXPORT
#define GLSLANG_EXPORT
#endif

namespace spv {

class spirvbin_base_t
{
public:
   enum Options {
      NONE          = 0,
      STRIP         = (1<<0),
      MAP_TYPES     = (1<<1),
      MAP_NAMES     = (1<<2),
      MAP_FUNCS     = (1<<3),
      DCE_FUNCS     = (1<<4),
      DCE_VARS      = (1<<5),
      DCE_TYPES     = (1<<6),
      OPT_LOADSTORE = (1<<7),
      OPT_FWD_LS    = (1<<8), // EXPERIMENTAL: PRODUCES INVALID SCHEMA-0 SPIRV
      MAP_ALL       = (MAP_TYPES | MAP_NAMES | MAP_FUNCS),
      DCE_ALL       = (DCE_FUNCS | DCE_VARS | DCE_TYPES),
      OPT_ALL       = (OPT_LOADSTORE),

      ALL_BUT_STRIP = (MAP_ALL | DCE_ALL | OPT_ALL),
      DO_EVERYTHING = (STRIP | ALL_BUT_STRIP)
   };
};

} // namespace SPV

#include <functional>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <cassert>

#include "spirv.hpp"

namespace spv {

static inline constexpr Id NoResult = 0;

// class to hold SPIR-V binary data for remapping, DCE, and debug stripping
class GLSLANG_EXPORT spirvbin_t : public spirvbin_base_t
{
public:
   spirvbin_t(int verbose = 0) : entryPoint(spv::NoResult), largestNewId(0), verbose(verbose), errorLatch(false)
   { }

   virtual ~spirvbin_t() { }

   // remap on an existing binary in memory
   void remap(std::vector<std::uint32_t>& spv, const std::vector<std::string>& whiteListStrings,
              std::uint32_t opts = DO_EVERYTHING);

   // remap on an existing binary in memory - legacy interface without white list
   void remap(std::vector<std::uint32_t>& spv, std::uint32_t opts = DO_EVERYTHING);

   // Type for error/log handler functions
   typedef std::function<void(const std::string&)> errorfn_t;
   typedef std::function<void(const std::string&)> logfn_t;

   // Register error/log handling functions (can be lambda fn / functor / etc)
   static void registerErrorHandler(errorfn_t handler) { errorHandler = handler; }
   static void registerLogHandler(logfn_t handler)     { logHandler   = handler; }

protected:
   // This can be overridden to provide other message behavior if needed
   virtual void msg(int minVerbosity, int indent, const std::string& txt) const;

private:
   // Local to global, or global to local ID map
   typedef std::unordered_map<spv::Id, spv::Id> idmap_t;
   typedef std::unordered_set<spv::Id>          idset_t;
   typedef std::unordered_map<spv::Id, int>     blockmap_t;

   void remap(std::uint32_t opts = DO_EVERYTHING);

   // Map of names to IDs
   typedef std::unordered_map<std::string, spv::Id> namemap_t;

   typedef std::uint32_t spirword_t;

   typedef std::pair<unsigned, unsigned> range_t;
   typedef std::function<void(spv::Id&)>                idfn_t;
   typedef std::function<bool(spv::Op, unsigned start)> instfn_t;

   // Special Values for ID map:
   static const spv::Id unmapped;     // unchanged from default value
   static const spv::Id unused;       // unused ID
   static const int     header_size;  // SPIR header = 5 words

   class id_iterator_t;

   // For mapping type entries between different shaders
   typedef std::vector<spirword_t>        typeentry_t;
   typedef std::map<spv::Id, typeentry_t> globaltypes_t;

   // A set that preserves position order, and a reverse map
   typedef std::set<int>                    posmap_t;
   typedef std::unordered_map<spv::Id, int> posmap_rev_t;

   // Maps and ID to the size of its base type, if known.
   typedef std::unordered_map<spv::Id, unsigned> typesize_map_t;

   // handle error
   void error(const std::string& txt) const { errorLatch = true; errorHandler(txt); }

   bool     isConstOp(spv::Op opCode)      const;
   bool     isTypeOp(spv::Op opCode)       const;
   bool     isStripOp(spv::Op opCode)      const;
   bool     isFlowCtrl(spv::Op opCode)     const;
   range_t  literalRange(spv::Op opCode)   const;
   range_t  typeRange(spv::Op opCode)      const;
   range_t  constRange(spv::Op opCode)     const;
   unsigned typeSizeInWords(spv::Id id)    const;
   unsigned idTypeSizeInWords(spv::Id id)  const;

   bool isStripOp(spv::Op opCode, unsigned start) const;

   spv::Id&        asId(unsigned word)                { return spv[word]; }
   const spv::Id&  asId(unsigned word)          const { return spv[word]; }
   spv::Op         asOpCode(unsigned word)      const { return opOpCode(spv[word]); }
   std::uint32_t   asOpCodeHash(unsigned word);
   spv::Decoration asDecoration(unsigned word)  const { return spv::Decoration(spv[word]); }
   unsigned        asWordCount(unsigned word)   const { return opWordCount(spv[word]); }
   spv::Id         asTypeConstId(unsigned word) const { return asId(word + (isTypeOp(asOpCode(word)) ? 1 : 2)); }
   unsigned        idPos(spv::Id id)            const;

   static unsigned opWordCount(spirword_t data) { return data >> spv::WordCountShift; }
   static spv::Op  opOpCode(spirword_t data)    { return spv::Op(data & spv::OpCodeMask); }

   // Header access & set methods
   spirword_t  magic()    const       { return spv[0]; } // return magic number
   spirword_t  bound()    const       { return spv[3]; } // return Id bound from header
   spirword_t  bound(spirword_t b)    { return spv[3] = b; }
   spirword_t  genmagic() const       { return spv[2]; } // generator magic
   spirword_t  genmagic(spirword_t m) { return spv[2] = m; }
   spirword_t  schemaNum() const      { return spv[4]; } // schema number from header

   // Mapping fns: get
   spv::Id     localId(spv::Id id) const { return idMapL[id]; }

   // Mapping fns: set
   inline spv::Id   localId(spv::Id id, spv::Id newId);
   void             countIds(spv::Id id);

   // Return next unused new local ID.
   // NOTE: boost::dynamic_bitset would be more efficient due to find_next(),
   // which std::vector<bool> doens't have.
   inline spv::Id   nextUnusedId(spv::Id id);

   void buildLocalMaps();
   std::string literalString(unsigned word) const; // Return literal as a std::string
   int literalStringWords(const std::string& str) const { return (int(str.size())+4)/4; }

   bool isNewIdMapped(spv::Id newId)   const { return isMapped(newId);            }
   bool isOldIdUnmapped(spv::Id oldId) const { return localId(oldId) == unmapped; }
   bool isOldIdUnused(spv::Id oldId)   const { return localId(oldId) == unused;   }
   bool isOldIdMapped(spv::Id oldId)   const { return !isOldIdUnused(oldId) && !isOldIdUnmapped(oldId); }
   bool isFunction(spv::Id oldId)      const { return fnPos.find(oldId) != fnPos.end(); }

   // bool    matchType(const globaltypes_t& globalTypes, spv::Id lt, spv::Id gt) const;
   // spv::Id findType(const globaltypes_t& globalTypes, spv::Id lt) const;
   std::uint32_t hashType(unsigned typeStart) const;

   spirvbin_t& process(instfn_t, idfn_t, unsigned begin = 0, unsigned end = 0);
   int         processInstruction(unsigned word, instfn_t, idfn_t);

   void        validate() const;
   void        mapTypeConst();
   void        mapFnBodies();
   void        optLoadStore();
   void        dceFuncs();
   void        dceVars();
   void        dceTypes();
   void        mapNames();
   void        foldIds();  // fold IDs to smallest space
   void        forwardLoadStores(); // load store forwarding (EXPERIMENTAL)
   void        offsetIds(); // create relative offset IDs

   void        applyMap();            // remap per local name map
   void        mapRemainder();        // map any IDs we haven't touched yet
   void        stripDebug();          // strip all debug info
   void        stripDeadRefs();       // strips debug info for now-dead references after DCE
   void        strip();               // remove debug symbols

   std::vector<spirword_t> spv;      // SPIR words

   std::vector<std::string> stripWhiteList;

   namemap_t               nameMap;  // ID names from OpName

   // Since we want to also do binary ops, we can't use std::vector<bool>.  we could use
   // boost::dynamic_bitset, but we're trying to avoid a boost dependency.
   typedef std::uint64_t bits_t;
   std::vector<bits_t> mapped; // which new IDs have been mapped
   static const int mBits = sizeof(bits_t) * 4;

   bool isMapped(spv::Id id) const  { return id < maxMappedId() && ((mapped[id/mBits] & (1LL<<(id%mBits))) != 0); }
   void setMapped(spv::Id id) { resizeMapped(id); mapped[id/mBits] |= (1LL<<(id%mBits)); }
   void resizeMapped(spv::Id id) { if (id >= maxMappedId()) mapped.resize(id/mBits+1, 0); }
   size_t maxMappedId() const { return mapped.size() * mBits; }

   // Add a strip range for a given instruction starting at 'start'
   // Note: avoiding brace initializers to please older versions os MSVC.
   void stripInst(unsigned start) { stripRange.push_back(range_t(start, start + asWordCount(start))); }

   // Function start and end.  use unordered_map because we'll have
   // many fewer functions than IDs.
   std::unordered_map<spv::Id, range_t> fnPos;

   // Which functions are called, anywhere in the module, with a call count
   std::unordered_map<spv::Id, int> fnCalls;

   posmap_t       typeConstPos;  // word positions that define types & consts (ordered)
   posmap_rev_t   idPosR;        // reverse map from IDs to positions
   typesize_map_t idTypeSizeMap; // maps each ID to its type size, if known.

   std::vector<spv::Id>  idMapL;   // ID {M}ap from {L}ocal to {G}lobal IDs

   spv::Id entryPoint;      // module entry point
   spv::Id largestNewId;    // biggest new ID we have mapped anything to

   // Sections of the binary to strip, given as [begin,end)
   std::vector<range_t> stripRange;

   // processing options:
   std::uint32_t options;
   int           verbose;     // verbosity level

   // Error latch: this is set if the error handler is ever executed.  It would be better to
   // use a try/catch block and throw, but that's not desired for certain environments, so
   // this is the alternative.
   mutable bool errorLatch;

   static errorfn_t errorHandler;
   static logfn_t   logHandler;
};

} // namespace SPV

#endif // SPIRVREMAPPER_H
