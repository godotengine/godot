----------------------------------------------------------------------------
-- LuaJIT module to save/list bytecode.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------
--
-- This module saves or lists the bytecode for an input file.
-- It's run by the -b command line option.
--
------------------------------------------------------------------------------

local jit = require("jit")
assert(jit.version_num == 20199, "LuaJIT core/library version mismatch")
local bit = require("bit")

-- Symbol name prefix for LuaJIT bytecode.
local LJBC_PREFIX = "luaJIT_BC_"

local type, assert = type, assert
local format = string.format
local tremove, tconcat = table.remove, table.concat

------------------------------------------------------------------------------

local function usage()
  io.stderr:write[[
Save LuaJIT bytecode: luajit -b[options] input output
  -l        Only list bytecode.
  -s        Strip debug info (default).
  -g        Keep debug info.
  -W        Generate 32 bit (non-GC64) bytecode.
  -X        Generate 64 bit (GC64) bytecode.
  -d        Generate bytecode in deterministic manner.
  -n name   Set module name (default: auto-detect from input name).
  -t type   Set output file type (default: auto-detect from output name).
  -a arch   Override architecture for object files (default: native).
  -o os     Override OS for object files (default: native).
  -F name   Override filename (default: input filename).
  -e chunk  Use chunk string as input.
  --        Stop handling options.
  -         Use stdin as input and/or stdout as output.

File types: c cc h obj o raw (default)
]]
  os.exit(1)
end

local function check(ok, ...)
  if ok then return ok, ... end
  io.stderr:write("luajit: ", ...)
  io.stderr:write("\n")
  os.exit(1)
end

local function readfile(ctx, input)
  if ctx.string then
    return check(loadstring(input, nil, ctx.mode))
  elseif ctx.filename then
    local data
    if input == "-" then
      data = io.stdin:read("*a")
    else
      local fp = assert(io.open(input, "rb"))
      data = assert(fp:read("*a"))
      assert(fp:close())
    end
    return check(load(data, ctx.filename, ctx.mode))
  else
    if input == "-" then input = nil end
    return check(loadfile(input, ctx.mode))
  end
end

local function savefile(name, mode)
  if name == "-" then return io.stdout end
  return check(io.open(name, mode))
end

local function set_stdout_binary(ffi)
  ffi.cdef[[int _setmode(int fd, int mode);]]
  ffi.C._setmode(1, 0x8000)
end

------------------------------------------------------------------------------

local map_type = {
  raw = "raw", c = "c", cc = "c", h = "h", o = "obj", obj = "obj",
}

local map_arch = {
  x86 =		{ e = "le", b = 32, m = 3, p = 0x14c, },
  x64 =		{ e = "le", b = 64, m = 62, p = 0x8664, },
  arm =		{ e = "le", b = 32, m = 40, p = 0x1c0, },
  arm64 =	{ e = "le", b = 64, m = 183, p = 0xaa64, },
  arm64be =	{ e = "be", b = 64, m = 183, },
  ppc =		{ e = "be", b = 32, m = 20, },
  mips =	{ e = "be", b = 32, m = 8, f = 0x50001006, },
  mipsel =	{ e = "le", b = 32, m = 8, f = 0x50001006, },
  mips64 =	{ e = "be", b = 64, m = 8, f = 0x80000007, },
  mips64el =	{ e = "le", b = 64, m = 8, f = 0x80000007, },
  mips64r6 =	{ e = "be", b = 64, m = 8, f = 0xa0000407, },
  mips64r6el =	{ e = "le", b = 64, m = 8, f = 0xa0000407, },
  riscv64 =    { e = "le", b = 64, m = 243, f = 0x00000004, },
}

local map_os = {
  linux = true, windows = true, osx = true, freebsd = true, netbsd = true,
  openbsd = true, dragonfly = true, solaris = true,
}

local function checkarg(str, map, err)
  str = str:lower()
  local s = check(map[str], "unknown ", err)
  return type(s) == "string" and s or str
end

local function detecttype(str)
  local ext = str:lower():match("%.(%a+)$")
  return map_type[ext] or "raw"
end

local function checkmodname(str)
  check(str:match("^[%w_.%-]+$"), "bad module name")
  return str:gsub("[%.%-]", "_")
end

local function detectmodname(str)
  if type(str) == "string" then
    local tail = str:match("[^/\\]+$")
    if tail then str = tail end
    local head = str:match("^(.*)%.[^.]*$")
    if head then str = head end
    str = str:match("^[%w_.%-]+")
  else
    str = nil
  end
  check(str, "cannot derive module name, use -n name")
  return str:gsub("[%.%-]", "_")
end

------------------------------------------------------------------------------

local function bcsave_tail(fp, output, s)
  local ok, err = fp:write(s)
  if ok and output ~= "-" then ok, err = fp:close() end
  check(ok, "cannot write ", output, ": ", err)
end

local function bcsave_raw(output, s)
  if output == "-" and jit.os == "Windows" then
    local ok, ffi = pcall(require, "ffi")
    check(ok, "FFI library required to write binary file to stdout")
    set_stdout_binary(ffi)
  end
  local fp = savefile(output, "wb")
  bcsave_tail(fp, output, s)
end

local function bcsave_c(ctx, output, s)
  local fp = savefile(output, "w")
  if ctx.type == "c" then
    fp:write(format([[
#ifdef __cplusplus
extern "C"
#endif
#ifdef _WIN32
__declspec(dllexport)
#endif
const unsigned char %s%s[] = {
]], LJBC_PREFIX, ctx.modname))
  else
    fp:write(format([[
#define %s%s_SIZE %d
static const unsigned char %s%s[] = {
]], LJBC_PREFIX, ctx.modname, #s, LJBC_PREFIX, ctx.modname))
  end
  local t, n, m = {}, 0, 0
  for i=1,#s do
    local b = tostring(string.byte(s, i))
    m = m + #b + 1
    if m > 78 then
      fp:write(tconcat(t, ",", 1, n), ",\n")
      n, m = 0, #b + 1
    end
    n = n + 1
    t[n] = b
  end
  bcsave_tail(fp, output, tconcat(t, ",", 1, n).."\n};\n")
end

local function bcsave_elfobj(ctx, output, s, ffi)
  ffi.cdef[[
typedef struct {
  uint8_t emagic[4], eclass, eendian, eversion, eosabi, eabiversion, epad[7];
  uint16_t type, machine;
  uint32_t version;
  uint32_t entry, phofs, shofs;
  uint32_t flags;
  uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstridx;
} ELF32header;
typedef struct {
  uint8_t emagic[4], eclass, eendian, eversion, eosabi, eabiversion, epad[7];
  uint16_t type, machine;
  uint32_t version;
  uint64_t entry, phofs, shofs;
  uint32_t flags;
  uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstridx;
} ELF64header;
typedef struct {
  uint32_t name, type, flags, addr, ofs, size, link, info, align, entsize;
} ELF32sectheader;
typedef struct {
  uint32_t name, type;
  uint64_t flags, addr, ofs, size;
  uint32_t link, info;
  uint64_t align, entsize;
} ELF64sectheader;
typedef struct {
  uint32_t name, value, size;
  uint8_t info, other;
  uint16_t sectidx;
} ELF32symbol;
typedef struct {
  uint32_t name;
  uint8_t info, other;
  uint16_t sectidx;
  uint64_t value, size;
} ELF64symbol;
typedef struct {
  ELF32header hdr;
  ELF32sectheader sect[6];
  ELF32symbol sym[2];
  uint8_t space[4096];
} ELF32obj;
typedef struct {
  ELF64header hdr;
  ELF64sectheader sect[6];
  ELF64symbol sym[2];
  uint8_t space[4096];
} ELF64obj;
]]
  local symname = LJBC_PREFIX..ctx.modname
  local ai = assert(map_arch[ctx.arch])
  local is64, isbe = ai.b == 64, ai.e == "be"

  -- Handle different host/target endianess.
  local function f32(x) return x end
  local f16, fofs = f32, f32
  if ffi.abi("be") ~= isbe then
    f32 = bit.bswap
    function f16(x) return bit.rshift(bit.bswap(x), 16) end
    if is64 then
      local two32 = ffi.cast("int64_t", 2^32)
      function fofs(x) return bit.bswap(x)*two32 end
    else
      fofs = f32
    end
  end

  -- Create ELF object and fill in header.
  local o = ffi.new(is64 and "ELF64obj" or "ELF32obj")
  local hdr = o.hdr
  if ctx.os == "bsd" or ctx.os == "other" then -- Determine native hdr.eosabi.
    local bf = assert(io.open("/bin/ls", "rb"))
    local bs = bf:read(9)
    bf:close()
    ffi.copy(o, bs, 9)
    check(hdr.emagic[0] == 127, "no support for writing native object files")
  else
    hdr.emagic = "\127ELF"
    hdr.eosabi = ({ freebsd=9, netbsd=2, openbsd=12, solaris=6 })[ctx.os] or 0
  end
  hdr.eclass = is64 and 2 or 1
  hdr.eendian = isbe and 2 or 1
  hdr.eversion = 1
  hdr.type = f16(1)
  hdr.machine = f16(ai.m)
  hdr.flags = f32(ai.f or 0)
  hdr.version = f32(1)
  hdr.shofs = fofs(ffi.offsetof(o, "sect"))
  hdr.ehsize = f16(ffi.sizeof(hdr))
  hdr.shentsize = f16(ffi.sizeof(o.sect[0]))
  hdr.shnum = f16(6)
  hdr.shstridx = f16(2)

  -- Fill in sections and symbols.
  local sofs, ofs = ffi.offsetof(o, "space"), 1
  for i,name in ipairs{
      ".symtab", ".shstrtab", ".strtab", ".rodata", ".note.GNU-stack",
    } do
    local sect = o.sect[i]
    sect.align = fofs(1)
    sect.name = f32(ofs)
    ffi.copy(o.space+ofs, name)
    ofs = ofs + #name+1
  end
  o.sect[1].type = f32(2) -- .symtab
  o.sect[1].link = f32(3)
  o.sect[1].info = f32(1)
  o.sect[1].align = fofs(8)
  o.sect[1].ofs = fofs(ffi.offsetof(o, "sym"))
  o.sect[1].entsize = fofs(ffi.sizeof(o.sym[0]))
  o.sect[1].size = fofs(ffi.sizeof(o.sym))
  o.sym[1].name = f32(1)
  o.sym[1].sectidx = f16(4)
  o.sym[1].size = fofs(#s)
  o.sym[1].info = 17
  o.sect[2].type = f32(3) -- .shstrtab
  o.sect[2].ofs = fofs(sofs)
  o.sect[2].size = fofs(ofs)
  o.sect[3].type = f32(3) -- .strtab
  o.sect[3].ofs = fofs(sofs + ofs)
  o.sect[3].size = fofs(#symname+2)
  ffi.copy(o.space+ofs+1, symname)
  ofs = ofs + #symname + 2
  o.sect[4].type = f32(1) -- .rodata
  o.sect[4].flags = fofs(2)
  o.sect[4].ofs = fofs(sofs + ofs)
  o.sect[4].size = fofs(#s)
  o.sect[5].type = f32(1) -- .note.GNU-stack
  o.sect[5].ofs = fofs(sofs + ofs + #s)

  -- Write ELF object file.
  local fp = savefile(output, "wb")
  fp:write(ffi.string(o, ffi.sizeof(o)-4096+ofs))
  bcsave_tail(fp, output, s)
end

local function bcsave_peobj(ctx, output, s, ffi)
  ffi.cdef[[
typedef struct {
  uint16_t arch, nsects;
  uint32_t time, symtabofs, nsyms;
  uint16_t opthdrsz, flags;
} PEheader;
typedef struct {
  char name[8];
  uint32_t vsize, vaddr, size, ofs, relocofs, lineofs;
  uint16_t nreloc, nline;
  uint32_t flags;
} PEsection;
typedef struct __attribute((packed)) {
  union {
    char name[8];
    uint32_t nameref[2];
  };
  uint32_t value;
  int16_t sect;
  uint16_t type;
  uint8_t scl, naux;
} PEsym;
typedef struct __attribute((packed)) {
  uint32_t size;
  uint16_t nreloc, nline;
  uint32_t cksum;
  uint16_t assoc;
  uint8_t comdatsel, unused[3];
} PEsymaux;
typedef struct {
  PEheader hdr;
  PEsection sect[2];
  // Must be an even number of symbol structs.
  PEsym sym0;
  PEsymaux sym0aux;
  PEsym sym1;
  PEsymaux sym1aux;
  PEsym sym2;
  PEsym sym3;
  uint32_t strtabsize;
  uint8_t space[4096];
} PEobj;
]]
  local symname = LJBC_PREFIX..ctx.modname
  local ai = assert(map_arch[ctx.arch])
  local is64 = ai.b == 64
  local symexport = "   /EXPORT:"..symname..",DATA "

  -- The file format is always little-endian. Swap if the host is big-endian.
  local function f32(x) return x end
  local f16 = f32
  if ffi.abi("be") then
    f32 = bit.bswap
    function f16(x) return bit.rshift(bit.bswap(x), 16) end
  end

  -- Create PE object and fill in header.
  local o = ffi.new("PEobj")
  local hdr = o.hdr
  hdr.arch = f16(assert(ai.p))
  hdr.nsects = f16(2)
  hdr.symtabofs = f32(ffi.offsetof(o, "sym0"))
  hdr.nsyms = f32(6)

  -- Fill in sections and symbols.
  o.sect[0].name = ".drectve"
  o.sect[0].size = f32(#symexport)
  o.sect[0].flags = f32(0x00100a00)
  o.sym0.sect = f16(1)
  o.sym0.scl = 3
  o.sym0.name = ".drectve"
  o.sym0.naux = 1
  o.sym0aux.size = f32(#symexport)
  o.sect[1].name = ".rdata"
  o.sect[1].size = f32(#s)
  o.sect[1].flags = f32(0x40300040)
  o.sym1.sect = f16(2)
  o.sym1.scl = 3
  o.sym1.name = ".rdata"
  o.sym1.naux = 1
  o.sym1aux.size = f32(#s)
  o.sym2.sect = f16(2)
  o.sym2.scl = 2
  o.sym2.nameref[1] = f32(4)
  o.sym3.sect = f16(-1)
  o.sym3.scl = 2
  o.sym3.value = f32(1)
  o.sym3.name = "@feat.00" -- Mark as SafeSEH compliant.
  ffi.copy(o.space, symname)
  local ofs = #symname + 1
  o.strtabsize = f32(ofs + 4)
  o.sect[0].ofs = f32(ffi.offsetof(o, "space") + ofs)
  ffi.copy(o.space + ofs, symexport)
  ofs = ofs + #symexport
  o.sect[1].ofs = f32(ffi.offsetof(o, "space") + ofs)

  -- Write PE object file.
  local fp = savefile(output, "wb")
  fp:write(ffi.string(o, ffi.sizeof(o)-4096+ofs))
  bcsave_tail(fp, output, s)
end

local function bcsave_machobj(ctx, output, s, ffi)
  ffi.cdef[[
typedef struct
{
  uint32_t magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags;
} mach_header;
typedef struct
{
  mach_header; uint32_t reserved;
} mach_header_64;
typedef struct {
  uint32_t cmd, cmdsize;
  char segname[16];
  uint64_t vmaddr, vmsize, fileoff, filesize;
  uint32_t maxprot, initprot, nsects, flags;
} mach_segment_command_64;
typedef struct {
  char sectname[16], segname[16];
  uint64_t addr, size;
  uint32_t offset, align, reloff, nreloc, flags;
  uint32_t reserved1, reserved2, reserved3;
} mach_section_64;
typedef struct {
  uint32_t cmd, cmdsize, symoff, nsyms, stroff, strsize;
} mach_symtab_command;
typedef struct {
  int32_t strx;
  uint8_t type, sect;
  uint16_t desc;
  uint64_t value;
} mach_nlist_64;
typedef struct {
  mach_header_64 hdr;
  mach_segment_command_64 seg;
  mach_section_64 sec;
  mach_symtab_command sym;
  mach_nlist_64 sym_entry;
  uint8_t space[4096];
} mach_obj_64;
]]
  local symname = '_'..LJBC_PREFIX..ctx.modname
  local cputype, cpusubtype = 0x01000007, 3
  if ctx.arch ~= "x64" then
    check(ctx.arch == "arm64", "unsupported architecture for OSX")
    cputype, cpusubtype = 0x0100000c, 0
  end
  local function aligned(v, a) return bit.band(v+a-1, -a) end

  -- Create Mach-O object and fill in header.
  local o = ffi.new("mach_obj_64")
  local mach_size = aligned(ffi.offsetof(o, "space")+#symname+2, 8)

  -- Fill in sections and symbols.
  o.hdr.magic = 0xfeedfacf
  o.hdr.cputype = cputype
  o.hdr.cpusubtype = cpusubtype
  o.hdr.filetype = 1
  o.hdr.ncmds = 2
  o.hdr.sizeofcmds = ffi.sizeof(o.seg)+ffi.sizeof(o.sec)+ffi.sizeof(o.sym)
  o.seg.cmd = 0x19
  o.seg.cmdsize = ffi.sizeof(o.seg)+ffi.sizeof(o.sec)
  o.seg.vmsize = #s
  o.seg.fileoff = mach_size
  o.seg.filesize = #s
  o.seg.maxprot = 1
  o.seg.initprot = 1
  o.seg.nsects = 1
  ffi.copy(o.sec.sectname, "__data")
  ffi.copy(o.sec.segname, "__DATA")
  o.sec.size = #s
  o.sec.offset = mach_size
  o.sym.cmd = 2
  o.sym.cmdsize = ffi.sizeof(o.sym)
  o.sym.symoff = ffi.offsetof(o, "sym_entry")
  o.sym.nsyms = 1
  o.sym.stroff = ffi.offsetof(o, "sym_entry")+ffi.sizeof(o.sym_entry)
  o.sym.strsize = aligned(#symname+2, 8)
  o.sym_entry.type = 0xf
  o.sym_entry.sect = 1
  o.sym_entry.strx = 1
  ffi.copy(o.space+1, symname)

  -- Write Mach-O object file.
  local fp = savefile(output, "wb")
  fp:write(ffi.string(o, mach_size))
  bcsave_tail(fp, output, s)
end

local function bcsave_obj(ctx, output, s)
  local ok, ffi = pcall(require, "ffi")
  check(ok, "FFI library required to write this file type")
  if output == "-" and jit.os == "Windows" then
    set_stdout_binary(ffi)
  end
  if ctx.os == "windows" then
    return bcsave_peobj(ctx, output, s, ffi)
  elseif ctx.os == "osx" then
    return bcsave_machobj(ctx, output, s, ffi)
  else
    return bcsave_elfobj(ctx, output, s, ffi)
  end
end

------------------------------------------------------------------------------

local function bclist(ctx, input, output)
  local f = readfile(ctx, input)
  require("jit.bc").dump(f, savefile(output, "w"), true)
end

local function bcsave(ctx, input, output)
  local f = readfile(ctx, input)
  local s = string.dump(f, ctx.mode)
  local t = ctx.type
  if not t then
    t = detecttype(output)
    ctx.type = t
  end
  if t == "raw" then
    bcsave_raw(output, s)
  else
    if not ctx.modname then ctx.modname = detectmodname(input) end
    if t == "obj" then
      bcsave_obj(ctx, output, s)
    else
      bcsave_c(ctx, output, s)
    end
  end
end

local function docmd(...)
  local arg = {...}
  local n = 1
  local list = false
  local ctx = {
    mode = "bt", arch = jit.arch, os = jit.os:lower(),
    type = false, modname = false, string = false,
  }
  local strip = "s"
  local gc64 = ""
  while n <= #arg do
    local a = arg[n]
    if type(a) == "string" and a:sub(1, 1) == "-" and a ~= "-" then
      tremove(arg, n)
      if a == "--" then break end
      for m=2,#a do
	local opt = a:sub(m, m)
	if opt == "l" then
	  list = true
	elseif opt == "s" then
	  strip = "s"
	elseif opt == "g" then
	  strip = ""
	elseif opt == "W" or opt == "X" then
	  gc64 = opt
	elseif opt == "d" then
	  ctx.mode = ctx.mode .. opt
	else
	  if arg[n] == nil or m ~= #a then usage() end
	  if opt == "e" then
	    if n ~= 1 then usage() end
	    ctx.string = true
	  elseif opt == "n" then
	    ctx.modname = checkmodname(tremove(arg, n))
	  elseif opt == "t" then
	    ctx.type = checkarg(tremove(arg, n), map_type, "file type")
	  elseif opt == "a" then
	    ctx.arch = checkarg(tremove(arg, n), map_arch, "architecture")
	  elseif opt == "o" then
	    ctx.os = checkarg(tremove(arg, n), map_os, "OS name")
	  elseif opt == "F" then
	    ctx.filename = "@"..tremove(arg, n)
	  else
	    usage()
	  end
	end
      end
    else
      n = n + 1
    end
  end
  ctx.mode = ctx.mode .. strip .. gc64
  if list then
    if #arg == 0 or #arg > 2 then usage() end
    bclist(ctx, arg[1], arg[2] or "-")
  else
    if #arg ~= 2 then usage() end
    bcsave(ctx, arg[1], arg[2])
  end
end

------------------------------------------------------------------------------

-- Public module functions.
return {
  start = docmd -- Process -b command line option.
}

