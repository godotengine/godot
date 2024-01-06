----------------------------------------------------------------------------
-- LuaJIT profiler.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------
--
-- This module is a simple command line interface to the built-in
-- low-overhead profiler of LuaJIT.
--
-- The lower-level API of the profiler is accessible via the "jit.profile"
-- module or the luaJIT_profile_* C API.
--
-- Example usage:
--
--   luajit -jp myapp.lua
--   luajit -jp=s myapp.lua
--   luajit -jp=-s myapp.lua
--   luajit -jp=vl myapp.lua
--   luajit -jp=G,profile.txt myapp.lua
--
-- The following dump features are available:
--
--   f  Stack dump: function name, otherwise module:line. Default mode.
--   F  Stack dump: ditto, but always prepend module.
--   l  Stack dump: module:line.
--   <number> stack dump depth (callee < caller). Default: 1.
--   -<number> Inverse stack dump depth (caller > callee).
--   s  Split stack dump after first stack level. Implies abs(depth) >= 2.
--   p  Show full path for module names.
--   v  Show VM states. Can be combined with stack dumps, e.g. vf or fv.
--   z  Show zones. Can be combined with stack dumps, e.g. zf or fz.
--   r  Show raw sample counts. Default: show percentages.
--   a  Annotate excerpts from source code files.
--   A  Annotate complete source code files.
--   G  Produce raw output suitable for graphical tools (e.g. flame graphs).
--   m<number> Minimum sample percentage to be shown. Default: 3.
--   i<number> Sampling interval in milliseconds. Default: 10.
--
----------------------------------------------------------------------------

-- Cache some library functions and objects.
local jit = require("jit")
local profile = require("jit.profile")
local vmdef = require("jit.vmdef")
local math = math
local pairs, ipairs, tonumber, floor = pairs, ipairs, tonumber, math.floor
local sort, format = table.sort, string.format
local stdout = io.stdout
local zone -- Load jit.zone module on demand.

-- Output file handle.
local out

------------------------------------------------------------------------------

local prof_ud
local prof_states, prof_split, prof_min, prof_raw, prof_fmt, prof_depth
local prof_ann, prof_count1, prof_count2, prof_samples

local map_vmmode = {
  N = "Compiled",
  I = "Interpreted",
  C = "C code",
  G = "Garbage Collector",
  J = "JIT Compiler",
}

-- Profiler callback.
local function prof_cb(th, samples, vmmode)
  prof_samples = prof_samples + samples
  local key_stack, key_stack2, key_state
  -- Collect keys for sample.
  if prof_states then
    if prof_states == "v" then
      key_state = map_vmmode[vmmode] or vmmode
    else
      key_state = zone:get() or "(none)"
    end
  end
  if prof_fmt then
    key_stack = profile.dumpstack(th, prof_fmt, prof_depth)
    key_stack = key_stack:gsub("%[builtin#(%d+)%]", function(x)
      return vmdef.ffnames[tonumber(x)]
    end)
    if prof_split == 2 then
      local k1, k2 = key_stack:match("(.-) [<>] (.*)")
      if k2 then key_stack, key_stack2 = k1, k2 end
    elseif prof_split == 3 then
      key_stack2 = profile.dumpstack(th, "l", 1)
    end
  end
  -- Order keys.
  local k1, k2
  if prof_split == 1 then
    if key_state then
      k1 = key_state
      if key_stack then k2 = key_stack end
    end
  elseif key_stack then
    k1 = key_stack
    if key_stack2 then k2 = key_stack2 elseif key_state then k2 = key_state end
  end
  -- Coalesce samples in one or two levels.
  if k1 then
    local t1 = prof_count1
    t1[k1] = (t1[k1] or 0) + samples
    if k2 then
      local t2 = prof_count2
      local t3 = t2[k1]
      if not t3 then t3 = {}; t2[k1] = t3 end
      t3[k2] = (t3[k2] or 0) + samples
    end
  end
end

------------------------------------------------------------------------------

-- Show top N list.
local function prof_top(count1, count2, samples, indent)
  local t, n = {}, 0
  for k in pairs(count1) do
    n = n + 1
    t[n] = k
  end
  sort(t, function(a, b) return count1[a] > count1[b] end)
  for i=1,n do
    local k = t[i]
    local v = count1[k]
    local pct = floor(v*100/samples + 0.5)
    if pct < prof_min then break end
    if not prof_raw then
      out:write(format("%s%2d%%  %s\n", indent, pct, k))
    elseif prof_raw == "r" then
      out:write(format("%s%5d  %s\n", indent, v, k))
    else
      out:write(format("%s %d\n", k, v))
    end
    if count2 then
      local r = count2[k]
      if r then
	prof_top(r, nil, v, (prof_split == 3 or prof_split == 1) and "  -- " or
			    (prof_depth < 0 and "  -> " or "  <- "))
      end
    end
  end
end

-- Annotate source code
local function prof_annotate(count1, samples)
  local files = {}
  local ms = 0
  for k, v in pairs(count1) do
    local pct = floor(v*100/samples + 0.5)
    ms = math.max(ms, v)
    if pct >= prof_min then
      local file, line = k:match("^(.*):(%d+)$")
      if not file then file = k; line = 0 end
      local fl = files[file]
      if not fl then fl = {}; files[file] = fl; files[#files+1] = file end
      line = tonumber(line)
      fl[line] = prof_raw and v or pct
    end
  end
  sort(files)
  local fmtv, fmtn = " %3d%% | %s\n", "      | %s\n"
  if prof_raw then
    local n = math.max(5, math.ceil(math.log10(ms)))
    fmtv = "%"..n.."d | %s\n"
    fmtn = (" "):rep(n).." | %s\n"
  end
  local ann = prof_ann
  for _, file in ipairs(files) do
    local f0 = file:byte()
    if f0 == 40 or f0 == 91 then
      out:write(format("\n====== %s ======\n[Cannot annotate non-file]\n", file))
      break
    end
    local fp, err = io.open(file)
    if not fp then
      out:write(format("====== ERROR: %s: %s\n", file, err))
      break
    end
    out:write(format("\n====== %s ======\n", file))
    local fl = files[file]
    local n, show = 1, false
    if ann ~= 0 then
      for i=1,ann do
	if fl[i] then show = true; out:write("@@ 1 @@\n"); break end
      end
    end
    for line in fp:lines() do
      if line:byte() == 27 then
	out:write("[Cannot annotate bytecode file]\n")
	break
      end
      local v = fl[n]
      if ann ~= 0 then
	local v2 = fl[n+ann]
	if show then
	  if v2 then show = n+ann elseif v then show = n
	  elseif show+ann < n then show = false end
	elseif v2 then
	  show = n+ann
	  out:write(format("@@ %d @@\n", n))
	end
	if not show then goto next end
      end
      if v then
	out:write(format(fmtv, v, line))
      else
	out:write(format(fmtn, line))
      end
    ::next::
      n = n + 1
    end
    fp:close()
  end
end

------------------------------------------------------------------------------

-- Finish profiling and dump result.
local function prof_finish()
  if prof_ud then
    profile.stop()
    local samples = prof_samples
    if samples == 0 then
      if prof_raw ~= true then out:write("[No samples collected]\n") end
      return
    end
    if prof_ann then
      prof_annotate(prof_count1, samples)
    else
      prof_top(prof_count1, prof_count2, samples, "")
    end
    prof_count1 = nil
    prof_count2 = nil
    prof_ud = nil
    if out ~= stdout then out:close() end
  end
end

-- Start profiling.
local function prof_start(mode)
  local interval = ""
  mode = mode:gsub("i%d*", function(s) interval = s; return "" end)
  prof_min = 3
  mode = mode:gsub("m(%d+)", function(s) prof_min = tonumber(s); return "" end)
  prof_depth = 1
  mode = mode:gsub("%-?%d+", function(s) prof_depth = tonumber(s); return "" end)
  local m = {}
  for c in mode:gmatch(".") do m[c] = c end
  prof_states = m.z or m.v
  if prof_states == "z" then zone = require("jit.zone") end
  local scope = m.l or m.f or m.F or (prof_states and "" or "f")
  local flags = (m.p or "")
  prof_raw = m.r
  if m.s then
    prof_split = 2
    if prof_depth == -1 or m["-"] then prof_depth = -2
    elseif prof_depth == 1 then prof_depth = 2 end
  elseif mode:find("[fF].*l") then
    scope = "l"
    prof_split = 3
  else
    prof_split = (scope == "" or mode:find("[zv].*[lfF]")) and 1 or 0
  end
  prof_ann = m.A and 0 or (m.a and 3)
  if prof_ann then
    scope = "l"
    prof_fmt = "pl"
    prof_split = 0
    prof_depth = 1
  elseif m.G and scope ~= "" then
    prof_fmt = flags..scope.."Z;"
    prof_depth = -100
    prof_raw = true
    prof_min = 0
  elseif scope == "" then
    prof_fmt = false
  else
    local sc = prof_split == 3 and m.f or m.F or scope
    prof_fmt = flags..sc..(prof_depth >= 0 and "Z < " or "Z > ")
  end
  prof_count1 = {}
  prof_count2 = {}
  prof_samples = 0
  profile.start(scope:lower()..interval, prof_cb)
  prof_ud = newproxy(true)
  getmetatable(prof_ud).__gc = prof_finish
end

------------------------------------------------------------------------------

local function start(mode, outfile)
  if not outfile then outfile = os.getenv("LUAJIT_PROFILEFILE") end
  if outfile then
    out = outfile == "-" and stdout or assert(io.open(outfile, "w"))
  else
    out = stdout
  end
  prof_start(mode or "f")
end

-- Public module functions.
return {
  start = start, -- For -j command line option.
  stop = prof_finish
}

