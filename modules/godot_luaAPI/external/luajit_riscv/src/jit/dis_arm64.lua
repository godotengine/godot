----------------------------------------------------------------------------
-- LuaJIT ARM64 disassembler module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
--
-- Contributed by Djordje Kovacevic and Stefan Pejic from RT-RK.com.
-- Sponsored by Cisco Systems, Inc.
----------------------------------------------------------------------------
-- This is a helper module used by the LuaJIT machine code dumper module.
--
-- It disassembles most user-mode AArch64 instructions.
-- NYI: Advanced SIMD and VFP instructions.
------------------------------------------------------------------------------

local type = type
local sub, byte, format = string.sub, string.byte, string.format
local match, gmatch, gsub = string.match, string.gmatch, string.gsub
local concat = table.concat
local bit = require("bit")
local band, bor, bxor, tohex = bit.band, bit.bor, bit.bxor, bit.tohex
local lshift, rshift, arshift = bit.lshift, bit.rshift, bit.arshift
local ror = bit.ror

------------------------------------------------------------------------------
-- Opcode maps
------------------------------------------------------------------------------

local map_adr = { -- PC-relative addressing.
  shift = 31, mask = 1,
  [0] = "adrDBx", "adrpDBx"
}

local map_addsubi = { -- Add/subtract immediate.
  shift = 29, mask = 3,
  [0] = "add|movDNIg", "adds|cmnD0NIg", "subDNIg", "subs|cmpD0NIg",
}

local map_logi = { -- Logical immediate.
  shift = 31, mask = 1,
  [0] = {
    shift = 22, mask = 1,
    [0] = {
      shift = 29, mask = 3,
      [0] = "andDNig", "orr|movDN0ig", "eorDNig", "ands|tstD0Nig"
    },
    false -- unallocated
  },
  {
    shift = 29, mask = 3,
    [0] = "andDNig", "orr|movDN0ig", "eorDNig", "ands|tstD0Nig"
  }
}

local map_movwi = { -- Move wide immediate.
  shift = 31, mask = 1,
  [0] = {
    shift = 22, mask = 1,
    [0] = {
      shift = 29, mask = 3,
      [0] = "movnDWRg", false, "movz|movDYRg", "movkDWRg"
    }, false -- unallocated
  },
  {
    shift = 29, mask = 3,
    [0] = "movnDWRg", false, "movz|movDYRg", "movkDWRg"
  },
}

local map_bitf = { -- Bitfield.
  shift = 31, mask = 1,
  [0] = {
    shift = 22, mask = 1,
    [0] = {
      shift = 29, mask = 3,
      [0] = "sbfm|sbfiz|sbfx|asr|sxtw|sxth|sxtbDN12w",
      "bfm|bfi|bfxilDN13w",
      "ubfm|ubfiz|ubfx|lsr|lsl|uxth|uxtbDN12w"
    }
  },
  {
    shift = 22, mask = 1,
    {
      shift = 29, mask = 3,
      [0] = "sbfm|sbfiz|sbfx|asr|sxtw|sxth|sxtbDN12x",
      "bfm|bfi|bfxilDN13x",
      "ubfm|ubfiz|ubfx|lsr|lsl|uxth|uxtbDN12x"
    }
  }
}

local map_datai = { -- Data processing - immediate.
  shift = 23, mask = 7,
  [0] = map_adr, map_adr, map_addsubi, false,
  map_logi, map_movwi, map_bitf,
  {
    shift = 15, mask = 0x1c0c1,
    [0] = "extr|rorDNM4w", [0x10080] = "extr|rorDNM4x",
    [0x10081] = "extr|rorDNM4x"
  }
}

local map_logsr = { -- Logical, shifted register.
  shift = 31, mask = 1,
  [0] = {
    shift = 15, mask = 1,
    [0] = {
      shift = 29, mask = 3,
      [0] = {
	shift = 21, mask = 1,
	[0] = "andDNMSg", "bicDNMSg"
      },
      {
	shift = 21, mask = 1,
	[0] = "orr|movDN0MSg", "orn|mvnDN0MSg"
      },
      {
	shift = 21, mask = 1,
	[0] = "eorDNMSg", "eonDNMSg"
      },
      {
	shift = 21, mask = 1,
	[0] = "ands|tstD0NMSg", "bicsDNMSg"
      }
    },
    false -- unallocated
  },
  {
    shift = 29, mask = 3,
    [0] = {
      shift = 21, mask = 1,
      [0] = "andDNMSg", "bicDNMSg"
    },
    {
      shift = 21, mask = 1,
      [0] = "orr|movDN0MSg", "orn|mvnDN0MSg"
    },
    {
      shift = 21, mask = 1,
      [0] = "eorDNMSg", "eonDNMSg"
    },
    {
      shift = 21, mask = 1,
      [0] = "ands|tstD0NMSg", "bicsDNMSg"
    }
  }
}

local map_assh = {
  shift = 31, mask = 1,
  [0] = {
    shift = 15, mask = 1,
    [0] = {
      shift = 29, mask = 3,
      [0] = {
	shift = 22, mask = 3,
	[0] = "addDNMSg", "addDNMSg", "addDNMSg", "addDNMg"
      },
      {
	shift = 22, mask = 3,
	[0] = "adds|cmnD0NMSg", "adds|cmnD0NMSg",
	      "adds|cmnD0NMSg", "adds|cmnD0NMg"
      },
      {
	shift = 22, mask = 3,
	[0] = "sub|negDN0MSg", "sub|negDN0MSg", "sub|negDN0MSg", "sub|negDN0Mg"
      },
      {
	shift = 22, mask = 3,
	[0] = "subs|cmp|negsD0N0MzSg", "subs|cmp|negsD0N0MzSg",
	      "subs|cmp|negsD0N0MzSg", "subs|cmp|negsD0N0Mzg"
      },
    },
    false -- unallocated
  },
  {
    shift = 29, mask = 3,
    [0] = {
      shift = 22, mask = 3,
      [0] = "addDNMSg", "addDNMSg", "addDNMSg", "addDNMg"
    },
    {
      shift = 22, mask = 3,
      [0] = "adds|cmnD0NMSg", "adds|cmnD0NMSg", "adds|cmnD0NMSg",
	    "adds|cmnD0NMg"
    },
    {
      shift = 22, mask = 3,
      [0] = "sub|negDN0MSg", "sub|negDN0MSg", "sub|negDN0MSg", "sub|negDN0Mg"
    },
    {
      shift = 22, mask = 3,
      [0] = "subs|cmp|negsD0N0MzSg", "subs|cmp|negsD0N0MzSg",
	    "subs|cmp|negsD0N0MzSg", "subs|cmp|negsD0N0Mzg"
    }
  }
}

local map_addsubsh = { -- Add/subtract, shifted register.
  shift = 22, mask = 3,
  [0] = map_assh, map_assh, map_assh
}

local map_addsubex = { -- Add/subtract, extended register.
  shift = 22, mask = 3,
  [0] = {
    shift = 29, mask = 3,
    [0] = "addDNMXg", "adds|cmnD0NMXg", "subDNMXg", "subs|cmpD0NMzXg",
  }
}

local map_addsubc = { -- Add/subtract, with carry.
  shift = 10, mask = 63,
  [0] = {
    shift = 29, mask = 3,
    [0] = "adcDNMg", "adcsDNMg", "sbc|ngcDN0Mg", "sbcs|ngcsDN0Mg",
  }
}

local map_ccomp = {
  shift = 4, mask = 1,
  [0] = {
    shift = 10, mask = 3,
    [0] = { -- Conditional compare register.
      shift = 29, mask = 3,
      "ccmnNMVCg", false, "ccmpNMVCg",
    },
    [2] = {  -- Conditional compare immediate.
      shift = 29, mask = 3,
      "ccmnN5VCg", false, "ccmpN5VCg",
    }
  }
}

local map_csel = { -- Conditional select.
  shift = 11, mask = 1,
  [0] = {
    shift = 10, mask = 1,
    [0] = {
      shift = 29, mask = 3,
      [0] = "cselDNMzCg", false, "csinv|cinv|csetmDNMcg", false,
    },
    {
      shift = 29, mask = 3,
      [0] = "csinc|cinc|csetDNMcg", false, "csneg|cnegDNMcg", false,
    }
  }
}

local map_data1s = { -- Data processing, 1 source.
  shift = 29, mask = 1,
  [0] = {
    shift = 31, mask = 1,
    [0] = {
      shift = 10, mask = 0x7ff,
      [0] = "rbitDNg", "rev16DNg", "revDNw", false, "clzDNg", "clsDNg"
    },
    {
      shift = 10, mask = 0x7ff,
      [0] = "rbitDNg", "rev16DNg", "rev32DNx", "revDNx", "clzDNg", "clsDNg"
    }
  }
}

local map_data2s = { -- Data processing, 2 sources.
  shift = 29, mask = 1,
  [0] = {
    shift = 10, mask = 63,
    false, "udivDNMg", "sdivDNMg", false, false, false, false, "lslDNMg",
    "lsrDNMg", "asrDNMg", "rorDNMg"
  }
}

local map_data3s = { -- Data processing, 3 sources.
  shift = 29, mask = 7,
  [0] = {
    shift = 21, mask = 7,
    [0] = {
      shift = 15, mask = 1,
      [0] = "madd|mulDNMA0g", "msub|mnegDNMA0g"
    }
  }, false, false, false,
  {
    shift = 15, mask = 1,
    [0] = {
      shift = 21, mask = 7,
      [0] = "madd|mulDNMA0g", "smaddl|smullDxNMwA0x", "smulhDNMx", false,
      false, "umaddl|umullDxNMwA0x", "umulhDNMx"
    },
    {
      shift = 21, mask = 7,
      [0] = "msub|mnegDNMA0g", "smsubl|smneglDxNMwA0x", false, false,
      false, "umsubl|umneglDxNMwA0x"
    }
  }
}

local map_datar = { -- Data processing, register.
  shift = 28, mask = 1,
  [0] = {
    shift = 24, mask = 1,
    [0] = map_logsr,
    {
      shift = 21, mask = 1,
      [0] = map_addsubsh, map_addsubex
    }
  },
  {
    shift = 21, mask = 15,
    [0] = map_addsubc, false, map_ccomp, false, map_csel, false,
    {
      shift = 30, mask = 1,
      [0] = map_data2s, map_data1s
    },
    false, map_data3s, map_data3s, map_data3s, map_data3s, map_data3s,
    map_data3s, map_data3s, map_data3s
  }
}

local map_lrl = { -- Load register, literal.
  shift = 26, mask = 1,
  [0] = {
    shift = 30, mask = 3,
    [0] = "ldrDwB", "ldrDxB", "ldrswDxB"
  },
  {
    shift = 30, mask = 3,
    [0] = "ldrDsB", "ldrDdB"
  }
}

local map_lsriind = { -- Load/store register, immediate pre/post-indexed.
  shift = 30, mask = 3,
  [0] = {
    shift = 26, mask = 1,
    [0] = {
      shift = 22, mask = 3,
      [0] = "strbDwzL", "ldrbDwzL", "ldrsbDxzL", "ldrsbDwzL"
    }
  },
  {
    shift = 26, mask = 1,
    [0] = {
      shift = 22, mask = 3,
      [0] = "strhDwzL", "ldrhDwzL", "ldrshDxzL", "ldrshDwzL"
    }
  },
  {
    shift = 26, mask = 1,
    [0] = {
      shift = 22, mask = 3,
      [0] = "strDwzL", "ldrDwzL", "ldrswDxzL"
    },
    {
      shift = 22, mask = 3,
      [0] = "strDszL", "ldrDszL"
    }
  },
  {
    shift = 26, mask = 1,
    [0] = {
      shift = 22, mask = 3,
      [0] = "strDxzL", "ldrDxzL"
    },
    {
      shift = 22, mask = 3,
      [0] = "strDdzL", "ldrDdzL"
    }
  }
}

local map_lsriro = {
  shift = 21, mask = 1,
  [0] = {  -- Load/store register immediate.
    shift = 10, mask = 3,
    [0] = { -- Unscaled immediate.
      shift = 26, mask = 1,
      [0] = {
	shift = 30, mask = 3,
	[0] = {
	  shift = 22, mask = 3,
	  [0] = "sturbDwK", "ldurbDwK"
	},
	{
	  shift = 22, mask = 3,
	  [0] = "sturhDwK", "ldurhDwK"
	},
	{
	  shift = 22, mask = 3,
	  [0] = "sturDwK", "ldurDwK"
	},
	{
	  shift = 22, mask = 3,
	  [0] = "sturDxK", "ldurDxK"
	}
      }
    }, map_lsriind, false, map_lsriind
  },
  {  -- Load/store register, register offset.
    shift = 10, mask = 3,
    [2] = {
      shift = 26, mask = 1,
      [0] = {
	shift = 30, mask = 3,
	[0] = {
	  shift = 22, mask = 3,
	  [0] = "strbDwO", "ldrbDwO", "ldrsbDxO", "ldrsbDwO"
	},
	{
	  shift = 22, mask = 3,
	  [0] = "strhDwO", "ldrhDwO", "ldrshDxO", "ldrshDwO"
	},
	{
	  shift = 22, mask = 3,
	  [0] = "strDwO", "ldrDwO", "ldrswDxO"
	},
	{
	  shift = 22, mask = 3,
	  [0] = "strDxO", "ldrDxO"
	}
      },
      {
	shift = 30, mask = 3,
	[2] = {
	  shift = 22, mask = 3,
	  [0] = "strDsO", "ldrDsO"
	},
	[3] = {
	  shift = 22, mask = 3,
	  [0] = "strDdO", "ldrDdO"
	}
      }
    }
  }
}

local map_lsp = { -- Load/store register pair, offset.
  shift = 22, mask = 1,
  [0] = {
    shift = 30, mask = 3,
    [0] = {
      shift = 26, mask = 1,
      [0] = "stpDzAzwP", "stpDzAzsP",
    },
    {
      shift = 26, mask = 1,
      "stpDzAzdP"
    },
    {
      shift = 26, mask = 1,
      [0] = "stpDzAzxP"
    }
  },
  {
    shift = 30, mask = 3,
    [0] = {
      shift = 26, mask = 1,
      [0] = "ldpDzAzwP", "ldpDzAzsP",
    },
    {
      shift = 26, mask = 1,
      [0] = "ldpswDAxP", "ldpDzAzdP"
    },
    {
      shift = 26, mask = 1,
      [0] = "ldpDzAzxP"
    }
  }
}

local map_ls = { -- Loads and stores.
  shift = 24, mask = 0x31,
  [0x10] = map_lrl, [0x30] = map_lsriro,
  [0x20] = {
    shift = 23, mask = 3,
    map_lsp, map_lsp, map_lsp
  },
  [0x21] = {
    shift = 23, mask = 3,
    map_lsp, map_lsp, map_lsp
  },
  [0x31] = {
    shift = 26, mask = 1,
    [0] = {
      shift = 30, mask = 3,
      [0] = {
	shift = 22, mask = 3,
	[0] = "strbDwzU", "ldrbDwzU"
      },
      {
	shift = 22, mask = 3,
	[0] = "strhDwzU", "ldrhDwzU"
      },
      {
	shift = 22, mask = 3,
	[0] = "strDwzU", "ldrDwzU"
      },
      {
	shift = 22, mask = 3,
	[0] = "strDxzU", "ldrDxzU"
      }
    },
    {
      shift = 30, mask = 3,
      [2] = {
	shift = 22, mask = 3,
	[0] = "strDszU", "ldrDszU"
      },
      [3] = {
	shift = 22, mask = 3,
	[0] = "strDdzU", "ldrDdzU"
      }
    }
  },
}

local map_datafp = { -- Data processing, SIMD and FP.
  shift = 28, mask = 7,
  { -- 001
    shift = 24, mask = 1,
    [0] = {
      shift = 21, mask = 1,
      {
	shift = 10, mask = 3,
	[0] = {
	  shift = 12, mask = 1,
	  [0] = {
	    shift = 13, mask = 1,
	    [0] = {
	      shift = 14, mask = 1,
	      [0] = {
		shift = 15, mask = 1,
		[0] = { -- FP/int conversion.
		  shift = 31, mask = 1,
		  [0] = {
		    shift = 16, mask = 0xff,
		    [0x20] = "fcvtnsDwNs", [0x21] = "fcvtnuDwNs",
		    [0x22] = "scvtfDsNw", [0x23] = "ucvtfDsNw",
		    [0x24] = "fcvtasDwNs", [0x25] = "fcvtauDwNs",
		    [0x26] = "fmovDwNs", [0x27] = "fmovDsNw",
		    [0x28] = "fcvtpsDwNs", [0x29] = "fcvtpuDwNs",
		    [0x30] = "fcvtmsDwNs", [0x31] = "fcvtmuDwNs",
		    [0x38] = "fcvtzsDwNs", [0x39] = "fcvtzuDwNs",
		    [0x60] = "fcvtnsDwNd", [0x61] = "fcvtnuDwNd",
		    [0x62] = "scvtfDdNw", [0x63] = "ucvtfDdNw",
		    [0x64] = "fcvtasDwNd", [0x65] = "fcvtauDwNd",
		    [0x68] = "fcvtpsDwNd", [0x69] = "fcvtpuDwNd",
		    [0x70] = "fcvtmsDwNd", [0x71] = "fcvtmuDwNd",
		    [0x78] = "fcvtzsDwNd", [0x79] = "fcvtzuDwNd"
		  },
		  {
		    shift = 16, mask = 0xff,
		    [0x20] = "fcvtnsDxNs", [0x21] = "fcvtnuDxNs",
		    [0x22] = "scvtfDsNx", [0x23] = "ucvtfDsNx",
		    [0x24] = "fcvtasDxNs", [0x25] = "fcvtauDxNs",
		    [0x28] = "fcvtpsDxNs", [0x29] = "fcvtpuDxNs",
		    [0x30] = "fcvtmsDxNs", [0x31] = "fcvtmuDxNs",
		    [0x38] = "fcvtzsDxNs", [0x39] = "fcvtzuDxNs",
		    [0x60] = "fcvtnsDxNd", [0x61] = "fcvtnuDxNd",
		    [0x62] = "scvtfDdNx", [0x63] = "ucvtfDdNx",
		    [0x64] = "fcvtasDxNd", [0x65] = "fcvtauDxNd",
		    [0x66] = "fmovDxNd", [0x67] = "fmovDdNx",
		    [0x68] = "fcvtpsDxNd", [0x69] = "fcvtpuDxNd",
		    [0x70] = "fcvtmsDxNd", [0x71] = "fcvtmuDxNd",
		    [0x78] = "fcvtzsDxNd", [0x79] = "fcvtzuDxNd"
		  }
		}
	      },
	      { -- FP data-processing, 1 source.
		shift = 31, mask = 1,
		[0] = {
		  shift = 22, mask = 3,
		  [0] = {
		    shift = 15, mask = 63,
		    [0] = "fmovDNf", "fabsDNf", "fnegDNf",
		    "fsqrtDNf", false, "fcvtDdNs", false, false,
		    "frintnDNf", "frintpDNf", "frintmDNf", "frintzDNf",
		    "frintaDNf", false, "frintxDNf", "frintiDNf",
		  },
		  {
		    shift = 15, mask = 63,
		    [0] = "fmovDNf", "fabsDNf", "fnegDNf",
		    "fsqrtDNf", "fcvtDsNd", false, false, false,
		    "frintnDNf", "frintpDNf", "frintmDNf", "frintzDNf",
		    "frintaDNf", false, "frintxDNf", "frintiDNf",
		  }
		}
	      }
	    },
	    { -- FP compare.
	      shift = 31, mask = 1,
	      [0] = {
		shift = 14, mask = 3,
		[0] = {
		  shift = 23, mask = 1,
		  [0] = {
		    shift = 0, mask = 31,
		    [0] = "fcmpNMf", [8] = "fcmpNZf",
		    [16] = "fcmpeNMf", [24] = "fcmpeNZf",
		  }
		}
	      }
	    }
	  },
	  { -- FP immediate.
	    shift = 31, mask = 1,
	    [0] = {
	      shift = 5, mask = 31,
	      [0] = {
		shift = 23, mask = 1,
		[0] = "fmovDFf"
	      }
	    }
	  }
	},
	{ -- FP conditional compare.
	  shift = 31, mask = 1,
	  [0] = {
	    shift = 23, mask = 1,
	    [0] = {
	      shift = 4, mask = 1,
	      [0] = "fccmpNMVCf", "fccmpeNMVCf"
	    }
	  }
	},
	{ -- FP data-processing, 2 sources.
	  shift = 31, mask = 1,
	  [0] = {
	    shift = 23, mask = 1,
	    [0] = {
	      shift = 12, mask = 15,
	      [0] = "fmulDNMf", "fdivDNMf", "faddDNMf", "fsubDNMf",
	      "fmaxDNMf", "fminDNMf", "fmaxnmDNMf", "fminnmDNMf",
	      "fnmulDNMf"
	    }
	  }
	},
	{ -- FP conditional select.
	  shift = 31, mask = 1,
	  [0] = {
	    shift = 23, mask = 1,
	    [0] = "fcselDNMCf"
	  }
	}
      }
    },
    { -- FP data-processing, 3 sources.
      shift = 31, mask = 1,
      [0] = {
	shift = 15, mask = 1,
	[0] = {
	  shift = 21, mask = 5,
	  [0] = "fmaddDNMAf", "fnmaddDNMAf"
	},
	{
	  shift = 21, mask = 5,
	  [0] = "fmsubDNMAf", "fnmsubDNMAf"
	}
      }
    }
  }
}

local map_br = { -- Branches, exception generating and system instructions.
  shift = 29, mask = 7,
  [0] = "bB",
  { -- Compare & branch, immediate.
    shift = 24, mask = 3,
    [0] = "cbzDBg", "cbnzDBg", "tbzDTBw", "tbnzDTBw"
  },
  { -- Conditional branch, immediate.
    shift = 24, mask = 3,
    [0] = {
      shift = 4, mask = 1,
      [0] = {
	shift = 0, mask = 15,
	[0] = "beqB", "bneB", "bhsB", "bloB", "bmiB", "bplB", "bvsB", "bvcB",
	"bhiB", "blsB", "bgeB", "bltB", "bgtB", "bleB", "balB"
      }
    }
  }, false, "blB",
  { -- Compare & branch, immediate.
    shift = 24, mask = 3,
    [0] = "cbzDBg", "cbnzDBg", "tbzDTBx", "tbnzDTBx"
  },
  {
    shift = 24, mask = 3,
    [0] = { -- Exception generation.
      shift = 0, mask = 0xe0001f,
      [0x200000] = "brkW"
    },
    { -- System instructions.
      shift = 0, mask = 0x3fffff,
      [0x03201f] = "nop"
    },
    { -- Unconditional branch, register.
      shift = 0, mask = 0xfffc1f,
      [0x1f0000] = "brNx", [0x3f0000] = "blrNx",
      [0x5f0000] = "retNx"
    },
  }
}

local map_init = {
  shift = 25, mask = 15,
  [0] = false, false, false, false, map_ls, map_datar, map_ls, map_datafp,
  map_datai, map_datai, map_br, map_br, map_ls, map_datar, map_ls, map_datafp
}

------------------------------------------------------------------------------

local map_regs = { x = {}, w = {}, d = {}, s = {} }

for i=0,30 do
  map_regs.x[i] = "x"..i
  map_regs.w[i] = "w"..i
  map_regs.d[i] = "d"..i
  map_regs.s[i] = "s"..i
end
map_regs.x[31] = "sp"
map_regs.w[31] = "wsp"
map_regs.d[31] = "d31"
map_regs.s[31] = "s31"

local map_cond = {
  [0] = "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc",
  "hi", "ls", "ge", "lt", "gt", "le", "al",
}

local map_shift = { [0] = "lsl", "lsr", "asr", "ror"}

local map_extend = {
  [0] = "uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx",
}

------------------------------------------------------------------------------

-- Output a nicely formatted line with an opcode and operands.
local function putop(ctx, text, operands)
  local pos = ctx.pos
  local extra = ""
  if ctx.rel then
    local sym = ctx.symtab[ctx.rel]
    if sym then
      extra = "\t->"..sym
    end
  end
  if ctx.hexdump > 0 then
    ctx.out(format("%08x  %s  %-5s %s%s\n",
      ctx.addr+pos, tohex(ctx.op), text, concat(operands, ", "), extra))
  else
    ctx.out(format("%08x  %-5s %s%s\n",
      ctx.addr+pos, text, concat(operands, ", "), extra))
  end
  ctx.pos = pos + 4
end

-- Fallback for unknown opcodes.
local function unknown(ctx)
  return putop(ctx, ".long", { "0x"..tohex(ctx.op) })
end

local function match_reg(p, pat, regnum)
  return map_regs[match(pat, p.."%w-([xwds])")][regnum]
end

local function fmt_hex32(x)
  if x < 0 then
    return tohex(x)
  else
    return format("%x", x)
  end
end

local imm13_rep = { 0x55555555, 0x11111111, 0x01010101, 0x00010001, 0x00000001 }

local function decode_imm13(op)
  local imms = band(rshift(op, 10), 63)
  local immr = band(rshift(op, 16), 63)
  if band(op, 0x00400000) == 0 then
    local len = 5
    if imms >= 56 then
      if imms >= 60 then len = 1 else len = 2 end
    elseif imms >= 48 then len = 3 elseif imms >= 32 then len = 4 end
    local l = lshift(1, len)-1
    local s = band(imms, l)
    local r = band(immr, l)
    local imm = ror(rshift(-1, 31-s), r)
    if len ~= 5 then imm = band(imm, lshift(1, l)-1) + rshift(imm, 31-l) end
    imm = imm * imm13_rep[len]
    local ix = fmt_hex32(imm)
    if rshift(op, 31) ~= 0 then
      return ix..tohex(imm)
    else
      return ix
    end
  else
    local lo, hi = -1, 0
    if imms < 32 then lo = rshift(-1, 31-imms) else hi = rshift(-1, 63-imms) end
    if immr ~= 0 then
      lo, hi = ror(lo, immr), ror(hi, immr)
      local x = immr == 32 and 0 or band(bxor(lo, hi), lshift(-1, 32-immr))
      lo, hi = bxor(lo, x), bxor(hi, x)
      if immr >= 32 then lo, hi = hi, lo end
    end
    if hi ~= 0 then
      return fmt_hex32(hi)..tohex(lo)
    else
      return fmt_hex32(lo)
    end
  end
end

local function parse_immpc(op, name)
  if name == "b" or name == "bl" then
    return arshift(lshift(op, 6), 4)
  elseif name == "adr" or name == "adrp" then
    local immlo = band(rshift(op, 29), 3)
    local immhi = lshift(arshift(lshift(op, 8), 13), 2)
    return bor(immhi, immlo)
  elseif name == "tbz" or name == "tbnz" then
    return lshift(arshift(lshift(op, 13), 18), 2)
  else
    return lshift(arshift(lshift(op, 8), 13), 2)
  end
end

local function parse_fpimm8(op)
  local sign = band(op, 0x100000) == 0 and 1 or -1
  local exp = bxor(rshift(arshift(lshift(op, 12), 5), 24), 0x80) - 131
  local frac = 16+band(rshift(op, 13), 15)
  return sign * frac * 2^exp
end

local function prefer_bfx(sf, uns, imms, immr)
  if imms < immr or imms == 31 or imms == 63 then
    return false
  end
  if immr == 0 then
    if sf == 0 and (imms == 7 or imms == 15) then
      return false
    end
    if sf ~= 0 and uns == 0 and (imms == 7 or imms == 15 or imms == 31) then
      return false
    end
  end
  return true
end

-- Disassemble a single instruction.
local function disass_ins(ctx)
  local pos = ctx.pos
  local b0, b1, b2, b3 = byte(ctx.code, pos+1, pos+4)
  local op = bor(lshift(b3, 24), lshift(b2, 16), lshift(b1, 8), b0)
  local operands = {}
  local suffix = ""
  local last, name, pat
  local map_reg
  ctx.op = op
  ctx.rel = nil
  last = nil
  local opat
  opat = map_init[band(rshift(op, 25), 15)]
  while type(opat) ~= "string" do
    if not opat then return unknown(ctx) end
    opat = opat[band(rshift(op, opat.shift), opat.mask)] or opat._
  end
  name, pat = match(opat, "^([a-z0-9]*)(.*)")
  local altname, pat2 = match(pat, "|([a-z0-9_.|]*)(.*)")
  if altname then pat = pat2 end
  if sub(pat, 1, 1) == "." then
    local s2, p2 = match(pat, "^([a-z0-9.]*)(.*)")
    suffix = suffix..s2
    pat = p2
  end

  local rt = match(pat, "[gf]")
  if rt then
    if rt == "g" then
      map_reg = band(op, 0x80000000) ~= 0 and map_regs.x or map_regs.w
    else
      map_reg = band(op, 0x400000) ~= 0 and map_regs.d or map_regs.s
    end
  end

  local second0, immr

  for p in gmatch(pat, ".") do
    local x = nil
    if p == "D" then
      local regnum = band(op, 31)
      x = rt and map_reg[regnum] or match_reg(p, pat, regnum)
    elseif p == "N" then
      local regnum = band(rshift(op, 5), 31)
      x = rt and map_reg[regnum] or match_reg(p, pat, regnum)
    elseif p == "M" then
      local regnum = band(rshift(op, 16), 31)
      x = rt and map_reg[regnum] or match_reg(p, pat, regnum)
    elseif p == "A" then
      local regnum = band(rshift(op, 10), 31)
      x = rt and map_reg[regnum] or match_reg(p, pat, regnum)
    elseif p == "B" then
      local addr = ctx.addr + pos + parse_immpc(op, name)
      ctx.rel = addr
      x = "0x"..tohex(addr)
    elseif p == "T" then
      x = bor(band(rshift(op, 26), 32), band(rshift(op, 19), 31))
    elseif p == "V" then
      x = band(op, 15)
    elseif p == "C" then
      x = map_cond[band(rshift(op, 12), 15)]
    elseif p == "c" then
      local rn = band(rshift(op, 5), 31)
      local rm = band(rshift(op, 16), 31)
      local cond = band(rshift(op, 12), 15)
      local invc = bxor(cond, 1)
      x = map_cond[cond]
      if altname and cond ~= 14 and cond ~= 15 then
	local a1, a2 = match(altname, "([^|]*)|(.*)")
	if rn == rm then
	  local n = #operands
	  operands[n] = nil
	  x = map_cond[invc]
	  if rn ~= 31 then
	    if a1 then name = a1 else name = altname end
	  else
	    operands[n-1] = nil
	    name = a2
	  end
	end
      end
    elseif p == "W" then
      x = band(rshift(op, 5), 0xffff)
    elseif p == "Y" then
      x = band(rshift(op, 5), 0xffff)
      local hw = band(rshift(op, 21), 3)
      if altname and (hw == 0 or x ~= 0) then
	name = altname
      end
    elseif p == "L" then
      local rn = map_regs.x[band(rshift(op, 5), 31)]
      local imm9 = arshift(lshift(op, 11), 23)
      if band(op, 0x800) ~= 0 then
	x = "["..rn..", #"..imm9.."]!"
      else
	x = "["..rn.."], #"..imm9
      end
    elseif p == "U" then
      local rn = map_regs.x[band(rshift(op, 5), 31)]
      local sz = band(rshift(op, 30), 3)
      local imm12 = lshift(rshift(lshift(op, 10), 20), sz)
      if imm12 ~= 0 then
	x = "["..rn..", #"..imm12.."]"
      else
	x = "["..rn.."]"
      end
    elseif p == "K" then
      local rn = map_regs.x[band(rshift(op, 5), 31)]
      local imm9 = arshift(lshift(op, 11), 23)
      if imm9 ~= 0 then
	x = "["..rn..", #"..imm9.."]"
      else
	x = "["..rn.."]"
      end
    elseif p == "O" then
      local rn, rm = map_regs.x[band(rshift(op, 5), 31)]
      local m = band(rshift(op, 13), 1)
      if m == 0 then
	rm = map_regs.w[band(rshift(op, 16), 31)]
      else
	rm = map_regs.x[band(rshift(op, 16), 31)]
      end
      x = "["..rn..", "..rm
      local opt = band(rshift(op, 13), 7)
      local s = band(rshift(op, 12), 1)
      local sz = band(rshift(op, 30), 3)
      -- extension to be applied
      if opt == 3 then
       if s == 0 then x = x.."]"
       else x = x..", lsl #"..sz.."]" end
      elseif opt == 2 or opt == 6 or opt == 7 then
	if s == 0 then x = x..", "..map_extend[opt].."]"
	else x = x..", "..map_extend[opt].." #"..sz.."]" end
      else
	x = x.."]"
      end
    elseif p == "P" then
      local sh = 2 + rshift(op, 31 - band(rshift(op, 26), 1))
      local imm7 = lshift(arshift(lshift(op, 10), 25), sh)
      local rn = map_regs.x[band(rshift(op, 5), 31)]
      local ind = band(rshift(op, 23), 3)
      if ind == 1 then
	x = "["..rn.."], #"..imm7
      elseif ind == 2 then
	if imm7 == 0 then
	  x = "["..rn.."]"
	else
	  x = "["..rn..", #"..imm7.."]"
	end
      elseif ind == 3 then
	x = "["..rn..", #"..imm7.."]!"
      end
    elseif p == "I" then
      local shf = band(rshift(op, 22), 3)
      local imm12 = band(rshift(op, 10), 0x0fff)
      local rn, rd = band(rshift(op, 5), 31), band(op, 31)
      if altname == "mov" and shf == 0 and imm12 == 0 and (rn == 31 or rd == 31) then
	name = altname
	x = nil
      elseif shf == 0 then
	x = imm12
      elseif shf == 1 then
	x = imm12..", lsl #12"
      end
    elseif p == "i" then
      x = "#0x"..decode_imm13(op)
    elseif p == "1" then
      immr = band(rshift(op, 16), 63)
      x = immr
    elseif p == "2" then
      x = band(rshift(op, 10), 63)
      if altname then
	local a1, a2, a3, a4, a5, a6 =
	  match(altname, "([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)|(.*)")
	local sf = band(rshift(op, 26), 32)
	local uns = band(rshift(op, 30), 1)
	if prefer_bfx(sf, uns, x, immr) then
	  name = a2
	  x = x - immr + 1
	elseif immr == 0 and x == 7 then
	  local n = #operands
	  operands[n] = nil
	  if sf ~= 0 then
	    operands[n-1] = gsub(operands[n-1], "x", "w")
	  end
	  last = operands[n-1]
	  name = a6
	  x = nil
	elseif immr == 0 and x == 15 then
	  local n = #operands
	  operands[n] = nil
	  if sf ~= 0 then
	    operands[n-1] = gsub(operands[n-1], "x", "w")
	  end
	  last = operands[n-1]
	  name = a5
	  x = nil
	elseif x == 31 or x == 63 then
	  if x == 31 and immr == 0 and name == "sbfm" then
	    name = a4
	    local n = #operands
	    operands[n] = nil
	    if sf ~= 0 then
	      operands[n-1] = gsub(operands[n-1], "x", "w")
	    end
	    last = operands[n-1]
	  else
	    name = a3
	  end
	  x = nil
	elseif band(x, 31) ~= 31 and immr == x+1 and name == "ubfm" then
	  name = a4
	  last = "#"..(sf+32 - immr)
	  operands[#operands] = last
	  x = nil
	elseif x < immr then
	  name = a1
	  last = "#"..(sf+32 - immr)
	  operands[#operands] = last
	  x = x + 1
	end
      end
    elseif p == "3" then
      x = band(rshift(op, 10), 63)
      if altname then
	local a1, a2 = match(altname, "([^|]*)|(.*)")
	if x < immr then
	  name = a1
	  local sf = band(rshift(op, 26), 32)
	  last = "#"..(sf+32 - immr)
	  operands[#operands] = last
	  x = x + 1
	else
	  name = a2
	  x = x - immr + 1
	end
      end
    elseif p == "4" then
      x = band(rshift(op, 10), 63)
      local rn = band(rshift(op, 5), 31)
      local rm = band(rshift(op, 16), 31)
      if altname and rn == rm then
	local n = #operands
	operands[n] = nil
	last = operands[n-1]
	name = altname
      end
    elseif p == "5" then
      x = band(rshift(op, 16), 31)
    elseif p == "S" then
      x = band(rshift(op, 10), 63)
      if x == 0 then x = nil
      else x = map_shift[band(rshift(op, 22), 3)].." #"..x end
    elseif p == "X" then
      local opt = band(rshift(op, 13), 7)
      -- Width specifier <R>.
      if opt ~= 3 and opt ~= 7 then
	last = map_regs.w[band(rshift(op, 16), 31)]
	operands[#operands] = last
      end
      x = band(rshift(op, 10), 7)
      -- Extension.
      if opt == 2 + band(rshift(op, 31), 1) and
	 band(rshift(op, second0 and 5 or 0), 31) == 31 then
	if x == 0 then x = nil
	else x = "lsl #"..x end
      else
	if x == 0 then x = map_extend[band(rshift(op, 13), 7)]
	else x = map_extend[band(rshift(op, 13), 7)].." #"..x end
      end
    elseif p == "R" then
      x = band(rshift(op,21), 3)
      if x == 0 then x = nil
      else x = "lsl #"..x*16 end
    elseif p == "z" then
      local n = #operands
      if operands[n] == "sp" then operands[n] = "xzr"
      elseif operands[n] == "wsp" then operands[n] = "wzr"
      end
    elseif p == "Z" then
      x = 0
    elseif p == "F" then
      x = parse_fpimm8(op)
    elseif p == "g" or p == "f" or p == "x" or p == "w" or
	   p == "d" or p == "s" then
      -- These are handled in D/N/M/A.
    elseif p == "0" then
      if last == "sp" or last == "wsp" then
	local n = #operands
	operands[n] = nil
	last = operands[n-1]
	if altname then
	  local a1, a2 = match(altname, "([^|]*)|(.*)")
	  if not a1 then
	    name = altname
	  elseif second0 then
	    name, altname = a2, a1
	  else
	    name, altname = a1, a2
	  end
	end
      end
      second0 = true
    else
      assert(false)
    end
    if x then
      last = x
      if type(x) == "number" then x = "#"..x end
      operands[#operands+1] = x
    end
  end

  return putop(ctx, name..suffix, operands)
end

------------------------------------------------------------------------------

-- Disassemble a block of code.
local function disass_block(ctx, ofs, len)
  if not ofs then ofs = 0 end
  local stop = len and ofs+len or #ctx.code
  ctx.pos = ofs
  ctx.rel = nil
  while ctx.pos < stop do disass_ins(ctx) end
end

-- Extended API: create a disassembler context. Then call ctx:disass(ofs, len).
local function create(code, addr, out)
  local ctx = {}
  ctx.code = code
  ctx.addr = addr or 0
  ctx.out = out or io.write
  ctx.symtab = {}
  ctx.disass = disass_block
  ctx.hexdump = 8
  return ctx
end

-- Simple API: disassemble code (a string) at address and output via out.
local function disass(code, addr, out)
  create(code, addr, out):disass()
end

-- Return register name for RID.
local function regname(r)
  if r < 32 then return map_regs.x[r] end
  return map_regs.d[r-32]
end

-- Public module functions.
return {
  create = create,
  disass = disass,
  regname = regname
}

