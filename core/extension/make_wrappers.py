proto_mod = """#define MODBIND$VER($RETTYPEm_name$ARG)\\
	virtual $RETVAL _##m_name($FUNCARGS) $CONST;\\
	_FORCE_INLINE_ virtual $RETVAL m_name($FUNCARGS) $CONST override {\\
		$RETX _##m_name($CALLARGS);\\
	}

"""


def generate_mod_version(argcount, const=False, returns=False):
    s = proto_mod
    sproto = str(argcount)
    method_info = ""
    if returns:
        sproto += "R"
        s = s.replace("$RETTYPE", "m_ret, ")
        s = s.replace("$RETVAL", "m_ret")
        s = s.replace("$RETX", "return")

    else:
        s = s.replace("$RETTYPE", "")
        s = s.replace("$RETVAL", "void")
        s = s.replace("$RETX ", "")

    if const:
        sproto += "C"
        s = s.replace("$CONST", "const")
    else:
        s = s.replace(" $CONST", "")

    s = s.replace("$VER", sproto)
    argtext = ""
    funcargs = ""
    callargs = ""

    for i in range(argcount):
        if i > 0:
            funcargs += ", "
            callargs += ", "

        argtext += ", m_type" + str(i + 1)
        funcargs += "m_type" + str(i + 1) + " arg" + str(i + 1)
        callargs += "arg" + str(i + 1)

    if argcount:
        s = s.replace("$ARG", argtext)
        s = s.replace("$FUNCARGS", funcargs)
        s = s.replace("$CALLARGS", callargs)
    else:
        s = s.replace("$ARG", "")
        s = s.replace("$FUNCARGS", funcargs)
        s = s.replace("$CALLARGS", callargs)

    return s


proto_ex = """#define EXBIND$VER($RETTYPEm_name$ARG)\\
	GDVIRTUAL$VER($RETTYPE_##m_name$ARG)\\
	virtual $RETVAL m_name($FUNCARGS) $CONST override {\\
		$RETPRE\\
		GDVIRTUAL_REQUIRED_CALL(_##m_name$CALLARGS$RETREF);\\
		$RETPOST\\
	}

"""


def generate_ex_version(argcount, const=False, returns=False):
    s = proto_ex
    sproto = str(argcount)
    method_info = ""
    if returns:
        sproto += "R"
        s = s.replace("$RETTYPE", "m_ret, ")
        s = s.replace("$RETVAL", "m_ret")
        s = s.replace("$RETPRE", "m_ret ret;\\\n\t\tZeroInitializer<m_ret>::initialize(ret);")
        s = s.replace("$RETPOST", "return ret;")

    else:
        s = s.replace("$RETTYPE", "")
        s = s.replace("$RETVAL", "void")
        s = s.replace("\t\t$RETPRE\\\n", "")
        s = s.replace("$RETPOST", "return;")

    if const:
        sproto += "C"
        s = s.replace("$CONST", "const")
    else:
        s = s.replace("$CONST ", "")

    s = s.replace("$VER", sproto)
    argtext = ""
    funcargs = ""
    callargs = ""

    for i in range(argcount):
        if i > 0:
            funcargs += ", "

        argtext += ", m_type" + str(i + 1)
        funcargs += "m_type" + str(i + 1) + " arg" + str(i + 1)
        callargs += ", arg" + str(i + 1)

    if argcount:
        s = s.replace("$ARG", argtext)
        s = s.replace("$FUNCARGS", funcargs)
        s = s.replace("$CALLARGS", callargs)
    else:
        s = s.replace("$ARG", "")
        s = s.replace("$FUNCARGS", funcargs)
        s = s.replace("$CALLARGS", callargs)

    if returns:
        s = s.replace("$RETREF", ", ret")
    else:
        s = s.replace("$RETREF", "")

    return s


def run(target, source, env):
    from methods import format_defines

    max_versions = 12

    txt = """/* THIS FILE IS GENERATED DO NOT EDIT */
#ifndef GDEXTENSION_WRAPPERS_GEN_H
#define GDEXTENSION_WRAPPERS_GEN_H

"""

    for i in range(max_versions + 1):
        txt += "/* Extension Wrapper " + str(i) + " Arguments */\n\n"
        txt += generate_ex_version(i, False, False)
        txt += generate_ex_version(i, False, True)
        txt += generate_ex_version(i, True, False)
        txt += generate_ex_version(i, True, True)

    for i in range(max_versions + 1):
        txt += "/* Module Wrapper " + str(i) + " Arguments */\n\n"
        txt += generate_mod_version(i, False, False)
        txt += generate_mod_version(i, False, True)
        txt += generate_mod_version(i, True, False)
        txt += generate_mod_version(i, True, True)

    txt += "#endif // GDEXTENSION_WRAPPERS_GEN_H\n"

    with open(target[0], "w") as f:
        f.write(format_defines(txt))


if __name__ == "__main__":
    from platform_methods import subprocess_main

    subprocess_main(globals())
