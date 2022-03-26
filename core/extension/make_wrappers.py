proto = """
#define EXBIND$VER($RETTYPE m_name$ARG) \\
GDVIRTUAL$VER($RETTYPE_##m_name$ARG)\\
virtual $RETVAL m_name($FUNCARGS) $CONST override { \\
    $RETPRE\\
    GDVIRTUAL_REQUIRED_CALL(_##m_name$CALLARGS$RETREF);\\
    $RETPOST\\
}
"""


def generate_version(argcount, const=False, returns=False):
    s = proto
    sproto = str(argcount)
    method_info = ""
    if returns:
        sproto += "R"
        s = s.replace("$RETTYPE", "m_ret, ")
        s = s.replace("$RETVAL", "m_ret")
        s = s.replace("$RETPRE", "m_ret ret; ZeroInitializer<m_ret>::initialize(ret);\\\n")
        s = s.replace("$RETPOST", "return ret;\\\n")

    else:
        s = s.replace("$RETTYPE", "")
        s = s.replace("$RETVAL", "void")
        s = s.replace("$RETPRE", "")
        s = s.replace("$RETPOST", "return;")

    if const:
        sproto += "C"
        s = s.replace("$CONST", "const")
    else:
        s = s.replace("$CONST", "")

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

    max_versions = 12

    txt = """
#ifndef GDEXTENSION_WRAPPERS_GEN_H
#define GDEXTENSION_WRAPPERS_GEN_H


"""

    for i in range(max_versions + 1):

        txt += "/* " + str(i) + " Arguments */\n\n"
        txt += generate_version(i, False, False)
        txt += generate_version(i, False, True)
        txt += generate_version(i, True, False)
        txt += generate_version(i, True, True)

    txt += "#endif"

    with open(target[0], "w") as f:
        f.write(txt)


if __name__ == "__main__":
    from platform_methods import subprocess_main

    subprocess_main(globals())
