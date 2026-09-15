import sys

try:
    sys.path.insert(0, "./")
    import methods
except ImportError:
    raise SystemExit(f'Generator script "{__file__}" must be run from repository root!')

import argparse

proto_mod = """
#define MODBIND$VER($RETTYPE m_name$ARG) \\
virtual $RETVAL _##m_name($FUNCARGS) $CONST; \\
_FORCE_INLINE_ virtual $RETVAL m_name($FUNCARGS) $CONST override { \\
    $RETX _##m_name($CALLARGS);\\
}
"""


def generate_mod_version(argcount, const=False, returns=False):
    s = proto_mod
    sproto = str(argcount)
    if returns:
        sproto += "R"
        s = s.replace("$RETTYPE", "m_ret, ")
        s = s.replace("$RETVAL", "m_ret")
        s = s.replace("$RETX", "return")

    else:
        s = s.replace("$RETTYPE", "")
        s = s.replace("$RETVAL", "void")
        s = s.replace("$RETX", "")

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


proto_ex = """
#define EXBIND$VER($RETTYPE m_name$ARG) \\
GDVIRTUAL$VER_REQUIRED($RETTYPE_##m_name$ARG)\\
virtual $RETVAL m_name($FUNCARGS) $CONST override { \\
    $RETPRE\\
    GDVIRTUAL_CALL(_##m_name$CALLARGS$RETREF);\\
    $RETPOST\\
}
"""


def generate_ex_version(argcount, const=False, returns=False):
    s = proto_ex
    sproto = str(argcount)
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


def wrapper(target: str) -> None:
    max_versions = 12

    txt = ""

    for i in range(max_versions + 1):
        txt += "\n/* Extension Wrapper " + str(i) + " Arguments */\n"
        txt += generate_ex_version(i, False, False)
        txt += generate_ex_version(i, False, True)
        txt += generate_ex_version(i, True, False)
        txt += generate_ex_version(i, True, True)

    for i in range(max_versions + 1):
        txt += "\n/* Module Wrapper " + str(i) + " Arguments */\n"
        txt += generate_mod_version(i, False, False)
        txt += generate_mod_version(i, False, True)
        txt += generate_mod_version(i, True, False)
        txt += generate_mod_version(i, True, True)

    with methods.generated_wrapper(target) as f:
        f.write(txt)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    wrapper_parser = subparsers.add_parser("wrapper")
    wrapper_parser.add_argument("target")

    args = vars(parser.parse_args())
    command = globals().get(args.pop("command"), {})
    command(**args)


if __name__ == "__main__":
    main()
