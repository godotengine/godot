from __future__ import print_function


def supported(result):
    return "supported" if result else "not supported"


def check_cxx11_thread_local(conf):
    print("Checking for `thread_local` support...", end=" ")
    result = conf.TryCompile("thread_local int foo = 0; int main() { return foo; }", ".cpp")
    print(supported(result))
    return bool(result)


def check_declspec_thread(conf):
    print("Checking for `__declspec(thread)` support...", end=" ")
    result = conf.TryCompile("__declspec(thread) int foo = 0; int main() { return foo; }", ".cpp")
    print(supported(result))
    return bool(result)


def check_gcc___thread(conf):
    print("Checking for `__thread` support...", end=" ")
    result = conf.TryCompile("__thread int foo = 0; int main() { return foo; }", ".cpp")
    print(supported(result))
    return bool(result)


def configure(conf):
    if check_cxx11_thread_local(conf):
        conf.env.Append(CPPDEFINES=["HAVE_CXX11_THREAD_LOCAL"])
    else:
        if conf.env.msvc:
            if check_declspec_thread(conf):
                conf.env.Append(CPPDEFINES=["HAVE_DECLSPEC_THREAD"])
        elif check_gcc___thread(conf):
            conf.env.Append(CPPDEFINES=["HAVE_GCC___THREAD"])
