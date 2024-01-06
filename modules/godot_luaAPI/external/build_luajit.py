import sys
import os
import platform

def run(cmd):
    print("Running: %s" % cmd)
    res = os.system(cmd)
    code = 0
    if (os.name == 'nt'):
        code = res
    else:
        code = os.WEXITSTATUS(res)
    if code != 0:
        print("Error: return code: " + str(code))
        sys.exit(code)

def build_luajit(env, extension=False):
    if extension or not env.msvc:
        os.chdir("luaJIT_riscv")
        
        # cross compile posix->windows
        if (os.name == 'posix') and env['platform'] == 'windows':
            host_arch = platform.machine()
            run("make clean")
            if (host_arch != env['arch']):
                if (host_arch == 'x86_64' and env['arch'] == 'x86_32'):
                    host_cc = env['luaapi_host_cc'] + ' -m32'
                    run('make HOST_CC="%s" CROSS="%s" BUILDMODE="static" TARGET_SYS=Windows' % (host_cc, env['CC'].replace("-gcc", "-").replace("-clang", "-")))

                else:
                    print("ERROR: Unsupported cross compile!")
                    sys.exit(-1)
            else:
                run('make HOST_CC="%s" CROSS="%s" BUILDMODE="static" TARGET_SYS=Windows' % (env['luaapi_host_cc'], env['CC'].replace("-gcc", "-").replace("-clang", "-")))

        elif env['platform']=='macos':
            run("make clean MACOSX_DEPLOYMENT_TARGET=10.12")
            arch = env['arch']
            if arch == "universal":
                run('make CC="%s" TARGET_FLAGS="-arch x86_64" MACOSX_DEPLOYMENT_TARGET=10.12' % (env['CC']))
                run('mv src/libluajit.a src/libluajit64.a')
                run('make clean MACOSX_DEPLOYMENT_TARGET=10.12')
                run('make CC="%s" TARGET_FLAGS="-arch arm64" MACOSX_DEPLOYMENT_TARGET=10.12' % (env['CC']))
                run('lipo -create src/libluajit.a src/libluajit64.a -output src/libluajit.a')
                run('rm src/libluajit64.a')
            else:
                run('make CC="%s" TARGET_FLAGS="-arch %s" MACOSX_DEPLOYMENT_TARGET=10.12' % (env['CC'], arch))
        elif env['platform']=='linuxbsd' or env['platform']=='linux':
            host_arch = platform.machine()
            run("make clean")
            if (host_arch != env['arch']):
                if (host_arch == 'x86_64' and env['arch'] == 'x86_32'):
                    host_cc = env['luaapi_host_cc'] + ' -m32'
                    run('make HOST_CC="%s" CROSS="%s" BUILDMODE="static"' % (host_cc, env['CC'].replace("-gcc", "-").replace("-clang", "-")))

                else:
                    print("ERROR: Unsupported cross compile!")
                    sys.exit(-1)

            else:
                run('make CC="%s" BUILDMODE="static"' % env['CC'])

        else:
            print("ERROR: Unsupported platform '%s'." % env['platform'])
            sys.exit(-1)
    else:
        os.chdir("luaJIT_riscv/src")
        run("msvcbuild static")
