#!/usr/bin/env python
import os
import sys

env = SConscript("godot-cpp/SConstruct")

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags

# Generate version header.
sys.path.append("./limboai")
import limboai_version

os.chdir("./limboai")
limboai_version.generate_module_version_header()
os.chdir("..")
sys.path.remove("./limboai")

# Tweak this if you want to use different folders, or more folders, to store your source code in.
env.Append(CPPPATH=["limboai/"])
env.Append(CPPDEFINES=["LIMBOAI_GDEXTENSION"])
sources = Glob("limboai/*.cpp")
sources += Glob("limboai/blackboard/*.cpp")
sources += Glob("limboai/blackboard/bb_param/*.cpp")
sources += Glob("limboai/bt/*.cpp")
sources += Glob("limboai/bt/tasks/*.cpp")
sources += Glob("limboai/bt/tasks/blackboard/*.cpp")
sources += Glob("limboai/bt/tasks/composites/*.cpp")
sources += Glob("limboai/bt/tasks/decorators/*.cpp")
sources += Glob("limboai/bt/tasks/scene/*.cpp")
sources += Glob("limboai/bt/tasks/utility/*.cpp")
sources += Glob("limboai/gdextension/*.cpp")
sources += Glob("limboai/editor/debugger/*.cpp")
sources += Glob("limboai/editor/*.cpp")
sources += Glob("limboai/hsm/*.cpp")
sources += Glob("limboai/util/*.cpp")

# Generate documentation header.
if env["target"] in ["editor", "template_debug"]:
    doc_data = env.GodotCPPDocData("limboai/gen/doc_data.gen.cpp", source=Glob("limboai/doc_classes/*.xml"))
    sources.append(doc_data)

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "demo/addons/limboai/bin/liblimboai.{}.{}.framework/liblimboai.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "demo/addons/limboai/bin/liblimboai{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)
