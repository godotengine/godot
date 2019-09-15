# config.py

# @author Cagdas Caglak <cagdascaglak@gmail.com>

def can_build(env, platform):
    return True

def configure(env):
    if env['platform'] == "android":
        env.android_add_dependency("implementation 'com.google.android.gms:play-services-location:17.0.0'") # Android location service dependency with androidx
    if env['platform'] == "iphone":
        env.Append(LINKFLAGS=['-ObjC', '-framework','CoreLocation'])

    pass
