def can_build(env, platform):
    if platform == 'windows' and not env.msvc:
        print("[FIREBASE][WARN] doesn't support cross compiling. skip this module. to use this module, use MSVC on windows")
        return False
    return True

def configure(env):
    if env['platform'] == 'android':
        env.android_add_gradle_classpath("com.google.gms:google-services:4.2.0")
        env.android_add_java_dir("android/java/src")
        env.android_add_res_dir("android/res")
        env.android_add_to_manifest("android/manifest/AndroidManifestChunk.xml")
        dependencies = [
                "implementation 'com.android.support:preference-v7:28.0.0'", # avoid dependencies conflict
                "implementation 'com.google.firebase:firebase-core:16.0.6'",
                "implementation 'com.google.firebase:firebase-auth:16.1.0'",
                "implementation 'com.google.android.gms:play-services-auth:16.0.1'",
                "implementation 'com.facebook.android:facebook-login:[4,5)'",
        ]
        for x in dependencies:
            env.android_dependencies.append(x)
