# Protect Godot Engine Core
-keep class com.godot.** { *; }
-keep class org.godotengine.** { *; }
-keep class ** extends org.godotengine.godot.plugin.GodotPlugin { *; }

# Protect methods exposed to Godot via @UsedByGodot
-keepattributes *Annotation*
-keepclassmembers class * {
    @org.godotengine.godot.plugin.UsedByGodot *;
}

# Protect JNI bindings
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep public class * extends android.app.Activity
-keep public class * extends android.app.Application
-keep public class * extends android.app.Service
