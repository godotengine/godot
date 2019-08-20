-dontwarn org.godotengine.godot.**

-keep class org.godotengine.godot.** {*;}
-keep class * extends java.util.ListResourceBundle {
   protected java.lang.Object[][] getContents();
}

-keepnames class * implements android.os.Parcelable {
   public static final ** CREATOR;
}
-keepattributes *Annotation*

-optimizations !code/allocation/variable
