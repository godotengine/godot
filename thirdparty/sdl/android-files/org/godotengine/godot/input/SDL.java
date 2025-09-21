package org.godotengine.godot.input;

import android.content.Context;

import java.lang.Class;
import java.lang.reflect.Method;

/**
    SDL library initialization
*/
public class SDL {

    // This function should be called first and sets up the native code
    // so it can call into the Java classes
    public static void setupJNI() {
        //SDLActivity.nativeSetupJNI();
        SDLAudioManager.nativeSetupJNI();
        SDLControllerManager.nativeSetupJNI();
    }

    // This function should be called each time the activity is started
    public static void initialize() {
        setContext(null);

        //SDLActivity.initialize();
        SDLAudioManager.initialize();
        SDLControllerManager.initialize();
    }

    // This function stores the current activity (SDL or not)
    public static void setContext(Context context) {
        SDLAudioManager.setContext(context);
        mContext = context;
    }

    public static Context getContext() {
        return mContext;
    }

    public static void loadLibrary(String libraryName) throws UnsatisfiedLinkError, SecurityException, NullPointerException {
        loadLibrary(libraryName, mContext);
    }

    public static void loadLibrary(String libraryName, Context context) throws UnsatisfiedLinkError, SecurityException, NullPointerException {

        if (libraryName == null) {
            throw new NullPointerException("No library name provided.");
        }

        try {
            // Let's see if we have ReLinker available in the project.  This is necessary for
            // some projects that have huge numbers of local libraries bundled, and thus may
            // trip a bug in Android's native library loader which ReLinker works around.  (If
            // loadLibrary works properly, ReLinker will simply use the normal Android method
            // internally.)
            //
            // To use ReLinker, just add it as a dependency.  For more information, see
            // https://github.com/KeepSafe/ReLinker for ReLinker's repository.
            //
            Class<?> relinkClass = context.getClassLoader().loadClass("com.getkeepsafe.relinker.ReLinker");
            Class<?> relinkListenerClass = context.getClassLoader().loadClass("com.getkeepsafe.relinker.ReLinker$LoadListener");
            Class<?> contextClass = context.getClassLoader().loadClass("android.content.Context");
            Class<?> stringClass = context.getClassLoader().loadClass("java.lang.String");

            // Get a 'force' instance of the ReLinker, so we can ensure libraries are reinstalled if
            // they've changed during updates.
            Method forceMethod = relinkClass.getDeclaredMethod("force");
            Object relinkInstance = forceMethod.invoke(null);
            Class<?> relinkInstanceClass = relinkInstance.getClass();

            // Actually load the library!
            Method loadMethod = relinkInstanceClass.getDeclaredMethod("loadLibrary", contextClass, stringClass, stringClass, relinkListenerClass);
            loadMethod.invoke(relinkInstance, context, libraryName, null, null);
        }
        catch (final Throwable e) {
            // Fall back
            try {
                System.loadLibrary(libraryName);
            }
            catch (final UnsatisfiedLinkError ule) {
                throw ule;
            }
            catch (final SecurityException se) {
                throw se;
            }
        }
    }

    protected static Context mContext;
}
