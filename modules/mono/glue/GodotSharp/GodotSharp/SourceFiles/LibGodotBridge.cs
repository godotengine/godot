#if LIBGODOT_ENABLED
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GodotPlugins.Game
{
    // Wrapper for Marshal.StringToHGlobalAnsi/Marshal.FreeHGlobal.
    public sealed class AnsiString : IDisposable
    {
        public nint Ptr { get; private set; } = nint.Zero;

        public AnsiString(string value)
        {
            Ptr = Marshal.StringToHGlobalAnsi(value);
        }

        ~AnsiString()
        {
            Dispose(false);
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        void Dispose(bool _)
        {
            if (Ptr != nint.Zero)
            {
                Marshal.FreeHGlobal(Ptr);
                Ptr = nint.Zero;
            }
        }

        public override string ToString()
        {
            return Marshal.PtrToStringAnsi(Ptr) ?? "Error";
        }

        public static implicit operator nint(AnsiString ansiString) => ansiString.Ptr;
    }

    public unsafe class GodotLibrary
    {
        public nint GetProcAddress { get; }
        public nint Token { get; }
        private delegate* unmanaged<nint, nint> GetProcAddressDelegate { get; }

        public GodotLibrary(nint getProcAddress, nint token)
        {
            GetProcAddress = getProcAddress;
            Token = token;
            GetProcAddressDelegate = (delegate* unmanaged<nint, nint>)getProcAddress;
        }

        public nint LoadFunction(string name)
        {
            using AnsiString ansiName = new(name);
            nint fnPtr = GetProcAddressDelegate(ansiName);
            // Console.WriteLine($"Loaded {name}");
            return fnPtr;
        }
    }

    public enum GDExtensionInitializationLevel
    {
        Core,
        Servers,
        Scene,
        Editor,
        Max,
    }

    public enum GDExtensionVariantType
    {
        Nil = 0,
        Bool = 1,
        Int = 2,
        Float = 3,
        String = 4,
        Vector2 = 5,
        Vector2I = 6,
        Rect2 = 7,
        Rect2I = 8,
        Vector3 = 9,
        Vector3I = 10,
        TRANSFORM2D = 11,
        Vector4 = 12,
        Vector4I = 13,
        Plane = 14,
        Quaternion = 15,
        Aabb = 16,
        Basis = 17,
        TRANSFORM3D = 18,
        Projection = 19,
        Color = 20,
        StringName = 21,
        NodePath = 22,
        Rid = 23,
        Object = 24,
        Callable = 25,
        Signal = 26,
        Dictionary = 27,
        Array = 28,
        PackedByteArray = 29,
        PackedInt32Array = 30,
        PackedInt64Array = 31,
        PackedFloat32Array = 32,
        PackedFloat64Array = 33,
        PackedStringArray = 34,
        PackedVector2Array = 35,
        PackedVector3Array = 36,
        PackedColorArray = 37,
        PackedVECTOR4Array = 38,
        VariantMax = 39,
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct GDExtensionInitialization
    {
        public GDExtensionInitializationLevel MinimumInitializationLevel;
        public nint Userdata;
        public delegate* unmanaged<nint, GDExtensionInitializationLevel, void> Initialize;
        public delegate* unmanaged<nint, GDExtensionInitializationLevel, void> Deinitialize;
    }

    public class StringName : IDisposable
    {
        private unsafe static class Bindings
        {
            public static delegate* unmanaged<nint, nint, byte, void> string_name_new_with_latin1_chars;
            public static delegate* unmanaged<nint, void> destructor;

            public static void Initialize()
            {
                string_name_new_with_latin1_chars = (delegate* unmanaged<nint, nint, byte, void>)SimpleInterface.Library.LoadFunction("string_name_new_with_latin1_chars");
                destructor = SimpleInterface.variant_get_ptr_destructor(GDExtensionVariantType.StringName);
            }
        }
        public nint Ptr { get; private set; }

        public unsafe StringName(string content)
        {
            using AnsiString ansiString = new(content);
            Ptr = Marshal.AllocHGlobal(Unsafe.SizeOf<nint>());
            Bindings.string_name_new_with_latin1_chars(Ptr, ansiString.Ptr, 0);
        }

        ~StringName()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private unsafe void Dispose(bool _)
        {
            if (Ptr != nint.Zero)
            {
                Bindings.destructor(Ptr);
                Marshal.FreeHGlobal(Ptr);
                Ptr = nint.Zero;
            }
        }

        public static void InitializeBindings()
        {
            Bindings.Initialize();
        }
    }

    // GodotInstance implementation.
    public unsafe partial class GodotInstance : IDisposable
    {
        // Import LibGodot functions.
        [LibraryImport("libgodot")]
        private static partial nint libgodot_create_godot_instance(int argc, nint* argv, nint initFunc);
        [LibraryImport("libgodot")]
        private static partial void libgodot_destroy_godot_instance(nint godotInstance);

        private static class Bindings
        {
            public static nint mbStart;
            public static nint mbIsStarted;
            public static nint mbIteration;
            public static nint mbFocusIn;
            public static nint mbFocusOut;
            public static nint mbPause;
            public static nint mbResume;

            public static void Initialize()
            {
                using StringName nativeName = new("GodotInstance");

                using StringName startName = new("start");
                // Console.WriteLine("GodotInstance before start");
                mbStart = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, startName.Ptr, 2240911060L);

                using StringName isStartedName = new("is_started");
                // Console.WriteLine("GodotInstance before is_started");
                mbIsStarted = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, isStartedName.Ptr, 2240911060L);

                using StringName iteratiomName = new("iteration");
                // Console.WriteLine("GodotInstance before iteration");
                mbIteration = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, iteratiomName.Ptr, 2240911060L);

                using StringName focusInName = new("focus_in");
                // Console.WriteLine("GodotInstance before focus_in");
                mbFocusIn = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, focusInName.Ptr, 3218959716L);

                using StringName focusOutName = new("focus_out");
                // Console.WriteLine("GodotInstance before focus_out");
                mbFocusOut = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, focusOutName.Ptr, 3218959716L);

                using StringName pauseName = new("pause");
                // Console.WriteLine("GodotInstance before pause");
                mbPause = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, pauseName.Ptr, 3218959716L);

                using StringName resumeName = new("resume");
                // Console.WriteLine("GodotInstance before resume");
                mbResume = SimpleInterface.classdb_get_method_bind(nativeName.Ptr, resumeName.Ptr, 3218959716L);
            }
        }

        private nint owner;

        private GodotInstance(nint owner)
        {
            this.owner = owner;
        }

        public static GodotInstance? CreateInstance(string[] args, delegate* unmanaged<nint, nint, GDExtensionInitialization*, byte> gdExtensionInit)
        {
            AnsiString[] argsAnsi = [.. args.Select(arg => new AnsiString(arg))];
            nint[] argsPtrs = [.. argsAnsi.Select(arg => arg.Ptr)];
            try
            {
                fixed (nint* argsBegin = argsPtrs)
                {
                    // Console.WriteLine($"Before libgodot_create_godot_instance {argsPtrs.Length}");
                    nint instance = libgodot_create_godot_instance(argsPtrs.Length, argsBegin, (nint)gdExtensionInit);
                    // Console.WriteLine("After libgodot_create_godot_instance");
                    if (instance == nint.Zero) { return null; }
                    return new GodotInstance(instance);
                }
            }
            finally
            {
                foreach (var arg in argsAnsi)
                {
                    arg.Dispose();
                }
            }
        }

        ~GodotInstance()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool _)
        {
            if (owner != nint.Zero)
            {
                libgodot_destroy_godot_instance(owner);
                owner = nint.Zero;
            }
        }

        public static void InitializeBindings()
        {
            Bindings.Initialize();
        }

        public bool Start()
        {
            nint* args = stackalloc nint[0];
            byte ret = 0;
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbStart, owner, args, (nint)(void*)&ret);
            return ret != 0;
        }
        public bool IsStarted()
        {
            nint* args = stackalloc nint[0];
            byte ret = 0;
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbIsStarted, owner, args, (nint)(void*)&ret);
            return ret != 0;
        }
        public bool Iteration()
        {
            nint* args = stackalloc nint[0];
            byte ret = 0;
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbIteration, owner, args, (nint)(void*)&ret);
            return ret != 0;
        }
        public void FocusIn()
        {
            nint* args = stackalloc nint[0];
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbFocusIn, owner, args, nint.Zero);
        }
        public void FocusOut()
        {
            nint* args = stackalloc nint[0];
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbFocusOut, owner, args, nint.Zero);
        }
        public void Pause()
        {
            nint* args = stackalloc nint[0];
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbPause, owner, args, nint.Zero);
        }
        public void Resume()
        {
            nint* args = stackalloc nint[0];
            SimpleInterface.object_method_bind_ptrcall(Bindings.mbResume, owner, args, nint.Zero);
        }
    }

    // Simple GDExtension interface. Loads only what is needed for GodotInstance
    public unsafe static class SimpleInterface
    {
        private static GodotLibrary? library = null;
        public static GodotLibrary Library => library ?? throw new NullReferenceException(nameof(library));

        public static delegate* unmanaged<GDExtensionVariantType, delegate* unmanaged<nint, void>> variant_get_ptr_destructor;
        public static delegate* unmanaged<nint, nint, long, nint> classdb_get_method_bind;
        public static delegate* unmanaged<nint, nint, nint*, nint, void> object_method_bind_ptrcall;

        public static void Initialize(nint getProcAddress, nint token)
        {
            // Console.WriteLine("Inside SimpleInterface.Initialize");
            library = new(getProcAddress, token);
            // Console.WriteLine("Create library wrapper");
            variant_get_ptr_destructor = (delegate* unmanaged<GDExtensionVariantType, delegate* unmanaged<nint, void>>)library.LoadFunction("variant_get_ptr_destructor");
            classdb_get_method_bind = (delegate* unmanaged<nint, nint, long, nint>)library.LoadFunction("classdb_get_method_bind");
            object_method_bind_ptrcall = (delegate* unmanaged<nint, nint, nint*, nint, void>)library.LoadFunction("object_method_bind_ptrcall");

            StringName.InitializeBindings();
            // Console.WriteLine("StringName after init");

            GodotInstance.InitializeBindings();
            // Console.WriteLine("GodotInstance after init");
        }
    }

    // GDExtensions related init
    public static partial class LibGodot
    {
        [UnmanagedCallersOnly]
        private static void Initialize(nint userdata, GDExtensionInitializationLevel level) { }

        [UnmanagedCallersOnly]
        private static void Deinitialize(nint userdata, GDExtensionInitializationLevel level) { }

        [UnmanagedCallersOnly]
        private static unsafe byte GDExtensionInit(nint getProcAddress, nint token, GDExtensionInitialization* initialization)
        {
            // Console.WriteLine("Inside GDExtensionInit");
            initialization->MinimumInitializationLevel = GDExtensionInitializationLevel.Scene;
            initialization->Initialize = &Initialize;
            initialization->Deinitialize = &Deinitialize;
            initialization->Userdata = nint.Zero;

            SimpleInterface.Initialize(getProcAddress, token);
            return 1;
        }

        public static unsafe GodotInstance? CreateGodotInstance(string[] args)
        {
            return GodotInstance.CreateInstance(args, &GDExtensionInit);
        }
    }
}
#endif
