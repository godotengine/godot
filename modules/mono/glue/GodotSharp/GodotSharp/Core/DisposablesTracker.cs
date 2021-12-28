using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Godot.NativeInterop;

#nullable enable

namespace Godot
{
    internal static class DisposablesTracker
    {
        static DisposablesTracker()
        {
            AssemblyLoadContext.Default.Unloading += _ => OnUnloading();
        }

        [UnmanagedCallersOnly]
        internal static void OnGodotShuttingDown()
        {
            try
            {
                OnUnloading();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
            }
        }

        private static void OnUnloading()
        {
            bool isStdoutVerbose;

            try
            {
                isStdoutVerbose = OS.IsStdoutVerbose();
            }
            catch (ObjectDisposedException)
            {
                // OS singleton already disposed. Maybe OnUnloading was called twice.
                isStdoutVerbose = false;
            }

            if (isStdoutVerbose)
                GD.Print("Unloading: Disposing tracked instances...");

            // Dispose Godot Objects first, and only then dispose other disposables
            // like StringName, NodePath, Godot.Collections.Array/Dictionary, etc.
            // The Godot Object Dispose() method may need any of the later instances.

            foreach (WeakReference<Object> item in GodotObjectInstances.Keys)
            {
                if (item.TryGetTarget(out Object? self))
                    self.Dispose();
            }

            foreach (WeakReference<IDisposable> item in OtherInstances.Keys)
            {
                if (item.TryGetTarget(out IDisposable? self))
                    self.Dispose();
            }

            if (isStdoutVerbose)
                GD.Print("Unloading: Finished disposing tracked instances.");
        }

        // ReSharper disable once RedundantNameQualifier
        private static ConcurrentDictionary<WeakReference<Godot.Object>, object?> GodotObjectInstances { get; } =
            new();

        private static ConcurrentDictionary<WeakReference<IDisposable>, object?> OtherInstances { get; } =
            new();

        public static WeakReference<Object> RegisterGodotObject(Object godotObject)
        {
            var weakReferenceToSelf = new WeakReference<Object>(godotObject);
            GodotObjectInstances.TryAdd(weakReferenceToSelf, null);
            return weakReferenceToSelf;
        }

        public static WeakReference<IDisposable> RegisterDisposable(IDisposable disposable)
        {
            var weakReferenceToSelf = new WeakReference<IDisposable>(disposable);
            OtherInstances.TryAdd(weakReferenceToSelf, null);
            return weakReferenceToSelf;
        }

        public static void UnregisterGodotObject(WeakReference<Object> weakReference)
        {
            if (!GodotObjectInstances.TryRemove(weakReference, out _))
                throw new ArgumentException("Godot Object not registered", nameof(weakReference));
        }

        public static void UnregisterDisposable(WeakReference<IDisposable> weakReference)
        {
            if (!OtherInstances.TryRemove(weakReference, out _))
                throw new ArgumentException("Disposable not registered", nameof(weakReference));
        }
    }
}
