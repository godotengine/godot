using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

#nullable enable

namespace Godot
{
    internal static class DisposablesTracker
    {
        [UnmanagedCallersOnly]
        internal static void OnGodotShuttingDown()
        {
            try
            {
                OnGodotShuttingDownImpl();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        private static void OnGodotShuttingDownImpl()
        {
            bool isStdoutVerbose;

            try
            {
                isStdoutVerbose = OS.IsStdOutVerbose();
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

            foreach (WeakReference<GodotObject> item in GodotObjectInstances)
            {
                if (item.TryGetTarget(out GodotObject? self))
                    self.Dispose();
            }

            foreach (WeakReference<IDisposable> item in OtherInstances)
            {
                if (item.TryGetTarget(out IDisposable? self))
                    self.Dispose();
            }

            if (isStdoutVerbose)
                GD.Print("Unloading: Finished disposing tracked instances.");
        }

        public sealed class Element<T> where T : class
        {
            public T Value;
            public Element<T>? NextElement;
            public Element<T>? LastElement;
        }

        public sealed class FirstNonDuplicateLinkedList<T> : IEnumerable<T> where T : class
        {

            private Element<T>? FirstElement;
            private Element<T>? LastElement;

            public void Remove(Element<T> value)
            {
                lock (this)
                {
                    Element<T>? lastElement = value.LastElement;
                    if (value == LastElement)
                    {
                        LastElement = lastElement;
                    }
                    if (lastElement is null)
                    {
                        FirstElement = value.NextElement;
                    }
                    else if (value.NextElement is not null)
                    {
                        value.NextElement.LastElement = lastElement;
                        lastElement.NextElement = value.NextElement;
                    }
                    value.NextElement = null;
                    value.LastElement = null;
                }
            }

            public Element<T> Add(T value)
            {
                lock (this)
                {
                    if (FirstElement is null)
                    {
                        return LastElement = FirstElement = new Element<T> { Value = value };
                    }
                    return LastElement = LastElement.NextElement = new Element<T> { Value = value, LastElement = LastElement };
                }
            }

            private sealed class Enumerator : IEnumerator<T>
            {
                public Enumerator(Element<T>? element)
                {
                    Element = element;
                    StartElement = element;
                }
                private Element<T>? StartElement;

                private Element<T>? Element;

                public T Current => Element.Value;

                object IEnumerator.Current => Current;

                public void Dispose()
                {
                    Element = null;
                }

                public bool MoveNext()
                {
                    Element = Element?.NextElement;
                    return Element is not null;
                }

                public void Reset()
                {
                    Element = StartElement;
                }
            }

            public IEnumerator<T> GetEnumerator()
            {
                return new Enumerator(FirstElement);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return new Enumerator(FirstElement);
            }
        }

        private static FirstNonDuplicateLinkedList<WeakReference<GodotObject>> GodotObjectInstances { get; } =
            new();

        private static FirstNonDuplicateLinkedList<WeakReference<IDisposable>> OtherInstances { get; } =
            new();

        public static DisposablesTracker.Element<WeakReference<GodotObject>> RegisterGodotObject(GodotObject godotObject)
        {
            var weakReferenceToSelf = new WeakReference<GodotObject>(godotObject);
            return GodotObjectInstances.Add(weakReferenceToSelf);
        }

        public static DisposablesTracker.Element<WeakReference<IDisposable>> RegisterDisposable(IDisposable disposable)
        {
            var weakReferenceToSelf = new WeakReference<IDisposable>(disposable);
            return OtherInstances.Add(weakReferenceToSelf);
        }

        public static void UnregisterGodotObject(GodotObject godotObject, DisposablesTracker.Element<WeakReference<GodotObject>> weakReferenceToSelf)
        {
            GodotObjectInstances.Remove(weakReferenceToSelf);
        }

        public static void UnregisterDisposable(DisposablesTracker.Element<WeakReference<IDisposable>> weakReference)
        {
            OtherInstances.Remove(weakReference);
        }
    }
}
