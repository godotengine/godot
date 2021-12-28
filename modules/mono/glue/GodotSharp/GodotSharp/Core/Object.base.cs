using System;
using System.Linq;
using System.Reflection;
using Godot.NativeInterop;

namespace Godot
{
    public partial class Object : IDisposable
    {
        private bool _disposed = false;
        private Type _cachedType = typeof(Object);

        internal IntPtr NativePtr;
        internal bool MemoryOwn;

        /// <summary>
        /// Constructs a new <see cref="Object"/>.
        /// </summary>
        public Object() : this(false)
        {
            if (NativePtr == IntPtr.Zero)
            {
                unsafe
                {
                    NativePtr = NativeCtor();
                }

                InteropUtils.TieManagedToUnmanaged(this, NativePtr,
                    NativeName, refCounted: false, GetType(), _cachedType);
            }
            else
            {
                InteropUtils.TieManagedToUnmanagedWithPreSetup(this, NativePtr);
            }

            _InitializeGodotScriptInstanceInternals();
        }

        internal void _InitializeGodotScriptInstanceInternals()
        {
            // Performance is not critical here as this will be replaced with source generators.
            Type top = GetType();
            Type native = InternalGetClassNativeBase(top);

            while (top != null && top != native)
            {
                foreach (var eventSignal in top.GetEvents(
                                 BindingFlags.DeclaredOnly | BindingFlags.Instance |
                                 BindingFlags.NonPublic | BindingFlags.Public)
                             .Where(ev => ev.GetCustomAttributes().OfType<SignalAttribute>().Any()))
                {
                    using var eventSignalName = new StringName(eventSignal.Name);
                    var eventSignalNameSelf = (godot_string_name)eventSignalName.NativeValue;
                    NativeFuncs.godotsharp_internal_object_connect_event_signal(NativePtr, eventSignalNameSelf);
                }

                top = top.BaseType;
            }
        }

        internal Object(bool memoryOwn)
        {
            MemoryOwn = memoryOwn;
        }

        /// <summary>
        /// The pointer to the native instance of this <see cref="Object"/>.
        /// </summary>
        public IntPtr NativeInstance => NativePtr;

        internal static IntPtr GetPtr(Object instance)
        {
            if (instance == null)
                return IntPtr.Zero;

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.NativePtr;
        }

        ~Object()
        {
            Dispose(false);
        }

        /// <summary>
        /// Disposes of this <see cref="Object"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes implementation of this <see cref="Object"/>.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (NativePtr != IntPtr.Zero)
            {
                if (MemoryOwn)
                {
                    MemoryOwn = false;
                    NativeFuncs.godotsharp_internal_refcounted_disposed(NativePtr, (!disposing).ToGodotBool());
                }
                else
                {
                    NativeFuncs.godotsharp_internal_object_disposed(NativePtr);
                }

                NativePtr = IntPtr.Zero;
            }

            _disposed = true;
        }

        /// <summary>
        /// Converts this <see cref="Object"/> to a string.
        /// </summary>
        /// <returns>A string representation of this object.</returns>
        public override string ToString()
        {
            NativeFuncs.godotsharp_object_to_string(GetPtr(this), out godot_string str);
            using (str)
                return Marshaling.ConvertStringToManaged(str);
        }

        /// <summary>
        /// Returns a new <see cref="SignalAwaiter"/> awaiter configured to complete when the instance
        /// <paramref name="source"/> emits the signal specified by the <paramref name="signal"/> parameter.
        /// </summary>
        /// <param name="source">
        /// The instance the awaiter will be listening to.
        /// </param>
        /// <param name="signal">
        /// The signal the awaiter will be waiting for.
        /// </param>
        /// <example>
        /// This sample prints a message once every frame up to 100 times.
        /// <code>
        /// public override void _Ready()
        /// {
        ///     for (int i = 0; i &lt; 100; i++)
        ///     {
        ///         await ToSignal(GetTree(), "process_frame");
        ///         GD.Print($"Frame {i}");
        ///     }
        /// }
        /// </code>
        /// </example>
        /// <returns>
        /// A <see cref="SignalAwaiter"/> that completes when
        /// <paramref name="source"/> emits the <paramref name="signal"/>.
        /// </returns>
        public SignalAwaiter ToSignal(Object source, StringName signal)
        {
            return new SignalAwaiter(source, signal, this);
        }

        internal static Type InternalGetClassNativeBase(Type t)
        {
            do
            {
                var assemblyName = t.Assembly.GetName();

                if (assemblyName.Name == "GodotSharp")
                    return t;

                if (assemblyName.Name == "GodotSharpEditor")
                    return t;
            } while ((t = t.BaseType) != null);

            return null;
        }

        internal static bool InternalIsClassNativeBase(Type t)
        {
            var assemblyName = t.Assembly.GetName();
            return assemblyName.Name == "GodotSharp" || assemblyName.Name == "GodotSharpEditor";
        }

        internal bool InternalGodotScriptCallViaReflection(string method, NativeVariantPtrArgs args, int argCount,
            out godot_variant ret)
        {
            // Performance is not critical here as this will be replaced with source generators.
            Type top = GetType();
            Type native = InternalGetClassNativeBase(top);

            while (top != null && top != native)
            {
                var methodInfo = top.GetMethod(method,
                    BindingFlags.DeclaredOnly | BindingFlags.Instance |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (methodInfo != null)
                {
                    var parameters = methodInfo.GetParameters();
                    int paramCount = parameters.Length;

                    if (argCount == paramCount)
                    {
                        object[] invokeParams = new object[paramCount];

                        for (int i = 0; i < paramCount; i++)
                        {
                            invokeParams[i] = Marshaling.ConvertVariantToManagedObjectOfType(
                                args[i], parameters[i].ParameterType);
                        }

                        object retObj = methodInfo.Invoke(this, invokeParams);

                        ret = Marshaling.ConvertManagedObjectToVariant(retObj);
                        return true;
                    }
                }

                top = top.BaseType;
            }

            ret = default;
            return false;
        }

        internal bool InternalGodotScriptSetFieldOrPropViaReflection(string name, in godot_variant value)
        {
            // Performance is not critical here as this will be replaced with source generators.
            Type top = GetType();
            Type native = InternalGetClassNativeBase(top);

            while (top != null && top != native)
            {
                var fieldInfo = top.GetField(name,
                    BindingFlags.DeclaredOnly | BindingFlags.Instance |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (fieldInfo != null)
                {
                    object valueManaged = Marshaling.ConvertVariantToManagedObjectOfType(value, fieldInfo.FieldType);
                    fieldInfo.SetValue(this, valueManaged);

                    return true;
                }

                var propertyInfo = top.GetProperty(name,
                    BindingFlags.DeclaredOnly | BindingFlags.Instance |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (propertyInfo != null)
                {
                    object valueManaged =
                        Marshaling.ConvertVariantToManagedObjectOfType(value, propertyInfo.PropertyType);
                    propertyInfo.SetValue(this, valueManaged);

                    return true;
                }

                top = top.BaseType;
            }

            return false;
        }

        internal bool InternalGodotScriptGetFieldOrPropViaReflection(string name, out godot_variant value)
        {
            // Performance is not critical here as this will be replaced with source generators.
            Type top = GetType();
            Type native = InternalGetClassNativeBase(top);

            while (top != null && top != native)
            {
                var fieldInfo = top.GetField(name,
                    BindingFlags.DeclaredOnly | BindingFlags.Instance |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (fieldInfo != null)
                {
                    object valueManaged = fieldInfo.GetValue(this);
                    value = Marshaling.ConvertManagedObjectToVariant(valueManaged);
                    return true;
                }

                var propertyInfo = top.GetProperty(name,
                    BindingFlags.DeclaredOnly | BindingFlags.Instance |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (propertyInfo != null)
                {
                    object valueManaged = propertyInfo.GetValue(this);
                    value = Marshaling.ConvertManagedObjectToVariant(valueManaged);
                    return true;
                }

                top = top.BaseType;
            }

            value = default;
            return false;
        }

        internal unsafe void InternalRaiseEventSignal(in godot_string_name eventSignalName, NativeVariantPtrArgs args,
            int argc)
        {
            // Performance is not critical here as this will be replaced with source generators.

            using var stringName = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(eventSignalName));
            string eventSignalNameStr = stringName.ToString();

            Type top = GetType();
            Type native = InternalGetClassNativeBase(top);

            while (top != null && top != native)
            {
                var foundEventSignals = top.GetEvents(
                        BindingFlags.DeclaredOnly | BindingFlags.Instance |
                        BindingFlags.NonPublic | BindingFlags.Public)
                    .Where(ev => ev.GetCustomAttributes().OfType<SignalAttribute>().Any())
                    .Select(ev => ev.Name);

                var fields = top.GetFields(
                    BindingFlags.DeclaredOnly | BindingFlags.Instance |
                    BindingFlags.NonPublic | BindingFlags.Public);

                var eventSignalField = fields
                    .Where(f => typeof(Delegate).IsAssignableFrom(f.FieldType))
                    .Where(f => foundEventSignals.Contains(f.Name))
                    .FirstOrDefault(f => f.Name == eventSignalNameStr);

                if (eventSignalField != null)
                {
                    var @delegate = (Delegate)eventSignalField.GetValue(this);

                    if (@delegate == null)
                        continue;

                    var delegateType = eventSignalField.FieldType;

                    var invokeMethod = delegateType.GetMethod("Invoke");

                    if (invokeMethod == null)
                        throw new MissingMethodException(delegateType.FullName, "Invoke");

                    var parameterInfos = invokeMethod.GetParameters();
                    var paramsLength = parameterInfos.Length;

                    if (argc != paramsLength)
                    {
                        throw new InvalidOperationException(
                            $"The event delegate expects {paramsLength} arguments, but received {argc}.");
                    }

                    var managedArgs = new object[argc];

                    for (int i = 0; i < argc; i++)
                    {
                        managedArgs[i] = Marshaling.ConvertVariantToManagedObjectOfType(
                            args[i], parameterInfos[i].ParameterType);
                    }

                    invokeMethod.Invoke(@delegate, managedArgs);
                    return;
                }

                top = top.BaseType;
            }
        }

        internal static unsafe IntPtr ClassDB_get_method(StringName type, string method)
        {
            IntPtr methodBind;
            fixed (char* methodChars = method)
            {
                var typeSelf = (godot_string_name)type.NativeValue;
                methodBind = NativeFuncs.godotsharp_method_bind_get_method(typeSelf, methodChars);
            }

            if (methodBind == IntPtr.Zero)
                throw new NativeMethodBindNotFoundException(type + "." + method);

            return methodBind;
        }

        internal static unsafe delegate* unmanaged<IntPtr> ClassDB_get_constructor(StringName type)
        {
            // for some reason the '??' operator doesn't support 'delegate*'
            var typeSelf = (godot_string_name)type.NativeValue;
            var nativeConstructor = NativeFuncs.godotsharp_get_class_constructor(typeSelf);

            if (nativeConstructor == null)
                throw new NativeConstructorNotFoundException(type);

            return nativeConstructor;
        }
    }
}
