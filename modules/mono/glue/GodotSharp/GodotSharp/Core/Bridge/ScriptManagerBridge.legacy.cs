using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;
using JetBrains.Annotations;

namespace Godot.Bridge;

#nullable enable

public static partial class ScriptManagerBridge
{
    [UnmanagedCallersOnly]
    internal static unsafe godot_bool LegacyCreateManagedForGodotObjectScriptInstance(IntPtr scriptPtr,
        IntPtr godotObject,
        godot_variant** args, int argCount)
    {
        try
        {
            // Performance is not critical here as this will be replaced with source generators.
            var scriptTypeMeta = GetOrResolveScriptTypeMeta(_scriptTypeBiMap.GetScriptType(scriptPtr));
            Type scriptType = scriptTypeMeta.Type;

            Debug.Assert(!scriptType.IsAbstract,
                $"Cannot create script instance. The class '{scriptType.FullName}' is abstract.");

            if (scriptTypeMeta.LegacyCreateManagedForGodotObjectScriptInstance == null)
            {
                throw new InvalidOperationException(
                    $"No legacy construction delegate available for class '{scriptType.FullName}'.");
            }

            bool found = scriptTypeMeta.LegacyCreateManagedForGodotObjectScriptInstance.Invoke(
                godotObject, new NativeVariantPtrArgs(args, argCount));

            if (!found)
            {
                if (argCount == 0)
                    throw new MissingMemberException(
                        $"Cannot create script instance. The class '{scriptType.FullName}' does not define a parameterless constructor.");

                throw new MissingMemberException(
                    $"The class '{scriptType.FullName}' does not define a constructor that takes {argCount} parameters.");
            }

            return godot_bool.True;
        }
        catch (Exception e)
        {
            ExceptionUtils.LogException(e);
            return godot_bool.False;
        }
    }

    [UnmanagedCallersOnly]
    internal static unsafe godot_bool LegacyCallStatic(IntPtr scriptPtr, godot_string_name* method,
        godot_variant** args, int argCount, godot_variant_call_error* refCallError, godot_variant* outRet)
    {
        try
        {
            var scriptTypeMeta = GetOrResolveScriptTypeMeta(_scriptTypeBiMap.GetScriptType(scriptPtr));

            if (scriptTypeMeta.LegacyInvokeGodotClassStaticMethod != null)
            {
                var invoked = scriptTypeMeta.LegacyInvokeGodotClassStaticMethod.Invoke(
                    CustomUnsafe.AsRef(method), new NativeVariantPtrArgs(args, argCount),
                    out godot_variant retValue);
                if (invoked)
                {
                    *outRet = retValue;
                    return godot_bool.True;
                }
            }
        }
        catch (Exception e)
        {
            ExceptionUtils.LogException(e);
            *outRet = default;
            return godot_bool.False;
        }

        *outRet = default;
        (*refCallError).Error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD;
        return godot_bool.False;
    }

    [RequiresUnreferencedCode(
        "Determining the script type meta on legacy code at runtime is not compatible with trimming.")]
    private static ScriptTypeMeta DetermineScriptTypeMetaOfLegacyType(Type legacyScriptType)
    {
        var type = legacyScriptType;

        var nativeType = InternalGetClassNativeBase(type);
        var nativeName = NativeClassGetNativeName(nativeType) ?? throw
            new InvalidOperationException($"Native name not found for native Godot type {nativeType}.");

        var scriptTypeMeta = new ScriptTypeMeta(type, nativeType, nativeName)
        {
            GetGodotClassTrampolines = FindGetGodotClassTrampolines(type),
            GetGodotMethodList = FindGetGodotMethodList(type),
            GetGodotSignalList = FindGetGodotSignalList(type),
            GetGodotPropertyList = FindGetGodotPropertyList(type),
            GetGodotRpcMethods = FindGetGodotRpcMethods(type),
            GetGodotPropertyDefaultValues = FindGetGodotPropertyDefaultValues(type),
            // No need to check for "HasGodotClassMethod" nor "HasGodotClassSignal",
            // as these always accompany "InvokeGodotClassMethod" and "RaiseGodotClassSignalCallbacks".
            ShouldFallbackToLegacyTrampolines =
                DoesUserScriptContainLegacyInstanceMethod("InvokeGodotClassMethod")
                || DoesUserScriptContainLegacyInstanceMethod("SetGodotClassPropertyValue")
                || DoesUserScriptContainLegacyInstanceMethod("GetGodotClassPropertyValue")
                || DoesUserScriptContainLegacyInstanceMethod("RaiseGodotClassSignalCallbacks")
        };

        if (scriptTypeMeta.GetGodotPropertyDefaultValues == null)
            scriptTypeMeta.LegacyGetGodotPropertyDefaultValues = FindLegacyGetGodotPropertyDefaultValues(type);

        if (scriptTypeMeta.GetGodotClassTrampolines == null)
        {
            scriptTypeMeta.LegacyInvokeGodotClassStaticMethod = FindLegacyInvokeGodotClassStaticMethod(type);

            if (type.IsAbstract)
            {
                scriptTypeMeta.LegacyCreateManagedForGodotObjectScriptInstance = bool (_, _)
                    => throw new InvalidOperationException(
                        $"Cannot create script instance. The class '{type.FullName}' is abstract.");
            }
            else
            {
                scriptTypeMeta.LegacyCreateManagedForGodotObjectScriptInstance = bool (godotObjectPtr, args)
                    => LegacyCreateManagedForGodotObjectScriptInstanceImpl(type, godotObjectPtr, args);
            }
        }

        const BindingFlags StaticReflFlags = BindingFlags.DeclaredOnly | BindingFlags.Static |
                                             BindingFlags.NonPublic | BindingFlags.Public;

        return scriptTypeMeta;

        static StringName? NativeClassGetNativeName(Type nativeClass) =>
            nativeClass.GetField("NativeName",
                BindingFlags.DeclaredOnly | BindingFlags.Static |
                BindingFlags.Public | BindingFlags.NonPublic)?.GetValue(null) as StringName;

        static Action<TrampolineCollectors, TrampolineCollectionOptions>? FindGetGodotClassTrampolines(Type type) =>
            (type.GetNestedType("GodotInternal")?.GetMethod("GetGodotClassTrampolines", StaticReflFlags)
             ?? type.GetMethod("GetGodotClassTrampolines", StaticReflFlags))
            ?.CreateDelegate<Action<TrampolineCollectors, TrampolineCollectionOptions>>();

        static Func<List<MethodInfo>?>? FindGetGodotMethodList(Type type) =>
            (type.GetNestedType("GodotInternal")?.GetMethod("GetGodotMethodList", StaticReflFlags)
             ?? type.GetMethod("GetGodotMethodList", StaticReflFlags))
            ?.CreateDelegate<Func<List<MethodInfo>?>>();

        static Func<List<MethodInfo>?>? FindGetGodotSignalList(Type type) =>
            (type.GetNestedType("GodotInternal")?.GetMethod("GetGodotSignalList", StaticReflFlags)
             ?? type.GetMethod("GetGodotSignalList", StaticReflFlags))
            ?.CreateDelegate<Func<List<MethodInfo>?>>();

        static Func<List<PropertyInfo>?>? FindGetGodotPropertyList(Type type) =>
            (type.GetNestedType("GodotInternal")?.GetMethod("GetGodotPropertyList", StaticReflFlags)
             ?? type.GetMethod("GetGodotPropertyList", StaticReflFlags))
            ?.CreateDelegate<Func<List<PropertyInfo>?>>();

        static Action<RpcMethodCollector>? FindGetGodotRpcMethods(Type type) =>
            (type.GetNestedType("GodotInternal")?.GetMethod("GetGodotRpcMethods", StaticReflFlags)
             ?? type.GetMethod("GetGodotRpcMethods", StaticReflFlags))
            ?.CreateDelegate<Action<RpcMethodCollector>>();

        static Func<Dictionary<StringName, Variant>?>? FindGetGodotPropertyDefaultValues(Type type)
        {
            try
            {
                return (type.GetNestedType("GodotInternal")
                            ?.GetMethod("GetGodotPropertyDefaultValues", StaticReflFlags)
                        ?? type.GetMethod("GetGodotPropertyDefaultValues", StaticReflFlags))
                    ?.CreateDelegate<Func<Dictionary<StringName, Variant>?>>();
            }
            // Wrong delegate signature, just return null.
            catch (InvalidCastException)
            {
                return null;
            }
        }

        static Func<Dictionary<StringName, object?>>? FindLegacyGetGodotPropertyDefaultValues(Type type)
        {
            try
            {
                return type.GetMethod("GetGodotPropertyDefaultValues", StaticReflFlags)
                    ?.CreateDelegate<Func<Dictionary<StringName, object?>>>();
            }
            // Wrong delegate signature, just return null.
            catch (InvalidCastException)
            {
                return null;
            }
        }

        static LegacyInvokeGodotClassStaticMethodDelegate? FindLegacyInvokeGodotClassStaticMethod(Type type) =>
            type.GetMethod("InvokeGodotClassStaticMethod", StaticReflFlags)
                ?.CreateDelegate<LegacyInvokeGodotClassStaticMethodDelegate>();

        bool DoesUserScriptContainLegacyInstanceMethod(string methodName)
        {
            var methodInfo = type.GetMethod(methodName,
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);

            if (methodInfo == null)
                return false;

            for (Type? top = type; top != null && top != nativeType; top = top.BaseType)
            {
                if (methodInfo.DeclaringType == top)
                    return true;
            }

            return false;
        }

        static bool LegacyCreateManagedForGodotObjectScriptInstanceImpl(Type type, IntPtr godotObjectPtr,
            NativeVariantPtrArgs args)
        {
            // This method is only used if the type is not abstract, so no need to check that.

            int argCount = args.Count;

            var ctor = type
                .GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(c => c.GetParameters().Length == argCount);

            if (ctor == null)
            {
                if (argCount == 0)
                {
                    throw new MissingMemberException(
                        $"Cannot create script instance. The class '{type.FullName}' does not define a parameterless constructor.");
                }

                throw new MissingMemberException(
                    $"The class '{type.FullName}' does not define a constructor that takes {argCount} parameters.");
            }

            var obj = (GodotObject)RuntimeHelpers.GetUninitializedObject(type);

            var parameters = ctor.GetParameters();
            int paramCount = parameters.Length;

            var invokeParams = new object?[paramCount];

            for (int i = 0; i < paramCount; i++)
            {
                invokeParams[i] = DelegateUtils.RuntimeTypeConversionHelper.ConvertToObjectOfType(
                    args[i], parameters[i].ParameterType);
            }

            obj.NativePtr = godotObjectPtr;

            _ = ctor.Invoke(obj, invokeParams);

            return true;
        }

        static bool IsNativeClass(Type t) =>
            ReferenceEquals(t.Assembly, typeof(GodotObject).Assembly) ||
            (ReflectionUtils.IsEditorHintCached && t.Assembly.GetName().Name == "GodotSharpEditor");

        static Type InternalGetClassNativeBase(Type t)
        {
            while (!IsNativeClass(t))
            {
                t = t.BaseType ?? throw
                    new InvalidOperationException("The script type does not derive from a native Godot type.");
            }

            return t;
        }
    }

    private static bool _legacyScriptTypeMetaResolverAlreadyEnabled;

    /// <summary>
    /// Enables support for legacy code that relies on runtime script type meta resolution.
    /// </summary>
    [RequiresUnreferencedCode(
        "Resolving of script type meta on legacy code at runtime is not compatible with trimming.")]
    [PublicAPI("Source generators depend on this for supporting legacy code.")]
    [Obsolete(
        "Use 'RegisterScriptType', 'RegisterScriptGenericTypeDefinition', and 'ScriptTypeMetaProviderAttribute' instead. "
        + "Source generators handle the boilerplate of type registration automatically.")]
    public static void EnableLegacyScriptTypeMetaResolver()
    {
        if (_legacyScriptTypeMetaResolverAlreadyEnabled)
            return;

        var typeOfGodotObject = typeof(GodotObject); // Captured by the resolver.

        AddLegacyScriptTypeMetaLookupDelegate(LegacyResolver);
        _legacyScriptTypeMetaResolverAlreadyEnabled = true;


        return;

        bool LegacyResolver(Type scriptType, [MaybeNullWhen(false)] out ScriptTypeMeta scriptTypeMeta)
        {
            if (!typeOfGodotObject.IsAssignableFrom(scriptType))
            {
                scriptTypeMeta = null;
                return false;
            }

            if (scriptType.GetCustomAttribute<ScriptTypeMetaProviderBaseAttribute>() is { } providerAttr)
                throw new InvalidOperationException(
                    $"The type '{scriptType.FullName}' is decorated with '{providerAttr.GetType().FullName}', " +
                    $"so its script type meta should have been provided by that attribute. " +
                    $"The presence of this exception indicates a bug in the script type meta resolution logic.");

            scriptTypeMeta = DetermineScriptTypeMetaOfLegacyType(scriptType);
            return true;
        }
    }

    /// <summary>
    /// Look for Godot script classes in the specified assembly.
    /// (legacy) Called from GodotPlugins.
    /// </summary>
    [RequiresUnreferencedCode(
        "Searching for script types in an assembly at runtime is not compatible with trimming. "
        + "Use 'RegisterScriptType' and 'RegisterScriptGenericTypeDefinition' instead. "
        + "Source generators handle the boilerplate of type registration automatically.")]
    [Obsolete(
        "Use 'RegisterScriptType', 'RegisterScriptGenericTypeDefinition', and 'ScriptTypeMetaProviderAttribute' instead. "
        + "Source generators handle the boilerplate of type registration automatically.")]
    [PublicAPI("ABI compatibility with legacy code.")]
    public static void LookupScriptsInAssembly(Assembly assembly)
    {
        // LookupScriptsInAssembly is called by legacy code, so we need to enable this to discover the ScriptTypeMeta.
        EnableLegacyScriptTypeMetaResolver();

        var assemblyHasScriptsAttr = assembly.GetCustomAttributes(inherit: false)
            .OfType<AssemblyHasScriptsAttribute>()
            .FirstOrDefault();

        if (assemblyHasScriptsAttr == null)
            return;

        if (assemblyHasScriptsAttr.RequiresLookup)
        {
            // This is supported for scenarios where specifying all types would be cumbersome,
            // such as when disabling C# source generators (for whatever reason) or when using a
            // language other than C# that has nothing similar to source generators to automate it.

            var typeOfGodotObject = typeof(GodotObject);

            foreach (var type in assembly.GetTypes())
            {
                if (type.IsNested)
                    continue;

                if (!typeOfGodotObject.IsAssignableFrom(type))
                    continue;

                LookupScriptForClass(type);
            }
        }
        else
        {
            // This is the most likely scenario as we use C# source generators

            var scriptTypes = assemblyHasScriptsAttr.ScriptTypes;

            if (scriptTypes != null)
            {
                foreach (var type in scriptTypes)
                {
                    LookupScriptForClass(type);
                }
            }
        }

        return;

        static void LookupScriptForClass(Type type)
        {
            var scriptPathAttr = type.GetCustomAttributes(inherit: false)
                .OfType<ScriptPathAttribute>()
                .FirstOrDefault();

            if (scriptPathAttr == null)
                return;

            var scriptTypeMeta = DetermineScriptTypeMetaOfLegacyType(type);

            try
            {
                _scriptTypeMetaMap.TryAdd(type, scriptTypeMeta);

                _pathTypeBiMap.Add(scriptPathAttr.Path, type);
            }
            finally
            {
                if (AlcReloadCfg.IsAlcReloadingEnabled)
                    AddTypeForAlcReloading(type);
            }
        }
    }
}
