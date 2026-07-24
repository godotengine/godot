using Godot;
using Godot.Bridge;
using GodotTools.Utils;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Xml.Linq;
using ReflectionFieldInfo = System.Reflection.FieldInfo;
using ReflectionPropertyInfo = System.Reflection.PropertyInfo;

namespace GodotTools
{
    public static class ScriptDoc
    {
        private static object? _scriptTypeBiMap;
        private static System.Reflection.MethodInfo? _getScriptTypeMethod;

        public static Type? GetScriptType(IntPtr scriptPtr)
        {
            if (_scriptTypeBiMap == null || _getScriptTypeMethod == null)
            {

                var scriptManagerBridge = typeof(ScriptManagerBridge).GetField("_scriptTypeBiMap", BindingFlags.NonPublic | BindingFlags.Static);
                _scriptTypeBiMap = scriptManagerBridge?.GetValue(null);
                _getScriptTypeMethod = _scriptTypeBiMap?.GetType().GetMethod("GetScriptType", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance);
            }
            var result = _getScriptTypeMethod?.Invoke(_scriptTypeBiMap, new object[] { scriptPtr });

            return result is Type t ? t : null;
        }

        /// <summary>
        /// Gets the default values for exported members of a Godot script type.
        /// </summary>
        /// <remarks>
        /// The Godot source generators emit a static <c>GetGodotPropertyDefaultValues</c> method on
        /// Godot script classes that returns a <c>System.Collections.Generic.Dictionary&lt;StringName, Variant&gt;</c>
        /// with the default values for each exported member (as evaluated from its initializer).
        /// This method invokes it via reflection so the documentation generator can surface those
        /// values in the inspector and built-in documentation, mirroring how GDScript exposes
        /// property defaults.
        /// </remarks>
        public static System.Collections.Generic.Dictionary<StringName, Variant>? GetPropertyDefaultValues(Type scriptType)
        {
            var getGodotPropertyDefaultValuesMethod = scriptType.GetMethod(
                "GetGodotPropertyDefaultValues",
                BindingFlags.DeclaredOnly | BindingFlags.Static |
                BindingFlags.NonPublic | BindingFlags.Public);

            if (getGodotPropertyDefaultValuesMethod == null)
                return null;

            try
            {
                var defaultValuesObj = getGodotPropertyDefaultValuesMethod.Invoke(null, null);
                return defaultValuesObj as System.Collections.Generic.Dictionary<StringName, Variant>;
            }
            catch
            {
                return null;
            }
        }

        internal static string GetTypeDocumentationId(Type type)
        {
            Type xmlType = type.IsConstructedGenericType ? type.GetGenericTypeDefinition() : type;
            string fullName = xmlType.FullName ?? throw new InvalidOperationException($"Unable to resolve full name for type '{type}'.");
            return $"T:{fullName.Replace('+', '.')}";
        }

        /// <summary>
        /// Returns <see langword="true"/> when the given member is marked with
        /// <c>[EditorBrowsable(EditorBrowsableState.Never)]</c>. The Godot source generators
        /// tag every method/property they emit with this attribute so the members stay hidden
        /// from IntelliSense; we reuse that signal to keep generated boilerplate (e.g.
        /// <c>GetGodotMethodList</c>, <c>InvokeGodotClassMethod</c>) out of the documentation.
        /// </summary>
        public static bool IsHiddenFromEditor(System.Reflection.MemberInfo member)
        {
            var attribute = member.GetCustomAttribute<System.ComponentModel.EditorBrowsableAttribute>(inherit: false);
            return attribute?.State == System.ComponentModel.EditorBrowsableState.Never;
        }

        internal static string GetPropertyDocumentationId(System.Reflection.PropertyInfo property)
            => $"P:{GetTypeDocumentationId(property.DeclaringType!).Substring(2)}.{property.Name}";

        internal static string GetFieldDocumentationId(System.Reflection.FieldInfo field)
            => $"F:{GetTypeDocumentationId(field.DeclaringType!).Substring(2)}.{field.Name}";

        /// <summary>
        /// Builds the XML documentation member id for a method, following the .NET XML
        /// documentation signature rules (e.g. <c>M:Namespace.Class.Method(System.Int32)</c>).
        /// </summary>
        internal static string GetMethodDocumentationId(System.Reflection.MethodInfo method)
        {
            string declaringId = GetTypeDocumentationId(method.DeclaringType!).Substring(2);

            // Method name includes the generic arity suffix for generic methods (e.g. ``1).
            string methodName = method.Name;
            if (method.IsGenericMethod)
            {
                methodName += "``" + method.GetGenericArguments().Length;
            }

            var sb = new System.Text.StringBuilder();
            sb.Append("M:").Append(declaringId).Append('.').Append(methodName);

            System.Reflection.ParameterInfo[] parameters = method.GetParameters();
            if (parameters.Length > 0)
            {
                sb.Append('(');
                for (int i = 0; i < parameters.Length; i++)
                {
                    if (i > 0)
                        sb.Append(',');
                    sb.Append(GetXmlParameterTypeString(parameters[i].ParameterType));
                }
                sb.Append(')');
            }

            // Specialization: methods whose return type differs only in return type need a
            // disambiguating suffix (e.g. M:C.M(System.Int32)~System.String). We omit this
            // since exported/public method overloads differing only by return type are rare
            // and the documentation lookup falls back gracefully when not found.

            return sb.ToString();
        }

        /// <summary>
        /// Returns the XML documentation string for a parameter type, matching the format
        /// used in member id signatures (fully qualified, TRANSREF for by-ref, array notation).
        /// </summary>
        private static string GetXmlParameterTypeString(Type type)
        {
            if (type.IsByRef)
                return GetXmlParameterTypeString(type.GetElementType()!) + "@";

            if (type.IsArray)
                return GetXmlParameterTypeString(type.GetElementType()!) + "[]";

            if (type.IsPointer)
                return GetXmlParameterTypeString(type.GetElementType()!) + "*";

            if (type.IsGenericParameter)
                return "``" + type.GenericParameterPosition;

            if (type.IsConstructedGenericType)
            {
                Type genericDef = type.GetGenericTypeDefinition();
                string defName = (genericDef.FullName ?? genericDef.Name).Replace('+', '.');
                Type[] args = type.GetGenericArguments();
                var sb = new System.Text.StringBuilder(defName).Append('{');
                for (int i = 0; i < args.Length; i++)
                {
                    if (i > 0)
                        sb.Append(',');
                    sb.Append(GetXmlParameterTypeString(args[i]));
                }
                return sb.Append('}').ToString();
            }

            return (type.FullName ?? type.Name).Replace('+', '.');
        }

        public static Dictionary<string, XElement> LoadMembersFromFile(string xmlPath)
        {
            if (!System.IO.File.Exists(xmlPath))
                throw new FileNotFoundException($"Required XML documentation file was not found: '{xmlPath}'.", xmlPath);

            using var stream = System.IO.File.OpenRead(xmlPath);
            XDocument document = XDocument.Load(stream);
            XElement? membersElement = document.Root?.Element("members");

            if (membersElement == null)
                throw new InvalidOperationException($"Invalid XML documentation format. Missing '<members>' in '{xmlPath}'.");

            var members = new Dictionary<string, XElement>(StringComparer.Ordinal);

            foreach (XElement member in membersElement.Elements("member"))
            {
                XAttribute? nameAttr = member.Attribute("name");
                if (nameAttr == null)
                    continue;

                members[nameAttr.Value] = member;
            }

            if (members.Count == 0)
                throw new InvalidOperationException($"XML documentation '{xmlPath}' does not contain any member entries.");

            return members;
        }

        public sealed class XmlDocumentationCache
        {
            public XmlDocumentationCache(Dictionary<string, System.Xml.Linq.XElement> members)
            {
                Members = members;
            }

            public Dictionary<string, System.Xml.Linq.XElement> Members { get; }
        }

        public sealed class AssemblyXmlDocumentationPath
        {
            public AssemblyXmlDocumentationPath(string xmlPath)
            {
                XmlPath = xmlPath;
            }

            public string XmlPath { get; }
        }

        public static readonly ConcurrentDictionary<string, XmlDocumentationCache> _xmlDocumentationCacheByPath =
            new(StringComparer.OrdinalIgnoreCase);

        public static readonly ConditionalWeakTable<Assembly, AssemblyXmlDocumentationPath> _xmlDocPathByAssembly =
            new();

        public static XmlDocumentationCache LoadXmlDocumentationCache(string xmlPath) => new(LoadMembersFromFile(xmlPath));

        public static XmlDocumentationCache? GetXmlDocumentationCache(Assembly assembly)
        {
            static bool TryGetXmlPathFromAssemblyLocation(Assembly targetAssembly, out string? xmlPath)
            {
                string assemblyPath = targetAssembly.Location;
                if (!string.IsNullOrEmpty(assemblyPath))
                {
                    string candidate = Path.ChangeExtension(assemblyPath, ".xml");
                    if (System.IO.File.Exists(candidate))
                    {
                        xmlPath = candidate;
                        return true;
                    }
                }

                xmlPath = null;
                return false;
            }

            if (_xmlDocPathByAssembly.TryGetValue(assembly, out AssemblyXmlDocumentationPath? knownPath) &&
                System.IO.File.Exists(knownPath.XmlPath))
            {
                return _xmlDocumentationCacheByPath.GetOrAdd(knownPath.XmlPath, LoadXmlDocumentationCache);
            }

            if (TryGetXmlPathFromAssemblyLocation(assembly, out string? resolvedXmlPath))
                return _xmlDocumentationCacheByPath.GetOrAdd(resolvedXmlPath!, LoadXmlDocumentationCache);

            // For assemblies loaded from stream (where Assembly.Location is empty),
            // try the project's known build output directory. In the editor,
            // assemblies are loaded via LoadFromStream to prevent file locking,
            // which causes Assembly.Location to return empty.
            if (TryGetXmlPathFromProjectBuildOutput(assembly, out string? projectXmlPath))
            {
                _xmlDocPathByAssembly.AddOrUpdate(assembly, new AssemblyXmlDocumentationPath(projectXmlPath!));
                return _xmlDocumentationCacheByPath.GetOrAdd(projectXmlPath!, LoadXmlDocumentationCache);
            }

            string? assemblySimpleName = assembly.GetName().Name;
            if (string.IsNullOrEmpty(assemblySimpleName))
                return null;

            string[] probeDirectories =
            {
                AppContext.BaseDirectory,
                System.Environment.CurrentDirectory,
            };

            foreach (string probeDirectory in probeDirectories)
            {
                if (string.IsNullOrEmpty(probeDirectory))
                    continue;

                string candidate = Path.Combine(probeDirectory, assemblySimpleName + ".xml");
                if (System.IO.File.Exists(candidate))
                    return _xmlDocumentationCacheByPath.GetOrAdd(candidate, LoadXmlDocumentationCache);
            }

            return null;
        }

        /// <summary>
        /// Tries to find the XML documentation file in the project's build output directory.
        /// This is needed because project assemblies are loaded from a memory stream
        /// (to avoid file locking), which causes <see cref="Assembly.Location"/> to be empty.
        /// </summary>
        private static bool TryGetXmlPathFromProjectBuildOutput(Assembly assembly, out string? xmlPath)
        {
            xmlPath = null;

            string? assemblyName = assembly.GetName().Name;
            if (string.IsNullOrEmpty(assemblyName))
                return false;

            try
            {
                // Derive the build output directory from the known metadata directory.
                // ResMetadataDir returns e.g. "res://.godot/mono/metadata".
                // The editor build output is at "res://.godot/mono/temp/bin/Debug/".
                string globalMetadataDir = ProjectSettings.GlobalizePath(
                    Internals.GodotSharpDirs.ResMetadataDir);
                string? monoDir = Path.GetDirectoryName(globalMetadataDir);
                if (string.IsNullOrEmpty(monoDir))
                    return false;

                string tempAssembliesDir = Path.Combine(monoDir, "temp", "bin", "Debug");
                string candidate = Path.Combine(tempAssembliesDir, assemblyName + ".xml");
                if (System.IO.File.Exists(candidate))
                {
                    xmlPath = candidate;
                    return true;
                }
            }
            catch
            {
                // Ignore errors when accessing project directories.
            }

            return false;
        }

        public static string? TryGetDocTag(
            XmlDocumentationCache? cache,
            Type contextType,
            string memberId,
            string tagName,
            Func<string?>? implicitInheritResolver = null,
            HashSet<string>? visited = null)
        {
            if (cache == null)
                return null;

            return XmlDocToBBCode.TryGetTagBbCode(
                cache.Members,
                contextType,
                memberId,
                tagName,
                implicitInheritResolver,
                visited);
        }

        /// <summary>
        /// Gets the full documentation description for a member, combining its
        /// <c>summary</c>, <c>remarks</c>, <c>param</c>, <c>returns</c> and <c>exception</c>
        /// XML documentation tags into a single BBCode string (sectioned like GDScript docs).
        /// </summary>
        public static string? TryGetFullDocDescription(
            XmlDocumentationCache? cache,
            Type contextType,
            string memberId,
            Func<string?>? implicitInheritResolver = null)
        {
            if (cache == null)
                return null;

            return XmlDocToBBCode.TryGetFullDescriptionBbCode(
                cache.Members,
                contextType,
                memberId,
                implicitInheritResolver);
        }

        public static string GetPropertyDocTypeName(Type type)
        {
            if (type.IsEnum)
                return "int";

            // void only applies to method return types (properties are never void).
            // Map it to the documentation convention used by GDScript.
            if (type == typeof(void))
                return "void";

            // Typed arrays use the "T[]" format, matching the GDScript documentation generator
            // (gdscript_docgen.cpp appends "[]" for typed arrays). This is the format the editor
            // documentation renderer (_add_type_to_rt in editor_help.cpp) recognizes via
            // ends_with("[]") to produce clickable type links for the element type.
            // e.g. float[] -> "float[]", Array<Vector3> -> "Vector3[]".
            if (type.IsArray)
            {
                Type? elementType = type.GetElementType();
                if (elementType != null)
                    return $"{GetPropertyDocTypeName(elementType)}[]";
            }

            // Godot typed collections.
            // Array<T> -> "T[]" (same typed-array format as above, for clickable element links).
            // Dictionary<TKey, TValue> -> "Dictionary[TKey, TValue]" (matches the renderer's
            // begins_with("Dictionary[") branch, which splits key/value into separate links).
            // Using these formats avoids the raw reflection names ("Array`1", "Dictionary`2")
            // leaking into the docs.
            if (type.IsConstructedGenericType)
            {
                Type genericDefinition = type.GetGenericTypeDefinition();
                Type[] genericArguments = type.GetGenericArguments();

                if (genericDefinition == typeof(Godot.Collections.Array<>))
                    return $"{GetPropertyDocTypeName(genericArguments[0])}[]";

                if (genericDefinition == typeof(Godot.Collections.Dictionary<,>))
                    return $"Dictionary[{GetPropertyDocTypeName(genericArguments[0])}, {GetPropertyDocTypeName(genericArguments[1])}]";
            }

            return Type.GetTypeCode(type) switch
            {
                TypeCode.Boolean => "bool",
                TypeCode.Char => "int",
                TypeCode.SByte => "int",
                TypeCode.Byte => "int",
                TypeCode.Int16 => "int",
                TypeCode.UInt16 => "int",
                TypeCode.Int32 => "int",
                TypeCode.UInt32 => "int",
                TypeCode.Int64 => "int",
                TypeCode.UInt64 => "int",
                TypeCode.Single => "float",
                TypeCode.Double => "float",
                TypeCode.String => "String",
                _ => type == typeof(Variant) ? "Variant" : type.Name,
            };
        }

        public static string? FindInheritedPropertyDocumentationId(ReflectionPropertyInfo property)
        {
            Type? baseType = property.DeclaringType?.BaseType;

            while (baseType != null)
            {
                ReflectionPropertyInfo? inherited = baseType.GetProperty(property.Name,
                    BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);

                if (inherited != null)
                    return XmlDocToBBCode.GetPropertyDocumentationId(inherited);

                baseType = baseType.BaseType;
            }

            return null;
        }

        public static string? FindInheritedFieldDocumentationId(ReflectionFieldInfo field)
        {
            Type? baseType = field.DeclaringType?.BaseType;

            while (baseType != null)
            {
                ReflectionFieldInfo? inherited = baseType.GetField(field.Name,
                    BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);

                if (inherited != null)
                    return XmlDocToBBCode.GetFieldDocumentationId(inherited);

                baseType = baseType.BaseType;
            }

            return null;
        }

        /// <summary>
        /// Clear cache when assemblies are reloaded
        /// </summary>
        public static void ClearXmlCache()
        {
            _xmlDocumentationCacheByPath.Clear();
            _xmlDocPathByAssembly.Clear();
        }
    }
}
