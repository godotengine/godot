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

        internal static string GetTypeDocumentationId(Type type)
        {
            Type xmlType = type.IsConstructedGenericType ? type.GetGenericTypeDefinition() : type;
            string fullName = xmlType.FullName ?? throw new InvalidOperationException($"Unable to resolve full name for type '{type}'.");
            return $"T:{fullName.Replace('+', '.')}";
        }

        internal static string GetPropertyDocumentationId(System.Reflection.PropertyInfo property)
            => $"P:{GetTypeDocumentationId(property.DeclaringType!).Substring(2)}.{property.Name}";

        internal static string GetFieldDocumentationId(System.Reflection.FieldInfo field)
            => $"F:{GetTypeDocumentationId(field.DeclaringType!).Substring(2)}.{field.Name}";

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

        public static string GetPropertyDocTypeName(Type type)
        {
            if (type.IsEnum)
                return "int";

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
