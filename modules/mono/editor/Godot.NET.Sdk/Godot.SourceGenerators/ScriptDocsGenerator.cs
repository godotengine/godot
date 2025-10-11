using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Godot.SourceGenerators.MarshalUtils;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptDocsGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotSourceGeneratorDisabled("ScriptDocs"))
                return;

            if (context.IsGodotToolsProject())
                return;

            // NOTE: NotNullWhen diagnostics don't work on projects targeting .NET Standard 2.0
            // ReSharper disable once ReplaceWithStringIsNullOrEmpty
            if (!context.TryGetGlobalAnalyzerProperty("GodotProjectDirBase64", out string? godotProjectDir) || godotProjectDir!.Length == 0)
            {
                if (!context.TryGetGlobalAnalyzerProperty("GodotProjectDir", out godotProjectDir) || godotProjectDir!.Length == 0)
                {
                    throw new InvalidOperationException("Property 'GodotProjectDir' is null or empty.");
                }
            }
            else
            {
                // Workaround for https://github.com/dotnet/roslyn/issues/51692
                godotProjectDir = Encoding.UTF8.GetString(Convert.FromBase64String(godotProjectDir));
            }

            INamedTypeSymbol[] godotClasses = context
                .Compilation.SyntaxTrees
                .SelectMany(tree =>
                    tree.GetRoot().DescendantNodes()
                        .OfType<ClassDeclarationSyntax>()
                        .SelectGodotScriptClasses(context.Compilation)
                        // Report and skip non-partial classes
                        .Where(x =>
                        {
                            if (x.cds.IsPartial())
                            {
                                if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out _))
                                {
                                    return false;
                                }

                                return true;
                            }

                            return false;
                        })
                        .Select(x => x.symbol)
                )
                .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
                .ToArray();

            if (godotClasses.Length > 0)
            {
                var typeCache = new MarshalUtils.TypeCache(context.Compilation);

                foreach (var godotClass in godotClasses)
                {
                    VisitGodotScriptClass(context, typeCache, godotClass, godotProjectDir);
                }
            }
        }

        private static void VisitGodotScriptClass(
            GeneratorExecutionContext context,
            MarshalUtils.TypeCache typeCache,
            INamedTypeSymbol symbol,
            string godotProjectDir
        )
        {
            symbol.GetDocumentationSummaryText(out string? briefDescription, out string? classDescription);

            StringBuilder docPropertyString = new StringBuilder();

            var members = symbol.GetMembers();

            var exportedProperties = members
                .Where(s => s.Kind == SymbolKind.Property)
                .Cast<IPropertySymbol>()
                .Where(s => s.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false))
                .ToArray();

            var exportedFields = members
                .Where(s => s.Kind == SymbolKind.Field && !s.IsImplicitlyDeclared)
                .Cast<IFieldSymbol>()
                .Where(s => s.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false))
                .ToArray();

            var enumRegistration = new HashSet<ITypeSymbol>();

            foreach (var property in exportedProperties)
            {
                GeneratePropertyDoc(docPropertyString, property, typeCache, symbol, enumRegistration);
            }

            foreach (var field in exportedFields)
            {
                GeneratePropertyDoc(docPropertyString, field, typeCache, symbol, enumRegistration);
            }

            StringBuilder docSignalString = new StringBuilder();

            var signalDelegateSymbols = members
                .Where(s => s.Kind == SymbolKind.NamedType)
                .Cast<INamedTypeSymbol>()
                .Where(namedTypeSymbol => namedTypeSymbol.TypeKind == TypeKind.Delegate)
                .Where(s => s.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotSignalAttribute() ?? false))
                .ToArray();

            foreach (var signalDelegateSymbol in signalDelegateSymbols)
            {
                GenerateSignalDoc(docSignalString, signalDelegateSymbol);
            }

            StringBuilder docEnumString = new StringBuilder();
            StringBuilder docConstantString = new StringBuilder();

            foreach (var enumType in enumRegistration)
            {
                GenerateEnumRegistration(docEnumString, docConstantString, enumType);
            }

            if (string.IsNullOrWhiteSpace(classDescription) && docPropertyString.Length == 0 && docSignalString.Length == 0 && docEnumString.Length == 0 && docConstantString.Length == 0)
            {
                // Script has no doc.
                return;
            }

            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol is { IsGlobalNamespace: false }
                ? namespaceSymbol.FullQualifiedNameOmitGlobal()
                : string.Empty;
            bool hasNamespace = classNs.Length != 0;

            bool isInnerClass = symbol.ContainingType != null;

            string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                                + "_ScriptDocs.generated";

            var source = new StringBuilder();

            if (hasNamespace)
            {
                source.Append("namespace ");
                source.Append(classNs);
                source.Append(" {\n\n");
            }

            if (isInnerClass)
            {
                var containingType = symbol.ContainingType;
                AppendPartialContainingTypeDeclarations(containingType);

                void AppendPartialContainingTypeDeclarations(INamedTypeSymbol? containingType)
                {
                    if (containingType == null)
                        return;

                    AppendPartialContainingTypeDeclarations(containingType.ContainingType);

                    source.Append("partial ");
                    source.Append(containingType.GetDeclarationKeyword());
                    source.Append(" ");
                    source.Append(containingType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
                    source.Append("\n{\n");
                }
            }

            source.Append("partial class ");
            source.Append(symbol.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
            source.Append("\n{\n");

            source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");
            source.Append("#if TOOLS\n");

            source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
            source.Append("    internal new static global::Godot.Collections.Dictionary GetGodotClassDocs()\n    {\n");
            source.Append("        var docs = new global::Godot.Collections.Dictionary();\n");
            source.Append("        docs.Add(\"name\",\"");

            string scriptPath = ScriptPathAttributeGenerator.RelativeToDir(symbol.DeclaringSyntaxReferences.First().SyntaxTree.FilePath, godotProjectDir);

            if (symbol.GetAttributes().Any(a => a.AttributeClass?.IsGodotGlobalClassAttribute() ?? false))
            {
                source.Append(symbol.Name);
            }
            else
            {
                source.Append($"\\\"{scriptPath}\\\"");
                if (isInnerClass)
                {
                    source.Append($".{symbol.Name}");
                }
            }
            source.Append("\");\n");
            source.Append("        docs.Add(\"brief_description\",@\"");
            source.Append(briefDescription);
            source.Append("\");\n");
            source.Append("        docs.Add(\"description\",@\"");
            source.Append(classDescription);
            source.Append("\");\n\n");

            if (docPropertyString.Length > 0)
            {
                source.Append("        var propertyDocs = new global::Godot.Collections.Array();\n");
                source.Append(docPropertyString);
                source.Append("        docs.Add(\"properties\", propertyDocs);\n\n");
            }

            if (docSignalString.Length > 0)
            {
                source.Append("        var signalDocs  = new global::Godot.Collections.Array();\n");
                source.Append(docSignalString);
                source.Append("        docs.Add(\"signals\", signalDocs);\n\n");
            }

            if (docEnumString.Length > 0)
            {
                source.Append("        var enumDocs = new global::Godot.Collections.Dictionary();\n");
                source.Append(docEnumString);
                source.Append("        docs.Add(\"enums\", enumDocs);\n\n");
            }

            if (docConstantString.Length > 0)
            {
                source.Append("        var constantDocs = new global::Godot.Collections.Array();\n");
                source.Append(docConstantString);
                source.Append("        docs.Add(\"constants\", constantDocs);\n\n");
            }

            source.Append("        docs.Add(\"is_script_doc\", true);\n\n");
            source.Append("        docs.Add(\"script_path\", \"").Append(scriptPath).Append("\");\n\n");

            source.Append("        return docs;\n    }\n\n");

            source.Append("#endif // TOOLS\n");

            source.Append("#pragma warning restore CS0109\n");

            source.Append("}\n"); // partial class

            if (isInnerClass)
            {
                var containingType = symbol.ContainingType;

                while (containingType != null)
                {
                    source.Append("}\n"); // outer class

                    containingType = containingType.ContainingType;
                }
            }

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private const string FallbackDoc = "There is currently no description for this property.";

        private static void GeneratePropertyDoc(StringBuilder docPropertyString, ISymbol symbol, TypeCache typeCache, ITypeSymbol containingType, HashSet<ITypeSymbol> enumRegistration)
        {
            var propertySymbol = symbol as IPropertySymbol;
            var fieldSymbol = symbol as IFieldSymbol;
            var memberType = propertySymbol?.Type ?? fieldSymbol!.Type;
            var typeInfo = new Dictionary<string, string>();
            ConvertManagedTypeToDocTypeString(memberType, typeCache, containingType, enumRegistration, typeInfo);
            docPropertyString.Append("        propertyDocs.Add(new global::Godot.Collections.Dictionary { { \"name\", PropertyName.")
                .Append(symbol.Name).Append(" }");
            foreach (var info in typeInfo)
            {
                docPropertyString.Append(", { @\"").Append(info.Key).Append("\", @\"").Append(info.Value).Append("\" }");
            }

            symbol.GetDocumentationSummaryText(out _, out string? text);
            if (!string.IsNullOrWhiteSpace(text))
            {
                docPropertyString.Append(", { \"description\", @\"").Append(text).Append("\" }");
            }
            else
            {
                docPropertyString.Append(", { \"description\", @\"").Append(FallbackDoc).Append("\" }");
            }

            docPropertyString.Append("});\n");
        }

        private static void GenerateEnumRegistration(StringBuilder enumRegistrationString, StringBuilder constantRegistrationString, ITypeSymbol enumType)
        {
            enumRegistrationString.Append("        enumDocs.Add(").Append("\"").Append(enumType.Name).Append("\", new global::Godot.Collections.Dictionary {");
            bool isFlags = enumType.GetAttributes().Any(a => a.AttributeClass?.ToDisplayString() == "System.FlagsAttribute");

            enumType.GetDocumentationSummaryText(out _, out string? text);
            if (!string.IsNullOrWhiteSpace(text))
            {
                enumRegistrationString.Append("{ \"description\", @\"").Append(text).Append("\" }");
            }
            else
            {
                enumRegistrationString.Append("{ \"description\", @\"").Append(FallbackDoc).Append("\" }");
            }

            enumRegistrationString.Append("});\n");

            var enumMembers = enumType.GetMembers().Where(m => m.Kind == SymbolKind.Field).Cast<IFieldSymbol>().Where(f => f.ConstantValue != null);
            foreach (var member in enumMembers)
            {
                constantRegistrationString.Append("        constantDocs.Add(new global::Godot.Collections.Dictionary {")
                    .Append(" { \"name\", @\"").Append(member.Name).Append("\" }, ")
                    .Append("{ \"value\", @\"").Append(member.ConstantValue).Append("\" }, ")
                    .Append("{ \"enumeration\", @\"").Append(enumType.Name).Append("\" }");

                if (isFlags)
                {
                    constantRegistrationString.Append(", { \"isFlags\", @\"true\" }");
                }

                member.GetDocumentationSummaryText(out _, out string? memberSummary);
                if (!string.IsNullOrWhiteSpace(memberSummary))
                {
                    constantRegistrationString.Append(", { \"description\", @\"").Append(memberSummary).Append("\" }");
                }
                else
                {
                    constantRegistrationString.Append(", { \"description\", @\"").Append(FallbackDoc).Append("\" }");
                }

                constantRegistrationString.Append("});\n");
            }
        }

        private static void GenerateSignalDoc(StringBuilder docSignalString, ISymbol symbol)
        {
            string signalName = symbol.Name;
            const string SignalDelegateSuffix = "EventHandler";
            signalName = signalName.Substring(0, signalName.Length - SignalDelegateSuffix.Length);
            docSignalString.Append("        signalDocs.Add(new global::Godot.Collections.Dictionary { { \"name\", SignalName.")
                .Append(signalName).Append(" }");
            symbol.GetDocumentationSummaryText(out _, out string? text);
            if (!string.IsNullOrWhiteSpace(text))
            {
                docSignalString.Append(", { \"description\", @\"").Append(text).Append("\" }");
            }
            else
            {
                docSignalString.Append(", { \"description\", @\"").Append(FallbackDoc).Append("\" }");
            }
            docSignalString.Append("});\n");
        }

        private static void ConvertManagedTypeToDocTypeString(ITypeSymbol typeSymbol, TypeCache typeCache,
            ITypeSymbol containingType, HashSet<ITypeSymbol> enumRegistration, Dictionary<string, string> typeInfo)
        {
            typeInfo["type"] = "Variant";
            var marshalType = ConvertManagedTypeToMarshalType(typeSymbol, typeCache);
            if (!marshalType.HasValue)
            {
                return;
            }

            var marshalTypeValue = marshalType.Value;
            string typeString;

            switch (marshalTypeValue)
            {
                case MarshalType.Boolean:
                    typeString = "bool";
                    break;
                case MarshalType.Char:
                case MarshalType.SByte:
                case MarshalType.Int16:
                case MarshalType.Int32:
                case MarshalType.Int64:
                case MarshalType.Byte:
                case MarshalType.UInt16:
                case MarshalType.UInt32:
                case MarshalType.UInt64:
                    typeString = "int";
                    break;
                case MarshalType.Single:
                case MarshalType.Double:
                    typeString = "float";
                    break;
                case MarshalType.String:
                    typeString = "String";
                    break;
                case MarshalType.Vector2:
                    typeString = "Vector2";
                    break;
                case MarshalType.Vector2I:
                    typeString = "Vector2i";
                    break;
                case MarshalType.Rect2:
                    typeString = "Rect2";
                    break;
                case MarshalType.Rect2I:
                    typeString = "Rect2i";
                    break;
                case MarshalType.Transform2D:
                    typeString = "Transform2D";
                    break;
                case MarshalType.Vector3:
                    typeString = "Vector3";
                    break;
                case MarshalType.Vector3I:
                    typeString = "Vector3i";
                    break;
                case MarshalType.Basis:
                    typeString = "Basis";
                    break;
                case MarshalType.Quaternion:
                    typeString = "Quaternion";
                    break;
                case MarshalType.Transform3D:
                    typeString = "Transform3D";
                    break;
                case MarshalType.Vector4:
                    typeString = "Vector4";
                    break;
                case MarshalType.Vector4I:
                    typeString = "Vector4i";
                    break;
                case MarshalType.Projection:
                    typeString = "Projection";
                    break;
                case MarshalType.Aabb:
                    typeString = "AABB";
                    break;
                case MarshalType.Color:
                    typeString = "Color";
                    break;
                case MarshalType.Plane:
                    typeString = "Plane";
                    break;
                case MarshalType.Callable:
                    typeString = "Callable";
                    break;
                case MarshalType.Signal:
                    typeString = "Signal";
                    break;
                case MarshalType.ByteArray:
                    typeString = "PackedByteArray";
                    break;
                case MarshalType.Int32Array:
                    typeString = "PackedInt32Array";
                    break;
                case MarshalType.Int64Array:
                    typeString = "PackedInt64Array";
                    break;
                case MarshalType.Float32Array:
                    typeString = "PackedFloat32Array";
                    break;
                case MarshalType.Float64Array:
                    typeString = "PackedFloat64Array";
                    break;
                case MarshalType.StringArray:
                    typeString = "PackedStringArray";
                    break;
                case MarshalType.Vector2Array:
                    typeString = "PackedVector2Array";
                    break;
                case MarshalType.Vector3Array:
                    typeString = "PackedVector3Array";
                    break;
                case MarshalType.Vector4Array:
                    typeString = "PackedVector4Array";
                    break;
                case MarshalType.ColorArray:
                    typeString = "PackedColorArray";
                    break;
                case MarshalType.Variant:
                    typeString = "Variant";
                    break;
                case MarshalType.StringName:
                    typeString = "StringName";
                    break;
                case MarshalType.NodePath:
                    typeString = "NodePath";
                    break;
                case MarshalType.Rid:
                    typeString = "RID";
                    break;
                case MarshalType.GodotDictionary:
                    typeString = "Dictionary";
                    break;
                case MarshalType.GodotArray:
                    typeString = "Array";
                    break;
                case MarshalType.SystemArrayOfStringName:
                    typeString = "StringName[]";
                    break;
                case MarshalType.SystemArrayOfNodePath:
                    typeString = "NodePath[]";
                    break;
                case MarshalType.SystemArrayOfRid:
                    typeString = "RID[]";
                    break;
                case MarshalType.GodotObjectOrDerived:
                    typeString = typeSymbol.Name;
                    break;
                case MarshalType.GodotObjectOrDerivedArray:
                {
                    var arrayType = (IArrayTypeSymbol)typeSymbol;
                    var elementType = arrayType.ElementType;
                    typeString = $"{elementType.Name}[]";
                    break;
                }
                case MarshalType.GodotGenericArray:
                {
                    var innerType = ((INamedTypeSymbol)typeSymbol).TypeArguments[0];
                    if (innerType.TypeKind == TypeKind.Enum)
                    {
                        enumRegistration.Add(innerType);
                        typeString = $"{containingType.Name}.{innerType.Name}[]";
                    }
                    else
                    {
                        var innerTypeInfo = new Dictionary<string, string>();
                        ConvertManagedTypeToDocTypeString(innerType, typeCache, containingType, enumRegistration, innerTypeInfo);
                        typeString = $"{innerTypeInfo["type"]}[]";
                    }
                    break;
                }
                case MarshalType.GodotGenericDictionary:
                {
                    var namedTypeSymbol = (INamedTypeSymbol)typeSymbol;
                    var keyType = namedTypeSymbol.TypeArguments[0];
                    var valueType = namedTypeSymbol.TypeArguments[1];
                    string keyTypeName;
                    string valueTypeName;
                    if (keyType.TypeKind == TypeKind.Enum)
                    {
                        enumRegistration.Add(keyType);
                        keyTypeName = $"{containingType.Name}.{keyType.Name}";
                    }
                    else
                    {
                        var innerTypeInfo = new Dictionary<string, string>();
                        ConvertManagedTypeToDocTypeString(keyType, typeCache, containingType, enumRegistration, innerTypeInfo);
                        keyTypeName = innerTypeInfo["type"];
                    }
                    if (valueType.TypeKind == TypeKind.Enum)
                    {
                        enumRegistration.Add(valueType);
                        valueTypeName = $"{containingType.Name}.{valueType.Name}";
                    }
                    else
                    {
                        var innerTypeInfo = new Dictionary<string, string>();
                        ConvertManagedTypeToDocTypeString(valueType, typeCache, containingType, enumRegistration, innerTypeInfo);
                        valueTypeName = innerTypeInfo["type"];
                    }
                    typeString = $"Dictionary[{keyTypeName}, {valueTypeName}]";
                    break;
                }
                case MarshalType.Enum:
                {
                    typeString = "int";
                    typeInfo["enumeration"] = $"{containingType.Name}.{typeSymbol.Name}";
                    enumRegistration.Add(typeSymbol);
                    var isFlags = typeSymbol.GetAttributes()
                        .Any(a => a.AttributeClass?.ToDisplayString() == "System.FlagsAttribute");
                    if (isFlags)
                    {
                        typeInfo["is_bitfield"] = "true";
                    }
                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException();
            }
            typeInfo["type"] = typeString;
        }
    }
}
