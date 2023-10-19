using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptPropertiesGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotSourceGeneratorDisabled("ScriptProperties"))
                return;

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
                                if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out var typeMissingPartial))
                                {
                                    Common.ReportNonPartialGodotScriptOuterClass(context, typeMissingPartial!);
                                    return false;
                                }

                                return true;
                            }

                            Common.ReportNonPartialGodotScriptClass(context, x.cds, x.symbol);
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
                    VisitGodotScriptClass(context, typeCache, godotClass);
                }
            }
        }

        private static void VisitGodotScriptClass(
            GeneratorExecutionContext context,
            MarshalUtils.TypeCache typeCache,
            INamedTypeSymbol symbol
        )
        {
            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedNameOmitGlobal() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            bool isInnerClass = symbol.ContainingType != null;

            string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                                + "_ScriptProperties.generated";

            var source = new StringBuilder();

            source.Append("using Godot;\n");
            source.Append("using Godot.NativeInterop;\n");
            source.Append("\n");

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
                    source.Append(containingType.NameWithTypeParameters());
                    source.Append("\n{\n");
                }
            }

            source.Append("partial class ");
            source.Append(symbol.NameWithTypeParameters());
            source.Append("\n{\n");

            var members = symbol.GetMembers();

            var propertySymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Property)
                .Cast<IPropertySymbol>()
                .Where(s => !s.IsIndexer && s.ExplicitInterfaceImplementations.Length == 0);

            var fieldSymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Field && !s.IsImplicitlyDeclared)
                .Cast<IFieldSymbol>();

            var godotClassProperties = propertySymbols.WhereIsGodotCompatibleType(typeCache).ToArray();
            var godotClassFields = fieldSymbols.WhereIsGodotCompatibleType(typeCache).ToArray();

            source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");

            source.Append("    /// <summary>\n")
                .Append("    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.\n")
                .Append("    /// </summary>\n");

            source.Append(
                $"    public new class PropertyName : {symbol.BaseType.FullQualifiedNameIncludeGlobal()}.PropertyName {{\n");

            // Generate cached StringNames for methods and properties, for fast lookup

            foreach (var property in godotClassProperties)
            {
                string propertyName = property.PropertySymbol.Name;

                source.Append("        /// <summary>\n")
                    .Append("        /// Cached name for the '")
                    .Append(propertyName)
                    .Append("' property.\n")
                    .Append("        /// </summary>\n");

                source.Append("        public new static readonly global::Godot.StringName ");
                source.Append(propertyName);
                source.Append(" = \"");
                source.Append(propertyName);
                source.Append("\";\n");
            }

            foreach (var field in godotClassFields)
            {
                string fieldName = field.FieldSymbol.Name;

                source.Append("        /// <summary>\n")
                    .Append("        /// Cached name for the '")
                    .Append(fieldName)
                    .Append("' field.\n")
                    .Append("        /// </summary>\n");

                source.Append("        public new static readonly global::Godot.StringName ");
                source.Append(fieldName);
                source.Append(" = \"");
                source.Append(fieldName);
                source.Append("\";\n");
            }

            source.Append("    }\n"); // class GodotInternal

            if (godotClassProperties.Length > 0 || godotClassFields.Length > 0)
            {
                bool isFirstEntry;

                // Generate SetGodotClassPropertyValue

                bool allPropertiesAreReadOnly = godotClassFields.All(fi => fi.FieldSymbol.IsReadOnly) &&
                                                godotClassProperties.All(pi => pi.PropertySymbol.IsReadOnly || pi.PropertySymbol.SetMethod!.IsInitOnly);

                if (!allPropertiesAreReadOnly)
                {
                    source.Append("    /// <inheritdoc/>\n");
                    source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                    source.Append("    protected override bool SetGodotClassPropertyValue(in godot_string_name name, ");
                    source.Append("in godot_variant value)\n    {\n");

                    isFirstEntry = true;
                    foreach (var property in godotClassProperties)
                    {
                        if (property.PropertySymbol.IsReadOnly || property.PropertySymbol.SetMethod!.IsInitOnly)
                            continue;

                        GeneratePropertySetter(property.PropertySymbol.Name,
                            property.PropertySymbol.Type, property.Type, source, isFirstEntry);
                        isFirstEntry = false;
                    }

                    foreach (var field in godotClassFields)
                    {
                        if (field.FieldSymbol.IsReadOnly)
                            continue;

                        GeneratePropertySetter(field.FieldSymbol.Name,
                            field.FieldSymbol.Type, field.Type, source, isFirstEntry);
                        isFirstEntry = false;
                    }

                    source.Append("        return base.SetGodotClassPropertyValue(name, value);\n");

                    source.Append("    }\n");
                }

                // Generate GetGodotClassPropertyValue
                bool allPropertiesAreWriteOnly = godotClassFields.Length == 0 && godotClassProperties.All(pi => pi.PropertySymbol.IsWriteOnly);

                if (!allPropertiesAreWriteOnly)
                {
                    source.Append("    /// <inheritdoc/>\n");
                    source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                    source.Append("    protected override bool GetGodotClassPropertyValue(in godot_string_name name, ");
                    source.Append("out godot_variant value)\n    {\n");

                    isFirstEntry = true;
                    foreach (var property in godotClassProperties)
                    {
                        if (property.PropertySymbol.IsWriteOnly)
                            continue;

                        GeneratePropertyGetter(property.PropertySymbol.Name,
                            property.PropertySymbol.Type, property.Type, source, isFirstEntry);
                        isFirstEntry = false;
                    }

                    foreach (var field in godotClassFields)
                    {
                        GeneratePropertyGetter(field.FieldSymbol.Name,
                            field.FieldSymbol.Type, field.Type, source, isFirstEntry);
                        isFirstEntry = false;
                    }

                    source.Append("        return base.GetGodotClassPropertyValue(name, out value);\n");

                    source.Append("    }\n");
                }
                // Generate GetGodotPropertyList

                const string dictionaryType = "global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>";

                source.Append("    /// <summary>\n")
                    .Append("    /// Get the property information for all the properties declared in this class.\n")
                    .Append("    /// This method is used by Godot to register the available properties in the editor.\n")
                    .Append("    /// Do not call this method.\n")
                    .Append("    /// </summary>\n");

                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");

                source.Append("    internal new static ")
                    .Append(dictionaryType)
                    .Append(" GetGodotPropertyList()\n    {\n");

                source.Append("        var properties = new ")
                    .Append(dictionaryType)
                    .Append("();\n");

                // To retain the definition order (and display categories correctly), we want to
                //  iterate over fields and properties at the same time, sorted by line number.
                var godotClassPropertiesAndFields = Enumerable.Empty<GodotPropertyOrFieldData>()
                    .Concat(godotClassProperties.Select(propertyData => new GodotPropertyOrFieldData(propertyData)))
                    .Concat(godotClassFields.Select(fieldData => new GodotPropertyOrFieldData(fieldData)))
                    .OrderBy(data => data.Symbol.Locations[0].Path())
                    .ThenBy(data => data.Symbol.Locations[0].StartLine());

                foreach (var member in godotClassPropertiesAndFields)
                {
                    foreach (var groupingInfo in DetermineGroupingPropertyInfo(member.Symbol))
                        AppendGroupingPropertyInfo(source, groupingInfo);

                    var propertyInfo = DeterminePropertyInfo(context, typeCache,
                        member.Symbol, member.Type);

                    if (propertyInfo == null)
                        continue;

                    AppendPropertyInfo(source, propertyInfo.Value);
                }

                source.Append("        return properties;\n");
                source.Append("    }\n");

                source.Append("#pragma warning restore CS0109\n");
            }

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

        private static void GeneratePropertySetter(
            string propertyMemberName,
            ITypeSymbol propertyTypeSymbol,
            MarshalType propertyMarshalType,
            StringBuilder source,
            bool isFirstEntry
        )
        {
            source.Append("        ");

            if (!isFirstEntry)
                source.Append("else ");

            source.Append("if (name == PropertyName.")
                .Append(propertyMemberName)
                .Append(") {\n")
                .Append("            this.")
                .Append(propertyMemberName)
                .Append(" = ")
                .AppendNativeVariantToManagedExpr("value", propertyTypeSymbol, propertyMarshalType)
                .Append(";\n")
                .Append("            return true;\n")
                .Append("        }\n");
        }

        private static void GeneratePropertyGetter(
            string propertyMemberName,
            ITypeSymbol propertyTypeSymbol,
            MarshalType propertyMarshalType,
            StringBuilder source,
            bool isFirstEntry
        )
        {
            source.Append("        ");

            if (!isFirstEntry)
                source.Append("else ");

            source.Append("if (name == PropertyName.")
                .Append(propertyMemberName)
                .Append(") {\n")
                .Append("            value = ")
                .AppendManagedToNativeVariantExpr("this." + propertyMemberName,
                    propertyTypeSymbol, propertyMarshalType)
                .Append(";\n")
                .Append("            return true;\n")
                .Append("        }\n");
        }

        private static void AppendGroupingPropertyInfo(StringBuilder source, PropertyInfo propertyInfo)
        {
            source.Append("        properties.Add(new(type: (global::Godot.Variant.Type)")
                .Append((int)VariantType.Nil)
                .Append(", name: \"")
                .Append(propertyInfo.Name)
                .Append("\", hint: (global::Godot.PropertyHint)")
                .Append((int)PropertyHint.None)
                .Append(", hintString: \"")
                .Append(propertyInfo.HintString)
                .Append("\", usage: (global::Godot.PropertyUsageFlags)")
                .Append((int)propertyInfo.Usage)
                .Append(", exported: true));\n");
        }

        private static void AppendPropertyInfo(StringBuilder source, PropertyInfo propertyInfo)
        {
            source.Append("        properties.Add(new(type: (global::Godot.Variant.Type)")
                .Append((int)propertyInfo.Type)
                .Append(", name: PropertyName.")
                .Append(propertyInfo.Name)
                .Append(", hint: (global::Godot.PropertyHint)")
                .Append((int)propertyInfo.Hint)
                .Append(", hintString: \"")
                .Append(propertyInfo.HintString)
                .Append("\", usage: (global::Godot.PropertyUsageFlags)")
                .Append((int)propertyInfo.Usage)
                .Append(", exported: ")
                .Append(propertyInfo.Exported ? "true" : "false")
                .Append("));\n");
        }

        private static IEnumerable<PropertyInfo> DetermineGroupingPropertyInfo(ISymbol memberSymbol)
        {
            foreach (var attr in memberSymbol.GetAttributes())
            {
                PropertyUsageFlags? propertyUsage = attr.AttributeClass?.FullQualifiedNameOmitGlobal() switch
                {
                    GodotClasses.ExportCategoryAttr => PropertyUsageFlags.Category,
                    GodotClasses.ExportGroupAttr => PropertyUsageFlags.Group,
                    GodotClasses.ExportSubgroupAttr => PropertyUsageFlags.Subgroup,
                    _ => null
                };

                if (propertyUsage is null)
                    continue;

                if (attr.ConstructorArguments.Length > 0 && attr.ConstructorArguments[0].Value is string name)
                {
                    string? hintString = null;
                    if (propertyUsage != PropertyUsageFlags.Category && attr.ConstructorArguments.Length > 1)
                        hintString = attr.ConstructorArguments[1].Value?.ToString();

                    yield return new PropertyInfo(VariantType.Nil, name, PropertyHint.None, hintString,
                        propertyUsage.Value, true);
                }
            }
        }

        private static PropertyInfo? DeterminePropertyInfo(
            GeneratorExecutionContext context,
            MarshalUtils.TypeCache typeCache,
            ISymbol memberSymbol,
            MarshalType marshalType
        )
        {
            var exportAttr = memberSymbol.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.IsGodotExportAttribute() ?? false);

            var propertySymbol = memberSymbol as IPropertySymbol;
            var fieldSymbol = memberSymbol as IFieldSymbol;

            if (exportAttr != null && propertySymbol != null)
            {
                if (propertySymbol.GetMethod == null)
                {
                    // This should never happen, as we filtered WriteOnly properties, but just in case.
                    Common.ReportExportedMemberIsWriteOnly(context, propertySymbol);
                    return null;
                }

                if (propertySymbol.SetMethod == null || propertySymbol.SetMethod.IsInitOnly)
                {
                    // This should never happen, as we filtered ReadOnly properties, but just in case.
                    Common.ReportExportedMemberIsReadOnly(context, propertySymbol);
                    return null;
                }
            }

            var memberType = propertySymbol?.Type ?? fieldSymbol!.Type;

            var memberVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(marshalType)!.Value;
            string memberName = memberSymbol.Name;

            if (exportAttr == null)
            {
                return new PropertyInfo(memberVariantType, memberName, PropertyHint.None,
                    hintString: null, PropertyUsageFlags.ScriptVariable, exported: false);
            }

            if (!TryGetMemberExportHint(typeCache, memberType, exportAttr, memberVariantType,
                    isTypeArgument: false, out var hint, out var hintString))
            {
                var constructorArguments = exportAttr.ConstructorArguments;

                if (constructorArguments.Length > 0)
                {
                    var hintValue = exportAttr.ConstructorArguments[0].Value;

                    hint = hintValue switch
                    {
                        null => PropertyHint.None,
                        int intValue => (PropertyHint)intValue,
                        _ => (PropertyHint)(long)hintValue
                    };

                    hintString = constructorArguments.Length > 1 ?
                        exportAttr.ConstructorArguments[1].Value?.ToString() :
                        null;
                }
                else
                {
                    hint = PropertyHint.None;
                }
            }

            var propUsage = PropertyUsageFlags.Default | PropertyUsageFlags.ScriptVariable;

            if (memberVariantType == VariantType.Nil)
                propUsage |= PropertyUsageFlags.NilIsVariant;

            return new PropertyInfo(memberVariantType, memberName,
                hint, hintString, propUsage, exported: true);
        }

        private static bool TryGetMemberExportHint(
            MarshalUtils.TypeCache typeCache,
            ITypeSymbol type, AttributeData exportAttr,
            VariantType variantType, bool isTypeArgument,
            out PropertyHint hint, out string? hintString
        )
        {
            hint = PropertyHint.None;
            hintString = null;

            if (variantType == VariantType.Nil)
                return true; // Variant, no export hint

            if (variantType == VariantType.Int &&
                type.IsValueType && type.TypeKind == TypeKind.Enum)
            {
                bool hasFlagsAttr = type.GetAttributes()
                    .Any(a => a.AttributeClass?.IsSystemFlagsAttribute() ?? false);

                hint = hasFlagsAttr ? PropertyHint.Flags : PropertyHint.Enum;

                var members = type.GetMembers();

                var enumFields = members
                    .Where(s => s.Kind == SymbolKind.Field && s.IsStatic &&
                                s.DeclaredAccessibility == Accessibility.Public &&
                                !s.IsImplicitlyDeclared)
                    .Cast<IFieldSymbol>().ToArray();

                var hintStringBuilder = new StringBuilder();
                var nameOnlyHintStringBuilder = new StringBuilder();

                // True: enum Foo { Bar, Baz, Qux }
                // True: enum Foo { Bar = 0, Baz = 1, Qux = 2 }
                // False: enum Foo { Bar = 0, Baz = 7, Qux = 5 }
                bool usesDefaultValues = true;

                for (int i = 0; i < enumFields.Length; i++)
                {
                    var enumField = enumFields[i];

                    if (i > 0)
                    {
                        hintStringBuilder.Append(",");
                        nameOnlyHintStringBuilder.Append(",");
                    }

                    string enumFieldName = enumField.Name;
                    hintStringBuilder.Append(enumFieldName);
                    nameOnlyHintStringBuilder.Append(enumFieldName);

                    long val = enumField.ConstantValue switch
                    {
                        sbyte v => v,
                        short v => v,
                        int v => v,
                        long v => v,
                        byte v => v,
                        ushort v => v,
                        uint v => v,
                        ulong v => (long)v,
                        _ => 0
                    };

                    uint expectedVal = (uint)(hint == PropertyHint.Flags ? 1 << i : i);
                    if (val != expectedVal)
                        usesDefaultValues = false;

                    hintStringBuilder.Append(":");
                    hintStringBuilder.Append(val);
                }

                hintString = !usesDefaultValues ?
                    hintStringBuilder.ToString() :
                    // If we use the format NAME:VAL, that's what the editor displays.
                    // That's annoying if the user is not using custom values for the enum constants.
                    // This may not be needed in the future if the editor is changed to not display values.
                    nameOnlyHintStringBuilder.ToString();

                return true;
            }

            if (variantType == VariantType.Object && type is INamedTypeSymbol memberNamedType)
            {
                if (memberNamedType.InheritsFrom("GodotSharp", "Godot.Resource"))
                {
                    hint = PropertyHint.ResourceType;
                    hintString = GetTypeName(memberNamedType);

                    return true;
                }

                if (memberNamedType.InheritsFrom("GodotSharp", "Godot.Node"))
                {
                    hint = PropertyHint.NodeType;
                    hintString = GetTypeName(memberNamedType);

                    return true;
                }
            }

            static string GetTypeName(INamedTypeSymbol memberSymbol)
            {
                if (memberSymbol.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotGlobalClassAttribute() ?? false))
                {
                    return memberSymbol.Name;
                }

                return memberSymbol.GetGodotScriptNativeClassName()!;
            }

            static bool GetStringArrayEnumHint(VariantType elementVariantType,
                AttributeData exportAttr, out string? hintString)
            {
                var constructorArguments = exportAttr.ConstructorArguments;

                if (constructorArguments.Length > 0)
                {
                    var presetHintValue = exportAttr.ConstructorArguments[0].Value;

                    PropertyHint presetHint = presetHintValue switch
                    {
                        null => PropertyHint.None,
                        int intValue => (PropertyHint)intValue,
                        _ => (PropertyHint)(long)presetHintValue
                    };

                    if (presetHint == PropertyHint.Enum)
                    {
                        string? presetHintString = constructorArguments.Length > 1 ?
                            exportAttr.ConstructorArguments[1].Value?.ToString() :
                            null;

                        hintString = (int)elementVariantType + "/" + (int)PropertyHint.Enum + ":";

                        if (presetHintString != null)
                            hintString += presetHintString;

                        return true;
                    }
                }

                hintString = null;
                return false;
            }

            if (!isTypeArgument && variantType == VariantType.Array)
            {
                var elementType = MarshalUtils.GetArrayElementType(type);

                if (elementType == null)
                    return false; // Non-generic Array, so there's no hint to add

                var elementMarshalType = MarshalUtils.ConvertManagedTypeToMarshalType(elementType, typeCache)!.Value;
                var elementVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(elementMarshalType)!.Value;

                bool isPresetHint = false;

                if (elementVariantType == VariantType.String || elementVariantType == VariantType.StringName)
                    isPresetHint = GetStringArrayEnumHint(elementVariantType, exportAttr, out hintString);

                if (!isPresetHint)
                {
                    bool hintRes = TryGetMemberExportHint(typeCache, elementType,
                        exportAttr, elementVariantType, isTypeArgument: true,
                        out var elementHint, out var elementHintString);

                    // Format: type/hint:hint_string
                    if (hintRes)
                    {
                        hintString = (int)elementVariantType + "/" + (int)elementHint + ":";

                        if (elementHintString != null)
                            hintString += elementHintString;
                    }
                    else
                    {
                        hintString = (int)elementVariantType + "/" + (int)PropertyHint.None + ":";
                    }
                }

                hint = PropertyHint.TypeString;

                return hintString != null;
            }

            if (!isTypeArgument && variantType == VariantType.PackedStringArray)
            {
                if (GetStringArrayEnumHint(VariantType.String, exportAttr, out hintString))
                {
                    hint = PropertyHint.TypeString;
                    return true;
                }
            }

            if (!isTypeArgument && variantType == VariantType.Dictionary)
            {
                // TODO: Dictionaries are not supported in the inspector
                return false;
            }

            return false;
        }
    }
}
