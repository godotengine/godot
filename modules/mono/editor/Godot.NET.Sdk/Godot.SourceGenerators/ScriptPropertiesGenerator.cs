using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
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
            bool isToolClass = symbol.GetAttributes().Any(a => a.AttributeClass?.IsGodotToolAttribute() ?? false);

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
                    source.Append(containingType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
                    source.Append("\n{\n");
                }
            }

            source.Append("partial class ");
            source.Append(symbol.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
            source.Append("\n{\n");

            var members = symbol.GetMembers()
                .Where(m => !m.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotIgnoreMemberAttribute() ?? false));

            var godotProperties = members
                .WhereIsSortedGodotProperties(
                    typeCache,
                    static members => members
                        .Where(s => !s.IsStatic && s.Kind == SymbolKind.Property)
                        .Cast<IPropertySymbol>()
                        .Where(s => !s.IsIndexer && s.ExplicitInterfaceImplementations.Length == 0)
                    ,
                    static members => members
                        .Where(s => !s.IsStatic && s.Kind == SymbolKind.Field && !s.IsImplicitlyDeclared)
                        .Cast<IFieldSymbol>())
                .ToArray();

            source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");

            source.Append("    /// <summary>\n")
                .Append("    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.\n")
                .Append("    /// </summary>\n");

            source.Append(
                $"    public new class PropertyName : {symbol.BaseType!.FullQualifiedNameIncludeGlobal()}.PropertyName {{\n");

            // Generate cached StringNames for methods and properties, for fast lookup

            foreach (var embed in godotProperties.Where(prop => prop.PropertyType == PropertyType.Embed))
            {
                if (!embed.PropertyTypeSymbol.IsReferenceType)
                    continue;

                source.Append("        /// <summary>\n")
                    .Append("        /// Cached name for the '")
                    .Append(embed.MemberName)
                    .Append("' embedded.\n")
                    .Append("        /// </summary>\n");

                source.Append("        public new static readonly global::Godot.StringName @").Append(embed.PropertyName).Append("_nonnull = \"");
                source.Append($"{embed.PropertyGroupName}/#nonnull");
                source.Append("\";\n");
            }

            foreach (var property in godotProperties.Where(prop => prop.PropertyType == PropertyType.Property))
            {
                source.Append("        /// <summary>\n")
                    .Append("        /// Cached name for the '")
                    .Append(property.MemberName)
                    .Append("' property.\n")
                    .Append("        /// </summary>\n");

                source.Append("        public new static readonly global::Godot.StringName @");
                source.Append(property.PropertyName);
                source.Append(" = \"");
                source.Append(property.PropertyNameHint);
                source.Append("\";\n");
            }

            foreach (var field in godotProperties.Where(prop => prop.PropertyType == PropertyType.Field))
            {
                source.Append("        /// <summary>\n")
                    .Append("        /// Cached name for the '")
                    .Append(field.MemberName)
                    .Append("' field.\n")
                    .Append("        /// </summary>\n");

                source.Append("        public new static readonly global::Godot.StringName @");
                source.Append(field.PropertyName);
                source.Append(" = \"");
                source.Append(field.PropertyNameHint);
                source.Append("\";\n");
            }

            source.Append("    }\n"); // class GodotInternal

            if (godotProperties.Length > 0)
            {

                // Generate SetGodotClassPropertyValue

                bool allPropertiesAreReadOnly = godotProperties.All(pi => pi.IsReadOnly || (pi.PropertyType == PropertyType.Embed && !pi.PropertyTypeSymbol.IsReferenceType));

                if (!allPropertiesAreReadOnly)
                {
                    source.Append("    /// <inheritdoc/>\n");
                    source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                    source.Append("    protected override bool SetGodotClassPropertyValue(in godot_string_name name, ");
                    source.Append("in godot_variant value)\n    {\n");

                    foreach (var embed in godotProperties.Where(prop => prop.PropertyType == PropertyType.Embed))
                    {
                        if (!embed.PropertyTypeSymbol.IsReferenceType)
                            continue;

                        source
                            .Append("        if (name == PropertyName.@")
                            .Append(embed.PropertyName)
                            .Append("_nonnull) {\n");
                        if (embed.InNullable)
                            source.Append("            if (").Append("this.@").Append(embed.ContainingProperty.MemberNameNullable).Append(" is not null) {\n    ");
                        source.Append("            this.@")
                            .Append(embed.MemberName)
                            .Append(" = ")
                            .AppendNativeVariantToManagedExpr("value", context.Compilation.GetSpecialType(SpecialType.System_Boolean), MarshalType.Boolean)
                            .Append(" ? ")
                            .Append("this.@")
                            .Append(embed.MemberName)
                            .Append(" ?? new() : null;\n");
                        if (embed.InNullable)
                            source.Append("            }\n");
                        source.Append("            return true;\n")
                            .Append("        }\n");
                    }

                    foreach (var property in godotProperties.Where(prop => prop.PropertyType == PropertyType.Property))
                    {
                        if (property.IsReadOnly)
                            continue;

                        GeneratePropertySetter(property, source);
                    }

                    foreach (var field in godotProperties.Where(prop => prop.PropertyType == PropertyType.Field))
                    {
                        if (field.IsReadOnly)
                            continue;

                        GeneratePropertySetter(field, source);
                    }

                    source.Append("        return base.SetGodotClassPropertyValue(name, value);\n");

                    source.Append("    }\n");
                }

                // Generate GetGodotClassPropertyValue
                bool allPropertiesAreWriteOnly = godotProperties.All(pi => pi.IsWriteOnly || (pi.PropertyType == PropertyType.Embed && !pi.PropertyTypeSymbol.IsReferenceType));

                if (!allPropertiesAreWriteOnly)
                {
                    source.Append("    /// <inheritdoc/>\n");
                    source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                    source.Append("    protected override bool GetGodotClassPropertyValue(in godot_string_name name, ");
                    source.Append("out godot_variant value)\n    {\n");

                    foreach (var embed in godotProperties.Where(prop => prop.PropertyType == PropertyType.Embed))
                    {
                        if (!embed.PropertyTypeSymbol.IsReferenceType)
                            continue;

                        source.Append("        if (name == PropertyName.@").Append(embed.PropertyName).Append("_nonnull) {\n");
                        source.Append("            value = ").AppendManagedToNativeVariantExpr($"this.@{embed.MemberNameNullable} is not null",
                            context.Compilation.GetSpecialType(SpecialType.System_Boolean), MarshalType.Boolean).Append(";\n");
                        source.Append("            return true;\n");
                        source.Append("        }\n");
                    }

                    foreach (var property in godotProperties.Where(prop => prop.PropertyType == PropertyType.Property))
                    {
                        if (property.IsWriteOnly)
                            continue;

                        GeneratePropertyGetter(property, source);
                    }

                    foreach (var field in godotProperties.Where(prop => prop.PropertyType == PropertyType.Field))
                    {
                        if (field.IsWriteOnly)
                            continue;

                        GeneratePropertyGetter(field, source);
                    }

                    source.Append("        return base.GetGodotClassPropertyValue(name, out value);\n");

                    source.Append("    }\n");
                }
                // Generate GetGodotPropertyList

                const string DictionaryType = "global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>";

                source.Append("    /// <summary>\n")
                    .Append("    /// Get the property information for all the properties declared in this class.\n")
                    .Append("    /// This method is used by Godot to register the available properties in the editor.\n")
                    .Append("    /// Do not call this method.\n")
                    .Append("    /// </summary>\n");

                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");

                source.Append("    internal new static ")
                    .Append(DictionaryType)
                    .Append(" GetGodotPropertyList()\n    {\n");

                source.Append("        var properties = new ")
                    .Append(DictionaryType)
                    .Append("();\n");


                foreach (var member in godotProperties)
                {
                    if (member.PropertyType == PropertyType.Embed)
                    {
                        string memberGroupName = member.PropertyGroupName;
                        AppendGroupingPropertyInfo(source, new PropertyInfo(
                            VariantType.Nil,
                            member.ContainingProperty is null ? memberGroupName : memberGroupName.Substring(memberGroupName.IndexOf("/") + 1),
                            PropertyHint.None,
                            $"{memberGroupName}/",
                            member.ContainingProperty is null ? PropertyUsageFlags.Group : PropertyUsageFlags.Subgroup,
                            true
                        ));
                        if (member.PropertyTypeSymbol.IsReferenceType)
                        {
                            AppendPropertyInfo(source, new PropertyInfo(VariantType.Bool, $"{member.PropertyName}_nonnull", PropertyHint.GroupEnable, "", PropertyUsageFlags.Default, true));
                        }
                        continue;
                    }

                    foreach (var groupingInfo in DetermineGroupingPropertyInfo(member))
                        AppendGroupingPropertyInfo(source, groupingInfo);

                    var propertyInfo = DeterminePropertyInfo(context, typeCache, member);

                    if (propertyInfo == null)
                        continue;

                    if (propertyInfo.Value.Hint == PropertyHint.ToolButton && !isToolClass)
                    {
                        context.ReportDiagnostic(Diagnostic.Create(
                            Common.OnlyToolClassesShouldUseExportToolButtonRule,
                            member.Symbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                            member.Symbol.ToDisplayString()
                        ));
                        continue;
                    }

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

        private static void GeneratePropertySetter(GodotPropertyData property, StringBuilder source)
        {
            source.Append("        ");

            source.Append("if (name == PropertyName.@")
                .Append(property.PropertyName)
                .Append(") {\n");
            if (property.InNullable)
                source.Append("            if (this.@")
                    .Append(property.ContainingProperty.MemberNameNullable)
                    .Append(" is not null) {\n    ");
            source.Append("            this.@")
                .Append(property.MemberName)
                .Append(" = ")
                .AppendNativeVariantToManagedExpr("value", property.PropertyTypeSymbol, property.MarshalType)
                .Append(";\n");
            if (property.InNullable)
                source.Append("            }\n");
            source.Append("            return true;\n")
                .Append("        }\n");
        }

        private static void GeneratePropertyGetter(GodotPropertyData property, StringBuilder source)
        {
            source.Append("        ");

            source.Append("if (name == PropertyName.@")
                .Append(property.PropertyName)
                .Append(") {\n")
                .Append("            value = ")
                .AppendManagedToNativeVariantExpr(
                    string.Concat("this.@", property.MemberNameNullable, property.InNullable ? " ?? default" : ""),
                    property.PropertyTypeSymbol, property.MarshalType)
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
                .Append(", name: PropertyName.@")
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

        private static IEnumerable<PropertyInfo> DetermineGroupingPropertyInfo(GodotPropertyData property)
        {
            foreach (var attr in property.Symbol.GetAttributes())
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

                    if (propertyUsage != PropertyUsageFlags.Category && property.ContainingProperty is not null)
                    {
                        string groupName = property.ContainingProperty.PropertyGroupName;
                        hintString = $"{groupName}/{hintString}";
                        int slashIndex = groupName.IndexOf("/");
                        if (slashIndex != -1)
                            groupName = groupName.Substring(slashIndex + 1);
                        else
                            groupName = "";
                        name = string.Join("/", new[] { groupName, name }.Where(s => !string.IsNullOrWhiteSpace(s)));
                        propertyUsage = PropertyUsageFlags.Subgroup;
                    }

                    yield return new PropertyInfo(VariantType.Nil, name, PropertyHint.None, hintString,
                        propertyUsage.Value, true);
                }
            }
        }

        private static PropertyInfo? DeterminePropertyInfo(
            GeneratorExecutionContext context,
            MarshalUtils.TypeCache typeCache,
            GodotPropertyData property
        )
        {
            var exportAttr = property.Symbol.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.IsGodotExportAttribute() ?? false);

            var exportToolButtonAttr = property.Symbol.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.IsGodotExportToolButtonAttribute() ?? false);

            if (exportAttr != null && exportToolButtonAttr != null)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    Common.ExportToolButtonShouldNotBeUsedWithExportRule,
                    property.Symbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                    property.Symbol.ToDisplayString()
                ));
                return null;
            }

            var propertySymbol = property.Symbol as IPropertySymbol;
            var fieldSymbol = property.Symbol as IFieldSymbol;

            if (exportAttr != null && propertySymbol != null)
            {
                if (propertySymbol.GetMethodOrBaseGetMethod() is null || propertySymbol.SetMethodOrBaseSetMethod() is not { IsInitOnly: false })
                {
                    // Exports can be neither read-only nor write-only but the diagnostic errors for properties are already
                    // reported by ScriptPropertyDefValGenerator.cs so just quit early here.
                    return null;
                }
            }

            if (exportToolButtonAttr != null && propertySymbol != null && propertySymbol.GetMethod == null)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    Common.ExportedPropertyIsWriteOnlyRule,
                    propertySymbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                    propertySymbol.ToDisplayString()
                ));
                return null;
            }

            if (exportToolButtonAttr != null && propertySymbol != null)
            {
                if (!PropertyIsExpressionBodiedAndReturnsNewCallable(context.Compilation, propertySymbol))
                {
                    context.ReportDiagnostic(Diagnostic.Create(
                        Common.ExportToolButtonMustBeExpressionBodiedProperty,
                        propertySymbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                        propertySymbol.ToDisplayString()
                    ));
                    return null;
                }

                static bool PropertyIsExpressionBodiedAndReturnsNewCallable(Compilation compilation, IPropertySymbol? propertySymbol)
                {
                    if (propertySymbol == null)
                    {
                        return false;
                    }

                    var propertyDeclarationSyntax = propertySymbol.DeclaringSyntaxReferences
                        .Select(r => r.GetSyntax() as PropertyDeclarationSyntax).FirstOrDefault();
                    if (propertyDeclarationSyntax == null || propertyDeclarationSyntax.Initializer != null)
                    {
                        return false;
                    }

                    if (propertyDeclarationSyntax.AccessorList != null)
                    {
                        var accessors = propertyDeclarationSyntax.AccessorList.Accessors;
                        foreach (var accessor in accessors)
                        {
                            if (!accessor.IsKind(SyntaxKind.GetAccessorDeclaration))
                            {
                                // Only getters are allowed.
                                return false;
                            }

                            if (!ExpressionBodyReturnsNewCallable(compilation, accessor.ExpressionBody))
                            {
                                return false;
                            }
                        }
                    }
                    else if (!ExpressionBodyReturnsNewCallable(compilation, propertyDeclarationSyntax.ExpressionBody))
                    {
                        return false;
                    }

                    return true;
                }

                static bool ExpressionBodyReturnsNewCallable(Compilation compilation, ArrowExpressionClauseSyntax? expressionSyntax)
                {
                    if (expressionSyntax == null)
                    {
                        return false;
                    }

                    var semanticModel = compilation.GetSemanticModel(expressionSyntax.SyntaxTree);

                    switch (expressionSyntax.Expression)
                    {
                        case ImplicitObjectCreationExpressionSyntax creationExpression:
                            // We already validate that the property type must be 'Callable'
                            // so we can assume this constructor is valid.
                            return true;

                        case ObjectCreationExpressionSyntax creationExpression:
                            var typeSymbol = semanticModel.GetSymbolInfo(creationExpression.Type).Symbol as ITypeSymbol;
                            if (typeSymbol != null)
                            {
                                return typeSymbol.FullQualifiedNameOmitGlobal() == GodotClasses.Callable;
                            }
                            break;

                        case InvocationExpressionSyntax invocationExpression:
                            var methodSymbol = semanticModel.GetSymbolInfo(invocationExpression).Symbol as IMethodSymbol;
                            if (methodSymbol != null && methodSymbol.Name == "From")
                            {
                                return methodSymbol.ContainingType.FullQualifiedNameOmitGlobal() == GodotClasses.Callable;
                            }
                            break;
                    }

                    return false;
                }
            }

            var memberType = propertySymbol?.Type ?? fieldSymbol!.Type;

            var memberVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(property.MarshalType)!.Value;
            string memberName = property.PropertyName;

            string? hintString = null;

            if (exportToolButtonAttr != null)
            {
                if (memberVariantType != VariantType.Callable)
                {
                    context.ReportDiagnostic(Diagnostic.Create(
                        Common.ExportToolButtonIsNotCallableRule,
                        property.Symbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                        property.Symbol.ToDisplayString()
                    ));
                    return null;
                }

                hintString = exportToolButtonAttr.ConstructorArguments[0].Value?.ToString() ?? "";
                foreach (var namedArgument in exportToolButtonAttr.NamedArguments)
                {
                    if (namedArgument is { Key: "Icon", Value.Value: string { Length: > 0 } })
                    {
                        hintString += $",{namedArgument.Value.Value}";
                    }
                }

                return new PropertyInfo(memberVariantType, memberName, PropertyHint.ToolButton,
                    hintString: hintString, PropertyUsageFlags.Editor, exported: true);
            }

            if (exportAttr == null)
            {
                return new PropertyInfo(memberVariantType, memberName, PropertyHint.None,
                    hintString: hintString, PropertyUsageFlags.ScriptVariable, exported: false);
            }

            if (!TryGetMemberExportHint(typeCache, memberType, exportAttr, memberVariantType,
                    isTypeArgument: false, out var hint, out hintString))
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
                if (TryGetNodeOrResourceType(exportAttr, out hint, out hintString))
                {
                    return true;
                }

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

            static bool TryGetNodeOrResourceType(AttributeData exportAttr, out PropertyHint hint, out string? hintString)
            {
                hint = PropertyHint.None;
                hintString = null;

                if (exportAttr.ConstructorArguments.Length <= 1) return false;

                var hintValue = exportAttr.ConstructorArguments[0].Value;

                var hintEnum = hintValue switch
                {
                    null => PropertyHint.None,
                    int intValue => (PropertyHint)intValue,
                    _ => (PropertyHint)(long)hintValue
                };

                if (!hintEnum.HasFlag(PropertyHint.NodeType) && !hintEnum.HasFlag(PropertyHint.ResourceType))
                    return false;

                var hintStringValue = exportAttr.ConstructorArguments[1].Value?.ToString();
                if (string.IsNullOrWhiteSpace(hintStringValue))
                {
                    return false;
                }

                hint = hintEnum;
                hintString = hintStringValue;

                return true;
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
                    return false; // Non-generic Array, so there's no hint to add.

                if (elementType.TypeKind == TypeKind.TypeParameter)
                    return false; // The generic is not constructed, we can't really hint anything.

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
                var elementTypes = MarshalUtils.GetGenericElementTypes(type);

                if (elementTypes == null)
                    return false; // Non-generic Dictionary, so there's no hint to add
                Debug.Assert(elementTypes.Length == 2);

                var keyElementMarshalType = MarshalUtils.ConvertManagedTypeToMarshalType(elementTypes[0], typeCache);
                var valueElementMarshalType = MarshalUtils.ConvertManagedTypeToMarshalType(elementTypes[1], typeCache);

                if (keyElementMarshalType == null || valueElementMarshalType == null)
                {
                    // To maintain compatibility with previous versions of Godot before 4.4,
                    // we must preserve the old behavior for generic dictionaries with non-marshallable
                    // generic type arguments.
                    return false;
                }

                var keyElementVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(keyElementMarshalType.Value)!.Value;
                var keyIsPresetHint = false;
                var keyHintString = (string?)null;

                if (keyElementVariantType == VariantType.String || keyElementVariantType == VariantType.StringName)
                    keyIsPresetHint = GetStringArrayEnumHint(keyElementVariantType, exportAttr, out keyHintString);

                if (!keyIsPresetHint)
                {
                    bool hintRes = TryGetMemberExportHint(typeCache, elementTypes[0],
                        exportAttr, keyElementVariantType, isTypeArgument: true,
                        out var keyElementHint, out var keyElementHintString);

                    // Format: type/hint:hint_string
                    if (hintRes)
                    {
                        keyHintString = (int)keyElementVariantType + "/" + (int)keyElementHint + ":";

                        if (keyElementHintString != null)
                            keyHintString += keyElementHintString;
                    }
                    else
                    {
                        keyHintString = (int)keyElementVariantType + "/" + (int)PropertyHint.None + ":";
                    }
                }

                var valueElementVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(valueElementMarshalType.Value)!.Value;
                var valueIsPresetHint = false;
                var valueHintString = (string?)null;

                if (valueElementVariantType == VariantType.String || valueElementVariantType == VariantType.StringName)
                    valueIsPresetHint = GetStringArrayEnumHint(valueElementVariantType, exportAttr, out valueHintString);

                if (!valueIsPresetHint)
                {
                    bool hintRes = TryGetMemberExportHint(typeCache, elementTypes[1],
                        exportAttr, valueElementVariantType, isTypeArgument: true,
                        out var valueElementHint, out var valueElementHintString);

                    // Format: type/hint:hint_string
                    if (hintRes)
                    {
                        valueHintString = (int)valueElementVariantType + "/" + (int)valueElementHint + ":";

                        if (valueElementHintString != null)
                            valueHintString += valueElementHintString;
                    }
                    else
                    {
                        valueHintString = (int)valueElementVariantType + "/" + (int)PropertyHint.None + ":";
                    }
                }

                hint = PropertyHint.TypeString;

                hintString = keyHintString != null && valueHintString != null ? $"{keyHintString};{valueHintString}" : null;
                return hintString != null;
            }

            return false;
        }
    }
}
