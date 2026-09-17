using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class GenericScriptTypeMetaProviderAnalyzer : DiagnosticAnalyzer
{
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(
            Common.GenericScriptTypeMetaProviderRuleGenericTarget,
            Common.GenericScriptTypeMetaProviderRuleTypeNotFound,
            Common.GenericScriptTypeMetaProviderRuleTypeNotNestedWithin,
            Common.GenericScriptTypeMetaProviderRuleMissingAssemblyName,
            Common.GenericScriptTypeMetaProviderRuleMissingInterface,
            Common.GenericScriptTypeMetaProviderRuleNoExtraGenerics
        );

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(AnalyzeNamedType, SymbolKind.NamedType);
    }

    private static void AnalyzeNamedType(SymbolAnalysisContext context)
    {
        var namedType = (INamedTypeSymbol)context.Symbol;

        var attr = namedType.GetAttributes().FirstOrDefault(a =>
            a.AttributeClass is
            {
                Name: "GenericScriptTypeMetaProviderAttribute",
                ContainingNamespace: { Name: "Godot", ContainingNamespace.IsGlobalNamespace: true }
            });

        if (attr == null)
            return;

        // The attribute must be applied to a generic type.
        if (!namedType.IsGenericType)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.GenericScriptTypeMetaProviderRuleGenericTarget,
                attr.ApplicationSyntaxReference?.GetSyntax().GetLocation(),
                namedType.Name));
            return;
        }

        // Extract the string argument from the attribute constructor.
        if (attr.ConstructorArguments.Length == 0 || attr.ConstructorArguments[0].Value
                is not string providerTypeAssemblyQualifiedName)
        {
            return;
        }

        // Find the provider type.
        // The string must be an assembly-qualified name in the same format as 'Type.GetType(string)',
        // e.g.: "Namespace.GenericScriptType`1.GodotInternal.MetaProvider, AssemblyName".
        if (FindProviderTypeFromQualifiedName(
                context.Compilation, potentialOuterType: namedType,
                providerTypeAssemblyQualifiedName, out bool usesExplicitAssemblyName
            ) is not { } resolvedProviderType)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.GenericScriptTypeMetaProviderRuleTypeNotFound,
                attr.ApplicationSyntaxReference?.GetSyntax().GetLocation(),
                providerTypeAssemblyQualifiedName));
            return;
        }

        // The provider type must be nested within the script type that the attribute is applied to.
        if (!IsNestedWithin(resolvedProviderType, potentialOuterType: namedType))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.GenericScriptTypeMetaProviderRuleTypeNotNestedWithin,
                attr.ApplicationSyntaxReference?.GetSyntax().GetLocation(),
                providerTypeAssemblyQualifiedName, namedType.Name));
            return;
        }

        // The assembly-qualified name must include the assembly name suffix (", AssemblyName").
        if (!usesExplicitAssemblyName)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.GenericScriptTypeMetaProviderRuleMissingAssemblyName,
                attr.ApplicationSyntaxReference?.GetSyntax().GetLocation(),
                providerTypeAssemblyQualifiedName, resolvedProviderType.ContainingAssembly.Identity.Name));
            return;
        }

        // The provider type must implement IScriptTypeMetaProvider.
        var interfaceSymbol = context.Compilation.GetTypeByMetadataName("Godot.IScriptTypeMetaProvider");
        if (interfaceSymbol != null && !resolvedProviderType.AllInterfaces.Contains(interfaceSymbol))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.GenericScriptTypeMetaProviderRuleMissingInterface,
                attr.ApplicationSyntaxReference?.GetSyntax().GetLocation(),
                resolvedProviderType.Name));
            return;
        }

        // There must be no additional generic parameters beyond the ones from the script type.
        var current = resolvedProviderType;
        while (current != null && !SymbolEqualityComparer.Default.Equals(current, namedType))
        {
            if (current.Arity > 0)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    Common.GenericScriptTypeMetaProviderRuleNoExtraGenerics,
                    attr.ApplicationSyntaxReference?.GetSyntax().GetLocation(),
                    current.Name));
                return;
            }

            current = current.ContainingType;
        }
    }

    private static bool IsNestedWithin(ITypeSymbol potentialInnerType, ITypeSymbol potentialOuterType)
    {
        var current = potentialInnerType.ContainingType;

        while (current != null)
        {
            if (SymbolEqualityComparer.Default.Equals(current, potentialOuterType))
                return true;

            current = current.ContainingType;
        }

        return false;
    }

    private static INamedTypeSymbol? FindProviderTypeFromQualifiedName(
        Compilation compilation,
        ITypeSymbol potentialOuterType,
        string qualifiedName,
        out bool usesExplicitAssemblyName)
    {
        // GetTypeByMetadataName doesn't work with the ", AssemblyName" suffix, so that case needs special handling.

        var potentialOuterTypeAssembly = potentialOuterType.ContainingAssembly;

        // Split the type and assembly ("Namespace.Class, AssemblyName" -> ["Namespace.Class", "AssemblyName"]).
        var parts = qualifiedName.Split(',');
        var typeMetadataName = parts[0].Trim();

        if (parts.Length == 1)
        {
            usesExplicitAssemblyName = false;
            return compilation.GetTypeByMetadataName(typeMetadataName);
        }

        usesExplicitAssemblyName = true;

        var assemblyName = parts[1].Trim();

        // Search within compilation's assembly first if it has the same name.
        if (potentialOuterTypeAssembly.Identity.Name.Equals(assemblyName, StringComparison.OrdinalIgnoreCase)
            && potentialOuterTypeAssembly.GetTypeByMetadataName(typeMetadataName) is { } found)
        {
            return found;
        }

        // Locate the specified assembly's symbol.
        var assemblySymbol = compilation.References
            .Select(compilation.GetAssemblyOrModuleSymbol)
            .OfType<IAssemblySymbol>()
            .FirstOrDefault(a => a.Identity.Name.Equals(assemblyName, StringComparison.OrdinalIgnoreCase));

        // Get the type from that specific assembly.
        return assemblySymbol?.GetTypeByMetadataName(typeMetadataName);
    }
}
