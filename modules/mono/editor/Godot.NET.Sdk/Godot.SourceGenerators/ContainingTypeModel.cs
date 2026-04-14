using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators;

internal record struct ContainingTypeModel(
    string DeclarationKeyword,
    string DisplayStringMinimallyQualifiedFormat
)
{
    public static ContainingTypeModel[]? GetContainingTypesFor(INamedTypeSymbol symbol)
    {
        if (symbol.ContainingType == null)
            return null;

        int containingTypeCount = 0;
        INamedTypeSymbol? containingType = symbol.ContainingType;

        while (containingType != null)
        {
            containingTypeCount++;
            containingType = containingType.ContainingType;
        }

        var containingTypeModels = new ContainingTypeModel[containingTypeCount];

        containingType = symbol.ContainingType;

        while (containingType != null)
        {
            containingTypeModels[--containingTypeCount] = new ContainingTypeModel(
                containingType.GetDeclarationKeyword(),
                containingType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)
            );
            containingType = containingType.ContainingType;
        }

        return containingTypeModels;
    }
}
