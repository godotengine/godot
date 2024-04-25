//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_UNIT_H_
#define MATERIALX_UNIT_H_

/// @file
/// Unit classes

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

class UnitConverter;
class LinearUnitConverter;
class UnitConverterRegistry;

/// A shared pointer to a UnitConverter
using UnitConverterPtr = shared_ptr<UnitConverter>;
/// A shared pointer to a const UnitConverter
using ConstUnitConverterPtr = shared_ptr<const UnitConverter>;

/// A shared pointer to a LinearUnitConverter
using LinearUnitConverterPtr = shared_ptr<LinearUnitConverter>;
/// A shared pointer to a const LinearUnitConverter
using ConstLinearUnitConverterPtr = shared_ptr<const LinearUnitConverter>;

/// A shared pointer to a UnitConverterRegistry
using UnitConverterRegistryPtr = shared_ptr<UnitConverterRegistry>;
/// A shared pointer to a const UnitConverterRegistry
using ConstUnitConverterRegistryPtr = shared_ptr<const UnitConverterRegistry>;

/// @class UnitConverter
/// An abstract base class for unit converters.
/// Each unit converter instance is responsible for a single unit type.
class MX_CORE_API UnitConverter
{
  public:
    UnitConverter() { }
    virtual ~UnitConverter() { }

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    virtual float convert(float input, const string& inputUnit, const string& outputUnit) const = 0;

    /// Given a unit name return a value that it can map to as an integer
    /// Returns -1 value if not found
    virtual int getUnitAsInteger(const string&) const { return -1; }

    /// Given an integer index return the unit name in the map used by the converter
    /// Returns Empty string if not found
    virtual string getUnitFromInteger(int) const { return EMPTY_STRING; }

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    virtual Vector2 convert(const Vector2& input, const string& inputUnit, const string& outputUnit) const = 0;

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    virtual Vector3 convert(const Vector3& input, const string& inputUnit, const string& outputUnit) const = 0;

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    virtual Vector4 convert(const Vector4& input, const string& inputUnit, const string& outputUnit) const = 0;

    /// Create unit definitions in a document based on the converter
    virtual void write(DocumentPtr doc) const = 0;
};

/// @class LinearUnitConverter
/// A converter class for linear units that require only a scalar multiplication.
class MX_CORE_API LinearUnitConverter : public UnitConverter
{
  public:
    virtual ~LinearUnitConverter() { }

    /// Creator
    static LinearUnitConverterPtr create(UnitTypeDefPtr UnitDef);

    /// Return the unit type string
    const string& getUnitType() const
    {
        return _unitType;
    }

    /// Create unit definitions in a document based on the converter
    void write(DocumentPtr doc) const override;

    /// @name Conversion
    /// @{

    /// Return the mappings from unit names to the scale value
    /// defined by a linear converter.
    const std::unordered_map<string, float>& getUnitScale() const
    {
        return _unitScale;
    }

    /// Ratio between the given unit to a desired unit
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    float conversionRatio(const string& inputUnit, const string& outputUnit) const;

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    float convert(float input, const string& inputUnit, const string& outputUnit) const override;

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    Vector2 convert(const Vector2& input, const string& inputUnit, const string& outputUnit) const override;

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    Vector3 convert(const Vector3& input, const string& inputUnit, const string& outputUnit) const override;

    /// Convert a given value in a given unit to a desired unit
    /// @param input Input value to convert
    /// @param inputUnit Unit of input value
    /// @param outputUnit Unit for output value
    Vector4 convert(const Vector4& input, const string& inputUnit, const string& outputUnit) const override;

    /// @}
    /// @name Shader Mapping
    /// @{

    /// Given a unit name return a value that it can map to as an integer.
    /// Returns -1 value if not found
    int getUnitAsInteger(const string& unitName) const override;

    /// Given an integer index return the unit name in the map used by the converter.
    /// Returns Empty string if not found
    virtual string getUnitFromInteger(int index) const override;

    /// @}

  private:
    LinearUnitConverter(UnitTypeDefPtr UnitDef);

  private:
    std::unordered_map<string, float> _unitScale;
    std::unordered_map<string, int> _unitEnumeration;
    string _unitType;
};

/// Map of unit converters
using UnitConverterPtrMap = std::unordered_map<string, UnitConverterPtr>;

/// @class UnitConverterRegistry
/// A registry for unit converters.
class MX_CORE_API UnitConverterRegistry
{
  public:
    virtual ~UnitConverterRegistry() { }

    /// Creator
    static UnitConverterRegistryPtr create();

    /// Add a unit converter for a given UnitDef.
    /// Returns false if a converter has already been registered for the given UnitDef
    bool addUnitConverter(UnitTypeDefPtr def, UnitConverterPtr converter);

    /// Remove a unit converter for a given UnitDef.
    /// Returns false if a converter does not exist for the given UnitDef
    bool removeUnitConverter(UnitTypeDefPtr def);

    /// Get a unit converter for a given UnitDef
    /// Returns any empty pointer if a converter does not exist for the given UnitDef
    UnitConverterPtr getUnitConverter(UnitTypeDefPtr def);

    /// Clear all unit converters from the registry.
    void clearUnitConverters();

    /// Given a unit name return a value that it can map to as an integer
    /// Returns -1 value if not found
    int getUnitAsInteger(const string& unitName) const;

    /// Create unit definitions in a document based on registered converters
    void write(DocumentPtr doc) const;

    /// Convert input values which have a source unit to a given target unit.
    /// Returns if any unit conversion occured.
    bool convertToUnit(DocumentPtr doc, const string& unitType, const string& targetUnit);

  private:
    UnitConverterRegistry(const UnitConverterRegistry&) = delete;
    UnitConverterRegistry() { }

    UnitConverterRegistry& operator=(const UnitConverterRegistry&) = delete;

  private:
    UnitConverterPtrMap _unitConverters;
};

MATERIALX_NAMESPACE_END

#endif
