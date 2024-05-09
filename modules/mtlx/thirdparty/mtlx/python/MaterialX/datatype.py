#!/usr/bin/env python
'''
Native Python helper functions for MaterialX data types.
'''

import sys

from .PyMaterialXCore import *


#--------------------------------------------------------------------------------
_typeToName = { int         : 'integer',
                float       : 'float',
                bool        : 'boolean',
                Color3      : 'color3',
                Color4      : 'color4',
                Vector2     : 'vector2',
                Vector3     : 'vector3',
                Vector4     : 'vector4',
                Matrix33    : 'matrix33',
                Matrix44    : 'matrix44',
                str         : 'string' }

if sys.version_info[0] < 3:
    _typeToName[long] = 'integer'
    _typeToName[unicode] = 'string'
else:
    _typeToName[bytes] = 'string'


#--------------------------------------------------------------------------------
def getTypeString(value):
    """Return the MaterialX type string associated with the given Python value
       If the type of the given Python value is not recognized by MaterialX,
       then None is returned.

       Examples:
           getTypeString(1.0) -> 'float'
           getTypeString(mx.Color3(1)) -> 'color3'"""

    valueType = type(value)
    if valueType in _typeToName:
        return _typeToName[valueType]
    if valueType in (tuple, list):
        if len(value):
            elemType = type(value[0])
            if elemType in _typeToName:
                return _typeToName[elemType] + 'array'
        return 'stringarray'
    return None

def getValueString(value):
    """Return the MaterialX value string associated with the given Python value
       If the type of the given Python value is not recognized by MaterialX,
       then None is returned

       Examples:
           getValueString(0.1) -> '0.1'
           getValueString(mx.Color3(0.1, 0.2, 0.3)) -> '0.1, 0.2, 0.3'"""

    typeString = getTypeString(value)
    if not typeString:
        return None
    method = globals()['TypedValue_' + typeString].createValue
    return method(value).getValueString()

def createValueFromStrings(valueString, typeString):
    """Convert a MaterialX value and type strings to the corresponding
       Python value.  If the given conversion cannot be performed, then None
       is returned.

       Examples:
           createValueFromStrings('0.1', 'float') -> 0.1
           createValueFromStrings('0.1, 0.2, 0.3', 'color3') -> mx.Color3(0.1, 0.2, 0.3)"""

    valueObj = Value.createValueFromStrings(valueString, typeString)
    if not valueObj:
        return None
    return valueObj.getData()


def isColorType(t):
    "Return True if the given type is a MaterialX color."
    return t in (Color3, Color4)

def isColorValue(value):
    "Return True if the given value is a MaterialX color."
    return isColorType(type(value))

def stringToBoolean(value):
    "Return boolean value found in a string. Throws and exception if a boolean value could not be parsed"
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    raise TypeError('Boolean value expected.')
