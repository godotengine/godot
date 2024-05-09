#!/usr/bin/env python
'''
Native Python wrappers for PyMaterialX, providing a more Pythonic interface
for Elements and Values.
'''

import warnings

from .PyMaterialXCore import *
from .PyMaterialXFormat import *
from .datatype import *
import os

#
# Element
#

def _isA(self, elementClass, category = ''):
    """Return True if this element is an instance of the given subclass.
       If a category string is specified, then both subclass and category
       matches are required."""
    if not isinstance(self, elementClass):
        return False
    if category and self.getCategory() != category:
        return False
    return True

def _addChild(self, elementClass, name, typeString = ''):
    "Add a child element of the given subclass, name, and optional type string."
    method = getattr(self.__class__, "_addChild" + elementClass.__name__)
    return method(self, name, typeString)

def _getChild(self, name):
    "Return the child element, if any, with the given name."
    if (name == None):
        return None
    return self._getChild(name)

def _getChildOfType(self, elementClass, name):
    "Return the child element, if any, with the given name and subclass."
    method = getattr(self.__class__, "_getChildOfType" + elementClass.__name__)
    return method(self, name)

def _getChildrenOfType(self, elementClass):
    """Return a list of all child elements that are instances of the given type.
       The returned list maintains the order in which children were added."""
    method = getattr(self.__class__, "_getChildrenOfType" + elementClass.__name__)
    return method(self)

def _removeChildOfType(self, elementClass, name):
    "Remove the typed child element, if any, with the given name."
    method = getattr(self.__class__, "_removeChildOfType" + elementClass.__name__)
    method(self, name)

Element.isA = _isA
Element.addChild = _addChild
Element.getChild = _getChild
Element.getChildOfType = _getChildOfType
Element.getChildrenOfType = _getChildrenOfType
Element.removeChildOfType = _removeChildOfType


#
# ValueElement
#

def _setValue(self, value, typeString = ''):
    "Set the typed value of an element."
    method = getattr(self.__class__, "_setValue" + getTypeString(value))
    method(self, value, typeString)

def _getValue(self):
    "Return the typed value of an element."
    value = self._getValue()
    return value.getData() if value else None

def _getDefaultValue(self):
    """Return the default value for this element."""
    value = self._getDefaultValue()
    return value.getData() if value else None

ValueElement.setValue = _setValue
ValueElement.getValue = _getValue
ValueElement.getDefaultValue = _getDefaultValue


#
# InterfaceElement
#

def _setInputValue(self, name, value, typeString = ''):
    """Set the typed value of an input by its name, creating a child element
       to hold the input if needed."""
    method = getattr(self.__class__, "_setInputValue" + getTypeString(value))
    return method(self, name, value, typeString)

def _getInputValue(self, name, target = ''):
    """Return the typed value of an input by its name, taking both the
       calling element and its declaration into account.  If the given
       input is not found, then None is returned."""
    value = self._getInputValue(name, target)
    return value.getData() if value else None

def _addParameter(self, name):
    """(Deprecated) Add a Parameter to this interface."""
    warnings.warn("This function is deprecated; parameters have been replaced with uniform inputs in 1.38.", DeprecationWarning, stacklevel = 2)
    return self.addInput(name)

def _getParameters(self):
    """(Deprecated) Return a vector of all Parameter elements."""
    warnings.warn("This function is deprecated; parameters have been replaced with uniform inputs in 1.38.", DeprecationWarning, stacklevel = 2)
    return list()

def _getActiveParameters(self):
    """(Deprecated) Return a vector of all parameters belonging to this interface, taking inheritance into account."""
    warnings.warn("This function is deprecated; parameters have been replaced with uniform inputs in 1.38.", DeprecationWarning, stacklevel = 2)
    return list()

def _setParameterValue(self, name, value, typeString = ''):
    """(Deprecated) Set the typed value of a parameter by its name."""
    warnings.warn("This function is deprecated; parameters have been replaced with uniform inputs in 1.38.", DeprecationWarning, stacklevel = 2)

def _getParameterValue(self, name, target = ''):
    """(Deprecated) Return the typed value of a parameter by its name."""
    warnings.warn("This function is deprecated; parameters have been replaced with uniform inputs in 1.38.", DeprecationWarning, stacklevel = 2)
    return None

def _getParameterValueString(self, name):
    """(Deprecated) Return the value string of a parameter by its name."""
    warnings.warn("This function is deprecated; parameters have been replaced with uniform inputs in 1.38.", DeprecationWarning, stacklevel = 2)
    return ""

def _addBindInput(self, name, type = DEFAULT_TYPE_STRING):
    """(Deprecated) Add a BindInput to this shader reference."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return self.addInput(name, type)

def _getBindInputs(self):
    """(Deprecated) Return a vector of all BindInput elements in this shader reference."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return self.getInputs()

def _addBindParam(self, name, type = DEFAULT_TYPE_STRING):
    """(Deprecated) Add a BindParam to this shader reference."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return self.addInput(name, type)

def _getBindParams(self):
    """(Deprecated) Return a vector of all BindParam elements in this shader reference."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return list()

def _getBindTokens(self):
    """(Deprecated) Return a vector of all BindToken elements in this shader reference."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return list()

InterfaceElement.setInputValue = _setInputValue
InterfaceElement.getInputValue = _getInputValue
InterfaceElement.addParameter = _addParameter
InterfaceElement.getParameters = _getParameters
InterfaceElement.getActiveParameters = _getActiveParameters
InterfaceElement.setParameterValue = _setParameterValue
InterfaceElement.getParameterValue = _getParameterValue
InterfaceElement.getParameterValueString = _getParameterValueString
InterfaceElement.addBindInput = _addBindInput
InterfaceElement.getBindInputs = _getBindInputs
InterfaceElement.addBindParam = _addBindParam
InterfaceElement.getBindParams = _getBindParams
InterfaceElement.getBindTokens = _getBindTokens


#
# Node
#

def _getReferencedNodeDef(self):
    "(Deprecated) Return the first NodeDef that declares this node."
    warnings.warn("This function is deprecated; call Node.getNodeDef instead.", DeprecationWarning, stacklevel = 2)
    return self.getNodeDef()

def _addShaderRef(self, name, nodeName):
    "(Deprecated) Add a shader reference to this material element."
    warnings.warn("This function is deprecated; material elements have been replaced with material nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return self.getParent().addNode(nodeName, name)

def _getShaderRefs(self):
    """(Deprecated) Return a vector of all shader references in this material element."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return getShaderNodes(self)

def _getActiveShaderRefs(self):
    """(Deprecated) Return a vector of all shader references in this material element, taking material inheritance into account."""
    warnings.warn("This function is deprecated; shader references have been replaced with shader nodes in 1.38.", DeprecationWarning, stacklevel = 2)
    return getShaderNodes(self)

Node.getReferencedNodeDef = _getReferencedNodeDef
Node.addShaderRef = _addShaderRef
Node.getShaderRefs = _getShaderRefs
Node.getActiveShaderRefs = _getActiveShaderRefs


#
# PropertySet
#

def _setPropertyValue(self, name, value, typeString = ''):
    """Set the typed value of a property by its name, creating a child element
       to hold the property if needed."""
    method = getattr(self.__class__, "_setPropertyValue" + getTypeString(value))
    return method(self, name, value, typeString)

def _getPropertyValue(self, name, target = ''):
    """Return the typed value of a property by its name.  If the given property
       is not found, then None is returned."""
    value = self._getPropertyValue(name)
    return value.getData() if value else None

PropertySet.setPropertyValue = _setPropertyValue
PropertySet.getPropertyValue = _getPropertyValue


#
# GeomInfo
#

def _setGeomPropValue(self, name, value, typeString = ''):
    """Set the value of a geomprop by its name, creating a child element
       to hold the geomprop if needed."""
    method = getattr(self.__class__, "_setGeomPropValue" + getTypeString(value))
    return method(self, name, value, typeString)

def _addGeomAttr(self, name):
    "(Deprecated) Add a geomprop to this element."
    warnings.warn("This function is deprecated; call GeomInfo.addGeomProp() instead", DeprecationWarning, stacklevel = 2)
    return self.addGeomProp(name)

def _setGeomAttrValue(self, name, value, typeString = ''):
    "(Deprecated) Set the value of a geomattr by its name."
    warnings.warn("This function is deprecated; call GeomInfo.setGeomPropValue() instead", DeprecationWarning, stacklevel = 2)
    return self.setGeomPropValue(name, value, typeString)

GeomInfo.setGeomPropValue = _setGeomPropValue
GeomInfo.addGeomAttr = _addGeomAttr
GeomInfo.setGeomAttrValue = _setGeomAttrValue


#
# Document
#

def _addMaterial(self, name):
    """(Deprecated) Add a material element to the document."""
    warnings.warn("This function is deprecated; call Document.addMaterialNode() instead.", DeprecationWarning, stacklevel = 2)
    return self.addMaterialNode(name)

def _getMaterials(self):
    """(Deprecated) Return a vector of all materials in the document."""
    warnings.warn("This function is deprecated; call Document.getMaterialNodes() instead.", DeprecationWarning, stacklevel = 2)
    return self.getMaterialNodes()

Document.addMaterial = _addMaterial
Document.getMaterials = _getMaterials


#
# Value
#

def _typeToName(t):
    "(Deprecated) Return the MaterialX type string associated with the given Python type."
    warnings.warn("This function is deprecated; call MaterialX.getTypeString instead.", DeprecationWarning, stacklevel = 2)
    return getTypeString(t())

def _valueToString(value):
    "(Deprecated) Convert a Python value to its correponding MaterialX value string."
    warnings.warn("This function is deprecated; call MaterialX.getValueString instead.", DeprecationWarning, stacklevel = 2)
    return getValueString(value)

def _stringToValue(string, t):
    "(Deprecated) Convert a MaterialX value string and Python type to the corresponding Python value."
    warnings.warn("This function is deprecated; call MaterialX.createValueFromStrings instead.", DeprecationWarning, stacklevel = 2)
    return createValueFromStrings(string, getTypeString(t()))

typeToName = _typeToName
valueToString = _valueToString
stringToValue = _stringToValue


#
# XmlIo
#

readFromXmlFile = readFromXmlFileBase


#
# Default Data Paths
#

def getDefaultDataSearchPath():
    """
    Return the default data search path.
    """
    return FileSearchPath(os.path.dirname(__file__))

def getDefaultDataLibraryFolders():
    """
    Return list of default data library folders
    """
    return [ 'libraries' ]
