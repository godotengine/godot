#!/usr/bin/env python
'''
Native Python wrappers for PyMaterialX and PyOpenColorIO, providing helper
functions for transforming MaterialX colors between OpenColorIO color spaces.

By default, the OpenColorIO configuration packaged with MaterialX Python will
be used, but clients may instead pass their own custom configurations to these
methods.
'''

import os

from .PyMaterialXCore import *


#--------------------------------------------------------------------------------
_defaultConfig = None
_defaultConfigFilename = 'config/config.ocio'
_validateDefaultConfig = False


#--------------------------------------------------------------------------------
def getColorSpaces(cms = 'ocio', config = None):
    """Return a list containing the names of all supported color spaces.
       By default, the OCIO color management system and default MaterialX
       config are used."""

    if cms != 'ocio':
        raise ValueError('Color management system is unrecognized: ' + cms)
    if config is None:
        config = getDefaultOCIOConfig()

    return [cs.getName() for cs in config.getColorSpaces()]

def transformColor(color, sourceColorSpace, destColorSpace, cms = 'ocio', config = None):
    """Given a MaterialX color and the names of two supported color spaces,
       transform the color from the source to the destination color space.
       By default, the OCIO color management system and default MaterialX
       config are used."""

    if cms != 'ocio':
        raise ValueError('Color management system is unrecognized: ' + cms)
    if config is None:
        config = getDefaultOCIOConfig()

    newColor = color
    processor = config.getProcessor(str(sourceColorSpace), str(destColorSpace))
    if isinstance(newColor, Color3):
        newColor = Color3(processor.applyRGB(newColor))
    elif isinstance(newColor, Color4):
        newColor = Color4(processor.applyRGBA(newColor))

    return newColor

def getDefaultOCIOConfig():
    """Return the default OCIO config packaged with this Python library.
       Raises ImportError if the PyOpenColorIO module cannot be imported."""
    global _defaultConfig

    if _defaultConfig is None:
        import PyOpenColorIO
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        configFilename = os.path.join(scriptDir, _defaultConfigFilename)
        _defaultConfig = PyOpenColorIO.Config.CreateFromFile(configFilename)
        if _validateDefaultConfig:
            _defaultConfig.sanityCheck()

    return _defaultConfig
