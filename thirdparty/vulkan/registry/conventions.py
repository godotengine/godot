#!/usr/bin/python3 -i
#
# Copyright (c) 2013-2019 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Base class for working-group-specific style conventions,
# used in generation.

from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})

class ConventionsBase(ABC):
    """WG-specific conventions."""

    @abstractmethod
    def formatExtension(self, name):
        """Mark up a name as an extension for the spec."""
        raise NotImplementedError

    @property
    @abstractmethod
    def null(self):
        """Preferred spelling of NULL."""
        raise NotImplementedError

    def makeProseList(self, elements, connective='and'):
        """Make a (comma-separated) list for use in prose.

        Adds a connective (by default, 'and')
        before the last element if there are more than 1.

        Override with a different method or different call to
        _implMakeProseList if you want to add a comma for two elements,
        or not use a serial comma.
        """
        return self._implMakeProseList(elements, connective)

    @property
    def struct_macro(self):
        """Get the appropriate format macro for a structure.

        May override.
        """
        return 'sname:'

    def makeStructName(self, name):
        """Prepend the appropriate format macro for a structure to a structure type name.

        Uses struct_macro, so just override that if you want to change behavior.
        """
        return self.struct_macro + name

    @property
    def external_macro(self):
        """Get the appropriate format macro for an external type like uint32_t.

        May override.
        """
        return 'basetype:'

    def makeExternalTypeName(self, name):
        """Prepend the appropriate format macro for an external type like uint32_t to a type name.

        Uses external_macro, so just override that if you want to change behavior.
        """
        return self.external_macro + name

    def _implMakeProseList(self, elements, connective, comma_for_two_elts=False, serial_comma=True):
        """Internal-use implementation to make a (comma-separated) list for use in prose.

        Adds a connective (by default, 'and')
        before the last element if there are more than 1,
        and only includes commas if there are more than 2
        (if comma_for_two_elts is False).

        Don't edit these defaults, override self.makeProseList().
        """
        assert(serial_comma)  # didn't implement what we didn't need
        my_elts = list(elements)
        if len(my_elts) > 1:
            my_elts[-1] = '{} {}'.format(connective, my_elts[-1])

        if not comma_for_two_elts and len(my_elts) <= 2:
            return ' '.join(my_elts)
        return ', '.join(my_elts)

    @property
    @abstractmethod
    def file_suffix(self):
        """Return suffix of generated Asciidoctor files"""
        raise NotImplementedError

    @abstractmethod
    def api_name(self, spectype = None):
        """Return API name"""
        raise NotImplementedError

    @property
    @abstractmethod
    def api_prefix(self):
        """Return API token prefix"""
        raise NotImplementedError

    @property
    @abstractmethod
    def api_version_prefix(self):
        """Return API core version token prefix"""
        raise NotImplementedError

    @property
    @abstractmethod
    def KHR_prefix(self):
        """Return extension name prefix for KHR extensions"""
        raise NotImplementedError

    @property
    @abstractmethod
    def EXT_prefix(self):
        """Return extension name prefix for EXT extensions"""
        raise NotImplementedError
