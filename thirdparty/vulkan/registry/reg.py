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

import copy
import re
import sys
import xml.etree.ElementTree as etree
from collections import defaultdict, namedtuple
from generator import OutputGenerator, write

# matchAPIProfile - returns whether an API and profile
#   being generated matches an element's profile
# api - string naming the API to match
# profile - string naming the profile to match
# elem - Element which (may) have 'api' and 'profile'
#   attributes to match to.
# If a tag is not present in the Element, the corresponding API
#   or profile always matches.
# Otherwise, the tag must exactly match the API or profile.
# Thus, if 'profile' = core:
#   <remove> with no attribute will match
#   <remove profile='core'> will match
#   <remove profile='compatibility'> will not match
# Possible match conditions:
#   Requested   Element
#   Profile     Profile
#   ---------   --------
#   None        None        Always matches
#   'string'    None        Always matches
#   None        'string'    Does not match. Can't generate multiple APIs
#                           or profiles, so if an API/profile constraint
#                           is present, it must be asked for explicitly.
#   'string'    'string'    Strings must match
#
#   ** In the future, we will allow regexes for the attributes,
#   not just strings, so that api="^(gl|gles2)" will match. Even
#   this isn't really quite enough, we might prefer something
#   like "gl(core)|gles1(common-lite)".
def matchAPIProfile(api, profile, elem):
    """Match a requested API & profile name to a api & profile attributes of an Element"""
    # Match 'api', if present
    elem_api = elem.get('api')
    if elem_api:
        if api is None:
            raise UserWarning("No API requested, but 'api' attribute is present with value '" +
                              elem_api + "'")
        elif api != elem_api:
            # Requested API doesn't match attribute
            return False
    elem_profile = elem.get('profile')
    if elem_profile:
        if profile is None:
            raise UserWarning("No profile requested, but 'profile' attribute is present with value '" +
                elem_profile + "'")
        elif profile != elem_profile:
            # Requested profile doesn't match attribute
            return False
    return True

# BaseInfo - base class for information about a registry feature
# (type/group/enum/command/API/extension).
#   required - should this feature be defined during header generation
#     (has it been removed by a profile or version)?
#   declared - has this feature been defined already?
#   elem - etree Element for this feature
#   resetState() - reset required/declared to initial values. Used
#     prior to generating a new API interface.
#   compareElem(info) - return True if self.elem and info.elem have the
#     same definition.
class BaseInfo:
    """Represents the state of a registry feature, used during API generation"""
    def __init__(self, elem):
        self.required = False
        self.declared = False
        self.elem = elem
    def resetState(self):
        self.required = False
        self.declared = False
    def compareElem(self, info):
        # Just compares the tag and attributes.
        # @@ This should be virtualized. In particular, comparing <enum>
        # tags requires special-casing on the attributes, as 'extnumber' is
        # only relevant when 'offset' is present.
        selfKeys = sorted(self.elem.keys())
        infoKeys = sorted(info.elem.keys())

        if selfKeys != infoKeys:
            return False

        # Ignore value of 'extname' and 'extnumber', as these will inherently
        # be different when redefining the same interface in different feature
        # and/or extension blocks.
        for key in selfKeys:
            if (key != 'extname' and key != 'extnumber' and
                (self.elem.get(key) != info.elem.get(key))):
                return False

        return True

# TypeInfo - registry information about a type. No additional state
#   beyond BaseInfo is required.
class TypeInfo(BaseInfo):
    """Represents the state of a registry type"""
    def __init__(self, elem):
        BaseInfo.__init__(self, elem)
        self.additionalValidity = []
        self.removedValidity = []
    def resetState(self):
        BaseInfo.resetState(self)
        self.additionalValidity = []
        self.removedValidity = []

# GroupInfo - registry information about a group of related enums
# in an <enums> block, generally corresponding to a C "enum" type.
class GroupInfo(BaseInfo):
    """Represents the state of a registry <enums> group"""
    def __init__(self, elem):
        BaseInfo.__init__(self, elem)

# EnumInfo - registry information about an enum
#   type - numeric type of the value of the <enum> tag
#     ( '' for GLint, 'u' for GLuint, 'ull' for GLuint64 )
class EnumInfo(BaseInfo):
    """Represents the state of a registry enum"""
    def __init__(self, elem):
        BaseInfo.__init__(self, elem)
        self.type = elem.get('type')
        if self.type is None:
            self.type = ''

# CmdInfo - registry information about a command
class CmdInfo(BaseInfo):
    """Represents the state of a registry command"""
    def __init__(self, elem):
        BaseInfo.__init__(self, elem)
        self.additionalValidity = []
        self.removedValidity = []
    def resetState(self):
        BaseInfo.resetState(self)
        self.additionalValidity = []
        self.removedValidity = []

# FeatureInfo - registry information about an API <feature>
# or <extension>
#   name - feature name string (e.g. 'VK_KHR_surface')
#   version - feature version number (e.g. 1.2). <extension>
#     features are unversioned and assigned version number 0.
#     ** This is confusingly taken from the 'number' attribute of <feature>.
#        Needs fixing.
#   number - extension number, used for ordering and for
#     assigning enumerant offsets. <feature> features do
#     not have extension numbers and are assigned number 0.
#   category - category, e.g. VERSION or khr/vendor tag
#   emit - has this feature been defined already?
class FeatureInfo(BaseInfo):
    """Represents the state of an API feature (version/extension)"""
    def __init__(self, elem):
        BaseInfo.__init__(self, elem)
        self.name = elem.get('name')
        # Determine element category (vendor). Only works
        # for <extension> elements.
        if elem.tag == 'feature':
            self.category = 'VERSION'
            self.version = elem.get('name')
            self.versionNumber = elem.get('number')
            self.number = "0"
            self.supported = None
        else:
            self.category = self.name.split('_', 2)[1]
            self.version = "0"
            self.versionNumber = "0"
            self.number = elem.get('number')
            # If there's no 'number' attribute, use 0, so sorting works
            if self.number is None:
                self.number = 0
            self.supported = elem.get('supported')
        self.emit = False

# Registry - object representing an API registry, loaded from an XML file
# Members
#   tree - ElementTree containing the root <registry>
#   typedict - dictionary of TypeInfo objects keyed by type name
#   groupdict - dictionary of GroupInfo objects keyed by group name
#   enumdict - dictionary of EnumInfo objects keyed by enum name
#   cmddict - dictionary of CmdInfo objects keyed by command name
#   apidict - dictionary of <api> Elements keyed by API name
#   extensions - list of <extension> Elements
#   extdict - dictionary of <extension> Elements keyed by extension name
#   gen - OutputGenerator object used to write headers / messages
#   genOpts - GeneratorOptions object used to control which
#     fetures to write and how to format them
#   emitFeatures - True to actually emit features for a version / extension,
#     or False to just treat them as emitted
#   breakPat - regexp pattern to break on when generatng names
# Public methods
#   loadElementTree(etree) - load registry from specified ElementTree
#   loadFile(filename) - load registry from XML file
#   setGenerator(gen) - OutputGenerator to use
#   breakOnName() - specify a feature name regexp to break on when
#     generating features.
#   parseTree() - parse the registry once loaded & create dictionaries
#   dumpReg(maxlen, filehandle) - diagnostic to dump the dictionaries
#     to specified file handle (default stdout). Truncates type /
#     enum / command elements to maxlen characters (default 80)
#   generator(g) - specify the output generator object
#   apiGen(apiname, genOpts) - generate API headers for the API type
#     and profile specified in genOpts, but only for the versions and
#     extensions specified there.
#   apiReset() - call between calls to apiGen() to reset internal state
# Private methods
#   addElementInfo(elem,info,infoName,dictionary) - add feature info to dict
#   lookupElementInfo(fname,dictionary) - lookup feature info in dict
class Registry:
    """Represents an API registry loaded from XML"""
    def __init__(self):
        self.tree         = None
        self.typedict     = {}
        self.groupdict    = {}
        self.enumdict     = {}
        self.cmddict      = {}
        self.apidict      = {}
        self.extensions   = []
        self.requiredextensions = [] # Hack - can remove it after validity generator goes away
        # ** Global types for automatic source generation **
        # Length Member data
        self.commandextensiontuple = namedtuple('commandextensiontuple',
                                       ['command',        # The name of the command being modified
                                        'value',          # The value to append to the command
                                        'extension'])     # The name of the extension that added it
        self.validextensionstructs = defaultdict(list)
        self.commandextensionsuccesses = []
        self.commandextensionerrors = []
        self.extdict      = {}
        # A default output generator, so commands prior to apiGen can report
        # errors via the generator object.
        self.gen          = OutputGenerator()
        self.genOpts      = None
        self.emitFeatures = False
        self.breakPat     = None
        # self.breakPat     = re.compile('VkFenceImportFlagBits.*')

    def loadElementTree(self, tree):
        """Load ElementTree into a Registry object and parse it"""
        self.tree = tree
        self.parseTree()

    def loadFile(self, file):
        """Load an API registry XML file into a Registry object and parse it"""
        self.tree = etree.parse(file)
        self.parseTree()

    def setGenerator(self, gen):
        """Specify output generator object. None restores the default generator"""
        self.gen = gen
        self.gen.setRegistry(self)

    # addElementInfo - add information about an element to the
    # corresponding dictionary
    #   elem - <type>/<enums>/<enum>/<command>/<feature>/<extension> Element
    #   info - corresponding {Type|Group|Enum|Cmd|Feature}Info object
    #   infoName - 'type' / 'group' / 'enum' / 'command' / 'feature' / 'extension'
    #   dictionary - self.{type|group|enum|cmd|api|ext}dict
    # If the Element has an 'api' attribute, the dictionary key is the
    # tuple (name,api). If not, the key is the name. 'name' is an
    # attribute of the Element
    def addElementInfo(self, elem, info, infoName, dictionary):
        # self.gen.logMsg('diag', 'Adding ElementInfo.required =',
        #     info.required, 'name =', elem.get('name'))
        api = elem.get('api')
        if api:
            key = (elem.get('name'), api)
        else:
            key = elem.get('name')
        if key in dictionary:
            if not dictionary[key].compareElem(info):
                self.gen.logMsg('warn', 'Attempt to redefine', key,
                                'with different value (this may be benign)')
            #else:
            #    self.gen.logMsg('warn', 'Benign redefinition of', key,
            #                    'with identical value')
        else:
            dictionary[key] = info

    # lookupElementInfo - find a {Type|Enum|Cmd}Info object by name.
    # If an object qualified by API name exists, use that.
    #   fname - name of type / enum / command
    #   dictionary - self.{type|enum|cmd}dict
    def lookupElementInfo(self, fname, dictionary):
        key = (fname, self.genOpts.apiname)
        if key in dictionary:
            # self.gen.logMsg('diag', 'Found API-specific element for feature', fname)
            return dictionary[key]
        if fname in dictionary:
            # self.gen.logMsg('diag', 'Found generic element for feature', fname)
            return dictionary[fname]

        return None

    def breakOnName(self, regexp):
        self.breakPat = re.compile(regexp)

    def parseTree(self):
        """Parse the registry Element, once created"""
        # This must be the Element for the root <registry>
        self.reg = self.tree.getroot()

        # Create dictionary of registry types from toplevel <types> tags
        # and add 'name' attribute to each <type> tag (where missing)
        # based on its <name> element.
        #
        # There's usually one <types> block; more are OK
        # Required <type> attributes: 'name' or nested <name> tag contents
        self.typedict = {}
        for type_elem in self.reg.findall('types/type'):
            # If the <type> doesn't already have a 'name' attribute, set
            # it from contents of its <name> tag.
            if type_elem.get('name') is None:
                type_elem.set('name', type_elem.find('name').text)
            self.addElementInfo(type_elem, TypeInfo(type_elem), 'type', self.typedict)

        # Create dictionary of registry enum groups from <enums> tags.
        #
        # Required <enums> attributes: 'name'. If no name is given, one is
        # generated, but that group can't be identified and turned into an
        # enum type definition - it's just a container for <enum> tags.
        self.groupdict = {}
        for group in self.reg.findall('enums'):
            self.addElementInfo(group, GroupInfo(group), 'group', self.groupdict)

        # Create dictionary of registry enums from <enum> tags
        #
        # <enums> tags usually define different namespaces for the values
        #   defined in those tags, but the actual names all share the
        #   same dictionary.
        # Required <enum> attributes: 'name', 'value'
        # For containing <enums> which have type="enum" or type="bitmask",
        # tag all contained <enum>s are required. This is a stopgap until
        # a better scheme for tagging core and extension enums is created.
        self.enumdict = {}
        for enums in self.reg.findall('enums'):
            required = (enums.get('type') is not None)
            for enum in enums.findall('enum'):
                enumInfo = EnumInfo(enum)
                enumInfo.required = required
                self.addElementInfo(enum, enumInfo, 'enum', self.enumdict)

        # Create dictionary of registry commands from <command> tags
        # and add 'name' attribute to each <command> tag (where missing)
        # based on its <proto><name> element.
        #
        # There's usually only one <commands> block; more are OK.
        # Required <command> attributes: 'name' or <proto><name> tag contents
        self.cmddict = {}
        # List of commands which alias others. Contains
        #   [ aliasName, element ]
        # for each alias
        cmdAlias = []
        for cmd in self.reg.findall('commands/command'):
            # If the <command> doesn't already have a 'name' attribute, set
            # it from contents of its <proto><name> tag.
            name = cmd.get('name')
            if name is None:
                name = cmd.set('name', cmd.find('proto/name').text)
            ci = CmdInfo(cmd)
            self.addElementInfo(cmd, ci, 'command', self.cmddict)
            alias = cmd.get('alias')
            if alias:
                cmdAlias.append([name, alias, cmd])

        # Now loop over aliases, injecting a copy of the aliased command's
        # Element with the aliased prototype name replaced with the command
        # name - if it exists.
        for (name, alias, cmd) in cmdAlias:
            if alias in self.cmddict:
                #@ pdb.set_trace()
                aliasInfo = self.cmddict[alias]
                cmdElem = copy.deepcopy(aliasInfo.elem)
                cmdElem.find('proto/name').text = name
                cmdElem.set('name', name)
                cmdElem.set('alias', alias)
                ci = CmdInfo(cmdElem)
                # Replace the dictionary entry for the CmdInfo element
                self.cmddict[name] = ci

                #@  newString = etree.tostring(base, encoding="unicode").replace(aliasValue, aliasName)
                #@elem.append(etree.fromstring(replacement))
            else:
                self.gen.logMsg('warn', 'No matching <command> found for command',
                    cmd.get('name'), 'alias', alias)

        # Create dictionaries of API and extension interfaces
        #   from toplevel <api> and <extension> tags.
        self.apidict = {}
        for feature in self.reg.findall('feature'):
            featureInfo = FeatureInfo(feature)
            self.addElementInfo(feature, featureInfo, 'feature', self.apidict)

            # Add additional enums defined only in <feature> tags
            # to the corresponding core type.
            # When seen here, the <enum> element, processed to contain the
            # numeric enum value, is added to the corresponding <enums>
            # element, as well as adding to the enum dictionary. It is
            # *removed* from the <require> element it is introduced in.
            # Not doing this will cause spurious genEnum()
            # calls to be made in output generation, and it's easier
            # to handle here than in genEnum().
            #
            # In lxml.etree, an Element can have only one parent, so the
            # append() operation also removes the element. But in Python's
            # ElementTree package, an Element can have multiple parents. So
            # it must be explicitly removed from the <require> tag, leading
            # to the nested loop traversal of <require>/<enum> elements
            # below.
            #
            # This code also adds a 'version' attribute containing the
            # api version.
            #
            # For <enum> tags which are actually just constants, if there's
            # no 'extends' tag but there is a 'value' or 'bitpos' tag, just
            # add an EnumInfo record to the dictionary. That works because
            # output generation of constants is purely dependency-based, and
            # doesn't need to iterate through the XML tags.
            for elem in feature.findall('require'):
                for enum in elem.findall('enum'):
                    addEnumInfo = False
                    groupName = enum.get('extends')
                    if groupName is not None:
                        # self.gen.logMsg('diag', 'Found extension enum',
                        #     enum.get('name'))
                        # Add version number attribute to the <enum> element
                        enum.set('version', featureInfo.version)
                        # Look up the GroupInfo with matching groupName
                        if groupName in self.groupdict:
                            # self.gen.logMsg('diag', 'Matching group',
                            #     groupName, 'found, adding element...')
                            gi = self.groupdict[groupName]
                            gi.elem.append(enum)
                            # Remove element from parent <require> tag
                            # This should be a no-op in lxml.etree
                            elem.remove(enum)
                        else:
                            self.gen.logMsg('warn', 'NO matching group',
                                groupName, 'for enum', enum.get('name'), 'found.')
                        addEnumInfo = True
                    elif enum.get('value') or enum.get('bitpos') or enum.get('alias'):
                        # self.gen.logMsg('diag', 'Adding extension constant "enum"',
                        #     enum.get('name'))
                        addEnumInfo = True
                    if addEnumInfo:
                        enumInfo = EnumInfo(enum)
                        self.addElementInfo(enum, enumInfo, 'enum', self.enumdict)

        self.extensions = self.reg.findall('extensions/extension')
        self.extdict = {}
        for feature in self.extensions:
            featureInfo = FeatureInfo(feature)
            self.addElementInfo(feature, featureInfo, 'extension', self.extdict)

            # Add additional enums defined only in <extension> tags
            # to the corresponding core type.
            # Algorithm matches that of enums in a "feature" tag as above.
            #
            # This code also adds a 'extnumber' attribute containing the
            # extension number, used for enumerant value calculation.
            for elem in feature.findall('require'):
                for enum in elem.findall('enum'):
                    addEnumInfo = False
                    groupName = enum.get('extends')
                    if groupName is not None:
                        # self.gen.logMsg('diag', 'Found extension enum',
                        #     enum.get('name'))

                        # Add <extension> block's extension number attribute to
                        # the <enum> element unless specified explicitly, such
                        # as when redefining an enum in another extension.
                        extnumber = enum.get('extnumber')
                        if not extnumber:
                            enum.set('extnumber', featureInfo.number)

                        enum.set('extname', featureInfo.name)
                        enum.set('supported', featureInfo.supported)
                        # Look up the GroupInfo with matching groupName
                        if groupName in self.groupdict:
                            # self.gen.logMsg('diag', 'Matching group',
                            #     groupName, 'found, adding element...')
                            gi = self.groupdict[groupName]
                            gi.elem.append(enum)
                            # Remove element from parent <require> tag
                            # This should be a no-op in lxml.etree
                            elem.remove(enum)
                        else:
                            self.gen.logMsg('warn', 'NO matching group',
                                groupName, 'for enum', enum.get('name'), 'found.')
                        addEnumInfo = True
                    elif enum.get('value') or enum.get('bitpos') or enum.get('alias'):
                        # self.gen.logMsg('diag', 'Adding extension constant "enum"',
                        #     enum.get('name'))
                        addEnumInfo = True
                    if addEnumInfo:
                        enumInfo = EnumInfo(enum)
                        self.addElementInfo(enum, enumInfo, 'enum', self.enumdict)

        # Construct a "validextensionstructs" list for parent structures
        # based on "structextends" tags in child structures
        disabled_types = []
        for disabled_ext in self.reg.findall('extensions/extension[@supported="disabled"]'):
            for type_elem in disabled_ext.findall("*/type"):
                disabled_types.append(type_elem.get('name'))
        for type_elem in self.reg.findall('types/type'):
            if type_elem.get('name') not in disabled_types:
                parentStructs = type_elem.get('structextends')
                if parentStructs is not None:
                    for parent in parentStructs.split(','):
                        # self.gen.logMsg('diag', type.get('name'), 'extends', parent)
                        self.validextensionstructs[parent].append(type_elem.get('name'))
        # Sort the lists so they don't depend on the XML order
        for parent in self.validextensionstructs:
            self.validextensionstructs[parent].sort()

    def dumpReg(self, maxlen = 120, filehandle = sys.stdout):
        """Dump all the dictionaries constructed from the Registry object"""
        write('***************************************', file=filehandle)
        write('    ** Dumping Registry contents **',     file=filehandle)
        write('***************************************', file=filehandle)
        write('// Types', file=filehandle)
        for name in self.typedict:
            tobj = self.typedict[name]
            write('    Type', name, '->', etree.tostring(tobj.elem)[0:maxlen], file=filehandle)
        write('// Groups', file=filehandle)
        for name in self.groupdict:
            gobj = self.groupdict[name]
            write('    Group', name, '->', etree.tostring(gobj.elem)[0:maxlen], file=filehandle)
        write('// Enums', file=filehandle)
        for name in self.enumdict:
            eobj = self.enumdict[name]
            write('    Enum', name, '->', etree.tostring(eobj.elem)[0:maxlen], file=filehandle)
        write('// Commands', file=filehandle)
        for name in self.cmddict:
            cobj = self.cmddict[name]
            write('    Command', name, '->', etree.tostring(cobj.elem)[0:maxlen], file=filehandle)
        write('// APIs', file=filehandle)
        for key in self.apidict:
            write('    API Version ', key, '->',
                etree.tostring(self.apidict[key].elem)[0:maxlen], file=filehandle)
        write('// Extensions', file=filehandle)
        for key in self.extdict:
            write('    Extension', key, '->',
                etree.tostring(self.extdict[key].elem)[0:maxlen], file=filehandle)
        # write('***************************************', file=filehandle)
        # write('    ** Dumping XML ElementTree **', file=filehandle)
        # write('***************************************', file=filehandle)
        # write(etree.tostring(self.tree.getroot(),pretty_print=True), file=filehandle)

    # typename - name of type
    # required - boolean (to tag features as required or not)
    def markTypeRequired(self, typename, required):
        """Require (along with its dependencies) or remove (but not its dependencies) a type"""
        self.gen.logMsg('diag', 'tagging type:', typename, '-> required =', required)
        # Get TypeInfo object for <type> tag corresponding to typename
        typeinfo = self.lookupElementInfo(typename, self.typedict)
        if typeinfo is not None:
            if required:
                # Tag type dependencies in 'alias' and 'required' attributes as
                # required. This DOES NOT un-tag dependencies in a <remove>
                # tag. See comments in markRequired() below for the reason.
                for attrib_name in [ 'requires', 'alias' ]:
                    depname = typeinfo.elem.get(attrib_name)
                    if depname:
                        self.gen.logMsg('diag', 'Generating dependent type',
                            depname, 'for', attrib_name, 'type', typename)
                        # Don't recurse on self-referential structures.
                        if typename != depname:
                            self.markTypeRequired(depname, required)
                        else:
                            self.gen.logMsg('diag', 'type', typename, 'is self-referential')
                # Tag types used in defining this type (e.g. in nested
                # <type> tags)
                # Look for <type> in entire <command> tree,
                # not just immediate children
                for subtype in typeinfo.elem.findall('.//type'):
                    self.gen.logMsg('diag', 'markRequired: type requires dependent <type>', subtype.text)
                    if typename != subtype.text:
                        self.markTypeRequired(subtype.text, required)
                    else:
                        self.gen.logMsg('diag', 'type', typename, 'is self-referential')
                # Tag enums used in defining this type, for example in
                #   <member><name>member</name>[<enum>MEMBER_SIZE</enum>]</member>
                for subenum in typeinfo.elem.findall('.//enum'):
                    self.gen.logMsg('diag', 'markRequired: type requires dependent <enum>', subenum.text)
                    self.markEnumRequired(subenum.text, required)
                # Tag type dependency in 'bitvalues' attributes as
                # required. This ensures that the bit values for a flag
                # are emitted
                depType = typeinfo.elem.get('bitvalues')
                if depType:
                    self.gen.logMsg('diag', 'Generating bitflag type',
                        depType, 'for type', typename)
                    self.markTypeRequired(depType, required)
                    group = self.lookupElementInfo(depType, self.groupdict)
                    if group is not None:
                        group.flagType = typeinfo

            typeinfo.required = required
        elif '.h' not in typename:
            self.gen.logMsg('warn', 'type:', typename , 'IS NOT DEFINED')

    # enumname - name of enum
    # required - boolean (to tag features as required or not)
    def markEnumRequired(self, enumname, required):
        self.gen.logMsg('diag', 'tagging enum:', enumname, '-> required =', required)
        enum = self.lookupElementInfo(enumname, self.enumdict)
        if enum is not None:
            enum.required = required
            # Tag enum dependencies in 'alias' attribute as required
            depname = enum.elem.get('alias')
            if depname:
                self.gen.logMsg('diag', 'Generating dependent enum',
                    depname, 'for alias', enumname, 'required =', enum.required)
                self.markEnumRequired(depname, required)
        else:
            self.gen.logMsg('warn', 'enum:', enumname , 'IS NOT DEFINED')

    # cmdname - name of command
    # required - boolean (to tag features as required or not)
    def markCmdRequired(self, cmdname, required):
        self.gen.logMsg('diag', 'tagging command:', cmdname, '-> required =', required)
        cmd = self.lookupElementInfo(cmdname, self.cmddict)
        if cmd is not None:
            cmd.required = required
            # Tag command dependencies in 'alias' attribute as required
            depname = cmd.elem.get('alias')
            if depname:
                self.gen.logMsg('diag', 'Generating dependent command',
                    depname, 'for alias', cmdname)
                self.markCmdRequired(depname, required)
            # Tag all parameter types of this command as required.
            # This DOES NOT remove types of commands in a <remove>
            # tag, because many other commands may use the same type.
            # We could be more clever and reference count types,
            # instead of using a boolean.
            if required:
                # Look for <type> in entire <command> tree,
                # not just immediate children
                for type_elem in cmd.elem.findall('.//type'):
                    self.gen.logMsg('diag', 'markRequired: command implicitly requires dependent type', type_elem.text)
                    self.markTypeRequired(type_elem.text, required)
        else:
            self.gen.logMsg('warn', 'command:', cmdname, 'IS NOT DEFINED')

    # featurename - name of the feature
    # feature - Element for <require> or <remove> tag
    # required - boolean (to tag features as required or not)
    def markRequired(self, featurename, feature, required):
        """Require or remove features specified in the Element"""
        self.gen.logMsg('diag', 'markRequired (feature = <too long to print>, required =', required, ')')

        # Loop over types, enums, and commands in the tag
        # @@ It would be possible to respect 'api' and 'profile' attributes
        #  in individual features, but that's not done yet.
        for typeElem in feature.findall('type'):
            self.markTypeRequired(typeElem.get('name'), required)
        for enumElem in feature.findall('enum'):
            self.markEnumRequired(enumElem.get('name'), required)
        for cmdElem in feature.findall('command'):
            self.markCmdRequired(cmdElem.get('name'), required)

        # Extensions may need to extend existing commands or other items in the future.
        # So, look for extend tags.
        for extendElem in feature.findall('extend'):
            extendType = extendElem.get('type')
            if extendType == 'command':
                commandName = extendElem.get('name')
                successExtends = extendElem.get('successcodes')
                if successExtends is not None:
                    for success in successExtends.split(','):
                        self.commandextensionsuccesses.append(self.commandextensiontuple(command=commandName,
                                                                                         value=success,
                                                                                         extension=featurename))
                errorExtends = extendElem.get('errorcodes')
                if errorExtends is not None:
                    for error in errorExtends.split(','):
                        self.commandextensionerrors.append(self.commandextensiontuple(command=commandName,
                                                                                      value=error,
                                                                                      extension=featurename))
            else:
                self.gen.logMsg('warn', 'extend type:', extendType, 'IS NOT SUPPORTED')

    # interface - Element for <version> or <extension>, containing
    #   <require> and <remove> tags
    # featurename - name of the feature
    # api - string specifying API name being generated
    # profile - string specifying API profile being generated
    def requireAndRemoveFeatures(self, interface, featurename, api, profile):
        """Process <require> and <remove> tags for a <version> or <extension>"""
        # <require> marks things that are required by this version/profile
        for feature in interface.findall('require'):
            if matchAPIProfile(api, profile, feature):
                self.markRequired(featurename, feature, True)
        # <remove> marks things that are removed by this version/profile
        for feature in interface.findall('remove'):
            if matchAPIProfile(api, profile, feature):
                self.markRequired(featurename, feature, False)

    def assignAdditionalValidity(self, interface, api, profile):
        # Loop over all usage inside all <require> tags.
        for feature in interface.findall('require'):
            if matchAPIProfile(api, profile, feature):
                for v in feature.findall('usage'):
                    if v.get('command'):
                        self.cmddict[v.get('command')].additionalValidity.append(copy.deepcopy(v))
                    if v.get('struct'):
                        self.typedict[v.get('struct')].additionalValidity.append(copy.deepcopy(v))

        # Loop over all usage inside all <remove> tags.
        for feature in interface.findall('remove'):
            if matchAPIProfile(api, profile, feature):
                for v in feature.findall('usage'):
                    if v.get('command'):
                        self.cmddict[v.get('command')].removedValidity.append(copy.deepcopy(v))
                    if v.get('struct'):
                        self.typedict[v.get('struct')].removedValidity.append(copy.deepcopy(v))

    # generateFeature - generate a single type / enum group / enum / command,
    # and all its dependencies as needed.
    #   fname - name of feature (<type>/<enum>/<command>)
    #   ftype - type of feature, 'type' | 'enum' | 'command'
    #   dictionary - of *Info objects - self.{type|enum|cmd}dict
    def generateFeature(self, fname, ftype, dictionary):
        #@ # Break to debugger on matching name pattern
        #@ if self.breakPat and re.match(self.breakPat, fname):
        #@    pdb.set_trace()

        self.gen.logMsg('diag', 'generateFeature: generating', ftype, fname)
        f = self.lookupElementInfo(fname, dictionary)
        if f is None:
            # No such feature. This is an error, but reported earlier
            self.gen.logMsg('diag', 'No entry found for feature', fname,
                            'returning!')
            return

        # If feature isn't required, or has already been declared, return
        if not f.required:
            self.gen.logMsg('diag', 'Skipping', ftype, fname, '(not required)')
            return
        if f.declared:
            self.gen.logMsg('diag', 'Skipping', ftype, fname, '(already declared)')
            return
        # Always mark feature declared, as though actually emitted
        f.declared = True

        # Determine if this is an alias, and of what, if so
        alias = f.elem.get('alias')
        if alias:
            self.gen.logMsg('diag', fname, 'is an alias of', alias)

        # Pull in dependent declaration(s) of the feature.
        # For types, there may be one type in the 'requires' attribute of
        #   the element, one in the 'alias' attribute, and many in
        #   embedded <type> and <enum> tags within the element.
        # For commands, there may be many in <type> tags within the element.
        # For enums, no dependencies are allowed (though perhaps if you
        #   have a uint64 enum, it should require that type).
        genProc = None
        followupFeature = None
        if ftype == 'type':
            genProc = self.gen.genType

            # Generate type dependencies in 'alias' and 'requires' attributes
            if alias:
                self.generateFeature(alias, 'type', self.typedict)
            requires = f.elem.get('requires')
            if requires:
                self.gen.logMsg('diag', 'Generating required dependent type',
                                requires)
                self.generateFeature(requires, 'type', self.typedict)

            # Generate types used in defining this type (e.g. in nested
            # <type> tags)
            # Look for <type> in entire <command> tree,
            # not just immediate children
            for subtype in f.elem.findall('.//type'):
                self.gen.logMsg('diag', 'Generating required dependent <type>',
                    subtype.text)
                self.generateFeature(subtype.text, 'type', self.typedict)

            # Generate enums used in defining this type, for example in
            #   <member><name>member</name>[<enum>MEMBER_SIZE</enum>]</member>
            for subtype in f.elem.findall('.//enum'):
                self.gen.logMsg('diag', 'Generating required dependent <enum>',
                    subtype.text)
                self.generateFeature(subtype.text, 'enum', self.enumdict)

            # If the type is an enum group, look up the corresponding
            # group in the group dictionary and generate that instead.
            if f.elem.get('category') == 'enum':
                self.gen.logMsg('diag', 'Type', fname, 'is an enum group, so generate that instead')
                group = self.lookupElementInfo(fname, self.groupdict)
                if alias is not None:
                    # An alias of another group name.
                    # Pass to genGroup with 'alias' parameter = aliased name
                    self.gen.logMsg('diag', 'Generating alias', fname,
                                    'for enumerated type', alias)
                    # Now, pass the *aliased* GroupInfo to the genGroup, but
                    # with an additional parameter which is the alias name.
                    genProc = self.gen.genGroup
                    f = self.lookupElementInfo(alias, self.groupdict)
                elif group is None:
                    self.gen.logMsg('warn', 'Skipping enum type', fname,
                        ': No matching enumerant group')
                    return
                else:
                    genProc = self.gen.genGroup
                    f = group

                    #@ The enum group is not ready for generation. At this
                    #@   point, it contains all <enum> tags injected by
                    #@   <extension> tags without any verification of whether
                    #@   they're required or not. It may also contain
                    #@   duplicates injected by multiple consistent
                    #@   definitions of an <enum>.

                    #@ Pass over each enum, marking its enumdict[] entry as
                    #@ required or not. Mark aliases of enums as required,
                    #@ too.

                    enums = group.elem.findall('enum')

                    self.gen.logMsg('diag', 'generateFeature: checking enums for group', fname)

                    # Check for required enums, including aliases
                    # LATER - Check for, report, and remove duplicates?
                    enumAliases = []
                    for elem in enums:
                        name = elem.get('name')

                        required = False

                        extname = elem.get('extname')
                        version = elem.get('version')
                        if extname is not None:
                            # 'supported' attribute was injected when the <enum> element was
                            # moved into the <enums> group in Registry.parseTree()
                            if self.genOpts.defaultExtensions == elem.get('supported'):
                                required = True
                            elif re.match(self.genOpts.addExtensions, extname) is not None:
                                required = True
                        elif version is not None:
                            required = re.match(self.genOpts.emitversions, version) is not None
                        else:
                            required = True

                        self.gen.logMsg('diag', '* required =', required, 'for', name)
                        if required:
                            # Mark this element as required (in the element, not the EnumInfo)
                            elem.set('required', 'true')
                            # If it's an alias, track that for later use
                            enumAlias = elem.get('alias')
                            if enumAlias:
                                enumAliases.append(enumAlias)
                    for elem in enums:
                        name = elem.get('name')
                        if name in enumAliases:
                            elem.set('required', 'true')
                            self.gen.logMsg('diag', '* also need to require alias', name)
            if f.elem.get('category') == 'bitmask':
                followupFeature = f.elem.get( 'bitvalues' )
        elif ftype == 'command':
            # Generate command dependencies in 'alias' attribute
            if alias:
                self.generateFeature(alias, 'command', self.cmddict)

            genProc = self.gen.genCmd
            for type_elem in f.elem.findall('.//type'):
                depname = type_elem.text
                self.gen.logMsg('diag', 'Generating required parameter type',
                                depname)
                self.generateFeature(depname, 'type', self.typedict)
        elif ftype == 'enum':
            # Generate enum dependencies in 'alias' attribute
            if alias:
                self.generateFeature(alias, 'enum', self.enumdict)
            genProc = self.gen.genEnum

        # Actually generate the type only if emitting declarations
        if self.emitFeatures:
            self.gen.logMsg('diag', 'Emitting', ftype, 'decl for', fname)
            genProc(f, fname, alias)
        else:
            self.gen.logMsg('diag', 'Skipping', ftype, fname,
                            '(should not be emitted)')

        if followupFeature :
            self.gen.logMsg('diag', 'Generating required bitvalues <enum>',
                followupFeature)
            self.generateFeature(followupFeature, "type", self.typedict)

    # generateRequiredInterface - generate all interfaces required
    # by an API version or extension
    #   interface - Element for <version> or <extension>
    def generateRequiredInterface(self, interface):
        """Generate required C interface for specified API version/extension"""

        # Loop over all features inside all <require> tags.
        for features in interface.findall('require'):
            for t in features.findall('type'):
                self.generateFeature(t.get('name'), 'type', self.typedict)
            for e in features.findall('enum'):
                self.generateFeature(e.get('name'), 'enum', self.enumdict)
            for c in features.findall('command'):
                self.generateFeature(c.get('name'), 'command', self.cmddict)

    # apiGen(genOpts) - generate interface for specified versions
    #   genOpts - GeneratorOptions object with parameters used
    #   by the Generator object.
    def apiGen(self, genOpts):
        """Generate interfaces for the specified API type and range of versions"""

        self.gen.logMsg('diag', '*******************************************')
        self.gen.logMsg('diag', '  Registry.apiGen file:', genOpts.filename,
                        'api:', genOpts.apiname,
                        'profile:', genOpts.profile)
        self.gen.logMsg('diag', '*******************************************')

        self.genOpts = genOpts
        # Reset required/declared flags for all features
        self.apiReset()

        # Compile regexps used to select versions & extensions
        regVersions = re.compile(self.genOpts.versions)
        regEmitVersions = re.compile(self.genOpts.emitversions)
        regAddExtensions = re.compile(self.genOpts.addExtensions)
        regRemoveExtensions = re.compile(self.genOpts.removeExtensions)
        regEmitExtensions = re.compile(self.genOpts.emitExtensions)

        # Get all matching API feature names & add to list of FeatureInfo
        # Note we used to select on feature version attributes, not names.
        features = []
        apiMatch = False
        for key in self.apidict:
            fi = self.apidict[key]
            api = fi.elem.get('api')
            if api == self.genOpts.apiname:
                apiMatch = True
                if regVersions.match(fi.name):
                    # Matches API & version #s being generated. Mark for
                    # emission and add to the features[] list .
                    # @@ Could use 'declared' instead of 'emit'?
                    fi.emit = (regEmitVersions.match(fi.name) is not None)
                    features.append(fi)
                    if not fi.emit:
                        self.gen.logMsg('diag', 'NOT tagging feature api =', api,
                            'name =', fi.name, 'version =', fi.version,
                            'for emission (does not match emitversions pattern)')
                    else:
                        self.gen.logMsg('diag', 'Including feature api =', api,
                            'name =', fi.name, 'version =', fi.version,
                            'for emission (matches emitversions pattern)')
                else:
                    self.gen.logMsg('diag', 'NOT including feature api =', api,
                        'name =', fi.name, 'version =', fi.version,
                        '(does not match requested versions)')
            else:
                self.gen.logMsg('diag', 'NOT including feature api =', api,
                    'name =', fi.name,
                    '(does not match requested API)')
        if not apiMatch:
            self.gen.logMsg('warn', 'No matching API versions found!')

        # Get all matching extensions, in order by their extension number,
        # and add to the list of features.
        # Start with extensions tagged with 'api' pattern matching the API
        # being generated. Add extensions matching the pattern specified in
        # regExtensions, then remove extensions matching the pattern
        # specified in regRemoveExtensions
        for (extName,ei) in sorted(self.extdict.items(),key = lambda x : x[1].number if x[1].number is not None else '0'):
            extName = ei.name
            include = False

            # Include extension if defaultExtensions is not None and if the
            # 'supported' attribute matches defaultExtensions. The regexp in
            # 'supported' must exactly match defaultExtensions, so bracket
            # it with ^(pat)$.
            pat = '^(' + ei.elem.get('supported') + ')$'
            if (self.genOpts.defaultExtensions and
                     re.match(pat, self.genOpts.defaultExtensions)):
                self.gen.logMsg('diag', 'Including extension',
                    extName, "(defaultExtensions matches the 'supported' attribute)")
                include = True

            # Include additional extensions if the extension name matches
            # the regexp specified in the generator options. This allows
            # forcing extensions into an interface even if they're not
            # tagged appropriately in the registry.
            if regAddExtensions.match(extName) is not None:
                self.gen.logMsg('diag', 'Including extension',
                    extName, '(matches explicitly requested extensions to add)')
                include = True
            # Remove extensions if the name matches the regexp specified
            # in generator options. This allows forcing removal of
            # extensions from an interface even if they're tagged that
            # way in the registry.
            if regRemoveExtensions.match(extName) is not None:
                self.gen.logMsg('diag', 'Removing extension',
                    extName, '(matches explicitly requested extensions to remove)')
                include = False

            # If the extension is to be included, add it to the
            # extension features list.
            if include:
                ei.emit = (regEmitExtensions.match(extName) is not None)
                features.append(ei)
                if not ei.emit:
                    self.gen.logMsg('diag', 'NOT tagging extension',
                        extName,
                        'for emission (does not match emitextensions pattern)')

                # Hack - can be removed when validity generator goes away
                # (Jon) I'm not sure what this does, or if it should respect
                # the ei.emit flag above.
                self.requiredextensions.append(extName)
            else:
                self.gen.logMsg('diag', 'NOT including extension',
                    extName, '(does not match api attribute or explicitly requested extensions)')

        # Sort the extension features list, if a sort procedure is defined
        if self.genOpts.sortProcedure:
            self.genOpts.sortProcedure(features)

        # Pass 1: loop over requested API versions and extensions tagging
        #   types/commands/features as required (in an <require> block) or no
        #   longer required (in an <remove> block). It is possible to remove
        #   a feature in one version and restore it later by requiring it in
        #   a later version.
        # If a profile other than 'None' is being generated, it must
        #   match the profile attribute (if any) of the <require> and
        #   <remove> tags.
        self.gen.logMsg('diag', 'PASS 1: TAG FEATURES')
        for f in features:
            self.gen.logMsg('diag', 'PASS 1: Tagging required and removed features for',
                f.name)
            self.requireAndRemoveFeatures(f.elem, f.name, self.genOpts.apiname, self.genOpts.profile)
            self.assignAdditionalValidity(f.elem, self.genOpts.apiname, self.genOpts.profile)

        # Pass 2: loop over specified API versions and extensions printing
        #   declarations for required things which haven't already been
        #   generated.
        self.gen.logMsg('diag', 'PASS 2: GENERATE INTERFACES FOR FEATURES')
        self.gen.beginFile(self.genOpts)
        for f in features:
            self.gen.logMsg('diag', 'PASS 2: Generating interface for',
                f.name)
            emit = self.emitFeatures = f.emit
            if not emit:
                self.gen.logMsg('diag', 'PASS 2: NOT declaring feature',
                    f.elem.get('name'), 'because it is not tagged for emission')
            # Generate the interface (or just tag its elements as having been
            # emitted, if they haven't been).
            self.gen.beginFeature(f.elem, emit)
            self.generateRequiredInterface(f.elem)
            self.gen.endFeature()
        self.gen.endFile()

    # apiReset - use between apiGen() calls to reset internal state
    def apiReset(self):
        """Reset type/enum/command dictionaries before generating another API"""
        for datatype in self.typedict:
            self.typedict[datatype].resetState()
        for enum in self.enumdict:
            self.enumdict[enum].resetState()
        for cmd in self.cmddict:
            self.cmddict[cmd].resetState()
        for cmd in self.apidict:
            self.apidict[cmd].resetState()

    # validateGroups - check that group= attributes match actual groups
    def validateGroups(self):
        """Validate group= attributes on <param> and <proto> tags"""
        # Keep track of group names not in <group> tags
        badGroup = {}
        self.gen.logMsg('diag', 'VALIDATING GROUP ATTRIBUTES')
        for cmd in self.reg.findall('commands/command'):
            proto = cmd.find('proto')
            # funcname = cmd.find('proto/name').text
            group = proto.get('group')
            if group is not None and group not in self.groupdict:
                # self.gen.logMsg('diag', '*** Command ', funcname, ' has UNKNOWN return group ', group)
                if group not in badGroup:
                    badGroup[group] = 1
                else:
                    badGroup[group] = badGroup[group] +  1

            for param in cmd.findall('param'):
                pname = param.find('name')
                if pname is not None:
                    pname = pname.text
                else:
                    pname = param.get('name')
                group = param.get('group')
                if group is not None and group not in self.groupdict:
                    # self.gen.logMsg('diag', '*** Command ', funcname, ' param ', pname, ' has UNKNOWN group ', group)
                    if group not in badGroup:
                        badGroup[group] = 1
                    else:
                        badGroup[group] = badGroup[group] +  1

        if badGroup:
            self.gen.logMsg('diag', 'SUMMARY OF UNRECOGNIZED GROUPS')
            for key in sorted(badGroup.keys()):
                self.gen.logMsg('diag', '    ', key, ' occurred ', badGroup[key], ' times')
