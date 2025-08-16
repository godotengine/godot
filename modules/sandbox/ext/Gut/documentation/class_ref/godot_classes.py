from collections import OrderedDict
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

from godot_consts import *
import logger as lgr
import bitwes


class State:
    def __init__(self) -> None:
        self.num_errors = 0
        self.num_warnings = 0
        self.classes: OrderedDict[str, ClassDef] = OrderedDict()
        self.current_class: str = ""

        # Additional content and structure checks and validators.
        self.script_language_parity_check: ScriptLanguageParityCheck = ScriptLanguageParityCheck()


    def parse_class(self, class_root: ET.Element, filepath: str) -> None:
        # -bitwes: remove quotes from class names, these appear when the script
        # does not have a class_name.  This prevents the quotes from appearing
        # in TOC and allows linking to scripts by path without having to use
        # quotes.
        class_name = class_root.attrib["name"].replace('"', "")
        self.current_class = class_name

        class_def = ClassDef(class_name)
        self.classes[class_name] = class_def
        class_def.filepath = filepath

        inherits = class_root.get("inherits")
        if inherits is not None:
            class_def.inherits = inherits

        class_def.deprecated = class_root.get("deprecated")
        class_def.experimental = class_root.get("experimental")

        brief_desc = class_root.find("brief_description")
        if brief_desc is not None and brief_desc.text:
            class_def.brief_description = brief_desc.text

        desc = class_root.find("description")
        if desc is not None and desc.text:
            class_def.description = desc.text

        keywords = class_root.get("keywords")
        if keywords is not None:
            class_def.keywords = keywords

        properties = class_root.find("members")
        if properties is not None:
            for property in properties:
                assert property.tag == "member"

                property_name = property.attrib["name"]
                if property_name in class_def.properties:
                    lgr.print_error(f'{class_name}.xml: Duplicate property "{property_name}".', self)
                    continue

                type_name = TypeName.from_element(property)
                setter = property.get("setter") or None  # Use or None so '' gets turned into None.
                getter = property.get("getter") or None
                default_value = property.get("default") or None
                if default_value is not None:
                    default_value = f"``{default_value}``"
                overrides = property.get("overrides") or None

                property_def = PropertyDef(
                    property_name, type_name, setter, getter, property.text, default_value, overrides
                )
                property_def.deprecated = property.get("deprecated")
                property_def.experimental = property.get("experimental")
                class_def.properties[property_name] = property_def

        constructors = class_root.find("constructors")
        if constructors is not None:
            for constructor in constructors:
                assert constructor.tag == "constructor"

                method_name = constructor.attrib["name"]
                qualifiers = constructor.get("qualifiers")

                return_element = constructor.find("return")
                if return_element is not None:
                    return_type = TypeName.from_element(return_element)
                else:
                    return_type = TypeName("void")

                params = self.parse_params(constructor, "constructor")

                desc_element = constructor.find("description")
                method_desc = None
                if desc_element is not None:
                    method_desc = desc_element.text

                method_def = MethodDef(method_name, return_type, params, method_desc, qualifiers)
                method_def.definition_name = "constructor"
                method_def.deprecated = constructor.get("deprecated")
                method_def.experimental = constructor.get("experimental")
                if method_name not in class_def.constructors:
                    class_def.constructors[method_name] = []

                class_def.constructors[method_name].append(method_def)

        methods = class_root.find("methods")
        if methods is not None:
            for method in methods:
                assert method.tag == "method"

                method_name = method.attrib["name"]
                qualifiers = method.get("qualifiers")

                return_element = method.find("return")
                if return_element is not None:
                    return_type = TypeName.from_element(return_element)

                else:
                    return_type = TypeName("void")

                params = self.parse_params(method, "method")

                desc_element = method.find("description")
                method_desc = None
                if desc_element is not None:
                    method_desc = desc_element.text

                method_def = MethodDef(method_name, return_type, params, method_desc, qualifiers)
                method_def.deprecated = method.get("deprecated")
                method_def.experimental = method.get("experimental")
                if method_name not in class_def.methods:
                    class_def.methods[method_name] = []

                class_def.methods[method_name].append(method_def)

        operators = class_root.find("operators")
        if operators is not None:
            for operator in operators:
                assert operator.tag == "operator"

                method_name = operator.attrib["name"]
                qualifiers = operator.get("qualifiers")

                return_element = operator.find("return")
                if return_element is not None:
                    return_type = TypeName.from_element(return_element)

                else:
                    return_type = TypeName("void")

                params = self.parse_params(operator, "operator")

                desc_element = operator.find("description")
                method_desc = None
                if desc_element is not None:
                    method_desc = desc_element.text

                method_def = MethodDef(method_name, return_type, params, method_desc, qualifiers)
                method_def.definition_name = "operator"
                method_def.deprecated = operator.get("deprecated")
                method_def.experimental = operator.get("experimental")
                if method_name not in class_def.operators:
                    class_def.operators[method_name] = []

                class_def.operators[method_name].append(method_def)

        constants = class_root.find("constants")
        if constants is not None:
            for constant in constants:
                assert constant.tag == "constant"

                constant_name = constant.attrib["name"]
                value = constant.attrib["value"]
                enum = constant.get("enum")
                is_bitfield = constant.get("is_bitfield") == "true"
                constant_def = ConstantDef(constant_name, value, constant.text, is_bitfield)
                constant_def.deprecated = constant.get("deprecated")
                constant_def.experimental = constant.get("experimental")
                if enum is None:
                    if constant_name in class_def.constants:
                        lgr.print_error(f'{class_name}.xml: Duplicate constant "{constant_name}".', self)
                        continue

                    class_def.constants[constant_name] = constant_def

                else:
                    if enum in class_def.enums:
                        enum_def = class_def.enums[enum]

                    else:
                        enum_def = EnumDef(enum, TypeName("int", enum), is_bitfield)
                        class_def.enums[enum] = enum_def

                    enum_def.values[constant_name] = constant_def

        annotations = class_root.find("annotations")
        if annotations is not None:
            for annotation in annotations:
                assert annotation.tag == "annotation"

                annotation_name = annotation.attrib["name"]
                qualifiers = annotation.get("qualifiers")

                params = self.parse_params(annotation, "annotation")

                desc_element = annotation.find("description")
                annotation_desc = None
                if desc_element is not None:
                    annotation_desc = desc_element.text

                annotation_def = AnnotationDef(annotation_name, params, annotation_desc, qualifiers)
                if annotation_name not in class_def.annotations:
                    class_def.annotations[annotation_name] = []

                class_def.annotations[annotation_name].append(annotation_def)

        signals = class_root.find("signals")
        if signals is not None:
            for signal in signals:
                assert signal.tag == "signal"

                signal_name = signal.attrib["name"]

                if signal_name in class_def.signals:
                    lgr.print_error(f'{class_name}.xml: Duplicate signal "{signal_name}".', self)
                    continue

                params = self.parse_params(signal, "signal")

                desc_element = signal.find("description")
                signal_desc = None
                if desc_element is not None:
                    signal_desc = desc_element.text

                signal_def = SignalDef(signal_name, params, signal_desc)
                signal_def.deprecated = signal.get("deprecated")
                signal_def.experimental = signal.get("experimental")
                class_def.signals[signal_name] = signal_def

        theme_items = class_root.find("theme_items")
        if theme_items is not None:
            for theme_item in theme_items:
                assert theme_item.tag == "theme_item"

                theme_item_name = theme_item.attrib["name"]
                theme_item_data_name = theme_item.attrib["data_type"]
                theme_item_id = "{}_{}".format(theme_item_data_name, theme_item_name)
                if theme_item_id in class_def.theme_items:
                    lgr.print_error(
                        f'{class_name}.xml: Duplicate theme property "{theme_item_name}" of type "{theme_item_data_name}".',
                        self,
                    )
                    continue

                default_value = theme_item.get("default") or None
                if default_value is not None:
                    default_value = f"``{default_value}``"

                theme_item_def = ThemeItemDef(
                    theme_item_name,
                    TypeName.from_element(theme_item),
                    theme_item_data_name,
                    theme_item.text,
                    default_value,
                )
                class_def.theme_items[theme_item_name] = theme_item_def

        tutorials = class_root.find("tutorials")
        if tutorials is not None:
            for link in tutorials:
                assert link.tag == "link"

                if link.text is not None:
                    class_def.tutorials.append((link.text.strip(), link.get("title", "")))

        self.current_class = ""

    def parse_params(self, root: ET.Element, context: str) -> List["ParameterDef"]:
        param_elements = root.findall("param")
        params: Any = [None] * len(param_elements)

        for param_index, param_element in enumerate(param_elements):
            param_name = param_element.attrib["name"]
            index = int(param_element.attrib["index"])
            type_name = TypeName.from_element(param_element)
            default = param_element.get("default")

            if param_name.strip() == "" or param_name.startswith("_unnamed_arg"):
                lgr.print_error(
                    f'{self.current_class}.xml: Empty argument name in {context} "{root.attrib["name"]}" at position {param_index}.',
                    self,
                )

            params[index] = ParameterDef(param_name, type_name, default)

        cast: List[ParameterDef] = params

        return cast

    def sort_classes(self) -> None:
        self.classes = OrderedDict(sorted(self.classes.items(), key=lambda t: t[0].lower()))


class TypeName:
    def __init__(self, type_name: str, enum: Optional[str] = None, is_bitfield: bool = False) -> None:
        self.type_name = type_name
        self.enum = enum
        self.is_bitfield = is_bitfield

    def to_rst(self, state: State) -> str:
        if self.enum is not None:
            return make_enum(self.enum, self.is_bitfield, state)
        elif self.type_name == "void":
            return "|void|"
        else:
            return make_type(self.type_name, state)

    @classmethod
    def from_element(cls, element: ET.Element) -> "TypeName":
        return cls(element.attrib["type"], element.get("enum"), element.get("is_bitfield") == "true")


class DefinitionBase:
    def __init__(
        self,
        definition_name: str,
        name: str,
    ) -> None:
        self.definition_name = definition_name
        self.name = name
        self.deprecated: Optional[str] = None
        self.experimental: Optional[str] = None
        self.description: Optional[str] = None

    # Checks the description for an annotation, returns if it is there,
    # optionally replaces the annotation in the description with replace_text.
    def desc_annotation(self, ann_text, replace_text=None):
        exists = False
        if(self.description != None):
            exists = ann_text in self.description
            if(exists and replace_text != None):
                self.description = self.description.replace(ann_text, replace_text)
        return exists

    def is_description_empty(self):
        return self.description is None or self.description.strip() == ""




class PropertyDef(DefinitionBase):
    def __init__(
        self,
        name: str,
        type_name: TypeName,
        setter: Optional[str],
        getter: Optional[str],
        text: Optional[str],
        default_value: Optional[str],
        overrides: Optional[str],
    ) -> None:
        super().__init__("property", name)

        self.type_name = type_name
        self.setter = setter
        self.getter = getter
        self.text = "!! No longer used, use description !!"
        self.default_value = default_value
        self.overrides = overrides
        self.description = text

        self.ignore = self.desc_annotation("@ignore")


class ParameterDef(DefinitionBase):
    def __init__(self, name: str, type_name: TypeName, default_value: Optional[str]) -> None:
        super().__init__("parameter", name)

        self.type_name = type_name
        self.default_value = default_value


class SignalDef(DefinitionBase):
    def __init__(self, name: str, parameters: List[ParameterDef], description: Optional[str]) -> None:
        super().__init__("signal", name)

        self.parameters = parameters
        self.description = description
        self.ignore = self.desc_annotation("@ignore")


class AnnotationDef(DefinitionBase):
    def __init__(
        self,
        name: str,
        parameters: List[ParameterDef],
        description: Optional[str],
        qualifiers: Optional[str],
    ) -> None:
        super().__init__("annotation", name)

        self.parameters = parameters
        self.description = description
        self.qualifiers = qualifiers


class MethodDef(DefinitionBase):
    def __init__(
        self,
        name: str,
        return_type: TypeName,
        parameters: List[ParameterDef],
        description: Optional[str],
        qualifiers: Optional[str],
    ) -> None:
        super().__init__("method", name)

        self.return_type = return_type
        self.parameters = parameters
        self.description = description
        self.qualifiers = qualifiers
        self.internal = self.desc_annotation("@internal", "[b]Internal use only.[/b]")
        self.ignore = self.desc_annotation("@ignore", None)



class ConstantDef(DefinitionBase):
    def __init__(self, name: str, value: str, text: Optional[str], bitfield: bool) -> None:
        super().__init__("constant", name)

        self.value = value
        self.text = text
        self.is_bitfield = bitfield


class EnumDef(DefinitionBase):
    def __init__(self, name: str, type_name: TypeName, bitfield: bool) -> None:
        super().__init__("enum", name)

        self.type_name = type_name
        self.values: OrderedDict[str, ConstantDef] = OrderedDict()
        self.is_bitfield = bitfield


class ThemeItemDef(DefinitionBase):
    def __init__(
        self, name: str, type_name: TypeName, data_name: str, text: Optional[str], default_value: Optional[str]
    ) -> None:
        super().__init__("theme property", name)

        self.type_name = type_name
        self.data_name = data_name
        self.text = text
        self.default_value = default_value


class ClassDef(DefinitionBase):
    def __init__(self, name: str) -> None:
        super().__init__("class", name)

        self.class_group = "variant"
        self.editor_class = self._is_editor_class()

        self.constants: OrderedDict[str, ConstantDef] = OrderedDict()
        self.enums: OrderedDict[str, EnumDef] = OrderedDict()
        self.properties: OrderedDict[str, PropertyDef] = OrderedDict()
        self.constructors: OrderedDict[str, List[MethodDef]] = OrderedDict()
        self.methods: OrderedDict[str, List[MethodDef]] = OrderedDict()
        self.operators: OrderedDict[str, List[MethodDef]] = OrderedDict()
        self.signals: OrderedDict[str, SignalDef] = OrderedDict()
        self.annotations: OrderedDict[str, List[AnnotationDef]] = OrderedDict()
        self.theme_items: OrderedDict[str, ThemeItemDef] = OrderedDict()
        self.inherits: Optional[str] = None
        self.brief_description: Optional[str] = None
        self.description: Optional[str] = None
        self.tutorials: List[Tuple[str, str]] = []
        self.keywords: Optional[str] = None

        # Used to match the class with XML source for output filtering purposes.
        self.filepath: str = ""
        self.ignore_uncommented = False

    def _is_editor_class(self) -> bool:
        if self.name.startswith("Editor"):
            return True
        if self.name in EDITOR_CLASSES:
            return True

        return False

    def update_class_group(self, state: State) -> None:
        group_name = "variant"

        if self.name.startswith("@"):
            group_name = "global"
        elif self.inherits:
            inherits = self.inherits.strip()

            while inherits in state.classes:
                if inherits == "Node":
                    group_name = "node"
                    break
                if inherits == "Resource":
                    group_name = "resource"
                    break
                if inherits == "Object":
                    group_name = "object"
                    break

                inode = state.classes[inherits].inherits
                if inode:
                    inherits = inode.strip()
                else:
                    break

        self.class_group = group_name


    def _strip_private_props(self):
        to_delete = []
        for key in self.properties.keys():
            if(key.startswith("_") and self.properties[key].is_description_empty()):
                to_delete.append(key)

        for del_me in to_delete:
            del self.properties[del_me]


    def _strip_private_methods(self):
        to_delete = []
        for key in self.methods.keys():
            if(key.startswith("_") and self.methods[key][0].is_description_empty()):
                to_delete.append(key)

        for del_me in to_delete:
            del self.methods[del_me]


    def strip_privates(self):
        self.ignore_uncommented = self.desc_annotation("@ignore-uncommented", "")
        self._strip_private_props()
        self._strip_private_methods()


# Checks if code samples have both GDScript and C# variations.
# For simplicity we assume that a GDScript example is always present, and ignore contexts
# which don't necessarily need C# examples.
class ScriptLanguageParityCheck:
    def __init__(self) -> None:
        self.hit_map: OrderedDict[str, List[Tuple[DefinitionBase, str]]] = OrderedDict()
        self.hit_count = 0

    def add_hit(self, class_name: str, context: DefinitionBase, error: str, state: State) -> None:
        if class_name in ["@GDScript", "@GlobalScope"]:
            return  # We don't expect these contexts to have parity.

        class_def = state.classes[class_name]
        if class_def.class_group == "variant" and class_def.name != "Object":
            return  # Variant types are replaced with native types in C#, we don't expect parity.

        self.hit_count += 1

        if class_name not in self.hit_map:
            self.hit_map[class_name] = []

        self.hit_map[class_name].append((context, error))


def make_type(klass: str, state: State) -> str:
    if klass.find("*") != -1:  # Pointer, ignore
        return f"``{klass}``"

    link_type = klass
    is_array = False

    if link_type.endswith("[]"):  # Typed array, strip [] to link to contained type.
        link_type = link_type[:-2]
        is_array = True

    if link_type in state.classes:
        type_rst = f":ref:`{link_type}<class_{link_type}>`"
        if is_array:
            type_rst = f":ref:`Array<class_Array>`\\[{type_rst}\\]"
        return type_rst

    # print_error(f'{state.current_class}.xml: Unresolved type "{link_type}".', state)
    # type_rst = f"``{link_type}``"
    type_rst = bitwes.make_type_link(link_type)
    if is_array:
        type_rst = f":ref:`Array<class_Array>`\\[{type_rst}\\]"
    return type_rst


def make_enum(t: str, is_bitfield: bool, state: State) -> str:
    p = t.find(".")
    if p >= 0:
        c = t[0:p]
        e = t[p + 1 :]
        # Variant enums live in GlobalScope but still use periods.
        if c == "Variant":
            c = "@GlobalScope"
            e = "Variant." + e
    else:
        c = state.current_class
        e = t
        if c in state.classes and e not in state.classes[c].enums:
            c = "@GlobalScope"

    if c in state.classes and e in state.classes[c].enums:
        if is_bitfield:
            if not state.classes[c].enums[e].is_bitfield:
                lgr.print_error(f'{state.current_class}.xml: Enum "{t}" is not bitfield.', state)
            return f"|bitfield|\\[:ref:`{e}<enum_{c}_{e}>`\\]"
        else:
            return f":ref:`{e}<enum_{c}_{e}>`"

    # Don't fail for `Vector3.Axis`, as this enum is a special case which is expected not to be resolved.
    if f"{c}.{e}" != "Vector3.Axis":
        lgr.print_error(f'{state.current_class}.xml: Unresolved enum "{t}".', state)

    return t
