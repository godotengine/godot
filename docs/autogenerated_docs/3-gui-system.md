# GUI System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [core/doc_data.cpp](https://github.com/godotengine/godot/blob/4219ce91/core/doc_data.cpp)
- [core/doc_data.h](https://github.com/godotengine/godot/blob/4219ce91/core/doc_data.h)
- [doc/class.xsd](https://github.com/godotengine/godot/blob/4219ce91/doc/class.xsd)
- [doc/classes/AcceptDialog.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/AcceptDialog.xml)
- [doc/classes/BaseButton.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/BaseButton.xml)
- [doc/classes/Button.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/Button.xml)
- [doc/classes/ButtonGroup.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/ButtonGroup.xml)
- [doc/classes/CheckBox.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/CheckBox.xml)
- [doc/classes/CheckButton.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/CheckButton.xml)
- [doc/classes/CodeEdit.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/CodeEdit.xml)
- [doc/classes/ColorPickerButton.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/ColorPickerButton.xml)
- [doc/classes/ItemList.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/ItemList.xml)
- [doc/classes/Label.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/Label.xml)
- [doc/classes/LineEdit.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/LineEdit.xml)
- [doc/classes/LinkButton.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/LinkButton.xml)
- [doc/classes/MenuButton.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/MenuButton.xml)
- [doc/classes/OptionButton.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/OptionButton.xml)
- [doc/classes/Popup.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/Popup.xml)
- [doc/classes/PopupMenu.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/PopupMenu.xml)
- [doc/classes/RichTextLabel.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/RichTextLabel.xml)
- [doc/classes/TextEdit.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/TextEdit.xml)
- [doc/classes/TextLine.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/TextLine.xml)
- [doc/classes/TextParagraph.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/TextParagraph.xml)
- [doc/classes/Tree.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/Tree.xml)
- [doc/classes/TreeItem.xml](https://github.com/godotengine/godot/blob/4219ce91/doc/classes/TreeItem.xml)
- [editor/editor_log.cpp](https://github.com/godotengine/godot/blob/4219ce91/editor/editor_log.cpp)
- [editor/editor_log.h](https://github.com/godotengine/godot/blob/4219ce91/editor/editor_log.h)
- [misc/extension_api_validation/4.1-stable_4.2-stable.expected](https://github.com/godotengine/godot/blob/4219ce91/misc/extension_api_validation/4.1-stable_4.2-stable.expected)
- [misc/extension_api_validation/4.2-stable_4.3-stable.expected](https://github.com/godotengine/godot/blob/4219ce91/misc/extension_api_validation/4.2-stable_4.3-stable.expected)
- [scene/gui/base_button.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/base_button.cpp)
- [scene/gui/base_button.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/base_button.h)
- [scene/gui/button.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/button.cpp)
- [scene/gui/button.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/button.h)
- [scene/gui/check_box.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/check_box.cpp)
- [scene/gui/check_box.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/check_box.h)
- [scene/gui/check_button.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/check_button.cpp)
- [scene/gui/check_button.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/check_button.h)
- [scene/gui/code_edit.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/code_edit.cpp)
- [scene/gui/code_edit.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/code_edit.h)
- [scene/gui/dialogs.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/dialogs.cpp)
- [scene/gui/dialogs.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/dialogs.h)
- [scene/gui/item_list.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/item_list.cpp)
- [scene/gui/item_list.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/item_list.h)
- [scene/gui/label.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/label.cpp)
- [scene/gui/label.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/label.h)
- [scene/gui/line_edit.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/line_edit.cpp)
- [scene/gui/line_edit.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/line_edit.h)
- [scene/gui/link_button.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/link_button.cpp)
- [scene/gui/link_button.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/link_button.h)
- [scene/gui/menu_button.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/menu_button.cpp)
- [scene/gui/menu_button.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/menu_button.h)
- [scene/gui/option_button.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/option_button.cpp)
- [scene/gui/option_button.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/option_button.h)
- [scene/gui/popup.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/popup.cpp)
- [scene/gui/popup.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/popup.h)
- [scene/gui/popup_menu.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/popup_menu.cpp)
- [scene/gui/popup_menu.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/popup_menu.h)
- [scene/gui/rich_text_label.compat.inc](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.compat.inc)
- [scene/gui/rich_text_label.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.cpp)
- [scene/gui/rich_text_label.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.h)
- [scene/gui/text_edit.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.cpp)
- [scene/gui/text_edit.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.h)
- [scene/gui/tree.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/tree.cpp)
- [scene/gui/tree.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/tree.h)
- [scene/property_list_helper.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/property_list_helper.cpp)
- [scene/property_list_helper.h](https://github.com/godotengine/godot/blob/4219ce91/scene/property_list_helper.h)
- [scene/resources/text_line.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/resources/text_line.cpp)
- [scene/resources/text_line.h](https://github.com/godotengine/godot/blob/4219ce91/scene/resources/text_line.h)
- [scene/resources/text_paragraph.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/resources/text_paragraph.cpp)
- [scene/resources/text_paragraph.h](https://github.com/godotengine/godot/blob/4219ce91/scene/resources/text_paragraph.h)
- [tests/display_server_mock.h](https://github.com/godotengine/godot/blob/4219ce91/tests/display_server_mock.h)
- [tests/scene/test_code_edit.h](https://github.com/godotengine/godot/blob/4219ce91/tests/scene/test_code_edit.h)
- [tests/scene/test_text_edit.h](https://github.com/godotengine/godot/blob/4219ce91/tests/scene/test_text_edit.h)
- [tests/scene/test_viewport.h](https://github.com/godotengine/godot/blob/4219ce91/tests/scene/test_viewport.h)
- [tests/test_macros.h](https://github.com/godotengine/godot/blob/4219ce91/tests/test_macros.h)
- [tests/test_main.cpp](https://github.com/godotengine/godot/blob/4219ce91/tests/test_main.cpp)

</details>



This document covers Godot's comprehensive GUI framework for creating user interfaces in both games and the editor. The system provides a rich set of controls for text display and editing, hierarchical data presentation, and specialized interfaces.

For information about the Scene System that GUI controls integrate with, see [Scene System](#2). For details about Editor-specific GUI components, see [Editor Architecture](#5).

## Architecture Overview

Godot's GUI system is built around the `Control` class as the foundation for all user interface elements. The system provides both basic text display controls and advanced editing components with features like syntax highlighting, rich text formatting, and multi-caret editing.

### Core GUI System Architecture

```mermaid
graph TB
    Control["Control<br/>(Base UI Class)"]
    
    subgraph "Text Controls"
        Label["Label<br/>(Simple Text Display)"]
        LineEdit["LineEdit<br/>(Single Line Input)"]
        TextEdit["TextEdit<br/>(Multi-line Editor)"]
        CodeEdit["CodeEdit<br/>(Code Editor)"]
        RichTextLabel["RichTextLabel<br/>(Formatted Text)"]
    end
    
    subgraph "Data Display Controls"
        Tree["Tree<br/>(Hierarchical Data)"]
        ItemList["ItemList<br/>(Selectable Lists)"]
    end
    
    subgraph "Specialized Controls"
        GraphEdit["GraphEdit<br/>(Node Graph Editor)"]
        GraphNode["GraphNode<br/>(Graph Node Element)"]
    end
    
    subgraph "Supporting Classes"
        TreeItem["TreeItem<br/>(Tree Node Data)"]
        TextParagraph["TextParagraph<br/>(Text Layout)"]
        SyntaxHighlighter["SyntaxHighlighter<br/>(Code Highlighting)"]
    end
    
    Control --> Label
    Control --> LineEdit
    Control --> TextEdit
    Control --> RichTextLabel
    Control --> Tree
    Control --> ItemList
    Control --> GraphEdit
    Control --> GraphNode
    
    TextEdit --> CodeEdit
    Tree --> TreeItem
    TextEdit --> TextParagraph
    CodeEdit --> SyntaxHighlighter
    RichTextLabel --> TextParagraph
```

Sources: [scene/gui/text_edit.h:40](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.h#L40), [scene/gui/rich_text_label.h:43](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.h#L43), [scene/gui/tree.h:46](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/tree.h#L46), [scene/gui/line_edit.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/line_edit.h), [scene/gui/label.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/label.h), [scene/gui/code_edit.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/code_edit.h)

## Text-Based Controls

### TextEdit - Advanced Text Editor

`TextEdit` is the most sophisticated text editing control, supporting multi-line text editing with advanced features like multiple carets, syntax highlighting, and extensive keyboard shortcuts.

#### TextEdit Core Features

```mermaid
graph LR
    TextEdit["TextEdit"]
    
    subgraph "Text Management"
        TextClass["Text<br/>(Internal Text Storage)"]
        LineData["Line<br/>(Per-Line Data)"]
        TextParagraph["TextParagraph<br/>(Text Layout)"]
    end
    
    subgraph "Caret System" 
        MultiCaret["Multiple Carets<br/>(Vector&lt;Caret&gt;)"]
        CaretBlink["Caret Blinking<br/>(Timer-based)"]
        Selection["Selection<br/>(Per-Caret Selection)"]
    end
    
    subgraph "Input Handling"
        IME["IME Support<br/>(Input Method Editor)"]
        AltInput["Alt Code Input<br/>(Unicode/OEM/Windows)"]
        VirtualKeyboard["Virtual Keyboard<br/>(Mobile Support)"]
    end
    
    TextEdit --> TextClass
    TextEdit --> MultiCaret
    TextEdit --> IME
    TextClass --> LineData
    TextClass --> TextParagraph
    MultiCaret --> Selection
    MultiCaret --> CaretBlink
```

The `TextEdit` class implements comprehensive text editing through several key systems:

- **Text Storage**: The internal `Text` class manages line data and text formatting
- **Multi-Caret Support**: Allows multiple cursors for simultaneous editing operations
- **Input Method Support**: Handles international text input through IME systems
- **Accessibility**: Full screen reader support with element-based navigation

Sources: [scene/gui/text_edit.h:40-454](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.h#L40-L454), [scene/gui/text_edit.cpp:47-79](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.cpp#L47-L79)

### CodeEdit - Programming-Focused Editor

`CodeEdit` extends `TextEdit` with programming-specific features like syntax highlighting, code completion, and brace matching.

```mermaid
graph TB
    TextEdit["TextEdit<br/>(Base Editor)"]
    CodeEdit["CodeEdit<br/>(Code-Specific Editor)"]
    
    subgraph "Code Features"
        SyntaxHL["Syntax Highlighting<br/>(SyntaxHighlighter)"]
        CodeCompletion["Code Completion<br/>(Auto-complete)"]
        BraceMatching["Brace Matching<br/>(Bracket Highlighting)"]
        LineNumbers["Line Numbers<br/>(Gutter Display)"]
        Folding["Code Folding<br/>(Collapsible Blocks)"]
    end
    
    subgraph "Editor Integration"
        Gutters["Gutter System<br/>(Margins & Annotations)"]
        Bookmarks["Bookmarks<br/>(Navigation Markers)"]
        Breakpoints["Breakpoints<br/>(Debug Integration)"]
    end
    
    TextEdit --> CodeEdit
    CodeEdit --> SyntaxHL
    CodeEdit --> CodeCompletion
    CodeEdit --> BraceMatching
    CodeEdit --> LineNumbers
    CodeEdit --> Folding
    CodeEdit --> Gutters
    CodeEdit --> Bookmarks
    CodeEdit --> Breakpoints
```

Sources: [scene/gui/code_edit.h:40-78](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/code_edit.h#L40-L78), [scene/gui/code_edit.cpp:44-56](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/code_edit.cpp#L44-L56)

### RichTextLabel - Formatted Text Display

`RichTextLabel` provides rich text display with BBCode markup support, images, and interactive elements.

#### RichTextLabel Item System

```mermaid
graph TB
    RichTextLabel["RichTextLabel"]
    
    subgraph "Content Items"
        ItemFrame["ITEM_FRAME<br/>(Container)"]
        ItemText["ITEM_TEXT<br/>(Text Content)"]
        ItemImage["ITEM_IMAGE<br/>(Embedded Images)"]
        ItemTable["ITEM_TABLE<br/>(Table Layout)"]
        ItemList["ITEM_LIST<br/>(Bullet/Numbered Lists)"]
    end
    
    subgraph "Formatting Items"
        ItemFont["ITEM_FONT<br/>(Font Changes)"]
        ItemColor["ITEM_COLOR<br/>(Text Color)"]
        ItemBold["ITEM_BOLD<br/>(Bold Formatting)"]
        ItemItalic["ITEM_ITALIC<br/>(Italic Formatting)"]
        ItemUnderline["ITEM_UNDERLINE<br/>(Underline)"]
    end
    
    subgraph "Interactive Items"
        ItemMeta["ITEM_META<br/>(Clickable Links)"]
        ItemHint["ITEM_HINT<br/>(Tooltips)"]
        ItemButton["Custom Buttons"]
    end
    
    RichTextLabel --> ItemFrame
    RichTextLabel --> ItemText
    RichTextLabel --> ItemImage
    RichTextLabel --> ItemTable
    RichTextLabel --> ItemList
    RichTextLabel --> ItemFont
    RichTextLabel --> ItemColor
    RichTextLabel --> ItemMeta
```

Sources: [scene/gui/rich_text_label.h:70-101](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.h#L70-L101), [scene/gui/rich_text_label.cpp:49-68](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.cpp#L49-L68)

## Data Display Controls

### Tree - Hierarchical Data Display

The `Tree` control displays hierarchical data using `TreeItem` objects, supporting multiple columns, custom cell types, and interactive elements.

#### Tree Architecture

```mermaid
graph TB
    Tree["Tree<br/>(Main Control)"]
    TreeItem["TreeItem<br/>(Data Node)"]
    
    subgraph "Cell System"
        CellString["CELL_MODE_STRING<br/>(Text Display)"]
        CellCheck["CELL_MODE_CHECK<br/>(Checkbox)"]
        CellRange["CELL_MODE_RANGE<br/>(Slider/SpinBox)"]
        CellIcon["CELL_MODE_ICON<br/>(Icon Display)"]
        CellCustom["CELL_MODE_CUSTOM<br/>(Custom Drawing)"]
    end
    
    subgraph "Tree Features"
        MultiSelect["Multi-Selection<br/>(SELECT_MULTI)"]
        Columns["Multi-Column<br/>(Column System)"]
        Hierarchy["Parent-Child<br/>(Tree Structure)"]
        Editing["Inline Editing<br/>(Cell Editing)"]
    end
    
    subgraph "Cell Components"
        CellButtons["Cell Buttons<br/>(Action Buttons)"]
        CellText["Text Buffer<br/>(TextParagraph)"]
        CellIcon["Icon Texture<br/>(Texture2D)"]
    end
    
    Tree --> TreeItem
    TreeItem --> CellString
    TreeItem --> CellCheck
    TreeItem --> CellRange
    TreeItem --> CellIcon
    TreeItem --> CellCustom
    Tree --> MultiSelect
    Tree --> Columns
    Tree --> Hierarchy
    Tree --> Editing
    TreeItem --> CellButtons
    TreeItem --> CellText
    TreeItem --> CellIcon
```

Sources: [scene/gui/tree.h:46-57](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/tree.h#L46-L57), [scene/gui/tree.cpp:50-105](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/tree.cpp#L50-L105)

### ItemList - Selectable Item Lists

`ItemList` provides a vertical list of selectable items with support for icons, multi-selection, and custom styling.

```mermaid
graph LR
    ItemList["ItemList"]
    
    subgraph "Item System"
        Item["Item<br/>(List Entry)"]
        ItemIcon["Icon<br/>(Texture2D)"]
        ItemText["Text<br/>(String + TextParagraph)"]
        ItemTooltip["Tooltip<br/>(Hover Text)"]
        ItemMetadata["Metadata<br/>(Variant)"]
    end
    
    subgraph "Display Modes"
        IconMode["Icon Mode<br/>(ICON_MODE_TOP/LEFT)"]
        SelectMode["Select Mode<br/>(Single/Multi)"]
        TextMode["Text Handling<br/>(AutoTranslate)"]
    end
    
    ItemList --> Item
    Item --> ItemIcon
    Item --> ItemText
    Item --> ItemTooltip
    Item --> ItemMetadata
    ItemList --> IconMode
    ItemList --> SelectMode
    ItemList --> TextMode
```

Sources: [scene/gui/item_list.h:46-89](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/item_list.h#L46-L89), [scene/gui/item_list.cpp:37-72](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/item_list.cpp#L37-L72)

## Specialized Controls

### GraphEdit - Visual Node Editor

`GraphEdit` provides a sophisticated node-based visual editor used extensively in Godot's shader editor and other visual programming interfaces.

#### GraphEdit System Architecture

```mermaid
graph TB
    GraphEdit["GraphEdit<br/>(Main Graph Container)"]
    GraphNode["GraphNode<br/>(Individual Node)"]
    
    subgraph "Connection System"
        Connection["Connection<br/>(Node Links)"]
        Line2D["Line2D<br/>(Visual Connection)"]
        ShaderMaterial["ShaderMaterial<br/>(Connection Styling)"]
    end
    
    subgraph "Graph Features"
        Minimap["GraphEditMinimap<br/>(Overview)"]
        ViewPanner["ViewPanner<br/>(Pan & Zoom)"]
        GridSystem["Grid System<br/>(Snapping & Visual)"]
        Selection["Selection System<br/>(Multi-Select)"]
    end
    
    subgraph "Node Features"
        Ports["Input/Output Ports<br/>(Connection Points)"]
        Resizing["Node Resizing<br/>(User Interaction)"]
        NodeTitle["Title Bar<br/>(Node Header)"]
        NodeContent["Node Content<br/>(Custom Controls)"]
    end
    
    GraphEdit --> GraphNode
    GraphEdit --> Connection
    GraphEdit --> Minimap
    GraphEdit --> ViewPanner
    GraphEdit --> GridSystem
    GraphEdit --> Selection
    
    Connection --> Line2D
    Connection --> ShaderMaterial
    
    GraphNode --> Ports
    GraphNode --> Resizing
    GraphNode --> NodeTitle
    GraphNode --> NodeContent
```

Sources: [scene/gui/graph_edit.h:67-246](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/graph_edit.h#L67-L246), [scene/gui/graph_edit.cpp:216-306](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/graph_edit.cpp#L216-L306), [scene/gui/graph_node.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/graph_node.h), [scene/gui/graph_node.cpp](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/graph_node.cpp)

## Text Processing and Layout

All text-based GUI controls rely on Godot's text processing system for layout, rendering, and internationalization support.

### Text Processing Pipeline

```mermaid
graph LR
    subgraph "Input Text"
        RawText["Raw Text<br/>(String)"]
        BBCode["BBCode<br/>(Markup)"]
        Unicode["Unicode<br/>(UTF-8/UTF-16)"]
    end
    
    subgraph "Processing Layer"
        TextServer["TextServer<br/>(Platform Text API)"]
        TextParagraph["TextParagraph<br/>(Layout Engine)"]
        TextLine["TextLine<br/>(Single Line)"]
    end
    
    subgraph "Rendering Output"
        CanvasItem["CanvasItem<br/>(GPU Rendering)"]
        AccessibilityRID["Accessibility<br/>(Screen Reader)"]
        Font["Font Rendering<br/>(Glyph Rasterization)"]
    end
    
    RawText --> TextServer
    BBCode --> TextServer
    Unicode --> TextServer
    
    TextServer --> TextParagraph
    TextServer --> TextLine
    
    TextParagraph --> CanvasItem
    TextParagraph --> AccessibilityRID
    TextParagraph --> Font
```

Sources: [scene/gui/text_edit.cpp:51-66](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.cpp#L51-L66), [scene/gui/rich_text_label.cpp:268-365](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/rich_text_label.cpp#L268-L365), [scene/resources/text_paragraph.h](https://github.com/godotengine/godot/blob/4219ce91/scene/resources/text_paragraph.h)

## Integration with Scene System

GUI controls are fully integrated with Godot's scene system, inheriting from `Control` which provides layout, theming, and event handling capabilities.

### Control Base Class Integration

```mermaid
graph TB
    Node["Node<br/>(Scene Tree Base)"]
    CanvasItem["CanvasItem<br/>(2D Rendering)"]
    Control["Control<br/>(GUI Base)"]
    
    subgraph "GUI Controls"
        TextControls["Text Controls<br/>(TextEdit, Label, etc.)"]
        DataControls["Data Controls<br/>(Tree, ItemList)"]
        SpecialControls["Special Controls<br/>(GraphEdit, etc.)"]
    end
    
    subgraph "Control Features"
        Layout["Layout System<br/>(Anchors, Margins)"]
        Theming["Theme System<br/>(Styling)"]
        Input["Input Handling<br/>(Events, Focus)"]
        Accessibility["Accessibility<br/>(Screen Readers)"]
    end
    
    Node --> CanvasItem
    CanvasItem --> Control
    
    Control --> TextControls
    Control --> DataControls
    Control --> SpecialControls
    
    Control --> Layout
    Control --> Theming
    Control --> Input
    Control --> Accessibility
```

Sources: [scene/gui/control.h](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/control.h), [scene/gui/text_edit.h:40](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/text_edit.h#L40), [scene/gui/tree.h:46](https://github.com/godotengine/godot/blob/4219ce91/scene/gui/tree.h#L46)