#!/usr/bin/env python3
"""Add STRUCT cases to all DataType::Kind switch statements"""

import re
from pathlib import Path

files_to_update = [
    "modules/gdscript/gdscript_analyzer.cpp",
    "modules/gdscript/gdscript_compiler.cpp",
    "modules/gdscript/gdscript_byte_codegen.cpp",
]

# Pattern: case ENUM followed by case RESOLVING or case VARIANT
pattern = r'(case GDScriptParser::DataType::ENUM:.*?\n)([ \t]*case GDScriptParser::DataType::(VARIANT|RESOLVING|UNRESOLVED))'

replacement = r'\1\2\t\tcase GDScriptParser::DataType::STRUCT:\n\3'

for file_path in files_to_update:
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️  {file_path} not found")
        continue
    
    content = path.read_text(encoding='utf-8')
    original = content
    
    # Add STRUCT after ENUM in switch statements
    content = re.sub(
        r'(case GDScriptParser::DataType::ENUM:[^\n]*\n)([ \t]*)(case GDScriptParser::DataType::(VARIANT|RESOLVING|UNRESOLVED))',
        r'\1\2case GDScriptParser::DataType::STRUCT:\n\2\3',
        content,
        flags=re.MULTILINE
    )
    
    if content != original:
        path.write_text(content, encoding='utf-8', newline='\n')
        changes = content.count('STRUCT:') - original.count('STRUCT:')
        print(f"✅ {file_path}: Added {changes} STRUCT cases")
    else:
        print(f"ℹ️  {file_path}: No changes needed")

print("\n✨ Done!")
