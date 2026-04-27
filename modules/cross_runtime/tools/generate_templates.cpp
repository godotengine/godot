/*
This generates the needed templates, that is cs/Bridge.cs, Bridge.cpp, memory_layout.h, tests/<STRUCT NAME>memory_contract_tests.cpp
*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <cctype>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Field {
    std::string name;
    std::string type;
    float default_value;
};

struct Schema {
    std::string struct_name;
    int entity_count;
    int base_offset;
    int worker_ready_offset;
    std::vector<Field> fields;
};

static std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void write_file(const fs::path& path, const std::string& content) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write: " + path.string());
    out << content;
}

static bool valid_identifier(const std::string& s) {
    if (s.empty()) return false;
    if (!std::isalpha(static_cast<unsigned char>(s[0])) && s[0] != '_') return false;
    for (char c : s)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') return false;
    return true;
}

static std::string to_upper(const std::string& s) {
    std::string r = s;
    for (char& c : r) c = std::toupper(static_cast<unsigned char>(c));
    return r;
}

static Schema load_schema(const fs::path& path) {
    json j = json::parse(read_file(path));

    if (!j.contains("struct_name")) throw std::runtime_error("missing 'struct_name'");
    if (!j.contains("entity_count")) throw std::runtime_error("missing 'entity_count'");
    if (!j.contains("base_offset")) throw std::runtime_error("missing 'base_offset'");
    if (!j.contains("worker_ready_offset")) throw std::runtime_error("missing 'worker_ready_offset'");
    if (!j.contains("fields") || !j["fields"].is_array() || j["fields"].empty())
        throw std::runtime_error("'fields' must be a non-empty array");

    Schema s;
    s.struct_name = j["struct_name"].get<std::string>();
    s.entity_count = j["entity_count"].get<int>();
    s.base_offset = j["base_offset"].get<int>();
    s.worker_ready_offset = j["worker_ready_offset"].get<int>();

    if (!valid_identifier(s.struct_name))
        throw std::runtime_error("invalid struct_name: " + s.struct_name);
    if (s.entity_count <= 0)
        throw std::runtime_error("entity_count must be > 0");
    if (s.base_offset < 0)
        throw std::runtime_error("base_offset must be >= 0");
    if (s.worker_ready_offset < 0)
        throw std::runtime_error("worker_ready_offset must be >= 0");

    for (const auto& item : j["fields"]) {
        if (!item.contains("name") || !item.contains("type"))
            throw std::runtime_error("each field needs 'name' and 'type'");
        Field f;
        f.name = item["name"].get<std::string>();
        f.type = item["type"].get<std::string>();
        f.default_value = item.value("default", 0.0f);

        if (!valid_identifier(f.name))
            throw std::runtime_error("invalid field name: " + f.name);
        if (f.type != "f32")
            throw std::runtime_error("only f32 supported, got: " + f.type);

        s.fields.push_back(f);
    }
    return s;
}

static std::unordered_map<std::string, int> compute_offsets(const Schema& s) {
    std::unordered_map<std::string, int> off;
    int cur = 0;
    for (const auto& f : s.fields) {
        off[f.name] = cur;
        cur += 4;
    }
    return off;
}

// memory_layout.h
static std::string gen_memory_layout_h(const Schema& s) {
    auto off = compute_offsets(s);
    int stride = static_cast<int>(s.fields.size()) * 4;
    std::ostringstream out;
    out << "#pragma once\n\n";
    out << "#include <cstdint>\n\n";
    out << "// -------------------------------------------------------------------\n";
    out << "// Automatically generated from your schema. Do not edit directly.\n";
    out << "// -------------------------------------------------------------------\n\n";
    out << "constexpr std::uint32_t ENTITY_COUNT    = " << s.entity_count << "u;\n";
    out << "constexpr std::uint32_t ENTITIES_OFFSET = 0x" << std::hex << std::uppercase << s.base_offset << std::dec << "u;\n";
    out << "constexpr std::uint32_t ENTITY_STRIDE   = " << stride << "u;\n";
    out << "constexpr std::uint32_t WORKER_READY_OFFSET = 0x" << std::hex << std::uppercase << s.worker_ready_offset << std::dec << "u;\n\n";
    out << "struct " << s.struct_name << " {\n";
    for (const auto& f : s.fields) out << "    float " << f.name << ";\n";
    out << "};\n\n";
    out << "// Byte offsets for each field\n";
    for (const auto& f : s.fields) {
        out << "constexpr std::uint32_t ENTITY_FIELD_" << to_upper(f.name) << "_OFFSET = " << off.at(f.name) << "u;\n";
    }
    out << "\n// Field order:\n";
    for (size_t i = 0; i < s.fields.size(); ++i)
        out << "//   " << i << " : " << s.fields[i].name << "\n";
    out << "\n";
    return out.str();
}

// bridge.cpp
static std::string gen_bridge_cpp(const Schema& s) {
    std::ostringstream out;
    out << "#include \"memory_layout.h\"\n";
    out << "#include <emscripten.h>\n";
    out << "#include <cstdio>\n\n";
    out << "EM_JS(void, js_bridge_tick, (int offset, int count, float dt), {\n";
    out << "    if (typeof window.godot_js_bridge_tick === \"function\") {\n";
    out << "        window.godot_js_bridge_tick(offset, count, dt);\n";
    out << "    } else {\n";
    out << "        console.error(\"window.godot_js_bridge_tick not found\");\n";
    out << "    }\n";
    out << "});\n\n";
    out << "extern \"C\" {\n\n";
    out << "static inline " << s.struct_name << "* entities() {\n";
    out << "    return reinterpret_cast<" << s.struct_name << "*>(ENTITIES_OFFSET);\n";
    out << "}\n\n";
    out << "void init_entities() {\n";
    out << "    std::printf(\"[C++] init_entities called\\n\");\n";
    out << "    " << s.struct_name << "* arr = entities();\n";
    out << "    for (std::uint32_t i = 0; i < ENTITY_COUNT; ++i) {\n";
    for (const auto& f : s.fields) {
        out << "        arr[i]." << f.name << " = " << f.default_value << "f;\n";
    }
    out << "    }\n";
    out << "}\n\n";
    out << "void tick_entities(float dt) {\n";
    out << "    js_bridge_tick(ENTITIES_OFFSET, ENTITY_COUNT, dt);\n";
    out << "}\n\n";
    out << "} // extern \"C\"\n";
    return out.str();
}

// template.cs
static std::string gen_cs_template(const Schema& s) {
    auto off = compute_offsets(s);
    int stride = static_cast<int>(s.fields.size()) * 4;
    std::ostringstream out;

    out << "using System;\n";
    out << "using System.Threading.Tasks;\n";
    out << "using System.Runtime.InteropServices.JavaScript;\n\n";

    out << "public static partial class Host\n";
    out << "{\n";
    out << "    [JSImport(\"bulkRead\", \"interop\")] public static partial void BulkRead(int srcOffset, byte[] destArray, int length);\n";
    out << "    [JSImport(\"bulkWrite\", \"interop\")] public static partial void BulkWrite(byte[] srcArray, int destOffset, int length);\n\n";
    out << "    [JSExport]\n";
    out << "    public static async Task InitInterop()\n";
    out << "    {\n";
    out << "        await JSHost.ImportAsync(\"interop\", \"/cs/interop.js\");\n";
    out << "        Console.WriteLine(\"C# interop imported (bulk version)\");\n";
    out << "    }\n";
    out << "}\n\n";

    out << "public static class Contract\n";
    out << "{\n";
    out << "    public const int EntityCount = " << s.entity_count << ";\n";
    out << "    public const int EntitiesOffset = 0x" << std::hex << std::uppercase << s.base_offset << std::dec << ";\n";
    out << "    public const int EntityStride = " << stride << ";\n";
    out << "    public const int EntityFieldCount = " << s.fields.size() << ";\n";
    out << "    public const string StructName = \"" << s.struct_name << "\";\n\n";
    out << "    public static readonly string[] FieldNames = { ";
    for (size_t i = 0; i < s.fields.size(); ++i) {
        out << "\"" << s.fields[i].name << "\"";
        if (i + 1 < s.fields.size()) out << ", ";
    }
    out << " };\n\n";
    out << "    public static readonly int[] FieldOffsets = { ";
    for (size_t i = 0; i < s.fields.size(); ++i) {
        out << off.at(s.fields[i].name);
        if (i + 1 < s.fields.size()) out << ", ";
    }
    out << " };\n\n";
    for (const auto& f : s.fields) {
        out << "    public const int " << to_upper(f.name) << "Offset = " << off.at(f.name) << ";\n";
    }
    out << "\n";
    out << "    public static int GetFieldOffset(int fieldIndex) => fieldIndex switch\n    {\n";
    for (size_t i = 0; i < s.fields.size(); ++i) {
        out << "        " << i << " => " << to_upper(s.fields[i].name) << "Offset,\n";
    }
    out << "        _ => throw new ArgumentOutOfRangeException(nameof(fieldIndex)),\n    };\n\n";
    out << "    public static string GetFieldName(int fieldIndex) => fieldIndex switch\n    {\n";
    for (size_t i = 0; i < s.fields.size(); ++i) {
        out << "        " << i << " => \"" << s.fields[i].name << "\",\n";
    }
    out << "        _ => throw new ArgumentOutOfRangeException(nameof(fieldIndex)),\n    };\n\n";
    out << "    public static int GetFieldOffset(string fieldName) => fieldName switch\n    {\n";
    for (const auto& f : s.fields) {
        out << "        \"" << f.name << "\" => " << to_upper(f.name) << "Offset,\n";
    }
    out << "        _ => throw new ArgumentOutOfRangeException(nameof(fieldName)),\n    };\n\n";
    out << "    public static float ReadFieldF32(int entityOffset, int fieldIndex) =>\n";
    out << "        Host.ReadF32(entityOffset + GetFieldOffset(fieldIndex));\n\n";
    out << "    public static void WriteFieldF32(int entityOffset, int fieldIndex, float value) =>\n";
    out << "        Host.WriteF32(entityOffset + GetFieldOffset(fieldIndex), value);\n";
    out << "}\n\n";

    out << "public static partial class GameLogic\n";
    out << "{\n";
    out << "    [JSExport]\n";
    out << "    public static void UpdateEntities(int entityBaseOffset, int entityCount, float dtSeconds)\n";
    out << "    {\n";
    out << "        // User code goes here – use Contract and Host.\n";
    out << "        Console.WriteLine($\"[C#] UpdateEntities called: count={entityCount}, dt={dtSeconds:F6}\");\n";
    out << "    }\n";
    out << "}\n";

    return out.str();
}


// memory_contract_tests.cpp
static std::string gen_memory_contract_tests_cpp(const Schema& s) {
    auto off = compute_offsets(s);
    const int stride = static_cast<int>(s.fields.size()) * 4;

    std::ostringstream out;
    out << "#include <cassert>\n";
    out << "#include <cstddef>\n";
    out << "#include <cstdint>\n";
    out << "#include <cstring>\n";
    out << "#include <iostream>\n\n";
    out << "#include \"../memory_layout.h\"\n\n";

    out << "#define CHECK(expr) do { \\\n";
    out << "    if (!(expr)) { \\\n";
    out << "        std::cerr << \"CHECK failed: \" << #expr << \"\\n\"; \\\n";
    out << "        return 1; \\\n";
    out << "    } \\\n";
    out << "} while (0)\n\n";

    out << "static int test_layout_contract() {\n";
    out << "    static_assert(sizeof(float) == 4, \"schema assumes 32-bit float\");\n";
    out << "    static_assert(sizeof(" << s.struct_name << ") == ENTITY_STRIDE, \"struct size must match stride\");\n";
    out << "    static_assert(offsetof(" << s.struct_name << ", " << s.fields[0].name << ") == ENTITY_FIELD_" << to_upper(s.fields[0].name) << "_OFFSET, \"field offset mismatch\");\n";
    for (size_t i = 1; i < s.fields.size(); ++i) {
        out << "    static_assert(offsetof(" << s.struct_name << ", " << s.fields[i].name << ") == ENTITY_FIELD_" << to_upper(s.fields[i].name) << "_OFFSET, \"field offset mismatch\");\n";
    }
    out << "    static_assert(ENTITIES_OFFSET % alignof(" << s.struct_name << ") == 0, \"entities must be aligned\");\n";
    out << "    static_assert(WORKER_READY_OFFSET % alignof(std::uint32_t) == 0, \"worker flag must be 32-bit aligned\");\n";
    out << "    static_assert(WORKER_READY_OFFSET >= ENTITIES_OFFSET + ENTITY_COUNT * ENTITY_STRIDE, \"worker flag overlaps entity storage\");\n";
    out << "    return 0;\n";
    out << "}\n\n";

    out << "static int test_field_roundtrip() {\n";
    out << "    alignas(" << s.struct_name << ") std::uint8_t raw[sizeof(" << s.struct_name << ")] = {};\n";
    out << "    auto* e = reinterpret_cast<" << s.struct_name << "*>(raw);\n";
    for (const auto& f : s.fields) {
        if (f.name == s.fields[0].name) {
            out << "    e->" << f.name << " = 1.25f;\n";
        } else if (f.name == s.fields[1].name) {
            out << "    e->" << f.name << " = 2.5f;\n";
        } else if (f.name == s.fields[2].name) {
            out << "    e->" << f.name << " = 3.75f;\n";
        } else if (f.name == s.fields[3].name) {
            out << "    e->" << f.name << " = 4.5f;\n";
        } else {
            out << "    e->" << f.name << " = 0.0f;\n";
        }
    }
    out << "\n";
    out << "    " << s.struct_name << " copy{};\n";
    out << "    std::memcpy(&copy, raw, sizeof(copy));\n";
    for (size_t i = 0; i < s.fields.size(); ++i) {
        out << "    CHECK(copy." << s.fields[i].name << " == e->" << s.fields[i].name << ");\n";
    }
    out << "    return 0;\n";
    out << "}\n\n";

    out << "static int test_index_math() {\n";
    out << "    for (std::uint32_t i = 0; i < ENTITY_COUNT; ++i) {\n";
    out << "        const std::uint32_t expected = ENTITIES_OFFSET + i * ENTITY_STRIDE;\n";
    out << "        const std::uint32_t actual = ENTITIES_OFFSET + i * static_cast<std::uint32_t>(sizeof(" << s.struct_name << "));\n";
    out << "        CHECK(expected == actual);\n";
    out << "    }\n";
    out << "    return 0;\n";
    out << "}\n\n";

    out << "int main() {\n";
    out << "    if (int r = test_layout_contract()) return r;\n";
    out << "    if (int r = test_field_roundtrip()) return r;\n";
    out << "    if (int r = test_index_math()) return r;\n";
    out << "    std::cout << \"memory contract tests passed\\n\";\n";
    out << "    return 0;\n";
    out << "}\n";

    return out.str();
}

int main() {
    try {
        Schema s = load_schema("schema.json");
        write_file(fs::path("") / ("memory_layout.h"), gen_memory_layout_h(s));
        write_file(fs::path("") / ("Bridge.cpp"), gen_bridge_cpp(s));
         write_file(fs::path("cs") / ("Bridge.cs"), gen_cs_template(s));
        write_file(fs::path("tests") / (s.struct_name + "_memory_contract_tests.cpp"), gen_memory_contract_tests_cpp(s));
        std::cout << "Generated All files\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
