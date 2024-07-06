// Copyright (c) 2014-2024 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and/or associated documentation files (the "Materials"),
// to deal in the Materials without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Materials, and to permit persons to whom the
// Materials are furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS KHRONOS
// STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
// HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
// IN THE MATERIALS.

//
// Print headers for SPIR-V in several languages.
//
// To change the header information, change the C++-built database in doc.*.
//
// Then, use "spriv -h <language>" - e.g, spriv.{h,hpp,lua,py,etc}:
// replace the auto-generated header, or "spirv -H" to generate all
// supported language headers to predefined names in the current directory.
//

#include <string>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <cctype>
#include <vector>
#include <utility>
#include <set>

#include "jsoncpp/dist/json/json.h"

#include "header.h"
#include "jsonToSpirv.h"

// snprintf and _snprintf are not quite the same, but close enough
// for our use.
#ifdef _MSC_VER
#pragma warning(disable:4996)
#define snprintf _snprintf
#endif

// This file converts SPIR-V definitions to an internal JSON
// representation, and then generates language specific
// data from that single internal form.

// Initially, the internal form is created from C++ data,
// though this can be changed to a JSON master in time.

namespace {
    class TPrinter {
    protected:
        TPrinter();

        static const int         DocMagicNumber = 0x07230203;
        static const int         DocVersion     = 0x00010600;
        static const int         DocRevision    = 1;
        #define DocRevisionString                "1"
        static const std::string DocCopyright;
        static const std::string DocComment1;
        static const std::string DocComment2;

        enum enumStyle_t {
            enumNoMask,
            enumCount,
            enumShift,
            enumMask,
            enumHex,
        };

        static std::string styleStr(enumStyle_t s) {
            return s == enumShift ? "Shift" :
                   s == enumMask  ? "Mask"  : "";
        }

        friend std::ostream& operator<<(std::ostream&, const TPrinter&);

        virtual void printAll(std::ostream&)      const;
        virtual void printComments(std::ostream&) const;
        virtual void printPrologue(std::ostream&) const { }
        virtual void printDefs(std::ostream&)     const;
        virtual void printEpilogue(std::ostream&) const { }
        virtual void printMeta(std::ostream&)     const;
        virtual void printTypes(std::ostream&)    const { }
        virtual void printUtility(std::ostream&)     const { };

        virtual std::string escapeComment(const std::string& s) const;

        // Default printComments() uses these comment strings
        virtual std::string commentBeg() const            { return ""; }
        virtual std::string commentEnd(bool isLast) const { return ""; }
        virtual std::string commentBOL() const            { return ""; }
        virtual std::string commentEOL(bool isLast) const { return ""; }

        typedef std::pair<unsigned, std::string> valpair_t;

        // for printing enum values
        virtual std::string enumBeg(const std::string&, enumStyle_t) const { return ""; }
        virtual std::string enumEnd(const std::string&, enumStyle_t, bool isLast = false) const {
            return "";
        }
        virtual std::string enumFmt(const std::string&, const valpair_t&,
                                    enumStyle_t, bool isLast = false) const {
            return "";
        }
        virtual std::string maxEnumFmt(const std::string& s, const valpair_t& v,
                               enumStyle_t style) const {
            return enumFmt(s, v, style, true);
        }

        virtual std::string fmtConstInt(unsigned val, const std::string& name,
                                        const char* fmt, bool isLast = false) const {
            return "";
        }

        std::vector<valpair_t> getSortedVals(const Json::Value&) const;

        virtual std::string indent(int count = 1) const {
            return std::string(count * 4, ' ');   // default indent level = 4
        }

        static std::string fmtNum(const char* fmt, unsigned val) {
            char buff[16]; // ample for 8 hex digits + 0x
            snprintf(buff, sizeof(buff), fmt, val);
            buff[sizeof(buff)-1] = '\0';  // MSVC doesn't promise null termination
            return buff;
        }

        static std::string fmtStyleVal(unsigned v, enumStyle_t style);

        // If the enum value name would start with a sigit, prepend the enum name.
        // E.g, "3D" -> "Dim3D".
        static std::string prependIfDigit(const std::string& ename, const std::string& vname) {
            return (std::isdigit(vname[0]) ? ename : std::string("")) + vname;
        }

        void addComment(Json::Value& node, const std::string& str);

        Json::Value spvRoot; // JSON SPIR-V data
    };

    // Format value as mask or value
    std::string TPrinter::fmtStyleVal(unsigned v, enumStyle_t style)
    {
        switch (style) {
        case enumMask:
            return fmtNum("0x%08x", 1<<v);
        case enumHex:
            return fmtNum("0x%08x", v);
        default:
            return std::to_string(v);
        }
    }

    const std::string TPrinter::DocCopyright =
R"(Copyright (c) 2014-2024 The Khronos Group Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and/or associated documentation files (the "Materials"),
to deal in the Materials without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Materials, and to permit persons to whom the
Materials are furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Materials.

MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS KHRONOS
STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/ 

THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
IN THE MATERIALS.
)";

    const std::string TPrinter::DocComment1 =
        "This header is automatically generated by the same tool that creates\n"
        "the Binary Section of the SPIR-V specification.\n";

    const std::string TPrinter::DocComment2 =
        "Enumeration tokens for SPIR-V, in various styles:\n"
        "  C, C++, C++11, JSON, Lua, Python, C#, D, Beef\n"
        "\n"
        "- C will have tokens with a \"Spv\" prefix, e.g.: SpvSourceLanguageGLSL\n"
        "- C++ will have tokens in the \"spv\" name space, e.g.: spv::SourceLanguageGLSL\n"
        "- C++11 will use enum classes in the spv namespace, e.g.: spv::SourceLanguage::GLSL\n"
        "- Lua will use tables, e.g.: spv.SourceLanguage.GLSL\n"
        "- Python will use dictionaries, e.g.: spv['SourceLanguage']['GLSL']\n"
        "- C# will use enum classes in the Specification class located in the \"Spv\" namespace,\n"
        "    e.g.: Spv.Specification.SourceLanguage.GLSL\n"
        "- D will have tokens under the \"spv\" module, e.g: spv.SourceLanguage.GLSL\n"
        "- Beef will use enum classes in the Specification class located in the \"Spv\" namespace,\n"
        "    e.g.: Spv.Specification.SourceLanguage.GLSL\n"
        "\n"
        "Some tokens act like mask values, which can be OR'd together,\n"
        "while others are mutually exclusive.  The mask-like ones have\n"
        "\"Mask\" in their name, and a parallel enum that has the shift\n"
        "amount (1 << x) for each corresponding enumerant.\n";

    // Construct
    TPrinter::TPrinter()
    {
        Json::Value& meta            = spvRoot["spv"]["meta"];
        Json::Value& enums           = spvRoot["spv"]["enum"];

        meta["MagicNumber"]          = DocMagicNumber;
        meta["Version"]              = DocVersion;
        meta["Revision"]             = DocRevision;
        meta["OpCodeMask"]           = 0xffff;
        meta["WordCountShift"]       = 16;

        int commentId = 0;
        addComment(meta["Comment"][commentId++], DocCopyright);
        addComment(meta["Comment"][commentId++], DocComment1);
        addComment(meta["Comment"][commentId++], DocComment2);

        for (int e = spv::OperandSource; e < spv::OperandOpcode; ++e) {
            auto& enumSet =  spv::OperandClassParams[e];
            const bool        mask     = enumSet.bitmask;
            const std::string enumName = enumSet.codeName;

            for (auto& enumRow : enumSet) {
                std::string name = enumRow.name;
                enums[e - spv::OperandSource]["Values"][name] = enumRow.value;
            }

            enums[e - spv::OperandSource]["Type"] = mask ? "Bit" : "Value";
            enums[e - spv::OperandSource]["Name"] = enumName;
        }

          // Instructions are in their own different table
        {
            auto& entry = enums[spv::OperandOpcode - spv::OperandSource];
            for (auto& enumRow : spv::InstructionDesc) {
                std::string name = enumRow.name;
                entry["Values"][name] = enumRow.value;
            }
            entry["Type"] = "Value";
            entry["Name"] = "Op";
        }
    }

    // Create comment
    void TPrinter::addComment(Json::Value& node, const std::string& str)
    {
        std::istringstream cstream(str);
        std::string        cline;

        int line = 0;
        while (std::getline(cstream, cline))  // fmt each line
            node[line++] = cline;
    }


    // Return a list of values sorted by enum value.  The std::vector
    // returned by value is okay in c++11 due to move semantics.
    std::vector<TPrinter::valpair_t>
    TPrinter::getSortedVals(const Json::Value& p) const
    {
        std::vector<valpair_t> values;

        for (auto e = p.begin(); e != p.end(); ++e)
            values.push_back(valpair_t(e->asUInt(), e.name()));

        // Use a stable sort because we might have aliases, e.g.
        // SubgropuBallot (might be in future core) vs. SubgroupBallotKHR.
        std::stable_sort(values.begin(), values.end());

        return values;
    }

    // Escape comment characters if needed
    std::string TPrinter::escapeComment(const std::string& s) const { return s; }

    // Format comments in language specific way
    void TPrinter::printComments(std::ostream& out) const
    {
        const int commentCount = spvRoot["spv"]["meta"]["Comment"].size();
        int commentNum = 0;

        for (const auto& comment : spvRoot["spv"]["meta"]["Comment"]) {
            out << commentBeg();

            for (int line = 0; line < int(comment.size()); ++line)
                out << commentBOL() << escapeComment(comment[line].asString()) <<
                    commentEOL((line+1) == comment.size()) << std::endl;

            out << commentEnd(++commentNum == commentCount) << std::endl;
        }
    }

    // Format header metadata
    void TPrinter::printMeta(std::ostream& out) const
    {
        const Json::Value& meta = spvRoot["spv"]["meta"];

        const auto print = [&](const char* name, const char* fmt, bool isLast) {
            out << fmtConstInt(meta[name].asUInt(), name, fmt, isLast);
        };

        print("MagicNumber",    "0x%08lx", false);
        print("Version",        "0x%08lx", false);
        print("Revision",       "%d",      false);
        print("OpCodeMask",     "0x%04x",  false);
        print("WordCountShift", "%d",      true);
    }

    // Format value definitions in language specific way
    void TPrinter::printDefs(std::ostream& out) const
    {
        const Json::Value& enums = spvRoot["spv"]["enum"];

        for (auto opClass = enums.begin(); opClass != enums.end(); ++opClass) {
            const bool isMask   = (*opClass)["Type"].asString() == "Bit";
            const auto opName   = (*opClass)["Name"].asString();
            const auto opPrefix = opName == "Op" ? "" : opName;

            for (enumStyle_t style = (isMask ? enumShift : enumCount);
                 style <= (isMask ? enumMask : enumCount); style = enumStyle_t(int(style)+1)) {

                out << enumBeg(opName, style);

                if (style == enumMask)
                    out << enumFmt(opPrefix, valpair_t(0, "MaskNone"), enumNoMask);

                const auto sorted = getSortedVals((*opClass)["Values"]);

                std::string maxEnum = maxEnumFmt(opName, valpair_t(0x7FFFFFFF, "Max"), enumHex);

                bool printMax = (style != enumMask && maxEnum.size() > 0);

                for (const auto& v : sorted)
                    out << enumFmt(opPrefix, v, style, !printMax && v.second == sorted.back().second);

                if (printMax)
                    out << maxEnum;

                auto nextOpClass = opClass;
                out << enumEnd(opName, style, ++nextOpClass == enums.end());
            }
        }
    }

    void TPrinter::printAll(std::ostream& out) const
    {
        printComments(out);
        printPrologue(out);
        printTypes(out);
        printMeta(out);
        printDefs(out);
        printUtility(out);
        printEpilogue(out);
    }

    // Stream entire header to output
    std::ostream& operator<<(std::ostream& out, const TPrinter &p)
    {
        p.printAll(out);
        return out;
    }

    // JSON printer.  Rather than use the default printer, we supply our own so
    // we can control the printing order within various containers.
    class TPrinterJSON final : public TPrinter {
    private:
        void printPrologue(std::ostream& out) const override { out << "{\n" + indent() + "\"spv\":\n" + indent() + "{\n"; }
        void printEpilogue(std::ostream& out) const override { out << indent() + "}\n}\n"; }

        std::string escapeComment(const std::string& s) const override {
            std::string newStr;
            for (auto c : s) {
                if (c == '"') {
                    newStr += '\\';
                    newStr += c;
                } else {
                    newStr += c;
                }
            }
            return newStr;
        }

        std::string fmtConstInt(unsigned val, const std::string& name,
                                const char* fmt, bool isLast) const override {
            return indent(3) + '"' + name + "\": " + fmtNum("%d", val) + (isLast ? "\n" : ",\n");
        }

        void printMeta(std::ostream& out) const override
        {
            out << indent(2) + "\"meta\":\n" + indent(2) + "{\n";
            printComments(out);
            TPrinter::printMeta(out);
            out << indent(2) + "},\n";
        }

        std::string commentBeg() const override            { return indent(4) + "[\n"; }
        std::string commentEnd(bool isLast) const override { return indent(4) + (isLast ? "]" : "],"); }
        std::string commentBOL() const override            { return indent(5) + '"'; }
        std::string commentEOL(bool isLast) const override { return (isLast ? "\"" : "\","); }

        void printComments(std::ostream& out) const override
        {
            out << indent(3) + "\"Comment\":\n" + indent(3) + "[\n";
            TPrinter::printComments(out);
            out << indent(3) + "],\n";
        }

        void printDefs(std::ostream& out) const override
        {
            out << indent(2) + "\"enum\":\n" + indent(2) + "[\n";
            TPrinter::printDefs(out);
            out << indent(2) + "]\n";
        }

        void printAll(std::ostream& out) const override
        {
            printPrologue(out);
            printMeta(out);
            printDefs(out);
            printEpilogue(out);
        }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            if (style == enumMask)
                return "";
            return indent(3) + "{\n" +
                indent(4) + "\"Name\": \"" + s + "\",\n" +
                indent(4) + "\"Type\": " + (style == enumShift ? "\"Bit\"" : "\"Value\"") + ",\n" +
                indent(4) + "\"Values\":\n" +
                indent(4) + "{\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            if (style == enumMask)
                return "";
            return indent(4) + "}\n" +
                   indent(3) + "}" + (isLast ? "" : ",") + "\n";
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            if (style == enumMask || style == enumNoMask)
                return "";
            return indent(5) + '"' + prependIfDigit(s, v.second) + "\": " + fmtNum("%d", v.first) +
                (isLast ? "\n" : ",\n");
        }
        std::string maxEnumFmt(const std::string& s, const valpair_t& v,
                               enumStyle_t style) const override {
            return "";
        }
    };

    // base for C and C++
    class TPrinterCBase : public TPrinter {
    protected:
        virtual void printPrologue(std::ostream& out) const override {
            out << "#ifndef spirv_" << headerGuardSuffix() << std::endl
                << "#define spirv_" << headerGuardSuffix() << std::endl
                << std::endl;
        }

        void printMeta(std::ostream& out) const override {
            out << "#define SPV_VERSION 0x" << std::hex << DocVersion << std::dec << "\n";
            out << "#define SPV_REVISION " << DocRevision << "\n";
            out << "\n";

            return TPrinter::printMeta(out);
        }

        virtual void printEpilogue(std::ostream& out) const override {
            out << "#endif" << std::endl;
        }

        virtual void printTypes(std::ostream& out) const override {
            out << "typedef unsigned int " << pre() << "Id;\n\n";
        }

        virtual std::string fmtConstInt(unsigned val, const std::string& name,
                                        const char* fmt, bool isLast) const override
        {
            return std::string("static const unsigned int ") + pre() + name +
                " = " + fmtNum(fmt, val) + (isLast ? ";\n\n" : ";\n");
        }

        virtual std::string pre() const { return ""; } // C name prefix
        virtual std::string headerGuardSuffix() const = 0;

        virtual std::string fmtEnumUse(const std::string& opPrefix, const std::string& name) const { return pre() + name; }

        virtual void printUtility(std::ostream& out) const override
        {
            out << "#ifdef SPV_ENABLE_UTILITY_CODE" << std::endl;
            out << "#ifndef __cplusplus" << std::endl;
            out << "#include <stdbool.h>" << std::endl;
            out << "#endif" << std::endl;

            printHasResultType(out);
            printStringFunctions(out);

            out << "#endif /* SPV_ENABLE_UTILITY_CODE */" << std::endl << std::endl;
        }

        void printHasResultType(std::ostream& out) const {
            const Json::Value& enums = spvRoot["spv"]["enum"];

            std::set<unsigned> seenValues;

            for (auto opClass = enums.begin(); opClass != enums.end(); ++opClass) {
                const auto opName   = (*opClass)["Name"].asString();
                if (opName != "Op") {
                    continue;
                }


                out << "inline void " << pre() << "HasResultAndType(" << pre() << opName << " opcode, bool *hasResult, bool *hasResultType) {" << std::endl;
                out << "    *hasResult = *hasResultType = false;" << std::endl;
                out << "    switch (opcode) {" << std::endl;
                out << "    default: /* unknown opcode */ break;" << std::endl;

                for (auto& inst : spv::InstructionDesc) {

                    // Filter out duplicate enum values, which would break the switch statement.
                    // These are probably just extension enums promoted to core.
                    if (seenValues.find(inst.value) != seenValues.end()) {
                        continue;
                    }
                    seenValues.insert(inst.value);

                    std::string name = inst.name;
                    out << "    case " << fmtEnumUse("Op", name) << ": *hasResult = " << (inst.hasResult() ? "true" : "false") << "; *hasResultType = " << (inst.hasType() ? "true" : "false") << "; break;" << std::endl;
                }

                out << "    }" << std::endl;
                out << "}" << std::endl;
            }
        }

        void printStringFunctions(std::ostream& out) const {
            const Json::Value& enums = spvRoot["spv"]["enum"];

            for (auto it = enums.begin(); it != enums.end(); ++it) {
                const auto type   = (*it)["Type"].asString();
                // Skip bitmasks
                if (type == "Bit") {
                    continue;
                }
                const auto name   = (*it)["Name"].asString();
                const auto sorted = getSortedVals((*it)["Values"]);

                std::set<unsigned> seenValues;
                std::string fullName = pre() + name;

                out << "inline const char* " << fullName << "ToString(" << fullName << " value) {" << std::endl;
                out << "    switch (value) {" << std::endl;
                for (const auto& v : sorted) {
                    // Filter out duplicate enum values, which would break the switch statement.
                    // These are probably just extension enums promoted to core.
                    if (seenValues.count(v.first)) {
                        continue;
                    }
                    seenValues.insert(v.first);

                    std::string label{name + v.second};
                    if (name == "Op") {
                        label = v.second;
                    }
                    out << "    " << "case " << pre() << label << ": return " << "\"" << v.second << "\";" << std::endl;
                }
                out << "    default: return \"Unknown\";" << std::endl;
                out << "    }" << std::endl;
                out << "}" << std::endl << std::endl;
            }
        }
    };

    // C printer
    class TPrinterC final : public TPrinterCBase {
    private:
        std::string commentBeg() const override            { return "/*\n"; }
        std::string commentEnd(bool isLast) const override { return "*/\n"; }
        std::string commentBOL() const override            { return "** ";  }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return std::string("typedef enum ") + pre() + s + styleStr(style) + "_ {\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return "} " + pre() + s + styleStr(style) + ";\n\n";
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            return indent() + pre() + s + v.second + styleStr(style) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        std::string pre() const override { return "Spv"; } // C name prefix
        std::string headerGuardSuffix() const override { return "H"; }
    };

    // C++ printer
    class TPrinterCPP : public TPrinterCBase {
    protected:
        void printMaskOperators(std::ostream& out, const std::string& specifiers) const {
            const Json::Value& enums = spvRoot["spv"]["enum"];

            out << "// Overload bitwise operators for mask bit combining\n\n";

            for (auto opClass = enums.begin(); opClass != enums.end(); ++opClass) {
                const bool isMask   = (*opClass)["Type"].asString() == "Bit";
                const auto opName   = (*opClass)["Name"].asString();

                if (isMask) {
                    const auto typeName = opName + styleStr(enumMask);

                    // Overload operator|
                    out << specifiers << " " << typeName << " operator|(" << typeName << " a, " << typeName << " b) { return " <<
                        typeName << "(unsigned(a) | unsigned(b)); }\n";
                    // Overload operator&
                    out << specifiers << " " << typeName << " operator&(" << typeName << " a, " << typeName << " b) { return " <<
                        typeName << "(unsigned(a) & unsigned(b)); }\n";
                    // Overload operator^
                    out << specifiers << " " << typeName << " operator^(" << typeName << " a, " << typeName << " b) { return " <<
                        typeName << "(unsigned(a) ^ unsigned(b)); }\n";
                    // Overload operator~
                    out << specifiers << " " << typeName << " operator~(" << typeName << " a) { return " <<
                        typeName << "(~unsigned(a)); }\n";
                }
            }
        }
    private:
        void printPrologue(std::ostream& out) const override {
            TPrinterCBase::printPrologue(out);
            out << "namespace spv {\n\n";
        }

        void printEpilogue(std::ostream& out) const override {
            printMaskOperators(out, "inline");
            out << "\n}  // end namespace spv\n\n";
            out << "#endif  // #ifndef spirv_" << headerGuardSuffix() << std::endl;
        }

        std::string commentBOL() const override { return "// "; }


        virtual std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return std::string("enum ") + s + styleStr(style) + " {\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return "};\n\n";
        }

        virtual std::string enumFmt(const std::string& s, const valpair_t& v,
                                    enumStyle_t style, bool isLast) const override {
            return indent() + s + v.second + styleStr(style) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        // The C++ and C++11 headers define types with the same name. So they
        // should use the same header guard.
        std::string headerGuardSuffix() const override { return "HPP"; }

        std::string operators;
    };

    // C++11 printer (uses enum classes)
    class TPrinterCPP11 final : public TPrinterCPP {
    private:
        void printEpilogue(std::ostream& out) const override {
            printMaskOperators(out, "constexpr");
            out << "\n}  // end namespace spv\n\n";
            out << "#endif  // #ifndef spirv_" << headerGuardSuffix() << std::endl;
        }
        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return std::string("enum class ") + s + styleStr(style) + " : unsigned {\n";
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            return indent() + prependIfDigit(s, v.second) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        // Add type prefix for scoped enum
        virtual std::string fmtEnumUse(const std::string& opPrefix, const std::string& name) const override { return opPrefix + "::" + name; }

        std::string headerGuardSuffix() const override { return "HPP"; }
    };

    // LUA printer
    class TPrinterLua final : public TPrinter {
    private:
        void printPrologue(std::ostream& out) const override { out << "spv = {\n"; }

        void printEpilogue(std::ostream& out) const override { out << "}\n"; }

        std::string commentBOL() const override { return "-- "; }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return indent() + s + styleStr(style) + " = {\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return indent() + "},\n\n";
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            return indent(2) + prependIfDigit(s, v.second) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        virtual std::string fmtConstInt(unsigned val, const std::string& name,
                                        const char* fmt, bool isLast) const override
        {
            return indent() + name + " = " + fmtNum(fmt, val) + (isLast ? ",\n\n" : ",\n");
        }
    };

    // Python printer
    class TPrinterPython final : public TPrinter {
    private:
        void printPrologue(std::ostream& out) const override { out << "spv = {\n"; }

        void printEpilogue(std::ostream& out) const override { out << "}\n"; }

        std::string commentBOL() const override { return "# "; }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return indent() + "'" + s + styleStr(style) + "'" + " : {\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return indent() + "},\n\n";
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            return indent(2) + "'" + prependIfDigit(s, v.second) + "'" + " : " + fmtStyleVal(v.first, style) + ",\n";
        }
        std::string maxEnumFmt(const std::string& s, const valpair_t& v,
                               enumStyle_t style) const override {
            return "";
        }
        std::string fmtConstInt(unsigned val, const std::string& name,
                                const char* fmt, bool isLast) const override
        {
            return indent() + "'" + name + "'" + " : " + fmtNum(fmt, val) + (isLast ? ",\n\n" : ",\n");
        }
    };

    // C# printer
    class TPrinterCSharp final : public TPrinter {
    private:
        std::string commentBOL() const override { return "// ";  }

        void printPrologue(std::ostream& out) const override {
            out << "namespace Spv\n{\n\n";
            out << indent() << "public static class Specification\n";
            out << indent() << "{\n";
        }

        void printEpilogue(std::ostream& out) const override {
            out << indent() << "}\n";
            out << "}\n";
        }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return indent(2) + "public enum " + s + styleStr(style) + "\n" + indent(2) + "{\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return indent(2) + "}" + + (isLast ? "\n" : "\n\n");
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            return indent(3) + prependIfDigit(s, v.second) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        std::string fmtConstInt(unsigned val, const std::string& name,
                                const char* fmt, bool isLast) const override {
            return indent(2) + std::string("public const uint ") + name +
                " = " + fmtNum(fmt, val) + (isLast ? ";\n\n" : ";\n");
        }
    };

    // D printer
    class TPrinterD final : public TPrinter {
    private:
        std::string commentBeg() const override            { return "/+\n"; }
        std::string commentBOL() const override            { return " + ";  }
        std::string commentEnd(bool isLast) const override { return " +/\n"; }

        void printPrologue(std::ostream& out) const override {
            out << "module spv;\n\n";
        }

        void printEpilogue(std::ostream& out) const override {
        }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return "enum " + s + styleStr(style) + " : uint\n{\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return std::string("}\n\n");
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
                            enumStyle_t style, bool isLast) const override {
            return indent() + prependIfDigit("_", v.second) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        std::string fmtConstInt(unsigned val, const std::string& name,
                                const char* fmt, bool isLast) const override {
            return std::string("enum uint ") + name +
                " = " + fmtNum(fmt, val) + (isLast ? ";\n\n" : ";\n");
        }
    };

    // Beef printer
    class TPrinterBeef final : public TPrinter {
    private:
        std::string commentBOL() const override { return "// "; }

        void printPrologue(std::ostream& out) const override {
            out << "namespace Spv\n{\n";
            out << indent() << "using System;\n\n";
            out << indent() << "public static class Specification\n";
            out << indent() << "{\n";
        }

        void printEpilogue(std::ostream& out) const override {
            out << indent() << "}\n";
            out << "}\n";
        }

        std::string enumBeg(const std::string& s, enumStyle_t style) const override {
            return indent(2) + "[AllowDuplicates, CRepr] public enum " + s + styleStr(style) + "\n" + indent(2) + "{\n";
        }

        std::string enumEnd(const std::string& s, enumStyle_t style, bool isLast) const override {
            return indent(2) + "}" + +(isLast ? "\n" : "\n\n");
        }

        std::string enumFmt(const std::string& s, const valpair_t& v,
            enumStyle_t style, bool isLast) const override {
            return indent(3) + prependIfDigit(s, v.second) + " = " + fmtStyleVal(v.first, style) + ",\n";
        }

        std::string fmtConstInt(unsigned val, const std::string& name,
            const char* fmt, bool isLast) const override {
            return indent(2) + std::string("public const uint32 ") + name +
                " = " + fmtNum(fmt, val) + (isLast ? ";\n\n" : ";\n");
        }
    };

} // namespace

namespace spv {
    void PrintAllHeaders()
    {
        // TODO: Once MSVC 2012 is no longer a factor, use brace initializers here
        std::vector<std::pair<TLanguage, std::string>> langInfo;

        langInfo.push_back(std::make_pair(ELangC,       "spirv.h"));
        langInfo.push_back(std::make_pair(ELangCPP,     "spirv.hpp"));
        langInfo.push_back(std::make_pair(ELangCPP11,   "spirv.hpp11"));
        langInfo.push_back(std::make_pair(ELangJSON,    "spirv.json"));
        langInfo.push_back(std::make_pair(ELangLua,     "spirv.lua"));
        langInfo.push_back(std::make_pair(ELangPython,  "spirv.py"));
        langInfo.push_back(std::make_pair(ELangCSharp,  "spirv.cs"));
        langInfo.push_back(std::make_pair(ELangD,       "spv.d"));
        langInfo.push_back(std::make_pair(ELangBeef,    "spirv.bf"));

        for (const auto& lang : langInfo) {
            std::ofstream out(lang.second, std::ios::out);

            if ((out.rdstate() & std::ifstream::failbit)) {
                std::cerr << "Unable to open file: " << lang.second << std::endl;
            } else {
                PrintHeader(lang.first, out);
            }
        }
    }

    // Print header for given language to given output stream
    void PrintHeader(TLanguage lang, std::ostream& out)
    {
        typedef std::unique_ptr<TPrinter> TPrinterPtr;
        TPrinterPtr p;

        switch (lang) {
            case ELangC:       p = TPrinterPtr(new TPrinterC);       break;
            case ELangCPP:     p = TPrinterPtr(new TPrinterCPP);     break;
            case ELangCPP11:   p = TPrinterPtr(new TPrinterCPP11);   break;
            case ELangJSON:    p = TPrinterPtr(new TPrinterJSON);    break;
            case ELangLua:     p = TPrinterPtr(new TPrinterLua);     break;
            case ELangPython:  p = TPrinterPtr(new TPrinterPython);  break;
            case ELangCSharp:  p = TPrinterPtr(new TPrinterCSharp);  break;
            case ELangD:       p = TPrinterPtr(new TPrinterD);       break;
            case ELangBeef:    p = TPrinterPtr(new TPrinterBeef);    break;
            case ELangAll:     PrintAllHeaders();                    break;
            default:
                std::cerr << "Unknown language." << std::endl;
                return;
        }

        // Print the data in the requested format
        if (p)
            out << *p << std::endl;

        // object is auto-deleted
    }

} // namespace spv
