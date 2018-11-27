#include "thirdparty/rapidjson/prettywriter.h"
#include "thirdparty/rapidjson/stringbuffer.h"
#include "modules/gdscript/gdscript_parser.h"

class GDScriptASTBuilder {
private:
	rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF16<wchar_t> >* m_writer;
	rapidjson::StringBuffer m_s;
	GDScriptParser m_parser;
	String m_error;
private:
	inline void key(const wchar_t* key) { m_writer->Key(key); }
	inline void key(const String& key) { m_writer->Key(key.c_str()); }
	inline void serialize(const wchar_t* str) { m_writer->String(str); }
	inline void serialize(const String& str) { serialize(str.c_str()); }
	inline void serialize(const StringName& str) { serialize(String(str)); }
	inline void serialize(int val) { m_writer->Int(val); }
	inline void serialize(bool val) { m_writer->Bool(val); }
	void serialize(real_t val);
	inline void startArray() { m_writer->StartArray(); }
	inline void endArray() { m_writer->EndArray(); }
	inline void startObject() { m_writer->StartObject(); }
	inline void endObject() { m_writer->EndObject(); }
	inline void null() { m_writer->Null(); }

	template<typename T>
	void serialize(const PoolVector<T>& poolvector) {
		startArray();
		for(int i = 0; i < poolvector.size(); i++) {
			serialize(T(poolvector[i]));
		}

		endArray();
	}

	template<typename Elem>
	void serialize(const Vector<Elem>& vector) {
		startArray();

		for(int i = 0; i < vector.size(); i++) {
			serialize(vector[i]);
		}

		endArray();
	}

	template<typename Elem>
	void serialize(const List<Elem>& list) {
		startArray();

		const typename List<Elem>::Element *elem = list.front();
		while(elem) {
			serialize(**elem);
			elem = elem->next();
		}

		endArray();
	}

	template<typename Key, typename Value>
	void serialize(const Map<Key, Value>& map) {
		startArray();
		const typename Map<Key, Value>::Element *elem = map.front();

		while(elem) {
			startObject();

			key(L"key");
			serialize(elem->key());

			key(L"value");
			serialize(elem->value());

			endObject();

			elem = elem->next();
		}
		endArray();
	}


	void serialize(GDScriptParser::OperatorNode::Operator op);
	void serialize(GDScriptFunctions::Function func);
	void serialize(Variant::Type type);
	void serialize(const Object* o);
	void serialize(const Vector2& vector2);
	void serialize(const Vector3& vector3);
	void serialize(const Rect2& rect2);
	void serialize(const AABB& rect2);
	void serialize(const Transform2D& t2);
	void serialize(const Plane& p);
	void serialize_quat(const Quat& p);
	void serialize_basis(const Basis& b);
	void serialize_transform(const Transform& t);
	void serialize(const Color& c);
	void serialize_nodepath(const NodePath& n);
	void serialize(const RID& r);
	void serialize(const Dictionary& d);
	void serialize(const Array& a);
	void serialize(const Variant& variant);
	void serialize(PropertyHint hint);
	void serialize(const PropertyInfo& pi);
	void serialize(MultiplayerAPI::RPCMode mode);
	void serialize(const GDScriptParser::ClassNode::Member& member);
	void serialize(const GDScriptParser::ClassNode::Constant& constant);
	void serialize(const GDScriptParser::ClassNode::Signal& signal);
	void serialize(const GDScriptParser::DataType& data_type);
	void node_prefix(const GDScriptParser::Node* node);
	void node_suffix(const GDScriptParser::Node* node);
	void serialize(const GDScriptParser::ClassNode* node);
	void serialize(const GDScriptParser::FunctionNode* node);
	void serialize(const GDScriptParser::BuiltInFunctionNode* node);
	void serialize(const GDScriptParser::BlockNode* node);
	void serialize(const GDScriptParser::IdentifierNode* node);
	void serialize(const GDScriptParser::TypeNode* node);
	void serialize(const GDScriptParser::ConstantNode* node);
	void serialize(const GDScriptParser::ArrayNode* node);
	void serialize(const GDScriptParser::DictionaryNode* node);
	void serialize(const GDScriptParser::SelfNode* node);
	void serialize(const GDScriptParser::OperatorNode* node);
	void serialize(const GDScriptParser::PatternNode::PatternType pattern);
	void serialize(const GDScriptParser::DictionaryNode::Pair& pair);
	void serialize(const GDScriptParser::PatternNode* node);
	void serialize(const GDScriptParser::PatternBranchNode* node);
	void serialize(const GDScriptParser::MatchNode* node);
	void serialize(const GDScriptParser::ControlFlowNode::CFType type);
	void serialize(const GDScriptParser::ControlFlowNode* node);
	void serialize(const GDScriptParser::LocalVarNode* node);
	void serialize(const GDScriptParser::CastNode* node);
	void serialize(const GDScriptParser::AssertNode* node);
	void serialize(const GDScriptParser::BreakpointNode* node);
	void serialize(const GDScriptParser::NewLineNode* node);
	void serialize(const GDScriptParser::Node* node);
public:
	GDScriptASTBuilder();
	~GDScriptASTBuilder();
	inline const String& GetError() const { return m_error; }
	inline const String GetResult() const { return m_s.GetString(); }

	bool build(const String& code);
};
