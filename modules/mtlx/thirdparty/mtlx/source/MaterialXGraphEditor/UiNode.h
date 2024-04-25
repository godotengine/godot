//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_UINODE_H
#define MATERIALX_UINODE_H

#include <MaterialXCore/Unit.h>

#include <imgui_node_editor.h>

namespace mx = MaterialX;
namespace ed = ax::NodeEditor;

class UiNode;
class UiPin;

using UiNodePtr = std::shared_ptr<UiNode>;
using UiPinPtr = std::shared_ptr<UiPin>;

// An edge between two UiNodes, storing the two nodes and connecting input.
class UiEdge
{
  public:
    UiEdge(UiNodePtr uiDown, UiNodePtr uiUp, mx::InputPtr input) :
        _uiDown(uiDown),
        _uiUp(uiUp),
        _input(input)
    {
    }
    mx::InputPtr getInput()
    {
        return _input;
    }
    UiNodePtr getDown()
    {
        return _uiDown;
    }
    UiNodePtr getUp()
    {
        return _uiUp;
    }
    std::string getInputName()
    {
        if (_input != nullptr)
        {
            return _input->getName();
        }
        else
        {
            return mx::EMPTY_STRING;
        }
    }
    UiNodePtr _uiDown;
    UiNodePtr _uiUp;
    mx::InputPtr _input;
};

// A connectable input or output pin of a UiNode.
class UiPin
{
  public:
    UiPin(int id, const char* name, const std::string& type, std::shared_ptr<UiNode> node, ed::PinKind kind, mx::InputPtr input, mx::OutputPtr output) :
        _pinId(id),
        _name(name),
        _type(type),
        _pinNode(node),
        _kind(kind),
        _input(input),
        _output(output),
        _connected(false)
    {
    }

    void setConnected(bool connected)
    {
        _connected = connected;
    }

    bool getConnected()
    {
        return _connected;
    }

    void addConnection(UiPinPtr pin)
    {
        for (size_t i = 0; i < _connections.size(); i++)
        {
            if (_connections[i]->_pinId == pin->_pinId)
            {
                return;
            }
        }
        _connections.push_back(pin);
    }

    void deleteConnection(UiPinPtr pin)
    {
        for (size_t i = 0; i < _connections.size(); i++)
        {
            if (_connections[i]->_pinId == pin->_pinId)
            {
                _connections.erase(_connections.begin() + i);
            }
        }
        for (size_t i = 0; i < pin->_connections.size(); i++)
        {
            if (pin->_connections[i]->_pinId == _pinId)
            {
                pin->_connections.erase(pin->_connections.begin() + i);
            }
        }
        if (pin->_connections.size() == 0)
        {
            pin->setConnected(false);
        }
        return;
    }

    const std::vector<UiPinPtr>& getConnections()
    {
        return _connections;
    }

  public:
    ed::PinId _pinId;
    std::string _name;
    std::string _type;
    std::shared_ptr<UiNode> _pinNode;
    ed::PinKind _kind;
    mx::InputPtr _input;
    mx::OutputPtr _output;
    std::vector<UiPinPtr> _connections;
    bool _connected;
};

// The visual representation of a node in a graph.
class UiNode
{
  public:
    UiNode();
    UiNode(const std::string& name, int id);
    ~UiNode(){};

    std::string getName()
    {
        return _name;
    }
    ImVec2 getPos()
    {
        return _nodePos;
    }
    int getInputConnect()
    {
        return _inputNodeNum;
    }
    int getId()
    {
        return _id;
    }
    const std::vector<UiNodePtr>& getOutputConnections()
    {
        return _outputConnections;
    }
    mx::NodePtr getNode()
    {
        return _currNode;
    }
    mx::InputPtr getInput()
    {
        return _currInput;
    }
    mx::OutputPtr getOutput()
    {
        return _currOutput;
    }

    void setName(const std::string& newName)
    {
        _name = newName;
    }
    void setPos(ImVec2 pos)
    {
        _nodePos = pos;
    }
    void setInputNodeNum(int num)
    {
        _inputNodeNum += num;
    }
    void setNode(mx::NodePtr node)
    {
        _currNode = node;
    }
    void setInput(mx::InputPtr input)
    {
        _currInput = input;
    }
    void setOutput(mx::OutputPtr output)
    {
        _currOutput = output;
    }
    void setOutputConnection(UiNodePtr connections)
    {
        _outputConnections.push_back(connections);
    }

    void setMessage(const std::string& message)
    {
        _message = message;
    }

    const std::string& getMessage()
    {
        return _message;
    }

    void setCategory(const std::string& category)
    {
        _category = category;
    }

    const std::string& getCategory()
    {
        return _category;
    }

    void setType(const std::string& type)
    {
        _type = type;
    }

    const std::string& getType()
    {
        return _type;
    }

    mx::NodeGraphPtr getNodeGraph()
    {
        return _currNodeGraph;
    }

    void setNodeGraph(mx::NodeGraphPtr nodeGraph)
    {
        _currNodeGraph = nodeGraph;
    }

    UiNodePtr getConnectedNode(const std::string& name);
    float getAverageY();
    float getMinX();
    int getEdgeIndex(int id, UiPinPtr pin);
    std::vector<UiEdge> edges;
    std::vector<UiPinPtr> inputPins;
    std::vector<UiPinPtr> outputPins;
    void removeOutputConnection(const std::string& name);
    mx::ElementPtr getMxElement();
    int _level;
    bool _showAllInputs;

  private:
    int _id;
    ImVec2 _nodePos;
    std::string _name;
    int _inputNodeNum;
    std::vector<std::pair<int, std::string>> _inputs;
    std::vector<std::pair<int, std::string>> _outputs;
    std::vector<UiNodePtr> _outputConnections;
    mx::NodePtr _currNode;
    mx::InputPtr _currInput;
    mx::OutputPtr _currOutput;
    std::string _category;
    std::string _message;
    std::string _type;
    mx::NodeGraphPtr _currNodeGraph;
};

#endif
