//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXView/Editor.h>

#include <MaterialXView/Viewer.h>

#include <nanogui/colorwheel.h>
#include <nanogui/slider.h>
#include <nanogui/vscrollpanel.h>

namespace
{

// Custom color picker with numeric entry and feedback.
class EditorColorPicker : public ng::ColorPicker
{
  public:
    EditorColorPicker(ng::Widget* parent, const ng::Color& color) :
        ng::ColorPicker(parent, color)
    {
        ng::Popup* popup = this->popup();
        ng::Widget* floatGroup = new ng::Widget(popup);
        auto layout = new ng::GridLayout(ng::Orientation::Horizontal, 2,
                                         ng::Alignment::Middle, 2, 2);
        layout->set_col_alignment({ ng::Alignment::Fill, ng::Alignment::Fill });
        layout->set_spacing(1, 1);
        floatGroup->set_layout(layout);

        const std::array<std::string, 4> COLOR_LABELS = { "Red", "Green", "Blue", "Alpha" };
        for (size_t i = 0; i < COLOR_LABELS.size(); i++)
        {
            new ng::Label(floatGroup, COLOR_LABELS[i]);
            _colorWidgets[i] = new ng::FloatBox<float>(floatGroup, color[i]);
            _colorWidgets[i]->set_editable(true);
            _colorWidgets[i]->set_alignment(ng::TextBox::Alignment::Right);
            _colorWidgets[i]->set_fixed_size(ng::Vector2i(70, 20));
            _colorWidgets[i]->set_font_size(15);
            _colorWidgets[i]->set_spinnable(true);
            _colorWidgets[i]->set_callback([this](float)
            {
                ng::Color value(_colorWidgets[0]->value(), _colorWidgets[1]->value(), _colorWidgets[2]->value(), _colorWidgets[3]->value());
                m_color_wheel->set_color(value);
                m_pick_button->set_background_color(value);
                m_pick_button->set_text_color(value.contrasting_color());
            });
        }

        // The color wheel does not handle alpha properly, so only
        // overwrite RGB in the callback.
        m_callback = [this](const ng::Color& value)
        {
            _colorWidgets[0]->set_value(value[0]);
            _colorWidgets[1]->set_value(value[1]);
            _colorWidgets[2]->set_value(value[2]);
        };
    }

  protected:
    // Additional numeric entry / feedback widgets
    ng::FloatBox<float>* _colorWidgets[4];
};

} // anonymous namespace

//
// PropertyEditor methods
//

PropertyEditor::PropertyEditor() :
    _window(nullptr),
    _container(nullptr),
    _gridLayout2(nullptr),
    _gridLayout3(nullptr),
    _visible(false),
    _fileDialogsForImages(true)
{
}

void PropertyEditor::create(Viewer& parent)
{
    ng::Window* parentWindow = parent.getWindow();

    // Remove the window associated with the form.
    // This is done by explicitly creating and owning the window
    // as opposed to having it being done by the form
    ng::Vector2i previousPosition(15, parentWindow->height());
    if (_window)
    {
        for (int i = 0; i < _window->child_count(); i++)
        {
            _window->remove_child_at(i);
        }
        // We don't want the property editor to move when
        // we update it's contents so cache any previous position
        // to use when we create a new window.
        previousPosition = _window->position();
        parent.remove_child(_window);
    }

    if (previousPosition.x() < 0)
        previousPosition.x() = 0;
    if (previousPosition.y() < 0)
        previousPosition.y() = 0;

    _window = new ng::Window(&parent, "Property Editor");
    _window->set_layout(new ng::GroupLayout());
    _window->set_position(previousPosition);
    _window->set_visible(_visible);

    ng::VScrollPanel* scroll_panel = new ng::VScrollPanel(_window);
    scroll_panel->set_fixed_height(300);
    _container = new ng::Widget(scroll_panel);
    _container->set_layout(new ng::GroupLayout(1, 1, 1, 1));

    // 2 cell layout for label plus value pair.
    _gridLayout2 = new ng::GridLayout(ng::Orientation::Horizontal, 2,
                                      ng::Alignment::Minimum, 2, 2);
    _gridLayout2->set_col_alignment({ ng::Alignment::Minimum, ng::Alignment::Maximum });

    // 3 cell layout for label plus widget value pair.
    _gridLayout3 = new ng::GridLayout(ng::Orientation::Horizontal, 3,
                                      ng::Alignment::Minimum, 2, 2);
    _gridLayout3->set_col_alignment({ ng::Alignment::Minimum, ng::Alignment::Maximum, ng::Alignment::Maximum });
}

void PropertyEditor::addItemToForm(const mx::UIPropertyItem& item, const std::string& group,
                                   ng::Widget* container, Viewer* viewer, bool editable)
{
    const mx::UIProperties& ui = item.ui;
    mx::ValuePtr value = item.variable->getValue();
    if (!value)
    {
        return;
    }

    std::string label = item.label;
    const std::string& unit = item.variable->getUnit();
    if (!unit.empty())
    {
        label += std::string(" (") + unit + std::string(")");
    }
    const std::string& path = item.variable->getPath();
    const mx::StringVec& enumeration = ui.enumeration;
    const std::vector<mx::ValuePtr> enumValues = ui.enumerationValues;

    if (!group.empty())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);
        ng::Label* groupLabel = new ng::Label(twoColumns, group);
        groupLabel->set_font_size(20);
        groupLabel->set_font("sans-bold");
        new ng::Label(twoColumns, "");
    }

    // Integer input. Can map to a combo box if an enumeration
    if (value->isA<int>())
    {
        const size_t INVALID_INDEX = std::numeric_limits<size_t>::max();
        auto indexInEnumeration = [&value, &enumValues, &enumeration]()
        {
            size_t index = 0;
            for (auto& enumValue : enumValues)
            {
                if (value->getValueString() == enumValue->getValueString())
                {
                    return index;
                }
                index++;
            }
            index = 0;
            for (auto& enumName : enumeration)
            {
                if (value->getValueString() == enumName)
                {
                    return index;
                }
                index++;
            }
            return std::numeric_limits<size_t>::max(); // INVALID_INDEX;
        };

        // Create a combo box. The items are the enumerations in order.
        const size_t valueIndex = indexInEnumeration();
        if (INVALID_INDEX != valueIndex)
        {
            ng::Widget* twoColumns = new ng::Widget(container);
            twoColumns->set_layout(_gridLayout2);

            new ng::Label(twoColumns, label);
            ng::ComboBox* comboBox = new ng::ComboBox(twoColumns, { "" });
            comboBox->set_enabled(editable);
            comboBox->set_items(enumeration);
            comboBox->set_selected_index(static_cast<int>(valueIndex));
            comboBox->set_fixed_size(ng::Vector2i(100, 20));
            comboBox->set_font_size(15);
            comboBox->set_callback([path, viewer, enumeration, enumValues](int index)
            {
                mx::MaterialPtr material = viewer->getSelectedMaterial();
                if (material)
                {
                    if (index >= 0 && static_cast<size_t>(index) < enumValues.size())
                    {
                        material->modifyUniform(path, enumValues[index]);
                    }
                    else if (index >= 0 && static_cast<size_t>(index) < enumeration.size())
                    {
                        material->modifyUniform(path, mx::Value::createValue(index), enumeration[index]);
                    }
                }
            });
        }
        else
        {
            ng::Widget* twoColumns = new ng::Widget(container);
            twoColumns->set_layout(_gridLayout2);

            new ng::Label(twoColumns, label);
            auto intVar = new ng::IntBox<int>(twoColumns);
            intVar->set_fixed_size(ng::Vector2i(100, 20));
            intVar->set_font_size(15);
            intVar->set_editable(editable);
            intVar->set_spinnable(editable);
            intVar->set_callback([intVar, path, viewer](int /*unclamped*/)
            {
                mx::MaterialPtr material = viewer->getSelectedMaterial();
                if (material)
                {
                    // https://github.com/wjakob/nanogui/issues/205
                    material->modifyUniform(path, mx::Value::createValue(intVar->value()));
                }
            });
            if (ui.uiMin)
            {
                intVar->set_min_value(ui.uiMin->asA<int>());
            }
            if (ui.uiMax)
            {
                intVar->set_max_value(ui.uiMax->asA<int>());
            }
            if (ui.uiStep)
            {
                intVar->set_value_increment(ui.uiStep->asA<int>());
            }
            intVar->set_value(value->asA<int>());
        }
    }

    // Float widget
    else if (value->isA<float>())
    {
        ng::Widget* threeColumns = new ng::Widget(container);
        threeColumns->set_layout(_gridLayout3);
        ng::FloatBox<float>* floatBox = createFloatWidget(threeColumns, label, value->asA<float>(), &ui, [viewer, path](float value)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                material->modifyUniform(path, mx::Value::createValue(value));
            }
        });
        floatBox->set_fixed_size(ng::Vector2i(100, 20));
        floatBox->set_editable(editable);
    }

    // Boolean widget
    else if (value->isA<bool>())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);

        bool v = value->asA<bool>();
        new ng::Label(twoColumns, label);
        ng::CheckBox* boolVar = new ng::CheckBox(twoColumns, "");
        boolVar->set_checked(v);
        boolVar->set_font_size(15);
        boolVar->set_callback([path, viewer](bool v)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                material->modifyUniform(path, mx::Value::createValue((float) v));
            }
        });
    }

    // Color3 input. Can map to a combo box if an enumeration
    else if (value->isA<mx::Color3>())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);

        // Determine if there is an enumeration for this
        mx::Color3 color = value->asA<mx::Color3>();
        int index = -1;
        if (!enumeration.empty() && !enumValues.empty())
        {
            index = 0;
            for (size_t i = 0; i < enumValues.size(); i++)
            {
                if (enumValues[i]->asA<mx::Color3>() == color)
                {
                    index = static_cast<int>(i);
                    break;
                }
            }
        }

        // Create a combo box. The items are the enumerations in order.
        if (index >= 0)
        {
            ng::ComboBox* comboBox = new ng::ComboBox(twoColumns, { "" });
            comboBox->set_enabled(editable);
            comboBox->set_items(enumeration);
            comboBox->set_selected_index(index);
            comboBox->set_font_size(15);
            comboBox->set_callback([path, enumValues, viewer](int index)
            {
                mx::MaterialPtr material = viewer->getSelectedMaterial();
                if (material)
                {
                    if (index < (int) enumValues.size())
                    {
                        material->modifyUniform(path, enumValues[index]);
                    }
                }
            });
        }
        else
        {
            mx::Color3 v = value->asA<mx::Color3>();
            ng::Color c(v[0], v[1], v[2], 1.0);

            new ng::Label(twoColumns, label);
            auto colorVar = new EditorColorPicker(twoColumns, c);
            colorVar->set_fixed_size({ 100, 20 });
            colorVar->set_font_size(15);
            colorVar->set_final_callback([path, viewer](const ng::Color& c)
            {
                mx::MaterialPtr material = viewer->getSelectedMaterial();
                if (material)
                {
                    mx::Vector3 v(c.r(), c.g(), c.b());
                    material->modifyUniform(path, mx::Value::createValue(v));
                }
            });
        }
    }

    // Color4 input
    else if (value->isA<mx::Color4>())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);

        new ng::Label(twoColumns, label);
        mx::Color4 v = value->asA<mx::Color4>();
        ng::Color c(v[0], v[1], v[2], v[3]);
        auto colorVar = new EditorColorPicker(twoColumns, c);
        colorVar->set_fixed_size({ 100, 20 });
        colorVar->set_font_size(15);
        colorVar->set_final_callback([path, viewer](const ng::Color& c)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector4 v(c.r(), c.g(), c.b(), c.w());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
    }

    // Vec 2 widget
    else if (value->isA<mx::Vector2>())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);

        mx::Vector2 v = value->asA<mx::Vector2>();
        new ng::Label(twoColumns, label + ".x");
        auto v1 = new ng::FloatBox<float>(twoColumns, v[0]);
        v1->set_fixed_size({ 100, 20 });
        v1->set_font_size(15);
        new ng::Label(twoColumns, label + ".y");
        auto v2 = new ng::FloatBox<float>(twoColumns, v[1]);
        v2->set_fixed_size({ 100, 20 });
        v2->set_font_size(15);
        v1->set_callback([v2, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector2 v(f, v2->value());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v1->set_spinnable(editable);
        v1->set_editable(editable);
        v2->set_callback([v1, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector2 v(v1->value(), f);
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v2->set_spinnable(editable);
        v2->set_editable(editable);
    }

    // Vec 3 input
    else if (value->isA<mx::Vector3>())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);

        mx::Vector3 v = value->asA<mx::Vector3>();
        new ng::Label(twoColumns, label + ".x");
        auto v1 = new ng::FloatBox<float>(twoColumns, v[0]);
        v1->set_fixed_size({ 100, 20 });
        v1->set_font_size(15);
        new ng::Label(twoColumns, label + ".y");
        auto v2 = new ng::FloatBox<float>(twoColumns, v[1]);
        v2->set_fixed_size({ 100, 20 });
        v2->set_font_size(15);
        new ng::Label(twoColumns, label + ".z");
        auto v3 = new ng::FloatBox<float>(twoColumns, v[2]);
        v3->set_fixed_size({ 100, 20 });
        v3->set_font_size(15);

        v1->set_callback([v2, v3, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector3 v(f, v2->value(), v3->value());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v1->set_spinnable(editable);
        v1->set_editable(editable);
        v2->set_callback([v1, v3, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector3 v(v1->value(), f, v3->value());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v2->set_spinnable(editable);
        v2->set_editable(editable);
        v3->set_callback([v1, v2, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector3 v(v1->value(), v2->value(), f);
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v3->set_spinnable(editable);
        v3->set_editable(editable);
    }

    // Vec 4 input
    else if (value->isA<mx::Vector4>())
    {
        ng::Widget* twoColumns = new ng::Widget(container);
        twoColumns->set_layout(_gridLayout2);

        mx::Vector4 v = value->asA<mx::Vector4>();
        new ng::Label(twoColumns, label + ".x");
        auto v1 = new ng::FloatBox<float>(twoColumns, v[0]);
        v1->set_fixed_size({ 100, 20 });
        v1->set_font_size(15);
        new ng::Label(twoColumns, label + ".y");
        auto v2 = new ng::FloatBox<float>(twoColumns, v[1]);
        v2->set_fixed_size({ 100, 20 });
        v1->set_font_size(15);
        new ng::Label(twoColumns, label + ".z");
        auto v3 = new ng::FloatBox<float>(twoColumns, v[2]);
        v3->set_fixed_size({ 100, 20 });
        v1->set_font_size(15);
        new ng::Label(twoColumns, label + ".w");
        auto v4 = new ng::FloatBox<float>(twoColumns, v[3]);
        v4->set_fixed_size({ 100, 20 });
        v1->set_font_size(15);

        v1->set_callback([v2, v3, v4, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector4 v(f, v2->value(), v3->value(), v4->value());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v1->set_spinnable(editable);
        v2->set_callback([v1, v3, v4, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector4 v(v1->value(), f, v3->value(), v4->value());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v2->set_spinnable(editable);
        v2->set_editable(editable);
        v3->set_callback([v1, v2, v4, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector4 v(v1->value(), v2->value(), f, v4->value());
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v3->set_spinnable(editable);
        v3->set_editable(editable);
        v4->set_callback([v1, v2, v3, path, viewer](float f)
        {
            mx::MaterialPtr material = viewer->getSelectedMaterial();
            if (material)
            {
                mx::Vector4 v(v1->value(), v2->value(), v3->value(), f);
                material->modifyUniform(path, mx::Value::createValue(v));
            }
        });
        v4->set_spinnable(editable);
        v4->set_editable(editable);
    }

    // String
    else if (value->isA<std::string>())
    {
        std::string v = value->asA<std::string>();
        if (!v.empty())
        {
            ng::Widget* twoColumns = new ng::Widget(container);
            twoColumns->set_layout(_gridLayout2);

            if (item.variable->getType() == mx::Type::FILENAME)
            {
                new ng::Label(twoColumns, label);
                ng::Button* buttonVar = new ng::Button(twoColumns, mx::FilePath(v).getBaseName());
                buttonVar->set_enabled(editable);
                buttonVar->set_font_size(15);
                buttonVar->set_callback([buttonVar, path, viewer]()
                {
                    mx::MaterialPtr material = viewer->getSelectedMaterial();
                    mx::ShaderPort* uniform = material ? material->findUniform(path) : nullptr;
                    if (uniform)
                    {
                        if (uniform->getType() == mx::Type::FILENAME)
                        {
                            mx::ImageHandlerPtr handler = viewer->getImageHandler();
                            if (handler)
                            {
                                mx::StringSet extensions = handler->supportedExtensions();
                                std::vector<std::pair<std::string, std::string>> filetypes;
                                for (const auto& extension : extensions)
                                {
                                    filetypes.emplace_back(extension, extension);
                                }
                                std::string filename = ng::file_dialog(filetypes, false);
                                if (!filename.empty())
                                {
                                    uniform->setValue(mx::Value::createValue<std::string>(filename));
                                    buttonVar->set_caption(mx::FilePath(filename).getBaseName());
                                    viewer->perform_layout();
                                }
                            }
                        }
                    }
                });
            }
            else
            {
                new ng::Label(twoColumns, label);
                ng::TextBox* stringVar = new ng::TextBox(twoColumns, v);
                stringVar->set_fixed_size({ 100, 20 });
                stringVar->set_font_size(15);
                stringVar->set_callback([path, viewer](const std::string& v)
                {
                    mx::MaterialPtr material = viewer->getSelectedMaterial();
                    mx::ShaderPort* uniform = material ? material->findUniform(path) : nullptr;
                    if (uniform)
                    {
                        uniform->setValue(mx::Value::createValue<std::string>(v));
                    }
                    return true;
                });
            }
        }
    }
}

void PropertyEditor::updateContents(Viewer* viewer)
{
    create(*viewer);

    mx::MaterialPtr material = viewer->getSelectedMaterial();
    mx::TypedElementPtr elem = material ? material->getElement() : nullptr;
    if (!material || !elem)
    {
        return;
    }

#ifndef MATERIALXVIEW_METAL_BACKEND
    // Bind and validate the shader
    material->bindShader();
#endif

    // Shading model display
    mx::NodePtr node = elem->asA<mx::Node>();
    if (node)
    {
        std::string shaderName = node->getCategory();
        std::vector<mx::NodePtr> shaderNodes = mx::getShaderNodes(node);
        if (!shaderNodes.empty())
        {
            shaderName = shaderNodes[0]->getCategory();
        }
        if (!shaderName.empty() && shaderName != "surface")
        {
            ng::Widget* twoColumns = new ng::Widget(_container);
            twoColumns->set_layout(_gridLayout2);
            ng::Label* modelLabel = new ng::Label(twoColumns, "Shading Model");
            modelLabel->set_font_size(20);
            modelLabel->set_font("sans-bold");
            ng::Label* nameLabel = new ng::Label(twoColumns, shaderName);
            nameLabel->set_font_size(20);
        }
    }

    bool addedItems = false;
    const mx::VariableBlock* publicUniforms = material->getPublicUniforms();
    if (publicUniforms)
    {
        mx::UIPropertyGroup groups;
        mx::UIPropertyGroup unnamedGroups;
        const std::string pathSeparator(":");
        mx::createUIPropertyGroups(elem->getDocument(), *publicUniforms, groups, unnamedGroups, pathSeparator);

        // First add items with named groups.
        std::string previousFolder;
        for (auto it = groups.begin(); it != groups.end(); ++it)
        {
            const std::string& folder = it->first;
            const mx::UIPropertyItem& item = it->second;

            // Verify that the uniform is editable, as some inputs may be optimized out during compilation.
            if (material->findUniform(item.variable->getPath()))
            {
                addItemToForm(item, (previousFolder == folder) ? mx::EMPTY_STRING : folder, _container, viewer, true);
                previousFolder.assign(folder);
                addedItems = true;
            }
        }

        // Then add items with unnamed groups.
        bool addedLabel = false;
        for (auto it2 = unnamedGroups.begin(); it2 != unnamedGroups.end(); ++it2)
        {
            const mx::UIPropertyItem& item = it2->second;
            if (material->findUniform(item.variable->getPath()))
            {
                addItemToForm(item, addedLabel ? mx::EMPTY_STRING : "Shader Inputs", _container, viewer, true);
                addedLabel = true;
                addedItems = true;
            }
        }
    }
    if (!addedItems)
    {
        new ng::Label(_container, "No Shader Inputs");
        new ng::Label(_container, "");
    }

    viewer->perform_layout();
}

ng::FloatBox<float>* createFloatWidget(ng::Widget* parent, const std::string& label, float value,
                                       const mx::UIProperties* ui, std::function<void(float)> callback)
{
    new ng::Label(parent, label);

    ng::Slider* slider = new ng::Slider(parent);
    slider->set_value(value);

    ng::FloatBox<float>* box = new ng::FloatBox<float>(parent, value);
    box->set_fixed_width(60);
    box->set_font_size(15);
    box->set_alignment(ng::TextBox::Alignment::Right);

    if (ui)
    {
        std::pair<float, float> range(0.0f, 1.0f);
        if (ui->uiMin)
        {
            box->set_min_value(ui->uiMin->asA<float>());
            range.first = ui->uiMin->asA<float>();
        }
        if (ui->uiMax)
        {
            box->set_max_value(ui->uiMax->asA<float>());
            range.second = ui->uiMax->asA<float>();
        }
        if (ui->uiSoftMin)
        {
            range.first = ui->uiSoftMin->asA<float>();
        }
        if (ui->uiSoftMax)
        {
            range.second = ui->uiSoftMax->asA<float>();
        }
        if (range.first != range.second)
        {
            slider->set_range(range);
        }
        if (ui->uiStep)
        {
            box->set_value_increment(ui->uiStep->asA<float>());
            box->set_spinnable(true);
            box->set_editable(true);
        }
    }

    slider->set_callback([box, callback](float value)
    {
        box->set_value(value);
        callback(value);
    });
    box->set_callback([slider, callback](float value)
    {
        slider->set_value(value);
        callback(value);
    });

    return box;
}

ng::IntBox<int>* createIntWidget(ng::Widget* parent, const std::string& label, int value,
                                 const mx::UIProperties* ui, std::function<void(int)> callback)
{
    new ng::Label(parent, label);

    ng::Slider* slider = new ng::Slider(parent);
    slider->set_value((float) value);

    ng::IntBox<int>* box = new ng::IntBox<int>(parent, value);
    box->set_fixed_width(60);
    box->set_font_size(15);
    box->set_alignment(ng::TextBox::Alignment::Right);
    if (ui)
    {
        std::pair<int, int> range(0, 1);
        if (ui->uiMin)
        {
            box->set_min_value(ui->uiMin->asA<int>());
            range.first = ui->uiMin->asA<int>();
        }
        if (ui->uiMax)
        {
            box->set_max_value(ui->uiMax->asA<int>());
            range.second = ui->uiMax->asA<int>();
        }
        if (ui->uiSoftMin)
        {
            range.first = ui->uiSoftMin->asA<int>();
        }
        if (ui->uiSoftMax)
        {
            range.second = ui->uiSoftMax->asA<int>();
        }
        if (range.first != range.second)
        {
            std::pair<float, float> float_range((float) range.first, (float) range.second);
            slider->set_range(float_range);
        }
        if (ui->uiStep)
        {
            box->set_value_increment(ui->uiStep->asA<int>());
            box->set_spinnable(true);
            box->set_editable(true);
        }
    }

    slider->set_callback([box, callback](float value)
    {
        box->set_value((int) value);
        callback((int) value);
    });
    box->set_callback([slider, callback](int value)
    {
        slider->set_value((float) value);
        callback(value);
    });

    return box;
}
