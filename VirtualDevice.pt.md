# Patch Note - Virtual Device Input System

## 1. Visão Geral

O sistema de **Virtual Input** introduz uma nova camada de abstração de hardware na Godot Engine, tratando controles de interface (UI) como dispositivos de entrada de primeira classe. Esta implementação integra profundamente interações táteis ao pipeline de entrada nativo, permitindo que a engine processe botões e joysticks virtuais com a mesma prioridade e estrutura de um gamepad físico ou teclado.

## 2. Motivação

A implementação de um sistema de entrada virtual nativo na Godot Engine é sustentada por quatro pilares fundamentais para facilitar o **port de jogos de PC e consoles para dispositivos touch**:

- **Performance Determinística**: Ao processar a detecção de toque e o despacho de eventos no core em C++, garantimos uma resposta imediata e fluida, essencial para manter a fidelidade da experiência original em telas sensíveis ao toque.
- **Unificação de Input**: Através dos novos eventos nativos, desenvolvedores podem mapear ações no `InputMap` que aceitam simultaneamente entradas físicas (gamepads/teclado) e virtuais. Isso permite que um jogo projetado para consoles funcione em dispositivos móveis com mudanças mínimas na lógica de gameplay.
- **Aceleração de Desenvolvimento (DX)**: A integração com o Inspector e o sistema de Temas permite que artistas e designers customizem a interface de controle sem tocar em código, utilizando o fluxo de trabalho padrão da engine.
- **Robustez no Multitoque**: O gerenciamento de estados de toque (tracking de IDs e captura de foco) é tratado nativamente, prevenindo conflitos comuns em implementações manuais de GUI.

## 3. Core/Input

O núcleo da mudança reside na expansão da classe `InputEvent` para suportar tipos de dados específicos para dispositivos virtuais:

- **`InputEventVirtualButton`**: Representa estados binários (pressionado/solto) vindos de dispositivos virtuais, incluindo suporte a `device_id` para múltiplos controles na mesma tela.
- **`InputEventVirtualMotion`**: Transporta dados de movimento relativo ou absoluto (eixos X/Y), ideal para sticks analógicos e áreas de deslizamento (touchpads).
- **Integração com o `Input` Singleton**: O método `Input.parse_input_event()` foi otimizado para rotear esses eventos, garantindo que o sistema de `is_action_pressed()` funcione de forma transparente com os novos nós virtuais.

## 4. Scene/GUI

A camada visual foi estruturada em uma hierarquia de classes flexível sob o módulo de interface:

- **`VirtualDevice` (Base)**: Classe abstrata que herda de `Control`. Gerencia o estado de foco tátil e provê a interface comum para o despacho de eventos.
- **`VirtualButton`**: Um controle otimizado para disparar ações discretas, suportando ícones, texto e estados visuais completos (Pressed, Hover, Disabled).
- **`VirtualJoystick`**: Implementação de stick analógico com zonas de deadzone e clamp configuráveis.
- **`VirtualJoystickDynamic`**: Uma área de captura que faz o joystick dinâmico aparecer exatamente onde o usuário tocou dentro da região.
- **`VirtualTouchPad`**: Área focada em movimento relativo (delta), funcionando de forma análoga a um trackpad de laptop, essencial para controle de câmera em 3D.

## 5. Doc/*.xml

A documentação de referência da API foi totalmente integrada ao sistema de ajuda interno da Godot. Cada novo nó e evento possui descrições detalhadas de propriedades, métodos e sinais, localizadas em:

- Cada nó em `scene/gui/` possui seu próprio arquivo de documentação XML dedicado.
- Arquivos do `core` como `Input` e `InputMap` foram atualizados para suportar os novos tipos de eventos e lógica de roteamento.
- Documentação sincronizada para: `VirtualButton`, `VirtualJoystick`, `VirtualJoystickDynamic`, `VirtualTouchPad`, `InputEventVirtualButton` e `InputEventVirtualMotion`.

Isso garante que o desenvolvedor tenha acesso imediato à documentação técnica diretamente pelo editor (F1) ou via documentação online oficial.

## 6. Scene/Theme

Para garantir que o sistema virtual se adapte a qualquer projeto, ele foi totalmente integrado ao **ThemeDB**:

- **Customização via Inspector**: Utilizando a macro `BIND_THEME_ITEM`, todas as propriedades visuais (StyleBoxes, Colors, Fonts) são expostas na seção "Theme Overrides" do Inspector.
- **Temas Padrão**: Definições iniciais consistentes foram adicionadas ao `default_theme.cpp`, fornecendo uma interface funcional e esteticamente agradável out-of-the-box, mantendo a coerência visual com os demais controles de UI da Godot.
