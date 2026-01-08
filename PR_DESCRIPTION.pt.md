# Input: Adição do sistema nativo de Dispositivo de Entrada Virtual (Virtual Device) para controles na tela

## Visão Geral

Este PR introduz um sistema nativo de **Dispositivo de Entrada Virtual** na Godot Engine. Ele fornece uma arquitetura robusta e de alta performance para controles na tela (joysticks, botões, touchpads), tratando-os como dispositivos de entrada de primeira classe. Esta implementação atende diretamente à necessidade antiga da comunidade por uma entrada móvel padronizada, como delineado nas **Godot Proposals [#13943](https://github.com/godotengine/godot-proposals/issues/13943)** e **[#11192](https://github.com/godotengine/godot-proposals/issues/11192)**.

## Intenções e Filosofia

A intenção principal deste sistema é fornecer uma **"Camada de Abstração de Hardware para UI"**.

Atualmente, os desenvolvedores precisam escolher entre os limitados nós `TouchScreenButton` ou joysticks complexos e fragmentados baseados em GDScript. Este PR muda isso ao:

1. **Unificar a Lógica de Entrada**: O código do jogo não deve se preocupar se um `InputEvent` vem de um Sony DualSense físico ou de um `VirtualJoystick` em um iPad. Ao emitir eventos nativos `InputEventVirtual*`, este sistema permite que o `InputMap` trate controles na tela exatamente como hardware.
2. **Performance (C++ Core)**: Ao mover o rastreamento de toque, cálculos de vetores e o despacho de eventos para o core da engine, alcançamos a menor latência possível e eliminamos o custo de processamento (frame-budget) de executar lógica de entrada complexa em GDScript.
3. **Experiência do Desenvolvedor (DX)**: Fornecer um conjunto de nós baseados em `Control` que respeitem o sistema de layout, o Inspector e o sistema de Temas permite que artistas e designers iterem em controles móveis sem escrever código repetitivo (boilerplate).

## Detalhamento Técnico

### 1. Expansão do Core de Entrada

A hierarquia de `InputEvent` foi expandida com dois tipos primários:

- **`InputEventVirtualButton`**: Representa estados digitais (Ligado/Desligado) de dispositivos virtuais. Suporta `device_id` para permitir múltiplos controles virtuais.
- **`InputEventVirtualMotion`**: Transporta dados analógicos (eixos X/Y) para joysticks, sliders ou touchpads.
- A integração no `Input.parse_input_event()` garante que `is_action_pressed()` e `get_vector()` funcionem perfeitamente tanto com entradas físicas quanto virtuais.

### 2. Arquitetura: Classe base `VirtualDevice`

Uma nova classe base abstrata `VirtualDevice` (herdando de `Control`) serve como ponte entre as interações de UI e o pipeline de Entrada. Ela gerencia:

- **Rastreamento de ID de Toque**: Lida automaticamente com o estado multi-toque e captura de foco em diferentes camadas de UI.
- **Despacho de Eventos**: Lógica para traduzir o `InputEventScreenTouch/Drag` bruto em eventos virtuais de alto nível.

### 3. Novos Nós de UI (`scene/gui/`)

O sistema inclui nós especializados baseados nos requisitos de jogos modernos:

- **`VirtualButton`**: Um botão otimizado para toque que suporta vários comportamentos de pressão e integração completa com temas.
- **`VirtualJoystick` & `VirtualJoystickDynamic`**: Joysticks analógicos de alta performance com suporte aos modos Fixo (Fixed) e Dinâmico (Dynamic). O `VirtualJoystickDynamic` introduz uma área de captura que faz o joystick aparecer no ponto de toque, com uma opção de **"Visível por Padrão"** que mantém o joystick em uma posição predefinida enquanto ocioso.
- **`VirtualDPad`**: Direcional digital padrão de 4 direções para jogos retrô e navegação de UI.
- **`VirtualTouchPad`**: Uma área de movimento relativo projetada especificamente para controles de câmera e lógica de olhar ao redor.

### 4. Integração com ThemeDB

Para garantir consistência visual, o sistema está totalmente integrado ao **Sistema de Temas** da Godot:

- Os novos nós usam `BIND_THEME_ITEM` para expor StyleBoxes, Cores e Fontes.
- Estéticas padrão foram adicionadas ao `default_theme.cpp`, garantindo um visual profissional "pronto para uso" (out-of-the-box), enquanto permanecem totalmente customizáveis via "Theme Overrides".

## Por que no Core?

Embora existam muitos addons de terceiros, eles carecem da integração profunda necessária para uma experiência de "primeira classe". Ao trazer isso para o core:

- Padronizamos o desenvolvimento móvel em toda a engine.
- Permitimos que os projetos de demonstração oficiais funcionem perfeitamente em dispositivos de toque.
- Fornecemos um caminho otimizado para performance que é difícil de alcançar apenas em GDScript.

## Documentação

- Referência XML completa atualizada para todas as novas classes e eventos em `doc/classes/`.
- Descrições atualizadas para `Input` e `InputMap` para incluir a lógica de roteamento de dispositivos virtuais.
