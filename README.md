# VisionRad: 3D Tumor Morphology Reconstruction & Tracking Platform
> **High-Performance Medical Imaging Architecture: Physics-Informed Ray Marching, Biaxial Back-Projection, Vulkan Compute, and Graph-Driven Spectral Deconvolution.**

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue.svg)](#)
[![Standards](https://img.shields.io/badge/DICOM-3.0%20(CT%20%7C%20CR%20%7C%20SEG)-green.svg)](#)
[![HPC](https://img.shields.io/badge/Core-C11%20%7C%20OpenMP%20%7C%20SIMD-orange.svg)](#)
[![GPU Compute](https://img.shields.io/badge/GPU-Vulkan%201.1%20%7C%20GLSL%20Compute-red.svg)](#)
[![Math](https://img.shields.io/badge/Algorithms-FFT%20%7C%20Ram--Lak%20%7C%20Graph%20Theory-purple.svg)](#)

---

##  Contexto & Propósito deste Repositório

> **Aviso de Propriedade Intelectual & Portfólio Técnico**:  
> Este repositório público foi estruturado como **portfólio técnico e arquitetural de engenharia**. O código-fonte integral, modelos matemáticos proprietários e dados de ensaios pré-clínicos são mantidos em repositório privado para proteção de propriedade intelectual. Este documento descreve de forma conceitual a **arquitetura de sistemas, escolhas teóricas, pipelines de computação de alto desempenho (HPC) e decisões de engenharia de software** adotadas no projeto **VisionRad**.

---

##  Visão Geral do Problema Clínico & Engenharia

O acompanhamento oncológico convencional por **Tomografia Computadorizada (TC 3D)** expõe o paciente a doses elevadas de radiação ionizante acumulada, além de apresentar alto custo hospitalar e limitações de agendamento frequente. Por outro lado, **radiografias digitais biaxiais convencionais (CR/DX frontal e lateral)** oferecem baixíssima dose de radiação e rápida aquisição, mas fornecem apenas projeções bidimensionais com perda de profundidade volumétrica.

O **VisionRad** aborda esse desafio clínico através de um framework de **Reconstrução Morfológica 3D Iterativa (SMT)**:
1. Ingestão da **Tomografia Computadorizada basal** e da máscara volumétrica segmentada do tumor (**DICOM SEG**).
2. Aquisição de apenas duas radiografias ortogonais de controle (**Frontal e Lateral**) no retorno clínico do paciente.
3. Síntese de radiografias digitalmente reconstruídas (**DRR - Digitally Reconstructed Radiographs**) de alta performance, simulando o transporte físico de radiação diretamente no volume tridimensional.
4. Identificação de divergências de atenuação e aplicação de **retroprojeção ortogonal biaxial com restrição volumétrica cruzada**, atualizando a geometria 3D tumoral e a distribuição radiodensa do tecido.
5. Resolução de casos complexos de **múltiplas lesões tumorais sobrepostas** por meio de **modelagem por Grafos de Oclusão** e **deconvolução espectral passa-alta no domínio da frequência (FFT 2D com filtro de rampa de Ram-Lak)**.
6. Exportação em conformidade estrita com o padrão **DICOM CT e DICOM SEG multi-frame**, viabilizando a reinjeção direta em estações de trabalho e sistemas PACS hospitalares.

---

##  Arquitetura Geral do Sistema

```
                  ┌────────────────────────────────────────┐
                  │    Hospital PACS / VNA Network         │
                  └──────────────────┬─────────────────────┘
                                     │ C-STORE / C-MOVE (TCP:11112)
                                     ▼
                  ┌────────────────────────────────────────┐
                  │         Módulo 1: c-dcm                │
                  │   DICOM Storage SCP/SCU (PyNetDICOM)   │
                  └──────────────────┬─────────────────────┘
                                     │ DICOM CT / SEG / CR
                                     ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                       manager.py (Orquestrador)                    │
    └──────────────┬─────────────────────────────────────┬───────────────┘
                   │                                     │
                   ▼                                     ▼
    ┌──────────────────────────────┐     ┌──────────────────────────────┐
    │    Módulo 2: core (C/CPU)    │     │  Módulo 3: gpu_core (Vulkan) │
    │  • Monte Carlo Ray Marching  │     │  • Headless Compute Pipeline │
    │  • OpenMP SIMD Batch (x8)    │ ou  │  • GLSL Ray Marching Shader  │
    │  • Klein-Nishina / Compton   │     │  • SSBO 16-bit CT Storage    │
    │  • Retroprojeção Biaxial 3D  │     │  • Dispatch Paralelo 2048²   │
    └──────────────┬───────────────┘     └──────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      │ Projeções & Diferenças 2D
                                      ▼
                   ┌──────────────────────────────────────┐
                   │     Módulo 4: tummor_mapping         │
                   │  • Detecção de Múltiplos Tumores     │
                   │  • Grafo de Alinhamento Oclusivo     │
                   │  • FFT 2D + Filtro Rampa (Ram-Lak)   │
                   │  • Separação Espectral de Manchas    │
                   └──────────────────┬───────────────────┘
                                      │ Morfologia 3D Atualizada
                                      ▼
                   ┌──────────────────────────────────────┐
                   │    DICOM Synthesis & Exporter        │
                   │  • Série CT Sintética Reconstruída   │
                   │  • DICOM SEG 1-bit Packed Multi-Frame│
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼ C-STORE SCU
                  ┌────────────────────────────────────────┐
                  │    Retorno ao PACS / Workstation       │
                  └────────────────────────────────────────┘
```

---

##  Módulos do Sistema

### 1. `c-dcm` — Gateway de Interoperabilidade PACS (DICOM Networking)
Responsável pela comunicação de rede e integração direta com a infraestrutura de diagnóstico por imagem do hospital.
- **Protocolo de Rede Médica**: Implementa um serviço **DICOM Storage SCP (Service Class Provider)** sobre TCP/IP nativo (porta padrão `11112`).
- **SOP Classes Suportadas**:
  - `CTImageStorage`
  - `ComputedRadiographyImageStorage`
  - `DigitalXRayImageStorageForPresentation`
  - `DigitalXRayImageStorageForProcessing`
- **Ingestão e Organização Canônica**: Processamento assíncrono de cabeçalhos de metadados, realizando a triagem estruturada dos exames de acordo com a hierarquia clínica padrão: Paciente, Estudo, Série e Instância.
- **Canal Bidirecional**: Estruturado para atuar também como **SCU (Service Class User)** para devolução dos volumes tomográficos reconstruídos e relatórios de segmentação ao PACS.

---

### 2. `core` — Motor Físico C Otimizado & Reconstrução Morfológica Biaxial
Núcleo computacional combinando modelagem física analítica em **C11 de baixo nível** e lógica de morfologia espacial em **Python**.

#### A. Simulação DRR por Ray Marching com Transporte Radiativo
Em vez de simplificações baseadas apenas em atenuação linear contínua, o traçador emprega uma abordagem física inspirada em princípios estocásticos de transporte de fótons:
- **Modelo de Interação Fóton-Matéria**:
  - Consideração explícita de **absorção fotoelétrica** combinada com **espalhamento incoerente de Compton**.
  - Emprego da formulação de **Klein-Nishina** para calcular a redistribuição relativística de energia e o espalhamento angular tridimensional sobre base ortonormal perpendicular à trajetória do feixe.
- **Engenharia de Baixo Nível & HPC**:
  - Paralelismo com **OpenMP** em grade de projeção de alta resolução ($2048 \times 2048$ detectores).
  - Estruturas de dados em memória C alinhadas estritamente a 32 bytes para máxima eficiência de acesso a registradores e alinhamento de cache.
  - Processamento em lotes contíguos de múltiplos raios simultâneos, reduzindo branches e facilitando auto-vetorização de hardware (SIMD).
  - Gerador de números pseudoaleatórios baseado em registradores locais de deslocamento (Xorshift), evitando overhead de chamadas de sistema ou bibliotecas externas.

#### B. Retificação Geométrica Anti-Cone Beam
Em radiografias clínicas, a geometria divergente do feixe cônico produz distorções de magnificação dependentes da profundidade da anatomia em relação ao ponto focal:
- **Correção Analítica de Magnificação**: O sistema estima a ampliação geométrica baseando-se na distância foco-detector e na posição do centro de massa tridimensional da lesão.
- **Normalização Espacial**: As radiografias clínicas de controle sofrem retificação anisotrópica antes da comparação direta com as projeções ortogonais virtuais.

#### C. Retroprojeção Biaxial & Evolução Morfológica 3D
- **Diferenciação com Filtragem Estocástica de Ruído**: Mapeamento diferencial entre radiografia virtual e clínica real, com isolamento dos picos de variação por limiarização dinâmica adaptada à variância estatística de fundo.
- **Envelope Geométrico por Tubos Ortogonais**: Geração de tubos de extrusão volumétrica correspondentes às projeções frontal e lateral. A interseção booleana 3D dos dois feixes define um envelope espacial estrito (*bounding envelope*), impedindo dispersões anatômicas indevidas.
- **Morfologia Tridimensional Direcionada**: Aplicação de operadores morfológicos volumétricos de expansão e erosão, condicionados exclusivamente aos voxels validados pela colisão ortogonal dos dois eixos.
- **Reatribuição de Densidade Tomográfica**: Voxels validados na expansão assumem a densidade mediana característica da lesão, enquanto voxels erodidos são restaurados à densidade radiológica do tecido pulmonar adjacente.

#### D. Exportador DICOM CT & SEG
- Síntese de novas séries axiais de tomografia computadorizada, preservando integralmente o sistema de coordenadas de referência espacial do paciente.
- Exportação de máscaras volumétricas no padrão **DICOM SEG multi-frame**, garantindo conformidade com o empacotamento binário de 1-bit exigido por visualizadores clínicos e sistemas de radioterapia.

---

### 3. `gpu_core` — Aceleração em GPU via Vulkan Compute & GLSL
Módulo dedicado a descarregar o processamento intensivo de raios e a reconstrução para arquitetura de GPU massivamente paralela.
- **Pipeline Vulkan Compute Puro (Headless)**:
  - Arquitetura focada estritamente em **filas de computação assíncrona** (`Compute Queues`), operando sem dependências de janelas, swapchains ou renderização gráfica tradicional.
- **Gerenciamento de Memória de Alta Eficiência**:
  - Utilização de buffers de transferência intermediários (*staging buffers*) mapeáveis pelo host e transferência assíncrona para buffers de alta performance em VRAM local da GPU.
  - Habilitação de extensões de armazenamento de 16 bits em GLSL, permitindo que volumes tomográficos densos sejam alocados em sua representação inteira nativa, reduzindo significativamente o consumo de VRAM.
- **Compute Shader GLSL de Ray Marching**:
  - Organização de *workgroups* locais otimizada para ocupação máxima de núcleos de sombreamento da GPU.
  - Execução paralela da física de atenuação, transporte de energia e espalhamento Compton diretamente em registradores gráficos.
- **Status do Módulo**: *Operacional*, com infraestrutura de alocação de buffers, despacho de pipeline e shaders principais integrados.

---

### 4. `tummor_mapping` — Grafo de Alinhamento & Deconvolução Espectral (FFT + Ram-Lak)
Em pacientes oncológicos apresentando **múltiplas lesões simultâneas ou metástases**, dois ou mais tumores podem compartilhar trajetórias de projeção nos detectores frontal ou lateral, gerando sobreposições de sombra que confundiriam algoritmos convencionais de segmentação.

#### A. Grafo Direcionado de Oclusão Espacial
- **Isolamento de Componentes Conexos**: Detecção e rotulagem individualizada de cada massa tumoral presente no volume de referência.
- **Varredura de Sombra Projetada**: Análise cumulativa das trajetórias de projeção ao longo dos eixos anatômicos independentes, identificando geometricamente se a sombra de um nódulo intercepta outro nódulo no plano de visão do detector.
- **Construção do Grafo de Oclusão**: Modela os tumores como vértices e as relações de interceptação como arestas direcionadas, determinando explicitamente quais lesões requerem desambiguação em cada eixo de projeção.

#### B. Deconvolução Espectral no Domínio da Frequência
Quando o grafo indica sobreposição entre múltiplas massas:
- **Condicionamento de Contorno**: Isolamento da região de interesse 2D com extensão de bordas para atenuação de artefatos de descontinuidade circular (*spectral leakage*).
- **Transformação para o Domínio da Frequência (FFT 2D)**: Mapeamento do sinal de diferença bidimensional para o espectro de frequências espaciais via Transformada Rápida de Fourier.
- **Filtragem Rampa Passa-Alta (Filtro de Ram-Lak)**:
  - Aplicação do filtro rampa característico de métodos de retroprojeção filtrada (FBP) em tomografia.
  - **Decisão Teórica**: O filtro atenua as componentes de baixa frequência associadas ao fundo difuso e à sobreposição ampla de sombras, enquanto amplifica os gradientes de alta frequência que demarcam as bordas e contornos individuais de cada nódulo.
- **Restauração e Atribuição Morfológica**: A transformada inversa recupera a projeção filtrada no espaço espacial, permitindo desacoplar as contribuições de densidade e atribuir com precisão a evolução dimensional à lesão correta.

---

### 5. `manager.py` — Orquestrador Central
Ponto de controle e coordenação global do sistema, responsável por conectar os fluxos de trabalho:
- Monitoramento de eventos de rede do servidor DICOM.
- Pré-processamento e verificação de integridade dos volumes e séries clínicas recebidas.
- Avaliação de carga e despacho dinâmico entre o motor de CPU (`core`) e o motor de GPU (`gpu_core`).
- Acionamento condicional do analisador de grafos e deconvolução espectral em exames multiloculares.
- Controle iterativo de convergência morfológica baseado em tolerância de resíduos de atenuação.
- Exportação automatizada dos exames sintéticos e despacho seguro para o repositório clínico.

---

##  Matriz de Status dos Módulos

| Módulo | Tecnologia Principal | Responsabilidade | Status |
|---|---|---|:---:|
| `c-dcm` | Python / PyNetDICOM | Servidor PACS C-STORE SCP/SCU e persistência clínica | **Operacional** |
| `core` | C11 (OpenMP) + Python | Ray Marching Monte Carlo, anti-cone, retroprojeção morfológica | **Operacional** |
| `tummor_mapping` | Python / SciPy / FFT | Grafo de oclusão de múltiplos tumores e filtro Ram-Lak | **Operacional** |
| `gpu_core` | Vulkan 1.1 / GLSL Compute | Ray Marching paralelo em GPU e aceleração via VRAM | **Operacional** |
| `manager.py` | Python / Async IO | Orquestração end-to-end e despacho dinâmico de recursos | *Em progresso (threads)* |

---

##  Destaques de Engenharia & Performance

- **Alinhamento e Estruturação de Memória**: Estruturas de dados em C alinhadas a 32 bytes, prevenindo desalinhamentos em instruções vetorizadas e otimizando o prefetching de linhas de cache.
- **Otimização de Largura de Banda de VRAM**: Emprego de tipos inteiros de 16 bits na GPU para acomodar matrizes tomográficas completas, reduzindo pela metade a demanda de tráfego de memória em comparação a floats de 32 bits.
- **Física de Transporte Radiativo em Tempo de Execução**: Modelagem física realista incorporando espalhamento Compton relativístico (Klein-Nishina) e atenuação fotoelétrica diretamente no loop de integração espacial.
- **Robustez com Padrões Médicos**: Suporte defensivo a descompactação de máscaras DICOM SEG de 1-bit, correção de geometrias poligonais ocas por fechamento topológico tridimensional e conversão precisa de metadados radiométricos para unidades Hounsfield (HU).

---

##  Stack Tecnológica

- **Linguagens**: C (C11), Python (3.11+), GLSL (Vulkan Shading Language).
- **APIs de Hardware & Gráficos**: Vulkan SDK 1.1+, OpenMP, ctypes.
- **Processamento de Imagens & Dados Médicos**: PyDICOM, PyNetDICOM, HighDICOM, SciPy, NumPy.
- **Computação Científica**: Fast Fourier Transform (FFT 2D), Teoria dos Grafos, Morfologia Matemática N-Dimensional.

---

##  Contato & Desenvolvimento

Desenvolvido como projeto de engenharia de software de alta performance e computação em imagens médicas. Para discussões técnicas, parcerias de pesquisa ou oportunidades profissionais:
- **Autor**: *Heitor Fernandes Silva*
- **E-mail**: *heitor.fernandes.cba@gmail.com*
