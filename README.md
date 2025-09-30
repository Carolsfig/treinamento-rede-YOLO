# Detecção de Objetos com YOLOv8: Meu Projeto de Treinamento Customizado

## Resumo do Projeto
Neste projeto, eu demonstro o processo completo de treinamento e validação de um modelo de detecção de objetos utilizando a arquitetura **YOLOv8**. O objetivo foi realizar **Transfer Learning** (Aprendizado por Transferência), partindo de um modelo YOLOv8 pré-treinado com o dataset COCO e especializando-o para detectar as 80 classes de um dataset público sobre objetos diversos, obtido através do Roboflow Universe.

Todo o workflow, desde a preparação dos dados até a inferência em novas imagens, foi executado por mim no ambiente **Google Colab**.

## Tecnologias que Utilizei
- **Python 3**
- **PyTorch**
- **Ultralytics YOLOv8**
- **Roboflow** (para gerenciamento e exportação do dataset)
- **Google Colab** (para treinamento com GPU)
- **Dataset Público do Roboflow Universe** (para o fine-tuning do modelo pré-treinado em COCO)

---

##  Workflow do Projeto

Eu dividi o projeto em quatro etapas principais para organizar o desenvolvimento:

### 1. Preparação do Dataset
Para este projeto, eu utilizei um dataset público que continha imagens com diversos objetos do cotidiano, incluindo as classes que eu tinha interesse em detectar, como `person` e `car`.

Diferente de uma abordagem de filtragem, eu optei por utilizar o dataset completo, com todas as suas classes originais. Meu objetivo com isso foi treinar um modelo capaz de identificar os objetos de interesse em um ambiente com maior variedade visual.

Para organizar e preparar estes dados para o treinamento, eu utilizei a plataforma **Roboflow**, onde realizei as seguintes etapas:

1.  **Upload e Organização:** Fiz o upload do dataset para a plataforma.
2.  **Divisão dos Dados:** Configurei o Roboflow para dividir automaticamente o conjunto de dados em pastas de **treino, validação e teste** (na proporção 70%/20%/10%), o que é uma prática essencial para avaliar o modelo de forma justa.
3.  **Exportação para o formato YOLOv8:** ExporteI o dataset já no formato exigido pelo YOLOv8. Essa etapa criou a estrutura de pastas e o arquivo `data.yaml` necessários, agilizando muito o início do treinamento.


### 2. Configuração do Ambiente e Treinamento
Para o treinamento, eu escolhi o **Google Colab** para ter acesso a uma GPU gratuita, acelerando o processo.

A técnica central que apliquei foi o **Transfer Learning**:
-   **Modelo Base:** Eu iniciei o projeto com o `yolov8n.pt`, um modelo leve e já pré-treinado no COCO. Isso me permitiu aproveitar o conhecimento que o modelo já possuía sobre formas e texturas básicas.
-   **Ajuste Fino (Fine-Tuning):** Em seguida, eu retreinei esse modelo com meu dataset customizado. Durante este processo, ajustei os pesos da rede para que ela se especializasse em detectar apenas as minhas classes de interesse.

O comando principal que utilizei para o treinamento foi:
```python
# Carreguei o modelo pré-treinado e iniciei o treinamento
model = YOLO('yolov8n.pt')
results = model.train(data='meu_dataset/data.yaml', epochs=25, imgsz=640)
```

### 3. Validação dos Resultados
Após o treinamento, o melhor modelo foi salvo como `best.pt`. Para garantir a qualidade do meu trabalho, eu avaliei a performance do modelo com base nas métricas geradas pela biblioteca `ultralytics`, como a matriz de confusão e as curvas de Precisão-Recall.

### 4. Inferência em Novas Imagens
Com meu modelo treinado e validado, a etapa final foi testá-lo em imagens que ele nunca tinha visto antes. Para isso, criei um script que carrega os pesos do `best.pt` e aplica a detecção, salvando a imagem resultante com as caixas delimitadoras e os rótulos.

O código abaixo foi o que usei para garantir que a detecção fosse executada e o resultado salvo corretamente:
```python
# Carreguei o meu modelo treinado
model = YOLO('runs/detect/train/weights/best.pt')

# Executei a predição e forcei o salvamento em uma pasta específica
results = model.predict(
    source='imagem_de_teste.jpg',
    save=True,
    project='runs/detect',
    name='predict_results',
    exist_ok=True
)
```

---

## 🚀 Como Executar o Projeto

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
    ```
2.  **Abra no Google Colab:** Faça o upload do arquivo `.ipynb` para o Google Colab e habilite o ambiente com GPU.
3.  **Configure sua Chave de API do Roboflow:**
    * Este projeto utiliza o gerenciador de Secrets do Colab para proteger a chave de API. Você precisará da sua própria chave do [Roboflow](https://roboflow.com/).
    * No Colab, clique no ícone de chave (🔑) na barra lateral esquerda.
    * Crie um novo secret com o nome `ROBOFLOW_API_KEY`.
    * No campo "valor", cole a sua chave de API privada do Roboflow.
4.  **Execute as Células:** Rode as células do notebook em ordem.

---

##  Conclusão e Próximos Passos

Com este projeto, demonstrei com sucesso a eficácia do Transfer Learning com YOLOv8 para criar um detector de objetos customizado. Aprendi na prática como gerenciar um pipeline de visão computacional, desde o tratamento dos dados até a avaliação do modelo final.

**Ideias para o futuro:**
-   Aumentar o número de épocas para um treinamento mais longo.
-   Testar um modelo base maior (ex: `yolov8m.pt`) para buscar mais performance.
-   Expandir o projeto para detectar mais classes de objetos.
-   Fazer o deploy do modelo em uma aplicação web ou em um dispositivo embarcado.
