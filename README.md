# Detecção de Objetos com YOLOv8: Meu Projeto de Treinamento Customizado

## Resumo do Projeto
Neste projeto, eu demonstro o processo completo de treinamento e validação de um modelo de detecção de objetos utilizando a arquitetura **YOLOv8**. Meu objetivo foi realizar **Transfer Learning** (Aprendizado por Transferência) a partir de um modelo pré-treinado, especializando-o para detectar um conjunto customizado de classes que selecionei do dataset COCO.

Todo o workflow, desde a preparação dos dados com o **Roboflow** até a inferência em novas imagens, foi executado por mim no ambiente **Google Colab**.

## Tecnologias que Utilizei
- **Python 3**
- **PyTorch**
- **Ultralytics YOLOv8**
- **Roboflow** (para preparação e anotação do dataset)
- **Google Colab** (para treinamento com GPU)
- **Dataset COCO** (como base para o meu conjunto de dados)

---

##  workflow-do-projeto Meu Workflow de Trabalho

Eu dividi o projeto em quatro etapas principais para organizar o desenvolvimento:

### 1. Preparação do Dataset com Roboflow
Percebi que o dataset COCO original era muito grande e continha muitas classes desnecessárias para o meu objetivo. Por isso, utilizei a plataforma **Roboflow** para otimizar este processo. Nela, eu:
1.  **Filtrei o Dataset:** Selecionei apenas as classes do meu interesse (`person` e `car`).
2.  **Dividi os Dados:** Configurei a plataforma para dividir automaticamente o conjunto de dados em treino, validação e teste (70%/20%/10%), um passo que considerei crucial para um treinamento de qualidade.
3.  **Exportei no Formato Correto:** Exportei o dataset já no formato específico do YOLOv8, o que me poupou muito tempo e evitou erros de formatação.

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

##  Como Executar o Projeto

Para replicar meu trabalho, siga os passos abaixo:

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
    ```
2.  **Abra no Google Colab:** Faça o upload do arquivo `.ipynb` para o Google Colab e certifique-se de habilitar o ambiente com GPU.
3.  **Prepare seu Dataset:** O snippet de download do Roboflow é único. Para rodar, você precisará criar sua própria conta, gerar seu dataset e colar seu código de download na célula correspondente.
4.  **Execute as Células:** Rode as células do notebook em ordem.

---

##  Conclusão e Próximos Passos

Com este projeto, demonstrei com sucesso a eficácia do Transfer Learning com YOLOv8 para criar um detector de objetos customizado. Aprendi na prática como gerenciar um pipeline de visão computacional, desde o tratamento dos dados até a avaliação do modelo final.

**Ideias para o futuro:**
-   Aumentar o número de épocas para um treinamento mais longo.
-   Testar um modelo base maior (ex: `yolov8m.pt`) para buscar mais performance.
-   Expandir o projeto para detectar mais classes de objetos.
-   Fazer o deploy do modelo em uma aplicação web ou em um dispositivo embarcado.
