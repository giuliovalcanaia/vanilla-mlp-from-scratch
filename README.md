# Perceptron
Uma implementação pura e minimalista de uma MLP (Multi-Layer Perceptron) com viés matemático destinada a reconhecer e classificar dígitos manuscritos numa imagem quadrada de $28px$.

# Matemática vs Computação
Redes neurais possuem uma íntima ligação com a álgebra linear, por isso neste primeiro momento vamos concentrar os nossos esforços em entender como essas duas áreas do conhecimento estão conectadas.

# Camada de entrada (Layer 0)
## Imagem
Todas as imagens utilizadas neste estudo seguem o mesmo padrão:
1. Dimensões: $28px \times 28px$
2. Tamanho total: $784$ pixels
3. Cor: escala de cinza

Para que a rede neural consiga ler e interpretar essas imagens, elas são convertidas e guardadas em uma representação matemática que será descrita a seguir. 

## Arquivo de entrada
Para o treinamento e teste da rede, serão utilizados dois datasets disponíveis neste repositório. Neles, cada imagem está armazenada em formato CSV, onde cada linha (exceto o cabeçalho) representa uma imagem individual.

A primeira coluna indica o algarismo representado, enquanto as colunas seguintes contêm os valores dos pixels. Esses valores variam em uma escala contínua entre 0 (preto) e 1 (branco) — com precisão de até 16 casas decimais —, mapeando as tonalidades em escala de cinza.

<div style="text-align: center;">
  <div style="display: inline-grid; grid-template-columns: 30px 400px; grid-template-rows: 20px 400px; gap: 0;">
    <div></div>
    <div style="display: grid; grid-template-columns: repeat(28, 1fr); align-items: end;">
      <div style="text-align: center; font-size: 8px;">1</div>
      <div style="text-align: center; font-size: 8px;">2</div>
      <div style="text-align: center; font-size: 8px;">3</div>
      <div style="text-align: center; font-size: 8px;">4</div>
      <div style="text-align: center; font-size: 8px;">5</div>
      <div style="text-align: center; font-size: 8px;">6</div>
      <div style="text-align: center; font-size: 8px;">7</div>
      <div style="text-align: center; font-size: 8px;">8</div>
      <div style="text-align: center; font-size: 8px;">9</div>
      <div style="text-align: center; font-size: 8px;">10</div>
      <div style="text-align: center; font-size: 8px;">11</div>
      <div style="text-align: center; font-size: 8px;">12</div>
      <div style="text-align: center; font-size: 8px;">13</div>
      <div style="text-align: center; font-size: 8px;">14</div>
      <div style="text-align: center; font-size: 8px;">15</div>
      <div style="text-align: center; font-size: 8px;">16</div>
      <div style="text-align: center; font-size: 8px;">17</div>
      <div style="text-align: center; font-size: 8px;">18</div>
      <div style="text-align: center; font-size: 8px;">19</div>
      <div style="text-align: center; font-size: 8px;">20</div>
      <div style="text-align: center; font-size: 8px;">21</div>
      <div style="text-align: center; font-size: 8px;">22</div>
      <div style="text-align: center; font-size: 8px;">23</div>
      <div style="text-align: center; font-size: 8px;">24</div>
      <div style="text-align: center; font-size: 8px;">25</div>
      <div style="text-align: center; font-size: 8px;">26</div>
      <div style="text-align: center; font-size: 8px;">27</div>
      <div style="text-align: center; font-size: 8px;">28</div>
    </div>
    <div style="display: grid; grid-template-rows: repeat(28, 1fr); justify-items: end; padding-right: 4px;">
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">1</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">2</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">3</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">4</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">5</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">6</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">7</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">8</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">9</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">10</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">11</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">12</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">13</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">14</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">15</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">16</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">17</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">18</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">19</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">20</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">21</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">22</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">23</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">24</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">25</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">26</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">27</div>
      <div style="text-align: center; font-size: 8px; line-height: calc(400px / 28);">28</div>
    </div>
    <div style="position: relative; width: 400px; height: 400px;">
      <img src="assets/linha_002_label_2.png" alt="Exemplo de linha do dataset" style="width: 100%; height: 100%; display: block; image-rendering: pixelated; image-rendering: crisp-edges;">
      <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; background-image: linear-gradient(to right, rgba(0,0,0,0.5) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,0,0,0.5) 1px, transparent 1px); background-size: calc(400px / 28) calc(400px / 28);"></div>
    </div>
  </div>
</div>

## Vetor da imagem
Matematicamente falando, é possível representar e guardar as informações contidas em cada imagem em um único vetor com $784$ dimensões. Aqui vamos chamá-lo de $\mathbf{x}$, um vetor coluna $\mathbf{x} \in \mathbb{R}^{784}$.


Para não ficar dúvidas e haver um melhor entendimento da disposição da imagem, que originalmente está representada em uma matriz e que precisa ser obrigatoriamente convertida para uma representação vetorial, abaixo segue uma descrição detalhada de como esta relação foi feita.

## Matriz original
A imagem de entrada é representada originalmente por uma matriz bidimensional $\mathbf{M} \in \mathbb{R}^{28 \times 28}$, onde $p$ representa cada um dos $784$ pixels da imagem com seu repectivo valor referente à coloração.

$$
\mathbf{M} = \begin{bmatrix}
p_{1,1} & p_{1,2} & \dots & p_{1,28} \\
p_{2,1} & p_{2,2} & \dots & p_{2,28} \\
\vdots & \vdots & \ddots & \vdots \\
p_{28,1} & p_{28,2} & \dots & p_{28,28}
\end{bmatrix}
$$

Para alimentação da layer de entrada, a matriz $\mathbf{M}$ é vetorizada através do mapeamento [ordem por linha](https://pt.wikipedia.org/wiki/Ordem_de_linha_e_de_coluna) (*Row-Major*), gerando o vetor coluna $\mathbf{x} \in \mathbb{R}^{784}$:


$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_{28} \\ \hline
x_{29} \\ x_{30} \\ \vdots \\ x_{56} \\ \hline
\vdots \\ \hline
x_{757} \\ \vdots \\ x_{784}
\end{bmatrix} =
\begin{bmatrix}
p_{1,1} \\ p_{1,2} \\ \vdots \\ p_{1,28} \\ \hline
p_{2,1} \\ p_{2,2} \\ \vdots \\ p_{2,28} \\ \hline
\vdots \\ \hline
p_{28,1} \\ \vdots \\ p_{28,28}
\end{bmatrix}
\begin{array}{l}
\left. \vphantom{\begin{matrix} p_{1,1} \\ p_{1,2} \\ \vdots \\ p_{1,28} \end{matrix}} \right\} \text{Linha 1 da imagem (28 pixels)} \\
\left. \vphantom{\begin{matrix} p_{2,1} \\ p_{2,2} \\ \vdots \\ p_{2,28} \end{matrix}} \right\} \text{Linha 2 da imagem (28 pixels)} \\
\vphantom{\begin{matrix} \vdots \end{matrix}} \\
\left. \vphantom{\begin{matrix} p_{28,1} \\ \vdots \\ p_{28,28} \end{matrix}} \right\} \text{Linha 28 da imagem (28 pixels)}
\end{array}
$$

A relação entre a posição unidimensional $i$ do vetor e os índices de linha $m$ e coluna $n$ da matriz é dada por:

$$i = (m - 1) \cdot 28 + n, \quad \text{para } m, n \in \{1, \dots, 28\}$$

## Termos: computação vs matemática
Uma vez que a imagem é transformada em um vetor-coluna, é possível dizer que esta é a representação computacional da nossa imagem. A única diferença se dá nos termos. Cada uma das coordenadas do vetor assume uma representação de um neurônio dentro dos termos da computação. Por isso, cada neurônio na camada de entrada nada mais é do que uma das coordenadas do vetor. 

Aproveitando que estamos falando de termos da computação, vamos introduzir um conceito chave para a compreensão de redes neurais: o *perceptron*.

### Perceptron
É o modelo mais básico e fundamental de um neurônio artificial. Ele foi proposto em 1958 pelo psicólogo estadunidense Frank Rosenblatt, e serve como o bloco de construção inicial para o que hoje conhecemos como Redes Neurais Artificiais e Aprendizado Profundo (Deep Learning). 
- Percep-: do inglês perception (percepção).

- -tron: sufixo derivado do grego que denota dispositivo, máquina, ferramenta ou dispositivo.

# Camada oculta (Hidden layer)
A camada oculta — ou o conjunto de camadas ocultas — é a estrutura intermediária de uma rede neural situada entre a camada de entrada e a camada de saída. O termo "oculta" deve-se ao fato de que seus valores intermediários não são observados diretamente no dataset de entrada e nem correspondem diretamente ao resultado final desejado (o rotulo/label de saída). 

Sua principal função é detectar padrões simples (bordas, retas e curvas), que serão usadas nas etapas subsequentes para reconhecer estruturas mais complexas, como laços fechados, ou traços verticais que caracterizam números específicos. 

Também se trata de um vetor, que geralmente possui menos dimensões que o vetor de entrada. 

## Teorema da Aproximação Universal
É o teorema que prova que uma única camada oculta com um número suficiente de neurônios pode aproximar qualquer função contínua com a precisão que for desejada. Duas demonstrações são as mais famosas:
- George Cybenko (1989): Provou o teorema para redes neurais usando funções de ativação sigmoides.
- Kurt Hornik (1991): Expandiu a prova mostrando que o poder de aproximação não vem da função de ativação específica (sigmoide, ReLU, etc.), mas sim da própria arquitetura de camadas ocultas aplicada aos dados.

Apesar da possibilidade de usar apenas uma camada oculta, é extremamente comum por questões práticas e de otimização, usarmos uma combinação de várias camadas ocultas. Problemas complexos costumam ser resolvidos de maneiras mais eficientes usando camadas profundas, ao invés de uma única camada extremamente larga. 

## O que temos até aqui?
Vamos criar uma rede com duas camadas ocultas: 
- Primeira: um vetor com 128 dimensões.
- Segunda: um vetor com 64 dimensões.

Esta configuração permite criar uma uma quantidade suficiente de conexões para conseguir uma excelente precisão na deteção de padrões, enquanto ainda permite manter a rede razoalvenmente simples. A disposição da quantidade de dimensões em formato de "funil", permite que a informação seja abstraida e compactada de maneira mais eficiente até chegar a camada final. 

$$\mathbf{x} =  \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_{783} \\ x_{784} \end{bmatrix}_{784 \times 1} \quad \longrightarrow \quad \mathbf{h}^{(1)} =  \begin{bmatrix} h^{(1)}_1 \\ h^{(1)}_2 \\ \vdots \\ h^{(1)}_{127} \\ h^{(1)}_{128} \end{bmatrix}_{128 \times 1} \quad \longrightarrow \quad \mathbf{h}^{(2)} =  \begin{bmatrix} h^{(2)}_1 \\ h^{(2)}_2 \\ \vdots \\ h^{(2)}_{63} \\ h^{(2)}_{64} \end{bmatrix}_{64 \times 1}$$

## Conexão entre camadas: Pesos e Viés
Os vetores de cada camada se conectam por meio de uma transformação. Essa transformação é composta e descrita pelas etapas abaixo.

### Matriz de Pesos: $\mathbf{W}$
Representa a força das conexões entre cada elemento do vetor da camada anterior e cada elemento do vetor atual. A matriz $\mathbf{W}$ tem tamanho $m \times n$ (número de neurônios da camada atual $\times$ número de entradas da camada anterior):


$$\mathbf{W} = \begin{bmatrix}  w_{1,1} & w_{1,2} & \cdots & w_{1,n} \\  w_{2,1} & w_{2,2} & \cdots & w_{2,n} \\  \vdots & \vdots & \ddots & \vdots \\  w_{m,1} & w_{m,2} & \cdots & w_{m,n}  \end{bmatrix}_{m \times n}$$


### Vetor de Viés: $\mathbf{b}$
Um vetor coluna que permite deslocar a função de ativação. O vetor de viés tem $m$ linhas e $1$ coluna, onde $m$ representa a quantidade de elementos do vetor daquela camada. No caso do nosso sistema ele vai ter 3 vetores de viés. 

$$\mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_{m} \end{bmatrix}_{m \times 1}$$

### Função de ativação: $g()$
Uma combinação de transformações lineares aplicadas à um vetor equivale à uma única transformação linear. Para que a rede neural aprenda padrões complexos aplicamos uma função de ativação não linear $g(\cdot)$ elemento a elemento. É essa não linearidade que permite à rede quebrar a limitação linear e aprender padrões complexos, conforme prevê o Teorema da Aproximação Universal.

#### ReLU
A função de ativação ReLU (Rectified Linear Unit, ou Unidade Linear Retificada) é descrita como uma função contínua por partes muito simples: ela retorna o próprio valor da entrada se ele for positivo, e retorna $0$ se for negativo.$$g(z) = \max(0, z)$$Na forma de função definida por partes:$$g(z) = \begin{cases} z & \text{se } z > 0 \\ 0 & \text{se } z \le 0 \end{cases}$$

# Camada de saída (Output Layer)
## Vetor de saída
Como o objetivo é classificar uma base de dados em dígitos de 0 a 9, então a ideia é que o vetor de saída tenha 10 dimensões ($\mathbb{R}^{10 \times 1}$). Cada posição $k$ do vetor representa a probabilidade da imagem ser o dígito $k$. Por isso, o vetor de saída possui uma construção muito semelhante aos vetores das camadas ocultas, com a principal diferença que a função de ativação é a Softmax.

#### Softmax
É uma transformação vetorial em que a saída de cada componente depende da soma de todos os elementos do vetor de pré-ativação $\mathbf{z}$. O objetivo principal da função Softmax é converter um vetor de pontuações brutas (logits) em uma distribuição de probabilidade válida.

$$g(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

- $\mathbf{z}$ (Vetor de Entrada ou Logits): É o vetor contendo todas as pontuações brutas calculadas pela camada anterior da rede neural ($\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$). Ele contém $K$ números reais (positivos, negativos ou zero).
- $z_i$ (Pontuação da Classe de Interesse):É o valor bruto específico pertencente à $i$-ésima classe que você deseja calcular a probabilidade individual.
- $z_j$ (Pontuação de Cada Classe):Representa o valor bruto da $j$-ésima classe dentro do somatório. O $j$ é apenas o índice que percorre todas as classes possíveis, de $1$ até $K$.
- $K$ (Número Total de Classes):É a quantidade total de opções ou categorias possíveis no problema. No nosso caso como estamos classificando dígitos de $0$ a $9$, $K = 10$.
- $i$ (Índice da Classe Requisitada):Identifica qual posição do vetor de saída está sendo calculada (onde $i \in \{1, 2, \dots, K\}$).


# Cálculo de Forward Propagation
Ao partir do vetor $x$ podemos chegar nos vetores subsequentes por meio da aplicação de duas etapas de cálculo:

1. 



# Referências
- [But what is a neural network? | Chapter 1, Deep learning](https://youtu.be/aircAruvnKk?si=O2XsR2W0H3gZzcDp)
- GONÇALVES, Juliana Brassolatti. Álgebra linear. Batatais: Claretiano, 2014. 183 p. ISBN 978-85-8377-357-3.
- GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. MIT Press, 2016. (Disponível gratuitamente em [deeplearningbook.org](https://www.deeplearningbook.org)).
