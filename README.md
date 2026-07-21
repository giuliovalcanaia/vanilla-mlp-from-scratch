# Vanilla Multi-Layer Perceptron from scratch
Uma implementação pura e minimalista de uma MLP (Multi-Layer Perceptron) destinada a reconhecer e classificar dígitos manuscritos numa imagem quadrada de $28px$.

# Matemática vs Computação
Redes neurais possuem uma íntima ligação com a álgebra linear, por isso neste primeiro momento vamos concentrar os nossos esforços em entender como essas duas áreas do conhecimento estão conectadas.

# Primeira layer
## Imagem de entrada
Todas as imagens utilizadas neste estudo seguem o mesmo padrão:
1. Dimensões: $28px \times 28px$
2. Preenchimento: em escala de cinza, onde $0$ representa o preto e $1$ o branco. E entre $0$ e $1$, os tons de cinza. 
3. Tamanho total: $784$ pixels

## Arquivo de entrada
Para treino e teste serão utilizados dois [datasets](https://pt.wikipedia.org/wiki/Conjunto_de_dados) que podem ser encontrados neste repositório. Eles estão em formato csv, e a representação da escala de cinza está disposta em valores que vão do 0 até o 255, totalizando assim 256 valores possíveis. Para fins didáticos, ressaltamos que, ao serem convertidos em vetores, esses valores sofrerão uma normalização: a escala original, que varia de 0 a 255, será convertida para uma escala contínua entre 0 e 1.

## Vetor da imagem
Matematicamente falando, é possível representar e guardar todas as informações continas em cada imagem do nosso estudo em um único vetor com $784$ dimensões. Aqui vamos chamá-lo de vetor $x$, com $784 \times 1$ dimensões.


Para não ficar dúvidas e haver um melhor entendimento da disposição da imagem, que originalmente está em formato de matriz e que precisa obrigatoriamente ser transposta para o formato de vetor, abaixo segue uma descrição detalhada de como esta transformação foi feita.

## Matriz original
A imagem de entrada é representada originalmente por uma matriz bidimensional $\mathbf{M} \in \mathbb{R}^{28 \times 28}$ com $784$ pixels:

$$
\mathbf{M} = \begin{bmatrix}
a_{1,1} & a_{1,2} & \dots & a_{1,28} \\
a_{2,1} & a_{2,2} & \dots & a_{2,28} \\
\vdots & \vdots & \ddots & \vdots \\
a_{28,1} & a_{28,2} & \dots & a_{28,28}
\end{bmatrix}
$$

Para alimentação da layer de entrada, a matriz $\mathbf{M}$ é vetorizada através do mapeamento [ordem por linha](https://pt.wikipedia.org/wiki/Ordem_de_linha_e_de_coluna) (*Row-Major*), gerando o vetor coluna $\mathbf{v} \in \mathbb{R}^{784 \times 1}$:


$$
\mathbf{v} =
\begin{bmatrix}
v_1 \\ v_2 \\ \vdots \\ v_{28} \\ \hline
v_{29} \\ v_{30} \\ \vdots \\ v_{56} \\ \hline
\vdots \\ \hline
v_{757} \\ \vdots \\ v_{784}
\end{bmatrix} =
\begin{bmatrix}
a_{1,1} \\ a_{1,2} \\ \vdots \\ a_{1,28} \\ \hline
a_{2,1} \\ a_{2,2} \\ \vdots \\ a_{2,28} \\ \hline
\vdots \\ \hline
a_{28,1} \\ \vdots \\ a_{28,28}
\end{bmatrix}
\begin{array}{l}
\left. \vphantom{\begin{matrix} p_{1,1} \\ p_{1,2} \\ \vdots \\ p_{1,28} \end{matrix}} \right\} \text{Linha 1 da imagem (28 pixels)} \\
\left. \vphantom{\begin{matrix} p_{2,1} \\ p_{2,2} \\ \vdots \\ p_{2,28} \end{matrix}} \right\} \text{Linha 2 da imagem (28 pixels)} \\
\vphantom{\begin{matrix} \vdots \end{matrix}} \\
\left. \vphantom{\begin{matrix} p_{28,1} \\ \vdots \\ p_{28,28} \end{matrix}} \right\} \text{Linha 28 da imagem (28 pixels)}
\end{array}
$$

Onde a relação entre a posição unidimensional $i$ do vetor e os índices de linha $m$ e coluna $n$ da matriz é dada por:

$$i = (m - 1) \cdot 28 + n, \quad \text{para } m, n \in \{1, \dots, 28\}$$

# Referências
- [But what is a neural network? | Chapter 1, Deep learning](https://youtu.be/aircAruvnKk?si=O2XsR2W0H3gZzcDp)
- GONÇALVES, Juliana Brassolatti. Álgebra linear. Batatais: Claretiano, 2014. 183 p. ISBN 978-85-8377-357-3.
