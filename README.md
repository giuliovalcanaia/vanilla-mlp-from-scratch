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
Para treino e teste serão utilizados dois [datasets](https://pt.wikipedia.org/wiki/Conjunto_de_dados) que podem ser encontrados neste repositório. Cada imagem está salva dentro deste dataset em formato csv. As linhas (com excessão da primeira que é o cabeçalho) representa uma imagem. A primeira coluna corresponde ao algarismo que está sendo representado, e as demais colunas são os valores dos pixels, sendo este um valor entre 0 (preto) e 1 (branco), podendo assumir qualquer valor entre esses limites (com uma precisão de 16 casas decimais, no caso deste estudo) para representar uma escala de cinza.

## Vetor da imagem
Matematicamente falando, é possível representar e guardar as informações continas em cada imagem em um único vetor com $784$ dimensões. Aqui vamos chamá-lo de vetor $v$, com $784 \times 1$ dimensões.


Para não ficar dúvidas e haver um melhor entendimento da disposição da imagem, que originalmente está representada em uma matriz e que precisa ser obrigatoriamente convertida para uma representação vetorial, abaixo segue uma descrição detalhada de como esta relação foi feita.

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

Para alimentação da layer de entrada, a matriz $\mathbf{M}$ é vetorizada através do mapeamento [ordem por linha](https://pt.wikipedia.org/wiki/Ordem_de_linha_e_de_coluna) (*Row-Major*), gerando o vetor coluna $\mathbf{v} \in \mathbb{R}^{784 \times 1}$, que aqui chamaremo de $\mathbf{v}^{(0)}$ por se tratar do vetor de entrada:


$$
\mathbf{v}^{(0)} =
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

## Nosso caso
Vamos criar uma rede com duas camadas ocultas: 
- Primeira: um vetor com 128 dimensões.
- Segunda: um vetor com 64 dimensões.

Esta configuração permite criar uma uma quantidade suficiente de conexões para conseguir uma excelente precisão na deteção de padrões, enquanto ainda permite manter a rede razoalvenmente simples. 

## Conexão entre camadas: Pesos e Viés
Os vetores de cada camada se conectam por meio de uma transformação linear. Essa transformação é composta e descrita pelas etapas abaixo.

### Matriz de Pesos($\mathbf{W}$)
Representa a força das conexões entre cada coordena do vetor da camada anterior e cada coordenada do vetor atual. 

- Vetor de Viés($\mathbf{b}$): Um vetor coluna que permite deslocar a função de ativação.
- Função de ativação: Uma combinação de transformações lineares aplicadas à um vetor equivale à uma única transformação linear. Para que a rede neural aprenda padrões complexos 



# Referências
- [But what is a neural network? | Chapter 1, Deep learning](https://youtu.be/aircAruvnKk?si=O2XsR2W0H3gZzcDp)
- GONÇALVES, Juliana Brassolatti. Álgebra linear. Batatais: Claretiano, 2014. 183 p. ISBN 978-85-8377-357-3.
