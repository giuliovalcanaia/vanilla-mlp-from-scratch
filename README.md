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


Para não ficar dúvidas haver um melhor entendimento da disposição da imagem, que originalmente está em formato de matriz e que precisa obrigatoriamente ser transposta para o formato de vetor, abaixo segue uma descrição detalhada de como esta transformação foi feita.


