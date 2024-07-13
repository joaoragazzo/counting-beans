# Contador de Feijões

Como projeto da disciplina de processamento de imagens, foi necessário construir um algoritmo que, através de manipulação da imagem, fosse
capaz de identificar e contar a quantidade de feijões na imagem. 

Durante o desenvolvimento, foram utilizadas diversas técnicas para atingir esse objetivo, tais como abertura, limiarização, função distância,
entre outras técnicas. 

Como forma de complementar o trabalho, também implementei uma forma de visualizar a contagem de feijões, através do arquivo `output.pgm`.

## Como utilizar

Para utilizar é bastante simples: basta executar o código passando como parâmetro a imagem que você deseja que seja processada.

```bash
python ./contafeijao.py <caminho_do_arquivo>
```

## Demonstração 

Uma imagem de entrada poderia ser, por exemplo:

<img src="/image_input.png">

E, por fim, a mensagem de saída é a seguinte:

<img src="/image_output.png">
