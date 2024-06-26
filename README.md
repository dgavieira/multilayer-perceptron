# Multilayer Perceptron

Este é um projeto de implementação de duas arquiteturas do Multilayer Perceptron. O objetivo é realizar a classificação quanto à existência ou não de câncer de mamas em amostras da base de Dados Mammographic Dataset, segundo o sistema Bi-RADS.
Este código trata-se da implementação do segundo trabalho da disciplina Aprendizado de Máquina do curso de mestrado em Engenharia Elétrica pelo Programa de Pós-Graduação em Engenharia Elétrica da Universidade Federal do Amazonas.

* Tema: Regressão Linear com Adaline
* Disciplina: PGENE 556 - Aprendizado de Máquina
* PPGEE - Programa de Pós Graduação em Engenharia Elétrica
* UFAM - Universidade Federal do Amazonas
* Autor: Diego Giovanni de ALcântara Vieira.

## Instalação

1. Clone este repositório:
```
git clone https://github.com/dgavieira/multilayer-perceptron
```

1. Construa a imagem

```
docker build -t perceptron -f Dockerfile .
```

## Uso

1. Execute o arquivo `main.py` para treinar o modelo e gerar os resultados.

```
docker run -it --rm -v $PWD:/tmp -w /tmp perceptron python ./script.py
```


## Contribuição

Contribuições são bem-vindas! Para sugestões, abra uma issue. Para mudanças significativas, abra um pull request.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.