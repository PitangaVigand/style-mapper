# Da Imagem Urbana a Inteligencia Computacional: Traducao de Padroes Arquitetonicos em Dados Urbanos

**Keywords:** API, patrimonio arquitetonico, smart cities, machine learning

## Introducao

O avanco das tecnologias digitais no setor AEC tem ampliado o uso de dados urbanos, sensores e inteligencia artificial, especialmente no contexto das smart cities. No entanto, essas abordagens frequentemente negligenciam a dimensao qualitativa da forma urbana, reduzindo a cidade a fluxos de dados e infraestruturas tecnologicas. Em contraste, autores como Kevin Lynch destacam que a experiencia urbana e estruturada por elementos perceptivos como imagem, legibilidade e identidade espacial, fundamentais para a construcao do significado urbano (Lynch, 1960).

Paralelamente, avancos significativos em aprendizado de maquina demonstram a possibilidade de identificar estilos arquitetonicos e padroes urbanos por meio de imagens, utilizando machine learning aplicado a bases como Google Street View (Sun et al., 2022; Xu et al., 2023; Wei et al., 2018). Essas abordagens indicam um potencial ainda pouco explorado: a identificacao das formas arquitetonicas em dados computacionais capazes de classificar estilos arquitetonicos em um contexto urbano.

Esta pesquisa investiga se ferramentas acessiveis de machine learning sao capazes de classificar padroes arquitetonicos com nivel de consistencia comparavel a leitura especializada, articulando imagem urbana, dados geoespaciais e cartografia digital. O objetivo e explorar o potencial dessas tecnologias como base para o desenvolvimento de ferramentas de leitura urbana, contribuindo para novas formas de inteligencia territorial.

## Metodologia

A metodologia esta estruturada em tres etapas: treinamento do modelo, coleta de dados urbanos e processamento geoespacial.

### 1. Treinamento do modelo

Na primeira etapa, e realizada a curadoria de um conjunto de imagens representativas de estilos arquitetonicos especificos. Esses dados sao utilizados para treinar um modelo de classificacao utilizando Teachable Machine (Carney et al., 2020), explorando sua viabilidade como ferramenta acessivel de aprendizado supervisionado.

### 2. Coleta de dados urbanos

Na segunda etapa, sao coletadas imagens urbanas por meio de scripts em Python utilizando a API do Google Street View. A coleta e orientada por coordenadas geograficas e metadados urbanos, permitindo a construcao de uma base espacial estruturada. Os dados sao organizados em formato GeoJSON, viabilizando sua integracao em sistemas de informacao geografica (IETF, 2016).

### 3. Processamento geoespacial e classificacao

Na terceira etapa, o modelo treinado e aplicado as imagens coletadas para identificar e classificar padroes arquitetonicos no territorio. Os resultados sao georreferenciados e exportados como dados cartograficos, possibilitando sua visualizacao e analise em escala urbana.

## Resultados

Os resultados preliminares indicam que modelos treinados com ferramentas acessiveis sao capazes de identificar padroes arquitetonicos com consistencia significativa. A integracao entre classificacao visual e dados geoespaciais permite mapear a distribuicao territorial de estilos arquitetonicos, evidenciando relacoes entre forma urbana e localizacao.

Espera-se, como desdobramento, o desenvolvimento de um prototipo de ferramenta de leitura urbana inspirado em plataformas digitais como ImagineRio, integrando classificacao automatizada e visualizacao cartografica.

## Discussao

A pesquisa contribui para o debate contemporaneo sobre o papel da inteligencia artificial no design ao deslocar seu uso da geracao formal para a interpretacao da cidade existente. O trabalho investiga como sistemas computacionais podem ampliar a capacidade humana de leitura urbana, atuando como mediadores entre imagem, dados e territorio.

Ao integrar visao computacional, Python e dados urbanos, a pesquisa propoe uma abordagem que articula percepcao qualitativa e analise quantitativa, respondendo as limitacoes das abordagens tradicionais de smart cities. O uso de ferramentas acessiveis sugere caminhos para democratizacao do acesso a analise urbana baseada em dados, com aplicacoes potenciais em planejamento participativo, preservacao do patrimonio e cartografias criticas.

Por fim, o trabalho reforca a necessidade de reposicionar o design computacional como pratica critica, capaz de integrar conhecimento historico e inovacao tecnologica.

## Estrutura do Projeto

Este repositorio organiza o fluxo principal da pesquisa em tres frentes:

- `gen_street_view_images_by_street/`: scripts para coleta de imagens urbanas a partir da API do Google Street View.
- `image_classifier/`: Aplicacao do modelo de classificacao arquitetonica.


## Como Executar

O fluxo abaixo foi validado para este projeto e evita os conflitos de versao encontrados com o modelo `keras_model.h5`.

### 1. Criar e ativar o ambiente virtual

Use Python `3.11`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

O `python --version` deve retornar `3.11.x`.

### 2. Instalar as dependencias do classificador

Se preferir instalar as bibliotecas a partir do arquivo de dependencias do repositorio, rode:

```powershell
python -m pip install -r requirements.txt
```

Instale exatamente estas versoes-base para o pipeline de classificacao:

```powershell
python -m pip install --upgrade pip
python -m pip uninstall -y keras tensorflow tf-keras numpy
python -m pip install --no-cache-dir "numpy<2" "tensorflow==2.15.1" "keras<2.16,>=2.15.0" pillow pytest shapely h5py
```

Essa combinacao e importante porque:

- `tensorflow==2.15.1` funciona corretamente com Python `3.11`
- `numpy<2` evita incompatibilidades binarias com TensorFlow `2.15`
- `keras 2.15.x` evita conflitos com `keras 3.x` ao abrir o modelo legado `keras_model.h5`

### 3. Rodar o teste do classificador isoladamente

Para a coleta de Street View funcionar, crie um arquivo `.env` em `Projects/style mapper/gen_street_view_images_by_street/.env` com a variavel:

```env
STREETVIEW_API_KEY=seu_token_aqui
```

Como este repositorio contem outros projetos com dependencias diferentes, execute o teste do classificador diretamente em vez de rodar a descoberta global do `pytest`:

```powershell
python -m pytest "Projects/style mapper/image_classifier/test_identify_image.py::test_classify_image" -q
```

Para rodar os dois testes do classificador:

```powershell
python -m pytest "Projects/style mapper/image_classifier/test_identify_image.py" -q
```

### 4. Observacao sobre o modelo legado

O classificador usa um modelo Teachable Machine salvo em formato HDF5 (`keras_model.h5`). O loader em `image_classifier/test_identify_image.py` aplica um pequeno patch de compatibilidade durante a leitura do arquivo para remover campos legados rejeitados por loaders mais novos.

## Referencias

Carney, M., Webster, B., Alvarado, I., Phillips, K., Howell, N., Griffith, J., Jongejan, J., Pitaru, A., & Chen, A. (2020). *Teachable Machine: Approachable web-based tool for exploring machine learning classification*. Extended Abstracts of the 2020 CHI Conference on Human Factors in Computing Systems (CHI EA '20). https://doi.org/10.1145/3334480.3382839

Cullen, G. (1961). *The concise townscape*. Architectural Press.

Internet Engineering Task Force (IETF). (2016). *The GeoJSON format (RFC 7946)*. https://doi.org/10.17487/RFC7946

Lynch, K. (1960). *The image of the city*. MIT Press.

Rice University, Spatial Studies Lab. (n.d.). *ImagineRio: A digital atlas*. https://imaginerio.org

Sun, M., Zhang, F., Duarte, F., & Ratti, C. (2022). *Understanding architecture age and style through deep learning*. *Cities, 128*, 103787. https://doi.org/10.1016/j.cities.2022.103787

Wei, X., Yu, K., Wang, Y., & Zhang, H. (2018). Referencia citada no resumo do projeto como base para reconhecimento visual e classificacao de padroes arquitetonicos.

Xu, H., Sun, H., Wang, L., Yu, X., & Li, T. (2023). *Urban architectural style recognition and dataset construction method under deep learning of street view images: A case study of Wuhan*. *ISPRS International Journal of Geo-Information, 12*(7), 264. https://doi.org/10.3390/ijgi12070264
#   s t y l e - m a p p e r  
 