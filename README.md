# BenaHdp

Software desktop para apoio ao trabalho com imagens biometricas, alinhamento de impressao digital, deteccao de minucias, matching visual e exportacao/importacao de resultados em ZIP.

## Estrutura

- `Aplicativo/`: versao empacotada para Windows, com o executavel e instrucoes de uso.
- `Código Fonte/`: script Python, recursos visuais/sonoros, exemplos e dependencias.
- `LICENSE`: licenca do projeto.

## Executar pelo aplicativo

Abra:

```powershell
Aplicativo\BenaHdp.exe
```

O executavel ja inclui as bibliotecas e os recursos necessarios.

## Executar pelo codigo fonte

```powershell
cd "Código Fonte"
python -m pip install -r requirements.txt
python Benahdp.py
```

## Principais recursos

- Carregamento de imagem base e imagem sobreposta.
- Alinhamento por deslocamento, rotacao, zoom e transparencia.
- Deteccao e edicao visual de minucias.
- Visualizacao de direcoes, grafos e matching entre pontos.
- Salvamento e carregamento de sessoes em arquivo `.zip`.

## Pastas de recursos

- `Complementos/`: sons e imagens auxiliares.
- `Fotos/`: imagens institucionais usadas pela interface.
- `Imagens/`: modelos visuais de padroes biometricos.
- `Biometric/`: imagens de exemplo/apoio para testes.
- `Exemplos/`: arquivo ZIP de exemplo gerado pelo aplicativo.
