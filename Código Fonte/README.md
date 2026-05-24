# BenaHdp - Codigo Fonte

Esta pasta contem o codigo Python e os recursos usados pelo aplicativo.

## Requisitos

- Python 3.10 ou superior.
- Dependencias listadas em `requirements.txt`.

## Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

## Executar

```powershell
python Benahdp.py
```

Execute o comando a partir desta pasta para manter os caminhos dos recursos organizados.

## Gerar executavel Windows

```powershell
python -m PyInstaller --noconfirm --onefile --windowed --name BenaHdp --add-data "Complementos;Complementos" --add-data "Fotos;Fotos" --add-data "Imagens;Imagens" --add-data "Biometric;Biometric" Benahdp.py
```

Depois de gerar, coloque `dist\BenaHdp.exe` na pasta `..\Aplicativo`.

## Conteudo

- `Benahdp.py`: arquivo principal do aplicativo.
- `Complementos/`: recursos sonoros e imagens auxiliares.
- `Fotos/`: imagens institucionais.
- `Imagens/`: padroes biometricos usados na interface.
- `Biometric/`: imagens de apoio.
- `Exemplos/`: arquivos de exemplo para teste.
