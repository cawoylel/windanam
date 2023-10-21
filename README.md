# winndam
A multi-dialectal speech recognition models for Fula varieties

## Environement
J'ai pris un espace de stockage 1000GB sur runpod, et j'ai créé une instance avec un gpu utilisant cet espace de stockage.
Une fois cela fait, je me connecte au serveur sur vscode. RunPod a fait un blog sur comment utiliser leurs serveurs sur VSCode:

https://blog.runpod.io/how-to-connect-vscode-to-runpod/?utm_term=&utm_campaign=Serverless+GPU&utm_source=adwords&utm_medium=ppc&hsa_acc=4558579452&hsa_cam=20156995097&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gclid=CjwKCAjw7c2pBhAZEiwA88pOF-hncmlsgnsjWUM4Y1uvi4yaj0ZcKeRTfU2NsHqcHv8jzj1o7FiNmBoCJ8gQAvD_BwE

## Data
### Téléchargement des données
Script: `src/download_dataset.ipynb`
J'ai téléchargé les données de FulaSpeechCorpora une fois, et plus besoin de le refaire.

### Augmentation des données
Scripts: `src/augment.py` (contient juste les fonctions d'augmentation) et `process.py` pour augmenter, processer les données et les enrigistrer seulement une fois.

On aura besoin des audios d'augmentation:

```bash
wget https://www.openslr.org/resources/17/musan.tar.gz
tar -zxvf musan.tar.gz
rm -r ./musan/speech
```


Dès que les données sont processées et téléchargées, je supprime `data/datasets` pour libérer l'espace. Le reste je ne travaille que avec `data/processed`

## Entraînement
Script: `model.py`

Moi j'utilise `tmux` pour lancer l'entraînement et pouvoir me déconnecter sans que celui-ci s'arrête. Installation de tmux:

```bash
apt-get update
apt-get install tmux
```

Créer une nouvelle session tmux:

```bash
tmux new -s training
```

Puis j'ai juste à lancer mon script d'entraînement

```bash
python model.py
```