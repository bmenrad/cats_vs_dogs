# Cats vs Dogs Klassifikationsprojekt

Dieses Projekt ist ein Python-basiertes Machine-Learning-Projekt, das ein **ResNet-Modell (TensorFlow)** verwendet, um Bilder von Katzen und Hunden zu klassifizieren. Zusätzlich enthält das Projekt eine **Web-App**, die das Modell für Vorhersagen über einen Browser verfügbar macht.

## Voraussetzungen

- Python 3.10 oder höher
- pip
- Git
- Git LFS
- Optional: Virtuelle Umgebung (`venv`) empfohlen

---

## Schritt-für-Schritt Installation

### 1. Repository klonen

    git clone https://github.com/bmenrad/cats_vs_dogs.git

    cd cats_vs_dogs

### 2. Virtuelle Umgebung erstellen (optional, empfohlen)

    python3 -m venv .venv

    source .venv/bin/activate

### 3. Git LFS installieren

    sudo apt update

    sudo apt install git-lfs

    git lfs install

### 4. Abhängigkeiten installieren

    pip install --upgrade pip

    pip install -r requirements.txt
#### Das Verzeichnis data/ muss Trainings- und Validierungsbilder enthalten, sortiert in cats/ und dogs/.

#### Die Funktion organize_dataset() in train_tf_model.py kann Rohdaten automatisch sortieren.
-----------------------------------------------------------------------------------------
### Modell trainieren

    python train_tf_model.py

#### Trainiert das TensorFlow ResNet-Modell mit den Daten in data/.

#### Speichert das Modell nach dem Training im Verzeichnis saved_resnet/.

#### Zeigt während des Trainings den Fortschritt und den Loss an.
<br>

### Web-App starten

    python app.py

#### Browser öffnen und zu http://localhost:5000 navigieren.

#### Bilder hochladen, um Vorhersagen zu erhalten (Cat oder Dog).

#### Die App verwendet das gespeicherte Modell in saved_resnet/.
-----------------------------------------------------------------------------------------
<br>

### GitHub Hinweise

#### Große Dateien (Modell) werden über Git LFS verwaltet (saved_resnet/variables/).
#### Virtuelle Umgebung (.venv/) ist in .gitignore enthalten und wird nicht hochgeladen.
<br>

### Lizenz

#### Dieses Projekt ist frei verwendbar.

<div style="text-align:center;">
  <img src="https://ben.forfiles.de/wp-content/uploads/2025/11/cats_vs_dogs-logo.png" 
       alt="Man vs Machine Logo" 
       width="400" />
</div>

<br>
