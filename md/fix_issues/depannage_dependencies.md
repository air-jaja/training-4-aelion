# Dépannage installation Deep Learning — groupe `dl` (uv)

Ce guide compile les erreurs rencontrées lors de l'ajout des dépendances de deep learning (`tensorflow`, `albumentations`, `opencv`, `scikit-image`, `pillow`) au projet `aelion-4-trainning`, avec `uv`.

**Contexte** : `uv`, Windows.

---

## Erreur 1 — `uv add cv2` : `No solution found when resolving dependencies`

```
uv add cv2 --group dl
  × No solution found when resolving dependencies for split (markers: python_full_version == '3.11.*' and
  │ sys_platform == 'win32'):
  ╰─▶ Because there are no versions of cv2 and aelion-4-trainning:dl depends on cv2, we can conclude that
      aelion-4-trainning:dl's requirements are unsatisfiable.
```

**Cause** : `cv2` n'est **pas un nom de package PyPI**. C'est uniquement le nom du module Python qu'on importe (`import cv2`) une fois OpenCV installé. Le package à installer sur PyPI s'appelle `opencv-python` (avec interface graphique) ou `opencv-python-headless` (sans dépendances GUI, recommandé en environnement serveur/CI/notebook).

**Vérification** : `cv2` n'existe tout simplement pas dans l'index PyPI (`https://pypi.org/project/cv2/` → 404).

**Solution** : ne pas ajouter `cv2`, utiliser le vrai nom de package.

```bash
uv add opencv-python-headless --group dl
```

**Dans ce projet précis** : `opencv-python-headless>=4.9` est **déjà présent** dans le groupe `dl` du `pyproject.toml`. Aucune action n'est donc nécessaire — il suffit de synchroniser l'environnement :

```bash
uv sync --group dl
```

Puis, dans le code :

```python
import cv2  # le nom d'import reste "cv2", même si le package s'appelle "opencv-python-headless"
```

**Règle générale** : le nom du package PyPI (`uv add ...`, `pip install ...`) et le nom du module importé (`import ...`) ne sont pas toujours identiques. Quelques cas fréquents :

| `import ...` | Package PyPI à installer |
|---|---|
| `import cv2` | `opencv-python` ou `opencv-python-headless` |
| `import sklearn` | `scikit-learn` |
| `import skimage` | `scikit-image` |
| `import PIL` / `from PIL import Image` | `pillow` |
| `import yaml` | `pyyaml` |

En cas de doute, chercher le package sur [pypi.org](https://pypi.org) plutôt que de deviner à partir du nom d'import.

---

## Point vérifié — `uv python pin` / `requires-python`

Vérification faite : `.python-version` = `3.11`, `uv python pin` = `3.11`, `python --version` = `3.11.15`. Tout est cohérent, ce n'était pas la source du problème. (Point initialement soulevé par erreur — pas de souci réel ici.)

---

## Erreur 2 — `ModuleNotFoundError: No module named 'cv2'` dans le notebook (VS Code)

```
----> 6 import cv2
ModuleNotFoundError: No module named 'cv2'
```

Cette erreur apparaît **après** avoir corrigé le nom du package (`opencv-python-headless` bien présent dans `pyproject.toml`, groupe `dl`), ce qui indique que le problème n'est **pas** un package manquant dans le projet, mais un **kernel Jupyter qui ne pointe pas vers le bon interpréteur**.

**Ce qui a pu se passer** : la commande `uv add cv2` a échoué (voir Erreur 1 — `cv2` n'existe pas sur PyPI), donc **aucune dépendance n'a été ajoutée ni installée** suite à cette commande. Le `ModuleNotFoundError` observé dans le notebook n'est donc pas une conséquence de cet échec, mais un problème préexistant : soit `opencv-python-headless` n'a jamais été synchronisé dans le venv, soit le notebook utilise un autre interpréteur Python que celui du projet.

### Étape 1 — Vérifier que le groupe `dl` est bien installé dans le venv du projet

```bash
uv sync --group dl
```

Puis vérifier directement via `uv run` (bypass tout problème de kernel) :

```bash
uv run python -c "import cv2; print(cv2.__version__)"
```

- Si ça fonctionne ici → le package est bien installé dans `.venv`, le problème vient du kernel utilisé par le notebook (étape 2).
- Si ça échoue encore ici → le `dl` n'a pas été synchronisé, relancer `uv sync --group dl --group ml --group dev` (ou `uv sync --all-groups`) et vérifier qu'aucune erreur ne s'affiche.

### Étape 2 — Vérifier le kernel sélectionné dans VS Code / Jupyter

Cause très fréquente : le notebook utilise un kernel Python différent du venv du projet (Python système, autre venv, base conda...).

- Dans VS Code : en haut à droite du notebook, cliquer sur le sélecteur de kernel → choisir l'interpréteur du projet, normalement :
  ```
  .venv\Scripts\python.exe   (chemin complet : E:\Projets\_workspace\aelion-4-trainning\.venv\Scripts\python.exe)
  ```
- Si ce kernel n'apparaît pas dans la liste, il faut d'abord enregistrer le venv comme kernel Jupyter (le groupe `dev` du projet contient déjà `ipykernel`) :
  ```bash
  uv run python -m ipykernel install --user --name aelion-4-trainning --display-name "Python (aelion-4-trainning)"
  ```
  Puis sélectionner `Python (aelion-4-trainning)` dans le sélecteur de kernel de VS Code, et redémarrer le kernel du notebook.
- Vérification directe dans une cellule du notebook, pour confirmer quel interpréteur est réellement utilisé :
  ```python
  import sys
  print(sys.executable)
  ```
  Le chemin affiché doit correspondre à `.venv\Scripts\python.exe` du projet. S'il pointe ailleurs (ex. `AppData\Local\Programs\Python\...` ou un autre venv), c'est confirmé : mauvais kernel sélectionné.

### Point de vigilance — pas de dégâts causés par `uv add cv2`

Comme `uv add cv2` a échoué **avant** toute modification effective (uv ne modifie `pyproject.toml`/`uv.lock` qu'après une résolution réussie), il n'y a normalement rien à annuler. Vérification par sécurité :

```bash
git diff pyproject.toml uv.lock
```

Si `git diff` ne montre aucune trace de `cv2`, le fichier est resté propre et il n'y a rien à corriger sur ce point.

---

## Erreur 3 — `ModuleNotFoundError: No module named 'sklearn'`

```
----> 11 from sklearn.model_selection import train_test_split
ModuleNotFoundError: No module named 'sklearn'
```

**Cause** : `scikit-learn` est déclaré dans le groupe `ml` du `pyproject.toml`, pas dans le groupe `dl`. Le notebook de prétraitement (`bottle_preprocessing.ipynb`) utilise `train_test_split` pour le split train/validation, mais seul `uv sync --group dl` avait été exécuté jusque-là → le groupe `ml` n'avait jamais été synchronisé dans le venv.

**Solution rapide** — synchroniser aussi le groupe `ml` :

```bash
uv sync --group dl --group ml
```

**Solution plus propre** — le split train/validation fait partie du prétraitement, pas de l'entraînement ML classique (`xgboost`, `optuna`, etc. du groupe `ml`). Autant ajouter `scikit-learn` au groupe `dl` directement, pour que le pipeline de prétraitement/deep learning ne dépende pas du groupe `ml` :

```bash
uv add scikit-learn --group dl
```

**Vérification** :

```bash
uv run python -c "import sklearn; print(sklearn.__version__)"
```

---

Une fois les points ci-dessus vérifiés :

```bash
uv venv          # recréer le venv si nécessaire (voir dépannage Windows .venv verrouillé)
uv sync --group dl --group ml --group dev
```

Vérification finale dans un notebook ou un script :

```python
import cv2, tensorflow, albumentations, skimage, PIL, sklearn
print("cv2:", cv2.__version__)
print("tensorflow:", tensorflow.__version__)
print("albumentations:", albumentations.__version__)
print("scikit-image:", skimage.__version__)
print("pillow:", PIL.__version__)
print("scikit-learn:", sklearn.__version__)
```
