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

## Point de vigilance associé — version Python pinnée vs `requires-python`

Le `pyproject.toml` fourni contraint le projet à :

```toml
requires-python = ">=3.11,<3.12"
```

Si un `uv python pin 3.12` a été fait sur la machine (pour un autre projet, ou par erreur), `uv` tentera de résoudre l'environnement avec Python 3.12, **incompatible avec cette contrainte** (`<3.12` exclut strictement 3.12). Cela peut provoquer des erreurs de résolution supplémentaires, potentiellement dans le même style que l'erreur `cv2` ci-dessus (message `No solution found... markers: python_full_version == '3.11.*'` qui laisse deviner que uv teste plusieurs versions candidates).

**Vérification** :

```bash
uv python pin        # affiche la version actuellement épinglée pour ce projet
cat .python-version   # si le fichier existe, montre la version pinnée localement
```

**Solution si le pin ne correspond pas à `requires-python`** :

```bash
uv python pin 3.11
```

Puis régénérer l'environnement proprement (cf. section suivante si `.venv` refuse de se supprimer sous Windows).

---

## Procédure de resynchronisation complète recommandée

Une fois les points ci-dessus vérifiés :

```bash
uv python pin 3.11
uv venv          # recréer le venv si nécessaire (voir dépannage Windows .venv verrouillé)
uv sync --group dl --group ml --group dev
```

Vérification finale dans un notebook ou un script :

```python
import cv2, tensorflow, albumentations, skimage, PIL
print("cv2:", cv2.__version__)
print("tensorflow:", tensorflow.__version__)
print("albumentations:", albumentations.__version__)
print("scikit-image:", skimage.__version__)
print("pillow:", PIL.__version__)
```
