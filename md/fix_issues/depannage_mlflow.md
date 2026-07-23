# Dépannage installation MLflow — compatibilité Python / protobuf / setuptools

Ce guide compile les erreurs rencontrées lors de l'installation de MLflow avec `uv` et leurs solutions, dans l'ordre où elles apparaissent typiquement.

**Contexte** : `uv`, Windows, Python 3.14.x puis repli sur 3.13.5.

---

## 0) Installation de base

```bash
uv add mlflow
```

---

## Erreur 1 — `ModuleNotFoundError: No module named 'distutils'`

```
File ".../mlflow/store/artifact/local_artifact_repo.py", line 1, in <module>
    import distutils.dir_util as dir_util
ModuleNotFoundError: No module named 'distutils'
```

**Cause** : `distutils` a été retiré de la bibliothèque standard à partir de Python 3.12. Certains modules internes de MLflow l'importent encore.

**Solution** : installer `setuptools`, qui fournit un shim `distutils` compatible.

```bash
uv add setuptools
```

---

## Erreur 2 — `ImportError: cannot import name 'service' from 'google.protobuf'`

```
File ".../mlflow/protos/service_pb2.py", line 11, in <module>
    from google.protobuf import service as _service
ImportError: cannot import name 'service' from 'google.protobuf'
```

**Cause** : le module `google.protobuf.service` a été retiré dans les versions récentes de `protobuf` (5.x+), alors que les fichiers `_pb2.py` compilés dans MLflow en dépendent encore.

**Solution** : épingler `protobuf` à une version antérieure à 5.

```bash
uv add "protobuf<5"
```

---

## Erreur 3 — `TypeError: Metaclasses with custom tp_new are not supported`

```
File ".../google/protobuf/internal/api_implementation.py", line 51, in <module>
    if _CanImport('google._upb._message'):
TypeError: Metaclasses with custom tp_new are not supported.
```

**Cause** : incompatibilité réelle entre l'extension C compilée de `protobuf` (`google._upb._message`) et **Python 3.14** — la metaclasse utilisée par cette extension n'est plus supportée par les nouvelles règles du C API de 3.14. Ce n'est pas un problème de version de `protobuf` mais de compatibilité de son extension native avec Python 3.14 (problème connu, déjà remonté côté `protobuf`/Red Hat).

**Deux solutions possibles** :

**Option A — contourner l'extension C (rapide, reste en Python 3.14)**

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
python -m mlflow --version
```

⚠️ Force l'implémentation Python pure de `protobuf` (plus lente, mais négligeable pour du tracking MLflow). À définir avant chaque lancement (notebook, serveur MLflow), ou à ajouter à un fichier `.env` chargé automatiquement par le projet.

**Option B — revenir temporairement à Python 3.13 (plus durable)**

```bash
uv python install 3.13.5
```

Si `pyproject.toml` fixe `requires-python = ">=3.14"`, l'assouplir d'abord :

```toml
[project]
requires-python = ">=3.13"
```

Puis :

```bash
uv python pin 3.13.5
```

Si le `.venv` existant ne peut pas être supprimé (`rmdir` refusé, fichier `python.exe` verrouillé) : fermer tous les terminaux/kernels Jupyter/IDE utilisant ce venv, vérifier qu'aucun processus ne le retient encore :

```powershell
Get-Process python | Where-Object { $_.Path -like "*<nom_du_projet>*" } | Stop-Process -Force
Remove-Item -Recurse -Force .\.venv\
```

Puis recréer l'environnement et réinstaller les dépendances :

```bash
uv sync
uv add mlflow "protobuf<5" setuptools
```

---

## Erreur 4 — `ModuleNotFoundError: No module named 'pkg_resources'`

```
File ".../mlflow/utils/autologging_utils/versioning.py", line 6, in <module>
    from pkg_resources import resource_filename
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause** : `pkg_resources` faisait partie de `setuptools`, mais a été **complètement retiré à partir de `setuptools` 82.0.0** (8 février 2026). Un `uv add setuptools` sans contrainte de version installe la dernière version, dépourvue de `pkg_resources`, ce qui casse MLflow qui en dépend encore en interne.

**Solution** : épingler `setuptools` à une version antérieure à 82.

```bash
uv add "setuptools<82"
```

---

## Avertissement résiduel (non bloquant) — dépréciation de `pkg_resources`

```
UserWarning: pkg_resources is deprecated as an API. ... Refrain from using this package or pin to Setuptools<81.
```

C'est un simple avertissement, pas une erreur : MLflow s'importe et fonctionne normalement. Il disparaîtra lorsque MLflow migrera vers `importlib.metadata`/`importlib.resources` (recommandation officielle de `setuptools`).

---

## Erreur 5 — `No module named mlflow.__main__; 'mlflow' is a package and cannot be directly executed`

```bash
python -m mlflow --version
# ...
# No module named mlflow.__main__; 'mlflow' is a package and cannot be directly executed
```

**Cause** : ce n'est pas une erreur d'installation. MLflow ne fournit pas de point d'entrée `__main__.py`, donc `python -m mlflow` ne fonctionne pas — contrairement à d'autres outils (`python -m pip`, par exemple).

**Solution** : utiliser directement la commande CLI installée par le package.

```bash
mlflow --version
```

Ou, dans le contexte `uv` :

```bash
uv run mlflow --version
```

---

## Erreur 6 — `AttributeError: 'Connection' object has no attribute 'connect'` (lancement du serveur)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```
```
File ".../mlflow/store/db_migrations/env.py", line 72, in run_migrations_online
    with connectable.connect() as connection:
AttributeError: 'Connection' object has no attribute 'connect'. Did you mean: 'connection'?
```

**Cause** : SQLAlchemy 2.0 a supprimé le support des connexions "branchées" — un objet `Connection` ne peut plus appeler `.connect()` pour créer une sous-connexion (autorisé sous SQLAlchemy 1.x). Le script de migration Alembic utilisé en interne par cette version de MLflow suppose encore ce comportement SQLAlchemy 1.x.

**Solution** : épingler `sqlalchemy` à une version antérieure à 2.0.

```bash
uv add "sqlalchemy<2"
```

⚠️ Si un fichier `mlflow.db` partiellement créé existe déjà suite à l'échec précédent, le supprimer avant de relancer pour repartir d'une base propre :

```powershell
Remove-Item .\mlflow.db
```

Puis relancer le serveur :

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

---

## Récapitulatif — commandes finales qui fonctionnent

```bash
# Environnement (si retour à 3.13 nécessaire à cause de l'erreur protobuf/Python 3.14)
uv python pin 3.13.5

# Dépendances avec versions compatibles épinglées
uv add mlflow
uv add "setuptools<82"
uv add "protobuf<5"
uv add "sqlalchemy<2"

# Vérification
uv run mlflow --version

# Lancement du serveur de tracking
uv run mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Si le projet reste en **Python 3.14**, ajouter également le contournement protobuf :

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
uv run mlflow --version
```

## Résumé des causes

| Erreur | Cause | Fix |
|---|---|---|
| `No module named 'distutils'` | `distutils` retiré depuis Python 3.12 | `uv add setuptools` |
| `cannot import name 'service' from 'google.protobuf'` | `protobuf` 5.x+ trop récent pour MLflow | `uv add "protobuf<5"` |
| `Metaclasses with custom tp_new are not supported` | Extension C de `protobuf` incompatible Python 3.14 | `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` ou repli Python 3.13 |
| `No module named 'pkg_resources'` | `setuptools` 82+ a retiré `pkg_resources` | `uv add "setuptools<82"` |
| `No module named mlflow.__main__` | MLflow n'a pas de point d'entrée `-m` | Utiliser `mlflow --version` (pas `python -m mlflow`) |
| `'Connection' object has no attribute 'connect'` | SQLAlchemy 2.0 a supprimé les connexions branchées, migration Alembic de MLflow suppose SQLAlchemy 1.x | `uv add "sqlalchemy<2"` |