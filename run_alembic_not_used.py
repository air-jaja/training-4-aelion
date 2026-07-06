import os
from alembic.config import Config
from alembic import command
from dotenv import load_dotenv

# Chargez les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérez les variables d'environnement
db_user = os.getenv("DB_USER", "indusense_user")
db_password = os.getenv("DB_PASSWORD", "ThEP@ssW0rd")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "indusense_db")

# Construisez l'URL de la base de données
db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Chargez la configuration Alembic
alembic_ini_path = os.path.join(os.path.dirname(__file__), "alembic.ini")
alembic_cfg = Config(alembic_ini_path)

# Remplacez l'URL de la base de données dans la configuration
alembic_cfg.set_main_option("sqlalchemy.url", db_url)

# Exécutez Alembic avec les arguments passés en ligne de commande
if __name__ == "__main__":
    command.main(argv=os.sys.argv[1:], config=alembic_cfg)