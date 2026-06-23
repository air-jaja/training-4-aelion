import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base

# Base class for SQLAlchemy models
Base = declarative_base()


# Model for anonymized incidents data
class Incident(Base):
    __tablename__ = "incidents"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    date = sqlalchemy.Column(sqlalchemy.DateTime)
    shift = sqlalchemy.Column(sqlalchemy.String)
    machine_id = sqlalchemy.Column(sqlalchemy.String)
    severity = sqlalchemy.Column(sqlalchemy.Integer)

    # Dynamic columns for failure types (type_*)
    # We'll handle these dynamically in the database_loader module

# Model for telemetry data
class Telemetry(Base):
    __tablename__ = "telemetry"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime)
    machine_id = sqlalchemy.Column(sqlalchemy.String)
    temperature_c = sqlalchemy.Column(sqlalchemy.Float)
    pressure_bar = sqlalchemy.Column(sqlalchemy.Float)
    voltage_mean_v = sqlalchemy.Column(sqlalchemy.Float)
    rotation_mean_rpm = sqlalchemy.Column(sqlalchemy.Float)
    pieces_produced = sqlalchemy.Column(sqlalchemy.Integer)
