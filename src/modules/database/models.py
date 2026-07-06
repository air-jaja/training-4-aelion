from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Date, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# Base class for SQLAlchemy models
Base = declarative_base()

# ======================
# BRONZE LAYER (Raw Data)
# ======================
class IncidentBronze(Base):
    __tablename__ = "incidents_bronze"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    shift = Column(String)
    machine_id = Column(String)
    severity = Column(String)
    surchauffe = Column(String)
    baisse_pression = Column(String)
    vibration = Column(String)
    bruit_mecanique = Column(String)
    surconsommation = Column(String)
    blocage_mecanique = Column(String)
    alarme_capteur = Column(String)
    arret_urgence = Column(String)
    defaut_qualite = Column(String)
    comment = Column(String)
    flag_ok = Column(String)

class TelemetryBronze(Base):
    __tablename__ = "telemetry_bronze"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    machine_id = Column(String)
    temperature_c = Column(String)
    pressure_bar = Column(String)
    voltage_mean_v = Column(String)
    rotation_mean_rpm = Column(String)
    pieces_produced = Column(String)
    flag_ok = Column(String)

# ======================
# REFERENTIAL DATA
# ======================
class Machine(Base):
    __tablename__ = "machine"
    machine_code = Column(String(16), primary_key=True)
    commissioning_date = Column(Date, nullable=False)
    max_daily_capacity = Column(Integer, nullable=False)
    max_hourly_capacity_pieces = Column(Integer, nullable=False)
    model = Column(String(32), nullable=False)
    production_line = Column(String(16), nullable=False, index=True)
    location = Column(String(16), nullable=False, index=True)
    criticality = Column(String(8), nullable=False)  # 'LOW', 'MEDIUM', 'HIGH'
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Maintenance(Base):
    __tablename__ = "maintenance"
    maintenance_id = Column(Integer, primary_key=True, index=True)
    machine_code = Column(String(16), ForeignKey("machine.machine_code"), nullable=False, index=True)
    maintenance_at = Column(DateTime(timezone=True), nullable=False)
    maintenance_type = Column(String, nullable=False)  # 'preventive', 'reactive', etc.
    action_type = Column(String, nullable=False)  # 'changement_suite_panne', etc.
    component = Column(String, nullable=False)  # 'joint hydraulique', etc.
    description = Column(Text, nullable=False)
    related_incident_id = Column(String)  # Référence à un incident (ex: 'INC-000426')
    duration_hours = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
