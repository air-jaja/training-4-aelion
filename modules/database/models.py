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
    related_incident_id = Column(String)  # Reference to an incident (e.g., 'INC-000426')
    duration_hours = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# ======================
# GOLD LAYER (Processed Data)
# ======================
class GoldDataset(Base):
    """
    SQLAlchemy model for the gold_dataset table.
    This table contains processed and enriched data for machine learning and analysis.
    It is loaded from the 'gold_dataset.csv' file in the INPUT_DATA_DIR.
    """
    __tablename__ = "gold_dataset"
    
    # Primary key (auto-incremented)
    id = Column(Integer, primary_key=True, index=True)
    
    # Machine identifier
    machine_id_std = Column(String(50))
    
    # Time window columns
    window_start = Column(String(50))
    window_end = Column(String(50))
    
    # 1-hour window features
    temp_mean_1h = Column(Float)
    temp_max_1h = Column(Float)
    pressure_mean_1h = Column(Float)
    pressure_max_1h = Column(Float)
    voltage_mean_1h = Column(Float)
    voltage_max_1h = Column(Float)
    rotation_mean_1h = Column(Float)
    rotation_max_1h = Column(Float)
    pieces_produced_sum_1h = Column(Integer)
    
    # 6-hour window features
    temp_mean_6h = Column(Float)
    temp_max_6h = Column(Float)
    temp_std_6h = Column(Float)
    pressure_mean_6h = Column(Float)
    pressure_max_6h = Column(Float)
    pressure_std_6h = Column(Float)
    voltage_mean_6h = Column(Float)
    voltage_max_6h = Column(Float)
    voltage_std_6h = Column(Float)
    rotation_mean_6h = Column(Float)
    rotation_max_6h = Column(Float)
    rotation_std_6h = Column(Float)
    
    # 12-hour window features
    temp_mean_12h = Column(Float)
    temp_max_12h = Column(Float)
    temp_std_12h = Column(Float)
    pressure_mean_12h = Column(Float)
    pressure_max_12h = Column(Float)
    pressure_std_12h = Column(Float)
    voltage_mean_12h = Column(Float)
    voltage_max_12h = Column(Float)
    voltage_std_12h = Column(Float)
    rotation_mean_12h = Column(Float)
    rotation_max_12h = Column(Float)
    rotation_std_12h = Column(Float)
    
    # 24-hour window features
    temp_mean_24h = Column(Float)
    temp_max_24h = Column(Float)
    temp_std_24h = Column(Float)
    pressure_mean_24h = Column(Float)
    pressure_max_24h = Column(Float)
    pressure_std_24h = Column(Float)
    voltage_mean_24h = Column(Float)
    voltage_max_24h = Column(Float)
    voltage_std_24h = Column(Float)
    rotation_mean_24h = Column(Float)
    rotation_max_24h = Column(Float)
    rotation_std_24h = Column(Float)
    
    # Trend features (6-hour)
    temp_trend_6h = Column(Float)
    pressure_trend_6h = Column(Float)
    voltage_trend_6h = Column(Float)
    rotation_trend_6h = Column(Float)
    
    # Z-score features
    temp_zscore_24h = Column(Float)
    temp_zscore_machine = Column(Float)
    pressure_zscore_machine = Column(Float)
    
    # Delta features (1-hour and 3-hour)
    temp_delta_1h = Column(Float)
    temp_delta_3h = Column(Float)
    pressure_delta_1h = Column(Float)
    pressure_delta_3h = Column(Float)
    rotation_delta_1h = Column(Float)
    rotation_delta_3h = Column(Float)
    voltage_delta_1h = Column(Float)
    voltage_delta_3h = Column(Float)
    
    # Production features
    pieces_produced_sum_24h = Column(Integer)
    capacity_utilization_pct = Column(Float)
    
    # Incident features (1-hour, 24-hour, 7-day)
    incident_count_1h = Column(Integer)
    incident_max_severity_1h = Column(Integer)
    incident_count_prev_24h = Column(Integer)
    incident_max_severity_prev_24h = Column(Integer)
    incident_count_prev_7d = Column(Integer)
    hours_since_last_incident = Column(Float)
    
    # Incident type flags (boolean)
    type_surchauffe = Column(Integer)
    type_baisse_pression = Column(Integer)
    type_vibration = Column(Integer)
    type_bruit_mecanique = Column(Integer)
    type_surconsommation = Column(Integer)
    type_blocage_mecanique = Column(Integer)
    type_alarme_capteur = Column(Integer)
    type_arret_urgence = Column(Integer)
    type_defaut_qualite = Column(Integer)
    
    # Incident type counts (previous 24 hours)
    type_surchauffe_count_prev_24h = Column(Integer)
    type_baisse_pression_count_prev_24h = Column(Integer)
    type_vibration_count_prev_24h = Column(Integer)
    type_bruit_mecanique_count_prev_24h = Column(Integer)
    type_surconsommation_count_prev_24h = Column(Integer)
    type_blocage_mecanique_count_prev_24h = Column(Integer)
    type_alarme_capteur_count_prev_24h = Column(Integer)
    type_arret_urgence_count_prev_24h = Column(Integer)
    type_defaut_qualite_count_prev_24h = Column(Integer)
    
    # Maintenance features
    days_since_last_maintenance = Column(Float)
    maintenance_count_prev_30d = Column(Integer)
    
    # Future incident features (target variables for ML)
    future_incident_count_6h = Column(Integer)
    label_failure_next_6h = Column(String(50))
    future_incident_count_12h = Column(Integer)
    label_failure_next_12h = Column(String(50))
    future_incident_count_24h = Column(Integer)
    label_failure_next_24h = Column(String(50))
    future_incident_count_48h = Column(Integer)
    label_failure_next_48h = Column(String(50))
    
    # Dataset split identifier
    split_set = Column(String(50))
