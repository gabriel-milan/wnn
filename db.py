from os import getenv
import numpy as np
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, Integer, String, Boolean, create_engine
from sqlalchemy_utils import database_exists, create_database


__all__ = ["DBManager"]

Base = declarative_base()


class EnsembleTable(Base):

    __tablename__ = "ensemble"

    # Columns
    id = Column(Integer, primary_key=True)
    mode = Column(String)
    n_best = Column(Integer)
    threshold = Column(Float)
    weights = Column(String)
    et1_val_sp = Column(Float)
    et2_val_sp = Column(Float)
    et3_val_sp = Column(Float)
    et4_val_sp = Column(Float)
    et5_val_sp = Column(Float)
    et1_val_pd = Column(Float)
    et2_val_pd = Column(Float)
    et3_val_pd = Column(Float)
    et4_val_pd = Column(Float)
    et5_val_pd = Column(Float)
    et1_val_fa = Column(Float)
    et2_val_fa = Column(Float)
    et3_val_fa = Column(Float)
    et4_val_fa = Column(Float)
    et5_val_fa = Column(Float)


class TrainTable(Base):

    __tablename__ = "train"

    # Columns
    id = Column(Integer, primary_key=True)
    wsd_cluster = Column(Boolean)
    feature_set = Column(String)
    random_seed = Column(Integer)
    train_validation_split = Column(Float)
    train_folds = Column(Integer)
    binarization = Column(String)
    binarization_threshold = Column(Integer)
    binarization_resolution = Column(Integer)
    wsd_address_size = Column(Integer)
    wsd_ignore_zero = Column(Boolean)
    wsd_verbose = Column(Boolean)
    clus_min_score = Column(Float)
    clus_threshold = Column(Integer)
    clus_discriminator_limit = Column(Integer)
    window_size = Column(Integer)
    constant_c = Column(Float)
    constant_k = Column(Float)
    et1_val_sp = Column(Float)
    et2_val_sp = Column(Float)
    et3_val_sp = Column(Float)
    et4_val_sp = Column(Float)
    et5_val_sp = Column(Float)
    et1_val_pd = Column(Float)
    et2_val_pd = Column(Float)
    et3_val_pd = Column(Float)
    et4_val_pd = Column(Float)
    et5_val_pd = Column(Float)
    et1_val_fa = Column(Float)
    et2_val_fa = Column(Float)
    et3_val_fa = Column(Float)
    et4_val_fa = Column(Float)
    et5_val_fa = Column(Float)

    def __repr__(self) -> str:
        return "<Wisard Train (accuracy={}, clus_wizard={}, address_size={}, binarization={}, binarization_threshold={}, binarization_resolution={})>".format(
            np.mean([self.et1_val_sp, self.et2_val_sp,
                    self.et3_val_sp, self.et4_val_sp, self.et5_val_sp]),
            self.wsd_cluster,
            self.wsd_address_size,
            self.binarization,
            self.binarization_threshold,
            self.binarization_resolution
        )


class DBManager:

    def __init__(self):
        connection_url: str = getenv("DB_CONNECTION_URL", "")
        if connection_url == "":
            raise Exception(
                "Environment variable DB_CONNECTION_URL not set.")
        self._engine = create_engine(connection_url)
        if not database_exists(self._engine.url):
            create_database(self._engine.url)
            Base.metadata.create_all(self._engine)

        self._session: Session = sessionmaker(bind=self._engine)()

    @property
    def session(self) -> Session:
        return self._session

    def add_ensemble_result(self, config: dict, scores: list):
        et1 = scores[0]
        et2 = scores[1]
        et3 = scores[2]
        et4 = scores[3]
        et5 = scores[4]
        ensemble_result = EnsembleTable(
            et1_val_sp=et1[0],
            et2_val_sp=et2[0],
            et3_val_sp=et3[0],
            et4_val_sp=et4[0],
            et5_val_sp=et5[0],
            et1_val_pd=et1[1],
            et2_val_pd=et2[1],
            et3_val_pd=et3[1],
            et4_val_pd=et4[1],
            et5_val_pd=et5[1],
            et1_val_fa=et1[2],
            et2_val_fa=et2[2],
            et3_val_fa=et3[2],
            et4_val_fa=et4[2],
            et5_val_fa=et5[2],
            **config
        )
        self._session.add(ensemble_result)
        self._session.commit()

    def add_train_result(self, config: dict, scores: list):
        et1 = scores[0]
        et2 = scores[1]
        et3 = scores[2]
        et4 = scores[3]
        et5 = scores[4]
        train_result = TrainTable(
            et1_val_sp=et1[0],
            et2_val_sp=et2[0],
            et3_val_sp=et3[0],
            et4_val_sp=et4[0],
            et5_val_sp=et5[0],
            et1_val_pd=et1[1],
            et2_val_pd=et2[1],
            et3_val_pd=et3[1],
            et4_val_pd=et4[1],
            et5_val_pd=et5[1],
            et1_val_fa=et1[2],
            et2_val_fa=et2[2],
            et3_val_fa=et3[2],
            et4_val_fa=et4[2],
            et5_val_fa=et5[2],
            **config
        )
        self._session.add(train_result)
        self._session.commit()
