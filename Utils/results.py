from sqlalchemy import Column, Integer, String,DateTime,Float
from Utils.database import Base


class Result(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, index=True)
    speed = Column(Float)
    created = Column(DateTime)
