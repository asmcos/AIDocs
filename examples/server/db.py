from sqlalchemy import create_engine,Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
import sys
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
class Image(Base):
    __tablename__ = 'img_info'

    id = Column(Integer, primary_key=True)
    imagename = Column(String(255))
    imagelabel = Column(Integer)

engine = create_engine('sqlite:///'+sys.path[0]+'/img.db')

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def insert(filename,label):
    img = Image(filename, label)
    session.add(img)
    session.commit()

def get_all():
    imgs = session.query(Image)
    for row in imgs: 
        print(row.id, row.imagename,row.imagelabel)
    return imgs

