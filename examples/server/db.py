from sqlalchemy import create_engine,Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
import sys
from sqlalchemy.orm import sessionmaker
# 1.3.0

Base = declarative_base()

class Image(Base):
    __tablename__ = 'img_info'

    id = Column(Integer, primary_key=True)
    imagename = Column(String(255))
    imagelabel = Column(Integer)

class predImage(Base):
    __tablename__ = 'predimg_info'

    id =        Column(Integer, primary_key=True)
    imagename = Column(String(255))
    imagelabel= Column(Integer)
    pred =      Column(Integer)

    def to_dict(self):
       return {c.name: getattr(self, c.name) for c in self.__table__.columns}


engine = create_engine('sqlite:///'+sys.path[0]+'/img.db',connect_args={"check_same_thread": False})

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def insert(filename,pred):
    img = predImage(imagename=filename, pred=pred)
    session.add(img)
    session.commit()

def update(imgid,y):
    img = session.query(predImage).get(imgid)
    img.imagelabel = y
    session.commit()

def deletepred(predid):
    preimg = session.query(predImage).get(imgid)
    session.delete(preimg)
    session.commit() 

def get_all_pred():
    imgs = session.query(predImage)
    return [img.to_dict() for img in imgs]

def makeImagefrompred(predId,label):
    predimg = session.query(predImage).get(imgid)
    if predimg:
        img = session.query(Image).filter_by(imagename=predimg.imagename).first()
        if img:
            img.imagelabel=label
        else:
            img = Image({'imagename':predimg.imagename,'imagelabel':label})
            session.add(img)
        session.commit()
    return "ok"

            
def get_all():
    imgs = session.query(Image)
    for row in imgs: 
        print(row.id, row.imagename,row.imagelabel)
    return imgs

