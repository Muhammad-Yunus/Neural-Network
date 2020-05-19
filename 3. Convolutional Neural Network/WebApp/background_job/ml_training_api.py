from init.init_celery_redis import make_celery
from flask import Flask

#------------------------------------------------------------
# setup for windows OS
import os
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
#------------------------------------------------------------

flask_app = Flask(__name__)
flask_app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)



celery = make_celery(flask_app)

@celery.task()
def add_together(a, b):
    return a + b