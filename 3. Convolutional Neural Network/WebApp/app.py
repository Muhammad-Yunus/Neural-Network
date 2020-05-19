#!venv/bin/python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, url_for, redirect, render_template, request, abort
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, \
    UserMixin, RoleMixin, login_required, current_user
from flask_security.utils import encrypt_password
import flask_admin
from flask_admin.contrib import sqla
from flask_admin import helpers as admin_helpers
from flask_admin import BaseView, AdminIndexView, expose

from tensorflow.compat.v1 import get_default_graph
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

# ML core service
from ml_core.preprocessing import TextPreprocessing
from ml_core.feature_engineering import featureEngineering
from ml_core.feature_selection import featureSelection
from ml_core.classification_predict import load_sparse
from ml_core.custom_metric import rec, prec, f1
from ml_core.classification_training import get_cnn_model
from ml_core.wrapper_training import run_ml_pipeline


import pickle

# Create Flask application
app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)



# load model
global model_wrapper
MODEL_FOLDER = "ml_core/model/"
MODEL_NAME = "cnn_model_training.h5"
CLASS_NAME = "cnn_class_training.pkl"

model_class = pickle.load(open(MODEL_FOLDER + CLASS_NAME,'rb'))
model_wrapper = KerasClassifier(build_fn=get_cnn_model, epochs=25, batch_size=6)
model_wrapper.model = load_model(MODEL_FOLDER + MODEL_NAME, custom_objects={"rec": rec, "prec": prec, "f1": f1})
model_wrapper.classes_ = model_class
model_wrapper.model.summary()

# Define models
roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

    def __str__(self):
        return self.name

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))

    def __str__(self):
        return self.email

class History(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    run_on = db.Column(db.DateTime())
    accuracy = db.Column(db.DECIMAL(20,16))
    precission = db.Column(db.DECIMAL(20,16))
    recall = db.Column(db.DECIMAL(20,16))

    def __str__(self):
        return self.run_on


# Setup Flask-Security
user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)


# Create customized model view class
class MyModelView(sqla.ModelView):

    def is_accessible(self):
        if not current_user.is_active or not current_user.is_authenticated:
            return False

        if current_user.has_role('superuser'):
            return True

        return False

    def _handle_view(self, name, **kwargs):
        """
        Override builtin _handle_view in order to redirect users when a view is not accessible.
        """
        if not self.is_accessible():
            if current_user.is_authenticated:
                # permission denied
                abort(403)
            else:
                # login
                return redirect(url_for('security.login', next=request.url))


    # can_edit = True
    edit_modal = True
    create_modal = True    
    can_export = True
    can_view_details = True
    details_modal = True

class UserView(MyModelView):
    column_editable_list = ['email', 'first_name', 'last_name']
    column_searchable_list = column_editable_list
    column_exclude_list = ['password']
    # form_excluded_columns = column_exclude_list
    column_details_exclude_list = column_exclude_list
    column_filters = column_editable_list

class HistoryView(MyModelView):
    column_editable_list = ['run_on', 'accuracy', 'precission', 'recall']
    column_searchable_list = column_editable_list
    column_exclude_list = []
    # form_excluded_columns = column_exclude_list
    column_details_exclude_list = column_exclude_list
    column_filters = column_editable_list

class About(BaseView):
    @expose('/')
    def index(self):
        return self.render('admin/about.html')

class Profile(BaseView):
    @expose('/')
    def index(self):
        user_id = request.args.get('user_id')
        profileData = User.query.filter_by(id=user_id).first()
        profileData.last_name = '' if profileData.last_name == None else profileData.last_name
        return self.render('admin/profile.html', profileData = profileData, show_feedback_success = False)
    
    def is_visible(self):
        return False
    
    @expose('/', methods=('GET', 'POST'))
    def on_model_change(self):
        user_id = request.form['id']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        active = True if request.form['active'] == 'on' else False
        UserUpdate = None
        UserSave = None

        if user_id != '':
            UserUpdate = User.query.filter_by(id=user_id).first()
            UserUpdate.first_name = first_name
            UserUpdate.last_name = last_name
            UserUpdate.email = email
            UserUpdate.active = active

            db.session.merge(UserUpdate)

            db.session.flush()
            db.session.refresh(UserUpdate)
            user_id = UserUpdate.id

            db.session.commit()
        else :
            UserSave = User(first_name=first_name, last_name=last_name, email=email, active=active)
            db.session.add(UserSave)

            db.session.flush()
            db.session.refresh(UserSave)
            user_id = UserSave.id

            db.session.commit()
        return self.render('admin/profile.html', 
            profileData = (UserUpdate if UserUpdate is not None else UserSave), 
            show_feedback_success = True)

class Preprocessing(BaseView):
    @expose('/')
    def index(self):
        return self.render('admin/preprocessing.html', 
                show_feedback_success = False, 
                documents = None, 
                tableRecords= None)
    
    @expose('/', methods=('GET', 'POST'))
    def on_after_submit(self):
        tableRecords = []
        documents = request.form['documents']
        tableRecords = TextPreprocessing(documents)
        return self.render('admin/preprocessing.html', 
                show_feedback_success = True,
                documents = documents, 
                tableRecords = tableRecords)

class FeatureEngineering(BaseView):
    @expose('/')
    def index(self):
        tableRecords = featureEngineering()
        return self.render('admin/feature_engineering.html', 
                show_feedback_success = False, 
                tableRecords= tableRecords)

class FeatureSelection(BaseView):
    @expose('/')
    def index(self):
        tableRecords = featureSelection()
        return self.render('admin/feature_selection.html', 
                tableRecords= tableRecords)

class Classification(BaseView):
    @expose('/')
    def index(self):
        X = load_sparse()
        Y = []

        # with graph.as_default():
        Y = model_wrapper.predict(X)
        E = model_wrapper.model.predict(X)
        print(E[0][0])
        return self.render('admin/classification.html', 
                tableRecords= zip(Y, E))

class TrainingModel(BaseView):
    @expose('/')
    def index(self):
        result = ml_pipeline_job.delay()
        result.wait()
        return self.render('admin/training.html')

# Flask views
@app.route('/')
def index():
    return render_template('index.html')

# Home views
class MyHomeView(AdminIndexView):
    @expose('/')
    def index(self):
        return self.render('admin/about.html')

# Create admin
admin = flask_admin.Admin(
    app,
    'Hoax Classifier',
    base_template='my_master.html',
    template_mode='bootstrap3',
    index_view=MyHomeView()
)


# Add model views
admin.add_view(About(name="About", endpoint='about', menu_icon_type='fa', menu_icon_value='fa-info-circle'))

admin.add_view(HistoryView(History, db.session, menu_icon_type='fa', menu_icon_value='fa-database', name="History"))
admin.add_view(Preprocessing(name="Preprocessing", endpoint='preprocessing', menu_icon_type='fa', menu_icon_value='fa-cogs'))
admin.add_view(FeatureEngineering(name="Feature Engineering", endpoint='feature_engineering', menu_icon_type='fa', menu_icon_value='fa-braille'))
admin.add_view(FeatureSelection(name="Feature Selection", endpoint='feature_selection', menu_icon_type='fa', menu_icon_value='fa-filter'))
admin.add_view(Classification(name="Classification Result", endpoint='classification', menu_icon_type='fa', menu_icon_value='fa-magic'))
admin.add_view(TrainingModel(name="Training Model", endpoint='raining_model', menu_icon_type='fa', menu_icon_value='fa-area-chart'))

admin.add_view(MyModelView(Role, db.session, menu_icon_type='fa', menu_icon_value='fa-server', name="Roles"))
admin.add_view(UserView(User, db.session, menu_icon_type='fa', menu_icon_value='fa-users', name="Users"))
admin.add_view(Profile(name="Profile", endpoint='profile'))


# define a context processor for merging flask-admin's template context into the
# flask-security views.
@security.context_processor
def security_context_processor():
    return dict(
        admin_base_template=admin.base_template,
        admin_view=admin.index_view,
        h=admin_helpers,
        get_url=url_for
    )


def build_sample_db():
    """
    Populate a small db with some example entries.
    """

    import string
    import random

    db.drop_all()
    db.create_all()

    with app.app_context():
        user_role = Role(name='user')
        super_user_role = Role(name='superuser')
        db.session.add(user_role)
        db.session.add(super_user_role)
        db.session.commit()

        test_user = user_datastore.create_user(
            first_name='Admin',
            email='admin',
            password=encrypt_password('admin'),
            roles=[user_role, super_user_role]
        )

        first_names = [
            'User'
        ]
        last_names = [
            'Test'
        ]

        for i in range(len(first_names)):
            tmp_email = first_names[i].lower() + "." + last_names[i].lower() + "@example.com"
            tmp_pass = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(10))
            user_datastore.create_user(
                first_name=first_names[i],
                last_name=last_names[i],
                email=tmp_email,
                password=encrypt_password(tmp_pass),
                roles=[user_role, ]
            )
        db.session.commit()
    return

if __name__ == '__main__':

    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))
    database_path = os.path.join(app_dir, app.config['DATABASE_FILE'])
    if not os.path.exists(database_path):
        print("-------------CREATE-------------")
        build_sample_db()
    else :
        print("-------------EXIST-------------")

    # Start app
    app.run(debug=False, threaded=False)