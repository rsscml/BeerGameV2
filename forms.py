from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, SubmitField, SelectField, SelectMultipleField, IntegerField, FloatField, FieldList, FormField, BooleanField
from wtforms.validators import InputRequired
import numpy as np
import random

class UserActionForm(FlaskForm):
    RetailerOrderQty = FloatField("Retailer Order Quantity", default=0)
    WholesalerOrderQty = FloatField("Wholesaler Order Quantity", default=0)
    DistributorOrderQty = FloatField("Distributor Order Quantity", default=0)
    submit = SubmitField("Submit")

class GameInitForm(FlaskForm):
    GameName = StringField("Game Id")
    PlanningPeriods = IntegerField("Planning Periods", default=30)
    MeanCustomerDemand = FloatField("Mean Customer Demand", default=random.randint(10, 30))
    submit = SubmitField("Start Game")

class HomeForm(FlaskForm):
    submit = SubmitField("Proceed")