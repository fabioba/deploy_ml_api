"""
This module tests the application deployed on Heroku

Author: Fabio Barbazza
Date: 1st of Feb, 2022
"""
from cgi import test
import requests
import json

def test_post_heroku():
    """
        Tests if the post requests works
    """
    try:

        url_inference='https://census-app-fb.herokuapp.com/inference'

        body={"age":31, "workclass":"Self-emp-not-inc",
            "fnlgt":45781, "education":"HS-grad","education_num":14, "marital_status":"Married-civ-spouse", 
            "occupation":"Tech-support","relationship":"Husband", "race":"White", "sex":"Male","capital_gain":14084,
            "capital_loss":0,"hours_per_week":50, "native_country":"United-States","salary":">50K"}

        r = requests.post(url_inference, json=body)
        print(r)

        assert r.status_code==200,'STATUS CODE != 200'
        assert r.content != None, 'content NONE'
        
        print('status code: {}'.format(r.status_code))
        print('content: {}'.format(r.content))

    except Exception as err:
        print(err)

if __name__=='__main__':
    test_post_heroku()