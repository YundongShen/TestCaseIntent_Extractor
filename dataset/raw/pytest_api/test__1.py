import numpy as np
import mlflow
import score
import random
import string
import requests
import subprocess
import time
import json

def test_score():
    LogisticRegression = mlflow.sklearn.load_model("mlruns/0/a211eafb22c94b63b49575f2a2f25a28/artifacts/model/")

    # generating random input string of length 100
    random_text = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(100))
    random_threshold = random.random()

    random_output = score.score(random_text, LogisticRegression, random_threshold)

    # format test
    assert type(random_output[0]) == bool
    assert type(random_output[1]) == np.float64
    
    # prediction value must be True or False
    assert random_output[0] in [True, False]

    # propensity score must be between 0 and 1
    assert 0 <= random_output[1] <= 1

    # setting threshold to 0, prediction should be True
    assert  score.score(random_text, LogisticRegression, 0)[0] == True

    # setting threshold to 1, prediction should be False
    assert  score.score(random_text, LogisticRegression, 1)[0] == False

    # obvious spam text should have prediction True
    spam_text = "PRIVATE! Your 2003 Account Statement for 07815296484 shows 800 un-redeemed S.I.M. points. Call 08718738001 Identifier Code 41782 Expires 18/11/04"
    assert score.score(spam_text, LogisticRegression, 0.5)[0] == True

    # obvious non-spam text should have prediction False
    non_spam_text = "No it was cancelled yeah baby! Well that sounds important so i understand my darlin give me a ring later on this fone love Kate x"
    assert score.score(non_spam_text, LogisticRegression, 0.5)[0] == False

def test_flask():
    flask_app = subprocess.Popen(["python3", "-m", "flask", "--app", "app", "run"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(10)

    # post request for a spam text
    spam_text = "PRIVATE! Your 2003 Account Statement for 07815296484 shows 800 un-redeemed S.I.M. points. Call 08718738001 Identifier Code 41782 Expires 18/11/04"
    r = requests.post('http://localhost:5000/score', data={"text": spam_text})

    response = json.loads(r.text)

    assert response["prediction"] == True
    assert response["propensity"] > 0.5

    flask_app.terminate()

def test_docker():
    # run the docker container
    subprocess.run(["docker", "run", "-d", "-p", "5000:5000", "--name", "aml-assignment", "aml-assignment"])
    time.sleep(10)

    # post request for a spam text
    spam_text = "PRIVATE! Your 2003 Account Statement for 07815296484 shows 800 un-redeemed S.I.M. points. Call 08718738001 Identifier Code 41782 Expires 18/11/04"
    r = requests.post('http://localhost:5000/score', data={"text": spam_text})

    response = json.loads(r.text)

    assert response["prediction"] == True
    assert response["propensity"] > 0.5

    # close the docker container
    subprocess.run(["docker", "stop", "aml-assignment"])
    subprocess.run(["docker", "rm", "aml-assignment"])
