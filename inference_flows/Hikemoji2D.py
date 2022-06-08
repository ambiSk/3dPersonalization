import requests
import time
import os
import csv

url_post = "http://10.20.6.17:5002/ds/internal/v1/selfie"
url_get = "http://10.20.6.17:5002/ds/internal/v1/selfie/%s?uid=test&msisdn=test&type=data"

headers = {}


def send_request(file_name, gender):
    payload = {'gender': gender,
               'version': 'v6-99999999',
               'uid': 'test',
               'msisdn': 'test',
               'app_name': 'rush',
               'type': 'data',
               'inference_type': '2D',
               'lod': 'a'}
    files = [
        ('file', (file_name.split("/")[-1], open(file_name, 'rb'), 'image/png'))
    ]

    response = requests.request("POST", url_post, headers=headers, data=payload, files=files)
    print(f"Request sent for {file_name}, {response.text}")
    return response.json()["id"]


def send_request_curl(file_name, gender):
    curl_req = f'''
    curl --location --request POST 'http://10.20.6.17:5002/ds/internal/v1/selfie' \
    --form 'file=@"{file_name}"' \
    --form 'gender="{gender}"' \
    --form 'version="v6-99999999"' \
    --form 'uid="test"' \
    --form 'msisdn="test"' \
    --form 'app_name="rush"' \
    --form 'type="data"' \
    --form 'inference_type="2D"' \
    --form 'lod="a"'
    '''

    response = eval(os.popen(curl_req).read())
    print(f"Request sent for {file_name}, {response}")
    return response["id"]


def get_data(request_id, gender):
    payload = {}

    response = requests.request("GET", url_get % request_id, headers=headers, data=payload)
    print("Response", response.text)
    response = response.json()["data"]

    if gender == 'male':
        return dict(gender = gender,
                Eyes=response.get("MaleEyes_Open_R", dict()).get("value", "").split("_")[0],
                Nose=response.get("MaleNose", dict()).get("value", ""),
                Lips=response.get("MaleLips", dict()).get("value", ""),
                Eyebrows=response.get("MaleEyebrows", dict()).get("value", ""),
                HairFront=response.get("MaleHairFront", dict()).get("value", ""),
                HairBack=response.get("MaleHairBack", dict()).get("value", ""),
                FaceShape=response.get("MaleFaceShape", dict()).get("value", ""),
                SkinColor=response.get("SkinColor", dict()),
                Eyewear=response.get("MaleEyewear", dict()).get("value", ""),
                Beard=response.get("MaleBasebeard", dict()).get("value", ""),
                Goatee=response.get("MaleGoatee", dict()).get("value", ""),
                Moustache=response.get("MaleMustache", dict()).get("value", ""))
                

    else:
        return dict(gender=gender,
                    Eyes=response.get("FemaleEyes_Open_R", dict()).get("value", "").split("_")[0],
                    Nose=response.get("FemaleNose", dict()).get("value", ""),
                    Lips=response.get("FemaleLips", dict()).get("value", ""),
                    Eyebrows=response.get("FemaleEyebrows", dict()).get("value", ""),
                    HairFront=response.get("FemaleHairFront", dict()).get("value", ""),
                    HairBack=response.get("FemaleHairBack", dict()).get("value", ""),
                    FaceShape=response.get("FemaleFaceShape", dict()).get("value", ""),
                    SkinColor=response.get("SkinColor", dict()),
                    Eyewear=response.get("FemaleEyewear", dict()).get("value", ""),
                    Beard=response.get("FemaleBasebeard", dict()).get("value", ""),
                    Goatee=response.get("FemaleGoatee", dict()).get("value", ""),
                    Moustache=response.get("FemaleMustache", dict()).get("value", ""))
                


ROWS_MALE = ["","image_name","gender","Eyes","Nose","Lips","Eyebrows","HairFront","HairBack",
             "FaceShape","SkinColor","Eyewear", "Beard", "Goatee", "Moustache"]
ROWS_FEMALE = ["","image_name","gender","Eyes","Nose","Lips","Eyebrows","HairFront","HairBack",
             "FaceShape","SkinColor","Eyewear", "Beard", "Goatee", "Moustache"]


def run_inference(image_path, gender='male'):
    if not (image_path.endswith("png") or image_path.endswith("jpg") or image_path.endswith("jpeg")):
        return
    request_id = send_request_curl(src_, gender)
    time.sleep(5)

    request_data = get_data(request_id, gender)

    return request_data
