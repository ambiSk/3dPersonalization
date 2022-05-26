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
        return [response.get("MaleEyes_Open_R", dict()).get("value", "").split("_")[0],
                response.get("MaleNose", dict()).get("value", ""),
                response.get("MaleLips", dict()).get("value", ""),
                response.get("MaleEyebrows", dict()).get("value", ""),
                response.get("MaleHairFront", dict()).get("value", ""),
                response.get("MaleHairBack", dict()).get("value", ""),
                response.get("MaleFaceShape", dict()).get("value", ""),
                response.get("SkinColor", dict()),
                response.get("MaleEyewear", dict()).get("value", ""),
                response.get("MaleBasebeard", dict()).get("value", ""),
                response.get("MaleGoatee", dict()).get("value", ""),
                response.get("MaleMustache", dict()).get("value", ""),
                ]

    else:
        return [response.get("FemaleEyes_Open_R", dict()).get("value", "").split("_")[0],
                response.get("FemaleNose", dict()).get("value", ""),
                response.get("FemaleLips", dict()).get("value", ""),
                response.get("FemaleEyebrows", dict()).get("value", ""),
                response.get("FemaleHairFront", dict()).get("value", ""),
                response.get("FemaleHairBack", dict()).get("value", ""),
                response.get("FemaleFaceShape", dict()).get("value", ""),
                response.get("SkinColor", dict()),
                response.get("FemaleEyewear", dict()).get("value", ""),
                response.get("FemaleBasebeard", dict()).get("value", ""),
                response.get("FemaleGoatee", dict()).get("value", ""),
                response.get("FemaleMustache", dict()).get("value", ""),
                ]


ROWS_MALE = ["","image_name","gender","Eyes","Nose","Lips","Eyebrows","HairFront","HairBack",
             "FaceShape","SkinColor","Eyewear", "Beard", "Goatee", "Moustache"]
ROWS_FEMALE = ["","image_name","gender","Eyes","Nose","Lips","Eyebrows","HairFront","HairBack",
             "FaceShape","SkinColor","Eyewear", "Beard", "Goatee", "Moustache"]


def run_inference(images_path, csv_file=None, gender='male'):
    if os.path.isfile(images_path):
        img_paths = [images_path, ]
    else:
        img_paths = [os.path.join(images_path, img_path) for img_path in os.listdir(images_path)]

    if csv_file is not None:
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        out_file = open(csv_file, "w")
        csvwriter = csv.writer(out_file)

        if gender == 'male':
            csvwriter.writerow(ROWS_MALE)
        elif gender == 'female':
            csvwriter.writerow(ROWS_FEMALE)

    for (i, src_) in enumerate(img_paths):
        if not (src_.endswith("png") or src_.endswith("jpg") or src_.endswith("jpeg")):
            continue

        request_id = send_request_curl(src_, gender)
        time.sleep(5)

        request_data = get_data(request_id, gender)

        if csv_file is not None:
            csvwriter.writerow([i, src_.split("/")[-1], gender] + request_data)

    if csv_file:
        out_file.close()

# if __name__ == "__main__":
#     run_inference('/Users/pankajdahiya/Downloads/Selfies', os.path.join(os.path.dirname(__file__), "component_values_maleV2.csv"), 'male')
