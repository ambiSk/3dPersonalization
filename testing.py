import requests
import argparse
import os
import time
import json
import base64


ml_hikemoji_post_url = "http://10.20.6.17:5002/ds/internal/v1/selfie"
ml_hikemoji_get_url = "http://10.20.6.17:5002/ds/internal/v1/selfie/"

###3d  dragonbones_sticker_creation_url =  "http://selfie-stitching-ds.hike.in:5000/processimage/getProfileImage"
dragonbones_sticker_creation_url =  "http://10.28.0.14:3001/processimage/getProfileImage"

ml_hikemoji_post_url_headers = {
    'cookie': 'user=SS86RZFX7do='
}

ml_hikemoji_get_url_headers = {
    'cookie': 'user=SS86RZFX7do='
}

dragonbones_sticker_creation_url_header = {
    'Content-Type': "application/json"
}

supported_ext = ["png", "jpg", "jpeg", "PNG", "JPEG", "JPG"]


def dragon_bones_create_sticker(final_dict,  output_file_name):

    final_dict = json.dumps(final_dict)
    base_64_response = requests.request("POST", dragonbones_sticker_creation_url, data=final_dict, headers=dragonbones_sticker_creation_url_header)

    if base_64_response.status_code == 200:
        base_64_image = json.loads(base_64_response.text)['avatar']
        image = base64.b64decode(base_64_image)
        with open(output_file_name, "wb") as photo_file:
            photo_file.write(image)
    else:
        print("RESPONSE DRAGON BONES:", base_64_response.status_code)

def change_data_format(avatar_dict, gender, version):

    final_dict = {}
    final_dict['gender'] = gender
    final_dict['version'] = version
    final_dict['data'] = {}

    data_dict = avatar_dict.get("data")

    if avatar_dict is None:
        return final_dict

    for component_keys in data_dict.keys():
        if component_keys == "SkinColor":
            continue
        else:
            final_dict["data"][component_keys] = {}
        if isinstance(data_dict[component_keys], dict):
            for subkeys in data_dict[component_keys].keys():
                if isinstance(data_dict[component_keys][subkeys], str):
                    if subkeys == "value":
                        final_dict["data"][component_keys]["name"] = data_dict[component_keys]["value"]
                    if subkeys == "color":
                        final_dict["data"][component_keys]["color"] = data_dict[component_keys]["color"]
                    if subkeys == "x_scale":
                        final_dict["data"][component_keys]["ratio"] = data_dict[component_keys]["x_scale"]

    return final_dict



def ml_hikemoji_post_call(file_path, payload):
    files =  {'file': open(file_path, 'rb')}
    response = requests.request("POST", ml_hikemoji_post_url, headers=ml_hikemoji_post_url_headers, data=payload, files=files)
    if response.status_code == 200:
        return response.text
    else:
        print("FAILED ML HIKEMOJI POST:", response.text, response.status_code)
        return None

def ml_hikemoji_get_call(ml_request_id, params_load=None):
    if params_load is None:
        params_load = {}
    response = requests.request("GET", ml_hikemoji_get_url + ml_request_id, headers=ml_hikemoji_get_url_headers, params=params_load)
    if response.status_code == 200 and json.loads(response.text)['data'] is not None:
        return response.text
    else:
        print("FAILED ML HIKEMOJI GET:", response.text, response.status_code)
        return None


arguments = argparse.ArgumentParser()
arguments.add_argument('--input_dir', type=str, default='./SelfiesUXR/')
arguments.add_argument('--gender', type=str, default='male')
arguments.add_argument('--version', type=str, default='v6-999999999')
arguments.add_argument('--uid', type=str, default='ml_infer_script')
arguments.add_argument('--msisdn', type=str, default='ml_infer_script')
arguments.add_argument('--inference_2d', type=str, default="no")

args = arguments.parse_args()

def main(input_dir, gender):
    
    version = args.version
    uid = args.uid
    msisdn = args.msisdn

    json_output_dir = input_dir + "output_json/"
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    sticker_output_dir = input_dir + "output_stickers/"
    if not os.path.exists(sticker_output_dir):
        os.makedirs(sticker_output_dir)

    payload = {'gender': gender, 'version': version, "uid":uid, "msisdn":msisdn}
    params_load = {"uid":uid, "msisdn":msisdn}

    for file in os.listdir(input_dir):
        print("Processing file", file)
        if os.path.isfile(input_dir + file) and ".DS_Store" not in file and file.split(".")[1] in supported_ext:
            output_json_path = json_output_dir + file.split(".")[0] + ".json"
            if os.path.exists(output_json_path):
                continue
            file_path = input_dir + file
            ml_request = ml_hikemoji_post_call(file_path=file_path, payload=payload)
            print("ML REQUEST ID:", ml_request)
            if ml_request is None:
                print("ML ID IS NONE - ABORTING")
                continue
            else:
                ml_request = json.loads(ml_request)
                ml_request_id  = ml_request.get("id")
                time.sleep(3)

            for _ in range(10):
                ml_output = ml_hikemoji_get_call(ml_request_id=ml_request_id, params_load=params_load)

                if ml_output is None:
                    time.sleep(0.5)
                    continue
                else:
                    ml_output_dict = json.loads(ml_output)
                    output_json_path = json_output_dir + file.split(".")[0] + ".json"
                    with open(output_json_path, "w") as f:
                        json.dump(ml_output_dict, f)
                    #break
                    # print(ml_output_dict)
                    output_sticker_path = sticker_output_dir + file.split(".")[0] + ".png"
                    dragon_bones_sticker_dict = change_data_format(ml_output_dict, gender, version.split("-")[0])
                    dragon_bones_create_sticker(dragon_bones_sticker_dict, output_sticker_path)
                    break
    print("DONE!")

if __name__== '__main__':
    main(args.input_dir, args.gender)