import os
import json
import pandas as pd

def get_preset_hair_color(gender):
    
    input_path = './input_images/output_json/'
    
    output_path = './input_files/' + gender + '/'

    columnvalues = ['image_name','gender','HairFront','HairBack','Eyebrows','SkinColor']
    df = pd.DataFrame(columns=columnvalues)

    if gender == 'male':
        st = 4
    else:
        st = 6

    for file in os.listdir(input_path):
        file_path = input_path + file

        component_data = json.load(open(file_path))
        component_data = component_data['data']

        target_list = ['hair','skincolor','eyes','eyebrow']

        eyes = 0
        valuedict = {key: 'null' for key in columnvalues}
        valuedict['image_name'] = file.split('.')[0] + '.png'
        valuedict['gender'] = gender

        for key,value in component_data.items():

            if any(s in key.lower() for s in target_list):

                if ('eyes' in key.lower() and eyes == 0) or ('eyes' not in key.lower()):
                    eyes = 1

                    if ('eyes' not in key.lower()):
                        if 'skincolor' in key.lower():
                            newkey = key[0:len(key)]
                            newvalue = value
                        else:
                            newkey = key[st:len(key)]
                            newvalue = value['value']
                    else:
                        newkey = key[st:key.find('_')]
                        newvalue = value['value'].split('_')[0]

                    if str(newkey) in valuedict:
                        valuedict[str(newkey)] = newvalue

        df = df.append([valuedict],ignore_index=True,sort=False)

    if gender == 'male':
        df.to_csv(output_path+'component_values_male.csv',index=False)
    else:
        df.to_csv(output_path+'component_values_female.csv',index=False)