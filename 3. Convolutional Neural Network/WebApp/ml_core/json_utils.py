import json

def readJson_config(Path, Name, Keys):
    with open(Path + Name) as json_config:
        json_object = json.load(json_config)

    param = []
    for item in json_object:
        if item in Keys:
            param.append(json_object[item])

    return param

def writeJson_config(Path, Name, Data, append):
    mode = 'a+' if append else 'w'
    full_path = Path + Name

    with open(full_path, mode=mode) as json_config:
        json.dump(Data, json.load(json_config) if append else json_config)
    
    return 'success' 