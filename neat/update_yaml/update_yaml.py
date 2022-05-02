import yaml

def do_update_yaml(input_path: str, keys: list, values: list) -> None:
    with open(input_path, 'r') as yaml_file:
        contents = yaml.load(yaml_file, Loader=yaml.FullLoader)

    newkeyvalues = tuple(zip(keys, values))
    for key, newvalue in newkeyvalues:
        print(f"Will set \"{key}\" to \"{newvalue}\".")

        try:
            # Parse the input key
            if len(key.split(":")) == 1:
                contents = update_keyvalue(contents, [key], newvalue)
            else:
                keylist = key.split(":")
                contents = update_keyvalue(contents, keylist, newvalue)
        except KeyError:
            print(f"Could not find \"{key}\"! Skipping.")
            continue

        print("Done.")

    with open(input_path, 'w') as yaml_file:
        yaml_file.write(yaml.dump(contents, default_flow_style=False, sort_keys=False))


def update_keyvalue(input_dict, keys, newvalue):
    """
    Function to update a provided key with a value.
    :param input_dict: YAML dict representation to be updated
    :param keys: list containing one or more keys. 
                    Multiple keys are nested, e.g. input_dict[key1][key2[key3].
    :param newvalue: value to set key to.
    :return: dict
    """
    new_dict = input_dict.copy()

    if len(keys) == 1 and keys[0] in new_dict:
        new_dict[keys[0]] = newvalue
        return new_dict
    else:
        i = 0
        current_dict = new_dict[keys[i]]
        while True:
            if i == len(keys)-2:
                current_dict[keys[i+1]] = newvalue
                return new_dict
            else:
                i = i+1
                current_dict = current_dict[keys[i]]
                
                

