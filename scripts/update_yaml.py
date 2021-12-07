import yaml
import collections
import click

"""
Updates one or more values for a one or more keys, 
with a provided path to a YAML file.
Will not replace keys found multiple times in the YAML.
Ignores keys in lists, even if they're dicts in lists.
"""


@click.command()
@click.option("--input_path",
               nargs=1,
               help="The path to the yaml to update.")
@click.option("--keys",
               callback=lambda _,__,x: x.split(',') if x else [],
               help="One or more keys to update the values for, comma-delimited.")
@click.option("--values",
               callback=lambda _,__,x: x.split(',') if x else [],
               help="One or more values, in the same order as keys, comma-delimited.")
def run(input_path, keys, values):

    with open(input_path, 'r') as yaml_file:
        contents = yaml.load(yaml_file, Loader=yaml.FullLoader)

    newkeyvalues = tuple(zip(keys, values))
    for key, newvalue in newkeyvalues:
        oldvalues = []
        print(f"Will set \"{key}\" to \"{newvalue}\".")
        for oldvalue in get_all_keyvalues(contents,key):
            oldvalues.append(oldvalue)
        if len(oldvalues) > 1:
            print(f"Found more than one value for \"{key}\"! Skipping.")
            continue
        elif len(oldvalues) == 0:
            print(f"Found no values for \"{key}\" in this yaml! Skipping.")
            continue
        else:
            contents = update_keyvalue(contents, key, newvalue)
            print("Done.")

    with open(input_path, 'w') as yaml_file:
        yaml_file.write(yaml.dump(contents, default_flow_style=False, sort_keys=False))


def update_keyvalue(input_dict, keyname, newvalue):
    """Function to update a provided key with a value.
    """
    new_dict = input_dict.copy()
    if keyname in new_dict:
        new_dict[keyname] = newvalue
        return new_dict
    for key, value in new_dict.items():
        if isinstance(value, dict):
            nested_dict = update_keyvalue(value, keyname, newvalue)
            if new_dict[key] != nested_dict:
                new_dict[key] = nested_dict
                return new_dict


def get_all_keyvalues(input_dict, keyname):
    """Generator function to iteratively search through
    all dict key value pairs.
    Returns all values for the given key.
    Useful for knowing if multiple values are present!
    Does not look within lists.
    """
    if keyname in input_dict:
        yield input_dict[keyname]
    for key, value in input_dict.items():
        if isinstance(value, dict):
            for item in get_all_keyvalues(value, keyname):
                yield item


if __name__ == '__main__':
  run()
