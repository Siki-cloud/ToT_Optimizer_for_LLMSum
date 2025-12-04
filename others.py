import json

def pre_processing():
    data_file = "data/component_library.json"
    with open(data_file,'r',encoding='utf-8') as file:
        data = json.load(file)
    diff_keys = []
    for item in data:
        key = item['id']
        if key not in diff_keys:
            diff_keys.append(key)
    print(f"{len(data)}. diff. keys:{len(diff_keys)}")


if __name__ == "__main__":
    pre_processing()